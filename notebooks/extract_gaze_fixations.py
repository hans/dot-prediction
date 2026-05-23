# %% [markdown]
# # Phase 2 — Tobii gaze projection + fixation extraction
#
# Per-sample gaze in iPad-screen and behavior-canvas coordinates, plus
# fixation events collapsed from Tobii's built-in classifier.
#
# Inputs:
#   - data/{subject}/tobii/{tobii_tsv}     — raw Tobii TSV
#   - results/{subject}/phase1c_per_frame.parquet — per-frame screen→frame H
#   - results/{subject}/video_alignment.json      — video_t ↔ behavior_t affine
#   - results/{subject}/trials_with_video.parquet — trial events
#
# Outputs (written to `out_dir`):
#   - gaze_per_sample.parquet
#   - fixation_events.parquet
#   - gaze_coverage_and_accuracy.png
#   - gaze_canvas_heatmap.png
#   - pre_click_gaze_trajectories.png
#   - saccade_psth_around_reveal.png
#   - gaze_screen_overlay.mp4

# %% tags=["parameters"]
import sys
from pathlib import Path

_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
    if Path("src").is_dir()
    else Path.cwd().parent
)
sys.path.insert(0, str(_ROOT / "src"))

subject = "EC347"
video_path = str(_ROOT / f"data/{subject}/tobii/scenevideo.mp4")
tsv_path = str(_ROOT / f"data/{subject}/tobii/EC347_B16_tobii.tsv")
phase1c_path = str(_ROOT / f"results/{subject}/phase1c_per_frame.parquet")
align_path = str(_ROOT / f"results/{subject}/video_alignment.json")
trials_path = str(_ROOT / f"results/{subject}/trials_with_video.parquet")
out_dir = str(_ROOT / f"results/{subject}/eyetrack")

URL_BAR_H_PX = 272
CANVAS_X_PAD_PX = 233
MAX_Y_COORD = 0.75
SCREEN_W_PX = 2388
SCREEN_H_PX = 1668
CLIP_START_S = 120.0
CLIP_DURATION_S = 30.0

# %%
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gaze_projection import (
    is_on_screen,
    project_video_to_screen,
    screen_to_canvas,
    tobii_ts_to_behavior_ms,
    tobii_ts_to_video_frame_frac,
)
from homography_solver import behavior_to_screen

out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

CANVAS_W_PX = SCREEN_W_PX - 2 * CANVAS_X_PAD_PX     # 1922
CANVAS_H_PX = SCREEN_H_PX - URL_BAR_H_PX             # 1396
CANVAS_USABLE_H_PX = CANVAS_H_PX * MAX_Y_COORD       # 1047 px (top-75%)

print(f"Canvas geometry: {CANVAS_W_PX} × {CANVAS_H_PX} px  "
      f"(usable y: 0..{CANVAS_USABLE_H_PX:.0f} px = top-{MAX_Y_COORD:.0%})")

# %% [markdown]
# ## Load inputs

# %%
with open(align_path) as f:
    align = json.load(f)
slope_ms_per_s = align["slope_ms_per_s"]
intercept_ms = align["intercept_ms"]
fps = align["fps"]
print(f"Alignment: slope={slope_ms_per_s:.6f} ms/s  intercept={intercept_ms:.1f} ms  fps={fps:.6f}")

per_frame_df = pd.read_parquet(phase1c_path)
trials_df = pd.read_parquet(trials_path)
print(f"Per-frame H: {len(per_frame_df)} rows  "
      f"({(per_frame_df.detection_status == 'no_screen').sum()} no_screen)")
print(f"Trials: {len(trials_df)} rows across {trials_df.trial_idx.nunique()} trials  "
      f"({trials_df.response_time.notna().sum()} with clicks)")

# Build per-frame H tensor (N_frames, 3, 3). NaN-filled rows mark no_screen frames.
_H_COLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
N_FRAMES = int(per_frame_df.frame_idx.max()) + 1
per_frame_h = np.full((N_FRAMES, 3, 3), np.nan)
for _, r in per_frame_df.iterrows():
    fidx = int(r.frame_idx)
    if pd.isna(r.h00):
        continue
    per_frame_h[fidx] = np.array([
        [r.h00, r.h01, r.h02],
        [r.h10, r.h11, r.h12],
        [r.h20, r.h21, r.h22],
    ])
n_valid_frames = (~np.isnan(per_frame_h).any(axis=(1, 2))).sum()
print(f"Per-frame H tensor: {per_frame_h.shape}  ({n_valid_frames}/{N_FRAMES} frames have H)")

# %%
# Read only the Tobii TSV columns we need; the file is 200+ MB.
_TSV_COLS = [
    "Recording timestamp", "Sensor",
    "Gaze point X", "Gaze point Y",
    "Validity left", "Validity right",
    "Eye movement type", "Eye movement type index",
    "Eye movement event duration",
    "Fixation point X", "Fixation point Y",
]
tobii_df = pd.read_csv(tsv_path, sep="\t", usecols=_TSV_COLS, low_memory=False)
print(f"Tobii TSV: {len(tobii_df)} total rows across all sensors")

# Filter to Eye Tracker rows — those are the gaze samples (Gyro/Accel rows
# carry no gaze information). Spec defines outputs in terms of gaze samples;
# downstream metrics (gaze_valid rate, fixation collapse) only make sense for
# Eye Tracker rows.
tobii_df = tobii_df[tobii_df["Sensor"] == "Eye Tracker"].reset_index(drop=True)
print(f"After Eye Tracker filter: {len(tobii_df)} samples  "
      f"({len(tobii_df) / (tobii_df['Recording timestamp'].max() / 1e6):.1f} Hz)")

# %% [markdown]
# ## Project every sample
#
# Vectorised pipeline: build per-sample H by element-wise lerp of the flanking
# integer-frame Hs, then batch-invert all per-sample H and apply to gaze.

# %%
ts_us = tobii_df["Recording timestamp"].values.astype(np.int64)
gx_video = tobii_df["Gaze point X"].values.astype(np.float64)
gy_video = tobii_df["Gaze point Y"].values.astype(np.float64)

behavior_t_ms = tobii_ts_to_behavior_ms(ts_us, slope_ms_per_s, intercept_ms)
video_frame_frac = tobii_ts_to_video_frame_frac(ts_us, fps)

# Per-sample lerp'd H, vectorised.
# Clip to valid frame range so floor_idx/ceil_idx don't OOB.
ff = np.clip(video_frame_frac, 0.0, N_FRAMES - 1)
floor_idx = np.floor(ff).astype(np.int64)
ceil_idx = np.minimum(floor_idx + 1, N_FRAMES - 1)
alpha = (ff - floor_idx).reshape(-1, 1, 1)
h_lo = per_frame_h[floor_idx]   # (N, 3, 3)
h_hi = per_frame_h[ceil_idx]
H_per_sample = (1.0 - alpha) * h_lo + alpha * h_hi

# homography_valid: neither flanker had NaN
homography_valid = ~(np.isnan(h_lo).any(axis=(1, 2)) | np.isnan(h_hi).any(axis=(1, 2)))
print(f"homography_valid: {homography_valid.mean():.1%}  ({homography_valid.sum()}/{len(homography_valid)})")

# gaze_valid: Validity columns are "Valid"/"Invalid" strings in this Tobii export
val_left = tobii_df["Validity left"].astype("string").values
val_right = tobii_df["Validity right"].astype("string").values
em_type_arr = tobii_df["Eye movement type"].astype("string").values
gaze_valid = (
    (val_left == "Valid")
    & (val_right == "Valid")
    & ~np.isnan(gx_video)
    & ~np.isnan(gy_video)
    & (em_type_arr != "EyesNotFound")
)
print(f"gaze_valid:        {gaze_valid.mean():.1%}  ({gaze_valid.sum()}/{len(gaze_valid)})")

# Project. project_video_to_screen handles NaN H and NaN gaze.
gx_screen, gy_screen = project_video_to_screen(gx_video, gy_video, H_per_sample)
# NaN out anything where either flag is false (spec: gx_screen=NaN if either is False).
mask_both = gaze_valid & homography_valid
gx_screen = np.where(mask_both, gx_screen, np.nan)
gy_screen = np.where(mask_both, gy_screen, np.nan)

on_screen = is_on_screen(gx_screen, gy_screen, screen_w_px=SCREEN_W_PX, screen_h_px=SCREEN_H_PX)
print(f"on_screen:         {on_screen.mean():.1%}  ({on_screen.sum()}/{len(on_screen)})")

gx_canvas, gy_canvas = screen_to_canvas(
    gx_screen, gy_screen, url_bar_h_px=URL_BAR_H_PX, canvas_x_pad_px=CANVAS_X_PAD_PX
)

# %%
def _fillna_int64(arr: pd.Series) -> np.ndarray:
    # Tobii exports leave em_type_index / duration NaN on EyesNotFound rows.
    # Fill with -1 so the column is int64 (parquet doesn't love mixed dtypes).
    return arr.fillna(-1).astype(np.int64).values


gaze_per_sample = pd.DataFrame({
    "tobii_ts_us": ts_us,
    "behavior_t_ms": behavior_t_ms,
    "video_frame_frac": video_frame_frac,
    "gx_video": gx_video,
    "gy_video": gy_video,
    "gx_screen": gx_screen,
    "gy_screen": gy_screen,
    "gx_canvas": gx_canvas,
    "gy_canvas": gy_canvas,
    "gaze_valid": gaze_valid,
    "homography_valid": homography_valid,
    "on_screen": on_screen,
    "em_type": tobii_df["Eye movement type"].astype("string").fillna("Unclassified").values,
    "em_type_index": _fillna_int64(tobii_df["Eye movement type index"]),
    "em_duration_ms": _fillna_int64(tobii_df["Eye movement event duration"]),
    "fixation_x_video": tobii_df["Fixation point X"].values.astype(np.float64),
    "fixation_y_video": tobii_df["Fixation point Y"].values.astype(np.float64),
})
gaze_per_sample.to_parquet(out_dir / "gaze_per_sample.parquet", index=False)
print(f"\nSaved gaze_per_sample.parquet  ({len(gaze_per_sample)} rows)")

# %% [markdown]
# ## Sanity prints

# %%
em_counts = gaze_per_sample.em_type.value_counts(dropna=False)
print("\nEye movement type counts:")
for k, v in em_counts.items():
    print(f"  {k:>15}  {v:>8}")

n_fix = int((gaze_per_sample.em_type == "Fixation").sum())
assert n_fix > 0, f"No Fixation rows found in Tobii TSV ({tsv_path}) — bailing out."

# Verify monotonicity of em_type_index within each em_type segment.
em_seg = gaze_per_sample.em_type.values
em_idx = gaze_per_sample.em_type_index.values
_bad_mono = 0
for t in ["Fixation", "Saccade"]:
    mask = em_seg == t
    if not mask.any():
        continue
    seg_idx = em_idx[mask]
    if not (np.diff(seg_idx) >= 0).all():
        _bad_mono += 1
print(f"em_type_index monotonicity check: {2 - _bad_mono}/2 em_types pass")

# %% [markdown]
# ## Collapse fixation events

# %%
fix_rows = gaze_per_sample[gaze_per_sample.em_type == "Fixation"].copy()
grouped = fix_rows.groupby("em_type_index", sort=True)


def _safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) > 0 else np.nan


def _agg_event(g: pd.DataFrame) -> dict:
    gv = g[g.gaze_valid]
    both = g[g.gaze_valid & g.homography_valid]
    return {
        "event_idx": int(g.em_type_index.iloc[0]),
        "start_behavior_t_ms": float(g.behavior_t_ms.min()),
        "end_behavior_t_ms": float(g.behavior_t_ms.max()),
        "duration_ms": int(g.em_duration_ms.iloc[0]),
        "n_samples": int(len(g)),
        "centroid_video_x": _safe_mean(gv.gx_video),
        "centroid_video_y": _safe_mean(gv.gy_video),
        "centroid_screen_x": _safe_mean(both.gx_screen),
        "centroid_screen_y": _safe_mean(both.gy_screen),
        "centroid_canvas_x": _safe_mean(both.gx_canvas),
        "centroid_canvas_y": _safe_mean(both.gy_canvas),
        "frac_homography_valid": float(g.homography_valid.mean()),
        "frac_on_screen": float(g.on_screen.mean()),
    }


fix_events = pd.DataFrame([_agg_event(g) for _, g in grouped])
assert len(fix_events) > 0, "Zero fixation events after collapsing — check Tobii classifier output."
fix_events.to_parquet(out_dir / "fixation_events.parquet", index=False)

n_high_homog = int((fix_events.frac_homography_valid > 0.8).sum())
print(f"\nFixation events: {len(fix_events)} total  "
      f"mean duration {fix_events.duration_ms.mean():.0f} ms  "
      f"frac_homog>0.8: {n_high_homog}/{len(fix_events)} "
      f"({n_high_homog/len(fix_events):.1%})")

# %% [markdown]
# ## Validation metric — click → pre-click gaze distance

# %%
clicks = trials_df[trials_df.response_time.notna()].copy()
print(f"\nValidating against {len(clicks)} click events.")

# Only consider gaze samples that are valid AND on-screen.
valid_mask = gaze_per_sample.gaze_valid & gaze_per_sample.homography_valid & gaze_per_sample.on_screen
g_valid = gaze_per_sample[valid_mask][["behavior_t_ms", "gx_canvas", "gy_canvas"]].values

distances = []
n_with_samples = 0
for _, click in clicks.iterrows():
    rt = float(click.response_time)
    lo, hi = rt - 300, rt - 50
    in_win = g_valid[(g_valid[:, 0] >= lo) & (g_valid[:, 0] <= hi)]
    if len(in_win) < 3:
        continue
    n_with_samples += 1
    mean_gx = in_win[:, 1].mean()
    mean_gy = in_win[:, 2].mean()
    click_canvas_x = float(click.response_x) * CANVAS_W_PX
    click_canvas_y = float(click.response_y) * (CANVAS_H_PX / MAX_Y_COORD)
    distances.append(np.hypot(mean_gx - click_canvas_x, mean_gy - click_canvas_y))

distances = np.array(distances)
if len(distances) > 0:
    med = float(np.median(distances))
    q1, q3 = np.percentile(distances, [25, 75])
    p90 = float(np.percentile(distances, 90))
    print(f"  N clicks with ≥3 valid samples in [click-300, click-50] ms: "
          f"{n_with_samples}/{len(clicks)}")
    print(f"  median: {med:.1f} canvas-px  "
          f"IQR=[{q1:.1f}, {q3:.1f}]  P90={p90:.1f}")
    pass_flag = med < 250
    print(f"  PASS/FAIL: {'PASS' if pass_flag else 'FAIL'}  "
          f"(threshold: median < 250 canvas-px)")
else:
    med = np.nan
    pass_flag = False
    print("  [WARN] No clicks had ≥3 valid samples in the window.")

# %% [markdown]
# ## Visualization 1 — coverage timeline + accuracy histogram

# %%
fig, (ax_cov, ax_hist) = plt.subplots(1, 2, figsize=(13, 4))

# (a) Stacked rates over time, 10 s bins.
bin_s = 10.0
last_t = float(gaze_per_sample.behavior_t_ms.max() - intercept_ms) / 1000  # video-time s
bins = np.arange(0, last_t + bin_s, bin_s)
video_t_s = gaze_per_sample.tobii_ts_us.values / 1e6
which_bin = np.digitize(video_t_s, bins) - 1
_keep = (which_bin >= 0) & (which_bin < len(bins) - 1)
flags = {
    "gaze_valid": gaze_per_sample.gaze_valid.values,
    "homography_valid": gaze_per_sample.homography_valid.values,
    "on_screen": gaze_per_sample.on_screen.values,
}
bin_rates = {}
n_per_bin = np.zeros(len(bins) - 1, dtype=np.int64)
for i in range(len(bins) - 1):
    sel = _keep & (which_bin == i)
    n = int(sel.sum())
    n_per_bin[i] = n
for name, arr in flags.items():
    rates = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if n_per_bin[i] > 0:
            sel = _keep & (which_bin == i)
            rates[i] = arr[sel].mean()
    bin_rates[name] = rates
bin_mid = (bins[:-1] + bins[1:]) / 2
for name, color in [("gaze_valid", "#4488ff"),
                    ("homography_valid", "#44cc66"),
                    ("on_screen", "#ff8844")]:
    ax_cov.plot(bin_mid, bin_rates[name], color=color, lw=1.5, label=name)
ax_cov.set_xlabel("video time (s)")
ax_cov.set_ylabel("rate")
ax_cov.set_ylim(-0.02, 1.05)
ax_cov.set_title(f"Per-bin flag rates (bin={bin_s:.0f} s)")
ax_cov.legend(loc="lower right", fontsize=8)
ax_cov.grid(alpha=0.3)

# (b) Click→gaze distance histogram.
if len(distances) > 0:
    ax_hist.hist(distances, bins=30, edgecolor="white", color="#4488ff")
    ax_hist.axvline(med, color="red", lw=1.5, label=f"median={med:.0f} px")
    ax_hist.axvline(250, color="black", lw=1, linestyle="--", label="250 px threshold")
    ax_hist.set_xlabel("click→gaze distance (canvas px)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Click→gaze validation (N={len(distances)} clicks)")
    ax_hist.legend(fontsize=8)
else:
    ax_hist.text(0.5, 0.5, "no validation samples", ha="center", va="center")
plt.tight_layout()
fig.savefig(out_dir / "gaze_coverage_and_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved gaze_coverage_and_accuracy.png")

# %% [markdown]
# ## Visualization 2 — canvas heatmap, 3 phases

# %%
# Phase assignment per gaze sample: pre-reveal / reveal→click / post-click.
# Iterate trials in trial_idx order; reveal_time and response_time are behavior-ms.
# - pre-reveal: samples BEFORE the first reveal of each trial (trial onset, tpt=0)
# - reveal→click: between reveal and click of the SAME tpt (clicks only)
# - post-click: after each click, before the next reveal in that trial OR
#               before the next trial's first reveal

phase_of_sample = np.full(len(gaze_per_sample), "", dtype=object)
beh_t = gaze_per_sample.behavior_t_ms.values

# Sort trials by reveal_time so we can assign phases in temporal order.
all_trial_events = trials_df.sort_values(["trial_idx", "tpt"]).reset_index(drop=True)
trial_starts = (
    all_trial_events.groupby("trial_idx")["reveal_time"].min().sort_values()
)
# pre-reveal: samples between previous-trial's last event and this trial's first reveal
prev_end = 0.0
for tidx in trial_starts.index:
    first_reveal = float(trial_starts[tidx])
    mask = (beh_t >= prev_end) & (beh_t < first_reveal)
    phase_of_sample[mask] = "pre-reveal"
    # Mark end of this trial as max event time
    trial_evts = all_trial_events[all_trial_events.trial_idx == tidx]
    prev_end = float(trial_evts[["reveal_time", "response_time"]].max().max())

# reveal→click for each tpt with a click; post-click between click and next reveal
for _, row in all_trial_events.iterrows():
    rev = float(row.reveal_time) if pd.notna(row.reveal_time) else None
    cli = float(row.response_time) if pd.notna(row.response_time) else None
    if rev is None or cli is None:
        continue
    if cli <= rev:
        continue
    mask = (beh_t >= rev) & (beh_t <= cli)
    phase_of_sample[mask] = "reveal-to-click"

    # post-click: from click to next event (next tpt reveal in the same trial,
    # or the next trial's first reveal).
    next_evts = all_trial_events[
        (all_trial_events.trial_idx > row.trial_idx)
        | ((all_trial_events.trial_idx == row.trial_idx) & (all_trial_events.tpt > row.tpt))
    ]
    if len(next_evts) > 0:
        next_t = float(next_evts.reveal_time.dropna().min())
    else:
        next_t = float(gaze_per_sample.behavior_t_ms.max())
    mask = (beh_t > cli) & (beh_t < next_t)
    phase_of_sample[mask] = "post-click"

phase_counts = pd.Series(phase_of_sample).value_counts(dropna=False)
print("\nSample counts by phase:")
print(phase_counts)

# Per-phase canvas-pixel heatmap (use fixation centroids — one point per fixation event,
# not raw per-sample, so the heatmap reflects where the eye dwelled).
# Tag each fixation event by phase: use start_behavior_t_ms.
fix_phase = np.full(len(fix_events), "", dtype=object)
for phase_label in ["pre-reveal", "reveal-to-click", "post-click"]:
    sample_in_phase = phase_of_sample == phase_label
    # Find behavior-ms ranges covered by this phase
    # (we just bucket fixation start times via per-sample phase via nearest-time lookup).
ft = fix_events.start_behavior_t_ms.values
# Order phase_of_sample by beh_t for fast lookup
order = np.argsort(beh_t)
beh_t_sorted = beh_t[order]
phase_sorted = np.asarray(phase_of_sample)[order]
idx = np.clip(np.searchsorted(beh_t_sorted, ft) - 1, 0, len(beh_t_sorted) - 1)
fix_phase = phase_sorted[idx]

fig2, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
phase_titles = ["pre-reveal", "reveal-to-click", "post-click"]
# Compute shared color scale across panels.
bin_edges_x = np.linspace(0, CANVAS_W_PX, 60)
bin_edges_y = np.linspace(0, CANVAS_H_PX, 45)
all_hists = []
for ph in phase_titles:
    sub = fix_events[fix_phase == ph]
    sub = sub.dropna(subset=["centroid_canvas_x", "centroid_canvas_y"])
    h, _, _ = np.histogram2d(
        sub.centroid_canvas_x.values, sub.centroid_canvas_y.values,
        bins=[bin_edges_x, bin_edges_y],
    )
    all_hists.append(h)
vmax = max(h.max() for h in all_hists) if any(h.max() > 0 for h in all_hists) else 1
norm = LogNorm(vmin=0.5, vmax=max(vmax, 1))

for ax, ph, h in zip(axes, phase_titles, all_hists):
    # Faint canvas backdrop (full rect + usable-y rect)
    ax.add_patch(plt.Rectangle((0, 0), CANVAS_W_PX, CANVAS_H_PX,
                                fill=False, edgecolor="#888888", lw=0.8))
    ax.add_patch(plt.Rectangle((0, 0), CANVAS_W_PX, CANVAS_USABLE_H_PX,
                                fill=False, edgecolor="#888888", lw=0.6, linestyle=":"))
    im = ax.imshow(
        h.T, origin="upper",
        extent=[0, CANVAS_W_PX, CANVAS_H_PX, 0],
        cmap="viridis", norm=norm, aspect="equal", interpolation="nearest",
    )
    sub_count = int(h.sum())
    ax.set_title(f"{ph}  (N={sub_count} fixations)", fontsize=10)
    ax.set_xlim(-50, CANVAS_W_PX + 50)
    ax.set_ylim(CANVAS_H_PX + 50, -50)
    ax.set_xlabel("canvas x (px)")
axes[0].set_ylabel("canvas y (px)")
fig2.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="fixation count")
fig2.suptitle("Fixation-centroid heatmap by trial phase", y=1.02)
fig2.savefig(out_dir / "gaze_canvas_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved gaze_canvas_heatmap.png")

# %% [markdown]
# ## Visualization 3 — pre-click gaze trajectories (4×5 grid)

# %%
rng = np.random.default_rng(seed=0)

# Find clicks with ≥10 gaze_valid & homography_valid samples in [rt-1000, rt].
# (We don't restrict to on_screen here — let trajectories show off-screen
# wandering, but the validation uses on-screen only.)
g_traj = gaze_per_sample[gaze_per_sample.gaze_valid & gaze_per_sample.homography_valid].copy()
g_traj_arr = g_traj[["behavior_t_ms", "gx_canvas", "gy_canvas"]].values

eligible = []
for _, click in clicks.iterrows():
    rt = float(click.response_time)
    in_win = g_traj_arr[(g_traj_arr[:, 0] >= rt - 1000) & (g_traj_arr[:, 0] <= rt)]
    if len(in_win) >= 10:
        eligible.append((click, in_win))
print(f"Pre-click trajectory candidates with ≥10 valid samples in [rt-1s, rt]: "
      f"{len(eligible)}/{len(clicks)}")

n_sub = min(20, len(eligible))
chosen_idx = rng.choice(len(eligible), size=n_sub, replace=False) if n_sub > 0 else []

fig3, axes = plt.subplots(4, 5, figsize=(18, 14.5))
axes_flat = axes.flatten()
for ax in axes_flat:
    ax.set_xticks([])
    ax.set_yticks([])

for plot_i, sel_i in enumerate(chosen_idx):
    ax = axes_flat[plot_i]
    click, in_win = eligible[sel_i]
    rt = float(click.response_time)
    trial_idx = int(click.trial_idx)
    tpt = int(click.tpt)

    # Canvas backdrop.
    ax.add_patch(plt.Rectangle((0, 0), CANVAS_W_PX, CANVAS_H_PX,
                                fill=False, edgecolor="#888888", lw=0.8))
    ax.add_patch(plt.Rectangle((0, 0), CANVAS_W_PX, CANVAS_USABLE_H_PX,
                                fill=False, edgecolor="#aaaaaa", lw=0.5, linestyle=":"))

    # Other previously-revealed dots in the same trial (small grey).
    same_trial = trials_df[(trials_df.trial_idx == trial_idx) & (trials_df.tpt < tpt)]
    for _, prev in same_trial.iterrows():
        px = float(prev.true_x) * CANVAS_W_PX
        py = float(prev.true_y) * (CANVAS_H_PX / MAX_Y_COORD)
        ax.scatter(px, py, s=30, color="#cccccc", edgecolor="#888888", lw=0.5, zorder=2)

    # Clicked dot — true position (large yellow).
    tx = float(click.true_x) * CANVAS_W_PX
    ty = float(click.true_y) * (CANVAS_H_PX / MAX_Y_COORD)
    ax.scatter(tx, ty, s=180, color="#ffe44a", edgecolor="#cc9900", lw=1.5, zorder=4)

    # Gaze trajectory: time-colored (t = -1s blue → t = click yellow).
    rel_t = (in_win[:, 0] - rt) / 1000  # in seconds, ranges [-1, 0]
    ax.scatter(in_win[:, 1], in_win[:, 2], c=rel_t, cmap="viridis",
               s=20, vmin=-1, vmax=0, zorder=3)

    # Click location (smaller blue dot, distinct from true).
    ax.scatter(float(click.response_x) * CANVAS_W_PX,
               float(click.response_y) * (CANVAS_H_PX / MAX_Y_COORD),
               marker="x", s=80, color="#0066cc", lw=2, zorder=5)

    ax.set_xlim(-50, CANVAS_W_PX + 50)
    ax.set_ylim(CANVAS_H_PX + 50, -50)
    ax.set_aspect("equal")
    ax.set_title(f"trial={trial_idx} tpt={tpt}", fontsize=9)

# Hide unused subplots
for j in range(n_sub, len(axes_flat)):
    axes_flat[j].axis("off")

fig3.suptitle(
    f"Pre-click gaze trajectories (random {n_sub} clicks, seed=0)\n"
    "gold = true dot, blue×= click, dots = gaze t=-1s (blue) → t=0 (yellow)",
    fontsize=11, y=1.0,
)
plt.tight_layout()
fig3.savefig(out_dir / "pre_click_gaze_trajectories.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved pre_click_gaze_trajectories.png  ({n_sub} subplots)")

# %% [markdown]
# ## Visualization 4 — saccade PSTH around reveal

# %%
# First-sample-of-event saccades = where em_type=="Saccade" and em_type_index changed
sac_mask = gaze_per_sample.em_type == "Saccade"
sac_df = gaze_per_sample[sac_mask].reset_index(drop=True)
# First sample of each saccade event:
is_first = sac_df.em_type_index.diff().fillna(1) != 0
sac_onsets = sac_df.loc[is_first.values, "behavior_t_ms"].values
print(f"\nSaccade onsets: {len(sac_onsets)}")

# Build Δt list across reveals.
reveals = trials_df.reveal_time.dropna().values
dts = []
for rt in reveals:
    delta = sac_onsets - rt
    in_win = delta[(delta >= -2000) & (delta <= 2000)]
    dts.extend(in_win)
dts = np.array(dts)
print(f"Saccade Δt (within ±2s of any reveal): {len(dts)}")

fig4, ax = plt.subplots(figsize=(9, 4))
ax.hist(dts, bins=np.arange(-2000, 2050, 50), edgecolor="white", color="#4488ff")
ax.axvline(0, color="red", lw=1.5, label="reveal onset")
ax.set_xlabel("Δt = saccade onset − reveal (ms)")
ax.set_ylabel("count")
ax.set_title(f"Saccade onsets relative to dot reveal (N={len(dts)})")
ax.legend()
fig4.savefig(out_dir / "saccade_psth_around_reveal.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"Saved saccade_psth_around_reveal.png")

# %% [markdown]
# ## Visualization 5 — gaze-on-screen overlay video (30 s clip)

# %%
def _pt(H_mat: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    v = H_mat @ np.array([xy[0], xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

clip_start_frame = int(round(CLIP_START_S * fps))
clip_n_frames = int(round(CLIP_DURATION_S * fps))
clip_end_frame = clip_start_frame + clip_n_frames

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_mp4 = str(out_dir / "gaze_screen_overlay.mp4")
writer = cv2.VideoWriter(out_mp4, fourcc, fps, (frame_w, frame_h))

screen_corners_xy = [
    (0.0, 0.0), (SCREEN_W_PX, 0.0),
    (SCREEN_W_PX, SCREEN_H_PX), (0.0, SCREEN_H_PX),
]

# Pre-sort beh_t for nearest-gaze lookup.
ts_us_arr = gaze_per_sample.tobii_ts_us.values
beh_t_arr = gaze_per_sample.behavior_t_ms.values
gx_v = gaze_per_sample.gx_video.values
gy_v = gaze_per_sample.gy_video.values
gv_flag = gaze_per_sample.gaze_valid.values

cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
n_written = 0
for fidx in tqdm(range(clip_start_frame, clip_end_frame), desc="overlay"):
    ok, frame = cap.read()
    if not ok:
        break

    frame_video_t_s = fidx / fps
    frame_beh_t_ms = frame_video_t_s * slope_ms_per_s + intercept_ms

    # Active trial: most recent reveal at or before this beh_t (any tpt).
    prior = trials_df[trials_df.reveal_time.notna() & (trials_df.reveal_time <= frame_beh_t_ms)]

    H_row = per_frame_df[per_frame_df.frame_idx == fidx]
    has_h = (len(H_row) > 0
             and not pd.isna(H_row.iloc[0].h00))
    if has_h:
        r = H_row.iloc[0]
        H_mat = np.array([
            [r.h00, r.h01, r.h02],
            [r.h10, r.h11, r.h12],
            [r.h20, r.h21, r.h22],
        ])
        # Screen rect (yellow polygon)
        pts = np.array([_pt(H_mat, c) for c in screen_corners_xy], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Active-trial previously-revealed dots (cyan)
        if len(prior) > 0:
            active_trial = int(prior.iloc[-1].trial_idx)
            same_trial = prior[prior.trial_idx == active_trial]
            for _, dot in same_trial.iterrows():
                sx, sy = behavior_to_screen(
                    float(dot.true_x), float(dot.true_y),
                    url_bar_h_px=URL_BAR_H_PX, canvas_x_pad_px=CANVAS_X_PAD_PX,
                    max_y_coord=MAX_Y_COORD,
                )
                fx, fy = _pt(H_mat, (sx, sy))
                cv2.circle(frame, (int(round(fx)), int(round(fy))),
                           12, (255, 255, 0), -1, cv2.LINE_AA)

    # Tobii gaze nearest to this frame's beh_t.
    i_near = int(np.argmin(np.abs(beh_t_arr - frame_beh_t_ms)))
    if not np.isnan(gx_v[i_near]) and not np.isnan(gy_v[i_near]):
        gxi, gyi = int(round(gx_v[i_near])), int(round(gy_v[i_near]))
        cv2.circle(frame, (gxi, gyi), 8, (255, 255, 255), -1, cv2.LINE_AA)
        if not gv_flag[i_near]:
            cv2.circle(frame, (gxi, gyi), 11, (0, 0, 255), 2, cv2.LINE_AA)

    caption = f"f={fidx}  beh_t={frame_beh_t_ms:.0f} ms"
    for thick, color in [(2, (255, 255, 255)), (1, (0, 0, 0))]:
        cv2.putText(frame, caption, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)

    writer.write(frame)
    n_written += 1

cap.release()
writer.release()
print(f"Saved gaze_screen_overlay.mp4  ({n_written} frames @ {fps:.1f} fps)")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 60)
print(f"Phase 2 — extract_gaze_fixations  (subject={subject})")
print("=" * 60)
print(f"Tobii samples (Eye Tracker rows): {len(gaze_per_sample)}")
print(f"  gaze_valid:        {gaze_per_sample.gaze_valid.mean():.1%}")
print(f"  homography_valid:  {gaze_per_sample.homography_valid.mean():.1%}")
print(f"  on_screen:         {gaze_per_sample.on_screen.mean():.1%}")
print(f"Fixation events: {len(fix_events)}")
print(f"  mean duration:     {fix_events.duration_ms.mean():.0f} ms")
print(f"  frac_homog>0.8:    {n_high_homog}/{len(fix_events)} "
      f"({n_high_homog/len(fix_events):.1%})")
print(f"Click→gaze validation:")
if len(distances) > 0:
    print(f"  median:            {med:.1f} canvas-px  (threshold 250 px)")
    print(f"  status:            {'PASS' if pass_flag else 'FAIL'}")
else:
    print(f"  status:            NO DATA")
print("=" * 60)
