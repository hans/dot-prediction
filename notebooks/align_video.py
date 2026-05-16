# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Video ↔ behavior alignment (interactive)
#
# Maps Tobii scene-camera frames to behavior TSV rows. Stars detected on the
# iPad screen act as visual fiducials.
#
# **Run this notebook interactively** — the anchor picker uses ipywidgets and
# does not work under headless `ploomber_engine`. Outputs (anchors JSON +
# `trials_with_video.parquet`) become inputs for downstream Snakemake rules.
#
# Workflow:
# 1. Load behavior + video; derive wall-clock prior offset from Tobii TSV
#    header (minute-precision).
# 2. Coarse-scan the video for detected stars (cached parquet).
# 3. Scrub frames in the picker. Detected blobs are drawn green; every nearby
#    behavior row's expected on-screen position is overlaid (coloured cross
#    labelled `trial.tpt`). Initially overlays are coarse (wall-clock prior);
#    they tighten as anchors accumulate.
# 4. For each anchor: scrub to the first frame the star appears, choose the
#    matching `(trial_idx, tpt)` from the dropdown, hit **Add anchor**.
# 5. After ≥ 2 anchors a slope+intercept fit is shown. Saving writes
#    `video_alignment.json` and `trials_with_video.parquet`.

# %% tags=["parameters"]
video_path = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4"
behavior_path = "/Users/jon/Projects/dot-prediction/data/EC347/behavior/data.csv"
tobii_tsv_path = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/EC347_B16_tobii.tsv"
anchors_path = "results/EC347/video_alignment_anchors.json"
alignment_path = "results/EC347/video_alignment.json"
trials_out = "results/EC347/trials_with_video.parquet"
scan_cache = "results/EC347/video_star_scan.parquet"
scan_step_s = 0.5
overlay_window_s = 120.0
# Minimum video-time span (s) before polyfit is allowed to estimate slope.
# With anchors clustered close together, frame quantization (±40 ms at 25 fps)
# swamps any drift signal and a free slope extrapolates wildly. Below this
# threshold the picker locks slope = 1000 ms/s and fits intercept only.
min_span_for_slope_fit_s = 300.0

# %%
import datetime
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..") / "src"))

import cv2
import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from screen_detection import detect_corners
from star_detector import detect_stars

# %% [markdown]
# ## Behavior
# Drop trial 0 (the seq_id=`example_line` demo). On EC347 the Tobii recording
# starts *after* the entire example trial finished, so none of trial 0's
# reveals are in the video and including them in the picker creates a
# misidentification trap.

# %%
trials = (
    pd.read_csv(behavior_path)
    .sort_values(["trial_idx", "tpt"])
    .reset_index(drop=True)
)
trials = trials[trials.trial_idx >= 1].reset_index(drop=True)
expt_start_ms = int(trials.expt_start_time.iloc[0])
print(
    f"{len(trials)} behavior rows across trials "
    f"{trials.trial_idx.min()}–{trials.trial_idx.max()}"
)

# %% [markdown]
# ## Video & wall-clock prior
# Parse Tobii recording-start UTC from the TSV header. Minute-precision only,
# so the resulting prior offset is good to ~30 s — enough to seed the picker.

# %%
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_frames} frames @ {fps:.3f} fps → {n_frames/fps:.1f} s")

tobii_hdr = pd.read_csv(tobii_tsv_path, sep="\t", nrows=1)
tobii_start_dt = datetime.datetime.strptime(
    f"{tobii_hdr['Recording date UTC'].iloc[0]} "
    f"{tobii_hdr['Recording start time UTC'].iloc[0]}",
    "%m/%d/%Y %H:%M:%S.%f",
).replace(tzinfo=datetime.timezone.utc)
tobii_start_ms = int(tobii_start_dt.timestamp() * 1000)
# The Tobii TSV's "Recording start time UTC" is not reliable on EC347 (likely
# file-save time). On this subject the recording started shortly after the
# example trial (trial 0, seq_id=example_line) ended — trial 0's last reveal
# is at expt_t=282,296 ms and trial 1's first reveal is at 329,680 ms, so
# video_t=0 falls in that gap. Empirically a star at video_t≈26 s matches
# trial 1 tpt 0, putting the intercept at ~303,300 ms.
prior_intercept_ms = 303_300  # ms since expt_start at video_t = 0 (EC347)
print(f"Tobii start UTC (from TSV header, may be wrong): {tobii_start_dt}")
print(f"Prior: reveal_time_ms ≈ video_t_s * 1000 + {prior_intercept_ms}")

# %% [markdown]
# ## Coarse star scan (cached)
# Reads every frame and runs `detect_stars` every `scan_step_s`. The detector
# is unreliable in this dark video, but a hit is still a useful navigation
# cue — clumps mark plausible trial intervals.

# %%
def coarse_scan():
    step_frames = max(1, int(round(scan_step_s * fps)))
    records = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step_frames == 0:
            blobs = detect_stars(frame)
            records.append({
                "frame_idx": fi,
                "video_t": fi / fps,
                "n_blobs": len(blobs),
                "blob_x": blobs[0][0] if blobs else np.nan,
                "blob_y": blobs[0][1] if blobs else np.nan,
            })
        fi += 1
    return pd.DataFrame(records)


scan_cache_p = Path(scan_cache)
if scan_cache_p.exists():
    scan = pd.read_parquet(scan_cache_p)
    print(f"Loaded cached scan ({len(scan)} samples) from {scan_cache_p}")
else:
    print(f"Running coarse scan (every {scan_step_s}s) — may take a few minutes…")
    scan = coarse_scan()
    scan_cache_p.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(scan_cache_p, index=False)
    print(f"Saved {scan_cache_p}")

hit_frames = scan.loc[scan.n_blobs > 0, "frame_idx"].to_numpy()
print(f"Frames with ≥1 blob: {len(hit_frames)} / {len(scan)}")

# %%
f, ax = plt.subplots(figsize=(14, 2))
ax.vlines(scan.loc[scan.n_blobs > 0, "video_t"], 0, 1, linewidth=0.6, alpha=0.6)
ax.set_yticks([])
ax.set_xlabel("video time (s)")
ax.set_title("Coarse scan: video times with ≥1 detected blob")
plt.tight_layout()

# %% [markdown]
# ## Anchor picker (interactive)

# %%
SCREEN_CORNERS = np.array(
    [[0, 0], [2388, 0], [2388, 1668], [0, 1668]], dtype=np.float32
)
OVERLAY_CMAP = plt.colormaps.get_cmap("tab10")

anchors_p = Path(anchors_path)
anchors_p.parent.mkdir(parents=True, exist_ok=True)
anchors = json.loads(anchors_p.read_text()) if anchors_p.exists() else []
print(f"Loaded {len(anchors)} anchor(s) from {anchors_path}")


def save_anchors():
    anchors_p.write_text(json.dumps(anchors, indent=2))


def current_fit():
    """Return (slope_ms_per_s, intercept_ms, residuals_ms).

    With no anchors, falls back to the wall-clock prior. With anchors whose
    video-time span is below ``min_span_for_slope_fit_s`` the slope is locked
    to 1000 ms/s and the intercept is fit as the mean residual — this avoids
    runaway extrapolation when all anchors sit in a small window. Past that
    span threshold, slope and intercept are jointly fit.
    """
    if not anchors:
        return 1000.0, float(prior_intercept_ms), np.array([])
    vt = np.array([a["video_t"] for a in anchors])
    et = np.array([a["expt_t_ms"] for a in anchors])
    span = float(vt.max() - vt.min())
    if len(anchors) >= 2 and span >= min_span_for_slope_fit_s:
        slope, intercept = np.polyfit(vt, et, 1)
    else:
        slope = 1000.0
        intercept = float(np.mean(et - 1000.0 * vt))
    residuals = et - (slope * vt + intercept)
    return float(slope), float(intercept), residuals


def video_to_expt_ms(vt_s, slope, intercept):
    return slope * vt_s + intercept


def expt_ms_to_video(et_ms, slope, intercept):
    return (et_ms - intercept) / slope


def read_frame(frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    return frame if ret else None


def render_frame(frame_idx, show_overlay=True):
    frame = read_frame(frame_idx)
    if frame is None:
        return None, None
    vis = frame.copy()
    corners = detect_corners(frame)
    blobs = detect_stars(frame)
    if corners is not None:
        cv2.polylines(
            vis, [corners.astype(np.int32).reshape(-1, 1, 2)],
            True, (0, 200, 0), 2,
        )
    for bx, by, br in blobs:
        cv2.circle(vis, (int(bx), int(by)), max(int(br), 8), (0, 255, 0), 2)
    slope, intercept, _ = current_fit()
    video_t = frame_idx / fps
    expt_t_now = video_to_expt_ms(video_t, slope, intercept)
    window_ms = overlay_window_s * 1000.0
    nearby = trials[
        (trials.reveal_time >= expt_t_now - window_ms)
        & (trials.reveal_time <= expt_t_now + window_ms)
    ].reset_index(drop=True)
    if show_overlay and corners is not None and len(nearby):
        H, _ = cv2.findHomography(SCREEN_CORNERS, corners)
        for i, row in nearby.iterrows():
            sx, sy = row.true_x * 2388.0, row.true_y * 1668.0
            p = (H @ np.array([sx, sy, 1.0])).reshape(3)
            fx, fy = p[0] / p[2], p[1] / p[2]
            rgba = OVERLAY_CMAP(i % 10)
            col = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            cv2.drawMarker(
                vis, (int(fx), int(fy)), col,
                markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2,
            )
            cv2.putText(
                vis, f"{int(row.trial_idx)}.{int(row.tpt)}",
                (int(fx) + 10, int(fy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
            )
    cv2.putText(
        vis,
        f"frame={frame_idx}  t={video_t:.3f}s  blobs={len(blobs)}  "
        f"corners={'OK' if corners is not None else 'FAIL'}",
        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    # Show distinct nearby trial names (seq_id) so the user can verify trial
    # indexing without relying on the (possibly broken) overlay crosses.
    if len(nearby):
        labels = [
            f"tr{int(t)}={s}"
            for t, s in nearby[["trial_idx", "seq_id"]].drop_duplicates().itertuples(index=False)
        ]
        cv2.putText(
            vis, " | ".join(labels),
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2,
        )
    return vis, nearby


def jpeg_bytes(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


# Initial frame — wall-clock prior for trial 1 first reveal.
init_frame = int(
    np.clip(expt_ms_to_video(trials.reveal_time.iloc[0], 1000.0, prior_intercept_ms) * fps,
            0, n_frames - 1)
)

frame_slider = W.IntSlider(
    value=init_frame, min=0, max=n_frames - 1, step=1,
    description="frame", continuous_update=False,
    layout=W.Layout(width="80%"),
)
fine = W.HBox([
    W.Button(description="-30 fr"),
    W.Button(description="-10 fr"),
    W.Button(description="-1 fr"),
    W.Button(description="+1 fr"),
    W.Button(description="+10 fr"),
    W.Button(description="+30 fr"),
    W.Button(description="↪ next hit"),
    W.Button(description="↩ prev hit"),
])
row_dropdown = W.Dropdown(
    description="row", layout=W.Layout(width="60%"),
)
add_btn = W.Button(description="Add anchor", button_style="success")
remove_btn = W.Button(description="Remove last", button_style="warning")
overlay_toggle = W.Checkbox(value=True, description="show predicted-star overlay")
status_html = W.HTML()
img_widget = W.Image(format="jpeg", layout=W.Layout(width="1000px"))


def refresh():
    vis, nearby = render_frame(frame_slider.value, show_overlay=overlay_toggle.value)
    if vis is None:
        status_html.value = "<span style='color:red'>Failed to read frame</span>"
        return
    img_widget.value = jpeg_bytes(vis)
    opts = []
    if nearby is not None:
        for _, row in nearby.iterrows():
            opts.append((
                f"trial {int(row.trial_idx)} [{row.seq_id}] tpt {int(row.tpt)}  "
                f"reveal={int(row.reveal_time)}ms  "
                f"({row.true_x:.3f},{row.true_y:.3f})",
                (int(row.trial_idx), int(row.tpt)),
            ))
    row_dropdown.options = opts
    slope, intercept, residuals = current_fit()
    rms = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else float("nan")
    anchor_rows = "".join(
        f"<tr><td>{i}</td><td>{a['video_t']:.3f}s</td>"
        f"<td>tr {a['trial_idx']}.{a['tpt']}</td>"
        f"<td>{a['expt_t_ms']:.0f}ms</td></tr>"
        for i, a in enumerate(anchors)
    )
    status_html.value = (
        f"<b>{len(anchors)} anchor(s)</b><br>"
        f"fit: <code>expt_ms = {slope:.4f} × video_s + {intercept:.1f}</code><br>"
        f"RMS residual: {rms:.1f} ms"
        + (f"<br><table border=1>{anchor_rows}</table>" if anchors else "")
    )


def _step(n):
    def cb(_):
        frame_slider.value = int(np.clip(frame_slider.value + n, 0, n_frames - 1))
    return cb


def _next_hit(_):
    nxt = hit_frames[hit_frames > frame_slider.value]
    if len(nxt):
        frame_slider.value = int(nxt[0])


def _prev_hit(_):
    prv = hit_frames[hit_frames < frame_slider.value]
    if len(prv):
        frame_slider.value = int(prv[-1])


for btn, delta in zip(fine.children[:6], [-30, -10, -1, 1, 10, 30]):
    btn.on_click(_step(delta))
fine.children[6].on_click(_next_hit)
fine.children[7].on_click(_prev_hit)


def on_add(_):
    sel = row_dropdown.value
    if sel is None:
        status_html.value = "<span style='color:red'>No row selected</span>"
        return
    trial_idx, tpt = sel
    row = trials[(trials.trial_idx == trial_idx) & (trials.tpt == tpt)].iloc[0]
    anchor = {
        "video_t": float(frame_slider.value) / fps,
        "frame_idx": int(frame_slider.value),
        "expt_t_ms": float(row.reveal_time),
        "trial_idx": int(trial_idx),
        "tpt": int(tpt),
    }
    # Replace any prior anchor on the same (trial, tpt).
    anchors[:] = [
        a for a in anchors
        if not (a["trial_idx"] == trial_idx and a["tpt"] == tpt)
    ]
    anchors.append(anchor)
    anchors.sort(key=lambda a: a["video_t"])
    save_anchors()
    refresh()


def on_remove(_):
    if anchors:
        anchors.pop()
        save_anchors()
        refresh()


add_btn.on_click(on_add)
remove_btn.on_click(on_remove)
frame_slider.observe(lambda change: refresh() if change["name"] == "value" else None,
                     names="value")
overlay_toggle.observe(lambda change: refresh() if change["name"] == "value" else None,
                       names="value")

display(W.VBox([
    frame_slider,
    fine,
    img_widget,
    W.HBox([row_dropdown, add_btn, remove_btn, overlay_toggle]),
    status_html,
]))
refresh()

# %% [markdown]
# ## Fit + save
# Linear regression over committed anchors. Slope deviates from 1000 ms/s by
# the iPad↔Tobii clock drift (typically a few hundred ppm at most).

# %%
slope, intercept, residuals = current_fit()
rms = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else None
print(f"slope     = {slope:.6f} ms/s   ({(slope - 1000) / 1000 * 1e6:+.1f} ppm)")
print(f"intercept = {intercept:.2f} ms")
print(f"n_anchors = {len(anchors)}    RMS residual = "
      f"{rms if rms is None else f'{rms:.1f}'} ms")

alignment = {
    "slope_ms_per_s": slope,
    "intercept_ms": intercept,
    "n_anchors": len(anchors),
    "rms_residual_ms": rms,
    "fps": float(fps),
    "expt_start_ms": expt_start_ms,
    "tobii_start_ms": tobii_start_ms,
    "anchors": anchors,
}
Path(alignment_path).parent.mkdir(parents=True, exist_ok=True)
Path(alignment_path).write_text(json.dumps(alignment, indent=2))
print(f"wrote {alignment_path}")

# %%
out = trials.copy()
out["video_t_reveal_s"] = expt_ms_to_video(out.reveal_time.values, slope, intercept)
out["video_t_response_s"] = expt_ms_to_video(out.response_time.values, slope, intercept)
out["video_frame_reveal"] = (out.video_t_reveal_s * fps).round().astype("Int64")
out["video_frame_response"] = (out.video_t_response_s * fps).round().astype("Int64")
Path(trials_out).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(trials_out, index=False)
print(f"wrote {trials_out} ({len(out)} rows)")

# %% [markdown]
# ## QC: predicted star vs detected blobs at sampled reveals
# Magenta tilted-cross = behavior-row projected into frame coords; green
# circles = detector hits. Tight agreement → alignment is sound.

# %%
qc = (out
      .dropna(subset=["video_frame_reveal"])
      .query("0 <= video_frame_reveal < @n_frames")
      .sample(min(8, len(out)), random_state=0))
cols = 4
rows_n = (len(qc) + cols - 1) // cols
fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n))
axes = np.array(axes).flatten()
for i, (_, row) in enumerate(qc.iterrows()):
    fi = int(row.video_frame_reveal)
    frame = read_frame(fi)
    if frame is None:
        axes[i].axis("off")
        continue
    vis = frame.copy()
    corners = detect_corners(frame)
    blobs = detect_stars(frame)
    if corners is not None:
        H, _ = cv2.findHomography(SCREEN_CORNERS, corners)
        p = (H @ np.array([row.true_x * 2388, row.true_y * 1668, 1.0])).reshape(3)
        cv2.drawMarker(
            vis, (int(p[0] / p[2]), int(p[1] / p[2])),
            (255, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=30, thickness=3,
        )
    for bx, by, br in blobs:
        cv2.circle(vis, (int(bx), int(by)), max(int(br), 8), (0, 255, 0), 2)
    axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[i].set_title(
        f"trial {int(row.trial_idx)} tpt {int(row.tpt)}\n"
        f"frame {fi}  t={row.video_t_reveal_s:.2f}s",
        fontsize=9,
    )
    axes[i].axis("off")
for ax in axes[len(qc):]:
    ax.axis("off")
plt.tight_layout()

# %%
cap.release()
