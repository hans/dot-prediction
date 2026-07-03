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
# # Branch comparison: phase1c homography + gaze
#
# Compares two completed pipeline runs (e.g. main vs a feature branch) across
# two levels:
#
# 1. **Homography quality** — from `phase1c_per_frame.parquet` alone. No video
#    needed; can run as soon as `phase1c_homography` finishes in both worktrees.
# 2. **Gaze quality** — from `gaze_per_sample.parquet`. Requires
#    `extract_gaze_fixations` to have completed in both worktrees.
#
# Pass the two result directories as parameters; outputs go to `out_dir`.

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

# Paths to the two results/{subject}/ directories being compared.
dir_a = str(_ROOT / f"results/{subject}")
dir_b = str(_ROOT / f"results/{subject}")   # replace with worktree path

label_a = "main"
label_b = "feat/issue-8"

# trials_with_video.parquet is identical in both worktrees (manual artifact).
trials_path = str(_ROOT / f"results/{subject}/trials_with_video.parquet")

out_dir = str(_ROOT / f"results/{subject}/comparison")

# Canvas geometry — must match config_eyetrack.yaml behavior_canvas values.
SCREEN_W_PX = 2388
SCREEN_H_PX = 1668
URL_BAR_H_PX = 272
CANVAS_X_PAD_PX = 233
MAX_Y_COORD = 0.75

# Pre-click window used by extract_gaze_fixations.
PRECLICK_WINDOW_MS = 500

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir_a = Path(dir_a)
dir_b = Path(dir_b)
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

CANVAS_W_PX = SCREEN_W_PX - 2 * CANVAS_X_PAD_PX   # 1922
CANVAS_H_PX = SCREEN_H_PX - URL_BAR_H_PX           # 1396

# %%
# ---------------------------------------------------------------------------
# Load phase1c parquets
# ---------------------------------------------------------------------------

p_a = pd.read_parquet(dir_a / "phase1c_per_frame.parquet")
p_b = pd.read_parquet(dir_b / "phase1c_per_frame.parquet")

print(f"{label_a}: {len(p_a)} frames, columns: {p_a.columns.tolist()}")
print(f"{label_b}: {len(p_b)} frames, columns: {p_b.columns.tolist()}")

# %%
# ---------------------------------------------------------------------------
# Helper: project a screen point through each frame's H
# ---------------------------------------------------------------------------

def project_col(df, sx, sy):
    """Return frame-px x-coordinate of (sx, sy) projected through each row's H."""
    v0 = df.h00 * sx + df.h01 * sy + df.h02
    v1 = df.h10 * sx + df.h11 * sy + df.h12
    v2 = df.h20 * sx + df.h21 * sy + df.h22
    return v0 / v2


def tr_x(df):
    """Top-right corner (W, 0) projected x — the high-leverage extrapolation."""
    valid = df[df.detection_status.isin(["detected", "interpolated", "extrapolated"])]
    return project_col(valid, float(SCREEN_W_PX), 0.0), valid.frame_idx


# %%
# ---------------------------------------------------------------------------
# Level 1: Homography quality
# ---------------------------------------------------------------------------

def h_summary(df, label):
    n = len(df)
    n_valid = df.detection_status.isin(["detected", "interpolated", "extrapolated"]).sum()
    n_detected = (df.detection_status == "detected").sum()

    tr, _ = tr_x(df)
    outliers_10k = (tr.abs() > 10_000).sum()
    outliers_100k = (tr.abs() > 100_000).sum()
    tr_p99 = tr.abs().quantile(0.99)
    tr_max = tr.abs().max()

    residuals = df.big_star_residual_px.dropna()

    row = {
        "branch": label,
        "total_frames": n,
        "valid_H_frames": n_valid,
        "valid_H_%": round(100 * n_valid / n, 1),
        "detected_frames": n_detected,
        "|TR_x|>10k": outliers_10k,
        "|TR_x|>100k": outliers_100k,
        "|TR_x| p99 (px)": round(tr_p99, 0),
        "|TR_x| max (px)": round(tr_max, 0),
        "big_star_residual median (px)": round(residuals.median(), 2) if len(residuals) else float("nan"),
        "big_star_residual N": len(residuals),
    }

    if "n_anchors_used" in df.columns:
        counts = df.n_anchors_used.value_counts().to_dict()
        det = df[df.detection_status == "detected"]
        five = (det.n_anchors_used == 5).sum()
        row["5-anchor frames"] = five
        row["5-anchor % (of detected)"] = round(100 * five / len(det), 1) if len(det) else float("nan")

    return row


rows = [h_summary(p_a, label_a), h_summary(p_b, label_b)]
h_table = pd.DataFrame(rows).set_index("branch").T
print("\n=== Homography quality ===")
print(h_table.to_string())
h_table.to_csv(out_dir / "h_summary.csv")

# %%
# ---------------------------------------------------------------------------
# Plot: TR_x over time — both branches
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

for ax, df, label in [(axes[0], p_a, label_a), (axes[1], p_b, label_b)]:
    tr, frames = tr_x(df)
    ax.plot(frames, tr, lw=0.4, alpha=0.6, color="steelblue")
    ax.axhline(10_000, color="orange", lw=0.8, ls="--", label="|TR_x|=10k")
    ax.axhline(-10_000, color="orange", lw=0.8, ls="--")
    ax.axhline(100_000, color="red", lw=0.8, ls="--", label="|TR_x|=100k")
    ax.axhline(-100_000, color="red", lw=0.8, ls="--")
    ax.set_ylabel("TR_x (frame px)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_ylim(-200_000, 200_000)

axes[1].set_xlabel("frame index")
fig.suptitle("Top-right corner projection (TR_x) — high-leverage extrapolation")
fig.tight_layout()
fig.savefig(out_dir / "tr_x_over_time.png", dpi=120)
plt.close(fig)
print("Saved tr_x_over_time.png")

# %%
# ---------------------------------------------------------------------------
# Plot: TR_x outlier histogram overlay
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
for df, label, color in [(p_a, label_a, "steelblue"), (p_b, label_b, "darkorange")]:
    tr, _ = tr_x(df)
    ax.hist(tr.clip(-50_000, 50_000), bins=200, alpha=0.5, label=label, color=color, density=True)
ax.axvline(10_000, color="red", lw=0.8, ls="--")
ax.axvline(-10_000, color="red", lw=0.8, ls="--")
ax.set_xlabel("TR_x clipped to ±50k (frame px)")
ax.set_ylabel("density")
ax.set_title("TR_x distribution")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "tr_x_hist.png", dpi=120)
plt.close(fig)
print("Saved tr_x_hist.png")

# %%
# ---------------------------------------------------------------------------
# Level 2: Gaze quality
# (skip gracefully if files are missing)
# ---------------------------------------------------------------------------

gaze_a_path = dir_a / "eyetrack" / "gaze_per_sample.parquet"
gaze_b_path = dir_b / "eyetrack" / "gaze_per_sample.parquet"

if not gaze_a_path.exists() or not gaze_b_path.exists():
    missing = [str(p) for p in [gaze_a_path, gaze_b_path] if not p.exists()]
    print(f"\nSkipping gaze comparison — files not yet present:\n  " + "\n  ".join(missing))
else:
    g_a = pd.read_parquet(gaze_a_path)
    g_b = pd.read_parquet(gaze_b_path)
    trials = pd.read_parquet(trials_path)

    # -----------------------------------------------------------------------
    # Gaze scalar summary
    # -----------------------------------------------------------------------

    def gaze_summary(g, label):
        valid = g[g.gaze_valid]
        return {
            "branch": label,
            "total_samples": len(g),
            "gaze_valid_%": round(100 * g.gaze_valid.mean(), 1),
            "homography_valid_%": round(100 * g.homography_valid.mean(), 1),
            "on_screen_%": round(100 * g.on_screen.mean(), 1),
        }

    def preclick_accuracy(g, trials, window_ms=PRECLICK_WINDOW_MS):
        """Mean gaze-to-target distance (canvas px) in the pre-click window."""
        clicks = trials[trials.response_time.notna()].copy()
        valid_mask = g.gaze_valid & g.homography_valid
        g_valid = g[valid_mask][["behavior_t_ms", "gx_canvas", "gy_canvas"]].values
        distances = []
        for _, click in clicks.iterrows():
            rt = float(click.response_time)
            window = g_valid[
                (g_valid[:, 0] >= rt - window_ms) & (g_valid[:, 0] <= rt)
            ]
            if len(window) < 5:
                continue
            mean_gx = window[:, 1].mean()
            mean_gy = window[:, 2].mean()
            click_cx = float(click.response_x) * CANVAS_W_PX
            click_cy = float(click.response_y) * (CANVAS_H_PX / MAX_Y_COORD)
            distances.append(float(np.hypot(mean_gx - click_cx, mean_gy - click_cy)))
        return pd.Series(distances)

    acc_a = preclick_accuracy(g_a, trials)
    acc_b = preclick_accuracy(g_b, trials)

    g_rows = []
    for g, label, acc in [(g_a, label_a, acc_a), (g_b, label_b, acc_b)]:
        row = gaze_summary(g, label)
        row["preclick_accuracy median (canvas px)"] = round(acc.median(), 1) if len(acc) else float("nan")
        row["preclick_accuracy N clicks"] = len(acc)
        row["preclick_accuracy <250px %"] = round(100 * (acc < 250).mean(), 1) if len(acc) else float("nan")
        g_rows.append(row)

    g_table = pd.DataFrame(g_rows).set_index("branch").T
    print("\n=== Gaze quality ===")
    print(g_table.to_string())
    g_table.to_csv(out_dir / "gaze_summary.csv")

    # -----------------------------------------------------------------------
    # Plot: pre-click accuracy histograms
    # -----------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))
    for acc, label, color in [(acc_a, label_a, "steelblue"), (acc_b, label_b, "darkorange")]:
        if len(acc):
            ax.hist(acc.clip(upper=1000), bins=60, alpha=0.5, label=f"{label} (n={len(acc)}, med={acc.median():.0f}px)", color=color, density=True)
    ax.axvline(250, color="red", lw=1, ls="--", label="250 px threshold")
    ax.set_xlabel("mean pre-click gaze distance to target (canvas px)")
    ax.set_ylabel("density")
    ax.set_title("Pre-click gaze accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "preclick_accuracy.png", dpi=120)
    plt.close(fig)
    print("Saved preclick_accuracy.png")

    # -----------------------------------------------------------------------
    # Matched-sample canvas shift: where does the PR move gaze?
    # -----------------------------------------------------------------------

    # Join on tobii_ts_us (same Tobii recording, same timestamps in both runs).
    both = g_a[g_a.homography_valid][["tobii_ts_us", "gx_canvas", "gy_canvas"]].merge(
        g_b[g_b.homography_valid][["tobii_ts_us", "gx_canvas", "gy_canvas"]],
        on="tobii_ts_us",
        suffixes=("_a", "_b"),
    )
    both["delta_x"] = both.gx_canvas_b - both.gx_canvas_a
    both["delta_y"] = both.gy_canvas_b - both.gy_canvas_a
    both["shift_mag"] = np.hypot(both.delta_x, both.delta_y)

    print(f"\n=== Canvas shift ({label_b} − {label_a}), N={len(both)} matched samples ===")
    print(both[["delta_x", "delta_y", "shift_mag"]].describe().round(1).to_string())
    both[["tobii_ts_us", "delta_x", "delta_y", "shift_mag"]].to_parquet(out_dir / "canvas_shift.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Shift magnitude CDF
    ax = axes[0]
    sorted_mag = np.sort(both.shift_mag)
    ax.plot(sorted_mag, np.linspace(0, 1, len(sorted_mag)))
    ax.axvline(sorted_mag[int(0.5 * len(sorted_mag))], color="red", lw=0.8, ls="--", label="median")
    ax.axvline(sorted_mag[int(0.9 * len(sorted_mag))], color="orange", lw=0.8, ls="--", label="p90")
    ax.set_xlabel("gaze shift magnitude (canvas px)")
    ax.set_ylabel("CDF")
    ax.set_title("Shift magnitude")
    ax.legend(fontsize=8)

    # Δx, Δy histograms
    ax = axes[1]
    ax.hist(both.delta_x.clip(-100, 100), bins=80, alpha=0.7, color="steelblue")
    ax.set_xlabel(f"Δgx_canvas ({label_b} − {label_a}, px)")
    ax.set_title("Canvas shift: x")

    ax = axes[2]
    ax.hist(both.delta_y.clip(-100, 100), bins=80, alpha=0.7, color="steelblue")
    ax.set_xlabel(f"Δgy_canvas ({label_b} − {label_a}, px)")
    ax.set_title("Canvas shift: y")

    fig.suptitle(f"Per-sample canvas shift: {label_b} − {label_a}")
    fig.tight_layout()
    fig.savefig(out_dir / "canvas_shift.png", dpi=120)
    plt.close(fig)
    print("Saved canvas_shift.png")

print(f"\nAll outputs written to {out_dir}/")

# %%
# ---------------------------------------------------------------------------
# Visual comparison HTML — pipeline-generated PNGs side by side
# ---------------------------------------------------------------------------

import base64

PIPELINE_IMAGES = [
    (
        "cascade_trajectory.png",
        "Corner tracking trajectory",
        "Box-corner and screen-corner Y positions vs frame index. "
        "Look for fewer sudden jumps in the right column (PR branch). "
        "Persistent drift or spikes in the left column are what the PR targets.",
    ),
    (
        "big_star_residual_hist.png",
        "big_star residual histogram",
        "Distribution of 4-anchor-only H reprojection error at 19 hand-labeled "
        "big_star frames. Both branches use the same 4-anchor refit here, so "
        "this is a sanity check that the PR didn't disturb box-corner calibration. "
        "Distributions should look similar.",
    ),
    (
        "big_star_residual_vs_frame.png",
        "big_star residual vs frame",
        "Same residuals plotted over time. Outlier frames that are high in both "
        "branches indicate genuinely hard frames; outliers only in main suggest "
        "the PR fixed a specific bad-regime interval.",
    ),
    (
        str(Path("eyetrack") / "gaze_coverage_and_accuracy.png"),
        "Gaze coverage and pre-click accuracy",
        "Left panel: homography_valid and on_screen rates over time in 10 s bins — "
        "PR should be equal or higher throughout. "
        "Right panel: pre-click gaze distance to target; PR acceptance gate is "
        "median < 250 canvas px (red line). Compare medians and tail shape.",
    ),
    (
        str(Path("eyetrack") / "pre_click_gaze_trajectories.png"),
        "Pre-click gaze trajectories",
        "20 sampled click events. Each panel: canvas with revealed dots, target "
        "(gold), gaze trajectory in the 1 s before click (purple→yellow), click "
        "location (blue ×). Look for tighter trajectories around the target dot "
        "in the PR branch, especially for dots near the top of the canvas.",
    ),
    (
        str(Path("eyetrack") / "gaze_canvas_heatmap.png"),
        "Gaze canvas heatmap",
        "Log-scaled fixation density by trial phase (pre-reveal / reveal-to-click "
        "/ post-click). Overall layout should be similar; differences indicate "
        "gaze positions shifted by the new H. Hot spots near the top edge are "
        "most sensitive to the TR/TL conditioning fix.",
    ),
    (
        str(Path("eyetrack") / "saccade_psth_around_reveal.png"),
        "Saccade PSTH around reveal",
        "Saccade-onset times relative to dot-reveal events (±2 s). "
        "Should look similar between branches — this is a downstream neural "
        "signal and large differences would warrant investigation.",
    ),
]


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _img_tag(path):
    if not Path(path).exists():
        return f'<div class="missing">image not found: {path}</div>'
    ext = Path(path).suffix.lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f'<img src="data:image/{mime};base64,{_b64(path)}" />'


rows_html = []
for rel_path, title, instructions in PIPELINE_IMAGES:
    path_a = dir_a / rel_path
    path_b = dir_b / rel_path
    rows_html.append(f"""
  <section>
    <h2>{title}</h2>
    <p class="instructions">{instructions}</p>
    <div class="pair">
      <figure>
        <figcaption>{label_a}</figcaption>
        {_img_tag(path_a)}
      </figure>
      <figure>
        <figcaption>{label_b}</figcaption>
        {_img_tag(path_b)}
      </figure>
    </div>
  </section>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Pipeline comparison: {label_a} vs {label_b}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1600px; margin: 0 auto; padding: 1rem 2rem; background: #111; color: #ddd; }}
  h1 {{ font-size: 1.3rem; color: #fff; border-bottom: 1px solid #444; padding-bottom: .4rem; }}
  h2 {{ font-size: 1rem; color: #adf; margin: 2rem 0 .2rem; }}
  p.instructions {{ font-size: .85rem; color: #aaa; margin: 0 0 .6rem; max-width: 900px; line-height: 1.5; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: .8rem; font-weight: bold; color: #fa0; margin-bottom: .3rem; }}
  img {{ width: 100%; border: 1px solid #333; }}
  .missing {{ color: #f88; font-style: italic; padding: .5rem; border: 1px dashed #f88; }}
</style>
</head>
<body>
<h1>Pipeline comparison — {label_a} (left) vs {label_b} (right) &nbsp;·&nbsp; subject {subject}</h1>
{"".join(rows_html)}
</body>
</html>"""

html_path = out_dir / "visual_comparison.html"
html_path.write_text(html)
print(f"Saved {html_path}")
