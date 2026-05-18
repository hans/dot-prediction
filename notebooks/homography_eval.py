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
# # Homography solver evaluation — EC347
#
# Runs the three-step sequential homography solver and validates against
# hand-labeled ground-truth correspondences.
#
# **Outputs** (all under `results/{subject}/homography_eval/`):
# - `homography_box_calibration.json` — calibrated photodiode screen-coords
# - `homography_per_frame.parquet` — per-frame H + big_star residuals
# - `big_star_residual_hist.png` — residual histogram
# - `frame_{idx}_overlay.jpg` — spot-check overlays (requires video)

# %% tags=["parameters"]
import json
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
labels_path = str(_ROOT / f"results/{subject}/homography_labels.parquet")
trials_path = str(_ROOT / f"results/{subject}/trials_with_video.parquet")
align_path = str(_ROOT / f"results/{subject}/video_alignment.json")
out_dir = str(_ROOT / f"results/{subject}/homography_eval")

SCREEN_W_PX = 2388
SCREEN_H_PX = 1668

# Safari URL bar height in physical screen pixels (empirically derived ≈272 px = 136 CSS px
# at 2×).  Corrects behavior coordinates which are normalised to the canvas below the URL bar.
# Sanity check: behavior_to_screen(0, MAX_Y_COORD) → (0, SCREEN_H_PX).
URL_BAR_H_PX = 272
# Symmetric left/right white padding inside the screen (empirically derived ≈233 px each side).
CANVAS_X_PAD_PX = 233
# Fraction of canvas height in which dots can appear (from trials_df["max_y"]).
MAX_Y_COORD = 0.75

# Spot-check frames: calibration anchor, mid-trial, inter-trial, head-jump
SPOT_CHECK_FRAMES = [664, 1550, 2288, 30125]

# Label colours from label_homography_correspondences.py
LABEL_COLORS_HEX = {
    "screen_tl": "#ff4444",
    "screen_tr": "#44ff44",
    "screen_br": "#4488ff",
    "screen_bl": "#ffcc00",
    "box_bl":    "#ff44ff",
    "box_br":    "#44ffff",
    "big_star":  "#ffffff",
}

# %%
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from homography_solver import (
    behavior_to_screen,
    big_star_residuals,
    calibrate_box_position,
    fit_per_frame_homography,
)

# Wrap injected strings (Snakemake passes paths as str) back to Path.
video_path, labels_path, trials_path, align_path, out_dir = (
    Path(p) for p in [video_path, labels_path, trials_path, align_path, out_dir]
)
out_dir.mkdir(parents=True, exist_ok=True)


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


LABEL_COLORS_BGR = {lt: _hex_to_bgr(c) for lt, c in LABEL_COLORS_HEX.items()}


# %% [markdown]
# ## Load data

# %%
labels_df = pd.read_parquet(labels_path)
trials_df = pd.read_parquet(trials_path)

with open(align_path) as f:
    align = json.load(f)
fps = align["fps"]
slope_ms_per_s = align["slope_ms_per_s"]
intercept_ms = align["intercept_ms"]


def frame_to_expt_ms(frame_idx: int) -> float:
    return slope_ms_per_s * (frame_idx / fps) + intercept_ms


print(f"Labels: {len(labels_df)} rows across "
      f"{labels_df.frame_idx.nunique()} frames")
print(f"Trials: {len(trials_df)} rows, "
      f"trial {trials_df.trial_idx.min()}–{trials_df.trial_idx.max()}")

# %% [markdown]
# ## Step 1: Calibrate photodiode-device screen position

# %%
calib = calibrate_box_position(
    labels_df, screen_w_px=SCREEN_W_PX, screen_h_px=SCREEN_H_PX
)
box_bl_screen = tuple(calib["box_bl_screen"])
box_br_screen = tuple(calib["box_br_screen"])

calib_json_path = out_dir / "homography_box_calibration.json"
# Convert int keys in per_frame_estimates to strings for JSON.
calib_out = dict(calib)
calib_out["per_frame_estimates"] = {
    str(k): v for k, v in calib["per_frame_estimates"].items()
}
with open(calib_json_path, "w") as f:
    json.dump(calib_out, f, indent=2)
print(f"Calibration saved → {calib_json_path}")

n_calib = len(calib["calibration_frames"])
print(f"\nCalibration frames (N={n_calib}): {calib['calibration_frames']}")
print(f"  box_bl screen-xy: ({box_bl_screen[0]:.1f}, {box_bl_screen[1]:.1f})")
print(f"  box_br screen-xy: ({box_br_screen[0]:.1f}, {box_br_screen[1]:.1f})")

sp = calib["spread_screen_px"]
print(f"\nSpread (screen-px):")
for corner in ["box_bl", "box_br"]:
    s = sp[corner]
    flag = " ← ⚠ LARGE" if max(s["max_minus_min_x"], s["max_minus_min_y"]) > 30 else ""
    print(f"  {corner}  max−min x={s['max_minus_min_x']:.1f}  y={s['max_minus_min_y']:.1f}"
          f"  IQR x={s['iqr_x']:.1f}  y={s['iqr_y']:.1f}{flag}")

if n_calib == 1:
    print("\n[FLAG] Only 1 calibration frame — spread metrics not meaningful. "
          "Label additional frames with all 4 iPad corners visible.")

# %% [markdown]
# ## Step 2: Per-frame homography

# %%
per_frame_h = fit_per_frame_homography(
    labels_df,
    box_bl_screen=box_bl_screen,
    box_br_screen=box_br_screen,
    screen_w_px=SCREEN_W_PX,
    screen_h_px=SCREEN_H_PX,
)

n_included = (per_frame_h.excluded_reason == "").sum()
n_excluded = (per_frame_h.excluded_reason != "").sum()
print(f"Frames included: {n_included}   excluded: {n_excluded}")
for _, row in per_frame_h[per_frame_h.excluded_reason != ""].iterrows():
    print(f"  frame {int(row.frame_idx)}: {row.excluded_reason}")

# %% [markdown]
# ## Step 3: Validation via big_star reprojection

# %%
residuals_df = big_star_residuals(
    per_frame_h, labels_df, trials_df,
    screen_w_px=SCREEN_W_PX, screen_h_px=SCREEN_H_PX,
    url_bar_h_px=URL_BAR_H_PX, canvas_x_pad_px=CANVAS_X_PAD_PX,
    max_y_coord=MAX_Y_COORD,
)

print(f"big_star validation frames: {len(residuals_df)}")
if len(residuals_df) > 0:
    med = residuals_df.residual_px.median()
    q1, q3 = residuals_df.residual_px.quantile([0.25, 0.75])
    print(f"  median={med:.2f} px   IQR=[{q1:.2f}, {q3:.2f}]   "
          f"max={residuals_df.residual_px.max():.2f}")
    if med > 20:
        print("  [FLAG] Median residual > 20 px — consider joint optimisation.")
    # Check systematic bias (all residuals in same direction).
    dx = residuals_df.predicted_frame_x - residuals_df.labeled_frame_x
    dy = residuals_df.predicted_frame_y - residuals_df.labeled_frame_y
    if dx.mean() / (dx.std() + 1e-9) > 2 or dy.mean() / (dy.std() + 1e-9) > 2:
        print("  [FLAG] Systematic bias detected in residual direction.")

# %% [markdown]
# ## Merge residuals into per_frame_h and save parquet

# %%
residual_lookup = residuals_df.set_index("frame_idx")["residual_px"].rename("big_star_residual_px")
per_frame_h = per_frame_h.join(residual_lookup, on="frame_idx")

parquet_path = out_dir / "homography_per_frame.parquet"
per_frame_h.to_parquet(parquet_path, index=False)
print(f"Per-frame H saved → {parquet_path}")

# %% [markdown]
# ## Numeric summary

# %%
print(f"\n{'frame_idx':>10}  {'included':>8}  {'big_star_residual_px':>22}  excluded_reason")
print("-" * 65)
for _, row in per_frame_h.sort_values("frame_idx").iterrows():
    included = row.excluded_reason == ""
    res = f"{row.big_star_residual_px:.1f}" if pd.notna(row.get("big_star_residual_px")) else "n/a"
    print(f"{int(row.frame_idx):>10}  {str(included):>8}  {res:>22}  {row.excluded_reason}")

# %% [markdown]
# ## Residual histogram

# %%
if len(residuals_df) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals_df.residual_px, bins=10, edgecolor="white", color="#4488ff")
    med = residuals_df.residual_px.median()
    q1, q3 = residuals_df.residual_px.quantile([0.25, 0.75])
    ax.axvline(med, color="red", linewidth=1.5, label=f"median={med:.1f} px")
    ax.set_xlabel("Residual (frame-px)")
    ax.set_ylabel("Count")
    ax.set_title("big_star reprojection residuals")
    ax.set_subtitle = None  # matplotlib 3.x
    fig.suptitle(
        f"N={len(residuals_df)}  median={med:.1f} px  IQR=[{q1:.1f}, {q3:.1f}]",
        fontsize=10, y=0.97,
    )
    ax.legend()
    hist_path = out_dir / "big_star_residual_hist.png"
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved → {hist_path}")
else:
    print("(no big_star residuals — skipping histogram)")

# %% [markdown]
# ## Spot-check overlays

# %%
def _project(H_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H_mat @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _draw_filled(img, xy, color, radius=7):
    cv2.circle(img, (int(round(xy[0])), int(round(xy[1]))), radius, color, -1, cv2.LINE_AA)


def _draw_open(img, xy, color, radius=11):
    cv2.circle(img, (int(round(xy[0])), int(round(xy[1]))), radius, color, 2, cv2.LINE_AA)


def _h_for_frame(frame_idx: int) -> np.ndarray | None:
    rows = per_frame_h[
        (per_frame_h.frame_idx == frame_idx) & (per_frame_h.excluded_reason == "")
    ]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return np.array(
        [[r.h00, r.h01, r.h02], [r.h10, r.h11, r.h12], [r.h20, r.h21, r.h22]]
    )


def _stars_visible_at(frame_idx: int) -> list[tuple[float, float]]:
    """Screen-px coords of all stars currently visible at frame_idx."""
    expt_t = frame_to_expt_ms(frame_idx)
    active = trials_df[
        (trials_df.trial_onset <= expt_t)
        & (trials_df.trial_offset >= expt_t)
        & (trials_df.reveal_time <= expt_t)
    ]
    return [
        behavior_to_screen(float(row.true_x), float(row.true_y),
                           screen_w_px=SCREEN_W_PX, screen_h_px=SCREEN_H_PX,
                           url_bar_h_px=URL_BAR_H_PX, canvas_x_pad_px=CANVAS_X_PAD_PX,
                           max_y_coord=MAX_Y_COORD)
        for _, row in active.iterrows()
    ]


def _trial_context(frame_idx: int) -> dict:
    prior = trials_df[
        trials_df.video_frame_reveal.notna()
        & (trials_df.video_frame_reveal <= frame_idx)
    ]
    if prior.empty:
        return {"trial_idx": None, "tpt": None}
    row = prior.loc[prior.video_frame_reveal.idxmax()]
    return {"trial_idx": int(row.trial_idx), "tpt": int(row.tpt)}


if not video_path.exists():
    print(f"[WARN] Video not found at {video_path!r} — skipping overlay images.")
else:
    cap = cv2.VideoCapture(str(video_path))

    # Screen positions of the 4 H-fitting anchors.
    anchor_screen_xy = {
        "screen_bl": (0.0, float(SCREEN_H_PX)),
        "screen_br": (float(SCREEN_W_PX), float(SCREEN_H_PX)),
        "box_bl":    box_bl_screen,
        "box_br":    box_br_screen,
    }

    for frame_idx in SPOT_CHECK_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Could not read frame {frame_idx}")
            continue

        H_mat = _h_for_frame(frame_idx)
        expt_t_ms = frame_to_expt_ms(frame_idx)
        ctx = _trial_context(frame_idx)

        # big_star residual for caption.
        res_row = residuals_df[residuals_df.frame_idx == frame_idx]
        res_str = f"{res_row.iloc[0].residual_px:.1f}px" if not res_row.empty else "n/a"

        # --- draw labeled points (filled dots) ---
        frame_labels = labels_df[labels_df.frame_idx == frame_idx]
        for _, lr in frame_labels.iterrows():
            if not lr.visible or lr.label_type not in LABEL_COLORS_BGR:
                continue
            color = LABEL_COLORS_BGR[lr.label_type]
            _draw_filled(frame, (lr.x_frame, lr.y_frame), color, radius=7)

        # --- draw back-projections (open circles) + residual arrows ---
        if H_mat is not None:
            # Anchor back-projections (always drawn).
            for lt, (sx, sy) in anchor_screen_xy.items():
                bp_x, bp_y = _project(H_mat, sx, sy)
                color = LABEL_COLORS_BGR[lt]
                _draw_open(frame, (bp_x, bp_y), color, radius=11)
                # Residual arrow from label → back-projection.
                lr_rows = frame_labels[frame_labels.label_type == lt]
                if not lr_rows.empty and bool(lr_rows.iloc[0].visible):
                    lr = lr_rows.iloc[0]
                    dx = bp_x - lr.x_frame
                    dy = bp_y - lr.y_frame
                    length = np.hypot(dx, dy)
                    if length * 5 >= 5:
                        tip = (lr.x_frame + dx * 5, lr.y_frame + dy * 5)
                        cv2.arrowedLine(
                            frame,
                            (int(round(lr.x_frame)), int(round(lr.y_frame))),
                            (int(round(tip[0])), int(round(tip[1]))),
                            (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3,
                        )

            # big_star back-projection (even if not labeled visible).
            if not res_row.empty:
                r = res_row.iloc[0]
                bp_x, bp_y = _project(H_mat, r.true_screen_x, r.true_screen_y)
                _draw_open(frame, (bp_x, bp_y), LABEL_COLORS_BGR["big_star"], radius=11)
                dx = bp_x - r.labeled_frame_x
                dy = bp_y - r.labeled_frame_y
                length = np.hypot(dx, dy)
                if length * 5 >= 5:
                    tip = (r.labeled_frame_x + dx * 5, r.labeled_frame_y + dy * 5)
                    cv2.arrowedLine(
                        frame,
                        (int(round(r.labeled_frame_x)), int(round(r.labeled_frame_y))),
                        (int(round(tip[0])), int(round(tip[1]))),
                        (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3,
                    )

            # All currently-on-screen stars (small white dots).
            for sx, sy in _stars_visible_at(frame_idx):
                fx, fy = _project(H_mat, sx, sy)
                _draw_filled(frame, (fx, fy), (255, 255, 255), radius=3)

        # Caption.
        trial_str = (
            f"{ctx['trial_idx']}.{ctx['tpt']}" if ctx["trial_idx"] is not None else "—"
        )
        caption = (
            f"frame {frame_idx}  expt_t={expt_t_ms:.0f}ms  "
            f"trial={trial_str}  big_star_residual={res_str}"
        )
        cv2.putText(
            frame, caption, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, caption, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA,
        )

        overlay_path = out_dir / f"frame_{frame_idx}_overlay.jpg"
        cv2.imwrite(str(overlay_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Overlay saved → {overlay_path}")

    cap.release()

# %% [markdown]
# ## Open questions to flag
#
# 1. **Box-position calibration quality** — check `spread_screen_px` above.
#    If max−min > 30 screen-px, the per-frame estimates disagree more than
#    expected for a mechanically-fixed device; escalate to joint optimisation.
#
# 2. **H quality vs frame_idx (head pose / time drift)** — residual vs
#    frame_idx below; look for systematic growth late in the video.
#
# 3. **Geometry sanity** — the calibrated `box_bl_screen` / `box_br_screen`
#    should be at y ≈ 490–520 screen-px (photodiode device extends ~30% down).

# %%
if len(residuals_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(residuals_df.frame_idx, residuals_df.residual_px, s=30, color="#4488ff")
    ax.axhline(20, color="red", linestyle="--", linewidth=1, label="20 px threshold")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("Residual (px)")
    ax.set_title("big_star residual vs frame index")
    ax.legend()
    drift_path = out_dir / "big_star_residual_vs_frame.png"
    fig.savefig(drift_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Drift plot saved → {drift_path}")
