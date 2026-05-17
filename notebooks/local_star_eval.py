# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Local star detection evaluation — Phase 1b
#
# Compares three H_rough modes for the position-conditioned local detector:
#
# 1. **raw** corners (per-frame `detect_corners`)
# 2. **smoothed** corners (rolling median over `corner_smoother`)
# 3. **smoothed + anchor** (smoothed corners refined by translating so the
#    global Phase-1a big-star detection coincides with the predicted location
#    of the freshly-revealed star — falls back to plain smoothed when no
#    global detection is available)
#
# ## Frames evaluated
#
# Hand-picked to span the difficulty range from the spec:
# - mid/late-trial clean
# - mid/late-trial with hand occlusion
# - early-trial (only the big star + 1–2 small ones)
# - frame near a head-motion jump
# - inter-trial gap (no stars; verifies empty return)

# %% tags=["parameters"]
subject = "EC347"
video_path = f"/Users/jon/Projects/dot-prediction/data/{subject}/tobii/scenevideo.mp4"
trials_path = f"results/{subject}/trials_with_video.parquet"
align_path = f"results/{subject}/video_alignment.json"
out_dir = f"results/{subject}/local_star_eval"
window_size_px = 40       # fixed window for raw / smoothed modes (covers H_rough translation error)
floor = 20.0              # R-B opponent floor, permissive vs global detector's 40
# Adaptive window for the anchor-corrected mode: trial-1 reveals cluster
# within ~30 frame-px, so a fixed 40-px window makes all predictions overlap
# and snap to the same blob. With anchor correction, H_rough residual is
# small, so window ≈ 6 × expected_radius_px (≈12-15 px for fresh stars,
# down to floor for very old ones) isolates each prediction.
anchor_adaptive_factor = 6.0
anchor_min_window_px = 10
anchor_max_window_px = 40
# Frames spanning the difficulty range
eval_frames = [
    659,    # trial 0 start — very early, big star only
    750,    # trial 0 mid — big + a few small stars
    1500,   # trial 1 mid, clean, multiple small stars
    1700,   # trial 1, hand occlusion across top
    1900,   # trial 1 late, multiple small stars + occlusion
    2150,   # trial 1 very late, many small stars
    2270,   # inter-trial gap between trial 1 and trial 2 (empty expected)
]
smoothing_window = 51
smoothing_half_pad = 60   # frames of context each side of eval_frames union

# %% [markdown]
# ## Setup

# %%
import json
import os
import sys
from pathlib import Path

# Locate project root from this script's location (../). When run as a
# Jupyter notebook, ``__file__`` is undefined; fall back to cwd in that case.
_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
)
os.chdir(_ROOT)  # so the "results/..." relative paths resolve correctly
sys.path.insert(0, str(_ROOT / "src"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corner_smoother import smooth_corners
from homography_refinement import anchor_translate
from local_star_detector import (
    _window_size_for,
    detect_in_windows,
    find_overlapping_peaks,
)
from predicted_positions import SCREEN_H, SCREEN_W, predicted_positions
from screen_detection import detect_corners
from star_detector import detect_stars

out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

trials = pd.read_parquet(trials_path)
align = json.loads(Path(align_path).read_text())
slope, intercept = align["slope_ms_per_s"], align["intercept_ms"]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

SCREEN_CORNERS = np.array([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]],
                         dtype=np.float32)


def frame_to_expt_t(fi: int) -> float:
    return slope * fi / fps + intercept


def read_frame(fi: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    return frame if ret else None


# %% [markdown]
# ## Cache raw corners over the eval window + smooth
#
# Smoothing 30k+ frames is slow; smoothing only the [lo, hi] window of interest
# keeps eval interactive. The trade-off is that the rolling median uses fewer
# neighbors near the edges of [lo, hi], but the half-pad of 60 frames gives
# the central frames a full 51-wide window.

# %%
lo = max(0, min(eval_frames) - smoothing_half_pad)
hi = max(eval_frames) + smoothing_half_pad
print(f"Caching raw corners for frames [{lo}, {hi}] ({hi-lo+1} frames)...")
raw_corners: list[np.ndarray | None] = []
cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
for _ in range(lo, hi + 1):
    ret, frm = cap.read()
    raw_corners.append(detect_corners(frm) if ret else None)
n_ok = sum(c is not None for c in raw_corners)
print(f"  raw detection rate: {n_ok}/{len(raw_corners)} "
      f"({100 * n_ok / len(raw_corners):.1f}%)")
smoothed = smooth_corners(raw_corners, window=smoothing_window)


# %% [markdown]
# ## Per-frame evaluation
#
# For each eval frame and each H_rough mode, run the predictor + local
# detector and record one row per (frame, tpt, mode) for clean aggregation.

# %%
def homography_from_corners(corners: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(SCREEN_CORNERS.astype(np.float32),
                              corners.astype(np.float32))
    return H


def big_star_anchor(predictions, global_blobs) -> tuple[tuple[float, float],
                                                        tuple[float, float]] | None:
    """Pick the freshly-revealed prediction and the nearest global blob.

    Returns (anchor_screen_xy, anchor_frame_xy) or None if either side is empty.
    """
    if not predictions or not global_blobs:
        return None
    newest = predictions[-1]   # predictions are reveal-time ordered, oldest→newest
    fx_pred, fy_pred = newest.frame_xy
    j = int(np.argmin([np.hypot(b[0] - fx_pred, b[1] - fy_pred)
                       for b in global_blobs]))
    return newest.screen_xy, (float(global_blobs[j][0]), float(global_blobs[j][1]))


def detect_for_mode(frame, preds, mode: str):
    """Detector call with per-mode window strategy.

    raw / smoothed: fixed 40-px window (no anchor, must absorb H_rough bias).
    smoothed+anchor: adaptive window scaled by expected_radius_px.
    """
    if mode == "smoothed+anchor":
        return detect_in_windows(
            frame, preds, window_size_px=window_size_px, floor=floor,
            adaptive_radius_factor=anchor_adaptive_factor,
            min_window_px=anchor_min_window_px,
            max_window_px=anchor_max_window_px,
        )
    return detect_in_windows(frame, preds, window_size_px=window_size_px,
                             floor=floor)


def evaluate_one(fi: int, mode: str, H: np.ndarray, frame: np.ndarray,
                 expt_t: float, global_blobs) -> list[dict]:
    """Run prediction + local detection for one (frame, mode); return CSV rows."""
    preds = predicted_positions(expt_t, trials, H)
    dets, unmatched = detect_for_mode(frame, preds, mode)
    det_by_tpt = {d.source_prediction.tpt: d for d in dets}

    rows = []
    # Anchor diagnostics: did we find the big-star blob this frame?
    has_anchor = big_star_anchor(preds, global_blobs) is not None

    for p in preds:
        d = det_by_tpt.get(p.tpt)
        rows.append(dict(
            frame_idx=fi, video_t_s=fi / fps, expt_t_ms=expt_t,
            trial_idx=p.trial_idx, tpt=p.tpt,
            age_s=p.age_s, expected_radius_px=p.expected_radius_px,
            predicted_frame_x=p.frame_xy[0], predicted_frame_y=p.frame_xy[1],
            detected_frame_x=(d.frame_xy_subpix[0] if d else np.nan),
            detected_frame_y=(d.frame_xy_subpix[1] if d else np.nan),
            confidence=(d.confidence if d else np.nan),
            equivalent_radius_px=(d.equivalent_radius_px if d else np.nan),
            source=("local" if d else "none"),
            H_rough_mode=mode,
            has_anchor=has_anchor,
        ))
    return rows


records: list[dict] = []
overlay_buffers: dict[int, np.ndarray] = {}

for fi in eval_frames:
    frame = read_frame(fi)
    if frame is None:
        print(f"frame {fi}: read fail"); continue
    expt_t = frame_to_expt_t(fi)
    print(f"\nframe {fi}: expt_t={expt_t:.0f} ms")

    # Global Phase-1a detection (stable per spec; not modified).
    global_blobs = detect_stars(frame)
    print(f"  global blobs: {len(global_blobs)}")

    cache_idx = fi - lo
    raw_c = raw_corners[cache_idx]
    smo_c = smoothed[cache_idx]

    modes: list[tuple[str, np.ndarray | None]] = []
    if raw_c is not None:
        modes.append(("raw", homography_from_corners(raw_c)))
    else:
        modes.append(("raw", None))
    modes.append(("smoothed", homography_from_corners(smo_c)))

    # smoothed + anchor (if global blob exists)
    H_smo = homography_from_corners(smo_c)
    preds_for_anchor = predicted_positions(expt_t, trials, H_smo)
    anchor = big_star_anchor(preds_for_anchor, global_blobs)
    if anchor is not None:
        H_anchor = anchor_translate(H_smo, anchor[0], anchor[1])
    else:
        H_anchor = H_smo  # fall back; result will be equal to plain smoothed
    modes.append(("smoothed+anchor", H_anchor))

    for mode, H in modes:
        if H is None:
            continue
        rows = evaluate_one(fi, mode, H, frame, expt_t, global_blobs)
        # Count peak-pixel overlaps once per (frame, mode), attach to each row
        preds = predicted_positions(expt_t, trials, H)
        dets, _ = detect_for_mode(frame, preds, mode)
        n_overlap_pairs = len(find_overlapping_peaks(dets))
        for r in rows:
            r["n_overlapping_peak_pairs"] = n_overlap_pairs
        records.extend(rows)

    # Build a multi-panel overlay for this frame
    panels = []
    for mode, H in modes:
        if H is None: continue
        preds = predicted_positions(expt_t, trials, H)
        dets, _ = detect_for_mode(frame, preds, mode)
        det_by_tpt = {d.source_prediction.tpt: d for d in dets}
        vis = frame.copy()
        # Search windows — actual window used for each prediction
        for p in preds:
            cx, cy = int(p.frame_xy[0]), int(p.frame_xy[1])
            if mode == "smoothed+anchor":
                w = _window_size_for(p, window_size_px, anchor_adaptive_factor,
                                     anchor_min_window_px, anchor_max_window_px)
            else:
                w = window_size_px
            half = w // 2
            cv2.rectangle(vis, (cx - half, cy - half), (cx + half, cy + half),
                          (60, 60, 60), 1)
            age_norm = min(p.age_s / 60, 1.0)
            col = (int(255 * age_norm), 0, int(255 * (1 - age_norm)))
            cv2.drawMarker(vis, (cx, cy), col, cv2.MARKER_TILTED_CROSS, 14, 1)
            d = det_by_tpt.get(p.tpt)
            if d:
                fx, fy = d.frame_xy_subpix
                cv2.circle(vis, (int(fx), int(fy)), 6, (0, 255, 0), 2)
            else:
                cv2.line(vis,
                         (cx - 6, cy - 6), (cx + 6, cy + 6), (0, 0, 200), 1)
                cv2.line(vis,
                         (cx - 6, cy + 6), (cx + 6, cy - 6), (0, 0, 200), 1)
        # Global blob detections (cyan)
        for bx, by, br in global_blobs:
            cv2.circle(vis, (int(bx), int(by)), max(int(br), 6), (0, 255, 255), 2)
        cv2.putText(vis, f"f{fi}  mode={mode}  preds={len(preds)}  loc_det={len(dets)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(vis)
    if panels:
        H_img = panels[0].shape[0]
        # Stack vertically
        cv2.imwrite(str(out_path / f"f{fi:05d}_overlay.jpg"),
                    np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])

cap.release()


eval_df = pd.DataFrame(records)
csv_path = out_path / "local_star_eval.csv"
eval_df.to_csv(csv_path, index=False)
print(f"\nSaved {len(eval_df)} rows → {csv_path}")


# %% [markdown]
# ## Detection rate per H_rough mode
#
# The "smoothed+anchor" overall rate is misleading because it pools frames
# where the Phase-1a global detector did find the big-star (anchor available
# → corrected H_rough) with frames where it didn't (no anchor → falls back
# to plain smoothed). Splitting by ``has_anchor`` surfaces the real finding:
# **on anchor-available frames the local detector recovers nearly all
# predicted stars; on anchor-unavailable frames it recovers nothing** —
# which is the Phase-1a bottleneck, not a Phase-1b failure.

# %%
print("Overall detection rate per H_rough mode (pooled, can be misleading):\n")
print(eval_df.groupby("H_rough_mode", observed=False)
      .apply(lambda g: (g.source == "local").mean(), include_groups=False)
      .rename("detect_rate"))

print("\nDetection rate per H_rough mode, split by anchor availability:\n")
print(eval_df.groupby(["H_rough_mode", "has_anchor"], observed=False)
      .apply(lambda g: (g.source == "local").mean(), include_groups=False)
      .rename("detect_rate"))


# %% [markdown]
# ## Detection rate per star age bucket
#
# Spec's primary axis: as stars age (and shrink), is detection rate falling off?

# %%
eval_df["age_bucket"] = pd.cut(
    eval_df["age_s"], bins=[-0.01, 2, 5, 10, 20, 30, 45, 60, 120],
    labels=["0-2", "2-5", "5-10", "10-20", "20-30", "30-45", "45-60", "60+"],
)

age_pivot = (eval_df
             .assign(detected=(eval_df["source"] == "local").astype(int))
             .groupby(["H_rough_mode", "age_bucket"], observed=False)["detected"]
             .agg(["sum", "size"])
             .assign(rate=lambda d: d["sum"] / d["size"]))
print(age_pivot)


# %% [markdown]
# ## Candidate frames — big star vs. small stars
#
# Per-frame snapshot showing where the predictor places the freshly-revealed
# big star (yellow circle, age <2 s) and all previously-revealed small stars
# (cyan crosses, sized by their expected radius). Local detections are drawn
# with a green outline; misses with a red X. Three H_rough modes side by side.

# %%
BIG_AGE_THRESHOLD_S = 2.0

def draw_candidate(frame, preds, dets, H_label):
    vis = frame.copy()
    det_by_tpt = {d.source_prediction.tpt: d for d in dets}
    if not preds:
        cv2.putText(vis, f"mode={H_label}  no predictions",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return vis
    newest_tpt = preds[-1].tpt
    for p in preds:
        px, py = int(p.frame_xy[0]), int(p.frame_xy[1])
        is_big = (p.tpt == newest_tpt) and (p.age_s < BIG_AGE_THRESHOLD_S)
        if is_big:
            cv2.circle(vis, (px, py), 18, (0, 255, 255), 3)
            cv2.putText(vis, f"BIG t{p.tpt} ({p.age_s:.1f}s)",
                        (px + 22, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1)
        else:
            cv2.drawMarker(vis, (px, py), (200, 200, 80),
                           cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.putText(vis, f"t{p.tpt}/{p.age_s:.0f}s",
                        (px + 8, py + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 80), 1)
        d = det_by_tpt.get(p.tpt)
        if d:
            cv2.circle(vis, (int(d.frame_xy_subpix[0]), int(d.frame_xy_subpix[1])),
                       7, (0, 255, 0), 2)
        else:
            cv2.line(vis, (px - 8, py - 8), (px + 8, py + 8), (0, 0, 220), 1)
            cv2.line(vis, (px - 8, py + 8), (px + 8, py - 8), (0, 0, 220), 1)
    n_det = sum(1 for p in preds if p.tpt in det_by_tpt)
    cv2.putText(vis,
                f"mode={H_label}  preds={len(preds)}  detected={n_det}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis


candidate_cap = cv2.VideoCapture(video_path)
for fi in eval_frames:
    candidate_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = candidate_cap.read()
    if not ret: continue
    expt_t = frame_to_expt_t(fi)
    global_blobs = detect_stars(frame)
    cache_idx = fi - lo
    raw_c = raw_corners[cache_idx]
    smo_c = smoothed[cache_idx]

    H_raw = homography_from_corners(raw_c) if raw_c is not None else None
    H_smo = homography_from_corners(smo_c)
    preds_smo = predicted_positions(expt_t, trials, H_smo)
    anchor = big_star_anchor(preds_smo, global_blobs)
    H_anchor = anchor_translate(H_smo, anchor[0], anchor[1]) if anchor else H_smo

    panels = []
    for label, H in [("raw", H_raw), ("smoothed", H_smo),
                     ("smoothed+anchor", H_anchor)]:
        if H is None:
            blank = np.full_like(frame, 30)
            cv2.putText(blank, f"mode={label}: no corners detected",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            panels.append(blank); continue
        preds = predicted_positions(expt_t, trials, H)
        dets, _ = detect_for_mode(frame, preds, label)
        panels.append(draw_candidate(frame, preds, dets, label))
    cv2.imwrite(str(out_path / f"candidate_f{fi:05d}.jpg"),
                np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
candidate_cap.release()
print(f"Candidate overlays in {out_path}/candidate_f*.jpg")


# %% [markdown]
# ## All-overlays inline display

# %%
fig_paths = sorted(out_path.glob("f*_overlay.jpg"))
for p in fig_paths:
    img = cv2.imread(str(p))
    fig, ax = plt.subplots(figsize=(12, 12 * img.shape[0] / img.shape[1]))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(p.name); ax.axis("off")
    plt.show()


# %% [markdown]
# ## Centroid stability across consecutive frames
#
# Pick a frame range inside one trial where H_rough+anchor is good, and track
# the same tpt over ~25 frames. The std of its detected centroid is the
# per-correspondence noise floor.

# %%
stab_frames = list(range(1485, 1515))  # ~30 frames around frame 1500
stab_tpt = 6   # newest at frame 1500
stab_records = []
stab_cap = cv2.VideoCapture(video_path)
for fi in stab_frames:
    stab_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = stab_cap.read()
    if not ret:
        continue
    expt_t = frame_to_expt_t(fi)
    global_blobs = detect_stars(frame)
    cache_idx = fi - lo
    if cache_idx < 0 or cache_idx >= len(smoothed):
        continue
    H_smo = homography_from_corners(smoothed[cache_idx])
    preds = predicted_positions(expt_t, trials, H_smo)
    anchor = big_star_anchor(preds, global_blobs)
    H_use = anchor_translate(H_smo, anchor[0], anchor[1]) if anchor else H_smo
    preds = predicted_positions(expt_t, trials, H_use)
    dets, _ = detect_for_mode(frame, preds, "smoothed+anchor")
    for d in dets:
        if d.source_prediction.tpt == stab_tpt:
            stab_records.append(dict(frame=fi,
                                     cx=d.frame_xy_subpix[0],
                                     cy=d.frame_xy_subpix[1]))
stab_cap.release()

stab = pd.DataFrame(stab_records)
if len(stab) >= 5:
    jx, jy = stab.cx.std(), stab.cy.std()
    rms = float(np.sqrt(jx ** 2 + jy ** 2))
    print(f"Centroid stability tpt={stab_tpt} over frames {stab.frame.min()}-{stab.frame.max()}")
    print(f"  std_x = {jx:.2f} px,  std_y = {jy:.2f} px,  RMS = {rms:.2f} px")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(stab.frame, stab.cx, ".-", label="cx")
    axes[0].plot(stab.frame, stab.cy, ".-", label="cy")
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("centroid (px)")
    axes[0].legend(); axes[0].set_title("Centroid over consecutive frames")
    axes[1].scatter(stab.cx - stab.cx.mean(), stab.cy - stab.cy.mean(),
                    s=25, alpha=0.7)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Δcx (px)"); axes[1].set_ylabel("Δcy (px)")
    axes[1].set_title(f"Scatter — RMS={rms:.2f} px")
    plt.tight_layout()
    plt.savefig(str(out_path / "centroid_stability.png"), dpi=100)
    plt.show()
else:
    print(f"Stability tpt={stab_tpt} only detected in {len(stab)} frames; "
          f"try a different (frame range, tpt) combination.")
