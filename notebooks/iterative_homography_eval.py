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
# # Iterative homography refinement — Phase 1c sanity checks
#
# Three visualizations for steps 1–3, before the iteration controller (step 4/5)
# is wired up:
#
# 1. **`detect_constellation` vs `detect_in_windows`** on the 7 Phase-1b eval
#    frames — validates the same-blob fix from step 3.
# 2. **Quality-gate rejection map** — which Phase-1b anchor-mode detections
#    survive `apply_quality_gates`, and why the rest were dropped.
# 3. **One-shot re-solve** on frame 1500 — builds a correspondence set per the
#    Change-1 table and checks whether one call to `solve_weighted_homography`
#    moves star predictions toward the detected blobs.

# %% tags=["parameters"]
subject = "EC347"
video_path = f"/Users/jon/Projects/dot-prediction/data/{subject}/tobii/scenevideo.mp4"
trials_path = f"results/{subject}/trials_with_video.parquet"
align_path = f"results/{subject}/video_alignment.json"
out_dir = f"results/{subject}/iterative_homography_eval"
floor = 20.0
# Same adaptive window params as Phase-1b smoothed+anchor mode
anchor_adaptive_factor = 6.0
anchor_min_window_px = 10
anchor_max_window_px = 40
eval_frames = [659, 750, 1500, 1700, 1900, 2150, 2270]
resolvable_frame = 1500  # clean mid-trial frame for the one-shot re-solve
smoothing_window = 51
smoothing_half_pad = 60

# %% [markdown]
# ## Setup

# %%
import json
import os
import sys
from pathlib import Path

def _find_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    # notebook mode: walk up from cwd until pyproject.toml is found
    p = Path.cwd()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return Path.cwd()

_ROOT = _find_root()
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src"))

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corner_smoother import smooth_corners
from homography_refinement import (
    Correspondence,
    apply_quality_gates,
    anchor_translate,
    detect_constellation,
    solve_weighted_homography,
)
from local_star_detector import detect_in_windows, find_overlapping_peaks
from predicted_positions import SCREEN_H, SCREEN_W, predicted_positions
from screen_detection import detect_corners
from star_detector import detect_stars

out_path = Path(out_dir)
out_path.mkdir(parents=True, exist_ok=True)

trials = pd.read_parquet(trials_path)
align = json.loads(Path(align_path).read_text())
slope, intercept = align["slope_ms_per_s"], align["intercept_ms"]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# [TL, TR, BR, BL] — matches detect_corners output order
SCREEN_CORNERS_NP = np.array(
    [[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]],
    dtype=np.float32,
)
CORNER_LABELS = ["TL", "TR", "BR", "BL"]

DETECTOR_KW = dict(
    floor=floor,
    adaptive_radius_factor=anchor_adaptive_factor,
    min_window_px=anchor_min_window_px,
    max_window_px=anchor_max_window_px,
)


def frame_to_expt_t(fi: int) -> float:
    return slope * fi / fps + intercept


def read_frame(fi: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    return frame if ret else None


def homography_from_corners(corners: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(
        SCREEN_CORNERS_NP.astype(np.float32),
        corners.astype(np.float32),
    )
    return H


def big_star_anchor(predictions, global_blobs):
    """Return (anchor_screen_xy, anchor_frame_xy) or None."""
    if not predictions or not global_blobs:
        return None
    newest = predictions[-1]
    fx_pred, fy_pred = newest.frame_xy
    j = int(np.argmin([np.hypot(b[0] - fx_pred, b[1] - fy_pred) for b in global_blobs]))
    return newest.screen_xy, (float(global_blobs[j][0]), float(global_blobs[j][1]))


def anchor_h_for(fi: int, frame: np.ndarray, cache_idx: int):
    """Return (H_smo, H_anchor, smo_c, raw_c, global_blobs, preds_anchor)."""
    smo_c = smoothed_corners[cache_idx]
    raw_c = raw_corners_list[cache_idx]
    H_smo = homography_from_corners(smo_c)
    global_blobs = detect_stars(frame)
    preds_smo = predicted_positions(frame_to_expt_t(fi), trials, H_smo)
    anchor = big_star_anchor(preds_smo, global_blobs)
    H_anchor = anchor_translate(H_smo, anchor[0], anchor[1]) if anchor else H_smo
    preds = predicted_positions(frame_to_expt_t(fi), trials, H_anchor)
    return H_smo, H_anchor, smo_c, raw_c, global_blobs, preds

# %% [markdown]
# ## Cache raw corners + smooth

# %%
lo = max(0, min(eval_frames) - smoothing_half_pad)
hi = max(eval_frames) + smoothing_half_pad
print(f"Caching raw corners [{lo}, {hi}]...")
raw_corners_list: list[np.ndarray | None] = []
cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
for _ in range(lo, hi + 1):
    ret, frm = cap.read()
    raw_corners_list.append(detect_corners(frm) if ret else None)
n_ok = sum(c is not None for c in raw_corners_list)
print(f"  {n_ok}/{len(raw_corners_list)} detected ({100*n_ok/len(raw_corners_list):.1f}%)")
smoothed_corners = smooth_corners(raw_corners_list, window=smoothing_window)

# %% [markdown]
# ## Viz 1 — `detect_constellation` vs `detect_in_windows`
#
# Both detectors run on the same anchor-corrected H and predictions (Phase-1b
# smoothed+anchor mode). Overlapping-peak detections in `detect_in_windows`
# output — same-blob snaps — are drawn in orange; `detect_constellation`
# should eliminate them.
#
# **Legend:** grey cross = prediction; green circle = unique detection;
# orange circle = overlapping-peak detection; red X = unmatched.

# %%
def draw_detector_panel(
    frame: np.ndarray,
    preds,
    dets,
    overlap_idxs: set[int],
    label: str,
    n_overlap_pairs: int,
) -> np.ndarray:
    vis = frame.copy()
    fi_txt = label.split()[0] if " " in label else ""
    det_by_tpt = {d.source_prediction.tpt: d for d in dets}
    for p in preds:
        cx, cy = int(p.frame_xy[0]), int(p.frame_xy[1])
        cv2.drawMarker(vis, (cx, cy), (110, 110, 110), cv2.MARKER_TILTED_CROSS, 12, 1)
    for i, d in enumerate(dets):
        fx, fy = int(d.frame_xy_subpix[0]), int(d.frame_xy_subpix[1])
        color = (0, 150, 255) if i in overlap_idxs else (0, 220, 0)
        cv2.circle(vis, (fx, fy), 7, color, 2)
    for p in preds:
        if p.tpt not in det_by_tpt:
            cx, cy = int(p.frame_xy[0]), int(p.frame_xy[1])
            cv2.line(vis, (cx - 7, cy - 7), (cx + 7, cy + 7), (0, 0, 210), 2)
            cv2.line(vis, (cx - 7, cy + 7), (cx + 7, cy - 7), (0, 0, 210), 2)
    suffix = f"  same_blob_pairs={n_overlap_pairs}" if n_overlap_pairs else ""
    cv2.putText(
        vis,
        f"{label}  det={len(dets)}/{len(preds)}{suffix}",
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
    )
    return vis


print(f"{'frame':>6}  {'preds':>5}  {'old_det':>7}  {'old_pairs':>9}  "
      f"{'new_det':>7}  {'new_unmatch':>11}")
for fi in eval_frames:
    frame = read_frame(fi)
    if frame is None:
        print(f"  frame {fi}: read fail")
        continue
    cache_idx = fi - lo
    _, H_anchor, _, _, _, preds = anchor_h_for(fi, frame, cache_idx)

    old_dets, _ = detect_in_windows(frame, preds, **DETECTOR_KW)
    old_pairs = find_overlapping_peaks(old_dets)
    old_overlap_idxs = {i for pair in old_pairs for i in pair}

    new_dets, new_unmatched = detect_constellation(frame, preds, **DETECTOR_KW)

    print(f"{fi:>6}  {len(preds):>5}  {len(old_dets):>7}  {len(old_pairs):>9}  "
          f"{len(new_dets):>7}  {len(new_unmatched):>11}")

    panel_old = draw_detector_panel(
        frame, preds, old_dets, old_overlap_idxs,
        f"f{fi}  detect_in_windows", len(old_pairs),
    )
    panel_new = draw_detector_panel(
        frame, preds, new_dets, set(),
        f"f{fi}  detect_constellation", 0,
    )
    cv2.imwrite(
        str(out_path / f"v1_f{fi:05d}_constellation.jpg"),
        np.vstack([panel_old, panel_new]),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

print(f"\nOverlays → {out_path}/v1_f*_constellation.jpg")

# %% [markdown]
# ### Inline: Viz 1 overlays

# %%
for fi in eval_frames:
    img_path = out_path / f"v1_f{fi:05d}_constellation.jpg"
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / img.shape[1]))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.name)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Viz 2 — Quality-gate rejection map
#
# Runs `apply_quality_gates` on `detect_in_windows` output (Phase-1b anchor
# mode). Shows which detections survive and the reason each rejected one was
# dropped.
#
# **Legend:** green circle = accepted; red X = radius_mismatch;
# orange X = same_blob.

# %%
def draw_gate_panel(
    frame: np.ndarray,
    preds,
    accepted,
    rejections,
    fi: int,
) -> np.ndarray:
    vis = frame.copy()
    det_by_tpt = {d.source_prediction.tpt: d for d in accepted}
    rej_by_tpt = {r.detection.source_prediction.tpt: r for r in rejections}
    for p in preds:
        cx, cy = int(p.frame_xy[0]), int(p.frame_xy[1])
        cv2.drawMarker(vis, (cx, cy), (110, 110, 110), cv2.MARKER_TILTED_CROSS, 12, 1)
    for d in accepted:
        fx, fy = int(d.frame_xy_subpix[0]), int(d.frame_xy_subpix[1])
        cv2.circle(vis, (fx, fy), 8, (0, 220, 0), 2)
    for r in rejections:
        fx = int(r.detection.frame_xy_subpix[0])
        fy = int(r.detection.frame_xy_subpix[1])
        color = (0, 0, 210) if r.reason == "radius_mismatch" else (0, 150, 255)
        cv2.line(vis, (fx - 8, fy - 8), (fx + 8, fy + 8), color, 2)
        cv2.line(vis, (fx - 8, fy + 8), (fx + 8, fy - 8), color, 2)
        cv2.putText(vis, r.reason[:4], (fx + 10, fy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    n_r = sum(1 for r in rejections if r.reason == "radius_mismatch")
    n_b = sum(1 for r in rejections if r.reason == "same_blob")
    cv2.putText(
        vis,
        (f"f{fi}  quality_gates  accepted={len(accepted)}/{len(accepted)+len(rejections)}"
         f"  radius_drop={n_r}  blob_drop={n_b}"),
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    return vis


print("Quality-gate survival:")
print(f"{'frame':>6}  {'det_in':>6}  {'accepted':>8}  {'r_miss':>6}  {'s_blob':>6}")
for fi in eval_frames:
    frame = read_frame(fi)
    if frame is None:
        continue
    cache_idx = fi - lo
    _, H_anchor, _, _, _, preds = anchor_h_for(fi, frame, cache_idx)

    dets, _ = detect_in_windows(frame, preds, **DETECTOR_KW)
    accepted, rejections = apply_quality_gates(dets)
    n_r = sum(1 for r in rejections if r.reason == "radius_mismatch")
    n_b = sum(1 for r in rejections if r.reason == "same_blob")
    print(f"{fi:>6}  {len(dets):>6}  {len(accepted):>8}  {n_r:>6}  {n_b:>6}")

    panel = draw_gate_panel(frame, preds, accepted, rejections, fi)
    cv2.imwrite(
        str(out_path / f"v2_f{fi:05d}_gates.jpg"),
        panel,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

print(f"\nOverlays → {out_path}/v2_f*_gates.jpg")

# %% [markdown]
# ### Inline: Viz 2 overlays

# %%
for fi in eval_frames:
    img_path = out_path / f"v2_f{fi:05d}_gates.jpg"
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    fig, ax = plt.subplots(figsize=(14, 7 * img.shape[0] / img.shape[1]))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.name)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Viz 3 — One-shot re-solve sanity check (frame 1500)
#
# Builds correspondences per the Change-1 table (smoothed BL/BR @ 1.0,
# smoothed TL/TR @ 0.3, raw corners conditional on |raw−smooth| thresholds,
# big star @ 0.9, gated small stars at detector-confidence weight), then
# calls `solve_weighted_homography`.
#
# **Outputs:**
# 1. Per-correspondence residual bar chart.
# 2. Frame overlay: orange crosses = H_anchor predictions, blue crosses =
#    H_refined predictions, arrows show the shift, green circles = detections.
# 3. Per-star prediction-error table (H_anchor → H_refined vs actual centroid).

# %%
fi = resolvable_frame
frame = read_frame(fi)
expt_t = frame_to_expt_t(fi)
cache_idx = fi - lo
H_smo, H_anchor, smo_c, raw_c, global_blobs, preds = anchor_h_for(fi, frame, cache_idx)

# Detect + gate small stars under H_anchor
dets, _ = detect_constellation(frame, preds, **DETECTOR_KW)
accepted_dets, rejections = apply_quality_gates(dets)
print(f"Frame {fi}: {len(dets)} detections → {len(accepted_dets)} post-gate "
      f"({len(rejections)} rejected)")

# Build correspondences per Change-1 table
correspondences: list[Correspondence] = []
labels: list[str] = []

# Smoothed corners (always included)
for i, clabel in enumerate(CORNER_LABELS):
    sc_xy = (float(SCREEN_CORNERS_NP[i][0]), float(SCREEN_CORNERS_NP[i][1]))
    sm_xy = (float(smo_c[i][0]), float(smo_c[i][1]))
    w = 1.0 if clabel in ("BL", "BR") else 0.3
    correspondences.append(Correspondence(sc_xy, sm_xy, w, "corner_smoothed"))
    labels.append(f"{clabel}_smo")

# Raw corners (conditional on |raw − smoothed| thresholds from Change-1)
if raw_c is not None:
    for i, clabel in enumerate(CORNER_LABELS):
        sc_xy = (float(SCREEN_CORNERS_NP[i][0]), float(SCREEN_CORNERS_NP[i][1]))
        raw_xy = (float(raw_c[i][0]), float(raw_c[i][1]))
        diff = float(np.hypot(raw_c[i][0] - smo_c[i][0], raw_c[i][1] - smo_c[i][1]))
        if clabel in ("BL", "BR") and diff < 10:
            correspondences.append(Correspondence(sc_xy, raw_xy, 0.5, "corner_raw"))
            labels.append(f"{clabel}_raw({diff:.1f}px)")
        elif clabel in ("TL", "TR") and diff < 5:
            correspondences.append(Correspondence(sc_xy, raw_xy, 0.1, "corner_raw"))
            labels.append(f"{clabel}_raw({diff:.1f}px)")

# Big star anchor
anchor = big_star_anchor(preds, global_blobs)
if anchor is not None:
    correspondences.append(Correspondence(anchor[0], anchor[1], 0.9, "big_star"))
    labels.append("big_star")

# Accepted small stars (weight = normalized confidence × 0.7)
for d in accepted_dets:
    w = float(d.confidence) / 255.0 * 0.7
    correspondences.append(
        Correspondence(d.source_prediction.screen_xy, d.frame_xy_subpix, w, "small_star")
    )
    labels.append(f"t{d.source_prediction.tpt}")

print(f"\nCorrespondences ({len(correspondences)} total):")
for c, lbl in zip(correspondences, labels):
    print(f"  {lbl:20s}  w={c.weight:.2f}  src={c.source}")

result = solve_weighted_homography(correspondences)
print(f"\nSolver: {result.method}")
print(f"Residuals (px) — median={np.nanmedian(result.residuals_px):.2f}  "
      f"max={np.nanmax(result.residuals_px):.2f}")

# %% [markdown]
# ### Residual bar chart

# %%
src_colors = {
    "corner_smoothed": "steelblue",
    "corner_raw": "cornflowerblue",
    "big_star": "gold",
    "small_star": "mediumseagreen",
}
bar_colors = [src_colors[c.source] for c in correspondences]
residuals = result.residuals_px

fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7 + 1), 4))
ax.bar(range(len(labels)), residuals, color=bar_colors, edgecolor="white", linewidth=0.4)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Reprojection error (px)")
ax.set_title(f"Frame {fi}: per-correspondence residuals after re-solve ({result.method})")
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax.axhline(3.0, color="tomato", linestyle="--", linewidth=0.8)
ax.text(len(labels) - 0.5, 1.05, "1 px", color="grey", fontsize=8, ha="right")
ax.text(len(labels) - 0.5, 3.05, "3 px", color="tomato", fontsize=8, ha="right")
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in src_colors.items()]
ax.legend(handles=legend_patches, fontsize=8)
plt.tight_layout()
plt.savefig(str(out_path / f"v3_f{fi:05d}_residuals.png"), dpi=120)
plt.show()

# %% [markdown]
# ### Prediction shift: H_anchor → H_refined

# %%
preds_refined = predicted_positions(expt_t, trials, result.H)
preds_by_tpt = {p.tpt: p for p in preds}
dets_by_tpt = {d.source_prediction.tpt: d for d in accepted_dets}

vis = frame.copy()

# Detections (green circles — reference)
for d in accepted_dets:
    fx, fy = int(d.frame_xy_subpix[0]), int(d.frame_xy_subpix[1])
    cv2.circle(vis, (fx, fy), 8, (0, 220, 0), 2)

# H_anchor predictions (orange crosses)
for p in preds:
    cx, cy = int(p.frame_xy[0]), int(p.frame_xy[1])
    cv2.drawMarker(vis, (cx, cy), (0, 150, 255), cv2.MARKER_CROSS, 14, 2)

# H_refined predictions (blue crosses) + arrows from anchor
for p_ref in preds_refined:
    rx, ry = int(p_ref.frame_xy[0]), int(p_ref.frame_xy[1])
    cv2.drawMarker(vis, (rx, ry), (255, 80, 0), cv2.MARKER_CROSS, 14, 2)
    p_anc = preds_by_tpt.get(p_ref.tpt)
    if p_anc is not None:
        ax_px, ay_px = int(p_anc.frame_xy[0]), int(p_anc.frame_xy[1])
        if np.hypot(rx - ax_px, ry - ay_px) > 1:
            cv2.arrowedLine(vis, (ax_px, ay_px), (rx, ry), (150, 150, 0), 1, tipLength=0.35)

# Smoothed screen corners (yellow squares)
for i, clabel in enumerate(CORNER_LABELS):
    fx, fy = int(smo_c[i][0]), int(smo_c[i][1])
    cv2.rectangle(vis, (fx - 6, fy - 6), (fx + 6, fy + 6), (50, 210, 210), 2)
    cv2.putText(vis, clabel, (fx + 8, fy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 210, 210), 1)

cv2.putText(
    vis,
    "green=detected  orange_cross=H_anchor  blue_cross=H_refined  arrow=shift",
    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
)
cv2.putText(
    vis,
    f"f{fi}  accepted={len(accepted_dets)}  method={result.method}",
    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
)
cv2.imwrite(
    str(out_path / f"v3_f{fi:05d}_reproject.jpg"),
    vis, [cv2.IMWRITE_JPEG_QUALITY, 92],
)

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
ax.set_title(f"Frame {fi}: prediction shift H_anchor → H_refined")
ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Per-star prediction error table

# %%
shifts = []
for p_ref in preds_refined:
    p_anc = preds_by_tpt.get(p_ref.tpt)
    d = dets_by_tpt.get(p_ref.tpt)
    if p_anc is None or d is None:
        continue
    dx, dy = d.frame_xy_subpix
    err_before = float(np.hypot(p_anc.frame_xy[0] - dx, p_anc.frame_xy[1] - dy))
    err_after = float(np.hypot(p_ref.frame_xy[0] - dx, p_ref.frame_xy[1] - dy))
    shifts.append(dict(
        tpt=p_ref.tpt,
        err_anchor_px=round(err_before, 2),
        err_refined_px=round(err_after, 2),
        improvement_px=round(err_before - err_after, 2),
    ))

if shifts:
    shift_df = pd.DataFrame(shifts).sort_values("tpt")
    print(shift_df.to_string(index=False))
    print(f"\nMedian improvement: {shift_df.improvement_px.median():.2f} px")
    print(f"Mean improvement:   {shift_df.improvement_px.mean():.2f} px")
    print(f"Fraction improved:  "
          f"{(shift_df.improvement_px > 0).mean():.0%}")
else:
    print("No matched (detected + refined) stars to compare.")

cap.release()
print("\nDone.")
