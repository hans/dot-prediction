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
# iterate_homography parameters (Viz 4+)
phase1b_csv = f"results/{subject}/local_star_eval/local_star_eval.csv"
k_max = 2
convergence_px = 0.5
window_size_px = 40  # fallback fixed window when adaptive is also set

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
    iterate_homography,
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

# %% [markdown]
# ## Viz 4 — Full `iterate_homography` evaluation
#
# Runs the complete iterate_homography loop on all 7 eval frames, recording
# per-(frame, mode, iteration_k, tpt) data. Two modes:
#
# - **anchor**: big-star anchor supplied from the global detector.
# - **no_anchor**: no anchor; starts from the corner-only H_v0.
#
# Frames that exit with `"no_predictions"` (e.g. inter-trial gap f2270) produce
# no rows.

# %%
def _project_corners_h(H: np.ndarray) -> np.ndarray:
    sc = SCREEN_CORNERS_NP.astype(np.float64)
    ones = np.ones((4, 1))
    h = (H @ np.hstack([sc, ones]).T).T
    return h[:, :2] / h[:, 2:3]


def _h_delta_px(H_in: np.ndarray, H_out: np.ndarray) -> float:
    proj_in = _project_corners_h(H_in)
    proj_out = _project_corners_h(H_out)
    return float(np.max(np.hypot(*(proj_out - proj_in).T)))


def rows_from_result(fi, expt_t, result, big_star_frame_xy, mode):
    """Flatten an IterationResult into one row per (iteration_k, tpt)."""
    rows = []
    for k, step in enumerate(result.steps):
        small_star_residuals = {}
        for c, res in zip(step.correspondences, step.solve_result.residuals_px):
            if c.source == "small_star":
                small_star_residuals[c.screen_xy] = float(res)

        accepted_by_tpt = {d.source_prediction.tpt: d for d in step.detections}
        rejected_by_tpt = {r.detection.source_prediction.tpt: r.reason
                           for r in step.rejections}
        h_delta = _h_delta_px(step.H_in, step.H_out)
        n_same_blob = sum(1 for r in step.rejections if r.reason == "same_blob")
        n_radius_rej = sum(1 for r in step.rejections if r.reason == "radius_mismatch")

        for pred in step.predictions:
            det = accepted_by_tpt.get(pred.tpt)
            rej_reason = rejected_by_tpt.get(pred.tpt)
            dist_anchor = (
                float(np.hypot(pred.frame_xy[0] - big_star_frame_xy[0],
                               pred.frame_xy[1] - big_star_frame_xy[1]))
                if big_star_frame_xy is not None else np.nan
            )
            if det is not None:
                status = "accepted"
            elif rej_reason is not None:
                status = rej_reason
            else:
                status = "unmatched"
            rows.append(dict(
                frame_idx=fi, expt_t_ms=expt_t, mode=mode, iteration_k=k,
                trial_idx=pred.trial_idx, tpt=pred.tpt,
                age_s=pred.age_s, expected_radius_px=pred.expected_radius_px,
                predicted_frame_x=pred.frame_xy[0], predicted_frame_y=pred.frame_xy[1],
                distance_from_anchor_px=dist_anchor,
                status=status, detected=det is not None,
                detected_frame_x=(det.frame_xy_subpix[0] if det else np.nan),
                detected_frame_y=(det.frame_xy_subpix[1] if det else np.nan),
                confidence=(det.confidence if det else np.nan),
                equivalent_radius_px=(det.equivalent_radius_px if det else np.nan),
                reprojection_error_px=small_star_residuals.get(pred.screen_xy, np.nan),
                h_delta_px=h_delta,
                anchor_less=result.anchor_less, iterations_run=result.iterations_run,
                convergence_reason=result.convergence_reason,
                n_same_blob_rejected=n_same_blob, n_radius_rejected=n_radius_rej,
            ))
    return rows


records4: list[dict] = []
frame_results4: dict = {}

for fi in eval_frames:
    frame = read_frame(fi)
    if frame is None:
        print(f"frame {fi}: read fail"); continue
    expt_t = frame_to_expt_t(fi)
    print(f"\nframe {fi}: expt_t={expt_t:.0f} ms")

    cache_idx = fi - lo
    raw_c = raw_corners_list[cache_idx]
    smo_c = smoothed_corners[cache_idx]
    if smo_c is None:
        print(f"  frame {fi}: no smoothed corners; skipping"); continue

    global_blobs = detect_stars(frame)
    print(f"  global blobs: {len(global_blobs)}")

    # Compute H_v0 from smoothed corners to generate initial predictions for anchor.
    H_v0_for_anchor, _ = cv2.findHomography(
        SCREEN_CORNERS_NP.astype(np.float32),
        smo_c.astype(np.float32),
        method=0,
    )
    preds_init = predicted_positions(expt_t, trials, H_v0_for_anchor)
    anchor = big_star_anchor(preds_init, global_blobs)
    big_screen_xy, big_frame_xy = anchor if anchor else (None, None)
    print(f"  anchor found: {anchor is not None}")

    sc64 = SCREEN_CORNERS_NP.astype(np.float64)

    result_anchor = iterate_homography(
        frame, smo_c, sc64, trials, expt_t,
        raw_corners=raw_c,
        big_star_screen_xy=big_screen_xy,
        big_star_frame_xy=big_frame_xy,
        k_max=k_max, convergence_px=convergence_px, floor=floor,
        window_size_px=window_size_px,
        adaptive_radius_factor=anchor_adaptive_factor,
    )
    print(f"  anchor: iters={result_anchor.iterations_run}, "
          f"reason={result_anchor.convergence_reason}, "
          f"anchor_less={result_anchor.anchor_less}")
    records4.extend(rows_from_result(fi, expt_t, result_anchor, big_frame_xy, "anchor"))

    result_noanchor = iterate_homography(
        frame, smo_c, sc64, trials, expt_t,
        raw_corners=raw_c,
        k_max=k_max, convergence_px=convergence_px, floor=floor,
        window_size_px=window_size_px,
        adaptive_radius_factor=anchor_adaptive_factor,
    )
    print(f"  no_anchor: iters={result_noanchor.iterations_run}, "
          f"reason={result_noanchor.convergence_reason}")
    records4.extend(rows_from_result(fi, expt_t, result_noanchor, big_frame_xy, "no_anchor"))

    frame_results4[fi] = {
        "anchor": result_anchor, "no_anchor": result_noanchor,
        "big_frame_xy": big_frame_xy, "global_blobs": global_blobs,
        "frame": frame, "smo_c": smo_c, "raw_c": raw_c,
    }

eval_df = pd.DataFrame(records4)
csv_path = out_path / "iterative_homography_eval.csv"
eval_df.to_csv(csv_path, index=False)
print(f"\nSaved {len(eval_df)} rows → {csv_path}")


# %% [markdown]
# ## Detection rate by iteration
#
# Within each mode, detection rate should be monotonically non-decreasing across
# iterations. Pooled over all eval frames that had predictions.

# %%
if not eval_df.empty:
    detect_by_iter = (
        eval_df
        .groupby(["mode", "iteration_k"])
        .apply(lambda g: g["detected"].mean(), include_groups=False)
        .rename("detect_rate")
        .reset_index()
    )
    print("Detection rate by (mode, iteration_k):\n")
    print(detect_by_iter.to_string(index=False))

    print("\nDetection rate per frame per iteration (anchor mode):\n")
    anchor_df = eval_df[eval_df["mode"] == "anchor"]
    if not anchor_df.empty:
        print(anchor_df.groupby(["frame_idx", "iteration_k"])
              .apply(lambda g: g["detected"].mean(), include_groups=False)
              .rename("detect_rate")
              .unstack("iteration_k")
              .to_string())
else:
    print("No data rows recorded (all frames had no predictions).")


# %% [markdown]
# ## Detection rate vs distance from anchor
#
# The key Phase 1b failure: under H_rough_anchor, predictions far from the anchor
# accumulated perspective residual. After full-perspective refinement this should
# be flat. Pooled over anchor-mode, non-anchor-less frames.

# %%
df_anchored = eval_df[
    (eval_df["mode"] == "anchor")
    & (~eval_df["anchor_less"])
    & eval_df["distance_from_anchor_px"].notna()
].copy()

dist_bins = [0, 50, 100, 150, 200, 300, 600]
dist_labels = ["0–50", "50–100", "100–150", "150–200", "200–300", "300+"]

if not df_anchored.empty:
    df_anchored["dist_bucket"] = pd.cut(
        df_anchored["distance_from_anchor_px"],
        bins=dist_bins, labels=dist_labels, right=False,
    )
    dist_rate = (
        df_anchored
        .groupby(["iteration_k", "dist_bucket"], observed=False)["detected"]
        .agg(["sum", "size"])
        .assign(rate=lambda d: d["sum"] / d["size"].clip(lower=1))
    )
    print("Detection rate by (iteration_k, distance_from_anchor bucket):\n")
    print(dist_rate.to_string())

    n_iters = df_anchored["iteration_k"].nunique()
    if n_iters > 0:
        fig, axes = plt.subplots(1, n_iters, figsize=(5 * n_iters, 4), sharey=True)
        if n_iters == 1:
            axes = [axes]
        for ax, (k, grp) in zip(axes, dist_rate.groupby("iteration_k")):
            grp_by_bucket = grp.droplevel("iteration_k")
            ax.bar(range(len(grp_by_bucket)), grp_by_bucket["rate"],
                   tick_label=grp_by_bucket.index.tolist())
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Distance from anchor (frame px)")
            ax.set_ylabel("Detection rate")
            ax.set_title(f"Iteration k={k}")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Detection rate vs distance from anchor (anchor mode)")
        plt.tight_layout()
        plt.savefig(str(out_path / "v4_detect_rate_vs_distance.png"), dpi=100)
        plt.show()
else:
    print("No anchor-mode, non-anchor-less rows with distance data.")


# %% [markdown]
# ## Same-blob-snapping rate per iteration
#
# Count of `"same_blob"` rejections per frame per iteration. Should drop between
# iterations as predictions converge onto distinct blobs under the improved H.

# %%
if not eval_df.empty:
    snap_df = (
        eval_df[["frame_idx", "mode", "iteration_k",
                  "n_same_blob_rejected", "n_radius_rejected"]]
        .drop_duplicates(subset=["frame_idx", "mode", "iteration_k"])
        .sort_values(["mode", "frame_idx", "iteration_k"])
    )
    print("Same-blob + radius rejections per (frame, mode, iteration_k):\n")
    print(snap_df.to_string(index=False))

    snap_pivot = (
        snap_df
        .groupby(["mode", "iteration_k"])[["n_same_blob_rejected", "n_radius_rejected"]]
        .mean()
        .round(2)
    )
    print("\nMean rejections per frame by (mode, iteration_k):\n")
    print(snap_pivot.to_string())


# %% [markdown]
# ## Homography reprojection error
#
# Per-correspondence reprojection error for accepted small-star correspondences.
# Lower is better; <2 px median is the target.

# %%
reproj_df = eval_df[
    (eval_df["status"] == "accepted")
    & eval_df["reprojection_error_px"].notna()
]

if not reproj_df.empty:
    reproj_stats = (
        reproj_df
        .groupby(["mode", "iteration_k"])["reprojection_error_px"]
        .agg(["median", lambda x: x.quantile(0.95)])
        .rename(columns={"median": "median_px", "<lambda_0>": "p95_px"})
        .round(3)
    )
    print("Reprojection error — accepted small-star correspondences:\n")
    print(reproj_stats.to_string())

    groups = list(reproj_df.groupby(["mode", "iteration_k"]))
    labels_box = [f"{mode}\nk={k}" for (mode, k), _ in groups]
    data_box = [grp["reprojection_error_px"].dropna().values for _, grp in groups]

    fig, ax = plt.subplots(figsize=(max(6, len(labels_box) * 1.5 + 1), 4))
    ax.boxplot(data_box, labels=labels_box, showfliers=False)
    ax.set_ylabel("Reprojection error (frame px)")
    ax.set_title("Small-star reprojection error per mode × iteration")
    ax.axhline(2.0, color="tomato", linestyle="--", linewidth=1, label="2 px target")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path / "v4_reprojection_error.png"), dpi=100)
    plt.show()
else:
    print("No accepted small-star correspondences with reprojection data.")


# %% [markdown]
# ## Side-by-side with Phase 1b
#
# Phase 1b's best mode is `smoothed+anchor`; Phase 1c uses the final iteration
# of the `anchor` mode. Compared per (frame_idx, tpt).

# %%
from pathlib import Path as _Path

_p1b_path = _Path(phase1b_csv)
if _p1b_path.exists() and not eval_df.empty:
    p1b = pd.read_csv(_p1b_path)
    p1b_best = (
        p1b[p1b["H_rough_mode"] == "smoothed+anchor"][["frame_idx", "tpt", "source"]]
        .assign(detected_1b=lambda d: d["source"] == "local")
    )

    # Phase 1c final iteration per frame (transform avoids index ambiguity)
    _anchor = eval_df[eval_df["mode"] == "anchor"].copy()
    _max_k = _anchor.groupby("frame_idx")["iteration_k"].transform("max")
    p1c_final = (
        _anchor[_anchor["iteration_k"] == _max_k]
        [["frame_idx", "tpt", "detected"]]
        .rename(columns={"detected": "detected_1c"})
    )

    comparison = p1b_best.merge(p1c_final, on=["frame_idx", "tpt"], how="outer")

    print("Detection rate comparison — Phase 1b (smoothed+anchor) vs Phase 1c (final):\n")
    summary = (
        comparison
        .groupby("frame_idx")[["detected_1b", "detected_1c"]]
        .mean()
        .round(3)
        .assign(delta=lambda d: (d["detected_1c"] - d["detected_1b"]).round(3))
    )
    print(summary.to_string())
    print(f"\nOverall — Phase 1b: {comparison['detected_1b'].mean():.3f}, "
          f"Phase 1c: {comparison['detected_1c'].mean():.3f}")
else:
    print(f"Phase 1b CSV not found at {phase1b_csv}; skipping comparison.")


# %% [markdown]
# ## Per-iteration overlays
#
# One JPEG per eval frame with anchor-mode iterations stacked vertically.
# Per panel:
# - **Gray rectangle**: adaptive search window.
# - **Colored cross**: predicted position (blue=fresh, red=old by age).
# - **Green circle**: accepted detection.
# - **Orange circle**: rejected detection (label = reason abbreviation).
# - **Cyan circle**: global big-star blob.
# - Header text: frame, mode, iteration k, detection count, H-delta.

# %%
def _draw_iter_panel(
    frame: np.ndarray,
    step,
    big_frame_xy,
    global_blobs,
    label: str,
) -> np.ndarray:
    vis = frame.copy()
    accepted_by_tpt = {d.source_prediction.tpt: d for d in step.detections}
    rejected_by_tpt = {r.detection.source_prediction.tpt: r for r in step.rejections}

    for pred in step.predictions:
        cx, cy = int(round(pred.frame_xy[0])), int(round(pred.frame_xy[1]))
        age_norm = min(pred.age_s / 60.0, 1.0)
        cross_col = (int(255 * (1 - age_norm)), 30, int(255 * age_norm))

        # Window outline (adaptive if possible)
        if anchor_adaptive_factor is not None and pred.expected_radius_px > 0:
            w = int(np.clip(
                anchor_adaptive_factor * pred.expected_radius_px,
                anchor_min_window_px, anchor_max_window_px,
            ))
        else:
            w = window_size_px
        half = w // 2
        cv2.rectangle(vis, (cx - half, cy - half), (cx + half, cy + half),
                      (70, 70, 70), 1)
        cv2.drawMarker(vis, (cx, cy), cross_col, cv2.MARKER_TILTED_CROSS, 12, 1)

        det = accepted_by_tpt.get(pred.tpt)
        rej = rejected_by_tpt.get(pred.tpt)
        if det is not None:
            fx, fy = int(round(det.frame_xy_subpix[0])), int(round(det.frame_xy_subpix[1]))
            cv2.circle(vis, (fx, fy), 8, (0, 220, 0), 2)
        elif rej is not None:
            fx = int(round(rej.detection.frame_xy_subpix[0]))
            fy = int(round(rej.detection.frame_xy_subpix[1]))
            cv2.circle(vis, (fx, fy), 8, (0, 140, 255), 2)
            cv2.putText(vis, rej.reason[:2], (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1)
        else:
            cv2.line(vis, (cx - 5, cy - 5), (cx + 5, cy + 5), (50, 50, 200), 1)
            cv2.line(vis, (cx - 5, cy + 5), (cx + 5, cy - 5), (50, 50, 200), 1)

    for bx, by, br in global_blobs:
        cv2.circle(vis, (int(bx), int(by)), max(int(br), 5), (255, 220, 0), 2)

    n_det = len(accepted_by_tpt)
    n_pred = len(step.predictions)
    h_delta = _h_delta_px(step.H_in, step.H_out)
    cv2.putText(vis, label, (14, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(vis, f"det={n_det}/{n_pred}  Hdelta={h_delta:.2f}px",
                (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 100), 2)
    return vis


for fi, fr_data in frame_results4.items():
    fr = fr_data["frame"]
    big_frame_xy = fr_data["big_frame_xy"]
    global_blobs = fr_data["global_blobs"]

    panels = []
    for mode_key, mode_label in [("anchor", "anchor"), ("no_anchor", "no_anchor")]:
        result = fr_data[mode_key]
        if not result.steps:
            blank = np.full_like(fr, 25)
            cv2.putText(blank, f"f{fi}  {mode_label}: {result.convergence_reason}",
                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
            panels.append(blank)
            continue
        for k, step in enumerate(result.steps):
            label = f"f{fi}  {mode_label}  k={k}  [{result.convergence_reason}]"
            panels.append(_draw_iter_panel(fr, step, big_frame_xy, global_blobs, label))

    if panels:
        cv2.imwrite(
            str(out_path / f"v4_f{fi:05d}_iters.jpg"),
            np.vstack(panels),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

print(f"Overlays written to {out_path}/v4_f*_iters.jpg")


# %% [markdown]
# ### Inline: iteration overlays

# %%
for p in sorted(out_path.glob("v4_f*_iters.jpg")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / img.shape[1]))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(p.name)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


cap.release()
print("\nDone.")
