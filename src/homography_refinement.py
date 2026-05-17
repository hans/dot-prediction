"""Refinement of a rough screen→frame homography.

Phase 1b discovered that the screen-corner detector consistently overshoots
the iPad's right edge on EC347 (likely a glare patch beyond the bezel),
producing a systematic ~140 px x-offset in the predicted star positions —
both for raw and rolling-median-smoothed corners. A single known
correspondence (typically the freshly-revealed big star, detected by the
Phase 1a global detector) can collapse the bulk of that error via a pure
translation: ``anchor_translate``.

Phase 1c re-solves the full homography from a weighted correspondence set
(corners + big star + confirmed small stars), iterating predict → detect →
re-solve until convergence. This module provides the building blocks:

* ``Correspondence`` — one (screen_xy, frame_xy, weight, source) tuple.
* ``solve_weighted_homography`` — weighted-DLT/RANSAC re-solve via the
  repeated-points trick, returning H plus per-correspondence residuals
  and an inlier mask.
* ``apply_quality_gates`` — radius-match + same-blob-snapping filter over
  ``LocalDetection`` outputs (with ``radius_match_ok`` and
  ``resolve_blob_conflicts`` as the per-gate helpers).
* ``build_correspondences`` — assembles the weighted correspondence list
  from smoothed/raw corners, a big-star anchor, and quality-gated small-
  star detections per the Change-1 weighting table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

try:
    from local_star_detector import (
        LocalDetection,
        _window_bounds,
        _window_size_for,
    )
    from predicted_positions import PredictedStar
except ImportError:
    from src.local_star_detector import (
        LocalDetection,
        _window_bounds,
        _window_size_for,
    )
    from src.predicted_positions import PredictedStar

CorrespondenceSource = Literal[
    "corner_smoothed",
    "corner_raw",
    "big_star",
    "small_star",
]


@dataclass(frozen=True)
class Correspondence:
    """One screen↔frame point pairing fed to the homography re-solver.

    Attributes:
        screen_xy: iPad screen-pixel coordinate (x, y).
        frame_xy: Scene-video frame-pixel coordinate (x, y).
        weight: Trust weight in [0, 1]. Materialised as repeated rows in
            the DLT system by ``solve_weighted_homography``.
        source: Tag for provenance / debugging — see CorrespondenceSource.
    """

    screen_xy: tuple[float, float]
    frame_xy: tuple[float, float]
    weight: float
    source: CorrespondenceSource


def anchor_translate(
    H: np.ndarray,
    anchor_screen_xy: tuple[float, float],
    anchor_frame_xy: tuple[float, float],
) -> np.ndarray:
    """Translate ``H`` so the anchor projects exactly to its known frame xy.

    Args:
        H: 3×3 homography mapping iPad screen-px → frame-px.
        anchor_screen_xy: Known screen-pixel position of a star.
        anchor_frame_xy: Known frame-pixel detection of the same star.

    Returns:
        New 3×3 homography ``H'`` such that
        ``H' @ [sx, sy, 1]ᵀ`` projects to ``anchor_frame_xy`` (up to numerical
        precision). All other points are shifted by the same translation
        ``(detected − predicted)``.
    """
    H = np.asarray(H, dtype=np.float64)
    sx, sy = anchor_screen_xy
    fx_target, fy_target = anchor_frame_xy
    h = H @ np.array([sx, sy, 1.0])
    fx_pred, fy_pred = h[0] / h[2], h[1] / h[2]
    dx, dy = fx_target - fx_pred, fy_target - fy_pred

    # Post-multiply translation in frame space: T @ H, so that for any input
    # (sx, sy), the projection becomes (H @ pt) + (dx, dy). A simple add to
    # H[:2, 2] is only correct for affine H (h31 = h32 = 0, h33 = 1).
    T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return T @ H


@dataclass(frozen=True)
class SolveResult:
    """Output of ``solve_weighted_homography``.

    Attributes:
        H: 3×3 homography mapping screen → frame.
        inlier_mask: Bool array of length ``len(correspondences)``. True
            where the correspondence was an inlier under RANSAC, all True
            when the solver fell back to least-squares.
        residuals_px: Float array of length ``len(correspondences)`` giving
            the Euclidean reprojection error in frame pixels (``||frame_xy −
            H @ screen_xy||``) per input correspondence.
        method: ``"ransac"`` or ``"lstsq"``.
    """

    H: np.ndarray
    inlier_mask: np.ndarray
    residuals_px: np.ndarray
    method: Literal["ransac", "lstsq"]


def _project_many(H: np.ndarray, screen_xy: np.ndarray) -> np.ndarray:
    """Project (N, 2) screen points through 3×3 ``H``; return (N, 2)."""
    ones = np.ones((len(screen_xy), 1), dtype=np.float64)
    h = (H @ np.hstack([screen_xy, ones]).T).T
    return h[:, :2] / h[:, 2:3]


def solve_weighted_homography(
    correspondences: list[Correspondence],
    *,
    weight_replication: int = 10,
    use_ransac: bool = False,
    ransac_threshold_px: float = 3.0,
) -> SolveResult:
    """Solve for screen→frame H from a weighted correspondence set.

    Weights are realised via the **repeated-points trick**: each correspondence
    contributes ``max(1, round(weight × weight_replication))`` rows to the DLT
    system. This is the spec's "simple first cut"; a hand-rolled weighted DLT
    can replace it later if continuous weights matter.

    Least-squares (``method=0``) is the default. Pre-flight check 1 shows
    that under pure Gaussian noise — the expected post-gate regime — lstsq
    has 3–12× lower p95 error than RANSAC at σ≥1 px. Set ``use_ransac=True``
    only if eval surfaces correspondences that survived the quality gates and
    destabilised H (see spec Risk section).

    Args:
        correspondences: Non-empty list of ``Correspondence``. Entries with
            ``weight <= 0`` are silently dropped.
        weight_replication: Multiplier converting continuous weights into
            integer row replication counts (default 10 → weights of 0.1
            granularity).
        use_ransac: If True, use ``cv2.RANSAC`` with ``ransac_threshold_px``.
            Default False (lstsq).
        ransac_threshold_px: Frame-pixel reprojection threshold for RANSAC.
            Ignored when ``use_ransac=False``.

    Returns:
        ``SolveResult`` with H, inlier mask aligned to the input list, and
        per-correspondence reprojection residuals (px).

    Raises:
        ValueError: Fewer than 4 usable (positive-weight) correspondences,
            or ``cv2.findHomography`` returned no solution.
    """
    usable = [c for c in correspondences if c.weight > 0]
    if len(usable) < 4:
        raise ValueError(
            f"solve_weighted_homography needs ≥4 positive-weight "
            f"correspondences; got {len(usable)}."
        )

    # Replicate each correspondence proportionally to its weight. Floor at 1
    # row so even very low-weight points contribute, matching the spec's
    # intent that low-weight ≠ excluded.
    repeated_screen: list[tuple[float, float]] = []
    repeated_frame: list[tuple[float, float]] = []
    origin_idx: list[int] = []
    for i, c in enumerate(usable):
        reps = max(1, int(round(c.weight * weight_replication)))
        repeated_screen.extend([c.screen_xy] * reps)
        repeated_frame.extend([c.frame_xy] * reps)
        origin_idx.extend([i] * reps)

    src = np.asarray(repeated_screen, dtype=np.float64)
    dst = np.asarray(repeated_frame, dtype=np.float64)

    if use_ransac:
        H, mask = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
        )
        method: Literal["ransac", "lstsq"] = "ransac"
    else:
        H, mask = cv2.findHomography(src.astype(np.float32),
                                     dst.astype(np.float32), method=0)
        method = "lstsq"

    if H is None:
        raise ValueError("cv2.findHomography returned no solution.")
    H = np.asarray(H, dtype=np.float64)

    # Fold the per-replicated-row inlier mask back to per-unique correspondence:
    # since all replicas of a correspondence share the same residual, any
    # disagreement is just a numerical edge case at the RANSAC threshold —
    # mark "any replica is inlier" as inlier for that correspondence.
    n = len(usable)
    inlier_mask = np.zeros(n, dtype=bool)
    if mask is None or not use_ransac:
        inlier_mask[:] = True
    else:
        mask_flat = mask.ravel().astype(bool)
        for row, oi in enumerate(origin_idx):
            if mask_flat[row]:
                inlier_mask[oi] = True

    # Per-correspondence reprojection residual in frame pixels.
    screen_unique = np.array([c.screen_xy for c in usable], dtype=np.float64)
    frame_unique = np.array([c.frame_xy for c in usable], dtype=np.float64)
    projected = _project_many(H, screen_unique)
    residuals = np.hypot(*(projected - frame_unique).T)

    # Pad the inlier mask / residuals back to the original input length
    # (entries with weight <= 0 → not inlier, NaN residual).
    full_inlier = np.zeros(len(correspondences), dtype=bool)
    full_residuals = np.full(len(correspondences), np.nan, dtype=np.float64)
    usable_idx = [i for i, c in enumerate(correspondences) if c.weight > 0]
    for k, oi in enumerate(usable_idx):
        full_inlier[oi] = inlier_mask[k]
        full_residuals[oi] = residuals[k]

    return SolveResult(
        H=H, inlier_mask=full_inlier, residuals_px=full_residuals, method=method
    )


# ---------------------------------------------------------------------------
# Quality gates on local-detector outputs (Phase 1c Step 2)
# ---------------------------------------------------------------------------

GateRejectionReason = Literal["radius_mismatch", "same_blob"]


@dataclass(frozen=True)
class GateRejection:
    """One detection that was filtered out by ``apply_quality_gates``.

    Attributes:
        detection: The ``LocalDetection`` that was rejected.
        reason: Which gate dropped it — ``"radius_mismatch"`` (equivalent
            radius too far from expected) or ``"same_blob"`` (clustered
            with another detection and lost the conflict-resolver tiebreak).
    """

    detection: LocalDetection
    reason: GateRejectionReason


def radius_match_ok(
    detection: LocalDetection,
    *,
    tau_radius: float = 1.5,
) -> bool:
    """True if the detection's equivalent radius matches its prediction.

    Uses the same relative-error formulation as Change 1's downweight rule:
    accept iff ``|equivalent_radius − expected_radius| / expected_radius ≤
    tau_radius``. With ``tau_radius=1.5`` this admits observed radii in
    ``[0, 2.5 × expected]`` (the spec's "factor of ~2.5"). Predictions with
    ``expected_radius_px ≤ 0`` skip the check (no model available).

    The local detector already applies a generous ``max_radius_factor=4.0``
    one-sided hard reject; this gate is the tighter per-frame filter that
    feeds the homography re-solver.
    """
    expected = detection.source_prediction.expected_radius_px
    if expected <= 0:
        return True
    rel_err = abs(detection.equivalent_radius_px - expected) / expected
    return rel_err <= tau_radius


def _cluster_indices_by_distance(
    points: np.ndarray, tau_px: float,
) -> list[list[int]]:
    """Single-link cluster (N, 2) ``points`` by Euclidean distance < tau_px.

    Returns a list of clusters, each a list of row indices into ``points``.
    """
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    tau_sq = tau_px * tau_px
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            if dx * dx + dy * dy < tau_sq:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def resolve_blob_conflicts(
    detections: list[LocalDetection],
    *,
    tau_centroid_px: float = 3.0,
) -> tuple[list[LocalDetection], list[LocalDetection]]:
    """Collapse same-blob-snapping clusters down to one detection each.

    Two detections belong to the same cluster if their ``frame_xy_subpix``
    are within ``tau_centroid_px`` (single-link). Within each cluster of
    size ≥ 2, the winner is the detection whose **source prediction's**
    ``frame_xy`` is closest to the cluster centroid (mean of the
    detections' sub-pixel centroids). All other detections in the cluster
    are rejected — per spec, no secondary-peak search is attempted in the
    same window.

    Args:
        detections: Local-detector outputs to filter.
        tau_centroid_px: Clustering distance threshold (default 3 px per
            the spec's τ_centroid).

    Returns:
        ``(accepted, rejected)`` partitioning the input list. Singleton
        clusters always end up in ``accepted``. Order within each list is
        the order of first appearance in ``detections``.
    """
    if not detections:
        return [], []

    centroids = np.array(
        [d.frame_xy_subpix for d in detections], dtype=np.float64,
    )
    clusters = _cluster_indices_by_distance(centroids, tau_centroid_px)

    accepted_idx: set[int] = set()
    for cluster in clusters:
        if len(cluster) == 1:
            accepted_idx.add(cluster[0])
            continue
        cx, cy = centroids[cluster].mean(axis=0)
        # Tiebreak by smallest squared distance from prediction to centroid.
        def dist_sq(i: int, cx: float = cx, cy: float = cy) -> float:
            px, py = detections[i].source_prediction.frame_xy
            return (px - cx) ** 2 + (py - cy) ** 2
        winner = min(cluster, key=dist_sq)
        accepted_idx.add(winner)

    accepted = [d for i, d in enumerate(detections) if i in accepted_idx]
    rejected = [d for i, d in enumerate(detections) if i not in accepted_idx]
    return accepted, rejected


def apply_quality_gates(
    detections: list[LocalDetection],
    *,
    tau_centroid_px: float = 3.0,
    tau_radius: float = 1.5,
) -> tuple[list[LocalDetection], list[GateRejection]]:
    """Filter local detections through Phase-1c quality gates.

    Gates, applied in order:

    1. **Radius match** (``radius_match_ok``) — drop detections whose
       equivalent radius is more than ``tau_radius`` away from the
       prediction's expected radius in relative-error terms.
    2. **Same-blob conflict** (``resolve_blob_conflicts``) — collapse
       clusters of detections within ``tau_centroid_px`` to a single
       winner (the one whose prediction is closest to the cluster
       centroid).

    Running radius first avoids the case where a radius-bad detection
    wins the centroid tiebreak over a radius-good one in the same
    cluster.

    Returns:
        ``(accepted, rejections)``. ``rejections`` tags each dropped
        detection with the gate that filtered it for diagnostics.
    """
    rejections: list[GateRejection] = []
    radius_ok: list[LocalDetection] = []
    for d in detections:
        if radius_match_ok(d, tau_radius=tau_radius):
            radius_ok.append(d)
        else:
            rejections.append(GateRejection(d, "radius_mismatch"))

    accepted, blob_rejected = resolve_blob_conflicts(
        radius_ok, tau_centroid_px=tau_centroid_px,
    )
    rejections.extend(GateRejection(d, "same_blob") for d in blob_rejected)
    return accepted, rejections


# ---------------------------------------------------------------------------
# Greedy constellation matcher (Phase 1c Step 3)
# ---------------------------------------------------------------------------


def _group_overlapping_windows(
    bounds: list[tuple[int, int, int, int]],
) -> list[list[int]]:
    """Single-link cluster of axis-aligned window rects by overlap."""
    n = len(bounds)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        x0i, y0i, x1i, y1i = bounds[i]
        for j in range(i + 1, n):
            x0j, y0j, x1j, y1j = bounds[j]
            if x0i < x1j and x0j < x1i and y0i < y1j and y0j < y1i:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _greedy_assign_in_union(
    redness: np.ndarray,
    group_preds: list[PredictedStar],
    group_bounds: list[tuple[int, int, int, int]],
    *,
    floor: float,
    max_radius_factor: float,
) -> tuple[list[LocalDetection], list[PredictedStar]]:
    """Find connected components in the union of windows, greedy-assign.

    Each component becomes a candidate blob; in descending-confidence order
    the blob is bound to the nearest unspent prediction (by centroid distance
    to ``prediction.frame_xy``) whose rectangular window contains the blob's
    centroid. A blob too large under the assigned prediction's radius model
    (``equivalent_radius > max_radius_factor × expected``) is rejected; the
    prediction is then left unmatched and no fallback blob is tried.
    """
    ux0 = min(b[0] for b in group_bounds)
    uy0 = min(b[1] for b in group_bounds)
    ux1 = max(b[2] for b in group_bounds)
    uy1 = max(b[3] for b in group_bounds)

    redness_crop = redness[uy0:uy1, ux0:ux1]
    h_c, w_c = redness_crop.shape
    in_union = np.zeros((h_c, w_c), dtype=bool)
    for x0, y0, x1, y1 in group_bounds:
        in_union[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0] = True

    mask = ((redness_crop > floor) & in_union).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    blobs: list[tuple[float, float, float, int, int, float]] = []
    for lbl in range(1, n_labels):
        ys, xs = np.nonzero(labels == lbl)
        weights = redness_crop[ys, xs]
        total = weights.sum()
        cx_local = float((xs * weights).sum() / total)
        cy_local = float((ys * weights).sum() / total)
        peak_idx = int(np.argmax(weights))
        confidence = float(weights[peak_idx])
        radius = float(np.sqrt(len(xs) / np.pi))
        blobs.append((
            confidence,
            ux0 + cx_local,
            uy0 + cy_local,
            ux0 + int(xs[peak_idx]),
            uy0 + int(ys[peak_idx]),
            radius,
        ))

    blobs.sort(key=lambda b: b[0], reverse=True)

    spent: set[int] = set()
    matched: dict[int, LocalDetection] = {}
    for confidence, cx, cy, peak_x, peak_y, radius in blobs:
        candidates: list[tuple[float, int]] = []
        for i, (x0, y0, x1, y1) in enumerate(group_bounds):
            if i in spent:
                continue
            if x0 <= cx < x1 and y0 <= cy < y1:
                px, py = group_preds[i].frame_xy
                candidates.append(((px - cx) ** 2 + (py - cy) ** 2, i))
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[0])
        _, idx = candidates[0]
        spent.add(idx)
        pred = group_preds[idx]
        if (
            pred.expected_radius_px > 0
            and radius > max_radius_factor * pred.expected_radius_px
        ):
            continue
        matched[idx] = LocalDetection(
            frame_xy_subpix=(cx, cy),
            confidence=confidence,
            equivalent_radius_px=radius,
            peak_xy=(peak_x, peak_y),
            source_prediction=pred,
        )

    detections = [matched[i] for i in range(len(group_preds)) if i in matched]
    unmatched = [p for i, p in enumerate(group_preds) if i not in matched]
    return detections, unmatched


def detect_constellation(
    frame: np.ndarray,
    predictions: list[PredictedStar],
    *,
    window_size_px: int = 40,
    floor: float = 20.0,
    max_radius_factor: float = 4.0,
    adaptive_radius_factor: float | None = None,
    min_window_px: int = 10,
    max_window_px: int = 60,
) -> tuple[list[LocalDetection], list[PredictedStar]]:
    """Position-conditioned local detection with greedy constellation matching.

    Drop-in replacement for ``local_star_detector.detect_in_windows`` that
    handles overlapping search windows: when two or more predictions' windows
    overlap, blobs in the union are extracted as connected components and
    greedy-assigned to predictions in descending-confidence order (each blob
    to the nearest unspent prediction whose window contains it).

    Compared to ``detect_in_windows`` this fixes the per-frame failure mode
    where N predictions snap to the same shared blob (one detection per
    prediction, all on the same peak). Independently it also fixes the rare
    case where a single non-overlapping window contains two distinct above-
    floor components — ``detect_in_windows`` averages their centroids; this
    function picks the highest-confidence component.

    Same-blob conflicts are now prevented at the source, but
    ``resolve_blob_conflicts`` is still useful downstream as a safety net for
    distinct blobs whose centroids happen to land within ``τ_centroid_px`` of
    each other across different overlap groups.

    Args:
        frame: BGR uint8 image, shape (H, W, 3).
        predictions: List of ``PredictedStar`` from ``predicted_positions``.
        window_size_px: Fixed square window side length (px), used when
            ``adaptive_radius_factor`` is None.
        floor: Minimum R−B opponent-channel value for a pixel to count as
            part of a blob.
        max_radius_factor: A blob is rejected after assignment when its
            equivalent radius exceeds this × the assigned prediction's
            ``expected_radius_px``. The prediction is then left unmatched
            (no fallback to a smaller blob in the same union).
        adaptive_radius_factor: When set, each prediction's window is
            ``factor × expected_radius_px`` clipped to
            ``[min_window_px, max_window_px]``.
        min_window_px: Lower clip on adaptive window size.
        max_window_px: Upper clip on adaptive window size.

    Returns:
        ``(detections, unmatched)`` partitioning ``predictions``: detections
        are in input-prediction order; off-frame and unmatched predictions
        accumulate in ``unmatched`` (also in input order). A prediction whose
        window is fully off-frame is reported as unmatched without any
        detection attempt.
    """
    if not predictions:
        return [], []

    h_img, w_img = frame.shape[:2]
    redness = (
        frame[:, :, 2].astype(np.float32) - frame[:, :, 0].astype(np.float32)
    )

    on_frame_bounds: list[tuple[int, int, int, int]] = []
    on_frame_preds: list[PredictedStar] = []
    off_frame_preds: list[PredictedStar] = []
    for pred in predictions:
        cx, cy = pred.frame_xy
        win = _window_size_for(
            pred, window_size_px, adaptive_radius_factor,
            min_window_px, max_window_px,
        )
        b = _window_bounds(cx, cy, win, w_img, h_img)
        if b is None:
            off_frame_preds.append(pred)
        else:
            on_frame_preds.append(pred)
            on_frame_bounds.append(b)

    groups = _group_overlapping_windows(on_frame_bounds)

    detections: list[LocalDetection] = []
    unmatched: list[PredictedStar] = list(off_frame_preds)
    for group in groups:
        group_preds = [on_frame_preds[i] for i in group]
        group_bounds = [on_frame_bounds[i] for i in group]
        d, u = _greedy_assign_in_union(
            redness, group_preds, group_bounds,
            floor=floor, max_radius_factor=max_radius_factor,
        )
        detections.extend(d)
        unmatched.extend(u)

    order = {id(p): i for i, p in enumerate(predictions)}
    detections.sort(key=lambda d: order[id(d.source_prediction)])
    unmatched.sort(key=lambda p: order[id(p)])
    return detections, unmatched


# ---------------------------------------------------------------------------
# Correspondence builder (Phase 1c Step 4)
# ---------------------------------------------------------------------------

# Corner indices follow screen_detection.detect_corners: [TL, TR, BR, BL].
_CORNER_TL, _CORNER_TR, _CORNER_BR, _CORNER_BL = 0, 1, 2, 3


def build_correspondences(
    smoothed_corners: np.ndarray,
    screen_corners: np.ndarray,
    *,
    raw_corners: np.ndarray | None = None,
    big_star_screen_xy: tuple[float, float] | None = None,
    big_star_frame_xy: tuple[float, float] | None = None,
    big_star_weight: float = 1.0,
    small_stars: list[LocalDetection] | None = None,
    raw_bl_br_tau_px: float = 10.0,
    raw_tl_tr_tau_px: float = 5.0,
    small_star_radius_blend_tau: float = 0.5,
    small_star_tau_radius: float = 1.5,
    small_star_confidence_scale: float = 255.0,
) -> list[Correspondence]:
    """Build the weighted correspondence set for the homography re-solver.

    Implements the Change-1 weighting table from the Phase-1c spec:

    +----------------------------+--------+----------------------------------------------+
    | Source                     | Weight | Conditions                                   |
    +============================+========+==============================================+
    | Smoothed BL, BR            | 1.0    | Always included.                             |
    +----------------------------+--------+----------------------------------------------+
    | Raw BL, BR                 | 0.5    | |raw − smoothed| < ``raw_bl_br_tau_px``.     |
    +----------------------------+--------+----------------------------------------------+
    | Smoothed TL, TR            | 0.3    | |raw − smoothed| < ``raw_tl_tr_tau_px``.     |
    +----------------------------+--------+----------------------------------------------+
    | Raw TL, TR                 | 0.1    | |raw − smoothed| < ``raw_tl_tr_tau_px``.     |
    +----------------------------+--------+----------------------------------------------+
    | Big star                   | caller | Both ``big_star_*`` args provided.           |
    +----------------------------+--------+----------------------------------------------+
    | Small stars (quality-gated)| varies | conf / scale × radius-match factor.          |
    +----------------------------+--------+----------------------------------------------+

    Corner ordering follows ``screen_detection.detect_corners``: [TL, TR, BR, BL].

    **Top-corner gating:** when ``raw_corners`` is absent (detection failed for
    this frame), top corners are omitted entirely — their quality cannot be
    assessed from the smoothed value alone. Bottom corners are always included at
    weight 1.0 regardless (spec: "Always included").

    **Small-star weights:** each detection's ``confidence`` (raw R−B peak value)
    is divided by ``small_star_confidence_scale`` (default 255) to normalise to
    [0, 1], then multiplied by a radius-match factor that is 1.0 for
    ``rel_err ≤ small_star_radius_blend_tau`` and linearly drops to 0.5 at
    ``small_star_tau_radius`` (matching the quality-gate cutoff).

    Args:
        smoothed_corners: (4, 2) float array [TL, TR, BR, BL] of rolling-median
            smoothed corner positions in frame pixels. Always required.
        screen_corners: (4, 2) float array [TL, TR, BR, BL] of corner positions
            in iPad screen pixels (typically the four screen-edge corners).
        raw_corners: Optional (4, 2) array of raw (unsmoothed) corner frame
            coordinates in the same [TL, TR, BR, BL] order. When *None*, raw
            correspondences are not added and top corners are omitted.
        big_star_screen_xy: iPad screen-pixel position of the anchor (big) star.
            If *None*, no big-star correspondence is added.
        big_star_frame_xy: Detected frame-pixel position of the big star.
        big_star_weight: Weight for the big-star correspondence (default 1.0).
        small_stars: Quality-gated ``LocalDetection`` outputs. Pass the list
            returned by ``apply_quality_gates``.
        raw_bl_br_tau_px: |raw − smoothed| threshold (px) below which raw BL/BR
            are added at weight 0.5. Detections ≥ this value are excluded
            (default 10 px per pre-flight check 2).
        raw_tl_tr_tau_px: |raw − smoothed| threshold (px) below which TL/TR
            correspondences (both smoothed and raw) are included. At or above
            this value both are dropped (default 5 px per pre-flight check 2).
        small_star_radius_blend_tau: Relative-error threshold below which small
            stars receive full weight (default 0.5); above this the weight is
            linearly blended down to 0.5× at ``small_star_tau_radius``.
        small_star_tau_radius: Upper bound of the linear blend range — should
            match the ``tau_radius`` passed to ``apply_quality_gates`` (default
            1.5). Detections above this would already have been filtered out by
            the gate; the parameter exists so the blend endpoint tracks the gate
            when both are tuned together.
        small_star_confidence_scale: Divisor applied to each detection's raw R−B
            confidence before weighting (default 255, the channel maximum).

    Returns:
        List of ``Correspondence`` objects ready for ``solve_weighted_homography``,
        ordered: bottom corners → top corners → big star → small stars.
    """
    corrs: list[Correspondence] = []

    # --- Bottom corners (BL, BR) ---
    # Smoothed: always at weight 1.0 (spec: "Always included").
    # Raw: at weight 0.5 when within raw_bl_br_tau_px of smoothed.
    for idx in (_CORNER_BL, _CORNER_BR):
        s_xy = (float(screen_corners[idx, 0]), float(screen_corners[idx, 1]))
        corrs.append(Correspondence(
            screen_xy=s_xy,
            frame_xy=(float(smoothed_corners[idx, 0]), float(smoothed_corners[idx, 1])),
            weight=1.0,
            source="corner_smoothed",
        ))
        if raw_corners is not None:
            delta = float(np.hypot(
                raw_corners[idx, 0] - smoothed_corners[idx, 0],
                raw_corners[idx, 1] - smoothed_corners[idx, 1],
            ))
            if delta < raw_bl_br_tau_px:
                corrs.append(Correspondence(
                    screen_xy=s_xy,
                    frame_xy=(float(raw_corners[idx, 0]), float(raw_corners[idx, 1])),
                    weight=0.5,
                    source="corner_raw",
                ))

    # --- Top corners (TL, TR) ---
    # Both smoothed (0.3) and raw (0.1) are included only when the raw-smoothed
    # delta is small enough to trust the detection. If raw_corners is absent we
    # cannot assess quality, so both are omitted conservatively (pre-flight
    # check 2 shows TR can be 235 px off on occlusion frames).
    if raw_corners is not None:
        for idx in (_CORNER_TL, _CORNER_TR):
            delta = float(np.hypot(
                raw_corners[idx, 0] - smoothed_corners[idx, 0],
                raw_corners[idx, 1] - smoothed_corners[idx, 1],
            ))
            if delta < raw_tl_tr_tau_px:
                s_xy = (float(screen_corners[idx, 0]), float(screen_corners[idx, 1]))
                corrs.append(Correspondence(
                    screen_xy=s_xy,
                    frame_xy=(float(smoothed_corners[idx, 0]), float(smoothed_corners[idx, 1])),
                    weight=0.3,
                    source="corner_smoothed",
                ))
                corrs.append(Correspondence(
                    screen_xy=s_xy,
                    frame_xy=(float(raw_corners[idx, 0]), float(raw_corners[idx, 1])),
                    weight=0.1,
                    source="corner_raw",
                ))

    # --- Big star (anchor) ---
    if big_star_screen_xy is not None and big_star_frame_xy is not None:
        corrs.append(Correspondence(
            screen_xy=big_star_screen_xy,
            frame_xy=big_star_frame_xy,
            weight=big_star_weight,
            source="big_star",
        ))

    # --- Small stars ---
    blend_range = small_star_tau_radius - small_star_radius_blend_tau
    for det in (small_stars or []):
        expected = det.source_prediction.expected_radius_px
        if expected > 0:
            rel_err = abs(det.equivalent_radius_px - expected) / expected
        else:
            rel_err = 0.0

        if rel_err <= small_star_radius_blend_tau:
            radius_factor = 1.0
        else:
            t = min(1.0, (rel_err - small_star_radius_blend_tau) / blend_range)
            radius_factor = 1.0 - 0.5 * t

        normalized_conf = min(1.0, max(0.0, det.confidence / small_star_confidence_scale))
        weight = normalized_conf * radius_factor

        corrs.append(Correspondence(
            screen_xy=det.source_prediction.screen_xy,
            frame_xy=det.frame_xy_subpix,
            weight=weight,
            source="small_star",
        ))

    return corrs
