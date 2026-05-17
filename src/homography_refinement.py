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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

try:
    from local_star_detector import LocalDetection
except ImportError:
    from src.local_star_detector import LocalDetection

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
    ransac_threshold_px: float = 3.0,
    weight_replication: int = 10,
    ransac_min_correspondences: int = 6,
) -> SolveResult:
    """Solve for screen→frame H from a weighted correspondence set.

    Weights are realised via the **repeated-points trick**: each correspondence
    contributes ``max(1, round(weight × weight_replication))`` rows to the DLT
    system. Stock ``cv2.findHomography`` then weights duplicates equally,
    which (for RANSAC) gives high-trust points proportionally more inlier
    voting power. This is the spec's "simple first cut"; a hand-rolled
    weighted DLT can replace it later if continuous weights matter.

    RANSAC is used when there are at least ``ransac_min_correspondences``
    unique correspondences (enough degrees of freedom to drop an outlier);
    below that, least-squares is used and the inlier mask is all-True.

    Args:
        correspondences: Non-empty list of ``Correspondence``. Entries with
            ``weight <= 0`` are silently dropped.
        ransac_threshold_px: Frame-pixel reprojection threshold passed to
            ``cv2.findHomography(..., method=cv2.RANSAC)``. Defaults to 3.0
            per spec.
        weight_replication: Multiplier converting continuous weights into
            integer row replication counts (default 10 → weights of 0.1
            granularity).
        ransac_min_correspondences: Minimum unique correspondences before
            switching from least-squares to RANSAC.

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

    use_ransac = len(usable) >= ransac_min_correspondences
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
