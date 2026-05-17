"""Position-conditioned local star detector.

Given a set of predicted star locations and a frame, search a small window
around each prediction and confirm or deny the star's presence. The position
prior carries the bulk of the discrimination, so the per-pixel threshold can
be much more permissive than in the global Phase-1a detector.

Method (per prediction)
-----------------------
1. Crop a (window_size, window_size) patch centred on the prediction.
2. Compute the R-B opponent channel (warm = positive).
3. Threshold at a permissive floor (default 20, vs 40 in the global detector).
   Below the floor → "not detected" for this prediction.
4. Intensity-weighted centroid over the above-floor pixels gives the
   sub-pixel star centre. This is the detection's frame-pixel xy.
5. The patch peak value is the confidence score. Equivalent radius is
   ``sqrt(above_floor_pixel_count / π)``.
6. Size-aware rejection: if the equivalent radius is more than
   ``max_radius_factor × expected_radius_px``, reject (likely a finger /
   glare blob saturating the window).

Window-overlap conflicts
------------------------
If two predictions land within ``window_size`` of each other, their search
patches overlap, and the same peak pixel could be the argmax for both. The
function flags this case by reporting the overlap in each ``LocalDetection``
but does NOT attempt to resolve it (the spec calls overlaps "usually means
H_rough is significantly off" — a diagnostic, not an error).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    # Notebooks / scripts: ``sys.path.insert(0, 'src')`` then bare imports.
    from predicted_positions import PredictedStar
except ImportError:
    # pytest / external callers: ``src`` is a namespace package.
    from src.predicted_positions import PredictedStar


@dataclass(frozen=True)
class LocalDetection:
    """Result of a successful local search around a prediction.

    Attributes:
        frame_xy_subpix: Intensity-weighted sub-pixel centroid (x, y) in
            frame coordinates.
        confidence: Peak R−B opponent-channel value inside the window
            (uint8 scale, so 0–255).
        equivalent_radius_px: ``sqrt(above_floor_pixel_count / π)``.
        peak_xy: Integer pixel coordinates of the peak (for overlap
            detection).
        source_prediction: The PredictedStar this detection corresponds to.
    """

    frame_xy_subpix: tuple[float, float]
    confidence: float
    equivalent_radius_px: float
    peak_xy: tuple[int, int]
    source_prediction: PredictedStar


def _window_bounds(cx: float, cy: float, win: int, W: int, H: int) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) clamped to the image. None if fully off-frame."""
    half = win // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + win
    y1 = y0 + win
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x0c >= x1c or y0c >= y1c:
        return None
    return x0c, y0c, x1c, y1c


def _window_size_for(
    pred: PredictedStar,
    window_size_px: int,
    adaptive_radius_factor: float | None,
    min_window_px: int,
    max_window_px: int,
) -> int:
    """Choose the square window side length for a single prediction.

    Fixed when ``adaptive_radius_factor`` is None; otherwise scaled to the
    expected blob diameter and clipped to [min_window_px, max_window_px].
    """
    if adaptive_radius_factor is None:
        return window_size_px
    w = int(round(adaptive_radius_factor * pred.expected_radius_px))
    return max(min_window_px, min(w, max_window_px))


def detect_in_windows(
    frame: np.ndarray,
    predictions: list[PredictedStar],
    window_size_px: int = 40,
    floor: float = 20.0,
    max_radius_factor: float = 4.0,
    adaptive_radius_factor: float | None = None,
    min_window_px: int = 10,
    max_window_px: int = 60,
) -> tuple[list[LocalDetection], list[PredictedStar]]:
    """Search the frame at each prediction for the predicted star.

    Args:
        frame: BGR image, uint8, shape (H, W, 3).
        predictions: List of PredictedStar from ``predicted_positions``.
        window_size_px: Side length of the square search window in frame px,
            used when ``adaptive_radius_factor`` is None. Default 40 —
            the spec's starting point. Tune up if H_rough has large bias
            and no anchor correction is applied.
        floor: Minimum R−B opponent-channel value inside the window for a
            pixel to count toward the centroid / area. The position prior
            does most discrimination, so this can be much lower than the
            global detector's 40.
        max_radius_factor: A blob is rejected when its equivalent radius is
            larger than this × ``prediction.expected_radius_px``. Tuned
            generously (4.0×) because the size model is coarse — the goal
            is to reject finger-sized blobs only.
        adaptive_radius_factor: When set, the window for each prediction
            becomes ``factor × expected_radius_px`` (clipped to
            [min_window_px, max_window_px]). Use with anchor-corrected
            H_rough where residual translation error is small: tightens
            the window to roughly the star's actual extent, eliminating
            the "7 predictions all snap to the same blob" failure mode
            when stars cluster within a fixed window's span.
        min_window_px: Lower clip on adaptive window size.
        max_window_px: Upper clip on adaptive window size.

    Returns:
        Tuple of (detections, unmatched_predictions). Each element of
        ``detections`` is a LocalDetection; ``unmatched_predictions`` is a
        list of predictions for which no valid blob was found.
    """
    H, W = frame.shape[:2]
    r = frame[:, :, 2].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    redness = r - b

    detections: list[LocalDetection] = []
    unmatched: list[PredictedStar] = []

    for pred in predictions:
        cx, cy = pred.frame_xy
        win = _window_size_for(pred, window_size_px, adaptive_radius_factor,
                               min_window_px, max_window_px)
        bounds = _window_bounds(cx, cy, win, W, H)
        if bounds is None:
            unmatched.append(pred)
            continue
        x0, y0, x1, y1 = bounds
        patch = redness[y0:y1, x0:x1]
        mask = patch > floor
        if not mask.any():
            unmatched.append(pred)
            continue

        # Intensity-weighted centroid in patch-local coords
        ys, xs = np.nonzero(mask)
        weights = patch[ys, xs]
        total = weights.sum()
        cx_patch = float((xs * weights).sum() / total)
        cy_patch = float((ys * weights).sum() / total)
        fx = x0 + cx_patch
        fy = y0 + cy_patch

        peak_local = int(np.argmax(patch))
        py, px = np.unravel_index(peak_local, patch.shape)
        peak_xy = (int(x0 + px), int(y0 + py))
        confidence = float(patch[py, px])

        area = int(mask.sum())
        equivalent_radius_px = float(np.sqrt(area / np.pi))

        if (
            pred.expected_radius_px > 0
            and equivalent_radius_px > max_radius_factor * pred.expected_radius_px
        ):
            # Reject — way larger than the prediction's expected size.
            unmatched.append(pred)
            continue

        detections.append(
            LocalDetection(
                frame_xy_subpix=(fx, fy),
                confidence=confidence,
                equivalent_radius_px=equivalent_radius_px,
                peak_xy=peak_xy,
                source_prediction=pred,
            )
        )

    return detections, unmatched


def find_overlapping_peaks(detections: list[LocalDetection]) -> list[tuple[int, int]]:
    """Return list of detection-index pairs (i, j) whose ``peak_xy`` coincide.

    A non-empty result is a hint that ``H_rough`` is significantly off at this
    frame — two predicted-star windows are landing on the same pixel.
    """
    by_peak: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(detections):
        by_peak.setdefault(d.peak_xy, []).append(i)
    return [(grp[a], grp[b])
            for grp in by_peak.values() if len(grp) > 1
            for a in range(len(grp)) for b in range(a + 1, len(grp))]
