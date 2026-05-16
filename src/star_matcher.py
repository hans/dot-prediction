"""Match detected star blobs to known star positions via a rough homography.

The matcher projects each known star (screen-pixel coordinates) into predicted
frame coordinates using ``H_rough``, then assigns detections to predictions
using nearest-neighbour within a configurable search radius.  Ambiguous
assignments (two detections competing for the same prediction) are resolved by
always taking the closer one; the displaced detection is then offered to its
next-nearest prediction.

Returns
-------
Three lists:
  correspondences  – (frame_xy, screen_xy) pairs, one per matched star.
  unmatched_detections  – frame-coord blobs with no close prediction (false
      positives: glare, UI artefacts, hand highlights, …).
  unmatched_predictions – screen-coord stars with no nearby detection (missed:
      hand occlusion, star too small, out-of-frame, …).

Coordinate conventions
----------------------
- ``screen_stars``: Nx2 float array of star positions in **iPad device pixels**
  (2388 × 1668 native resolution, landscape).  Convert from the normalised
  task-log coordinates with ``true_x * 2388, true_y * 1668``.
- ``H_rough``: 3×3 float64 homography mapping **screen → frame** pixel coords.
  Build from the existing brightness-detector corners:
  ``cv2.findHomography(SCREEN_CORNERS, detected_frame_corners)``
  where ``SCREEN_CORNERS = [[0,0],[2388,0],[2388,1668],[0,1668]]``.
- ``detections``: list of (x, y, radius) as returned by ``detect_stars()``.
"""

import numpy as np

# Default nearest-neighbour search radius (frame pixels).  H_rough typically
# has 30-100 px error (top-corner undershoot from the brightness detector), so
# 100 px gives comfortable headroom while staying selective enough to avoid
# cross-assignment on closely-spaced stars.
_DEFAULT_RADIUS: float = 100.0


def match_stars(
    detections: list[tuple[float, float, float]],
    screen_stars: np.ndarray,
    H_rough: np.ndarray,
    search_radius: float = _DEFAULT_RADIUS,
) -> tuple[
    list[tuple[tuple[float, float], tuple[float, float]]],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
]:
    """Match blob detections to known star screen positions.

    Args:
        detections: List of (x, y, radius) blobs from detect_stars().
        screen_stars: float array of shape (N, 2), star positions in iPad
            device pixels.
        H_rough: 3×3 homography, screen-pixel → frame-pixel.
        search_radius: Maximum frame-pixel distance for a valid match.

    Returns:
        Tuple of:
          - correspondences: list of ((frame_x, frame_y), (screen_x, screen_y))
          - unmatched_detections: list of (x, y, radius) – likely false positives
          - unmatched_predictions: list of (screen_x, screen_y) – likely misses
    """
    if len(screen_stars) == 0 or H_rough is None:
        return [], list(detections), []

    screen_stars = np.asarray(screen_stars, dtype=np.float64)
    if screen_stars.ndim == 1:
        screen_stars = screen_stars.reshape(1, 2)

    # Project screen coords → frame coords via H_rough
    n_pred = len(screen_stars)
    ones = np.ones((n_pred, 1), dtype=np.float64)
    pts_h = np.hstack([screen_stars, ones])          # (N, 3)
    proj_h = (H_rough @ pts_h.T).T                   # (N, 3)
    predicted = proj_h[:, :2] / proj_h[:, 2:3]      # (N, 2) frame coords

    if len(detections) == 0:
        return [], [], [(float(s[0]), float(s[1])) for s in screen_stars]

    det_xy = np.array([[d[0], d[1]] for d in detections], dtype=np.float64)  # (M, 2)

    # Build distance matrix: (M detections) x (N predictions)
    diff = det_xy[:, None, :] - predicted[None, :, :]   # (M, N, 2)
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])        # (M, N)

    # Greedy nearest-neighbour with conflict resolution
    matched_det: set[int] = set()
    matched_pred: set[int] = set()
    # Sort all (det, pred) pairs by distance
    pairs = sorted(
        [(dist[m, n], m, n) for m in range(len(detections)) for n in range(n_pred)],
        key=lambda x: x[0],
    )

    correspondences: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for d, m, n in pairs:
        if d > search_radius:
            break  # remaining pairs all exceed radius (sorted)
        if m in matched_det or n in matched_pred:
            continue
        matched_det.add(m)
        matched_pred.add(n)
        frame_xy = (float(det_xy[m, 0]), float(det_xy[m, 1]))
        screen_xy = (float(screen_stars[n, 0]), float(screen_stars[n, 1]))
        correspondences.append((frame_xy, screen_xy))

    unmatched_detections = [
        detections[m] for m in range(len(detections)) if m not in matched_det
    ]
    unmatched_predictions = [
        (float(screen_stars[n, 0]), float(screen_stars[n, 1]))
        for n in range(n_pred)
        if n not in matched_pred
    ]

    return correspondences, unmatched_detections, unmatched_predictions
