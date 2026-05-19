"""Detect one photodiode-device (box) corner via Harris + color prior."""

import cv2
import numpy as np


def _nms_candidates(
    harris: np.ndarray,
    min_response: float,
    nms_radius: int = 3,
) -> list[tuple[int, int, float]]:
    """Return NMS local maxima above threshold as (py, px, score) triples."""
    kernel = np.ones((2 * nms_radius + 1, 2 * nms_radius + 1), np.uint8)
    dilated = cv2.dilate(harris, kernel)
    mask = (harris == dilated) & (harris >= min_response)
    ys, xs = np.where(mask)
    return [(int(y), int(x), float(harris[y, x])) for y, x in zip(ys, xs)]


def _color_prior_score(
    frame_bgr: np.ndarray,
    fy: int,
    fx: int,
    corner: str,
    r: int = 8,
) -> float:
    """Soft color prior score for a candidate corner at frame coords (fx, fy).

    Scores how well the local HSV appearance matches the expected signature
    of the photodiode device corner:
      - Dark device casing above/left
      - Blue-purple canvas background (OpenCV H≈126) to one side
      - White canvas padding (low S, high V) below

    Returns a value in [0, 1]; higher = more consistent with expected appearance.
    """
    fH, fW = frame_bgr.shape[:2]

    def region_mean(y0: int, y1: int, x0: int, x1: int):
        y0_, y1_ = max(0, fy + y0), min(fH, fy + y1)
        x0_, x1_ = max(0, fx + x0), min(fW, fx + x1)
        if y0_ >= y1_ or x0_ >= x1_:
            return None
        patch = frame_bgr[y0_:y1_, x0_:x1_]
        return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).mean(axis=(0, 1)).astype(float)

    scores: list[float] = []

    if corner == "br":
        # top-left: dark device body (low V)
        m = region_mean(-r, 0, -r, 0)
        if m is not None:
            scores.append(1.0 - m[2] / 255.0)

        # top-right: blue-purple canvas (OpenCV H ≈ 126, high S)
        m = region_mean(-r, 0, 0, r)
        if m is not None:
            h_dist = min(abs(m[0] - 126.0), 180.0 - abs(m[0] - 126.0))
            scores.append(max(0.0, 1.0 - h_dist / 40.0) * (m[1] / 255.0))

        # bot-left: white canvas padding (low S, high V)
        m = region_mean(0, r, -r, 0)
        if m is not None:
            scores.append((1.0 - m[1] / 255.0) * (m[2] / 255.0))

    elif corner == "bl":
        # top-left: dark casing/edge (low V)
        m = region_mean(-r, 0, -r, 0)
        if m is not None:
            scores.append(1.0 - m[2] / 255.0)

        # top-right: dark device body (low V)
        m = region_mean(-r, 0, 0, r)
        if m is not None:
            scores.append(1.0 - m[2] / 255.0)

        # bot-right: white canvas padding (low S, high V)
        m = region_mean(0, r, 0, r)
        if m is not None:
            scores.append((1.0 - m[1] / 255.0) * (m[2] / 255.0))

    return float(np.mean(scores)) if scores else 0.0


def detect_box_corner(
    frame: np.ndarray,
    predicted_xy: tuple[float, float],
    corner: str,
    x_half: int = 20,
    y_above: int | None = None,
    y_below: int = 20,
    min_harris_response: float = 0.05,
) -> tuple[float, float] | None:
    """Detect one box corner in a local search window using Harris + color prior.

    Harris detects corner candidates; the color prior ranks them by how well
    each matches the expected appearance (dark device casing above, white canvas
    padding below, blue-purple canvas to one side).

    Args:
        frame: BGR image, uint8, shape (H, W, 3).
        predicted_xy: Expected (x, y) in frame pixel coordinates.
        corner: "bl" or "br" — controls window geometry and color prior.
        x_half: Half-width of the search window (symmetric in x).
        y_above: Pixels above predicted_xy included in window. Defaults to
            20 for "br" (symmetric per spec) and 5 for "bl" (asymmetric per spec).
        y_below: Pixels below predicted_xy included in window.
        min_harris_response: Minimum normalised Harris response to accept.

    Returns:
        (x, y) in frame pixel coordinates, or None.
    """
    if not (np.isfinite(predicted_xy[0]) and np.isfinite(predicted_xy[1])):
        return None

    if y_above is None:
        y_above = 20

    fH, fW = frame.shape[:2]
    cx = int(round(predicted_xy[0]))
    cy = int(round(predicted_xy[1]))

    x0 = max(0, cx - x_half)
    x1 = min(fW, cx + x_half)
    y0 = max(0, cy - y_above)
    y1 = min(fH, cy + y_below)

    if x0 >= x1 or y0 >= y1:
        return None

    patch = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(gray.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    harris /= harris.max() + 1e-10

    candidates = _nms_candidates(harris, min_harris_response)
    if not candidates:
        return None

    best_py, best_px = None, None
    best_score = -1.0

    for py, px, h_score in candidates:
        c_score = _color_prior_score(frame, y0 + py, x0 + px, corner)
        if c_score > best_score:
            best_score = c_score
            best_py, best_px = py, px

    if best_py is None:
        return None

    return float(x0 + best_px), float(y0 + best_py)
