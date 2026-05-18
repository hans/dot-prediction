"""Detect one photodiode-device (box) corner via Harris in a local window."""

import cv2
import numpy as np


def detect_box_corner(
    frame: np.ndarray,
    predicted_xy: tuple[float, float],
    corner: str,
    x_half: int = 20,
    y_above: int = 5,
    y_below: int = 20,
    min_harris_response: float = 0.05,
) -> tuple[float, float] | None:
    """Detect one box corner in an asymmetric local search window.

    The window is shifted downward relative to the prediction (y_above < y_below)
    to exclude a competing strong corner that exists ~10 px above the true
    device-bottom feature. See spec § Empirical characterisation for details.

    Args:
        frame: BGR image, uint8, shape (H, W, 3).
        predicted_xy: Expected (x, y) in frame pixel coordinates.
        corner: "bl" or "br" — reserved for future per-corner window tuning.
        x_half: Half-width of the search window (symmetric in x).
        y_above: Pixels above predicted_xy included in window.
        y_below: Pixels below predicted_xy included in window.
        min_harris_response: Minimum normalised Harris response to accept
            (0–1 after dividing by the patch maximum). Rejects uniform patches.

    Returns:
        (x, y) in frame pixel coordinates, or None.
    """
    if not (np.isfinite(predicted_xy[0]) and np.isfinite(predicted_xy[1])):
        return None

    H, W = frame.shape[:2]
    cx = int(round(predicted_xy[0]))
    cy = int(round(predicted_xy[1]))

    x0 = max(0, cx - x_half)
    x1 = min(W, cx + x_half)
    y0 = max(0, cy - y_above)
    y1 = min(H, cy + y_below)

    if x0 >= x1 or y0 >= y1:
        return None

    patch = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(gray.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    harris /= harris.max() + 1e-10

    py, px = np.unravel_index(harris.argmax(), harris.shape)
    if harris[py, px] < min_harris_response:
        return None

    return float(x0 + px), float(y0 + py)
