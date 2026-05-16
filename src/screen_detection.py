"""Detect the 4 corners of the iPad screen in a video frame."""

import cv2
import numpy as np

# Minimum contour area in pixels. The iPad screen occupies roughly
# 25–35% of the 1920×1080 frame ≈ 500k–700k px. Use a conservative floor.
_MIN_AREA = 80_000

# Brightness threshold (0–255 grayscale). The iPad screen is much brighter
# than the near-black room. Tuned from inspecting real frames.
_THRESH = 50

# Morphological close kernel size (px). Must be large enough to fill the
# blue task background (which sits above threshold but may have thin dark
# gaps from content elements like dots/stars).
_CLOSE_K = 51


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in [TL, TR, BR, BL] order.

    Args:
        pts: shape (4, 2) or (4, 1, 2) float array of corner points.

    Returns:
        float32 array of shape (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)          # x + y
    d = pts[:, 0] - pts[:, 1]   # x - y
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmax(d)],   # TR: largest x-y (large x, small y)
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmin(d)],   # BL: smallest x-y (small x, large y)
    ], dtype=np.float32)


def detect_corners(frame: np.ndarray) -> np.ndarray | None:
    """Detect the 4 corners of the iPad screen in a video frame.

    Args:
        frame: BGR image of shape (H, W, 3), uint8.

    Returns:
        float32 array of shape (4, 2) with corners in [TL, TR, BR, BL]
        order (video-pixel coordinates), or None if no screen is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold: screen pixels are brighter than _THRESH in near-dark room
    _, binary = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)

    # Close to fill gaps within the screen content area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_CLOSE_K, _CLOSE_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only large contours; take the largest
    large = [c for c in contours if cv2.contourArea(c) > _MIN_AREA]
    if not large:
        return None
    largest = max(large, key=cv2.contourArea)
    largest = cv2.convexHull(largest)

    # Approximate to a polygon; require exactly 4 sides.
    # Try increasing epsilon until we get a 4-vertex result (handles motion blur
    # and slight curve in the contour that can produce 5+ vertices at 0.02).
    peri = cv2.arcLength(largest, True)
    approx = None
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        candidate = cv2.approxPolyDP(largest, eps * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break
    if approx is None:
        return None

    return order_corners(approx.reshape(4, 2).astype(np.float32))
