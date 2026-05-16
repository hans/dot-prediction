import numpy as np
import pytest
import cv2
from src.screen_detection import detect_corners, order_corners

# Realistic-looking perspective quad that mimics the iPad's position in the video
# (off-axis, slightly rotated, as if viewed from above at an angle)
IPAD_CORNERS = np.array([
    [580,  370],  # TL
    [1060, 330],  # TR
    [1090, 600],  # BR
    [510,  590],  # BL
], dtype=np.float32)


def _make_frame(corners, brightness=200, frame_shape=(1080, 1920, 3)):
    """Synthetic dark frame with a bright filled quadrilateral."""
    frame = np.zeros(frame_shape, dtype=np.uint8)
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (brightness, brightness, brightness))
    return frame


def test_detect_corners_returns_array():
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    assert result.shape == (4, 2)
    assert result.dtype == np.float32


def test_detect_corners_ordering():
    """TL has smallest x+y, BR has largest x+y."""
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    sums = result.sum(axis=1)
    assert np.argmin(sums) == 0, "index 0 should be TL (smallest x+y)"
    assert np.argmax(sums) == 2, "index 2 should be BR (largest x+y)"


def test_detect_corners_accuracy():
    """Detected corners should be within 10 px of the ground truth."""
    ordered_gt = order_corners(IPAD_CORNERS)
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    assert np.allclose(result, ordered_gt, atol=10), (
        f"Detected corners too far from ground truth.\n"
        f"Expected:\n{ordered_gt}\nGot:\n{result}"
    )


def test_detect_corners_dark_frame_returns_none():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert detect_corners(frame) is None


def test_detect_corners_tiny_bright_region_returns_none():
    """A small bright blob (not a screen) should be filtered out."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:510, 900:910] = 255  # tiny 10×10 blob
    assert detect_corners(frame) is None


def test_order_corners_canonical():
    pts = np.array([[100, 50], [200, 50], [200, 150], [100, 150]], dtype=np.float32)
    ordered = order_corners(pts)
    np.testing.assert_array_equal(ordered[0], [100, 50])   # TL
    np.testing.assert_array_equal(ordered[1], [200, 50])   # TR
    np.testing.assert_array_equal(ordered[2], [200, 150])  # BR
    np.testing.assert_array_equal(ordered[3], [100, 150])  # BL
