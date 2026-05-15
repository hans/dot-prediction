"""Unit tests for star_detector.detect_stars().

All tests use synthetic frames — no real video required.
"""

import cv2
import numpy as np
import pytest
from src.star_detector import detect_stars

# Frame dimensions matching the Tobii scene camera
W, H = 1920, 1080


def _blue_frame(blob_center=None, blob_radius=20, blob_color=(120, 175, 180)):
    """Create a synthetic frame: blue background + optional warm-white blob.

    The background mimics the task's blue gradient (BGR B≈140,G≈50,R≈50 →
    HSV H≈120, S≈163, V≈140).

    ``blob_color`` is BGR; default (B=120, G=175, R=180) matches observed star
    pixel values (R > G > B, HSV H≈28, S≈85, V≈180 — warm-white).
    """
    # Pure blue background in BGR: B=140, G=50, R=50 → HSV H≈120
    frame = np.full((H, W, 3), (140, 50, 50), dtype=np.uint8)

    if blob_center is not None:
        cx, cy = blob_center
        cv2.circle(frame, (cx, cy), blob_radius, blob_color, -1)

    return frame


def _has_blob_near(blobs, cx, cy, radius=30):
    """Return True if any blob centroid is within ``radius`` px of (cx, cy)."""
    for bx, by, _r in blobs:
        if np.hypot(bx - cx, by - cy) < radius:
            return True
    return False


# ── basic contract ──────────────────────────────────────────────────────────

def test_returns_list():
    frame = _blue_frame()
    result = detect_stars(frame)
    assert isinstance(result, list)


def test_empty_on_pure_blue():
    """No blobs on a uniform blue frame."""
    frame = _blue_frame()
    assert detect_stars(frame) == []


def test_blob_tuple_shape():
    """Each returned element is a 3-tuple of floats."""
    cx, cy = W // 2, H // 2
    frame = _blue_frame(blob_center=(cx, cy), blob_radius=20)
    blobs = detect_stars(frame)
    assert len(blobs) >= 1
    for b in blobs:
        assert len(b) == 3
        assert all(isinstance(v, float) for v in b)


def test_radius_positive():
    """Radius is strictly positive."""
    cx, cy = W // 2, H // 2
    frame = _blue_frame(blob_center=(cx, cy), blob_radius=20)
    blobs = detect_stars(frame)
    assert all(r > 0 for _, _, r in blobs)


# ── detection accuracy ───────────────────────────────────────────────────────

def test_detects_large_blob():
    """A warm-white circle of radius 20 on a blue background is detected."""
    cx, cy = 900, 540
    frame = _blue_frame(blob_center=(cx, cy), blob_radius=20)
    blobs = detect_stars(frame)
    assert _has_blob_near(blobs, cx, cy), f"No blob near ({cx},{cy}), got {blobs}"


def test_centroid_accuracy():
    """Detected centroid is within 10 px of the true blob centre."""
    cx, cy = 750, 400
    frame = _blue_frame(blob_center=(cx, cy), blob_radius=22)
    blobs = detect_stars(frame)
    assert blobs, "Expected at least one detection"
    closest = min(blobs, key=lambda b: np.hypot(b[0] - cx, b[1] - cy))
    dist = np.hypot(closest[0] - cx, closest[1] - cy)
    assert dist < 10, f"Centroid error {dist:.1f} px exceeds 10 px"


def test_radius_roughly_correct():
    """Returned radius is within 50% of the synthetic blob radius."""
    blob_r = 20
    frame = _blue_frame(blob_center=(W // 2, H // 2), blob_radius=blob_r)
    blobs = detect_stars(frame)
    assert blobs
    detected_r = blobs[0][2]
    assert 0.5 * blob_r <= detected_r <= 2.0 * blob_r, (
        f"Expected radius ≈ {blob_r}, got {detected_r:.1f}"
    )


def test_rejects_small_noise():
    """A warm blob of radius 5 below min_area is rejected."""
    frame = _blue_frame(blob_center=(500, 300), blob_radius=5)
    # area of r=5 circle ≈ 78 px² — below default min_area=100
    blobs = detect_stars(frame)
    assert not _has_blob_near(blobs, 500, 300, radius=20), (
        "Small noise blob should be filtered out"
    )


def test_min_area_parameter():
    """Lowering min_area to 30 lets a r=5 blob through."""
    frame = _blue_frame(blob_center=(500, 300), blob_radius=5)
    blobs = detect_stars(frame, min_area=30)
    assert _has_blob_near(blobs, 500, 300, radius=20), (
        "With min_area=30 the small blob should be detected"
    )


def test_dark_frame_returns_empty():
    """All-black frame: no blue background → no display region → no detections."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert detect_stars(frame) == []


def test_multiple_blobs_detected():
    """Two warm blobs far apart are both detected."""
    p1, p2 = (400, 300), (1400, 700)
    frame = _blue_frame()
    for px, py in (p1, p2):
        cv2.circle(frame, (px, py), 22, (120, 175, 180), -1)  # warm-white blob
    blobs = detect_stars(frame)
    assert _has_blob_near(blobs, *p1), f"Blob near {p1} missing"
    assert _has_blob_near(blobs, *p2), f"Blob near {p2} missing"


def test_bezel_not_detected():
    """White pixels outside the blue content region (bezels) are not returned."""
    # Add a white strip on both sides (simulated bezels), no blob in blue area
    frame = _blue_frame()
    frame[:, :80] = (255, 255, 255)   # left bezel
    frame[:, -80:] = (255, 255, 255)  # right bezel
    blobs = detect_stars(frame)
    # No blob should be near the bezel areas
    for bx, _, _ in blobs:
        assert bx > 80 and bx < W - 80, (
            f"Bezel false positive at x={bx:.0f}"
        )
