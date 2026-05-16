"""Unit tests for local_star_detector.detect_in_windows().

All tests use synthetic frames + synthetic predictions — no real video.
"""

import cv2
import numpy as np
import pytest

from src.local_star_detector import detect_in_windows, find_overlapping_peaks
from src.predicted_positions import PredictedStar

W, H = 1920, 1080


def _blue_frame() -> np.ndarray:
    """Blue background, like the iPad task display."""
    return np.full((H, W, 3), (140, 50, 50), dtype=np.uint8)


def _paint_blob(frame: np.ndarray, cx: int, cy: int, radius: int = 5,
                color=(120, 175, 180)) -> None:
    cv2.circle(frame, (cx, cy), radius, color, -1)


def _pred(cx: float, cy: float, expected_radius_px: float = 5.0,
          age_s: float = 1.0, tpt: int = 0) -> PredictedStar:
    return PredictedStar(
        trial_idx=1, tpt=tpt,
        screen_xy=(cx, cy),
        frame_xy=(cx, cy),
        age_s=age_s,
        expected_radius_px=expected_radius_px,
    )


# ── return-type contract ─────────────────────────────────────────────────────

def test_returns_pair_of_lists():
    frame = _blue_frame()
    dets, unmatched = detect_in_windows(frame, [])
    assert dets == [] and unmatched == []


def test_no_predictions_returns_empty():
    frame = _blue_frame()
    _paint_blob(frame, 500, 500)
    dets, unmatched = detect_in_windows(frame, [])
    assert dets == [] and unmatched == []


# ── basic detection ──────────────────────────────────────────────────────────

def test_blob_at_prediction_detected():
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=5)
    dets, unmatched = detect_in_windows(frame, [_pred(800, 600)])
    assert len(dets) == 1
    assert unmatched == []


def test_centroid_near_blob_center():
    """Sub-pixel centroid is within 2 px of the painted blob center."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=5)
    dets, _ = detect_in_windows(frame, [_pred(800, 600)])
    fx, fy = dets[0].frame_xy_subpix
    assert np.hypot(fx - 800, fy - 600) < 2.0


def test_missing_blob_is_unmatched():
    frame = _blue_frame()  # no blob painted
    dets, unmatched = detect_in_windows(frame, [_pred(800, 600)])
    assert dets == []
    assert len(unmatched) == 1


def test_blob_offset_within_window_still_detected():
    """Prediction 15 px off the actual blob is fine within a 40-px window."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=5)
    dets, unmatched = detect_in_windows(frame, [_pred(815, 605)], window_size_px=40)
    assert len(dets) == 1
    fx, fy = dets[0].frame_xy_subpix
    assert np.hypot(fx - 800, fy - 600) < 3.0


def test_blob_outside_window_not_detected():
    """Prediction 60 px off → blob outside the 40-px window → not detected."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=5)
    dets, unmatched = detect_in_windows(frame, [_pred(870, 670)], window_size_px=40)
    assert dets == []
    assert len(unmatched) == 1


def test_window_size_parameter():
    """A larger window can recover a blob the default window misses."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=5)
    pred = _pred(860, 660)  # 60×sqrt(2) px off — outside 40-px but inside 200
    dets_small, _ = detect_in_windows(frame, [pred], window_size_px=40)
    dets_big, _ = detect_in_windows(frame, [pred], window_size_px=200)
    assert dets_small == []
    assert len(dets_big) == 1


# ── confidence & radius ──────────────────────────────────────────────────────

def test_confidence_increases_with_blob_brightness():
    """A brighter (more saturated R) blob yields a higher confidence."""
    f_dim = _blue_frame()
    _paint_blob(f_dim, 800, 600, radius=5, color=(100, 140, 150))  # R-B = 50
    f_bright = _blue_frame()
    _paint_blob(f_bright, 800, 600, radius=5, color=(80, 200, 230))  # R-B = 150

    pred = [_pred(800, 600)]
    d_dim, _ = detect_in_windows(f_dim, pred)
    d_bri, _ = detect_in_windows(f_bright, pred)
    assert d_dim and d_bri
    assert d_bri[0].confidence > d_dim[0].confidence


def test_equivalent_radius_scales_with_blob_size():
    """A larger painted blob gives a larger equivalent_radius_px."""
    f_small = _blue_frame()
    _paint_blob(f_small, 800, 600, radius=3)
    f_big = _blue_frame()
    _paint_blob(f_big, 800, 600, radius=8)

    pred_small = _pred(800, 600, expected_radius_px=3.0)
    pred_big = _pred(800, 600, expected_radius_px=8.0)

    d_s, _ = detect_in_windows(f_small, [pred_small])
    d_b, _ = detect_in_windows(f_big, [pred_big])
    assert d_s and d_b
    assert d_b[0].equivalent_radius_px > d_s[0].equivalent_radius_px


# ── size-aware rejection ─────────────────────────────────────────────────────

def test_oversized_blob_rejected():
    """A blob much larger than expected (e.g. a finger) is rejected."""
    frame = _blue_frame()
    # Big warm region — radius 30 — vs prediction expecting 5
    _paint_blob(frame, 800, 600, radius=30, color=(120, 175, 180))
    pred = _pred(800, 600, expected_radius_px=5.0)
    dets, unmatched = detect_in_windows(frame, [pred], window_size_px=100,
                                        max_radius_factor=4.0)
    # Inside a 100×100 window a r=30 blob has area ≈ 2800 → radius ≈ 30,
    # which is 6× the 5-px expected → over the 4× factor → rejected.
    assert dets == []
    assert len(unmatched) == 1


def test_max_radius_factor_disables_rejection():
    """A large factor (e.g. 1e6) keeps even very oversized blobs."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=30)
    pred = _pred(800, 600, expected_radius_px=5.0)
    dets, _ = detect_in_windows(frame, [pred], window_size_px=100,
                                max_radius_factor=1e6)
    assert len(dets) == 1


# ── edge cases ───────────────────────────────────────────────────────────────

def test_prediction_off_frame_is_unmatched():
    """Predicted xy fully outside frame → unmatched, no crash."""
    frame = _blue_frame()
    pred = _pred(-100, -100)
    dets, unmatched = detect_in_windows(frame, [pred])
    assert dets == []
    assert len(unmatched) == 1


def test_prediction_at_frame_edge_partial_window():
    """Prediction near a corner: window is clipped to the image, still works."""
    frame = _blue_frame()
    _paint_blob(frame, 5, 5, radius=4)
    pred = _pred(5, 5)
    dets, _ = detect_in_windows(frame, [pred], window_size_px=40)
    assert len(dets) == 1
    fx, fy = dets[0].frame_xy_subpix
    assert np.hypot(fx - 5, fy - 5) < 3.0


def test_floor_parameter():
    """A dim blob is detected only when ``floor`` is permissive."""
    frame = _blue_frame()
    # R-B contrast for (120,130,140) is 140-120 = 20.
    _paint_blob(frame, 800, 600, radius=6, color=(120, 130, 140))
    pred = _pred(800, 600)
    d_strict, _ = detect_in_windows(frame, [pred], floor=50)
    d_loose, _ = detect_in_windows(frame, [pred], floor=10)
    assert d_strict == []
    assert len(d_loose) == 1


# ── overlap diagnostic ──────────────────────────────────────────────────────

def test_overlapping_peaks_flagged():
    """Two predictions whose windows hit the same peak pixel are flagged."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=4)
    pred_a = _pred(795, 605, tpt=0)
    pred_b = _pred(805, 595, tpt=1)
    dets, _ = detect_in_windows(frame, [pred_a, pred_b], window_size_px=40)
    assert len(dets) == 2
    overlaps = find_overlapping_peaks(dets)
    # Same blob → same peak xy → exactly one collision pair.
    assert len(overlaps) == 1
    i, j = overlaps[0]
    assert {dets[i].peak_xy, dets[j].peak_xy} == {dets[i].peak_xy}  # equal


def test_no_overlap_for_disjoint_blobs():
    """Two predictions with their own blobs produce no overlap flags."""
    frame = _blue_frame()
    _paint_blob(frame, 400, 400, radius=4)
    _paint_blob(frame, 1200, 700, radius=4)
    dets, _ = detect_in_windows(frame, [_pred(400, 400), _pred(1200, 700)])
    assert len(dets) == 2
    assert find_overlapping_peaks(dets) == []
