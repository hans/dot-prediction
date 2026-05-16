"""Unit tests for star_matcher.match_stars().

All tests use synthetic detections and homographies — no real video required.
"""

import numpy as np
import pytest
from src.star_matcher import match_stars

# Identity homography: screen coords == frame coords
H_IDENTITY = np.eye(3, dtype=np.float64)

# Simple 2D translation homography: frame = screen + (100, 50)
H_SHIFT = np.array([
    [1, 0, 100],
    [0, 1,  50],
    [0, 0,   1],
], dtype=np.float64)


def _det(x, y, r=15.0):
    """Helper: single detection tuple."""
    return (float(x), float(y), float(r))


def _stars(*xys):
    """Helper: Nx2 screen-star array."""
    return np.array(xys, dtype=np.float64)


# ── return-type contract ─────────────────────────────────────────────────────

def test_returns_three_lists():
    corr, unmatched_det, unmatched_pred = match_stars([], _stars(), H_IDENTITY)
    assert isinstance(corr, list)
    assert isinstance(unmatched_det, list)
    assert isinstance(unmatched_pred, list)


def test_empty_inputs():
    corr, ud, up = match_stars([], _stars(), H_IDENTITY)
    assert corr == [] and ud == [] and up == []


def test_no_detections_all_unmatched_predictions():
    stars = _stars((100, 200), (300, 400))
    corr, ud, up = match_stars([], stars, H_IDENTITY)
    assert corr == []
    assert ud == []
    assert len(up) == 2


def test_no_predictions_all_unmatched_detections():
    dets = [_det(100, 200), _det(300, 400)]
    corr, ud, up = match_stars(dets, _stars(), H_IDENTITY)
    assert corr == []
    assert len(ud) == 2
    assert up == []


# ── basic matching ───────────────────────────────────────────────────────────

def test_exact_match_identity():
    """Detection exactly at the projected position → single correspondence."""
    stars = _stars((500, 300))
    dets = [_det(500, 300)]
    corr, ud, up = match_stars(dets, stars, H_IDENTITY)
    assert len(corr) == 1
    assert ud == []
    assert up == []
    (fx, fy), (sx, sy) = corr[0]
    assert fx == pytest.approx(500.0)
    assert fy == pytest.approx(300.0)
    assert sx == pytest.approx(500.0)
    assert sy == pytest.approx(300.0)


def test_match_with_offset_homography():
    """Detection displaced by H_SHIFT still matches when within search_radius."""
    stars = _stars((400, 300))
    # After H_SHIFT: projected frame pos = (500, 350)
    dets = [_det(505, 355)]  # 7 px off → within default radius
    corr, ud, up = match_stars(dets, stars, H_SHIFT)
    assert len(corr) == 1
    assert ud == []
    assert up == []


def test_detection_outside_radius_is_unmatched():
    """Detection > search_radius from prediction is an unmatched detection."""
    stars = _stars((400, 300))
    # Projected = (500, 350). Detection 200 px away.
    dets = [_det(700, 350)]
    corr, ud, up = match_stars(dets, stars, H_SHIFT, search_radius=100.0)
    assert len(corr) == 0
    assert len(ud) == 1
    assert len(up) == 1


def test_multiple_stars_matched():
    """Three stars with nearby detections all match correctly."""
    star_positions = [(200, 200), (600, 400), (1000, 300)]
    stars = _stars(*star_positions)
    # Detections ≈5 px off each star
    dets = [_det(sx + 5, sy - 3) for sx, sy in star_positions]
    corr, ud, up = match_stars(dets, stars, H_IDENTITY)
    assert len(corr) == 3
    assert ud == []
    assert up == []


def test_partial_match():
    """Two stars, one detection: one match, one missed star, no false positives."""
    stars = _stars((300, 300), (700, 500))
    dets = [_det(303, 298)]  # near first star only
    corr, ud, up = match_stars(dets, stars, H_IDENTITY, search_radius=50.0)
    assert len(corr) == 1
    assert len(ud) == 0   # detection was matched
    assert len(up) == 1   # second star unmatched


def test_false_positive_detection():
    """One star, two detections: correct one matches, other is false positive."""
    stars = _stars((500, 400))
    dets = [_det(502, 401), _det(900, 700)]  # second is far away
    corr, ud, up = match_stars(dets, stars, H_IDENTITY, search_radius=50.0)
    assert len(corr) == 1
    assert len(ud) == 1   # (900, 700) is unmatched detection
    assert len(up) == 0


# ── conflict resolution ──────────────────────────────────────────────────────

def test_conflict_closer_wins():
    """Two detections compete for one star: closer one wins."""
    stars = _stars((500, 400))
    close = _det(502, 400)   # 2 px away
    far   = _det(520, 400)   # 20 px away
    corr, ud, up = match_stars([close, far], stars, H_IDENTITY, search_radius=50.0)
    assert len(corr) == 1
    (fx, fy), _ = corr[0]
    # Matched detection should be the closer one
    assert fx == pytest.approx(502.0)
    assert len(ud) == 1   # the far detection is unmatched


def test_two_stars_two_detections_no_swap():
    """Greedy nearest-neighbour assigns each detection to its correct star."""
    # Stars at (200,200) and (800,200). Detections each close to their own star.
    stars = _stars((200, 200), (800, 200))
    dets = [_det(205, 200), _det(795, 200)]
    corr, ud, up = match_stars(dets, stars, H_IDENTITY)
    assert len(corr) == 2
    assert ud == []
    assert up == []
    # Verify assignments (by checking frame_x close to screen_x)
    for (fx, _), (sx, _) in corr:
        assert abs(fx - sx) < 20


# ── edge cases ───────────────────────────────────────────────────────────────

def test_single_star_array():
    """screen_stars passed as 1-D array (shape (2,)) is handled."""
    stars = np.array([500.0, 300.0])
    dets = [_det(500, 300)]
    corr, ud, up = match_stars(dets, stars, H_IDENTITY)
    assert len(corr) == 1


def test_custom_search_radius():
    """search_radius=10 rejects a 30 px miss; search_radius=50 accepts it."""
    stars = _stars((500, 400))
    dets = [_det(530, 400)]  # 30 px off
    corr_tight, _, _ = match_stars(dets, stars, H_IDENTITY, search_radius=10.0)
    corr_loose, _, _ = match_stars(dets, stars, H_IDENTITY, search_radius=50.0)
    assert len(corr_tight) == 0
    assert len(corr_loose) == 1
