"""Unit tests for homography_refinement.anchor_translate()."""

import numpy as np
import pytest

from src.homography_refinement import anchor_translate


def _project(H, sx, sy):
    h = H @ np.array([sx, sy, 1.0])
    return h[0] / h[2], h[1] / h[2]


def test_anchor_lands_on_target():
    """After correction, the anchor projects exactly to its frame target."""
    # Translation-only H (predicts +100, +50)
    H = np.array([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=np.float64)
    anchor_s = (500.0, 400.0)
    anchor_f_target = (650.0, 480.0)  # 50, 30 off the H prediction
    H_new = anchor_translate(H, anchor_s, anchor_f_target)
    fx, fy = _project(H_new, *anchor_s)
    assert fx == pytest.approx(650.0, abs=1e-6)
    assert fy == pytest.approx(480.0, abs=1e-6)


def test_other_points_translate_by_same_offset():
    """Every projected point shifts by exactly (dx, dy) = target − predicted."""
    H = np.array([[2, 0, 100], [0, 2, 50], [0, 0, 1]], dtype=np.float64)
    anchor_s = (200.0, 300.0)
    fx_pred, fy_pred = _project(H, *anchor_s)
    target = (fx_pred + 17.0, fy_pred - 9.0)
    H_new = anchor_translate(H, anchor_s, target)

    # For an arbitrary other point, the projection shifts by the same (17, -9).
    other = (1000.0, 800.0)
    before = _project(H, *other)
    after = _project(H_new, *other)
    assert after[0] - before[0] == pytest.approx(17.0, abs=1e-6)
    assert after[1] - before[1] == pytest.approx(-9.0, abs=1e-6)


def test_perspective_homography_preserved():
    """Translation correction does not change the perspective component (h31, h32)."""
    H = np.array([
        [1.5, 0.1, 100],
        [0.05, 1.4, 50],
        [1e-4, 2e-4, 1.0],
    ], dtype=np.float64)
    H_new = anchor_translate(H, (300.0, 400.0), (700.0, 800.0))
    # Last row (perspective) must be unchanged
    assert H_new[2, 0] == pytest.approx(H[2, 0])
    assert H_new[2, 1] == pytest.approx(H[2, 1])
    assert H_new[2, 2] == pytest.approx(H[2, 2])


def test_zero_correction_when_anchor_already_matches():
    """If the prediction already hits the target, H is unchanged."""
    H = np.array([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=np.float64)
    anchor_s = (200.0, 200.0)
    target = _project(H, *anchor_s)  # exact match
    H_new = anchor_translate(H, anchor_s, target)
    np.testing.assert_allclose(H_new, H, atol=1e-9)
