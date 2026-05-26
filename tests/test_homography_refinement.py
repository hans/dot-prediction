"""Unit tests for homography_refinement."""

import numpy as np
import pytest

from src.homography_refinement import (
    Correspondence,
    anchor_translate,
    solve_weighted_homography,
)


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


# ---------------------------------------------------------------------------
# solve_weighted_homography
# ---------------------------------------------------------------------------

# A non-trivial true H with translation, scale, and modest perspective —
# representative of the real EC347 screen→frame mapping.
_H_TRUE = np.array([
    [0.55, 0.03, 380.0],
    [0.02, 0.58, 210.0],
    [3e-5, 1e-5, 1.0],
], dtype=np.float64)


def _make_correspondences(
    screen_pts: np.ndarray,
    H: np.ndarray = _H_TRUE,
    weight: float = 1.0,
    source: str = "small_star",
    frame_offsets: np.ndarray | None = None,
) -> list[Correspondence]:
    ones = np.ones((len(screen_pts), 1))
    proj = (H @ np.hstack([screen_pts, ones]).T).T
    frame_pts = proj[:, :2] / proj[:, 2:3]
    if frame_offsets is not None:
        frame_pts = frame_pts + frame_offsets
    return [
        Correspondence(
            screen_xy=(float(s[0]), float(s[1])),
            frame_xy=(float(f[0]), float(f[1])),
            weight=weight,
            source=source,  # type: ignore[arg-type]
        )
        for s, f in zip(screen_pts, frame_pts)
    ]


def _max_corner_reproj_error(H_est: np.ndarray) -> float:
    screen_corners = np.array(
        [[0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0]],
    )
    ones = np.ones((4, 1))
    pred = (_H_TRUE @ np.hstack([screen_corners, ones]).T).T
    pred = pred[:, :2] / pred[:, 2:3]
    got = (H_est @ np.hstack([screen_corners, ones]).T).T
    got = got[:, :2] / got[:, 2:3]
    return float(np.hypot(*(pred - got).T).max())


def test_solver_recovers_clean_homography_via_ransac():
    """6 noise-free unit-weight correspondences → H within sub-pixel."""
    screen = np.array([
        [0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0],
        [1200.0, 800.0], [800.0, 1200.0],
    ])
    corrs = _make_correspondences(screen)
    res = solve_weighted_homography(corrs)
    assert res.method == "ransac"
    assert res.inlier_mask.all()
    assert res.residuals_px.max() < 1e-3
    assert _max_corner_reproj_error(res.H) < 1e-2


def test_solver_least_squares_path_with_four_points():
    """Exactly 4 correspondences → least-squares (no RANSAC)."""
    screen = np.array(
        [[0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0]],
    )
    corrs = _make_correspondences(screen)
    res = solve_weighted_homography(corrs)
    assert res.method == "lstsq"
    assert res.inlier_mask.all()
    assert res.residuals_px.max() < 1e-3


def test_solver_ransac_rejects_outlier():
    """A wildly wrong correspondence is flagged as outlier; H stays clean."""
    screen = np.array([
        [0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0],
        [1200.0, 800.0], [800.0, 1200.0], [500.0, 500.0],
    ])
    offsets = np.zeros((7, 2))
    offsets[-1] = [80.0, -60.0]  # last point: 100 px off, pure noise
    corrs = _make_correspondences(screen, frame_offsets=offsets)
    res = solve_weighted_homography(corrs, ransac_threshold_px=3.0)
    assert res.method == "ransac"
    assert not res.inlier_mask[-1], "outlier should be excluded"
    assert res.inlier_mask[:-1].all(), "clean points should all be inliers"
    # The clean-point fit dominates because the outlier is excluded.
    assert _max_corner_reproj_error(res.H) < 1.0


def test_solver_weight_biases_solution():
    """High weight on a biased point pulls H; low weight leaves H clean.

    Set up: 4 corners (clean, weight 1.0) + 1 interior point with a 4 px
    frame offset. With a high weight, the solver should accommodate the
    biased point (raising clean-point residuals). With a tiny weight,
    the clean fit dominates.
    """
    screen_clean = np.array(
        [[0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0]],
    )
    clean = _make_correspondences(screen_clean, weight=1.0)

    biased_screen = np.array([[1200.0, 800.0]])
    biased_high = _make_correspondences(
        biased_screen, weight=1.0, frame_offsets=np.array([[4.0, 0.0]]),
    )
    biased_low = _make_correspondences(
        biased_screen, weight=0.05, frame_offsets=np.array([[4.0, 0.0]]),
    )

    res_high = solve_weighted_homography(clean + biased_high)
    res_low = solve_weighted_homography(clean + biased_low)

    # High weight: the biased point's residual should be much smaller than
    # the raw 4 px offset because the solver moved H to accommodate it.
    assert res_high.residuals_px[-1] < 3.5
    # Low weight: the biased point should still carry ~the full 4 px offset
    # (the clean corners dominate the fit).
    assert res_low.residuals_px[-1] > 3.5


def test_solver_raises_on_too_few_correspondences():
    """<4 positive-weight correspondences → ValueError."""
    screen = np.array([[0.0, 0.0], [2388.0, 0.0], [1200.0, 800.0]])
    corrs = _make_correspondences(screen)
    with pytest.raises(ValueError, match="≥4"):
        solve_weighted_homography(corrs)


def test_solver_drops_zero_weight_correspondences():
    """Zero-weight entries are skipped; their slots get NaN residual."""
    screen = np.array([
        [0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0],
        [1200.0, 800.0],
    ])
    corrs = _make_correspondences(screen)
    # Zero-out the last entry's weight.
    corrs[-1] = Correspondence(
        screen_xy=corrs[-1].screen_xy,
        frame_xy=corrs[-1].frame_xy,
        weight=0.0,
        source=corrs[-1].source,
    )
    res = solve_weighted_homography(corrs)
    assert not res.inlier_mask[-1]
    assert np.isnan(res.residuals_px[-1])
    # The remaining 4 clean points solve via least-squares.
    assert res.method == "lstsq"
    assert _max_corner_reproj_error(res.H) < 1e-2
