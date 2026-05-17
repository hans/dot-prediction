"""Unit tests for homography_refinement."""

import numpy as np
import pytest

from src.homography_refinement import (
    Correspondence,
    GateRejection,
    anchor_translate,
    apply_quality_gates,
    radius_match_ok,
    resolve_blob_conflicts,
    solve_weighted_homography,
)
from src.local_star_detector import LocalDetection
from src.predicted_positions import PredictedStar


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


# ---------------------------------------------------------------------------
# Quality gates (Phase 1c Step 2)
# ---------------------------------------------------------------------------

def _pred(screen_xy: tuple[float, float], frame_xy: tuple[float, float],
          expected_radius_px: float = 5.0, tpt: int = 0) -> PredictedStar:
    return PredictedStar(
        trial_idx=1, tpt=tpt,
        screen_xy=screen_xy, frame_xy=frame_xy,
        age_s=1.0, expected_radius_px=expected_radius_px,
    )


def _det(prediction: PredictedStar, frame_xy_subpix: tuple[float, float],
         equivalent_radius_px: float = 5.0,
         confidence: float = 100.0) -> LocalDetection:
    return LocalDetection(
        frame_xy_subpix=frame_xy_subpix,
        confidence=confidence,
        equivalent_radius_px=equivalent_radius_px,
        peak_xy=(int(frame_xy_subpix[0]), int(frame_xy_subpix[1])),
        source_prediction=prediction,
    )


# radius_match_ok ------------------------------------------------------------

def test_radius_match_ok_within_band():
    """Observed radius within relative-error band passes."""
    pred = _pred((100, 100), (500, 500), expected_radius_px=5.0)
    # 5.0 ± 50% → well within tau_radius=1.5
    assert radius_match_ok(_det(pred, (500, 500), equivalent_radius_px=7.5))
    assert radius_match_ok(_det(pred, (500, 500), equivalent_radius_px=2.5))


def test_radius_match_ok_rejects_oversized():
    """Observed > 2.5× expected (relative err > 1.5) → reject."""
    pred = _pred((100, 100), (500, 500), expected_radius_px=5.0)
    # 5.0 → 15.0: relative err = 2.0 > 1.5
    assert not radius_match_ok(_det(pred, (500, 500), equivalent_radius_px=15.0))


def test_radius_match_ok_accepts_when_expected_is_zero():
    """No usable size model → gate is permissive."""
    pred = _pred((100, 100), (500, 500), expected_radius_px=0.0)
    assert radius_match_ok(_det(pred, (500, 500), equivalent_radius_px=20.0))


# resolve_blob_conflicts -----------------------------------------------------

def test_resolve_blob_no_conflict_passes_through():
    """Detections spaced > tau apart all survive."""
    p1 = _pred((100, 100), (200, 200))
    p2 = _pred((300, 300), (600, 600))
    dets = [_det(p1, (201, 201)), _det(p2, (601, 601))]
    accepted, rejected = resolve_blob_conflicts(dets, tau_centroid_px=3.0)
    assert len(accepted) == 2
    assert rejected == []


def test_resolve_blob_picks_closest_prediction_to_centroid():
    """Two snapped-together detections → winner is prediction nearest the
    cluster centroid; loser is rejected."""
    # Both detections at ~ (500.0, 500.0) — same blob.
    p_near = _pred((100, 100), (500.5, 500.5))  # close to centroid
    p_far = _pred((300, 300), (510.0, 510.0))   # far from centroid
    d_near = _det(p_near, (500.0, 500.0))
    d_far = _det(p_far, (500.5, 500.5))
    accepted, rejected = resolve_blob_conflicts(
        [d_near, d_far], tau_centroid_px=3.0,
    )
    assert accepted == [d_near]
    assert rejected == [d_far]


def test_resolve_blob_handles_multiple_clusters_independently():
    """Two distinct clusters; each resolved on its own."""
    # Cluster A around (500, 500): two detections snap together; p_a1 closer.
    p_a1 = _pred((100, 100), (500.0, 500.0))
    p_a2 = _pred((110, 110), (510.0, 510.0))
    d_a1 = _det(p_a1, (500.0, 500.0))
    d_a2 = _det(p_a2, (500.5, 500.5))
    # Cluster B around (800, 800): three detections, p_b2 closest.
    p_b1 = _pred((200, 200), (810.0, 810.0))
    p_b2 = _pred((210, 210), (800.0, 800.0))
    p_b3 = _pred((220, 220), (805.0, 815.0))
    d_b1 = _det(p_b1, (800.5, 800.5))
    d_b2 = _det(p_b2, (800.0, 800.0))
    d_b3 = _det(p_b3, (801.0, 800.5))
    # Singleton C far from both.
    p_c = _pred((400, 400), (1500.0, 1500.0))
    d_c = _det(p_c, (1500.0, 1500.0))

    dets = [d_a1, d_a2, d_b1, d_b2, d_b3, d_c]
    accepted, rejected = resolve_blob_conflicts(dets, tau_centroid_px=3.0)
    assert set(accepted) == {d_a1, d_b2, d_c}
    assert set(rejected) == {d_a2, d_b1, d_b3}


def test_resolve_blob_empty_input():
    accepted, rejected = resolve_blob_conflicts([])
    assert accepted == []
    assert rejected == []


def test_resolve_blob_single_link_chain():
    """Three detections in a single-link chain (each < tau from the next)
    cluster together, even though end-to-end exceeds tau."""
    p1 = _pred((100, 100), (500.0, 500.0))
    p2 = _pred((110, 110), (502.0, 500.0))
    p3 = _pred((120, 120), (504.0, 500.0))
    d1 = _det(p1, (500.0, 500.0))
    d2 = _det(p2, (502.0, 500.0))   # 2 px from d1, 2 px from d3
    d3 = _det(p3, (504.0, 500.0))   # 4 px from d1 — would be a separate cluster
                                    # under complete-link, but joined here.
    accepted, rejected = resolve_blob_conflicts([d1, d2, d3], tau_centroid_px=3.0)
    # Cluster centroid = (502, 500); p2 is closest, wins.
    assert accepted == [d2]
    assert set(rejected) == {d1, d3}


# apply_quality_gates --------------------------------------------------------

def test_apply_quality_gates_clean_pass_through():
    p1 = _pred((100, 100), (200, 200))
    p2 = _pred((300, 300), (600, 600))
    dets = [_det(p1, (200, 200)), _det(p2, (600, 600))]
    accepted, rejections = apply_quality_gates(dets)
    assert len(accepted) == 2
    assert rejections == []


def test_apply_quality_gates_rejects_radius_mismatch():
    """Oversized blob is dropped with reason='radius_mismatch'."""
    p_good = _pred((100, 100), (200, 200), expected_radius_px=5.0)
    p_bad = _pred((300, 300), (600, 600), expected_radius_px=5.0)
    d_good = _det(p_good, (200, 200), equivalent_radius_px=5.0)
    d_bad = _det(p_bad, (600, 600), equivalent_radius_px=20.0)
    accepted, rejections = apply_quality_gates([d_good, d_bad])
    assert accepted == [d_good]
    assert rejections == [GateRejection(d_bad, "radius_mismatch")]


def test_apply_quality_gates_reports_same_blob_rejections():
    """Cluster losers are reported with reason='same_blob'."""
    p_near = _pred((100, 100), (500.5, 500.5))
    p_far = _pred((300, 300), (510.0, 510.0))
    d_near = _det(p_near, (500.0, 500.0))
    d_far = _det(p_far, (500.5, 500.5))
    accepted, rejections = apply_quality_gates([d_near, d_far])
    assert accepted == [d_near]
    assert rejections == [GateRejection(d_far, "same_blob")]


def test_apply_quality_gates_radius_runs_before_blob_resolver():
    """A radius-bad detection cannot win the centroid tiebreak: it is
    dropped first, so the radius-good one survives even if the bad one
    was closer to the cluster centroid."""
    # Cluster centroid will be ~ (500.25, 500.0). p_bad's prediction is at
    # the centroid; p_good's prediction is 5 px away. Without the radius
    # gate, p_bad would win — but p_bad has equivalent_radius=20 (>> 2.5×5).
    p_good = _pred((100, 100), (505.0, 500.0), expected_radius_px=5.0)
    p_bad = _pred((300, 300), (500.25, 500.0), expected_radius_px=5.0)
    d_good = _det(p_good, (500.0, 500.0), equivalent_radius_px=5.0)
    d_bad = _det(p_bad, (500.5, 500.0), equivalent_radius_px=20.0)
    accepted, rejections = apply_quality_gates([d_good, d_bad])
    assert accepted == [d_good]
    reasons = sorted(r.reason for r in rejections)
    assert reasons == ["radius_mismatch"]
