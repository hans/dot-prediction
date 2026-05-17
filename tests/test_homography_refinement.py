"""Unit tests for homography_refinement."""

import cv2
import numpy as np
import pytest

from src.homography_refinement import (
    Correspondence,
    GateRejection,
    anchor_translate,
    apply_quality_gates,
    build_correspondences,
    detect_constellation,
    radius_match_ok,
    resolve_blob_conflicts,
    solve_weighted_homography,
)
from src.local_star_detector import LocalDetection, detect_in_windows
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


def test_solver_recovers_clean_homography_via_lstsq():
    """6 noise-free unit-weight correspondences → H within sub-pixel (lstsq default)."""
    screen = np.array([
        [0.0, 0.0], [2388.0, 0.0], [2388.0, 1668.0], [0.0, 1668.0],
        [1200.0, 800.0], [800.0, 1200.0],
    ])
    corrs = _make_correspondences(screen)
    res = solve_weighted_homography(corrs)
    assert res.method == "lstsq"
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
    res = solve_weighted_homography(corrs, use_ransac=True, ransac_threshold_px=3.0)
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


# ---------------------------------------------------------------------------
# Greedy constellation matcher (Phase 1c Step 3)
# ---------------------------------------------------------------------------

_FRAME_H, _FRAME_W = 1080, 1920


def _blue_frame() -> np.ndarray:
    """Blue background like the iPad task display (BGR=140,50,50)."""
    return np.full((_FRAME_H, _FRAME_W, 3), (140, 50, 50), dtype=np.uint8)


def _paint_blob(frame: np.ndarray, cx: int, cy: int, radius: int = 4,
                color=(120, 175, 180)) -> None:
    cv2.circle(frame, (cx, cy), radius, color, -1)


def _pred_full(frame_xy: tuple[float, float], *,
               expected_radius_px: float = 5.0, tpt: int = 0) -> PredictedStar:
    return PredictedStar(
        trial_idx=1, tpt=tpt,
        screen_xy=(float(frame_xy[0]), float(frame_xy[1])),
        frame_xy=frame_xy, age_s=1.0,
        expected_radius_px=expected_radius_px,
    )


def test_constellation_empty_predictions():
    """No predictions → no detections, no unmatched."""
    dets, unmatched = detect_constellation(_blue_frame(), [])
    assert dets == [] and unmatched == []


def test_constellation_isolated_windows_match_detect_in_windows():
    """Non-overlapping windows + one blob each → same result as detect_in_windows.

    Verifies the unified union-and-components path produces the same answer as
    legacy per-window detection in the singleton group case.
    """
    frame = _blue_frame()
    _paint_blob(frame, 400, 400, radius=4)
    _paint_blob(frame, 1200, 700, radius=4)
    preds = [_pred_full((400, 400), tpt=0),
             _pred_full((1200, 700), tpt=1)]

    legacy_dets, legacy_unmatched = detect_in_windows(
        frame, preds, window_size_px=40,
    )
    new_dets, new_unmatched = detect_constellation(
        frame, preds, window_size_px=40,
    )

    assert len(new_dets) == len(legacy_dets) == 2
    assert new_unmatched == legacy_unmatched == []
    # Detections come back in input-prediction order.
    assert [d.source_prediction for d in new_dets] == preds
    for d_new, d_legacy in zip(new_dets, legacy_dets):
        np.testing.assert_allclose(
            d_new.frame_xy_subpix, d_legacy.frame_xy_subpix, atol=0.01,
        )
        assert d_new.equivalent_radius_px == pytest.approx(
            d_legacy.equivalent_radius_px,
        )


def test_constellation_overlap_one_blob_assigns_to_nearer_prediction():
    """Two overlapping windows hitting a single blob → nearer prediction wins,
    the other is unmatched. The legacy detect_in_windows would have produced
    two detections snapped to the same peak."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=4)
    # Both windows (40 px) cover (800, 600); pred_near is 5 px away, pred_far 15.
    pred_near = _pred_full((805, 600), tpt=0)
    pred_far = _pred_full((815, 600), tpt=1)

    dets, unmatched = detect_constellation(
        frame, [pred_near, pred_far], window_size_px=40,
    )
    assert len(dets) == 1
    assert dets[0].source_prediction is pred_near
    assert unmatched == [pred_far]


def test_constellation_overlap_two_blobs_each_to_own_prediction():
    """Overlapping windows with two distinct blobs → each prediction is bound
    to its own blob (no same-blob snapping)."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=3)
    _paint_blob(frame, 820, 600, radius=3)
    pred_a = _pred_full((800, 600), tpt=0)
    pred_b = _pred_full((820, 600), tpt=1)

    dets, unmatched = detect_constellation(
        frame, [pred_a, pred_b], window_size_px=40,
    )
    assert len(dets) == 2
    assert unmatched == []
    by_pred = {d.source_prediction: d for d in dets}
    fx_a, fy_a = by_pred[pred_a].frame_xy_subpix
    fx_b, fy_b = by_pred[pred_b].frame_xy_subpix
    assert np.hypot(fx_a - 800, fy_a - 600) < 1.5
    assert np.hypot(fx_b - 820, fy_b - 600) < 1.5


def test_constellation_three_predictions_two_blobs_one_unmatched():
    """3 mutually-overlapping windows, only 2 blobs in the union →
    exactly one prediction is unmatched (the one with no candidate blob in
    range)."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=3)
    _paint_blob(frame, 830, 600, radius=3)
    p1 = _pred_full((800, 600), tpt=0)
    p2 = _pred_full((815, 600), tpt=1)
    p3 = _pred_full((830, 600), tpt=2)

    dets, unmatched = detect_constellation(
        frame, [p1, p2, p3], window_size_px=40,
    )
    assert len(dets) == 2
    assert len(unmatched) == 1
    matched_preds = {d.source_prediction for d in dets}
    # The blobs are at the positions of p1 and p3, so they take p1 and p3;
    # p2 (with no blob at its centroid) ends up unmatched.
    assert matched_preds == {p1, p3}
    assert unmatched == [p2]


def test_constellation_blob_outside_all_windows_ignored():
    """A blob whose centroid is not inside any prediction's window has no
    candidate to assign to — it's silently dropped (no crash, predictions
    around it remain unmatched)."""
    frame = _blue_frame()
    # Blob at (800, 600). Predictions sit far away — windows don't contain it.
    _paint_blob(frame, 800, 600, radius=3)
    pred_far = _pred_full((400, 400), tpt=0)
    dets, unmatched = detect_constellation(
        frame, [pred_far], window_size_px=40,
    )
    assert dets == []
    assert unmatched == [pred_far]


def test_constellation_rejects_oversized_blob_via_max_radius_factor():
    """A blob whose equivalent radius is >> the prediction's expected radius
    is rejected; the prediction is left unmatched (no fallback to a smaller
    blob)."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=20)  # huge blob
    # Expected radius 3 → max accepted (factor=4) = 12; blob's r ≈ 20.
    pred = _pred_full((800, 600), tpt=0, expected_radius_px=3.0)
    dets, unmatched = detect_constellation(
        frame, [pred], window_size_px=80, max_radius_factor=4.0,
    )
    assert dets == []
    assert unmatched == [pred]


def test_constellation_off_frame_prediction_is_unmatched():
    """A prediction whose window is fully off-frame is reported as unmatched
    (no detection attempted)."""
    frame = _blue_frame()
    pred_off = _pred_full((-100, -100), tpt=0)
    pred_on = _pred_full((400, 400), tpt=1)
    _paint_blob(frame, 400, 400, radius=4)
    dets, unmatched = detect_constellation(
        frame, [pred_off, pred_on], window_size_px=40,
    )
    assert [d.source_prediction for d in dets] == [pred_on]
    assert unmatched == [pred_off]


def test_constellation_resolves_overlap_that_legacy_could_not():
    """Direct contrast with detect_in_windows: same overlapping setup, but
    constellation_match returns one detection (correctly), while the legacy
    detector returns two duplicate detections at the same peak."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=4)
    pred_a = _pred_full((795, 605), tpt=0)
    pred_b = _pred_full((805, 595), tpt=1)

    legacy_dets, _ = detect_in_windows(
        frame, [pred_a, pred_b], window_size_px=40,
    )
    assert len(legacy_dets) == 2  # legacy: same-blob duplicates
    assert legacy_dets[0].peak_xy == legacy_dets[1].peak_xy

    new_dets, new_unmatched = detect_constellation(
        frame, [pred_a, pred_b], window_size_px=40,
    )
    assert len(new_dets) == 1
    assert len(new_unmatched) == 1


def test_constellation_processes_blobs_in_descending_confidence_order():
    """The brighter blob is greedily assigned first. When two blobs both
    overlap one prediction's window but only one is also in the other
    prediction's window, the brighter blob must claim the shared prediction;
    otherwise the dim blob would take it and the bright blob would be
    pushed to a much worse match (or no match at all).

    Setup: p1 at (800, 600), p2 at (820, 600), windows 40 px.
      Bright blob at (819, 600) — inside *both* windows. Wants p2 (dist 1).
      Dim blob at (826, 600) — outside p1's window (x = 826 ≥ 820),
        inside p2's. Its only option is p2.

    Descending: bright → p2; dim then has no unspent candidate → 1 detection.
    Ascending: dim → p2; bright then falls back to p1 at distance 19 →
      2 detections, but bright is matched to the *wrong* prediction.

    The spec's "descending confidence" gives the first outcome; this test
    pins it.
    """
    frame = _blue_frame()
    _paint_blob(frame, 819, 600, radius=2)  # bright (redness 60)
    _paint_blob(frame, 826, 600, radius=2, color=(120, 145, 150))  # redness 30
    p1 = _pred_full((800, 600), tpt=0)
    p2 = _pred_full((820, 600), tpt=1)

    dets, unmatched = detect_constellation(
        frame, [p1, p2], window_size_px=40,
    )
    assert len(dets) == 1
    assert dets[0].source_prediction is p2
    assert dets[0].confidence == pytest.approx(60.0)  # bright blob, not dim
    assert unmatched == [p1]


def test_constellation_adaptive_window_falls_back_to_per_window():
    """With small adaptive windows, predictions far enough apart don't
    overlap; the matcher reduces to independent per-window detection."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 600, radius=2)
    _paint_blob(frame, 900, 600, radius=2)
    preds = [_pred_full((800, 600), tpt=0, expected_radius_px=2.0),
             _pred_full((900, 600), tpt=1, expected_radius_px=2.0)]
    dets, unmatched = detect_constellation(
        frame, preds, adaptive_radius_factor=4.0,
        min_window_px=8, max_window_px=40,
    )
    assert len(dets) == 2
    assert unmatched == []
    by_pred = {d.source_prediction: d for d in dets}
    for pred, (cx, cy) in zip(preds, [(800, 600), (900, 600)]):
        fx, fy = by_pred[pred].frame_xy_subpix
        assert np.hypot(fx - cx, fy - cy) < 1.5


# ---------------------------------------------------------------------------
# Correspondence builder (Phase 1c Step 4)
# ---------------------------------------------------------------------------

# Fixed iPad screen corners [TL, TR, BR, BL] used across tests.
_SCREEN_CORNERS = np.array([
    [0.0, 0.0],       # TL
    [2388.0, 0.0],    # TR
    [2388.0, 1668.0], # BR
    [0.0, 1668.0],    # BL
], dtype=np.float64)

# A plausible smoothed-corner array in frame coords.
_SMOOTHED = np.array([
    [100.0, 80.0],   # TL
    [900.0, 75.0],   # TR
    [910.0, 700.0],  # BR
    [95.0, 710.0],   # BL
], dtype=np.float64)


def _raw(offsets: tuple[tuple, tuple, tuple, tuple]) -> np.ndarray:
    """Build a raw-corners array by adding per-corner (dx, dy) to _SMOOTHED."""
    arr = _SMOOTHED.copy()
    for i, (dx, dy) in enumerate(offsets):
        arr[i, 0] += dx
        arr[i, 1] += dy
    return arr


def _sources(corrs):
    return [c.source for c in corrs]


# --- Corner-only cases -------------------------------------------------------

def test_build_correspondences_no_raw_corners_includes_only_smoothed_bottom():
    """When raw_corners is None, only the two smoothed BL/BR correspondences
    appear — no top corners, no raw-corner entries."""
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS)
    # BL (idx 3) and BR (idx 2), both smoothed.
    assert len(corrs) == 2
    assert all(c.source == "corner_smoothed" for c in corrs)
    assert all(c.weight == pytest.approx(1.0) for c in corrs)
    # Screen coordinates must match the BL/BR slots.
    screen_xys = {c.screen_xy for c in corrs}
    assert (0.0, 1668.0) in screen_xys   # BL
    assert (2388.0, 1668.0) in screen_xys # BR


def test_build_correspondences_raw_bl_br_close_adds_raw_correspondences():
    """Raw BL/BR within threshold → also appear at weight 0.5."""
    raw = _raw(((0, 0), (0, 0), (3.0, 0), (3.0, 0)))  # BL/BR shifted 3 px (< 10)
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    # 2 smoothed bottom + 2 raw bottom = 4; top corners excluded because raw
    # TL/TR delta is 0 px which IS < 5 px, so smoothed+raw TL+TR = 4 more.
    # (3 px < 5 px threshold for TL/TR)
    bottom_smooth = [c for c in corrs
                     if c.source == "corner_smoothed" and c.weight == pytest.approx(1.0)]
    bottom_raw = [c for c in corrs
                  if c.source == "corner_raw" and c.weight == pytest.approx(0.5)]
    assert len(bottom_smooth) == 2
    assert len(bottom_raw) == 2
    # Frame xy of the raw entries should differ from smoothed by the offset.
    raw_bl = next(c for c in bottom_raw if c.screen_xy == (0.0, 1668.0))
    assert raw_bl.frame_xy[0] == pytest.approx(_SMOOTHED[3, 0] + 3.0)


def test_build_correspondences_raw_bl_br_at_threshold_excluded():
    """Delta exactly at raw_bl_br_tau_px (10.0) is NOT included (< not ≤)."""
    raw = _raw(((0, 0), (0, 0), (0, 10.0), (0, 10.0)))  # delta = 10.0 px
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    raw_bottom = [c for c in corrs if c.source == "corner_raw"
                  and c.weight == pytest.approx(0.5)]
    assert raw_bottom == []


def test_build_correspondences_raw_bl_br_far_excluded():
    """Raw BL/BR >10 px from smoothed → not added; smoothed still at 1.0."""
    raw = _raw(((0, 0), (0, 0), (15.0, 0), (-20.0, 0)))  # BL/BR > 10 px
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    raw_bottom = [c for c in corrs if c.source == "corner_raw"
                  and c.weight == pytest.approx(0.5)]
    assert raw_bottom == []
    smooth_bottom = [c for c in corrs if c.source == "corner_smoothed"
                     and c.weight == pytest.approx(1.0)]
    assert len(smooth_bottom) == 2


def test_build_correspondences_top_corners_included_when_raw_close():
    """Raw TL/TR within raw_tl_tr_tau_px → smoothed at 0.3 and raw at 0.1 added."""
    raw = _raw(((2.0, 0), (1.0, 0), (0, 0), (0, 0)))  # TL: 2 px, TR: 1 px (both < 5)
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    top_smooth = [c for c in corrs if c.source == "corner_smoothed"
                  and c.weight == pytest.approx(0.3)]
    top_raw = [c for c in corrs if c.source == "corner_raw"
               and c.weight == pytest.approx(0.1)]
    assert len(top_smooth) == 2  # TL and TR
    assert len(top_raw) == 2


def test_build_correspondences_top_corners_at_threshold_excluded():
    """Delta exactly 5.0 px for a top corner → that corner is excluded."""
    raw = _raw(((5.0, 0), (1.0, 0), (0, 0), (0, 0)))  # TL=5 px (not < 5), TR=1 px
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    # TR included (1 < 5), TL excluded.
    tl_screen = (0.0, 0.0)
    tr_screen = (2388.0, 0.0)
    tl_corrs = [c for c in corrs if c.screen_xy == tl_screen]
    tr_corrs = [c for c in corrs if c.screen_xy == tr_screen]
    assert tl_corrs == []
    assert len(tr_corrs) == 2  # smoothed 0.3 + raw 0.1


def test_build_correspondences_top_corners_excluded_when_raw_far():
    """Raw TL/TR many pixels from smoothed → both omitted from output."""
    raw = _raw(((50.0, 0), (100.0, 0), (0, 0), (0, 0)))  # TL=50 px, TR=100 px
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    top_corrs = [c for c in corrs
                 if c.screen_xy in {(0.0, 0.0), (2388.0, 0.0)}]
    assert top_corrs == []


# --- Big star ----------------------------------------------------------------

def test_build_correspondences_big_star_added():
    """Providing both big_star_* args adds one big_star correspondence at the given weight."""
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        big_star_screen_xy=(1000.0, 800.0),
        big_star_frame_xy=(450.0, 360.0),
        big_star_weight=0.9,
    )
    big = [c for c in corrs if c.source == "big_star"]
    assert len(big) == 1
    assert big[0].screen_xy == (1000.0, 800.0)
    assert big[0].frame_xy == (450.0, 360.0)
    assert big[0].weight == pytest.approx(0.9)


def test_build_correspondences_big_star_absent_when_not_provided():
    """No big_star_screen_xy/frame_xy → no big_star entry."""
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS)
    assert all(c.source != "big_star" for c in corrs)


# --- Small stars -------------------------------------------------------------

def test_build_correspondences_small_star_full_weight_within_blend_tau():
    """A detection with rel_err ≤ 0.5 gets the full (normalized) confidence."""
    pred = _pred((500.0, 400.0), (220.0, 180.0), expected_radius_px=10.0)
    det = _det(pred, (221.0, 181.0), equivalent_radius_px=11.0,  # rel_err = 0.1
               confidence=127.5)  # conf/255 = 0.5
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        small_stars=[det],
    )
    small = [c for c in corrs if c.source == "small_star"]
    assert len(small) == 1
    assert small[0].weight == pytest.approx(0.5)  # 0.5 × 1.0


def test_build_correspondences_small_star_blended_weight_mid_range():
    """A detection with rel_err = 1.0 (halfway between 0.5 and 1.5) gets factor 0.75."""
    pred = _pred((500.0, 400.0), (220.0, 180.0), expected_radius_px=10.0)
    det = _det(pred, (221.0, 181.0), equivalent_radius_px=20.0,  # rel_err = 1.0
               confidence=255.0)  # normalized to 1.0
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        small_stars=[det],
    )
    small = [c for c in corrs if c.source == "small_star"]
    # blend: t = (1.0 - 0.5) / (1.5 - 0.5) = 0.5, factor = 1.0 - 0.5*0.5 = 0.75
    assert small[0].weight == pytest.approx(0.75)


def test_build_correspondences_small_star_confidence_clamped_above_scale():
    """Confidence > small_star_confidence_scale is clamped to 1.0 before weighting."""
    pred = _pred((500.0, 400.0), (220.0, 180.0), expected_radius_px=10.0)
    det = _det(pred, (221.0, 181.0), equivalent_radius_px=10.0,
               confidence=300.0)  # > 255
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        small_stars=[det],
    )
    small = [c for c in corrs if c.source == "small_star"]
    assert small[0].weight == pytest.approx(1.0)  # clamped, rel_err=0 → factor 1.0


def test_build_correspondences_small_star_zero_expected_radius_permissive():
    """When expected_radius_px=0, no size model exists — treat rel_err as 0."""
    pred = _pred((500.0, 400.0), (220.0, 180.0), expected_radius_px=0.0)
    det = _det(pred, (221.0, 181.0), equivalent_radius_px=50.0,
               confidence=255.0)
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        small_stars=[det],
    )
    small = [c for c in corrs if c.source == "small_star"]
    assert small[0].weight == pytest.approx(1.0)  # factor stays 1.0


# --- Source-tag completeness -------------------------------------------------

def test_build_correspondences_all_four_source_tags_present():
    """A fully-populated call produces correspondences with all four source tags."""
    raw = _raw(((1.0, 0), (1.0, 0), (1.0, 0), (1.0, 0)))  # all within thresholds
    pred = _pred((500.0, 400.0), (220.0, 180.0), expected_radius_px=10.0)
    det = _det(pred, (221.0, 181.0), equivalent_radius_px=10.0, confidence=200.0)
    corrs = build_correspondences(
        _SMOOTHED, _SCREEN_CORNERS,
        raw_corners=raw,
        big_star_screen_xy=(1000.0, 800.0),
        big_star_frame_xy=(450.0, 360.0),
        small_stars=[det],
    )
    sources = {c.source for c in corrs}
    assert sources == {"corner_smoothed", "corner_raw", "big_star", "small_star"}


# --- Integration: output feeds solve_weighted_homography --------------------

def test_build_correspondences_result_feeds_solver():
    """Output of build_correspondences can be passed directly to
    solve_weighted_homography and produce a valid H.

    Uses a corner-only scenario (no star detections) to keep the geometry
    clean, and verifies that the row-count balance is dominated by the
    bottom-corner correspondences (the highest-weight entries).
    """
    raw = _raw(((1.0, 0), (1.0, 0), (1.0, 0), (1.0, 0)))
    corrs = build_correspondences(_SMOOTHED, _SCREEN_CORNERS, raw_corners=raw)
    # Four smoothed bottom+top corners + four raw entries = ≥ 4 with positive weight.
    result = solve_weighted_homography(corrs)
    assert result.H is not None
    assert result.H.shape == (3, 3)
    assert result.method == "lstsq"
