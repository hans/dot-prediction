"""Fixture-only tests for homography_solver (no video I/O)."""

import numpy as np
import pandas as pd
import pytest

from src.homography_solver import (
    behavior_to_screen,
    big_star_residuals,
    calibrate_box_position,
    fit_homography,
    fit_per_frame_homography,
    loo_residuals,
)

SCREEN_W = 2388
SCREEN_H = 1668

# Known projective transform (non-trivial perspective component).
_H_TRUE = np.array(
    [
        [0.30, 0.01, 200.0],
        [0.02, 0.25, 150.0],
        [1e-5, 2e-5, 1.0],
    ],
    dtype=np.float64,
)
_H_TRUE /= _H_TRUE[2, 2]

# Calibrated box corner positions in screen-px (arbitrary but realistic).
_BOX_BL_SCREEN = (50.0, 490.0)
_BOX_BR_SCREEN = (2330.0, 490.0)

# Screen corners in TL/TR/BR/BL order (matches the labeling notebook).
_SCREEN_CORNERS_XYSRC = [
    (0.0, 0.0),
    (float(SCREEN_W), 0.0),
    (float(SCREEN_W), float(SCREEN_H)),
    (0.0, float(SCREEN_H)),
]


def _project(H_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H_mat @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _label_row(
    frame_idx: int,
    label_type: str,
    x_frame: float,
    y_frame: float,
    visible: bool = True,
    quality: str | None = "confident",
) -> dict:
    return {
        "frame_idx": frame_idx,
        "label_type": label_type,
        "visible": visible,
        "x_frame": x_frame,
        "y_frame": y_frame,
        "quality": quality if visible else None,
        "notes": "",
        "saved_at": "2026-01-01T00:00:00+00:00",
    }


def _calib_labels(frame_idx: int) -> list[dict]:
    """Synthetic labels for one calibration frame using _H_TRUE."""
    label_types_screen = [
        ("screen_tl", _SCREEN_CORNERS_XYSRC[0]),
        ("screen_tr", _SCREEN_CORNERS_XYSRC[1]),
        ("screen_br", _SCREEN_CORNERS_XYSRC[2]),
        ("screen_bl", _SCREEN_CORNERS_XYSRC[3]),
    ]
    rows = []
    for lt, (sx, sy) in label_types_screen:
        fx, fy = _project(_H_TRUE, sx, sy)
        rows.append(_label_row(frame_idx, lt, fx, fy))

    bl_fx, bl_fy = _project(_H_TRUE, *_BOX_BL_SCREEN)
    br_fx, br_fy = _project(_H_TRUE, *_BOX_BR_SCREEN)
    rows.append(_label_row(frame_idx, "box_bl", bl_fx, bl_fy))
    rows.append(_label_row(frame_idx, "box_br", br_fx, br_fy))
    return rows


def _per_frame_labels(frame_idx: int) -> list[dict]:
    """Synthetic labels for Step-2 fitting (only the 4 required anchors)."""
    rows = []
    for lt, (sx, sy) in [
        ("screen_bl", _SCREEN_CORNERS_XYSRC[3]),
        ("screen_br", _SCREEN_CORNERS_XYSRC[2]),
        ("box_bl", _BOX_BL_SCREEN),
        ("box_br", _BOX_BR_SCREEN),
    ]:
        fx, fy = _project(_H_TRUE, sx, sy)
        rows.append(_label_row(frame_idx, lt, fx, fy))
    return rows


def _per_frame_labels_5pt(frame_idx: int, bs_screen: tuple[float, float]) -> list[dict]:
    """Synthetic labels for Step-2 fitting plus a big_star anchor."""
    rows = _per_frame_labels(frame_idx)
    fx, fy = _project(_H_TRUE, *bs_screen)
    rows.append(_label_row(frame_idx, "big_star", fx, fy))
    return rows


# ---------------------------------------------------------------------------
# calibrate_box_position
# ---------------------------------------------------------------------------


def test_calibrate_recovers_screen_coords_single_frame():
    """With one calibration frame and exact labels, box screen-coords are recovered."""
    labels_df = pd.DataFrame(_calib_labels(100))
    result = calibrate_box_position(
        labels_df, screen_w_px=SCREEN_W, screen_h_px=SCREEN_H
    )

    np.testing.assert_allclose(result["box_bl_screen"], _BOX_BL_SCREEN, atol=1e-4)
    np.testing.assert_allclose(result["box_br_screen"], _BOX_BR_SCREEN, atol=1e-4)
    assert result["calibration_frames"] == [100]


def test_calibrate_median_across_multiple_frames():
    """With multiple identical frames, median equals the per-frame estimate."""
    rows = _calib_labels(100) + _calib_labels(200)
    labels_df = pd.DataFrame(rows)
    result = calibrate_box_position(
        labels_df, screen_w_px=SCREEN_W, screen_h_px=SCREEN_H
    )

    np.testing.assert_allclose(result["box_bl_screen"], _BOX_BL_SCREEN, atol=1e-4)
    assert set(result["calibration_frames"]) == {100, 200}
    spread = result["spread_screen_px"]
    # Two identical frames → zero spread on every coordinate.
    assert spread["box_bl"]["max_minus_min_x"] == pytest.approx(0.0, abs=1e-6)
    assert spread["box_bl"]["max_minus_min_y"] == pytest.approx(0.0, abs=1e-6)
    assert spread["box_br"]["max_minus_min_x"] == pytest.approx(0.0, abs=1e-6)


def test_calibrate_raises_when_no_qualifying_frame():
    """Raises ValueError if no frame has all required labels visible."""
    labels_df = pd.DataFrame([_label_row(100, "screen_bl", 200.0, 150.0)])
    with pytest.raises(ValueError):
        calibrate_box_position(labels_df)


def test_calibrate_skips_frame_with_invisible_label():
    """A frame where one of the 6 required labels is not-visible is skipped."""
    rows = _calib_labels(100)
    # Mark screen_tl as not-visible in frame 100 — should be skipped.
    for r in rows:
        if r["label_type"] == "screen_tl":
            r["visible"] = False
            r["quality"] = None
    # Add frame 200 which is fully valid.
    rows += _calib_labels(200)
    labels_df = pd.DataFrame(rows)
    result = calibrate_box_position(
        labels_df, screen_w_px=SCREEN_W, screen_h_px=SCREEN_H
    )

    assert result["calibration_frames"] == [200]


# ---------------------------------------------------------------------------
# fit_per_frame_homography
# ---------------------------------------------------------------------------


def test_fit_recovers_true_H():
    """Recovered H equals _H_TRUE (normalised) within numerical precision."""
    labels_df = pd.DataFrame(_per_frame_labels(100))
    result = fit_per_frame_homography(
        labels_df,
        _BOX_BL_SCREEN,
        _BOX_BR_SCREEN,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
    )

    included = result[result.excluded_reason == ""]
    assert len(included) == 1, "Expected exactly one included frame"

    row = included.iloc[0]
    H_rec = np.array(
        [
            [row.h00, row.h01, row.h02],
            [row.h10, row.h11, row.h12],
            [row.h20, row.h21, row.h22],
        ]
    )
    H_exp = _H_TRUE / _H_TRUE[2, 2]
    np.testing.assert_allclose(H_rec, H_exp, atol=1e-5)
    assert row.n_correspondences == 4


def test_fit_missing_anchor_excluded():
    """Frame missing a required anchor gets excluded_reason set and NaN H."""
    rows = [
        _label_row(100, "screen_bl", 200.0, 150.0),
        _label_row(100, "screen_br", 800.0, 155.0),
        # box_bl and box_br intentionally absent
    ]
    labels_df = pd.DataFrame(rows)
    result = fit_per_frame_homography(labels_df, _BOX_BL_SCREEN, _BOX_BR_SCREEN)

    assert len(result) == 1
    row = result.iloc[0]
    assert row.excluded_reason != ""
    assert np.isnan(row.h00)
    assert row.n_correspondences == 0


def test_fit_not_visible_anchor_excluded():
    """Frame where a required anchor is labeled not-visible is excluded."""
    rows = _per_frame_labels(100)
    for r in rows:
        if r["label_type"] == "screen_bl":
            r["visible"] = False
            r["quality"] = None
    labels_df = pd.DataFrame(rows)
    result = fit_per_frame_homography(labels_df, _BOX_BL_SCREEN, _BOX_BR_SCREEN)

    row = result.iloc[0]
    assert row.excluded_reason == "missing_screen_bl"


def test_fit_approximate_quality_excluded_by_default():
    """Frames with quality='approximate' are excluded from the default quality set."""
    rows = _per_frame_labels(100)
    for r in rows:
        if r["label_type"] == "screen_bl":
            r["quality"] = "approximate"
    labels_df = pd.DataFrame(rows)
    result = fit_per_frame_homography(labels_df, _BOX_BL_SCREEN, _BOX_BR_SCREEN)

    assert result.iloc[0].excluded_reason == "missing_screen_bl"


def test_fit_approximate_quality_included_when_relaxed():
    """Frames with quality='approximate' are included when include_qualities is relaxed."""
    rows = _per_frame_labels(100)
    for r in rows:
        r["quality"] = "approximate"
    labels_df = pd.DataFrame(rows)
    result = fit_per_frame_homography(
        labels_df,
        _BOX_BL_SCREEN,
        _BOX_BR_SCREEN,
        include_qualities={"confident", "approximate"},
    )

    assert result.iloc[0].excluded_reason == ""
    assert not np.isnan(result.iloc[0].h00)


# ---------------------------------------------------------------------------
# 5-anchor fit + LOO residuals
# ---------------------------------------------------------------------------


_BIG_STAR_SCREEN = (1200.0, 800.0)  # roughly screen interior


def test_fit_homography_recovers_true_H_4pt():
    """Direct fit_homography helper recovers H from 4 exact correspondences."""
    screen_pts = np.array(
        [
            [0.0, float(SCREEN_H)],
            [float(SCREEN_W), float(SCREEN_H)],
            list(_BOX_BL_SCREEN),
            list(_BOX_BR_SCREEN),
        ]
    )
    frame_pts = np.array([_project(_H_TRUE, sx, sy) for sx, sy in screen_pts])
    H = fit_homography(screen_pts, frame_pts)
    np.testing.assert_allclose(H, _H_TRUE / _H_TRUE[2, 2], atol=1e-5)


def test_fit_per_frame_homography_recovers_true_H_5pt():
    """5-anchor fit includes big_star and recovers H exactly."""
    labels_df = pd.DataFrame(_per_frame_labels_5pt(100, _BIG_STAR_SCREEN))
    result = fit_per_frame_homography(
        labels_df,
        _BOX_BL_SCREEN,
        _BOX_BR_SCREEN,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        include_big_star=True,
        big_star_screen_lookup={100: _BIG_STAR_SCREEN},
    )
    row = result.iloc[0]
    assert row.excluded_reason == ""
    assert row.n_correspondences == 5
    H_rec = np.array(
        [
            [row.h00, row.h01, row.h02],
            [row.h10, row.h11, row.h12],
            [row.h20, row.h21, row.h22],
        ]
    )
    np.testing.assert_allclose(H_rec, _H_TRUE / _H_TRUE[2, 2], atol=1e-5)


def test_fit_per_frame_homography_falls_back_to_4pt_without_big_star_lookup():
    """include_big_star=True but no lookup entry → 4-anchor fit."""
    labels_df = pd.DataFrame(_per_frame_labels_5pt(100, _BIG_STAR_SCREEN))
    result = fit_per_frame_homography(
        labels_df,
        _BOX_BL_SCREEN,
        _BOX_BR_SCREEN,
        include_big_star=True,
        big_star_screen_lookup={},  # no entry for frame 100
    )
    assert result.iloc[0].n_correspondences == 4
    assert result.iloc[0].excluded_reason == ""


def test_fit_per_frame_homography_4pt_when_include_big_star_false():
    """include_big_star=False ignores hand-labeled big_star — legacy semantics."""
    labels_df = pd.DataFrame(_per_frame_labels_5pt(100, _BIG_STAR_SCREEN))
    result = fit_per_frame_homography(
        labels_df,
        _BOX_BL_SCREEN,
        _BOX_BR_SCREEN,
        include_big_star=False,
        big_star_screen_lookup={100: _BIG_STAR_SCREEN},
    )
    assert result.iloc[0].n_correspondences == 4


def test_loo_residuals_zero_on_exact_5pt_fit():
    """5 exact correspondences → all LOO residuals are ~0."""
    screen_pts = np.array(
        [
            [0.0, float(SCREEN_H)],
            [float(SCREEN_W), float(SCREEN_H)],
            list(_BOX_BL_SCREEN),
            list(_BOX_BR_SCREEN),
            list(_BIG_STAR_SCREEN),
        ]
    )
    frame_pts = np.array([_project(_H_TRUE, sx, sy) for sx, sy in screen_pts])
    loo = loo_residuals(screen_pts, frame_pts)
    assert len(loo) == 5
    for r in loo:
        assert r == pytest.approx(0.0, abs=1e-3)


def test_loo_residuals_nan_when_fewer_than_5():
    """Below 5 anchors LOO is underdetermined — returns NaN per anchor."""
    screen_pts = np.array(
        [
            [0.0, float(SCREEN_H)],
            [float(SCREEN_W), float(SCREEN_H)],
            list(_BOX_BL_SCREEN),
            list(_BOX_BR_SCREEN),
        ]
    )
    frame_pts = np.array([_project(_H_TRUE, sx, sy) for sx, sy in screen_pts])
    loo = loo_residuals(screen_pts, frame_pts)
    assert len(loo) == 4
    for r in loo:
        assert np.isnan(r)


def test_loo_residuals_detects_outlier_anchor():
    """Perturbing one anchor → its LOO residual recovers the shift exactly.

    With 5 anchors and only one perturbed (big_star, by 20 px), the LOO that
    drops big_star refits from 4 *clean* exact correspondences → recovers the
    true H → projected big_star sits at the true (un-shifted) frame-xy →
    residual to the shifted input equals the shift magnitude (20 px). The
    other LOOs include the perturbed anchor in the fit and absorb it,
    producing larger residuals at the dropped clean anchor.
    """
    screen_pts = np.array(
        [
            [0.0, float(SCREEN_H)],
            [float(SCREEN_W), float(SCREEN_H)],
            list(_BOX_BL_SCREEN),
            list(_BOX_BR_SCREEN),
            list(_BIG_STAR_SCREEN),
        ]
    )
    frame_pts = np.array([_project(_H_TRUE, sx, sy) for sx, sy in screen_pts])
    # Shift big_star (index 4) by 20 frame-px.
    frame_pts[4] = (frame_pts[4][0] + 20.0, frame_pts[4][1])
    loo = loo_residuals(screen_pts, frame_pts)
    # Dropping the outlier recovers true H; residual to shifted input == shift.
    assert loo[4] == pytest.approx(20.0, abs=0.01)
    # All other LOO residuals are non-zero (the fit was contaminated by the
    # outlier) and demonstrate the diagnostic signal.
    for i in range(4):
        assert loo[i] > 1.0


# ---------------------------------------------------------------------------
# big_star_residuals
# ---------------------------------------------------------------------------


def _make_per_frame_h(frame_idx: int, H_mat: np.ndarray) -> pd.DataFrame:
    H_mat = H_mat / H_mat[2, 2]
    return pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                "h00": H_mat[0, 0],
                "h01": H_mat[0, 1],
                "h02": H_mat[0, 2],
                "h10": H_mat[1, 0],
                "h11": H_mat[1, 1],
                "h12": H_mat[1, 2],
                "h20": H_mat[2, 0],
                "h21": H_mat[2, 1],
                "h22": H_mat[2, 2],
                "n_correspondences": 4,
                "excluded_reason": "",
            }
        ]
    )


def test_behavior_to_screen():
    """behavior_to_screen applies URL-bar, x-padding, and max_y normalisation."""
    URL_H, X_PAD, MY = 272, 233, 0.75

    # true_y=max_y_coord → screen bottom; true_x=0 → left canvas edge.
    sx, sy = behavior_to_screen(
        0.0,
        MY,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=URL_H,
        canvas_x_pad_px=X_PAD,
        max_y_coord=MY,
    )
    assert sy == pytest.approx(SCREEN_H, abs=1.0)
    assert sx == pytest.approx(X_PAD, abs=1e-6)

    # true_x=1 → right canvas edge.
    sx1, _ = behavior_to_screen(
        1.0,
        0.0,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=URL_H,
        canvas_x_pad_px=X_PAD,
        max_y_coord=MY,
    )
    assert sx1 == pytest.approx(SCREEN_W - X_PAD, abs=1e-6)

    # true_y=0 → url_bar_h_px.
    _, sy0 = behavior_to_screen(
        0.0,
        0.0,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=URL_H,
        canvas_x_pad_px=X_PAD,
        max_y_coord=MY,
    )
    assert sy0 == pytest.approx(URL_H, abs=1e-6)

    # Degenerate: url_bar_h_px=0, canvas_x_pad_px=0, max_y_coord=1 → old formula.
    sx, sy = behavior_to_screen(
        0.5,
        0.4,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=0,
        canvas_x_pad_px=0,
        max_y_coord=1.0,
    )
    assert sx == pytest.approx(0.5 * SCREEN_W, abs=1e-6)
    assert sy == pytest.approx(0.4 * SCREEN_H, abs=1e-6)


def test_big_star_residual_zero_with_true_H():
    """When H and big_star screen-xy are exact, residual = 0."""
    frame_idx = 100
    true_x, true_y = 0.5, 0.4
    # Use degenerate url_bar_h=0, max_y=1 so star_sx/sy = true * screen dims.
    star_sx = true_x * SCREEN_W
    star_sy = true_y * SCREEN_H
    star_fx, star_fy = _project(_H_TRUE, star_sx, star_sy)

    per_frame_h = _make_per_frame_h(frame_idx, _H_TRUE)
    labels_df = pd.DataFrame([_label_row(frame_idx, "big_star", star_fx, star_fy)])
    trials_df = pd.DataFrame(
        [
            {
                "trial_idx": 1,
                "tpt": 0,
                "true_x": true_x,
                "true_y": true_y,
                "video_frame_reveal": frame_idx - 10,
            }
        ]
    )

    result = big_star_residuals(
        per_frame_h,
        labels_df,
        trials_df,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=0,
        canvas_x_pad_px=0,
        max_y_coord=1.0,
    )

    assert len(result) == 1
    assert result.iloc[0]["residual_px"] == pytest.approx(0.0, abs=1e-4)


def test_big_star_skips_excluded_frame():
    """Frame with a valid big_star label but excluded H is omitted from output."""
    frame_idx = 100
    star_fx, star_fy = 300.0, 250.0

    per_frame_h = pd.DataFrame(
        [
            {
                "frame_idx": frame_idx,
                **{
                    col: float("nan")
                    for col in [
                        "h00",
                        "h01",
                        "h02",
                        "h10",
                        "h11",
                        "h12",
                        "h20",
                        "h21",
                        "h22",
                    ]
                },
                "n_correspondences": 0,
                "excluded_reason": "missing_screen_bl",
            }
        ]
    )
    labels_df = pd.DataFrame([_label_row(frame_idx, "big_star", star_fx, star_fy)])
    trials_df = pd.DataFrame(
        [
            {
                "trial_idx": 1,
                "tpt": 0,
                "true_x": 0.5,
                "true_y": 0.4,
                "video_frame_reveal": frame_idx - 5,
            }
        ]
    )

    result = big_star_residuals(
        per_frame_h,
        labels_df,
        trials_df,
        url_bar_h_px=0,
        canvas_x_pad_px=0,
        max_y_coord=1.0,
    )
    assert result.empty


def test_big_star_skips_not_visible():
    """big_star label marked not-visible is not included in residuals."""
    frame_idx = 100
    per_frame_h = _make_per_frame_h(frame_idx, _H_TRUE)
    labels_df = pd.DataFrame(
        [_label_row(frame_idx, "big_star", 300.0, 250.0, visible=False)]
    )
    trials_df = pd.DataFrame(
        [
            {
                "trial_idx": 1,
                "tpt": 0,
                "true_x": 0.5,
                "true_y": 0.4,
                "video_frame_reveal": frame_idx - 5,
            }
        ]
    )

    result = big_star_residuals(
        per_frame_h,
        labels_df,
        trials_df,
        url_bar_h_px=0,
        canvas_x_pad_px=0,
        max_y_coord=1.0,
    )
    assert result.empty


def test_big_star_uses_most_recent_reveal():
    """When multiple reveals exist before frame_idx, the most recent is used."""
    frame_idx = 100
    true_x, true_y = 0.6, 0.3
    star_sx = true_x * SCREEN_W
    star_sy = true_y * SCREEN_H
    star_fx, star_fy = _project(_H_TRUE, star_sx, star_sy)

    per_frame_h = _make_per_frame_h(frame_idx, _H_TRUE)
    labels_df = pd.DataFrame([_label_row(frame_idx, "big_star", star_fx, star_fy)])
    # Two reveals: the later one (frame 90) has the matching true_x/true_y.
    trials_df = pd.DataFrame(
        [
            {
                "trial_idx": 1,
                "tpt": 0,
                "true_x": 0.1,
                "true_y": 0.1,
                "video_frame_reveal": 50,
            },
            {
                "trial_idx": 1,
                "tpt": 1,
                "true_x": true_x,
                "true_y": true_y,
                "video_frame_reveal": 90,
            },
        ]
    )

    result = big_star_residuals(
        per_frame_h,
        labels_df,
        trials_df,
        screen_w_px=SCREEN_W,
        screen_h_px=SCREEN_H,
        url_bar_h_px=0,
        canvas_x_pad_px=0,
        max_y_coord=1.0,
    )
    assert len(result) == 1
    assert result.iloc[0]["residual_px"] == pytest.approx(0.0, abs=1e-4)
    assert result.iloc[0]["tpt"] == 1
