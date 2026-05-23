"""Unit tests for src/gaze_projection.py."""

from __future__ import annotations

import numpy as np
import pytest

from gaze_projection import (
    is_on_screen,
    lerp_homography_at_frac,
    project_video_to_screen,
    screen_to_canvas,
    tobii_ts_to_behavior_ms,
    tobii_ts_to_video_frame_frac,
)
from homography_solver import behavior_to_screen


def test_tobii_ts_to_behavior_ms_linear() -> None:
    ts_us = np.array([0, 1_000_000, 2_500_000], dtype=np.int64)
    out = tobii_ts_to_behavior_ms(ts_us, slope_ms_per_s=1000.03, intercept_ms=303290.7)
    expected = np.array([303290.7, 304290.73, 305790.775])
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_tobii_ts_to_video_frame_frac_zero_and_boundary() -> None:
    fps = 24.954763666979154
    # ts=0 → frame 0
    assert tobii_ts_to_video_frame_frac(np.array([0]), fps)[0] == 0.0
    # ts at exactly one frame's worth of µs → 1.0 (integer)
    one_frame_us = int(round(1e6 / fps))
    out = tobii_ts_to_video_frame_frac(np.array([one_frame_us]), fps)
    np.testing.assert_allclose(out[0], 1.0, atol=2e-3)  # int rounding


def _sample_H() -> np.ndarray:
    # Mild perspective transform — not identity, not degenerate.
    return np.array(
        [
            [1.2, 0.05, 30.0],
            [0.04, 1.1, 20.0],
            [1e-4, 2e-4, 1.0],
        ]
    )


def test_lerp_homography_at_frac_endpoints_and_midpoint() -> None:
    h_lo = np.full((3, 3), 1.0)
    h_hi = np.full((3, 3), 3.0)
    per_frame = np.stack([h_lo, h_hi], axis=0)  # frames 0, 1

    np.testing.assert_array_equal(lerp_homography_at_frac(per_frame, 0.0), h_lo)
    np.testing.assert_array_equal(lerp_homography_at_frac(per_frame, 1.0), h_hi)
    np.testing.assert_array_equal(
        lerp_homography_at_frac(per_frame, 0.5), np.full((3, 3), 2.0)
    )


def test_lerp_homography_at_frac_nan_flanker_returns_none() -> None:
    h_good = _sample_H()
    h_bad = np.full((3, 3), np.nan)
    per_frame = np.stack([h_good, h_bad], axis=0)
    assert lerp_homography_at_frac(per_frame, 0.7) is None
    # Even at exactly-integer fractional positions, NaN flanker rules out.
    per_frame2 = np.stack([h_good, h_bad, h_good], axis=0)
    assert lerp_homography_at_frac(per_frame2, 1.0) is None  # ceil==floor=1, bad


def test_lerp_homography_at_frac_last_frame_no_oob() -> None:
    h0 = _sample_H()
    h1 = _sample_H() * 1.1
    per_frame = np.stack([h0, h1], axis=0)
    # frame_frac at exactly N-1 should not index past the array.
    np.testing.assert_array_equal(lerp_homography_at_frac(per_frame, 1.0), h1)
    # Above N-1 clips to N-1.
    np.testing.assert_array_equal(lerp_homography_at_frac(per_frame, 5.0), h1)


def test_project_video_to_screen_roundtrip_single_H() -> None:
    H = _sample_H()
    # Push known screen point through forward H to get a frame point.
    s_pt = np.array([500.0, 800.0, 1.0])
    f_vec = H @ s_pt
    fx, fy = f_vec[0] / f_vec[2], f_vec[1] / f_vec[2]
    # Now invert via the helper.
    sx, sy = project_video_to_screen(np.array([fx]), np.array([fy]), H)
    np.testing.assert_allclose([sx[0], sy[0]], [500.0, 800.0], atol=1e-6)


def test_project_video_to_screen_batched_H() -> None:
    H1 = _sample_H()
    H2 = _sample_H() * np.array([[1.0, 1.0, 1.2], [1.0, 1.0, 0.9], [1.0, 1.0, 1.0]])
    H_batch = np.stack([H1, H2], axis=0)

    # Different forward-projected screen points per H.
    s_pts = np.array([[400.0, 700.0, 1.0], [1200.0, 900.0, 1.0]])
    f_vecs = np.einsum("nij,nj->ni", H_batch, s_pts)
    fx = f_vecs[:, 0] / f_vecs[:, 2]
    fy = f_vecs[:, 1] / f_vecs[:, 2]

    sx, sy = project_video_to_screen(fx, fy, H_batch)
    np.testing.assert_allclose(sx, [400.0, 1200.0], atol=1e-6)
    np.testing.assert_allclose(sy, [700.0, 900.0], atol=1e-6)


def test_project_video_to_screen_nan_propagation() -> None:
    H = _sample_H()
    gx = np.array([100.0, np.nan, 200.0])
    gy = np.array([50.0, 60.0, np.nan])
    sx, sy = project_video_to_screen(gx, gy, H)
    assert np.isnan(sx[1]) and np.isnan(sy[1])
    assert np.isnan(sx[2]) and np.isnan(sy[2])
    assert not np.isnan(sx[0]) and not np.isnan(sy[0])


def test_project_video_to_screen_batched_nan_H() -> None:
    H_good = _sample_H()
    H_bad = np.full((3, 3), np.nan)
    H_batch = np.stack([H_good, H_bad], axis=0)
    sx, sy = project_video_to_screen(np.array([10.0, 10.0]), np.array([20.0, 20.0]), H_batch)
    assert not np.isnan(sx[0])
    assert np.isnan(sx[1]) and np.isnan(sy[1])


def test_screen_to_canvas_roundtrip_with_behavior_to_screen() -> None:
    """canvas → screen (behavior_to_screen) → canvas should recover input."""
    URL = 272
    PAD = 233
    MAX_Y = 0.75
    grid_x = np.linspace(0.05, 0.95, 5)
    grid_y = np.linspace(0.05, 0.70, 5)
    for tx in grid_x:
        for ty in grid_y:
            sx, sy = behavior_to_screen(
                tx, ty,
                url_bar_h_px=URL, canvas_x_pad_px=PAD, max_y_coord=MAX_Y,
            )
            cx, cy = screen_to_canvas(np.array(sx), np.array(sy),
                                      url_bar_h_px=URL, canvas_x_pad_px=PAD)
            # canvas_w = 2388 - 2*233 = 1922; canvas_h = 1668 - 272 = 1396
            # behavior x normalised by canvas_w; behavior y by (canvas_h / MAX_Y)
            recovered_tx = float(cx) / 1922
            recovered_ty = float(cy) / (1396 / MAX_Y)
            np.testing.assert_allclose(recovered_tx, tx, atol=1e-9)
            np.testing.assert_allclose(recovered_ty, ty, atol=1e-9)


def test_is_on_screen_boundaries() -> None:
    sx = np.array([-1.0, 0.0, 1194.0, 2388.0, 2389.0, np.nan])
    sy = np.array([100.0, 100.0, 834.0, 1668.0, 100.0, 100.0])
    out = is_on_screen(sx, sy)
    np.testing.assert_array_equal(out, [False, True, True, True, False, False])


def test_is_on_screen_y_boundary() -> None:
    sx = np.array([100.0, 100.0, 100.0])
    sy = np.array([-1.0, 1668.0, 1669.0])
    out = is_on_screen(sx, sy)
    np.testing.assert_array_equal(out, [False, True, False])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
