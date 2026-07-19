"""Unit tests for big_star_detector — synthetic frames + synthetic trials."""

import cv2
import numpy as np
import pandas as pd

from src.big_star_detector import active_dot_screen_xy, detect_big_star
from src.homography_solver import behavior_to_screen

W, H = 1920, 1080


def _blue_frame() -> np.ndarray:
    return np.full((H, W, 3), (140, 50, 50), dtype=np.uint8)


def _paint_blob(
    frame: np.ndarray, cx: int, cy: int, radius: int = 8, color=(120, 175, 180)
) -> None:
    cv2.circle(frame, (cx, cy), radius, color, -1)


# ── active_dot_screen_xy ────────────────────────────────────────────────────


def _trials_row(
    trial_idx: int, tpt: int, true_x: float, true_y: float, video_frame_reveal
) -> dict:
    return {
        "trial_idx": trial_idx,
        "tpt": tpt,
        "true_x": true_x,
        "true_y": true_y,
        "video_frame_reveal": video_frame_reveal,
    }


def test_active_dot_screen_xy_picks_max_reveal():
    """Most-recently-revealed dot (max video_frame_reveal ≤ frame) wins."""
    trials_df = pd.DataFrame(
        [
            _trials_row(1, 0, 0.2, 0.2, 50),
            _trials_row(1, 1, 0.5, 0.4, 100),
            _trials_row(1, 2, 0.8, 0.6, 200),
        ]
    )
    result = active_dot_screen_xy(trials_df, 150, behavior_to_screen)
    assert result is not None
    (sx, sy), reveal_frame, trial_idx, tpt = result
    assert reveal_frame == 100
    assert tpt == 1
    expected_sx, expected_sy = behavior_to_screen(0.5, 0.4)
    assert sx == expected_sx
    assert sy == expected_sy


def test_active_dot_screen_xy_none_before_first_reveal():
    """Frames before any reveal return None."""
    trials_df = pd.DataFrame([_trials_row(1, 0, 0.5, 0.4, 100)])
    assert active_dot_screen_xy(trials_df, 50, behavior_to_screen) is None


def test_active_dot_screen_xy_handles_nan_reveal():
    """Rows with NaN video_frame_reveal are skipped."""
    trials_df = pd.DataFrame(
        [
            _trials_row(1, 0, 0.5, 0.4, pd.NA),
            _trials_row(1, 1, 0.6, 0.5, 80),
        ]
    )
    trials_df["video_frame_reveal"] = trials_df["video_frame_reveal"].astype("Int64")
    result = active_dot_screen_xy(trials_df, 150, behavior_to_screen)
    assert result is not None
    _, reveal_frame, _, tpt = result
    assert reveal_frame == 80
    assert tpt == 1


# ── detect_big_star ─────────────────────────────────────────────────────────


def test_detect_big_star_finds_warm_blob():
    """Identity H, blob painted at predicted location → sub-pixel detection."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 500, radius=8)
    H_identity = np.eye(3, dtype=np.float64)
    xy = detect_big_star(
        frame,
        H_identity,
        screen_xy=(800.0, 500.0),
        reveal_frame=10,
        current_frame=20,
        fps=30.0,
    )
    assert xy is not None
    fx, fy = xy
    assert np.hypot(fx - 800, fy - 500) < 2.0


def test_detect_big_star_returns_none_without_blob():
    """No warm blob in window → None."""
    frame = _blue_frame()
    H_identity = np.eye(3, dtype=np.float64)
    xy = detect_big_star(
        frame,
        H_identity,
        screen_xy=(800.0, 500.0),
        reveal_frame=10,
        current_frame=20,
        fps=30.0,
    )
    assert xy is None


def test_detect_big_star_tolerates_small_prediction_offset():
    """Blob 10 px from predicted location is still inside the search window."""
    frame = _blue_frame()
    _paint_blob(frame, 810, 510, radius=8)
    H_identity = np.eye(3, dtype=np.float64)
    xy = detect_big_star(
        frame,
        H_identity,
        screen_xy=(800.0, 500.0),
        reveal_frame=10,
        current_frame=20,
        fps=30.0,
        window_size_px=60,
    )
    assert xy is not None
    fx, fy = xy
    assert np.hypot(fx - 810, fy - 510) < 3.0


def test_detect_big_star_handles_off_frame_prediction():
    """Predicted xy outside frame → None (window fully out of bounds)."""
    frame = _blue_frame()
    H_identity = np.eye(3, dtype=np.float64)
    xy = detect_big_star(
        frame,
        H_identity,
        screen_xy=(-500.0, -500.0),
        reveal_frame=10,
        current_frame=20,
    )
    assert xy is None


def test_detect_big_star_uses_default_radius_when_no_reveal_frame():
    """Without reveal_frame the default expected radius is used (no crash)."""
    frame = _blue_frame()
    _paint_blob(frame, 800, 500, radius=8)
    H_identity = np.eye(3, dtype=np.float64)
    xy = detect_big_star(
        frame,
        H_identity,
        screen_xy=(800.0, 500.0),
        reveal_frame=None,
        current_frame=20,
    )
    assert xy is not None
