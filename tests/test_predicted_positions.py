"""Unit tests for predicted_positions.predicted_positions().

All tests use synthetic trial DataFrames and homographies — no real video.
"""

import numpy as np
import pandas as pd
import pytest

from src.predicted_positions import (
    SCREEN_H,
    SCREEN_W,
    expected_screen_radius_px,
    predicted_positions,
)


# Identity homography: screen-px == frame-px.
H_IDENT = np.eye(3, dtype=np.float64)
# Pure 2-px-per-screen-px scale homography.
H_SCALE_2X = np.diag([2.0, 2.0, 1.0])
# Translation: project to (+100, +50).
H_SHIFT = np.array([[1, 0, 100], [0, 1, 50], [0, 0, 1]], dtype=np.float64)


_COLS = ["trial_idx", "tpt", "trial_onset", "trial_offset",
         "reveal_time", "true_x", "true_y"]


def _trial(trial_idx: int, onset: float, offset: float,
           tpts: list[tuple[int, float, float, float]]) -> pd.DataFrame:
    """Build a small trials_df for one trial.

    Each tpt is (tpt, reveal_time_ms, true_x_norm, true_y_norm).
    """
    rows = []
    for tpt, reveal_t, tx, ty in tpts:
        rows.append(dict(
            trial_idx=trial_idx, tpt=tpt, trial_onset=onset, trial_offset=offset,
            reveal_time=reveal_t, true_x=tx, true_y=ty,
        ))
    return pd.DataFrame(rows, columns=_COLS)


# ── return-type contract ─────────────────────────────────────────────────────

def test_returns_list():
    df = _trial(1, 0, 1000, [(0, 100, 0.5, 0.5)])
    out = predicted_positions(500, df, H_IDENT)
    assert isinstance(out, list)


def test_empty_trial_df():
    df = _trial(1, 0, 0, [])  # no rows
    assert predicted_positions(500, df, H_IDENT) == []


def test_no_active_trial_returns_empty():
    """Timestamp before any trial onset → empty result."""
    df = _trial(1, 1000, 2000, [(0, 1100, 0.5, 0.5)])
    assert predicted_positions(500, df, H_IDENT) == []


def test_no_active_trial_after_offset_returns_empty():
    """Timestamp past the trial offset → empty result (iPad cleared)."""
    df = _trial(1, 1000, 2000, [(0, 1100, 0.5, 0.5)])
    assert predicted_positions(3000, df, H_IDENT) == []


def test_no_revealed_before_first_reveal():
    """Inside the trial but before the first reveal → empty."""
    df = _trial(1, 1000, 5000, [(0, 1500, 0.5, 0.5)])
    assert predicted_positions(1200, df, H_IDENT) == []


# ── Rule A (all-revealed-so-far) ─────────────────────────────────────────────

def test_only_revealed_stars_returned():
    """At t=1600, tpts 0 and 1 are revealed; tpt 2 is not."""
    df = _trial(1, 1000, 5000, [
        (0, 1100, 0.2, 0.3),
        (1, 1500, 0.4, 0.5),
        (2, 1700, 0.6, 0.7),
    ])
    out = predicted_positions(1600, df, H_IDENT)
    assert {p.tpt for p in out} == {0, 1}


def test_order_is_reveal_time_ascending():
    """Returned list is oldest-first."""
    df = _trial(1, 1000, 5000, [
        (2, 1400, 0.6, 0.7),
        (0, 1100, 0.2, 0.3),
        (1, 1300, 0.4, 0.5),
    ])
    out = predicted_positions(1500, df, H_IDENT)
    assert [p.tpt for p in out] == [0, 1, 2]


def test_revealed_at_exact_t_included():
    """Star revealed at exactly t is visible (inclusive)."""
    df = _trial(1, 1000, 5000, [(0, 1500, 0.5, 0.5)])
    out = predicted_positions(1500, df, H_IDENT)
    assert len(out) == 1 and out[0].tpt == 0


# ── projection / age ─────────────────────────────────────────────────────────

def test_screen_xy_scaled_to_device_pixels():
    df = _trial(1, 0, 10000, [(0, 0, 0.5, 0.5)])
    out = predicted_positions(100, df, H_IDENT)
    sx, sy = out[0].screen_xy
    assert sx == pytest.approx(SCREEN_W * 0.5)
    assert sy == pytest.approx(SCREEN_H * 0.5)


def test_frame_xy_identity_homography():
    """Identity homography: frame_xy == screen_xy."""
    df = _trial(1, 0, 10000, [(0, 0, 0.25, 0.75)])
    out = predicted_positions(100, df, H_IDENT)
    sx, sy = out[0].screen_xy
    fx, fy = out[0].frame_xy
    assert (fx, fy) == pytest.approx((sx, sy))


def test_frame_xy_with_translation():
    df = _trial(1, 0, 10000, [(0, 0, 0.0, 0.0)])  # corner at origin
    out = predicted_positions(100, df, H_SHIFT)
    assert out[0].frame_xy == pytest.approx((100.0, 50.0))


def test_age_in_seconds():
    df = _trial(1, 0, 10000, [(0, 1000, 0.5, 0.5)])
    out = predicted_positions(3500, df, H_IDENT)
    assert out[0].age_s == pytest.approx(2.5)


# ── size model ───────────────────────────────────────────────────────────────

def test_expected_screen_radius_decreases_with_age():
    """The size model is monotonically non-increasing."""
    ages = [0, 5, 15, 30, 45, 60, 90]
    sizes = [expected_screen_radius_px(a) for a in ages]
    for a, b in zip(sizes, sizes[1:]):
        assert a >= b, f"Non-monotonic at {a}→{b}"


def test_expected_screen_radius_endpoints():
    assert expected_screen_radius_px(0) == pytest.approx(25.0)
    assert expected_screen_radius_px(30) == pytest.approx(15.0)
    assert expected_screen_radius_px(60) == pytest.approx(10.0)


def test_expected_radius_px_scales_with_homography():
    """A 2× scale homography → expected_radius_px = 2× screen-px radius at age 0."""
    df = _trial(1, 0, 10000, [(0, 0, 0.5, 0.5)])
    out_ident = predicted_positions(0, df, H_IDENT)
    out_scale = predicted_positions(0, df, H_SCALE_2X)
    # Local scale (frame-px per screen-px) is 1 for identity, 2 for scale.
    assert out_scale[0].expected_radius_px == pytest.approx(
        2.0 * out_ident[0].expected_radius_px,
        rel=1e-4,
    )


# ── multi-trial behaviour ────────────────────────────────────────────────────

def test_only_active_trial_stars_returned():
    """A second trial's reveals are ignored even if reveal_time <= t."""
    df = pd.concat([
        _trial(1, 0, 1000, [(0, 100, 0.2, 0.3)]),
        _trial(2, 2000, 3000, [(0, 2100, 0.6, 0.7)]),
    ], ignore_index=True)
    out_t500 = predicted_positions(500, df, H_IDENT)
    assert {p.trial_idx for p in out_t500} == {1}

    # Inter-trial gap → nothing visible
    assert predicted_positions(1500, df, H_IDENT) == []

    # Second trial active
    out_t2500 = predicted_positions(2500, df, H_IDENT)
    assert {p.trial_idx for p in out_t2500} == {2}
