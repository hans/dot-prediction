import numpy as np
import pytest
from src.corner_smoother import smooth_corners


def _c(val):
    """All 4 corners at (val, val), float32 (4,2)."""
    return np.full((4, 2), val, dtype=np.float32)


GOOD  = _c(500.0)
GOOD2 = _c(505.0)   # slight camera drift
BAD   = _c(0.0)     # totally wrong (hand occlusion failure)


# --- output contract ---

def test_output_shape():
    out = smooth_corners([GOOD] * 20, window=5)
    assert out.shape == (20, 4, 2)
    assert out.dtype == np.float32


def test_no_nans_in_output():
    raw = [GOOD, None, GOOD2, None, GOOD]
    out = smooth_corners(raw, window=3)
    assert not np.any(np.isnan(out))


def test_all_none_raises():
    with pytest.raises(ValueError, match="no valid"):
        smooth_corners([None, None, None], window=3)


# --- interpolation (None gaps) ---

def test_interior_none_interpolated():
    """A single None in the middle is filled by linear interpolation."""
    raw = [_c(100.0)] * 5 + [None] + [_c(100.0)] * 5
    out = smooth_corners(raw, window=1)   # window=1 disables smoothing
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[5], 100.0, atol=1.0)


def test_leading_none_edge_filled():
    """NaN at the start is forward-filled from the first valid value."""
    raw = [None, None] + [_c(200.0)] * 8
    out = smooth_corners(raw, window=1)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[0], 200.0, atol=1.0)
    np.testing.assert_allclose(out[1], 200.0, atol=1.0)


def test_trailing_none_edge_filled():
    """NaN at the end is back-filled from the last valid value."""
    raw = [_c(150.0)] * 8 + [None, None]
    out = smooth_corners(raw, window=1)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[-1], 150.0, atol=1.0)
    np.testing.assert_allclose(out[-2], 150.0, atol=1.0)


# --- rolling median (outlier rejection) ---

def test_outlier_absorbed_by_median():
    """A single wrong detection in a window of good frames is ignored."""
    # 25 good frames with 1 bad frame near the centre
    raw = [GOOD] * 12 + [BAD] + [GOOD] * 12
    out = smooth_corners(raw, window=25)
    # The frame at the bad position should be close to GOOD, not BAD
    np.testing.assert_allclose(out[12], 500.0, atol=10.0)


def test_constant_input_unchanged():
    """Constant signal passes through interpolation + median unchanged."""
    raw = [_c(300.0)] * 50
    out = smooth_corners(raw, window=5)
    np.testing.assert_allclose(out, 300.0, atol=1e-3)


def test_smoothing_reduces_noise():
    """Rolling median reduces per-frame jitter."""
    rng = np.random.default_rng(42)
    base = np.full((4, 2), 500.0, dtype=np.float32)
    raw = [base + rng.normal(0, 20, (4, 2)).astype(np.float32) for _ in range(200)]
    out = smooth_corners(raw, window=25)
    assert out.std() < 10.0, f"Smoothed std {out.std():.1f} too high"


def test_window_1_returns_interpolated_only():
    """Window=1 skips smoothing; Nones are filled but jitter is kept."""
    raw = [GOOD, None, GOOD2]
    out = smooth_corners(raw, window=1)
    assert out.shape == (3, 4, 2)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[0], 500.0, atol=1e-3)
    np.testing.assert_allclose(out[2], 505.0, atol=1e-3)
