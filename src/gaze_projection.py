"""Tobii gaze projection to screen and canvas coordinates.

Pure functions used by the `extract_gaze_fixations` Snakemake rule. Reads
nothing, writes nothing — the notebook composes them with I/O.

The screen-coord system is the iPad's native 2388×1668 pixel rectangle (origin
at the screen top-left). The canvas-coord system is the inner Safari canvas
(1922×1396 px, origin at canvas top-left), inset by the URL bar at the top and
symmetric padding on the sides. See `behavior_to_screen` in
`homography_solver.py` for the screen↔canvas transform.

TODO: revisit fixation classification with I-DT/I-VT — currently we rely on
Tobii's built-in `Eye movement type`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_H_COLS_NON_NORMALIZED = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]


def smooth_homography_elements(
    per_frame_h: pd.DataFrame,
    window: int = 5,
    min_valid: int = 3,
) -> pd.DataFrame:
    """Centered rolling-median smoothing of H elements across frames.

    Phase 1c's per-frame H is fit independently from 4 noisy correspondences
    that are clustered near the screen bottom and the photodiode device. The
    screen TR/TL corners are far from any anchor, so sub-pixel anchor jitter
    is amplified into huge perspective-element swings (TR_x std measured in
    the thousands of px). Smoothing h00..h21 with a centered rolling median
    over `window` consecutive frames knocks the high-frequency jitter down
    without lagging the polygon.

    Args:
        per_frame_h: DataFrame with `frame_idx` + h00..h22 columns. NaN-filled
            h-rows mark frames where Phase 1c could not produce an H
            (no_screen, geometric_invalid, …).
        window: Centered rolling window size; default 5 frames.
        min_valid: Minimum non-NaN values in window required to emit a
            smoothed value (otherwise NaN).

    Returns:
        Copy of `per_frame_h` with h00..h21 replaced by smoothed values.
        `h22` is set to 1.0 (perspective normalization preserved). All other
        columns are passed through unchanged.
    """
    out = per_frame_h.sort_values("frame_idx").reset_index(drop=True).copy()
    for col in _H_COLS_NON_NORMALIZED:
        out[col] = (
            out[col]
            .rolling(window=window, min_periods=min_valid, center=True)
            .median()
        )
    # h22 is the perspective-normalization element; Phase 1c always emits 1.0.
    # Smoothing it would propagate NaN into otherwise-valid frames at boundaries.
    out["h22"] = np.where(out[_H_COLS_NON_NORMALIZED].isna().any(axis=1), np.nan, 1.0)
    return out


def tobii_ts_to_behavior_ms(
    ts_us: np.ndarray, slope_ms_per_s: float, intercept_ms: float
) -> np.ndarray:
    """Map Tobii recording timestamps (µs) → behavior time (ms).

    `video_t_s = ts_us / 1e6`, then `behavior_t_ms = video_t_s * slope + intercept`.
    """
    return (np.asarray(ts_us, dtype=np.float64) / 1e6) * slope_ms_per_s + intercept_ms


def tobii_ts_to_video_frame_frac(ts_us: np.ndarray, fps: float) -> np.ndarray:
    """Map Tobii recording timestamps (µs) → fractional video frame index."""
    return (np.asarray(ts_us, dtype=np.float64) / 1e6) * fps


def lerp_homography_at_frac(
    per_frame_h: np.ndarray, frame_frac: float
) -> np.ndarray | None:
    """Element-wise linear interpolation between flanking integer-frame Hs.

    Args:
        per_frame_h: shape (N_frames, 3, 3); NaN-filled rows mark frames where
            Phase 1c could not produce an H (no_screen).
        frame_frac: fractional frame index. Clipped to [0, N-1].

    Returns:
        Interpolated (3, 3) H, or None if either flanking integer-frame H
        contains a NaN.
    """
    n_frames = per_frame_h.shape[0]
    if n_frames == 0:
        return None
    f = float(np.clip(frame_frac, 0.0, n_frames - 1))
    floor_idx = int(np.floor(f))
    ceil_idx = min(floor_idx + 1, n_frames - 1)
    alpha = f - floor_idx
    h_lo = per_frame_h[floor_idx]
    h_hi = per_frame_h[ceil_idx]
    if np.isnan(h_lo).any() or np.isnan(h_hi).any():
        return None
    if floor_idx == ceil_idx:
        return h_lo.copy()
    return (1.0 - alpha) * h_lo + alpha * h_hi


def project_video_to_screen(
    gx: np.ndarray, gy: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project video-px gaze to screen-px via inverse H (screen→frame).

    `H` may be a single (3, 3) or a batch of shape (N, 3, 3). The latter
    handles per-sample homographies in a single vectorised pass.

    Returns (sx, sy) with NaN propagation: any input NaN → NaN output.
    """
    gx_arr = np.asarray(gx, dtype=np.float64)
    gy_arr = np.asarray(gy, dtype=np.float64)
    H_arr = np.asarray(H, dtype=np.float64)

    if H_arr.ndim == 2:
        H_inv = np.linalg.inv(H_arr)
        ones = np.ones_like(gx_arr)
        v = np.stack([gx_arr, gy_arr, ones], axis=-1)
        out = v @ H_inv.T
        sx = out[..., 0] / out[..., 2]
        sy = out[..., 1] / out[..., 2]
    else:
        # Per-sample batched inversion. Where H contains NaN, inv → NaN,
        # which propagates to the output. Mask out those rows first to
        # avoid LinAlgError on singular slices.
        nan_mask = np.isnan(H_arr).any(axis=(1, 2))
        H_safe = H_arr.copy()
        H_safe[nan_mask] = np.eye(3)
        H_inv = np.linalg.inv(H_safe)
        H_inv[nan_mask] = np.nan

        ones = np.ones_like(gx_arr)
        v = np.stack([gx_arr, gy_arr, ones], axis=-1)  # (N, 3)
        # (N, 3, 3) @ (N, 3, 1) → (N, 3, 1)
        out = np.einsum("nij,nj->ni", H_inv, v)
        sx = out[..., 0] / out[..., 2]
        sy = out[..., 1] / out[..., 2]

    bad = np.isnan(gx_arr) | np.isnan(gy_arr)
    sx = np.where(bad, np.nan, sx)
    sy = np.where(bad, np.nan, sy)
    return sx, sy


def screen_to_canvas(
    sx: np.ndarray | float,
    sy: np.ndarray | float,
    url_bar_h_px: int = 272,
    canvas_x_pad_px: int = 233,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `homography_solver.behavior_to_screen`'s canvas-pixel step.

    Subtract the canvas inset to convert iPad-screen px to canvas-tl-origin px.
    The output canvas rectangle is 1922×1396 px (screen 2388×1668 minus the
    URL bar and symmetric x-padding).
    """
    sx_arr = np.asarray(sx, dtype=np.float64)
    sy_arr = np.asarray(sy, dtype=np.float64)
    return sx_arr - canvas_x_pad_px, sy_arr - url_bar_h_px


def is_on_screen(
    sx: np.ndarray,
    sy: np.ndarray,
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
) -> np.ndarray:
    """Strict iPad-screen rectangle test. NaN → False."""
    sx_arr = np.asarray(sx, dtype=np.float64)
    sy_arr = np.asarray(sy, dtype=np.float64)
    on = (
        (sx_arr >= 0.0)
        & (sx_arr <= screen_w_px)
        & (sy_arr >= 0.0)
        & (sy_arr <= screen_h_px)
    )
    # NaN comparisons return False already, so on stays False where gaze invalid.
    return on
