"""Smooth per-frame iPad corner detections across time.

Two-step pipeline:
1. Linear interpolation across None (detection-failure) gaps, with edge-fill
   for any leading/trailing gaps.
2. Rolling median per coordinate to absorb outlier wrong-but-not-None
   detections (e.g. hand occlusion). Median is robust as long as <50% of
   frames in the window are bad — which holds for click events < 2 s.
"""

import numpy as np
import pandas as pd


def smooth_corners(
    raw: list,
    window: int = 51,
) -> np.ndarray:
    """Smooth a sequence of per-frame corner detections.

    Args:
        raw: List of length n_frames. Each element is a float32 array of
            shape (4, 2) from detect_corners(), or None.
        window: Rolling median window in frames. Default 51 ≈ 2 s at 25 fps.

    Returns:
        float32 array of shape (n_frames, 4, 2). No NaN. Corners are in the
        same [TL, TR, BR, BL] order as detect_corners() output.

    Raises:
        ValueError: If every frame is None (no valid detection anywhere).
    """
    n = len(raw)
    if n == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    # Build (n, 4, 2) with NaN where detection failed
    stack = np.full((n, 4, 2), np.nan, dtype=np.float32)
    for i, corners in enumerate(raw):
        if corners is not None:
            stack[i] = corners

    if np.all(np.isnan(stack)):
        raise ValueError("smooth_corners: no valid detections in input (all None)")

    # Reshape to (n, 8) — treat each of the 8 scalar series independently
    flat = stack.reshape(n, 8)

    # Step 1: interpolate NaN gaps, edge-fill leading/trailing NaN
    df = pd.DataFrame(flat)
    df = df.interpolate(method="linear", axis=0).ffill().bfill()
    interpolated = df.to_numpy(dtype=np.float32)

    # Step 2: rolling median to absorb outlier wrong detections
    # min_periods=1 avoids NaN at edges; center=True uses symmetric window
    df2 = pd.DataFrame(interpolated)
    smoothed = (
        df2.rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float32)
    )

    return smoothed.reshape(n, 4, 2)
