"""Smooth per-frame iPad corner detections across time.

Three-step pipeline:
1. Linear interpolation across None (detection-failure) gaps, with edge-fill
   for any leading/trailing gaps.
2. Rolling median per coordinate to absorb outlier wrong-but-not-None
   detections (e.g. hand occlusion). Median is robust as long as <50% of
   frames in the window are bad — which holds for click events < 2 s.
3. Velocity clamp: cap frame-to-frame corner movement to max_drift_px_per_s.
   The anchor resets at each post-gap boundary so that occlusion recovery
   (large but legitimate jump) is never penalised by the clamp.
"""

import numpy as np
import pandas as pd


def smooth_corners(
    raw: list,
    window: int = 75,
    max_drift_px_per_s: float | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Smooth a sequence of per-frame corner detections.

    Args:
        raw: List of length n_frames. Each element is a float32 array of
            shape (4, 2) from detect_corners(), or None.
        window: Rolling median window in frames. Default 75 ≈ 3 s at 25 fps.
        max_drift_px_per_s: If set (with fps), cap per-frame corner movement
            to this many pixels per second within each contiguous detection run.
            The clamp anchor resets after any None-gap.
        fps: Frames per second of the source video. Required when
            max_drift_px_per_s is set.

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

    # Step 3: velocity clamp — cap corner movement within contiguous runs.
    # The anchor resets at each post-gap frame so a legitimate position jump
    # after a long occlusion isn't mistaken for fast drift.
    if max_drift_px_per_s is not None and fps is not None:
        max_delta = max_drift_px_per_s / fps  # pixels per frame, per axis
        none_flags = [r is None for r in raw]
        # Frames that start a new contiguous detection run (first valid after gap)
        post_gap = {
            i for i in range(n)
            if not none_flags[i] and (i == 0 or none_flags[i - 1])
        }
        clamped = smoothed.copy()
        for i in range(1, n):
            if i in post_gap:
                continue  # reset anchor; don't penalise post-occlusion jump
            delta = clamped[i] - clamped[i - 1]
            clamped[i] = clamped[i - 1] + np.clip(delta, -max_delta, max_delta)
        smoothed = clamped

    return smoothed.reshape(n, 4, 2)
