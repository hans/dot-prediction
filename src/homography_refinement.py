"""Light-weight refinement of a rough screen→frame homography.

Phase 1b discovered that the screen-corner detector consistently overshoots
the iPad's right edge on EC347 (likely a glare patch beyond the bezel),
producing a systematic ~140 px x-offset in the predicted star positions —
both for raw and rolling-median-smoothed corners. A single known
correspondence (typically the freshly-revealed big star, detected by the
Phase 1a global detector) can collapse the bulk of that error via a pure
translation.

This module provides one helper: ``anchor_translate`` adjusts the homography
so that one screen-coord anchor projects exactly to its known frame-coord
location. The perspective component is unchanged, so residual error grows
with distance from the anchor — the eval framework should quantify this.
"""

from __future__ import annotations

import numpy as np


def anchor_translate(
    H: np.ndarray,
    anchor_screen_xy: tuple[float, float],
    anchor_frame_xy: tuple[float, float],
) -> np.ndarray:
    """Translate ``H`` so the anchor projects exactly to its known frame xy.

    Args:
        H: 3×3 homography mapping iPad screen-px → frame-px.
        anchor_screen_xy: Known screen-pixel position of a star.
        anchor_frame_xy: Known frame-pixel detection of the same star.

    Returns:
        New 3×3 homography ``H'`` such that
        ``H' @ [sx, sy, 1]ᵀ`` projects to ``anchor_frame_xy`` (up to numerical
        precision). All other points are shifted by the same translation
        ``(detected − predicted)``.
    """
    H = np.asarray(H, dtype=np.float64)
    sx, sy = anchor_screen_xy
    fx_target, fy_target = anchor_frame_xy
    h = H @ np.array([sx, sy, 1.0])
    fx_pred, fy_pred = h[0] / h[2], h[1] / h[2]
    dx, dy = fx_target - fx_pred, fy_target - fy_pred

    # Post-multiply translation in frame space: T @ H, so that for any input
    # (sx, sy), the projection becomes (H @ pt) + (dx, dy). A simple add to
    # H[:2, 2] is only correct for affine H (h31 = h32 = 0, h33 = 1).
    T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return T @ H
