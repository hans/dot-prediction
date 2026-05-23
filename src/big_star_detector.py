"""Per-frame big_star detection for use as a 5th homography anchor.

The "big_star" is the most-recently-revealed dot in the experiment — bright,
warm-coloured, near the centre of the iPad screen. Promoting it from a held-out
validation signal (Phase 1c v1) to a fit anchor dramatically improves the
conditioning of the per-frame H fit: with only 4 anchors clustered near the
screen bottom + photodiode device, screen TR/TL are high-leverage extrapolations.
Adding a 5th anchor in the screen interior bounds the TR/TL projection error.

The detector wraps :func:`local_star_detector.detect_in_windows` with a single
``PredictedStar`` constructed from the prior frame's H and the active dot's
screen-coordinate position (computed via ``trials_df`` + ``behavior_to_screen``).
When the local search fails (no warm blob in window, blob too large, prediction
off-frame), the caller falls back to a 4-anchor fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    # Notebook / script usage with `sys.path.insert(0, 'src')`.
    from local_star_detector import detect_in_windows
    from predicted_positions import PredictedStar, expected_screen_radius_px
except ImportError:
    # pytest / external callers: ``src`` is a namespace package.
    from src.local_star_detector import detect_in_windows
    from src.predicted_positions import PredictedStar, expected_screen_radius_px


def active_dot_screen_xy(
    trials_df: pd.DataFrame,
    frame_idx: int,
    behavior_to_screen,
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
    url_bar_h_px: int = 272,
    canvas_x_pad_px: int = 233,
    max_y_coord: float = 0.75,
) -> tuple[tuple[float, float], int, int, int] | None:
    """Return the active dot's iPad screen-pixel xy at ``frame_idx``.

    The active dot is the most-recently-revealed trial point with
    ``video_frame_reveal <= frame_idx``. Returns None when no dot has been
    revealed yet (pre-experiment frames).

    Args:
        trials_df: trials_with_video.parquet contents — must contain
            ``video_frame_reveal``, ``true_x``, ``true_y``, ``trial_idx``,
            ``tpt``.
        frame_idx: Current video frame index.
        behavior_to_screen: The ``homography_solver.behavior_to_screen``
            function, injected for parameter consistency with the caller.

    Returns:
        ``((sx, sy), reveal_frame, trial_idx, tpt)`` or ``None`` if no
        active dot exists at this frame.
    """
    prior = trials_df[
        trials_df.video_frame_reveal.notna()
        & (trials_df.video_frame_reveal <= frame_idx)
    ]
    if prior.empty:
        return None
    active = prior.loc[prior.video_frame_reveal.idxmax()]
    sx, sy = behavior_to_screen(
        float(active.true_x),
        float(active.true_y),
        screen_w_px=screen_w_px,
        screen_h_px=screen_h_px,
        url_bar_h_px=url_bar_h_px,
        canvas_x_pad_px=canvas_x_pad_px,
        max_y_coord=max_y_coord,
    )
    return (
        (sx, sy),
        int(active.video_frame_reveal),
        int(active.trial_idx),
        int(active.tpt),
    )


def detect_big_star(
    frame: np.ndarray,
    H_prior: np.ndarray,
    screen_xy: tuple[float, float],
    reveal_frame: int | None,
    current_frame: int,
    fps: float = 30.0,
    window_size_px: int = 60,
    floor: float = 20.0,
    max_radius_factor: float = 4.0,
) -> tuple[float, float] | None:
    """Detect the big_star at ``current_frame`` via local search around H_prior projection.

    Args:
        frame: BGR scene-video frame.
        H_prior: 3×3 homography from a recent good fit (typically the previous
            frame's H). Used only to predict the search-window centre — small
            translation/rotation errors in H_prior are tolerated by the local
            search window.
        screen_xy: Active dot's true position in iPad screen-px.
        reveal_frame: Video frame where the dot was revealed (for age → size).
            ``None`` skips the age-based size model and uses a generous default.
        current_frame: Current video frame index.
        fps: Video frame rate, used to convert frame offset to age in seconds.
        window_size_px: Local-search window side length in frame px. Wider than
            the local_star_detector default (40) because H_prior may be off by
            10-20 px at the screen interior.
        floor: R-B opponent-channel floor for the centroid; same default as
            local_star_detector.
        max_radius_factor: Reject blobs more than this × ``expected_radius_px``.

    Returns:
        Sub-pixel (x, y) in frame px, or None on no detection / rejection.
    """
    # Predict frame-px location via the prior H.
    p = H_prior @ np.array([screen_xy[0], screen_xy[1], 1.0])
    if abs(p[2]) < 1e-12:
        return None
    pred_xy = (float(p[0] / p[2]), float(p[1] / p[2]))

    # Expected blob size in screen-px from the age model; scaled to frame-px
    # via the local Jacobian determinant of H_prior at the predicted point.
    if reveal_frame is not None:
        age_s = max(0.0, (current_frame - reveal_frame) / float(fps))
        screen_r = expected_screen_radius_px(age_s)
    else:
        screen_r = 25.0  # generous default

    # 1-px finite-difference Jacobian to convert screen-px → frame-px scale.
    sx, sy = screen_xy
    dx = H_prior @ np.array([sx + 1.0, sy, 1.0])
    dy = H_prior @ np.array([sx, sy + 1.0, 1.0])
    if abs(dx[2]) < 1e-12 or abs(dy[2]) < 1e-12:
        return None
    dx_xy = (dx[0] / dx[2] - pred_xy[0], dx[1] / dx[2] - pred_xy[1])
    dy_xy = (dy[0] / dy[2] - pred_xy[0], dy[1] / dy[2] - pred_xy[1])
    det = abs(dx_xy[0] * dy_xy[1] - dx_xy[1] * dy_xy[0])
    scale = float(np.sqrt(det)) if det > 0 else 1.0
    expected_radius_px = screen_r * scale

    pred = PredictedStar(
        trial_idx=0,
        tpt=0,
        screen_xy=screen_xy,
        frame_xy=pred_xy,
        age_s=0.0,
        expected_radius_px=expected_radius_px,
    )

    detections, _ = detect_in_windows(
        frame,
        [pred],
        window_size_px=window_size_px,
        floor=floor,
        max_radius_factor=max_radius_factor,
    )
    if not detections:
        return None
    return detections[0].frame_xy_subpix
