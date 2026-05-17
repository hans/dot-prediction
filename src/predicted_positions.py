"""Position prior for the local star detector.

For a given video-frame timestamp, returns the list of stars that should be
visible on the iPad (from the behavioral log) along with their predicted
locations in frame coordinates (via the supplied rough homography).

Visibility rule
---------------
**Rule A** ("all-revealed-so-far"): inside an active trial
(``trial_onset <= t <= trial_offset``), every dot whose ``reveal_time <= t``
is visible. Confirmed empirically on EC347 trial 1 — the oldest small star
remains detectable >30 s after reveal, well past any click event.

Between trials (no active trial), nothing is visible — the iPad is cleared.

Size model
----------
Newest star at ~25 screen-px radius, decaying to ~10 screen-px by 60 s.
Translated to frame-px via the local scale of ``H_rough`` at the predicted
location (linearised Jacobian determinant ⁻⁰·⁵). The model is intentionally
coarse — the local detector uses the size only for sanity checks
(reject blobs much larger than expected) and confidence weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCREEN_W: int = 2388
SCREEN_H: int = 1668


@dataclass(frozen=True)
class PredictedStar:
    """Single predicted star.

    Attributes:
        trial_idx: Trial id from the behavioral log.
        tpt: Time-point index within the trial.
        screen_xy: (x, y) in iPad device pixels, [0, 2388] × [0, 1668].
        frame_xy: (x, y) in scene-video frame pixels (projected via H_rough).
        age_s: Seconds elapsed since this dot was revealed.
        expected_radius_px: Expected blob radius in frame pixels.
    """

    trial_idx: int
    tpt: int
    screen_xy: tuple[float, float]
    frame_xy: tuple[float, float]
    age_s: float
    expected_radius_px: float


def _project(H: np.ndarray, xy: np.ndarray) -> tuple[float, float]:
    h = H @ np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    return float(h[0] / h[2]), float(h[1] / h[2])


def _local_frame_per_screen(H: np.ndarray, screen_xy: tuple[float, float]) -> float:
    """Average linear scale (frame-px per screen-px) at ``screen_xy``.

    Uses a 1-px finite-difference Jacobian at the projection point. Returns the
    geometric mean of the two principal singular values, i.e. ``sqrt(|det J|)``.
    """
    sx, sy = screen_xy
    p0x, p0y = _project(H, np.array([sx, sy]))
    dxx, dxy = _project(H, np.array([sx + 1.0, sy]))
    dyx, dyy = _project(H, np.array([sx, sy + 1.0]))
    j11, j21 = dxx - p0x, dxy - p0y
    j12, j22 = dyx - p0x, dyy - p0y
    det = abs(j11 * j22 - j12 * j21)
    return float(np.sqrt(det))


def expected_screen_radius_px(age_s: float) -> float:
    """Expected star radius in **iPad screen pixels** as a function of age.

    Piecewise-linear: 25 px at age 0, 15 px at age 30 s, 10 px at age 60 s,
    floor at 8 px beyond. The exact values are uncertain — the local detector
    must tolerate ±50% error in this estimate.
    """
    if age_s <= 0:
        return 25.0
    if age_s <= 30.0:
        return 25.0 - (25.0 - 15.0) * (age_s / 30.0)
    if age_s <= 60.0:
        return 15.0 - (15.0 - 10.0) * ((age_s - 30.0) / 30.0)
    return 8.0


def predicted_positions(
    frame_t_ms: float,
    trials_df: pd.DataFrame,
    H_rough: np.ndarray,
) -> list[PredictedStar]:
    """Predict which stars are visible at ``frame_t_ms`` and where in frame.

    Args:
        frame_t_ms: Frame timestamp in the experiment clock (ms). Convert from
            video frame index with the alignment slope/intercept upstream.
        trials_df: Behavior log with columns ``trial_idx``, ``tpt``,
            ``trial_onset``, ``trial_offset``, ``reveal_time``, ``true_x``,
            ``true_y``. Coordinates are normalised [0,1]; this function scales
            by (SCREEN_W, SCREEN_H) internally.
        H_rough: 3×3 homography mapping iPad device-pixel coords → frame
            pixel coords (e.g. from ``cv2.findHomography(SCREEN_CORNERS,
            detected_frame_corners)``).

    Returns:
        List of PredictedStar, in reveal-time order (oldest first, newest
        last). Empty list if no trial is active at ``frame_t_ms``.
    """
    # Find the trial active at this timestamp. Rows are duplicated per (trial,
    # tpt), so trial_onset/trial_offset are constant within a trial — picking
    # any row is sufficient.
    active = trials_df[
        (trials_df["trial_onset"] <= frame_t_ms)
        & (trials_df["trial_offset"] >= frame_t_ms)
    ]
    if active.empty:
        return []
    trial_idx = int(active["trial_idx"].iloc[0])

    visible = active[active["reveal_time"] <= frame_t_ms].sort_values("reveal_time")
    if visible.empty:
        return []

    H = np.asarray(H_rough, dtype=np.float64)
    out: list[PredictedStar] = []
    for _, row in visible.iterrows():
        screen_xy = (float(row["true_x"]) * SCREEN_W, float(row["true_y"]) * SCREEN_H)
        frame_xy = _project(H, np.array(screen_xy))
        age_s = (frame_t_ms - float(row["reveal_time"])) / 1000.0
        scale = _local_frame_per_screen(H, screen_xy)
        expected_radius_px = expected_screen_radius_px(age_s) * scale
        out.append(
            PredictedStar(
                trial_idx=trial_idx,
                tpt=int(row["tpt"]),
                screen_xy=screen_xy,
                frame_xy=frame_xy,
                age_s=age_s,
                expected_radius_px=expected_radius_px,
            )
        )
    return out
