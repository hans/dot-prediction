"""Per-frame screen→frame homography solver via hand-labeled correspondences.

Three-step pipeline:
  1. calibrate_box_position   — back-project photodiode-device corners to
                                screen-coords using frames with all 4 iPad
                                corners visible.
  2. fit_per_frame_homography — DLT on 4 points (BL/BR + calibrated box).
  3. big_star_residuals       — held-out validation via labeled big_star.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


# --- internal helper -------------------------------------------------------

def _pt(H: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    """Perspective-divide H @ [x, y, 1]ᵀ → (x', y')."""
    v = H @ np.array([xy[0], xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


# --- public API ------------------------------------------------------------

def calibrate_box_position(
    labels_df: pd.DataFrame,
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
) -> dict:
    """Calibrate the photodiode-device corners' screen-coord positions.

    Auto-selects every frame in labels_df with all 4 iPad corners + both
    box corners visible (any quality, including 'approximate' — TL/TR are
    intrinsically approximate due to photodiode occlusion).

    Returns:
        {
          'box_bl_screen': [median_x, median_y],
          'box_br_screen': [median_x, median_y],
          'calibration_frames': [frame_idx, ...],
          'per_frame_estimates': {frame_idx: {'box_bl_screen': [...], ...}},
          'spread_screen_px': {
            'box_bl': {'iqr_x': ..., 'max_minus_min_x': ..., 'iqr_y': ..., 'max_minus_min_y': ...},
            'box_br': {...},
          },
          'notes': str,
        }

    Raises:
        ValueError: if no frames qualify.
    """
    required = ["screen_tl", "screen_tr", "screen_br", "screen_bl", "box_bl", "box_br"]

    # Screen corners in TL/TR/BR/BL order — matches SCREEN_CORNERS in the
    # labeling notebook so the ordering convention is consistent.
    screen_corners = np.array(
        [[0, 0], [screen_w_px, 0], [screen_w_px, screen_h_px], [0, screen_h_px]],
        dtype=np.float32,
    )

    calib_frames: list[int] = []
    per_frame_label_map: dict[int, dict] = {}

    for frame_idx, group in labels_df.groupby("frame_idx"):
        by_type = {row.label_type: row for _, row in group.iterrows()}
        per_frame_label_map[int(frame_idx)] = by_type
        if all(
            lt in by_type and bool(by_type[lt].visible)
            for lt in required
        ):
            calib_frames.append(int(frame_idx))

    if not calib_frames:
        raise ValueError(
            "No frames qualify for box-position calibration. Need every frame "
            "to have all 4 iPad corners (screen_tl/tr/br/bl) and both box "
            "corners (box_bl/br) labeled visible."
        )

    per_frame_estimates: dict[int, dict] = {}
    for frame_idx in calib_frames:
        by_type = per_frame_label_map[frame_idx]
        frame_pts = np.array(
            [
                [by_type["screen_tl"].x_frame, by_type["screen_tl"].y_frame],
                [by_type["screen_tr"].x_frame, by_type["screen_tr"].y_frame],
                [by_type["screen_br"].x_frame, by_type["screen_br"].y_frame],
                [by_type["screen_bl"].x_frame, by_type["screen_bl"].y_frame],
            ],
            dtype=np.float32,
        )
        H_s2f, _ = cv2.findHomography(screen_corners, frame_pts)
        H_f2s = np.linalg.inv(H_s2f)

        bl_sx, bl_sy = _pt(H_f2s, (by_type["box_bl"].x_frame, by_type["box_bl"].y_frame))
        br_sx, br_sy = _pt(H_f2s, (by_type["box_br"].x_frame, by_type["box_br"].y_frame))
        per_frame_estimates[frame_idx] = {
            "box_bl_screen": [bl_sx, bl_sy],
            "box_br_screen": [br_sx, br_sy],
        }

    bl_xs = np.array([v["box_bl_screen"][0] for v in per_frame_estimates.values()])
    bl_ys = np.array([v["box_bl_screen"][1] for v in per_frame_estimates.values()])
    br_xs = np.array([v["box_br_screen"][0] for v in per_frame_estimates.values()])
    br_ys = np.array([v["box_br_screen"][1] for v in per_frame_estimates.values()])

    def _spread(arr: np.ndarray) -> dict:
        q1, q3 = np.percentile(arr, [25, 75])
        return {
            "iqr": float(q3 - q1),
            "max_minus_min": float(arr.max() - arr.min()),
        }

    return {
        "box_bl_screen": [float(np.median(bl_xs)), float(np.median(bl_ys))],
        "box_br_screen": [float(np.median(br_xs)), float(np.median(br_ys))],
        "calibration_frames": sorted(calib_frames),
        "per_frame_estimates": per_frame_estimates,
        "spread_screen_px": {
            "box_bl": {
                "iqr_x": _spread(bl_xs)["iqr"],
                "max_minus_min_x": _spread(bl_xs)["max_minus_min"],
                "iqr_y": _spread(bl_ys)["iqr"],
                "max_minus_min_y": _spread(bl_ys)["max_minus_min"],
            },
            "box_br": {
                "iqr_x": _spread(br_xs)["iqr"],
                "max_minus_min_x": _spread(br_xs)["max_minus_min"],
                "iqr_y": _spread(br_ys)["iqr"],
                "max_minus_min_y": _spread(br_ys)["max_minus_min"],
            },
        },
        "notes": (
            "TL labels are mostly approximate (photodiode occlusion); "
            "per-frame spread is the quality signal."
        ),
    }


def fit_per_frame_homography(
    labels_df: pd.DataFrame,
    box_bl_screen: tuple[float, float],
    box_br_screen: tuple[float, float],
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
    include_qualities: set[str] = {"confident"},
) -> pd.DataFrame:
    """Fit a 4-point DLT homography for every labeled frame.

    Uses screen_bl, screen_br, and the calibrated box corners (box_bl_screen,
    box_br_screen) as the 4 correspondences. big_star is intentionally held
    out for downstream validation.

    Returns one row per labeled frame with columns:
        frame_idx, h00–h22, n_correspondences, excluded_reason.
    H is row-major, normalised so h22 = 1. excluded_reason is the empty
    string when the frame is included; otherwise 'missing_<anchor>'.
    """
    required_anchors = ["screen_bl", "screen_br", "box_bl", "box_br"]

    screen_pts = np.array(
        [
            [0.0, screen_h_px],              # screen_bl
            [float(screen_w_px), screen_h_px],  # screen_br
            list(box_bl_screen),             # box_bl calibrated
            list(box_br_screen),             # box_br calibrated
        ],
        dtype=np.float32,
    )

    _nan = float("nan")
    _h_cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]

    rows: list[dict] = []
    for frame_idx, group in labels_df.groupby("frame_idx"):
        by_type = {r.label_type: r for _, r in group.iterrows()}

        excluded_reason = ""
        for anchor in required_anchors:
            r = by_type.get(anchor)
            if r is None or not bool(r.visible) or r.quality not in include_qualities:
                excluded_reason = f"missing_{anchor}"
                break

        if excluded_reason:
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    **{col: _nan for col in _h_cols},
                    "n_correspondences": 0,
                    "excluded_reason": excluded_reason,
                }
            )
            continue

        frame_pts = np.array(
            [
                [by_type["screen_bl"].x_frame, by_type["screen_bl"].y_frame],
                [by_type["screen_br"].x_frame, by_type["screen_br"].y_frame],
                [by_type["box_bl"].x_frame, by_type["box_bl"].y_frame],
                [by_type["box_br"].x_frame, by_type["box_br"].y_frame],
            ],
            dtype=np.float32,
        )
        H_mat, _ = cv2.findHomography(screen_pts, frame_pts)
        H_mat = H_mat / H_mat[2, 2]

        rows.append(
            {
                "frame_idx": int(frame_idx),
                "h00": H_mat[0, 0], "h01": H_mat[0, 1], "h02": H_mat[0, 2],
                "h10": H_mat[1, 0], "h11": H_mat[1, 1], "h12": H_mat[1, 2],
                "h20": H_mat[2, 0], "h21": H_mat[2, 1], "h22": H_mat[2, 2],
                "n_correspondences": 4,
                "excluded_reason": "",
            }
        )

    return pd.DataFrame(rows)


def behavior_to_screen(
    true_x: float,
    true_y: float,
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
    url_bar_h_px: int = 272,
    canvas_x_pad_px: int = 233,
    max_y_coord: float = 0.75,
) -> tuple[float, float]:
    """Convert normalised behavior coordinates to iPad screen-pixel coordinates.

    The canvas is inset from the screen on all sides:
      - Top: Safari URL bar (url_bar_h_px ≈ 272 physical px = 136 CSS px at 2×).
      - Left/right: symmetric white padding (canvas_x_pad_px ≈ 233 px each side).
      - Bottom: aligns with screen bottom.
    true_y is normalised to [0, max_y_coord] of the canvas height (dots are
    restricted to the top 75% of the canvas).

    Both offsets empirically derived by back-projecting labeled big_star positions
    through the 4-corner H. Sanity checks:
      behavior_to_screen(0, 0.75) → (233, 1668)   [canvas bottom-left]
      behavior_to_screen(1, 0.75) → (2155, 1668)  [canvas bottom-right]
    """
    canvas_h = screen_h_px - url_bar_h_px
    canvas_w = screen_w_px - 2 * canvas_x_pad_px
    sx = canvas_x_pad_px + true_x * canvas_w
    sy = url_bar_h_px + true_y * canvas_h / max_y_coord
    return sx, sy


def big_star_residuals(
    per_frame_h: pd.DataFrame,
    labels_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    screen_w_px: int = 2388,
    screen_h_px: int = 1668,
    url_bar_h_px: int = 272,
    canvas_x_pad_px: int = 233,
    max_y_coord: float = 0.75,
) -> pd.DataFrame:
    """Compute held-out big_star reprojection residuals.

    For each frame where big_star is labeled visible & confident, finds the
    most recently revealed trial point, projects its true screen-xy through H,
    and measures the distance to the labeled frame position.

    Returns columns:
        frame_idx, trial_idx, tpt, true_screen_x, true_screen_y,
        predicted_frame_x, predicted_frame_y,
        labeled_frame_x, labeled_frame_y, residual_px.
    """
    star_labels = labels_df[
        (labels_df.label_type == "big_star")
        & (labels_df.visible == True)
        & (labels_df.quality == "confident")
    ]

    included_h = per_frame_h[per_frame_h.excluded_reason == ""]

    rows: list[dict] = []
    for _, label_row in star_labels.iterrows():
        frame_idx = int(label_row.frame_idx)

        h_rows = included_h[included_h.frame_idx == frame_idx]
        if h_rows.empty:
            continue

        h_row = h_rows.iloc[0]
        H_mat = np.array(
            [
                [h_row.h00, h_row.h01, h_row.h02],
                [h_row.h10, h_row.h11, h_row.h12],
                [h_row.h20, h_row.h21, h_row.h22],
            ]
        )

        prior = trials_df[
            trials_df.video_frame_reveal.notna()
            & (trials_df.video_frame_reveal <= frame_idx)
        ]
        if prior.empty:
            continue

        active = prior.loc[prior.video_frame_reveal.idxmax()]
        true_sx, true_sy = behavior_to_screen(
            float(active.true_x), float(active.true_y),
            screen_w_px=screen_w_px,
            screen_h_px=screen_h_px,
            url_bar_h_px=url_bar_h_px,
            canvas_x_pad_px=canvas_x_pad_px,
            max_y_coord=max_y_coord,
        )

        pred_fx, pred_fy = _pt(H_mat, (true_sx, true_sy))
        labeled_fx = float(label_row.x_frame)
        labeled_fy = float(label_row.y_frame)
        residual_px = float(np.hypot(pred_fx - labeled_fx, pred_fy - labeled_fy))

        rows.append(
            {
                "frame_idx": frame_idx,
                "trial_idx": int(active.trial_idx),
                "tpt": int(active.tpt),
                "true_screen_x": true_sx,
                "true_screen_y": true_sy,
                "predicted_frame_x": pred_fx,
                "predicted_frame_y": pred_fy,
                "labeled_frame_x": labeled_fx,
                "labeled_frame_y": labeled_fy,
                "residual_px": residual_px,
            }
        )

    return pd.DataFrame(rows)
