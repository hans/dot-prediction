# %% [markdown]
# # Phase 1c — box corner detector evaluation
#
# Runs the full Phase-1c pipeline:
#   1. Preflight: detector accuracy against 26 hand-labeled frames.
#   2. Full-video detection (forward + backward passes from seed frame 664).
#   3. Interpolation pass for gaps.
#   4. big_star held-out residual validation.
#   5. Spot-check overlays (frames 664, 1550, 2288, 30125).

# %% tags=["parameters"]
import sys
from pathlib import Path
from tqdm.auto import tqdm, trange

_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd()
    if Path("src").is_dir()
    else Path.cwd().parent
)
sys.path.insert(0, str(_ROOT / "src"))

subject = "EC347"
video_path = _ROOT / f"data/{subject}/tobii/scenevideo.mp4"
labels_path = str(_ROOT / f"results/{subject}/homography_labels.parquet")
calibration_path = str(_ROOT / f"results/{subject}/homography_eval/homography_box_calibration.json")
screen_corners_path = str(_ROOT / f"results/{subject}/screen_corners.parquet")
trials_path = str(_ROOT / f"results/{subject}/trials_with_video.parquet")
out_per_frame = str(_ROOT / f"results_scratch/{subject}/phase1c_per_frame.parquet")
out_calibration = str(_ROOT / f"results_scratch/{subject}/phase1c_calibration_used.json")

SCREEN_W_PX = 2388
SCREEN_H_PX = 1668
URL_BAR_H_PX = 272
CANVAS_X_PAD_PX = 233
MAX_Y_COORD = 0.75

SEED_FRAME = 664
# Frames to render as spot-check overlays. Empty list → auto-select seed frame
# + top-8 worst-residual frames after the run completes.
SPOT_CHECK_FRAMES: list[int] = []

out_trajectory = str(Path(out_per_frame).parent / "cascade_trajectory.png")
out_residual_hist = str(Path(out_per_frame).parent / "big_star_residual_hist.png")
out_residual_scatter = str(Path(out_per_frame).parent / "big_star_residual_vs_frame.png")

LABEL_COLORS_HEX = {
    "screen_bl": "#ffcc00",
    "screen_br": "#4488ff",
    "box_bl":    "#ff44ff",
    "box_br":    "#44ffff",
    "big_star":  "#ffffff",
}

# %%
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from box_corner_detector import detect_box_corner
from homography_solver import behavior_to_screen

video_path = Path(video_path)
labels_path = Path(labels_path)
calibration_path = Path(calibration_path)
screen_corners_path = Path(screen_corners_path)
trials_path = Path(trials_path)
out_per_frame = Path(out_per_frame)
out_calibration = Path(out_calibration)
out_per_frame.parent.mkdir(parents=True, exist_ok=True)
out_dir = out_per_frame.parent


def _pt(H: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    v = H @ np.array([xy[0], xy[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def _mat(row) -> np.ndarray:
    return np.array([
        [row["h00"], row["h01"], row["h02"]],
        [row["h10"], row["h11"], row["h12"]],
        [row["h20"], row["h21"], row["h22"]],
    ])


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


LABEL_COLORS_BGR = {lt: _hex_to_bgr(c) for lt, c in LABEL_COLORS_HEX.items()}

# %% [markdown]
# ## Load data

# %%
labels_df = pd.read_parquet(labels_path)
trials_df = pd.read_parquet(trials_path)
screen_corners_df = pd.read_parquet(screen_corners_path).set_index("frame_idx")

with open(calibration_path) as f:
    calib = json.load(f)
box_bl_screen = tuple(calib["box_bl_screen"])
box_br_screen = tuple(calib["box_br_screen"])

print(f"Labels: {len(labels_df)} rows across {labels_df.frame_idx.nunique()} frames")
print(f"Screen corners: {len(screen_corners_df)} frames, "
      f"{screen_corners_df.no_screen.sum()} no-screen")
print(f"box_bl_screen = ({box_bl_screen[0]:.1f}, {box_bl_screen[1]:.1f})")
print(f"box_br_screen = ({box_br_screen[0]:.1f}, {box_br_screen[1]:.1f})")

# %%
# Save copy of calibration for provenance.
with open(out_calibration, "w") as f:
    json.dump(calib, f, indent=2)
print(f"Calibration copy saved → {out_calibration}")

# %% [markdown]
# ## Preflight: per-frame detection error against hand labels
#
# Run the detector on all labeled frames using the **true label positions as the
# prediction** (zero prediction error). This isolates detector quality from H-tracking.

# %%
labeled_frames = (
    labels_df[labels_df.label_type.isin(["box_bl", "box_br"])]
    .groupby("frame_idx")
    .filter(lambda g: set(g.label_type) >= {"box_bl", "box_br"})
    .frame_idx.unique()
)
labeled_frames = sorted(labeled_frames)

if not video_path.exists():
    print(f"[WARN] Video not found at {video_path!r} — skipping preflight.")
    preflight_rows = []
else:
    cap_pf = cv2.VideoCapture(str(video_path))
    preflight_rows = []

    for fidx in labeled_frames:
        cap_pf.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap_pf.read()
        if not ok:
            continue

        by_type = {r.label_type: r for _, r in labels_df[labels_df.frame_idx == fidx].iterrows()}
        if "box_bl" not in by_type or "box_br" not in by_type:
            continue

        bl_label = (by_type["box_bl"].x_frame, by_type["box_bl"].y_frame)
        br_label = (by_type["box_br"].x_frame, by_type["box_br"].y_frame)

        bl_det = detect_box_corner(frame, bl_label, "bl")
        br_det = detect_box_corner(frame, br_label, "br")

        bl_err = float(np.hypot(bl_det[0] - bl_label[0], bl_det[1] - bl_label[1])) if bl_det else None
        br_err = float(np.hypot(br_det[0] - br_label[0], br_det[1] - br_label[1])) if br_det else None

        note = "seed frame" if fidx == SEED_FRAME else ""
        if (bl_err is not None and bl_err > 8) or (br_err is not None and br_err > 8):
            note += (" | " if note else "") + "FLAG >8px"
        preflight_rows.append({
            "frame_idx": fidx,
            "box_bl_error_px": bl_err,
            "box_br_error_px": br_err,
            "note": note,
        })

    cap_pf.release()

preflight_df = pd.DataFrame(preflight_rows)
print(f"\nPreflight: {len(preflight_df)} labeled frames\n")
print(f"{'frame_idx':>10}  {'box_bl_err':>12}  {'box_br_err':>12}  note")
print("-" * 60)
for _, row in preflight_df.iterrows():
    bl_s = f"{row.box_bl_error_px:.1f}" if row.box_bl_error_px is not None else "None"
    br_s = f"{row.box_br_error_px:.1f}" if row.box_br_error_px is not None else "None"
    print(f"{int(row.frame_idx):>10}  {bl_s:>12}  {br_s:>12}  {row.note}")

if len(preflight_df) > 0:
    valid_bl = preflight_df.box_bl_error_px.dropna()
    valid_br = preflight_df.box_br_error_px.dropna()
    print(f"\nbox_bl: median={valid_bl.median():.1f} px  max={valid_bl.max():.1f} px  "
          f">5px: {(valid_bl > 5).sum()}/{len(valid_bl)}")
    print(f"box_br: median={valid_br.median():.1f} px  max={valid_br.max():.1f} px  "
          f">5px: {(valid_br > 5).sum()}/{len(valid_br)}")

flag_count = preflight_df["note"].str.contains("FLAG").sum() if len(preflight_df) > 0 else 0
if flag_count > 0:
    print(f"\n[FLAG] {flag_count} frames with error > 8 px. "
          "Check against known outliers (f2288, f9124, f10562, f19671, f20192, "
          "f30125, f30135, f30175, f30950) before proceeding.")

# %% [markdown]
# ## Full-video detection
#
# Forward pass (frames 664..N−1) then backward pass (663..0), both seeded from
# frame 664's hand-labeled H.

# %%
_H_COLS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
_NAN = float("nan")
_SCREEN_PTS = np.array([
    [0.0, float(SCREEN_H_PX)],           # screen_bl
    [float(SCREEN_W_PX), float(SCREEN_H_PX)],  # screen_br
    list(box_bl_screen),                  # box_bl calibrated
    list(box_br_screen),                  # box_br calibrated
], dtype=np.float64)


def _empty_result(t: int) -> dict:
    return {
        "frame_idx": t,
        **{k: _NAN for k in _H_COLS},
        "detection_status": "",
        "detection_reason": "",
        "screen_bl_x": _NAN, "screen_bl_y": _NAN,
        "screen_br_x": _NAN, "screen_br_y": _NAN,
        "box_bl_x": _NAN, "box_bl_y": _NAN,
        "box_br_x": _NAN, "box_br_y": _NAN,
    }


def _process_frame(
    frame_bgr: np.ndarray,
    t: int,
    H_prev: np.ndarray,
    box_labels: dict | None = None,
) -> tuple[dict, np.ndarray | None]:
    """Run one detection step. Returns (result_dict, new_H or None).

    If box_labels contains an entry for frame t, use the labeled positions
    directly instead of running the detector (re-anchors the tracker).
    box_labels format: {frame_idx: {'bl': (x, y), 'br': (x, y)}}
    """
    r = _empty_result(t)

    # Use hand labels as detection result when available (re-anchor).
    if box_labels and t in box_labels:
        lbl = box_labels[t]
        box_bl_det = lbl.get("bl")
        box_br_det = lbl.get("br")
    else:
        box_bl_pred = _pt(H_prev, box_bl_screen)
        box_br_pred = _pt(H_prev, box_br_screen)
        box_bl_det = detect_box_corner(frame_bgr, box_bl_pred, "bl")
        box_br_det = detect_box_corner(frame_bgr, box_br_pred, "br")

    # Step C: screen corners
    if t not in screen_corners_df.index:
        r["detection_status"] = "no_screen"
        r["detection_reason"] = "no_screen"
        return r, None
    sc = screen_corners_df.loc[t]
    if bool(sc.no_screen):
        r["detection_status"] = "no_screen"
        r["detection_reason"] = "no_screen"
        return r, None
    if pd.isna(sc.screen_bl_x):
        r["detection_status"] = "missing_screen_bl"
        r["detection_reason"] = "missing_screen_bl"
        return r, None
    if pd.isna(sc.screen_br_x):
        r["detection_status"] = "missing_screen_br"
        r["detection_reason"] = "missing_screen_br"
        return r, None

    r["screen_bl_x"] = float(sc.screen_bl_x)
    r["screen_bl_y"] = float(sc.screen_bl_y)
    r["screen_br_x"] = float(sc.screen_br_x)
    r["screen_br_y"] = float(sc.screen_br_y)

    # Step D: fit H
    if box_bl_det is None:
        r["detection_status"] = "missing_box_bl"
        r["detection_reason"] = "missing_box_bl"
        return r, None
    if box_br_det is None:
        r["detection_status"] = "missing_box_br"
        r["detection_reason"] = "missing_box_br"
        return r, None

    r["box_bl_x"], r["box_bl_y"] = box_bl_det
    r["box_br_x"], r["box_br_y"] = box_br_det

    # Geometric sanity check: bl must be meaningfully left of br (device width
    # appears ≈48–56 px in frame), corners on same horizontal edge (small y-diff).
    # Skip for label-override frames (trusted ground truth).
    is_label_frame = box_labels is not None and t in box_labels
    horiz_sep = box_br_det[0] - box_bl_det[0]
    vert_spread = abs(box_bl_det[1] - box_br_det[1])
    if not is_label_frame and (horiz_sep < 15 or horiz_sep > 80 or vert_spread > 30):
        r["detection_status"] = "geometric_invalid"
        r["detection_reason"] = f"horiz_sep={horiz_sep:.0f},vert_spread={vert_spread:.0f}"
        return r, None

    frame_pts = np.array([
        [sc.screen_bl_x, sc.screen_bl_y],
        [sc.screen_br_x, sc.screen_br_y],
        list(box_bl_det),
        list(box_br_det),
    ], dtype=np.float64)

    H_new, _ = cv2.findHomography(_SCREEN_PTS, frame_pts)
    if H_new is None:
        r["detection_status"] = "missing_homography_failed"
        r["detection_reason"] = "missing_homography_failed"
        return r, None
    H_new = H_new / H_new[2, 2]

    r.update({
        "h00": H_new[0, 0], "h01": H_new[0, 1], "h02": H_new[0, 2],
        "h10": H_new[1, 0], "h11": H_new[1, 1], "h12": H_new[1, 2],
        "h20": H_new[2, 0], "h21": H_new[2, 1], "h22": H_new[2, 2],
    })
    r["detection_status"] = "detected"
    r["detection_reason"] = ""
    return r, H_new


# %%
# Build seed H from frame 664 hand labels.
seed_labels = {r.label_type: r for _, r in labels_df[labels_df.frame_idx == SEED_FRAME].iterrows()}
seed_frame_pts = np.array([
    [seed_labels["screen_bl"].x_frame, seed_labels["screen_bl"].y_frame],
    [seed_labels["screen_br"].x_frame, seed_labels["screen_br"].y_frame],
    [seed_labels["box_bl"].x_frame,    seed_labels["box_bl"].y_frame],
    [seed_labels["box_br"].x_frame,    seed_labels["box_br"].y_frame],
], dtype=np.float64)
H_seed, _ = cv2.findHomography(_SCREEN_PTS, seed_frame_pts)
H_seed = H_seed / H_seed[2, 2]

seed_result = _empty_result(SEED_FRAME)
seed_result.update({
    "h00": H_seed[0, 0], "h01": H_seed[0, 1], "h02": H_seed[0, 2],
    "h10": H_seed[1, 0], "h11": H_seed[1, 1], "h12": H_seed[1, 2],
    "h20": H_seed[2, 0], "h21": H_seed[2, 1], "h22": H_seed[2, 2],
    "detection_status": "detected",
    "detection_reason": "",
    "screen_bl_x": seed_labels["screen_bl"].x_frame,
    "screen_bl_y": seed_labels["screen_bl"].y_frame,
    "screen_br_x": seed_labels["screen_br"].x_frame,
    "screen_br_y": seed_labels["screen_br"].y_frame,
    "box_bl_x": seed_labels["box_bl"].x_frame,
    "box_bl_y": seed_labels["box_bl"].y_frame,
    "box_br_x": seed_labels["box_br"].x_frame,
    "box_br_y": seed_labels["box_br"].y_frame,
})
print(f"Seed H built from frame {SEED_FRAME} hand labels.")

# %%
# Build label-override dict: use trusted box_bl/box_br labels to re-anchor
# the tracker at frames where hand labels are available and visible.
# big_star validation frames are EXCLUDED from the override so that the held-out
# residual measures real tracker quality rather than label-override quality.
_big_star_val_frames = set(
    int(f) for f in labels_df[
        (labels_df.label_type == "big_star")
        & (labels_df.visible == True)
        & (labels_df.quality == "confident")
    ].frame_idx.unique()
)

_box_labels_df = labels_df[
    labels_df.label_type.isin(["box_bl", "box_br"]) & (labels_df.visible == True)
]
_box_labels: dict[int, dict] = {}
for fidx, grp in _box_labels_df.groupby("frame_idx"):
    if int(fidx) in _big_star_val_frames:
        continue  # held out for validation — do not override tracker
    entry: dict = {}
    for _, row in grp.iterrows():
        key = "bl" if row.label_type == "box_bl" else "br"
        entry[key] = (float(row.x_frame), float(row.y_frame))
    if "bl" in entry and "br" in entry:
        _box_labels[int(fidx)] = entry
n_heldout = len(_big_star_val_frames)
print(f"Label-override dict built: {len(_box_labels)} frames with both bl+br labels.")
print(f"  ({n_heldout} big_star validation frames held out — tracker must predict at those frames)")

# %%
if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(str(video_path))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_frames} frames")

results: dict[int, dict] = {SEED_FRAME: seed_result}

# Forward pass: frames SEED_FRAME+1 … N-1 (sequential read).
cap.set(cv2.CAP_PROP_POS_FRAMES, SEED_FRAME + 1)
H_prev = H_seed
for t in trange(SEED_FRAME + 1, n_frames):
    ok, frame = cap.read()
    if not ok:
        print(f"[WARN] Could not read frame {t}; stopping forward pass.")
        break
    r, H_new = _process_frame(frame, t, H_prev, _box_labels)
    results[t] = r
    if H_new is not None:
        H_prev = H_new

print(f"Forward pass complete ({SEED_FRAME+1}..{n_frames-1}).")

# Backward pass: frames SEED_FRAME-1 … 0 (per-frame seeks).
H_prev = H_seed
for t in trange(SEED_FRAME - 1, -1, -1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, frame = cap.read()
    if not ok:
        r = _empty_result(t)
        r["detection_status"] = "no_screen"
        r["detection_reason"] = "read_failed"
        results[t] = r
        continue
    r, H_new = _process_frame(frame, t, H_prev, _box_labels)
    results[t] = r
    if H_new is not None:
        H_prev = H_new

cap.release()
print(f"Backward pass complete ({SEED_FRAME-1}..0).")

# %% [markdown]
# ## Interpolation pass
#
# Linearly interpolate H elements across contiguous runs of non-detected,
# non-no_screen frames that are flanked by detected frames on both sides.
# Leading/trailing edge runs are filled from the nearest detected frame.

# %%
per_frame_df = (
    pd.DataFrame([results[t] for t in range(n_frames)])
    .sort_values("frame_idx")
    .reset_index(drop=True)
)

# %% [markdown]
# ## B3: temporal-jump rejection
#
# Flag detected frames where box_bl_y deviates > _DRIFT_PERSIST_PX from a
# rolling median of the preceding 100 detected frames. Flagged frames are
# marked "temporal_jump" so the interpolation pass treats them as non-detected
# and fills them from neighboring good detections.

# %%
_JUMP_THRESHOLD_PX = 25   # flag single-frame delta > this (unused below but documents intent)
_DRIFT_PERSIST_PX  = 20   # flag frames deviating > this from pre-jump level

_b3_det_mask = per_frame_df["detection_status"] == "detected"
_b3_det_df   = per_frame_df.loc[_b3_det_mask, ["box_bl_x", "box_bl_y"]].copy()

flagged: set[int] = set()
bl_y_vals = _b3_det_df["box_bl_y"].values
bl_y_idx  = _b3_det_df.index.values

for _i in range(50, len(bl_y_vals)):
    pre = [bl_y_vals[_j] for _j in range(max(0, _i - 100), _i)
           if bl_y_idx[_j] not in flagged]
    if len(pre) < 10:
        continue
    ref = np.median(pre)
    if abs(bl_y_vals[_i] - ref) > _DRIFT_PERSIST_PX:
        flagged.add(bl_y_idx[_i])

per_frame_df.loc[list(flagged), "detection_status"] = "temporal_jump"
per_frame_df.loc[list(flagged), "detection_reason"] = "temporal_jump"
print(f"B3: flagged {len(flagged)} frames as temporal_jump")

# %%
det_mask = per_frame_df.detection_status == "detected"
no_screen_mask = per_frame_df.detection_status == "no_screen"
need_fill = ~det_mask & ~no_screen_mask

det_frames = per_frame_df.loc[det_mask, "frame_idx"].values
det_positions = per_frame_df.index[det_mask].values

for i in per_frame_df.index[need_fill]:
    t = int(per_frame_df.loc[i, "frame_idx"])
    left_m = det_frames < t
    right_m = det_frames > t

    if left_m.any() and right_m.any():
        lp = det_positions[left_m][-1]
        rp = det_positions[right_m][0]
        t_l = int(per_frame_df.loc[lp, "frame_idx"])
        t_r = int(per_frame_df.loc[rp, "frame_idx"])
        alpha = (t - t_l) / (t_r - t_l)
        for col in _H_COLS:
            per_frame_df.loc[i, col] = (1 - alpha) * per_frame_df.loc[lp, col] + alpha * per_frame_df.loc[rp, col]
        per_frame_df.loc[i, "detection_status"] = "interpolated"
    elif left_m.any():
        lp = det_positions[left_m][-1]
        for col in _H_COLS:
            per_frame_df.loc[i, col] = per_frame_df.loc[lp, col]
        per_frame_df.loc[i, "detection_status"] = "extrapolated"
    elif right_m.any():
        rp = det_positions[right_m][0]
        for col in _H_COLS:
            per_frame_df.loc[i, col] = per_frame_df.loc[rp, col]
        per_frame_df.loc[i, "detection_status"] = "extrapolated"

print("Interpolation pass complete.")

# %% [markdown]
# ## big_star held-out residual

# %%
star_labels = labels_df[
    (labels_df.label_type == "big_star")
    & (labels_df.visible == True)
    & (labels_df.quality == "confident")
]

h_lookup = per_frame_df[
    per_frame_df.detection_status.isin(["detected", "interpolated", "extrapolated"])
].set_index("frame_idx")

residual_rows = []
for _, label_row in star_labels.iterrows():
    fidx = int(label_row.frame_idx)
    if fidx not in h_lookup.index:
        continue
    h_row = h_lookup.loc[fidx]
    H_mat = _mat(h_row)

    prior = trials_df[
        trials_df.video_frame_reveal.notna()
        & (trials_df.video_frame_reveal <= fidx)
    ]
    if prior.empty:
        continue
    active = prior.loc[prior.video_frame_reveal.idxmax()]
    true_sx, true_sy = behavior_to_screen(
        float(active.true_x), float(active.true_y),
        screen_w_px=SCREEN_W_PX, screen_h_px=SCREEN_H_PX,
        url_bar_h_px=URL_BAR_H_PX, canvas_x_pad_px=CANVAS_X_PAD_PX,
        max_y_coord=MAX_Y_COORD,
    )
    pred_fx, pred_fy = _pt(H_mat, (true_sx, true_sy))
    labeled_fx = float(label_row.x_frame)
    labeled_fy = float(label_row.y_frame)
    residual_rows.append({
        "frame_idx": fidx,
        "residual_px": float(np.hypot(pred_fx - labeled_fx, pred_fy - labeled_fy)),
        "true_screen_x": true_sx, "true_screen_y": true_sy,
        "predicted_frame_x": pred_fx, "predicted_frame_y": pred_fy,
        "labeled_frame_x": labeled_fx, "labeled_frame_y": labeled_fy,
    })

residuals_df = pd.DataFrame(residual_rows)

# Tag each residual row: was the box position tracker-predicted or label-overridden?
# Seed frame uses H_seed (built from labels), all others use tracker unless in _box_labels.
if len(residuals_df) > 0:
    residuals_df["box_override"] = residuals_df["frame_idx"].apply(
        lambda f: (int(f) in _box_labels) or (int(f) == SEED_FRAME)
    )

# Attach big_star_residual_px to per_frame_df.
per_frame_df["big_star_residual_px"] = _NAN
if len(residuals_df) > 0:
    res_lookup = residuals_df.set_index("frame_idx")["residual_px"]
    per_frame_df = per_frame_df.set_index("frame_idx")
    per_frame_df["big_star_residual_px"] = res_lookup
    per_frame_df = per_frame_df.reset_index()

# %% [markdown]
# ## Save per-frame parquet

# %%
per_frame_df.to_parquet(out_per_frame, index=False)
print(f"Per-frame parquet saved → {out_per_frame}  ({len(per_frame_df)} rows)")

# %% [markdown]
# ## Detection status summary

# %%
status_counts = per_frame_df.detection_status.value_counts()
n_total = len(per_frame_df)
n_no_screen = int(status_counts.get("no_screen", 0))
n_eligible = n_total - n_no_screen

print(f"\n{'Status':<18}  {'Count':>8}  {'%total':>8}  {'%eligible':>10}")
print("-" * 52)
for status in ["detected", "interpolated", "extrapolated", "no_screen"]:
    n = int(status_counts.get(status, 0))
    pct_tot = 100 * n / n_total if n_total > 0 else 0
    pct_elig = 100 * n / n_eligible if n_eligible > 0 and status != "no_screen" else float("nan")
    pct_elig_s = f"{pct_elig:.1f}%" if not np.isnan(pct_elig) else "—"
    print(f"{status:<18}  {n:>8}  {pct_tot:>7.1f}%  {pct_elig_s:>10}")

# Other missing statuses
other = [s for s in status_counts.index if s not in ["detected","interpolated","extrapolated","no_screen"]]
for s in other:
    n = int(status_counts[s])
    pct = 100 * n / n_total
    print(f"  {s:<16}  {n:>8}  {pct:>7.1f}%")

n_detected = int(status_counts.get("detected", 0))
det_rate = n_detected / n_eligible if n_eligible > 0 else 0
if det_rate < 0.85:
    print(f"\n[FLAG] detected rate among eligible frames = {det_rate:.1%} < 85% threshold.")

# %% [markdown]
# ## big_star residual histogram

# %%
if len(residuals_df) > 0:
    med = residuals_df.residual_px.median()
    q1, q3 = residuals_df.residual_px.quantile([0.25, 0.75])
    print(f"\nbig_star validation: N={len(residuals_df)}  "
          f"median={med:.2f} px  IQR=[{q1:.2f}, {q3:.2f}]  "
          f"max={residuals_df.residual_px.max():.2f}")

    # Report tracker-only frames (the real metric) vs. override/seed frames.
    if "box_override" in residuals_df.columns:
        tracker_res = residuals_df[~residuals_df.box_override]
        override_res = residuals_df[residuals_df.box_override]
        if len(tracker_res) > 0:
            tm = tracker_res.residual_px.median()
            print(f"  tracker-only (N={len(tracker_res)}): median={tm:.2f} px  "
                  f"max={tracker_res.residual_px.max():.2f}  "
                  f">20px: {(tracker_res.residual_px > 20).sum()}/{len(tracker_res)}")
        else:
            print("  [WARN] No tracker-only big_star frames — all validation frames are overridden.")
        if len(override_res) > 0:
            print(f"  label-override/seed (N={len(override_res)}): "
                  f"median={override_res.residual_px.median():.2f} px  "
                  f"(measures screen-corner quality, not tracker)")

    # Per-frame table for tracker-only frames.
    if "box_override" in residuals_df.columns and len(tracker_res) > 0:
        print(f"\n  {'frame':>7}  {'residual_px':>12}")
        print("  " + "-" * 22)
        for _, rr in tracker_res.sort_values("residual_px").iterrows():
            flag = "  [FLAG]" if rr.residual_px > 20 else ""
            print(f"  {int(rr.frame_idx):>7}  {rr.residual_px:>12.2f}{flag}")

    tm_val = tracker_res.residual_px.median() if "box_override" in residuals_df.columns and len(tracker_res) > 0 else med
    if tm_val > 20:
        print("  [FLAG] Tracker-only median > 20 px — Phase-1c box corner detector has systematic bias.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals_df.residual_px, bins=10, edgecolor="white", color="#4488ff")
    ax.axvline(med, color="red", linewidth=1.5, label=f"median={med:.1f} px")
    ax.set_xlabel("Residual (frame-px)")
    ax.set_ylabel("Count")
    ax.set_title("Phase-1c big_star reprojection residuals")
    fig.suptitle(
        f"N={len(residuals_df)}  median={med:.1f} px  IQR=[{q1:.1f}, {q3:.1f}]",
        fontsize=10, y=0.97,
    )
    ax.legend()
    hist_path = Path(out_residual_hist)
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved → {hist_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.scatter(residuals_df.frame_idx, residuals_df.residual_px, s=30, color="#4488ff")
    ax2.axhline(20, color="red", linestyle="--", linewidth=1, label="20 px threshold")
    ax2.set_xlabel("frame_idx")
    ax2.set_ylabel("Residual (px)")
    ax2.set_title("Phase-1c big_star residual vs frame index")
    ax2.legend()
    drift_path = Path(out_residual_scatter)
    fig2.savefig(drift_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"Drift plot saved → {drift_path}")
else:
    print("(no big_star residuals — skipping histogram)")

# %% [markdown]
# ## Trajectory plot

# %%
_traj_labels_bl = labels_df[labels_df.label_type == "box_bl"][["frame_idx", "y_frame"]]
_traj_labels_br = labels_df[labels_df.label_type == "box_br"][["frame_idx", "y_frame"]]

_snap_lo, _snap_hi = 750, 1200
fig_t, (ax_full_t, ax_zoom_t) = plt.subplots(2, 1, figsize=(14, 8))
for ax_t, xlim_t, title_t in [
    (ax_full_t, None, "Box corner y-positions vs screen corner y-positions (full video)"),
    (ax_zoom_t, (_snap_lo, _snap_hi), f"Zoomed: frames {_snap_lo}–{_snap_hi}"),
]:
    _sub = per_frame_df.copy()
    if xlim_t:
        _sub = _sub[(_sub.frame_idx >= xlim_t[0]) & (_sub.frame_idx <= xlim_t[1])]
    _det = _sub[_sub.detection_status.isin(["detected", "interpolated", "extrapolated"])]
    ax_t.plot(_det.frame_idx, _det.box_br_y, color="cyan", lw=0.8, alpha=0.8, label="box_br_y")
    ax_t.plot(_det.frame_idx, _det.box_bl_y, color="magenta", lw=0.8, alpha=0.8, label="box_bl_y")
    ax_t.plot(_sub.frame_idx, _sub.screen_bl_y, color="gold", lw=0.7, alpha=0.6, label="screen_bl_y")
    ax_t.plot(_sub.frame_idx, _sub.screen_br_y, color="dodgerblue", lw=0.7, alpha=0.6, label="screen_br_y")
    _lbl_sub = _traj_labels_bl if xlim_t is None else _traj_labels_bl[
        (_traj_labels_bl.frame_idx >= xlim_t[0]) & (_traj_labels_bl.frame_idx <= xlim_t[1])]
    _lbr_sub = _traj_labels_br if xlim_t is None else _traj_labels_br[
        (_traj_labels_br.frame_idx >= xlim_t[0]) & (_traj_labels_br.frame_idx <= xlim_t[1])]
    ax_t.scatter(_lbr_sub.frame_idx, _lbr_sub.y_frame, color="dodgerblue", marker="s", s=40, zorder=5, label="box_br label")
    ax_t.scatter(_lbl_sub.frame_idx, _lbl_sub.y_frame, color="magenta", marker="s", s=40, zorder=5, label="box_bl label")
    if xlim_t is None:
        ax_t.axvspan(_snap_lo, _snap_hi, alpha=0.08, color="red")
    ax_t.set_xlabel("frame_idx")
    ax_t.set_ylabel("frame y-coordinate")
    ax_t.set_title(title_t)
    ax_t.legend(loc="upper right", fontsize=7, ncol=3 if xlim_t is None else 2)
plt.tight_layout()
traj_path = Path(out_trajectory)
fig_t.savefig(traj_path, dpi=120, bbox_inches="tight")
plt.close(fig_t)
print(f"Trajectory plot saved → {traj_path}")

# %% [markdown]
# ## Spot-check overlays

# %%
# Auto-select: seed frame + top-8 worst-residual frames (or the caller-supplied list).
if SPOT_CHECK_FRAMES:
    _spot_frames = list(SPOT_CHECK_FRAMES)
else:
    _top_res = (
        residuals_df.sort_values("residual_px", ascending=False).head(8).frame_idx.tolist()
        if len(residuals_df) > 0 else []
    )
    _spot_frames = sorted(set([SEED_FRAME] + [int(f) for f in _top_res]))
print(f"Spot-check frames: {_spot_frames}")


def _draw_filled(img, xy, color, radius=7):
    cv2.circle(img, (int(round(xy[0])), int(round(xy[1]))), radius, color, -1, cv2.LINE_AA)


def _draw_open(img, xy, color, radius=11):
    cv2.circle(img, (int(round(xy[0])), int(round(xy[1]))), radius, color, 2, cv2.LINE_AA)


anchor_screen_xy = {
    "screen_bl": (0.0, float(SCREEN_H_PX)),
    "screen_br": (float(SCREEN_W_PX), float(SCREEN_H_PX)),
    "box_bl":    box_bl_screen,
    "box_br":    box_br_screen,
}

pf_lookup = per_frame_df.set_index("frame_idx")

if not video_path.exists():
    print(f"[WARN] Video not found — skipping overlays.")
else:
    cap_ov = cv2.VideoCapture(str(video_path))

    for fidx in _spot_frames:
        cap_ov.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap_ov.read()
        if not ok:
            print(f"[WARN] Could not read frame {fidx}")
            continue

        if fidx not in pf_lookup.index:
            print(f"[WARN] Frame {fidx} not in per_frame_df")
            continue
        pf_row = pf_lookup.loc[fidx]
        status = pf_row.detection_status

        # Draw hand labels (filled dots).
        frame_labels = labels_df[labels_df.frame_idx == fidx]
        for _, lr in frame_labels.iterrows():
            if not lr.visible or lr.label_type not in LABEL_COLORS_BGR:
                continue
            _draw_filled(frame, (lr.x_frame, lr.y_frame), LABEL_COLORS_BGR[lr.label_type])

        # Draw back-projections if H is available.
        if status in ("detected", "interpolated", "extrapolated"):
            H_mat = _mat(pf_row)
            for lt, (sx, sy) in anchor_screen_xy.items():
                bp_x, bp_y = _pt(H_mat, (sx, sy))
                _draw_open(frame, (bp_x, bp_y), LABEL_COLORS_BGR.get(lt, (128, 128, 128)))

            res_row = residuals_df[residuals_df.frame_idx == fidx] if len(residuals_df) > 0 else pd.DataFrame()
            if not res_row.empty:
                r = res_row.iloc[0]
                bp_x, bp_y = _pt(H_mat, (r.true_screen_x, r.true_screen_y))
                _draw_open(frame, (bp_x, bp_y), LABEL_COLORS_BGR["big_star"])

        res_str = (
            f"{residuals_df[residuals_df.frame_idx==fidx].iloc[0].residual_px:.1f}px"
            if len(residuals_df) > 0 and not residuals_df[residuals_df.frame_idx==fidx].empty
            else "n/a"
        )
        caption = f"frame {fidx}  detection_status={status}  big_star_residual={res_str}"
        for thick, color in [(2, (255, 255, 255)), (1, (0, 0, 0))]:
            cv2.putText(frame, caption, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)

        overlay_path = out_dir / f"frame_{fidx}_overlay.jpg"
        cv2.imwrite(str(overlay_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"Overlay saved → {overlay_path}")

    cap_ov.release()

# %% [markdown]
# ## Open questions
#
# 1. **Outlier frames** — if the preflight table shows exactly the 9 documented
#    outliers (f2288, f9124, f10562, f19671, f20192, f30125, f30135, f30175,
#    f30950) and no new ones, proceed. New failures → investigate before
#    accepting outputs.
#
# 2. **min_harris_response calibration** — print the distribution of normalised
#    Harris responses. If many cluster near 0.05, tune the threshold.
#
# 3. **big_star residual by detection_status** — if interpolated frames have
#    markedly higher residual than detected frames, consider a larger search window
#    as a recovery pass.
#
# 4. **Detection rate vs time** — if failures cluster late in the session (head
#    pose drift), consider per-segment calibration.
