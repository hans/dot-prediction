"""
Regenerate cascade_trajectory.png from current per-frame parquet and extract
frame crops around the 750-1200 snap-onset region for manual annotation.

Usage:
    uv run python notebooks/trajectory_and_annotation_frames.py
"""
# %% tags=["parameters"]
import sys
from pathlib import Path

_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in dir()
    else Path("..").resolve()
)
sys.path.insert(0, str(_ROOT / "src"))

subject = "EC347"
video_path = _ROOT / f"data/{subject}/tobii/scenevideo.mp4"
per_frame_path = _ROOT / f"results_scratch/{subject}/phase1c_per_frame.parquet"
labels_path = _ROOT / f"results/{subject}/homography_labels.parquet"
out_dir = _ROOT / f"results_scratch/{subject}"

SNAP_LO = 750
SNAP_HI = 1200
SNAP_STEP = 25  # extract one frame every 25 frames in the snap region

# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pf = pd.read_parquet(per_frame_path)
labels_df = pd.read_parquet(labels_path)

# ── 1. Trajectory plot ────────────────────────────────────────────────────────

box_bl_label_frames = labels_df[labels_df.label_type == "box_bl"][["frame_idx", "x_frame", "y_frame"]].rename(columns={"x_frame": "box_bl_lx", "y_frame": "box_bl_ly"})
box_br_label_frames = labels_df[labels_df.label_type == "box_br"][["frame_idx", "x_frame", "y_frame"]].rename(columns={"x_frame": "box_br_lx", "y_frame": "box_br_ly"})

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
ax_full, ax_zoom = axes

for ax, xlim, title in [
    (ax_full, None, "Box corner y-positions vs screen corner y-positions (full video)"),
    (ax_zoom, (SNAP_LO, SNAP_HI), f"Zoomed: frames {SNAP_LO}-{SNAP_HI} — snap onset"),
]:
    sub = pf.copy()
    if xlim:
        sub = sub[(sub.frame_idx >= xlim[0]) & (sub.frame_idx <= xlim[1])]

    # Stored tracker positions
    detected = sub[sub.detection_status.isin(["detected", "interpolated", "extrapolated"])]
    ax.plot(detected.frame_idx, detected.box_br_y, color="cyan", lw=0.8, alpha=0.8, label="box_br_y (stored)")
    ax.plot(detected.frame_idx, detected.box_bl_y, color="magenta", lw=0.8, alpha=0.8, label="box_bl_y (stored)")
    ax.plot(sub.frame_idx, sub.screen_bl_y, color="gold", lw=0.7, alpha=0.6, label="screen_bl_y")
    ax.plot(sub.frame_idx, sub.screen_br_y, color="dodgerblue", lw=0.7, alpha=0.6, label="screen_br_y")

    # Highlight no-screen gaps
    no_screen = sub[sub.detection_status == "no_screen"]
    if len(no_screen):
        for _, seg_start in no_screen.groupby((no_screen.frame_idx.diff() != 1).cumsum()).apply(lambda g: g.iloc[[0, -1]]).groupby(level=0):
            x0, x1 = seg_start.frame_idx.min(), seg_start.frame_idx.max()
            ax.axvspan(x0, x1, alpha=0.15, color="gray", label="no_screen gap" if x0 == no_screen.frame_idx.iloc[0] else "")

    # Hand labels as scatter
    lbl_sub = box_bl_label_frames.copy()
    lbl_br_sub = box_br_label_frames.copy()
    if xlim:
        lbl_sub = lbl_sub[(lbl_sub.frame_idx >= xlim[0]) & (lbl_sub.frame_idx <= xlim[1])]
        lbl_br_sub = lbl_br_sub[(lbl_br_sub.frame_idx >= xlim[0]) & (lbl_br_sub.frame_idx <= xlim[1])]
    ax.scatter(lbl_br_sub.frame_idx, lbl_br_sub.box_br_ly, color="dodgerblue", marker="s", s=40, zorder=5, label="box_br label")
    ax.scatter(lbl_sub.frame_idx, lbl_sub.box_bl_ly, color="magenta", marker="s", s=40, zorder=5, label="box_bl label")

    # Mark snap region in full plot
    if xlim is None:
        ax.axvspan(SNAP_LO, SNAP_HI, alpha=0.08, color="red")
        ax.axvline(SNAP_LO, color="red", lw=0.8, linestyle="--")
        ax.axvline(SNAP_HI, color="red", lw=0.8, linestyle="--")

    ax.set_xlabel("frame_idx")
    ax.set_ylabel("frame y-coordinate")
    ax.set_title(title)
    if xlim is None:
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    else:
        ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
out_path = out_dir / "cascade_trajectory.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Trajectory plot saved → {out_path}")

# ── 2. Extract frame crops for annotation ────────────────────────────────────

snap_frames = list(range(SNAP_LO, SNAP_HI + 1, SNAP_STEP))
# Also include any existing label frames in the region so we can see alignment
existing_in_region = sorted(
    int(f) for f in labels_df[
        (labels_df.frame_idx >= SNAP_LO) & (labels_df.frame_idx <= SNAP_HI)
    ].frame_idx.unique()
)
snap_frames = sorted(set(snap_frames) | set(existing_in_region))
print(f"\nExtracting {len(snap_frames)} frames in {SNAP_LO}-{SNAP_HI} for annotation:")
print(f"  {snap_frames}")

if not video_path.exists():
    print(f"[WARN] Video not found at {video_path} — skipping frame extraction.")
else:
    pf_lookup = pf.set_index("frame_idx")
    labels_lookup = labels_df.set_index("frame_idx")

    LABEL_COLORS_BGR = {
        "screen_bl": (0, 204, 255),
        "screen_br": (255, 136, 0),
        "box_bl":    (0, 200, 255),
        "box_br":    (180, 0, 255),
        "big_star":  (255, 255, 255),
    }

    cap = cv2.VideoCapture(str(video_path))
    snap_out_dir = out_dir / "snap_onset_frames"
    snap_out_dir.mkdir(exist_ok=True)

    for fidx in snap_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [WARN] Could not read frame {fidx}")
            continue

        vis = frame.copy()

        # Draw tracker prediction (open circle) if available
        if fidx in pf_lookup.index:
            row = pf_lookup.loc[fidx]
            if row.detection_status in ("detected", "interpolated", "extrapolated"):
                for col, color in [("box_bl", (0, 200, 255)), ("box_br", (180, 0, 255))]:
                    x, y = row[f"{col}_x"], row[f"{col}_y"]
                    if pd.notna(x) and pd.notna(y):
                        cv2.circle(vis, (int(x), int(y)), 12, color, 2, cv2.LINE_AA)

        # Draw hand labels (filled dot) if available
        if fidx in labels_lookup.index:
            frame_labels = labels_lookup.loc[[fidx]]
            for _, lr in frame_labels.iterrows():
                if lr.get("visible", True) and lr.label_type in LABEL_COLORS_BGR:
                    cv2.circle(vis, (int(lr.x_frame), int(lr.y_frame)), 8,
                               LABEL_COLORS_BGR[lr.label_type], -1, cv2.LINE_AA)

        status = pf_lookup.loc[fidx].detection_status if fidx in pf_lookup.index else "unknown"
        cv2.putText(vis, f"frame {fidx}  status={status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        out_file = snap_out_dir / f"snap_f{fidx:05d}.jpg"
        cv2.imwrite(str(out_file), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])

    cap.release()
    print(f"\nFrame crops saved → {snap_out_dir}/")
    print(f"Use label_homography_correspondences.py to annotate box_bl/box_br at these frames.")
