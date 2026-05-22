"""Snakemake script: run Phase-1a screen corner detector on full video, save parquet.

Outputs one row per frame with smoothed BL/BR screen-corner frame-coordinates.
Frames where detect_corners() returned None get NaN xy and no_screen=True.
"""

import cv2
import numpy as np
import pandas as pd

from screen_detection import detect_corners
from corner_smoother import smooth_corners

video_path = snakemake.input.video
out_path = snakemake.output.parquet

cap = cv2.VideoCapture(video_path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw = []
for _ in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break
    raw.append(detect_corners(frame))
cap.release()

n_frames = len(raw)
no_screen_mask = np.array([r is None for r in raw], dtype=bool)

# smooth_corners interpolates gaps from None frames; we re-apply NaN afterward.
# [TL, TR, BR, BL] ordering → index 2=BR, 3=BL.
smoothed = smooth_corners(raw)  # (n_frames, 4, 2), float32

screen_bl = smoothed[:, 3, :].copy()  # BL
screen_br = smoothed[:, 2, :].copy()  # BR

screen_bl[no_screen_mask] = np.nan
screen_br[no_screen_mask] = np.nan

df = pd.DataFrame({
    "frame_idx": np.arange(n_frames, dtype=np.int64),
    "screen_bl_x": screen_bl[:, 0].astype(np.float64),
    "screen_bl_y": screen_bl[:, 1].astype(np.float64),
    "screen_br_x": screen_br[:, 0].astype(np.float64),
    "screen_br_y": screen_br[:, 1].astype(np.float64),
    "no_screen": no_screen_mask,
})
df.to_parquet(out_path, index=False)
print(f"Saved {out_path}  ({n_frames} frames, {int(no_screen_mask.sum())} no-screen)")
