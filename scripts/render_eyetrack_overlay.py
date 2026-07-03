#!/usr/bin/env python3
"""
Render eyetracking overlay samples on the EC347 first-person scene video.

Uses fixation-snapping: during fixations the dot holds at the I-VT fixation
centroid; during saccades / invalid samples it is hidden (or shown dimly).
This mirrors what Tobii Pro Lab's gaze-replay view does.

Three visual variants are produced from the same 20-second clip:
  fixation_dot     — dot at fixation centroid, blanked during saccades/invalid
  fixation_sized   — dot radius scales with fixation duration
  fixation_saccade — fixation dot + small dim indicator during saccades

Output: results/eyetrack_sample_<variant>.mp4
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii")
VIDEO    = DATA_DIR / "scenevideo.mp4"
TSV      = DATA_DIR / "EC347_B16_tobii.tsv"

WORKTREE = Path(__file__).resolve().parent.parent
OUT_DIR  = WORKTREE / "results"

CLIP_START    = 120.0  # seconds into video
CLIP_DURATION = 20.0   # seconds per sample

W, H = 1920, 1080


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TSV_COLS = [
    "Recording timestamp",   # microseconds
    "Sensor",
    "Eye movement type",     # Fixation | Saccade | EyesNotFound | Unclassified
    "Gaze point X",          # raw gaze, pixels
    "Gaze point Y",
    "Fixation point X",      # I-VT centroid, pixels
    "Fixation point Y",
    "Eye movement event duration",  # microseconds
    "Validity left",
    "Validity right",
]

def load_tsv() -> pd.DataFrame:
    """Load Eye Tracker rows from the Tobii TSV export."""
    df = pd.read_csv(TSV, sep="\t", usecols=TSV_COLS)
    df = df[df["Sensor"] == "Eye Tracker"].copy()
    df["timestamp"] = df["Recording timestamp"] / 1e6   # µs → s
    df["valid"] = (df["Validity left"] == "Valid") | (df["Validity right"] == "Valid")
    df["fix_dur_s"] = df["Eye movement event duration"] / 1e6
    return df.reset_index(drop=True)


def build_frame_index(tsv: pd.DataFrame, fps: float, start_frame: int, n_frames: int) -> dict:
    """Pre-compute per-frame overlay state.

    Returns a dict with arrays of length n_frames:
      type    — 'fixation' | 'saccade' | 'blank'
      fix_x   — fixation centroid X (pixels), nan if not fixation
      fix_y   — fixation centroid Y (pixels), nan if not fixation
      gaze_x  — raw gaze X (pixels), nan if invalid
      gaze_y  — raw gaze Y (pixels), nan if invalid
      fix_dur — fixation duration in seconds, nan if not fixation
    """
    MAX_GAP = 0.04  # s — if nearest sample is further away, treat as blank

    frame_times = (start_frame + np.arange(n_frames)) / fps
    ts = tsv["timestamp"].to_numpy()

    idx = np.searchsorted(ts, frame_times)
    idx = np.clip(idx, 0, len(ts) - 1)
    before = np.clip(idx - 1, 0, len(ts) - 1)
    closer = np.abs(ts[before] - frame_times) < np.abs(ts[idx] - frame_times)
    idx = np.where(closer, before, idx)

    gap   = np.abs(ts[idx] - frame_times)
    valid = tsv["valid"].to_numpy()[idx] & (gap <= MAX_GAP)
    etype = tsv["Eye movement type"].to_numpy()[idx]

    def col_or_nan(col):
        vals = tsv[col].to_numpy(dtype=float)[idx]
        vals[~valid] = np.nan
        return vals

    fix_x   = np.where(valid & (etype == "Fixation"), tsv["Fixation point X"].to_numpy(dtype=float)[idx], np.nan)
    fix_y   = np.where(valid & (etype == "Fixation"), tsv["Fixation point Y"].to_numpy(dtype=float)[idx], np.nan)
    gaze_x  = np.where(valid & (etype == "Saccade"),  tsv["Gaze point X"].to_numpy(dtype=float)[idx], np.nan)
    gaze_y  = np.where(valid & (etype == "Saccade"),  tsv["Gaze point Y"].to_numpy(dtype=float)[idx], np.nan)
    fix_dur = np.where(valid & (etype == "Fixation"),  tsv["fix_dur_s"].to_numpy(dtype=float)[idx], np.nan)

    frame_type = np.where(
        valid & (etype == "Fixation"), "fixation",
        np.where(valid & (etype == "Saccade"), "saccade", "blank")
    )

    return dict(type=frame_type, fix_x=fix_x, fix_y=fix_y,
                gaze_x=gaze_x, gaze_y=gaze_y, fix_dur=fix_dur)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

# Fixation radius range for the sized variant (duration → radius mapping)
MIN_R, MAX_R = 12, 40
MIN_DUR, MAX_DUR = 0.05, 1.0   # seconds

def _fix_radius(dur: float) -> int:
    t = np.clip((dur - MIN_DUR) / (MAX_DUR - MIN_DUR), 0, 1)
    return int(MIN_R + t * (MAX_R - MIN_R))

def draw_fixation(frame, x, y, radius=20):
    import cv2
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(frame, (ix, iy), radius,     (0, 0, 220),     -1)
    cv2.circle(frame, (ix, iy), radius + 2, (255, 255, 255),  2)

def draw_saccade_indicator(frame, x, y):
    import cv2
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(frame, (ix, iy), 8, (80, 80, 200), -1)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def ffmpeg_writer(out_path: Path, fps: float):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def render_clip(variant: str, idx: dict, fps: float, start_frame: int, n_frames: int, out_path: Path):
    import cv2

    cap  = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    proc = ffmpeg_writer(out_path, fps)

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ftype   = idx["type"][fi]
        fix_x   = idx["fix_x"][fi]
        fix_y   = idx["fix_y"][fi]
        gaze_x  = idx["gaze_x"][fi]
        gaze_y  = idx["gaze_y"][fi]
        fix_dur = idx["fix_dur"][fi]

        if variant == "fixation_dot":
            if ftype == "fixation":
                draw_fixation(frame, fix_x, fix_y)

        elif variant == "fixation_sized":
            if ftype == "fixation":
                r = _fix_radius(fix_dur) if not np.isnan(fix_dur) else 20
                draw_fixation(frame, fix_x, fix_y, radius=r)

        elif variant == "fixation_saccade":
            if ftype == "fixation":
                draw_fixation(frame, fix_x, fix_y)
            elif ftype == "saccade" and not np.isnan(gaze_x):
                draw_saccade_indicator(frame, gaze_x, gaze_y)

        proc.stdin.write(frame.tobytes())

        if fi % 100 == 0:
            print(f"  {variant}: {fi / n_frames * 100:.0f}%", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import cv2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TSV...")
    tsv = load_tsv()
    print(f"  {len(tsv):,} eye-tracker samples | {tsv['timestamp'].iloc[0]:.1f}s – {tsv['timestamp'].iloc[-1]:.1f}s")
    print("  types:", tsv["Eye movement type"].value_counts().to_dict())

    cap = cv2.VideoCapture(str(VIDEO))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = int(CLIP_START * fps)
    n_frames    = min(int(CLIP_DURATION * fps), total_frames - start_frame)
    print(f"  Video: {fps:.2f} fps | clip: {CLIP_START}s–{CLIP_START+CLIP_DURATION}s ({n_frames} frames)")

    print("Indexing by frame...")
    idx = build_frame_index(tsv, fps, start_frame, n_frames)
    for label, val in [("fixation", "fixation"), ("saccade", "saccade"), ("blank", "blank")]:
        pct = (idx["type"] == val).mean() * 100
        print(f"  {label}: {pct:.1f}%")

    for variant in ("fixation_dot", "fixation_sized", "fixation_saccade"):
        print(f"\nRendering [{variant}]...")
        out = OUT_DIR / f"eyetrack_sample_{variant}.mp4"
        render_clip(variant, idx, fps, start_frame, n_frames, out)

    print("\nDone. Clips written to:", OUT_DIR)


if __name__ == "__main__":
    main()
