# Screen-Gaze Video Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a side-by-side video: left panel shows the original scene with the detected screen outline and fixation dot; right panel shows the perspective-corrected iPad view with the gaze dot in screen space.

**Architecture:** One script composes the three upstream modules (`screen_detection`, `corner_smoother`, `gaze_screen`) into a full rendering pipeline. Frames are decoded with OpenCV, annotated, and piped through ffmpeg for h264 output. The output is 1920×540: two 960×540 panels side by side. The left panel is the original frame scaled 50%; the right panel is the iPad screen warped to fill 960×540 via a rendering homography.

**Tech Stack:** Python 3.14, OpenCV (`cv2`), NumPy, pandas, ffmpeg (subprocess pipe). Run everything with `uv run`.

---

## Context

- **Worktree root:** `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/ec348-eyetrack-overlay/`
- **Depends on:** All three upstream plans must be complete:
  - `src/screen_detection.py` — `detect_corners(frame)`
  - `src/corner_smoother.py` — `smooth_corners(raw, window)`
  - `src/gaze_screen.py` — `load_fixations()`, `build_homography()`, `map_gaze_to_screen()`
- **Video:** `/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4` — 1920×1080, 24.95 fps, 21 min 17 s
- **TSV:** `/Users/jon/Projects/dot-prediction/data/EC347/tobii/EC347_B16_tobii.tsv`
- **Output:** `results/screen_gaze_overlay.mp4`
- **All commands use `uv run`** from the worktree root.

---

## Output layout

```
┌──────────────────┬──────────────────┐  total: 1920×540
│  Original frame  │  Rectified iPad  │
│  960×540         │  960×540         │
│  (scaled 50%)    │  (warpPerspect.) │
│  + screen outline│  + gaze dot      │
│  + gaze dot      │                  │
└──────────────────┴──────────────────┘
```

The left panel is the original 1920×1080 frame scaled exactly 50% (`cv2.resize` with `INTER_AREA`). The right panel uses a "rendering homography" that maps the detected iPad corners in video space to the rectangle `(0,0)–(960,540)`.

**Two distinct homographies:**
- **Analysis H** (`build_homography` from `gaze_screen.py`): maps video corners → iPad native (2388×1668). Used to transform gaze point to iPad screen coordinates (stored per frame, not used for rendering).
- **Render H**: maps video corners → 960×540 render panel. Used by `warpPerspective` to produce the rectified view. Built inline in the renderer with `cv2.getPerspectiveTransform`.

---

## Gaze lookup per frame

The Tobii gaze data is at 100 Hz (~10 ms intervals). Video is at ~25 fps (~40 ms intervals). For each video frame at time `t`:

1. Find the nearest gaze sample (binary search by timestamp).
2. If the nearest sample is more than 40 ms away: no gaze overlay.
3. If `is_fixation` is True: use `fix_x`, `fix_y` (the I-VT centroid).
4. If `is_saccade` is True: use `gaze_x`, `gaze_y` (raw gaze).
5. Otherwise (EyesNotFound, Unclassified): no gaze overlay.

---

## Gaze dot appearance

| Condition | Left panel (video space) | Right panel (screen space) |
|-----------|--------------------------|---------------------------|
| Fixation  | Red filled circle r=15, white outline r=17 | Same, scaled to panel |
| Saccade   | Small dim blue circle r=7 | Same |
| None      | No dot | No dot |

For the right panel: transform the gaze point through the **render H** (not the analysis H) to get its position in the 960×540 panel. This keeps left and right panels visually consistent.

---

## File Structure

| Path | Purpose |
|------|---------|
| `scripts/render_screen_gaze.py` | Full rendering pipeline |

No new library code — this script is pure composition of upstream modules.

---

## Task 1: Two-second smoke test

Build the renderer and confirm it produces valid output on a 2-second clip before committing to a full render.

**Files:**
- Create: `scripts/render_screen_gaze.py`

- [ ] **Step 1: Write the renderer**

Create `scripts/render_screen_gaze.py`:

```python
#!/usr/bin/env python3
"""
Render a side-by-side video: original scene + perspective-corrected iPad view,
both with fixation-snap gaze overlay.

Usage:
    uv run python scripts/render_screen_gaze.py [--start S] [--duration D] [--output PATH]

Defaults: start=120s, duration=30s, output=results/screen_gaze_overlay.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.screen_detection import detect_corners
from src.corner_smoother import smooth_corners
from src.gaze_screen import load_fixations, build_homography, map_gaze_to_screen, IPAD_W, IPAD_H

VIDEO = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
TSV   = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/EC347_B16_tobii.tsv")

# Output panel dimensions
PANEL_W, PANEL_H = 960, 540   # each panel
OUT_W = PANEL_W * 2           # 1920
OUT_H = PANEL_H               # 540

MAX_GAZE_GAP_S = 0.04         # ignore gaze if nearest sample is further than this


# ---------------------------------------------------------------------------
# Gaze lookup
# ---------------------------------------------------------------------------

def build_gaze_index(fixations, fps: float, start_frame: int, n_frames: int) -> dict:
    """Pre-compute per-frame gaze state.

    Returns a dict with numpy arrays of length n_frames:
        ftype  — object array: 'fixation' | 'saccade' | 'blank'
        gaze_x — float32, video-pixel X (NaN when blank)
        gaze_y — float32, video-pixel Y (NaN when blank)
    """
    frame_times = (start_frame + np.arange(n_frames)) / fps
    ts = fixations["timestamp"].to_numpy()

    idx = np.searchsorted(ts, frame_times)
    idx = np.clip(idx, 0, len(ts) - 1)
    before = np.clip(idx - 1, 0, len(ts) - 1)
    closer = np.abs(ts[before] - frame_times) < np.abs(ts[idx] - frame_times)
    idx = np.where(closer, before, idx)

    gap   = np.abs(ts[idx] - frame_times)
    in_range = gap <= MAX_GAZE_GAP_S

    is_fix = fixations["is_fixation"].to_numpy()[idx]
    is_sac = fixations["is_saccade"].to_numpy()[idx]
    fix_x  = fixations["fix_x"].to_numpy(dtype=float)[idx]
    fix_y  = fixations["fix_y"].to_numpy(dtype=float)[idx]
    gaze_x = fixations["gaze_x"].to_numpy(dtype=float)[idx]
    gaze_y = fixations["gaze_y"].to_numpy(dtype=float)[idx]

    ftype  = np.where(in_range & is_fix, "fixation",
             np.where(in_range & is_sac, "saccade", "blank"))
    gx = np.where(ftype == "fixation", fix_x,
         np.where(ftype == "saccade",  gaze_x, np.nan)).astype(np.float32)
    gy = np.where(ftype == "fixation", fix_y,
         np.where(ftype == "saccade",  gaze_y, np.nan)).astype(np.float32)

    return dict(ftype=ftype, gaze_x=gx, gaze_y=gy)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_screen_outline(frame: np.ndarray, corners: np.ndarray) -> None:
    """Draw a green quadrilateral outline on the frame (in-place)."""
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 0), thickness=2)


def draw_gaze_dot(panel: np.ndarray, x: float, y: float, ftype: str) -> None:
    """Draw fixation or saccade indicator on a panel (in-place)."""
    ix, iy = int(round(x)), int(round(y))
    h, w = panel.shape[:2]
    if not (0 <= ix < w and 0 <= iy < h):
        return
    if ftype == "fixation":
        cv2.circle(panel, (ix, iy), 15, (0, 0, 220), -1)
        cv2.circle(panel, (ix, iy), 17, (255, 255, 255), 2)
    elif ftype == "saccade":
        cv2.circle(panel, (ix, iy), 7, (180, 80, 80), -1)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def build_left_panel(frame: np.ndarray, corners: np.ndarray,
                     gaze_x: float, gaze_y: float, ftype: str) -> np.ndarray:
    """Scale original frame to 960×540, draw outline and gaze dot."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    # Scale corners to panel coords (50%)
    scaled_corners = corners * 0.5
    draw_screen_outline(panel, scaled_corners)
    if ftype != "blank" and not np.isnan(gaze_x):
        draw_gaze_dot(panel, gaze_x * 0.5, gaze_y * 0.5, ftype)
    return panel


def build_right_panel(frame: np.ndarray, corners: np.ndarray,
                      gaze_x: float, gaze_y: float, ftype: str) -> np.ndarray:
    """Warp iPad screen area to 960×540, draw gaze dot in panel coords."""
    dst_corners = np.array([
        [0,       0      ],
        [PANEL_W, 0      ],
        [PANEL_W, PANEL_H],
        [0,       PANEL_H],
    ], dtype=np.float32)
    H_render = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
    panel = cv2.warpPerspective(frame, H_render, (PANEL_W, PANEL_H))

    if ftype != "blank" and not np.isnan(gaze_x):
        pt = np.array([[gaze_x, gaze_y]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt.reshape(1, 1, 2), H_render).reshape(2)
        draw_gaze_dot(panel, float(mapped[0]), float(mapped[1]), ftype)

    return panel


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def render(start_s: float, duration_s: float, out_path: Path) -> None:
    print("Loading fixations...")
    fixations = load_fixations(TSV)
    print(f"  {len(fixations):,} eye-tracker rows")

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_s * fps)
    n_frames = min(int(duration_s * fps), total_frames - start_frame)
    print(f"  Video: {fps:.2f} fps | clip: {start_s}s–{start_s+duration_s}s ({n_frames} frames)")

    # Pass 1: detect corners for every frame
    print("Pass 1/2 — detecting screen corners...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    raw_corners: list = []
    for fi in range(n_frames):
        ret, frame = cap.read()
        raw_corners.append(detect_corners(frame) if ret else None)
        if fi % 200 == 0:
            detected = sum(1 for c in raw_corners if c is not None)
            print(f"  frame {fi}/{n_frames} — detected {detected} so far")

    n_det = sum(1 for c in raw_corners if c is not None)
    print(f"  Detection rate: {n_det}/{n_frames} ({n_det/n_frames*100:.1f}%)")

    print("Smoothing corners...")
    smoothed = smooth_corners(raw_corners, window=10)

    # Gaze index
    gaze_idx = build_gaze_index(fixations, fps, start_frame, n_frames)

    # Pass 2: render
    print("Pass 2/2 — rendering...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{OUT_W}x{OUT_H}", "-pix_fmt", "bgr24", "-r", str(fps),
         "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-pix_fmt", "yuv420p",
         str(out_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        corners = smoothed[fi]
        ftype   = gaze_idx["ftype"][fi]
        gaze_x  = float(gaze_idx["gaze_x"][fi])
        gaze_y  = float(gaze_idx["gaze_y"][fi])

        left  = build_left_panel(frame, corners, gaze_x, gaze_y, ftype)
        right = build_right_panel(frame, corners, gaze_x, gaze_y, ftype)
        combined = np.concatenate([left, right], axis=1)

        proc.stdin.write(combined.tobytes())

        if fi % 200 == 0:
            print(f"  rendered {fi}/{n_frames} ({fi/n_frames*100:.0f}%)")

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"\nDone → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output",   type=Path,
                        default=Path(__file__).resolve().parent.parent / "results" / "screen_gaze_overlay.mp4")
    args = parser.parse_args()
    render(args.start, args.duration, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run 2-second smoke test**

```bash
uv run python scripts/render_screen_gaze.py --start 120 --duration 2 \
    --output results/screen_gaze_smoke.mp4
```

Expected output:
```
Loading fixations...
  127,467 eye-tracker rows
  Video: 24.95 fps | clip: 120.0s–122.0s (50 frames)
Pass 1/2 — detecting screen corners...
  Detection rate: 48/50 (96.0%)
Smoothing corners...
Pass 2/2 — rendering...
Done → .../results/screen_gaze_smoke.mp4
```

- [ ] **Step 3: Inspect smoke test output**

```bash
ffprobe results/screen_gaze_smoke.mp4 2>&1 | grep -E "Duration|Video:"
```

Expected:
```
Duration: 00:00:02...
Video: h264, yuv420p, 1920x540
```

Open `results/screen_gaze_smoke.mp4` and verify:
- Left panel: original frame at 50% scale, green quadrilateral drawn on the iPad screen edges, red dot visible on screen
- Right panel: iPad screen fills the full 960×540 panel (perspective-corrected), red dot visible
- If corners look wrong: check `results/debug_detection/` images from the detection plan and re-tune `_THRESH`/`_CLOSE_K`

- [ ] **Step 4: Commit**

```bash
git add scripts/render_screen_gaze.py
git commit -m "feat: add screen-gaze side-by-side video renderer"
```

---

## Task 2: Full 30-second render

- [ ] **Step 1: Render the 30-second clip**

```bash
uv run python scripts/render_screen_gaze.py --start 120 --duration 30 \
    --output results/screen_gaze_overlay.mp4
```

Expected duration: ~2–4 minutes (two-pass: detection then render).

- [ ] **Step 2: Inspect output**

```bash
ffprobe results/screen_gaze_overlay.mp4 2>&1 | grep -E "Duration|Video:"
```

Expected:
```
Duration: 00:00:30...
Video: h264, yuv420p, 1920x540
```

Watch the video and verify:
- Right panel stays stable (corners smoothed — no jitter)
- Gaze dot disappears briefly when subject's hand covers corners (expected)
- Gaze dot holds during fixations, jumps during saccades

- [ ] **Step 3: Done**

The 30-second render is the final deliverable for this plan. No additional commit needed — `scripts/render_screen_gaze.py` was already committed in Task 1.

---

## Troubleshooting

**Right panel is black or severely distorted:** The corners are wrong. Check the debug images from `scripts/debug_screen_detection.py`. The most common cause is `detect_corners` returning the laptop corners instead of the iPad (only occurs at t<60s — use `--start 120` or later).

**Gaze dot appears off-screen in right panel:** The homography is slightly off due to corner detection error. This reduces after smoothing. If systematic, check whether the TSV gaze coordinates are in the expected 1920×1080 space (confirmed: `Fixation point X` max ≈ 1090, within 1920).

**Detection rate < 80%:** Lower `_THRESH` in `src/screen_detection.py` from 50 to 40 and re-run.

**Output video has wrong frame rate or duration:** Check that `fps` from `cap.get(cv2.CAP_PROP_FPS)` is being passed correctly to ffmpeg `-r`. Should be `24.95`.
