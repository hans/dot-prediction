# Gaze-to-Screen Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each video frame, use the smoothed screen corners to compute a perspective homography and transform the Tobii fixation gaze point from video-pixel coordinates into iPad screen coordinates (0–2388, 0–1668).

**Architecture:** Two functions: `build_homography` computes a 3×3 perspective matrix from the 4 detected corners to the iPad's native pixel rectangle. `map_gaze_to_screen` applies that matrix to a gaze point. A separate `load_fixations` function parses the Tobii TSV export and returns a clean DataFrame. The final per-frame assembly happens in the video renderer (next plan).

**Tech Stack:** Python 3.14, OpenCV (`cv2`), NumPy, pandas. Run everything with `uv run`.

---

## Context

- **Worktree root:** `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/ec348-eyetrack-overlay/`
- **Depends on:** `src/screen_detection.py` and `src/corner_smoother.py` (Plans `2026-05-15-screen-detection.md` and `2026-05-15-corner-smoothing.md`) — must be complete.
- **Tobii TSV:** `/Users/jon/Projects/dot-prediction/data/EC347/tobii/EC347_B16_tobii.tsv`
  - 442,972 rows, tab-separated. Relevant columns:
    - `Recording timestamp` — microseconds from recording start
    - `Sensor` — filter to `"Eye Tracker"` rows only
    - `Eye movement type` — `"Fixation"` | `"Saccade"` | `"EyesNotFound"` | `"Unclassified"`
    - `Fixation point X` / `Fixation point Y` — I-VT centroid in video-pixel coords (1920×1080 space), populated only during fixations
    - `Gaze point X` / `Gaze point Y` — raw gaze in video-pixel coords, populated during saccades and fixations
    - `Validity left` / `Validity right` — `"Valid"` | `"Invalid"`
    - `Eye movement event duration` — microseconds
- **iPad native resolution:** 2388 × 1668 (11-inch iPad Pro M2, landscape)
- **Video:** 1920×1080, 24.95 fps
- **All commands use `uv run`** from the worktree root.

---

## Coordinate systems

```
Video frame (pixels):   (0,0) top-left, (1919,1079) bottom-right
iPad screen (pixels):   (0,0) top-left, (2387,1667) bottom-right
                         Maps from video TL→TR→BR→BL corners
```

The homography `H` satisfies:  
`screen_pt_homogeneous = H @ video_pt_homogeneous`

---

## File Structure

| Path | Purpose |
|------|---------|
| `src/gaze_screen.py` | `load_fixations()`, `build_homography()`, `map_gaze_to_screen()` |
| `tests/test_gaze_screen.py` | Unit tests |

---

## Task 1: Write tests

**Files:**
- Create: `tests/test_gaze_screen.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gaze_screen.py`:

```python
import numpy as np
import pytest
from src.gaze_screen import build_homography, map_gaze_to_screen

# Synthetic axis-aligned corners (flat rectangle, no perspective):
# TL=(100,100), TR=(500,100), BR=(500,400), BL=(100,400)
FLAT_CORNERS = np.array([
    [100, 100],
    [500, 100],
    [500, 400],
    [100, 400],
], dtype=np.float32)

IPAD_W, IPAD_H = 2388, 1668


def test_build_homography_shape():
    H = build_homography(FLAT_CORNERS, IPAD_W, IPAD_H)
    assert H.shape == (3, 3)
    assert H.dtype == np.float64


def test_corners_map_to_ipad_corners():
    """The 4 detected corners should map to the 4 corners of the iPad rectangle."""
    H = build_homography(FLAT_CORNERS, IPAD_W, IPAD_H)
    expected = np.array([
        [0,       0      ],  # TL
        [IPAD_W,  0      ],  # TR
        [IPAD_W,  IPAD_H ],  # BR
        [0,       IPAD_H ],  # BL
    ], dtype=np.float32)
    result = map_gaze_to_screen(FLAT_CORNERS, H)
    np.testing.assert_allclose(result, expected, atol=1.0)


def test_center_maps_to_center():
    """The centre of the video-space screen region maps to the iPad centre."""
    H = build_homography(FLAT_CORNERS, IPAD_W, IPAD_H)
    cx = (FLAT_CORNERS[:, 0].mean())
    cy = (FLAT_CORNERS[:, 1].mean())
    result = map_gaze_to_screen(np.array([[cx, cy]], dtype=np.float32), H)
    np.testing.assert_allclose(result[0, 0], IPAD_W / 2, atol=5.0)
    np.testing.assert_allclose(result[0, 1], IPAD_H / 2, atol=5.0)


def test_map_gaze_single_point():
    H = build_homography(FLAT_CORNERS, IPAD_W, IPAD_H)
    pt = np.array([[100.0, 100.0]], dtype=np.float32)  # TL corner
    result = map_gaze_to_screen(pt, H)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1.0)


def test_map_gaze_batch():
    H = build_homography(FLAT_CORNERS, IPAD_W, IPAD_H)
    pts = FLAT_CORNERS.copy()  # 4 points
    result = map_gaze_to_screen(pts, H)
    assert result.shape == (4, 2)


def test_build_homography_perspective():
    """Perspective (skewed) corners should still produce a valid homography."""
    skewed = np.array([
        [580,  370],   # TL — realistic position from actual video
        [1060, 330],   # TR
        [1090, 600],   # BR
        [510,  590],   # BL
    ], dtype=np.float32)
    H = build_homography(skewed, IPAD_W, IPAD_H)
    assert H is not None
    assert H.shape == (3, 3)
    # TL corner should map to (0, 0)
    tl = map_gaze_to_screen(skewed[[0]], H)
    np.testing.assert_allclose(tl[0], [0.0, 0.0], atol=2.0)
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
uv run pytest tests/test_gaze_screen.py -v
```

Expected: `ImportError` — `src/gaze_screen.py` doesn't exist yet.

---

## Task 2: Implement `build_homography` and `map_gaze_to_screen`

**Files:**
- Create: `src/gaze_screen.py`

- [ ] **Step 1: Write the implementation**

Create `src/gaze_screen.py`:

```python
"""Map Tobii gaze coordinates from video-pixel space to iPad screen space.

Functions:
    load_fixations  — parse Tobii TSV export
    build_homography — compute 3×3 perspective transform from screen corners
    map_gaze_to_screen — apply homography to gaze point(s)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# iPad 11-inch Pro M2, landscape
IPAD_W = 2388
IPAD_H = 1668

_TSV_COLS = [
    "Recording timestamp",
    "Sensor",
    "Eye movement type",
    "Fixation point X",
    "Fixation point Y",
    "Gaze point X",
    "Gaze point Y",
    "Eye movement event duration",
    "Validity left",
    "Validity right",
]


def load_fixations(tsv_path: Path) -> pd.DataFrame:
    """Load Eye Tracker rows from the Tobii TSV export.

    Args:
        tsv_path: Path to the `*_tobii.tsv` file.

    Returns:
        DataFrame with columns:
            timestamp   — float, seconds from recording start
            is_fixation — bool
            is_saccade  — bool
            is_valid    — bool (at least one eye valid)
            fix_x       — float, fixation centroid X in video pixels (NaN if not fixation)
            fix_y       — float, fixation centroid Y in video pixels (NaN if not fixation)
            gaze_x      — float, raw gaze X in video pixels (NaN if invalid)
            gaze_y      — float, raw gaze Y in video pixels (NaN if invalid)
            fix_dur_s   — float, fixation duration in seconds (NaN if not fixation)
    """
    df = pd.read_csv(tsv_path, sep="\t", usecols=_TSV_COLS)
    df = df[df["Sensor"] == "Eye Tracker"].copy()

    df["timestamp"]   = df["Recording timestamp"] / 1e6
    df["is_valid"]    = (df["Validity left"] == "Valid") | (df["Validity right"] == "Valid")
    df["is_fixation"] = df["Eye movement type"] == "Fixation"
    df["is_saccade"]  = df["Eye movement type"] == "Saccade"
    df["fix_x"]       = np.where(df["is_fixation"], df["Fixation point X"],  np.nan)
    df["fix_y"]       = np.where(df["is_fixation"], df["Fixation point Y"],  np.nan)
    df["gaze_x"]      = np.where(df["is_valid"],    df["Gaze point X"],      np.nan)
    df["gaze_y"]      = np.where(df["is_valid"],    df["Gaze point Y"],      np.nan)
    df["fix_dur_s"]   = np.where(df["is_fixation"], df["Eye movement event duration"] / 1e6, np.nan)

    return df[["timestamp", "is_fixation", "is_saccade", "is_valid",
               "fix_x", "fix_y", "gaze_x", "gaze_y", "fix_dur_s"]].reset_index(drop=True)


def build_homography(
    corners: np.ndarray,
    ipad_w: int = IPAD_W,
    ipad_h: int = IPAD_H,
) -> np.ndarray:
    """Compute a 3×3 perspective transform from video corners to iPad screen.

    Args:
        corners: float32 array of shape (4, 2) with corners in
            [TL, TR, BR, BL] order (video-pixel coordinates).
        ipad_w: iPad screen width in pixels. Default 2388.
        ipad_h: iPad screen height in pixels. Default 1668.

    Returns:
        3×3 float64 homography matrix H such that
        `screen_pt = H @ [vx, vy, 1]` (in homogeneous coords).
    """
    src = corners.astype(np.float32)
    dst = np.array([
        [0,      0     ],
        [ipad_w, 0     ],
        [ipad_w, ipad_h],
        [0,      ipad_h],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def map_gaze_to_screen(
    gaze_px: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Transform gaze point(s) from video-pixel space to iPad screen space.

    Args:
        gaze_px: float array of shape (N, 2) — (x, y) in video pixels.
        H: 3×3 homography from `build_homography`.

    Returns:
        float32 array of shape (N, 2) — (x, y) in iPad screen pixels.
        Points outside the screen will have coordinates outside [0, ipad_w]
        or [0, ipad_h] — callers should clamp if needed.
    """
    pts = gaze_px.astype(np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)
```

- [ ] **Step 2: Run tests — confirm they pass**

```bash
uv run pytest tests/test_gaze_screen.py -v
```

Expected output:
```
test_gaze_screen.py::test_build_homography_shape PASSED
test_gaze_screen.py::test_corners_map_to_ipad_corners PASSED
test_gaze_screen.py::test_center_maps_to_center PASSED
test_gaze_screen.py::test_map_gaze_single_point PASSED
test_gaze_screen.py::test_map_gaze_batch PASSED
test_gaze_screen.py::test_build_homography_perspective PASSED
6 passed
```

- [ ] **Step 3: Commit**

```bash
git add src/gaze_screen.py tests/test_gaze_screen.py
git commit -m "feat: add gaze-to-screen homography and fixation TSV loader"
```

---

## Task 3: Integration smoke test

Verify that real fixation data transforms into plausible iPad screen coordinates.

- [ ] **Step 1: Run the smoke test**

```bash
uv run python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
from src.gaze_screen import load_fixations, build_homography, map_gaze_to_screen, IPAD_W, IPAD_H

TSV = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/EC347_B16_tobii.tsv")

# Load fixations
fix = load_fixations(TSV)
print(f"Total eye-tracker rows: {len(fix)}")
print(f"Fixation rows: {fix['is_fixation'].sum()}")
print(f"fix_x range: {fix['fix_x'].min():.0f} – {fix['fix_x'].max():.0f}")
print(f"fix_y range: {fix['fix_y'].min():.0f} – {fix['fix_y'].max():.0f}")

# Use the approximate real corners from a typical task frame (t≈180s)
# (These are ground-truth approximations from visual inspection;
#  the actual renderer will supply smoothed corners from detect_corners.)
approx_corners = np.array([
    [630,  370],
    [1055, 335],
    [1080, 595],
    [525,  580],
], dtype=np.float32)

H = build_homography(approx_corners, IPAD_W, IPAD_H)

fix_pts = fix[fix["is_fixation"]][["fix_x", "fix_y"]].dropna().to_numpy(dtype=np.float32)
screen_pts = map_gaze_to_screen(fix_pts, H)

pct_in_screen = (
    (screen_pts[:, 0] >= 0) & (screen_pts[:, 0] <= IPAD_W) &
    (screen_pts[:, 1] >= 0) & (screen_pts[:, 1] <= IPAD_H)
).mean() * 100

print(f"\nScreen-space fixations (N={len(screen_pts)}):")
print(f"  x range: {screen_pts[:,0].min():.0f} – {screen_pts[:,0].max():.0f}  (iPad: 0–{IPAD_W})")
print(f"  y range: {screen_pts[:,1].min():.0f} – {screen_pts[:,1].max():.0f}  (iPad: 0–{IPAD_H})")
print(f"  % within screen bounds: {pct_in_screen:.1f}%")
EOF
```

Expected output (approximate):
```
Total eye-tracker rows: 127467
Fixation rows: 95876
fix_x range: 530 – 1090
fix_y range: 330 – 610
Screen-space fixations (N=...):
  x range: -200 – 2600  (iPad: 0–2388)
  y range: -100 – 1800  (iPad: 0–1668)
  % within screen bounds: ~85–95%
```

The out-of-bounds points are expected: they correspond to fixations during screen transitions, or when the subject's eye was near the edge of the display. The per-frame renderer (next plan) uses the per-frame corners from `smooth_corners`, not these fixed approximate corners, so accuracy will be higher.
