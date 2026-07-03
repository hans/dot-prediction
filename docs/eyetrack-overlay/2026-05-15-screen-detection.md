# Screen Corner Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a function that takes one BGR video frame and returns the 4 corners of the iPad screen in video-pixel coordinates, or `None` if detection fails.

**Architecture:** Brightness thresholding works because the recording is in a near-dark room with the iPad as the only light source. After t≈60s the laptop that was also visible leaves frame. Convert to grayscale → threshold → morphological close to fill the content area → find the largest 4-sided contour → return ordered corners.

**Tech Stack:** Python 3.14, OpenCV (`cv2`), NumPy. Run everything with `uv run`.

---

## Context

- **Worktree root:** `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/ec348-eyetrack-overlay/`
- **Video:** `/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4` — 1920×1080, 24.95 fps, 21 min 17 s
- **iPad:** 11-inch iPad Pro M2, native 2388×1668, landscape orientation, lying on a table
- **Screen appearance:** consistently bright (blue/purple task background + white bezels) against near-black room. White bezels on left and right are the most reliably bright features.
- **All commands use `uv run`** from the worktree root.

---

## File Structure

| Path | Purpose |
|------|---------|
| `src/screen_detection.py` | `detect_corners(frame)` + `order_corners(pts)` |
| `tests/test_screen_detection.py` | Unit tests with synthetic frames |
| `scripts/debug_screen_detection.py` | Visual QC: run detection on real video frames, save annotated images |

---

## Task 1: Project structure + stub

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_screen_detection.py`
- Create: `src/screen_detection.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_screen_detection.py`:

```python
import numpy as np
import pytest
import cv2
from src.screen_detection import detect_corners, order_corners

# Realistic-looking perspective quad that mimics the iPad's position in the video
# (off-axis, slightly rotated, as if viewed from above at an angle)
IPAD_CORNERS = np.array([
    [580,  370],  # TL
    [1060, 330],  # TR
    [1090, 600],  # BR
    [510,  590],  # BL
], dtype=np.float32)


def _make_frame(corners, brightness=200, frame_shape=(1080, 1920, 3)):
    """Synthetic dark frame with a bright filled quadrilateral."""
    frame = np.zeros(frame_shape, dtype=np.uint8)
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (brightness, brightness, brightness))
    return frame


def test_detect_corners_returns_array():
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    assert result.shape == (4, 2)
    assert result.dtype == np.float32


def test_detect_corners_ordering():
    """TL has smallest x+y, BR has largest x+y."""
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    sums = result.sum(axis=1)
    assert np.argmin(sums) == 0, "index 0 should be TL (smallest x+y)"
    assert np.argmax(sums) == 2, "index 2 should be BR (largest x+y)"


def test_detect_corners_accuracy():
    """Detected corners should be within 10 px of the ground truth."""
    ordered_gt = order_corners(IPAD_CORNERS)
    frame = _make_frame(IPAD_CORNERS)
    result = detect_corners(frame)
    assert result is not None
    assert np.allclose(result, ordered_gt, atol=10), (
        f"Detected corners too far from ground truth.\n"
        f"Expected:\n{ordered_gt}\nGot:\n{result}"
    )


def test_detect_corners_dark_frame_returns_none():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert detect_corners(frame) is None


def test_detect_corners_tiny_bright_region_returns_none():
    """A small bright blob (not a screen) should be filtered out."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:510, 900:910] = 255  # tiny 10×10 blob
    assert detect_corners(frame) is None


def test_order_corners_canonical():
    pts = np.array([[100, 50], [200, 50], [200, 150], [100, 150]], dtype=np.float32)
    ordered = order_corners(pts)
    np.testing.assert_array_equal(ordered[0], [100, 50])   # TL
    np.testing.assert_array_equal(ordered[1], [200, 50])   # TR
    np.testing.assert_array_equal(ordered[2], [200, 150])  # BR
    np.testing.assert_array_equal(ordered[3], [100, 150])  # BL
```

- [ ] **Step 3: Run tests — confirm they fail**

```bash
uv run pytest tests/test_screen_detection.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src/screen_detection.py` doesn't exist yet.

---

## Task 2: Implement `order_corners` and `detect_corners`

**Files:**
- Create: `src/screen_detection.py`

- [ ] **Step 1: Write the implementation**

Create `src/screen_detection.py`:

```python
"""Detect the 4 corners of the iPad screen in a video frame."""

import cv2
import numpy as np

# Minimum contour area in pixels. The iPad screen occupies roughly
# 25–35% of the 1920×1080 frame ≈ 500k–700k px. Use a conservative floor.
_MIN_AREA = 80_000

# Brightness threshold (0–255 grayscale). The iPad screen is much brighter
# than the near-black room. Tuned from inspecting real frames.
_THRESH = 50

# Morphological close kernel size (px). Must be large enough to fill the
# blue task background (which sits above threshold but may have thin dark
# gaps from content elements like dots/stars).
_CLOSE_K = 51


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in [TL, TR, BR, BL] order.

    Args:
        pts: shape (4, 2) or (4, 1, 2) float array of corner points.

    Returns:
        float32 array of shape (4, 2).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)          # x + y
    d = pts[:, 0] - pts[:, 1]   # x - y
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmax(d)],   # TR: largest x-y (large x, small y)
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmin(d)],   # BL: smallest x-y (small x, large y)
    ], dtype=np.float32)


def detect_corners(frame: np.ndarray) -> np.ndarray | None:
    """Detect the 4 corners of the iPad screen in a video frame.

    Args:
        frame: BGR image of shape (H, W, 3), uint8.

    Returns:
        float32 array of shape (4, 2) with corners in [TL, TR, BR, BL]
        order (video-pixel coordinates), or None if no screen is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold: screen pixels are brighter than _THRESH in near-dark room
    _, binary = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)

    # Close to fill gaps within the screen content area
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_CLOSE_K, _CLOSE_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only large contours; take the largest
    large = [c for c in contours if cv2.contourArea(c) > _MIN_AREA]
    if not large:
        return None
    largest = max(large, key=cv2.contourArea)

    # Approximate to a polygon; require exactly 4 sides.
    # Try increasing epsilon until we get a 4-vertex result (handles motion blur
    # and slight curve in the contour that can produce 5+ vertices at 0.02).
    peri = cv2.arcLength(largest, True)
    approx = None
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        candidate = cv2.approxPolyDP(largest, eps * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break
    if approx is None:
        return None

    return order_corners(approx.reshape(4, 2).astype(np.float32))
```

- [ ] **Step 2: Run tests — confirm they pass**

```bash
uv run pytest tests/test_screen_detection.py -v
```

Expected output:
```
test_screen_detection.py::test_detect_corners_returns_array PASSED
test_screen_detection.py::test_detect_corners_ordering PASSED
test_screen_detection.py::test_detect_corners_accuracy PASSED
test_screen_detection.py::test_detect_corners_dark_frame_returns_none PASSED
test_screen_detection.py::test_detect_corners_tiny_bright_region_returns_none PASSED
test_screen_detection.py::test_order_corners_canonical PASSED
6 passed
```

- [ ] **Step 3: Commit**

```bash
git add src/screen_detection.py tests/__init__.py tests/test_screen_detection.py
git commit -m "feat: add iPad screen corner detection via brightness thresholding"
```

---

## Task 3: Debug visualization script

Run detection on real frames and save annotated images to `results/debug_detection/`. No unit tests — this is for visual QC.

**Files:**
- Create: `scripts/debug_screen_detection.py`

- [ ] **Step 1: Write the script**

Create `scripts/debug_screen_detection.py`:

```python
"""Visual QC: run screen detection on real video frames and save annotated images."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.screen_detection import detect_corners

VIDEO  = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
OUT    = Path(__file__).resolve().parent.parent / "results" / "debug_detection"

# Sample timestamps (seconds) spread across the recording
SAMPLE_TIMES = [30, 60, 90, 120, 180, 240, 300, 400, 600, 900]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detected = 0

    for t in SAMPLE_TIMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            print(f"t={t}s: could not read frame")
            continue

        corners = detect_corners(frame)
        vis = frame.copy()

        if corners is not None:
            detected += 1
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            labels = ["TL", "TR", "BR", "BL"]
            for (x, y), label in zip(corners.astype(int), labels):
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(vis, label, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            status = "OK"
        else:
            cv2.putText(vis, "DETECTION FAILED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            status = "FAIL"

        out_path = OUT / f"t{t:04d}s_{status}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"t={t:4d}s: {status}  → {out_path.name}")

    cap.release()
    print(f"\nDetected: {detected}/{len(SAMPLE_TIMES)} frames")
    print(f"Images saved to: {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
uv run python scripts/debug_screen_detection.py
```

Expected: output like:
```
t=  30s: OK  → t0030s_OK.jpg
t=  60s: OK  → t0060s_OK.jpg
...
Detected: 9/10 frames
Images saved to: .../results/debug_detection
```

- [ ] **Step 3: Visually inspect the output images**

Open the images in `results/debug_detection/`. Verify:
- Green quadrilateral outline sits tightly on the iPad screen edges
- Red corner dots land at the actual screen corners
- TL/TR/BR/BL labels are correctly ordered

If detection fails on frames where the laptop is still visible (t=30s), that is acceptable — the analysis will use t>60s. If detection fails on task frames (t≥90s), tune `_THRESH` or `_CLOSE_K` in `src/screen_detection.py`.

**Tuning guide:**
- Detection misses screen (returns None): lower `_THRESH` (try 40) or increase `_CLOSE_K` (try 71)
- Detects wrong region: raise `_THRESH` (try 70) or increase `_MIN_AREA`
- Polygon has ≠4 sides: increase `approxPolyDP` epsilon from `0.02` to `0.03`

- [ ] **Step 4: Commit**

```bash
git add scripts/debug_screen_detection.py
git commit -m "feat: add debug visualization script for screen corner detection"
```
