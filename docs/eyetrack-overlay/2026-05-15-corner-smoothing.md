# Corner Trajectory Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given per-frame raw corner detections (some `None` where detection failed, some with wrong corners due to hand occlusion), produce a complete, smooth corner trajectory for every frame.

**Architecture:** Two-step offline approach. Step 1 — interpolation: stack raw detections into a (n_frames, 4, 2) float array with `NaN` where detection is `None`, then linearly interpolate across NaN gaps and edge-fill any leading/trailing NaN. This handles genuine drop-outs (detection returned None). Step 2 — smoothing: apply a rolling median (not mean) per coordinate series with a ~2-second window. Rolling median ignores outlier wrong detections (wrong-but-not-None frames from hand occlusion) as long as fewer than half the frames in the window are bad — which holds for dot-click events that last <2 s.

**Tech Stack:** Python 3.14, NumPy, pandas (for interpolation and rolling). All commands use `uv run` from the worktree root.

---

## Context

- **Worktree root:** `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/ec348-eyetrack-overlay/`
- **Depends on:** `src/screen_detection.py` (already implemented) — `detect_corners(frame)` returns `np.ndarray | None` with shape (4, 2) float32.
- **Observed failure modes from visual QC:**
  - Most failures are **wrong-but-not-None** (hand occlusion pulls corners inside screen; detect_corners returns something)
  - Occasional **None** (arm fully bisects screen, contour too fragmented)
  - The convex hull fix (already committed) handles mild single-corner occlusion; rolling median handles the rest
- **Video:** `/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4` — ~24.95 fps, 21 min 17 s
- **Output:** `smooth_corners(raw) -> np.ndarray` — shape (n_frames, 4, 2) float32, no NaN, same [TL, TR, BR, BL] order as input.
- **Default window:** 51 frames ≈ 2 s at 25 fps. Dot-click events last ~0.5–1.5 s, so >50% of frames in the window are clean — the median ignores the bad ones.

---

## File Structure

| Path | Purpose |
|------|---------|
| `src/corner_smoother.py` | `smooth_corners(raw, window)` |
| `tests/test_corner_smoother.py` | Unit tests |
| `scripts/debug_smooth_detection.py` | Visual QC: before/after comparison on real video |

---

## Task 1: Write tests

**Files:**
- Create: `tests/test_corner_smoother.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corner_smoother.py`:

```python
import numpy as np
import pytest
from src.corner_smoother import smooth_corners


def _c(val):
    """All 4 corners at (val, val), float32 (4,2)."""
    return np.full((4, 2), val, dtype=np.float32)


GOOD  = _c(500.0)
GOOD2 = _c(505.0)   # slight camera drift
BAD   = _c(0.0)     # totally wrong (hand occlusion failure)


# --- output contract ---

def test_output_shape():
    out = smooth_corners([GOOD] * 20, window=5)
    assert out.shape == (20, 4, 2)
    assert out.dtype == np.float32


def test_no_nans_in_output():
    raw = [GOOD, None, GOOD2, None, GOOD]
    out = smooth_corners(raw, window=3)
    assert not np.any(np.isnan(out))


def test_all_none_raises():
    with pytest.raises(ValueError, match="no valid"):
        smooth_corners([None, None, None], window=3)


# --- interpolation (None gaps) ---

def test_interior_none_interpolated():
    """A single None in the middle is filled by linear interpolation."""
    raw = [_c(100.0)] * 5 + [None] + [_c(100.0)] * 5
    out = smooth_corners(raw, window=1)   # window=1 disables smoothing
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[5], 100.0, atol=1.0)


def test_leading_none_edge_filled():
    """NaN at the start is forward-filled from the first valid value."""
    raw = [None, None] + [_c(200.0)] * 8
    out = smooth_corners(raw, window=1)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[0], 200.0, atol=1.0)
    np.testing.assert_allclose(out[1], 200.0, atol=1.0)


def test_trailing_none_edge_filled():
    """NaN at the end is back-filled from the last valid value."""
    raw = [_c(150.0)] * 8 + [None, None]
    out = smooth_corners(raw, window=1)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[-1], 150.0, atol=1.0)
    np.testing.assert_allclose(out[-2], 150.0, atol=1.0)


# --- rolling median (outlier rejection) ---

def test_outlier_absorbed_by_median():
    """A single wrong detection in a window of good frames is ignored."""
    # 25 good frames with 1 bad frame near the centre
    raw = [GOOD] * 12 + [BAD] + [GOOD] * 12
    out = smooth_corners(raw, window=25)
    # The frame at the bad position should be close to GOOD, not BAD
    np.testing.assert_allclose(out[12], 500.0, atol=10.0)


def test_constant_input_unchanged():
    """Constant signal passes through interpolation + median unchanged."""
    raw = [_c(300.0)] * 50
    out = smooth_corners(raw, window=5)
    np.testing.assert_allclose(out, 300.0, atol=1e-3)


def test_smoothing_reduces_noise():
    """Rolling median reduces per-frame jitter."""
    rng = np.random.default_rng(42)
    base = np.full((4, 2), 500.0, dtype=np.float32)
    raw = [base + rng.normal(0, 20, (4, 2)).astype(np.float32) for _ in range(200)]
    out = smooth_corners(raw, window=25)
    assert out.std() < 10.0, f"Smoothed std {out.std():.1f} too high"


def test_window_1_returns_interpolated_only():
    """Window=1 skips smoothing; Nones are filled but jitter is kept."""
    raw = [GOOD, None, GOOD2]
    out = smooth_corners(raw, window=1)
    assert out.shape == (3, 4, 2)
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out[0], 500.0, atol=1e-3)
    np.testing.assert_allclose(out[2], 505.0, atol=1e-3)
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
uv run pytest tests/test_corner_smoother.py -v
```

Expected: `ImportError` — `src/corner_smoother.py` doesn't exist yet.

---

## Task 2: Implement `smooth_corners`

**Files:**
- Create: `src/corner_smoother.py`

- [ ] **Step 1: Write the implementation**

Create `src/corner_smoother.py`:

```python
"""Smooth per-frame iPad corner detections across time.

Two-step pipeline:
1. Linear interpolation across None (detection-failure) gaps, with edge-fill
   for any leading/trailing gaps.
2. Rolling median per coordinate to absorb outlier wrong-but-not-None
   detections (e.g. hand occlusion). Median is robust as long as <50% of
   frames in the window are bad — which holds for click events < 2 s.
"""

import numpy as np
import pandas as pd


def smooth_corners(
    raw: list,
    window: int = 51,
) -> np.ndarray:
    """Smooth a sequence of per-frame corner detections.

    Args:
        raw: List of length n_frames. Each element is a float32 array of
            shape (4, 2) from detect_corners(), or None.
        window: Rolling median window in frames. Default 51 ≈ 2 s at 25 fps.

    Returns:
        float32 array of shape (n_frames, 4, 2). No NaN. Corners are in the
        same [TL, TR, BR, BL] order as detect_corners() output.

    Raises:
        ValueError: If every frame is None (no valid detection anywhere).
    """
    n = len(raw)
    if n == 0:
        return np.empty((0, 4, 2), dtype=np.float32)

    # Build (n, 4, 2) with NaN where detection failed
    stack = np.full((n, 4, 2), np.nan, dtype=np.float32)
    for i, corners in enumerate(raw):
        if corners is not None:
            stack[i] = corners

    if np.all(np.isnan(stack)):
        raise ValueError("smooth_corners: no valid detections in input (all None)")

    # Reshape to (n, 8) — treat each of the 8 scalar series independently
    flat = stack.reshape(n, 8)

    # Step 1: interpolate NaN gaps, edge-fill leading/trailing NaN
    df = pd.DataFrame(flat)
    df = df.interpolate(method="linear", axis=0).ffill().bfill()
    interpolated = df.to_numpy(dtype=np.float32)

    # Step 2: rolling median to absorb outlier wrong detections
    # min_periods=1 avoids NaN at edges; center=True uses symmetric window
    df2 = pd.DataFrame(interpolated)
    smoothed = (
        df2.rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float32)
    )

    return smoothed.reshape(n, 4, 2)
```

- [ ] **Step 2: Run tests — confirm all pass**

```bash
uv run pytest tests/test_corner_smoother.py -v
```

Expected:
```
test_corner_smoother.py::test_output_shape PASSED
test_corner_smoother.py::test_no_nans_in_output PASSED
test_corner_smoother.py::test_all_none_raises PASSED
test_corner_smoother.py::test_interior_none_interpolated PASSED
test_corner_smoother.py::test_leading_none_edge_filled PASSED
test_corner_smoother.py::test_trailing_none_edge_filled PASSED
test_corner_smoother.py::test_outlier_absorbed_by_median PASSED
test_corner_smoother.py::test_constant_input_unchanged PASSED
test_corner_smoother.py::test_smoothing_reduces_noise PASSED
test_corner_smoother.py::test_window_1_returns_interpolated_only PASSED
10 passed
```

- [ ] **Step 3: Commit**

```bash
git add src/corner_smoother.py tests/test_corner_smoother.py
git commit -m "feat: add corner trajectory smoothing with interpolation + rolling median"
```

---

## Task 3: QC script — dense detection + before/after comparison

Run detection on every frame from t=0–500 s (covers all original QC timestamps), apply smoothing, save annotated images showing raw (dashed red) vs smoothed (solid green).

**Files:**
- Create: `scripts/debug_smooth_detection.py`

- [ ] **Step 1: Write the script**

Create `scripts/debug_smooth_detection.py`:

```python
"""QC: compare raw vs smoothed corner detection at known-bad timestamps.

Runs detect_corners on every frame from T_START to T_END, applies
smooth_corners, then saves annotated images at SAMPLE_TIMES showing:
  - dashed red  = raw single-frame detection
  - solid green = smoothed corners
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.screen_detection import detect_corners
from src.corner_smoother import smooth_corners

VIDEO = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
OUT   = Path(__file__).resolve().parent.parent / "results" / "debug_smoothing"

T_START = 0    # seconds — start of dense detection window
T_END   = 500  # seconds — covers all original QC timestamps up to t=400 s

# Timestamps to visualise (subset of original debug_screen_detection times)
SAMPLE_TIMES = [30, 60, 90, 120, 180, 240, 300, 400]

SMOOTH_WINDOW = 51  # frames (~2 s at 25 fps)


def _draw_quad(img, corners, color, thickness, dashed=False):
    pts = corners.astype(np.int32)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        if dashed:
            for s in range(0, 10, 2):
                a = (int(p1[0] + (p2[0] - p1[0]) * s / 10),
                     int(p1[1] + (p2[1] - p1[1]) * s / 10))
                b = (int(p1[0] + (p2[0] - p1[0]) * (s + 1) / 10),
                     int(p1[1] + (p2[1] - p1[1]) * (s + 1) / 10))
                cv2.line(img, a, b, color, thickness)
        else:
            cv2.line(img, p1, p2, color, thickness)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_start = int(T_START * fps)
    frame_end   = int(T_END   * fps)
    n_frames    = frame_end - frame_start

    print(f"Extracting detections: t={T_START}–{T_END}s ({n_frames} frames) …")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    abs_frames: list[int] = []
    raw_detections: list[np.ndarray | None] = []

    for idx in range(n_frames):
        abs_frames.append(frame_start + idx)
        ret, frame = cap.read()
        if not ret:
            raw_detections.append(None)
            continue
        raw_detections.append(detect_corners(frame))
        if idx % 1000 == 0:
            print(f"  {idx}/{n_frames} ({100*idx/n_frames:.0f}%)")

    n_detected = sum(1 for d in raw_detections if d is not None)
    print(f"Raw detection rate: {n_detected}/{n_frames} ({100*n_detected/n_frames:.1f}%)")

    print("Smoothing …")
    smoothed_arr = smooth_corners(raw_detections, window=SMOOTH_WINDOW)  # (n, 4, 2)

    # Map absolute frame number → (raw, smoothed)
    lookup = {
        abs_frames[i]: (raw_detections[i], smoothed_arr[i])
        for i in range(n_frames)
    }

    print("Saving annotated images …")
    for t in SAMPLE_TIMES:
        abs_f = int(t * fps)
        if abs_f not in lookup:
            print(f"t={t}s: out of range, skipping")
            continue

        raw_c, sm_c = lookup[abs_f]

        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_f)
        ret, frame = cap.read()
        if not ret:
            print(f"t={t}s: could not re-read frame")
            continue

        vis = frame.copy()

        if raw_c is not None:
            _draw_quad(vis, raw_c, (0, 0, 255), thickness=2, dashed=True)
            for x, y in raw_c.astype(int):
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        _draw_quad(vis, sm_c, (0, 255, 0), thickness=3, dashed=False)
        for (x, y), label in zip(sm_c.astype(int), ["TL", "TR", "BR", "BL"]):
            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        raw_tag = "ok" if raw_c is not None else "none"
        cv2.putText(vis, f"raw={raw_tag}  smoothed=ok", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        out_path = OUT / f"t{t:04d}s.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"t={t:4d}s: raw={raw_tag:<4s}  → {out_path.name}")

    cap.release()
    print(f"\nImages saved to: {OUT}")
    print("Legend: dashed red = raw, solid green = smoothed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
uv run python scripts/debug_smooth_detection.py
```

Expected: ~3–5 min runtime for ~12,500 frames. Output:
```
Raw detection rate: 11800/12480 (94.6%)
...
t=  90s: raw=ok   → t0090s.jpg
t= 120s: raw=ok   → t0120s.jpg
...
Images saved to: .../results/debug_smoothing
```

- [ ] **Step 3: Visually inspect the output**

Open images in `results/debug_smoothing/`. For each timestamp:
- **Dashed red** = raw single-frame detection (what we had before)
- **Solid green** = smoothed corners

Success criteria: at known-bad timestamps (t=90, 120, 180, 240, 300s), the green quad sits accurately on the actual screen edges, even when the red quad is wrong or absent.

If the green quad is still wrong at most timestamps, the window is too small relative to how long the hand stays on screen. Increase `SMOOTH_WINDOW = 101` (~4 s) and re-run. Also increase it in `src/corner_smoother.py`'s default if you change it here.

- [ ] **Step 4: Commit**

```bash
git add scripts/debug_smooth_detection.py
git commit -m "feat: add QC script for raw vs smoothed corner comparison"
```

---

## Task 4: Integration smoke test

Confirm the full pipeline (detect + smooth) runs end-to-end on a real clip and produces a NaN-free output array.

- [ ] **Step 1: Run the smoke test**

```bash
uv run python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import cv2
import numpy as np
from src.screen_detection import detect_corners
from src.corner_smoother import smooth_corners

VIDEO = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4"
START_S    = 120.0
DURATION_S = 30.0

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_S * fps))
n = int(DURATION_S * fps)

raw = []
for _ in range(n):
    ret, frame = cap.read()
    raw.append(detect_corners(frame) if ret else None)
cap.release()

n_det = sum(1 for c in raw if c is not None)
print(f"Detected:      {n_det}/{n} frames ({n_det/n*100:.1f}%)")

out = smooth_corners(raw, window=51)
print(f"Output shape:  {out.shape}")
print(f"NaN count:     {int(np.isnan(out).sum())}")
print(f"TL x range:    {out[:,0,0].min():.0f}–{out[:,0,0].max():.0f}")
print(f"TL y range:    {out[:,0,1].min():.0f}–{out[:,0,1].max():.0f}")
import numpy as np
EOF
```

Expected (values approximate):
```
Detected:      710/748 frames (94.9%)
Output shape:  (748, 4, 2)
NaN count:     0
TL x range:    590–660
TL y range:    340–410
```
