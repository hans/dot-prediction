# Phase 1c — automated box corner detector + full-video homography

## Context and motivation

The Phase-1a corner detector is reliable for the **bottom-left and bottom-right
iPad screen corners** (`screen_bl`, `screen_br`) but not for TL/TR (occluded by
the photodiode device). The homography solver (Phase handoff) showed that a
4-correspondence H using `screen_bl`, `screen_br`, `box_bl`, `box_br` is
accurate and stable (median big-star residual well under the 20 px escalation
threshold).

Phase 1c closes the loop: **automate `box_bl` and `box_br` detection** so that a
4-correspondence `H_screen→frame` can be produced for every frame in the video,
not just the 26 hand-labeled ones.

The downstream consumer is the eyetracking gaze-projection pipeline
(`extract_gaze_fixations`).

---

## Canonical feature definition

The canonical feature for both box corners is **the bottom edge of the
photodiode device, where the casing contacts the iPad screen surface** — i.e.,
the topmost dark→bright transition directly below the device body.

| corner | frame-coordinate meaning |
|---|---|
| `box_bl` | left endpoint of the bottom edge (dark→bright, leftmost) |
| `box_br` | right endpoint of the bottom edge (dark→bright, rightmost) |

The surrounding visual context is consistent across the recording:

| direction from device | appearance |
|---|---|
| above | dark (iPad case) |
| left | dark (iPad case) |
| below | bright (iPad screen) |
| right | bright (iPad screen, vertical strip) |

---

## Empirical characterisation (26-frame preflight)

Before fixing the detection approach, the detector was validated against all 26
hand-labeled frames with `box_bl` and `box_br` confident and visible.

**Harris argmax in a symmetric ±40 px window is broken.**
For `box_br`, Harris finds the wrong corner ~40 px away in most frames (a corner
high in the device body rather than the bottom-right edge). For `box_bl`, it
finds a corner 10–14 px above the true feature in 19/26 frames. The 80 px
window spec'd in the first draft was an untested assumption and is replaced below.

**Asymmetric windows fix the common case.**

After switching to asymmetric windows that exclude the competing-corner region:

| detector | window | median error | max error | >5 px frames |
|---|---|---|---|---|
| `box_bl` asymmetric | x ±20, y [−5, +20] | **3.7 px** | 17.7 px | 9/26 |
| `box_br` symmetric | x ±20, y ±20 | **3.9 px** | 22.6 px | 7/26 |

**Outlier frames.** A shared set of ~9 frames (f2288, f9124, f10562, f19671,
f20192, f30125, f30135, f30175, f30950) have errors of 13–22 px for one or both
corners. Visual inspection shows Harris finds an internal device corner instead
of the true edge endpoint. These frames are scattered (not consecutive), so the
interpolation step covers them without requiring explicit fixes. During
implementation, print a per-frame detection error table against the 26 labeled
frames to verify no new failure mode has emerged.

**Label-precision note.** In the outlier frames, the Harris-detected position
sits at the visible dark→bright transition while several labels appear to be
placed a few pixels into the dark region above it. If the per-frame detection
table in the implementation notebook shows systematic bias (detected always
below the label by a fixed offset), re-examine those labels in
`notebooks/label_homography_correspondences.ipynb` and move them to the exact
dark→bright transition.

---

## Inputs

| file | description |
|---|---|
| `data/{subject}/tobii/scenevideo.mp4` | raw scene video |
| `results/{subject}/homography_labels.parquet` | hand labels (seed + validation) |
| `results/{subject}/homography_eval/homography_box_calibration.json` | calibrated box corner screen coords |
| `results/{subject}/screen_corners.parquet` | per-frame `screen_bl`, `screen_br` — see below |

### Phase-1a screen corners parquet (prerequisite)

A new Snakemake rule `detect_screen_corners` must produce
`results/{subject}/screen_corners.parquet` **before** Phase 1c can run. This
rule applies the smoothed corner detector (Phase 1a + `corner_smoother`) to the
full video and saves only BL and BR (TL/TR are discarded):

| column | type | description |
|---|---|---|
| `frame_idx` | int | 0 … N−1 |
| `screen_bl_x`, `screen_bl_y` | float64 | smoothed BL frame coords; NaN if detection failed |
| `screen_br_x`, `screen_br_y` | float64 | smoothed BR frame coords; NaN if detection failed |

NaN rows where Phase 1a returned no contour (screen not visible or not detectable)
must be distinguished from frames where Phase 1a ran but BL/BR were missing.
Add a `no_screen` bool column: `True` when `detect_corners` returned `None`;
`False` (but NaN xy) only when the contour was found but couldn't be assigned
a corner.

This parquet is an explicit dependency of the Phase-1c notebook rule.

---

## Algorithm

### Bootstrap

Seed from **frame 664** (earliest labeled frame with all 4 anchor labels confident
and visible). Use the hand-labeled positions directly — no detector for this
frame.

**Backward pass**: after processing frames ≥ 664 forward, run the identical loop
in reverse from 663 to 0, seeding from the frame-664 H. This covers the ~26 s
of video before the first labeled frame.

### Per-frame detection loop

For each frame `t` (forward and backward passes):

**Step A — predict box corner positions**

```python
H_prev = H from most recent successfully-detected frame (initially = seed H)
box_bl_pred = perspective_transform(H_prev, box_bl_screen_calibrated)
box_br_pred = perspective_transform(H_prev, box_br_screen_calibrated)
```

**Step B — detect box corners locally**

```python
box_bl_det = detect_box_corner(frame, box_bl_pred, corner="bl")
box_br_det = detect_box_corner(frame, box_br_pred, corner="br")
```

See module spec below. Returns a frame-xy or `None` if the Harris response
is below `min_harris_response` after normalisation.

**Step C — check screen corners**

Look up `screen_corners.parquet` for `frame_idx == t`.

If `no_screen is True` → set `detection_status = "no_screen"`. Do **not**
update `H_prev`. Do **not** emit an H for this frame.

If `screen_bl` or `screen_br` is NaN → `detection_status = "missing_screen_bl"`
(or `_br`).

**Step D — fit H**

If all four anchors are available:

```python
screen_pts = [screen_BL_coords, screen_BR_coords,
              box_bl_screen_calibrated, box_br_screen_calibrated]
frame_pts  = [screen_bl_det, screen_br_det, box_bl_det, box_br_det]
H, _ = cv2.findHomography(np.array(screen_pts, dtype=np.float64),
                          np.array(frame_pts,  dtype=np.float64))
```

Update `H_prev = H`. Set `detection_status = "detected"`.

If any anchor is `None` or NaN: set `detection_status = "missing_<name>"`.
Do **not** update `H_prev`.

### Interpolation pass

After both forward and backward passes, `detection_status` is one of:
`"detected"`, `"missing_*"`, or `"no_screen"`.

For each contiguous run of non-detected, non-`no_screen` frames flanked by
detected frames on both sides, linearly interpolate each of the 8 non-trivial H
elements (h22=1 by construction):

```python
alpha = (t - t_left) / (t_right - t_left)   # 0 < alpha < 1
H_t[i,j] = (1 - alpha) * H_left[i,j] + alpha * H_right[i,j]
```

Set `detection_status = "interpolated"` for these rows.

For runs at the leading or trailing edge (no flanking detected frame on one
side), fill with the nearest detected H and set `detection_status =
"extrapolated"`.

`no_screen` frames are **never** filled — they retain NaN H elements.

---

## New module: `src/box_corner_detector.py`

```python
def detect_box_corner(
    frame: np.ndarray,
    predicted_xy: tuple[float, float],
    corner: str,                   # "bl" or "br"
    x_half: int = 20,
    y_above: int = 5,
    y_below: int = 20,
    min_harris_response: float = 0.05,
) -> tuple[float, float] | None:
    """Detect one box corner in an asymmetric local search window.

    The window is shifted downward relative to the prediction (y_above < y_below)
    to exclude a competing strong corner that exists ~10 px above the true
    device-bottom feature. See spec § Empirical characterisation for details.

    Args:
        frame: BGR image, uint8, shape (H, W, 3).
        predicted_xy: Expected (x, y) in frame pixel coordinates.
        corner: "bl" or "br" — reserved for future per-corner window tuning.
        x_half: Half-width of the search window (symmetric in x).
        y_above: Pixels above predicted_xy included in window.
        y_below: Pixels below predicted_xy included in window.
        min_harris_response: Minimum normalised Harris response to accept
            (0–1 after dividing by the patch maximum). Rejects uniform patches.

    Returns:
        (x, y) in frame pixel coordinates, or None.
    """
```

Implementation:
1. Compute clamped window `[cx−x_half : cx+x_half, cy−y_above : cy+y_below]`.
   Return `None` if window is fully outside frame.
2. Convert to grayscale.
3. `harris = cv2.cornerHarris(gray.astype(np.float32), blockSize=3, ksize=3, k=0.04)`
4. Normalise: `harris /= (harris.max() + 1e-10)`.
5. `py, px = np.unravel_index(harris.argmax(), harris.shape)`.
6. If `harris[py, px] < min_harris_response`: return `None`.
7. Return `(float(x0 + px), float(y0 + py))`.

No subpixel refinement in this version.

---

## Tests: `tests/test_box_corner_detector.py`

- **Synthetic corner, centred prediction**: 200×200 image with a sharp dark
  upper-left / bright lower-right corner at (100, 100). Predict (100, 100).
  Assert detected within 2 px.
- **Offset prediction (±15 px in both axes)**: Same synthetic image, predict at
  (85, 85) then (115, 115). Assert detected within 2 px of (100, 100) with the
  default asymmetric window.
- **Uniform patch** (no corner): Assert returns `None`.
- **Prediction outside frame**: Assert returns `None`.
- **`corner="bl"` vs `"br"`**: Both accept valid synthetic corners without error
  (hook for future per-corner tuning).

---

## Output: `results/{subject}/phase1c_per_frame.parquet`

One row per frame (frame_idx 0 … N−1, every integer).

| column | type | description |
|---|---|---|
| `frame_idx` | int | |
| `h00`–`h22` | float64 | 9 H elements, row-major, h22=1; NaN for `no_screen` frames |
| `detection_status` | str | source of the emitted H: `"detected"` / `"interpolated"` / `"extrapolated"` / `"no_screen"` |
| `detection_reason` | str | empty when `detected`; else `"missing_screen_bl"` / `"missing_box_br"` / `"no_screen"` / etc. |
| `screen_bl_x`, `screen_bl_y` | float64 | frame-xy used in fit; NaN if missing |
| `screen_br_x`, `screen_br_y` | float64 | same |
| `box_bl_x`, `box_bl_y` | float64 | same |
| `box_br_x`, `box_br_y` | float64 | same |
| `big_star_residual_px` | float64 | held-out validation against hand labels only (NaN for non-labeled frames) |

`big_star_residual_px` is populated **only** for the 26 hand-labeled frames
with `big_star` visible and confident. It is **not** populated using Phase-1b
auto-detections — that would conflate two different signals.

### `results/{subject}/phase1c_calibration_used.json`

Copy of the calibrated box corner screen coords used, for provenance.

---

## Validation notebook: `notebooks/phase1c_eval.py`

### Preflight: per-frame detection error against hand labels

Before running on the full video, run the detector on all 26 labeled frames
using the **true label positions as the prediction** (zero prediction error).
Print a table:

| frame_idx | box_bl_error_px | box_br_error_px | note |
|---|---|---|---|
| 664 | … | … | seed frame |
| … | | | |

Flag any frame with error > 8 px. If the pattern matches the 9 outliers
documented in § Empirical characterisation above and no new failures appear,
proceed to full-video run. If new failures appear, investigate before
continuing.

### Full-video detection summary

Print detection-status counts and rates:

```
detected:      N  (XX.X%)
interpolated:  N  (XX.X%)
extrapolated:  N  (XX.X%)
no_screen:     N  (XX.X%)
missing_*:     N  (XX.X% — broken down by anchor)
```

Flag if `detected` rate < 85% of non-`no_screen` frames.

### big_star held-out residual

Reproduce the `big_star_residual_hist.png` from the homography-solver
evaluation, now using Phase-1c H estimates for the same 19 validation frames.
Median should be comparable to the hand-labeled solver. If markedly worse,
the box corner detector has systematic bias.

### Spot-check overlays (same 4 frames as homography solver: 664, 1550, 2288, 30125)

Same overlay format as the homography-solver eval. Add to the caption:
`detection_status=detected/interpolated`.

---

## Snakemake rules

```python
rule detect_screen_corners:
    input:
        video="data/{subject}/tobii/scenevideo.mp4",
    output:
        parquet="results/{subject}/screen_corners.parquet",
    script:
        "scripts/detect_screen_corners.py"


rule phase1c_homography:
    input:
        video="data/{subject}/tobii/scenevideo.mp4",
        labels="results/{subject}/homography_labels.parquet",
        calibration="results/{subject}/homography_eval/homography_box_calibration.json",
        screen_corners="results/{subject}/screen_corners.parquet",
        trials="results/{subject}/trials_with_video.parquet",
        notebook="notebooks/phase1c_eval.py",
    output:
        per_frame="results/{subject}/phase1c_per_frame.parquet",
        calibration_used="results/{subject}/phase1c_calibration_used.json",
        notebook="results/{subject}/notebooks/phase1c_eval.ipynb",
    run:
        run_notebook(
            input.notebook,
            output.notebook,
            parameters=dict(
                subject=wildcards.subject,
                video_path=input.video,
                labels_path=input.labels,
                calibration_path=input.calibration,
                screen_corners_path=input.screen_corners,
                trials_path=input.trials,
                out_per_frame=output.per_frame,
                out_calibration=output.calibration_used,
            ),
        )
```

Add `phase1c_homography` outputs to `rule all` once the downstream
`extract_gaze_fixations` rule is ready to consume them.

---

## Out of scope

- **TL / TR detection.** BL + BR + box_bl + box_br = 4 correspondences → fully
  determined H. TL/TR are not needed.
- **big_star as a 5th correspondence.** Held out for validation.
- **Joint per-frame H + box position re-optimization.** Out of scope unless
  Phase-1c big_star residuals are markedly worse than the hand-labeled solver.
- **Subpixel corner refinement (`cornerSubPix`).** Add if residuals warrant.
- **Subjects other than EC347.** Functions parameterised by subject.

---

## Open questions (flag in notebook output if encountered)

1. **Are the 9 outlier frames actually mis-labeled?** The preflight table will
   show. If detected position is consistently a few px below the label for those
   frames, the labels are in the dark region rather than at the dark→bright
   transition. Fix: re-open the labeling notebook and move those labels to the
   exact transition point.
2. **Is `min_harris_response = 0.05` well-calibrated?** Print the distribution
   of normalised Harris responses across all detected frames. If many cluster
   near 0.05, tune it.
3. **Does big_star residual correlate with detection_status?** If interpolated
   frames have markedly higher residual than detected frames, consider a larger
   search window as a recovery pass for those frames.
4. **Does detection rate correlate with head position / time?** If failures
   cluster late in the session (where the device angle may have shifted), a
   per-segment calibration might be needed.
