# Eyetracking Pipeline Outputs

Reference for all files produced by `Snakefile_eyetrack`. All paths are relative to the repo root; `{subject}` is the subject ID (e.g. `EC347`).

---

## Coordinate systems

| Space | Origin | Extent | Notes |
|---|---|---|---|
| **video** | (0,0) top-left | 1920 × 1080 px | Tobii scene-camera frame |
| **screen** | (0,0) top-left | 2388 × 1668 px | iPad 11-in Pro M2, landscape |
| **canvas** | (0,0) top-left of content area | 1922 × 1396 px | `gx_canvas = gx_screen − CANVAS_X_PAD_PX (233)`, `gy_canvas = gy_screen − URL_BAR_H_PX (272)` |

Video → screen projection: per-sample bilinearly-interpolated inverse homography `H` (computed in `phase1c_homography`, applied in `extract_gaze_fixations`). Source: `src/gaze_projection.py`.

---

## Primary data outputs

### `results/{subject}/eyetrack/gaze_per_sample.parquet`

One row per Tobii Eye Tracker sample (~100 Hz). Non-Eye-Tracker rows (Gyro, Accel) are excluded.

| Column | dtype | Description |
|---|---|---|
| `tobii_ts_us` | int64 | Tobii recording timestamp (µs from recording start) |
| `behavior_t_ms` | float64 | Task clock time (ms) via video-alignment affine |
| `video_frame_frac` | float64 | Fractional video frame index |
| `gx_video` | float64 | Gaze X in video px (raw from Tobii) |
| `gy_video` | float64 | Gaze Y in video px |
| `gx_screen` | float64 | Projected gaze X in iPad screen px; NaN if `gaze_valid` or `homography_valid` is False |
| `gy_screen` | float64 | Projected gaze Y in iPad screen px; NaN under same conditions |
| `gx_canvas` | float64 | Gaze X in behavior canvas px |
| `gy_canvas` | float64 | Gaze Y in behavior canvas px |
| `gaze_valid` | bool | Both eyes Valid AND gaze not NaN AND em_type ≠ EyesNotFound |
| `homography_valid` | bool | Both flanking per-frame Hs are non-NaN |
| `on_screen` | bool | `gx_screen`/`gy_screen` within iPad screen bounds |
| `em_type` | string | `Fixation` / `Saccade` / `EyesNotFound` / `Unclassified` |
| `em_type_index` | int64 | Event index within em_type run (−1 if not available) |
| `em_duration_ms` | int64 | Raw Tobii `Eye movement event duration` value (units: µs despite `_ms` suffix; −1 if not available) |
| `fixation_x_video` | float64 | Tobii I-VT fixation centroid X in video px (NaN if not a Fixation row) |
| `fixation_y_video` | float64 | Tobii I-VT fixation centroid Y in video px |

**Source:** `notebooks/extract_gaze_fixations.py` → `rule extract_gaze_fixations`

---

### `results/{subject}/eyetrack/fixation_events.parquet`

One row per Tobii fixation event (collapsed from `gaze_per_sample` rows where `em_type == "Fixation"`).

| Column | dtype | Description |
|---|---|---|
| `event_idx` | int | `em_type_index` of this fixation event |
| `start_behavior_t_ms` | float64 | Start of event on task clock (ms) |
| `end_behavior_t_ms` | float64 | End of event on task clock (ms) |
| `duration_ms` | int | Duration from Tobii classifier (µs raw value; see note on `em_duration_ms` above) |
| `n_samples` | int | Number of Tobii samples in this event |
| `centroid_video_x` | float64 | Mean gaze X in video px (over `gaze_valid` samples) |
| `centroid_video_y` | float64 | Mean gaze Y in video px |
| `centroid_screen_x` | float64 | Mean projected X in screen px (over `gaze_valid & homography_valid` samples) |
| `centroid_screen_y` | float64 | Mean projected Y in screen px |
| `centroid_canvas_x` | float64 | Mean X in canvas px |
| `centroid_canvas_y` | float64 | Mean Y in canvas px |
| `frac_homography_valid` | float64 | Fraction of samples with valid homography |
| `frac_on_screen` | float64 | Fraction of samples with gaze on screen |

**Source:** `notebooks/extract_gaze_fixations.py` → `rule extract_gaze_fixations`

---

### `results/{subject}/phase1c_per_frame.parquet`

Per-video-frame homography matrices (3×3) used to project gaze. One row per frame.

**Source:** `notebooks/phase1c_eval.py` → `rule phase1c_homography`

### `results/{subject}/phase1c_calibration_used.json`

The box-calibration parameters selected for Phase 1c (which big-star anchor set, smoothing window, etc.).

---

## Visualizations from `extract_gaze_fixations`

All written to `results/{subject}/eyetrack/`.

### `pre_click_gaze_trajectories.png`

4×5 grid of 20 randomly-sampled click events (seed=0). Each panel shows the behavior canvas (1922×1396 px), previously-revealed dots in the trial, the target dot (gold circle), the gaze trajectory in the 1-second pre-click window (viridis color: purple = −1 s, yellow = 0 s), and the click location (blue ×). Only clicks with ≥10 valid+homography-valid samples in the window are eligible.

**This plot is part of the pipeline** — produced by `rule extract_gaze_fixations`.

### `gaze_coverage_and_accuracy.png`

Two-panel figure:
- **Left:** Time series of `gaze_valid`, `homography_valid`, and `on_screen` rates in 10-second bins across the full session.
- **Right:** Histogram of pre-click gaze distance to the target dot in canvas px, with median annotated and a 250-px threshold line.

### `gaze_canvas_heatmap.png`

3-panel log-scaled fixation-centroid heatmap in canvas coordinates, one panel per trial phase: pre-reveal / reveal-to-click / post-click.

### `saccade_psth_around_reveal.png`

Histogram of saccade-onset times relative to dot-reveal events (±2 s window, 50 ms bins).

---

## Video outputs

### `results/{subject}/eyetrack/gaze_screen_overlay.mp4`

30-second clip of scene video with overlays:
- Cyan polygon: projected screen boundary
- Cyan filled circles: previously-revealed behavior dots projected back to video
- White circle (red ring if invalid): raw Tobii gaze dot

Clip window: `clip_start_s` to `clip_start_s + clip_duration_s` from `config_eyetrack.yaml` (default 120–150 s). Codec: `mp4v` via `cv2.VideoWriter`.

**Source:** `notebooks/extract_gaze_fixations.py` → `rule extract_gaze_fixations`

### `results/{subject}/eyetrack/screen_gaze_overlay.mp4`

Side-by-side 1920×540 video:
- **Left panel (960×540):** Original frame at 50% scale, with detected screen outline (green polygon) and gaze dot overlay.
- **Right panel (960×540):** Perspective-rectified iPad screen view via `warpPerspective`, with gaze dot in panel coordinates.

Gaze dot appearance: fixation = red filled circle (r=15) with white outline; saccade = small dim-blue circle (r=7). Codec: h264/yuv420p via ffmpeg pipe.

**Source:** `scripts/render_screen_gaze.py` → `rule render_screen_gaze`

---

## Homography diagnostics

### `results/{subject}/homography_eval/homography_box_calibration.json`

Fitted box-to-screen homography calibration from hand-labeled correspondences.

### `results/{subject}/homography_eval/homography_per_frame.parquet`

Raw per-frame homography estimates before Phase 1c smoothing/refit.

### `results/{subject}/cascade_trajectory.png`

Two-panel plot of box-corner and screen-corner Y positions vs. frame index (full video + zoomed snap-onset region), with hand-label ground truth overlaid. Diagnostic for Phase 1c corner tracking quality.

### `results/{subject}/big_star_residual_hist.png` / `big_star_residual_vs_frame.png`

Histogram and scatter plot of big-star anchor back-projected residuals (px). Used to assess homography accuracy.

---

## Rule dependency graph

```
detect_screen_corners ──┐
homography_solver ───────┤
                         ▼
                  phase1c_homography ──► extract_gaze_fixations
                                                │
                                                ├── gaze_per_sample.parquet
                                                ├── fixation_events.parquet
                                                ├── pre_click_gaze_trajectories.png
                                                ├── gaze_coverage_and_accuracy.png
                                                ├── gaze_canvas_heatmap.png
                                                ├── saccade_psth_around_reveal.png
                                                └── gaze_screen_overlay.mp4

render_screen_gaze ─────────────────────────── screen_gaze_overlay.mp4
```

The `rule all` target requires `screen_gaze_overlay.mp4`, `gaze_per_sample.parquet`, and `fixation_events.parquet`.
