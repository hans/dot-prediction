# Eyetracking Pipeline Architecture

High-level overview of `Snakefile_eyetrack`. See `pipeline-outputs.md` for all output file schemas.

---

## Goal

Map every Tobii gaze sample (100 Hz, video-pixel coordinates) to iPad screen coordinates (2388×1668 px) and behavior canvas coordinates, using a per-frame perspective homography derived from the scene video.

---

## Inputs

| File | Source | Description |
|---|---|---|
| `{tobii_dir}/scenevideo.mp4` | Tobii export | 1920×1080, ~24.95 fps |
| `{tobii_dir}/{subject}_tobii.tsv` | Tobii export | Raw gaze + fixation/saccade events at ~100 Hz |
| `results/{subject}/homography_labels.parquet` | Hand-labeled | Manually annotated video-px ↔ screen-px correspondences |
| `results/{subject}/trials_with_video.parquet` | `align_behavior` | Behavior trial table with video-frame alignment |
| `results/{subject}/video_alignment.json` | `align_video` notebook | Affine mapping Tobii timestamps → behavior task clock |

---

## Rule chain

```
[scene video]──────────────────────────────────────────────┐
                                                            │
                     detect_screen_corners ─────────────────┤
                                                            │
[hand labels] ──── homography_solver ──────────────────────┤
[trials]                                                    ▼
                                              phase1c_homography
                                                            │
[Tobii TSV] ──────────────────────────────────────────────►│
[video_alignment.json] ────────────────────────────────────►│
                                                            ▼
                                              extract_gaze_fixations
                                           (gaze_per_sample, fixation_events,
                                            all diagnostic plots + video)

[scene video] ─── render_screen_gaze ──────────────────────►  screen_gaze_overlay.mp4
[Tobii TSV] ──────────────────────────────────────────────►   (independent QC video)
```

`rule all` requires: `screen_gaze_overlay.mp4`, `gaze_per_sample.parquet`, `fixation_events.parquet`.

---

## Stage-by-stage description

### 1. `detect_screen_corners`

**Script:** `scripts/detect_screen_corners.py` → `src/screen_detection.py`

Runs a corner detector over every frame of the scene video to find the four corners of the iPad screen. Returns a parquet with one row per frame and columns `[tl_x, tl_y, tr_x, ...]`; rows where detection fails are NaN.

Detection uses a two-step approach: edge detection followed by `src/screen_detection.py:detect_corners`, which applies a color prior (`src/box_corner_detector.py`) and non-maximum suppression to resolve ambiguous candidates.

**Output:** `results/{subject}/screen_corners.parquet`

---

### 2. `homography_solver`

**Notebook:** `notebooks/homography_eval.py` → `src/homography_solver.py`

Fits a box-calibration homography from hand-labeled correspondences (`homography_labels.parquet`). The calibration maps box-corner positions (in the physical rig coordinate system) to screen pixel positions. With the calibration and the per-frame detected corners it computes a raw per-frame homography series `homography_per_frame.parquet`.

Key library functions:
- `homography_solver.calibrate_box_position` — fits box-to-screen affine from labels
- `homography_solver.fit_per_frame_homography` — applies calibration to detected corners frame by frame
- `homography_solver.big_star_residuals` — back-projects big-star anchors to assess accuracy

**Outputs:** `homography_box_calibration.json`, `homography_per_frame.parquet`

---

### 3. `phase1c_homography`

**Notebook:** `notebooks/phase1c_eval.py` → `src/gaze_projection.py`, `src/homography_refinement.py`

The core homography estimation stage. Produces the smooth per-frame `H` matrices used by gaze projection.

Two-path cascade:
1. **Box corners** (when visible): raw H from `homography_solver` stage.
2. **Screen corners** (`screen_corners.parquet`): fills gaps when the box is occluded.

After combining the two paths, a rolling-median smoother (`src/corner_smoother.py:smooth_corners`) removes jitter. An anchor-refit pass (`src/gaze_projection.py:smooth_anchors_then_refit`) then re-fits Hs using locally-detected big-star positions rather than just corner estimates, reducing back-projection error.

The notebook also produces three diagnostic plots to assess the cascade:
- `cascade_trajectory.png` — corner Y-positions vs. frame (box path vs. screen path, with ground truth)
- `big_star_residual_hist.png` / `big_star_residual_vs_frame.png` — back-projection residuals of big-star anchors

**Outputs:** `phase1c_per_frame.parquet`, `phase1c_calibration_used.json`, diagnostic PNGs

---

### 4. `extract_gaze_fixations`

**Notebook:** `notebooks/extract_gaze_fixations.py` → `src/gaze_projection.py`

The main projection stage. Projects every Tobii Eye Tracker sample to screen and canvas coordinates, collapses fixation events, and generates all downstream visualizations.

**Projection pipeline (vectorised):**

```
Tobii timestamp (µs)
       │
       ▼  tobii_ts_to_video_frame_frac()
fractional frame index
       │
       ▼  bilinear interpolation of per-frame H matrices
per-sample H  (N × 3 × 3)
       │
       ▼  project_video_to_screen()   [batched cv2.perspectiveTransform]
gx_screen, gy_screen  (screen px)
       │
       ▼  screen_to_canvas()
gx_canvas, gy_canvas  (canvas px)
```

Validity gating:
- `gaze_valid`: both Tobii eyes Valid AND gaze not NaN AND em_type ≠ EyesNotFound
- `homography_valid`: both flanking integer-frame Hs are non-NaN (i.e. the frame had a good detection)
- `gx_screen`/`gy_screen` are set to NaN unless both flags are true

The Tobii I-VT fixation classifier output (`em_type`, `em_type_index`, fixation centroid columns) is carried through unchanged; no re-classification is done at this stage.

**Outputs:** `gaze_per_sample.parquet`, `fixation_events.parquet`, all PNGs in `eyetrack/`, `gaze_screen_overlay.mp4`

---

### 5. `render_screen_gaze` (independent)

**Script:** `scripts/render_screen_gaze.py` → `src/screen_detection.py`, `src/corner_smoother.py`, `src/gaze_screen.py`

Standalone two-pass renderer for QC. Pass 1 runs corner detection on the clip; pass 2 renders a side-by-side 1920×540 video (left: original with outline + gaze dot, right: perspective-rectified iPad view). This rule runs independently of the main chain and shares no intermediates with it.

**Output:** `results/{subject}/eyetrack/screen_gaze_overlay.mp4`

---

## Key source modules

| Module | Role |
|---|---|
| `src/gaze_projection.py` | Per-sample H interpolation, screen/canvas projection, anchor-refit smoother |
| `src/homography_solver.py` | Box calibration, per-frame H fitting, behavior↔screen affine |
| `src/screen_detection.py` | iPad corner detection (color prior + NMS) |
| `src/box_corner_detector.py` | Low-level corner candidate scoring |
| `src/corner_smoother.py` | Rolling-median corner smoother |
| `src/homography_refinement.py` | Anchor-translate refit |
| `src/local_star_detector.py` | Detect behavior dots in local image windows |
| `src/star_detector.py` / `star_matcher.py` | Global star detection and matching to predicted positions |

---

## Configuration

`config_eyetrack.yaml` (overridable with `config_eyetrack.local.yaml`):

```yaml
subjects:
  EC347:
    tobii_dir: ...      # path to Tobii export directory
    scene_video: ...    # scene video filename
    tobii_tsv: ...      # TSV filename

clip_start_s: 120.0     # start of QC video clip
clip_duration_s: 30.0   # duration of QC clip
smoothing_window: 10    # rolling-median window for corner smoother

behavior_canvas:
  url_bar_h_px: 272     # Safari URL bar height in screen px
  x_pad_px: 233         # horizontal padding from screen edge to canvas
  max_y: 0.75           # upper fraction of canvas used for dot placement
```

To run a new subject: add an entry under `subjects:` (or in a local override file), ensure `homography_labels.parquet` exists, then run:

```bash
uv run snakemake -s Snakefile_eyetrack --configfile config_eyetrack.yaml -j1
```
