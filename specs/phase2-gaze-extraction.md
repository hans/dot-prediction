# Phase 2 — Tobii gaze projection + fixation extraction

## Context

Phase 1c produced per-frame `H_screen→frame` for the full scene video
(`results/{subject}/phase1c_per_frame.parquet`). Phase 2 closes the eyetracking
loop: project every Tobii gaze sample into screen-pixel and behavior-canvas
coordinates, and extract fixation events using Tobii's built-in classification.

Downstream consumers: gaze-aligned ERPs, model-based fixation analyses
(LoT-predicted look targets), saccade-PSTH around dot reveal.

## Verified assumptions (do not re-derive)

1. **Tobii gaze coords are 1920×1080 video pixels** (matches `Recording media
   width/height` and the actual `scenevideo.mp4` dimensions). The inverse of
   `H_screen→frame` from `phase1c_per_frame.parquet` is the right path to
   screen pixels.
2. **`recording_ts_us / 1e6 ≡ video_t_s`** — Tobii Pro Glasses record gaze and
   scene video on a shared clock. Last Tobii ts = 1277.23 s vs video duration
   1277.07 s; agreement is within sub-frame precision. No separate
   Tobii↔video calibration is needed.
3. **Behavior time is the canonical analysis axis.** `behavior_t_ms =
   video_t_s * slope_ms_per_s + intercept_ms` using
   `results/{subject}/video_alignment.json` (slope ≈ 1000.03, intercept ≈
   303290.7 for EC347).
4. **Tobii's built-in `Eye movement type` is acceptable** for this phase.
   Values seen: Fixation, Saccade, Unclassified, EyesNotFound. `Eye movement
   type index` is unique per event and groups consecutive samples. Verify in
   implementation by asserting monotonicity of `em_type_index` per em_type
   segment. If zero rows have `em_type == "Fixation"`, fail loudly.

## Inputs

| file | description |
|---|---|
| `data/{subject}/tobii/{tobii_tsv}` | Raw Tobii TSV (442 972 rows for EC347; ~100 Hz) |
| `results/{subject}/phase1c_per_frame.parquet` | Per-frame H, from Phase 1c |
| `results/{subject}/video_alignment.json` | video_t ↔ behavior_t affine |
| `results/{subject}/trials_with_video.parquet` | Trial events; columns include `reveal_time`, `response_time` (behavior ms), `response_x`, `response_y` (canvas normalized), `true_x`, `true_y` (revealed-dot canvas normalized) |
| `data/{subject}/tobii/scenevideo.mp4` | For overlay video output only |

## Pipeline

For each Tobii TSV row:

1. `video_t_s = recording_ts_us / 1e6`
2. `behavior_t_ms = video_t_s * slope_ms_per_s + intercept_ms`
3. `video_frame_frac = video_t_s * fps` (fps from `video_alignment.json`)
4. **Homography lookup**: `H_t = (1 - α) * H[floor] + α * H[ceil]` element-wise,
   α = frac(`video_frame_frac`). If either `H[floor]` or `H[ceil]` contains
   NaN (e.g., `no_screen` frame), set `H_t = NaN` and `homography_valid =
   False`. Otherwise `homography_valid = True`. **Do not re-implement Phase
   1c's gap-filling** — `phase1c_per_frame` already has an H for every
   integer frame. This lerp handles only the sub-frame interpolation between
   successive integer-frame Hs.
5. **Project gaze**: `(gx_video, gy_video, 1)ᵀ` through `H_t⁻¹` (inverse of
   screen→frame H). Output `(gx_screen, gy_screen)` in iPad screen pixels.
6. **Screen→canvas** (inverse of `behavior_to_screen` from
   `src/homography_solver.py`):
   - `gx_canvas = gx_screen - canvas_x_pad_px` (canvas px, origin at canvas
     top-left; canvas width = 1922)
   - `gy_canvas = gy_screen - url_bar_h_px` (canvas height = 1396)
7. **Flags**:
   - `gaze_valid`: `em_type != "EyesNotFound"` AND `Validity left == 0` AND
     `Validity right == 0` AND `gaze_x`/`gaze_y` not NaN. (Tobii encodes
     0 = valid.)
   - `homography_valid`: from step 4.
   - `on_screen`: `0 ≤ gx_screen ≤ 2388` AND `0 ≤ gy_screen ≤ 1668`. Strict
     screen rect, no URL-bar trim.

## Outputs

### `results/{subject}/eyetrack/gaze_per_sample.parquet`

One row per Tobii sample. Columns:

| column | type | description |
|---|---|---|
| `tobii_ts_us` | int64 | Raw Tobii `Recording timestamp` (µs) |
| `behavior_t_ms` | float64 | Canonical behavior time (ms) |
| `video_frame_frac` | float64 | Fractional video frame |
| `gx_video`, `gy_video` | float64 | Raw Tobii gaze in video px; NaN if missing |
| `gx_screen`, `gy_screen` | float64 | Inverse-H projection; NaN if `homography_valid` is False or `gaze_valid` is False |
| `gx_canvas`, `gy_canvas` | float64 | Canvas-px (origin = canvas TL); NaN if either above is NaN |
| `gaze_valid` | bool | |
| `homography_valid` | bool | |
| `on_screen` | bool | |
| `em_type` | str | One of Fixation/Saccade/Unclassified/EyesNotFound |
| `em_type_index` | int64 | Unique event id |
| `em_duration_ms` | int64 | Event duration as reported by Tobii |
| `fixation_x_video`, `fixation_y_video` | float64 | Tobii's fixation centroid in video px (only populated for `em_type=Fixation` rows) |

### `results/{subject}/eyetrack/fixation_events.parquet`

Collapsed by `em_type_index` where `em_type == "Fixation"`. One row per
fixation event. Columns:

| column | type | description |
|---|---|---|
| `event_idx` | int64 | = `em_type_index` |
| `start_behavior_t_ms`, `end_behavior_t_ms` | float64 | First/last sample's behavior time |
| `duration_ms` | int64 | Tobii-reported duration (constant across event samples) |
| `n_samples` | int | Number of Tobii rows in event |
| `centroid_video_x`, `centroid_video_y` | float64 | Mean of `gx_video`, `gy_video` (gaze-valid samples) |
| `centroid_screen_x`, `centroid_screen_y` | float64 | Mean of `gx_screen`, `gy_screen` (both flags true) |
| `centroid_canvas_x`, `centroid_canvas_y` | float64 | Mean of `gx_canvas`, `gy_canvas` (both flags true) |
| `frac_homography_valid` | float64 | Fraction of event samples with `homography_valid` |
| `frac_on_screen` | float64 | Fraction of event samples with `on_screen` |

If the parquet is empty (zero fixations), raise an assertion in the notebook
with a clear message.

## Module: `src/gaze_projection.py`

Pure functions, no I/O:

```python
def tobii_ts_to_behavior_ms(ts_us: np.ndarray, slope_ms_per_s: float, intercept_ms: float) -> np.ndarray: ...
def tobii_ts_to_video_frame_frac(ts_us: np.ndarray, fps: float) -> np.ndarray: ...
def lerp_homography_at_frac(per_frame_h: np.ndarray, frame_frac: float) -> np.ndarray | None:
    """per_frame_h has shape (N_frames, 3, 3) — NaN for no_screen frames.
    Returns (3,3) or None if either flanking frame has NaN H."""
def project_video_to_screen(gx: np.ndarray, gy: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply H_inverse to (gx, gy) to get screen coords."""
def screen_to_canvas(sx, sy, url_bar_h_px=272, canvas_x_pad_px=233) -> tuple: ...
def is_on_screen(sx, sy, screen_w_px=2388, screen_h_px=1668) -> np.ndarray: ...
```

Vectorise where natural; the per-sample loop is fine if it's clear. EC347 has
~443k samples; aim for the per-sample step to complete in < 1 min.

## Tests: `tests/test_gaze_projection.py`

- `tobii_ts_to_behavior_ms` linear with known slope/intercept.
- `tobii_ts_to_video_frame_frac`: ts=0 → 0; ts at exact frame boundary →
  integer.
- `lerp_homography_at_frac` at α=0 returns `H_left`; at α=1 returns `H_right`;
  at α=0.5 returns the midpoint element-wise. NaN flanker → None.
- `project_video_to_screen` round-trip: take a known screen point, push
  through forward H, push back through `H_inv` → original within 1e-6.
- `screen_to_canvas`: round-trip with `behavior_to_screen` (canvas → screen →
  canvas → behavior) recovers the input on a 5×5 grid.
- `is_on_screen` boundary cases.
- NaN propagation: invalid gaze in → NaN out.

## Notebook: `notebooks/extract_gaze_fixations.py` (jupytext py:percent)

Per the user's "isolate new analyses" preference: fresh notebook, fresh
module, no edits to Phase 1c code or the homography solver. Import
`behavior_to_screen` from `homography_solver` for reference but don't modify
it.

Notebook sections:

1. **Parameters** (`tags=["parameters"]`): paths + canvas constants
   (`URL_BAR_H_PX=272`, `CANVAS_X_PAD_PX=233`, `MAX_Y_COORD=0.75`,
   `SCREEN_W_PX=2388`, `SCREEN_H_PX=1668`).
2. **Load inputs.** Read the TSV (only the columns we need; that file is
   200+ MB). Read the Phase 1c parquet, alignment JSON, and trials parquet.
3. **Project every sample.** Build the `gaze_per_sample` dataframe. Print:
   total samples, `gaze_valid` rate, `homography_valid` rate, `on_screen`
   rate.
4. **Sanity prints.** Distinct `em_type` values + counts. Assert at least one
   Fixation row.
5. **Collapse fixation events.** Build the `fixation_events` dataframe.
   Print: total events, mean duration, fraction with
   `frac_homography_valid > 0.8`.
6. **Validation metric.** For each click event in `trials_with_video` (rows
   where `response_time` is not NaN), compute the mean gaze position in
   canvas px over the window `[response_time − 300, response_time − 50]` ms,
   restricted to `gaze_valid & homography_valid & on_screen` samples.
   Convert the click `(response_x, response_y)` to canvas px via
   `(response_x * 1922, response_y * 1396 / 0.75)`. Compute the Euclidean
   distance. Aggregate: median, IQR, 90th percentile,
   `n_clicks_with_enough_samples` (≥3 valid samples in window). Fail-flag the
   session if median > 250 canvas px.
7. **Visualizations** — 5 fixed artifacts (write to `out_dir`):
   - `gaze_coverage_and_accuracy.png` — two-panel: (a) stacked-area timeline
     of `gaze_valid`/`homography_valid`/`on_screen` rates over the session
     (bin = 10 s); (b) histogram of the click→gaze validation distance.
   - `gaze_canvas_heatmap.png` — 2D fixation-centroid heatmap (use
     `np.histogram2d` + log scale) on a canvas backdrop (1922×1396 px, draw a
     faint URL-bar-trimmed rect), faceted into three phases: **pre-reveal**
     (samples before the first reveal of each trial), **reveal→click**
     (between reveal and click of the *same* tpt — only for tpts with a
     click), **post-click** (after each click, before the next reveal). Three
     side-by-side panels sharing a color scale.
   - `pre_click_gaze_trajectories.png` — 4×5 grid, 20 random click events
     (`seed=0` for reproducibility) from rows where `response_time` is not
     NaN AND there are ≥10 `gaze_valid`/`homography_valid` samples in
     `[response_time − 1000, response_time]` ms. Each subplot: canvas
     backdrop with the clicked dot (large yellow circle) and any other
     previously-revealed dots in the same trial (small grey circles); gaze
     trajectory in canvas px colored by time (viridis colormap, t = −1 s
     blue → t = click yellow); title `trial_idx={i} tpt={j}`. If fewer than
     20 such clicks exist, use what's available and label.
   - `saccade_psth_around_reveal.png` — for each reveal event in
     `trials_with_video`, find Tobii rows where `em_type == "Saccade"` and
     the sample is the first row of the event (`em_type_index` change).
     Compute `Δt = saccade_onset_behavior_ms − reveal_time_ms`. Histogram all
     Δt in the window [−2000, +2000] ms, bin = 50 ms. Single panel, with a
     vertical line at 0.
   - `gaze_screen_overlay.mp4` — 30 s clip starting at `video_t = 120 s`
     (matches the existing `clip_start_s`). Use `cv2.VideoWriter`. For each
     frame in the window:
     - draw the screen rect (4 corners via `H @ screen_corners`) as a yellow
       polygon;
     - draw each previously-revealed dot in the active trial via
       `behavior_to_screen` → frame px (`H @ screen_xy`); cyan filled circle,
       radius 12;
     - draw the Tobii gaze sample nearest to this frame's `behavior_t` (raw
       video-px) as a white circle, radius 8, with a red outline if
       `gaze_valid` is False;
     - frame counter and `behavior_t` in top-left text.
   - All PNGs at 150 DPI.
8. **Summary print** of pass/fail flags.

## Snakemake rule (in `Snakefile_eyetrack`)

```python
rule extract_gaze_fixations:
    input:
        video=lambda wc: f"{subj(wc)['tobii_dir']}/{subj(wc)['scene_video']}",
        tsv=lambda wc: f"{subj(wc)['tobii_dir']}/{subj(wc)['tobii_tsv']}",
        phase1c=rules.phase1c_homography.output.per_frame,
        align="results/{subject}/video_alignment.json",
        trials="results/{subject}/trials_with_video.parquet",
        notebook="notebooks/extract_gaze_fixations.py",
    output:
        per_sample="results/{subject}/eyetrack/gaze_per_sample.parquet",
        fixation_events="results/{subject}/eyetrack/fixation_events.parquet",
        coverage="results/{subject}/eyetrack/gaze_coverage_and_accuracy.png",
        heatmap="results/{subject}/eyetrack/gaze_canvas_heatmap.png",
        trajectories="results/{subject}/eyetrack/pre_click_gaze_trajectories.png",
        psth="results/{subject}/eyetrack/saccade_psth_around_reveal.png",
        overlay_mp4="results/{subject}/eyetrack/gaze_screen_overlay.mp4",
        notebook="results/{subject}/notebooks/extract_gaze_fixations.ipynb",
    run:
        canvas = config.get("behavior_canvas", {})
        run_notebook(
            input.notebook,
            output.notebook,
            parameters=dict(
                subject=wildcards.subject,
                video_path=input.video,
                tsv_path=input.tsv,
                phase1c_path=input.phase1c,
                align_path=input.align,
                trials_path=input.trials,
                out_dir=str(Path(output.per_sample).parent),
                URL_BAR_H_PX=canvas.get("url_bar_h_px", 272),
                CANVAS_X_PAD_PX=canvas.get("x_pad_px", 233),
                MAX_Y_COORD=canvas.get("max_y", 0.75),
            ),
        )
```

Add to `rule all`'s expand list.

## Validation criterion (PASS/FAIL flag)

Session median click→gaze canvas-px distance over `[click − 300, click − 50]`
ms < 250 px → PASS. Expected ~50–150 px. Print prominently in notebook
output. Do not fail the rule if the criterion is not met — just flag.

## Out of scope (do NOT implement)

- Re-classifying fixations (we use Tobii's built-in). Add a TODO comment
  noting we may revisit with I-DT/I-VT later.
- Smoothing gaze, gap-filling missing samples.
- Per-trial PNGs (we use a 4×5 grid).
- Subjects other than EC347.
- Modifying `render_screen_gaze.py` or any Phase 1c code.
- Saccade-target inference (which dot was being looked at).
