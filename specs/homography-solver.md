# Homography solver — per-frame screen↔frame mapping

## Status

- **Branch:** `trusting-cheese` (current worktree).
- **Prereqs done:** Hand-labeled ground-truth correspondences exist at
  `results/EC347/homography_labels.parquet` (28 frames). See
  `notebooks/label_homography_correspondences.py`. Coverage summary:

  | label type   | visible | not_visible | comment |
  |---|---|---|---|
  | `screen_bl`  | 27 | 1 | reliable |
  | `screen_br`  | 28 | 0 | reliable |
  | `box_bl`     | 26 | 2 | reliable (see note below) |
  | `box_br`     | 26 | 2 | reliable |
  | `big_star`   | 19 | 9 | the 9 not-visible are inter-trial / pre-trial-1 |
  | `screen_tl`  | 4  | 1 | mostly occluded by photodiode device — **don't rely on** |
  | `screen_tr`  | 3  | 2 | mostly occluded — **don't rely on** |

  Of the TL labels, 3/4 are marked `approximate`. Only frame 475 has a
  `confident` TL — but that frame has `screen_bl` not-visible, so it can't
  be used for a 4-corner H fit either.

- **Important semantics note:** `box_bl`/`box_br` in the labels file are
  **the bottom-left and bottom-right corners of the physical photodiode
  device** (the sensor clamped onto the top of the iPad), *not* the
  on-screen signal-box corners that the original spec assumed. The signal
  box itself is too small (~1 frame-px tall) to label by eye. The photodiode
  device is mechanically fixed to the iPad → its screen-coord position is
  constant across frames, but **unknown** until we calibrate it.

## Goal

Given the labeled set, produce a `H_screen→frame` 3×3 homography for each
labeled frame. Outputs feed downstream:

1. **Sanity check for existing detectors.** The Phase-1a/1b corner detector +
   `homography_refinement.anchor_translate` produces a homography from
   automated corner detection; comparing that to our hand-labeled H gives
   the reprojection error of the existing pipeline.
2. **Foundation for a corner-free homography pipeline.** If the photodiode
   device's screen position is well-calibrated, *any* future frame with
   `screen_bl`/`screen_br` + `box_bl`/`box_br` automatically detected gives
   a 4-correspondence H — no TL/TR detection needed. (Downstream work.)

## Approach: sequential, with joint as fallback

Two steps, both closed-form. If the per-frame `big_star` reprojection error
shows systematic bias, escalate to a joint optimization in a follow-up
(scope below).

### Step 1: Calibrate photodiode-device screen position

**Use every frame with all 4 iPad corners + both photodiode corners
labeled visible**, regardless of quality. The photodiode device physically
occludes the iPad's top-left corner on essentially every frame, so the
labeler's `screen_tl` (and often `screen_tr`) labels are mostly
`approximate` guesses. We intentionally **relax the quality filter for
this calibration step only** — multiple noisy estimates averaged are
better than one less-noisy estimate, and the per-frame H fitting in Step
2 doesn't use TL/TR at all, so the relaxation doesn't bleed downstream.

Procedure: for each calibration frame independently,

```python
# Fit H_screen→frame from the 4 iPad corners.
H_s2f = cv2.findHomography(screen_corners_iPad, frame_corners_labeled)[0]
H_f2s = np.linalg.inv(H_s2f)
# Back-project the photodiode corner labels to screen-coords.
box_bl_screen_i = perspective_transform(H_f2s, label_box_bl_frame_xy)
box_br_screen_i = perspective_transform(H_f2s, label_box_br_frame_xy)
```

Aggregate across the N calibration frames:

- Reported calibrated position = `median` over per-frame estimates (robust
  to one outlier frame).
- Spread = `IQR` and `max − min` per coordinate; reported as a quality
  signal. If max−min > 30 screen-px, the calibration is shaky and we
  should escalate (see open questions).

Saved to `results/EC347/homography_box_calibration.json`:

```json
{"box_bl_screen": [median_x, median_y],
 "box_br_screen": [median_x, median_y],
 "calibration_frames": [664, ...],
 "per_frame_estimates": {"664": {"box_bl_screen": [...], "box_br_screen": [...]}, ...},
 "spread_screen_px": {"box_bl": {"iqr_x": ..., "max_minus_min_x": ..., "iqr_y": ..., ...},
                      "box_br": {...}},
 "notes": "TL labels are mostly approximate (photodiode occlusion); per-frame spread is the quality signal."}
```

**Single-frame floor:** at minimum we have frame 664 (the only one in the
current label set with all 4 corners visible). The spec assumes the
labeler will add ~2–5 more frames with all-4 corners labeled (even with
TL/TR marked `approximate`) before this solver is implemented. If only
frame 664 is available when implementation starts, proceed with N=1 and
flag the missing spread metric in the calibration JSON.

### Step 2: Per-frame homography

For each labeled frame with all 4 anchor labels (`screen_bl`, `screen_br`,
`box_bl`, `box_br`) visible and quality ∈ {`confident`}:

```python
screen_pts = [screen_BL, screen_BR, box_bl_screen_calibrated, box_br_screen_calibrated]
frame_pts  = [label_screen_bl, label_screen_br, label_box_bl, label_box_br]
H = cv2.findHomography(screen_pts, frame_pts)[0]
```

`cv2.findHomography` is closed-form (DLT) on exactly 4 points. No optimizer.

**Frames skipped:** any frame missing any of the 4 anchor labels (visible &
confident). From current data: frames 475, 500 (head-jump pre-trial-1) are
both missing one or more — 26 frames remain.

Saved to `results/EC347/homography_per_frame.parquet`:

| column | type | description |
|---|---|---|
| `frame_idx` | int | |
| `h00`–`h22` | float64 | 9 elements of H (row-major, normalised so h22=1) |
| `n_correspondences` | int | always 4 in sequential approach |
| `big_star_residual_px` | float | distance between labeled big_star and H@true_xy; NaN if big_star not visible |
| `excluded_reason` | str | empty when included; else "missing_screen_bl" / etc. |

### Step 3: Validation via big_star reprojection

For each frame with `big_star` visible & confident:

```python
true_screen_xy = trials.loc[(trial_idx, tpt), ["true_x","true_y"]] * (2388, 1668)
predicted_frame_xy = perspective_transform(H, true_screen_xy)
residual_px = ||predicted_frame_xy − labeled_big_star_frame_xy||
```

**Held-out signal:** `big_star` is *not* used to fit H, so this is genuine
out-of-sample validation. 19 frames worth.

**Escalation criterion → joint optimization:** if median residual > 20
frame-px or there's a systematic bias (e.g., all residuals point in the same
direction, suggesting box calibration error), escalate. The joint version
(out of scope for this spec — see below) would jointly estimate per-frame
H + the shared 4 photodiode-position scalars across all frames.

## Module structure

Following `[[isolate new analyses]]` — new modules and notebook, no edits to
existing detectors. Module name picked to avoid collision with existing
`src/homography_refinement.py`:

- `src/homography_solver.py` — pure functions:
  ```python
  def calibrate_box_position(
      labels_df: pd.DataFrame,
      screen_w_px: int = 2388,
      screen_h_px: int = 1668,
  ) -> dict:
      """Auto-selects every frame in labels_df with all 4 iPad corners +
      both box corners visible (any quality, including 'approximate' —
      photodiode occlusion makes TL/TR labels intrinsically approximate).
      Returns {'box_bl_screen': (median_x, median_y),
               'box_br_screen': (...),
               'calibration_frames': [...],
               'per_frame_estimates': {frame_idx: {...}, ...},
               'spread_screen_px': {...}}.
      Raises ValueError if no frames qualify."""

  def fit_per_frame_homography(
      labels_df: pd.DataFrame,
      box_bl_screen: tuple[float, float],
      box_br_screen: tuple[float, float],
      screen_w_px: int = 2388,
      screen_h_px: int = 1668,
      include_qualities: set[str] = {"confident"},
  ) -> pd.DataFrame:
      """Returns one row per labeled frame; H elements + reason if excluded.
      Each H is computed only from BL/BR/box_bl/box_br (big_star is
      held-out)."""

  def big_star_residuals(
      per_frame_h: pd.DataFrame,
      labels_df: pd.DataFrame,
      trials_df: pd.DataFrame,
      screen_w_px: int = 2388,
      screen_h_px: int = 1668,
  ) -> pd.DataFrame:
      """Per-frame: predicted vs labeled big_star frame-xy + residual_px."""
  ```
- `tests/test_homography_solver.py` — fixtures-only (no video I/O):
  - `calibrate_box_position` on a synthesised frame with known H and known
    box-corner screen-coords recovers the box-corners within numerical
    precision.
  - `fit_per_frame_homography` on a synthesised dataset where the true H is
    known recovers H within numerical precision.
  - `big_star_residuals`: with the true H and the true big_star screen-xy,
    residual = 0.
  - Edge case: frame missing one anchor label → row has `excluded_reason`
    populated and no H.
- `notebooks/homography_eval.py` — jupytext py:percent. Loads everything,
  calls the three solver functions, emits the visual + numeric outputs
  below.

## Validation outputs (the spot-check)

All written to `results/EC347/homography_eval/`. Spot-check frame set:

| frame | category | why |
|---|---|---|
| 664   | calibration anchor | the source for box calibration; H should fit perfectly |
| 1550  | mid-trial with big_star | typical case; tests held-out validation |
| 2288  | inter-trial (no big_star) | tests H quality without the validation signal |
| 30125 | head-motion jump | tests stability across head pose change |

For each spot-check frame, write `frame_{idx}_overlay.jpg`:

- The original frame (BGR).
- **Labeled points**, filled dots, colors matching
  `LABEL_COLORS` in `notebooks/label_homography_correspondences.py`:
  - `screen_bl` #ffcc00, `screen_br` #4488ff (filled)
  - `box_bl` #ff44ff, `box_br` #44ffff (filled)
  - `big_star` #ffffff (filled, when visible)
- **Back-projections from H**, open circles of the same color, slightly
  larger — should sit on top of the corresponding labeled points:
  - screen_bl/br: H @ iPad corner screen-coords
  - box_bl/br: H @ calibrated box screen-coords
  - big_star: H @ `(trial.true_x, trial.true_y) * (2388, 1668)` (when
    available — even on frames where it wasn't labeled visible)
- **All currently-on-screen stars**, small filled white circles at H @
  screen-coord — sanity check beyond what was labeled. Visibility rule: any
  trial whose `(trial.trial_onset ≤ expt_t ≤ trial.trial_offset)` and whose
  `tpt`'s `reveal_time ≤ expt_t`. (Same rule as Phase 1b's predicted_positions.)
- **Residual arrows** from label → back-projection, magnified ×5 for
  visibility, drawn in red. Skip arrows where length × 5 < 5 px (no point).
- Caption at top-left: `frame {idx}  expt_t={ms:.0f}ms  trial={t}.{tpt}
  big_star_residual={r:.1f}px` (or `n/a`).

Also write `big_star_residual_hist.png`:
- Histogram of `big_star_residual_px` across all 19 validation frames.
- Vertical line at the median.
- X-axis in frame-px.
- Subtitle: median + interquartile range numerics.

Per-frame numeric summary printed in the notebook output:

| frame_idx | included | big_star_residual_px | excluded_reason |
|---|---|---|---|
| 475 | False | n/a | missing_screen_bl |
| 500 | False | n/a | missing_screen_tl |
| 664 | True | … | |
| … | | | |

## Out of scope

- **Joint per-frame H + shared box-position optimization.** A separate spec
  if validation fails. Would parameterize as `8N + 4` scalars (per-frame H
  with h33=1 + 4 photodiode-position scalars) and use
  `scipy.optimize.least_squares` (Levenberg-Marquardt) over per-frame
  reprojection residuals weighted by quality.
- **Automated detection of any of the 7 label types.** The point of this
  solver is to consume hand labels, not to replace them. The downstream
  detector pipeline is what would eventually consume this H to validate
  itself.
- **Snakemake rule.** Following the same pattern as
  `notebooks/label_homography_correspondences.py` (interactive-only) —
  the homography eval notebook is run by hand. If/when joint optimization
  arrives, that can be a Snakemake rule because it's deterministic and
  batch.
- **Subjects other than EC347.** Single-subject scope; the functions are
  parameterised by subject so they generalise without code change.
- **Modifying the labeling notebook or labels file.** If the labels turn out
  to be insufficient (e.g., calibration is too noisy), the corrective move
  is to label more frames in a separate labeling session, not to edit this
  solver's logic.

## Open questions (flag in notebook output if encountered)

1. **Is the box-position calibration good enough?** Two signals: (a) the
   spread of per-frame box-position estimates from Step 1 (if max−min > 30
   screen-px the calibration frames disagree more than makes sense for a
   mechanically-fixed device), and (b) the big_star residual histogram from
   Step 3. If either fails, the corrective moves are: escalate to joint
   optimization (let the 19 big_star observations correct the box-position
   estimate jointly with per-frame H), or relabel the calibration frames
   more carefully.
2. **Does H quality correlate with head pose / time?** Plot residual vs
   `frame_idx`. If there's drift (e.g., residual grows late in the video
   because the photodiode has shifted on its clamp), the "constant
   photodiode screen-coord position" assumption is wrong and we need a
   per-segment calibration. Doubt this is real but worth a single plot.
3. **Is the box calibration consistent with the original spec's
   geometry?** The spec assumed "30 screen-px visible strip" at the top of
   the screen; sanity-check confirms the photodiode bottom is at y ≈ 510
   screen-px instead — so the *photodiode device* extends ~30% down the
   screen, and the on-screen signal-box that it sits over is a separate,
   much smaller thing. Update the original homography-labeling spec's
   `BOX_VISIBLE_STRIP_HEIGHT_SY` mention if this gets confused later.
