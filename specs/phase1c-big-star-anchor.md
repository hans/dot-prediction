# Phase 1c — big_star as 5th anchor in per-frame H fit

## Context and motivation

The original Phase 1c pipeline (`specs/phase1c-box-corner-detector.md`) fits a
per-frame screen→frame homography from **4 anchors**:

| anchor | screen-px position | frame-px source |
|---|---|---|
| `screen_bl` | (0, H) | `screen_corners.parquet` |
| `screen_br` | (W, H) | `screen_corners.parquet` |
| `box_bl` | calibrated | `box_corner_detector` |
| `box_br` | calibrated | `box_corner_detector` |

All four anchors live near the **screen bottom + photodiode device** (the box
corners are inside the screen near `y = H`). The `screen_tl` and `screen_tr`
corners are not directly observed — they are reached by extrapolating the H
across the iPad's full height. This makes the projection at TR/TL very
high-leverage: sub-pixel jitter in any of the 4 anchors balloons into
thousands of frame-px of TR/TL error, producing the catastrophic polygon
"jumps" visible in the overlay rendering.

PR #5 (H-element smoothing) and PR #7 (anchor-refit smoothing) both work
around this by smoothing in the downstream Phase 2 step, but neither addresses
the conditioning problem at the source. **A 5th anchor in the screen interior
constrains the projection across the whole iPad**, dramatically reducing
TR/TL leverage.

The `big_star` — the most-recently-revealed dot in the experiment — is:

- Detectable per frame (warm-coloured blob on a strongly blue background,
  same as the small stars in Phase 1a).
- Near the centre of the iPad screen (canvas-relative coordinates).
- Already used as a held-out validation signal in
  `homography_solver.big_star_residuals()` and
  `notebooks/phase1c_eval.py`.

Phase 1c v2 promotes `big_star` to a **5th fit anchor**.

---

## Anchor geometry comparison

| variant | n_anchors | anchor coverage of iPad screen |
|---|---|---|
| v1 (box-corner detector) | 4 | bottom band only |
| v2 (this spec) | 5 | bottom band + central screen point |

Adding any single point above `y ≈ H − box_height` would constrain TR/TL.
`big_star` is the cheapest such point — it requires no new detector module
and no new label, only a per-frame screen-xy lookup from
`trials_with_video.parquet`.

---

## Algorithm

Per frame `t`, in addition to the existing 4-anchor pipeline:

**Step E1 — find the active dot**

Look up the most-recently-revealed trial point with
`video_frame_reveal <= t`. If no such row exists (pre-experiment frames),
proceed with the existing 4-anchor fit and tag `n_anchors_used = 4`.

```python
active = trials_df[trials_df.video_frame_reveal <= t].loc[
    trials_df.video_frame_reveal.idxmax()
]
screen_xy = behavior_to_screen(active.true_x, active.true_y)
```

**Step E2 — detect big_star in frame**

Project `screen_xy` through `H_prev` (the most recent successfully-fit H,
which carries the screen-interior geometry from the prior frame) to predict
the frame-px location of the big_star, then run the local star detector:

```python
predicted_xy = _project(H_prev, screen_xy)
big_star_det = local_star_detector.detect_in_windows(
    frame, [PredictedStar(..., frame_xy=predicted_xy, ...)],
    window_size_px=60, floor=20.0,
)
```

`H_prev` (not the just-fit 4-anchor `H_new` for this frame) is used as the
prediction H because `H_new` is exactly what we're trying to correct — its
TR/TL drift would re-enter the prediction. `H_prev`'s screen-interior
projection is much more stable, so it's a robust window centre.

The local detector returns `None` when the window contains no warm blob
brighter than `floor`, when the blob is unreasonably large relative to the
expected radius, or when the predicted window is fully off-frame.

**Step E3 — refit H with 5 anchors**

If `big_star_det` is non-None, the screen↔frame correspondence list extends
to 5 points and OpenCV's `findHomography` overdetermines the 8-DOF model
(rank-12 vs. 8). The resulting H is the best least-squares fit consistent
with all 5 points.

If `big_star_det` is None, fall back to the 4-anchor fit. Tag the frame's
`n_anchors_used = 4` and `detection_reason` records why
(`"no_active_dot"` or `"big_star_not_detected"`).

```python
screen_pts = [screen_bl, screen_br, box_bl, box_br, screen_xy]  # length 4 or 5
frame_pts  = [screen_bl_det, screen_br_det, box_bl_det, box_br_det, big_star_det]
H_new, _ = cv2.findHomography(screen_pts, frame_pts)
```

**Step E4 — leave-one-out validation**

On 5-anchor frames only, compute LOO reprojection residuals: for each anchor
`i`, refit H from the remaining 4 points, project `screen_pts[i]` through
the LOO H, measure distance to `frame_pts[i]`. Saved as columns
`loo_screen_bl_px`, `loo_screen_br_px`, `loo_box_bl_px`, `loo_box_br_px`,
`loo_big_star_px`.

These are stored per-frame and aggregated in the eval notebook. A median
`loo_big_star_px > 20 px` indicates the big_star is being detected on the
wrong blob (or that the 4 box-corner anchors have drifted enough that
adding big_star contradicts them).

---

## Acceptance criteria

From issue #8:

- Polygon overlay (especially TR/TL corners) is tight across the full
  video. Whole-video TR_x catastrophic outliers (O(1M px) at handful of
  frames in v1) should disappear.
- Click→canvas validation median (downstream Phase 2 metric) ideally under
  250 canvas-px (v1: 322; the gate is < 250). Verified after Phase 2 reruns
  against the new `phase1c_per_frame.parquet`.
- No regression on `homography_valid` rate or `on_screen` rate.

Internal checks emitted by `notebooks/phase1c_eval.py`:

- Fraction of `detected` frames with `n_anchors_used = 5` should be high
  during trial windows (the period of valid experiment behaviour). Outside
  trials, 4-anchor fallback is expected and not a regression.
- Median `loo_big_star_px` on 5-anchor frames should be small (< 5 px) —
  the in-fit big_star is consistent with the other 4 anchors.

---

## Schema changes

`results/{subject}/phase1c_per_frame.parquet` adds these columns (existing
columns unchanged):

| column | type | description |
|---|---|---|
| `big_star_x`, `big_star_y` | float64 | Detected big_star frame-px position; NaN when 4-anchor fallback. |
| `n_anchors_used` | int | 4 or 5; 0 for non-detected frames. |
| `loo_screen_bl_px`, `loo_screen_br_px`, `loo_box_bl_px`, `loo_box_br_px`, `loo_big_star_px` | float64 | LOO reprojection residual per anchor. All NaN when `n_anchors_used != 5`. |

Existing `big_star_residual_px` is **repurposed**: with big_star in the fit,
the saved H trivially projects the *detected* big_star onto itself. To
preserve the column's "honesty check at hand-labeled big_star frames"
semantics, it is now computed via a 4-anchor refit at those frames:

1. Drop big_star from the 5 anchors, refit `H_4pt` from screen_bl/br +
   box_bl/br only (using the saved frame-px positions).
2. Project the labeled `true_xy` through `H_4pt`.
3. Residual = distance to the labeled big_star frame-xy.

This is exactly what the v1 metric measured (4-anchor H's accuracy at the
hand-labeled big_star) and remains comparable across versions.

`detection_status` semantics unchanged. A new `detection_reason` value
(`"big_star_not_detected"` or `"no_active_dot"`) is recorded on frames where
the 4-anchor fallback fired.

---

## New module: `src/big_star_detector.py`

```python
def active_dot_screen_xy(
    trials_df: pd.DataFrame, frame_idx: int, behavior_to_screen, ...
) -> tuple[tuple[float, float], int, int, int] | None:
    """Most-recently-revealed dot's screen-xy + (reveal_frame, trial_idx, tpt).
    None if no dot revealed yet."""

def detect_big_star(
    frame: np.ndarray, H_prior: np.ndarray, screen_xy: tuple[float, float],
    reveal_frame: int | None, current_frame: int, fps: float = 30.0,
    window_size_px: int = 60, floor: float = 20.0,
    max_radius_factor: float = 4.0,
) -> tuple[float, float] | None:
    """Wrap local_star_detector.detect_in_windows for one predicted star."""
```

Implementation: predict frame-xy via `H_prior @ [sx, sy, 1]`, compute
expected blob radius via the existing age-based size model
(`predicted_positions.expected_screen_radius_px`) scaled by the local
Jacobian determinant of `H_prior` at `screen_xy`, and forward to
`detect_in_windows` with a single `PredictedStar`.

---

## New helpers in `src/homography_solver.py`

```python
def fit_homography(screen_pts, frame_pts) -> np.ndarray | None:
    """Thin findHomography wrapper, normalises h22 = 1."""

def loo_residuals(screen_pts, frame_pts) -> list[float]:
    """N≥5: leave each anchor out, refit, project, measure frame-px residual.
    N<5: returns all NaN (4-pt DLT is exactly determined)."""
```

`fit_per_frame_homography` accepts an opt-in `include_big_star=True` flag and
optional `big_star_screen_lookup: dict[frame_idx, (sx, sy)]` for callers that
work from hand-labeled data (e.g. `notebooks/homography_eval.py`). The
production per-frame fit lives in `notebooks/phase1c_eval.py` and uses the
detector path described above.

---

## Tests

`tests/test_homography_solver.py`:

- `test_fit_recovers_true_H_5pt` — 5 exact correspondences, fit recovers
  `_H_TRUE` within numerical precision; `n_correspondences == 5`.
- `test_loo_residuals_zero_on_exact_fit` — 5 exact correspondences →
  every LOO residual is ~0.
- `test_loo_residuals_nan_when_fewer_than_5` — 4 inputs → all NaN.

`tests/test_big_star_detector.py` (new):

- `test_detect_big_star_finds_warm_blob` — synthetic blue frame + warm blob,
  identity H, prediction at blob centre → detector returns sub-pixel xy
  near the blob.
- `test_detect_big_star_missing_returns_none` — no blob painted → None.
- `test_active_dot_screen_xy_picks_max_reveal` — multiple revealed rows,
  function returns the one with max `video_frame_reveal`.
- `test_active_dot_screen_xy_none_before_first_reveal` — frame_idx before
  any reveal → None.

---

## Out of scope

- **Detecting big_star from a global blob scan** (vs. local search). The
  global path is more robust to tracking loss but adds ~100 ms/frame and
  isn't needed: the 4-anchor H gives a reliable enough prediction for the
  local detector at the screen interior.
- **Using small stars as held-out validation** to replace the legacy
  `big_star_residual_px`. The 4-anchor refit at hand-labeled frames serves
  the same purpose and requires no new detector wiring.
- **Smoothing changes in Phase 2** (PR #5 / #7). With big_star in the fit
  the bad-regime amplitude shrinks, but the smoothing window stays as-is
  until Phase 2 is re-validated.
