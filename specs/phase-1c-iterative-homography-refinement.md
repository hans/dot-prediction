# Phase 1c — iterative homography refinement with star detections

## Status

- **Branch:** `iterative-homography-refi` (worktree:
  `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/iterative-homography-refi/`).
- **What's been done:**
  - Pre-flight checks (3 sanity checks). Results below — they change the
    design in 3 material ways. Script: `scripts/preflight_phase1c.py`;
    outputs: `results/EC347/preflight_phase1c/`.
  - **Step 1 — weighted re-solver + `Correspondence` dataclass**
    (`src/homography_refinement.py`). `solve_weighted_homography` uses the
    repeated-points trick for weighting (default K=10, weight 0.1
    granularity). **Defaults to least-squares (`method=0`)**; RANSAC is
    opt-in via `use_ransac=True` with `ransac_threshold_px=3.0`. Default
    switched from auto-RANSAC-at-N≥6 to lstsq per pre-flight check 1, which
    showed RANSAC's 3 px inlier filter has 3–12× worse p95 than lstsq under
    the post-gate Gaussian-noise regime (it randomly excludes clean points
    near the threshold and lands on degenerate subsets). RANSAC remains
    available as a watchlist for frames where a bad correspondence survives
    the gates. Returns `SolveResult(H, inlier_mask, residuals_px, method)`
    with masks/residuals aligned to the *input* list (zero-weight entries →
    False / NaN). Tests: `tests/test_homography_refinement.py` covers clean
    recovery (lstsq), least-squares path, RANSAC outlier rejection (opt-in),
    weight-biasing, <4-point rejection, zero-weight handling.
  - **Step 2 — detection quality gates** (`src/homography_refinement.py`).
    Three new symbols: `radius_match_ok(det, *, tau_radius=1.5)` —
    relative-error gate `|obs−exp|/exp ≤ τ` (matches Change-1's downweight
    formula; permissive when `expected_radius_px ≤ 0`);
    `resolve_blob_conflicts(dets, *, tau_centroid_px=3.0)` — single-link
    cluster on `frame_xy_subpix`, per-cluster winner is the detection whose
    source prediction's `frame_xy` is closest to the cluster centroid
    (singletons pass through); `apply_quality_gates(dets, ...)` — radius
    gate then resolver (in that order, so a radius-bad detection can't win
    a centroid tiebreak), returning `(accepted, [GateRejection(d, reason)])`
    with reasons `"radius_mismatch"` / `"same_blob"`. Tests cover both gates
    individually and end-to-end (22/22 passing in the file; 102/102 across
    the project).
  - **Step 3 — greedy constellation matcher** (`src/homography_refinement.py`).
    New entry point `detect_constellation(frame, predictions, ...)` — a drop-in
    replacement for `local_star_detector.detect_in_windows` that fixes the
    same-blob duplicate-detection failure: predictions whose windows overlap
    are grouped (single-link on rectangle intersection), the union mask is
    connected-component-labelled at the same R−B floor as the per-window
    detector, and components are greedy-assigned in *descending confidence*
    order to the nearest unspent prediction whose rectangular window
    contains the component centroid. The `max_radius_factor` reject still
    fires post-assignment (and leaves the prediction unmatched, no fallback
    blob). Singleton groups go through the same path — a minor refinement
    over `detect_in_windows`, which averaged multi-component centroids.
    Tests: 11 new (`test_homography_refinement.py::test_constellation_*`),
    including a confidence-ordering pin verified by flipping the sort
    direction. 113/113 across the project.
  - **Step 4 — correspondence builder** (`src/homography_refinement.py`).
    New entry point `build_correspondences(smoothed_corners, screen_corners, ...)`.
    Implements the Change-1 weighting table: smoothed BL/BR always at 1.0;
    raw BL/BR at 0.5 if `|raw−smoothed| < 10 px`; smoothed TL/TR at 0.3 and
    raw at 0.1 if `|raw−smoothed| < 5 px`, otherwise both top-corner entries
    omitted entirely (including when `raw_corners=None`); big-star at
    caller-supplied weight (default 1.0); small stars at
    `confidence/255 × radius_match_factor` where the factor linearly
    blends from 1.0 at `rel_err ≤ 0.5` down to 0.5 at `tau_radius=1.5`.
    All thresholds are keyword parameters. 15 new tests cover boundary deltas,
    clamped confidence, zero-expected-radius, all four source tags, and
    end-to-end feed into `solve_weighted_homography`. 128/128 across project.
  - **Step 5 — `iterate_homography()` controller** (`src/homography_refinement.py`).
    New entry point that orchestrates the predict → detect → gate → re-solve loop.
    Starts from `H_v0 = findHomography(screen_corners, smoothed_corners)` (lstsq,
    all 4 corners equal weight), optionally anchor-translates via a caller-supplied
    big-star pair, then iterates up to `k_max=2` times. Each iteration: calls
    `_predicted_positions` with the current H, runs `detect_constellation` + `apply_quality_gates`,
    builds correspondences via `build_correspondences`, and re-solves with
    `solve_weighted_homography`. Terminates with one of five `ConvergenceReason`
    values: `"k_max"`, `"converged"` (max corner displacement < `convergence_px=0.5`
    px), `"no_new_stars"` (accepted count didn't grow, checked from iter 2 onward),
    `"no_predictions"` (empty `predicted_positions` result), or `"solve_failed"`
    (<4 usable correspondences). Returns `IterationResult(H_refined, correspondences,
    anchor_less, iterations_run, convergence_reason, steps)` with `steps:
    list[IterationStep]` for per-iteration eval (each step records `H_in`, `H_out`,
    `predictions`, `detections`, `rejections`, `correspondences`, `solve_result`).
    Caller is responsible for matching the global big-star detection to the correct
    `PredictedStar.screen_xy` before calling. Note: distance-from-anchor window
    term from the spec is deferred; pass `adaptive_radius_factor` to approximate
    `base × expected_radius` only. 14 new tests cover all 5 termination paths, both
    anchor states, step field completeness, H-projection accuracy, and
    correspondence fallback. 142/142 across the project.
- **Where to resume:** Step 6 — `notebooks/iterative_homography_eval.py` evaluation notebook.

## Goal (recap)

For each frame within a trial, produce:

1. A homography `H_refined` that is more accurate than `H_rough_anchor` — usable
   downstream as the per-frame homography.
2. A set of confirmed star correspondences (screen_xy, frame_xy, confidence)
   including the big star and as many small stars as can be reliably detected,
   with same-blob-snapping and finger FPs explicitly rejected.

Target on a clean mid-trial frame: 6+ correct small-star detections (in
addition to the big star) with distinct centroids and sub-pixel offsets
matching expected positions under `H_refined`. Detection rate independent
of distance-from-anchor (the main Phase 1b failure mode this phase fixes).

## What Phase 1b provides

| File | Purpose |
|---|---|
| `src/star_detector.py` | `detect_stars(frame)` — global yellow-blob detector. Phase 1a, stable. |
| `src/screen_detection.py` | `detect_corners(frame)` — `[TL, TR, BR, BL]` or `None`. |
| `src/corner_smoother.py` | `smooth_corners(per_frame_corners)` — interp + rolling median. |
| `src/predicted_positions.py` | `predicted_positions(t_ms, trials_df, H)` → `[PredictedStar]` with `screen_xy`, `frame_xy`, `age_s`, `expected_radius_px`. |
| `src/local_star_detector.py` | `detect_in_windows(frame, predictions, ...)` → `[LocalDetection]` + unmatched. Has adaptive-window support. |
| `src/homography_refinement.py` | `anchor_translate(H, anchor_screen, anchor_frame)` — Phase 1b's single-correspondence translation. Phase 1c extends this module. |
| `notebooks/local_star_eval.py` | Phase 1b eval. Pattern to follow for the new notebook. |
| `results/EC347/local_star_eval/*.csv` | Phase 1b per-frame detection records for the 7 representative frames; baseline for the side-by-side comparison. |

`H_rough` is still not persisted to disk — computed on the fly via
`cv2.findHomography(SCREEN_CORNERS, detect_corners(frame))`, optionally
smoothed and anchor-translated.

## Pre-flight findings

Three checks were run on EC347. Outputs in `results/EC347/preflight_phase1c/`,
driver in `scripts/preflight_phase1c.py`.

### 1. `findHomography` numerical stability

500 trials of perturbing frame-side correspondences with Gaussian noise and
re-solving H; measured = max corner re-projection error (px) of the 4 screen
corners under the noisy-solved H vs. clean-solved H. A second sweep was run
with `method=cv2.RANSAC, ransacReprojThreshold=3.0` (the production solver
path for N≥6) to check whether RANSAC's inlier filtering helps or hurts under
pure Gaussian noise (no true outliers).

**lstsq (method=0):**

| Inputs | σ_in = 0.25 px | σ_in = 0.5 px | σ_in = 1.0 px | σ_in = 2.0 px |
|---|---:|---:|---:|---:|
| 4 corners only | 0.48 / 0.71 | 0.99 / 1.45 | 1.86 / 2.94 | 3.78 / 5.91 |
| 4 corners + 6 stars | 0.40 / 0.64 | 0.82 / 1.29 | 1.62 / 2.73 | 3.18 / 5.20 |
| 4 corners + 10 stars | 0.37 / 0.61 | 0.77 / 1.29 | 1.45 / 2.43 | 3.03 / 5.00 |
| **6 stars only (no corners)** | **3.26 / 6.50** | 6.36 / 13.59 | 12.82 / 28.92 | 25.76 / 57.41 |
| **10 stars only (no corners)** | **2.09 / 4.38** | 4.08 / 8.79 | 7.97 / 17.64 | 16.81 / 35.99 |

**RANSAC (ransacReprojThreshold=3.0), N≥6 cases only:**

| Inputs | σ_in = 0.25 px | σ_in = 0.5 px | σ_in = 1.0 px | σ_in = 2.0 px |
|---|---:|---:|---:|---:|
| 4 corners + 6 stars | 0.42 / 0.64 | 0.80 / 1.30 | **1.74 / 6.91** | **4.88 / 61.3** |
| 4 corners + 10 stars | 0.65 / 2.38 | 1.39 / 5.41 | **3.31 / 13.2** | **13.5 / 60.9** |
| 6 stars only (no corners) | 3.55 / 10.1 | 9.09 / 34.9 | 18.9 / 73.3 | 50.4 / 349 |
| 10 stars only (no corners) | 2.14 / 4.12 | 4.12 / 9.29 | 9.25 / 21.7 | 26.4 / 80.0 |

Values are median / p95 max-corner-error in frame px.

**Implications:**
- Sub-pixel input noise on corners + stars amplifies ~1.6× into corner
  re-projection error under lstsq — a comfortable noise floor. Iteration
  converges.
- **Stars without corners are an order of magnitude worse**, because EC347's
  star constellation fits inside the central ~70% of the screen — DLT cannot
  extrapolate well to the corners. **The original spec's "drop corners
  entirely once enough stars are confirmed" fallback is rejected**; corners
  are always needed to pin the perspective at the screen edges.
- More than 6 stars adds little numerical benefit under lstsq (1.45 vs 1.62
  px at σ=1.0). More stars buys robustness to bad correspondences, not noise
  reduction.
- **RANSAC hurts under pure Gaussian noise.** With no true outliers, the 3 px
  threshold randomly excludes good correspondences when noise pushes their
  reprojection error past the cutoff, occasionally leaving a near-degenerate
  point set. At σ=1.0 the p95 is 2.5–5× worse than lstsq; at σ=2.0 it is
  10–12× worse. The median is similar at σ≤0.5 but diverges at higher noise.
  **If the quality gates do their job** (blob-conflict + radius check remove
  the bad correspondences before the re-solve), the remaining inputs are clean
  Gaussian noise — exactly the regime where lstsq beats RANSAC. Use lstsq as
  the default; fall back to RANSAC only if eval shows catastrophic H estimates
  that survived the gates (see Risk section).

### 2. Bottom-corner trustworthiness

Per-frame `|raw - smoothed|` for each of the 4 corners, on the 7 Phase-1b eval
frames. Smoothed = rolling median over 51 frames (the existing pipeline).

| corner | median | mean | max |
|---:|---:|---:|---:|
| BL | 1.71 | 2.10 | 5.39 |
| BR | 3.00 | 3.96 | 7.81 |
| TL | 2.24 | 8.84 | 23.02 |
| TR | 9.63 | 63.52 | **234.62** |

The 234-px TR outlier is frame 1700 (hand occlusion across top of screen).
BL/BR survive that frame at 5.39 and 7.81 px — clipped by the same occlusion
but not catastrophically.

**Implications:**
- The spec's "bottom corners always trustworthy" assumption is *mostly*
  correct but breaks on occluded frames. Need a per-frame guard:
  **if `|raw - smoothed| > 10 px` for BL or BR, use the smoothed value
  (not raw) and downweight that corner** in the re-solve, rather than dropping
  it (we still need 4-corner spatial coverage per check 1).
- Top corners are gated more strictly: include with low weight only if
  `|raw - smoothed| < ~5 px`. On frame 1700 they're useless and must be
  excluded entirely.
- The smoothed value is always available, so corners never disappear from
  the correspondence set — they may just contribute at lower weight.

### 3. Star-position stationarity within a trial

`results/EC347/trials_with_video.parquet`: 300 rows, exactly one per
`(trial_idx, tpt)`. By data structure, `(true_x, true_y)` cannot vary
within a trial. ✓ Constellation matching can rely on this.

## Design (updated per pre-flight)

The structure of the original spec stands; the three changes below are
material. Anything not called out is unchanged from the original spec
(reproduced verbatim at the end of this document).

### Change 1 — Re-solver inputs

| Source | Confidence | Conditions |
|---|---|---|
| Smoothed BL, BR (from `corner_smoother`) | **1.0** always | Always included. |
| Raw BL, BR | 0.5 | Use only if `\|raw − smoothed\| < 10 px` for that corner this frame; else fall back to smoothed alone for that corner. |
| Smoothed TL, TR | 0.3 | Include unless flagged as bad. Spec follows. |
| Raw TL, TR | 0.1 | Use only if `\|raw − smoothed\| < 5 px`. On hand-occlusion frames (TR often >100 px off), drop entirely. |
| Big star (from global detector) | from detector | Anchor. Drop frame to anchor-less fallback if absent. |
| Small stars (from local detector) | from detector × radius-match | Downweight when `\|equivalent_radius − expected_radius\| / expected_radius` exceeds ~0.5. |

**Never drop all corners.** Pre-flight check 1 shows stars-only solves
have 5-10× worse extrapolation error at the screen edges than star+corner
solves, because the constellation is centrally concentrated. The corners
do real work pinning the perspective extrapolation.

Solve via `method=0` (lstsq) regardless of correspondence count. Pre-flight
check 1 shows that under pure Gaussian noise — the expected regime after the
quality gates remove bad correspondences — lstsq has the same median error as
RANSAC at low noise (σ≤0.5 px) but 3–12× lower p95 at σ≥1 px. RANSAC's
inlier-filtering randomly excludes good points when noise pushes their
reprojection past the 3 px threshold, occasionally landing on a degenerate
subset. Weighting via repeated points is a simple first cut; hand-rolled
weighted DLT can come later if needed.

**RANSAC watchlist:** If eval surfaces frames where a bad correspondence
clearly slipped through the gates and destabilized H, switch those cases to
`ransacReprojThreshold=3.0`. Do not enable RANSAC globally before seeing that
evidence — the p95 penalty on clean frames is too high.

### Change 2 — Iteration count

`K_max = 2`, not 3, as the working default. Rationale: under the noise-floor
measured in check 1, one re-solve with sub-pixel inputs already produces a H
that's within ~1-2 px at the corners. A second iteration with the now-better
H may add 1-2 more confirmed small stars but rarely changes the H itself.
Add a third iteration only if eval shows iteration 2 still adds stars.

Termination unchanged: stop if no new small stars confirmed, k ≥ K_max,
or `H_v(k+1) − H_vk` Frobenius-norm-equivalent (max corner displacement
< 0.5 px under the two H's).

### Change 3 — Constellation matching: greedy only

The original spec's "Strong case: affine-RANSAC backstop" is dropped from
scope. Pre-flight check 1 shows that as long as the 4 corners are in the
correspondence set, modest noise in star positions doesn't destabilize the
H solve, so a misassigned star is bounded in impact and will get rejected
by RANSAC at the next iteration. If eval surfaces a frame where greedy
clearly produces a wrong configuration, revisit.

## Pipeline (unchanged from original)

Per frame:

**Iteration 0 (initialization):**
- Detect screen corners → smooth → solve `H_v0 = H_rough_smoothed`.
- Detect big star globally → if found, anchor-translate `H_v0` by it → `H_v1`.
- If no big star, fall back to `H_v0` and flag the frame as anchor-less
  (lower confidence ceiling).

**Iteration k (k ≥ 1):**
- Predict all small-star positions using `H_vk` + behavioral log.
- Run local detector at each predicted position with adaptive window.
- Apply quality gates (next section).
- Build correspondence set per the table in *Change 1*.
- Re-solve homography → `H_v(k+1)`.
- Termination: stop if (a) no new small stars confirmed this iteration,
  (b) k ≥ K_max, or (c) `H_v(k+1)` is within ε of `H_vk`.

**Output:** `H_refined = H_vk_final`, plus the final correspondence list
with per-correspondence confidence and source.

## Quality gates on detections (unchanged)

A small-star detection is *accepted* only if all of:

1. **Distinct centroid** — centroid ≥ τ_centroid = 3 px from every other
   accepted detection.
2. **Equivalent radius matches expected** — within a factor of ~2.5
   (`τ_radius = 1.5`).
3. **Annular background check** — optional, only if 1+2 aren't enough.
   Mean R−B in an annulus at ~2× blob radius should be negative
   (surrounded by blue, not skin-tone).

**Conflict resolver** for same-blob snapping: group detections within
τ_centroid; the prediction whose center is closest to the shared centroid
wins; others → not-detected (do not look for a secondary peak in the same
window).

## Constellation-based assignment

- **Non-overlapping windows:** independent per-window detection as before.
- **Overlapping windows:** find all peaks in the union; greedy assign in
  descending confidence order to the nearest prediction within window radius;
  remove assigned peak/prediction; repeat.
- (Affine-RANSAC backstop dropped per Change 3 above.)

## Adaptive window sizing (unchanged)

`window_radius = base_factor × expected_radius_px + α × distance_from_anchor_px`
with `α ≈ 0.3` at iteration 1, tightening to ~0.1 at iteration 2 as H
improves.

## Evaluation

Same 7 frames from Phase 1b. Add 2-3 more if the spec frames don't expose
the failure modes:

- A frame with significant BR-corner glare (perspective residual large
  pre-iteration)
- A frame with the big star near a corner (anchor far from most small
  stars, so initial perspective residual is large)

Report **per-iteration**, not just final:

- **Detection rate by iteration** — monotonically increase or plateau.
- **Detection rate vs distance-from-anchor** — the key metric. Should be
  flat under `H_v_final` even if downward-sloping under `H_v1`.
- **Same-blob-snapping rate** — count rejections per iteration. Should
  drop sharply between iterations.
- **Homography reprojection error** per correspondence — most direct
  measure of "is H actually good now."
- **Side-by-side with Phase 1b** — using the per-frame CSV the existing
  local_star_eval emits.

Per-iteration overlays for ≥2 frames showing how predictions move between
iterations and which detections get added/rejected.

## File layout

- Extend `src/homography_refinement.py` with: `IterationResult` dataclass,
  `iterate_homography()` entry point, quality-gate helpers, weighted
  re-solver, greedy constellation matcher. Keep `anchor_translate` as-is
  (used at iteration 0).
- New `notebooks/iterative_homography_eval.py` (jupytext .py:percent
  format, matching `local_star_eval.py`).
- Pre-flight script: `scripts/preflight_phase1c.py` (this exists).

## Risks (updated)

- **Bottom corners not as reliable as assumed on occlusion frames.**
  Pre-flight check 2 confirmed: on hand-occlusion frame 1700, raw BL/BR
  deviate ~5–8 px from smoothed. The Change-1 outlier guard (`|raw − smoothed|
  > 10 px` → fall back to smoothed for that corner) handles this. Watch
  reprojection error on BR specifically.
- **Iteration could diverge if a bad detection slips through the quality
  gates.** Plausible mitigations: opt-in RANSAC in the re-solver
  (`use_ransac=True`, available but off by default per Change 1), and
  discarding any iteration whose H moves the bottom corners > 5 px from
  the smoothed value.
- **Configurations where iteration "works" but answer is wrong.** Same-blob
  snapping at iteration 0 could produce 7 "detections" all on the same
  big-star cluster. The conflict resolver should kill this; verify on the
  eval frames that it does.
- **Anchor unavailable.** If global detector misses the big star (mostly
  late-trial occlusion frames), iteration starts with no anchor. Plan:
  flag the frame and use the smoothed H_v0 alone — the small-star detector
  may still find a few stars to bootstrap H_v1.
- **Expected gain.** Realistically: from honest-3/7 → honest-5/7 or 6/7 on
  clean frames. If after iteration we're still at 3/7, something deeper is
  wrong — most likely the global big-star detector or screen corners, not
  the iteration logic.

## What success looks like

- Clean mid-trial frames: 6–7 of 7 expected small stars detected with
  distinct centroids; sub-pixel offsets matching expected positions under
  `H_refined` to within 2 px; reprojection error on the big star ≤ 1 px.
- Detection rate independent of distance-from-anchor.
- On hand-occluded frames: detect the stars that aren't covered; cleanly
  report the rest as not-detected. Refined H still accurate because
  4–6 visible stars + 2 corners is plenty.

If we hit this, downstream homography solving is essentially trivial —
`H_refined` *is* the homography we want.

---

## Original spec (verbatim, for traceability)

### Context

Phase 1b built a position-conditioned local detector for small stars, with
three H_rough modes (raw / smoothed / smoothed+anchor). Evaluation on
representative frames showed that even under smoothed+anchor — the best
mode — only ~3 of 7 expected small stars are honestly detected on clean
frames, and 2 of those 3 detections are same-blob-snapped to the big-star
cluster rather than locking onto distinct small stars. The remaining 4
misses are the 4 predictions farthest from the anchor.

The diagnosis: anchor-translate corrects H_rough's translation error (which
came from a corner-detector bias) but leaves a 5–10 frame-px perspective
residual, because the *perspective* component of H_rough is determined by
the relative configuration of the 4 detected corners, and the corner
detector has known systematic biases (e.g., a glare patch pushing BR
outward on EC347). The constellation of small-star predictions therefore
has approximately the right shape but is squished/sheared by a few percent
relative to ground truth.

This phase implements iterative homography refinement: use confident
detections to re-solve H, then re-predict and re-detect under the better
H, repeating until convergence. The output is both a better set of star
correspondences and a better homography, which together unblock the
downstream homography-solver phase.

### Scope

Build:

1. **Iteration controller** — orchestrates the predict → detect → re-solve
   → re-predict loop, including convergence/termination logic.
2. **Quality gates on detections** — same-blob-snapping detector,
   equivalent-radius vs expected-radius check, finger-FP filter via
   annular blue coverage or analogous.
3. **Homography re-solver** — weighted least-squares (or RANSAC)
   homography from a variable-size correspondence set: 2 trusted corners
   + big star + confirmed small stars.
4. **Constellation-based assignment** — greedy or Hungarian matching with
   the relative-position prior from the behavioral log, used when window
   enlargement causes prediction-window overlap.
5. **Per-frame output** with the refined H and the full set of
   correspondences for downstream use.

Do **not** in this phase:

- Modify the corner detector or screen brightness detector
- Modify the global big-star detector
- Modify the local opponent-channel detector itself, beyond changing how
  its inputs (window centers, window sizes) are computed
- Build the downstream gaze-projection pipeline
- Add temporal smoothing across frames within a trial (that's Phase 1d
  or 2)

### Homography re-solver (original recommendation — see Change 1)

Inputs: list of (screen_xy, frame_xy, confidence) correspondences.

- 2 bottom corners → confidence 1.0 (trusted per existing pipeline).
- Top corners → **excluded by default** (the systematic biases are the
  original problem). Include only as a low-confidence fallback if total
  correspondences < 4 after detection.
- Big star → confidence from global detector.
- Small stars → confidence from local detector, downweighted if
  `equivalent_radius_px` deviates from `expected_radius_px`.

If RANSAC rejects more than ~30% of correspondences as outliers, flag the
frame as low-quality and consider falling back to H_anchored from
iteration 0.
