# Phase 1b — small-star detection via position-conditioned local search

## Status at handoff

- **Branch:** `small-star-local-search`. Reset to current `main` (which contains
  the merged eyetrack PR #1 — Phase 1a). No code committed yet for Phase 1b.
- **Worktree:** `/Users/jon/.superset/worktrees/8cf2b0ca-f274-4f35-82b2-46edc185b2f7/small-star-local-search/`.
  `data/` and `results/` are symlinks to the primary worktree
  (`/Users/jon/Projects/dot-prediction/`). `config.local.yaml` is copied
  (gitignored). `vendor/EMU_data_collection` submodule is initialised.
- **What's been done this session:** orientation + one scratch frame render
  exposing an H_rough quality problem (see "Open empirical questions" below).
- **Where to resume:** the empirical pre-flight step. Do not write src code
  until visibility + size models are established. Scratch frames in
  `/tmp/star_eval/`.

## Goal (recap)

Phase 1a's global yellow-blob detector reliably finds the big (most-recently
revealed) star but misses small (shrunken, previously-revealed) stars. With
only 1 detection per frame and 3 screen corners (TL, TR, BR — BL often
occluded), we lack the 4 correspondences for a homography solve. This phase
adds **position-conditioned local search**: for each star the behavioral log
says should be on screen at time t, look in a small window around its
H_rough-predicted location.

**This phase remains detection-only.** No homography solving, no integration
with downstream pipeline. The goal is to go from "1 star detected per frame"
to "most stars present detected per frame," then evaluate.

The original full spec from the user is reproduced below in "Original spec"
verbatim — read it. This document layers what I learned on top.

## What Phase 1a provides

| File | Purpose |
|---|---|
| `src/star_detector.py` | `detect_stars(frame) → [(x, y, radius)]`. Global yellow-blob detector. Stable — do **not** modify. |
| `src/screen_detection.py` | `detect_corners(frame) → np.ndarray(4,2)` in `[TL, TR, BR, BL]` order, or `None`. |
| `src/corner_smoother.py` | `smooth_corners(list_of_per_frame_corners) → (n,4,2)`. Linear interp + rolling median. |
| `src/star_matcher.py` | `match_stars(detections, screen_stars, H_rough, search_radius)` — projects screen→frame and greedy-NN matches. Useful prior art. |
| `notebooks/align_video.py` | Interactive video↔behavior alignment picker. Writes `results/{subject}/video_alignment.json` and `trials_with_video.parquet`. |
| `notebooks/star_detection_eval.py` | Phase 1a's eval notebook. Pattern to follow. |
| `Snakefile_eyetrack`, `config_eyetrack.yaml` | Separate Snakemake DAG for eyetrack work. |

H_rough is **not** persisted to disk anywhere; the existing notebooks compute
it on the fly via `cv2.findHomography(SCREEN_CORNERS, detect_corners(frame))`.
For Phase 1b you can do the same. Whether to use raw vs smoothed corners is an
eval question (see "Detector design").

## Data

For subject `EC347`:

- Video: `data/EC347/tobii/scenevideo.mp4` (24.95 fps).
- Behavior raw: `data/EC347/behavior/data.csv`.
- Aligned trials: `results/EC347/trials_with_video.parquet` (300 rows, one per
  (trial, tpt)). Key columns:
  - `trial_idx`, `tpt`, `seq_id`
  - `true_x`, `true_y`: **normalised [0,1]** screen coords. Multiply by
    `(2388, 1668)` to get **iPad device pixels** (landscape; DPR=2 already
    applied — these are device-px, not CSS-px).
  - `reveal_time` (ms, expt clock — when the dot appeared)
  - `response_time` (ms, expt clock — when participant clicked this dot; NaN
    for the first 3 auto-revealed tpts of each trial, which are 0-1-2)
  - `trial_onset`, `trial_offset`, `end_of_auto_reveal`
  - `video_t_reveal_s`, `video_frame_reveal` — the alignment-mapped values
- Alignment: `results/EC347/video_alignment.json`. RMS residual = 12.7 ms over
  18 anchors → frame mapping is tight, can be trusted. Use
  `expt_t_ms = slope * (frame_idx / fps) + intercept`
  with `slope ≈ 1000.0325`, `intercept ≈ 303290.7`.

Trial 1 example (15 tpts, mid-experiment):

| tpt | reveal_time (ms) | response_time | video_frame |
|---|---|---|---|
| 0 | 329680 | NaN (auto) | 659 |
| 1 | 331182 | NaN (auto) | 696 |
| 2 | 332683 | NaN (auto) | 733 |
| 3 | 342490 | 340988 | 978 |
| … | … | … | … |
| 14 | 390824 | 389321 | 2184 |

Trial 1 onset = 328678 ms, offset = 391825 ms, `end_of_auto_reveal` = 332683
ms (= tpt 2's reveal_time).

Note `response_time < reveal_time` for tpts 3+ — `response_time` is the click
that *triggers* the next reveal, so the row's `response_time` is for the
*previous* dot. (Verify this; based on dataframe structure but I haven't
confirmed against task source.)

## Open empirical questions (do these FIRST)

### Q1: Visibility model

The spec says "previously-revealed stars are still visible but shrunk." But
the exact rule isn't documented. Candidate rules:

- **A**: all dots in the current trial that have been revealed by frame_t are
  visible until trial_offset.
- **B**: all dots revealed-but-not-yet-clicked. (Once clicked, gone.)
- **C**: time-decay lifetime — visible for N seconds after reveal then fade
  out.

**How to settle**: render 3 mid-trial frames, overlay the candidate rule's
predicted positions on top of the actual frame, see which visually matches the
faintly-visible small stars on the iPad.

The task UI source isn't in `vendor/EMU_data_collection` (that submodule is
ECoG-only). Asking the user or reading another repo may be the fastest path
if visual inspection is ambiguous. Probably ask Jon if you can't tell.

### Q2: Size-vs-age function

For each visible star, what's its expected radius in screen-px (and therefore,
after H_rough, frame-px)?

Spec claims: newest ~20–30 screen-px, oldest ~10 screen-px. Establish the
empirical curve by measuring stars in a few mid-trial frames and plotting
radius vs (frame_t − reveal_t).

### Q3: H_rough quality — pre-flight observation worth heeding

I rendered frame 1300 (mid-trial 1) and overlaid all 6 revealed-so-far stars
as predicted by `cv2.findHomography(SCREEN_CORNERS, detect_corners(frame))`.
The predictions were **all clustered together on the right side of the iPad
screen**, while the actually-visible stars (and the big-star detection) were
near the center-left. See `/tmp/star_eval/f01300_all_revealed.jpg`.

Hypothesis: the per-frame `detect_corners` returns a polygon that extends past
the iPad on the right side due to a reflection/glare patch that exceeds the
brightness threshold. That distorts the homography badly.

**Implications:**

- Don't trust raw `detect_corners` output blindly. Sanity-check the corners
  before using H_rough.
- Consider using **smoothed** corners from `corner_smoother.smooth_corners(...)`
  applied over a window of frames around the target — much more robust to
  per-frame outliers.
- The eval's "detection rate per H_rough quality" axis was already in the spec
  for a reason; you'll likely find that smoothed corners are required for
  Phase 1b to be useful at all.

Recommended order: build the detector against smoothed-corner H_rough first
(treat raw corners as a sub-mode for measuring fragility). The existing eval
notebook pattern can drive the smoothed pipeline:

```python
# Approximate sketch
raw_corners = []
for fi in range(n_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); _, frm = cap.read()
    raw_corners.append(detect_corners(frm))
smoothed = smooth_corners(raw_corners)  # (n_frames, 4, 2)
# Per frame: H_rough = cv2.findHomography(SCREEN_CORNERS, smoothed[fi])
```

Caveat: smoothing 20k+ frames is slow. Either cache it
(`results/EC347/smoothed_corners.npy`) or only compute over the window needed
for eval.

## Recommended module structure

(Per advisor input + the [[isolate new analyses]] memory rule — no touching of
`star_detector.py`.)

- `src/predicted_positions.py` — pure function
  `(frame_t_ms, trials_df, H_rough) → list[PredictedStar]` where
  `PredictedStar` has `(screen_xy, frame_xy, expected_radius_px, age_s, trial_idx, tpt)`.
  Encapsulates the visibility + size model from Q1/Q2.
- `src/local_star_detector.py` — pure function
  `(frame, predictions, window_size_px) → (list[LocalDetection], list[unmatched_predictions])`.
  Each `LocalDetection` has `(frame_xy_subpix, confidence, source_prediction)`.
  Reuses the `redness = R − B` opponent-channel logic from `star_detector.py`
  but skips the morphological closing and min-area filter — at 40×40 scale
  there's not enough context for either to help; the position prior carries
  the discrimination.
- `tests/test_predicted_positions.py`, `tests/test_local_star_detector.py` —
  fixture-based, like `tests/test_star_detector.py`.
- `notebooks/local_star_eval.py` — combines everything; emits eval CSV +
  overlay images per the eval framework below.

**No Snakemake rule.** Spec is explicit: detection-only, no downstream
integration.

## Detector design (after empirical phase)

- Window default **40×40 frame-px** centered on H_rough-predicted xy.
- Use the `R−B` opponent channel from `star_detector.py`. Skip
  `cv2.GaussianBlur(_, sigma=30)` background subtraction inside the window —
  too little context. Either no background subtraction, or use a small
  σ ≈ 5 within-window.
- Threshold: very permissive — the position prior is doing the
  discrimination. Suggest starting at `redness > 20` (vs 40 globally) and
  tune.
- **Sub-pixel centroid**: intensity-weighted average over pixels above the
  floor, not argmax. Critical for tiny stars.
- **Confidence score**: peak opponent-channel value within the window,
  optionally normalised by expected-radius area so a 4-px hit at a 4-px
  prediction site scores higher than at a 15-px prediction site.
- **Size-aware rejection**: reject if detected blob area >> expected area
  derived from `expected_radius_px`. This is the finger-occlusion case.
- **Out-of-frame guard**: if predicted xy falls outside `[0, W) × [0, H)`,
  return not-detected without indexing.
- **Window-overlap flag**: if the same peak pixel wins for two different
  predictions, flag (it usually means H_rough is significantly off).

## Eval framework

Pick ~10 frames spanning the difficulty range (from the original spec):

1. Mid/late-trial, clean, many small stars
2. Mid/late-trial with hand occlusion over part of the iPad
3. Early-trial: big star + 1–2 small ones
4. A frame where H_rough's top corners are noticeably under-shot
5. A frame near a head-motion jump
6. A frame with no stars visible (verify empty-return graceful)

Render overlays showing: predicted positions, detection windows, detected
centroids, unmatched predictions, source (global vs local).

Save eval stats to CSV columns: `frame_idx`, `video_t`, `trial_idx`, `tpt`,
`age_s`, `expected_radius_px`, `predicted_frame_x/y`, `detected_frame_x/y`,
`confidence`, `source` (`global`/`local`/`none`), `H_rough_mode`
(`raw`/`smoothed`).

Report metrics from the spec:
- Detection rate per star age bucket
- Detection rate per H_rough quality (raw vs smoothed; clean vs occluded)
- False-positive rate inside windows (manual labelling required)
- Centroid stability across consecutive frames on a stationary star

## Constraints from memory (CLAUDE auto-memory)

- Use `uv run python …`, not `.venv/bin/python` or `python3`.
- New analyses go in **new** files (don't mutate `star_detector.py` etc.).
- User prefers terse work, no superpowers skill invocation at session start.
- `subtrial_onset = reveal` in this codebase (just FYI for any cross-reference
  to ECoG-side notebooks; not directly relevant to Phase 1b).
- EC347 video alignment: the empirical intercept is ~303,300 ms; the TSV
  header start time is unreliable. The committed `video_alignment.json`
  already reflects this.

## Pre-flight checks status

- ✓ Video↔behavior alignment exists and is tight (RMS 12.7 ms).
- ✓ Behavior log columns + DPR convention understood (`true_x * 2388`).
- ✗ H_rough quality — **observed to be bad on at least one mid-trial frame**;
  smoothed corners likely required (see Q3).

## Where to resume

1. Render a couple more mid-trial frames with the same scratch script
   pattern, this time also rendering the smoothed-corners overlay alongside
   raw-corners — confirm smoothed fixes the H_rough problem.
2. With smoothed H_rough working, do the visibility-rule comparison (Q1) and
   size measurement (Q2). The user can probably answer Q1 quickly if visual
   inspection is ambiguous.
3. Call the advisor with empirical findings before writing `src/predicted_positions.py`.

## Scratch artifacts

- `/tmp/star_eval/f01300_all_revealed.jpg`, `f01900_all_revealed.jpg`,
  `f02150_all_revealed.jpg` — overlay rendered with "all-revealed-so-far"
  candidate visibility rule + per-frame `detect_corners` (i.e. raw H_rough,
  which we now believe is unreliable). Frame 1300 shows the problem
  clearly — predicted positions clustered on right edge, big star detected
  in center.

The scratch script used to produce them is in the conversation transcript;
re-pasting here for convenience:

```python
import cv2, json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')
from screen_detection import detect_corners
from star_detector import detect_stars

trials = pd.read_parquet('results/EC347/trials_with_video.parquet')
align = json.loads(open('results/EC347/video_alignment.json').read())
slope, intercept = align['slope_ms_per_s'], align['intercept_ms']
cap = cv2.VideoCapture('data/EC347/tobii/scenevideo.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
SCREEN = np.array([[0,0],[2388,0],[2388,1668],[0,1668]], dtype=np.float32)

for fi in [1300, 1900, 2150]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); _, frame = cap.read()
    expt_t = slope * fi / fps + intercept
    t_in = trials[(trials.trial_onset <= expt_t) & (trials.trial_offset >= expt_t)]
    trial_idx = int(t_in.iloc[0].trial_idx)
    revealed = trials[(trials.trial_idx == trial_idx) & (trials.reveal_time <= expt_t)]
    corners = detect_corners(frame)
    blobs = detect_stars(frame)
    vis = frame.copy()
    if corners is not None:
        cv2.polylines(vis, [corners.astype(np.int32).reshape(-1,1,2)], True, (0,200,0), 2)
        H, _ = cv2.findHomography(SCREEN, corners)
        for _, row in revealed.iterrows():
            p = (H @ np.array([row.true_x*2388, row.true_y*1668, 1.0])).reshape(3)
            fx, fy = p[0]/p[2], p[1]/p[2]
            age_ms = expt_t - row.reveal_time
            age_norm = min(age_ms / 30_000, 1.0)
            col = (int(255*age_norm), 0, int(255*(1-age_norm)))
            cv2.drawMarker(vis, (int(fx), int(fy)), col, cv2.MARKER_TILTED_CROSS, 24, 2)
            cv2.putText(vis, f"t{int(row.tpt)}/{int(age_ms/1000)}s", (int(fx)+8, int(fy)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    for bx, by, br in blobs:
        cv2.circle(vis, (int(bx), int(by)), max(int(br),6), (0,255,255), 2)
    cv2.imwrite(f'/tmp/star_eval/f{fi:05d}_all_revealed.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
cap.release()
```

---

## Original spec (verbatim, for reference)

Phase 1b — small-star detection via position-conditioned local search

### Context

Phase 1a built a global yellow-blob detector that reliably finds the large
(most-recently-revealed) star but misses the small (shrunken,
previously-revealed) stars. Targeting only the big star leaves us with
insufficient correspondences for a homography solve (need 4, have 3 with
bottom corners). This phase extends detection to the small stars using a
position-conditioned local search, leveraging the now-available video↔behavioral-log
alignment.

The video↔behavioral-log alignment means: for any frame timestamp, we can
look up from the behavioral log exactly which stars should be visible on the
iPad at that moment, and what their screen-pixel coordinates are. Combined
with the rough homography H_rough from the existing brightness detector, we
can predict where each star should appear in frame coordinates.

This phase remains detection-only. No homography solving, no integration with
downstream pipeline. The goal is to get from "1 star detected per frame" to
"most of the stars present detected per frame," then evaluate.

### Key insight

A global yellow-blob detector cannot distinguish a 3–5 frame-px star from
sensor noise or stray gradient artifacts without false positives. A
position-conditioned detector — one that searches only in a small window
around a predicted star location — can use a much more permissive threshold
because the position prior is doing most of the discrimination work. We don't
need to find small stars anywhere in the frame; we need to confirm or deny
their presence at a small number of specific predicted locations.

### Scope

Build:

- **Predicted-position generator**: for a given frame, takes the frame
  timestamp + behavioral log + H_rough, returns a list of
  `(screen_xy, predicted_frame_xy, expected_size)` for every star that should
  be visible in that frame.
- **Local star confirmer**: for each predicted frame_xy, searches a small
  window around the predicted location and returns either a refined
  detection `(frame_xy, confidence)` or a "not detected" verdict.
- **Combined output**: for each frame, a list of
  `(screen_xy, frame_xy, confidence, source)` where source is either
  `"global"` (from the existing big-star detector) or `"local"` (from the
  new conditioned detector), plus a list of expected-but-not-detected stars.

Do not in this phase:

- Solve homographies
- Filter detections against the finger-FP problem from Phase 1a beyond what
  the position prior naturally gives us (the position prior should mostly
  handle it — fingers don't appear at predicted star locations except when
  occluding a star, in which case the right answer is "not detected" anyway)
- Modify the global big-star detector; treat it as a stable input

### Design notes

**Window size.** H_rough has known error characteristics — the top corners
can be off by tens of frame-px under hand occlusion, the bottom corners are
usually good. The window size should be large enough to cover H_rough's
prediction error but small enough that the position prior remains
discriminative. Suggest starting at ±20 frame-px (40×40 window) and tuning.
Stars near the top of the iPad may need larger windows than stars near the
bottom, given the asymmetric H_rough error pattern; worth checking but don't
pre-optimize.

**Detection within window.** Reuse the opponent-channel approach from
Phase 1a (`(R+G)/2 − B` or whatever you settled on) but with a much lower
threshold. Within a 40×40 window centered on a predicted star, "the
brightest yellow-ish spot above a low floor" is likely the star. If no pixel
exceeds the floor, the star is occluded or off-screen — report not-detected.

**Expected size matters.** The behavioral log tells you roughly how long ago
each star was revealed, which determines its size. The newly-revealed star
is ~20–30 screen-px; older stars shrink to ~10 screen-px. Translating
screen-px to frame-px via H_rough gives an expected blob size. Use this:

- as a sanity check (reject candidates much larger than expected — that's
  probably a finger)
- to set scale-appropriate DoG / smoothing parameters if you use them
- to weight confidence (a 4-px blob found where we expect a 4-px star is
  more trustworthy than a 4-px blob found where we expect a 15-px star)

**Sub-pixel centroid.** Once a star is found within a window, compute its
centroid by intensity-weighted average of the opponent-channel values, not
just the argmax pixel. This matters for the eventual homography accuracy —
small stars give noisy centroids, and any sub-pixel refinement helps.

**Confidence score.** Report something interpretable per detection. The
opponent-channel peak value within the window, normalized somehow, is a
reasonable start. Downstream RANSAC for the homography will use this to
weight correspondences.

**Edge cases worth handling:**

- Predicted position outside the frame bounds (H_rough is wrong, or the star
  is genuinely off-screen) — report not-detected, don't crash.
- Two predicted positions whose windows overlap — fine, just search each
  independently, but flag if the same pixel is the peak for two different
  predictions (probably means H_rough is significantly off).
- The behavioral log indicating zero stars visible (early-trial pre-first-star,
  or inter-trial intervals) — return empty list.

### Evaluation

Use roughly the same evaluation framing as Phase 1a, with the spans of
difficulty extended:

- **Detection rate per star age.** For a clean mid-trial frame (no occlusion,
  H_rough good), what fraction of the stars predicted to be visible are
  detected? Break this down by star age / size — we expect the
  most-recently-revealed star to be ~100%, and detection rate to decay for
  older shrunken stars. We want to know where the cliff is.
- **Detection rate per H_rough quality.** On frames where H_rough is known to
  be bad (the analogues of the t=900s case), what's the detection rate? If
  the window is too small to cover H_rough's error, detection rate
  collapses; this tells us whether window size needs to be adaptive or
  whether we need a better H_rough.
- **False positive rate inside windows.** Within a 40×40 window centered on
  a predicted star, how often does the local detector lock onto something
  that isn't a star (a finger crossing through that exact location, glare,
  gradient artifact)? This should be low because the position prior is
  strong, but worth measuring.
- **Centroid stability across consecutive frames.** Pick a star that's
  stationary on the iPad for many frames within a trial. How much does its
  detected frame-coord centroid jitter frame-to-frame? This is the
  per-correspondence noise that will feed into the homography solve.

Pick ~10 frames spanning the difficulty range:

- Mid/late-trial, clean, many small stars visible
- Mid/late-trial with hand occlusion over part of the iPad
- Early-trial with just the big star + 1–2 small ones
- A frame where H_rough's top corners are noticeably under-shot
- A frame near a head-motion jump
- A frame with no stars visible (verify the system gracefully returns empty)

Render overlays showing: predicted star positions (from H_rough +
behavioral log), detection windows, detected centroids, and unmatched
predictions. Save eval stats to CSV for manual labeling like in Phase 1a.

### Pre-flight checks

- Confirm the video↔behavioral-log alignment is accurate to within a frame
  or two. If it's off by a few seconds, predicted star positions will be
  wildly wrong (a recently-revealed star will be missing from predictions,
  or a predicted star will not yet exist). Spot-check on a few frames by
  eye before trusting the alignment.
- Confirm star coordinates from the behavioral log are in iPad screen-pixel
  coords (2388×1668 device-px). The CSS-px vs device-px DPR=2 issue from
  earlier still applies.
- Confirm we can read the existing per-frame H_rough output cleanly. *(Note:
  no such artifact is saved; compute on the fly per existing notebooks.)*
