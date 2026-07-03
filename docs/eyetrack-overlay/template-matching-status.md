# iPad screen detection — current status

## Goal
Per-frame homography mapping iPad screen-pixel coordinates (2388×1668, landscape) → scene-video frame coordinates (1920×1080). Used to project eye-tracking gaze onto known screen content. Only frames *during trials* matter (inter-trial intervals can be skipped).

## Recording setup
- Head-mounted Tobii scene camera, 1920×1080 @ 24.95 fps, 21 min 17 s total
- 11" iPad Pro M2 (landscape, native 2388×1668), lying on a table in a near-dark room
- Subject is a child running a guessing-game task
- Camera POV shifts constantly with head motion → per-frame homography required
- iPad itself may shift slightly when pressed

## Stimulus content on the iPad
- **Background:** a static diagonal gradient image (purple top-right → blue bottom-left), drawn onto an HTML canvas at full-screen size. The canvas does *not* fill the entire iPad screen — there's ~220 device-px white padding on the left and right (from `box_width=100` CSS px, DPR=2) and ~17 px top/bottom (from `screen_fill=0.98`). Body background is white.
- **Dots:** yellow stars (one initially, more added every ~1s up to ~12) at known screen-pixel coordinates.
- **Photodiode sync box:** small white rectangle (`position:fixed`, top-left, 100×200 CSS px = 200×400 device px) that toggles black/white as sync events fire.
- **Physical photodiode device:** a dark-cased hardware device clamped onto the iPad over the sync box. Occludes a region we estimate at roughly 0–350 × 0–550 in screen-pixel coords (top-left, padded). The clamp extends past the iPad bezel.

## Template we've built
`data/template_padded_2388x1668.png`: 2388×1668 image, white background, with the gradient image stretched into the central 1948×1635 sub-rectangle at (220, 16). This matches the actual iPad layout when rendered (verified visually against a video frame).

## Existing baseline detector (brightness)
`src/screen_detection.py`: grayscale → threshold @ 50 → morph close (51×51) → convex hull → `approxPolyDP` until 4 vertices → order corners [TL, TR, BR, BL]. Works on a fraction of frames; fails (returns None) or under-shoots when hand occlusion makes the bright region non-rectangular. A separate agent is implementing temporal smoothing on top of this.

## Current strategy: brightness seed → ECC alignment with masked occlusion zones

Per-frame pipeline:

1. Brightness detector → 4-corner seed in frame-coords.
2. `H_init = getPerspectiveTransform(template_corners, seed_corners)`.
3. Define a "top-of-iPad" occlusion zone in screen-coords: `(0, 0, 2388, 900)` — the entire top ~54% of the screen, where both photodiode and hand occlusion can live.
4. Project the zone through `H_init` to get a quad in frame-coords.
5. **Dilate the projected zone asymmetrically** around its centroid: top corners pushed outward by 2.5×, bottom corners 1.0× (no change). Rationale: the brightness seed mainly under-shoots the *top* corners because both the photodiode and the hand pull the detected bright region downward. Bottom corners are usually accurate.
6. Compute residual `|warped_template_gray - frame_gray|` at `H_init`, Gaussian-blur with a 11×11 kernel.
7. Build `inputMask`: 0 (ignore) where `(inside dilated zone) AND (residual > 40)`; 255 elsewhere.
8. Run `cv2.findTransformECC(template_gray, frame_gray, H_init, MOTION_HOMOGRAPHY, mask=mask)` at 0.5× scale.

Source: `scripts/feasibility_ecc.py`. Debug renders: `results/ec348_feasibility/`.

## Where this lands on 3 sample frames

| t | brightness seed | ECC cc | ECC corner shift from seed | Visual assessment |
|---|---|---|---|---|
| 120s | OK but a bit under-shot | 0.94 | 45 px | Top edge still not at true iPad top; geometry close to seed |
| 240s | OK | **0.97** | 44 px | Clean fit, looks correct |
| 600s | detector returned None | — | — | Skipped |
| 900s | under-shot, hand crossing most of top | **failed** | — | After most recent settings: "Iterations do not converge / Images may be uncorrelated or non-overlapped" |

t=900s converged before we tightened the threshold; tightening pushed it over the edge into non-convergence.

## What we've tried, in order

1. **No mask, ECC alone.** ECC shrinks the template away from the dark photodiode device → green quad lands *inside* the iPad face, missing the top portion. cc=0.94 but visually wrong.
2. **Fixed photodiode mask** (rectangular, projected from screen-coords (0,0,350,550) via `H_init`). Fixes the photodiode-specific failure on t=240s (perfect). Doesn't help with hand occlusion on t=120s/900s.
3. **Globally aggressive dynamic residual mask** (mask any pixel anywhere with high template/frame mismatch). Catastrophic: the bezels get masked because of small H_init misregistration, ECC has no signal left, throws "non-overlapped" error on all frames.
4. **Residual mask restricted to fixed zones** (PD + HAND, both projected from screen-coords, not dilated). Helps but the HAND zone derived from a seed that under-shot the iPad doesn't cover where the hand actually is. t=900s still fails.
5. **Same as #4 but with symmetric uniform 1.5× dilation of projected zones.** Slight improvement.
6. **Asymmetric dilation: top 2.2× / bottom 1.2×.** t=900s improves to 52 px shift (was 124 px), top edge near the actual iPad top. Still not great.
7. **Single combined top-zone (0,0,2388,900) + asymmetric 2.5×/1.0× dilation + lower residual threshold (60→40).** t=120s and t=240s stay decent, t=900s falls off the cliff into non-convergence — the more aggressive mask removes too much signal.

## Failure modes that keep surfacing

- **Template mismatch warps the template.** Anything in the frame that doesn't match the template (photodiode device, hand, hair, glare) creates a localized high-residual region. ECC's gradient descent finds an H that warps the template *away* from those regions, shrinking it inward. This is the dominant failure mode.
- **Seed dependency.** The brightness detector under-shoots when hand occlusion makes the bright region irregular. Zones derived from the seed are correspondingly under-shot. Dilation helps but doesn't eliminate the dependency.
- **Mask too aggressive → ECC has insufficient signal.** If the combined mask (zone × residual threshold) removes enough of the bezel/gradient cues, ECC can't converge.
- **Masking the gradient interior loses orientation information.** The diagonal gradient is what gives ECC sub-pixel positional accuracy; masking the top half removes much of that signal.

## Open questions / branches worth exploring

1. **Is ECC the right tool?** Alternatives: feature matching (ORB/SIFT) with RANSAC homography — naturally rejects occluders as outliers, doesn't need a "zone." But the iPad gradient is smooth and may have few distinctive features for ORB to lock onto.
2. **Should we abandon the seed entirely?** The brightness detector is fragile. Options: (a) use a constant default H_init (assuming roughly centered iPad), (b) use feature matching for initialization, (c) deeply over-shoot the bounding box and let ECC find the actual fit.
3. **Iterated/RANSAC-style ECC.** Run ECC, compute residual on the result, mask outliers, re-run. This is essentially robust regression but more expensive.
4. **Smoothing as the answer, not per-frame perfection.** Accept that ~20% of frames will have bad fits, detect them (low cc, large jump from neighbors, degenerate quad shape), reject and interpolate from clean frames. The parallel agent's smoothing pass on the brightness detector might already be enough.
5. **Downstream accuracy requirement.** What's the eye-tracking pixel-error budget? If it's ~50 px, the current approach is probably already good enough for most frames; if it's ~5 px, we need a fundamentally more accurate method.

## Files
- `scripts/build_screen_template.py` — builds the template from `purp.png`
- `scripts/feasibility_ecc.py` — current implementation
- `src/screen_detection.py` — brightness-based detector (existing, being smoothed in parallel)
- `data/template_padded_2388x1668.png`, `data/template_full_2388x1668.png` — templates
- `data/purp.png` — source gradient (800×960, served at `static/imgs/purp.png` by the task)
- `results/ec348_feasibility/` — debug renders (per-frame quads + mask overlays)
