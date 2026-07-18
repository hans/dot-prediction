"""Phase 1c pre-flight checks.

Validates 3 spec assumptions before building the iterative refinement loop:

1. ``cv2.findHomography`` stability with 4-10 correspondences under sub-pixel
   input perturbation. The iteration re-solves H from a variable-sized set
   each round; if the solve is numerically delicate at low N, the iteration
   will jitter rather than converge.

2. Bottom-corner trustworthiness — the spec proposes treating BR/BL corners
   as confidence 1.0 inputs to the re-solve. Verify on representative frames
   that the per-frame raw detection of BR/BL is close to the rolling-median
   smoothed value (a proxy for "ground truth"). Large deviations mean the
   bottom corners are not always trustworthy and weighting needs work.

3. Star-position stationarity within a trial — the constellation-matching
   step assumes the relative configuration of small stars in a trial is
   fixed once revealed. The behavior log structure makes this trivially
   true, but verify explicitly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

from corner_smoother import smooth_corners
from predicted_positions import SCREEN_H, SCREEN_W, predicted_positions
from screen_detection import detect_corners

SUBJECT = "EC347"
VIDEO = f"/Users/jon/Projects/dot-prediction/data/{SUBJECT}/tobii/scenevideo.mp4"
TRIALS = f"results/{SUBJECT}/trials_with_video.parquet"
ALIGN = f"results/{SUBJECT}/video_alignment.json"
OUT_DIR = Path(f"results/{SUBJECT}/preflight_phase1c"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same eval frames as Phase 1b, plus a couple of extras the spec calls out
# (BR-on-glare frame, big-star-near-corner frame). The frame numbers below
# come from inspection of Phase 1b candidate overlays; if the spec frames
# need different indices we can amend after seeing what these reveal.
EVAL_FRAMES = [659, 750, 1500, 1700, 1900, 2150, 2270]


# ---------------------------------------------------------------------------
# Check 1: findHomography numerical stability
# ---------------------------------------------------------------------------

def _project(H, pts):
    """Project (N, 2) screen points through 3x3 H, return (N, 2)."""
    ones = np.ones((len(pts), 1))
    proj = (H @ np.hstack([pts, ones]).T).T
    return proj[:, :2] / proj[:, 2:3]


def check1_findhomography_stability():
    """How much does H wobble under sub-pixel perturbation of inputs?

    Setup: pick a representative H (Phase 1b smoothed corners from frame 1500),
    project a configurable number of known screen points to get synthetic
    frame correspondences, add Gaussian noise to the frame side, re-solve,
    and measure the projection error of the 4 screen corners under the
    re-solved H. Report median / p95 over many noise realizations.

    Runs each N>=6 case under both method=0 (lstsq) and RANSAC
    (ransacReprojThreshold=3.0) so the stability conclusions match the actual
    production solver path used by solve_weighted_homography.
    """
    print("\n=== Check 1: findHomography stability under input perturbation ===\n")

    rng = np.random.default_rng(0)

    # Take a real H from frame 1500 of EC347 (Phase 1b reported clean
    # detection there). Use the bare smoothed corners — perfect input.
    cap = cv2.VideoCapture(VIDEO)
    # Build a tiny smoothing window
    fi_ref = 1500
    raw_window: list = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi_ref - 60)
    for _ in range(121):
        ret, frm = cap.read()
        raw_window.append(detect_corners(frm) if ret else None)
    cap.release()
    smo = smooth_corners(raw_window, window=51)
    ref_corners = smo[60]  # the corners at frame 1500

    screen_corners = np.array(
        [[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]], dtype=np.float64
    )
    H_true, _ = cv2.findHomography(screen_corners.astype(np.float32),
                                   ref_corners.astype(np.float32))

    # Star locations — sample N points across the screen in a roughly typical
    # constellation. 4 corner-only and {6, 10} extra-stars scenarios.
    extra_screen = np.array([
        [SCREEN_W * x, SCREEN_H * y]
        for (x, y) in [(0.30, 0.30), (0.70, 0.30), (0.50, 0.50),
                       (0.25, 0.70), (0.75, 0.70), (0.50, 0.25),
                       (0.20, 0.50), (0.80, 0.50), (0.35, 0.85), (0.65, 0.85)]
    ])

    cases = [
        ("4 corners only", screen_corners),
        ("4 corners + 6 stars", np.vstack([screen_corners, extra_screen[:6]])),
        ("4 corners + 10 stars", np.vstack([screen_corners, extra_screen[:10]])),
        ("6 stars only (no corners)", extra_screen[:6]),
        ("10 stars only (no corners)", extra_screen[:10]),
    ]

    # Production solver: lstsq when N<6, RANSAC(3.0) when N>=6.
    # Run both for N>=6 cases so the stability conclusions match reality.
    solvers = [
        ("lstsq", 0, None),
        ("RANSAC", cv2.RANSAC, 3.0),
    ]

    noise_sigmas_px = [0.25, 0.5, 1.0, 2.0]
    n_trials = 500

    proj_true = _project(H_true, screen_corners)

    print(f"{'Case':<28s} {'solver':<6s} {'σ_in':>6s}  {'med_corner_err':>14s}  {'p95_corner_err':>14s}  {'fail_rate':>9s}")
    for label, screen_pts in cases:
        n_pts = len(screen_pts)
        frame_pts_clean = _project(H_true, screen_pts)
        for solver_label, method, ransac_thresh in solvers:
            if method == cv2.RANSAC and n_pts < 6:
                continue
            for sigma in noise_sigmas_px:
                errs = []
                failures = 0
                for _ in range(n_trials):
                    noise = rng.normal(scale=sigma, size=frame_pts_clean.shape)
                    noisy = frame_pts_clean + noise
                    if ransac_thresh is not None:
                        H_solved, _ = cv2.findHomography(
                            screen_pts.astype(np.float32),
                            noisy.astype(np.float32),
                            method=method,
                            ransacReprojThreshold=ransac_thresh,
                        )
                    else:
                        H_solved, _ = cv2.findHomography(
                            screen_pts.astype(np.float32),
                            noisy.astype(np.float32),
                            method=method,
                        )
                    if H_solved is None:
                        failures += 1
                        continue
                    # Measure: re-project the 4 *true* screen corners under
                    # H_solved and compare with their true frame positions.
                    proj_under = _project(H_solved, screen_corners)
                    d = np.hypot(*(proj_under - proj_true).T)
                    errs.append(d.max())
                arr = np.array(errs)
                print(
                    f"{label:<28s} {solver_label:<6s} {sigma:>6.2f}  "
                    f"{np.median(arr):>14.3f}  {np.quantile(arr, 0.95):>14.3f}  "
                    f"{failures / n_trials:>9.1%}"
                )

    print(
        "\n  Interpretation: max corner re-projection error (px) under H_solved.\n"
        "  Sub-px input noise should give sub-px-ish corner error for 6+ point\n"
        "  cases; 4-pt should amplify noise but stay bounded. >2x amplification\n"
        "  suggests RANSAC or weighting is needed even on clean data.\n"
        "  RANSAC rows show how the production solver path behaves; compare with\n"
        "  lstsq rows to see whether inlier filtering helps or hurts under pure\n"
        "  Gaussian noise (no outliers) vs. the original lstsq results."
    )


# ---------------------------------------------------------------------------
# Check 2: bottom-corner trustworthiness on real frames
# ---------------------------------------------------------------------------

def check2_bottom_corner_trust():
    """Compare raw vs smoothed BR/BL corner detections on the eval frames."""
    print("\n=== Check 2: bottom-corner trustworthiness ===\n")

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lo = min(EVAL_FRAMES) - 60
    hi = max(EVAL_FRAMES) + 60
    print(f"Caching raw corners for frames [{lo}, {hi}] ({hi-lo+1} frames)...")
    raw: list = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for _ in range(lo, hi + 1):
        ret, frm = cap.read()
        raw.append(detect_corners(frm) if ret else None)
    smoothed = smooth_corners(raw, window=51)

    # Per-frame snapshot of (raw vs smoothed) for each of 4 corners
    label_idx = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
    rows = []
    for fi in EVAL_FRAMES:
        i = fi - lo
        raw_c = raw[i]
        smo_c = smoothed[i]
        if raw_c is None:
            rows.append(dict(frame=fi, corner="ALL", raw=None,
                             dx=np.nan, dy=np.nan, dist=np.nan))
            continue
        for name, k in label_idx.items():
            dx = float(raw_c[k, 0] - smo_c[k, 0])
            dy = float(raw_c[k, 1] - smo_c[k, 1])
            rows.append(dict(frame=fi, corner=name,
                             raw_x=float(raw_c[k, 0]), raw_y=float(raw_c[k, 1]),
                             smo_x=float(smo_c[k, 0]), smo_y=float(smo_c[k, 1]),
                             dx=dx, dy=dy, dist=float(np.hypot(dx, dy))))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "corner_raw_vs_smoothed.csv", index=False)
    print("Per-corner |raw - smoothed| (px) by frame:\n")
    pivot = df.pivot(index="frame", columns="corner", values="dist")
    if "ALL" in pivot.columns:
        pivot = pivot.drop(columns="ALL")
    print(pivot.round(2).to_string())

    print("\nPer-corner |raw - smoothed| (px) summary across all eval frames:\n")
    summ = df.groupby("corner")["dist"].agg(["count", "median", "mean", "max"]).round(2)
    print(summ)

    # Render an overlay of raw + smoothed BR/BL on each frame, with a zoom
    # crop around each bottom corner.
    cap2 = cv2.VideoCapture(VIDEO)
    for fi in EVAL_FRAMES:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap2.read()
        if not ret:
            continue
        i = fi - lo
        raw_c = raw[i]
        smo_c = smoothed[i]
        # Crops around BR and BL (60x60 px); show raw (red) and smoothed (cyan)
        crops = []
        for name in ["BL", "BR"]:
            k = label_idx[name]
            cx_s, cy_s = int(smo_c[k, 0]), int(smo_c[k, 1])
            half = 50
            x0 = max(0, cx_s - half); y0 = max(0, cy_s - half)
            x1 = min(frame.shape[1], cx_s + half); y1 = min(frame.shape[0], cy_s + half)
            crop = frame[y0:y1, x0:x1].copy()
            # Smoothed (cyan)
            cv2.drawMarker(crop, (cx_s - x0, cy_s - y0), (255, 255, 0),
                           cv2.MARKER_CROSS, 16, 2)
            if raw_c is not None:
                cx_r, cy_r = int(raw_c[k, 0]), int(raw_c[k, 1])
                if 0 <= cx_r - x0 < crop.shape[1] and 0 <= cy_r - y0 < crop.shape[0]:
                    cv2.drawMarker(crop, (cx_r - x0, cy_r - y0), (0, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 16, 2)
            # Caption
            cv2.putText(crop, f"f{fi} {name}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            crops.append(crop)
        # Resize to same h for hstack
        h_min = min(c.shape[0] for c in crops)
        crops = [c[:h_min, :, :] for c in crops]
        cv2.imwrite(str(OUT_DIR / f"corners_f{fi:05d}.jpg"),
                    np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap2.release()

    print(f"\n  Overlays in {OUT_DIR}/corners_f*.jpg")
    print(
        "  Interpretation: a few pixels of raw-vs-smoothed offset is fine\n"
        "  (it's the smoothing absorbing per-frame jitter), but if any BL/BR\n"
        "  raw detection is >10 px off the smoothed value on these frames,\n"
        "  the 'bottom corners always in' assumption needs an outlier guard."
    )


# ---------------------------------------------------------------------------
# Check 3: star stationarity within a trial
# ---------------------------------------------------------------------------

def check3_star_stationarity():
    """Verify the behavior log has stable (true_x, true_y) per (trial, tpt)."""
    print("\n=== Check 3: star-position stationarity within a trial ===\n")
    trials = pd.read_parquet(TRIALS)
    grp = trials.groupby(["trial_idx", "tpt"])
    sizes = grp.size()
    n_dup = int((sizes > 1).sum())
    print(f"Total (trial_idx, tpt) groups: {len(sizes)}")
    print(f"Groups with >1 row (potential moving stars): {n_dup}")
    if n_dup > 0:
        # Check coord variance within those groups
        spread = grp[["true_x", "true_y"]].std().fillna(0)
        spread["dist"] = np.hypot(spread["true_x"], spread["true_y"])
        nonzero = spread[spread["dist"] > 1e-9]
        print(f"  Groups with coord variation > 0: {len(nonzero)}")
        if len(nonzero):
            print(nonzero.head(10))
    else:
        print("  ✓ Each (trial, tpt) has exactly one row → stars are stationary by construction.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check3_star_stationarity()   # cheapest first
    check1_findhomography_stability()
    check2_bottom_corner_trust()
    print(f"\nDone. Outputs in {OUT_DIR}/")
