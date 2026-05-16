"""Detect yellow-star blobs in a Tobii scene-camera frame.

Stars appear as warm-coloured blobs (R > B) against the strongly blue/purple
gradient background (R < B).  Rather than hard absolute-HSV thresholds — which
are sensitive to exposure and the exact camera white-balance — the detector uses
local colour contrast:

  redness  = R_channel - B_channel          (warm = positive, blue = negative)
  local_bg = GaussianBlur(redness, σ=30)    (slow background estimate)
  contrast  = redness - local_bg             (how much warmer than surroundings?)

A star pixel sits on a strongly blue background, so its local contrast is very
large (~120–160).  Browser-chrome pixels (URL bar, favicons) sit on a neutral
white background whose local redness is near zero, giving contrast < 20 — a
10× separation that makes the threshold easy to set.

Detection pipeline
------------------
1. Compute per-pixel local R-B contrast (Gaussian background subtraction).
2. Candidate pixels: contrast > threshold AND redness > 0 (pixel is actually
   warm, not merely less-blue-than-surroundings).
3. Restrict to the "display content" region built by morphologically closing the
   blue-background mask — keeps out the dark desk and camera frame.
4. Run connected-components; return blobs above a minimum area.
"""

import cv2
import numpy as np

# Blue gradient background thresholds (HSV, OpenCV 0-180 hue scale).
_BLUE_H_LO: int = 90
_BLUE_H_HI: int = 145
_BLUE_S_MIN: int = 30
_BLUE_V_MIN: int = 50

# Morphological close kernel radius (px).  Must span the largest star hole
# (~22 px) so the blue mask covers the full display content region.
_CLOSE_RADIUS: int = 30

# Gaussian sigma for local background estimation.  Large enough to make the
# background estimate insensitive to the star itself (~10–20 px radius) while
# staying well within the scale of the blue content region.
_BG_SIGMA: int = 30

# Local R-B contrast threshold.  Stars on blue background: ~120–160.
# Browser-chrome (URL bar) on white background: ~7–20.
_CONTRAST_THRESH: float = 40.0

# Minimum blob area (px²).  Stars in frame coords: ~100–1700 px² depending on
# iPad distance.  Set well below 100 so the smallest visible star passes.
_MIN_AREA: int = 100

# Minimum fraction of blue pixels in the annular region [2r, 4r] around a
# candidate blob.  Stars sit on a solid blue background (≈1.0); fingers score
# lower because the dark finger body fills part of the annulus.
_MIN_BLUE_FRACTION: float = 0.5


def detect_stars(
    frame: np.ndarray,
    min_area: int = _MIN_AREA,
    min_blue_fraction: float = _MIN_BLUE_FRACTION,
) -> list[tuple[float, float, float]]:
    """Detect star blobs in a Tobii scene-camera frame.

    Args:
        frame: BGR image, uint8, shape (H, W, 3).
        min_area: Minimum blob area in pixels.
        min_blue_fraction: Minimum fraction of blue pixels (from the raw HSV
            mask, before morphological closing) in the annular region at
            [2r, 4r] from each candidate centroid.  Only pixels inside the
            display region count toward both numerator and denominator, so
            blobs near screen edges are not unfairly penalised.

    Returns:
        List of (x, y, radius) tuples in frame-pixel coordinates, one per
        detected blob. ``radius`` is the equivalent circle radius
        (``sqrt(area / π)``). Returns an empty list when nothing is found.
    """
    H_frame, W_frame = frame.shape[:2]

    # Step 1 — display-content mask via blue background
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        (int(_BLUE_H_LO), int(_BLUE_S_MIN), int(_BLUE_V_MIN)),
        (int(_BLUE_H_HI), 255, 255),
    )
    k_size = _CLOSE_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    display_region = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # Step 2 — local warm-colour contrast (R - B relative to local background)
    r = frame[:, :, 2].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    redness = r - b  # warm = positive, blue = negative
    local_bg = cv2.GaussianBlur(redness, (0, 0), _BG_SIGMA)
    contrast = redness - local_bg

    # Step 3 — candidate pixels: warm relative to local background AND
    # intrinsically warm (R > B), masked to the display content region
    warm_local = ((contrast > _CONTRAST_THRESH) & (redness > 0)).astype(np.uint8) * 255
    candidates = cv2.bitwise_and(warm_local, display_region)

    # Step 4 — connected components
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates, connectivity=8
    )

    ys, xs = np.ogrid[:H_frame, :W_frame]

    blobs: list[tuple[float, float, float]] = []
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        radius = float(np.sqrt(area / np.pi))

        # Blue-coverage check: annulus [2r, 4r] around the blob should be
        # predominantly blue in the raw mask (before MORPH_CLOSE fills holes).
        # Denominator is restricted to display_region so edge blobs aren't
        # penalised for annulus pixels that fall outside the screen content.
        inner_r = 2.0 * radius
        outer_r = 4.0 * radius
        dist = np.hypot(xs - cx, ys - cy)
        annular_mask = ((dist >= inner_r) & (dist < outer_r)).astype(np.uint8) * 255
        annular_in_display = cv2.bitwise_and(annular_mask, display_region)
        denom = np.count_nonzero(annular_in_display)
        if denom > 0:
            numer = np.count_nonzero(cv2.bitwise_and(blue_mask, annular_in_display))
            if numer / denom < min_blue_fraction:
                continue

        blobs.append((cx, cy, radius))

    return blobs
