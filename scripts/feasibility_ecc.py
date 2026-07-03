"""Feasibility check: does ECC alignment of the gradient template converge?

Pipeline:
  1. Read a sample frame at SAMPLE_T seconds (well past the t=60s laptop exit).
  2. Use the existing brightness detector to get a rough corner seed.
  3. Build an initial homography H_init that maps the padded template
     (2388x1668, screen-pixel coords) to the four detected video-pixel corners.
  4. Run cv2.findTransformECC with MOTION_HOMOGRAPHY to refine H.
  5. Save before/after visualizations + print convergence stats.

If ECC converges sensibly and refined corners visibly hug the screen edges
better than the seed, template matching is viable on this data.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.screen_detection import detect_corners

VIDEO = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
TEMPLATE = ROOT / "data" / "template_padded_2388x1668.png"
OUT = ROOT / "results" / "ec348_feasibility"

# Sample several frames so we get a feel for behavior across the recording
SAMPLE_TIMES_S = [120, 240, 600, 900]

# Downsample factor for ECC (both template and frame scaled the same amount).
# Smaller = faster but less precise. 0.5 is a reasonable starting point.
ECC_SCALE = 0.5

ECC_ITERS = 200
ECC_EPS = 1e-6

# Occlusion zone in iPad screen-pixel coords (landscape 2388x1668). The whole
# upper portion of the iPad is potentially occluded — by the photodiode device
# (always top-left) or the subject's hand (can reach across most of the top).
# Residual-based masking applies *only* inside this zone; outside it we trust
# the signal, so bezel-edge cues stay available for alignment.
TOP_OCCLUSION_ZONE = (0, 0, 2388, 900)   # full width, top ~54% of screen

# Residual threshold: pixels inside the zone with mismatch above this (vs. the
# warped template at H_init) get masked. Lower = more aggressive.
RESIDUAL_THRESHOLD = 40
RESIDUAL_BLUR_KSIZE = 11

# Brightness-seeded zones inherit the seed's errors. The seed mainly misses on
# the *top* corners — the photodiode and hand both pull the detected quad
# downward. So push the top edge of the projected zone outward aggressively;
# the bottom edge stays put so we don't mask the unoccluded lower iPad.
# Corner order from project_screen_rect is [TL, TR, BR, BL].
ZONE_DILATION_TOP = 2.5     # TL, TR — push way out
ZONE_DILATION_BOTTOM = 1.0  # BR, BL — leave alone


def read_frame(cap: cv2.VideoCapture, t_seconds: float) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_seconds * fps))
    ret, frame = cap.read()
    return frame if ret else None


def homography_template_to_frame(template_shape, frame_corners):
    """Solve H mapping template-pixel corners → frame-pixel corners."""
    H, W = template_shape[:2]
    template_corners = np.array(
        [[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(template_corners, frame_corners), template_corners


def project_screen_rect(rect_screen, H_init):
    """Project a (x, y, w, h) rectangle in screen-pixel coords through H_init
    to get its quadrilateral in frame-pixel coords as int32 (4, 2)."""
    x, y, rw, rh = rect_screen
    pts = np.array(
        [[[x, y]], [[x + rw, y]], [[x + rw, y + rh]], [[x, y + rh]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(pts, H_init).reshape(-1, 2).astype(np.int32)


def dilate_quad_asymmetric(quad, top_factor, bottom_factor):
    """Scale a quad outward from its centroid. Corners assumed to be in
    [TL, TR, BR, BL] order: TL/TR are pushed by `top_factor`, BR/BL by
    `bottom_factor`. Returns int32 (4, 2)."""
    quad_f = quad.astype(np.float32)
    centroid = quad_f.mean(axis=0)
    factors = np.array(
        [top_factor, top_factor, bottom_factor, bottom_factor], dtype=np.float32
    )[:, None]
    return (centroid + (quad_f - centroid) * factors).astype(np.int32)


def build_occlusion_zone(frame_shape, H_init):
    """Binary frame-coord mask, 255 inside the dilated projected top-occlusion
    zone, 0 elsewhere. Residual masking applies only inside this zone."""
    h, w = frame_shape[:2]
    zone = np.zeros((h, w), dtype=np.uint8)
    projected = project_screen_rect(TOP_OCCLUSION_ZONE, H_init)
    dilated = dilate_quad_asymmetric(
        projected, ZONE_DILATION_TOP, ZONE_DILATION_BOTTOM
    )
    cv2.fillPoly(zone, [dilated], 255)
    return zone


def build_constrained_residual_mask(template_bgr, frame_bgr, H_init):
    """Mask high-residual pixels, but ONLY within the photodiode/hand zones.
    Outside those zones, the mask is always 255 (don't mask), preserving the
    bezel/gradient signal that ECC needs."""
    h, w = frame_bgr.shape[:2]
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    warped = cv2.warpPerspective(
        template_gray, H_init, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )
    residual = cv2.absdiff(warped, frame_gray)
    residual = cv2.GaussianBlur(
        residual, (RESIDUAL_BLUR_KSIZE, RESIDUAL_BLUR_KSIZE), 0
    )

    zone = build_occlusion_zone(frame_bgr.shape, H_init)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[(zone > 0) & (residual > RESIDUAL_THRESHOLD)] = 0
    return mask, zone


def run_ecc(template_gray, frame_gray, H_init, scale, input_mask=None):
    """Run ECC at a downsampled resolution and return the refined homography
    in the original coordinate system. Returns (H_refined, cc) or (None, error)."""
    # Downsample
    t_small = cv2.resize(
        template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    f_small = cv2.resize(
        frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    mask_small = None
    if input_mask is not None:
        mask_small = cv2.resize(
            input_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )

    # H_init maps template-px → frame-px. After downsampling both:
    #   t_small-px → frame-px via H_init scaled in (because t_small = template/scale)
    #   then frame-px → f_small-px via scale.
    # Net: H_small = S * H_init * S^-1 where S = diag(scale, scale, 1).
    S = np.diag([scale, scale, 1.0]).astype(np.float32)
    H_init_small = (S @ H_init @ np.linalg.inv(S)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_ITERS,
        ECC_EPS,
    )
    try:
        cc, H_small = cv2.findTransformECC(
            templateImage=t_small,
            inputImage=f_small,
            warpMatrix=H_init_small.copy(),
            motionType=cv2.MOTION_HOMOGRAPHY,
            criteria=criteria,
            inputMask=mask_small,
            gaussFiltSize=5,
        )
    except cv2.error as e:
        return None, str(e)

    H_refined = (np.linalg.inv(S) @ H_small @ S).astype(np.float32)
    return H_refined, float(cc)


def project_corners(template_shape, H):
    H_img, W_img = template_shape[:2]
    corners = np.array(
        [[[0, 0]], [[W_img, 0]], [[W_img, H_img]], [[0, H_img]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, H)
    return projected.reshape(4, 2)


def overlay_template(frame_bgr, template_bgr, H, alpha=0.4):
    """Warp template to frame coords and alpha-blend."""
    h, w = frame_bgr.shape[:2]
    warped = cv2.warpPerspective(
        template_bgr, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    # Build a mask of where the template actually maps (non-zero) to limit blend
    mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    blend = frame_bgr.astype(np.float32) * (1 - alpha * mask) + warped.astype(
        np.float32
    ) * (alpha * mask)
    return np.clip(blend, 0, 255).astype(np.uint8)


def annotate_with_masks(frame, mask, zone, H_init):
    """Render zone outlines + active mask onto a copy of the frame.
    - Zone outlines: thin magenta polygons (where masking is *eligible*)
    - Active mask:   yellow tint (where ECC actually ignored pixels)
    """
    vis = frame.copy()
    # Yellow tint for actually-masked pixels
    overlay = np.zeros_like(frame)
    overlay[mask == 0] = (0, 255, 255)  # BGR yellow
    vis = cv2.addWeighted(vis, 1.0, overlay, 0.55, 0)
    # Dilated zone: translucent magenta fill + thick outline so it's actually
    # visible against the dark scene
    projected = project_screen_rect(TOP_OCCLUSION_ZONE, H_init)
    dilated = dilate_quad_asymmetric(
        projected, ZONE_DILATION_TOP, ZONE_DILATION_BOTTOM
    )
    zone_fill = np.zeros_like(frame)
    cv2.fillPoly(zone_fill, [dilated], (255, 0, 255))
    vis = cv2.addWeighted(vis, 1.0, zone_fill, 0.15, 0)
    cv2.polylines(vis, [dilated.reshape(-1, 1, 2)],
                  isClosed=True, color=(255, 0, 255), thickness=4)
    cv2.putText(vis, "ZONE", tuple(dilated[0] + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return vis


def draw_quad(img, corners, color, thickness=3, label=None):
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        cx, cy = corners.mean(axis=0).astype(int)
        cv2.putText(
            img, label, (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )


def process_one(cap, t, template_bgr, template_gray):
    print(f"\n=== t={t}s ===")
    frame = read_frame(cap, t)
    if frame is None:
        print("  could not read frame")
        return

    # Seed corners from brightness detector
    seed = detect_corners(frame)
    if seed is None:
        print("  brightness detector returned None — skip")
        return
    print(f"  seed corners:\n{seed}")

    # Initial homography
    H_init, _ = homography_template_to_frame(template_bgr.shape, seed)

    # Build occlusion mask: residual-based, restricted to photodiode + hand zones
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, zone = build_constrained_residual_mask(template_bgr, frame, H_init)

    start = time.time()
    H_refined, info = run_ecc(
        template_gray, frame_gray, H_init, ECC_SCALE, input_mask=mask
    )
    elapsed = time.time() - start

    if H_refined is None:
        masked_pct = 100.0 * (mask == 0).sum() / mask.size
        print(f"  ECC failed (masked {masked_pct:.1f}% of frame): {info}  (after {elapsed:.1f}s)")
        vis = annotate_with_masks(frame, mask, zone, H_init)
        draw_quad(vis, seed, (0, 0, 255), label="seed (ECC failed)")
        cv2.imwrite(str(OUT / f"t{t:04d}s_failed.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        return

    refined_corners = project_corners(template_bgr.shape, H_refined)
    pixel_shift = np.linalg.norm(refined_corners - seed, axis=1).mean()
    print(f"  ECC cc={info:.4f}, mean corner shift={pixel_shift:.1f} px, t={elapsed:.1f}s")

    # Visualizations
    vis_quads = annotate_with_masks(frame, mask, zone, H_init)
    draw_quad(vis_quads, seed, (0, 0, 255), label="seed")
    draw_quad(vis_quads, refined_corners, (0, 255, 0), label="ECC")
    cv2.imwrite(str(OUT / f"t{t:04d}s_quads.jpg"), vis_quads,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    vis_seed_overlay = overlay_template(frame, template_bgr, H_init, alpha=0.45)
    cv2.imwrite(str(OUT / f"t{t:04d}s_overlay_seed.jpg"), vis_seed_overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    vis_ecc_overlay = overlay_template(frame, template_bgr, H_refined, alpha=0.45)
    cv2.imwrite(str(OUT / f"t{t:04d}s_overlay_ecc.jpg"), vis_ecc_overlay,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    cv2.imwrite(str(OUT / f"t{t:04d}s_frame.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template_bgr = cv2.imread(str(TEMPLATE))
    if template_bgr is None:
        sys.exit(f"could not read template {TEMPLATE}")
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    print(f"template shape: {template_bgr.shape}")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"could not open video {VIDEO}")

    for t in SAMPLE_TIMES_S:
        process_one(cap, t, template_bgr, template_gray)

    cap.release()
    print(f"\nResults in {OUT}")


if __name__ == "__main__":
    main()
