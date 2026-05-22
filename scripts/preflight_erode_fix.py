"""
Preflight comparison: current detect_corners vs. morphological-open variant.

Reads specific frames from the EC347 scene video and renders side-by-side crops
showing the binary mask and detected BR corner for both strategies.

Usage:
    uv run python scripts/preflight_erode_fix.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

VIDEO = Path("data/EC347/tobii/scenevideo.mp4")
OUT_DIR = Path("results_scratch/EC347")

# Frames: f500 is clean, the rest show wrist false-detection
FRAMES = [500, 9860, 10902, 13837, 17294, 18885]

# screen_detection constants
_THRESH = 50
_CLOSE_K = 51
_OPEN_K = 31   # erosion kernel for the open; strip thin peninsulas
_MIN_AREA = 80_000


def _binary_mask(gray, open_k=None):
    _, binary = cv2.threshold(gray, _THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_CLOSE_K, _CLOSE_K))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if open_k is not None:
        h, w = binary.shape
        ok = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ok)
        # Apply open only in the BR quadrant — leaves TL/TR/BL contour intact
        br_mask = np.zeros_like(binary)
        br_mask[h // 2:, w // 2:] = 255
        binary = np.where(br_mask > 0, opened, binary)
    return binary


LABELS = ["TL", "TR", "BR", "BL"]
COLORS = [(255, 128, 0), (0, 128, 255), (0, 0, 255), (255, 0, 255)]  # BGR


def _detect_all(binary):
    """Return (4,2) corners in [TL,TR,BR,BL] order, or None."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    large = [c for c in contours if cv2.contourArea(c) > _MIN_AREA]
    if not large:
        return None
    largest = max(large, key=cv2.contourArea)
    largest = cv2.convexHull(largest)
    peri = cv2.arcLength(largest, True)
    approx = None
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        candidate = cv2.approxPolyDP(largest, eps * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break
    if approx is None:
        return None
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmax(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmin(d)],   # BL
    ], dtype=np.float32)


def _crop(img, cx, cy, half=300):
    h, w = img.shape[:2]
    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(w, int(cx + half))
    y1 = min(h, int(cy + half))
    return img[y0:y1, x0:x1]


def process_frame(frame_idx, cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"  [WARN] could not read frame {frame_idx}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Current strategy (close only)
    bin_cur = _binary_mask(gray, open_k=None)
    corners_cur = _detect_all(bin_cur)

    # Proposed strategy (close + open)
    bin_new = _binary_mask(gray, open_k=_OPEN_K)
    corners_new = _detect_all(bin_new)

    # Draw all 4 corners on frame copies
    frame_cur = frame.copy()
    frame_new = frame.copy()

    def _draw_corners(img, corners):
        if corners is None:
            return
        for (x, y), label, color in zip(corners, LABELS, COLORS):
            cv2.circle(img, (int(x), int(y)), 12, color, 3)
            cv2.putText(img, label, (int(x) + 14, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    _draw_corners(frame_cur, corners_cur)
    _draw_corners(frame_new, corners_new)

    # Choose crop center: use current BR if available, else image centre
    if corners_cur is not None:
        br_cur = corners_cur[2]
        cx_c, cy_c = int(br_cur[0]), int(br_cur[1])
    else:
        cx_c, cy_c = frame.shape[1] // 2, frame.shape[0] // 2

    SCALE = 0.35
    overview_cur = cv2.resize(frame_cur, None, fx=SCALE, fy=SCALE)
    overview_new = cv2.resize(frame_new, None, fx=SCALE, fy=SCALE)

    crop_cur = _crop(frame_cur, cx_c, cy_c)
    crop_new = _crop(frame_new, cx_c, cy_c)

    # Binary mask crops around BR for detail
    bin_cur_bgr = cv2.cvtColor(bin_cur, cv2.COLOR_GRAY2BGR)
    bin_new_bgr = cv2.cvtColor(bin_new, cv2.COLOR_GRAY2BGR)
    mask_cur = _crop(bin_cur_bgr, cx_c, cy_c)
    mask_new = _crop(bin_new_bgr, cx_c, cy_c)

    def _pad(img, h, w):
        ph = max(0, h - img.shape[0])
        pw = max(0, w - img.shape[1])
        return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

    label_h = 40
    def _label(img, text, color):
        out = np.zeros((label_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return np.vstack([out, img])

    # Target width: widest of the two overviews
    target_w = max(overview_cur.shape[1], overview_new.shape[1])

    # Overview row (full frame, scaled)
    ov_cur = _pad(overview_cur, overview_cur.shape[0], target_w)
    ov_new = _pad(overview_new, overview_new.shape[0], target_w)
    ov_h = max(ov_cur.shape[0], ov_new.shape[0])
    ov_cur = _pad(ov_cur, ov_h, target_w)
    ov_new = _pad(ov_new, ov_h, target_w)
    overview_row = np.hstack([
        _label(ov_cur, f"CURRENT  f{frame_idx}", (100, 100, 255)),
        _label(ov_new, f"PROPOSED (open k={_OPEN_K})  f{frame_idx}", (100, 255, 100)),
    ])

    # Detail row (BR crop + mask)
    detail_h = max(crop_cur.shape[0], crop_new.shape[0], mask_cur.shape[0], mask_new.shape[0])
    detail_w = max(crop_cur.shape[1], crop_new.shape[1])
    crop_cur = _pad(crop_cur, detail_h, detail_w)
    crop_new = _pad(crop_new, detail_h, detail_w)
    mask_cur = _pad(mask_cur, detail_h, detail_w)
    mask_new = _pad(mask_new, detail_h, detail_w)

    col_cur = _label(np.vstack([crop_cur, mask_cur]), "BR detail + mask (current)", (100, 100, 255))
    col_new = _label(np.vstack([crop_new, mask_new]), "BR detail + mask (proposed)", (100, 255, 100))
    detail_col_w = max(col_cur.shape[1], col_new.shape[1])
    col_cur = _pad(col_cur, col_cur.shape[0], detail_col_w)
    col_new = _pad(col_new, col_new.shape[0], detail_col_w)
    detail_row = np.hstack([col_cur, col_new])

    # Pad overview and detail rows to same width before stacking vertically
    combined_w = max(overview_row.shape[1], detail_row.shape[1])
    overview_row = _pad(overview_row, overview_row.shape[0], combined_w)
    detail_row = _pad(detail_row, detail_row.shape[0], combined_w)

    out = np.vstack([overview_row, detail_row])

    def _fmt_corners(corners):
        if corners is None:
            return "None"
        return "  ".join(f"{l}({int(x)},{int(y)})" for (x, y), l in zip(corners, LABELS))

    status_lines = [
        f"current:  {_fmt_corners(corners_cur)}",
        f"proposed: {_fmt_corners(corners_new)}",
    ]
    status_h = 30 * len(status_lines) + 10
    status = np.zeros((status_h, out.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(status_lines):
        color = (100, 100, 255) if i == 0 else (100, 255, 100)
        cv2.putText(status, line, (10, 28 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
    out = np.vstack([out, status])

    dst = OUT_DIR / f"preflight_erode_f{frame_idx}.jpg"
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    br_cur = corners_cur[2] if corners_cur is not None else None
    br_new = corners_new[2] if corners_new is not None else None
    br_str_cur = f"({int(br_cur[0])},{int(br_cur[1])})" if br_cur is not None else "None"
    br_str_new = f"({int(br_new[0])},{int(br_new[1])})" if br_new is not None else "None"
    print(f"  f{frame_idx:6d}: current BR={br_str_cur}  proposed BR={br_str_new}  -> {dst.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO}")
        sys.exit(1)

    print(f"Video: {VIDEO}  ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames)")
    print(f"Strategy: close k={_CLOSE_K}  →  open k={_OPEN_K} (proposed addition)")
    print()
    for fi in FRAMES:
        process_frame(fi, cap)

    cap.release()
    print()
    print(f"Open results with:")
    print(f"  open {OUT_DIR}/preflight_erode_f*.jpg")


if __name__ == "__main__":
    main()
