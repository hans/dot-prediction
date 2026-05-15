"""QC: compare raw vs smoothed corner detection at known-bad timestamps.

Runs detect_corners on every frame from T_START to T_END, applies
smooth_corners, then saves annotated images at SAMPLE_TIMES showing:
  - dashed red  = raw single-frame detection
  - solid green = smoothed corners
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.screen_detection import detect_corners
from src.corner_smoother import smooth_corners

VIDEO = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
OUT   = Path(__file__).resolve().parent.parent / "results" / "debug_smoothing"

T_START = 0    # seconds — start of dense detection window
T_END   = 500  # seconds — covers all original QC timestamps up to t=400 s

# Timestamps to visualise (subset of original debug_screen_detection times)
SAMPLE_TIMES = [30, 60, 90, 120, 180, 240, 300, 400]

SMOOTH_WINDOW = 51  # frames (~2 s at 25 fps)


def _draw_quad(img, corners, color, thickness, dashed=False):
    pts = corners.astype(np.int32)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        if dashed:
            for s in range(0, 10, 2):
                a = (int(p1[0] + (p2[0] - p1[0]) * s / 10),
                     int(p1[1] + (p2[1] - p1[1]) * s / 10))
                b = (int(p1[0] + (p2[0] - p1[0]) * (s + 1) / 10),
                     int(p1[1] + (p2[1] - p1[1]) * (s + 1) / 10))
                cv2.line(img, a, b, color, thickness)
        else:
            cv2.line(img, p1, p2, color, thickness)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_start = int(T_START * fps)
    frame_end   = int(T_END   * fps)
    n_frames    = frame_end - frame_start

    print(f"Extracting detections: t={T_START}–{T_END}s ({n_frames} frames) …")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    abs_frames: list[int] = []
    raw_detections: list[np.ndarray | None] = []

    for idx in range(n_frames):
        abs_frames.append(frame_start + idx)
        ret, frame = cap.read()
        if not ret:
            raw_detections.append(None)
            continue
        raw_detections.append(detect_corners(frame))
        if idx % 1000 == 0:
            print(f"  {idx}/{n_frames} ({100*idx/n_frames:.0f}%)")

    n_detected = sum(1 for d in raw_detections if d is not None)
    print(f"Raw detection rate: {n_detected}/{n_frames} ({100*n_detected/n_frames:.1f}%)")

    print("Smoothing …")
    smoothed_arr = smooth_corners(raw_detections, window=SMOOTH_WINDOW)  # (n, 4, 2)

    # Map absolute frame number → (raw, smoothed)
    lookup = {
        abs_frames[i]: (raw_detections[i], smoothed_arr[i])
        for i in range(n_frames)
    }

    print("Saving annotated images …")
    for t in SAMPLE_TIMES:
        abs_f = int(t * fps)
        if abs_f not in lookup:
            print(f"t={t}s: out of range, skipping")
            continue

        raw_c, sm_c = lookup[abs_f]

        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_f)
        ret, frame = cap.read()
        if not ret:
            print(f"t={t}s: could not re-read frame")
            continue

        vis = frame.copy()

        if raw_c is not None:
            _draw_quad(vis, raw_c, (0, 0, 255), thickness=2, dashed=True)
            for x, y in raw_c.astype(int):
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

        _draw_quad(vis, sm_c, (0, 255, 0), thickness=3, dashed=False)
        for (x, y), label in zip(sm_c.astype(int), ["TL", "TR", "BR", "BL"]):
            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        raw_tag = "ok" if raw_c is not None else "none"
        cv2.putText(vis, f"raw={raw_tag}  smoothed=ok", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        out_path = OUT / f"t{t:04d}s.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"t={t:4d}s: raw={raw_tag:<4s}  → {out_path.name}")

    cap.release()
    print(f"\nImages saved to: {OUT}")
    print("Legend: dashed red = raw, solid green = smoothed")


if __name__ == "__main__":
    main()
