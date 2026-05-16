"""Visual QC: run screen detection on real video frames and save annotated images."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.screen_detection import detect_corners

VIDEO  = Path("/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4")
OUT    = Path(__file__).resolve().parent.parent / "results" / "debug_detection"

# Sample timestamps (seconds) spread across the recording
SAMPLE_TIMES = [30, 60, 90, 120, 180, 240, 300, 400, 600, 900]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detected = 0

    for t in SAMPLE_TIMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            print(f"t={t}s: could not read frame")
            continue

        corners = detect_corners(frame)
        vis = frame.copy()

        if corners is not None:
            detected += 1
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            labels = ["TL", "TR", "BR", "BL"]
            for (x, y), label in zip(corners.astype(int), labels):
                cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(vis, label, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            status = "OK"
        else:
            cv2.putText(vis, "DETECTION FAILED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            status = "FAIL"

        out_path = OUT / f"t{t:04d}s_{status}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"t={t:4d}s: {status}  → {out_path.name}")

    cap.release()
    print(f"\nDetected: {detected}/{len(SAMPLE_TIMES)} frames")
    print(f"Images saved to: {OUT}")


if __name__ == "__main__":
    main()
