# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Star detection evaluation — Phase 1
#
# Evaluates whether yellow-star blobs on the iPad task display are reliably
# detectable from the Tobii scene-camera video.  Matching against known star
# positions requires behavior-to-video alignment (a separate pipeline step) and
# is out of scope here.  Correctness is assessed visually from the overlay
# images.
#
# ## What this produces
# - Overlay images (screen outline + green circles for each detected blob) at
#   sample frames spread across the recording.
# - A centroid-stability plot: detected centroid of the largest blob over ~1 s
#   of consecutive frames within one trial, quantifying per-frame jitter.

# %% tags=["parameters"]
video_path   = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4"
out_dir      = "results/star_detection_eval"
# Video times (seconds) to sample; spread across the full recording.
# Adjust to frames that are known to contain visible stars.
sample_times = [120, 170, 240, 300, 400, 500]
# For centroid stability: sample this many consecutive frames starting here.
stability_t0_s   = 300.0
stability_nframes = 50

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..") / "src"))

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from screen_detection import detect_corners
from star_detector import detect_stars

out_path = Path(out_dir)
out_path.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_frames} frames @ {fps:.2f} fps → {n_frames/fps:.0f} s")
print(f"Output directory: {out_path.resolve()}")


# %% [markdown]
# ## Detection overlays

# %%
def grab_frame(vid_s: float):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_s * fps))
    ret, frame = cap.read()
    return frame if ret else None


def render_overlay(frame, corners, blobs) -> np.ndarray:
    vis = frame.copy()
    if corners is not None:
        cv2.polylines(vis, [corners.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 200, 0), 2)
    for bx, by, br in blobs:
        cv2.circle(vis, (int(bx), int(by)), max(int(br), 6), (0, 255, 0), 2)
        cv2.putText(vis, f"{int(np.pi * br**2)}", (int(bx) + int(br) + 3, int(by)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return vis


summary = []
for vid_s in sample_times:
    frame = grab_frame(vid_s)
    if frame is None:
        print(f"t={vid_s}s: cannot read frame"); continue

    corners = detect_corners(frame)
    blobs   = detect_stars(frame)

    overlay = render_overlay(frame, corners, blobs)
    screen_ok = corners is not None
    cv2.putText(overlay,
                f"t={vid_s}s  screen={'OK' if screen_ok else 'FAIL'}  blobs={len(blobs)}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out_file = out_path / f"t{int(vid_s):04d}s.jpg"
    cv2.imwrite(str(out_file), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    summary.append(dict(t=vid_s, screen_ok=screen_ok, n_blobs=len(blobs),
                        blob_areas=[int(np.pi * b[2]**2) for b in blobs]))
    print(f"t={vid_s:4d}s  screen={'OK' if screen_ok else 'FAIL'}"
          f"  blobs={len(blobs)}  areas={[int(np.pi*b[2]**2) for b in blobs]}")

# %%
# Display overlays inline
n = len(summary)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
axes = np.array(axes).flatten()

for i, info in enumerate(summary):
    img_path = out_path / f"t{int(info['t']):04d}s.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        axes[i].axis("off"); continue
    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"t={info['t']}s  blobs={info['n_blobs']}", fontsize=9)
    axes[i].axis("off")

for ax in axes[n:]:
    ax.axis("off")

legend = [mpatches.Patch(color="lime", label="detected blob (area px² labelled)")]
fig.legend(handles=legend, loc="lower center", fontsize=10)
fig.suptitle("Star detection overlays — visual QC", fontsize=13)
plt.tight_layout()
plt.savefig(str(out_path / "overview.png"), dpi=80, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Centroid stability
#
# Sample ~2 s of consecutive frames and track the largest detected blob.
# Jitter in the centroid bounds the per-correspondence accuracy that will feed
# the homography solve in Phase 2.

# %%
frame_step = 1.0 / fps
records = []

for i in range(stability_nframes):
    vid_s = stability_t0_s + i * frame_step
    frame = grab_frame(vid_s)
    if frame is None:
        continue
    blobs = detect_stars(frame)
    if not blobs:
        continue
    # track the largest blob
    best = max(blobs, key=lambda b: b[2])
    records.append({"i": i, "t": vid_s, "cx": best[0], "cy": best[1], "r": best[2]})

cap.release()

if not records:
    print("No detections in stability window — pick a different stability_t0_s")
else:
    import pandas as pd
    stab = pd.DataFrame(records)
    jx, jy = stab.cx.std(), stab.cy.std()
    rms = float(np.sqrt(jx**2 + jy**2))
    print(f"Tracked {len(stab)}/{stability_nframes} frames")
    print(f"Centroid std:  x={jx:.2f} px,  y={jy:.2f} px")
    print(f"RMS jitter:    {rms:.2f} px")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(stab.i, stab.cx, ".-", label="cx")
    axes[0].plot(stab.i, stab.cy, ".-", label="cy")
    axes[0].set_xlabel("frame index")
    axes[0].set_ylabel("centroid (px)")
    axes[0].set_title(f"Centroid over {stability_nframes} frames  (t₀={stability_t0_s:.0f}s)")
    axes[0].legend()

    axes[1].scatter(stab.cx - stab.cx.mean(), stab.cy - stab.cy.mean(), s=25, alpha=0.7)
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].axvline(0, color="grey", lw=0.5)
    axes[1].set_xlabel("Δcx (px)")
    axes[1].set_ylabel("Δcy (px)")
    axes[1].set_title(f"Centroid scatter  RMS={rms:.2f} px")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(str(out_path / "centroid_stability.png"), dpi=100)
    plt.show()
    print(f"Saved {out_path}/centroid_stability.png")
