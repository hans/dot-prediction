# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scout: big-star-near-corner frames (interactive)
#
# Finds reveal frames where the big star appears close to a screen corner
# in iPad screen space. When the anchor is corner-proximate, most small stars
# are far away — the worst case for iterative refinement, since the
# anchor-translation step corrects only global translation, not perspective.
#
# **Workflow:**
# 1. The overview plot ranks all reveal subtrials by proximity to the nearest
#    screen corner (computed purely from behavior data — no video needed).
# 2. The interactive scrubber shows each frame with corners + detected star
#    overlaid so you can verify the candidate visually.
# 3. Candidates are scanned and cached on first run (~300 frames, fast).
#
# Run interactively — widgets don't work under headless execution.

# %% tags=["parameters"]
video_path = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4"
trials_path = "results/EC347/trials_with_video.parquet"
reveal_scan_cache = "results/EC347/reveal_frame_scan.parquet"

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..") / "src"))

import cv2
import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from screen_detection import detect_corners
from star_detector import detect_stars

SCREEN_W = 2388
SCREEN_H = 1668
SCREEN_CORNERS_PX = np.array(
    [[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]], dtype=np.float32
)
CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# %% [markdown]
# ## Load behavior and compute corner proximity
# `true_x`, `true_y` are normalised screen coordinates → multiply by
# screen dimensions to get px. Distance to the nearest screen corner tells
# us how extreme the anchor position is.

# %%
trials = pd.read_parquet(trials_path).dropna(subset=["video_frame_reveal"])
trials = trials[trials["video_frame_reveal"] >= 0].copy()
trials["star_sx"] = trials["true_x"] * SCREEN_W
trials["star_sy"] = trials["true_y"] * SCREEN_H

for ci, label in enumerate(CORNER_LABELS):
    cx, cy = SCREEN_CORNERS_PX[ci]
    trials[f"dist_{label}"] = np.hypot(trials["star_sx"] - cx, trials["star_sy"] - cy)

trials["dist_nearest_corner"] = trials[[f"dist_{l}" for l in CORNER_LABELS]].min(axis=1)
trials["nearest_corner"] = trials[[f"dist_{l}" for l in CORNER_LABELS]].idxmin(axis=1).str.replace("dist_", "")
trials["video_frame_reveal"] = trials["video_frame_reveal"].astype(int)

print(f"{len(trials)} reveal rows  (trials {trials.trial_idx.min()}–{trials.trial_idx.max()})")
print("\nNearest-corner distance summary (screen px):")
print(trials["dist_nearest_corner"].describe().round(1))

# %% [markdown]
# ## Overview: star screen positions coloured by corner proximity

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sc = ax.scatter(
    trials["star_sx"], trials["star_sy"],
    c=trials["dist_nearest_corner"], cmap="plasma_r",
    s=18, alpha=0.7,
)
for (cx, cy), label in zip(SCREEN_CORNERS_PX, CORNER_LABELS):
    ax.plot(cx, cy, "k+", markersize=12)
    ax.annotate(label, (cx, cy), textcoords="offset points", xytext=(8, 4), fontsize=9)
plt.colorbar(sc, ax=ax, label="dist to nearest corner (px)")
ax.set_xlim(-50, SCREEN_W + 50)
ax.set_ylim(SCREEN_H + 50, -50)
ax.set_xlabel("screen x (px)")
ax.set_ylabel("screen y (px)")
ax.set_title("All reveal positions (brighter = closer to a corner)")
ax.set_aspect("equal")

ax = axes[1]
ax.hist(trials["dist_nearest_corner"], bins=40, color="steelblue")
ax.axvline(500, color="red", linestyle="--", label="500 px threshold")
close = (trials["dist_nearest_corner"] < 500).sum()
ax.set_xlabel("distance to nearest screen corner (px)")
ax.set_ylabel("reveal count")
ax.set_title(f"{close} reveals within 500 px of a corner")
ax.legend()

plt.tight_layout()

# %% [markdown]
# ## Scan reveal frames (cached)
# Detects corners and the big star on every reveal frame. ~300 frames → fast.

# %%
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_video_frames} frames @ {fps:.3f} fps")

cache_p = Path(reveal_scan_cache)

if cache_p.exists():
    rscan = pd.read_parquet(cache_p)
    print(f"Loaded reveal scan ({len(rscan)} rows) from {cache_p}")
else:
    print(f"Scanning {len(trials)} reveal frames…")
    rows = []
    for _, row in trials.iterrows():
        fi = int(row["video_frame_reveal"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            rows.append({"frame_idx": fi, "trial_idx": row["trial_idx"], "tpt": row["tpt"]})
            continue
        corners = detect_corners(frame)
        stars = detect_stars(frame)
        rec = {
            "frame_idx": fi,
            "trial_idx": int(row["trial_idx"]),
            "tpt": int(row["tpt"]),
        }
        if corners is not None:
            for ci, label in enumerate(CORNER_LABELS):
                rec[f"c{label}_x"] = float(corners[ci, 0])
                rec[f"c{label}_y"] = float(corners[ci, 1])
        else:
            for label in CORNER_LABELS:
                rec[f"c{label}_x"] = np.nan
                rec[f"c{label}_y"] = np.nan
        if stars:
            bx, by, br = stars[0]
            rec["star_x"] = float(bx)
            rec["star_y"] = float(by)
            rec["star_r"] = float(br)
        else:
            rec["star_x"] = rec["star_y"] = rec["star_r"] = np.nan
        rows.append(rec)
    rscan = pd.DataFrame(rows)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    rscan.to_parquet(cache_p, index=False)
    print(f"Saved {cache_p}")

# Merge detection results back onto trials for the interactive picker
reveal_df = trials.merge(rscan, on=["frame_idx", "trial_idx", "tpt"], how="left")
reveal_df = reveal_df.sort_values("dist_nearest_corner").reset_index(drop=True)
print(f"\nTop 10 closest-to-corner reveals:")
cols = ["trial_idx", "tpt", "nearest_corner", "dist_nearest_corner", "frame_idx"]
print(reveal_df.head(10)[cols].to_string(index=False))

# %% [markdown]
# ## Interactive scrubber
# Rows are sorted by `dist_nearest_corner` (ascending) so stepping forward
# moves toward less extreme positions. Use the jump buttons to skip to the
# next frame within the threshold distance.

# %%
def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _render(idx):
    row = reveal_df.iloc[idx]
    fi = int(row["frame_idx"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        return None
    vis = frame.copy()

    has_corners = not np.isnan(row["cTL_x"])
    has_star = not np.isnan(row["star_x"])

    if has_corners:
        fc = np.array([
            [row["cTL_x"], row["cTL_y"]],
            [row["cTR_x"], row["cTR_y"]],
            [row["cBR_x"], row["cBR_y"]],
            [row["cBL_x"], row["cBL_y"]],
        ], dtype=np.int32)
        cv2.polylines(vis, [fc.reshape(-1, 1, 2)], True, (0, 200, 0), 2)
        for pt in fc:
            cv2.circle(vis, tuple(pt), 5, (0, 200, 0), -1)

    if has_star:
        sx, sy, sr = int(row["star_x"]), int(row["star_y"]), max(int(row["star_r"]), 8)
        cv2.circle(vis, (sx, sy), sr, (0, 255, 255), 2)
        cv2.drawMarker(vis, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    # Project expected star position from screen space if corners available
    if has_corners:
        fc_f = np.array([
            [row["cTL_x"], row["cTL_y"]],
            [row["cTR_x"], row["cTR_y"]],
            [row["cBR_x"], row["cBR_y"]],
            [row["cBL_x"], row["cBL_y"]],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(SCREEN_CORNERS_PX, fc_f)
        if H is not None:
            p = H @ np.array([row["star_sx"], row["star_sy"], 1.0])
            ex, ey = int(p[0] / p[2]), int(p[1] / p[2])
            cv2.drawMarker(vis, (ex, ey), (255, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 2)

    label_info = (
        f"trial {int(row['trial_idx'])} tpt {int(row['tpt'])}  "
        f"frame {fi}  t={fi/fps:.2f}s  "
        f"star_screen=({row['star_sx']:.0f},{row['star_sy']:.0f})  "
        f"dist_nearest={row['dist_nearest_corner']:.0f} px ({row['nearest_corner']})"
    )
    cv2.putText(vis, label_info, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


n_rows = len(reveal_df)
threshold_slider = W.FloatSlider(
    value=500.0, min=100.0, max=1500.0, step=50.0,
    description="threshold px", continuous_update=False,
    layout=W.Layout(width="55%"),
)
row_slider = W.IntSlider(
    value=0, min=0, max=n_rows - 1, step=1,
    description="rank (↑ closer to corner)", continuous_update=False,
    layout=W.Layout(width="80%"),
)
fine_btns = W.HBox([
    W.Button(description="-10"),
    W.Button(description="-1"),
    W.Button(description="+1"),
    W.Button(description="+10"),
    W.Button(description="↪ next close"),
    W.Button(description="↩ prev close"),
])
status_html = W.HTML()
img_widget = W.Image(format="jpeg", layout=W.Layout(width="1000px"))


def _refresh():
    idx = row_slider.value
    vis = _render(idx)
    if vis is None:
        status_html.value = "<span style='color:red'>Failed to read frame</span>"
        return
    img_widget.value = _jpeg(vis)
    row = reveal_df.iloc[idx]
    thr = threshold_slider.value
    close = (reveal_df["dist_nearest_corner"] <= thr).sum()
    color = "red" if row["dist_nearest_corner"] <= thr else "black"
    status_html.value = (
        f"<b>Rank {idx + 1}/{n_rows}</b> &nbsp;|&nbsp; "
        f"trial {int(row['trial_idx'])} tpt {int(row['tpt'])} &nbsp;|&nbsp; "
        f"nearest corner: <b>{row['nearest_corner']}</b> &nbsp;|&nbsp; "
        f"distance: <span style='color:{color}'><b>{row['dist_nearest_corner']:.0f} px</b></span> "
        f"&nbsp;|&nbsp; {close} rows ≤ threshold &nbsp;|&nbsp; "
        f"corners: {'OK' if not np.isnan(row['cTL_x']) else 'MISS'}  "
        f"star: {'OK' if not np.isnan(row['star_x']) else 'MISS'}"
    )


def _step(n):
    def cb(_):
        row_slider.value = int(np.clip(row_slider.value + n, 0, n_rows - 1))
    return cb


def _next_close(_):
    thr = threshold_slider.value
    idx = row_slider.value
    close_idx = reveal_df.index[reveal_df["dist_nearest_corner"] <= thr].tolist()
    nxt = [i for i in close_idx if i > idx]
    if nxt:
        row_slider.value = nxt[0]


def _prev_close(_):
    thr = threshold_slider.value
    idx = row_slider.value
    close_idx = reveal_df.index[reveal_df["dist_nearest_corner"] <= thr].tolist()
    prv = [i for i in close_idx if i < idx]
    if prv:
        row_slider.value = prv[-1]


for btn, delta in zip(fine_btns.children[:4], [-10, -1, 1, 10]):
    btn.on_click(_step(delta))
fine_btns.children[4].on_click(_next_close)
fine_btns.children[5].on_click(_prev_close)

row_slider.observe(lambda c: _refresh() if c["name"] == "value" else None, names="value")
threshold_slider.observe(lambda c: _refresh() if c["name"] == "value" else None, names="value")

display(W.VBox([
    threshold_slider,
    row_slider,
    fine_btns,
    img_widget,
    status_html,
]))
_refresh()

# %%
cap.release()
