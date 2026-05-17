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
# # Scout: corner deviation / glare frames (interactive)
#
# Scans the full video for corner detections, smooths them, and shows
# per-frame |raw − smoothed| for each corner. Use the scrubber and jump
# buttons to find frames where a corner (typically BR) deviates significantly
# — these are candidates for the "BR-corner glare" eval frame the spec calls
# for.
#
# **First run is slow** (full scan of the video). Results are cached so
# subsequent runs start instantly. Run interactively — widgets don't work
# under headless execution.

# %% tags=["parameters"]
video_path = "/Users/jon/Projects/dot-prediction/data/EC347/tobii/scenevideo.mp4"
corner_scan_cache = "results/EC347/corner_scan_full.parquet"
smooth_window = 51

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

from corner_smoother import smooth_corners
from screen_detection import detect_corners

# %% [markdown]
# ## Full-video corner scan (cached)
# Detects corners on every frame. Takes ~10 minutes on a 35 k-frame video;
# loads in seconds on subsequent runs.

# %%
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_frames} frames @ {fps:.3f} fps → {n_frames / fps:.1f} s")

cache_p = Path(corner_scan_cache)

if cache_p.exists():
    scan = pd.read_parquet(cache_p)
    print(f"Loaded cached scan ({len(scan)} frames) from {cache_p}")
else:
    print(f"Scanning {n_frames} frames for corners — please wait…")
    CORNER_KEYS = ["TL_x", "TL_y", "TR_x", "TR_y", "BR_x", "BR_y", "BL_x", "BL_y"]
    rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        c = detect_corners(frame)
        if c is not None:
            rows.append({
                "frame_idx": fi,
                "TL_x": float(c[0, 0]), "TL_y": float(c[0, 1]),
                "TR_x": float(c[1, 0]), "TR_y": float(c[1, 1]),
                "BR_x": float(c[2, 0]), "BR_y": float(c[2, 1]),
                "BL_x": float(c[3, 0]), "BL_y": float(c[3, 1]),
            })
        else:
            rows.append({k: np.nan for k in CORNER_KEYS} | {"frame_idx": fi})
        if fi % 2000 == 0 and fi > 0:
            pct = fi / n_frames * 100
            print(f"  {fi}/{n_frames}  ({pct:.0f}%)")
    scan = pd.DataFrame(rows)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(cache_p, index=False)
    print(f"Saved {cache_p}")

assert len(scan) == n_frames, "scan row count mismatch — delete cache and re-run"

# %% [markdown]
# ## Smooth corners and compute per-frame deviation

# %%
CORNER_NAMES = ["TL", "TR", "BR", "BL"]

raw = []
for _, row in scan.iterrows():
    if np.isnan(row["TL_x"]):
        raw.append(None)
    else:
        raw.append(np.array([
            [row["TL_x"], row["TL_y"]],
            [row["TR_x"], row["TR_y"]],
            [row["BR_x"], row["BR_y"]],
            [row["BL_x"], row["BL_y"]],
        ], dtype=np.float32))

smoothed = smooth_corners(raw, window=smooth_window)  # (n_frames, 4, 2)

dev = {"frame_idx": scan["frame_idx"].values}
for ci, name in enumerate(CORNER_NAMES):
    rx = scan[f"{name}_x"].values
    ry = scan[f"{name}_y"].values
    sx = smoothed[:, ci, 0]
    sy = smoothed[:, ci, 1]
    dev[f"{name}_dev"] = np.where(np.isnan(rx), np.nan, np.hypot(rx - sx, ry - sy))
dev = pd.DataFrame(dev)

print("Deviation summary (px):")
for name in CORNER_NAMES:
    col = dev[f"{name}_dev"].dropna()
    print(f"  {name}: median={col.median():.1f}  p95={col.quantile(0.95):.1f}  max={col.max():.1f}")

# %% [markdown]
# ## Overview: deviation time series

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
for ax, name, color in zip(axes, CORNER_NAMES, ["C0", "C1", "C2", "C3"]):
    ax.plot(dev["frame_idx"], dev[f"{name}_dev"], linewidth=0.3, color=color, label=name)
    ax.axhline(10, color="red", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"{name} dev (px)")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("frame index")
fig.suptitle("Per-corner |raw − smoothed| deviation  (red dashed = 10 px guard)")
plt.tight_layout()

# %% [markdown]
# ## Interactive frame scrubber
# **Red** outline = raw corners detected this frame.
# **Cyan** outline = smoothed corners.
# Deviation numbers are shown in the top-left; values exceeding the threshold
# are highlighted in red in the status bar.

# %%
def _high_frames(threshold_px, corner):
    col = dev[f"{corner}_dev"]
    return dev.loc[col > threshold_px, "frame_idx"].values


def _read_frame(fi):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ret, frame = cap.read()
    return frame if ret else None


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _render(fi, show_smoothed):
    frame = _read_frame(fi)
    if frame is None:
        return None
    vis = frame.copy()
    row = scan.iloc[fi]
    has_raw = not np.isnan(row["TL_x"])
    if has_raw:
        rc = np.array([
            [row["TL_x"], row["TL_y"]],
            [row["TR_x"], row["TR_y"]],
            [row["BR_x"], row["BR_y"]],
            [row["BL_x"], row["BL_y"]],
        ], dtype=np.int32)
        cv2.polylines(vis, [rc.reshape(-1, 1, 2)], True, (0, 0, 200), 2)
        for pt in rc:
            cv2.circle(vis, tuple(pt), 5, (0, 0, 200), -1)
    if show_smoothed:
        sc = smoothed[fi].astype(np.int32)
        cv2.polylines(vis, [sc.reshape(-1, 1, 2)], True, (0, 200, 200), 2)
        for pt in sc:
            cv2.circle(vis, tuple(pt), 5, (0, 200, 200), -1)
    dev_row = dev.iloc[fi]
    parts = [f"frame={fi}  t={fi/fps:.2f}s  raw={'OK' if has_raw else 'MISS'}"]
    for name in CORNER_NAMES:
        d = dev_row[f"{name}_dev"]
        parts.append(f"{name}:{d:.1f}" if not np.isnan(d) else f"{name}:--")
    cv2.putText(vis, "  ".join(parts), (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis


threshold_slider = W.FloatSlider(
    value=10.0, min=2.0, max=60.0, step=1.0,
    description="threshold px", continuous_update=False,
    layout=W.Layout(width="55%"),
)
corner_dd = W.Dropdown(options=CORNER_NAMES, value="BR", description="corner")
frame_slider = W.IntSlider(
    value=0, min=0, max=n_frames - 1, step=1,
    description="frame", continuous_update=False,
    layout=W.Layout(width="80%"),
)
fine_btns = W.HBox([
    W.Button(description="-30 fr"),
    W.Button(description="-10 fr"),
    W.Button(description="-1 fr"),
    W.Button(description="+1 fr"),
    W.Button(description="+10 fr"),
    W.Button(description="+30 fr"),
    W.Button(description="↪ next glare"),
    W.Button(description="↩ prev glare"),
])
smo_toggle = W.Checkbox(value=True, description="show smoothed (cyan)")
status_html = W.HTML()
img_widget = W.Image(format="jpeg", layout=W.Layout(width="1000px"))


def _refresh():
    fi = frame_slider.value
    vis = _render(fi, smo_toggle.value)
    if vis is None:
        status_html.value = "<span style='color:red'>Failed to read frame</span>"
        return
    img_widget.value = _jpeg(vis)
    thr = threshold_slider.value
    dev_row = dev.iloc[fi]
    parts = []
    for name in CORNER_NAMES:
        d = dev_row[f"{name}_dev"]
        if np.isnan(d):
            parts.append(f"<b>{name}:</b> --")
        else:
            color = "red" if d > thr else "black"
            parts.append(f"<b>{name}:</b> <span style='color:{color}'>{d:.1f} px</span>")
    status_html.value = " &nbsp;|&nbsp; ".join(parts)


def _step(n):
    def cb(_):
        frame_slider.value = int(np.clip(frame_slider.value + n, 0, n_frames - 1))
    return cb


def _next_glare(_):
    hi = _high_frames(threshold_slider.value, corner_dd.value)
    nxt = hi[hi > frame_slider.value]
    if len(nxt):
        frame_slider.value = int(nxt[0])


def _prev_glare(_):
    hi = _high_frames(threshold_slider.value, corner_dd.value)
    prv = hi[hi < frame_slider.value]
    if len(prv):
        frame_slider.value = int(prv[-1])


for btn, delta in zip(fine_btns.children[:6], [-30, -10, -1, 1, 10, 30]):
    btn.on_click(_step(delta))
fine_btns.children[6].on_click(_next_glare)
fine_btns.children[7].on_click(_prev_glare)

frame_slider.observe(lambda c: _refresh() if c["name"] == "value" else None, names="value")
threshold_slider.observe(lambda c: _refresh() if c["name"] == "value" else None, names="value")
smo_toggle.observe(lambda c: _refresh() if c["name"] == "value" else None, names="value")

display(W.VBox([
    W.HBox([threshold_slider, corner_dd]),
    frame_slider,
    fine_btns,
    img_widget,
    smo_toggle,
    status_html,
]))
_refresh()

# %%
cap.release()
