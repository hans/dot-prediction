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
# # Hand-labeled homography correspondences
#
# Interactive notebook for placing ground-truth points on Tobii scene-camera
# frames. Labels feed a downstream homography-evaluation step — we'll compute
# reprojection error of candidate homographies against this labeled set.
#
# **Per frame we label up to 7 points:**
# - `screen_tl`, `screen_tr`, `screen_br`, `screen_bl` — iPad corners
# - `box_bl`, `box_br` — photodiode-box visible-strip bottom corners
# - `big_star` — freshly-revealed large-star centroid (when on screen)
#
# Each label is either visible (with frame-coords) or marked not-visible. A
# per-label quality flag (`confident` / `approximate` / `occluded`) records
# how trustworthy the placement is.
#
# **Workflow:**
# 1. Run the candidate-picker (below) to get a tagged list of ~40 frames
#    spanning the difficulty range.
# 2. Trim to ~25 frames by editing `frame_indices` in the parameters cell;
#    re-execute.
# 3. Label using the interactive UI. Auto-saves to
#    `results/{subject}/homography_labels.parquet` on every action — reopening
#    the notebook resumes where you left off.
# 4. Re-run the summary section at the end to see completion rates and the
#    box-bottom screen-coord back-projection.

# %%
# %matplotlib widget
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Locate project root (../ from this notebook). When run as a Jupyter notebook
# __file__ is undefined; fall back to cwd.
_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd().parent
)
sys.path.insert(0, str(_ROOT / "src"))

# %% tags=["parameters"]
# Convention (matching align_video.py): absolute paths for source data,
# project-relative for results/. When running interactively, start Jupyter
# from the project root (or a worktree with a results/ symlink to it).
subject = "EC347"
video_path = _ROOT / f"/Users/jon/Projects/dot-prediction/data/{subject}/tobii/scenevideo.mp4"
trials_path = _ROOT / f"results/{subject}/trials_with_video.parquet"
photodiode_edges_path = _ROOT / f"results/{subject}/photodiode_edges.parquet"
align_path = _ROOT / f"results/{subject}/video_alignment.json"
labels_out = _ROOT / f"results/{subject}/homography_labels.parquet"

# Frame indices to label. Empty → notebook stops after the candidate-picker
# and prompts you to populate this list. Fill in ~25 frames spanning the
# tagged categories printed by the picker.
frame_indices: list[int] = [
      475, 500,           # head-jumps (pre-trial-1)
      664, 685, 731,      # early tr1 + box black/white right after tr1 start
      1550,               # mid tr1
      2189, 2288,         # late tr1 + intertrial tr1→tr2
      3037, 3060,         # box black/white (tr2)
      5811, 6104,         # box black/white (tr4)
      9124, 9222,         # box black/white (tr6)
      9432, 9963, 10498, 10562,   # early/mid/late/intertrial tr7
      18973, 19671, 20129, 20192, # early/mid/late/intertrial tr13
      30125, 30135, 30175,        # head-jumps + intertrial near tr19→tr20
      30232, 30950, 31423,        # early/mid/late tr20

    # manually added based on trajectory results
    900, 4000, 7500, 26000, 5000, 5500, 5750,
]

# Candidate-picker knobs
picker_corner_stride = 25         # frames between detect_corners samples for head-motion proxy
picker_n_per_category = 4         # target frames per category
picker_head_motion_min_disp = 8.0 # frame-px displacement threshold for "head jump"

# Zoom panel initial view (frame-px). Use the matplotlib toolbar's pan/zoom
# buttons (top of the figure) to refine — your zoom persists across frame
# navigation, and the "reset zoom" button below the figure returns to these
# defaults. NB: toggle out of pan/zoom mode before clicking to label.
zoom_init_x_range = (200, 1100)
zoom_init_y_range = (50, 600)

# Approximate screen geometry (px). iPad Pro 11" native portrait orientation.
SCREEN_W_PX = 2388
SCREEN_H_PX = 1668

# Photodiode-box rough screen-coord estimate (used only for back-projection
# sanity check in the summary; corner labels themselves are not assumed.)
BOX_TOP_LEFT_SX = 0.0
BOX_TOP_LEFT_SY = 0.0
BOX_VISIBLE_STRIP_HEIGHT_SY = 30.0  # spec estimate

# %% [markdown]
# ## Setup

# %%
import cv2
import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from screen_detection import detect_corners

# %% [markdown]
# ## Load context
# Behavior trials, photodiode edges, video alignment, video metadata.

# %%
trials = pd.read_parquet(trials_path)
edges = pd.read_parquet(photodiode_edges_path)
with open(align_path) as f:
    align = json.load(f)
slope_ms_per_s = align["slope_ms_per_s"]
intercept_ms = align["intercept_ms"]

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {n_frames} frames @ {fps:.3f} fps → {n_frames/fps:.1f} s")
print(f"Trials: {trials.trial_idx.min()}–{trials.trial_idx.max()}, {len(trials)} rows")
print(f"Photodiode edges: {len(edges)} ({(edges.kind == 'rise').sum()} rise, {(edges.kind == 'fall').sum()} fall)")
print(f"Alignment: expt_ms = {slope_ms_per_s:.4f} × video_s + {intercept_ms:.1f}")


# %% [markdown]
# ### Helpers: time / state lookup
# - `frame_to_expt_ms`: video frame → experiment-clock ms (via alignment)
# - `photodiode_state_at`: most recent edge before a given expt time → "white"
#   (last edge was a rise = box went bright) or "black" (last edge was a fall).

# %%
def frame_to_expt_ms(frame_idx: int) -> float:
    video_t_s = frame_idx / fps
    return slope_ms_per_s * video_t_s + intercept_ms


def expt_ms_to_frame(expt_t_ms: float) -> int:
    video_t_s = (expt_t_ms - intercept_ms) / slope_ms_per_s
    return int(round(video_t_s * fps))


# Photodiode `t_peak` is in seconds since expt_start (see detect_photodiode_edges
# notebook). To compare against frame_to_expt_ms output (which is in
# expt-clock ms since the same expt_start), multiply by 1000.
_edges_sorted = edges.sort_values("t_peak").reset_index(drop=True)
_edge_t_ms = (_edges_sorted.t_peak.values * 1000.0).astype(np.float64)
_edge_kind = _edges_sorted.kind.values


def photodiode_state_at(expt_t_ms: float) -> str:
    """Return 'white' or 'black' based on the most recent edge.

    Returns 'unknown' if expt_t_ms precedes the first edge.
    """
    idx = np.searchsorted(_edge_t_ms, expt_t_ms, side="right") - 1
    if idx < 0:
        return "unknown"
    return "white" if _edge_kind[idx] == "rise" else "black"


def trial_context_at(frame_idx: int) -> dict:
    """Return {trial_idx, tpt, time_since_reveal_ms} for the most recent reveal
    at or before this frame, or {trial_idx: None, ...} if no prior reveal."""
    prior = trials[trials.video_frame_reveal <= frame_idx]
    if len(prior) == 0:
        return {"trial_idx": None, "tpt": None, "time_since_reveal_ms": None}
    row = prior.sort_values("video_frame_reveal").iloc[-1]
    expt_t_ms_now = frame_to_expt_ms(frame_idx)
    return {
        "trial_idx": int(row.trial_idx),
        "tpt": int(row.tpt),
        "time_since_reveal_ms": float(expt_t_ms_now - row.reveal_time),
        "true_x": float(row.true_x),
        "true_y": float(row.true_y),
    }


# %% [markdown]
# ## Candidate-frame picker
# Picks frames across categories to span the difficulty range. Output is a
# tagged list — review and trim to ~25 entries by editing `frame_indices` in
# the parameters cell.

# %%
def _trial_first_reveal_frame(t_idx: int) -> int | None:
    sub = trials[trials.trial_idx == t_idx].sort_values("tpt")
    return int(sub.video_frame_reveal.iloc[0]) if len(sub) else None


def _trial_last_reveal_frame(t_idx: int) -> int | None:
    sub = trials[trials.trial_idx == t_idx].sort_values("tpt")
    return int(sub.video_frame_reveal.iloc[-1]) if len(sub) else None


def _midpoint_between_reveals(t_idx: int, tpt_a: int) -> int | None:
    """Frame halfway between tpt_a's reveal and tpt_a+1's reveal — a 'stable
    dot displayed' frame for trial t_idx."""
    sub = trials[(trials.trial_idx == t_idx)].sort_values("tpt")
    if tpt_a + 1 not in sub.tpt.values or tpt_a not in sub.tpt.values:
        return None
    f_a = int(sub[sub.tpt == tpt_a].video_frame_reveal.iloc[0])
    f_b = int(sub[sub.tpt == tpt_a + 1].video_frame_reveal.iloc[0])
    return (f_a + f_b) // 2


def _pick_clean_midtrial(n: int) -> list[tuple[int, str]]:
    """Mid-trial 'stable display' frames spread across trials."""
    out = []
    trial_ids = sorted(trials.trial_idx.unique())
    # Evenly sample across trials, picking the midpoint between tpt 7 and 8
    # (a roughly middle reveal) of each chosen trial.
    chosen = np.linspace(0, len(trial_ids) - 1, n, dtype=int)
    for ci in chosen:
        t = trial_ids[ci]
        f = _midpoint_between_reveals(t, 7)
        if f is not None:
            out.append((f, f"clean_midtrial_tr{t}"))
    return out


def _pick_early_trial(n: int) -> list[tuple[int, str]]:
    """Tpt 0 frames — fresh big-star reveal on a mostly empty screen."""
    out = []
    trial_ids = sorted(trials.trial_idx.unique())
    chosen = np.linspace(0, len(trial_ids) - 1, n, dtype=int)
    for ci in chosen:
        t = trial_ids[ci]
        f = _trial_first_reveal_frame(t)
        if f is not None:
            # +5 frames so the big star is fully rendered, not mid-fade-in
            out.append((f + 5, f"early_tr{t}_tpt0"))
    return out


def _pick_late_trial(n: int) -> list[tuple[int, str]]:
    """Last-tpt frames — many small dots accumulated."""
    out = []
    trial_ids = sorted(trials.trial_idx.unique())
    chosen = np.linspace(0, len(trial_ids) - 1, n, dtype=int)
    for ci in chosen:
        t = trial_ids[ci]
        f = _trial_last_reveal_frame(t)
        if f is not None:
            out.append((f + 5, f"late_tr{t}_lasttpt"))
    return out


def _pick_inter_trial(n: int) -> list[tuple[int, str]]:
    """Frames in the gap between trials — no stars expected on screen."""
    out = []
    trial_ids = sorted(trials.trial_idx.unique())
    chosen = np.linspace(0, len(trial_ids) - 2, n, dtype=int)
    for ci in chosen:
        t = trial_ids[ci]
        t_next = trial_ids[ci + 1] if ci + 1 < len(trial_ids) else None
        if t_next is None:
            continue
        f_end = _trial_last_reveal_frame(t)
        f_start = _trial_first_reveal_frame(t_next)
        if f_end is None or f_start is None or f_start - f_end < 30:
            continue
        mid = (f_end + f_start) // 2
        out.append((mid, f"intertrial_tr{t}_to_tr{t_next}"))
    return out


def _pick_box_state(want: str, n: int) -> list[tuple[int, str]]:
    """Frames at well-known photodiode states. Picks frames a fixed delay
    after each edge so the box state is stable (not mid-transition)."""
    kind_want = "rise" if want == "white" else "fall"
    edges_of_kind = _edges_sorted[_edges_sorted.kind == kind_want]
    # Stable interval after the edge: 200 ms is well past any transition.
    out = []
    chosen = np.linspace(0, len(edges_of_kind) - 1, n * 3, dtype=int)
    for ci in chosen:
        if len(out) >= n:
            break
        e = edges_of_kind.iloc[ci]
        expt_t_ms = e.t_peak * 1000.0 + 200.0
        f = expt_ms_to_frame(expt_t_ms)
        if 0 <= f < n_frames:
            out.append((f, f"box_{want}_after_edge_t{e.t_peak:.1f}s"))
    return out[:n]


def _pick_head_motion(n: int, stride: int, min_disp: float) -> list[tuple[int, str]]:
    """Sample detect_corners every `stride` frames; find spikes in frame-to-frame
    BL displacement (head jumps). Picks frames near the largest spikes."""
    print(f"  scanning corners every {stride} frames for head-motion candidates...")
    sample_idx = np.arange(0, n_frames, stride)
    bl_x, bl_y = [], []
    for f in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ret, frame = cap.read()
        if not ret:
            bl_x.append(np.nan); bl_y.append(np.nan); continue
        c = detect_corners(frame)
        if c is None:
            bl_x.append(np.nan); bl_y.append(np.nan)
        else:
            bl_x.append(float(c[3, 0])); bl_y.append(float(c[3, 1]))
    bl_x = np.array(bl_x); bl_y = np.array(bl_y)
    disp = np.sqrt(np.diff(bl_x) ** 2 + np.diff(bl_y) ** 2)
    valid = ~np.isnan(disp) & (disp >= min_disp)
    if not valid.any():
        print(f"  no jumps ≥ {min_disp} px found")
        return []
    # Top-n peaks by displacement
    order = np.argsort(disp[valid])[::-1]
    picked_sample_idx = np.where(valid)[0][order[:n]]
    out = []
    for si in picked_sample_idx:
        f = int(sample_idx[si + 1])  # the AFTER-jump frame
        out.append((f, f"head_jump_disp{disp[si]:.1f}px"))
    return out


def pick_candidates() -> pd.DataFrame:
    """Returns a DataFrame of candidate frames with category tags."""
    rows = []
    rows += _pick_early_trial(picker_n_per_category)
    rows += _pick_clean_midtrial(picker_n_per_category)
    rows += _pick_late_trial(picker_n_per_category)
    rows += _pick_inter_trial(picker_n_per_category)
    rows += _pick_box_state("white", picker_n_per_category)
    rows += _pick_box_state("black", picker_n_per_category)
    rows += _pick_head_motion(
        picker_n_per_category, picker_corner_stride, picker_head_motion_min_disp
    )
    df = pd.DataFrame(rows, columns=["frame_idx", "category"])
    df = df.drop_duplicates(subset=["frame_idx"]).reset_index(drop=True)
    # Annotate with trial context + photodiode state for review
    df["expt_t_ms"] = df.frame_idx.map(frame_to_expt_ms)
    df["photodiode_state"] = df.expt_t_ms.map(photodiode_state_at)
    df["trial_idx"] = df.frame_idx.map(lambda f: trial_context_at(f).get("trial_idx"))
    df["tpt"] = df.frame_idx.map(lambda f: trial_context_at(f).get("tpt"))
    df["t_since_reveal_s"] = df.frame_idx.map(
        lambda f: (trial_context_at(f).get("time_since_reveal_ms") or 0) / 1000.0
    )
    return df.sort_values("frame_idx").reset_index(drop=True)


# Skip the picker (which does video I/O for head-motion scan, ~1 min) once
# the user has populated frame_indices. Set candidates = None as a marker.
if not frame_indices:
    candidates = pick_candidates()
    print(f"\n{len(candidates)} candidate frames:")
    print(candidates.to_string(index=False))
    print(
        "\nframe_indices is empty — stopping here. Copy a curated subset of "
        "frame_idx values into the frame_indices parameter and re-run."
    )
else:
    candidates = None
    print(f"frame_indices populated ({len(frame_indices)} frames) — skipping picker.")

# %% [markdown]
# ## Label store: load + persist
# Labels are stored on disk as a parquet keyed by `(frame_idx, label_type)`.
# In-memory, we hold a dict `{(frame_idx, label_type): record}` that's the
# source of truth during a session. Every UI action upserts into this dict
# and writes the entire dict back to parquet — small file, ~25 frames × 7
# label types = 175 rows max, so write cost is negligible.

# %%
LABEL_TYPES = ["screen_tl", "screen_tr", "screen_br", "screen_bl",
               "box_bl", "box_br", "big_star"]
QUALITY_LEVELS = ["confident", "approximate", "occluded"]

_labels_path = Path(labels_out)
_labels_path.parent.mkdir(parents=True, exist_ok=True)


def _empty_record(frame_idx: int, label_type: str) -> dict:
    return {
        "frame_idx": int(frame_idx),
        "label_type": label_type,
        "visible": None,        # None = not labeled, True = visible, False = marked not-visible
        "x_frame": None,
        "y_frame": None,
        "quality": None,
        "notes": "",
        "saved_at": None,
    }


def load_labels() -> dict:
    if not _labels_path.exists():
        return {}
    df = pd.read_parquet(_labels_path)
    out = {}
    for _, r in df.iterrows():
        key = (int(r.frame_idx), r.label_type)
        out[key] = {
            "frame_idx": int(r.frame_idx),
            "label_type": r.label_type,
            "visible": None if pd.isna(r.visible) else bool(r.visible),
            "x_frame": None if pd.isna(r.x_frame) else float(r.x_frame),
            "y_frame": None if pd.isna(r.y_frame) else float(r.y_frame),
            "quality": None if pd.isna(r.quality) else str(r.quality),
            "notes": "" if pd.isna(r.notes) else str(r.notes),
            "saved_at": None if pd.isna(r.saved_at) else str(r.saved_at),
        }
    return out


def save_labels(store: dict) -> None:
    if not store:
        # Write empty file so reopens don't error
        pd.DataFrame(columns=[
            "frame_idx", "label_type", "visible", "x_frame", "y_frame",
            "quality", "notes", "saved_at",
        ]).to_parquet(_labels_path, index=False)
        return
    df = pd.DataFrame(list(store.values()))
    df = df.sort_values(["frame_idx", "label_type"]).reset_index(drop=True)
    df.to_parquet(_labels_path, index=False)


labels = load_labels()
print(f"Loaded {len(labels)} existing label records from {_labels_path}")

# %% [markdown]
# ## Frame & zoom helpers

# %%
_frame_cache: dict[int, np.ndarray] = {}


def read_frame(frame_idx: int) -> np.ndarray | None:
    if frame_idx in _frame_cache:
        return _frame_cache[frame_idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    # cv2 reads BGR; convert for matplotlib display
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _frame_cache[frame_idx] = rgb
    return rgb


# %% [markdown]
# ## Interactive labeling UI

# %%
if frame_indices:
    # Per-frame UI state
    _state = {
        "frame_pos": 0,                # index into frame_indices
        "active_label": LABEL_TYPES[0],
        "active_quality": QUALITY_LEVELS[0],
        "current_crop": None,          # (x0, y0, x1, y1) for current frame
    }

    # ── Widgets ──────────────────────────────────────────────────────────
    label_dd = W.Dropdown(
        options=LABEL_TYPES, value=LABEL_TYPES[0], description="label:",
        layout=W.Layout(width="220px"),
    )
    quality_dd = W.Dropdown(
        options=QUALITY_LEVELS, value=QUALITY_LEVELS[0], description="quality:",
        layout=W.Layout(width="200px"),
    )
    prev_btn = W.Button(description="◀ prev", layout=W.Layout(width="80px"))
    next_btn = W.Button(description="next ▶", layout=W.Layout(width="80px"))
    jump_field = W.IntText(
        value=frame_indices[0], description="jump:",
        layout=W.Layout(width="160px"),
    )
    jump_btn = W.Button(description="go", layout=W.Layout(width="50px"))
    frame_counter = W.HTML()
    context_html = W.HTML()
    label_status_html = W.HTML()

    # Per-label-type "not visible" buttons
    not_visible_btns = {
        lt: W.Button(
            description=f"× {lt}", layout=W.Layout(width="110px"),
            button_style="warning",
        )
        for lt in LABEL_TYPES
    }
    clear_btn = W.Button(
        description="clear label", layout=W.Layout(width="120px"),
    )
    reset_zoom_btn = W.Button(
        description="reset zoom", layout=W.Layout(width="120px"),
    )

    # ── Figure ───────────────────────────────────────────────────────────
    # Toolbar is left visible so the user can pan/zoom on the right axes via
    # the standard matplotlib controls. Toggle out of pan/zoom mode before
    # clicking to label, otherwise the click is consumed as a zoom action.
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Read first frame so we know the frame dimensions for axes extents.
    _first_frame = read_frame(frame_indices[0])
    if _first_frame is None:
        raise RuntimeError(f"could not read first frame ({frame_indices[0]})")
    _FRAME_H, _FRAME_W = _first_frame.shape[:2]

    # Persistent artists (one per label type, in each axes) — updated in-place
    LABEL_COLORS = {
        "screen_tl": "#ff4444", "screen_tr": "#44ff44",
        "screen_br": "#4488ff", "screen_bl": "#ffcc00",
        "box_bl": "#ff44ff", "box_br": "#44ffff",
        "big_star": "#ffffff",
    }
    _full_scatters = {
        lt: ax_full.plot([], [], "o", color=LABEL_COLORS[lt], markersize=8,
                         markeredgecolor="black", markeredgewidth=1)[0]
        for lt in LABEL_TYPES
    }
    _zoom_scatters = {
        lt: ax_zoom.plot([], [], "o", color=LABEL_COLORS[lt], markersize=12,
                         markeredgecolor="black", markeredgewidth=1.5)[0]
        for lt in LABEL_TYPES
    }
    # Both axes show the full frame; the right axes is initially limited to
    # the top-left region. User pans/zooms that region via the matplotlib
    # toolbar. Click events return data coords = full-frame coords on both
    # axes, so label coordinates are correct at any zoom level.
    _full_img = ax_full.imshow(_first_frame, extent=(0, _FRAME_W, _FRAME_H, 0))
    _zoom_img = ax_zoom.imshow(_first_frame, extent=(0, _FRAME_W, _FRAME_H, 0))
    ax_full.set_xlim(0, _FRAME_W); ax_full.set_ylim(_FRAME_H, 0)
    ax_zoom.set_xlim(*zoom_init_x_range)
    ax_zoom.set_ylim(zoom_init_y_range[1], zoom_init_y_range[0])  # inverted for image coords
    ax_full.set_title("full frame")
    ax_zoom.set_title("zoom (pan/zoom via toolbar above)")
    for a in (ax_full, ax_zoom):
        a.set_xticks([]); a.set_yticks([])

    # ── Render ───────────────────────────────────────────────────────────
    def _current_frame_idx() -> int:
        return frame_indices[_state["frame_pos"]]

    def _ctx_summary(frame_idx: int) -> str:
        ctx = trial_context_at(frame_idx)
        state = photodiode_state_at(frame_to_expt_ms(frame_idx))
        if ctx["trial_idx"] is None or ctx["time_since_reveal_ms"] is None:
            trial_line = "trial: <i>(before first reveal)</i>"
        else:
            trial_line = (
                f"trial <b>{ctx['trial_idx']}</b>, tpt <b>{ctx['tpt']}</b>, "
                f"t-since-reveal <b>{ctx['time_since_reveal_ms']/1000:.2f}s</b>"
            )
        return (
            f"<div>frame <b>{frame_idx}</b> "
            f"(video_t={frame_idx/fps:.2f}s)</div>"
            f"<div>{trial_line}</div>"
            f"<div>photodiode: <b style='color:"
            f"{('black' if state == 'black' else '#999')};"
            f"background-color:{('#eee' if state == 'black' else 'white')}'>"
            f"{state}</b></div>"
        )

    def _label_summary(frame_idx: int) -> str:
        parts = []
        for lt in LABEL_TYPES:
            rec = labels.get((frame_idx, lt))
            if rec is None or rec["visible"] is None:
                parts.append(f"<span style='color:#999'>{lt}: —</span>")
            elif rec["visible"] is False:
                parts.append(
                    f"<span style='color:#c80'>{lt}: not visible</span>"
                )
            else:
                parts.append(
                    f"<span style='color:#080'>{lt}: "
                    f"({rec['x_frame']:.0f},{rec['y_frame']:.0f}) "
                    f"[{rec['quality']}]</span>"
                )
        return " | ".join(parts)

    def refresh():
        frame_idx = _current_frame_idx()
        frame = read_frame(frame_idx)
        if frame is None:
            context_html.value = (
                f"<span style='color:red'>could not read frame {frame_idx}</span>"
            )
            return
        # set_data preserves the user's xlim/ylim — zoom persists across
        # frame navigation. Use the reset_zoom button to return to default.
        _full_img.set_data(frame)
        _zoom_img.set_data(frame)
        # Draw existing labels
        for lt in LABEL_TYPES:
            rec = labels.get((frame_idx, lt))
            if rec and rec["visible"]:
                xs, ys = [rec["x_frame"]], [rec["y_frame"]]
            else:
                xs, ys = [], []
            _full_scatters[lt].set_data(xs, ys)
            _zoom_scatters[lt].set_data(xs, ys)
        fig.canvas.draw_idle()
        frame_counter.value = (
            f"<b>{_state['frame_pos'] + 1} / {len(frame_indices)}</b>"
        )
        context_html.value = _ctx_summary(frame_idx)
        label_status_html.value = _label_summary(frame_idx)
        jump_field.value = frame_idx

    # ── Event handlers ───────────────────────────────────────────────────
    def on_click(event):
        if event.inaxes not in (ax_full, ax_zoom):
            return
        if event.xdata is None or event.ydata is None:
            return
        frame_idx = _current_frame_idx()
        x, y = float(event.xdata), float(event.ydata)
        lt = _state["active_label"]
        rec = _empty_record(frame_idx, lt)
        rec.update({
            "visible": True,
            "x_frame": x,
            "y_frame": y,
            "quality": _state["active_quality"],
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        labels[(frame_idx, lt)] = rec
        save_labels(labels)
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_label_dd(change):
        if change["name"] == "value":
            _state["active_label"] = change["new"]

    def on_quality_dd(change):
        if change["name"] == "value":
            _state["active_quality"] = change["new"]

    def on_prev(_):
        if _state["frame_pos"] > 0:
            _state["frame_pos"] -= 1
            refresh()

    def on_next(_):
        if _state["frame_pos"] < len(frame_indices) - 1:
            _state["frame_pos"] += 1
            refresh()

    def on_jump(_):
        target = int(jump_field.value)
        if target in frame_indices:
            _state["frame_pos"] = frame_indices.index(target)
            refresh()
        else:
            context_html.value += (
                f"<br><span style='color:red'>frame {target} not in "
                f"frame_indices</span>"
            )

    def _make_not_visible_handler(lt):
        def handler(_):
            frame_idx = _current_frame_idx()
            rec = _empty_record(frame_idx, lt)
            rec.update({
                "visible": False,
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            labels[(frame_idx, lt)] = rec
            save_labels(labels)
            refresh()
        return handler

    def on_clear(_):
        frame_idx = _current_frame_idx()
        lt = _state["active_label"]
        if (frame_idx, lt) in labels:
            del labels[(frame_idx, lt)]
            save_labels(labels)
            refresh()

    def on_reset_zoom(_):
        ax_zoom.set_xlim(*zoom_init_x_range)
        ax_zoom.set_ylim(zoom_init_y_range[1], zoom_init_y_range[0])
        fig.canvas.draw_idle()

    label_dd.observe(on_label_dd, names="value")
    quality_dd.observe(on_quality_dd, names="value")
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    jump_btn.on_click(on_jump)
    clear_btn.on_click(on_clear)
    reset_zoom_btn.on_click(on_reset_zoom)
    for lt, btn in not_visible_btns.items():
        btn.on_click(_make_not_visible_handler(lt))

    # ── Layout ───────────────────────────────────────────────────────────
    nav_row = W.HBox([prev_btn, next_btn, jump_field, jump_btn, frame_counter])
    label_row = W.HBox([label_dd, quality_dd, clear_btn, reset_zoom_btn])
    not_visible_row = W.VBox([
        W.HTML("<b>mark not visible:</b>"),
        W.HBox(list(not_visible_btns.values())),
    ])

    display(W.VBox([
        nav_row,
        context_html,
        label_row,
        fig.canvas,
        not_visible_row,
        label_status_html,
    ]))
    refresh()

# %% [markdown]
# ## Summary
# Run after labeling to see completion rates and box-bottom screen-coord
# back-projection. Re-runnable as labeling progresses.

# %%
if labels:
    df = pd.DataFrame(list(labels.values()))
    df = df.sort_values(["frame_idx", "label_type"]).reset_index(drop=True)

    # Per-frame completion
    frames_with_labels = sorted(df.frame_idx.unique())
    print(f"=== completion across {len(frames_with_labels)} frames ===")
    for lt in LABEL_TYPES:
        sub = df[df.label_type == lt]
        n_total = len(sub)
        n_visible = int((sub.visible == True).sum())
        n_not_visible = int((sub.visible == False).sum())
        print(f"  {lt:12s}: visible={n_visible:3d}  not_visible={n_not_visible:3d}  total_labeled={n_total:3d}")

    # All-4 screen corners labeled visible. Reindex to the full expected column
    # set so frames with only 2-of-4 labeled don't get counted as fully visible.
    pivot_screen = df[df.label_type.str.startswith("screen_")].pivot_table(
        index="frame_idx", columns="label_type", values="visible", aggfunc="first"
    ).reindex(columns=["screen_tl", "screen_tr", "screen_br", "screen_bl"])
    all_4_screen_visible = (pivot_screen == True).all(axis=1).sum() if len(pivot_screen) else 0
    print(f"\nFrames with all 4 screen corners visible: {all_4_screen_visible}")

    # Both box corners labeled visible.
    pivot_box = df[df.label_type.str.startswith("box_")].pivot_table(
        index="frame_idx", columns="label_type", values="visible", aggfunc="first"
    ).reindex(columns=["box_bl", "box_br"])
    both_box_visible = (pivot_box == True).all(axis=1).sum() if len(pivot_box) else 0
    print(f"Frames with both box corners visible: {both_box_visible}")

    # ── Box identifiability by photodiode state ──────────────────────────
    box_df = df[df.label_type.str.startswith("box_")].copy()
    box_df["expt_t_ms"] = box_df.frame_idx.map(frame_to_expt_ms)
    box_df["state"] = box_df.expt_t_ms.map(photodiode_state_at)
    print("\n=== box-corner identifiability by photodiode state ===")
    for state in ["white", "black", "unknown"]:
        sub = box_df[box_df.state == state]
        if not len(sub):
            continue
        n_visible = int((sub.visible == True).sum())
        n_total = len(sub)
        print(f"  state={state}: visible {n_visible}/{n_total} ({100*n_visible/n_total:.0f}%)")

    # ── Visible-strip height ─────────────────────────────────────────────
    box_pairs = pivot_box[(pivot_box == True).all(axis=1)].index
    heights = []
    for f in box_pairs:
        bl = labels.get((f, "box_bl"))
        br = labels.get((f, "box_br"))
        if bl and br and bl["visible"] and br["visible"]:
            # The "visible strip" is bounded by the iPad top edge and the box's
            # bottom edge. The bl/br points trace the bottom edge. Strip height
            # in frame-px ≈ distance from box bottom to iPad TL.
            screen_tl = labels.get((f, "screen_tl"))
            if screen_tl and screen_tl["visible"]:
                mid_y = (bl["y_frame"] + br["y_frame"]) / 2
                heights.append(mid_y - screen_tl["y_frame"])
    if heights:
        print(f"\n=== visible-strip height (frame-px, box-bottom vs screen-TL) ===")
        print(f"  n={len(heights)}  median={np.median(heights):.1f}  mean={np.mean(heights):.1f}  "
              f"min={np.min(heights):.1f}  max={np.max(heights):.1f}")

    # ── Box-bottom screen-coord back-projection ──────────────────────────
    # For frames with all 4 screen corners + both box corners visible, fit
    # H from screen→frame and back-project box_bl, box_br to screen coords.
    SCREEN_CORNERS = np.array(
        [[0, 0], [SCREEN_W_PX, 0], [SCREEN_W_PX, SCREEN_H_PX], [0, SCREEN_H_PX]],
        dtype=np.float32,
    )
    back_proj = []
    for f in frames_with_labels:
        need = ["screen_tl", "screen_tr", "screen_br", "screen_bl", "box_bl", "box_br"]
        rs = {lt: labels.get((f, lt)) for lt in need}
        if not all(r and r["visible"] for r in rs.values()):
            continue
        frame_pts = np.array([
            [rs["screen_tl"]["x_frame"], rs["screen_tl"]["y_frame"]],
            [rs["screen_tr"]["x_frame"], rs["screen_tr"]["y_frame"]],
            [rs["screen_br"]["x_frame"], rs["screen_br"]["y_frame"]],
            [rs["screen_bl"]["x_frame"], rs["screen_bl"]["y_frame"]],
        ], dtype=np.float32)
        H_screen_to_frame, _ = cv2.findHomography(SCREEN_CORNERS, frame_pts)
        H_frame_to_screen = np.linalg.inv(H_screen_to_frame)
        for box_lt in ["box_bl", "box_br"]:
            p_f = np.array([rs[box_lt]["x_frame"], rs[box_lt]["y_frame"], 1.0])
            p_s = H_frame_to_screen @ p_f
            sx, sy = p_s[0] / p_s[2], p_s[1] / p_s[2]
            back_proj.append({
                "frame_idx": f, "box_corner": box_lt,
                "screen_x": sx, "screen_y": sy,
                "state": photodiode_state_at(frame_to_expt_ms(f)),
            })
    if back_proj:
        bp = pd.DataFrame(back_proj)
        print(f"\n=== box corners back-projected to screen-coords (n={len(bp)}) ===")
        print(f"  (spec estimate: box at (0,0) screen-coord, visible strip ~{BOX_VISIBLE_STRIP_HEIGHT_SY} screen-px tall)")
        for corner in ["box_bl", "box_br"]:
            sub = bp[bp.box_corner == corner]
            print(f"  {corner}:")
            print(f"    screen_x: median={sub.screen_x.median():.1f}  min={sub.screen_x.min():.1f}  max={sub.screen_x.max():.1f}")
            print(f"    screen_y: median={sub.screen_y.median():.1f}  min={sub.screen_y.min():.1f}  max={sub.screen_y.max():.1f}")
else:
    print("(no labels yet)")

# %%
