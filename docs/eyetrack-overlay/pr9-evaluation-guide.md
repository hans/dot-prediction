# PR #9 evaluation guide — big_star as 5th homography anchor

Walks through manually evaluating whether PR #9 (`feat/issue-8`) improves
gaze pipeline quality relative to main. Two worktrees run the pipeline
independently; a comparison notebook quantifies the differences.

---

## 1. Set up worktrees

```bash
# From the repo root (main worktree)
git worktree add ../dot-prediction-bigstar feat/issue-8
```

You now have:
- `/path/to/dot-prediction/` — main branch
- `/path/to/dot-prediction-bigstar/` — PR #9

Both share the same git objects and the same `data/` (via your normal data
path configuration). Each has its own `results/` directory.

---

## 2. Run both pipelines

Run these independently in each worktree. Both need the manual prerequisites
(`video_alignment.json`, `trials_with_video.parquet`,
`homography_labels.parquet`) to already exist — copy them from main into the
bigstar worktree's `results/EC347/` if they aren't there.

```bash
# In each worktree:
uv run snakemake -s Snakefile_eyetrack --configfile config_eyetrack.yaml \
    results/EC347/eyetrack/gaze_per_sample.parquet \
    results/EC347/eyetrack/fixation_events.parquet \
    -j1
```

Level 1 (homography quality) can be evaluated as soon as
`results/EC347/phase1c_per_frame.parquet` exists in both worktrees, before
`extract_gaze_fixations` completes.

---

## 3. Visual check — pipeline-generated plots

Each worktree produces these automatically. Open them side by side.

### `results/EC347/cascade_trajectory.png`

Box-corner and screen-corner Y positions vs frame index. Look for:
- Fewer sudden jumps in the PR branch
- Smoother tracking across the full video

### `results/EC347/big_star_residual_hist.png` and `big_star_residual_vs_frame.png`

At 19 hand-labeled frames: how far is the 4-anchor-only H from the labeled
big_star position. Both branches use the same 4-anchor refit here so this is
directly comparable. Lower is better; the PR should be roughly similar (this
metric is not what the PR improves — it's a sanity check that the PR didn't
break the box-corner calibration).

### `results/EC347/eyetrack/gaze_coverage_and_accuracy.png`

Two panels:
- **Left:** `homography_valid` and `on_screen` rates over time. PR should be
  equal or higher.
- **Right:** Pre-click gaze distance to target (canvas px), with a 250 px
  threshold line. PR acceptance criterion: median < 250 px.

### `results/EC347/eyetrack/q.png`

20 sampled clicks with gaze trajectories overlaid on the canvas. Look for
whether trajectories in the PR branch are better localized around the target
dot, especially for dots revealed near the top of the canvas (where the PR's
5th anchor matters most).

---

## 4. Run the comparison notebook

`compare_branches.py` is a jupytext percent-format notebook. Run it with
ploomber_engine (the same runner used by the Snakefile):

```bash
uv run python - <<'EOF'
from ploomber_engine import execute_notebook
import jupytext, tempfile
from pathlib import Path

nb = jupytext.read("notebooks/compare_branches.py")
with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
    tmp = Path(f.name)
jupytext.write(nb, tmp)

execute_notebook(str(tmp), "results/EC347/comparison/compare_branches.ipynb",
    parameters=dict(
        dir_a="/path/to/dot-prediction/results/EC347",
        label_a="main",
        dir_b="/path/to/dot-prediction-bigstar/results/EC347",
        label_b="feat/issue-8",
        trials_path="/path/to/dot-prediction/results/EC347/trials_with_video.parquet",
        out_dir="/path/to/dot-prediction/results/EC347/comparison",
    ))
tmp.unlink()
EOF
```

Or open `notebooks/compare_branches.py` directly in Jupyter (jupytext will
pair it as a notebook) and edit the `parameters` cell before running.

---

## 5. Interpret the comparison outputs

All written to `results/EC347/comparison/`.

### `h_summary.csv` — homography scalar table

| Metric | What to look for |
|---|---|
| `valid_H_%` | Should be equal or higher in PR |
| `\|TR_x\|>10k` | Should be lower in PR — this is the primary fix |
| `\|TR_x\|>100k` | Should be lower in PR |
| `\|TR_x\| p99` | Should be substantially lower in PR |
| `big_star_residual median` | Should be similar (≤ ~15 px either branch) |
| `5-anchor % (of detected)` | PR only; target ≥ 80% during trial windows |

### `tr_x_over_time.png`

TR_x (top-right corner projected x) vs frame index for each branch. The main
branch should show large spikes (O(100k–1M px)); the PR branch should show
the same spikes reduced in height or gone. If PR spikes are comparable to
main, the big_star detection rate may be too low during the bad-regime frames.

### `tr_x_hist.png`

Distribution overlay. PR branch tail should be substantially shorter.

### `gaze_summary.csv` — gaze scalar table

| Metric | What to look for |
|---|---|
| `homography_valid_%` | Equal or higher in PR |
| `on_screen_%` | Equal or higher in PR |

Pre-click accuracy is **not** included here. The PR uses big_star as a
calibration anchor, and pre-click gaze is always near the big_star (the dot
being clicked), so accuracy at that location is guaranteed by the fit rather
than demonstrated on held-out data. The right panel of
`gaze_coverage_and_accuracy.png` shows it but treat it as informational only.

### `canvas_shift.png` and `canvas_shift.parquet`

Where and by how much the PR changes projected gaze positions. Key questions:
- Is the shift concentrated near the top of the canvas? (Expected: the 5th
  anchor improves TR/TL conditioning, so the largest shifts should be near the
  top edge where extrapolation error was worst.)
- Is the shift distribution narrow (most samples shift < ~10 px)? Large
  widespread shifts would suggest the PR is moving gaze in ways not explained
  by the TR/TL fix alone.

---

## 6. Pass / fail criteria (from issue #8)

| Criterion | Source | Notes |
|---|---|---|
| TR_x catastrophic outliers reduced | `h_summary.csv` | Primary metric — `\|TR_x\|>100k` lower in PR |
| `homography_valid` no regression | `gaze_summary.csv` | PR ≥ main |
| `on_screen` no regression | `gaze_summary.csv` | PR ≥ main |
| `big_star_residual` no regression | `h_summary.csv` | PR median within ~5 px of main |
| Pre-click gaze accuracy | `gaze_coverage_and_accuracy.png` (right panel) | Informational only — circular for PR branch; big_star is both calibration anchor and click target |
