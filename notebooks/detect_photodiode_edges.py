# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dot-prediction
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detect photodiode edges
#
# Emits one row per detected rise/fall on the task window. `t_peak` and `idx`
# are relative to the start of the task window (`expt_start_time` subtracted).

# %% tags=["parameters"]
photodiode_path = "results/EC347/photodiode.parquet"
meta_path = "results/EC347/photodiode_meta.json"
edges_out = "results/EC347/photodiode_edges.parquet"
expt_start_time = 30
expt_end_time = 1289
detector = {"smooth_env_s": 0.01, "min_sep_s": 0.1, "prom_z": 3.5}

# %%
import sys
sys.path.append("src")

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from photodiode import detect_events_env_direction

# %% [markdown]
# ## Load the raw photodiode trace

# %%
df = pd.read_parquet(photodiode_path)
with open(meta_path) as f:
    meta = json.load(f)
fs = meta["fs"]
fs, len(df), df.t.iloc[0], df.t.iloc[-1]

# %% [markdown]
# ## Crop to the task window
# `expt_start_time` / `expt_end_time` are relative to the start of the block
# recording, but `df.t` holds absolute Neuralynx timestamps — rebase before
# masking.

# %%
t_block = df["t"].values - df["t"].iloc[0]
mask = (t_block >= expt_start_time) & (t_block <= expt_end_time)
sub = df.loc[mask].reset_index(drop=True)
t_rel = sub["t"].values - sub["t"].iloc[0]  # rezero for plotting
len(sub), t_block[mask][0], t_block[mask][-1]

# %% [markdown]
# ### QC: raw trace
# Scroll through with the slider below — if edges are missed or spurious, adjust
# `detector` params (`prom_z`, `smooth_env_s`, `min_sep_s`).

# %%
f, ax = plt.subplots(figsize=(15, 3))
n_plot = min(len(sub), int(60 * fs))  # first 60 s
ax.plot(t_rel[:n_plot], sub["x"].values[:n_plot])
ax.set_xlabel("time (s, rezeroed)")
ax.set_ylabel("photodiode (a.u.)")
ax.set_title(f"first {n_plot / fs:.1f} s of task window")

# %% [markdown]
# ## Detect edges

# %%
edges = detect_events_env_direction(sub["x"].to_numpy(), fs=fs, **dict(detector))
n_rise = len(edges["rise"]["t_peaks"])
n_fall = len(edges["fall"]["t_peaks"])
print(f"{n_rise} rises, {n_fall} falls")

# %% [markdown]
# ### QC: detected edges overlaid on the trace
# Rise = subject click (screen going black). Fall = reveal (screen going white).

# %%
f, ax = plt.subplots(figsize=(15, 3))
ax.plot(t_rel[:n_plot], sub["x"].values[:n_plot])
for t in edges["rise"]["t_peaks"]:
    if t < n_plot / fs:
        ax.axvline(t, color="C1", alpha=0.5)
for t in edges["fall"]["t_peaks"]:
    if t < n_plot / fs:
        ax.axvline(t, color="C2", alpha=0.5)
ax.set_xlabel("time (s, rezeroed)")
ax.set_title(f"rise (C1) / fall (C2) edges, first {n_plot / fs:.1f} s")

# %% [markdown]
# ### QC: rise → fall spacing
# A fall should follow the preceding rise by ~1.5 s within a trial. Anything
# far from that is either an inter-trial gap or a miss.

# %%
n_pairs = min(n_rise, n_fall)
reveal_dt = (
    edges["fall"]["t_peaks"][1:n_pairs] - edges["rise"]["t_peaks"][: n_pairs - 1]
)
f, ax = plt.subplots(figsize=(10, 3))
ax.hist(reveal_dt, bins=80)
ax.axvline(1.5, color="red", linestyle="--", label="expected 1.5 s")
ax.set_xlabel("Δt from rise_i to fall_{i+1} (s)")
ax.legend()
outliers = np.where(~np.isclose(reveal_dt, 1.5, atol=0.1))[0]
print(f"{len(outliers)} edge pairs with reveal ≠ ~1.5s (likely inter-trial gaps)")

# %% [markdown]
# ### QC: fall → rise (response duration)

# %%
response_dt = edges["rise"]["t_peaks"][:n_pairs] - edges["fall"]["t_peaks"][:n_pairs]
f, ax = plt.subplots(figsize=(10, 3))
ax.hist(response_dt, bins=80)
ax.set_xlabel("Δt from fall_i to rise_i (s) [≈ RT]")

# %% [markdown]
# ## Save

# %%
from pathlib import Path

rows = []
for kind in ("rise", "fall"):
    for t_peak, t_half, idx in zip(
        edges[kind]["t_peaks"],
        edges[kind]["t_half"],
        edges[kind]["idx_peaks"],
    ):
        rows.append(
            {
                "kind": kind,
                "t_peak": float(t_peak),
                "t_half": float(t_half),
                "idx": int(idx),
            }
        )
out_df = pd.DataFrame(rows).sort_values("t_peak").reset_index(drop=True)

Path(edges_out).parent.mkdir(parents=True, exist_ok=True)
out_df.to_parquet(edges_out, index=False)
print(f"wrote {edges_out} ({len(out_df)} edges)")
