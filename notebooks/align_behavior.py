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
# # Align behavior with photodiode edges
#
# Executed by Snakemake's `align_behavior` rule via ploomber_engine. Inputs /
# output are injected into the `parameters` cell below.

# %% tags=["parameters"]
behavior_path = "data/EC347/behavior/data.csv"
model_outputs_path = "data/EC347/model_outputs/model_outputs.csv"
edges_path = "results/EC347/photodiode_edges.parquet"
trials_out = "results/EC347/trials.parquet"

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Behavior
# Drop trial 0 (warmup; ECoG wasn't recorded for it). Compute a per-row reaction
# time (response_time minus the preceding reveal_time).

# %%
trials = pd.read_csv(behavior_path).sort_values(["trial_idx", "tpt"])
trials["prev_reveal_time"] = trials.reveal_time.shift()
trials["rt_duration"] = trials.response_time - trials.prev_reveal_time
trials = trials[trials.trial_idx != 0].reset_index(drop=True)
trials.head()

# %% [markdown]
# ## Photodiode edges
# Each (fall_i, rise_i) pair brackets one behavioral subtrial: screen turns
# white at `fall_i`, subject clicks at `rise_i`.

# %%
edges = pd.read_parquet(edges_path)
rise = edges[edges.kind == "rise"].reset_index(drop=True)
fall = edges[edges.kind == "fall"].reset_index(drop=True)

n_pairs = min(len(rise), len(fall))
anin_bounds = np.column_stack(
    [fall.t_peak.values[:n_pairs], rise.t_peak.values[:n_pairs]]
)
anin_durations = anin_bounds[:, 1] - anin_bounds[:, 0]
len(rise), len(fall)

# %% [markdown]
# ### QC: edge spacing
# Within a trial, a fall should follow the previous rise by ~1.5 s (the reveal
# duration). Anomalies here usually mean a missed edge.

# %%
f, ax = plt.subplots(figsize=(10, 3))
dts = fall.t_peak.values[1:n_pairs] - rise.t_peak.values[: n_pairs - 1]
ax.hist(dts, bins=80)
ax.axvline(1.5, color="red", linestyle="--", label="expected 1.5 s")
ax.set_xlabel("Δt from rise_i to fall_{i+1} (s)")
ax.legend()
outliers = np.where(~np.isclose(dts, 1.5, atol=0.1))[0]
print(f"{len(outliers)} edge pairs with reveal ≠ ~1.5s (likely inter-trial gaps)")

# %% [markdown]
# ## Alignment loop
# For each trial we consume N-3 ANIN-detected response durations, check they
# agree with the behaviorally reported RTs, then record the subtrial onsets/
# offsets in the photodiode timebase. The first ANIN duration of each trial
# also contains the three intro reveals (1+1.5+1.5 = 4 s) which we strip.

# %%
assert trials.trial_idx.is_monotonic_increasing

anin_offset = 0
subtrial_onset_all, subtrial_offset_all = [], []
for trial_idx, rows in trials.groupby("trial_idx"):
    n_responses = len(rows) - 3  # first 3 subtrials are intro reveals (no click)
    proposed = anin_durations[anin_offset : anin_offset + n_responses].copy()
    behavioral_rts = rows.iloc[3:].rt_duration.to_numpy() / 1000.0
    proposed[0] -= 4.0  # strip intro-reveal portion from first response window
    np.testing.assert_allclose(proposed, behavioral_rts, atol=0.1)

    onsets = [float(fall.t_peak.iloc[anin_offset + i]) for i in range(n_responses)]
    offsets = [float(rise.t_peak.iloc[anin_offset + i]) for i in range(n_responses)]

    anin_offset += n_responses
    np.testing.assert_allclose(
        anin_durations[anin_offset],
        1.0,
        atol=0.1,
        err_msg=f"refresh flip at idx {anin_offset} (trial {trial_idx}) != ~1.0s",
    )
    anin_offset += 1

    # Prepend the three intro reveals, each 1.5 s, starting 1 s after the first fall.
    t0 = onsets[0]
    onsets = [t0, t0 + 1.5, t0 + 2.5, t0 + 4.0] + onsets[1:]
    offsets = [t0 + 1.0, t0 + 3.0, t0 + 4.0, offsets[0]] + offsets[1:]

    subtrial_onset_all.extend(onsets)
    subtrial_offset_all.extend(offsets)

trials["subtrial_onset"] = subtrial_onset_all
trials["subtrial_offset"] = subtrial_offset_all
trials["subtrial_duration"] = trials.subtrial_offset - trials.subtrial_onset
trials["baseline_onset"] = (
    trials.groupby("trial_idx")["subtrial_onset"].transform("min") - 1.0
)
trials["baseline_offset"] = trials.groupby("trial_idx")["subtrial_onset"].transform(
    "min"
)
trials["baseline_duration"] = trials.baseline_offset - trials.baseline_onset

# %% [markdown]
# ### QC: RT consistency
# Photodiode-derived subtrial duration should match the behavioral RT.

# %%
pd.testing.assert_series_equal(
    trials.dropna(subset=["rt_duration"])["subtrial_duration"],
    trials.dropna(subset=["rt_duration"])["rt_duration"] / 1000,
    check_names=False,
    atol=0.1,
)
print("subtrial_duration == rt_duration ✓")

# %% [markdown]
# ### QC: trial-bounds overlay
# Ground-truth black periods (from behavior log) vs. ANIN-detected black
# periods (photodiode fall→rise). These should line up.

# %%
def _behavioral_trial_bounds(rows):
    bounds = [(rows.trial_onset.iloc[0], rows[~rows.correct.isna()].response_time.iloc[0])]
    bounds.extend(
        np.column_stack(
            [rows.reveal_time.shift().values[4:], rows.response_time.values[4:]]
        )
    )
    bounds.append((rows.reveal_time.iloc[-1], rows.trial_offset.iloc[0]))
    return np.array(bounds)


behavioral_bounds = np.concatenate(
    trials.groupby("trial_idx").apply(_behavioral_trial_bounds).tolist()
)
t_origin_beh = behavioral_bounds[0, 0]
behavioral_bounds_s = (behavioral_bounds - t_origin_beh) / 1000.0

anin_bounds_s = anin_bounds - anin_bounds[0, 0]

f, ax = plt.subplots(figsize=(14, 3))
for start, end in behavioral_bounds_s:
    ax.axvspan(start, end, ymin=0.0, ymax=0.5, color="C0", alpha=0.3)
for start, end in anin_bounds_s:
    ax.axvspan(start, end, ymin=0.5, ymax=1.0, color="C1", alpha=0.3)
ax.text(0.005, 0.25, "behavioral (ground truth)", color="C0",
        transform=ax.get_yaxis_transform(), fontsize=10)
ax.text(0.005, 0.75, "ANIN (photodiode)", color="C1",
        transform=ax.get_yaxis_transform(), fontsize=10)
ax.set_xlabel("time (s, rezeroed)")
ax.set_yticks([])

# %% [markdown]
# ## Derived metadata
# L2 response error + equal-frequency bin, available to downstream analyses as
# `epochs.metadata` columns for contrast definitions.

# %%
trials["l2_error"] = np.sqrt(
    (trials.response_x - trials.true_x) ** 2
    + (trials.response_y - trials.true_y) ** 2
)
trials["l2_error_bin"] = pd.qcut(trials["l2_error"], q=5, labels=False)
trials["l2_error_previous_bin"] = trials.groupby("trial_idx")["l2_error_bin"].shift()

f, ax = plt.subplots(figsize=(6, 3))
ax.hist(trials.l2_error.dropna(), bins=40)
ax.set_xlabel("L2 error (px)")
ax.set_ylabel("count")

# %% [markdown]
# ## LoT marginalized prediction error
# Merge the posterior-weighted expected PE from the LoT particle model onto
# each subtrial so downstream analyses can contrast neural activity by the
# model's *expected* surprise at reveal. Model-outputs rows are at
# ``(seq_id, tpt, model, model_particle)`` grain; `model_marg_*` is constant
# within a `(seq_id, tpt, model)` group so we take the first row.

# %%
model_outputs = pd.read_csv(model_outputs_path)
lot = model_outputs[model_outputs.model == "LoT"]
lot_marg = (
    lot.groupby(["seq_id", "tpt", "seq_attempt"], as_index=False)[
        ["model_marg_prediction_error", "model_marg_relative_prediction_error"]
    ]
    .first()
)

trials = trials.merge(lot_marg, on=["seq_id", "tpt", "seq_attempt"], how="left")

# Tercile bin for the binary high/low contrast (parallels l2_error_bin with q=5).
trials["model_marg_prediction_error_bin"] = pd.qcut(
    trials["model_marg_prediction_error"], q=3, labels=False
)

# QC: every click subtrial (tpt>=3) should have a marg PE.
missing = trials[(trials.tpt >= 3) & trials.model_marg_prediction_error.isna()]
assert missing.empty, (
    f"{len(missing)} subtrials with tpt>=3 are missing model_marg_prediction_error"
)

f, ax = plt.subplots(figsize=(6, 3))
ax.hist(trials.model_marg_prediction_error.dropna(), bins=40)
ax.set_xlabel("LoT marginalized PE")
ax.set_ylabel("count")

# %%
from pathlib import Path
Path(trials_out).parent.mkdir(parents=True, exist_ok=True)
trials.to_parquet(trials_out, index=False)
print(f"wrote {trials_out} ({len(trials)} rows)")
