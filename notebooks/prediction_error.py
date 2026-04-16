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
# # Prediction-error contrasts
#
# Tests whether reveal-locked ECoG tracks the LoT model's posterior-weighted
# expected prediction error at each subtrial. `model_marg_prediction_error`
# (model-based expected surprise) is shown side-by-side with the subject's
# L2 click error (`prediction_error`, model-free). See
# `specs/high-level-science.md` claim #2.
#
# Subset: click subtrials (`tpt >= 3`) with non-null marginalized PE.

# %% tags=["parameters"]
epochs_path = "results/EC347/epochs/hga-epo.fif"
# Plotted window (seconds; t=0 = reveal), matching erps.py.
tmin = -1.0
tmax = 1.5

# %%
import sys
sys.path.append("src")

# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import permutation_cluster_test

from viz import plot_beta_traces, plot_epochs

# %%
epochs = mne.read_epochs(epochs_path, preload=True)
epochs_pe = epochs[
    "tpt >= 3 and model_marg_prediction_error == model_marg_prediction_error"
]
epochs_pe

# %% [markdown]
# ## Binary tercile contrast — LoT marginalized PE
# Top vs. bottom tercile of `model_marg_prediction_error`.

# %%
plot_epochs(
    epochs_pe,
    None,
    {
        "low_margPE": "model_marg_prediction_error_bin == 0",
        "high_margPE": "model_marg_prediction_error_bin == 2",
    },
    tmin=tmin,
    tmax=tmax,
)

# %% [markdown]
# ## Binary quintile contrast — subject L2 click error
# Comparison panel using the model-free surprise proxy
# (same `l2_error_bin` quintile split as `erps.py`).

# %%
plot_epochs(
    epochs_pe,
    None,
    {
        "low_l2": "l2_error_bin == 0",
        "high_l2": "l2_error_bin == 4",
    },
    tmin=tmin,
    tmax=tmax,
)

# %% [markdown]
# ## Cluster-based permutation stats — margPE binary contrast
# Per-channel permutation F-test on top vs. bottom margPE tercile, restricted
# to the post-reveal window [0, 1.0 s] — every trial has non-nan samples
# there. Covariate-adjusted regression-style cluster stats would require a
# custom permutation loop; we use the binary contrast as the formal test.

# %%
cluster_tmin, cluster_tmax = 0.0, 1.0
sub = epochs_pe.copy().crop(cluster_tmin, cluster_tmax)
hi = sub["model_marg_prediction_error_bin == 2"].get_data()
lo = sub["model_marg_prediction_error_bin == 0"].get_data()
assert not np.isnan(hi).any() and not np.isnan(lo).any()

n_channels = hi.shape[1]
n_times_cp = hi.shape[2]
sig_mask_margpe = np.zeros((n_channels, n_times_cp), dtype=bool)
cluster_p_margpe = np.full(n_channels, np.nan)
for ch in range(n_channels):
    _, clusters, cpv, _ = permutation_cluster_test(
        [hi[:, ch, :], lo[:, ch, :]],
        n_permutations=1000,
        out_type="indices",
        seed=0,
        verbose=False,
    )
    for c_idx, p in zip(clusters, cpv):
        if p < 0.05:
            sig_mask_margpe[ch, c_idx[0]] = True
    if len(cpv):
        cluster_p_margpe[ch] = cpv.min()

print(
    f"margPE: {(cluster_p_margpe < 0.05).sum()}/{n_channels} channels with a cluster at p<0.05"
)

# %% [markdown]
# ## Parametric regression — LoT marginalized PE (covariate-adjusted)
# For each (channel, time) fit
# ``y ~ margPE + tpt + rt_duration + intercept`` across subtrials. Plot β of
# `margPE`. Variable-length epochs are handled per-timepoint by dropping NaN
# trials. Pre-reveal (t<0) should be near zero — sanity null.

# %%
md = epochs_pe.metadata
y_all = epochs_pe.get_data()  # (n_trials, n_channels, n_times)
times = epochs_pe.times
n_trials, n_channels, n_times = y_all.shape


def regress_beta(regressor_col):
    X_full = np.column_stack(
        [
            np.ones(n_trials),
            md[regressor_col].to_numpy(),
            md["tpt"].to_numpy(),
            md["rt_duration"].to_numpy(),
        ]
    )
    X_valid_rows = ~np.isnan(X_full).any(axis=1)
    beta_of_interest = np.full((n_channels, n_times), np.nan)
    for t in range(n_times):
        y_t = y_all[:, :, t]  # (n_trials, n_channels)
        valid = X_valid_rows & ~np.isnan(y_t).any(axis=1)
        if valid.sum() < X_full.shape[1] + 1:
            continue
        b, *_ = np.linalg.lstsq(X_full[valid], y_t[valid], rcond=None)
        beta_of_interest[:, t] = b[1]  # margPE / subject-PE column
    return beta_of_interest


beta_margpe = regress_beta("model_marg_prediction_error")
plot_slice = slice(np.searchsorted(times, tmin), np.searchsorted(times, tmax))
plot_beta_traces(
    beta_margpe[:, plot_slice], times[plot_slice], ylabel="β(LoT margPE)"
)

# %% [markdown]
# ## Parametric regression — subject L2 click error (covariate-adjusted)
# Same design with `l2_error` (subject click error) as the regressor of
# interest. This is the model-free surprise proxy.

# %%
beta_subj = regress_beta("l2_error")
plot_beta_traces(
    beta_subj[:, plot_slice], times[plot_slice], ylabel="β(subject L2)"
)
