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

# %% [markdown]
# ## Summary across channels
#
# Channel-level views of the cluster test and the two regressions. All
# across-channel plots share a single row ordering — descending peak
# |β(margPE)| over the post-reveal window [0, 1] s — so rows in the margPE
# and subject-L2 heatmaps are directly comparable.

# %%
# Indices into the full epoch time axis.
t_post_lo = np.searchsorted(times, 0.0)
t_post_hi = np.searchsorted(times, 1.0)

peak_abs_margpe = np.nanmax(np.abs(beta_margpe[:, t_post_lo:t_post_hi]), axis=1)
peak_abs_subj = np.nanmax(np.abs(beta_subj[:, t_post_lo:t_post_hi]), axis=1)
chan_order = np.argsort(-peak_abs_margpe)

# Project the binary-contrast significance mask back onto the full time axis,
# then into the plotting window.
cp_lo = np.searchsorted(times, cluster_tmin)
sig_mask_full = np.zeros((n_channels, n_times), dtype=bool)
sig_mask_full[:, cp_lo : cp_lo + n_times_cp] = sig_mask_margpe
sig_mask_plot = sig_mask_full[:, plot_slice]
times_plot = times[plot_slice]

# %% [markdown]
# ### Significant-cluster inventory (binary margPE contrast, p<0.05)

# %%
import pandas as pd

cluster_rows = []
for ch in range(n_channels):
    if not sig_mask_margpe[ch].any():
        continue
    on = np.flatnonzero(sig_mask_margpe[ch])
    # Group consecutive indices into (start, stop) windows.
    splits = np.where(np.diff(on) > 1)[0]
    groups = np.split(on, splits + 1)
    for g in groups:
        cluster_rows.append({
            "channel": ch,
            "t_start": float(times[cp_lo + g[0]]),
            "t_stop": float(times[cp_lo + g[-1]]),
            "duration_s": float(times[cp_lo + g[-1]] - times[cp_lo + g[0]]),
            "cluster_p_min_in_ch": float(cluster_p_margpe[ch]),
            "peak_abs_beta_margpe": float(peak_abs_margpe[ch]),
        })
cluster_table = pd.DataFrame(cluster_rows).sort_values("cluster_p_min_in_ch")
cluster_table

# %% [markdown]
# ### β(LoT margPE; t) across channels
# Rows sorted by peak |β(margPE)| in [0, 1] s. Diverging colormap centered at
# 0. t=0 marks reveal.

# %%
def _beta_heatmap(beta_matrix, row_order, title):
    data = beta_matrix[:, plot_slice][row_order]
    vmax = np.nanpercentile(np.abs(data), 99)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.08 * n_channels)))
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        extent=[times_plot[0], times_plot[-1], n_channels - 0.5, -0.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("time from reveal (s)")
    ax.set_ylabel("channel (sorted by peak |β(margPE)|)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="β")
    return fig, ax


_beta_heatmap(beta_margpe, chan_order, "β(LoT margPE; t) — covariate-adjusted")

# %% [markdown]
# ### β(subject L2; t) across channels — same row order
# Directly comparable to the margPE heatmap above.

# %%
_beta_heatmap(beta_subj, chan_order, "β(subject L2; t) — covariate-adjusted")

# %% [markdown]
# ### Binary-contrast significance map
# White = time-points in a significant margPE cluster (p<0.05). Same row
# order as the β heatmaps.

# %%
fig, ax = plt.subplots(figsize=(10, max(3, 0.08 * n_channels)))
ax.imshow(
    sig_mask_plot[chan_order].astype(float),
    aspect="auto",
    interpolation="nearest",
    extent=[times_plot[0], times_plot[-1], n_channels - 0.5, -0.5],
    cmap="Greys",
    vmin=0,
    vmax=1,
)
ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("time from reveal (s)")
ax.set_ylabel("channel (same order)")
ax.set_title("Significant margPE cluster membership (binary, p<0.05)")

# %% [markdown]
# ### Top-K channels — β traces with cluster shading
# Top-K by peak |β(margPE)| in [0, 1] s. Each facet overlays β(margPE) and
# β(subject L2) on the same axis; shaded band marks the binary-contrast
# margPE cluster window for that channel.

# %%
K = min(6, n_channels)
top_channels = chan_order[:K].tolist()
n_cols = 2
n_rows = (K + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.2 * n_rows), sharex=True)
for i, ch in enumerate(top_channels):
    ax = axes.flat[i]
    ax.plot(times_plot, beta_margpe[ch, plot_slice], color="C0", label="margPE")
    ax.plot(times_plot, beta_subj[ch, plot_slice], color="C1", label="subject L2")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    if sig_mask_plot[ch].any():
        ymin, ymax = ax.get_ylim()
        ax.fill_between(
            times_plot, ymin, ymax,
            where=sig_mask_plot[ch],
            color="C3", alpha=0.15, step="mid", linewidth=0,
        )
        ax.set_ylim(ymin, ymax)
    p = cluster_p_margpe[ch]
    p_str = f"cluster p={p:.3g}" if not np.isnan(p) else "no cluster"
    ax.set_title(f"ch {ch} | peak|β|={peak_abs_margpe[ch]:.3f} | {p_str}")
    if i == 0:
        ax.legend(fontsize=8, loc="upper right")
for j in range(K, n_rows * n_cols):
    axes.flat[j].axis("off")
fig.supxlabel("time from reveal (s)")
fig.supylabel("β")
fig.tight_layout()

# %% [markdown]
# ### Per-channel effect scatter
# Peak |β(margPE)| vs peak |β(subject L2)| in [0, 1] s. Channels with a
# significant margPE cluster are highlighted. Points above the unity line
# mean the model-based regressor explains more neural variance than the
# model-free one, at that channel's peak.

# %%
sig_channels = cluster_p_margpe < 0.05
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(
    peak_abs_subj[~sig_channels], peak_abs_margpe[~sig_channels],
    color="lightgray", s=20, edgecolor="none", label="ns",
)
ax.scatter(
    peak_abs_subj[sig_channels], peak_abs_margpe[sig_channels],
    color="C3", s=35, edgecolor="black", linewidth=0.5, label="cluster p<0.05",
)
lim = float(np.nanmax([peak_abs_subj.max(), peak_abs_margpe.max()]) * 1.05)
ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.set_xlabel("peak |β(subject L2)| in [0, 1] s")
ax.set_ylabel("peak |β(LoT margPE)| in [0, 1] s")
ax.legend(fontsize=8)
