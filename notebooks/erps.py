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
# # ERP contrasts
#
# Reads epochs produced by the `epoch_ecog` rule and defines contrasts via
# `epochs.metadata.query(...)`. Edit and add new contrasts here freely.

# %% tags=["parameters"]
epochs_path = "results/EC347/epochs/hga-epo.fif"
channel_indices = [40, 41, 66, 67, 69, 70]
# Plotted window (seconds; t=0 = reveal). Epochs extend past the click by
# `buffer_after/final_hz` and into the baseline by `buffer_before/final_hz`,
# but we only want to look at the ERP vicinity.
tmin = -1.0
tmax = 1.5

# %%
import sys
sys.path.append("src")

# %%
import mne

from viz import plot_epochs

# %%
epochs = mne.read_epochs(epochs_path, preload=True)
epochs

# %% [markdown]
# ## All subtrials (no contrast)

# %%
plot_epochs(epochs, channel_indices, tmin=tmin, tmax=tmax)

# %% [markdown]
# ## Previous-trial-error contrast
# Contrast subtrials whose preceding response had low vs. high L2 error.

# %%
plot_epochs(
    epochs,
    channel_indices,
    {
        "low": "l2_error_previous_bin == 0",
        "high": "l2_error_previous_bin == 4",
    },
    tmin=tmin,
    tmax=tmax,
)

# %% [markdown]
# ## Early vs. late subtrials within a trial

# %%
plot_epochs(
    epochs,
    channel_indices,
    {
        "early": "tpt in [3, 4, 5]",
        "late": "tpt in [12, 13, 14]",
    },
    tmin=tmin,
    tmax=tmax,
)
