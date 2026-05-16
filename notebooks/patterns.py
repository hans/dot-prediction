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
# # Pattern analyses: RSA + PCA decoding
#
# Reads epochs produced by the `epoch_ecog` rule.

# %% tags=["parameters"]
epochs_path = "results/EC347/epochs/hga-epo.fif"

# %%
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from loguru import logger as L
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# %%
epochs = mne.read_epochs(epochs_path, preload=True)
epochs


# %% [markdown]
# ## Representational similarity
#
# Averages each epoch over the analysis window, optionally reduces via PCA,
# then builds a condition × condition distance matrix (mean pairwise distance
# between trials in conditions `i` and `j`).

# %%
def compute_rsa(
    epochs: mne.Epochs,
    rsa_tmin: float,
    rsa_tmax: float,
    conditions: dict[str, str],
    channel_idxs: list[int],
    pca_dim: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    ret = np.zeros((len(conditions), len(conditions)))

    data = epochs.get_data()
    smin, smax = epochs.time_as_index([rsa_tmin, rsa_tmax])
    data = data[:, channel_idxs, :][:, :, smin:smax].mean(axis=2)
    data = data.reshape(data.shape[0], -1)

    data = StandardScaler().fit_transform(data)
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        data = pca.fit_transform(data)
        L.info(
            f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%"
            f" for {pca_dim} components"
        )

    condition_epochs = {
        cond: epochs.metadata.query(expr).index for cond, expr in conditions.items()
    }

    for i, (_, idxs_i) in enumerate(tqdm(condition_epochs.items())):
        for j, (_, idxs_j) in enumerate(condition_epochs.items()):
            ret[i, j] = cdist(data[idxs_i], data[idxs_j], metric=metric).mean()

    return ret


# %% [markdown]
# ### Early vs. late subtrial

# %%
compute_rsa(
    epochs,
    0.0,
    0.5,
    conditions={"early": "tpt in [3, 4, 5]", "late": "tpt in [12, 13, 14]"},
    channel_idxs=list(range(40)),
    pca_dim=20,
)

# %% [markdown]
# ### Low vs. high previous-trial error

# %%
compute_rsa(
    epochs,
    -0.2,
    0.2,
    conditions={
        "low": "l2_error_previous_bin == 0",
        "high": "l2_error_previous_bin == 4",
    },
    channel_idxs=list(range(40)),
    pca_dim=20,
)

# %% [markdown]
# ## PCA decoding scratch
#
# Fit PCA on a subset of channels over a response window; visualize how the
# top-10 stimulus patterns separate in the first two PC dimensions.

# %%
pca_channels = list(range(20, 41))
pca_tmin, pca_tmax = 0.0, 0.5
pca_dim = 10

patterns = epochs.metadata.seq_id.unique()[:10]
mask = epochs.metadata.seq_id.isin(patterns)
epoch_idxs = np.where(mask.values)[0]
labels = epochs.metadata.loc[mask, "seq_id"].values

smin, smax = epochs.time_as_index([pca_tmin, pca_tmax])
data = epochs.get_data()[epoch_idxs][:, pca_channels, smin:smax]
data = data.reshape(data.shape[0], -1)
data = StandardScaler().fit_transform(data)

pca = PCA(n_components=pca_dim)
proj = pca.fit_transform(data)
L.info(
    f"PCA explained variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%"
    f" for {pca_dim} components"
)

# %%
f, ax = plt.subplots(figsize=(10, 10))
for label in np.unique(labels):
    idxs = np.where(labels == label)[0]
    ax.scatter(proj[idxs, 0], proj[idxs, 1], label=label)
ax.legend()
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
