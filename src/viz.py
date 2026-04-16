import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns



def plot_epochs(epochs: mne.EpochsArray,
                channel_indices: list[int] = None,
                condition_queries: dict[str] = None,
                tmin=None,
                tmax=None,
                smoke_test=False,
                **facet_kwargs,
                ) -> sns.FacetGrid:
    """
    Plot faceted epochs data, one electrode per facet.
    Optionally contrast epochs by conditions specified in condition_queries.
    `tmin` / `tmax` crop the plotted window (in seconds, with t=0 at reveal).
    """

    if condition_queries is None:
        # Use a string key so groupby('condition') picks it up — it drops NaN keys.
        condition_queries = {"all": None}
    else:
        assert epochs.metadata is not None, "Epochs must have metadata to use condition queries."

    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    smin = 0
    if tmin is not None:
        assert tmin >= epochs.times[0], "tmin precedes the minimum time of the epochs."
        smin = np.searchsorted(epochs.times, tmin)
    smax = len(epochs.times)
    if tmax is not None:
        assert tmax <= epochs.times[-1], "tmax exceeds the maximum time of the epochs."
        smax = np.searchsorted(epochs.times, tmax)

    condition_data = {}
    for label, query in condition_queries.items():
        if query is None:
            selected_epochs_md = epochs.metadata
        else:
            selected_epochs_md = epochs.metadata.query(query)
        condition_data[label] = selected_epochs_md
    all_epochs = pd.concat(condition_data, names=["condition", "epoch_idx"])

    # make a dummy df to generate facetgrid
    electrode_df = pd.DataFrame({
        'channel_idx': channel_indices if channel_indices is not None else list(range(epoch_data.shape[1])),
    })
    # plot_df is a cross product of electrode_df and condition_data
    plot_df = pd.merge(
        electrode_df,
        all_epochs.reset_index(),
        how="cross"
    )

    facet_kwargs = {
        **dict(col_wrap=2, sharey=False, height=3),
        **facet_kwargs
    }

    g = sns.FacetGrid(plot_df, col="channel_idx", **facet_kwargs)

    palette = sns.color_palette(n_colors=len(condition_queries))
    condition_colors = dict(zip(condition_queries.keys(), palette))
    show_legend = len(condition_queries) > 1

    def plot_facet(data, **kwargs):
        ax = plt.gca()
        channel_idx = data['channel_idx'].iloc[0]

        for condition, cond_df in data.groupby('condition'):
            epoch_indices = cond_df['epoch_idx'].values

            times = epochs.times[smin:smax]
            data_ij = epoch_data[epoch_indices, channel_idx, smin:smax]

            data_ij_mean = np.nanmean(data_ij, axis=0)
            data_ij_sem = np.nanstd(data_ij, axis=0) / np.sqrt(len(epoch_indices))

            color = condition_colors[condition]
            ax.plot(times, data_ij_mean, color=color, label=str(condition))
            ax.fill_between(times,
                            data_ij_mean - data_ij_sem,
                            data_ij_mean + data_ij_sem,
                            color=color, alpha=0.3)

        # t=0 is the photodiode fall — i.e. screen-turns-white, the reveal.
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        if show_legend:
            ax.legend(fontsize=8)

    g.map_dataframe(plot_facet)
    g.set_axis_labels("time from reveal (s)", "activity (z)")
    g.set_titles(col_template="channel {col_name}")

    return g


def plot_beta_traces(beta: np.ndarray,
                     times: np.ndarray,
                     channel_indices: list[int] = None,
                     sig_mask: np.ndarray = None,
                     ylabel: str = "β",
                     **facet_kwargs,
                     ) -> sns.FacetGrid:
    """
    Plot per-channel β(t) traces with shaded significance bands.

    - ``beta``: shape (n_channels, n_times).
    - ``sig_mask``: optional boolean array, same shape as ``beta``; True marks
      significant time-points (e.g. time-points inside a cluster with
      cluster-permutation p < alpha). Shaded vertically on each facet.
    """
    n_channels, n_times = beta.shape
    assert times.shape == (n_times,), (times.shape, n_times)
    if sig_mask is not None:
        assert sig_mask.shape == beta.shape, (sig_mask.shape, beta.shape)

    if channel_indices is None:
        channel_indices = list(range(n_channels))

    plot_df = pd.DataFrame({"channel_idx": channel_indices})

    facet_kwargs = {**dict(col_wrap=2, sharey=False, height=3), **facet_kwargs}
    g = sns.FacetGrid(plot_df, col="channel_idx", **facet_kwargs)

    def plot_facet(data, **kwargs):
        ax = plt.gca()
        ch = int(data["channel_idx"].iloc[0])
        ax.plot(times, beta[ch], color="C0")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        if sig_mask is not None and sig_mask[ch].any():
            ymin, ymax = ax.get_ylim()
            ax.fill_between(times, ymin, ymax,
                            where=sig_mask[ch], color="C3", alpha=0.15,
                            step="mid", linewidth=0)
            ax.set_ylim(ymin, ymax)

    g.map_dataframe(plot_facet)
    g.set_axis_labels("time from reveal (s)", ylabel)
    g.set_titles(col_template="channel {col_name}")

    return g