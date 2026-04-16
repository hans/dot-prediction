import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns



def plot_epochs(epochs: mne.EpochsArray,
                channel_indices: list[int] = None,
                condition_queries: dict[str] = None,
                tmax=None,
                smoke_test=False,
                **facet_kwargs,
                ) -> sns.FacetGrid:
    """
    Plot faceted epochs data, one electrode per facet.
    Optionally contrast epochs by conditions specified in condition_queries.
    """

    if condition_queries is None:
        # Use a string key so groupby('condition') picks it up — it drops NaN keys.
        condition_queries = {"all": None}
    else:
        assert epochs.metadata is not None, "Epochs must have metadata to use condition queries."

    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    smax = None
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

            times = epochs.times
            data_ij = epoch_data[epoch_indices, channel_idx]
            if smax is not None:
                times = times[:smax]
                data_ij = data_ij[:, :smax]

            data_ij_mean = np.nanmean(data_ij, axis=0)
            data_ij_sem = np.nanstd(data_ij, axis=0) / np.sqrt(len(epoch_indices))

            color = condition_colors[condition]
            ax.plot(times, data_ij_mean, color=color, label=str(condition))
            ax.fill_between(times,
                            data_ij_mean - data_ij_sem,
                            data_ij_mean + data_ij_sem,
                            color=color, alpha=0.3)

        if show_legend:
            ax.legend(fontsize=8)

    g.map_dataframe(plot_facet)

    return g