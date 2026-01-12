import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns



def plot_epochs(epochs: mne.EpochsArray,
                channel_indices: list[int] = None,
                condition_queries: dict[str] = None,
                smoke_test=False):
    """
    Plot faceted epochs data, one electrode per facet.
    Optionally contrast epochs by conditions specified in condition_queries.
    """

    if condition_queries is not None:
        assert epochs.metadata is not None, "Epochs must have metadata to use condition queries."
    else:
        condition_queries = {None: None}
    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

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
        'channel_idx': channel_indices if channel_indices is not None else range(data.shape[1]),
    })
    # plot_df is a cross product of electrode_df and condition_data
    plot_df = pd.merge(
        electrode_df,
        all_epochs.reset_index(),
        how="cross"
    )
    
    ####

    g = sns.FacetGrid(plot_df, col="channel_idx", col_wrap=2, sharey=False, height=3)

    def plot_facet(data, color, **kwargs):
        ax = plt.gca()
        channel_idx = data['channel_idx'].iloc[0]

        for condition, condition_data in data.groupby('condition'):
            epoch_indices = condition_data['epoch_idx'].values
            data_ij = epoch_data[epoch_indices, channel_idx]

            data_ij_mean = np.mean(data_ij, axis=0)
            data_ij_sem = np.std(data_ij, axis=0) / np.sqrt(len(epoch_indices))
            times = epochs.times

            ax.plot(times, data_ij_mean, label=str(condition) if condition is not None else "all", **kwargs)
            ax.fill_between(times,
                            data_ij_mean - data_ij_sem,
                            data_ij_mean + data_ij_sem,
                            alpha=0.3)
            
    g.map_dataframe(plot_facet)

    return g