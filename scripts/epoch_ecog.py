"""Epoch preprocessed Hilbert-envelope ECoG around subtrial onsets, one FIF per band.

Uses all subtrials with a valid `rt_duration`; downstream contrasts are defined
by filtering on `epochs.metadata`.
"""
import h5py
import pandas as pd

from neuralynx_to_mne import epoch_ecog_mne

trials = pd.read_parquet(snakemake.input.trials)
trials = trials.dropna(subset=["rt_duration"]).reset_index(drop=True)

with h5py.File(snakemake.input.ecog_h5, "r") as hf:
    ecog_data = hf["ds_X_abs"][:]
    freq_centers = hf.attrs["filter_center"]

bands = list(snakemake.params.bands)
epochs_per_band = epoch_ecog_mne(
    ecog_data=ecog_data,
    start_times=trials.subtrial_onset.to_numpy(),
    end_times=trials.subtrial_offset.to_numpy(),
    baseline_times=trials[["baseline_onset", "baseline_offset"]].to_numpy(),
    frequency_bands=bands,
    freq_centers=freq_centers,
    epoch_metadata=trials,
    final_hz=snakemake.params.final_hz,
    buffer_before=snakemake.params.buffer_before,
    buffer_after=snakemake.params.buffer_after,
)

outputs = list(snakemake.output.epochs)
assert len(outputs) == len(bands), (bands, outputs)
for band, epochs, outpath in zip(bands, epochs_per_band, outputs):
    epochs.save(outpath, overwrite=True, split_size="2GB")
    print(f"wrote {outpath}")
