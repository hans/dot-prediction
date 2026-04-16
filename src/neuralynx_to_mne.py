from typing import Optional

import mne
import numpy as np
import pandas as pd
from scipy.stats import zscore
from loguru import logger as L

from ecog import get_band_indices, check_frequency_band


def check_presentation_df(presentation_df):
    # Validate required columns
    required_columns = ['stimulus', 'stim_index', 'trial_index', 'block', 'wav_path', 'stimulus_start', 'stimulus_stop', 'score']
    missing_columns = [col for col in required_columns if col not in presentation_df.columns]

    if missing_columns:
        L.warning(f"Missing required columns: {missing_columns}")

    # Extract timing information
    start_times = np.array(presentation_df["stimulus_start"])
    end_times = np.array(presentation_df["stimulus_stop"])

    L.info(f"📊 Stimulus timing summary:")
    L.info(f"   Number of trials: {len(start_times)}")
    L.info(f"   Start time range: {start_times.min():.1f} - {start_times.max():.1f} seconds")
    L.info(f"   Duration range: {(end_times - start_times).min():.1f} - {(end_times - start_times).max():.1f} seconds")
    L.info(f"   Average duration: {(end_times - start_times).mean():.1f} seconds")
    # Display first few rows
    L.info(f"\n📋 First 5 trials:")
    L.info(f"\n{presentation_df.head()}")

    return presentation_df


def epoch_ecog(ecog_data: np.ndarray,
               start_times: np.ndarray,
               end_times: np.ndarray,
               final_hz, buffer_before, buffer_after,
               baseline_times: Optional[np.ndarray] = None,
               ) -> tuple[np.ndarray, list]:
    """
    Segments ECoG data around stimulus events and z-scores each epoch.
    
    Parameters:
    - ecog_data: np.ndarray of shape [frequencies, channels, time]
    - start_times: np.ndarray of stimulus start times (in seconds)
    - end_times: np.ndarray of stimulus end times (in seconds)
    - final_hz: final sampling rate (in Hz)
    - buffer_before: number of samples to include before stimulus onset
    - buffer_after: number of samples to include after stimulus offset
    - baseline_times: (optional) np.ndarray of baseline period times (in seconds)
        of shape [n_trials, 2] indicating start and end times of baseline periods.
    
    Returns:
    - ecog_concat: np.ndarray of shape [n_trials, frequencies, channels, max_epoch_length]
    - valid_trials: list of indices of trials that were successfully segmented
    """

    assert len(start_times) == len(end_times), "Number of start and end times do not match"
    if baseline_times is not None:
        assert len(baseline_times) == len(start_times), "Number of baseline times do not match with start times"

    ecog_data_seg = []
    valid_trials = []

    for si, (sT, eT) in enumerate(zip(start_times, end_times)):
        # Convert time to samples (at final sampling rate)
        start_idx = int(sT * final_hz) - buffer_before
        end_idx = int(eT * final_hz) + buffer_after

        b_start_idx, b_end_idx = None, None
        if baseline_times is not None:
            b_start_T, b_end_T = baseline_times[si]
            b_start_idx = int(b_start_T * final_hz)
            b_end_idx = int(b_end_T * final_hz)

            if b_start_idx < 0 or b_end_idx >= ecog_data.shape[2]:
                L.warning(f"Baseline period for trial {si} is out of data bounds. Skipping.")
                continue

            if b_end_idx > start_idx:
                L.warning(f"Trial {si} overlaps with baseline period. Skipping.")
                continue
    
        # Check bounds
        if start_idx >= 0 and end_idx < ecog_data.shape[2]:
            # Extract epoch and z-score each frequency band and channel independently
            epoch = ecog_data[:, :, start_idx:end_idx]

            if b_start_idx is not None and b_end_idx is not None:
                # Remove baseline period from epoch
                baseline_data = ecog_data[:, :, b_start_idx:b_end_idx]
                baseline_mean = np.mean(baseline_data, axis=-1, keepdims=True)
                epoch = epoch - baseline_mean

            epoch_zscored = zscore(epoch, axis=-1)  # Z-score along time axis
            ecog_data_seg.append(epoch_zscored)
            valid_trials.append(si)

    # Pad segments to maximum length
    max_len = max(d.shape[-1] for d in ecog_data_seg)
    n_freq, n_ch = ecog_data_seg[0].shape[:2]

    # Create concatenated array filled with NaN for padding
    ecog_concat = np.full((len(ecog_data_seg), n_freq, n_ch, max_len), np.nan, dtype=float)
    for i, d in enumerate(ecog_data_seg):
        ecog_concat[i, :, :, :d.shape[-1]] = d

    return ecog_concat, valid_trials


def epoch_ecog_mne(ecog_data: np.ndarray,
                   start_times: np.ndarray,
                   end_times: np.ndarray,
                   frequency_bands: list[str],
                   freq_centers: np.ndarray,
                   final_hz, buffer_before, buffer_after,
                   baseline_times: Optional[np.ndarray] = None,
                   epoch_metadata: Optional[pd.DataFrame] = None,
                   event_id: Optional[dict] = None) -> list[mne.EpochsArray]:
    """
    Segments ECoG data around stimulus events and z-scores each epoch, 
    returning an MNE EpochsArray of band-averaged ERPs.

    Parameters:
    - ecog_data: np.ndarray of shape [frequencies, channels, time]
    - start_times: np.ndarray of stimulus start times (in seconds)
    - end_times: np.ndarray of stimulus end times (in seconds)
    - final_hz: final sampling rate (in Hz)
    - freq_centers: np.ndarray of frequency band centers, one per frequency bin
        (first axis of `ecog_data`)
    - buffer_before: number of samples to include before stimulus onset
    - buffer_after: number of samples to include after stimulus offset
    - baseline_times: (optional) np.ndarray of baseline period times (in seconds)
        of shape [n_trials, 2] indicating start and end times of baseline periods.

    Returns:
    - epochs: list of mne.EpochsArray objects containing the segmented and z-scored data for each frequency band
    """
    frequency_bands = [check_frequency_band(band) for band in frequency_bands]

    ecog_concat, valid_trials = epoch_ecog(
        ecog_data,
        start_times,
        end_times,
        final_hz,
        buffer_before,
        buffer_after,
        baseline_times
    )

    n_trials, n_freq, n_ch, n_times = ecog_concat.shape

    data_per_band = []
    for frequency_band in frequency_bands:
        fb_start_idx, fb_end_idx = get_band_indices(frequency_band, freq_centers)
        band_data_i = np.nanmean(ecog_concat[:, fb_start_idx:fb_end_idx, :, :], axis=1)  # Average over frequency band
        data_per_band.append(band_data_i)

    # Build per-epoch metadata (keep only valid trials and reset index)
    if epoch_metadata is not None:
        assert len(epoch_metadata) == len(start_times), "Epoch metadata length does not match number of trials"
        metadata_for_epochs = epoch_metadata.reset_index(drop=True)
    else:
        metadata_for_epochs = None

    # Create a simple events array for the epochs.
    # Using epoch-index as the "sample" column (0..n_epochs-1), zeros for the middle column,
    # and a single event code (1) for all epochs; adjust if you have real sample indices/event codes.
    events = np.column_stack((
        np.arange(n_trials, dtype=int),
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int),
    ))

    # Create MNE EpochsArray for each frequency band
    epochs_list = []
    for band_idx, frequency_band in enumerate(frequency_bands):
        band_data = data_per_band[band_idx]  # Shape: [n_trials, n_ch, n_times]

        info = mne.create_info(
            ch_names=[f"ECoG_{frequency_band}_{i}" for i in range(n_ch)],
            sfreq=final_hz,
            ch_types=['ecog'] * n_ch
        )

        epochs = mne.EpochsArray(
            band_data,
            info,
            tmin=-buffer_before / final_hz,
            events=events,
            event_id=event_id,          # may be None
            metadata=metadata_for_epochs,
            verbose=False
        )

        epochs_list.append(epochs)

    return epochs_list