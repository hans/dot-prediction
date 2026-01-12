import numpy as np


# Define standard EEG/ECoG frequency bands
hz_dict = {
    "theta": [4, 8],
    "alpha": [8, 12], 
    "beta": [12, 28],
    "gamma": [28, 56],
    "hga": [70, 150],  # High-gamma activity
    "hf": [64, 116]    # High-frequency
}

# Function to get band indices
def get_band_indices(band_name, freq_bands, hz_dict=hz_dict):
    """Get start and end indices for a frequency band."""
    fmin, fmax = hz_dict[band_name.lower()]
    start_idx = np.searchsorted(freq_bands, fmin, side='left')
    end_idx = np.searchsorted(freq_bands, fmax, side='right')
    return start_idx, end_idx


def check_frequency_band(band_name):
    """Check if the frequency band name is valid."""
    if band_name.lower() not in hz_dict:
        raise ValueError(f"Frequency band '{band_name}' is not recognized. Valid bands are: {list(hz_dict.keys())}")
    return band_name.lower()