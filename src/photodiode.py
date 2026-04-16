import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def detect_events_env_direction(
    x,
    fs,
    smooth_env_s=0.02,
    min_sep_s=0.15,
    prom_z=3.5,
    smooth_x_s=0.01,
    direction_window_s=0.02,
    return_derivative=False,
):
    """Detect rising/falling edges on a noisy photodiode trace.

    Peaks are picked on the Hilbert envelope of the detrended signal (thresholded
    by robust z-score); direction is classified by the sign of the slope of the
    smoothed signal across `direction_window_s` around each peak.

    Returns a dict with 'rise' and 'fall' keys; each maps to arrays of peak
    times (seconds), FWHM widths (seconds), and peak indices.
    """
    x = np.asarray(x)
    x_d = signal.detrend(x, type="linear")

    env = np.abs(signal.hilbert(x_d))
    sigma_env = max(1, int(round(smooth_env_s * fs)))
    env_s = gaussian_filter1d(env, sigma=sigma_env)

    med = np.median(env_s)
    mad = np.median(np.abs(env_s - med)) + 1e-12
    z = (env_s - med) / (1.4826 * mad)

    distance = max(1, int(round(min_sep_s * fs)))
    idx_peaks, _ = signal.find_peaks(z, prominence=prom_z, distance=distance)

    if idx_peaks.size > 0:
        widths, _, _, _ = signal.peak_widths(z, idx_peaks, rel_height=0.5)
        t_half = widths / fs
    else:
        t_half = np.array([])

    sigma_x = max(1, int(round(smooth_x_s * fs)))
    x_s = gaussian_filter1d(x_d, sigma=sigma_x)
    dx = np.gradient(x_s) * fs

    w = max(1, int(round(direction_window_s * fs)))

    rise_t, rise_idx, rise_t_half = [], [], []
    fall_t, fall_idx, fall_t_half = [], [], []
    for k, i0 in enumerate(idx_peaks):
        i0 = int(i0)
        signed_change = x_s[min(len(x_s) - 1, i0 + w)] - x_s[max(0, i0 - w)]
        sign = 1 if signed_change >= 0 else -1
        th = float(t_half[k]) if k < len(t_half) else float("nan")
        if sign > 0:
            rise_idx.append(i0)
            rise_t.append(i0 / fs)
            rise_t_half.append(th)
        else:
            fall_idx.append(i0)
            fall_t.append(i0 / fs)
            fall_t_half.append(th)

    out = {
        "rise": {
            "t_peaks": np.asarray(rise_t, dtype=float),
            "t_half": np.asarray(rise_t_half, dtype=float),
            "idx_peaks": np.asarray(rise_idx, dtype=int),
        },
        "fall": {
            "t_peaks": np.asarray(fall_t, dtype=float),
            "t_half": np.asarray(fall_t_half, dtype=float),
            "idx_peaks": np.asarray(fall_idx, dtype=int),
        },
    }
    return (out, x_s, dx) if return_derivative else out
