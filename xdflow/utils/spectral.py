from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def bandpass_filter(data, lowcut, highcut, order=4, fs=500, causal=False, axis=-1):
    """

    Args:
      data:
      lowcut:
      highcut:
      order: (Default value = 4)
      fs: Default value = 500)
      causal: Default value = False)
      axis: Default value = -1)

    Returns:

    """
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="ba")
    if causal:
        y = lfilter(b, a, data, axis=axis)
    else:
        y = filtfilt(b, a, data, axis=axis)
    return y


def get_remove_freq_ranges(num_bands_remove, freqs, remove_high=True):
    """Removes a specified number of frequency bands from the frequency ranges dictionary, starting with high or low frequency.

    Args:
      num_bands_remove: Number of frequency bands to remove
      freqs: Dictionary of frequency ranges (e.g., {'theta': (4, 8), 'beta': (13, 30)})
      remove_high: Boolean indicating whether to remove high (Default value = True)

    Returns:
      Modified frequency ranges dictionary with the specified number of frequency bands removed.

    """
    if freqs is None:
        raise ValueError("freqs parameter is required - provide a dictionary of frequency ranges")
    freq_bands_can_remove = list(freqs.keys())

    # Reverse the list if removing high-frequency bands
    if remove_high:
        freq_bands_can_remove = freq_bands_can_remove[::-1]

    for i in range(num_bands_remove):
        if i < len(freq_bands_can_remove):
            freqs.pop(freq_bands_can_remove[i], None)

    return freqs


def get_freq_band_indices(frequencies, low, high):
    """Returns the indices of the beginning and end of a frequency band.

    Args:
      frequencies: Sorted array of frequencies
      low: Lower bound of the frequency band
      high: Upper bound of the frequency band

    Returns:
      List with start and end indices of the frequency band.

    """
    low_index = np.searchsorted(frequencies, low, side="left")
    high_index = np.searchsorted(frequencies, high, side="right")
    return [low_index, high_index]


def check_sampling_rate(data: Any, fs: float, name: str) -> None:
    """Raise when no evidence of the data's sampling rate supports ``fs``. Evidence is ``attrs["sfreq"]`` (set by
    ``from_mne_epochs``) and the spacing of a time coordinate in seconds or milliseconds (without units: seconds, ms
    or µs, unless the spacing is a whole number, as for sample indices). Either can be stale (a step that decimates keeps attrs; time converted to ms may keep
    units of seconds), so ``fs`` passes if it matches any of them. Filtering at the wrong rate shifts every frequency
    without any other sign."""
    rates = []
    try:
        if data.attrs.get("sfreq") is not None:
            rates.append(float(data.attrs["sfreq"]))
    except (TypeError, ValueError):  # not a number ("250 Hz")
        pass
    time = data.coords.get("time")
    if time is not None and time.size > 1 and np.issubdtype(time.dtype, np.number):
        spacing = float(np.median(np.diff(np.asarray(time.values, dtype=float))))
        units = time.attrs.get("units")
        if spacing > 0 and units in ("s", "ms"):
            rates.append({"s": 1.0, "ms": 1000.0}[units] / spacing)
        elif spacing > 0 and units is None and (rates or not float(spacing).is_integer()):
            # Without units: seconds, ms or µs. An integer spacing alone is likely sample indices, not a time.
            rates.extend(per_second / spacing for per_second in (1.0, 1e3, 1e6))
    if rates and not any(np.isclose(rate, float(fs), rtol=1e-3) for rate in rates):
        evidence = ", ".join(f"{rate:g} Hz" for rate in rates)
        raise ValueError(
            f"{name} has fs={fs} Hz, but the data is sampled at {evidence} (attrs['sfreq'], time coordinate)."
        )
