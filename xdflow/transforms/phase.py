"""
Transforms for Hilbert phase-based features across frequency bands.

This module provides a transform that mirrors the legacy
`ConstSegPhaseFeaturizer` behavior in the new Transform framework.
"""

from typing import Any

import numpy as np
import xarray as xr
from scipy.signal import hilbert

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.spectral import bandpass_filter, get_remove_freq_ranges


class HilbertPhaseTransform(Transform):
    """
    Compute instantaneous phase via Hilbert transform per frequency band and extract features.

    Two modes are supported:
    - 'timepoints': Extract phase at timepoints for each channel and band. By default, uses the 'time'
      coordinate from input data (assumed to be in milliseconds). Can optionally use regularly spaced
      synthetic timepoints.
    - 'relative_average': Compute channel-wise phases relative to the average phase across channels,
      then average over a specified time window (in ms).

    Input dims must include ('trial', 'channel', 'time'). The transform removes the 'time' dimension and adds a 'freq_band' dimension
    and optionally a 'timepoint' dimension, depending on the mode. When using 'timepoints' mode with
    use_time_coord=True (default), the input data must have a 'time' coordinate, expressed in
    milliseconds.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    DEFAULT_FREQ_RANGES: dict[str, tuple[float, float]] = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "beta": (12.0, 30.0),
        "low_gamma": (30.0, 55.0),
        "high_gamma": (65.0, 100.0),
    }
    # Output dims are dynamic based on configuration

    def __init__(
        self,
        fs: int,
        mode: str = "timepoints",
        timepoints_step_ms: int = 100,
        timepoints_start_ms: int | None = None,
        timepoints_end_ms: int | None = None,
        use_time_coord: bool = True,
        num_lf_bands_remove: int = 0,
        num_hf_bands_remove: int = 1,
        lfp_pad_ms_at_ends: int = 0,
        n_jobs: int | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
    ):
        """
        Args:
            fs: Sampling frequency in Hz.
            mode: 'timepoints' or 'relative_average'.
            timepoints_step_ms: Step between timepoints to extract (ms), used when mode='timepoints' and use_time_coord=False.
            timepoints_start_ms: Optional start bound (ms) to begin extracting/averaging within the effective region.
                - If use_time_coord=True (or a 'time' coordinate exists), interpreted in the same absolute units as the
                  'time' coordinate (assumed ms) after padding is excluded.
                - If no 'time' coordinate, interpreted relative to the trimmed region start (0 ms after padding).
            timepoints_end_ms: Optional end bound (ms) to stop extracting/averaging within the effective region.
                Same interpretation rules as timepoints_start_ms.
            use_time_coord: If True (default), use the 'time' coordinate from input data for timepoint selection.
                If False, generate timepoints synthetically based on fs and timepoints_step_ms.
            num_lf_bands_remove: Remove this many low-frequency bands from default ranges.
            num_hf_bands_remove: Remove this many high-frequency bands from default ranges.
            lfp_pad_ms_at_ends: Assumed pre-existing padding present at both start and end of each trial segment.
                We do NOT add any new samples; instead we ignore the first/last lfp_pad_ms_at_ends when extracting
                timepoints or computing relative-average features. If total duration is T ms and this value is P ms,
                the effective region is [P, T - P). When use_time_coord=True, padding is applied to the time coordinate values.
            n_jobs: Number of parallel jobs for trial processing. If None, runs sequentially.
                Ignored when the input is dask-backed, in which case Dask controls parallelism.
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
            transform_sel: Optional transform selection to apply before transforming.
            transform_drop_sel: Optional transform drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        if mode not in ("timepoints", "relative_average"):
            raise ValueError("mode must be either 'timepoints' or 'relative_average'")

        self.fs = fs
        self.mode = mode
        self.timepoints_step_ms = int(timepoints_step_ms)
        self.timepoints_start_ms = None if timepoints_start_ms is None else int(timepoints_start_ms)
        self.timepoints_end_ms = None if timepoints_end_ms is None else int(timepoints_end_ms)
        self.use_time_coord = use_time_coord
        self.num_lf_bands_remove = int(num_lf_bands_remove)
        self.num_hf_bands_remove = int(num_hf_bands_remove)
        self.lfp_pad_ms_at_ends = int(lfp_pad_ms_at_ends)
        self.n_jobs = n_jobs

        if not self.freq_ranges:
            raise ValueError("No frequency ranges available after filtering configuration")

    @property
    def freq_ranges(self) -> dict[str, tuple[float, float]]:
        """Frequency bands to compute, derived on read from the band-removal arguments."""
        freq_ranges = get_remove_freq_ranges(self.num_hf_bands_remove, self.DEFAULT_FREQ_RANGES.copy())
        if self.num_lf_bands_remove > 0:
            freq_ranges = get_remove_freq_ranges(self.num_lf_bands_remove, freq_ranges, remove_high=False)
        return freq_ranges

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        dims = [d for d in input_dims if d != "time"]
        dims.append("freq_band")
        if self.mode == "timepoints":
            dims.append("timepoint")
        # Canonical order
        canonical = ["trial", "channel", "freq_band", "timepoint"]
        return tuple(d for d in canonical if d in dims)

    def _determine_timepoints(self, data: xr.DataArray, n_time: int, pad_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Determine timepoints and indices for timepoints mode."""
        if self.use_time_coord:
            # Use time coordinate from input data, but still respect timepoints_step_ms
            if "time" not in data.coords:
                raise ValueError("use_time_coord=True but input data has no 'time' coordinate")

            time_coord = data.coords["time"]
            # Treat the time coordinate as milliseconds regardless of dtype
            time_ms = time_coord.values.astype(float)

            # Apply padding constraints to determine valid time range
            if self.lfp_pad_ms_at_ends > 0:
                time_min = time_ms.min() + self.lfp_pad_ms_at_ends
                time_max = time_ms.max() - self.lfp_pad_ms_at_ends
            else:
                time_min = time_ms.min()
                time_max = time_ms.max()

            # Apply optional explicit start/end bounds (absolute ms in coord space)
            start_ms = time_min if self.timepoints_start_ms is None else max(time_min, float(self.timepoints_start_ms))
            end_ms = time_max if self.timepoints_end_ms is None else min(time_max, float(self.timepoints_end_ms))
            # Allow empty selection if bounds collapse after padding

            # Generate timepoints starting from start_ms, stepping by timepoints_step_ms
            timepoints_ms = np.arange(start_ms, end_ms + self.timepoints_step_ms, self.timepoints_step_ms)
            # Remove any that exceed time_max
            timepoints_ms = timepoints_ms[timepoints_ms <= end_ms]

            # When using time coord, we don't need array indices - return timepoints as both
            return timepoints_ms, timepoints_ms
        else:
            # Index-based timepoint generation using timepoints_step_ms
            if self.timepoints_step_ms <= 0:
                raise ValueError("timepoints_step_ms must be > 0 for mode='timepoints'")

            # Get the actual time coordinate to determine start/end
            if "time" in data.coords:
                time_coord = data.coords["time"]
                time_ms = time_coord.values.astype(float)

                # Apply padding constraints to determine valid time range
                if self.lfp_pad_ms_at_ends > 0:
                    time_min = time_ms.min() + self.lfp_pad_ms_at_ends
                    time_max = time_ms.max() - self.lfp_pad_ms_at_ends
                else:
                    time_min = time_ms.min()
                    time_max = time_ms.max()

                # Apply optional explicit start/end bounds (absolute ms in coord space)
                start_ms = (
                    time_min if self.timepoints_start_ms is None else max(time_min, float(self.timepoints_start_ms))
                )
                end_ms = time_max if self.timepoints_end_ms is None else min(time_max, float(self.timepoints_end_ms))
                # Allow empty interior if bounds collapse after padding; downstream logic yields NaNs

                # Generate timepoints starting from start_ms, stepping by timepoints_step_ms
                timepoints_ms = np.arange(start_ms, end_ms + self.timepoints_step_ms, self.timepoints_step_ms)
                # Remove any that exceed end_ms
                timepoints_ms = timepoints_ms[timepoints_ms <= end_ms]

                # Find closest indices for these timepoints
                time_idx = []
                for tp in timepoints_ms:
                    closest_idx = np.argmin(np.abs(time_ms - tp))
                    time_idx.append(closest_idx)
                time_idx = np.array(time_idx)

            else:
                # Fallback: synthetic timepoint generation relative to trimmed region (0..T)
                default_start_idx = pad_samples
                default_end_idx = max(pad_samples, n_time - pad_samples)

                # Optional explicit start/end (relative to trimmed region in ms)
                if self.timepoints_start_ms is not None:
                    default_start_idx = pad_samples + int(round(self.timepoints_start_ms * self.fs / 1000))
                if self.timepoints_end_ms is not None:
                    default_end_idx = pad_samples + int(round(self.timepoints_end_ms * self.fs / 1000))

                # Clamp to valid bounds
                start_idx = max(pad_samples, min(default_start_idx, n_time))
                end_idx = max(start_idx, min(default_end_idx, n_time))

                # Convert timepoints_step_ms to sample steps
                step_samples = int(round(self.timepoints_step_ms * self.fs / 1000))
                if step_samples <= 0:
                    step_samples = 1

                # Generate sample indices by stepping
                time_idx = np.arange(start_idx, end_idx, step_samples)

                # Convert back to time values in ms for the output coordinates
                timepoints_ms = ((time_idx - pad_samples) * 1000 / self.fs).astype(float)

            return time_idx, timepoints_ms

    def _parallelize_across_trials(
        self,
        data: xr.DataArray,
        items: list[tuple[str, tuple[float, float]]],
        time_idx: np.ndarray,
        timepoints_ms: np.ndarray,
        compute_band_features,
    ) -> list[tuple[str, xr.DataArray]]:
        """Parallelize computation across trials."""
        from collections import defaultdict

        from joblib import Parallel, delayed

        def compute_trial_all_bands(trial_idx: int) -> tuple[int, dict[str, xr.DataArray]]:
            trial_data = data.isel(trial=trial_idx)
            trial_results = {}

            for band_name, (f_low, f_high) in items:
                if f_high >= self.fs / 2:
                    raise ValueError(
                        f"High frequency {f_high} Hz for band '{band_name}' exceeds Nyquist frequency {self.fs / 2} Hz"
                    )

                if self.mode == "timepoints" and self.use_time_coord:
                    # For coordinate-based selection, compute full phases then select
                    da_band = xr.apply_ufunc(
                        lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                        trial_data,
                        input_core_dims=[["channel", "time"]],
                        output_core_dims=[["channel", "time"]],
                        dask="forbidden",  # Single trial, no need for dask
                        output_dtypes=[data.dtype],
                        keep_attrs=True,
                    )
                    # Select the desired timepoints using coordinate values
                    da_band = da_band.sel(time=timepoints_ms, method="nearest")
                    # Rename time dim to timepoint and assign proper coordinates
                    da_band = da_band.rename({"time": "timepoint"})
                    da_band = da_band.assign_coords(timepoint=("timepoint", timepoints_ms))
                elif self.mode == "timepoints":
                    # For index-based selection (legacy mode)
                    output_core_dims = [["channel", "timepoint"]]
                    da_band = xr.apply_ufunc(
                        lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                        trial_data,
                        input_core_dims=[["channel", "time"]],
                        output_core_dims=output_core_dims,
                        dask="forbidden",  # Single trial, no need for dask
                        output_dtypes=[data.dtype],
                        output_sizes={"timepoint": len(time_idx)},
                        keep_attrs=True,
                    )
                    da_band = da_band.assign_coords(timepoint=("timepoint", timepoints_ms))
                else:
                    # relative_average mode
                    output_core_dims = [["channel"]]
                    da_band = xr.apply_ufunc(
                        lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                        trial_data,
                        input_core_dims=[["channel", "time"]],
                        output_core_dims=output_core_dims,
                        dask="forbidden",  # Single trial, no need for dask
                        output_dtypes=[data.dtype],
                        keep_attrs=True,
                    )

                trial_results[band_name] = da_band

            return trial_idx, trial_results

        # Process trials in parallel
        trial_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(compute_trial_all_bands)(trial_idx) for trial_idx in range(data.sizes["trial"])
        )

        # Reorganize results by band
        band_trial_results = defaultdict(list)
        for trial_idx, bands_dict in trial_results:
            for band_name, band_result in bands_dict.items():
                band_trial_results[band_name].append((trial_idx, band_result))

        # Concatenate trials for each band
        results = []
        for band_name, _ in items:
            sorted_trial_results = sorted(band_trial_results[band_name], key=lambda x: x[0])
            trial_arrays = [result for _, result in sorted_trial_results]
            band_result = xr.concat(trial_arrays, dim="trial")
            results.append((band_name, band_result))
        return results

    def _transform(self, container: DataContainer, **kwargs: Any) -> DataContainer:
        data = container.data
        if not np.issubdtype(data.dtype, np.floating):
            # Phases are floats; outputs are cast to the input dtype below, which would truncate integers
            data = data.astype(np.float64)

        # Validate input
        for dim in ("trial", "channel", "time"):
            if dim not in data.dims:
                raise ValueError(f"Input data must have '{dim}' dimension")
        if data.sizes["time"] == 0:
            raise ValueError("Cannot compute phases on data with zero time points")

        n_time = int(data.sizes["time"])
        pad_samples = int(round(self.lfp_pad_ms_at_ends * self.fs / 1000))

        # Determine timepoints (ms) and indices when needed
        if self.mode == "timepoints":
            time_idx, timepoints_ms = self._determine_timepoints(data, n_time, pad_samples)
        else:
            # Relative average over a bounded window within the trimmed region
            if self.use_time_coord and ("time" in data.coords):
                time_coord = data.coords["time"]
                time_ms = time_coord.values.astype(float)

                # Effective bounds after padding
                if self.lfp_pad_ms_at_ends > 0:
                    time_min = time_ms.min() + self.lfp_pad_ms_at_ends
                    time_max = time_ms.max() - self.lfp_pad_ms_at_ends
                else:
                    time_min = time_ms.min()
                    time_max = time_ms.max()

                start_ms = (
                    time_min if self.timepoints_start_ms is None else max(time_min, float(self.timepoints_start_ms))
                )
                end_ms = time_max if self.timepoints_end_ms is None else min(time_max, float(self.timepoints_end_ms))
                # Allow empty selection if bounds collapse after padding
                # Allow empty interior if bounds collapse after padding; downstream logic yields NaNs

                # Map ms bounds to index bounds (end exclusive)
                start_idx = int(np.argmin(np.abs(time_ms - start_ms)))
                end_idx_incl = int(np.argmin(np.abs(time_ms - end_ms)))
                end_idx = min(n_time, end_idx_incl + 1)
                # Ensure within trimmed region indices
                start_idx = max(start_idx, pad_samples)
                end_idx = min(end_idx, n_time - pad_samples)
                end_idx = max(end_idx, start_idx)  # non-negative window length
            else:
                # Index-based: interpret start/end relative to trimmed region 0..T
                default_start_idx = pad_samples
                default_end_idx = max(pad_samples, n_time - pad_samples)
                if self.timepoints_start_ms is not None:
                    default_start_idx = pad_samples + int(round(self.timepoints_start_ms * self.fs / 1000))
                if self.timepoints_end_ms is not None:
                    default_end_idx = pad_samples + int(round(self.timepoints_end_ms * self.fs / 1000))

                start_idx = max(pad_samples, min(default_start_idx, n_time))
                end_idx = max(start_idx, min(default_end_idx, n_time))

        # Per-band computation function applied per trial
        def compute_band_features(trial_data: np.ndarray, f_low: float, f_high: float):
            # trial_data: (channel, time)
            if trial_data.size == 0:
                raise ValueError("Empty trial data")

            # Bandpass filter then Hilbert on the original signal
            filtered = bandpass_filter(trial_data, f_low, f_high, fs=self.fs, axis=-1)
            analytic = hilbert(filtered, axis=-1)
            phases = np.angle(analytic)

            if self.mode == "timepoints":
                if self.use_time_coord:
                    # For coordinate-based selection, we need to select at the xarray level, not here
                    # This function will be called via xarray.apply_ufunc which handles the selection
                    return phases
                else:
                    # For index-based selection, use integer indices
                    out = phases[:, time_idx]
                    return out.astype(trial_data.dtype, copy=False)
            else:
                # Relative to average phase across channels, then average over time window
                window = phases[:, start_idx:end_idx]
                if window.shape[1] == 0:
                    return np.full((phases.shape[0],), np.nan, dtype=phases.dtype)
                complex_phases = np.exp(1j * window)
                avg_complex = np.mean(complex_phases, axis=0)  # (time,)
                avg_phase = np.angle(avg_complex)  # (time,)
                relative = np.exp(1j * (window - avg_phase[np.newaxis, :]))
                out = np.angle(np.mean(relative, axis=1))  # circular mean over time, (channel,)
                return out.astype(trial_data.dtype, copy=False)

        # Compute features for each band, concatenating along new freq_band dim
        def compute_one_band(band_name: str, f_low: float, f_high: float) -> tuple[str, xr.DataArray]:
            if f_high >= self.fs / 2:
                raise ValueError(
                    f"High frequency {f_high} Hz for band '{band_name}' exceeds Nyquist frequency {self.fs / 2} Hz"
                )

            if self.mode == "timepoints" and self.use_time_coord:
                # For coordinate-based selection, compute full phases then select
                da_band = xr.apply_ufunc(
                    lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                    data,
                    input_core_dims=[["channel", "time"]],
                    output_core_dims=[["channel", "time"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[data.dtype],
                    keep_attrs=True,
                )
                # Select the desired timepoints using coordinate values
                da_band = da_band.sel(time=timepoints_ms, method="nearest")
                # Rename time dim to timepoint and assign proper coordinates
                da_band = da_band.rename({"time": "timepoint"})
                da_band = da_band.assign_coords(timepoint=("timepoint", timepoints_ms))
            elif self.mode == "timepoints":
                # For index-based selection (legacy mode)
                output_core_dims = [["channel", "timepoint"]]
                da_band = xr.apply_ufunc(
                    lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                    data,
                    input_core_dims=[["channel", "time"]],
                    output_core_dims=output_core_dims,
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[data.dtype],
                    output_sizes={"timepoint": len(time_idx)},
                    keep_attrs=True,
                )
                da_band = da_band.assign_coords(timepoint=("timepoint", timepoints_ms))
            else:
                # relative_average mode
                output_core_dims = [["channel"]]
                da_band = xr.apply_ufunc(
                    lambda trial_arr, f_low=f_low, f_high=f_high: compute_band_features(trial_arr, f_low, f_high),
                    data,
                    input_core_dims=[["channel", "time"]],
                    output_core_dims=output_core_dims,
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[data.dtype],
                    keep_attrs=True,
                )
            return band_name, da_band

        # Detect whether input is dask-backed
        try:
            import dask.array as da  # type: ignore

            is_dask = isinstance(data.data, da.Array)
        except ImportError:
            is_dask = False

        items: list[tuple[str, tuple[float, float]]] = list(self.freq_ranges.items())
        if (self.n_jobs is not None) and (self.n_jobs > 1) and (not is_dask):
            results = self._parallelize_across_trials(data, items, time_idx, timepoints_ms, compute_band_features)
        else:
            # Sequential processing - compute one band at a time
            results = [compute_one_band(band_name, f_low, f_high) for band_name, (f_low, f_high) in items]

        band_names, band_results = zip(*results)

        # Concatenate bands
        feat = xr.concat(list(band_results), dim="freq_band")
        feat = feat.assign_coords(freq_band=("freq_band", list(band_names)))

        # Preserve key trial/channel coords
        original_coords = {
            "trial": data.coords.get("trial", None),
            "channel": data.coords.get("channel", None),
        }
        # Preserve any trial-aligned coordinates to avoid dropping metadata.
        for coord_name, coord in data.coords.items():
            if coord_name in original_coords:
                continue
            if "trial" in coord.dims:
                original_coords[coord_name] = ("trial", coord.values)

        feat = feat.assign_coords({k: v for k, v in original_coords.items() if v is not None})

        # Reorder to canonical expected dims
        expected_dims = self.get_expected_output_dims(tuple(data.dims))
        feat = feat.transpose(*expected_dims)

        return DataContainer(feat)
