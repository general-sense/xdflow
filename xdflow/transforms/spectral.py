import logging
from typing import Any

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.spectral import bandpass_filter, check_sampling_rate

# Suppress spectral_connectivity logger messages for a cleaner output
spectral_logger = logging.getLogger("spectral_connectivity")
spectral_logger.setLevel(logging.CRITICAL + 1)
spectral_logger.addHandler(logging.NullHandler())
spectral_logger.propagate = False


def _spectral_connectivity() -> tuple[Any, Any, Any]:
    """Import spectral_connectivity on use: only MultiTaperTransform needs the 'spectral' extra."""
    try:
        from spectral_connectivity import Connectivity, Multitaper
        from spectral_connectivity.transforms import prepare_time_series  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MultiTaperTransform requires the 'spectral' extra: pip install 'xdflow[spectral]'"
        ) from exc
    return Connectivity, Multitaper, prepare_time_series


class MultiTaperTransform(Transform):
    """
    Computes a time-frequency representation using the multitaper method.

    This transform wraps the `spectral_connectivity.Multitaper` method, applying
    it across trials in a DataContainer. It offers extensive flexibility,
    allowing the user to retain all dimensions (tapers, time windows, frequencies)
    or selectively average over them.

    This class is the updated-pipeline equivalent of `ConstSegMTFeaturizer` and `compute_tf_representations`.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    # Output dimensions are dynamic based on the transform's parameters.

    def __init__(
        self,
        fs: int,
        time_halfbandwidth_product: float = 3.5,
        num_time_windows: int = 4,  # Changed from time_window_duration
        window_step_frac: float = 0.5,
        n_fft: int = 512,
        detrend: str = "constant",
        log_transform: bool = True,
        avg_over_tapers: bool = True,
        avg_over_time_windows: bool = False,
        avg_within_freq_bands: bool = False,
        freq_ranges: dict[str, tuple[float, float]] | None = None,
        n_jobs: int = 1,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        """
        Initializes the MultiTaperTransform.

        Args:
            fs: Sampling frequency in Hz.
            time_halfbandwidth_product: Time-bandwidth product for multitaper analysis.
            num_time_windows: Number of time windows to split the data into.
            window_step_frac: Fraction of the time window to step forward for each
                        subsequent window. E.g., 0.5 means 50% overlap step. Must be > 0 and <= 1.
            n_fft: Number of points for the Fast Fourier Transform.
            detrend: Type of detrending to apply before analysis (e.g., "constant", "linear").
            log_transform: If True, applies a log10 transform to the power.
            avg_over_tapers: If True, averages the power across tapers.
            avg_over_time_windows: If True, averages the power across time windows.
            avg_within_freq_bands: If True, averages power within each specified `freq_ranges` band.
            freq_ranges: A dictionary defining frequency bands for averaging, e.g.,
                         {"gamma": (40, 80)}. Required if `avg_within_freq_bands` is True.
            n_jobs: Number of parallel jobs to run. If > 1, trials will be processed in parallel.
            sel: Optional label selection applied before transforming.
            drop_sel: Optional labels to drop before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        _spectral_connectivity()  # fail at construction, not mid-pipeline, if the extra is missing
        if not 0 < window_step_frac <= 1:
            raise ValueError("`window_step_frac` must be in the range (0, 1].")

        if num_time_windows < 1:
            raise ValueError("`num_time_windows` must be a positive integer.")

        self.fs = fs
        self.time_halfbandwidth_product = time_halfbandwidth_product
        n_tapers = time_halfbandwidth_product * 2 - 1
        if n_tapers % 1 != 0:
            raise ValueError("`time_halfbandwidth_product` must be a multiple of 0.5.")
        self.n_tapers = int(n_tapers)
        self.num_time_windows = num_time_windows  # Changed parameter name
        self.window_step_frac = window_step_frac
        self.n_fft = n_fft
        self.detrend = detrend
        self.log_transform = log_transform
        self.avg_over_tapers = avg_over_tapers
        self.avg_over_time_windows = avg_over_time_windows
        self.avg_within_freq_bands = avg_within_freq_bands
        self.freq_ranges = freq_ranges
        self.n_jobs = n_jobs

        if self.avg_within_freq_bands and not self.freq_ranges:
            raise ValueError("`freq_ranges` must be provided when `avg_within_freq_bands` is True.")

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions based on the transform's configuration.
        """
        # Start with the full potential set of dimensions
        dims = set(input_dims)
        dims.remove("time")
        dims.update(["taper", "time_window", "frequency"])

        # Remove dimensions that are being averaged over
        if self.avg_over_tapers:
            dims.discard("taper")
        if self.avg_over_time_windows:
            dims.discard("time_window")
        if self.avg_within_freq_bands:
            dims.discard("frequency")
            dims.add("freq_band")

        # Define a canonical order for the output dimensions
        canonical_order = ["trial", "channel", "freq_band", "frequency", "time_window", "taper"]
        return tuple(d for d in canonical_order if d in dims)

    def _process_single_trial(
        self, trial_lfp: np.ndarray, window_duration: float, step_size_seconds: float, frequencies: np.ndarray
    ) -> np.ndarray:
        """
        Processes a single trial to compute time-frequency power.
        This method is designed to be called in parallel.

        Args:
            trial_lfp: The LFP data for a single trial (time, channel).
            window_duration: The calculated duration of each time window.
            step_size_seconds: The step size between windows in seconds.
            frequencies: The array of frequency values.

        Returns:
            The processed power data for one trial as a numpy array.
        """
        Connectivity, Multitaper, prepare_time_series = _spectral_connectivity()
        # trial_lfp comes in as (channel, time); transpose to (time, channel)
        time_series_2d = trial_lfp.T
        # spectral_connectivity>=2 requires explicit 3D input with clarified axis.
        time_series = prepare_time_series(time_series_2d, axis="signals")

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=self.fs,
            time_halfbandwidth_product=self.time_halfbandwidth_product,
            time_window_duration=window_duration,
            time_window_step=step_size_seconds,
            n_fft_samples=self.n_fft,
            detrend_type=self.detrend,
        )

        # Compute power for this trial
        c = Connectivity.from_multitaper(m)
        fourier_coeffs = c.fourier_coefficients.squeeze(axis=1)
        fourier_coeffs = fourier_coeffs[:, :, : self.n_fft // 2, :]  # Cut out negative frequencies
        power_np = (fourier_coeffs * fourier_coeffs.conjugate()).real
        # Shape is (channel, taper, freq, time_window)
        power_transposed = np.transpose(power_np, (3, 1, 2, 0))  # channel, taper, freq, time_window
        current_dims = ["channel", "taper", "frequency", "time_window"]

        # Apply log transform if needed
        if self.log_transform:
            power_transposed = np.log10(power_transposed + np.finfo(np.float32).eps, dtype=np.float32)

        # Apply averaging operations in order, keeping track of current dimensions
        processed_power = power_transposed

        # Average over tapers if needed
        if self.avg_over_tapers:
            taper_axis = current_dims.index("taper")
            processed_power = processed_power.mean(axis=taper_axis, dtype=np.float32)
            current_dims.remove("taper")

        # Average over time windows if needed
        if self.avg_over_time_windows:
            time_axis = current_dims.index("time_window")
            processed_power = processed_power.mean(axis=time_axis, dtype=np.float32)
            current_dims.remove("time_window")

        # Average frequency bins within each requested band.
        if self.avg_within_freq_bands:
            if self.freq_ranges is None:
                raise ValueError("`freq_ranges` must be provided when `avg_within_freq_bands` is True.")
            freq_axis = current_dims.index("frequency")
            band_results = []
            for _band_name, (f_low, f_high) in self.freq_ranges.items():
                # Find frequency indices for this band
                freq_mask = (frequencies >= f_low) & (frequencies <= f_high)
                freq_indices = np.where(freq_mask)[0]

                if freq_indices.size == 0:
                    # Skip bands that have no frequency bins
                    continue

                # Average within this frequency band.
                band_power = np.take(processed_power, freq_indices, axis=freq_axis).mean(
                    axis=freq_axis, dtype=np.float32
                )
                band_results.append(band_power)

            if not band_results:
                raise ValueError("No valid frequency bands found in freq_ranges for given n_fft and fs.")

            processed_power = np.stack(band_results, axis=freq_axis)
            # Replace the 'frequency' dim with 'freq_band' in our dim tracker
            current_dims[freq_axis] = "freq_band"

        # Reorder axes to a canonical order to match final_dims used in _transform
        # Current dims are tracked in `current_dims`. We want:
        #   ("channel", "freq_band"|"frequency", ["time_window" if present], ["taper" if present])
        canonical_dims = ["channel"]
        canonical_dims.append("freq_band" if self.avg_within_freq_bands else "frequency")
        if not self.avg_over_time_windows:
            canonical_dims.append("time_window")
        if not self.avg_over_tapers:
            canonical_dims.append("taper")

        # Compute permutation from current_dims to canonical_dims
        axes_order = [current_dims.index(dim) for dim in canonical_dims]
        processed_power = np.transpose(processed_power, axes_order)

        return processed_power

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the multitaper time-frequency transformation, with optional parallelization.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)
        """
        data = container.data
        check_sampling_rate(data, self.fs, type(self).__name__)

        # --- 1. SETUP (This is done once, outside the parallel loop) ---
        total_time_sec = data["time"].shape[0] / self.fs
        num_full_windows = 1 + (self.num_time_windows - 1) * self.window_step_frac
        window_duration = total_time_sec / num_full_windows
        step_size_seconds = window_duration * self.window_step_frac

        frequencies = np.fft.fftfreq(self.n_fft, 1 / self.fs)[: self.n_fft // 2]

        time_window_centers = []
        for i in range(self.num_time_windows):
            center_time = (i * step_size_seconds) + (window_duration / 2)
            time_window_centers.append(center_time)
        time_window_centers = np.array(time_window_centers)

        taper_indices = np.arange(self.n_tapers)

        # Calculate final array shape and dimensions based on averaging options
        final_shape = [data.sizes["trial"], data.sizes["channel"]]
        final_dims = ["trial", "channel"]
        coords = {}

        # Determine final shape and dimensions based on averaging
        if self.avg_within_freq_bands:
            if self.freq_ranges is None:
                raise ValueError("`freq_ranges` must be provided when `avg_within_freq_bands` is True.")
            # Filter bands that actually have frequency support
            band_names: list[str] = []
            for band_name, (f_low, f_high) in self.freq_ranges.items():
                freq_mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.any(freq_mask):
                    band_names.append(band_name)

            if not band_names:
                raise ValueError("No valid frequency bands found in freq_ranges for given n_fft and fs.")

            final_shape.append(len(band_names))
            final_dims.append("freq_band")
            coords["freq_band"] = band_names
        else:
            final_shape.append(len(frequencies))
            final_dims.append("frequency")
            coords["frequency"] = frequencies

        if not self.avg_over_time_windows:
            final_shape.append(len(time_window_centers))
            final_dims.append("time_window")
            coords["time_window"] = time_window_centers

        if not self.avg_over_tapers:
            # Placeholder; will be overwritten with actual size after computation
            final_shape.append(self.n_tapers)
            final_dims.append("taper")
            coords["taper"] = taper_indices

        # --- 2. EXECUTION (Parallel or Serial) ---
        trial_values = data["trial"].values

        if self.n_jobs > 1:
            # Parallel execution: use lists + np.stack (required for joblib)
            # print(f"Running multitaper transform in parallel on {self.n_jobs} cores...")
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_single_trial)(
                    trial_lfp=data.sel(trial=trial_val).values,
                    window_duration=window_duration,
                    step_size_seconds=step_size_seconds,
                    frequencies=frequencies,
                )
                for trial_val in trial_values
            )
            # Stack results along the first axis to create the 'trial' dimension
            tf_data_np = np.stack(results_list, axis=0).astype(np.float32)
        else:
            # Serial execution: use preallocation (memory efficient, original approach)
            # print("Running multitaper transform serially...")
            tf_data_np = np.zeros(final_shape, dtype=np.float32)

            for trial_idx, trial_val in enumerate(trial_values):
                processed_power = self._process_single_trial(
                    trial_lfp=data.sel(trial=trial_val).values,
                    window_duration=window_duration,
                    step_size_seconds=step_size_seconds,
                    frequencies=frequencies,
                )
                tf_data_np[trial_idx] = processed_power

        # Set up final coordinates
        original_coords = {
            "trial": data.coords["trial"],
            "channel": data.coords["channel"],
        }

        # keep all coords that are trial-based, important for meta data
        for coord_name, coord_data in data.coords.items():
            if (coord_name != "trial") and ("trial" in coord_data.dims):
                original_coords[coord_name] = coord_data

        # Sync any coordinate lengths that depend on computed array sizes (e.g., 'taper')
        final_sizes = dict(zip(final_dims, tf_data_np.shape))
        if "taper" in final_sizes and ("taper" not in coords or len(coords.get("taper", [])) != final_sizes["taper"]):
            coords["taper"] = np.arange(final_sizes["taper"])  # type: ignore[index]

        tf_data = xr.DataArray(tf_data_np, dims=final_dims, coords={**original_coords, **coords}, attrs=data.attrs)

        new_container = DataContainer(tf_data)
        return new_container


class BandpassFilterTransform(Transform):  # TODO: rename or combine with BandpassFilterBandTransform
    """
    Apply bandpass filtering, replacing the original data with the filtered data.
    The dimensions stay the same (useful for reference-sensor filtering).

    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")
    _supports_transform_sel: bool = True

    def __init__(
        self,
        lowcut: float,
        highcut: float,
        fs: float,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut

        # Validate frequency bounds
        if self.highcut >= self.fs / 2:
            raise ValueError(f"High frequency {self.highcut} Hz exceeds Nyquist frequency {self.fs / 2} Hz")

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return self.output_dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Apply bandpass filtering across frequency bands.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)
        """
        data = container.data

        # Validate required dimensions
        if "time" not in data.dims:
            raise ValueError("Input data must have 'time' dimension for filtering")

        if data.sizes["time"] == 0:
            raise ValueError("Cannot filter data with zero time points")
        check_sampling_rate(data, self.fs, type(self).__name__)

        # Apply bandpass filter
        filtered_data = xr.apply_ufunc(
            lambda x: bandpass_filter(x, self.lowcut, self.highcut, fs=self.fs),
            data,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=[data.dtype],
            keep_attrs=True,
        )

        return DataContainer(filtered_data)
