import importlib.util

import numpy as np
import pytest
import xarray as xr

from xdflow.core import DataContainer
from xdflow.transforms.spectral import MultiTaperTransform

# Only MultiTaperTransform needs the optional `spectral` extra; the module itself imports without it.
requires_spectral = pytest.mark.skipif(
    importlib.util.find_spec("spectral_connectivity") is None, reason="spectral-connectivity is not installed"
)


FREQ_RANGES = {
    "theta": (4.0, 8.0),
    "beta": (12.0, 30.0),
}


@requires_spectral
def test_multitaper_avg_within_freq_bands_expected_dims():
    transform = MultiTaperTransform(
        fs=500,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges=FREQ_RANGES,
    )

    assert transform.avg_within_freq_bands is True
    assert transform.get_expected_output_dims(("trial", "channel", "time")) == ("trial", "channel", "freq_band")


@requires_spectral
def test_multitaper_clone_preserves_avg_within_freq_bands():
    transform = MultiTaperTransform(
        fs=500,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges=FREQ_RANGES,
    )

    cloned = transform.clone()

    assert cloned.avg_within_freq_bands is True
    assert cloned.freq_ranges == FREQ_RANGES


@requires_spectral
def test_multitaper_preserves_existing_history():
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.normal(size=(2, 2, 200)),
        dims=("trial", "channel", "time"),
        coords={
            "trial": [0, 1],
            "channel": ["ch0", "ch1"],
            "time": np.arange(200) / 100,
            "stimulus": ("trial", ["A", "B"]),
        },
        attrs={
            "data_history": [{"class": "PreviousTransform", "params": {}}],
            "sampling_frequency_hz": 100,
        },
    )
    container = DataContainer(data)
    transform = MultiTaperTransform(
        fs=100,
        n_fft=128,
        avg_over_time_windows=True,
        avg_within_freq_bands=True,
        freq_ranges={"theta": (4.0, 8.0), "beta": (12.0, 30.0)},
    )

    result = transform.transform(container)

    assert result.data.attrs["sampling_frequency_hz"] == 100
    assert [entry["class"] for entry in result.data.attrs["data_history"]] == [
        "PreviousTransform",
        "MultiTaperTransform",
    ]
    assert [entry["class"] for entry in container.data.attrs["data_history"]] == ["PreviousTransform"]


def test_bandpass_filter_needs_no_spectral_extra_and_explicit_fs(monkeypatch):
    import importlib
    import sys

    import numpy as np
    import pytest
    import xarray as xr

    from xdflow.core.data_container import DataContainer

    monkeypatch.setitem(sys.modules, "spectral_connectivity", None)  # as if the extra were not installed
    monkeypatch.delitem(sys.modules, "xdflow.transforms.spectral", raising=False)
    spectral = importlib.import_module("xdflow.transforms.spectral")

    with pytest.raises(TypeError):
        spectral.BandpassFilterTransform(8.0, 30.0)  # fs is required: a default would silently mis-filter
    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(2, 1, 400)),
        dims=("trial", "channel", "time"),
        coords={"trial": [0, 1], "channel": [0], "time": np.arange(400) / 160},
    )
    filtered = spectral.BandpassFilterTransform(8.0, 30.0, fs=160).transform(DataContainer(data))
    assert filtered.data.shape == data.shape

    with pytest.raises(ModuleNotFoundError, match="spectral"):
        spectral.MultiTaperTransform(fs=160)


def test_bandpass_refuses_a_sampling_rate_that_contradicts_the_data():
    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.spectral import BandpassFilterTransform

    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(4, 2, 500)),
        dims=("trial", "channel", "time"),
        coords={"time": np.arange(500) / 250.0},
        attrs={"sfreq": 250.0},
    )
    with pytest.warns(UserWarning, match="250"):
        BandpassFilterTransform(8, 30, fs=160).transform(DataContainer(data))
    BandpassFilterTransform(8, 30, fs=250).transform(DataContainer(data))


def test_bandpass_checks_the_rate_only_when_data_and_metadata_agree():
    """Stale metadata (after decimation, or time converted to ms) must not refuse a correct fs."""
    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.spectral import BandpassFilterTransform

    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(4, 2, 800)),
        dims=("trial", "channel", "time"),
        coords={"time": ("time", np.arange(800) / 160.0, {"units": "s"})},
        attrs={"sfreq": 160.0},
    )
    BandpassFilterTransform(8, 30, fs=80).transform(DataContainer(data.isel(time=slice(None, None, 2))))
    in_ms = data.assign_coords(time=data.time * 1000)  # keeps units="s"
    BandpassFilterTransform(8, 30, fs=160).transform(DataContainer(in_ms))
    with pytest.warns(UserWarning, match="160"):
        BandpassFilterTransform(8, 30, fs=250).transform(DataContainer(data))


def test_bandpass_refuses_an_fs_that_no_evidence_supports():
    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.spectral import BandpassFilterTransform

    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(4, 2, 1000)),
        dims=("trial", "channel", "time"),
        coords={"time": ("time", np.arange(1000) / 500.0, {"units": "s"})},
        attrs={"sfreq": 250.0},  # stale
    )
    with pytest.warns(UserWarning, match="100"):
        BandpassFilterTransform(8, 30, fs=100).transform(DataContainer(data))
    BandpassFilterTransform(8, 30, fs=500).transform(DataContainer(data))  # the time coordinate supports it
    in_ms = data.assign_coords(time=("time", np.arange(1000) * 2.0, {"units": "ms"}))
    BandpassFilterTransform(8, 30, fs=500).transform(DataContainer(in_ms.assign_attrs(sfreq=1.0)))


def test_bandpass_checks_fs_against_a_time_coordinate_without_units():
    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.spectral import BandpassFilterTransform

    def data(time):
        return DataContainer(
            xr.DataArray(np.zeros((2, 3, time.size)), dims=("trial", "channel", "time"), coords={"time": time})
        )

    seconds_50hz = data(np.arange(128) / 50.0)
    with pytest.warns(UserWarning, match="100"):
        BandpassFilterTransform(1, 15, fs=100).transform(seconds_50hz)
    BandpassFilterTransform(1, 15, fs=50).transform(seconds_50hz)
    BandpassFilterTransform(8, 30, fs=256).transform(data(np.arange(256) * 1000 / 256))  # ms
    BandpassFilterTransform(8, 30, fs=160).transform(data(np.arange(320.0)))  # sample indices: nothing to check
    with pytest.raises(ValueError, match="fs"):
        BandpassFilterTransform(1, 15, fs=None)


@pytest.mark.parametrize("case", ["rounded_ms", "float32_session_time", "decimated_sample_indices"])
def test_correct_rates_on_ordinary_time_coordinates_do_not_warn(case):
    import warnings

    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.spectral import BandpassFilterTransform

    if case == "rounded_ms":  # 256 Hz, timestamps exported rounded to whole ms
        time, attrs, fs, units = np.round(np.arange(512) * 1000 / 256), {}, 256, "ms"
    elif case == "float32_session_time":  # absolute time from 1800 s, stored as float32
        time, attrs, fs, units = (1800 + np.arange(2000) / 1000).astype(np.float32), {}, 1000, "s"
    else:  # sample indices decimated by 4 with xarray; attrs['sfreq'] left at the original rate
        time, attrs, fs, units = np.arange(0, 4000, 4.0), {"sfreq": 1000.0}, 250, None
    coord = ("time", time, {"units": units}) if units else ("time", time)
    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(2, 2, time.size)),
        dims=("trial", "channel", "time"),
        coords={"time": coord},
        attrs=attrs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BandpassFilterTransform(8, 30, fs=fs).transform(DataContainer(data))
