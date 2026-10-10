import numpy as np
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.transforms.phase import HilbertPhaseTransform

FS = 500
T = np.arange(1000) / FS


def test_integer_input_gives_float_phases():
    rng = np.random.default_rng(0)
    signal = (1000 * np.sin(2 * np.pi * 6 * T))[None, None, :] + rng.normal(0, 50, size=(2, 3, 1000))
    data = xr.DataArray(
        signal, dims=("trial", "channel", "time"), coords={"trial": [0, 1], "channel": [0, 1, 2], "time": T * 1000}
    )
    transform = HilbertPhaseTransform(fs=FS, mode="timepoints", num_hf_bands_remove=4)

    from_float = transform.transform(DataContainer(data)).data
    from_int = transform.transform(DataContainer(data.astype(np.int16))).data

    assert np.issubdtype(from_int.dtype, np.floating)
    np.testing.assert_allclose(from_int.values, from_float.values, atol=0.05)


def test_relative_average_is_a_circular_mean():
    rng = np.random.default_rng(0)
    base = np.sin(2 * np.pi * 6 * T)
    signal = np.stack([base, base, -base]) + rng.normal(0, 1e-3, size=(3, 1000))
    data = xr.DataArray(
        signal[None], dims=("trial", "channel", "time"), coords={"trial": [0], "channel": [0, 1, 2], "time": T * 1000}
    )
    out = HilbertPhaseTransform(
        fs=FS, mode="relative_average", num_hf_bands_remove=4, lfp_pad_ms_at_ends=200
    ).transform(DataContainer(data))

    # the antiphase channel sits at +/-pi relative to the channel average
    np.testing.assert_allclose(np.abs(out.data.sel(channel=2).values), np.pi, atol=0.01)
