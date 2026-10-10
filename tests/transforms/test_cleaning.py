"""Tests for cleaning transforms."""

import numpy as np
import pytest
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.transforms.cleaning import CARTransform, RegressOutReferenceTransform, RemoveOutliersTransform


class TestCARTransform:
    def test_default_applies_common_average_across_all_channels(self, data_container_factory):
        container = data_container_factory(n_trials=4, n_channels=5, n_time=6)

        result = CARTransform().transform(container)

        np.testing.assert_allclose(result.data.mean(dim="channel").values, 0.0, atol=1e-12)
        assert result.data.dims == container.data.dims
        np.testing.assert_array_equal(result.data.coords["channel"].values, container.data.coords["channel"].values)

    def test_excluded_channels_are_left_unchanged(self, data_container_factory):
        container = data_container_factory(n_trials=4, n_channels=4, n_time=6)

        result = CARTransform(excluded_channels=["ch2"]).transform(container)

        np.testing.assert_allclose(
            result.data.sel(channel="ch2").values,
            container.data.sel(channel="ch2").values,
        )
        signal_data = result.data.sel(channel=["ch0", "ch1", "ch3"])
        np.testing.assert_allclose(signal_data.mean(dim="channel").values, 0.0, atol=1e-12)

    def test_none_method_leaves_data_unchanged(self, data_container_factory):
        container = data_container_factory(n_trials=4, n_channels=5, n_time=6)

        result = CARTransform(car_method="none").transform(container)

        np.testing.assert_allclose(result.data.values, container.data.values)

    def test_invalid_car_method_raises(self, data_container_factory):
        container = data_container_factory(n_trials=4, n_channels=5, n_time=6)

        with pytest.raises(ValueError, match="Must be one of 'all' or 'none'"):
            CARTransform(car_method="invalid").transform(container)

    def test_immutability(self, data_container_factory, assert_transform_immutability):
        container = data_container_factory(n_trials=4, n_channels=5, n_time=6)

        assert_transform_immutability(CARTransform(), container)


class TestRemoveOutliersTransform:
    def test_clips_outliers_per_channel(self, data_container_factory):
        container = data_container_factory(n_trials=4, n_channels=3, n_time=5)
        data = container.data.copy(deep=True)
        outlier_coord = {"trial": 0, "channel": "ch0", "time": data.coords["time"].values[0]}
        data.loc[outlier_coord] = 100.0

        result = RemoveOutliersTransform(per_dim="channel", std_threshold=1.0).transform(DataContainer(data))

        assert result.data.sel(outlier_coord).item() < 100.0
        assert result.data.dims == container.data.dims

    def test_per_dim_is_public_constructor_parameter(self):
        transform = RemoveOutliersTransform(per_dim=["channel", "time"], std_threshold=4.0)

        params = transform.get_params(deep=False)
        cloned = transform.clone()

        assert params["per_dim"] == ["channel", "time"]
        assert set(params) == {
            "per_dim",
            "std_threshold",
            "use_fit",
            "sel",
            "drop_sel",
            "transform_sel",
            "transform_drop_sel",
        }
        assert cloned.per_dim == ["channel", "time"]
        assert cloned.std_threshold == 4.0


def test_regress_out_reference_use_fit_learns_from_fit_data_only():
    import warnings

    import xarray as xr

    rng = np.random.default_rng(0)

    def make(gain):
        ref = rng.normal(size=(10, 1, 50))
        values = np.concatenate([ref, gain * ref + rng.normal(size=(10, 2, 50))], axis=1)
        return DataContainer(
            xr.DataArray(
                values,
                dims=("trial", "channel", "time"),
                coords={"trial": np.arange(10), "channel": ["ref", "a", "b"], "time": np.arange(50)},
            )
        )

    train, test = make(0.5), make(2.0)
    fitted = RegressOutReferenceTransform("ref", use_fit=True)
    assert fitted.is_stateful
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted.fit(train)
        from_train = fitted.transform(test).data

    with pytest.warns(UserWarning, match="across trials"):
        from_test = RegressOutReferenceTransform("ref").transform(test).data

    # the train-fitted gain (0.5) leaves part of the test signal's 2.0 * ref behind
    assert not np.allclose(from_train.sel(channel="a"), from_test.sel(channel="a"))


def test_regress_out_reference_refuses_channels_it_was_not_fitted_on():
    def signals(channels):
        ref = np.arange(10.0).reshape(2, 5)
        return DataContainer(
            xr.DataArray(
                np.stack([ref] * len(channels), axis=1),
                dims=("trial", "channel", "time"),
                coords={"channel": channels},
            )
        )

    fitted = RegressOutReferenceTransform("ref", use_fit=True).fit(signals(["ref", "a"]))
    with pytest.raises(ValueError, match="'b'"):
        fitted.transform(signals(["ref", "a", "b"]))
