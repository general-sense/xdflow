"""Tests for normalization transforms (Demean and ZScore)."""

import numpy as np
import pytest

from xdflow.transforms.normalization import DemeanTransform, ZScoreTransform


class TestDemeanTransform:
    """Test DemeanTransform functionality."""

    def test_demean_per_time(self, data_container_factory):
        """Test demeaning per time point (compute mean over trial/channel)."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        # per_dim="time" means: demean PER time point (compute mean over trial, channel)
        transform = DemeanTransform(per_dim="time")

        result = transform.fit_transform(container)

        # Mean over (trial, channel) should be ~0 for each time point
        trial_channel_means = result.data.mean(dim=["trial", "channel"])
        np.testing.assert_array_almost_equal(trial_channel_means.values, 0, decimal=10)

    def test_demean_per_channel(self, data_container_factory):
        """Test demeaning per channel (compute mean over trial/time)."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        # per_dim="channel" means: demean PER channel (compute mean over trial, time)
        transform = DemeanTransform(per_dim="channel")

        result = transform.fit_transform(container)

        # Mean over (trial, time) should be ~0 for each channel
        trial_time_means = result.data.mean(dim=["trial", "time"])
        np.testing.assert_array_almost_equal(trial_time_means.values, 0, decimal=10)

    def test_demean_stateful(self, data_container_factory):
        """Test stateful demeaning (fit on train, apply to test)."""
        train_container = data_container_factory(n_trials=10, n_channels=8, n_time=100, seed=42)
        test_container = data_container_factory(n_trials=5, n_channels=8, n_time=100, seed=43)

        transform = DemeanTransform(per_dim="time", use_fit=True)

        # Fit on train
        transform.fit(train_container)

        # Transform test
        result = transform.transform(test_container)

        # Test data should be shifted by train mean (not centered to 0)
        assert result.data.shape == test_container.data.shape
        assert transform.is_stateful


class TestZScoreTransform:
    """Test ZScoreTransform functionality."""

    def test_zscore_per_time(self, data_container_factory):
        """Test z-scoring per time point (compute stats over trial/channel)."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        # per_dim="time" means: z-score PER time point
        transform = ZScoreTransform(per_dim="time")

        result = transform.fit_transform(container)

        # Mean over (trial, channel) should be ~0, std should be ~1
        trial_channel_means = result.data.mean(dim=["trial", "channel"])
        trial_channel_stds = result.data.std(dim=["trial", "channel"])

        np.testing.assert_array_almost_equal(trial_channel_means.values, 0, decimal=10)
        np.testing.assert_array_almost_equal(trial_channel_stds.values, 1, decimal=1)

    def test_zscore_per_channel(self, data_container_factory):
        """Test z-scoring per channel (compute stats over trial/time)."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        # per_dim="channel" means: z-score PER channel
        transform = ZScoreTransform(per_dim="channel")

        result = transform.fit_transform(container)

        # Mean over (trial, time) should be ~0, std should be ~1
        trial_time_means = result.data.mean(dim=["trial", "time"])
        trial_time_stds = result.data.std(dim=["trial", "time"])

        np.testing.assert_array_almost_equal(trial_time_means.values, 0, decimal=10)
        np.testing.assert_array_almost_equal(trial_time_stds.values, 1, decimal=1)

    def test_zscore_stateful(self, data_container_factory):
        """Test stateful z-scoring (fit on train, apply to test)."""
        train_container = data_container_factory(n_trials=10, n_channels=8, n_time=100, seed=42)
        test_container = data_container_factory(n_trials=5, n_channels=8, n_time=100, seed=43)

        transform = ZScoreTransform(per_dim="time", use_fit=True)

        # Fit on train
        transform.fit(train_container)

        # Transform test using train statistics
        result = transform.transform(test_container)

        # Test data should NOT be perfectly standardized (using train stats)
        assert result.data.shape == test_container.data.shape
        assert transform.is_stateful


class TestNormalizationComparison:
    """Test differences between Demean and ZScore."""

    def test_zscore_scales_demean_doesnt(self, data_container_factory):
        """Test that ZScore scales variance but Demean doesn't."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)

        demean = DemeanTransform(per_dim="time")
        zscore = ZScoreTransform(per_dim="time")

        demean_result = demean.fit_transform(container)
        zscore_result = zscore.fit_transform(container)

        # Both should have mean ~0 (per time point, over trial/channel)
        demean_means = demean_result.data.mean(dim=["trial", "channel"])
        zscore_means = zscore_result.data.mean(dim=["trial", "channel"])
        np.testing.assert_array_almost_equal(demean_means.values, 0, decimal=10)
        np.testing.assert_array_almost_equal(zscore_means.values, 0, decimal=10)

        # ZScore should have std ~1 (per time point)
        zscore_stds = zscore_result.data.std(dim=["trial", "channel"])
        np.testing.assert_array_almost_equal(zscore_stds.values, 1, decimal=1)

        # Demean should NOT have std ~1
        demean_stds = demean_result.data.std(dim=["trial", "channel"])
        assert not np.allclose(demean_stds.values, 1, atol=0.2)

    def test_per_dim_is_public_constructor_parameter(self):
        """Test per_dim is exposed through params and clone."""
        transform = ZScoreTransform(per_dim="time", use_fit=True)

        params = transform.get_params(deep=False)
        cloned = transform.clone()

        assert params["per_dim"] == "time"
        assert set(params) == {
            "per_dim",
            "use_fit",
            "fit_sel",
            "sel",
            "drop_sel",
            "transform_sel",
            "transform_drop_sel",
        }
        assert cloned.per_dim == "time"
        assert cloned.use_fit is True

    def test_per_dim_accepts_multiple_dimensions(self, data_container_factory):
        """Test z-scoring per channel/time coordinate."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        transform = ZScoreTransform(per_dim=["channel", "time"])

        result = transform.fit_transform(container)

        trial_means = result.data.mean(dim="trial")
        trial_stds = result.data.std(dim="trial")
        np.testing.assert_array_almost_equal(trial_means.values, 0, decimal=10)
        np.testing.assert_array_almost_equal(trial_stds.values, 1, decimal=1)

    def test_set_params_updates_per_dim_behavior(self, data_container_factory):
        """Test tuning-style per_dim updates change the statistics dimensions."""
        container = data_container_factory(n_trials=5, n_channels=8, n_time=100)
        transform = ZScoreTransform(per_dim="time")

        transform.set_params(per_dim="channel")
        result = transform.fit_transform(container)

        trial_time_means = result.data.mean(dim=["trial", "time"])
        np.testing.assert_array_almost_equal(trial_time_means.values, 0, decimal=10)


def test_stateless_normalization_warns_only_when_pooling_trials():
    import warnings

    import numpy as np
    import xarray as xr

    from xdflow.core.data_container import DataContainer

    data = DataContainer(xr.DataArray(np.random.default_rng(0).normal(size=(4, 3)), dims=("trial", "channel")))
    with pytest.warns(UserWarning, match="across trials"):
        ZScoreTransform(per_dim="channel").transform(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ZScoreTransform(per_dim="trial").transform(data)
        ZScoreTransform(per_dim="channel", use_fit=True).fit_transform(data)


def test_per_dim_must_name_an_existing_dim():
    import xarray as xr

    from xdflow.core.data_container import DataContainer

    data = DataContainer(xr.DataArray(np.zeros((4, 3)), dims=("trial", "flat_channel__time")))
    with pytest.raises(ValueError, match="not found in data dims"):
        ZScoreTransform(per_dim="feature", use_fit=True).fit_transform(data)
    with pytest.raises(ValueError, match="not found in data dims"):
        DemeanTransform(per_dim="feature").transform(data)
