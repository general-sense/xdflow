"""Tests for PCA transform."""

import numpy as np

from xdflow.transforms.pca import GlobalFeaturePCA


class TestPCABasic:
    """Test basic PCA functionality."""

    def test_pca_creation(self):
        """Test PCA instantiation."""
        pca = GlobalFeaturePCA(n_components=5)
        assert pca.n_components == 5
        assert pca.is_stateful  # PCA is stateful

    def test_pca_fit_transform(self, timeseries_container):
        """Test PCA fit_transform."""
        pca = GlobalFeaturePCA(n_components=5)
        result = pca.fit_transform(timeseries_container)

        # Check output shape
        assert result.shape[0] == timeseries_container.shape[0]  # trials preserved
        assert result.shape[1] == 5  # reduced to 5 components

    def test_pca_separate_fit_transform(self, timeseries_container):
        """Test separate fit and transform calls."""
        pca = GlobalFeaturePCA(n_components=3)

        # Fit
        pca.fit(timeseries_container)

        # Transform
        result = pca.transform(timeseries_container)
        assert result.shape[1] == 3

    def test_pca_variance_explained(self, timeseries_container):
        """Test that PCA reduces dimensions."""
        # timeseries_container has shape (trial, channel, time)
        # After PCA it will be (trial, n_components)
        pca = GlobalFeaturePCA(n_components=5)
        result = pca.fit_transform(timeseries_container)

        assert result.shape[1] == 5


class TestPCAReproducibility:
    """Test PCA reproducibility."""

    def test_deterministic_output(self, timeseries_container):
        """Test that PCA produces deterministic results."""
        # GlobalFeaturePCA is deterministic (uses SVD, not randomized)
        pca1 = GlobalFeaturePCA(n_components=5)
        pca2 = GlobalFeaturePCA(n_components=5)

        result1 = pca1.fit_transform(timeseries_container)
        result2 = pca2.fit_transform(timeseries_container)

        np.testing.assert_array_almost_equal(result1.values, result2.values, decimal=10)


class TestPCACloning:
    """Test PCA cloning behavior."""

    def test_clone_unfitted(self):
        """Test cloning an unfitted PCA."""
        pca = GlobalFeaturePCA(n_components=5)
        cloned = pca.clone()

        assert cloned.n_components == pca.n_components
        assert cloned is not pca

    def test_clone_fitted(self, timeseries_container):
        """Test that cloning a fitted PCA creates an unfitted copy."""
        pca = GlobalFeaturePCA(n_components=5)
        pca.fit(timeseries_container)

        cloned = pca.clone()
        assert not hasattr(cloned, "_fitted")  # Should be unfitted


def test_fitted_projection_follows_channel_labels_not_positions():
    import numpy as np
    import xarray as xr

    from xdflow.core.data_container import DataContainer
    from xdflow.transforms.pca import GlobalFeaturePCA
    from xdflow.transforms.zca import GlobalZCAWhitening

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3, 20)) * np.array([1.0, 5.0, 0.2])[None, :, None]
    data = xr.DataArray(
        X,
        dims=("trial", "channel", "time"),
        coords={"trial": np.arange(30), "channel": ["a", "b", "c"], "time": np.arange(20)},
    )
    reordered = data.sel(channel=["c", "a", "b"])

    for transform in (GlobalFeaturePCA(n_components=2), GlobalZCAWhitening(n_components=3)):
        transform.fit(DataContainer(data))
        expected = transform.transform(DataContainer(data)).data
        actual = transform.transform(DataContainer(reordered)).data
        np.testing.assert_allclose(actual.values, expected.values)


def test_zca_whitening_fitted_before_the_whitening_fix_refuses_to_transform():
    import numpy as np
    import pytest

    from xdflow.transforms.zca import ZCAWhitening

    zca = ZCAWhitening().fit(np.random.default_rng(0).normal(size=(50, 3)))
    del zca.zca_version_  # as unpickled from xdflow < 0.3

    with pytest.raises(ValueError, match="refit"):
        zca.transform(np.zeros((2, 3)))
