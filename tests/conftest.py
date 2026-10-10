"""Pytest configuration and fixtures for xdflow tests."""

import numpy as np
import pytest
import xarray as xr

from xdflow.core.data_container import DataContainer


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Keep xdflow's on-disk caches per test, so tests never hit results cached by other runs."""
    monkeypatch.setenv("XDFLOW_CACHE_DIR", str(tmp_path / "xdflow_cache"))


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_data_array():
    """Create a simple 2D DataArray for testing."""
    data = xr.DataArray(
        np.random.randn(100, 10),
        dims=["trial", "feature"],
        coords={"trial": range(100), "feature": range(10)},
    )
    return data


@pytest.fixture
def simple_container(simple_data_array):
    """Create a simple DataContainer for testing."""
    return DataContainer(simple_data_array)


@pytest.fixture
def timeseries_data_array():
    """Create a 3D time-series DataArray (trials × channels × time)."""
    n_trials = 50
    n_channels = 8
    n_timepoints = 100

    data = xr.DataArray(
        np.random.randn(n_trials, n_channels, n_timepoints),
        dims=["trial", "channel", "time"],
        coords={
            "trial": range(n_trials),
            "channel": [f"ch{i}" for i in range(n_channels)],
            "time": np.linspace(0, 1, n_timepoints),
        },
    )
    return data


@pytest.fixture
def timeseries_container(timeseries_data_array):
    """Create a time-series DataContainer for testing."""
    return DataContainer(timeseries_data_array)


@pytest.fixture
def labeled_data_array():
    """Create a DataArray with categorical labels for classification."""
    n_trials = 120
    n_features = 20

    # Create balanced classes
    labels = np.repeat([0, 1, 2], n_trials // 3)

    data = xr.DataArray(
        np.random.randn(n_trials, n_features),
        dims=["trial", "feature"],
        coords={"trial": range(n_trials), "feature": range(n_features), "label": ("trial", labels)},
    )
    return data


@pytest.fixture
def labeled_container(labeled_data_array):
    """Create a labeled DataContainer for classification testing."""
    return DataContainer(labeled_data_array)


@pytest.fixture
def multi_session_data_array():
    """Create a DataArray with session/subject structure."""
    n_trials_per_session = 30
    n_sessions = 4
    n_features = 15

    n_trials = n_trials_per_session * n_sessions

    sessions = np.repeat(range(n_sessions), n_trials_per_session)
    subjects = np.repeat([0, 0, 1, 1], n_trials_per_session)  # 2 sessions per subject

    data = xr.DataArray(
        np.random.randn(n_trials, n_features),
        dims=["trial", "feature"],
        coords={
            "trial": range(n_trials),
            "feature": range(n_features),
            "session": ("trial", sessions),
            "subject": ("trial", subjects),
        },
    )
    return data


@pytest.fixture
def multi_session_container(multi_session_data_array):
    """Create a multi-session DataContainer for CV testing."""
    return DataContainer(multi_session_data_array)


@pytest.fixture
def data_container_factory():
    """
    Factory fixture to create a DataContainer with specified parameters.
    This allows for flexible creation of test data with different shapes,
    dimensions, and properties.
    """

    def _create_data_container(
        n_trials=10,
        n_channels=8,
        n_time=100,
        seed=42,
    ):
        """
        Args:
            n_trials (int): Number of trials.
            n_channels (int): Number of channels.
            n_time (int): Number of time points.
            seed (int): Random seed for reproducibility.

        Returns:
            DataContainer: A new DataContainer instance.
        """
        rng = np.random.default_rng(seed)
        dims = ("trial", "channel", "time")
        shape = (n_trials, n_channels, n_time)
        data = rng.standard_normal(shape)

        # Create coordinates
        trials = np.arange(n_trials)
        times = np.linspace(0, 1, n_time)
        channels = [f"ch{i}" for i in range(n_channels)]

        coords = {
            "trial": trials,
            "channel": channels,
            "time": times,
        }

        data_array = xr.DataArray(data, dims=dims, coords=coords)
        return DataContainer(data_array)

    return _create_data_container


@pytest.fixture
def assert_transform_immutability():
    """
    Fixture that provides a function to test that transforms do not mutate their input.

    This enforces the Transform-Responsibility Model: transforms must not mutate
    the input DataContainer or its underlying data array.

    Usage:
        def test_my_transform_immutability(data_container_factory, assert_transform_immutability):
            container = data_container_factory()
            transform = MyTransform()
            assert_transform_immutability(transform, container)
    """

    def _assert_immutability(transform, container, **transform_kwargs):
        """
        Tests that a transform does not mutate the input container in-place.

        Args:
            transform: The transform instance to test
            container: DataContainer to test with
            **transform_kwargs: Additional kwargs to pass to transform()

        Raises:
            AssertionError: If the transform mutates its input
        """
        import copy

        # Create multiple ways to detect mutation

        # 1. Hash of the underlying numpy data
        original_data_bytes = container.data.values.tobytes()
        original_data_hash = hash(original_data_bytes)

        # 2. Deep copy for comparison
        original_data_copy = copy.deepcopy(container.data.values)

        # 3. Reference check - the data array should be a different object
        original_data_id = id(container.data.values)

        # 4. Coordinate preservation
        original_coords = {
            k: v.values.copy() if hasattr(v, "values") else copy.deepcopy(v) for k, v in container.data.coords.items()
        }

        # Fit the transform if it's stateful
        if hasattr(transform, "is_stateful") and transform.is_stateful:
            # Fit should also not mutate
            transform.fit(container, **transform_kwargs)

            # Re-check after fit
            post_fit_hash = hash(container.data.values.tobytes())
            assert original_data_hash == post_fit_hash, (
                f"Transform '{type(transform).__name__}' mutated input data during fit()!"
            )

        # Run the transform
        result_container = transform.transform(container, **transform_kwargs)

        # Check 1: Hash comparison
        post_transform_hash = hash(container.data.values.tobytes())
        assert original_data_hash == post_transform_hash, (
            f"Transform '{type(transform).__name__}' mutated input data during transform()!"
        )

        # Check 2: Element-wise comparison
        assert np.array_equal(container.data.values, original_data_copy, equal_nan=True), (
            f"Transform '{type(transform).__name__}' mutated input data values!"
        )

        # Check 3: Ensure result is a new container instance
        assert id(result_container) != id(container), (
            f"Transform '{type(transform).__name__}' returned the same container instance!"
        )

        # Check 4: Ensure result has new data array (not just a view)
        assert id(result_container.data.values) != original_data_id, (
            f"Transform '{type(transform).__name__}' returned a view of the original data!"
        )

        # Check 5: Verify coordinates weren't mutated
        for coord_name, orig_coord_values in original_coords.items():
            if coord_name in container.data.coords:
                current_values = (
                    container.data.coords[coord_name].values
                    if hasattr(container.data.coords[coord_name], "values")
                    else container.data.coords[coord_name]
                )
                # Handle different dtypes - strings don't support equal_nan
                if np.issubdtype(current_values.dtype, np.number):
                    assert np.array_equal(current_values, orig_coord_values, equal_nan=True), (
                        f"Transform '{type(transform).__name__}' mutated coordinate '{coord_name}'!"
                    )
                else:
                    assert np.array_equal(current_values, orig_coord_values), (
                        f"Transform '{type(transform).__name__}' mutated coordinate '{coord_name}'!"
                    )

    return _assert_immutability
