"""Tests for sklearn CV adapter."""

import numpy as np
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.cv.sklearn_adapter import SklearnCVAdapter, set_cv_container


class FakeCV:
    """Minimal CV implementation for adapter testing."""

    def _split_holdout(self, container: DataContainer):
        trials = container.data.trial.values
        return trials, []

    def _get_splits(self, container: DataContainer, train_val_indices):
        trials = list(train_val_indices)
        mid = len(trials) // 2
        yield trials[:mid], trials[mid:]


def test_sklearn_adapter_split_indices():
    """Adapter should map trial labels to 0-based indices."""
    data = xr.DataArray(
        np.random.randn(4, 2),
        dims=("trial", "feature"),
        coords={"trial": [10, 11, 12, 13], "feature": [0, 1]},
    )
    container = DataContainer(data)

    adapter = SklearnCVAdapter(FakeCV())

    with set_cv_container(container):
        splits = list(adapter.split(np.zeros((4, 2))))

    assert len(splits) == 1
    train_idx, val_idx = splits[0]
    assert np.array_equal(train_idx, np.array([0, 1]))
    assert np.array_equal(val_idx, np.array([2, 3]))


def test_split_count_follows_the_current_container():
    from xdflow.cv import LeaveGroupOutValidator

    def make(n_sessions):
        n = 10 * n_sessions
        return DataContainer(
            xr.DataArray(
                np.zeros((n, 2)),
                dims=("trial", "feature"),
                coords={
                    "trial": np.arange(n),
                    "feature": [0, 1],
                    "y": ("trial", np.tile(["a", "b"], n // 2)),
                    "session": ("trial", np.repeat([f"s{i}" for i in range(n_sessions)], 10)),
                },
            )
        )

    adapter = SklearnCVAdapter(LeaveGroupOutValidator(group_coord="session", verbose=False))
    for n_sessions in (3, 5):
        with set_cv_container(make(n_sessions)):
            assert adapter.get_n_splits() == len(list(adapter.split(None))) == n_sessions
