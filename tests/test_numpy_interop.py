"""from_numpy wraps sample-first arrays with named dims and per-sample coordinates."""

import numpy as np
import pandas as pd
import pytest

from xdflow.interop import _as_container, from_numpy


def test_names_dims_and_attaches_coordinates():
    X = np.zeros((6, 2, 3))
    data = from_numpy(
        X,
        ("trial", "channel", "time"),
        coords={"channel": ["Cz", "Pz"]},
        metadata=pd.DataFrame({"session": [1, 1, 2, 2, 3, 3]}),
        label=list("ababab"),
        subject=7,
    ).data
    assert data.dims == ("trial", "channel", "time")
    np.testing.assert_array_equal(data.trial.values, np.arange(6))
    assert list(data.channel.values) == ["Cz", "Pz"]
    assert data.coords["session"].dims == ("trial",)
    np.testing.assert_array_equal(data.coords["subject"].values, np.full(6, 7))
    assert from_numpy(np.zeros((4, 5))).data.dims == ("trial", "dim_1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dims": ("trial",)}, "1 dim names"),
        ({"dims": ("sample", "feature")}, "must be named 'trial'"),
        ({"label": [1, 2]}, r"expected \(4,\)"),
        ({"metadata": pd.DataFrame({"a": [1, 2]})}, "2 rows for 4 samples"),
        ({"coords": {"dim_1": [0, 1]}}, "2 values for a dim of size 5"),
        ({"coords": {"trial": [0, 1, 2, 3]}}, "clashes"),
    ],
)
def test_rejects_inconsistent_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        from_numpy(np.zeros((4, 5)), **kwargs)


def test_as_container_refuses_to_overwrite_existing_coordinates():
    data = from_numpy(np.zeros((4, 5)), label=[0, 1, 0, 1])
    with pytest.raises(ValueError, match="already exist"):
        _as_container(data, label=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="only applies to NumPy"):
        _as_container(data, dims=("trial", "x"))
