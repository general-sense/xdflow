"""Aligner and AdaptWrapperTransform hyperparameters survive set_params and clone."""

import numpy as np
import pytest
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.transforms.adapt_wrapper import AdaptWrapperTransform
from xdflow.transforms.domain_adaptation import CoralAligner, DummyAligner, ProcrustesAligner


class _StubAdaptEstimator:
    """Stands in for an `adapt` estimator; the real ones need the optional `adapt` package."""

    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_

    def fit(self, X, y, Xt, yt):
        return self

    def transform(self, X):
        return X


class _SeededStub(_StubAdaptEstimator):
    def __init__(self, lambda_=1.0, random_state=None):
        super().__init__(lambda_)
        self.random_state = random_state


def _adapt_wrapper(cls=_StubAdaptEstimator, **kwargs):
    return AdaptWrapperTransform(
        adapt_estimator_cls=cls,
        sample_dim="trial",
        target_coord="stimulus",
        group_coord="session",
        target_group="b",
        **kwargs,
    )


def _two_sessions():
    n = 8
    return DataContainer(
        xr.DataArray(
            np.random.default_rng(0).normal(size=(n, 3)),
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(n),
                "feature": np.arange(3),
                "stimulus": ("trial", np.array(["x", "y"] * (n // 2))),
                "session": ("trial", np.array(["a"] * (n // 2) + ["b"] * (n // 2))),
            },
        )
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: ProcrustesAligner(target_coord="stimulus", sample_dim="trial", group_coord="session", target_group="a"),
        lambda: CoralAligner(sample_dim="trial", group_coord="session", target_group="a"),
    ],
    ids=["Procrustes", "Coral"],
)
def test_aligner_strategy_args_set_after_construction_reach_the_fit(make):
    aligner = make()
    aligner.set_params(target_group="b", n_jobs=2)
    for obj in (aligner, aligner.clone().clone()):
        assert obj.get_params()["target_group"] == "b"
        obj.fit(_two_sessions())
        assert obj.strategy.target_group == "b"
        assert obj.strategy.n_jobs == 2


def test_dummy_aligner_keeps_sel_on_clone():
    assert DummyAligner(sel={"stimulus": ["x"]}).clone().sel == {"stimulus": ["x"]}


def test_adapt_wrapper_set_params_reaches_what_fit_builds():
    transform = _adapt_wrapper(lambda_=0.5)
    transform.set_params(lambda_=0.9)
    assert transform.get_params()["lambda_"] == 0.9
    assert transform.clone().fit(_two_sessions()).adapt_estimator.lambda_ == 0.9


def test_adapt_wrapper_gives_n_jobs_to_the_strategy_alone():
    transform = _adapt_wrapper(lambda_=0.5, n_jobs=4)  # used to reach the estimator constructor and raise
    assert "n_jobs" not in transform._adapt_estimator_kwargs
    transform.set_params(n_jobs=8)
    assert transform.clone().fit(_two_sessions()).strategy.n_jobs == 8


def test_adapt_wrapper_rejects_nested_estimator_spelling():
    with pytest.raises(ValueError):
        _adapt_wrapper(lambda_=0.5).set_params(adapt_estimator__lambda_=0.9)


def test_adapt_wrapper_passes_random_state_to_an_estimator_that_takes_one():
    transform = _adapt_wrapper(_SeededStub)
    transform.set_params(random_state=7)
    assert transform.clone().fit(_two_sessions()).adapt_estimator.random_state == 7


def _session_data(target_classes=("a", "b", "c")):
    rng = np.random.default_rng(0)
    y = np.r_[np.tile(["a", "b", "c"], 10), np.tile(list(target_classes), 10)]
    session = np.repeat(["A", "B"], 30)
    perm = rng.permutation(60)
    return DataContainer(
        xr.DataArray(
            rng.normal(size=(60, 4)),
            dims=("trial", "feature"),
            coords={"trial": np.arange(60), "feature": np.arange(4), "y": ("trial", y), "session": ("trial", session)},
        ).isel(trial=perm)
    )


def test_aligners_keep_input_trial_order():
    data = _session_data()
    out = ProcrustesAligner("y", "trial", group_coord="session", target_group="A").fit_transform(data).data

    np.testing.assert_array_equal(out["trial"].values, data.data["trial"].values)


def test_aligners_refuse_groups_with_different_classes():
    with pytest.raises(ValueError, match="same 'y' classes"):
        ProcrustesAligner("y", "trial", group_coord="session", target_group="A").fit(_session_data(("a", "b", "d")))


def test_aligners_keep_order_without_unique_trial_labels():
    data = _session_data()
    duplicated = DataContainer(data.data.assign_coords(trial=np.zeros(data.data.sizes["trial"], dtype=int)))
    out = ProcrustesAligner("y", "trial", group_coord="session", target_group="A").fit_transform(duplicated).data

    np.testing.assert_array_equal(out["y"].values, duplicated.data["y"].values)
    assert not any(str(c).startswith("_xdflow_input_position") for c in out.coords)


def test_aligner_pickled_with_legacy_strategy_kwargs_still_clones():
    import pickle

    aligner = CoralAligner(sample_dim="trial", group_coord="session", target_group="A")
    state = pickle.loads(pickle.dumps(aligner)).__dict__.copy()
    # xdflow < 0.3 kept strategy arguments in _strategy_kwargs instead of attributes
    legacy = {name: state.pop(name) for name in ("group_coord", "target_group", "n_jobs", "adapt_sel")}
    state["_strategy_kwargs"] = legacy
    restored = CoralAligner.__new__(CoralAligner)
    restored.__setstate__(state)

    assert restored.clone().target_group == "A"
