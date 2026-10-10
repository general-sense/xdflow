"""SplitterValidator runs xdflow cross-validation over any scikit-learn splitter or explicit index pairs."""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold, TimeSeriesSplit

from xdflow import DataContainer, Pipeline, Transform
from xdflow.cv import SplitterValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b"])[np.arange(60) % 2]
    values = rng.normal(size=(60, 4))
    values[labels == "a", 0] += 1.5
    # Trial labels deliberately differ from positions, to catch position/label mix-ups.
    return xr.DataArray(
        values,
        dims=("trial", "feature"),
        coords={
            "trial": np.arange(60) * 10 + 7,
            "label": ("trial", labels),
            "subject": ("trial", np.repeat(np.arange(6), 10)),
        },
    )


def _pipeline():
    return Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])


def _run(validator, data):
    validator.set_pipeline(_pipeline())
    return validator.cross_validate(DataContainer(data))


@pytest.mark.parametrize("splitter", [GroupKFold(3), LeaveOneGroupOut()])
def test_group_splitters_never_share_a_group(data, splitter):
    validator = SplitterValidator(splitter, group_coord="subject", verbose=False)
    _run(validator, data)
    subject = data.coords["subject"].to_series()
    for train, test in validator.split_trials_:
        assert set(subject[train]).isdisjoint(subject[test])
    assert len(validator.split_trials_) == (3 if isinstance(splitter, GroupKFold) else 6)


def test_stratified_splitter_receives_the_target(data):
    validator = SplitterValidator(StratifiedKFold(5), verbose=False)
    _run(validator, data)
    labels = data.coords["label"].to_series()
    for _, test in validator.split_trials_:
        assert (labels[test] == "a").sum() == (labels[test] == "b").sum()


def test_index_pairs_are_positions_and_every_fold_is_scored(data):
    pairs = [(np.arange(30), np.arange(30, 60)), (np.arange(30, 60), np.arange(30))]
    validator = SplitterValidator(pairs, verbose=False)
    _run(validator, data)
    np.testing.assert_array_equal(validator.split_trials_[0][1], data.trial.values[30:])
    assert len(validator.cv_scores_) == 2


def test_order_respecting_splitter(data):
    validator = SplitterValidator(TimeSeriesSplit(3), verbose=False)
    _run(validator, data)
    for train, test in validator.split_trials_:
        assert train.max() < test.min()


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        ([(np.arange(31), np.arange(30, 60))], "both training and test"),
        ([(np.arange(30), np.arange(30, 61))], r"in \[0, 60\)"),
        ([(np.arange(30.0), np.arange(30, 60))], "integer positions"),
    ],
)
def test_malformed_index_pairs_raise(data, pairs, message):
    with pytest.raises(ValueError, match=message):
        _run(SplitterValidator(pairs, verbose=False), data)


def test_group_splitter_without_group_coord_explains(data):
    with pytest.raises(ValueError, match="group_coord"):
        _run(SplitterValidator(LeaveOneGroupOut(), verbose=False), data)


def test_generator_of_pairs_is_reusable(data):
    pairs = ((np.arange(30), np.arange(30, 60)) for _ in range(1))
    validator = SplitterValidator(pairs, verbose=False)
    first = _run(validator, data)
    assert _run(validator, data) == first


class _Reverse(Transform):
    """A stateless step that reverses trial order."""

    def _transform(self, container, **kwargs):
        return DataContainer(container.data.isel(trial=slice(None, None, -1)))

    def get_expected_output_dims(self, input_dims):
        return input_dims


def test_positions_refer_to_input_order_after_a_reordering_step(data):
    pairs = [(np.arange(40), np.arange(40, 60))]
    validator = SplitterValidator(pairs, verbose=False)
    validator.set_pipeline(Pipeline("p", [("reverse", _Reverse()), ("lr", _pipeline().steps[0].transform)]))
    validator.cross_validate(DataContainer(data))
    np.testing.assert_array_equal(validator.split_trials_[0][1], data.trial.values[40:])
