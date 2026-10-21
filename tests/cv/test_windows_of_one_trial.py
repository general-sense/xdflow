"""Validators keep windows cut from one trial (orig_trial) on one side of every split and of the holdout."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from xdflow import GroupedKFoldValidator, KFoldValidator, LeaveGroupOutValidator, Pipeline, SampledDomainKFoldValidator
from xdflow.interop import from_numpy
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _windows():
    rng = np.random.default_rng(0)
    n, w = 80, 4
    subject = np.repeat(np.arange(4), n // 4 * w)
    return from_numpy(
        rng.normal(size=(n * w, 3)),
        label=np.repeat(rng.integers(0, 2, n), w),
        orig_trial=np.repeat(np.arange(n), w),
        subject=subject,
        domain=np.where(subject == 0, "target", "source"),
    )


@pytest.mark.parametrize(
    "validator",
    [
        KFoldValidator(4, test_size=0.25, verbose=False),
        GroupedKFoldValidator(4, test_size=0.25, group_coord="subject", verbose=False),
        SampledDomainKFoldValidator(
            domain_coord="domain", target_domains="target", n_splits=4, test_size=0.25, verbose=False
        ),
    ],
    ids=lambda v: type(v).__name__,
)
def test_windows_of_one_trial_stay_together(validator):
    data = _windows()
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    validator.cross_validate(data)
    source = data.data.orig_trial.to_series()
    holdout = set(source.loc[validator.holdout_trial_labels_])
    for train, test in validator.split_trials_:
        train_sources, test_sources = set(source.loc[train]), set(source.loc[test])
        assert not train_sources & test_sources
        assert not (train_sources | test_sources) & holdout


def test_holdout_ids_can_be_a_single_value_dates_or_nested():
    from sklearn.model_selection import GridSearchCV

    from xdflow.cv import SklearnCVAdapter

    data = _windows()
    data = data.__class__(
        data.data.assign_coords(subject=("trial", np.array(["s0", "s1", "s2", "s3"])[data.data.subject.values]))
    )
    validator = LeaveGroupOutValidator(group_coord="subject", test_group_ids="s3", verbose=False)
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    validator.cross_validate(data)
    assert set(data.data.subject.sel(trial=validator.holdout_trial_labels_).values) == {"s3"}
    # nested: the inner adapter sees fold training data without the held-out subject
    inner = SklearnCVAdapter(LeaveGroupOutValidator(group_coord="subject", test_group_ids=["s3"], verbose=False))
    search = GridSearchCV(LogisticRegression(), {"C": [1.0]}, cv=inner)
    outer = LeaveGroupOutValidator(group_coord="subject", test_group_ids=["s3"], verbose=False)
    outer.set_pipeline(
        Pipeline("q", [("gs", SKLearnPredictor.from_estimator(search, sample_dim="trial", target_coord="label"))])
    )
    outer.cross_validate(data)
    days = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]")
    dated = data.__class__(data.data.assign_coords(day=("trial", days[data.data.subject.str[1].astype(int).values])))
    validator = LeaveGroupOutValidator(group_coord="day", test_group_ids=[days[-1]], verbose=False)
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    validator.cross_validate(dated)
    assert set(dated.data.subject.sel(trial=validator.holdout_trial_labels_).values) == {"s3"}


def _fitted(validator, data):
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    return validator.cross_validate(data)


def test_trial_numbers_restarting_in_each_subject_are_different_trials():
    # orig_trial 0..3 in every subject: ten subjects of four trials, not four trials
    data = from_numpy(
        np.random.default_rng(0).normal(size=(40, 2)),
        label=np.tile([0, 1], 20),
        orig_trial=np.tile(np.arange(4), 10),
        subject=np.repeat(np.arange(10), 4),
        session=np.zeros(40),
    )
    _fitted(GroupedKFoldValidator(5, group_coord="subject", stratify_by_group=False, verbose=False), data)


def _straddling(n=20, **coords):
    return from_numpy(
        np.random.default_rng(0).normal(size=(2 * n, 2)),
        label=np.repeat(np.arange(n) % 2, 2),
        orig_trial=np.repeat(np.arange(n), 2),
        **{name: np.resize(values, 2 * n) for name, values in coords.items()},
    )


@pytest.mark.parametrize(
    "validator, coord",
    [
        (
            SampledDomainKFoldValidator(domain_coord="domain", target_domains="target", n_splits=2, verbose=False),
            "domain",
        ),
        (GroupedKFoldValidator(2, group_coord="part", test_size=0.25, test_groups=["b"], verbose=False), "part"),
    ],
    ids=["domain", "holdout groups"],
)
def test_windows_of_one_trial_across_a_split_boundary_are_refused(validator, coord):
    # each trial has windows on both sides: training on its source (or non-test-group) window tests on the trial
    with pytest.raises(ValueError, match=f"different '{coord}'"):
        _fitted(validator, _straddling(domain=["source", "target"], part=["a", "b"]))


def test_windows_of_one_trial_in_unused_domains_or_filtered_cv_groups_are_fine():
    domains = SampledDomainKFoldValidator(
        domain_coord="domain", source_domains=["source"], target_domains=["target"], n_splits=2, verbose=False
    )
    _fitted(domains, _straddling(domain=["target", "ignored", "source", "source"]))  # the other window is unused
    groups = GroupedKFoldValidator(
        2, group_coord="part", train_groups=["a"], val_groups=["b"], stratify_by_group=False, verbose=False
    )
    _fitted(groups, _straddling(part=["a", "b"]))  # whole trials are split, then filtered


def test_subject_and_session_names_cannot_merge_trials():
    data = from_numpy(
        np.zeros((4, 1)),
        orig_trial=[0, 0, 0, 0],
        subject=["a::b", "a::b", "a", "a"],
        session=["c", "c", "b::c", "b::c"],
    )
    groups = KFoldValidator(verbose=False)._resolve_orig_trial_groups(data.data)
    assert groups[0] == groups[1] != groups[2] == groups[3]


def test_labels_varying_within_a_trial_disable_stratification():
    labels = np.zeros((12, 2), dtype=int)
    labels[:3, 1] = 1  # class 1 only on the second window of three trials
    data = from_numpy(
        np.random.default_rng(0).normal(size=(24, 2)),
        label=labels.ravel(),
        orig_trial=np.repeat(np.arange(12), 2),
        subject=np.zeros(24),
    )
    validator = GroupedKFoldValidator(3, group_coord="subject", stratify_by_group=False, verbose=False)
    with pytest.warns(UserWarning, match="vary within windows of one trial"):
        _fitted(validator, data)
