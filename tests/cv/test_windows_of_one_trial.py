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


def test_holdout_ids_that_match_no_group_are_refused():
    validator = LeaveGroupOutValidator(group_coord="subject", test_group_ids=[50], verbose=False)
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    with pytest.raises(ValueError, match="50"):
        validator.cross_validate(_windows())
