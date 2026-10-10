"""A parameter set after construction must reach the estimator that each CV fold actually fits.

CV clones the stateful pipeline once per fold, and SKLearnTransform.clone() rebuilds the estimator from
`_estimator_kwargs`; writes that only reached the live estimator were silently reverted in every fold.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.tuning.base import Tuner

N_TRIALS = 24
CONSTRUCTED_WITH = 1
TUNED_TO = 99

#: Fold fits append here (module-level: clone copies param values, so a list param would not be shared).
FITTED_WITH: list[int] = []


class _RecordingEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, marker: int = CONSTRUCTED_WITH):
        self.marker = marker

    def fit(self, X, y):
        FITTED_WITH.append(self.marker)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _container():
    return DataContainer(
        xr.DataArray(
            np.random.default_rng(0).normal(size=(N_TRIALS, 3)),
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(N_TRIALS),
                "feature": np.arange(3),
                "stimulus": ("trial", np.array(["a", "b"] * (N_TRIALS // 2))),
            },
        )
    )


def _pipeline():
    FITTED_WITH.clear()
    clf = SKLearnPredictor(
        estimator_cls=_RecordingEstimator, sample_dim="trial", target_coord="stimulus", marker=CONSTRUCTED_WITH
    )
    return Pipeline(name="recording_pipeline", steps=[("clf", clf)])


@pytest.mark.parametrize("write", ["set_params", "setattr"])
def test_every_fold_fits_the_value_set_after_construction(write):
    pipeline = _pipeline()
    if write == "set_params":
        pipeline.set_params(clf__marker=TUNED_TO)
    else:
        pipeline.get_transform_from_name("clf").marker = TUNED_TO

    validator = KFoldValidator(n_splits=3, test_size=0.2, use_stateful_fit_cache=False, verbose=False)
    validator.pipeline = pipeline
    validator.cross_validate(_container(), verbose=False)

    assert FITTED_WITH and set(FITTED_WITH) == {TUNED_TO}


def test_untouched_construction_value_is_still_what_folds_fit():
    validator = KFoldValidator(n_splits=3, test_size=0.2, use_stateful_fit_cache=False, verbose=False)
    validator.pipeline = _pipeline()
    validator.cross_validate(_container(), verbose=False)

    assert set(FITTED_WITH) == {CONSTRUCTED_WITH}


def test_the_tuners_seed_reaches_a_model_inside_an_automatic_wrapper():
    predictor = SKLearnPredictor(
        estimator_cls=Ridge, sample_dim="trial", target_coord=["a", "b"], is_classifier=False, random_state=0
    )
    Tuner._inject_random_seeds_into_pipelines(SimpleNamespace(pipelines_to_tune={"p": predictor}, random_seed=123))

    assert predictor.clone().clone().estimator.estimator.random_state == 123


def test_bare_param_reaches_model_inside_an_automatic_wrapper():
    predictor = SKLearnPredictor(
        estimator_cls=Ridge, sample_dim="trial", target_coord=["a", "b"], is_classifier=False, alpha=1.0
    )
    predictor.set_params(alpha=123.0)

    assert predictor.estimator.estimator.alpha == 123.0
    assert predictor.clone().estimator.estimator.alpha == 123.0


def test_the_tuners_seed_reaches_an_unfitted_group_apply_template():
    from xdflow.composite import GroupApplyTransform

    group_apply = GroupApplyTransform(
        "session", SKLearnPredictor(estimator_cls=Ridge, sample_dim="trial", target_coord="a", is_classifier=False)
    )
    Tuner._inject_random_seeds_into_pipelines(SimpleNamespace(pipelines_to_tune={"p": group_apply}, random_seed=123))

    assert group_apply.clone().transform_template.estimator.random_state == 123
