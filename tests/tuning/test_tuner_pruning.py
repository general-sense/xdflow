import numpy as np
import optuna
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.tuning.base import Tuner


def _container() -> DataContainer:
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b", "c"])[np.arange(60) % 3]
    return DataContainer(
        xr.DataArray(
            rng.normal(size=(60, 4)),
            dims=("trial", "feature"),
            coords={"trial": np.arange(60), "feature": np.arange(4), "stimulus": ("trial", labels)},
        )
    )


def _tuner(estimator_cls, pruner=None) -> Tuner:
    clf = SKLearnPredictor(estimator_cls, sample_dim="trial", target_coord="stimulus", max_iter=200)
    return Tuner(
        pipelines_to_tune=Pipeline("p", [("clf", clf)]),
        cv_strategy=KFoldValidator(n_splits=3, use_stateful_fit_cache=False, verbose=False),
        param_grid={"p": {"clf": {"C": ("log", 1e-2, 1.0)}}},
        initial_data_container=_container(),
        pruner=pruner,
        verbose=0,
        use_mlflow=False,
        use_cache=False,
    )


def test_fold_scores_reach_the_pruner():
    tuner = _tuner(LogisticRegression, pruner=optuna.pruners.MedianPruner(n_startup_trials=100))
    tuner.tune(n_trials=2)

    assert all(len(trial.intermediate_values) == 3 for trial in tuner.study.trials)


def test_tune_names_the_error_when_every_trial_is_pruned():
    class DivergingLR(LogisticRegression):
        def fit(self, X, y, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

    with pytest.raises(ValueError, match="(?s)No tuning trial completed.*SVD did not converge"):
        _tuner(DivergingLR).tune(n_trials=2)
