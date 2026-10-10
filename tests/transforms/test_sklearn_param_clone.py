"""SKLearnTransform estimator parameters set after construction survive clone and reach every fold fit."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms import multi_output_wrapper
from xdflow.transforms.multi_output_wrapper import MultiOutputRegressorFactory
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer
from xdflow.tuning.base import Tuner

# --- clone() carries what set_params / attribute writes put on the estimator ---


def test_estimator_params_set_after_construction_survive_clone():
    predictor = SKLearnPredictor(
        RandomForestClassifier, sample_dim="trial", target_coord="stimulus", n_estimators=10, max_depth=3
    )
    predictor.set_params(n_estimators=57)
    predictor.min_samples_leaf = 4  # the __setattr__ proxy

    params = predictor.clone().clone().estimator.get_params()
    assert params["n_estimators"] == 57
    assert params["min_samples_leaf"] == 4
    assert params["max_depth"] == 3


def test_nested_estimator_spelling_survives_clone():
    predictor = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="y", is_classifier=False, alpha=1.0)
    predictor.set_params(estimator__alpha=4.0)
    assert predictor.clone().estimator.alpha == 4.0


def test_named_constructor_args_forwarded_to_the_estimator_survive_clone():
    transformer = SKLearnTransformer(PCA, sample_dim="trial", n_components=3)
    transformer.set_params(n_components=5)
    assert transformer.clone().estimator.n_components == 5


def test_clone_preserves_set_params_auto_multi_output():
    predictor = SKLearnPredictor(Ridge, sample_dim="trial", target_coord=["a", "b"], alpha=1.0)
    assert predictor.multi_output
    predictor.set_params(alpha=5.0)  # flat name reaches the inner model
    cloned = predictor.clone()
    assert isinstance(cloned.estimator, MultiOutputRegressor)
    assert cloned.estimator.estimator.alpha == 5.0


def test_multi_output_factory_as_estimator_cls_unwraps():
    predictor = SKLearnPredictor(
        MultiOutputRegressorFactory(Ridge), sample_dim="trial", target_coord=["a", "b"], alpha=7.0
    )
    assert predictor.multi_output is False  # the construction path the flag does not cover
    kwargs = predictor._get_clone_kwargs()
    assert kwargs["alpha"] == 7.0
    assert "estimator" not in kwargs
    predictor.set_params(alpha=2.0)
    assert predictor.clone().estimator.estimator.alpha == 2.0


@pytest.mark.parametrize("factory_name", ["MultiOutputRegressorFactory", "MultiOutputClassifierFactory"])
def test_multi_output_factory_preserves_live_inner_params(factory_name):
    is_classifier = factory_name == "MultiOutputClassifierFactory"
    predictor = SKLearnPredictor(
        getattr(multi_output_wrapper, factory_name)(LogisticRegression if is_classifier else Ridge),
        sample_dim="trial",
        target_coord=["a", "b"],
        is_multilabel=is_classifier,
        is_classifier=is_classifier,
    )
    predictor.set_params(random_state=17)
    assert predictor.clone().clone().estimator.estimator.random_state == 17


def test_multi_output_regressor_class_itself_as_estimator_cls_clones():
    predictor = SKLearnPredictor(
        MultiOutputRegressor,
        estimator=Ridge(alpha=7.0),
        sample_dim="trial",
        target_coord=["a", "b"],
        is_classifier=False,
        multi_output=True,
    )
    predictor.set_params(estimator__alpha=0.9, n_jobs=2)
    clone = predictor.clone()
    assert isinstance(clone.estimator, MultiOutputRegressor)
    assert clone.estimator.estimator.alpha == 0.9
    assert clone.estimator.n_jobs == 2
    assert clone.estimator.estimator is not predictor.estimator.estimator


def test_partial_estimator_preserves_live_params_and_drops_fitted_state():
    factory = partial(Ridge, alpha=2.0)
    predictor = SKLearnPredictor(factory, sample_dim="trial", target_coord="stimulus", is_classifier=False)
    predictor.set_params(alpha=5.0, random_state=17)
    predictor.estimator.fit(np.arange(12).reshape(6, 2), np.arange(6))
    cloned = predictor.clone().clone()
    assert cloned.estimator_cls is factory
    assert cloned.estimator.alpha == 5.0
    assert cloned.estimator.random_state == 17
    assert not hasattr(cloned.estimator, "coef_")
    assert factory.keywords == {"alpha": 2.0}


def test_fit_time_multi_output_rewrap_uses_current_params():
    n = 12
    container = DataContainer(
        xr.DataArray(
            np.random.default_rng(0).normal(size=(n, 3)),
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(n),
                "feature": np.arange(3),
                "a_conc": ("trial", np.linspace(0, 1, n)),
                "b_conc": ("trial", np.linspace(1, 0, n)),
            },
        )
    )
    predictor = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="*_conc", is_classifier=False, alpha=1.0)
    predictor.set_params(alpha=5.0)
    predictor.fit(container)
    assert predictor.multi_output
    assert predictor.estimator.estimator.alpha == 5.0


# --- end to end: what each CV fold actually fits ---

FITTED_WITH: list[int] = []


class _RecordingEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, marker: int = 1):
        self.marker = marker

    def fit(self, X, y):
        FITTED_WITH.append(self.marker)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _container():
    n = 24
    return DataContainer(
        xr.DataArray(
            np.random.default_rng(0).normal(size=(n, 3)),
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(n),
                "feature": np.arange(3),
                "stimulus": ("trial", np.array(["a", "b"] * (n // 2))),
            },
        )
    )


@pytest.mark.parametrize("write", ["set_params", "setattr"])
def test_every_fold_fits_the_value_set_after_construction(write):
    FITTED_WITH.clear()
    clf = SKLearnPredictor(_RecordingEstimator, sample_dim="trial", target_coord="stimulus", marker=1)
    pipeline = Pipeline(name="p", steps=[("clf", clf)])
    if write == "set_params":
        pipeline.set_params(clf__marker=99)
    else:
        clf.marker = 99
    validator = KFoldValidator(n_splits=3, random_state=0, use_stateful_fit_cache=False)
    validator.set_pipeline(pipeline)
    validator.cross_validate(_container(), verbose=False)
    assert FITTED_WITH and set(FITTED_WITH) == {99}


@pytest.mark.parametrize("target_coord", ["y", ["a", "b"]])
def test_tuner_seed_reaches_the_model_through_clones(target_coord):
    predictor = SKLearnPredictor(
        Ridge, sample_dim="trial", target_coord=target_coord, is_classifier=False, random_state=0
    )
    Tuner._inject_random_seeds_into_pipelines(SimpleNamespace(pipelines_to_tune={"p": predictor}, random_seed=123))
    estimator = predictor.clone().clone().estimator
    estimator = getattr(estimator, "estimator", estimator)
    assert estimator.random_state == 123


def test_writes_through_an_automatically_added_wrapper_are_refused():
    predictor = SKLearnPredictor(Ridge, sample_dim="trial", target_coord=["a", "b"], is_classifier=False, alpha=2.0)
    assert predictor.multi_output
    with pytest.raises(ValueError, match="automatic MultiOutput wrapper"):
        predictor.set_params(estimator__n_jobs=3)
    assert predictor.estimator.n_jobs is None


def test_callable_factories_clone_with_their_live_params():

    factory_built = SKLearnPredictor(
        lambda: Ridge(alpha=2.0), sample_dim="trial", target_coord="a", is_classifier=False
    )
    factory_built.set_params(alpha=5.0)
    assert factory_built.clone().estimator.alpha == 5.0

    wrapped = SKLearnPredictor(
        partial(MultiOutputRegressor, estimator=Ridge()),
        sample_dim="trial",
        target_coord=["a", "b"],
        is_classifier=False,
    )
    wrapped.set_params(estimator__alpha=3.0)
    assert wrapped.clone().estimator.estimator.alpha == 3.0


def test_factories_that_swallow_kwargs_or_get_wrapped_keep_live_params():
    swallowing = SKLearnPredictor(lambda **kw: Ridge(), sample_dim="trial", target_coord="a", is_classifier=False)
    swallowing.set_params(alpha=42.0)
    assert swallowing.clone().estimator.alpha == 42.0

    wrapped = SKLearnPredictor(
        lambda: Ridge(), sample_dim="trial", target_coord=["a", "b"], is_classifier=False, multi_output=True
    )
    wrapped.set_params(alpha=3.0)
    assert wrapped.clone().estimator.estimator.alpha == 3.0


def test_auto_wrapped_meta_estimator_keeps_its_own_estimator_param():
    from sklearn.ensemble import BaggingRegressor

    predictor = SKLearnPredictor(
        BaggingRegressor,
        sample_dim="trial",
        target_coord=["a", "b"],
        is_classifier=False,
        estimator=Ridge(alpha=3.0),
        n_estimators=4,
    )

    for candidate in (predictor, predictor.clone()):
        assert isinstance(candidate.estimator.estimator, BaggingRegressor)
        assert candidate.estimator.estimator.estimator.alpha == 3.0
