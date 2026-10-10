"""Tests for ensemble predictors."""

import numpy as np
import xarray as xr

from xdflow.composite.ensemble import EnsemblePredictor
from xdflow.core.base import Predictor
from xdflow.core.data_container import DataContainer


class FixedProbaPredictor(Predictor):
    """Predictor that returns fixed probabilities for testing."""

    is_stateful: bool = True
    input_dims = ()
    output_dims = ()

    def __init__(self, sample_dim: str, target_coord: str, proba_values, **kwargs):
        super().__init__(sample_dim=sample_dim, target_coord=target_coord, is_classifier=True, **kwargs)
        self.proba_values = np.asarray(proba_values, dtype=float)

    def _fit(self, container: DataContainer, **kwargs) -> "FixedProbaPredictor":
        return self

    def _predict(self, data: xr.DataArray, **kwargs) -> np.ndarray:
        n_samples = data.sizes[self.sample_dim]
        return np.zeros(n_samples, dtype=int)

    def _predict_proba(self, data: xr.DataArray, **kwargs):
        n_samples = data.sizes[self.sample_dim]
        probs = np.tile(self.proba_values, (n_samples, 1))
        class_labels = np.arange(self.proba_values.shape[0])
        return probs, class_labels


def _make_container(n_trials: int = 6) -> DataContainer:
    data = xr.DataArray(
        np.random.randn(n_trials, 4),
        dims=("trial", "feature"),
        coords={"trial": np.arange(n_trials), "feature": np.arange(4)},
    )
    labels = np.array(["a", "b"] * (n_trials // 2) + (["a"] if n_trials % 2 else []))
    data.coords["label"] = ("trial", labels[:n_trials])
    return DataContainer(data)


def test_ensemble_uncertainty_outputs():
    """Ensemble should return probability + uncertainty containers with expected shapes."""
    container = _make_container()

    member_a = FixedProbaPredictor(sample_dim="trial", target_coord="label", proba_values=[0.7, 0.3])
    member_b = FixedProbaPredictor(sample_dim="trial", target_coord="label", proba_values=[0.2, 0.8])

    ensemble = EnsemblePredictor(
        members=[("a", member_a), ("b", member_b)],
        sample_dim="trial",
        target_coord="label",
        weights=[0.5, 0.5],
        normalize_weights=True,
    )

    ensemble.fit(container)

    proba, aleatoric, epistemic = ensemble.predict_proba_with_uncertainty_components(container)
    proba_std, std_container = ensemble.predict_proba_with_std(container)

    assert proba.data.dims == ("trial", "class")
    assert aleatoric.data.dims == ("trial",)
    assert epistemic.data.dims == ("trial",)
    assert std_container.data.dims == ("trial", "class")

    assert np.all(np.isfinite(proba.data.values))
    assert np.all(np.isfinite(aleatoric.data.values))
    assert np.all(np.isfinite(epistemic.data.values))
    assert np.all(np.isfinite(std_container.data.values))


def _sklearn_container(n: int = 40, labels=("a", "b")) -> DataContainer:
    rng = np.random.default_rng(0)
    y = np.array([labels[i % len(labels)] for i in range(n)])
    X = rng.normal(size=(n, 3))
    X[y == labels[1], 0] += 3.0
    return DataContainer(
        xr.DataArray(
            X,
            dims=("trial", "feature"),
            coords={"trial": np.arange(n), "feature": np.arange(3), "y": ("trial", y), "v": ("trial", X[:, 0])},
        )
    )


def test_regression_ensemble_averages_members():
    from sklearn.dummy import DummyRegressor

    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    def member(value):
        return SKLearnPredictor(
            DummyRegressor,
            sample_dim="trial",
            target_coord="v",
            is_classifier=False,
            strategy="constant",
            constant=value,
        )

    ensemble = EnsemblePredictor(
        [("m10", member(10.0)), ("m0", member(0.0)), ("m3", member(3.0))], sample_dim="trial", target_coord="v"
    )
    data = _sklearn_container()
    ensemble.fit(data)

    np.testing.assert_allclose(ensemble.predict(data).data.values, 13.0 / 3)


def test_score_based_weights_do_not_depend_on_label_type():
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    weights = []
    for labels in [("a", "b"), (1, 2)]:
        data = _sklearn_container(labels=labels)
        ensemble = EnsemblePredictor(
            [
                ("good", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="y")),
                (
                    "bad",
                    SKLearnPredictor(DummyClassifier, sample_dim="trial", target_coord="y", strategy="most_frequent"),
                ),
            ],
            sample_dim="trial",
            target_coord="y",
            weighting_strategy="score_based",
        )
        ensemble.fit(data)
        weights.append(ensemble.weights)

    np.testing.assert_allclose(weights[0], weights[1])
    assert weights[0][0] > weights[0][1]
