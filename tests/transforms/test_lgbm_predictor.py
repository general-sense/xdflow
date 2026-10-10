"""Tests for LGBMPredictor with early stopping support."""

import numpy as np
import pytest
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.transforms.lgbm_predictor import LGBMPredictor

lightgbm = pytest.importorskip("lightgbm")
LGBMClassifier = lightgbm.LGBMClassifier
LGBMRegressor = lightgbm.LGBMRegressor


@pytest.fixture
def classification_data():
    """Create synthetic classification data for testing."""
    np.random.seed(42)
    n_trials = 200
    n_features = 20
    n_classes = 4

    X = np.random.randn(n_trials, n_features)
    y = np.random.choice([f"class_{i}" for i in range(n_classes)], size=n_trials)

    data = xr.DataArray(
        X,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "label": ("trial", y),
        },
    )

    return DataContainer(data)


@pytest.fixture
def regression_data():
    """Create synthetic regression data for testing."""
    np.random.seed(42)
    n_trials = 200
    n_features = 20

    X = np.random.randn(n_trials, n_features)
    y = 2.5 * X[:, 0] - 1.3 * X[:, 1] + 0.8 * X[:, 2] + np.random.randn(n_trials) * 0.5

    data = xr.DataArray(
        X,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "target": ("trial", y),
        },
    )

    return DataContainer(data)


class TestLGBMPredictorInitialization:
    """Test initialization and parameter validation."""

    def test_basic_initialization_classifier(self):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=50,
            validation_size=0.2,
            validation_seed=42,
            n_estimators=100,
        )

        assert predictor.early_stopping_rounds == 50
        assert predictor.validation_size == 0.2
        assert predictor.validation_seed == 42
        assert predictor.is_classifier is True

    def test_basic_initialization_regressor(self):
        predictor = LGBMPredictor(
            LGBMRegressor,
            sample_dim="trial",
            target_coord="target",
            early_stopping_rounds=100,
            validation_size=0.15,
            eval_metric="rmse",
            n_estimators=500,
            is_classifier=False,
        )

        assert predictor.early_stopping_rounds == 100
        assert predictor.validation_size == 0.15
        assert predictor.eval_metric == "rmse"
        assert predictor.is_classifier is False

    def test_without_early_stopping(self):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            n_estimators=50,
        )

        assert predictor.early_stopping_rounds is None
        assert predictor.validation_size == 0.2
        assert predictor.validation_seed is None

    def test_invalid_early_stopping_rounds(self):
        with pytest.raises(ValueError, match="early_stopping_rounds must be a positive integer"):
            LGBMPredictor(
                LGBMClassifier,
                sample_dim="trial",
                target_coord="label",
                early_stopping_rounds=-10,
            )

        with pytest.raises(ValueError, match="early_stopping_rounds must be a positive integer"):
            LGBMPredictor(
                LGBMClassifier,
                sample_dim="trial",
                target_coord="label",
                early_stopping_rounds=0,
            )

    def test_invalid_validation_size(self):
        with pytest.raises(ValueError, match="validation_size must be between 0.0 and 1.0"):
            LGBMPredictor(
                LGBMClassifier,
                sample_dim="trial",
                target_coord="label",
                early_stopping_rounds=50,
                validation_size=0.0,
            )

        with pytest.raises(ValueError, match="validation_size must be between 0.0 and 1.0"):
            LGBMPredictor(
                LGBMClassifier,
                sample_dim="trial",
                target_coord="label",
                early_stopping_rounds=50,
                validation_size=1.0,
            )

        with pytest.raises(ValueError, match="validation_size must be between 0.0 and 1.0"):
            LGBMPredictor(
                LGBMClassifier,
                sample_dim="trial",
                target_coord="label",
                early_stopping_rounds=50,
                validation_size=1.5,
            )


class TestLGBMPredictorFitting:
    """Test fitting with early stopping."""

    def test_fit_classifier_with_early_stopping(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=10,
            validation_size=0.2,
            validation_seed=42,
            n_estimators=1000,
            learning_rate=0.1,
        )

        predictor.fit(classification_data)

        assert hasattr(predictor, "best_iteration_")
        assert predictor.best_iteration_ < 1000

    def test_fit_regressor_with_early_stopping(self, regression_data):
        predictor = LGBMPredictor(
            LGBMRegressor,
            sample_dim="trial",
            target_coord="target",
            early_stopping_rounds=20,
            validation_size=0.25,
            validation_seed=123,
            eval_metric="rmse",
            n_estimators=1000,
            learning_rate=0.05,
            is_classifier=False,
        )

        predictor.fit(regression_data)

        assert hasattr(predictor, "best_iteration_")
        assert predictor.best_iteration_ < 1000

    def test_fit_without_early_stopping(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            n_estimators=50,
        )

        predictor.fit(classification_data)
        predictions = predictor.predict(classification_data)

        assert predictions.data.ndim == 1
        assert len(predictions.data) == len(classification_data.data)

    def test_predict_after_early_stopping(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=10,
            validation_size=0.2,
            n_estimators=500,
        )

        predictor.fit(classification_data)
        predictions = predictor.predict(classification_data)

        assert predictions.data.ndim == 1
        assert len(predictions.data) == len(classification_data.data)
        assert "trial" in predictions.data.dims


class TestLGBMPredictorCloning:
    """Test cloning preserves early stopping parameters."""

    def test_clone_preserves_early_stopping_params(self):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=75,
            validation_size=0.18,
            validation_seed=999,
            eval_metric="multi_logloss",
            verbose_eval=10,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
        )

        cloned = predictor.clone()

        assert cloned.early_stopping_rounds == 75
        assert cloned.validation_size == 0.18
        assert cloned.validation_seed == 999
        assert cloned.eval_metric == "multi_logloss"
        assert cloned.verbose_eval == 10

        assert cloned.estimator.n_estimators == 200
        assert cloned.estimator.learning_rate == 0.05
        assert cloned.estimator.max_depth == 5

    def test_clone_is_unfitted(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=10,
            n_estimators=100,
        )

        predictor.fit(classification_data)
        assert hasattr(predictor, "best_iteration_")

        cloned = predictor.clone()
        assert not hasattr(cloned, "best_iteration_")

    def test_get_params_includes_early_stopping(self):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=50,
            validation_size=0.2,
            validation_seed=42,
        )

        params = predictor.get_params(deep=False)

        assert "early_stopping_rounds" in params
        assert "validation_size" in params
        assert "validation_seed" in params
        assert "eval_metric" in params
        assert "verbose_eval" in params


class TestLGBMPredictorVerboseEval:
    """Test verbose evaluation logging."""

    def test_verbose_eval_int(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=10,
            verbose_eval=50,
            n_estimators=200,
        )

        predictor.fit(classification_data)
        assert hasattr(predictor, "best_iteration_")

    def test_verbose_eval_bool(self, classification_data):
        predictor = LGBMPredictor(
            LGBMClassifier,
            sample_dim="trial",
            target_coord="label",
            early_stopping_rounds=10,
            verbose_eval=True,
            n_estimators=50,
        )

        predictor.fit(classification_data)
        assert hasattr(predictor, "best_iteration_")


def test_early_stopping_with_multiple_regression_targets():
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    data = DataContainer(
        xr.DataArray(
            X,
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(200),
                "feature": np.arange(4),
                "a": ("trial", 2 * X[:, 0]),
                "b": ("trial", X[:, 1] - X[:, 2]),
            },
        )
    )
    predictor = LGBMPredictor(
        LGBMRegressor,
        sample_dim="trial",
        target_coord=["a", "b"],
        is_classifier=False,
        early_stopping_rounds=5,
        validation_seed=0,
        verbose=-1,
    )
    predictor.fit(data)

    assert predictor.predict(data).data.shape == (200, 2)
    assert len(predictor.best_iteration_) == 2


def test_early_stopping_with_a_target_pattern():
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    data = DataContainer(
        xr.DataArray(
            X,
            dims=("trial", "feature"),
            coords={
                "trial": np.arange(60),
                "feature": np.arange(3),
                "out_a": ("trial", X[:, 0]),
                "out_b": ("trial", X[:, 1] ** 2),
            },
        )
    )
    predictor = LGBMPredictor(
        LGBMRegressor, sample_dim="trial", target_coord="out_*", early_stopping_rounds=3, validation_seed=0, verbose=-1
    )
    predictor.fit(data)

    assert predictor.predict(data).data.shape == (60, 2)
