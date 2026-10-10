"""Tests for SKLearn wrapper transforms."""

import numpy as np
import pytest
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from xdflow.core.data_container import DataContainer
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer


class TestSKLearnTransformer:
    """Test SKLearnTransformer functionality."""

    def test_pca_transformer(self, data_container_factory):
        """Test SKLearnTransformer with PCA."""
        # Need 2D data (samples × features)
        container = data_container_factory(n_trials=50, n_channels=10, n_time=1)
        # Average over time to get 2D
        container_2d = DataContainer(container.data.mean(dim="time"))

        transform = SKLearnTransformer(estimator_cls=PCA, n_components=5, sample_dim="trial")

        result = transform.fit_transform(container_2d)

        # Should reduce to 5 components
        assert result.data.shape == (50, 5)
        assert "trial" in result.data.dims

    def test_pca_separate_fit_transform(self, data_container_factory):
        """Test separate fit and transform calls."""
        train_container = data_container_factory(n_trials=50, n_channels=10, n_time=1, seed=42)
        test_container = data_container_factory(n_trials=20, n_channels=10, n_time=1, seed=43)

        train_2d = DataContainer(train_container.data.mean(dim="time"))
        test_2d = DataContainer(test_container.data.mean(dim="time"))

        transform = SKLearnTransformer(estimator_cls=PCA, n_components=5, sample_dim="trial")

        # Fit on train
        transform.fit(train_2d)

        # Transform both
        train_result = transform.transform(train_2d)
        test_result = transform.transform(test_2d)

        assert train_result.data.shape == (50, 5)
        assert test_result.data.shape == (20, 5)

    def test_is_stateful(self):
        """Test that SKLearnTransformer is stateful."""
        transform = SKLearnTransformer(estimator_cls=PCA, n_components=5, sample_dim="trial")
        assert transform.is_stateful


class TestSKLearnPredictor:
    """Test SKLearnPredictor functionality."""

    def test_logistic_regression_classifier(self, data_container_factory):
        """Test SKLearnPredictor with logistic regression."""
        # Create labeled data
        container = data_container_factory(n_trials=60, n_channels=10, n_time=1)
        container_2d = DataContainer(container.data.mean(dim="time"))

        # Add class labels (3 classes, 20 samples each)
        labels = np.repeat(["class_a", "class_b", "class_c"], 20)
        container_2d.data.coords["label"] = ("trial", labels)

        predictor = SKLearnPredictor(
            estimator_cls=LogisticRegression, sample_dim="trial", target_coord="label", max_iter=200
        )

        result = predictor.fit_transform(container_2d)

        # Should return predictions
        assert "label" in result.data.coords
        assert result.data.sizes["trial"] == 60
        # Predictions should be one of the three classes
        unique_preds = np.unique(result.data.coords["label"].values)
        assert len(unique_preds) <= 3

    def test_decision_tree_classifier(self, data_container_factory):
        """Test SKLearnPredictor with decision tree."""
        container = data_container_factory(n_trials=60, n_channels=10, n_time=1)
        container_2d = DataContainer(container.data.mean(dim="time"))

        # Add binary labels
        labels = np.repeat([0, 1], 30)
        container_2d.data.coords["target"] = ("trial", labels)

        predictor = SKLearnPredictor(
            estimator_cls=DecisionTreeClassifier, sample_dim="trial", target_coord="target", max_depth=3
        )

        result = predictor.fit_transform(container_2d)

        # Should return predictions
        assert "target" in result.data.coords
        assert result.data.sizes["trial"] == 60
        # Predictions should be 0 or 1
        unique_preds = np.unique(result.data.coords["target"].values)
        assert set(unique_preds).issubset({0, 1})

    def test_predict_method(self, data_container_factory):
        """Test that predict method works on separate train/test data."""
        train_container = data_container_factory(n_trials=60, n_channels=10, n_time=1, seed=42)
        test_container = data_container_factory(n_trials=20, n_channels=10, n_time=1, seed=43)

        train_2d = DataContainer(train_container.data.mean(dim="time"))
        test_2d = DataContainer(test_container.data.mean(dim="time"))

        # Add labels to train
        train_labels = np.repeat([0, 1, 2], 20)
        train_2d.data.coords["label"] = ("trial", train_labels)

        # Add dummy labels to test (required by the API, but won't be used)
        # In real use, you'd provide the true labels for evaluation
        test_labels = np.repeat([0, 1, 2], 20)[:20]  # Just placeholder labels
        test_2d.data.coords["label"] = ("trial", test_labels)

        predictor = SKLearnPredictor(
            estimator_cls=LogisticRegression, sample_dim="trial", target_coord="label", max_iter=200
        )

        # Fit on train
        predictor.fit(train_2d)

        # Predict on test
        predictions = predictor.predict(test_2d)

        # Should return predictions in 1D format
        assert predictions.data.sizes["trial"] == 20
        assert predictions.data.ndim == 1
        # Predictions should be one of the three classes
        unique_preds = np.unique(predictions.data.values)
        assert len(unique_preds) <= 3
        assert set(unique_preds).issubset({0, 1, 2})

    def test_is_stateful(self):
        """Test that SKLearnPredictor is stateful."""
        predictor = SKLearnPredictor(estimator_cls=LogisticRegression, sample_dim="trial", target_coord="label")
        assert predictor.is_stateful

    def test_multilabel_classifier_predicts_target_matrix(self):
        rng = np.random.default_rng(123)
        n_trials = 80
        target_a = np.tile([0, 1], n_trials // 2)
        target_b = np.tile([0, 0, 1, 1], n_trials // 4)
        X = np.column_stack(
            [
                target_a + rng.normal(scale=0.1, size=n_trials),
                target_b + rng.normal(scale=0.1, size=n_trials),
                rng.normal(size=n_trials),
            ]
        )
        container = DataContainer(
            xr.DataArray(
                X,
                dims=("trial", "feature"),
                coords={
                    "trial": np.arange(n_trials),
                    "feature": np.arange(X.shape[1]),
                    "target__a": ("trial", target_a),
                    "target__b": ("trial", target_b),
                },
            )
        )

        predictor = SKLearnPredictor(
            estimator_cls=LogisticRegression,
            sample_dim="trial",
            target_coord=["target__a", "target__b"],
            is_classifier=True,
            is_multilabel=True,
            max_iter=200,
        )

        predictor.fit(container)
        predictions = predictor.predict(container)
        probabilities = predictor.predict_proba(container)

        assert predictions.data.dims == ("trial", "target")
        assert probabilities.data.dims == ("trial", "target")
        assert list(predictions.data.coords["target"].values) == ["target__a", "target__b"]
        assert predictor.encoder is None
        assert predictor.get_labels() == ["target__a", "target__b"]


class TestSKLearnRegression:
    """Test SKLearnPredictor with regression tasks."""

    def test_ridge_regression(self, data_container_factory):
        """Test SKLearnPredictor with Ridge regression."""
        container = data_container_factory(n_trials=50, n_channels=10, n_time=1)
        container_2d = DataContainer(container.data.mean(dim="time"))

        # Add continuous target values
        targets = np.random.randn(50)
        container_2d.data.coords["target"] = ("trial", targets)

        predictor = SKLearnPredictor(estimator_cls=Ridge, sample_dim="trial", target_coord="target", alpha=1.0)

        result = predictor.fit_transform(container_2d)

        # Should return predictions
        assert "target" in result.data.coords
        assert result.data.sizes["trial"] == 50
        # Predictions should be continuous values
        assert result.data.coords["target"].dtype in [np.float32, np.float64]

    def test_multi_target_list_regression(self, data_container_factory):
        """Test multi-target regression with explicit target list."""
        container = data_container_factory(n_trials=40, n_channels=6, n_time=1)
        container_2d = DataContainer(container.data.mean(dim="time"))

        target_a = np.linspace(0, 1, 40)
        target_b = np.linspace(1, 2, 40)
        container_2d.data.coords["target_a"] = ("trial", target_a)
        container_2d.data.coords["target_b"] = ("trial", target_b)

        predictor = SKLearnPredictor(
            estimator_cls=Ridge,
            sample_dim="trial",
            target_coord=["target_a", "target_b"],
            is_classifier=False,
            alpha=1.0,
        )

        result = predictor.fit_transform(container_2d)

        assert result.data.dims == ("trial", "target")
        assert list(result.data.coords["target"].values) == ["target_a", "target_b"]
        assert predictor.is_multi_target

    def test_multi_target_pattern_regression(self, data_container_factory):
        """Test multi-target regression with target pattern."""
        container = data_container_factory(n_trials=30, n_channels=5, n_time=1)
        container_2d = DataContainer(container.data.mean(dim="time"))

        container_2d.data.coords["target_x"] = ("trial", np.linspace(0, 1, 30))
        container_2d.data.coords["target_y"] = ("trial", np.linspace(1, 2, 30))

        predictor = SKLearnPredictor(
            estimator_cls=Ridge,
            sample_dim="trial",
            target_coord="target_*",
            is_classifier=False,
            alpha=1.0,
        )

        result = predictor.fit_transform(container_2d)

        assert result.data.dims == ("trial", "target")
        assert set(result.data.coords["target"].values) == {"target_x", "target_y"}
        assert predictor.is_multi_target


def test_fitted_estimators_see_features_by_label_not_position():
    from sklearn.linear_model import LogisticRegression

    from xdflow.composite import Pipeline
    from xdflow.transforms.basic_transforms import FlattenTransform
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    rng = np.random.default_rng(0)
    labels = np.array(["a", "b"] * 30)
    values = rng.normal(size=(60, 4, 5))
    values[labels == "b", 0] += 1.0
    data = DataContainer(
        xr.DataArray(
            values,
            dims=("trial", "channel", "time"),
            coords={
                "trial": np.arange(60),
                "channel": ["c0", "c1", "c2", "c3"],
                "time": np.arange(5),
                "y": ("trial", labels),
            },
        )
    )
    pipeline = Pipeline(
        "flat_lr",
        [
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            ("clf", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="y", proba=True)),
        ],
    ).fit(data)

    reordered = DataContainer(data.data.isel(channel=[3, 1, 0, 2]))
    np.testing.assert_allclose(pipeline.predict_proba(reordered).data.values, pipeline.predict_proba(data).data.values)
    with pytest.raises(Exception, match="missing"):
        pipeline.predict(DataContainer(data.data.isel(channel=[0, 1, 2])))
