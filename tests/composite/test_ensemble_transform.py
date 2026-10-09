"""
Tests for EnsemblePredictor and UncertaintyEnsemble.
"""

import numpy as np
import pytest
import xarray as xr
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xdflow.composite import Pipeline
from xdflow.composite.base import TransformStep
from xdflow.composite.ensemble import EnsembleMember, EnsemblePredictor
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.basic_transforms import IdentityTransform


class MockTransform(Transform):
    """Simple mock transform for testing."""

    is_stateful = False
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "features")

    def __init__(self, name, multiplier=1.0, add_constant=0.0):
        super().__init__()
        self.name = name
        self.multiplier = multiplier
        self.add_constant = add_constant
        self.fitted = False

    def _fit(self, container, **kwargs):
        self.fitted = True
        return self

    def _transform(self, container, **kwargs):
        data = container.data * self.multiplier + self.add_constant
        flattened = data.stack(features=("channel", "time"))
        result_data = flattened.transpose("trial", "features")
        return DataContainer(result_data)

    def get_expected_output_dims(self, input_dims):
        return self.output_dims

    def get_params(self, deep=False):
        return {"name": self.name, "multiplier": self.multiplier, "add_constant": self.add_constant}


class MockStatefulTransform(MockTransform):
    """Mock stateful transform for testing."""

    is_stateful = True

    def __init__(self, name, multiplier=1.0, add_constant=0.0):
        super().__init__(name, multiplier, add_constant)
        self.mean_value = None

    def _fit(self, container, **kwargs):
        self.fitted = True
        self.mean_value = float(container.data.mean())
        return self

    def _transform(self, container, **kwargs):
        if self.mean_value is None:
            raise ValueError("Transform must be fitted before transform")
        centered_data = container.data - self.mean_value
        flattened = (
            (centered_data * self.multiplier + self.add_constant)
            .stack(features=("channel", "time"))
            .transpose("trial", "features")
        )
        return DataContainer(flattened)


class MockPredictor(Predictor):
    """Mock predictor for testing."""

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial",)

    def __init__(self, name="mock_predictor", prediction_bias=0.0, is_classifier: bool = True):
        self.name = name
        self.prediction_bias = prediction_bias
        self.fitted = False
        self.feature_mean = None
        encoder = LabelEncoder() if is_classifier else None
        super().__init__(
            sample_dim="trial",
            target_coord="label",
            is_classifier=is_classifier,
            encoder=encoder,
        )

    def _fit(self, container, **kwargs):
        self.fitted = True
        self.feature_mean = float(container.data.mean())
        return self

    def _predict(self, data, **kwargs):
        if self.feature_mean is None:
            raise ValueError("Predictor must be fitted before predict")
        predictions = (data.mean(dim=["channel", "time"]) > self.feature_mean + self.prediction_bias).astype(int)
        if self.encoder is not None and hasattr(self.encoder, "classes_") and len(self.encoder.classes_) > 0:
            predictions = predictions % len(self.encoder.classes_)
        return predictions.values

    def _predict_proba(self, data, **kwargs):
        if self.feature_mean is None:
            raise ValueError("Predictor must be fitted before predict_proba")
        mean_vals = data.mean(dim=["channel", "time"]).values
        threshold = self.feature_mean + self.prediction_bias
        distances = mean_vals - threshold

        classes = self.encoder.classes_
        n_classes = len(classes)
        probabilities = np.zeros((len(mean_vals), n_classes))

        for i, dist in enumerate(distances):
            if dist > 0:
                probabilities[i, 1 % n_classes] = 0.7
                probabilities[i, 0] = 0.3
            else:
                probabilities[i, 0] = 0.8
                probabilities[i, 1 % n_classes] = 0.2

        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        class_indices = np.arange(len(classes))
        return probabilities, class_indices

    def get_expected_output_dims(self, input_dims):
        return self.output_dims

    def get_params(self, deep=False):
        return {"name": self.name, "prediction_bias": self.prediction_bias, "is_classifier": self.is_classifier}


class MarkerTransform(Transform):
    """Transform that annotates data to prove inner pipelines ran."""

    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")

    def _fit(self, container, **kwargs):
        return self

    def _transform(self, container, **kwargs):
        annotated = container.data.assign_attrs(inner_marker=True)
        return DataContainer(annotated)

    def get_expected_output_dims(self, input_dims):
        return self.output_dims


class MarkerAwarePredictor(MockPredictor):
    """Predictor that demands upstream marker annotation."""

    def _predict(self, data, **kwargs):
        if not data.attrs.get("inner_marker"):
            raise RuntimeError("inner_marker attribute missing; inner pipeline likely skipped.")
        return super()._predict(data, **kwargs)

    def _predict_proba(self, data, **kwargs):
        if not data.attrs.get("inner_marker"):
            raise RuntimeError("inner_marker attribute missing; inner pipeline likely skipped.")
        return super()._predict_proba(data, **kwargs)


@pytest.fixture
def sample_container():
    np.random.seed(42)
    data = xr.DataArray(
        np.random.rand(10, 5, 8),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(10),
            "channel": np.arange(5),
            "time": np.arange(8),
            "label": ("trial", ["class_a"] * 5 + ["class_b"] * 5),
            "session": ("trial", ["session_1"] * 10),
        },
        attrs={"data_history": []},
    )
    return DataContainer(data)


@pytest.fixture
def sample_predictor_container():
    np.random.seed(42)
    data = xr.DataArray(
        np.random.rand(20, 3, 6),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(20),
            "channel": np.arange(3),
            "time": np.arange(6),
            "label": ("trial", ["class_a"] * 10 + ["class_b"] * 10),
            "session": ("trial", ["session_1"] * 20),
        },
        attrs={"data_history": []},
    )
    return DataContainer(data)


class TestEnsemblePredictorBasics:
    def test_initialization_with_tuples(self):
        predictor1 = MockPredictor("predictor1", prediction_bias=0.0)
        predictor2 = MockPredictor("predictor2", prediction_bias=0.1)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="uniform",
        )

        assert len(ensemble.members) == 2
        assert ensemble.members[0].name == "p1"
        assert ensemble.members[1].name == "p2"
        assert ensemble.weights == [0.5, 0.5]

    def test_initialization_with_ensemble_members(self):
        predictor1 = MockPredictor("predictor1")
        predictor2 = MockPredictor("predictor2")

        member1 = EnsembleMember("m1", predictor1, weight=0.3)
        member2 = EnsembleMember("m2", predictor2, weight=0.7)

        ensemble = EnsemblePredictor(
            members=[member1, member2],
            sample_dim="trial",
            target_coord="label",
            normalize_weights=False,
        )

        assert ensemble.weights == [0.3, 0.7]

    def test_initialization_with_transform_steps(self):
        predictor1 = MockPredictor("predictor1")
        predictor2 = MockPredictor("predictor2")

        step1 = TransformStep("step1", predictor1)
        step2 = TransformStep("step2", predictor2)

        ensemble = EnsemblePredictor(members=[step1, step2], sample_dim="trial", target_coord="label")

        assert len(ensemble.members) == 2
        assert ensemble.members[0].name == "step1"
        assert ensemble.members[1].name == "step2"

    def test_initialization_with_custom_weights(self):
        predictor1 = MockPredictor("predictor1")
        predictor2 = MockPredictor("predictor2")

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.3, 0.7],
        )

        assert ensemble.weights == [0.3, 0.7]

    def test_initialization_errors(self):
        predictor1 = MockPredictor("predictor1")

        with pytest.raises(ValueError, match="At least one ensemble member"):
            EnsemblePredictor(members=[], sample_dim="trial", target_coord="label")

        with pytest.raises(ValueError, match="Number of weights .* must match"):
            EnsemblePredictor(
                members=[("p1", predictor1)],
                sample_dim="trial",
                target_coord="label",
                weights=[0.3, 0.7],
            )

        with pytest.raises(ValueError, match="Invalid member type"):
            EnsemblePredictor(members=["invalid_member"], sample_dim="trial", target_coord="label")

    def test_duplicate_names_error(self):
        predictor1 = MockPredictor("predictor1")
        predictor2 = MockPredictor("predictor2")

        with pytest.raises(ValueError, match="Ensemble member names must be unique"):
            EnsemblePredictor(
                members=[("same_name", predictor1), ("same_name", predictor2)],
                sample_dim="trial",
                target_coord="label",
            )


class TestEnsemblePredictorFunctionality:
    def test_predict_basic(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.1)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.6, 0.4],
        )

        ensemble.fit(sample_predictor_container)
        result = ensemble.predict(sample_predictor_container)

        assert isinstance(result, DataContainer)
        assert result.data.dims == ("trial",)
        assert result.data.sizes["trial"] == 20

    def test_predict_proba_basic(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.1)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.6, 0.4],
            normalize_outputs=False,
        )

        ensemble.fit(sample_predictor_container)
        result = ensemble.predict_proba(sample_predictor_container)

        assert isinstance(result, DataContainer)
        assert len(result.data.dims) == 2
        assert result.data.sizes["trial"] == 20

    def test_uncertainty_components_basic(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.1)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.6, 0.4],
        )

        ensemble.fit(sample_predictor_container)
        proba_dc, aleatoric_dc, epistemic_dc = ensemble.predict_proba_with_uncertainty_components(
            sample_predictor_container
        )

        assert isinstance(proba_dc, DataContainer)
        assert len(proba_dc.data.dims) == 2
        assert proba_dc.data.sizes["trial"] == sample_predictor_container.data.sizes["trial"]

        for dc in (aleatoric_dc, epistemic_dc):
            assert isinstance(dc, DataContainer)
            assert dc.data.dims == ("trial",)
            assert dc.data.sizes["trial"] == sample_predictor_container.data.sizes["trial"]
            assert np.all(np.isfinite(dc.data.values))
            assert np.all(dc.data.values >= -1e-8)

    def test_predict_proba_with_std_basic(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.05)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.7, 0.3],
        )

        ensemble.fit(sample_predictor_container)
        proba_dc, std_dc = ensemble.predict_proba_with_std(sample_predictor_container)

        assert isinstance(proba_dc, DataContainer)
        assert isinstance(std_dc, DataContainer)
        assert proba_dc.data.dims == ("trial", "class")
        assert std_dc.data.dims == ("trial", "class")
        assert proba_dc.data.sizes == std_dc.data.sizes
        assert np.all(std_dc.data.values >= 0)
        assert std_dc.data.name == "proba_std"

    def test_predict_proba_with_std_matches_weighted_stats(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=-0.05)
        weights = [0.2, 0.8]

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=weights,
        )

        ensemble.fit(sample_predictor_container)
        proba_dc, std_dc = ensemble.predict_proba_with_std(sample_predictor_container)

        member_probs = []
        for member in ensemble.members:
            member_proba_dc = member.predictor.predict_proba(sample_predictor_container)
            member_probs.append(member_proba_dc.data.transpose("trial", "class").values)
        probs = np.stack(member_probs, axis=0)

        weights_arr = np.asarray(ensemble.weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        unweighted_mean = probs.mean(axis=0)
        manual_var = np.mean((probs - unweighted_mean[np.newaxis, :, :]) ** 2, axis=0)
        manual_std = np.sqrt(manual_var)

        np.testing.assert_allclose(
            std_dc.data.transpose("trial", "class").values,
            manual_std,
            atol=1e-8,
        )

        _, stderr_dc = ensemble.predict_proba_with_std(sample_predictor_container, return_stderr=True)
        manual_stderr = manual_std / np.sqrt(probs.shape[0])
        np.testing.assert_allclose(
            stderr_dc.data.transpose("trial", "class").values,
            manual_stderr,
            atol=1e-8,
        )
        assert stderr_dc.data.name == "proba_std_error"


def test_parallel_execution(sample_predictor_container):
    predictor1 = MockPredictor("p1", prediction_bias=0.0)
    predictor2 = MockPredictor("p2", prediction_bias=0.1)

    ensemble = EnsemblePredictor(
        members=[("p1", predictor1), ("p2", predictor2)],
        sample_dim="trial",
        target_coord="label",
        n_jobs=2,
    )

    ensemble.fit(sample_predictor_container)
    result = ensemble.predict(sample_predictor_container)

    assert isinstance(result, DataContainer)
    assert result.data.sizes["trial"] == 20


class TestEnsembleWeightingStrategies:
    def test_uniform_weighting(self, sample_predictor_container):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")
        predictor3 = MockPredictor("p3")

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2), ("p3", predictor3)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="uniform",
        )

        expected_weights = [1 / 3, 1 / 3, 1 / 3]
        assert np.allclose(ensemble.weights, expected_weights)

    def test_score_based_weighting(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.5)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="score_based",
            scoring_func=accuracy_score,
            calibration_container=sample_predictor_container,
        )

        ensemble.fit(sample_predictor_container)
        weights = ensemble.weights
        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0)

    def test_custom_weighting(self, sample_predictor_container):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")

        custom_weights = [0.8, 0.2]
        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=custom_weights,
            weighting_strategy="custom",
        )

        assert ensemble.weights == custom_weights

    def test_score_transform_function(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.1)

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="score_based",
            scoring_func=accuracy_score,
            scoring_transform_func=lambda x: x**2,
            calibration_container=sample_predictor_container,
        )

        ensemble.fit(sample_predictor_container)
        weights = ensemble.weights
        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0)


class TestEnsembleValidation:
    def test_incompatible_target_coords(self):
        predictor1 = MockPredictor("p1")
        predictor1.target_coord = "label"
        predictor1.sample_dim = "trial"

        predictor2 = MockPredictor("p2")
        predictor2.target_coord = "different_target"
        predictor2.sample_dim = "trial"

        with pytest.raises(ValueError, match="Inconsistent target_coord"):
            EnsemblePredictor(
                members=[("p1", predictor1), ("p2", predictor2)], sample_dim="trial", target_coord="label"
            )

    def test_predict_proba_with_non_classifiers(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", is_classifier=False)
        predictor2 = MockPredictor("p2")

        with pytest.raises(ValueError, match="Mixed classifier/regressor predictors"):
            EnsemblePredictor(
                members=[("p1", predictor1), ("p2", predictor2)], sample_dim="trial", target_coord="label"
            )

    def test_invalid_weighting_strategy(self):
        predictor1 = MockPredictor("p1")

        with pytest.raises(ValueError, match="Unknown weighting strategy"):
            ensemble = EnsemblePredictor(
                members=[("p1", predictor1)],
                sample_dim="trial",
                target_coord="label",
                weighting_strategy="invalid_strategy",
            )
            ensemble._compute_weights(None)


class TestEnsembleEdgeCases:
    def test_single_member_ensemble(self, sample_predictor_container):
        predictor = MockPredictor("single", prediction_bias=0.0)

        ensemble = EnsemblePredictor(
            members=[("single", predictor)], sample_dim="trial", target_coord="label", weights=[1.0]
        )

        ensemble.fit(sample_predictor_container)
        result = ensemble.predict(sample_predictor_container)

        assert isinstance(result, DataContainer)
        assert ensemble.weights == [1.0]

    def test_zero_weights_error(self):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")

        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            EnsemblePredictor(
                members=[("p1", predictor1), ("p2", predictor2)],
                sample_dim="trial",
                target_coord="label",
                weights=[0.0, 0.0],
            )

    def test_weight_normalization(self):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=[3.0, 7.0],
            normalize_weights=True,
        )

        expected_weights = [0.3, 0.7]
        assert np.allclose(ensemble.weights, expected_weights)

    def test_no_weight_normalization(self):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")

        original_weights = [3.0, 7.0]
        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weights=original_weights,
            normalize_weights=False,
        )

        assert ensemble.weights == original_weights

    def test_output_normalization(self, sample_predictor_container):
        predictor1_norm = MockPredictor("p1")
        predictor2_norm = MockPredictor("p2")
        predictor1_no_norm = MockPredictor("p1")
        predictor2_no_norm = MockPredictor("p2")

        ensemble_normalized = EnsemblePredictor(
            members=[("p1", predictor1_norm), ("p2", predictor2_norm)],
            sample_dim="trial",
            target_coord="label",
            normalize_outputs=False,
        )

        ensemble_non_normalized = EnsemblePredictor(
            members=[("p1", predictor1_no_norm), ("p2", predictor2_no_norm)],
            sample_dim="trial",
            target_coord="label",
            normalize_outputs=False,
        )

        ensemble_normalized.fit(sample_predictor_container)
        ensemble_non_normalized.fit(sample_predictor_container)

        result_norm = ensemble_normalized.predict_proba(sample_predictor_container)
        result_non_norm = ensemble_non_normalized.predict_proba(sample_predictor_container)

        if len(result_norm.data.dims) > 1:
            last_dim = result_norm.data.dims[-1]
            sums = result_norm.data.sum(dim=last_dim)
            assert np.allclose(sums.values, 1.0)
        if len(result_non_norm.data.dims) > 1:
            last_dim = result_non_norm.data.dims[-1]
            sums = result_non_norm.data.sum(dim=last_dim)
            assert np.allclose(sums.values, 1.0)


class TestEnsembleRepresentation:
    def test_repr(self):
        predictor1 = MockPredictor("p1")
        predictor2 = MockPredictor("p2")

        ensemble = EnsemblePredictor(
            members=[("p1", predictor1), ("p2", predictor2)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="uniform",
        )

        repr_str = repr(ensemble)
        assert "EnsemblePredictor" in repr_str
        assert "p1" in repr_str
        assert "p2" in repr_str
        assert "MockPredictor" in repr_str
        assert "uniform" in repr_str


class TestEnsembleWithPipelines:
    def _make_pipeline(self, name: str, predictor: Predictor) -> Pipeline:
        return Pipeline(name=name, steps=[("identity", IdentityTransform()), ("predict", predictor)])

    def test_cross_validate_pipeline_ending_in_ensemble(self, sample_predictor_container):
        """A terminal EnsemblePredictor must satisfy the Pipeline.fit() contract used by CV (fit_transform)."""
        ensemble = EnsemblePredictor(
            members=[("p1", MockPredictor("p1")), ("p2", MockPredictor("p2", prediction_bias=0.1))],
            sample_dim="trial",
            target_coord="label",
        )
        cv = KFoldValidator(n_splits=3, use_stateful_fit_cache=False, verbose=False)
        cv.set_pipeline(Pipeline("ensemble_cv", [("ensemble", ensemble)]))

        score = cv.cross_validate(sample_predictor_container, verbose=False)

        assert isinstance(score, float)
        assert len(cv.cv_scores_) == 3

    def test_pipeline_members_predict(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.2)

        pipe1 = self._make_pipeline("pipe1", predictor1)
        pipe2 = self._make_pipeline("pipe2", predictor2)

        ensemble = EnsemblePredictor(
            members=[("pipe1", pipe1), ("pipe2", pipe2)],
            sample_dim="trial",
            target_coord="label",
            weights=[0.6, 0.4],
        )

        ensemble.fit(sample_predictor_container)
        result = ensemble.predict(sample_predictor_container)

        assert isinstance(result, DataContainer)
        assert result.data.dims == ("trial",)
        assert result.data.sizes["trial"] == sample_predictor_container.data.sizes["trial"]
        assert ensemble.members[0].predictor is predictor1
        assert ensemble.members[1].predictor is predictor2

    def test_pipeline_members_predict_proba(self, sample_predictor_container):
        predictor1 = MockPredictor("p1", prediction_bias=0.0)
        predictor2 = MockPredictor("p2", prediction_bias=0.1)

        pipe1 = self._make_pipeline("pipe1", predictor1)
        pipe2 = self._make_pipeline("pipe2", predictor2)

        ensemble = EnsemblePredictor(
            members=[("pipe1", pipe1), ("pipe2", pipe2)],
            sample_dim="trial",
            target_coord="label",
            normalize_outputs=False,
        )

        ensemble.fit(sample_predictor_container)
        proba = ensemble.predict_proba(sample_predictor_container)

        assert isinstance(proba, DataContainer)
        assert len(proba.data.dims) == 2
        assert proba.data.sizes["trial"] == sample_predictor_container.data.sizes["trial"]

    def test_score_based_weighting_with_pipelines(self, sample_predictor_container):
        predictor_better = MockPredictor("better", prediction_bias=0.0)
        predictor_worse = MockPredictor("worse", prediction_bias=0.5)

        pipe_better = self._make_pipeline("pipe_better", predictor_better)
        pipe_worse = self._make_pipeline("pipe_worse", predictor_worse)

        ensemble = EnsemblePredictor(
            members=[("pipe_better", pipe_better), ("pipe_worse", pipe_worse)],
            sample_dim="trial",
            target_coord="label",
            weighting_strategy="score_based",
            scoring_func=accuracy_score,
            calibration_container=sample_predictor_container,
        )

        ensemble.fit(sample_predictor_container)
        weights = ensemble.weights

        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] > 0

    def test_nested_pipeline_members_use_encoded_entrypoint(self, sample_predictor_container):
        marker_predictor = MarkerAwarePredictor("marker_sensitive")
        inner_pipeline = Pipeline(
            name="inner_pipe",
            steps=[("marker", MarkerTransform()), ("predict", marker_predictor)],
        )
        outer_pipeline = Pipeline(
            name="outer_pipe",
            steps=[("identity", IdentityTransform()), ("inner", inner_pipeline)],
        )

        ensemble = EnsemblePredictor(
            members=[("outer_pipe", outer_pipeline)],
            sample_dim="trial",
            target_coord="label",
        )

        ensemble.fit(sample_predictor_container)

        prediction = ensemble.predict(sample_predictor_container)
        assert isinstance(prediction, DataContainer)

        proba = ensemble.predict_proba(sample_predictor_container)
        assert isinstance(proba, DataContainer)
        assert "class" in proba.data.dims
