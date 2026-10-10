import numpy as np
import pytest
import xarray as xr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-style classifier that records the dtype of y."""

    def __init__(self):
        self.saw_y_dtype_ = None
        self.saw_y_is_int_ = None
        self.classes_ = None

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.saw_y_dtype_ = y_arr.dtype
        self.saw_y_is_int_ = np.issubdtype(y_arr.dtype, np.integer)
        self.classes_ = np.unique(y_arr)
        return self

    def predict(self, X):
        n = X.shape[0]
        return np.zeros(n, dtype=int)


@pytest.fixture
def labeled_container():
    data = xr.DataArray(
        np.random.randn(20, 3, 5),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(20),
            "channel": np.arange(3),
            "time": np.arange(5),
            "label": ("trial", np.array(["a", "b"] * 10)),
        },
    )
    return DataContainer(data)


def test_predictor_fit_transform_uses_encoder(labeled_container):
    """Pipeline.fit routes Predictor through fit() so y is encoded to ints."""
    pipeline = Pipeline(
        name="predictor_encoding_path",
        steps=[
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            (
                "clf",
                SKLearnPredictor(
                    estimator_cls=RecordingClassifier,
                    sample_dim="trial",
                    target_coord="label",
                    encoder=LabelEncoder(),
                ),
            ),
        ],
    )

    pipeline.fit(labeled_container)

    predictor = pipeline.steps[-1].transform
    assert predictor.estimator.saw_y_is_int_ is True


def test_predictor_get_labels_returns_encoder_classes(labeled_container):
    """Predictor.get_labels exposes fitted encoder order."""
    pipeline = Pipeline(
        name="predictor_label_access",
        steps=[
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            (
                "clf",
                SKLearnPredictor(
                    estimator_cls=RecordingClassifier,
                    sample_dim="trial",
                    target_coord="label",
                    encoder=LabelEncoder(),
                ),
            ),
        ],
    )

    pipeline.fit(labeled_container)
    predictor = pipeline.predictive_transform

    assert predictor.get_labels() == list(predictor.encoder.classes_)


def test_pipeline_get_labels_proxies_predictor(labeled_container):
    """Pipeline.get_labels returns the final predictor's labels."""
    pipeline = Pipeline(
        name="pipeline_label_access",
        steps=[
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            (
                "clf",
                SKLearnPredictor(
                    estimator_cls=RecordingClassifier,
                    sample_dim="trial",
                    target_coord="label",
                    encoder=LabelEncoder(),
                ),
            ),
        ],
    )

    pipeline.fit(labeled_container)
    expected = list(pipeline.predictive_transform.encoder.classes_)

    assert pipeline.get_labels() == expected


def test_pipeline_get_labels_requires_predictor():
    """Calling get_labels on a pipeline without a predictor should raise."""
    pipeline = Pipeline(
        name="feature_only",
        steps=[("flatten", FlattenTransform(dims=("channel", "time")))],
    )

    with pytest.raises(TypeError):
        pipeline.get_labels()


def test_predictor_get_labels_rejects_regressor():
    """Regressors calling get_labels should raise."""
    predictor = SKLearnPredictor(
        estimator_cls=LinearRegression,
        sample_dim="trial",
        target_coord="target",
        is_classifier=False,
    )

    with pytest.raises(TypeError):
        predictor.get_labels()


def test_proba_auto_adds_encoder():
    """Constructor auto-provides an encoder for classifiers when proba=True."""
    predictor = SKLearnPredictor(
        estimator_cls=LogisticRegression,
        sample_dim="trial",
        target_coord="label",
        proba=True,
    )
    assert isinstance(predictor.encoder, LabelEncoder)


class ProbaNoClassesEstimator(ClassifierMixin, BaseEstimator):
    """Estimator with predict_proba but without classes_."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0
        return probs


def test_proba_without_estimator_classes_raises_at_transform(labeled_container):
    """Even with an encoder, transform should fail if estimator lacks classes_ for alignment."""
    pipeline = Pipeline(
        name="proba_no_classes_alignment",
        steps=[
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            (
                "clf",
                SKLearnPredictor(
                    estimator_cls=ProbaNoClassesEstimator,
                    sample_dim="trial",
                    target_coord="label",
                    proba=True,
                ),
            ),
        ],
    )

    with pytest.raises(TransformError):
        pipeline.fit_transform(labeled_container)


def test_regressor_cannot_accept_encoder():
    """Regressors must not accept encoders at construction time."""
    with pytest.raises(ValueError):
        SKLearnPredictor(
            estimator_cls=LinearRegression,
            sample_dim="trial",
            target_coord="target",
            encoder=LabelEncoder(),
        )


def test_clone_does_not_share_a_fitted_encoder():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    encoder = LabelEncoder().fit(["a", "b"])
    predictor = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="y", encoder=encoder)
    clone = predictor.clone()

    assert clone.encoder is not predictor.encoder
    np.testing.assert_array_equal(clone.encoder.classes_, ["a", "b"])
    predictor.encoder.fit(["c", "d"])
    np.testing.assert_array_equal(clone.encoder.classes_, ["a", "b"])
