"""make_pipeline names steps after their class, like scikit-learn's."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline as sklearn_make_pipeline
from sklearn.preprocessing import StandardScaler

from xdflow.composite.pipeline import make_pipeline
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.function_step import step
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer


@step
def rectify(data):
    return abs(data)


def test_step_names():
    pipeline = make_pipeline(
        ZScoreTransform(per_dim="trial"),
        ZScoreTransform(per_dim="trial"),
        rectify(),
        FlattenTransform(dims=("channel", "time")),
        SKLearnTransformer(StandardScaler, sample_dim="trial"),
        SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"),
    )
    assert [s.name for s in pipeline.steps] == [
        "zscore-1",
        "zscore-2",
        "rectify",
        "flatten",
        "standardscaler",
        "logisticregression",
    ]
    assert pipeline.name == "pipeline"


def test_wrapped_estimator_instance_is_named_after_it():
    wrapped = SKLearnPredictor.from_estimator(
        sklearn_make_pipeline(StandardScaler(), LogisticRegression()), sample_dim="trial", target_coord="label"
    )
    assert make_pipeline(wrapped).steps[0].name == "pipeline"


def test_rejects_classes_and_estimators():
    with pytest.raises(TypeError, match=r"rectify\(\)"):
        make_pipeline(rectify)
    with pytest.raises(TypeError, match="from_estimator"):
        make_pipeline(LogisticRegression())
