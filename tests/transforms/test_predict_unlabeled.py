"""A fitted classifier predicts on new data that has no label coordinate (the deployment case)."""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow import DataContainer, Pipeline
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _data(with_labels: bool) -> DataContainer:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(40, 3))
    labels = np.array(["rest", "move"])[np.arange(40) % 2]
    values[labels == "move", 0] += 2.0
    coords = {"trial": np.arange(40), "feature": ["a", "b", "c"]}
    if with_labels:
        coords["label"] = ("trial", labels)
    return DataContainer(xr.DataArray(values, dims=("trial", "feature"), coords=coords))


@pytest.fixture
def fitted() -> Pipeline:
    pipeline = Pipeline(
        "clf",
        [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label", proba=False))],
    )
    return pipeline.fit(_data(with_labels=True))


def test_predict_and_predict_proba_without_labels(fitted):
    unlabeled, labeled = _data(with_labels=False), _data(with_labels=True)

    predictions = fitted.predict(unlabeled).data
    assert "label" not in predictions.coords
    np.testing.assert_array_equal(predictions.values, fitted.predict(labeled).data.values)

    proba = fitted.predict_proba(unlabeled).data
    np.testing.assert_allclose(proba.values, fitted.predict_proba(labeled).data.values)
    assert list(proba.coords["class"].values) == fitted.get_labels()


def test_labels_are_kept_when_present(fitted):
    labeled = _data(with_labels=True)
    predictions = fitted.predict(labeled).data
    np.testing.assert_array_equal(predictions.coords["label"].values, labeled.data.coords["label"].values)


def test_fit_still_requires_labels():
    pipeline = Pipeline("clf", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    with pytest.raises(Exception, match="label"):
        pipeline.fit(_data(with_labels=False))


def test_ensemble_predicts_without_labels():
    from xdflow import EnsemblePredictor

    members = [
        ("a", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label")),
        ("b", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label", C=0.1)),
    ]
    ensemble = EnsemblePredictor(members, sample_dim="trial", target_coord="label")
    pipeline = Pipeline("ens", [("ens", ensemble)]).fit(_data(with_labels=True))
    labeled, unlabeled = _data(with_labels=True), _data(with_labels=False)
    np.testing.assert_array_equal(pipeline.predict(unlabeled).data.values, pipeline.predict(labeled).data.values)
    np.testing.assert_allclose(
        pipeline.predict_proba(unlabeled).data.values, pipeline.predict_proba(labeled).data.values
    )
