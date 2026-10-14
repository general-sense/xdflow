"""to_sklearn: an xdflow pipeline behind the scikit-learn estimator API."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score

from xdflow.composite.pipeline import make_pipeline
from xdflow.evaluate import evaluate
from xdflow.sklearn_api import XDFlowClassifier, XDFlowRegressor, to_sklearn
from xdflow.transforms.basic_transforms import CropTimeTransform, FlattenTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

DIMS = ("trial", "channel", "time")
COORDS = {"channel": ["C3", "C4"], "time": np.linspace(0.0, 0.9, 10)}


@pytest.fixture
def X():  # noqa: N802 - scikit-learn's name for the feature array
    rng = np.random.default_rng(0)
    values = rng.normal(size=(80, 2, 10))
    values[np.arange(80) % 2 == 0, 0, 5:] += 1.0  # signal only late in the trial
    return values


@pytest.fixture
def y():
    return np.array(["a", "b"])[np.arange(80) % 2]


def _classifier_pipeline():
    return make_pipeline(
        CropTimeTransform(0.5, 0.9),
        FlattenTransform(dims=("channel", "time")),
        SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"),
    )


def test_estimator_type_and_scoring(X, y):
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS)
    assert isinstance(clf, XDFlowClassifier) and is_classifier(clf)
    scores = cross_val_score(clf, X, y, cv=4)
    assert scores.mean() > 0.8
    assert cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean() > 0.8


def test_same_scores_as_evaluate_on_the_same_split(X, y):
    groups = np.repeat(np.arange(8), 10)
    via_sklearn = cross_val_score(
        to_sklearn(_classifier_pipeline(), DIMS, COORDS), X, y, groups=groups, cv=GroupKFold(4)
    )
    from xdflow.interop import from_numpy

    data = from_numpy(X, DIMS, coords=COORDS, label=y, group=groups)
    via_xdflow = evaluate(_classifier_pipeline(), data, groups="group", cv=GroupKFold(4))
    np.testing.assert_allclose(via_sklearn, via_xdflow.fold_scores)


def test_parallel_cross_validation(X, y):
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS)
    np.testing.assert_allclose(cross_val_score(clf, X, y, cv=4, n_jobs=2), cross_val_score(clf, X, y, cv=4))


def test_grid_search_reaches_the_xdflow_step(X, y):
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS)
    search = GridSearchCV(clf, {"pipeline__logisticregression__C": [1e-4, 1.0]}, cv=3).fit(X, y)
    assert search.best_params_ == {"pipeline__logisticregression__C": 1.0}
    assert search.best_estimator_.pipeline_.steps[-1].transform.estimator.C == 1.0
    assert clf.pipeline.steps[-1].transform.estimator.C == 1.0  # the template was not touched


def test_clone_is_independent_and_unfitted(X, y):
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS).fit(X, y)
    copy = clone(clf)
    assert copy.pipeline is not clf.pipeline
    assert not hasattr(copy, "pipeline_")


def test_predict_proba_columns_follow_classes(X, y):
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS).fit(X, y)
    np.testing.assert_array_equal(clf.classes_, ["a", "b"])
    proba = clf.predict_proba(X)
    assert proba.shape == (80, 2)
    np.testing.assert_array_equal(clf.classes_[proba.argmax(1)], clf.predict(X))


def test_regressor(X):
    target = X[:, 0, 5:].mean(-1)
    pipeline = make_pipeline(
        FlattenTransform(dims=("channel", "time")), SKLearnPredictor(Ridge, sample_dim="trial", target_coord="value")
    )
    reg = to_sklearn(pipeline, DIMS)
    assert isinstance(reg, XDFlowRegressor) and is_regressor(reg)
    assert cross_val_score(reg, X, target, cv=3).mean() > 0.9


def test_requires_a_predictor_pipeline():
    with pytest.raises(TypeError, match="ends with a predictor"):
        to_sklearn(make_pipeline(FlattenTransform(dims=("channel", "time"))))


def test_predictions_follow_input_order_when_a_step_reorders_trials(X, y):
    from xdflow.transforms.function_step import step

    @step
    def reverse(data):
        return data.isel(trial=slice(None, None, -1))

    plain = to_sklearn(_classifier_pipeline(), DIMS, COORDS).fit(X, y)
    reordering = make_pipeline(reverse(), *[s.transform for s in _classifier_pipeline().steps])
    shuffled = to_sklearn(reordering, DIMS, COORDS).fit(X, y)
    np.testing.assert_array_equal(shuffled.predict(X), plain.predict(X))
    np.testing.assert_allclose(shuffled.predict_proba(X), plain.predict_proba(X))


def test_classes_come_from_the_training_labels(X):
    from xdflow.interop import from_numpy

    three = np.arange(80) % 3
    template = _classifier_pipeline().fit(from_numpy(X, DIMS, coords=COORDS, label=three))  # already fitted
    clf = to_sklearn(template, DIMS, COORDS).fit(X[three != 2], three[three != 2])
    np.testing.assert_array_equal(clf.classes_, [0, 1])
    assert clf.predict_proba(X).shape == (80, 2)


def test_predict_proba_with_bool_labels_has_a_column_per_class(X):
    y = np.arange(80) % 2 == 0
    clf = to_sklearn(_classifier_pipeline(), DIMS, COORDS).fit(X, y)
    reference = LogisticRegression().fit(X[:, :, 5:].reshape(80, -1), y)
    np.testing.assert_array_equal(clf.classes_, [False, True])
    np.testing.assert_allclose(clf.predict_proba(X), reference.predict_proba(X[:, :, 5:].reshape(80, -1)), atol=1e-6)


def test_continuous_targets_are_refused_by_a_classifier(X):
    with pytest.raises(ValueError, match="continuous"):
        to_sklearn(_classifier_pipeline(), DIMS, COORDS).fit(X, np.linspace(0, 1, 80))
