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


def test_new_data_with_dims_in_another_order_is_put_back_in_fit_order(X, y):
    import xarray as xr
    from sklearn.pipeline import make_pipeline as sk_make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    flatten = FunctionTransformer(lambda a: a.reshape(len(a), -1))
    predictor = SKLearnPredictor.from_estimator(
        sk_make_pipeline(flatten, LogisticRegression()), sample_dim="trial", target_coord="label"
    )
    data = xr.DataArray(X, dims=DIMS, coords={**COORDS, "label": ("trial", y)})
    fitted = make_pipeline(predictor).fit(DataContainer(data))
    unlabeled = data.drop_vars("label")
    expected = fitted.predict(DataContainer(unlabeled)).data.values
    transposed = fitted.predict(DataContainer(unlabeled.transpose("trial", "time", "channel"))).data.values
    np.testing.assert_array_equal(transposed, expected)


def test_new_data_with_an_extra_singleton_dim_is_aligned(X, y):
    import xarray as xr
    from sklearn.pipeline import make_pipeline as sk_make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    flatten = FunctionTransformer(lambda a: a.reshape(len(a), -1))
    predictor = SKLearnPredictor.from_estimator(
        sk_make_pipeline(flatten, LogisticRegression()), sample_dim="trial", target_coord="label"
    )
    data = xr.DataArray(X, dims=DIMS, coords={**COORDS, "label": ("trial", y)})
    fitted = make_pipeline(predictor).fit(DataContainer(data))
    unlabeled = data.drop_vars("label")
    expected = fitted.predict_proba(DataContainer(unlabeled)).data.values
    odd = unlabeled.expand_dims("band").transpose("trial", "time", "band", "channel")
    np.testing.assert_allclose(fitted.predict_proba(DataContainer(odd)).data.values, expected)


def test_new_data_with_renamed_dims_or_dropped_labels_is_refused(X, y):
    import xarray as xr
    from sklearn.pipeline import make_pipeline as sk_make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    flatten = FunctionTransformer(lambda a: a.reshape(len(a), -1))
    predictor = SKLearnPredictor.from_estimator(
        sk_make_pipeline(flatten, LogisticRegression()), sample_dim="trial", target_coord="label"
    )
    data = xr.DataArray(X, dims=DIMS, coords={**COORDS, "label": ("trial", y)})
    fitted = make_pipeline(predictor).fit(DataContainer(data))
    unlabeled = data.drop_vars("label")
    with pytest.raises(ValueError, match="dims"):
        fitted.predict(DataContainer(unlabeled.rename(time="samples")))
    with pytest.raises(ValueError, match="channel"):
        fitted.predict(DataContainer(unlabeled.drop_vars("channel")))


def test_multilabel_target_seen_with_one_class_has_its_true_probability():
    import xarray as xr
    from sklearn.dummy import DummyClassifier

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    data = xr.DataArray(
        np.zeros((12, 2)),
        dims=("trial", "feature"),
        coords={"a": ("trial", np.zeros(12, int)), "b": ("trial", np.arange(12) % 2)},
    )
    predictor = SKLearnPredictor(DummyClassifier, sample_dim="trial", target_coord=["a", "b"], is_multilabel=True)
    proba = make_pipeline(predictor).fit(DataContainer(data)).predict_proba(DataContainer(data)).data.values
    np.testing.assert_allclose(proba[0], [0.0, 0.5])


@pytest.mark.parametrize("estimator", ["knn", "tree"])
def test_native_multilabel_estimator_with_a_one_class_target(estimator):
    import xarray as xr
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    data = xr.DataArray(
        np.random.default_rng(0).normal(size=(12, 2)),
        dims=("trial", "feature"),
        coords={"a": ("trial", np.zeros(12, int)), "b": ("trial", np.arange(12) % 2)},
    )
    model = KNeighborsClassifier(3) if estimator == "knn" else DecisionTreeClassifier(random_state=0)
    predictor = SKLearnPredictor.from_estimator(model, sample_dim="trial", target_coord=["a", "b"], is_multilabel=True)
    proba = make_pipeline(predictor).fit(DataContainer(data)).predict_proba(DataContainer(data)).data.values
    np.testing.assert_allclose(proba[:, 0], 0.0)  # "a" was never positive in training


@pytest.mark.parametrize("codes", [(0, 1), (1, 2), ("no", "yes")])
def test_multilabel_probability_is_of_the_larger_class(codes):
    import xarray as xr

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    rng = np.random.default_rng(0)
    on = np.arange(40) % 2 == 1
    values = rng.normal(size=(40, 2)) + on[:, None] * 3.0
    target = np.where(on, codes[1], codes[0])
    data = xr.DataArray(values, dims=("trial", "feature"), coords={"a": ("trial", target), "b": ("trial", target)})
    predictor = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord=["a", "b"], is_multilabel=True)
    fitted = make_pipeline(predictor).fit(DataContainer(data))
    proba = fitted.predict_proba(DataContainer(data)).data.values
    assert (proba[on] > 0.5).all() and (proba[~on] < 0.5).all()  # P(codes[1]), as in 0.3.0


def test_estimator_outputs_outside_the_labels_are_refused_not_coerced():
    import xarray as xr
    from sklearn.base import BaseEstimator, ClassifierMixin

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    class Odd(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), "apple_extra")  # "apple" after a cast to the labels' dtype

    data = xr.DataArray(
        np.zeros((6, 2)), dims=("trial", "feature"), coords={"y": ("trial", np.array(["apple", "berry"] * 3))}
    )
    predictor = SKLearnPredictor.from_estimator(Odd(), sample_dim="trial", target_coord="y", fit_on_labels=True)
    with pytest.raises(Exception, match="not among the training labels: \\['apple_extra'\\]"):
        make_pipeline(predictor).fit(DataContainer(data)).predict(DataContainer(data.drop_vars("y")))


def test_predictors_pickled_before_fit_on_labels_still_clone():
    import pickle

    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    predictor = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label")
    del predictor.fit_on_labels  # as saved by xdflow 0.3.0
    assert pickle.loads(pickle.dumps(predictor)).clone().fit_on_labels is False


@pytest.mark.parametrize("case", ["ints_for_bools", "column_vector"])
def test_estimator_outputs_equal_to_the_labels_are_accepted(case):
    import xarray as xr
    from sklearn.base import BaseEstimator, ClassifierMixin

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    labels = np.array([False, True] * 3) if case == "ints_for_bools" else np.array(["left", "right"] * 3)

    class Model(ClassifierMixin, BaseEstimator):  # like XGBoost (0/1 for bools) or CatBoost (shape (n, 1))
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            if case == "ints_for_bools":
                return (np.arange(len(X)) % 2).astype(int)
            return labels[: len(X)].reshape(-1, 1)

    data = xr.DataArray(np.zeros((6, 2)), dims=("trial", "feature"), coords={"y": ("trial", labels)})
    predictor = SKLearnPredictor.from_estimator(Model(), sample_dim="trial", target_coord="y", fit_on_labels=True)
    predicted = make_pipeline(predictor).fit(DataContainer(data)).predict(DataContainer(data.drop_vars("y")))
    np.testing.assert_array_equal(predicted.data.values, labels)


def test_a_multilabel_target_seen_with_one_class_reads_its_classes_not_its_columns():
    import xarray as xr
    from sklearn.neural_network import MLPClassifier

    from xdflow import DataContainer
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.normal(size=(20, 2)),
        dims=("trial", "feature"),
        coords={"a": ("trial", np.ones(20, int)), "b": ("trial", np.arange(20) % 2)},
    )
    predictor = SKLearnPredictor(  # wrapped per target in a MultiOutputClassifier
        MLPClassifier, sample_dim="trial", target_coord=["a", "b"], is_multilabel=True, max_iter=50, random_state=0
    )
    proba = make_pipeline(predictor).fit(DataContainer(data)).predict_proba(DataContainer(data)).data.values
    np.testing.assert_allclose(proba[:, 0], 1.0)
