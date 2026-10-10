"""evaluate: one call from data and a model to fold, group and trial-level results."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline as sklearn_make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from xdflow import DataContainer, KFoldValidator
from xdflow.composite.pipeline import make_pipeline
from xdflow.evaluate import evaluate
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.function_step import step
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _flatten(a):
    return a.reshape(len(a), -1)


@step
def rectify(data):
    return abs(data)


@pytest.fixture
def X():  # noqa: N802 - scikit-learn's name for the feature array
    rng = np.random.default_rng(0)
    values = rng.normal(size=(120, 2, 10))
    values[np.arange(120) % 2 == 0, 0] += 0.6
    return values


@pytest.fixture
def y():
    return np.array(["move", "rest"])[np.arange(120) % 2]


@pytest.fixture
def info():
    return pd.DataFrame({"subject": np.repeat(np.arange(6), 20)})


@pytest.fixture
def model():
    return sklearn_make_pipeline(FunctionTransformer(_flatten), StandardScaler(), LogisticRegression())


def test_matches_cross_val_score_on_the_same_split(model, X, y, info):
    groups = info.subject.to_numpy()
    expected = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(3))
    result = evaluate(model, X, y, groups=groups, cv=GroupKFold(3))
    np.testing.assert_allclose(result.fold_scores, expected)
    assert result.metric == "accuracy"
    assert result.score == pytest.approx(expected.mean())


def test_shared_subjects_warn_and_groups_silence_it(model, X, y, info):
    with pytest.warns(UserWarning, match="groups='subject'"):
        shuffled = evaluate(model, X, y, metadata=info)
    assert not shuffled.split.ok
    assert "trials shuffled" in shuffled.cv_description

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grouped = evaluate(model, X, y, groups="subject", metadata=info, cv=6)
    assert grouped.split.ok
    assert grouped.cv_description == "leave one subject out (6 folds)"
    assert len(grouped.fold_scores) == 6


def test_more_groups_than_folds_are_grouped_into_folds(model, X, y, info):
    result = evaluate(model, X, y, groups="subject", metadata=info, cv=3)
    assert len(result.fold_scores) == 3
    assert "stratified by class" in result.cv_description
    subjects = info.subject.to_numpy()
    for train, test in result.validator.split_trials_:
        assert set(subjects[train]).isdisjoint(subjects[test])


def test_predictions_and_scores_by_group(model, X, y, info):
    result = evaluate(model, X, y, groups="subject", metadata=info, cv=6)
    predictions = result.predictions
    assert len(predictions) == 120
    assert {"subject", "y_true", "y_pred", "fold"} <= set(predictions.columns)
    np.testing.assert_array_equal(predictions.sort_values("trial").y_true.to_numpy(), y)

    by_subject = result.by("subject")
    assert list(by_subject.index) == list(range(6))
    assert (by_subject.n_trials == 20).all()
    # Leave-one-subject-out: each subject is one fold, so its score is that fold's score.
    np.testing.assert_allclose(by_subject["accuracy"].to_numpy(), result.fold_scores)
    assert "By subject" in repr(result)


def test_xdflow_steps_before_an_sklearn_model_run_once(X, y, info):
    data = DataContainer(
        xr.DataArray(
            X,
            dims=("trial", "channel", "time"),
            coords={"trial": np.arange(120), "label": ("trial", y), "subject": ("trial", info.subject.to_numpy())},
        )
    )
    result = evaluate(
        [rectify(), FlattenTransform(dims=("channel", "time")), LogisticRegression()], data, groups="subject"
    )
    text = repr(result)
    assert "Computed once: rectify, flatten." in text
    assert "Refit in each fold: logisticregression." in text


def test_final_model_predicts_unlabeled_data(model, X, y):
    result = evaluate(model, X, y, cv=3)
    fitted = result.fit_final()
    new = DataContainer(xr.DataArray(X[:5], dims=("trial", "dim_1", "dim_2"), coords={"trial": np.arange(5)}))
    assert set(fitted.predict(new).data.values) <= {"move", "rest"}


def test_xdflow_pipeline_target_and_regression_default(X):
    target = X[:, 0].mean(-1)
    pipeline = make_pipeline(
        FlattenTransform(dims=("channel", "time")), SKLearnPredictor(Ridge, sample_dim="trial", target_coord="value")
    )
    data = DataContainer(
        xr.DataArray(X, dims=("trial", "channel", "time"), coords={"trial": np.arange(120), "value": ("trial", target)})
    )
    result = evaluate(pipeline, data, cv=3)
    assert result.metric == "r2"
    assert result.score > 0.9
    with pytest.raises(ValueError, match="predicts 'value'"):
        evaluate(pipeline, data, "other")


def test_mne_epochs_input():
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(0)
    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    events = np.column_stack([np.arange(40) * 100, np.zeros(40, int), 1 + np.arange(40) % 2])
    values = rng.normal(size=(40, 2, 50)) * 1e-6
    values[events[:, 2] == 1, 0] *= 3
    epochs = mne.EpochsArray(values, info, events=events, event_id={"left": 1, "right": 2}, verbose=False)
    model = sklearn_make_pipeline(FunctionTransformer(lambda a: np.log(a.var(-1))), LogisticRegression())
    result = evaluate(model, epochs, cv=4)
    assert result.score > 0.8
    assert set(result.predictions.y_true) == {"left", "right"}


def test_conflicting_or_missing_arguments(model, X, y, info):
    with pytest.raises(ValueError, match="Pass y"):
        evaluate(model, X)
    with pytest.raises(ValueError, match="at least 2"):
        evaluate(model, X, y, groups=np.zeros(120))
    with pytest.raises(ValueError, match="set them there"):
        evaluate(model, X, y, groups="subject", metadata=info, cv=KFoldValidator(verbose=False))
    with pytest.raises(ValueError, match="at least 2 folds"):
        evaluate(model, X, y, cv=1)
    with pytest.raises(TypeError, match="last step"):
        evaluate([StandardScaler(), LogisticRegression()], X, y)


@pytest.mark.parametrize("scoring", ["f1", "balanced_accuracy", "f1_macro"])
def test_scoring_names_mean_what_they_mean_in_scikit_learn(model, X, info, scoring):
    binary = np.arange(120) % 2  # sklearn's "f1" is binary F1 with positive label 1
    groups = info.subject.to_numpy()
    expected = cross_val_score(model, X, binary, groups=groups, cv=GroupKFold(3), scoring=scoring)
    result = evaluate(model, X, binary, groups=groups, cv=GroupKFold(3), scoring=scoring)
    np.testing.assert_allclose(result.fold_scores, expected)
    assert result.metric == scoring
    assert list(result.by("group").columns) == [scoring, "n_trials"]


def test_probability_metrics_are_refused_with_a_pointer(model, X, y):
    with pytest.raises(ValueError, match="scoring_needs_proba"):
        evaluate(model, X, y, scoring="roc_auc")


def test_single_subject_data_does_not_warn(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate(model, X, y, metadata=pd.DataFrame({"subject": np.full(120, "S1")}), cv=3)
    assert result.split.shared_coords == ["subject"]


def test_a_validator_can_be_passed_as_cv(model, X, y):
    result = evaluate(model, X, y, cv=KFoldValidator(n_splits=3, verbose=False))
    assert len(result.fold_scores) == 3
    assert result.metric == "f1_weighted"  # the validator's own default


@pytest.mark.filterwarnings("ignore:F-score is ill-defined")
def test_a_fold_with_one_class_scores_like_scikit_learn():
    from sklearn.dummy import DummyClassifier

    X = np.arange(24.0).reshape(12, 2)
    y = np.r_[np.zeros(4, int), np.arange(8) % 2]
    pairs = [(np.arange(4, 12), np.arange(4))]  # trained on both classes, tested on negatives only
    model = DummyClassifier(strategy="constant", constant=0)
    expected = cross_val_score(model, X, y, cv=pairs, scoring="f1")
    assert expected.tolist() == [0.0]
    np.testing.assert_allclose(evaluate(model, X, y, cv=pairs, scoring="f1").fold_scores, expected)
