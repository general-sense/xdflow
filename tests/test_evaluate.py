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


@pytest.mark.parametrize("scoring", [["accuracy", "f1"], {"acc": "accuracy"}])
def test_scoring_that_is_neither_a_name_nor_a_function_is_refused(model, X, y, scoring):
    with pytest.raises(TypeError, match="scoring"):
        evaluate(model, X, y, cv=3, scoring=scoring)


def test_scikit_learn_scorer_objects_score_like_cross_val_score(model, X, y):
    from sklearn.metrics import fbeta_score, make_scorer
    from sklearn.model_selection import StratifiedKFold

    scorer = make_scorer(fbeta_score, beta=2, pos_label="move")
    expected = cross_val_score(model, X, y, cv=StratifiedKFold(3), scoring=scorer)
    result = evaluate(model, X, y, cv=StratifiedKFold(3), scoring=scorer)
    np.testing.assert_allclose(result.fold_scores, expected)
    assert result.metric == "fbeta_score"


def test_binary_f1_with_binary_folds_of_multiclass_labels_scores_like_scikit_learn():
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier

    codes = np.zeros(12, int)
    codes[[1, 2, 7]], codes[3] = 1, 2
    X = np.random.default_rng(0).normal(size=(12, 3)) + codes[:, None]
    y = np.array([1, 2, 3])[codes]  # three classes overall, two in each fold
    model = DecisionTreeClassifier(random_state=0)
    expected = cross_val_score(model, X, y, cv=KFold(4), scoring="f1", error_score="raise")
    np.testing.assert_allclose(evaluate(model, X, y, cv=KFold(4), scoring="f1").fold_scores, expected)


@pytest.mark.parametrize("target", ["continuous", "nan"])
def test_classifier_targets_scikit_learn_rejects_are_rejected(X, target):
    y = np.linspace(0, 1, 120) if target == "continuous" else np.where(np.arange(120) % 5, np.arange(120) % 2, np.nan)
    from sklearn.model_selection import KFold

    with pytest.raises(ValueError, match="continuous|NaN"):
        evaluate(LogisticRegression(), X.reshape(120, -1), y, cv=KFold(3), scoring=lambda t, p: float(np.mean(t == p)))


def test_cv_none_means_the_default_folds(model, X, y):
    np.testing.assert_allclose(evaluate(model, X, y, cv=None).fold_scores, evaluate(model, X, y).fold_scores)


def test_an_xdflow_pipeline_takes_y_as_an_array(X, y):
    pipeline = make_pipeline(
        FlattenTransform(dims=("dim_1", "dim_2")),
        SKLearnPredictor.from_estimator(LogisticRegression(), sample_dim="trial", target_coord="label"),
    )
    expected = evaluate(pipeline, X, metadata=pd.DataFrame({"label": y}), cv=GroupKFold(3), groups=np.arange(120) % 6)
    result = evaluate(pipeline, X, y, cv=GroupKFold(3), groups=np.arange(120) % 6)
    np.testing.assert_allclose(result.fold_scores, expected.fold_scores)


def test_predicting_on_a_bare_array_explains_how_to_wrap_it(model, X, y):
    final = evaluate(model, X, y, cv=3).fit_final()
    for data in (X, xr.DataArray(X, dims=("trial", "channel", "time"))):
        with pytest.raises(TypeError, match="from_numpy"):
            final.predict(data)
        with pytest.raises(TypeError, match="from_numpy"):
            final.predict_proba(data)


def test_reusing_a_validator_leaves_earlier_results_intact(model, X, y):
    from sklearn.dummy import DummyClassifier

    validator = KFoldValidator(n_splits=3, verbose=False)
    first = evaluate(model, X, y, cv=validator)
    scores = list(first.fold_scores)
    evaluate(DummyClassifier(), X, y, cv=validator)
    assert first.validator.cv_scores_ == scores
    assert first.fit_final().steps[0].name == "pipeline"


def test_groups_split_apart_by_the_splitter_warn(model, X, y, info):
    from sklearn.model_selection import KFold

    with pytest.warns(UserWarning, match="puts trials of the same 'group'"):
        evaluate(model, X, y, groups=info.subject.to_numpy(), cv=KFold(3, shuffle=True, random_state=0))


def test_summary_mean_is_the_mean_of_fold_scores(model, X, y):
    validator = KFoldValidator(n_splits=7, verbose=False, pooling_score_weight=1.0, scoring="accuracy")
    result = evaluate(model, X, y, cv=validator)
    assert repr(result).startswith(f"{result.metric} {np.mean(result.fold_scores):.3f} ±")


def test_metadata_indexed_by_the_trial_labels_in_another_order_is_refused():
    """Position or label? Either reading is plausible, so the rows must be put in the data's order first."""
    rng = np.random.default_rng(0)
    trials = rng.permutation(60) + 1000
    data = xr.DataArray(
        rng.normal(size=(60, 3)),
        dims=("trial", "feature"),
        coords={"trial": trials, "label": ("trial", np.array(["a", "b"])[np.arange(60) % 2])},
    )
    info = pd.DataFrame({"subject": np.repeat(np.arange(6), 10)}, index=np.sort(trials))
    with pytest.raises(ValueError, match="order"):
        evaluate(LogisticRegression(), data, groups="subject", metadata=info, cv=3)
    result = evaluate(LogisticRegression(), data, groups="subject", metadata=info.loc[trials], cv=3)
    got = result.predictions.set_index("trial")["subject"].loc[info.index]
    np.testing.assert_array_equal(got.to_numpy(), info["subject"].to_numpy())
    with pytest.raises(ValueError, match="order"):  # reordered positional labels: also ambiguous
        reordered = data.assign_coords(trial=np.arange(60)).isel(trial=rng.permutation(60))
        evaluate(LogisticRegression(), reordered, groups="subject", metadata=info.set_axis(np.arange(60)), cv=3)


def test_a_negated_scorer_is_named_like_scikit_learn():
    from sklearn.metrics import get_scorer

    X, y = np.random.default_rng(0).normal(size=(60, 3)), np.arange(60.0)
    result = evaluate(Ridge(), X, y, cv=3, scoring=get_scorer("neg_mean_squared_error"))
    assert result.metric == "neg_mean_squared_error"


def test_by_keeps_trials_with_a_missing_value(model, X, y):
    condition = np.array(["A", "B", None], dtype=object)[np.arange(120) % 3]
    result = evaluate(model, X, y, cv=3, metadata=pd.DataFrame({"condition": condition}))
    assert result.by("condition")["n_trials"].sum() == 120


def test_summary_counts_trials_once_when_test_sets_overlap(model, X, y):
    from sklearn.model_selection import ShuffleSplit

    result = evaluate(model, X, y, cv=ShuffleSplit(10, test_size=0.5, random_state=0))
    assert "120 trials, 600 predictions" in repr(result)


def test_y_given_twice_is_refused(model, X, y):
    with pytest.raises(ValueError, match="both"):
        evaluate(model, X, y, cv=3, metadata=pd.DataFrame({"y": y}))


def test_metadata_rows_match_numpy_rows_by_position(model, X, y):
    shuffled = pd.DataFrame({"subject": np.repeat(np.arange(6), 20), "y": y}).sample(frac=1, random_state=0)
    result = evaluate(model, X, shuffled["y"].to_numpy(), groups="subject", metadata=shuffled[["subject"]], cv=3)
    np.testing.assert_array_equal(
        result.predictions.sort_values("trial")["subject"].to_numpy(), shuffled["subject"].to_numpy()
    )


def test_validator_hint_names_a_group_splitter(model, X, y, info):
    with pytest.warns(UserWarning, match="GroupKFold"):
        evaluate(model, X, y, metadata=info, cv=KFoldValidator(n_splits=3, verbose=False))


@pytest.mark.parametrize("kind", ["nan", "categorical", "timezone", "range_index"])
def test_metadata_label_checks_compare_labels_by_value(kind):
    from xdflow.interop import _as_container

    labels = {
        "nan": pd.Index([2.0, np.nan, 1.0]),
        "categorical": pd.CategoricalIndex(["b", "a", "c"], categories=["c", "b", "a"]),
        "timezone": pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"], tz="UTC"),
        "range_index": pd.Index([2, 0, 1]),
    }[kind]
    data = xr.DataArray(np.arange(3.0)[:, None], dims=("trial", "f"), coords={"trial": labels.to_numpy()})
    if kind == "range_index":  # rows already in the data's order, positional: accepted
        rows = pd.DataFrame({"subject": [20, 0, 10]})
    elif kind == "nan":  # same labels, other order: refused
        rows = pd.DataFrame({"subject": [10, 20, 0]}, index=pd.Index([1.0, 2.0, np.nan]))
    else:  # same labels in the same order (other categories / time zone): accepted
        index = labels.set_categories(["a", "b", "c"]) if kind == "categorical" else labels.tz_convert("US/Eastern")
        rows = pd.DataFrame({"subject": [20, 0, 10]}, index=index)
    if kind == "nan":
        with pytest.raises(ValueError, match="order"):
            _as_container(data, metadata=rows)
    else:
        np.testing.assert_array_equal(_as_container(data, metadata=rows).data.subject.values, [20, 0, 10])


def test_metadata_rows_by_position_when_labels_only_partly_match():
    from xdflow.interop import _as_container

    data = xr.DataArray(np.arange(3.0)[:, None], dims=("trial", "f"), coords={"trial": ["a", "a", "b"]})
    rows = pd.DataFrame({"subject": [1, 2, 3]}, index=["a", "b", "c"])
    np.testing.assert_array_equal(_as_container(data, metadata=rows).data.subject.values, [1, 2, 3])
    data = xr.DataArray(np.arange(3.0)[:, None], dims=("trial", "f"), coords={"trial": [0.5, 1.5, 2.5]})
    rows = pd.DataFrame({"subject": [1, 2, 3]}, index=pd.IntervalIndex.from_tuples([(1, 2), (2, 3), (0, 1)]))
    np.testing.assert_array_equal(_as_container(data, metadata=rows).data.subject.values, [1, 2, 3])


def test_a_non_default_range_index_names_trials():
    from xdflow.interop import _as_container

    data = xr.DataArray(np.arange(3.0)[:, None], dims=("trial", "f"), coords={"trial": [10, 20, 30]})
    with pytest.raises(ValueError, match="order"):
        _as_container(data, metadata=pd.DataFrame({"subject": [3, 2, 1]}, index=pd.RangeIndex(30, 0, -10)))


def test_series_keyed_by_the_trial_labels_in_another_order_is_refused():
    rng = np.random.default_rng(0)
    trials = np.array([f"id{i:03d}" for i in rng.permutation(40)])
    data = xr.DataArray(rng.normal(size=(40, 3)), dims=("trial", "f"), coords={"trial": trials})
    y = pd.Series(np.arange(40) % 2, index=np.sort(trials))
    with pytest.raises(ValueError, match="order"):
        evaluate(LogisticRegression(), data, y, cv=3)
    result = evaluate(LogisticRegression(), data, y.loc[trials], cv=3)  # rows in the data's order are fine
    got = result.predictions.set_index("trial")["y_true"].loc[y.index]
    np.testing.assert_array_equal(got.to_numpy(), y.to_numpy())


def test_a_metadata_column_named_sample_weight_does_not_weight_a_scikit_learn_model(model, X, y):
    from sklearn.model_selection import StratifiedKFold

    weights = np.random.default_rng(0).uniform(0.1, 5, size=120)
    expected = cross_val_score(model, X, y, cv=StratifiedKFold(5))
    result = evaluate(model, X, y, cv=StratifiedKFold(5), metadata=pd.DataFrame({"sample_weight": weights}))
    np.testing.assert_allclose(result.fold_scores, expected)


def test_a_coordinate_over_trial_and_channel_does_not_break_prediction(model, X, y):
    data = xr.DataArray(
        X,
        dims=("trial", "channel", "time"),
        coords={"label": ("trial", y), "impedance": (("trial", "channel"), np.ones((120, 2)))},
    )
    evaluate(model, data, "label", cv=3)


@pytest.mark.parametrize("labels", [np.array([1, 2, 3]), np.array(["a", "b", "c"])])
def test_class_naming_parameters_mean_what_they_mean_in_scikit_learn(labels):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = labels[np.arange(120) % 3]
    X[y == labels[1], 0] += 1.0
    cv = StratifiedKFold(5)
    for model in (
        LogisticRegression(class_weight={labels[1]: 20.0}),
        sklearn_make_pipeline(StandardScaler(), LogisticRegression(class_weight={labels[2]: 5.0})),
        DummyClassifier(strategy="constant", constant=labels[1]),
    ):
        np.testing.assert_allclose(evaluate(model, X, y, cv=cv).fold_scores, cross_val_score(model, X, y, cv=cv))


def test_a_pooled_only_score_ignores_undefined_fold_scores():
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import KFold

    from xdflow import SplitterValidator

    validator = SplitterValidator(KFold(3), scoring="r2", pooling_score_weight=1.0, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate(DummyRegressor(), np.arange(6.0).reshape(3, 2), np.arange(3.0), cv=validator)
    assert result.score == pytest.approx(result.validator.oof_score_)


def test_scikit_learn_models_see_the_labels_themselves():
    """Anything inside the model that names a class (pos_label in a nested scorer, class_weight in a
    meta-estimator) means what it means in scikit-learn, and fails where scikit-learn fails."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = np.array([1, 2])[(X[:, 0] + rng.normal(size=120) > 0.3).astype(int)]
    cv = StratifiedKFold(4)
    for model in (
        GridSearchCV(LogisticRegression(), {"C": [0.01, 1.0]}, scoring="recall", cv=3),
        OneVsRestClassifier(LogisticRegression(class_weight={0: 1.0, 1: 10.0})),
        LogisticRegression(class_weight={1: 1.0, 2: 1.0, 0: 50.0}),
    ):
        np.testing.assert_allclose(
            evaluate(model, X, y, cv=cv).fold_scores,
            cross_val_score(model, X, y, cv=cv, scoring="accuracy"),
            err_msg=str(model),
        )
    labels = y.astype(str)
    with pytest.raises(Exception, match="class_weight|classes"):
        cross_val_score(LogisticRegression(class_weight={1: 20.0}), X, labels, cv=cv, error_score="raise")
    with pytest.raises(Exception, match="class_weight|classes"):
        evaluate(LogisticRegression(class_weight={1: 20.0}), X, labels, cv=cv)


def test_windows_of_one_trial_stay_together_with_integer_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.repeat(rng.integers(1, 3, 25), 4)  # 4 windows per trial, integer labels
    windows = pd.DataFrame({"orig_trial": np.repeat(np.arange(25), 4)})
    result = evaluate(LogisticRegression(), X, y, metadata=windows)
    assert "orig_trial kept together" in result.cv_description
    folds = result.predictions.groupby("orig_trial")["fold"].nunique()
    assert (folds == 1).all()


@pytest.mark.parametrize("name", ["voting", "bagging", "ovr", "calibrated", "grid", "unused_key", "constant_array"])
def test_meta_estimators_score_like_scikit_learn(name):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import BaggingClassifier, VotingClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.multiclass import OneVsRestClassifier

    X = np.random.default_rng(1).normal(size=(90, 3))
    y = np.tile([1, 2, 3], 30)
    X[:, 0] += 0.5 * y
    model = {
        "voting": VotingClassifier([("a", LogisticRegression(class_weight={1: 10})), ("b", LogisticRegression())]),
        "bagging": BaggingClassifier(LogisticRegression(class_weight={1: 10}), n_estimators=3, random_state=0),
        "ovr": OneVsRestClassifier(LogisticRegression(class_weight={1: 10})),
        "calibrated": CalibratedClassifierCV(LogisticRegression(class_weight={2: 10}), cv=2, ensemble=False),
        "grid": GridSearchCV(LogisticRegression(), {"class_weight": [{2: 10}]}, cv=2),
        "unused_key": LogisticRegression(class_weight={1: 1, 2: 1, 3: 1, 0: 10}),
        "constant_array": DummyClassifier(strategy="constant", constant=[2]),
    }[name]
    cv = StratifiedKFold(3)
    expected = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    np.testing.assert_allclose(evaluate(model, X, y, cv=cv).fold_scores, expected)


def test_epochs_metadata_series_is_matched_by_position():
    mne = pytest.importorskip("mne")

    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    events = np.c_[np.arange(6) * 100, np.zeros(6, int), np.ones(6, int)]
    epochs = mne.EpochsArray(
        np.random.default_rng(0).normal(size=(6, 2, 50)),
        info,
        events=events,
        metadata=pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]}),
        verbose=False,
    )[[2, 0, 1, 5, 3, 4]]
    from xdflow.interop import _as_container

    container = _as_container(epochs, y=epochs.metadata["target"])
    np.testing.assert_array_equal(container.data.y.values, epochs.metadata["target"].to_numpy())


def test_a_stacking_classifier_with_the_default_final_estimator_is_accepted():
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import StratifiedKFold

    X = np.random.default_rng(0).normal(size=(60, 3))
    y = np.arange(60) % 2
    X[y == 1, 0] += 1
    model = StackingClassifier([("lr", LogisticRegression())])
    cv = StratifiedKFold(3)
    np.testing.assert_allclose(
        evaluate(model, X, y, cv=cv).fold_scores, cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    )


def test_a_dataframe_with_non_numeric_columns_is_refused():
    X = pd.DataFrame({"a": np.arange(12.0), "kind": pd.Categorical(list("xyz") * 4)})
    with pytest.raises(TypeError, match="kind"):
        evaluate(LogisticRegression(), X, np.arange(12) % 2, cv=3)


def test_many_classes_are_cross_validated():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(150), 6)
    X = rng.normal(size=(900, 2)) + y[:, None]
    assert len(evaluate(LogisticRegression(max_iter=50), X, y, cv=3).fold_scores) == 3


def test_metric_functions_that_need_scores_are_refused(model, X, y):
    from sklearn.metrics import roc_auc_score

    with pytest.raises(ValueError, match="roc_auc_score"):
        evaluate(model, X, y, cv=3, scoring=roc_auc_score)


@pytest.mark.parametrize("name", ["fold", "y_pred", "y_true"])
def test_metadata_columns_named_like_result_columns_are_refused(model, X, y, name):
    with pytest.raises(ValueError, match=name):
        evaluate(model, X, y, cv=3, metadata=pd.DataFrame({name: np.arange(120) % 5}))


def test_an_ensemble_of_members_that_reorder_trials_keeps_predictions_on_their_trials():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    from xdflow import EnsemblePredictor, Pipeline
    from xdflow.transforms.function_step import step as make_step

    @make_step
    def reverse(data):
        return data.isel(trial=slice(None, None, -1))

    labels = np.array(["a", "b"])[np.arange(60) % 2]
    X = (labels == "b").astype(float)[:, None] + np.zeros((60, 2))

    def member():
        knn = SKLearnPredictor(KNeighborsClassifier, sample_dim="trial", target_coord="y", n_neighbors=1)
        return Pipeline("m", [("reverse", reverse()), ("knn", knn)])

    ensemble = EnsemblePredictor([("one", member()), ("two", member())], sample_dim="trial", target_coord="y")
    result = evaluate(Pipeline("ens", [("ens", ensemble)]), X, labels, cv=StratifiedKFold(3))
    np.testing.assert_allclose(result.fold_scores, 1.0)


def test_by_reports_undefined_metrics_as_nan():
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold

    y = np.arange(12) % 2
    result = evaluate(
        DummyClassifier(strategy="constant", constant=0),
        np.zeros((12, 2)),
        y,
        cv=StratifiedKFold(3),
        scoring="precision",
        metadata=pd.DataFrame({"subject": np.repeat([0, 1], 6)}),
    )
    assert result.by("subject")["precision"].isna().all()  # no positive predictions: precision is undefined
