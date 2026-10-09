"""Tests for cross-validator scoring behavior."""

from functools import partial

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator, SampledDomainKFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _create_eval_data(task: str = "classification") -> DataContainer:
    n_trials = 90
    n_features = 6
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_trials, n_features))

    if task == "classification":
        target = rng.choice(["A", "B", "C"], size=n_trials)
    else:
        target = rng.normal(loc=0.0, scale=1.0, size=n_trials)

    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "stimulus": ("trial", target),
            "session": ("trial", rng.choice(["s1", "s2", "s3"], size=n_trials)),
            "event_type": ("trial", ["stim"] * n_trials),
        },
    )
    return DataContainer(da)


def test_auto_metric_selection_classification():
    data = _create_eval_data(task="classification")
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="stimulus",
        max_iter=200,
    )
    pipeline = Pipeline("clf_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3)
    cv.pipeline = pipeline

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "f1_weighted"
    assert cv.oof_score_ == cv.oof_f1_score_


def test_cross_validate_does_not_accumulate_history_on_input():
    data = _create_eval_data(task="classification")
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="stimulus",
        max_iter=200,
    )
    pipeline = Pipeline("clf_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3)
    cv.pipeline = pipeline

    for _ in range(3):
        cv.cross_validate(data, verbose=False)

    assert data.data.attrs["data_history"] == []


def test_auto_metric_selection_regression():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="stimulus",
        is_classifier=False,
        alpha=1.0,
    )
    pipeline = Pipeline("reg_pipeline", [("reg", reg)])
    cv = KFoldValidator(n_splits=3)
    cv.pipeline = pipeline

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "r2"
    assert np.isfinite(cv.oof_score_)


def test_custom_string_metrics():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="stimulus", is_classifier=False)
    pipeline = Pipeline("metrics_pipeline", [("reg", reg)])

    cv_mse = KFoldValidator(n_splits=3, scoring="mse")
    cv_mse.pipeline = pipeline
    cv_mse.cross_validate(data, verbose=False)
    assert cv_mse.metric_name_ == "mse"

    cv_rmse = KFoldValidator(n_splits=3, scoring="rmse")
    cv_rmse.pipeline = pipeline
    cv_rmse.cross_validate(data, verbose=False)
    assert cv_rmse.metric_name_ == "rmse"
    assert abs((abs(cv_rmse.oof_score_) ** 2) - abs(cv_mse.oof_score_)) < 0.05

    cv_mae = KFoldValidator(n_splits=3, scoring="mae")
    cv_mae.pipeline = pipeline
    cv_mae.cross_validate(data, verbose=False)
    assert cv_mae.metric_name_ == "mae"


def test_container_aware_scorer():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="stimulus", is_classifier=False)
    pipeline = Pipeline("custom_pipeline", [("reg", reg)])

    called = {"count": 0}

    def container_metric(y_true, y_pred, container):
        assert container is not None
        assert "trial" in container.data.dims
        called["count"] += 1
        return -np.mean(np.abs(y_true - y_pred))

    cv = KFoldValidator(n_splits=3, scoring=container_metric)
    cv.pipeline = pipeline
    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "custom"
    assert called["count"] > 0


def test_stratify_coord_runs():
    data = _create_eval_data(task="classification")
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="stimulus",
        max_iter=200,
    )
    pipeline = Pipeline("stratify_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3, stratify_coord="stimulus")
    cv.pipeline = pipeline

    score = cv.cross_validate(data, verbose=False)

    assert np.isfinite(score)
    assert cv.stratify_coord == "stimulus"


def _create_multilabel_data() -> DataContainer:
    n_trials = 90
    rng = np.random.default_rng(123)
    target_a = np.tile([0, 1], n_trials // 2)
    target_b = np.tile([0, 0, 1], n_trials // 3)
    data = np.column_stack(
        [
            target_a + rng.normal(scale=0.15, size=n_trials),
            target_b + rng.normal(scale=0.15, size=n_trials),
            rng.normal(size=n_trials),
        ]
    )
    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(data.shape[1]),
            "target__a": ("trial", target_a),
            "target__b": ("trial", target_b),
            "event_type": ("trial", ["stim"] * n_trials),
        },
    )
    return DataContainer(da)


def test_multilabel_probability_scoring():
    data = _create_multilabel_data()
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord=["target__a", "target__b"],
        is_classifier=True,
        is_multilabel=True,
        max_iter=200,
    )
    pipeline = Pipeline("multilabel_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3, scoring="ap_macro", verbose=False)
    cv.pipeline = pipeline

    score = cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "ap_macro"
    assert np.isfinite(score)
    assert len(cv.oof_probabilities_) == 3
    assert np.concatenate(cv.oof_probabilities_).shape == (90, 2)


def test_sampled_domain_kfold_splits_target_validation_and_sampled_target_training():
    n_source = 20
    n_target = 30
    n_trials = n_source + n_target
    rng = np.random.default_rng(123)
    labels = np.tile(["a", "b"], n_trials // 2)
    da = xr.DataArray(
        rng.normal(size=(n_trials, 4)),
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(4),
            "stimulus": ("trial", labels),
            "domain": ("trial", np.array(["source"] * n_source + ["target"] * n_target)),
        },
    )
    data = DataContainer(da)
    clf = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus", max_iter=200)
    pipeline = Pipeline("domain_pipeline", [("clf", clf)])
    cv = SampledDomainKFoldValidator(
        domain_coord="domain",
        target_domains="target",
        label_coord="stimulus",
        default_samples_per_label=1,
        n_splits=3,
        verbose=False,
    )
    cv.pipeline = pipeline

    splits = list(cv._get_splits(data, data.data.trial.values))

    assert len(splits) == 3
    for train_indices, validation_indices in splits:
        train_domains = data.data.sel(trial=train_indices).coords["domain"].values
        validation_domains = data.data.sel(trial=validation_indices).coords["domain"].values
        assert set(validation_domains) == {"target"}
        assert np.sum(train_domains == "source") == n_source
        assert np.sum(train_domains == "target") == 2


def _clf_cv(scoring) -> KFoldValidator:
    clf = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus", max_iter=200)
    cv = KFoldValidator(n_splits=3, test_size=0.2, scoring=scoring, use_stateful_fit_cache=False, verbose=False)
    cv.pipeline = Pipeline("clf_pipeline", [("clf", clf)])
    return cv


@pytest.mark.parametrize(
    "scoring",
    [accuracy_score, partial(f1_score, average="macro"), "accuracy"],
    ids=["sklearn_metric", "partial_metric", "accuracy_string"],
)
def test_sklearn_metrics_score_without_container(scoring):
    """Keyword-only params (sklearn's sample_weight, ...) are not a third positional slot for the container."""
    data = _create_eval_data(task="classification")
    cv = _clf_cv(scoring)

    score = cv.cross_validate(data, verbose=False)
    holdout = cv.score_on_holdout(data)

    assert 0.0 <= score <= 1.0
    assert 0.0 <= holdout <= 1.0


def test_keyword_only_container_scorer_receives_container():
    data = _create_eval_data(task="classification")
    seen = []

    def scorer(y_true, y_pred, *, container):
        seen.append(container.data.sizes["trial"])
        return float(accuracy_score(y_true, y_pred))

    cv = _clf_cv(scorer)
    cv.cross_validate(data, verbose=False)

    assert len(seen) == 3 and all(n == len(y) for n, y in zip(seen, cv.true_labels_))


def test_duplicate_trial_labels_are_rejected_upfront():
    data = _create_eval_data(task="classification")
    duplicated = DataContainer(data.data.assign_coords(trial=np.zeros(data.data.sizes["trial"], dtype=int)))

    with pytest.raises(ValueError, match="Trial labels must be unique"):
        _clf_cv("f1_weighted").cross_validate(duplicated, verbose=False)


def test_switching_from_container_scorer_to_builtin_resets_calling_mode():
    cv = _clf_cv(lambda y_true, y_pred, container: 1.0)
    cv._get_scoring_func()
    cv.scoring = "accuracy"
    scoring_func, _, _ = cv._get_scoring_func()

    assert cv._call_scoring_func(scoring_func, np.arange(3), np.arange(3), None) == 1.0


def test_confusion_matrix_keeps_evaluated_classes_after_finalize():
    data = _create_eval_data(task="classification")
    cv = _clf_cv("f1_weighted")
    cv.cross_validate(data, verbose=False)
    before = cv.oof_confusion_matrix_.sum()
    dropped_class = data.data["stimulus"].values[0]

    cv.finalize_pipeline(DataContainer(data.data.isel(trial=data.data["stimulus"].values != dropped_class)))

    assert cv.oof_confusion_matrix_.sum() == before


def test_scoring_resolves_predictor_inside_unfitted_group_apply():
    from xdflow.composite import GroupApplyTransform

    data = _create_eval_data(task="classification")
    per_session = GroupApplyTransform(
        "session", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus", max_iter=200)
    )
    cv = KFoldValidator(n_splits=3, test_size=0.2, scoring="accuracy", verbose=False)
    cv.set_pipeline(Pipeline("per_session", [("model", per_session)]))

    score = cv.cross_validate(data, verbose=False)
    holdout = cv.score_on_holdout(data)

    assert cv.metric_name_ == "accuracy"
    assert 0.0 <= score <= 1.0 and 0.0 <= holdout <= 1.0


def test_oof_predictions_container_keeps_trial_coords():
    data = _create_eval_data(task="classification")
    cv = _clf_cv("f1_weighted")
    cv.cross_validate(data, verbose=False)

    oof = cv.oof_predictions_container_.data

    np.testing.assert_array_equal(oof.values, np.concatenate(cv.oof_predictions_))
    np.testing.assert_array_equal(oof["stimulus"].values, np.concatenate(cv.true_labels_))
    assert oof["trial"].to_index().is_unique
    assert set(np.unique(oof["fold"].values)) == set(range(len(cv.cv_scores_)))
    # trial ids map back to the input data
    np.testing.assert_array_equal(data.data.sel(trial=oof["trial"].values)["stimulus"].values, oof["stimulus"].values)
