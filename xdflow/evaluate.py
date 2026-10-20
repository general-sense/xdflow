"""One-call cross-validation of a model on structured data."""

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier as is_sklearn_classifier
from sklearn.base import is_regressor as is_sklearn_regressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.multiclass import check_classification_targets

from xdflow.composite import Pipeline
from xdflow.composite.pipeline import make_pipeline
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator
from xdflow.cv.kfold import KFoldValidator
from xdflow.cv.splitter import SplitterValidator
from xdflow.diagnostics import _NOTHING_TO_AUDIT, SplitReport, audit_splits, grouping_coords
from xdflow.interop import _as_container
from xdflow.transforms.sklearn_transform import SKLearnPredictor


@dataclass(repr=False)
class EvaluationResult:
    """Cross-validation results, with every scored trial's prediction and metadata.

    Attributes:
        score: Mean score over folds (for a validator passed as ``cv``, its ``score_``, which blends in the
            pooled score when its ``pooling_score_weight`` is set).
        metric: Name of the metric.
        fold_scores: Score of each fold.
        predictions: One row per scored trial: its trial coordinates (labels, subject, session...), ``y_true``,
            ``y_pred`` and ``fold``.
        split: What the folds share between training and test sets (subjects, sessions, duplicate samples).
        cv_description: The split strategy in words.
        validator: The cross-validator that ran, for advanced use (fold splits, confusion matrices...).
        pipeline: The evaluated (unfitted) xdflow pipeline.
    """

    score: float
    metric: str
    fold_scores: list[float]
    predictions: pd.DataFrame
    split: SplitReport | None
    cv_description: str
    validator: CrossValidator
    pipeline: Pipeline
    groups: str | None = None
    _data: DataContainer | None = field(default=None, repr=False)

    def by(self, coord: str) -> pd.DataFrame:
        """Score per value of a trial coordinate (e.g. ``"subject"`` or ``"session"``), from out-of-fold
        predictions. Groups whose metric is undefined (e.g. a single class for some metrics) get NaN."""
        if coord not in self.predictions.columns:
            raise ValueError(f"'{coord}' is not a trial coordinate; available: {list(self.predictions.columns)}.")
        scoring_func, _, needs_proba = self.validator._get_scoring_func()
        metric = self.metric
        if needs_proba or self.validator._scoring_accepts_container:
            raise ValueError(f"by() needs a metric computed from predicted labels; '{metric}' is not.")
        rows = []
        for value, group in self.predictions.groupby(coord, sort=True, dropna=False):
            y_true, y_pred = np.stack(group["y_true"].to_numpy()), np.stack(group["y_pred"].to_numpy())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    score = float(scoring_func(y_true, y_pred))
                except ValueError:
                    score = np.nan
            if any(issubclass(w.category, UndefinedMetricWarning) for w in caught):
                score = np.nan  # scikit-learn's fallback (e.g. precision 0 with no positive prediction) is not a score
            rows.append({coord: value, metric: score, "n_trials": len(group)})
        return pd.DataFrame(rows).set_index(coord)

    def fit_final(self, verbose: bool = False) -> Pipeline:
        """Fit the pipeline on all the data, for prediction on new data."""
        if self._data is None:
            raise ValueError("The evaluated data is not available.")
        return self.validator.finalize_pipeline(self._data, verbose=verbose)

    def __repr__(self) -> str:
        stateless, stateful = self.validator._auto_detect_pipeline_parts(self.pipeline)
        once = ", ".join(_label(s) for s in stateless.steps) if stateless is not None else "nothing"
        per_fold = ", ".join(_label(s) for s in stateful.steps) if stateful is not None else "nothing"
        # Test sets may overlap (ShuffleSplit): count trials once.
        n_trials = self.predictions["trial"].nunique() if "trial" in self.predictions else len(self.predictions)
        scored = f"{n_trials} trials" + (
            f", {len(self.predictions)} predictions" if n_trials < len(self.predictions) else ""
        )
        lines = [
            f"{self.metric} {np.mean(self.fold_scores):.3f} ± {np.std(self.fold_scores):.3f} "
            f"(mean ± sd over {len(self.fold_scores)} folds, {scored})",
            f"Split: {self.cv_description}",
            f"Computed once: {once}. Refit in each fold: {per_fold}.",
        ]
        if self.groups is not None:
            table = self.by(self.groups)
            lines += ["", f"By {self.groups}:", table.round(3).to_string(max_rows=12)]
        if self.split is not None:
            checked = ", ".join(self.split.table["check"]) if not self.split.table.empty else "nothing"
            clean = (
                f"nothing shared between training and test sets ({checked})."
                if checked != "identical samples"
                else _NOTHING_TO_AUDIT
            )
            lines += [""] + (
                [f"Split check: {clean}"]
                if self.split.ok
                else ["Split check:"] + [f"  - {finding}" for finding in self.split.findings]
            )
        return "\n".join(lines)


def evaluate(
    model: Any,
    X: Any,
    y: Any = None,
    *,
    groups: Any = None,
    cv: Any = 5,
    scoring: Any = None,
    metadata: pd.DataFrame | None = None,
    dims: tuple[str, ...] | None = None,
    random_state: int = 0,
    cache: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """Cross-validate a model and report the score per fold, per group and per trial.

    Steps that learn nothing from data run once on all trials; the rest are refit inside each training fold,
    so nothing from a test fold reaches the model that scores it.

    Args:
        model: A scikit-learn estimator or ``Pipeline`` (used as is), an xdflow ``Pipeline`` or predictor, or a
            list of steps: xdflow transforms (e.g. filters, ``@step`` functions) followed by a final
            scikit-learn estimator or xdflow predictor.
        X: MNE ``Epochs``, ``xr.DataArray``, ``DataContainer``, or a NumPy array with samples first.
        y: Target: a trial coordinate name, or one value per sample. Defaults to the xdflow pipeline's target,
            or the ``"label"`` coordinate (as made by ``from_mne_epochs``).
        groups: Trial coordinate name (``"subject"``), or one value per sample. Trials of a group are never
            split between training and test sets: the score is for new groups (e.g. new subjects).
        cv: Number of folds; or a scikit-learn splitter or ``(train, test)`` index pairs; or an xdflow
            validator. With ``groups`` and an int, each group is held out in turn when there are no more
            groups than folds, otherwise groups are split into ``cv`` folds stratified by class.
        scoring: A scikit-learn scoring name computed from predictions (``"accuracy"``, ``"balanced_accuracy"``,
            ``"f1_macro"``, ``"r2"``, ``"neg_mean_absolute_error"``...) or scorer (``make_scorer(...)``), with
            scikit-learn's meaning, or a callable ``(y_true, y_pred) -> float``. Defaults to accuracy for
            classifiers and R² for regressors.
        metadata: DataFrame with one row per sample; its columns become trial coordinates usable as ``y``
            and ``groups``.
        dims: Dim names for NumPy input, starting with ``"trial"`` (e.g. ``("trial", "channel", "time")``).
        random_state: Seed for fold shuffling.
        cache: Reuse per-fold fits cached on disk from earlier identical runs (fits that depend on functions
            from your own code are recomputed unless ``XDFLOW_CACHE_USER_CODE=1``). A reused fit of an
            estimator without ``random_state`` repeats one random draw, so leave it off for seed studies.
        verbose: Print progress.

    Returns:
        An ``EvaluationResult``; print it for a summary, or use ``.by("subject")`` and ``.predictions``.

    Example:
        >>> result = evaluate(make_pipeline(CSP(6, log=True), LinearDiscriminantAnalysis()), epochs, groups="subject")
    """
    if cv is None:
        cv = 5  # scikit-learn's default
    # Target values given as an array are named after what an xdflow model predicts ("label"...), else "y".
    model_target = model.final_target_coord if isinstance(model, Pipeline) else getattr(model, "target_coord", None)
    y_coord = model_target if isinstance(model, (Pipeline, Predictor)) and isinstance(model_target, str) else "y"
    container = _as_container(
        X,
        dims=dims,
        metadata=metadata,
        **{y_coord: None if isinstance(y, str) else y},
        group=None if isinstance(groups, str) else groups,
    )
    data = container.data
    reserved = sorted({"fold", "y_true", "y_pred"} & set(_trial_coords(data)))
    if reserved:  # they would be overwritten in result.predictions and by()
        raise ValueError(f"Trial coordinates {reserved} are names evaluate uses for its results; rename them.")
    y_name = y if isinstance(y, str) else (y_coord if y is not None else None)
    groups_name = groups if isinstance(groups, str) else ("group" if groups is not None else None)

    pipeline = _as_pipeline(model, y_name, data)
    target = pipeline.final_target_coord
    targets = target if isinstance(target, list) else [target]
    for name in [*targets, groups_name]:
        if name is not None and (name not in data.coords or data.coords[name].dims != ("trial",)):
            raise ValueError(f"'{name}' must be a trial coordinate; trial coordinates: {_trial_coords(data)}.")
    predictor = pipeline.predictive_transform
    is_classifier = bool(getattr(predictor, "is_classifier", False))
    if is_classifier and isinstance(target, str) and not getattr(predictor, "is_multilabel", False):
        check_classification_targets(data.coords[target].values)  # as scikit-learn does: no continuous or NaN labels

    # scikit-learn scorer objects (make_scorer, get_scorer) score like their names; other callables are metrics.
    is_scorer = callable(scoring) and type(scoring).__module__.startswith("sklearn.metrics")
    if (
        not is_scorer
        and getattr(scoring, "__name__", None) in _SCORE_METRICS
        and str(getattr(scoring, "__module__", "")).startswith("sklearn.metrics")
    ):  # a (y_true, y_pred) callable receives predicted labels, which these metrics would silently accept
        raise ValueError(
            f"scoring={scoring.__name__} needs probabilities or decision scores; evaluate scores predicted labels. "
            "Pass its name (e.g. 'roc_auc') to see why, or use a validator with scoring_needs_proba=True."
        )
    metric: str | None = scoring if isinstance(scoring, str) else getattr(scoring, "__name__", None)
    if is_scorer:
        metric = getattr(getattr(scoring, "_score_func", None), "__name__", "custom")
        if getattr(scoring, "_sign", 1) < 0:  # greater_is_better=False: scores are negated, as in "neg_..." names
            metric = f"neg_{metric}"
    if isinstance(cv, CrossValidator):
        metric = None  # the validator's own scoring applies (passing scoring as well is rejected below)
    elif scoring is None and isinstance(target, str):
        metric = "accuracy" if is_classifier else "r2"
    if isinstance(metric, str) and (scoring is None or isinstance(scoring, str) or is_scorer):
        # Classes come from all labels, as an estimator's classes_ come from its training labels: a fold whose
        # trials are all of one class still scores like scikit-learn (e.g. binary F1 of 0, not an error).
        classes = np.unique(data.coords[target].values) if is_classifier and isinstance(target, str) else None
        scoring = _sklearn_metric(scoring if is_scorer else metric, is_classifier, classes, name=metric)
    validator, description = _as_validator(
        cv, groups_name, data, is_classifier, target, scoring, random_state, cache, verbose
    )
    validator.set_pipeline(pipeline)
    validator.cross_validate(container, verbose=verbose)

    split = _audit(validator, data, groups_name)
    if groups_name is not None and split is not None and groups_name in split.shared_coords:
        warnings.warn(
            f"The split ({description}) puts trials of the same '{groups_name}' in training and test sets. Use a "
            "group splitter (GroupKFold, LeaveOneGroupOut...) or an int cv.",
            UserWarning,
            stacklevel=2,
        )
    if groups_name is None and split is not None:
        # Warn only about units the data could be split by (a single subject is shared by every split).
        splittable = [name for name in split.shared_coords if name in grouping_coords(data, min_values=2)]
        if splittable:
            finding = split.findings[split.shared_coords.index(splittable[0])]
            hint = (
                f" Use a validator that keeps each {splittable[0]} on one side, e.g. "
                f"SplitterValidator(GroupKFold(5), group_coord='{splittable[0]}'),"
                if isinstance(cv, CrossValidator)
                else f" Pass groups='{splittable[0]}'"
            ) + f" to score on new {splittable[0]} values instead."
            warnings.warn(finding + hint, UserWarning, stacklevel=2)

    predictions = _predictions_frame(validator)
    return EvaluationResult(
        score=float(validator.score_),
        metric=metric or validator.metric_name_,
        fold_scores=[float(score) for score in validator.cv_scores_],
        predictions=predictions,
        split=split,
        cv_description=description,
        validator=validator,
        pipeline=pipeline,
        groups=groups_name,
        _data=container,
    )


class _Predictions(BaseEstimator):
    """Stands in for a fitted estimator so scikit-learn's scorers can score predictions already made."""

    def __init__(self, y_pred: Any = None, classes: Any = None):
        self.y_pred = y_pred
        self.classes = classes
        self.classes_ = classes

    def predict(self, X: Any) -> Any:
        return self.y_pred


class _PredictedLabels(ClassifierMixin, _Predictions):
    pass


class _PredictedValues(RegressorMixin, _Predictions):
    pass


_SCORE_METRICS = {"roc_auc_score", "log_loss", "average_precision_score", "brier_score_loss", "top_k_accuracy_score"}


def _sklearn_metric(scoring: Any, is_classifier: bool, classes: Any = None, name: str | None = None) -> Any:
    """A ``(y_true, y_pred) -> float`` metric with the meaning scikit-learn gives the scoring name or scorer."""
    scorer = get_scorer(scoring)  # raises, listing the valid names, for an unknown one
    name = name or str(scoring)

    def score(y_true: Any, y_pred: Any, known: Any) -> float:
        if is_classifier:
            seen = np.unique(np.concatenate([np.ravel(y_true), np.ravel(y_pred)]))
            stand_in = _PredictedLabels(y_pred, seen if known is None else np.union1d(known, seen))
        else:
            stand_in = _PredictedValues(y_pred)
        return float(scorer(stand_in, None, y_true))

    def metric(y_true: Any, y_pred: Any) -> float:  # two arguments: the validator passes no container
        return score(y_true, y_pred, classes)

    try:  # metrics that need probabilities or decision scores ask the stand-in for them
        score(np.array([0, 1, 1]), np.array([0, 1, 0]), None) if is_classifier else score(np.zeros(3), np.ones(3), None)
    except AttributeError as exc:
        raise ValueError(
            f"scoring='{name}' needs probabilities or decision scores; evaluate scores predicted labels. Use a "
            "validator with scoring_needs_proba=True and pass it as cv."
        ) from exc
    except ValueError:
        pass  # e.g. a positive label absent from the probe values; real labels are checked when scoring
    metric.__name__ = name
    return metric


def _label(step: Any) -> str:
    """Step name, plus the inner steps of a wrapped scikit-learn Pipeline: "pipeline (csp, lda)"."""
    estimator = getattr(getattr(step.transform, "_base_estimator_cls", None), "estimator", None)
    inner = getattr(estimator, "steps", None)
    return f"{step.name} ({', '.join(name for name, _ in inner)})" if inner else step.name


def _trial_coords(data) -> list[str]:
    return [str(name) for name, coord in data.coords.items() if coord.dims == ("trial",) and name != "trial"]


def _as_pipeline(model: Any, y_name: str | None, data) -> Pipeline:
    if isinstance(model, Pipeline) or isinstance(model, Predictor):
        pipeline = model if isinstance(model, Pipeline) else Pipeline("model", [("model", model)])
        if not pipeline.is_predictor:
            raise ValueError("The xdflow pipeline must end with a predictor.")
        target = pipeline.final_target_coord
        if y_name is not None and target != y_name:
            raise ValueError(f"The model predicts '{target}' but y is '{y_name}'; leave y unset or make them match.")
        return pipeline
    if isinstance(model, Transform):
        raise ValueError(f"{type(model).__name__} is not a predictor; end the model with an estimator.")

    if y_name is None:
        if "label" not in data.coords:
            raise ValueError(
                f"Pass y: a trial coordinate name or one value per sample. Trial coordinates: {_trial_coords(data)}."
            )
        y_name = "label"
    steps = list(model) if isinstance(model, (list, tuple)) else [model]
    if not steps:
        raise ValueError("model is an empty list.")
    *head, last = steps
    for item in head:
        if not isinstance(item, Transform):
            raise TypeError(
                f"Only the last step of a model list can be a scikit-learn estimator; got {type(item).__name__} "
                "earlier. Put scikit-learn preprocessing inside the final estimator's Pipeline."
            )
    if not isinstance(last, Transform):
        # predict may appear only once fitted (StackingClassifier): the estimator type says it is a model
        if not (
            hasattr(last, "fit")
            and (hasattr(last, "predict") or is_sklearn_classifier(last) or is_sklearn_regressor(last))
        ):
            raise TypeError(f"Expected a scikit-learn estimator or xdflow predictor, got {type(last).__name__}.")
        # Fit as cross_val_score would: on the labels themselves, and without a sample_weight coordinate.
        last = SKLearnPredictor.from_estimator(
            last, sample_dim="trial", target_coord=y_name, sample_weight_coord=None, fit_on_labels=True
        )
    elif isinstance(last, Predictor) and getattr(last, "target_coord", y_name) != y_name:
        raise ValueError(f"The final predictor predicts '{last.target_coord}' but y is '{y_name}'.")
    if not head and isinstance(last, Pipeline):
        return last
    return make_pipeline(*head, last, name="model")


def _as_validator(
    cv: Any,
    groups_name: str | None,
    data,
    is_classifier: bool,
    target: Any,
    scoring: Any,
    random_state: int,
    cache: bool,
    verbose: bool,
) -> tuple[CrossValidator, str]:
    if isinstance(cv, CrossValidator):
        if groups_name is not None or scoring is not None:
            raise ValueError("groups and scoring come from the validator you passed; set them there instead.")
        # A copy, so reusing cv cannot change this result; shallow, as cross_validate rebinds every result.
        return copy.copy(cv), type(cv).__name__

    if scoring is None and isinstance(target, str):
        scoring = "accuracy" if is_classifier else "r2"
    options: dict[str, Any] = dict(scoring=scoring, use_stateful_fit_cache=cache, verbose=verbose)

    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, (int, np.integer)):
        description = f"{cv!r}" + (f", groups = {groups_name}" if groups_name else "")
        return SplitterValidator(cv, group_coord=groups_name, **options), description

    n_splits = int(cv)
    if n_splits < 2:
        raise ValueError(f"cv must be at least 2 folds; got {n_splits}.")
    if groups_name is None:
        stratified = "stratified by class, " if is_classifier and isinstance(target, str) else ""
        validator = KFoldValidator(n_splits=n_splits, shuffle=True, random_state=random_state, **options)
        windows = (
            ", windows of one orig_trial kept together"
            if validator._resolve_orig_trial_groups(data) is not None
            else ""
        )
        return validator, f"{n_splits}-fold, {stratified}trials shuffled{windows}"

    n_groups = len(pd.unique(np.asarray(data.coords[groups_name].values)))
    if n_groups < 2:
        raise ValueError(f"groups '{groups_name}' has {n_groups} value; at least 2 are needed to hold one out.")
    grouped = (
        f"grouped ({n_groups} groups)" if groups_name == "group" else f"grouped by {groups_name} ({n_groups} values)"
    )
    if n_groups <= n_splits:
        splitter, description = LeaveOneGroupOut(), f"leave one {groups_name} out ({n_groups} folds)"
    elif is_classifier and isinstance(target, str):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state)
        description = f"{n_splits}-fold, {grouped}, stratified by class"
    else:
        splitter = GroupKFold(n_splits, shuffle=True, random_state=random_state)
        description = f"{n_splits}-fold, {grouped}"
    return SplitterValidator(splitter, group_coord=groups_name, **options), description


def _audit(validator: CrossValidator, data, groups_name: str | None) -> SplitReport | None:
    coords = grouping_coords(data)
    if groups_name and groups_name not in coords:
        coords.append(groups_name)
    try:
        samples = data.transpose("trial", ...).values.reshape(data.sizes["trial"], -1)
        return audit_splits(data, validator.split_trials_, coords, samples=samples)
    except ValueError:  # a stateless step renamed or created trials; the splits cannot be mapped back
        return None


def _predictions_frame(validator: CrossValidator) -> pd.DataFrame:
    predictions = validator.oof_predictions_container_.data
    frame = pd.DataFrame(
        {
            str(name): coord.values
            for name, coord in predictions.coords.items()
            if coord.dims == ("trial",) and name != "fold"
        }
    )
    y_true, y_pred = np.concatenate(validator.true_labels_), np.concatenate(validator.oof_predictions_)
    frame["y_true"] = list(y_true) if y_true.ndim > 1 else y_true
    frame["y_pred"] = list(y_pred) if y_pred.ndim > 1 else y_pred
    frame["fold"] = predictions.coords["fold"].values
    return frame
