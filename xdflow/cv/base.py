"""
Cross-validation framework for pipeline evaluation.
"""

import gc
import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import xarray as xr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xdflow.composite import Pipeline
from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.transforms import Predictor
from xdflow.utils.cache_utils import (
    _get_object_metadata,
    _installed_code_fingerprint,
    cache_user_code,
    get_module_hashes_for_object,
    references_user_code,
    save_cached_object,
    try_load_cached_object,
)
from xdflow.utils.target_utils import extract_target_array
from xdflow.utils.visualizations import plot_confusion_matrix


def _scorer_container_mode(scorer: Callable) -> str:
    """Return how a scorer accepts the optional validation container: "positional", "keyword" or "none".

    Keyword-only parameters such as sklearn's ``sample_weight`` do not count as a third positional slot,
    so ``accuracy_score`` or ``partial(f1_score, average="macro")`` are called as ``(y_true, y_pred)``.
    """
    sig = getattr(scorer, "__signature__", None) or inspect.signature(scorer)
    params = list(sig.parameters.values())
    positional_params = [
        p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional_params) >= 3 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return "positional"
    if any(
        p.name == "container" and p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    ) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return "keyword"
    return "none"


def _make_multioutput_scorer(metric_func, name, *, negate: bool = False):
    def _scorer(y_true, y_pred):
        base = metric_func(y_true, y_pred, multioutput="uniform_average")
        return -base if negate else base

    return lambda: (_scorer, name, False)


def _cleanup_torch_memory() -> None:
    """Best-effort cleanup of PyTorch CUDA caches after a fold."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def iter_predictors(transform: Transform | None) -> Iterator[Predictor]:
    """Yield all Predictor instances within a transform recursively."""
    from xdflow.composite import CompositeTransform
    from xdflow.transforms import Transform as TransformType

    if transform is None:
        return

    if isinstance(transform, Predictor):
        yield transform
        return

    if isinstance(transform, CompositeTransform):
        children = list(transform.children)
        # GroupApplyTransform's children exist only after fit; its template is what the groups are cloned from
        template = getattr(transform, "transform_template", None)
        if isinstance(template, TransformType) and not any(child is template for child in children):
            children.append(template)
        for child in children:
            if isinstance(child, TransformType):
                yield from iter_predictors(child)


def fit_label_encoders(pipeline: Transform, container: DataContainer) -> None:
    """Fit provided encoders in-place for all predictors on the full dataset.

    This learns the label vocabulary globally (including holdout if present)
    without leaking feature-target relationships across folds.
    """
    for predictor in iter_predictors(pipeline):
        if not getattr(predictor, "is_classifier", False):
            continue
        if getattr(predictor, "is_multilabel", False):
            continue
        target_coord = predictor.target_coord  # required arg for predictors
        encoder = predictor.encoder  # required for classifiers, should error otherwise

        if target_coord not in container.data.coords:
            raise ValueError(
                f"The target_coord '{target_coord}' specified in a predictor was not found in the data container's coordinates."
            )
        labels = container.data.coords[target_coord].values
        unique_labels = np.unique(labels)
        if hasattr(encoder, "classes_"):
            existing = getattr(encoder, "classes_", None)
            try:
                if existing is not None and np.array_equal(existing, unique_labels):
                    continue  # already fitted with correct classes
            except Exception:
                pass
        if not hasattr(encoder, "fit"):
            raise TypeError(f"Encoder for target '{target_coord}' must have a fit method")
        encoder.fit(labels)


class CrossValidator(ABC):
    """Base class for evaluating a predictive pipeline with held-out data.

    A cross-validator owns the train/validation/holdout splitting strategy and
    evaluates a complete `Pipeline`. Stateless pipeline steps can be run once
    before fold splitting, while stateful steps are cloned and fitted on each
    fold's training data. This keeps expensive deterministic preprocessing out
    of the per-fold loop when possible.

    Results are stored on the instance after evaluation, including fold scores,
    out-of-fold predictions, optional probabilities, and holdout predictions.
    Scorers may accept either `(y_true, y_pred)` or
    `(y_true, y_pred, container)` when coordinate-aware scoring is needed.

    Subclasses define only the split policy by implementing `_split_holdout` and
    `_get_splits`.
    """

    _STRING_SCORERS = {
        "r2": _make_multioutput_scorer(r2_score, "r2"),
        "r2_score": _make_multioutput_scorer(r2_score, "r2"),
        # Error metrics return the negative value so that higher is better (compatible with maximization)
        "mse": _make_multioutput_scorer(mean_squared_error, "mse", negate=True),
        "mean_squared_error": _make_multioutput_scorer(mean_squared_error, "mse", negate=True),
        "rmse": lambda: (
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")),
            "rmse",
            False,
        ),
        "root_mean_squared_error": lambda: (
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")),
            "rmse",
            False,
        ),
        "mae": _make_multioutput_scorer(mean_absolute_error, "mae", negate=True),
        "mean_absolute_error": _make_multioutput_scorer(mean_absolute_error, "mae", negate=True),
        "f1": lambda: (lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"), "f1_weighted", False),
        "f1_weighted": lambda: (
            lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
            "f1_weighted",
            False,
        ),
        "f1_macro": lambda: (lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), "f1_macro", False),
        "f1_micro": lambda: (lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"), "f1_micro", False),
        "accuracy": lambda: (lambda y_true, y_pred: accuracy_score(y_true, y_pred), "accuracy", False),
        "f1_samples": lambda: (
            lambda y_true, y_pred: f1_score(y_true, y_pred, average="samples", zero_division=0),
            "f1_samples",
            False,
        ),
        "hamming_loss": lambda: (lambda y_true, y_pred: -hamming_loss(y_true, y_pred), "hamming_loss", False),
        "subset_accuracy": lambda: (lambda y_true, y_pred: accuracy_score(y_true, y_pred), "subset_accuracy", False),
        "ap_macro": lambda: (
            lambda y_true, y_proba: average_precision_score(y_true, y_proba, average="macro"),
            "ap_macro",
            True,
        ),
        "ap_samples": lambda: (
            lambda y_true, y_proba: average_precision_score(y_true, y_proba, average="samples"),
            "ap_samples",
            True,
        ),
        "ap_micro": lambda: (
            lambda y_true, y_proba: average_precision_score(y_true, y_proba, average="micro"),
            "ap_micro",
            True,
        ),
    }

    def __init__(
        self,
        pooling_score_weight: float = 0.0,
        use_stateful_fit_cache: bool = True,
        release_fold_memory: bool = False,
        scoring: str | Callable | None = None,
        scoring_needs_proba: bool = False,
        stratify_coord: str | None = None,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        verbose: bool = True,
    ):
        """Initialize scoring, caching, and split-independent CV options.

        Args:
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
                Must be between 0.0 and 1.0.
                Higher values give more weight to folds with more trials.
            use_stateful_fit_cache: Whether to cache stateful transforms during CV.
            release_fold_memory: Whether to aggressively release per-fold objects and
                clear PyTorch CUDA caches after each fold.
            scoring: Metric name or callable. If None, classification defaults
                to `"f1_weighted"` and regression defaults to `"r2"`. Callable
                scorers may accept `(y_true, y_pred)` or
                `(y_true, y_pred, container)`.
            scoring_needs_proba: Whether a custom scorer expects probabilities from
                predict_proba instead of hard predictions.
            exclude_intertrial_from_scoring: If True, automatically remove any trials whose
                event_type coordinate is "intertrial" from CV/holdout scoring.
            exclude_offsets_from_scoring: If True, remove trials whose time_offset_ms
                coordinate is not 0 from CV/holdout scoring.
            stratify_coord: Optional coordinate name to use for stratified splits. If set,
                holdout and CV splits will stratify on this coordinate (must be present
                in the data). For multi-target/regression tasks, this allows stratifying
                on a categorical coord such as stimulus.
            verbose: Whether to print verbose output specific to the cross-validator.
                Verbosity of transforms is separetely controlled by class-level function arguments.

        Raises:
            ValueError: If pooling_score_weight is not between 0.0 and 1.0
        """
        if not 0.0 <= pooling_score_weight <= 1.0:
            raise ValueError("pooling_score_weight must be between 0.0 and 1.0")

        self.pooling_score_weight = pooling_score_weight
        self.scoring = scoring
        self.scoring_needs_proba = bool(scoring_needs_proba)

        # Results from cross-validation
        self.cv_scores_ = []
        self.oof_predictions_ = []  # Out-of-fold predictions
        self.oof_probabilities_ = []  # Out-of-fold probabilities/scores
        self.true_labels_ = []
        self.oof_containers_: list[DataContainer] = []  # Per-fold scored predictions with their trial coords
        self.split_trials_: list[tuple[np.ndarray, np.ndarray]] = []  # (train, validation) trial labels per fold
        # Class order at evaluation time; finalize_pipeline may later refit the template's encoder
        self.oof_class_labels_: list[Any] | None = None
        self.holdout_class_labels_: list[Any] | None = None

        # Holdout test set management
        # NOTE: `holdout_trial_labels_` stores trial labels in the stateless-preprocessed space.
        #       Trials are assumed to retain their original labels, so these map directly to the raw container.
        self.holdout_trial_labels_: np.ndarray | None = None
        self.holdout_score_: float | None = None
        self.holdout_pred_labels_: np.ndarray | None = None  # Holdout test predictions
        self.holdout_probabilities_: np.ndarray | None = None  # Holdout test probabilities/scores
        self.holdout_true_labels_: np.ndarray | None = None  # Holdout test true labels
        self.holdout_container_: DataContainer | None = None  # Holdout test container (for container-aware scorers)
        self.holdout_scoring_mask_: np.ndarray | None = None  # Mask used by container-aware scorer

        # Set by the user before evaluation (avoid property setter recursion)
        self._pipeline: Pipeline | None = None
        self.final_target_coord_: str | list[str] | None = None
        self.use_stateful_fit_cache = use_stateful_fit_cache
        self.release_fold_memory = release_fold_memory
        self.stratify_coord = stratify_coord
        self.exclude_intertrial_from_scoring = exclude_intertrial_from_scoring
        self.exclude_offsets_from_scoring = exclude_offsets_from_scoring

        # Resolved scoring function (set after pipeline is known)
        self._scoring_func: Callable | None = None
        self._metric_name: str | None = None
        self._scoring_needs_proba = False
        self._scoring_accepts_container = False
        self._scoring_container_mode = "none"

        self.verbose = verbose

    @abstractmethod
    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Abstract method to perform initial split into train/validation and holdout sets.

        Args:
            container: DataContainer to split

        Returns:
            Tuple of (train_val_indices, holdout_indices)
        """
        raise NotImplementedError

    @abstractmethod
    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Abstract method to generate train/validation splits for cross-validation.

        Args:
            container: DataContainer to split
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) tuples
        """
        raise NotImplementedError

    def _snapshot_class_labels(self) -> list[Any] | None:
        """Current class order of a single-label classifier, else None."""
        final_predictor = self._get_final_predictor()
        if final_predictor is None or not final_predictor.is_classifier or final_predictor.is_multilabel:
            return None
        return final_predictor.get_labels()

    def _confusion_labels(self, evaluated_labels: list[Any] | None) -> list[Any]:
        """Class order used when the results were produced, so confusion matrices keep rows for absent classes
        and are not reshaped by a later refit of the template encoder."""
        if evaluated_labels is not None:
            return evaluated_labels
        labels = self._snapshot_class_labels()
        if labels is None:
            raise ValueError("Confusion matrix is only available for single-label classification.")
        return labels

    def _get_final_predictor(self) -> Predictor | None:
        """
        Return the final predictor from the pipeline, preferring predictive_transform when available.
        """
        predictive = getattr(self._pipeline, "predictive_transform", None)
        if predictive is not None:
            return predictive

        final_predictor = None
        for predictor in self._iter_predictors(self._pipeline):
            final_predictor = predictor
        return final_predictor

    @staticmethod
    def _validate_stratify_labels(
        labels: np.ndarray, n_splits: int | None, test_size: float | None, context: str
    ) -> None:
        """Validate that stratification labels are suitable for splitting."""
        unique, counts = np.unique(labels, return_counts=True)
        n_unique = len(unique)
        min_count = counts.min() if counts.size else 0

        if n_unique > 100:
            raise ValueError(
                f"Stratification coord in {context} has too many unique values ({n_unique}). "
                "This looks continuous; bin the coord before stratifying."
            )

        if n_splits is not None and min_count < n_splits:
            raise ValueError(
                f"Stratification coord in {context} has classes with fewer than n_splits={n_splits} samples "
                f"(min_count={min_count}). Bin or drop rare classes before stratifying."
            )

        if test_size is not None and test_size > 0 and min_count < 2:
            raise ValueError(
                f"Stratification coord in {context} has classes with fewer than 2 samples (min_count={min_count}). "
                "train_test_split with stratify requires at least 2 samples per class. Bin or filter before stratifying."
            )

    def _resolve_orig_trial_groups(self, data) -> np.ndarray | None:
        """Return per-trial grouping values based on orig_trial, if present."""
        if "orig_trial" not in data.coords:
            return None

        if data.coords["orig_trial"].dims != ("trial",):
            raise ValueError(
                f"orig_trial coordinate must have dimension ('trial',), but has {data.coords['orig_trial'].dims}."
            )

        if len(np.unique(data.coords["trial"].values)) == len(np.unique(data.coords["orig_trial"].values)):
            return None

        orig_values = data.coords["orig_trial"].values
        if "session" in data.coords and data.coords["session"].dims == ("trial",):
            session_values = data.coords["session"].values
            return np.asarray([f"{session}::{orig}" for session, orig in zip(session_values, orig_values)], dtype=str)

        return orig_values.astype(str)

    @staticmethod
    def _get_unique_groups_and_labels(
        group_values: np.ndarray, labels: np.ndarray | None, context: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return unique groups and optional group-level labels for stratification."""

        def _normalize_group(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return value

        unique_groups = np.unique(group_values)
        if labels is None:
            return unique_groups, None

        group_labels = []
        for group in unique_groups:
            group_norm = _normalize_group(group)
            group_mask = np.array([_normalize_group(v) == group_norm for v in group_values])
            group_label_values = labels[group_mask]
            unique_labels = np.unique(group_label_values)
            if len(unique_labels) != 1:
                warnings.warn(
                    f"Stratification coord in {context} varies within split groups. "
                    "Disabling stratification for this split.",
                    stacklevel=2,
                )
                return unique_groups, None
            group_labels.append(unique_labels[0])

        return unique_groups, np.array(group_labels, dtype=object)

    # ------------------------------
    # Scoring Function Resolution
    # ------------------------------
    def _get_scoring_func(self) -> tuple[Callable, str, bool]:
        """
        Resolve the scoring function and metric name based on the scoring parameter and task type.

        For custom callable scorers, automatically detects whether the scorer accepts
        a container argument (3 parameters) or just labels (2 parameters). The container
        provides access to coordinates for filtering/weighting during scoring.

        Returns
        -------
        tuple[Callable, str, bool]
            A tuple of (scoring_function, metric_name, needs_proba)
        """
        # Resolved on every call (cheap) so changes to `scoring` or the pipeline are never served stale.
        # Built-in scorers take (y_true, y_pred); only a custom callable below may accept the container.
        self._scoring_container_mode = "none"
        self._scoring_accepts_container = False
        # Get the final predictor to determine task type
        final_predictor = self._get_final_predictor()

        if final_predictor is None:
            raise ValueError("Pipeline must end with a Predictor to use scoring")

        # If user provided a custom scoring function
        if callable(self.scoring):
            self._scoring_func = self.scoring
            self._metric_name = "custom"
            self._scoring_container_mode = _scorer_container_mode(self.scoring)
            self._scoring_accepts_container = self._scoring_container_mode != "none"
            self._scoring_needs_proba = bool(self.scoring_needs_proba)
            if self._scoring_needs_proba and not final_predictor.is_classifier:
                raise ValueError(
                    "Custom scoring configured with scoring_needs_proba=True, "
                    "but the final predictor is not a classifier."
                )

            return self._scoring_func, self._metric_name, self._scoring_needs_proba

        # If user provided a string metric name
        if isinstance(self.scoring, str):
            metric_name = self.scoring.lower()
            if metric_name not in self._STRING_SCORERS:
                raise ValueError(
                    f"Unknown scoring metric: {self.scoring}. Supported: {sorted(self._STRING_SCORERS.keys())}"
                )

            self._scoring_func, self._metric_name, self._scoring_needs_proba = self._STRING_SCORERS[metric_name]()
            if self._scoring_needs_proba and not final_predictor.is_classifier:
                raise ValueError(
                    f"Scoring metric '{self._metric_name}' requires probability scores, "
                    "but the final predictor is not a classifier."
                )
            return self._scoring_func, self._metric_name, self._scoring_needs_proba

        if self.scoring is not None:
            raise TypeError(
                f"scoring must be a metric name, a callable or None; got {type(self.scoring).__name__}. "
                "Score one metric at a time."
            )

        # Auto-select based on is_classifier and is_multilabel
        if final_predictor.is_classifier:
            if getattr(final_predictor, "is_multilabel", False):
                self._scoring_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average="samples", zero_division=0)
                self._metric_name = "f1_samples"
            else:
                self._scoring_func = lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")
                self._metric_name = "f1_weighted"
            self._scoring_needs_proba = False
        else:
            # Assume regression (is_regressor = not is_classifier)
            # Use multioutput="uniform_average" to handle both single and multi-target
            self._scoring_func = lambda y_true, y_pred: r2_score(y_true, y_pred, multioutput="uniform_average")
            self._metric_name = "r2"
            self._scoring_needs_proba = False

        if self._scoring_func is None or self._metric_name is None:
            raise RuntimeError("Failed to resolve scoring function.")
        return self._scoring_func, self._metric_name, self._scoring_needs_proba

    def _extract_targets(self, predictor: Predictor, container: DataContainer) -> np.ndarray:
        """
        Resolve and stack target coordinates from a container, handling multi-target predictors.
        """
        target_spec = (
            predictor.target_coord_list
            if predictor and hasattr(predictor, "target_coord_list") and predictor.target_coord_list
            else self.final_target_coord_
        )
        if target_spec is None:
            raise ValueError("Could not resolve target coordinates for scoring.")
        return extract_target_array(target_spec, container.data, validate=False)

    def _filter_scoring_inputs(
        self,
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
        container: DataContainer,
        *,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray, DataContainer, np.ndarray | None]:
        """Optionally drop rows from scoring inputs based on configured filters."""
        if not self.exclude_intertrial_from_scoring and not self.exclude_offsets_from_scoring:
            return pred_labels, true_labels, container, None

        exclude_intertrial_mask = np.ones(pred_labels.shape[0], dtype=bool)
        if self.exclude_intertrial_from_scoring:
            if "event_type" not in container.data.coords:
                warnings.warn(
                    f"exclude_intertrial_from_scoring is enabled but event_type coord missing during {context}; "
                    "keeping all trials.",
                    stacklevel=2,
                )
            else:
                event_types = np.asarray(container.data.coords["event_type"].values)
                if event_types.shape[0] != pred_labels.shape[0]:
                    raise ValueError(
                        f"Mismatch between event_type coord ({event_types.shape[0]}) and predictions "
                        f"({pred_labels.shape[0]}) during {context} scoring."
                    )
                exclude_intertrial_mask = event_types != "intertrial"

        exclude_offsets_mask = np.ones(pred_labels.shape[0], dtype=bool)
        if self.exclude_offsets_from_scoring:
            if "time_offset_ms" not in container.data.coords:
                warnings.warn(
                    f"exclude_offsets_from_scoring is enabled but time_offset_ms coord missing during {context}; "
                    "keeping all trials.",
                    stacklevel=2,
                )
            else:
                time_offsets = np.asarray(container.data.coords["time_offset_ms"].values)
                exclude_offsets_mask = time_offsets == 0

        mask = exclude_intertrial_mask & exclude_offsets_mask
        if mask.all():
            return pred_labels, true_labels, container, None

        if not mask.any():
            raise ValueError(
                "Scoring filters removed all validation samples. Ensure at least one scored sample exists per split."
            )

        filtered_pred = pred_labels[mask]
        filtered_true = true_labels[mask]
        selected_positions = np.nonzero(mask)[0]
        filtered_container = DataContainer(container.data.isel(trial=selected_positions))
        return filtered_pred, filtered_true, filtered_container, mask

    # ------------------------------
    # Encoder discovery and injection
    # ------------------------------
    def _iter_predictors(self, transform: Transform | None) -> Iterator[Predictor]:
        """Yield all Predictor instances within a transform recursively."""
        yield from iter_predictors(transform)

    def _find_and_fit_encoders(self, pipeline: Pipeline, container: DataContainer):
        """Fit provided encoders in-place for all predictors on the full dataset.

        This learns the label vocabulary globally (including holdout if present)
        without leaking feature-target relationships across folds.
        """
        fit_label_encoders(pipeline, container)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise ValueError("Pipeline must be set before calling this method. Use set_pipeline() first.")
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline):
        self.set_pipeline(pipeline)  # force checks and actions on pipeline when setting

    def set_pipeline(self, pipeline: Pipeline):
        """
        Set the pipeline to be used for cross-validation.

        Args:
            pipeline: Pipeline to be used for cross-validation
        """
        if pipeline is None:
            raise ValueError("Pipeline cannot be None")

        # Assign internal attribute to avoid triggering setter recursion
        self._pipeline = pipeline

        # Last step of pipeline must be an predictor
        if not pipeline.is_predictor:
            raise ValueError("The last pipeline step must be a Predictor")
        self.final_target_coord_ = pipeline.final_target_coord
        self.holdout_trial_labels_ = None

        # Early validation: disallow multi-target classification
        final_predictor = self._get_final_predictor()
        if final_predictor is None:
            raise ValueError("Pipeline must end with a Predictor to use cross-validation.")
        if (
            final_predictor.is_classifier
            and isinstance(self.final_target_coord_, list)
            and not getattr(final_predictor, "is_multilabel", False)
        ):
            raise ValueError("Multi-target classification is not supported; use a single target_coord for classifiers.")

    def _auto_detect_pipeline_parts(self, pipeline):
        """
        Automatically detect and split pipeline into stateless and stateful parts.

        Args:
            pipeline: The complete pipeline to split

        Returns:
            Tuple of (stateless_pipeline, stateful_pipeline)
        """
        # Find the index of the first stateful step
        first_stateful_idx = -1
        for i, step in enumerate(pipeline.steps):
            if step.transform.is_stateful:
                first_stateful_idx = i
                break

        if first_stateful_idx == -1:
            # All steps are stateless
            stateless_pipeline = pipeline
            stateful_pipeline = None  # cannot have pipeline with no steps
        elif first_stateful_idx == 0:
            # All steps are stateful
            stateless_pipeline = None  # cannot have pipeline with no steps
            stateful_pipeline = pipeline
        else:
            # Split at first stateful step
            stateless_steps = pipeline.steps[:first_stateful_idx]
            stateful_steps = pipeline.steps[first_stateful_idx:]

            stateless_pipeline = Pipeline(
                name=f"{pipeline.name}_stateless", steps=stateless_steps, use_cache=pipeline.use_cache
            )
            # Important: keep cache disabled for stateful slice to avoid skipping fit on cache hits
            stateful_pipeline = Pipeline(name=f"{pipeline.name}_stateful", steps=stateful_steps, use_cache=False)

        return stateless_pipeline, stateful_pipeline

    def _log_pipeline_structure(self, pipeline: Pipeline, stateless_pipeline, stateful_pipeline):
        """
        Log information about the pipeline structure and any post-cache context.

        Args:
            pipeline: The full pipeline
            stateless_pipeline: The stateless portion of the pipeline
            stateful_pipeline: The stateful portion of the pipeline
        """
        # Clarify when running on a dynamic slice (after tuner caching)
        is_post_cache = getattr(pipeline, "is_post_cache_pipeline", False)
        if is_post_cache:
            cached_prefix = getattr(pipeline, "cached_prefix_step_names", [])
            origin_name = getattr(pipeline, "origin_pipeline_name", pipeline.name)
            if self.verbose:
                print(f"  - Context: post-cache pipeline of '{origin_name}' (cached stateless steps: {cached_prefix})")
                print(f"  - Stateless (cached before CV): {len(cached_prefix)}")

        if self.verbose:
            print(
                f"  - Stateless (in this slice): {len(stateless_pipeline.steps) if stateless_pipeline is not None else 0}"
            )
            print(
                f"  - Stateful (in this slice): {len(stateful_pipeline.steps) if stateful_pipeline is not None else 0}"
            )

    def _run_stateless_preprocessing(
        self, stateless_pipeline, initial_container: DataContainer, verbose: bool = False
    ) -> DataContainer:
        """
        Run stateless preprocessing on the entire dataset.

        Args:
            stateless_pipeline: The stateless portion of the pipeline
            initial_container: The input DataContainer
            verbose: Whether to enable verbose logging of fit_transform

        Returns:
            Preprocessed DataContainer
        """
        if stateless_pipeline is not None:
            if self.verbose:
                print("Running stateless preprocessing on entire dataset...")
            return stateless_pipeline.fit_transform(initial_container, verbose=verbose)
        else:
            if self.verbose:
                print("No stateless steps detected, using original data...")
            return initial_container

    def _fit_stateful_pipeline_with_cache(self, fold_pipeline, train_container: DataContainer, verbose: bool = False):
        """
        Fit a stateful pipeline with optional caching.

        Args:
            fold_pipeline: The pipeline to fit
            train_container: Training data container
            verbose: Whether to enable verbose logging of fit

        Returns:
            Fitted pipeline (either newly fitted or loaded from cache)
        """
        # Fits that depend on user code are not reused: a key cannot cover everything that code depends on.
        use_cache = self.use_stateful_fit_cache and (
            cache_user_code() or not references_user_code(fold_pipeline, train_container)
        )
        key_dict = None
        if use_cache:
            # Snapshot the key before fitting: it references live objects that fit mutates, and the
            # save below must use the same key as the lookup.
            key_dict = _get_object_metadata(
                {
                    "kind": "stateful_fit",
                    "pipeline_name": getattr(fold_pipeline, "name", "pipeline"),
                    "stateful_config": (
                        fold_pipeline.get_params(deep=True) if hasattr(fold_pipeline, "get_params") else None
                    ),
                    "module_hashes": get_module_hashes_for_object(fold_pipeline),
                    "installed_packages": _installed_code_fingerprint(),
                    "train_signature": train_container,
                }
            )
        if key_dict is not None:
            cached = try_load_cached_object(prefix="stateful_fit", key_dict=key_dict)
            if cached is None:
                fold_pipeline.fit(train_container, verbose=verbose)
                save_cached_object(prefix="stateful_fit", key_dict=key_dict, obj=fold_pipeline)
            else:
                fold_pipeline = cached
        else:
            # Fit without caching
            fold_pipeline.fit(train_container, verbose=verbose)

        return fold_pipeline

    def _call_scoring_func(
        self, scoring_func: Callable, true_labels: np.ndarray, predictions: np.ndarray, container: DataContainer
    ) -> float:
        """Call a scorer, passing the scoring container in the style its signature accepts."""
        if self._scoring_container_mode == "keyword":
            return scoring_func(true_labels, predictions, container=container)
        if self._scoring_container_mode == "positional":
            return scoring_func(true_labels, predictions, container)
        return scoring_func(true_labels, predictions)

    def _process_cv_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        preprocessed_data: DataContainer,
        stateful_pipeline,
        verbose: bool = False,
        pruning_callback: Callable[[int, float], None] | None = None,
    ):
        """
        Process a single cross-validation fold.

        Args:
            fold_idx: Index of the current fold
            train_indices: Training trial indices for this fold
            validation_indices: Validation trial indices for this fold
            preprocessed_data: Preprocessed data container
            stateful_pipeline: The stateful portion of the pipeline
            verbose: Whether to enable verbose logging of fitting and predicting
        """
        if self.verbose:
            print(f"  Processing fold {fold_idx + 1}...")

        # Create train/validation containers by selecting trials from preprocessed data
        train_container = DataContainer(preprocessed_data.data.sel(trial=train_indices))
        validation_container = DataContainer(preprocessed_data.data.sel(trial=validation_indices))

        # Create a fresh copy of the stateful pipeline for this fold
        fold_pipeline = stateful_pipeline.clone()

        # Fit the pipeline (with optional caching)
        fold_pipeline = self._fit_stateful_pipeline_with_cache(fold_pipeline, train_container, verbose)

        # Predict on validation data
        validation_results_container = fold_pipeline.predict(validation_container, verbose=verbose)
        scoring_func, metric_name, needs_proba = self._get_scoring_func()
        validation_proba = None
        if needs_proba:
            validation_proba = fold_pipeline.predict_proba(validation_container, verbose=verbose).data.values

        # Extract predictions
        pred_labels = validation_results_container.data.values

        final_predictor = fold_pipeline.predictive_transform
        if final_predictor is None:
            raise ValueError("Fold pipeline must expose a predictive transform.")
        true_labels = self._extract_targets(cast(Predictor, final_predictor), validation_results_container)

        pred_labels, true_labels, scoring_container, scoring_mask = self._filter_scoring_inputs(
            pred_labels,
            true_labels,
            validation_results_container,
            context="cross-validation",
        )
        scoring_values = pred_labels
        if needs_proba:
            if validation_proba is None:
                raise RuntimeError("Scoring requires probabilities, but none were produced.")
            scoring_values = validation_proba if scoring_mask is None else validation_proba[scoring_mask]

        # Calculate fold score using the selected scoring function
        fold_score = self._call_scoring_func(scoring_func, true_labels, scoring_values, scoring_container)
        self.cv_scores_.append(fold_score)
        self.oof_predictions_.append(pred_labels)
        if needs_proba:
            self.oof_probabilities_.append(scoring_values)
        self.true_labels_.append(true_labels)
        self.oof_containers_.append(scoring_container)

        if self.verbose:
            print(f"    Fold {fold_idx + 1} {metric_name}: {fold_score:.4f}")

        if pruning_callback is not None:
            pruning_callback(fold_idx, fold_score)

        if self.release_fold_memory:
            del fold_pipeline, validation_results_container, validation_container, train_container
            gc.collect()
            _cleanup_torch_memory()

    def cross_validate(
        self,
        initial_container: DataContainer,
        verbose: bool = False,
        pruning_callback: Callable[[int, float], None] | None = None,
        **kwargs,
    ) -> float:
        """
        Runs the full cross-validation process on the train and validation sets. Held out test set is not used here.

        This method automatically detects stateless and stateful pipeline components and
        executes them optimally: stateless parts run once, stateful parts per fold.

        Args:
            initial_container: Input DataContainer to cross-validate on
            verbose: Whether to enable verbose logging in transforms
            **kwargs: Additional arguments passed to splitting methods

        Returns:
            Mean cross-validation score

        Raises:
            ValueError: If no pipeline is assigned
        """
        # Splits select trials by label, so duplicated labels would put the same label on both sides of a split
        if "trial" in initial_container.data.indexes and not initial_container.data.indexes["trial"].is_unique:
            raise ValueError(
                "Trial labels must be unique for cross-validation (e.g. renumber trials after concatenating sessions)."
            )

        # Fit encoders globally on initial data
        self._find_and_fit_encoders(self.pipeline, initial_container)

        # Step 1: Auto-detect and split pipeline into stateless and stateful parts
        if self.verbose:
            print("Auto-detecting pipeline structure...")
        stateless_pipeline, stateful_pipeline = self._auto_detect_pipeline_parts(self.pipeline)

        # Step 2: Log pipeline structure information
        self._log_pipeline_structure(self.pipeline, stateless_pipeline, stateful_pipeline)

        # Step 3: Run stateless preprocessing once before the CV loop
        preprocessed_data = self._run_stateless_preprocessing(stateless_pipeline, initial_container, verbose)

        # Step 4: Split into train/validation and holdout sets using preprocessed data
        train_val_indices, holdout_indices = self._split_holdout(preprocessed_data)
        self.holdout_trial_labels_ = holdout_indices

        # Step 5: Generate cross-validation splits on train/validation data only
        splits = self._get_splits(preprocessed_data, train_val_indices)

        # Step 6: Reset evaluation metrics
        self.cv_scores_ = []
        self.oof_predictions_ = []
        self.oof_probabilities_ = []
        self.true_labels_ = []
        self.oof_containers_ = []
        self.split_trials_ = []
        self.oof_class_labels_ = self._snapshot_class_labels()

        # Step 7: Run cross-validation loop with stateful pipeline
        if stateful_pipeline is not None:
            if self.verbose:
                print("Running cross-validation with stateful pipeline...")
            for fold_idx, (train_indices, validation_indices) in enumerate(splits):
                self.split_trials_.append((np.asarray(train_indices), np.asarray(validation_indices)))
                self._process_cv_fold(
                    fold_idx,
                    train_indices,
                    validation_indices,
                    preprocessed_data,
                    stateful_pipeline,
                    verbose,
                    pruning_callback=pruning_callback,
                )

            # Print summary of fold scores
            if self.verbose and self.cv_scores_:
                _, metric_name, _ = self._get_scoring_func()
                print("\nCross-validation summary:")
                print(f"  Individual fold {metric_name} scores: {[f'{score:.4f}' for score in self.cv_scores_]}")
                print(f"  Mean {metric_name}: {self.mean_cv_score_:.4f}")
                print(f"  Std {metric_name}: {np.std(self.cv_scores_):.4f}")
        else:
            # Edge case: no stateful steps (shouldn't happen with predictors, but handle gracefully)
            raise ValueError("Pipeline must contain at least one stateful step (typically a Predictor)")

        return self.score_

    def finalize_pipeline(self, container: DataContainer, verbose: bool = False) -> "Pipeline":
        """
        Finalizes a model for production by fitting on the entire provided container.

        Args:
            container: DataContainer to fit the final model on
            verbose: Whether to enable verbose logging in transforms

        Returns:
            The fitted pipeline object, ready for inference.
        """
        # Fit encoders globally before cloning so the clone carries (a copy of) the fitted class order
        self._find_and_fit_encoders(self.pipeline, container)
        fitted_pipeline = self.pipeline.clone()

        fitted_pipeline.fit(container, verbose=verbose)
        prepare_for_inference = getattr(fitted_pipeline, "prepare_for_inference", None)
        if callable(prepare_for_inference):
            prepare_for_inference()

        return fitted_pipeline

    def score_on_holdout(self, initial_container: DataContainer, verbose: bool = False) -> float:
        """
        Performs the final evaluation on the held-out test set.

        Args:
            initial_container: The original DataContainer used in cross_validate()
            verbose: Whether to enable verbose logging in transforms

        Returns:
            Final holdout test score

        Raises:
            ValueError: If holdout indices don't exist (cross_validate() not called first)
        """

        # Ensure encoders are fitted in-place on predictors
        self._find_and_fit_encoders(self.pipeline, initial_container)

        # Run the same preprocessing as in cross_validate()
        stateless_pipeline, stateful_pipeline = self._auto_detect_pipeline_parts(self.pipeline)
        assert stateful_pipeline is not None, "There must be at least one stateful step in the pipeline, for fitting."

        if stateless_pipeline is not None:
            preprocessed_data = stateless_pipeline.fit_transform(initial_container, verbose=verbose)
        else:
            preprocessed_data = initial_container

        if self.holdout_trial_labels_ is None:
            warnings.warn(
                "cross_validate() not called first so no holdout indices available. Calculating holdout indices now.",
                stacklevel=2,
            )  # necessary since Tuner makes a deepcopy of validators
            train_val_indices, holdout_indices = self._split_holdout(preprocessed_data)
            self.holdout_trial_labels_ = holdout_indices

        if len(self.holdout_trial_labels_) == 0:
            raise ValueError("No holdout data available for testing.")

        # Create train/validation container (all data except holdout)
        all_trials = preprocessed_data.data.trial.values
        train_val_mask = ~np.isin(all_trials, self.holdout_trial_labels_)
        train_val_indices = all_trials[train_val_mask]

        train_val_container = DataContainer(preprocessed_data.data.sel(trial=train_val_indices))
        test_container = DataContainer(preprocessed_data.data.sel(trial=self.holdout_trial_labels_))

        # Fit the stateful part of the pipeline on the train/validation data (no caching for holdout)
        stateful_pipeline_fitted = stateful_pipeline.clone()
        stateful_pipeline_fitted.fit(train_val_container, verbose=verbose)

        # Predict on the holdout test set
        test_results_container = stateful_pipeline_fitted.predict(test_container, verbose=verbose)
        scoring_func, _, needs_proba = self._get_scoring_func()
        holdout_probabilities = None
        if needs_proba:
            holdout_probabilities = stateful_pipeline_fitted.predict_proba(test_container, verbose=verbose).data.values

        final_predictor = stateful_pipeline_fitted.predictive_transform
        if final_predictor is None:
            raise ValueError("Stateful pipeline must expose a predictive transform.")
        pred_labels = test_results_container.data.values
        true_labels = self._extract_targets(cast(Predictor, final_predictor), test_results_container)

        pred_labels, true_labels, scoring_container, scoring_mask = self._filter_scoring_inputs(
            pred_labels,
            true_labels,
            test_results_container,
            context="holdout",
        )
        scoring_values = pred_labels
        if needs_proba:
            if holdout_probabilities is None:
                raise RuntimeError("Scoring requires probabilities, but none were produced.")
            scoring_values = holdout_probabilities if scoring_mask is None else holdout_probabilities[scoring_mask]

        # Store the holdout container and filtered predictions for container-aware scorers
        self.holdout_container_ = scoring_container
        self.holdout_pred_labels_ = pred_labels
        self.holdout_class_labels_ = self._snapshot_class_labels()
        self.holdout_probabilities_ = scoring_values if needs_proba else holdout_probabilities
        self.holdout_true_labels_ = true_labels
        self.holdout_scoring_mask_ = None

        # Calculate and store final score using the selected scoring function
        self.holdout_score_ = self._call_scoring_func(
            scoring_func, self.holdout_true_labels_, scoring_values, scoring_container
        )

        return self.holdout_score_

    def compute_holdout_scoring_mask(self, mask_func: Callable[[DataContainer], np.ndarray]) -> np.ndarray:
        """
        Compute and store the mask used by a container-aware scorer.

        This method should be called after score_on_holdout() when using a custom
        container-aware scorer that filters samples. The mask will be used to generate
        filtered confusion matrices that match the scoring logic.

        Args:
            mask_func: Function that takes a DataContainer and returns a boolean mask array.
                      Should implement the same filtering logic as the custom scorer.
                      Example: lambda c: (c.coords['concentration_bin'] == 'conc_2p4')

        Returns:
            The computed boolean mask array

        Raises:
            ValueError: If no holdout container is available, or if mask shape/dtype is invalid

        Example:
            >>> cv.score_on_holdout(data_container)
            >>> cv.compute_holdout_scoring_mask(lambda c: c.coords['concentration_bin'] == 'target')
            >>> cm = cv.holdout_confusion_matrix_  # Now filtered to match scorer
        """
        if self.holdout_container_ is None:
            raise ValueError("No holdout container available. Run score_on_holdout() first.")
        if self.holdout_pred_labels_ is None:
            raise ValueError("No holdout predictions available. Run score_on_holdout() first.")

        mask = mask_func(self.holdout_container_)

        # Validate mask
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)

        if mask.shape != self.holdout_pred_labels_.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match holdout predictions shape "
                f"{self.holdout_pred_labels_.shape}. The mask must have one boolean value per sample."
            )

        if mask.dtype != np.bool_:
            # Try to convert to boolean
            try:
                mask = mask.astype(np.bool_)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Mask must be boolean or convertible to boolean, got dtype {mask.dtype}") from e

        self.holdout_scoring_mask_ = mask
        return self.holdout_scoring_mask_

    @property
    def holdout_confusion_matrix_(self) -> np.ndarray:
        """
        Calculate confusion matrix from holdout test predictions.

        If a scoring mask has been set via compute_holdout_scoring_mask(), the confusion
        matrix will be computed only on the filtered samples, matching the scorer's logic.

        Returns:
            Confusion matrix as numpy array

        Raises:
            ValueError: If no holdout predictions available or task is not classification
        """
        # Check if this is a classification task
        final_predictor = self._get_final_predictor()

        if final_predictor and (not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)):
            raise ValueError("Confusion matrix is only available for classification tasks.")

        if self.holdout_pred_labels_ is None:
            raise ValueError("No holdout predictions available. Run score_on_holdout() first.")
        if self.holdout_true_labels_ is None:
            raise ValueError("No holdout labels available. Run score_on_holdout() first.")

        # Apply scoring mask if available
        if self.holdout_scoring_mask_ is not None:
            y_true = self.holdout_true_labels_[self.holdout_scoring_mask_]
            y_pred = self.holdout_pred_labels_[self.holdout_scoring_mask_]
            return confusion_matrix(y_true, y_pred, labels=self._confusion_labels(self.holdout_class_labels_))

        return confusion_matrix(
            self.holdout_true_labels_,
            self.holdout_pred_labels_,
            labels=self._confusion_labels(self.holdout_class_labels_),
        )

    @property
    def holdout_confusion_matrix_normalized_(self) -> np.ndarray:
        """
        Calculate normalized confusion matrix from holdout test predictions.

        If a scoring mask has been set via compute_holdout_scoring_mask(), the confusion
        matrix will be computed only on the filtered samples, matching the scorer's logic.

        Returns:
            Normalized confusion matrix as numpy array (rows sum to 1)

        Raises:
            ValueError: If no holdout predictions available or task is not classification
        """
        # Check if this is a classification task
        final_predictor = self._get_final_predictor()

        if final_predictor and (not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)):
            raise ValueError("Confusion matrix is only available for classification tasks.")

        if self.holdout_pred_labels_ is None:
            raise ValueError("No holdout predictions available. Run score_on_holdout() first.")
        if self.holdout_true_labels_ is None:
            raise ValueError("No holdout labels available. Run score_on_holdout() first.")

        # Apply scoring mask if available
        if self.holdout_scoring_mask_ is not None:
            y_true = self.holdout_true_labels_[self.holdout_scoring_mask_]
            y_pred = self.holdout_pred_labels_[self.holdout_scoring_mask_]
            return confusion_matrix(
                y_true, y_pred, labels=self._confusion_labels(self.holdout_class_labels_), normalize="true"
            )

        return confusion_matrix(
            self.holdout_true_labels_,
            self.holdout_pred_labels_,
            labels=self._confusion_labels(self.holdout_class_labels_),
            normalize="true",
        )

    def _compute_oof_metric(self, scoring_func: Callable) -> float:
        """
        Helper method to compute a metric on out-of-fold predictions.

        Args:
            scoring_func: Function with signature (y_true, y_pred) -> float

        Returns:
            Score calculated using the provided scoring function

        Raises:
            ValueError: If no out-of-fold predictions available
        """
        if not self.oof_predictions_:
            raise ValueError("No out-of-fold predictions available. Run cross_validate() first.")

        # Concatenate all out-of-fold predictions and true labels
        all_predictions = np.concatenate(self.oof_predictions_)
        all_true_labels = np.concatenate(self.true_labels_)

        return scoring_func(all_true_labels, all_predictions)

    @property
    def metric_name_(self) -> str:
        """
        Get the name of the scoring metric used for evaluation.

        Returns:
            Name of the metric (e.g., 'r2', 'mse', 'f1_weighted', 'custom')
        """
        _, metric_name, _ = self._get_scoring_func()
        return metric_name

    @property
    def oof_score_(self) -> float:
        """
        Calculate the selected metric score from out-of-fold predictions.

        Note: For scorers that require a container argument, OOF scoring is not
        possible since predictions come from multiple folds. In this case,
        returns the mean CV score as a fallback.

        Returns:
            Score calculated using the selected scoring function

        Raises:
            ValueError: If no out-of-fold predictions available
        """
        scoring_func, _, needs_proba = self._get_scoring_func()

        # If scorer needs container, we can't compute OOF score (predictions are from multiple folds)
        # Fall back to mean CV score instead
        if self._scoring_accepts_container:
            warnings.warn(
                "OOF scoring not available for container-aware scorers. Returning mean CV score instead.",
                UserWarning,
                stacklevel=2,
            )
            return self.mean_cv_score_

        if needs_proba:
            if not self.oof_probabilities_:
                raise ValueError("No out-of-fold probabilities available. Run cross_validate() first.")
            all_probabilities = np.concatenate(self.oof_probabilities_)
            all_true_labels = np.concatenate(self.true_labels_)
            return scoring_func(all_true_labels, all_probabilities)

        return self._compute_oof_metric(scoring_func)

    @property
    def mean_cv_score_(self) -> float:
        """
        Get the mean cross-validation score across all folds.

        Returns:
            Mean of scores from all cross-validation folds

        Raises:
            ValueError: If no cross-validation scores available
        """
        if not self.cv_scores_:
            raise ValueError("No cross-validation scores available. Run cross_validate() first.")

        return np.mean(self.cv_scores_)

    @property
    def oof_f1_score_(self) -> float:
        """
        Calculate F1 score from out-of-fold predictions.

        Convenience property for classification tasks that always returns weighted F1 score,
        regardless of the configured scoring metric.

        Returns:
            Weighted F1 score across all out-of-fold predictions

        Raises:
            ValueError: If no out-of-fold predictions available
        """
        return self._compute_oof_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"))

    def _compute_holdout_metric(self, scoring_func: Callable) -> float:
        """
        Helper method to compute a metric on holdout predictions.

        Args:
            scoring_func: Function with signature (y_true, y_pred) -> float

        Returns:
            Score calculated using the provided scoring function

        Raises:
            ValueError: If no holdout predictions available
        """
        if self.holdout_pred_labels_ is None:
            raise ValueError("No holdout predictions available. Run score_on_holdout() first.")

        return scoring_func(self.holdout_true_labels_, self.holdout_pred_labels_)

    @property
    def holdout_f1_score_(self) -> float:
        """
        Calculate F1 score from holdout predictions.

        Convenience property for classification tasks that always returns weighted F1 score,
        regardless of the configured scoring metric.

        Returns:
            Weighted F1 score from holdout predictions

        Raises:
            ValueError: If no holdout predictions available
        """
        return self._compute_holdout_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"))

    @property
    def mean_cv_f1_score_(self) -> float:
        """
        Get the mean cross-validation F1 score across all folds.

        Returns:
            Mean of F1 scores from all cross-validation folds

        Raises:
            ValueError: If no cross-validation scores available
        """
        if not self.cv_scores_:
            raise ValueError("No cross-validation scores available. Run cross_validate() first.")

        return np.mean(self.cv_scores_)

    @property
    def oof_predictions_container_(self) -> DataContainer:
        """Out-of-fold predictions for every scored validation trial, in one container.

        Keeps each trial's coordinates (labels, groups such as subject or session) plus a `fold` coordinate,
        so CV results can be broken down by any metadata, e.g. accuracy per subject.
        """
        if not self.oof_containers_:
            raise ValueError("No out-of-fold predictions available. Run cross_validate() first.")
        folds = [
            container.data.assign_coords(fold=("trial", np.full(container.data.sizes["trial"], fold_idx)))
            for fold_idx, container in enumerate(self.oof_containers_)
        ]
        return DataContainer(xr.concat(folds, dim="trial"))

    @property
    def oof_confusion_matrix_(self) -> np.ndarray:
        """
        Calculate confusion matrix from out-of-fold predictions.

        Returns:
            Confusion matrix as numpy array

        Raises:
            ValueError: If no out-of-fold predictions available or task is not classification
        """
        # Check if this is a classification task
        final_predictor = None
        for predictor in self._iter_predictors(self._pipeline):
            final_predictor = predictor

        if final_predictor and (not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)):
            raise ValueError("Confusion matrix is only available for classification tasks.")

        if not self.oof_predictions_:
            raise ValueError("No out-of-fold predictions available. Run cross_validate() first.")

        # Concatenate all out-of-fold predictions and true labels
        all_predictions = np.concatenate(self.oof_predictions_)
        all_true_labels = np.concatenate(self.true_labels_)

        return confusion_matrix(all_true_labels, all_predictions, labels=self._confusion_labels(self.oof_class_labels_))

    @property
    def oof_confusion_matrix_normalized_(self) -> np.ndarray:
        """
        Calculate normalized confusion matrix from out-of-fold predictions.

        Returns:
            Normalized confusion matrix as numpy array (rows sum to 1)

        Raises:
            ValueError: If no out-of-fold predictions available or task is not classification
        """
        # Check if this is a classification task
        final_predictor = None
        for predictor in self._iter_predictors(self._pipeline):
            final_predictor = predictor

        if final_predictor and (not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)):
            raise ValueError("Confusion matrix is only available for classification tasks.")

        if not self.oof_predictions_:
            raise ValueError("No out-of-fold predictions available. Run cross_validate() first.")

        # Concatenate all out-of-fold predictions and true labels
        all_predictions = np.concatenate(self.oof_predictions_)
        all_true_labels = np.concatenate(self.true_labels_)

        return confusion_matrix(
            all_true_labels, all_predictions, labels=self._confusion_labels(self.oof_class_labels_), normalize="true"
        )

    def get_fold_scores(self) -> list:
        """
        Get individual fold scores.

        Returns:
            List of scores for each cross-validation fold

        Raises:
            ValueError: If no cross-validation scores available
        """
        if not self.cv_scores_:
            raise ValueError("No cross-validation scores available. Run cross_validate() first.")

        return self.cv_scores_.copy()

    @property
    def score_(self) -> float:
        """
        Calculate the final CV score based on the pooling_score_weight.

        Blends the average fold score and pooled out-of-fold score using:
        score = (1 - pooling_score_weight) * mean_cv_f1_score_ + pooling_score_weight * oof_f1_score_

        When pooling_score_weight = 0.0: Returns average fold score (standard behavior)
        When pooling_score_weight = 1.0: Returns pooled OOF score
        When pooling_score_weight = 0.5: Returns equal blend of both

        Returns:
            Final blended cross-validation score

        Raises:
            ValueError: If no cross-validation scores are available
        """
        if not self.cv_scores_:
            raise ValueError("No CV scores available. Run cross_validate() first.")

        mean_score = self.mean_cv_score_
        if self.pooling_score_weight == 0:
            return mean_score  # the pooled score is not needed, and may be undefined (binary F1 of 3 classes)
        pooled_score = self.oof_score_
        if self.pooling_score_weight == 1:
            return pooled_score  # fold scores do not count, and may be undefined (R² of one trial)

        final_score = (1 - self.pooling_score_weight) * mean_score + self.pooling_score_weight * pooled_score
        return final_score

    def get_holdout_container(self, initial_container: DataContainer, *, verbose: bool = False) -> DataContainer:
        """
        Return the holdout trials from the original data container.

        This helper returns the raw-space slice referenced by `holdout_trial_labels_`.

        Args:
            initial_container: The original DataContainer provided to cross_validate()/score_on_holdout().
            verbose: Whether to enable verbose logging in transforms.

        Returns:
            DataContainer containing only the holdout trials in raw space.

        Raises:
            ValueError: If no holdout data is available (e.g., test_size not set).
        """
        if self.holdout_trial_labels_ is None:
            self._find_and_fit_encoders(self.pipeline, initial_container)
            stateless_pipeline, _ = self._auto_detect_pipeline_parts(self.pipeline)
            if stateless_pipeline is not None:
                preprocessed_data = stateless_pipeline.fit_transform(initial_container, verbose=verbose)
            else:
                preprocessed_data = initial_container
            _, holdout_indices = self._split_holdout(preprocessed_data)
            self.holdout_trial_labels_ = holdout_indices

        if len(self.holdout_trial_labels_) == 0:
            raise ValueError("No holdout data available for testing.")

        # Select directly from the original container using trial labels
        holdout_da = initial_container.data.sel(trial=self.holdout_trial_labels_)
        return DataContainer(holdout_da)

    def plot_confusion_matrix(self, use_holdout: bool = True, normalize: bool = True, title_info: str = "", **kwargs):
        """
        Plot the confusion matrix.

        Note: Only works for classification tasks.

        Raises:
            ValueError: If the pipeline is not a classifier
        """
        # Check if this is a classification task
        final_predictor = self.pipeline.predictive_transform
        if final_predictor and (not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)):
            raise ValueError("Confusion matrix plotting is only available for classification tasks.")

        if use_holdout:
            conf_matrix = self.holdout_confusion_matrix_normalized_ if normalize else self.holdout_confusion_matrix_
            f1_score = self.holdout_f1_score_
        else:
            conf_matrix = self.oof_confusion_matrix_normalized_ if normalize else self.oof_confusion_matrix_
            f1_score = self.oof_f1_score_

        # Use classes from the final predictor's encoder
        predictive_transform = self.pipeline.predictive_transform
        if predictive_transform is None or predictive_transform.encoder is None:
            raise ValueError("Pipeline predictor must have a fitted encoder before plotting confusion matrix.")
        evaluated_labels = self.holdout_class_labels_ if use_holdout else self.oof_class_labels_
        classes = evaluated_labels if evaluated_labels is not None else predictive_transform.encoder.classes_

        if title_info:
            title_info = f"{title_info},"

        plot_confusion_matrix(conf_matrix, classes, title=f"{title_info} F1 score: {f1_score:.4f}", **kwargs)
