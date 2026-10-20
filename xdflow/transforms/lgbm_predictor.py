"""
LightGBM predictor with built-in early stopping support.

This module provides a specialized predictor for LightGBM models that handles
early stopping parameters in __init__ and properly configures validation sets
during fitting.
"""

import warnings
from typing import Any

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - depends on optional extra
    lgb = None
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from xdflow.core.data_container import DataContainer
from xdflow.transforms.sklearn_transform import SKLearnPredictor


class LGBMPredictor(SKLearnPredictor):
    """
    LightGBM predictor with built-in early stopping support.

    Extends SKLearnPredictor to handle early stopping parameters in __init__ and
    automatically create validation splits during fitting. Works with both
    LGBMClassifier and LGBMRegressor.

    Parameters
    ----------
    estimator_cls : Type[BaseEstimator]
        LightGBM estimator class (LGBMClassifier or LGBMRegressor).
    sample_dim : str
        Name of the sample dimension.
    target_coord : Union[str, List[str]]
        Target coordinate name (or list/pattern for multi-target).
    early_stopping_rounds : Optional[int], default=None
        Number of rounds with no improvement before stopping.
        - None: disabled (no validation split created)
        - Positive int: enabled (creates validation split automatically)
        - Note: 0 raises ValueError; use None to disable
    validation_size : float, default=0.2
        Proportion of training data for validation (0.0-1.0).
        Only used when early_stopping_rounds is set.
    validation_seed : Optional[int], default=None
        Random seed for reproducible validation splits.
    eval_metric : Optional[str], default=None
        Metric for early stopping. If None, LightGBM auto-selects based on objective:
        - 'binary' → 'binary_logloss'
        - 'multiclass' → 'multi_logloss'
        - 'regression' → 'l2' (RMSE)
        Common overrides: 'auc', 'rmse', 'mae'
    verbose_eval : Union[int, bool], default=False
        Logging frequency (int > 0) or disable (False/0).
    **kwargs
        Standard predictor params (encoder, proba, is_classifier, multi_output, etc.)
        and LightGBM hyperparameters (n_estimators, learning_rate, max_depth, etc.).

    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> predictor = LGBMPredictor(
    ...     LGBMClassifier,
    ...     sample_dim='trial',
    ...     target_coord='stimulus',
    ...     early_stopping_rounds=50,
    ...     n_estimators=1000
    ... )
    >>> predictor.fit(train_data)
    >>> print(f"Stopped at iteration: {predictor.best_iteration_}")

    Notes
    -----
    - Validation split uses stratification for classifiers when possible
    - best_iteration_ attribute set after fitting with early stopping
    - Sample weights automatically split along with data if provided
    """

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str | list[str],
        early_stopping_rounds: int | None = None,
        validation_size: float = 0.2,
        validation_seed: int | None = None,
        eval_metric: str | None = None,
        verbose_eval: int | bool = False,
        encoder: LabelEncoder | None = None,
        proba: bool = False,
        is_classifier: bool | None = None,
        multi_output: bool = False,
        is_multilabel: bool = False,
        sel: dict | None = None,
        drop_sel: dict | None = None,
        sample_weight_coord: str | None = "sample_weight",
        fit_on_labels: bool = False,
        **kwargs: Any,
    ):
        """Initialize LGBMPredictor with early stopping parameters."""
        if lgb is None:
            raise ImportError(
                "LightGBM is required for LGBMPredictor. "
                "Install with: pip install xdflow[lightgbm] or pip install lightgbm"
            )
        # Validate early stopping parameters
        if early_stopping_rounds is not None:
            if not isinstance(early_stopping_rounds, int) or early_stopping_rounds <= 0:
                raise ValueError(
                    f"early_stopping_rounds must be a positive integer or None, got {early_stopping_rounds}"
                )
            if not 0.0 < validation_size < 1.0:
                raise ValueError(f"validation_size must be between 0.0 and 1.0, got {validation_size}")

        # Store early stopping parameters as public attributes (required for cloning)
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size
        self.validation_seed = validation_seed
        self.eval_metric = eval_metric
        self.verbose_eval = verbose_eval

        # Initialize parent class
        super().__init__(
            estimator_cls=estimator_cls,
            sample_dim=sample_dim,
            target_coord=target_coord,
            encoder=encoder,
            proba=proba,
            is_classifier=is_classifier,
            multi_output=multi_output,
            is_multilabel=is_multilabel,
            sel=sel,
            drop_sel=drop_sel,
            sample_weight_coord=sample_weight_coord,
            fit_on_labels=fit_on_labels,
            **kwargs,
        )

    def _fit(self, container: DataContainer, **kwargs) -> "LGBMPredictor":
        """
        Fit the LightGBM estimator with optional early stopping.

        If early_stopping_rounds is set, this method:
        1. Splits the training data into train/validation sets
        2. Fits the estimator with eval_set and early stopping callbacks
        3. Stores the best iteration in self.best_iteration_

        If early_stopping_rounds is None, delegates to parent class fit.
        """
        # If early stopping is disabled, use parent implementation
        if self.early_stopping_rounds is None:
            return super()._fit(container, **kwargs)

        # Early stopping is enabled - need to create validation set
        data = container.data
        self._record_fit_labels(data)
        X, sample_index = self._prepare_data(data)

        # Extract target values
        y = None
        if self.target_coord:
            # Resolve target coordinates (handles patterns, lists, and single coords)
            resolved_targets = self._resolve_target_coords(data)

            # Store resolved targets for prediction (every fit, so a changed target_coord is not left stale)
            self._resolved_target_coords = resolved_targets
            self._wrap_for_resolved_targets(resolved_targets)

            # Extract target values
            if len(resolved_targets) == 1:
                # Single target (as the estimator sees it: labels or codes, see fit_on_labels)
                y = self._estimator_target(data.coords[resolved_targets[0]].values)
            else:
                # Multi-target: stack all target coordinates
                target_arrays = []
                for target in resolved_targets:
                    target_arrays.append(data.coords[target].values)
                # Stack into shape (n_samples, n_targets)
                y = np.column_stack(target_arrays)

        if y is None:
            raise ValueError(
                "Early stopping requires a target coordinate (supervised learning). target_coord must be specified."
            )

        # Extract sample weights if provided
        sample_weight = self._extract_sample_weights(data, sample_index)

        # Split into train/validation sets
        split_kwargs: dict[str, Any] = {
            "test_size": self.validation_size,
            "random_state": self.validation_seed,
        }

        # Use stratified split for classifiers if possible
        if self.is_classifier and y.ndim == 1:
            # Check if we have enough samples per class for stratification
            unique, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            min_samples_per_class = max(1, int(np.ceil(self.validation_size * len(y) / len(unique))))

            if min_count >= 2 and min_count >= min_samples_per_class:
                split_kwargs["stratify"] = y
            else:
                warnings.warn(
                    f"Cannot use stratified split: some classes have too few samples "
                    f"(min={min_count}). Using random split instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Perform the split
        if sample_weight is not None:
            X_train, X_valid, y_train, y_valid, sw_train, _ = train_test_split(X, y, sample_weight, **split_kwargs)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, **split_kwargs)
            sw_train = None

        # Build fit kwargs
        fit_kwargs: dict[str, Any] = {}

        # Add sample weights to fit kwargs if supported
        if sw_train is not None:
            if self._supports_fit_param("sample_weight"):
                fit_kwargs["sample_weight"] = sw_train
            else:
                warnings.warn(
                    f"Ignoring sample weights because "
                    f"{self.estimator.__class__.__name__}.fit does not accept 'sample_weight'.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Add eval_set for early stopping
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]

        # Add eval_metric if specified
        if self.eval_metric is not None:
            fit_kwargs["eval_metric"] = self.eval_metric

        # Build callbacks list
        callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]

        # Add logging callback if verbose_eval is enabled
        if self.verbose_eval:
            period = self.verbose_eval if isinstance(self.verbose_eval, int) else 1
            callbacks.append(lgb.log_evaluation(period=period))

        fit_kwargs["callbacks"] = callbacks

        if y_train.ndim == 2 and isinstance(self.estimator, (MultiOutputClassifier, MultiOutputRegressor)):
            # MultiOutput* forwards fit kwargs unchanged, so every per-target model would be handed the 2D
            # validation labels; fit each target's model with its own validation column instead.
            estimators = []
            for k in range(y_train.shape[1]):
                target_kwargs = {**fit_kwargs, "eval_set": [(X_valid, y_valid[:, k])]}
                estimators.append(sklearn_clone(self.estimator.estimator).fit(X_train, y_train[:, k], **target_kwargs))
            self.estimator.estimators_ = estimators
            self.best_iteration_ = [getattr(est, "best_iteration_", None) for est in estimators]
            return self

        # Fit the estimator
        self.estimator.fit(X_train, y_train, **fit_kwargs)

        # Store best iteration if available
        if hasattr(self.estimator, "best_iteration_"):
            self.best_iteration_ = self.estimator.best_iteration_

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters including early stopping parameters.

        Returns parameters from the wrapper, early stopping config, and wrapped estimator.
        """
        params = super().get_params(deep=deep)
        return params

    def clone(self):
        """
        Return a fresh instance with the same constructor parameters.

        Ensures early stopping parameters are preserved in the cloned instance.
        """
        return super().clone()
