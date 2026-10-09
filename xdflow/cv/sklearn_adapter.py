"""
Adapters to convert CrossValidator classes to sklearn-compatible CV splitters.

This allows using custom CrossValidator classes with sklearn's built-in CV models
like LogisticRegressionCV, RidgeCV, LassoCV, etc.
"""

import contextvars
from contextlib import contextmanager

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from xdflow.core.data_container import DataContainer

# Context variable to pass container through sklearn's fit
_current_container: contextvars.ContextVar[DataContainer | None] = contextvars.ContextVar(
    "current_container", default=None
)


class SklearnCVAdapter(BaseCrossValidator):
    """
    Adapter that converts a CrossValidator to sklearn-compatible CV splitter.

    This allows using custom CrossValidator classes (LeaveGroupOutValidator, etc.)
    with sklearn models that accept a cv parameter (LogisticRegressionCV, RidgeCV, etc.).

    The adapter uses a context variable to receive the DataContainer during fit(),
    allowing it to work in nested CV scenarios where the container may change.
    It is intended for normal pipeline usage (including tuning) because
    SKLearnTransform automatically wraps estimator.fit with set_cv_container
    when it detects a SklearnCVAdapter. For standalone sklearn usage, the
    context manager must be set explicitly.
    """

    def __init__(self, cross_validator):
        """
        Initialize the adapter.

        Parameters
        ----------
        cross_validator : CrossValidator
            The custom cross validator to adapt
        """
        self.cross_validator = cross_validator

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        # Get container from context variable
        container = _current_container.get()
        if container is None:
            raise ValueError(
                "No container found in context. Use set_cv_container(container) context manager before calling fit()."
            )

        # Get all trial indices (for non-holdout splits)
        all_trials = container.data.trial.values

        # Get train/val split (excluding holdout)
        train_val_indices, _ = self.cross_validator._split_holdout(container)

        # Create mapping from trial IDs to 0-based indices
        trial_to_idx = {trial: idx for idx, trial in enumerate(all_trials)}

        # Get splits from the cross validator
        splits = self.cross_validator._get_splits(container, train_val_indices)

        for train_trials, val_trials in splits:
            # Convert trial IDs to 0-based indices
            train_indices = np.array([trial_to_idx[t] for t in train_trials])
            val_indices = np.array([trial_to_idx[t] for t in val_trials])
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        """
        # Not cached: the adapter is reused across containers (e.g. CV folds) whose split counts can differ.
        # Get container from context variable
        container = _current_container.get()
        if container is None:
            raise ValueError(
                "No container found in context. Use set_cv_container(container) "
                "context manager before calling get_n_splits()."
            )

        # Count the splits
        train_val_indices, _ = self.cross_validator._split_holdout(container)
        return sum(1 for _ in self.cross_validator._get_splits(container, train_val_indices))


@contextmanager
def set_cv_container(container: DataContainer):
    """
    Context manager to set the DataContainer for SklearnCVAdapter.

    This must be used when fitting sklearn models that use SklearnCVAdapter
    for cross-validation, unless the estimator is wrapped in SKLearnTransform
    (which will set the context automatically).
    """
    token = _current_container.set(container)
    try:
        yield
    finally:
        _current_container.reset(token)
