"""Use an xdflow pipeline as a scikit-learn estimator (``cross_val_score``, ``GridSearchCV``, MOABB...)."""

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import fit_label_encoders
from xdflow.interop import from_numpy


class _PipelineEstimator(BaseEstimator):
    """Shared fit/predict for ``XDFlowClassifier`` and ``XDFlowRegressor``."""

    def __init__(self, pipeline: Pipeline, dims: tuple[str, ...] | None = None, coords: dict[str, Any] | None = None):
        self.pipeline = pipeline
        self.dims = dims
        self.coords = coords

    def __sklearn_clone__(self) -> "_PipelineEstimator":
        # xdflow pipelines clone themselves (fresh, unfitted, current hyperparameters); sklearn's generic
        # clone would try to rebuild them from get_params(). Deep-copied, as scikit-learn copies parameters, so
        # editing a clone's mutable parameters (a dict) cannot change this estimator.
        return type(self)(pipeline=copy.deepcopy(self.pipeline.clone()), dims=self.dims, coords=self.coords)

    def _target(self) -> str:
        target = self.pipeline.final_target_coord
        if not isinstance(target, str):
            raise ValueError(f"to_sklearn supports pipelines with one target coordinate; got {target!r}.")
        return target

    def _container(self, X: Any, y: Any = None) -> DataContainer:
        trial_coords = {} if y is None else {self._target(): np.asarray(y)}
        return from_numpy(X, self.dims, coords=self.coords, **trial_coords)

    def fit(self, X: Any, y: Any) -> "_PipelineEstimator":
        """Fit a fresh copy of the pipeline on sample-first ``X`` and targets ``y``."""
        if not self.pipeline.is_predictor:
            raise ValueError("The xdflow pipeline must end with a predictor.")
        container = self._container(X, y)
        pipeline = self.pipeline.clone()
        fit_label_encoders(pipeline, container)  # classes come from this y, not from an earlier fit of the template
        self.pipeline_ = pipeline.fit(container)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        container = self._container(X)
        return np.asarray(_in_input_order(self.pipeline_.predict(container).data, container).values)


class XDFlowClassifier(ClassifierMixin, _PipelineEstimator):
    """An xdflow classification pipeline behind the scikit-learn estimator API.

    ``X`` is a sample-first NumPy array; each call wraps it with ``dims`` and ``coords`` (e.g. channel names
    and time points) so the pipeline's steps can address dimensions by name. Build one with ``to_sklearn``.
    """

    def fit(self, X: Any, y: Any) -> "XDFlowClassifier":
        check_classification_targets(y)  # as scikit-learn classifiers do: no continuous or NaN labels
        super().fit(X, y)
        self.classes_ = np.asarray(self.pipeline_.get_labels())
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, columns in the order of ``classes_``."""
        check_is_fitted(self, "pipeline_")
        container = self._container(X)
        proba = _in_input_order(self.pipeline_.predict_proba(container).data, container)
        class_dims = [dim for dim in proba.dims if dim != "trial"]
        if len(class_dims) != 1:
            raise ValueError(f"Expected probabilities with dims ('trial', <class dim>); got {proba.dims}.")
        class_dim = class_dims[0]
        if class_dim in proba.coords:  # by position: .sel would read bool classes as a mask
            columns = pd.Index(proba[class_dim].values).get_indexer(self.classes_)
            if (columns < 0).any():
                raise ValueError(f"Probabilities have no column for classes {list(self.classes_[columns < 0])}.")
            proba = proba.isel({class_dim: columns})
        return np.asarray(proba.transpose("trial", class_dim).values)


def _in_input_order(result: Any, container: DataContainer) -> Any:
    """One output per input sample, in the input's order (a step may reorder trials)."""
    trials = container.data.trial.values
    if "trial" not in result.dims or result.sizes["trial"] != trials.size or set(result.trial.values) != set(trials):
        raise ValueError("The pipeline must return exactly one prediction per input sample.")
    return result.sel(trial=trials)


class XDFlowRegressor(RegressorMixin, _PipelineEstimator):
    """An xdflow regression pipeline behind the scikit-learn estimator API. Build one with ``to_sklearn``."""


def to_sklearn(
    pipeline: Pipeline, dims: tuple[str, ...] | None = None, coords: dict[str, Any] | None = None
) -> XDFlowClassifier | XDFlowRegressor:
    """Wrap an xdflow pipeline as a scikit-learn estimator taking sample-first NumPy arrays.

    The result works anywhere scikit-learn estimators do: ``cross_val_score``, ``GridSearchCV`` (parameters
    are ``"pipeline__<step>__<param>"``), MOABB evaluations, or another scikit-learn ``Pipeline``.

    Args:
        pipeline: An xdflow pipeline ending with a predictor (one target coordinate).
        dims: Names of every axis of ``X``, starting with ``"trial"``, e.g. ``("trial", "channel", "time")``.
        coords: Coordinates for the non-trial dims, e.g. ``{"channel": ch_names, "time": times}``; needed by
            steps that select by label (cropping a time window, picking channels).

    Example:
        >>> clf = to_sklearn(pipeline, ("trial", "channel", "time"), {"channel": ch_names, "time": epochs.times})
        >>> cross_val_score(clf, X, y, groups=subjects, cv=GroupKFold(5))
    """
    if not isinstance(pipeline, Pipeline) or not pipeline.is_predictor:
        raise TypeError("to_sklearn expects an xdflow Pipeline that ends with a predictor.")
    is_classifier = bool(getattr(pipeline.predictive_transform, "is_classifier", False))
    estimator_cls = XDFlowClassifier if is_classifier else XDFlowRegressor
    return estimator_cls(pipeline=pipeline, dims=dims, coords=coords)
