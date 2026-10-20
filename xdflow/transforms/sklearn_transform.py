import inspect
import warnings
from inspect import Parameter, signature
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone
from sklearn.base import is_classifier as sklearn_is_classifier
from sklearn.base import is_regressor as sklearn_is_regressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

from xdflow.core.base import Predictor, SampleWeightMixin, Transform
from xdflow.core.data_container import DataContainer
from xdflow.transforms.multi_output_wrapper import MultiOutputClassifierFactory, MultiOutputRegressorFactory
from xdflow.utils.inspection import build_estimator, collect_super_init_param_names
from xdflow.utils.target_utils import extract_target_array, resolve_target_coords

# Fitted CV-diagnostic attributes that predict/predict_proba/transform never read but that can dwarf
# the model in a pickle (LogisticRegressionCV.coefs_paths_ is n_classes x n_folds x n_Cs x n_features).
# Deliberately excludes `scores_`: SelectKBest reads it in transform().
_INFERENCE_UNUSED_ESTIMATOR_ATTRS: tuple[str, ...] = (
    "coefs_paths_",  # LogisticRegressionCV
    "cv_results_",  # GridSearchCV / RidgeCV(store_cv_results=True)
    "mse_path_",  # LassoCV / ElasticNetCV
)


class SKLearnTransform(Transform, SampleWeightMixin):
    """Adapt a scikit-learn estimator to the XDFlow transform API.

    The wrapped estimator receives a two-dimensional matrix with samples along
    `sample_dim`. If `target_coord` is provided, the coordinate values are
    extracted and passed as `y` during fit; otherwise the estimator is treated as
    unsupervised. Sample weights can be forwarded from a coordinate when the
    estimator's `fit` method accepts `sample_weight`.

    Keyword arguments not owned by the wrapper or its parent classes are passed
    to the estimator constructor and preserved for cloning.

    Notes:
        This class participates in cooperative multiple inheritance with
        `Predictor` through `SKLearnPredictor`. Wrapper hyperparameters should be
        explicit attributes; estimator parameters live only on the wrapped
        estimator, which `get_params`, `set_params` and `clone` all read.
    """

    is_stateful: bool = True
    _supports_transform_sel: bool = False  # should not support transform_sel because dims are changed

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str | None = None,
        _estimator_instance: BaseEstimator
        | None = None,  # SKLearnPredictor needs to instantiate the estimator; only used internally, not by user
        sample_weight_coord: str | None = "sample_weight",
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the wrapper and the underlying estimator.

        Args:
            estimator_cls: Uninitialized scikit-learn estimator class.
            sample_dim: Dimension whose entries are samples.
            target_coord: Optional coordinate used as supervised target `y`.
            _estimator_instance: Pre-built estimator used internally by
                `SKLearnPredictor` after task-type wrapping.
            sample_weight_coord: Coordinate containing optional sample weights.
                Set to None to disable sample-weight forwarding.
            sel: Label selection applied before fitting or transforming.
            drop_sel: Label selection dropped before fitting or transforming.
            **kwargs: Parent-class options and estimator constructor arguments.
        """
        # For cooperative inheritance, we pass all kwargs up the chain.
        # for SKLearnPredictor, SKLearnTransform is initialized first, then Predictor.
        super().__init__(sample_dim=sample_dim, target_coord=target_coord, sel=sel, drop_sel=drop_sel, **kwargs)

        parent_param_names = collect_super_init_param_names(type(self), SKLearnTransform)
        self.estimator_cls = estimator_cls  # needed for clone
        self.sample_weight_coord = sample_weight_coord
        self._fit_param_support_cache: dict[str, bool] = {}

        # Estimator kwargs are everything not used by Transform or Predictor. They are not kept:
        # from here on the live estimator is the one source of truth for its parameters.
        if _estimator_instance is not None:
            self.estimator: Any = _estimator_instance
        else:
            self.estimator: Any = build_estimator(
                estimator_cls, {k: v for k, v in kwargs.items() if k not in parent_param_names}
            )

        if not hasattr(self.estimator, "fit"):
            raise TypeError(
                f"The provided estimator class '{estimator_cls.__name__}' must produce an object with a 'fit' method."
            )

        self.sample_dim = sample_dim
        self.target_coord = target_coord

    def _prepare_data(self, data: xr.DataArray) -> tuple[np.ndarray, pd.Index]:
        """
        Validates that data is 2D and returns it in the expected format.
        """
        if data.ndim != 2:
            raise ValueError(
                f"Input data must be 2D (samples, features). Received data with {data.ndim} dimensions: {data.dims}"
            )

        if data.dims[0] != self.sample_dim:
            data = data.transpose(self.sample_dim, ...)

        return data.values, data.coords[self.sample_dim]

    def _resolve_target_coords(self, data: xr.DataArray) -> list[str]:
        """
        Resolve target coordinates from string, list, or pattern.

        Returns:
            List of resolved target coordinate names
        """
        if self.target_coord is None:
            raise ValueError("target_coord must be set to resolve supervised targets.")
        return resolve_target_coords(self.target_coord, data)

    def _wrap_for_resolved_targets(self, resolved_targets: list[str]) -> None:
        """Wrap the estimator for multi-target fitting when a target pattern resolved to several targets.

        Pattern-based target specs (e.g. ``"out_*"``) are resolved only at fit time, after construction-time
        wrapping was decided.
        """
        if (
            hasattr(self, "multi_output")
            and not self.multi_output  # Not already wrapped (only for Predictors)
            and hasattr(self, "is_classifier")
            and len(resolved_targets) > 1  # Pattern resolved to multiple targets
            and isinstance(self.target_coord, str)
            and "*" in self.target_coord  # Was a pattern
            and hasattr(self, "_base_estimator_cls")
            and inspect.isclass(self._base_estimator_cls)  # Skip lambdas/callables
        ):
            live_params = sklearn_clone(self.estimator).get_params(deep=False)
            if getattr(self, "is_multilabel", False):
                estimator_cls = MultiOutputClassifierFactory(cast(type[BaseEstimator], self._base_estimator_cls))
                self.estimator = estimator_cls(**live_params)
                self.multi_output = True
            elif not self.is_classifier:
                estimator_cls = MultiOutputRegressorFactory(cast(type[BaseEstimator], self._base_estimator_cls))
                self.estimator = estimator_cls(**live_params)
                self.multi_output = True

    def _record_fit_labels(self, data: xr.DataArray) -> None:
        """Remember the labels of every non-sample dimension seen at fit (e.g. features or channels)."""
        self._fit_labels = {
            dim: data.indexes[dim] for dim in data.dims if dim != self.sample_dim and dim in data.indexes
        }
        self._fit_dims = data.dims

    def _align_to_fit(self, data: xr.DataArray) -> xr.DataArray:
        """Order every non-sample dimension as it was at fit.

        The estimator sees values by position, so input whose features or channels arrive in another order
        would otherwise be silently mis-assigned. Labels missing from the new data raise.
        """
        fit_dims = getattr(self, "_fit_dims", None)  # absent in models pickled before dim order was recorded
        if fit_dims is not None and set(data.dims) == set(fit_dims) and data.dims != fit_dims:
            data = data.transpose(*fit_dims)  # N-D estimators read axes by position, like features
        for dim, fitted in getattr(self, "_fit_labels", {}).items():
            if dim not in data.indexes or data.indexes[dim].equals(fitted):
                continue
            positions = data.indexes[dim].get_indexer(fitted)
            if (positions < 0).any():
                missing = list(fitted[positions < 0][:5])
                raise ValueError(
                    f"{type(self).__name__} was fitted with '{dim}' labels missing from this data: {missing}"
                )
            data = data.isel({dim: positions})
        return data

    def _fit(self, container: DataContainer, **kwargs) -> "SKLearnTransform":
        """Fits the scikit-learn estimator (supervised or unsupervised)."""
        data = container.data
        self._record_fit_labels(data)
        X, sample_index = self._prepare_data(data)

        y = None
        if self.target_coord:
            # Resolve target coordinates (handles patterns, lists, and single coords)
            resolved_targets = self._resolve_target_coords(data)
            self._resolved_target_coords = resolved_targets
            y = extract_target_array(resolved_targets, data, validate=False)

            self._wrap_for_resolved_targets(resolved_targets)

        # If supervised, delegate any encoding decisions to Predictor upstream
        fit_kwargs = self._build_fit_kwargs(data, sample_index)

        # Check if estimator uses SklearnCVAdapter and automatically set container context
        from xdflow.cv.sklearn_adapter import set_cv_container

        cv_param = getattr(self.estimator, "cv", None)
        uses_cv_adapter = cv_param is not None and hasattr(cv_param, "cross_validator")

        if uses_cv_adapter:
            # Automatically set container context for SklearnCVAdapter
            with set_cv_container(container):
                if y is not None:
                    self.estimator.fit(X, y, **fit_kwargs)
                else:
                    self.estimator.fit(X, **fit_kwargs)
        else:
            # Normal fit without context
            if y is not None:
                self.estimator.fit(X, y, **fit_kwargs)
            else:
                self.estimator.fit(X, **fit_kwargs)
        return self

    def _clone_estimator_cls(self) -> Any:
        """The estimator class ``clone()`` rebuilds from (the unwrapped class after automatic multi-output wrapping)."""
        if hasattr(self, "_base_estimator_cls") and hasattr(self, "multi_output"):
            return self._base_estimator_cls
        return self.estimator_cls

    def _param_estimator(self) -> Any:
        """The estimator whose parameters this wrapper exposes and clones.

        An automatic ``MultiOutput*`` wrapper (``multi_output``, a factory, or a fit-time pattern) is
        looked through, because ``clone()`` rebuilds it from the inner model's parameters. Only when
        ``estimator_cls`` is ``MultiOutputRegressor``/``MultiOutputClassifier`` itself does ``clone()``
        need the wrapper's own parameters.
        """
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            return None
        if self._clone_estimator_cls() in (MultiOutputClassifier, MultiOutputRegressor):
            return estimator
        # Look through wrappers xdflow added (automatic multi_output or its factories); a user-supplied wrapper
        # factory such as functools.partial(MultiOutputRegressor, ...) is cloned with the wrapper's own params.
        added_by_xdflow = getattr(self, "multi_output", False) or isinstance(
            self._clone_estimator_cls(), (MultiOutputClassifierFactory, MultiOutputRegressorFactory)
        )
        if added_by_xdflow and isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor)):
            return estimator.estimator
        return estimator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Wrapper parameters plus the live parameters of the wrapped estimator."""
        params = super().get_params(deep=deep)
        estimator = self._param_estimator()
        if estimator is not None:
            params.update(estimator.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> "SKLearnTransform":
        """Route estimator parameters to the live estimator and the rest to the wrapper."""
        estimator = self._param_estimator()
        owned = estimator.get_params(deep=True) if estimator is not None else {}
        if estimator is not None and estimator is not self.__dict__.get("estimator"):
            # Automatic MultiOutput wrapper: `estimator__x` means the inner model's x. The wrapper itself is
            # rebuilt by clone(), so writes to its own params would be silently lost.
            translated = {}
            for key, value in params.items():
                if key.startswith("estimator__"):
                    key = key[len("estimator__") :]
                    if key not in owned:
                        raise ValueError(
                            f"'estimator__{key}' targets the automatic MultiOutput wrapper, which clone() rebuilds; "
                            "set the inner model's parameters instead."
                        )
                translated[key] = value
            params = translated
        estimator_params = {k: v for k, v in params.items() if k in owned}
        if estimator_params:
            estimator.set_params(**estimator_params)
        super().set_params(**{k: v for k, v in params.items() if k not in owned})
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy assignment of estimator parameters (``wrapper.alpha = 5``) onto the live estimator."""
        estimator = self._param_estimator()
        if (
            estimator is not None
            and name != "estimator"
            and not name.startswith("_")
            and name in estimator.get_params()
        ):
            estimator.set_params(**{name: value})
        else:
            super().__setattr__(name, value)

    def _get_clone_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs plus the estimator's *current* parameters.

        Read live rather than from a construction-time snapshot, so ``set_params`` survives ``clone()``
        (cross-validation clones once per fold). ``sklearn.base.clone`` drops fitted state and copies
        nested estimators instead of sharing them with the original.
        """
        kwargs = super()._get_clone_kwargs()
        if "estimator_cls" in kwargs:  # subclasses that fix their estimator class do not take it
            kwargs["estimator_cls"] = self._clone_estimator_cls()
        return {**kwargs, **sklearn_clone(self._param_estimator()).get_params(deep=False)}

    def _build_fit_kwargs(self, data: xr.DataArray, sample_index: pd.Index) -> dict[str, Any]:
        """Collect optional keyword arguments to forward to estimator.fit."""
        fit_kwargs: dict[str, Any] = {}

        sample_weight = self._extract_sample_weights(data, sample_index)
        if sample_weight is not None:
            if self._supports_fit_param("sample_weight"):
                fit_kwargs["sample_weight"] = sample_weight
            else:
                warnings.warn(
                    f"Ignoring coordinate '{self.sample_weight_coord}' because "
                    f"{self.estimator.__class__.__name__}.fit does not accept 'sample_weight'.",
                    RuntimeWarning,
                )
        return fit_kwargs

    def _supports_fit_param(self, param_name: str) -> bool:
        """Check whether the wrapped estimator accepts a specific fit parameter."""
        if param_name not in self._fit_param_support_cache:
            try:
                sig = signature(self.estimator.fit)
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Could not inspect signature for {self.estimator.__class__.__name__}.fit "
                    f"(error: {exc}); assuming it does not accept '{param_name}'.",
                    RuntimeWarning,
                )
                self._fit_param_support_cache[param_name] = False
            else:
                params = sig.parameters
                accepts = param_name in params or any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
                self._fit_param_support_cache[param_name] = accepts
        return self._fit_param_support_cache[param_name]

    def _disable_for_inference(self) -> None:
        """Drop CV-diagnostic estimator attributes during ``prepare_for_inference`` to keep pickles small."""
        for attr in _INFERENCE_UNUSED_ESTIMATOR_ATTRS:
            if hasattr(self.estimator, attr):
                delattr(self.estimator, attr)

    # _extract_sample_weights is inherited from SampleWeightMixin


class SKLearnTransformer(SKLearnTransform):
    """Adapt a scikit-learn transformer to return a `DataContainer`.

    The estimator must implement `fit` and `transform`. Input data is arranged as
    `(sample_dim, features)`. The transformed matrix is returned with
    `sample_dim` preserved and a new feature-like dimension named by
    `output_dim_name`.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str | None = None,
        output_dim_name: str = "component",
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        sample_weight_coord: str | None = "sample_weight",
        **kwargs: Any,
    ):
        """Initialize a transformer wrapper.

        Args:
            estimator_cls: Uninitialized scikit-learn transformer class.
            sample_dim: Dimension whose entries are samples.
            target_coord: Optional supervised target coordinate for estimators
                whose `fit` accepts `y`.
            output_dim_name: Name of the non-sample output dimension.
            sel: Label selection applied before fitting or transforming.
            drop_sel: Label selection dropped before fitting or transforming.
            sample_weight_coord: Coordinate containing optional sample weights.
            **kwargs: Parent-class options and estimator constructor arguments.
        """
        super().__init__(
            estimator_cls=estimator_cls,
            sample_dim=sample_dim,
            target_coord=target_coord,
            sel=sel,
            drop_sel=drop_sel,
            sample_weight_coord=sample_weight_coord,
            **kwargs,
        )
        self.output_dim_name = output_dim_name
        if not hasattr(self.estimator, "transform"):
            raise TypeError(
                f"The provided estimator class '{self.estimator.__class__.__name__}' must have a 'transform' method."
            )

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Applies the estimator's transform method to the data."""
        data = self._align_to_fit(container.data)
        X, sample_index = self._prepare_data(data)

        transformed_data = self.estimator.transform(X)
        assert transformed_data.ndim == 2, (
            f"Sklearn estimators should output 2D data, but got {transformed_data.ndim} dimensions."
        )

        output_coords = {self.sample_dim: sample_index.values}
        output_dims = [self.sample_dim, self.output_dim_name]

        # Preserve all coordinates that are associated with the sample dimension
        for coord_name, coord_data in data.coords.items():
            if coord_name != self.sample_dim and self.sample_dim in coord_data.dims:
                output_coords[coord_name] = coord_data
        output_coords[self.output_dim_name] = np.arange(transformed_data.shape[1])
        output_da = xr.DataArray(transformed_data, dims=output_dims, coords=output_coords, attrs=data.attrs)

        return DataContainer(output_da)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Determines the expected output dimensions."""
        if len(input_dims) != 2:
            raise ValueError(f"Expected 2 input dimensions, but got {len(input_dims)}")
        return (self.sample_dim, self.output_dim_name)


class _EstimatorTemplate:
    """Picklable factory building fresh clones of an estimator instance, with parameter overrides."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.__name__ = type(estimator).__name__

    def __call__(self, **params: Any) -> BaseEstimator:
        return sklearn_clone(self.estimator).set_params(**params)

    def __repr__(self) -> str:
        return f"_EstimatorTemplate({self.estimator!r})"


class SKLearnPredictor(SKLearnTransform, Predictor):
    """Adapt a scikit-learn estimator to the XDFlow predictor API.

    The estimator is fitted on a two-dimensional matrix plus target coordinate
    values, then exposed through `predict`, `predict_proba`, and `transform`.
    Classifier or regressor mode is auto-detected from the estimator when
    possible, or can be supplied with `is_classifier`.

    Multi-target regression and multilabel classification can be wrapped with
    scikit-learn's multi-output estimators by setting `multi_output=True`, or by
    passing multiple target coordinates where wrapping can be inferred.
    """

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str | list[str],
        encoder: LabelEncoder | None = None,
        proba: bool = False,
        is_classifier: bool | None = None,
        multi_output: bool = False,
        is_multilabel: bool = False,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        sample_weight_coord: str | None = "sample_weight",
        **kwargs: Any,
    ):
        """Initialize a predictor wrapper.

        Args:
            estimator_cls: Uninitialized scikit-learn estimator class.
            sample_dim: Dimension whose entries are samples.
            target_coord: Target coordinate name, target coordinate list, or
                wildcard pattern resolved during fit.
            encoder: Optional label encoder for single-label classifiers.
            proba: Whether `transform` should call `predict_proba` instead of
                `predict`.
            is_classifier: Explicitly set classifier or regressor mode. If None,
                task type is inferred from the estimator.
            multi_output: Whether to wrap the estimator for multi-target
                regression or multilabel classification.
            is_multilabel: Whether targets are multiple binary coordinates.
            sel: Label selection applied before fitting or transforming.
            drop_sel: Label selection dropped before fitting or transforming.
            sample_weight_coord: Coordinate containing optional sample weights.
            **kwargs: Estimator constructor arguments plus parent-class options.
        """
        parent_param_names = collect_super_init_param_names(type(self), SKLearnTransform)
        # Separate kwargs for the estimator from the rest
        estimator_kwargs = {k: v for k, v in kwargs.items() if k not in parent_param_names}

        # Store the original estimator class before potential wrapping
        self._base_estimator_cls = estimator_cls

        _base_instance = build_estimator(estimator_cls, estimator_kwargs)

        # Auto-detect or use manual override before deciding on multi-output wrapping.
        if is_classifier is None:
            if is_multilabel:
                is_classifier = True
            elif sklearn_is_classifier(_base_instance):
                is_classifier = True
            elif sklearn_is_regressor(_base_instance):
                is_classifier = False
            else:
                raise ValueError(
                    f"Could not auto-detect task type for {self._base_estimator_cls.__name__}. "
                    "Please explicitly specify is_classifier=True or is_classifier=False."
                )

        has_multiple_targets = isinstance(target_coord, (list, tuple)) and len(target_coord) > 1
        supports_multi_output_task = (not is_classifier) or is_multilabel
        if not multi_output and inspect.isclass(estimator_cls) and has_multiple_targets and supports_multi_output_task:
            multi_output = True

        self.multi_output = multi_output

        if multi_output:
            from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

            from xdflow.transforms.multi_output_wrapper import (
                MultiOutputClassifierFactory,
                MultiOutputRegressorFactory,
            )

            _already_multioutput = estimator_cls in (MultiOutputRegressor, MultiOutputClassifier) or isinstance(
                estimator_cls, (MultiOutputRegressorFactory, MultiOutputClassifierFactory)
            )
            if not _already_multioutput:
                original_cls = estimator_cls
                factory = MultiOutputClassifierFactory if is_multilabel else MultiOutputRegressorFactory
                estimator_cls = factory(cast(type[BaseEstimator], original_cls))
                if kwargs.get("verbose", False):
                    print(
                        f"[SKLearnPredictor] Wrapping {original_cls.__name__} with "
                        f"{'MultiOutputClassifier' if is_multilabel else 'MultiOutputRegressor'} "
                        "for multi-target prediction"
                    )

            _estimator_instance = build_estimator(estimator_cls, estimator_kwargs)
        else:
            _estimator_instance = _base_instance

        if multi_output and is_classifier and not is_multilabel:
            raise ValueError(
                "multi_output=True with is_classifier=True requires is_multilabel=True. "
                "Multi-output classification is only supported in multilabel mode."
            )

        if proba and not hasattr(_estimator_instance, "predict_proba"):
            raise AttributeError(
                f"Estimator '{_estimator_instance.__class__.__name__}' has no method 'predict_proba' but 'proba' was set to True."
            )

        # Correctly initialize both parent classes for cooperative multiple inheritance
        super().__init__(
            estimator_cls=estimator_cls,
            _estimator_instance=_estimator_instance,
            sample_dim=sample_dim,
            target_coord=target_coord,
            encoder=encoder,
            is_classifier=is_classifier,
            is_multilabel=is_multilabel,
            proba=proba,
            sel=sel,
            drop_sel=drop_sel,
            sample_weight_coord=sample_weight_coord,
            **kwargs,
        )

    # _transform implemented by Predictor

    @classmethod
    def from_estimator(
        cls, estimator: BaseEstimator, sample_dim: str, target_coord: str | list[str], **kwargs: Any
    ) -> "SKLearnPredictor":
        """Wrap an existing scikit-learn estimator or ``Pipeline`` instance, e.g. one you already use.

        The instance is a template: it is cloned, never fitted or mutated, and its parameters stay tunable
        through ``set_params`` (``"lda__solver"`` for a step of a wrapped Pipeline). Input with more than two
        dims is passed sample-first, so pipelines that take ``(trial, channel, time)`` arrays, such as
        ``make_pipeline(mne.decoding.CSP(), LinearDiscriminantAnalysis())``, work unchanged.

        Args:
            estimator: An unfitted scikit-learn estimator or Pipeline.
            sample_dim: Dimension whose entries are samples.
            target_coord: Target coordinate name(s).
            **kwargs: Other ``SKLearnPredictor`` options (``proba``, ``is_classifier``, ...).
        """
        return cls(_EstimatorTemplate(estimator), sample_dim=sample_dim, target_coord=target_coord, **kwargs)

    def _prepare_data(self, data: xr.DataArray) -> tuple[np.ndarray, pd.Index]:
        """Sample dimension first. Input with more than two dims is passed through as is, for estimators
        (or Pipelines) that take N-D arrays; estimators that need 2-D input reject it themselves."""
        if self.sample_dim not in data.dims:
            raise ValueError(f"Sample dimension '{self.sample_dim}' not found in data dims {data.dims}.")
        data = data.transpose(self.sample_dim, ...)
        return data.values, data.coords[self.sample_dim]

    def _predict(self, data: xr.DataArray, **kwargs) -> np.ndarray:
        """
        Implements the core prediction logic, returning the predictions (based on target_coord values)
        """
        if not hasattr(self.estimator, "predict"):
            raise AttributeError(f"Estimator '{self.estimator.__class__.__name__}' has no method 'predict'.")
        X, _ = self._prepare_data(self._align_to_fit(data))
        return self.estimator.predict(X)

    def _predict_proba(self, data: xr.DataArray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Implements the core probability prediction logic. Returns raw probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - probabilities: For single-label, shape (n_samples, n_classes). For multilabel,
                  shape (n_samples, n_targets) with positive-class probabilities.
                - class_labels: Single-label estimator classes, or target indices for multilabel.
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Estimator '{self.estimator.__class__.__name__}' has no method 'predict_proba'.")
        X, _ = self._prepare_data(self._align_to_fit(data))
        raw_probabilities = self.estimator.predict_proba(X)

        if self.is_multilabel:
            if isinstance(raw_probabilities, list):
                probabilities = np.column_stack([p[:, 1] if p.shape[1] == 2 else p[:, -1] for p in raw_probabilities])
            else:
                probabilities = raw_probabilities
            return probabilities, np.arange(probabilities.shape[1])

        return raw_probabilities, self.estimator.classes_

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions for the `.transform()` method.
        The public `.predict()` method will always produce a 1D output.
        """
        if self.sample_dim not in input_dims:
            raise ValueError(f"Input dims {input_dims} must include the sample dimension '{self.sample_dim}'.")

        if self.proba and not self.is_multilabel:
            return (self.sample_dim, "class")
        elif self.is_multi_target or self.is_multilabel:
            return (self.sample_dim, "target")
        else:
            return (self.sample_dim, "prediction")
