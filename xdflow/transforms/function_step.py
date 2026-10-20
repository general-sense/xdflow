"""Turn plain functions into pipeline steps."""

import inspect
from types import FunctionType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer

# Names the generated class already uses; a function parameter with one of these names would shadow it.
_RESERVED_PARAM_NAMES = {
    "func",
    "fit_func",
    "sel",
    "drop_sel",
    "transform_sel",
    "transform_drop_sel",
    "is_stateful",
    "input_dims",
    "output_dims",
}


def step(
    func: FunctionType | None = None,
    *,
    fit: FunctionType | None = None,
    input_dims: tuple[str, ...] = (),
    output_dims: tuple[str, ...] | None = None,
    check_independence: bool = True,
) -> Any:
    """Decorator that turns a function of an ``xr.DataArray`` into a pipeline step.

    The function's first parameter receives the data as an ``xr.DataArray``; its other parameters become the
    step's hyperparameters. The decorated name is a ``Transform`` class: call it with the hyperparameters to
    get a step, which clones and tunes like any built-in transform (``"power__band"`` for a step named
    ``"power"``). The original function stays available as ``.func``. Its results are not reused from the disk
    cache unless ``XDFLOW_CACHE_USER_CODE=1`` (see ``xdflow.utils.cache_utils.cache_user_code``).

    The function returns an ``xr.DataArray``, or a NumPy array with the same shape as its input (it is given
    the input's dims and coords). It receives a read-only view of the data, so in-place edits raise instead of
    changing the caller's data.

    **Steps without ``fit``** must compute each trial's output from that trial alone (filtering, cropping,
    band power...). That is what lets cross-validation run them once on all trials. Each call runs a quick
    check, rerunning the function on three trials alone and comparing their values and per-trial coordinates;
    a function that uses other trials (e.g. subtracting the mean over trials) raises an error that points to
    ``fit``.

    **Steps with ``fit``** learn from the training trials. ``fit(data, **hyperparameters)`` returns any state
    (e.g. a mean or a filter); the step function then receives it as its ``state`` argument. Cross-validation
    calls ``fit`` on each training fold only.

    Args:
        func: The function to wrap (when used as ``@step`` without arguments).
        fit: Function learning state from training data; makes the step stateful. It receives the data and
            the hyperparameters it names, and the step function must take a ``state`` parameter.
        input_dims: Dims the function requires, checked when the pipeline is built (optional).
        output_dims: Dims the function returns, used by pipeline dimension checks (optional).
        check_independence: Verify on every call that each trial's output depends on that trial only.

    Example:
        >>> @step
        ... def log_power(data, band=(8.0, 30.0)):
        ...     return np.log((data.sel(freq=slice(*band)) ** 2).mean("freq"))
        >>>
        >>> def channel_means(data):
        ...     return data.mean(("trial", "time"))
        >>>
        >>> @step(fit=channel_means)
        ... def recenter(data, state):
        ...     return data - state
    """

    def build(fn: FunctionType) -> type[Transform]:
        return _make_step_class(
            fn, fit, tuple(input_dims), None if output_dims is None else tuple(output_dims), check_independence
        )

    if func is not None:
        return build(func)
    return build


def _make_step_class(
    fn: FunctionType,
    fit_fn: FunctionType | None,
    input_dims: tuple[str, ...],
    output_dims: tuple[str, ...] | None,
    check_independence: bool,
) -> type[Transform]:
    if not inspect.isfunction(fn):
        raise TypeError(f"@step expects a function defined with def or lambda, got {fn!r}.")
    if fit_fn is not None and not inspect.isfunction(fit_fn):
        raise TypeError(f"fit must be a function defined with def or lambda, got {fit_fn!r}.")
    parameters = list(inspect.signature(fn).parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(f"The first parameter of {fn.__name__}() must take the data positionally.")
    hyperparameters = parameters[1:]
    if fit_fn is not None:
        if "state" not in {parameter.name for parameter in hyperparameters}:
            raise TypeError(f"{fn.__name__}() must take a 'state' parameter to receive what fit returns.")
        hyperparameters = [parameter for parameter in hyperparameters if parameter.name != "state"]
    for parameter in hyperparameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"{fn.__name__}() uses *{parameter.name}; step hyperparameters must be named so they can be "
                "cloned and tuned."
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"{fn.__name__}() parameter '{parameter.name}' is positional-only; make it named.")
        if parameter.name in _RESERVED_PARAM_NAMES or (fit_fn is None and parameter.name == "state"):
            raise TypeError(f"{fn.__name__}() parameter '{parameter.name}' clashes with a Transform attribute.")
    param_names = tuple(parameter.name for parameter in hyperparameters)

    fit_param_names: tuple[str, ...] = ()
    if fit_fn is not None:
        fit_parameters = list(inspect.signature(fit_fn).parameters.values())
        if not fit_parameters:
            raise TypeError(f"{fit_fn.__name__}() must take the training data as its first parameter.")
        fit_param_names = tuple(parameter.name for parameter in fit_parameters[1:])
        unknown = sorted(set(fit_param_names) - set(param_names))
        if unknown:
            raise TypeError(
                f"{fit_fn.__name__}() takes {unknown}, which {fn.__name__}() does not; hyperparameters are declared "
                f"on {fn.__name__}() and passed to both."
            )

    # Hyperparameters keep their order, kinds and defaults; "state" is dropped (fit supplies it, not the user).
    init_signature = inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [parameter.replace(annotation=inspect.Parameter.empty) for parameter in hyperparameters]
        + [
            inspect.Parameter("sel", inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("drop_sel", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
    )

    def init(self, *args: Any, **kwargs: Any) -> None:
        _FunctionStep.__init__(self, *args, **kwargs)

    init.__signature__ = init_signature  # ty: ignore[unresolved-attribute]  # read by clone() and set_params()

    namespace = {
        "__init__": init,
        "__module__": fn.__module__,
        "__qualname__": fn.__qualname__,
        "__doc__": fn.__doc__,
        "is_stateful": fit_fn is not None,
        "input_dims": input_dims,
        "output_dims": output_dims or (),
        "func": staticmethod(fn),
        "_fn": staticmethod(fn),
        "_fit_fn": staticmethod(fit_fn) if fit_fn is not None else None,
        "_param_names": param_names,
        "_fit_param_names": fit_param_names,
        "_init_signature": init_signature,
        "_declared_output_dims": output_dims,
        "_check_independence": check_independence,
    }
    return type(fn.__name__, (_FunctionStep,), namespace)


class _FunctionStep(Transform):
    """Base of the classes made by ``@step``; the subclass supplies the functions and their parameter names."""

    is_stateful = False
    _fn: FunctionType
    _fit_fn: FunctionType | None
    _param_names: tuple[str, ...]
    _fit_param_names: tuple[str, ...]
    _init_signature: inspect.Signature
    _declared_output_dims: tuple[str, ...] | None
    _check_independence: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        bound = self._init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        super().__init__(sel=bound.arguments["sel"], drop_sel=bound.arguments["drop_sel"])
        for name in self._param_names:
            setattr(self, name, bound.arguments[name])
        self._attach_functions()

    def _attach_functions(self) -> None:
        # Public, so cache keys hash the functions' code: editing them (e.g. in a notebook) invalidates cached results.
        self.func = type(self)._fn
        if type(self)._fit_fn is not None:
            self.fit_func = type(self)._fit_fn

    def _params(self, names: tuple[str, ...]) -> dict[str, Any]:
        return {name: getattr(self, name) for name in names}

    def _fit(self, container: DataContainer, **kwargs: Any) -> "_FunctionStep":
        fit_fn = type(self)._fit_fn
        if fit_fn is None:
            return self
        self._state = self._call(fit_fn, container.data, self._params(self._fit_param_names))
        return self

    def _transform(self, container: DataContainer, **kwargs: Any) -> DataContainer:
        data = container.data
        params = self._params(self._param_names)
        if self.is_stateful:
            if not hasattr(self, "_state"):
                raise RuntimeError(f"{self._fn.__name__} step is not fitted; call fit() first.")
            params["state"] = self._state
        result = self._as_output(self._call(self.func, data, params), data)
        if self._check_independence:
            self._verify_independence(data, params, result)
        return DataContainer(result)

    def _call(self, fn: FunctionType, data: xr.DataArray, params: dict[str, Any]) -> Any:
        try:
            return fn(_read_only(data), **params)
        except ValueError as exc:
            if "read-only" not in str(exc):
                raise
            raise ValueError(
                f"{fn.__name__}() modified its input in place. Steps must not change the data they receive; "
                "compute a new array instead (e.g. `data = data * 2` rather than `data *= 2`)."
            ) from exc

    def _as_output(self, result: Any, data: xr.DataArray) -> xr.DataArray:
        name = self._fn.__name__
        if isinstance(result, DataContainer):
            result = result.data
        if isinstance(result, np.ndarray):
            if result.shape != data.shape:
                raise ValueError(
                    f"{name}() returned a NumPy array of shape {result.shape} for input of shape {data.shape} "
                    f"{data.dims}. Return an xr.DataArray when the shape changes, so its dims are named."
                )
            result = data.copy(deep=False, data=result)
        if not isinstance(result, xr.DataArray):
            raise TypeError(f"{name}() must return an xr.DataArray or NumPy array, got {type(result).__name__}.")
        return _writable(result)  # it may hold (slices of) the read-only input: don't pass the lock downstream

    def _verify_independence(self, data: xr.DataArray, params: dict[str, Any], result: xr.DataArray) -> None:
        """Rerun on three trials alone and compare with their output when all trials were present.

        A quick check on every call, not a proof: it catches outputs (values or coordinates) that change with the
        other trials present, such as anything computed over the trial dim.
        """
        if "trial" not in data.dims or data.sizes["trial"] < 3:
            return
        name = self._fn.__name__
        hint = (
            "If it uses other trials, give it a fit function (@step(fit=...)); if it does not, disable this "
            "check with @step(check_independence=False)."
        )
        if "trial" not in result.dims:
            raise ValueError(f"{name}() returned no 'trial' dim, so its output is not computed per trial. {hint}")
        unlabeled = "trial" not in data.indexes  # no trial labels: trials are matched by position
        if unlabeled:
            if "trial" in result.indexes or result.sizes["trial"] != data.sizes["trial"]:
                raise ValueError(
                    f"{name}() changed the trials of data without a 'trial' coordinate, so they cannot be checked; "
                    f"label the trials (data.assign_coords(trial=...)). {hint}"
                )
            result = result.assign_coords(trial=np.arange(data.sizes["trial"]))
        trials = np.arange(data.sizes["trial"]) if unlabeled else data.trial.values
        if not (pd.Index(trials).is_unique and pd.Index(result.trial.values).is_unique):
            raise ValueError(f"Trial labels must be unique to check {name}(); relabel the trials. {hint}")
        kept = trials[pd.Index(result.trial.values).get_indexer(trials) >= 0]
        if kept.size == 0:
            raise ValueError(f"{name}() returned none of its input trials, so they cannot be checked. {hint}")
        # Interior trials: on ordered data the first and last trials hold the extremes, which a global min/max
        # normalisation would reproduce on any subset containing them.
        labels = list(dict.fromkeys([kept[kept.size // 4], kept[kept.size // 2], kept[(3 * kept.size) // 4]]))
        subset = data.isel(trial=labels) if unlabeled else data.sel(trial=labels)  # unlabeled, as in the first call
        try:
            alone = self._as_output(self._call(self.func, subset, params), subset)
        except Exception as exc:
            raise ValueError(f"Could not run {name}() on 3 trials alone ({exc}). {hint}") from exc
        if unlabeled and "trial" not in alone.indexes and alone.sizes.get("trial") == len(labels):
            alone = alone.assign_coords(trial=labels)
        if "trial" not in alone.dims or not set(labels) <= set(alone.trial.values):
            raise ValueError(f"{name}() keeps different trials when run on 3 trials alone. {hint}")
        alone, together = alone.sel(trial=labels), result.sel(trial=labels)
        if set(alone.dims) == set(together.dims):
            alone = alone.transpose(*together.dims)
        # Every coordinate: statistics over trials can hide in a non-trial one (a channel's mean over trials).
        coords = [str(coord) for coord in together.coords if coord != "trial"]
        same = _same(alone.variable, together.variable) and all(
            coord in alone.coords and _same(alone.coords[coord].variable, together.coords[coord].variable)
            for coord in coords
        )
        if not same:
            raise ValueError(
                f"{name}() gives a different output for a trial depending on which other trials are present, so it "
                "learns from the data (e.g. a mean over trials) or is random. Run on all trials before the data is "
                "split, it would leak test trials into training. Move what it learns into a fit function: "
                "@step(fit=...)."
            )

    def _log_history(self, source_container: DataContainer, container: DataContainer) -> DataContainer:
        container = super()._log_history(source_container, container)
        # The class name identifies the functions; the function objects would make the output unpicklable (their
        # module attribute is the class), which breaks caching.
        params = container.data.attrs["data_history"][-1]["params"]
        params.pop("func", None)
        params.pop("fit_func", None)
        return container

    def get_expected_output_dims(self, input_dims: tuple[str, ...], /) -> tuple[str, ...]:
        if self._declared_output_dims is None:
            raise NotImplementedError(
                f"Output dims of {self._fn.__name__}() are not declared; use @step(output_dims=(...)) to enable "
                "pipeline dimension checks."
            )
        return self._declared_output_dims

    def __getstate__(self) -> dict[str, Any]:
        # The functions are pickled through the class (its module attribute is the class, not the function).
        state = self.__dict__.copy()
        state.pop("func", None)
        state.pop("fit_func", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._attach_functions()


def _read_only(data: xr.DataArray) -> xr.DataArray:
    """The same data with its values and non-index coordinates behind read-only views, and copies of its
    (single-level) indexes, whose buffers pandas would otherwise share with the caller."""
    values = data.values.view()
    values.flags.writeable = False  # a view with its own flag: the caller's arrays stay writable
    locked = data.copy(deep=False, data=values)
    for name, index in data.indexes.items():
        if name in data.dims and not isinstance(index, pd.MultiIndex):
            locked = locked.assign_coords({name: (name, index.copy(deep=True), dict(data.coords[name].attrs))})
    for name, coord in data.coords.items():
        if name not in data.indexes and isinstance(coord.variable.data, np.ndarray):
            view = coord.variable.data.view()
            view.flags.writeable = False
            locked = locked.assign_coords({name: coord.variable.copy(deep=False, data=view)})
    return locked


def _writable(result: xr.DataArray) -> xr.DataArray:
    if isinstance(result.variable.data, np.ndarray) and not result.variable.data.flags.writeable:
        result = result.copy(deep=False, data=result.values.copy())
    for name, coord in result.coords.items():
        if isinstance(coord.variable.data, np.ndarray) and not coord.variable.data.flags.writeable:
            result = result.assign_coords({name: coord.variable.copy(deep=True)})
    return result


def _same(a: xr.Variable, b: xr.Variable) -> bool:
    """Equal shapes and values. Floating values may differ by a few hundred ulps, since batch size can change the
    last bits of FFTs and reductions; the absolute slack scales with the typical (median) magnitude, so one large
    feature does not hide differences in the others. Missing values compare equal."""
    if a.shape != b.shape:
        return False
    left, right = np.asarray(a.values), np.asarray(b.values)
    if np.issubdtype(right.dtype, np.inexact) and np.issubdtype(left.dtype, np.inexact):
        rtol = min(1e3 * float(np.finfo(right.dtype).eps), 1e-3)
        magnitudes = np.abs(right[np.isfinite(right)])
        scale = float(np.median(magnitudes)) if magnitudes.size else 0.0
        return bool(np.allclose(left, right, rtol=rtol, atol=rtol * scale, equal_nan=True))
    return bool(pd.Series(left.ravel()).equals(pd.Series(right.ravel())))
