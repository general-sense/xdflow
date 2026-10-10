"""Every transform honours the constructor contract that `clone()` and the tuner depend on.

The tuner clones a template per trial, applies suggestions with `set_params`, then clones again for
every CV fold. That only works if a transform keeps each constructor argument where `clone()` finds
it, and `set_params` changes the same state a fresh construction would. Hand-picked clone tests
missed two transforms whose `clone()` always raised, so this file walks every `Transform` subclass:
a new transform is covered the day it is added.

For each concrete class, built from `_build_kwargs`:

- `clone()` works, and `get_params()` reports every named constructor argument;
- for each argument `_vary` can change, `set_params(arg=new)` leaves the object, and its clone, in
  the same state as a fresh construction with `arg=new`. Comparing whole-object state catches values
  computed in `__init__` without having to know which attribute fit reads. The build-time arguments
  in `_BUILD_TIME_ARGS` are left out;
- an argument a parent class accepts through `**kwargs` survives `clone()`;
- an unknown constructor argument, and an unknown `set_params` name, raise.
"""

import dataclasses
import importlib
import inspect
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import xdflow
from xdflow.core.base import Transform

_ROOT = Path(xdflow.__file__).parent


def _import_transform_modules() -> None:
    for package in ("transforms", "composite"):
        for path in sorted((_ROOT / package).rglob("*.py")):
            try:
                importlib.import_module(".".join(("xdflow", *path.relative_to(_ROOT).with_suffix("").parts)))
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] == "xdflow":
                    raise
                # optional extra (lightgbm, nitime, ...) not installed: its transforms are not covered


def _subclasses(cls: type) -> set[type]:
    found = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


_import_transform_modules()

# Base classes: each concrete aligner supplies its _STRATEGY_MAP; each @step function its own subclass
_NOT_CONCRETE = {"AdaptiveTransform", "_FunctionStep"}
CLASSES = sorted(
    (
        c
        for c in _subclasses(Transform)
        if c.__module__.startswith("xdflow.")
        and ".tests." not in c.__module__
        and not inspect.isabstract(c)
        and c.__name__ not in _NOT_CONCRETE
    ),
    key=lambda c: c.__name__,
)


# --- building each class ----------------------------------------------------------------------


class _StubAdaptEstimator:
    """Stands in for an `adapt` estimator; the real ones need the optional `adapt` package."""

    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_

    def fit(self, X, y, Xt, yt):
        return self

    def transform(self, X):
        return X


def _identity():
    from xdflow.transforms.basic_transforms import IdentityTransform

    return IdentityTransform()


def _ensemble_members():
    from sklearn.linear_model import LogisticRegression

    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    return [
        SKLearnPredictor(estimator_cls=LogisticRegression, sample_dim="trial", target_coord="stimulus", proba=True)
        for _ in range(2)
    ]


def _cls(module: str, name: str) -> Callable[[], type]:
    return lambda: getattr(importlib.import_module(module), name)


# Values for required arguments, by name: a new transform whose required arguments all appear here is
# covered without an entry of its own.
_BY_NAME: dict[str, Callable[[], Any]] = {
    "adapt_estimator_cls": lambda: _StubAdaptEstimator,
    "anchor_labels": lambda: ["x"],
    "by_dim": lambda: "trial",
    "per_dim": lambda: "channel",
    "grid_layout": lambda: np.arange(4).reshape(2, 2),
    "reference_channel": lambda: "ref",
    "choices": lambda: {"a": _identity(), "b": _identity()},
    "class_coord": lambda: "stimulus",
    "coord_name": lambda: "stimulus",
    "dim": lambda: "flat_channel__time",
    "dims": lambda: ["channel"],
    "end": lambda: 100,
    "domain_coord": lambda: "session",
    "estimator_cls": _cls("sklearn.linear_model", "Ridge"),
    "freq_bands": lambda: [(4.0, 8.0), (8.0, 12.0)],
    "fs": lambda: 1000.0,
    "func": lambda: lambda container: container,
    "group_coord": lambda: "session",
    "highcut": lambda: 10.0,
    "join_dim": lambda: "feature",
    "lowcut": lambda: 1.0,
    "members": _ensemble_members,
    "n_trials": lambda: 5,
    "name": lambda: "p",
    "phases_to_grab": lambda: [0, 1],
    "rename_map": lambda: {"a": "b"},
    "sample_dim": lambda: "trial",
    "sfreq": lambda: 1000.0,
    "source_domain": lambda: "s1",
    "source_domain_label": lambda: "s1",
    "start": lambda: 0,
    "steps": lambda: [("id", _identity())],
    "target_coord": lambda: "stimulus",
    "target_domain": lambda: "s2",
    "target_domain_label": lambda: "s2",
    "target_domains": lambda: ["s2"],
    "target_group": lambda: "s2",
    "transform_template": _identity,
    "transforms_list": lambda: [("a", _identity())],
}

_PCA_FRACTION = {"pca_frac_to_keep": lambda: 0.9}
# Where the name alone does not give a valid value, or an optional argument must be set to build.
_BY_CLASS: dict[str, dict[str, Callable[[], Any]]] = {
    "BandwiseFeaturePCA": _PCA_FRACTION,
    "CCAAligner": {"group_coord": lambda: "session"},
    "CoralAligner": {"group_coord": lambda: "session", "target_group": lambda: "s2"},
    "FlattenTransform": {"dims": lambda: ("channel", "time")},
    "GCCAAligner": {"group_coord": lambda: "session"},
    "GlobalColoringProjection": _PCA_FRACTION,
    "GlobalFeaturePCA": _PCA_FRACTION,
    "GlobalZCAWhitening": _PCA_FRACTION,
    "JDAAligner": {"group_coord": lambda: "session"},
    "KCCAAligner": {"group_coord": lambda: "session"},
    "LGBMPredictor": {"estimator_cls": _cls("lightgbm", "LGBMClassifier")},
    "LocalZCAWhitening": _PCA_FRACTION,
    "MCCAAligner": {"group_coord": lambda: "session"},
    "MultiBandFilterTransform": {"freq_bands": lambda: [("theta", (4.0, 8.0)), ("alpha", (8.0, 12.0))]},
    "ProcrustesAligner": {"group_coord": lambda: "session", "target_group": lambda: "s2"},
    "SAAligner": {"group_coord": lambda: "session", "target_group": lambda: "s2"},
    "SKLearnPredictor": {"estimator_cls": _cls("sklearn.linear_model", "LogisticRegression")},
    "SKLearnTransformer": {"estimator_cls": _cls("sklearn.decomposition", "PCA")},
    "TransformUnion": {"from_dims": lambda: ["feature"]},
    "ZCATransform": {"n_components": lambda: 3},
}


# Arguments that decide how `__init__` builds the object: which estimator, whether there is an
# encoder, which branches a composite has. `set_params` cannot re-apply them, and nothing tunes them,
# so they are set at construction. Keyed by the class that declares them.
_BUILD_TIME_ARGS: dict[str, set[str]] = {
    "AdaptiveTransform": {"strategy_name"},
    "EnsemblePredictor": {"normalize_weights"},
    "OptionalTransform": {"transform_template", "use", "name", "skip_name", "identity_rename"},
    "Predictor": {"is_classifier", "is_multilabel"},
    "SKLearnPredictor": {"multi_output"},
    "SKLearnTransform": {"estimator_cls"},
    "UnionWithInput": {"transform_template", "join_dim", "to_dim", "name"},
}


def _build_time_args(cls: type) -> set[str]:
    return set().union(*(_BUILD_TIME_ARGS.get(base.__name__, set()) for base in cls.__mro__))


def _named_args(cls: type) -> list[str]:
    return [
        name
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and not name.startswith("_") and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]


def _build_kwargs(cls: type) -> dict[str, Any]:
    overrides = _BY_CLASS.get(cls.__name__, {})
    kwargs = {name: make() for name, make in overrides.items()}
    for name, p in inspect.signature(cls.__init__).parameters.items():
        if name in kwargs or name == "self" or p.default is not p.empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name not in _BY_NAME:
            raise KeyError(f"{cls.__name__}: no value for required argument {name!r}; add one to _BY_NAME or _BY_CLASS")
        kwargs[name] = _BY_NAME[name]()
    return kwargs


def _build(cls: type, **changes: Any) -> Transform:
    try:
        return cls(**{**_build_kwargs(cls), **changes})
    except ModuleNotFoundError as exc:  # e.g. CSPTransform without the optional `mne` extra
        pytest.skip(f"{cls.__name__} needs an optional dependency: {exc}")


# --- comparing state --------------------------------------------------------------------------


def _state(value: Any, _path: frozenset[int] = frozenset()) -> Any:
    """A structural summary of `value` that two independently built objects can be compared by."""
    if id(value) in _path:
        return "<cycle>"
    path = _path | {id(value)}
    if isinstance(value, Transform):
        return (type(value).__name__, {k: _state(v, path) for k, v in sorted(vars(value).items())})
    if isinstance(value, BaseEstimator):
        return (type(value).__name__, _state(value.get_params(deep=False), path))
    if isinstance(value, dict):
        return {repr(k): _state(v, path) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_state(v, path) for v in value])
    if isinstance(value, (set, frozenset)):
        return ("set", sorted(repr(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tolist())
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, (str, int, float, bool, type(None), np.generic, np.dtype)):
        return value
    if isinstance(value, type) or inspect.isroutine(value):
        return getattr(value, "__qualname__", repr(value))
    if dataclasses.is_dataclass(value):
        return _state(dataclasses.asdict(value), path)
    if hasattr(value, "__dict__"):
        return (type(value).__name__, _state(vars(value), path))
    return type(value).__name__  # locks, dtypes, modules: identity is not configuration


def _differing_attributes(obj: Transform, reference: Transform) -> list[str]:
    ours, theirs = _state(obj)[1], _state(reference)[1]
    return sorted(k for k in ours.keys() | theirs.keys() if ours.get(k, "<missing>") != theirs.get(k, "<missing>"))


_NO_VARY = object()


def _vary(value: Any) -> Any:
    """A different value of the same type, or `_NO_VARY` when one cannot be made up generically."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float) and math.isfinite(value):
        return value / 2 if value else 0.5
    if isinstance(value, str):
        return value + "_x"
    return _NO_VARY


# --- the contract -----------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_builds_clones_and_reports_every_constructor_argument(cls):
    obj = _build(cls)
    obj.clone()
    reported = obj.get_params(deep=False)
    missing = [name for name in _named_args(cls) if name not in reported]
    assert not missing, f"get_params() does not report constructor arguments {missing}"


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_set_params_leaves_the_state_a_fresh_construction_has(cls):
    current = _build(cls).get_params(deep=False)
    build_time = _build_time_args(cls)
    mismatches = []
    for name in _named_args(cls):
        if name in build_time:
            continue
        new = _vary(current.get(name))
        if new is _NO_VARY:
            continue
        try:
            fresh = _build(cls, **{name: new})
        except (TypeError, ValueError, AssertionError, KeyError):
            continue  # not a value the constructor accepts, so there is nothing to compare against
        obj = _build(cls)
        obj.set_params(**{name: new})
        for label, candidate in (("the object", obj), ("its clone", obj.clone())):
            differing = _differing_attributes(candidate, fresh)
            if differing:
                mismatches.append(f"{name}={new!r}: {label} differs from a fresh construction in {differing}")
    assert not mismatches, "\n".join(mismatches)


def _parent_arg_names(cls: type) -> set[str]:
    names = set()
    for base in cls.__mro__[1:]:
        if "__init__" in vars(base):
            names |= set(_named_args(base))
        names |= set(getattr(base, "_cooperative_init_kwarg_names", ()))
    return names


# Values for parent arguments that a subclass may take through **kwargs. `calibrated_thresholds` is
# left out on purpose: it is set by calibration and `clone()` deliberately drops it.
_PARENT_VALUES: dict[str, Callable[[], Any]] = {
    "allow_unknown_targets": lambda: False,
    "drop_sel": lambda: {"stimulus": ["b"]},
    "sample_weight_coord": lambda: "weight",
    "sel": lambda: {"stimulus": ["a"]},
    "unknown_target_encoding": lambda: -7,
}


@pytest.mark.parametrize(
    "cls",
    [c for c in CLASSES if any(p.kind == p.VAR_KEYWORD for p in inspect.signature(c.__init__).parameters.values())],
    ids=lambda c: c.__name__,
)
def test_parent_arguments_taken_through_kwargs_survive_clone(cls):
    named, lost = set(_named_args(cls)), []
    for name in sorted((_parent_arg_names(cls) - named) & _PARENT_VALUES.keys()):
        value = _PARENT_VALUES[name]()
        try:
            obj = _build(cls, **{name: value})
        except (TypeError, ValueError):
            continue  # the class does not accept it
        if _state(obj.clone().get_params(deep=False).get(name)) != _state(value):
            lost.append(name)
    assert not lost, f"accepted through **kwargs but dropped by clone(): {lost}"


# Classes that hand any unclaimed keyword argument to a wrapped constructor that only runs at fit,
# so a wrong name fails there: loud, but late.
_WRAPPED_AT_FIT = {
    "AdaptWrapperTransform": "passed to an adapt estimator built only at fit, which reports a wrong name then",
}
_UNKNOWN_KWARG_EXCEPTIONS = {
    **_WRAPPED_AT_FIT,
    "LGBMPredictor": "LightGBM accepts any keyword as a booster parameter and ignores an unknown one",
}


def _expected_to_fail(reasons: dict[str, str]) -> list:
    """`CLASSES`, with the listed ones marked as known exceptions. Strict: once one starts passing,
    the test fails until its entry is removed, so an exception cannot outlive its reason."""
    return [
        pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=reasons[c.__name__]))
        if c.__name__ in reasons
        else c
        for c in CLASSES
    ]


@pytest.mark.parametrize("cls", _expected_to_fail(_UNKNOWN_KWARG_EXCEPTIONS), ids=lambda c: c.__name__)
def test_an_unknown_constructor_argument_raises(cls):
    with pytest.raises(TypeError):
        _build(cls, contract_typo_arg=1)


@pytest.mark.parametrize("cls", _expected_to_fail(_WRAPPED_AT_FIT), ids=lambda c: c.__name__)
def test_an_unknown_set_params_name_raises(cls):
    obj = _build(cls)
    with pytest.raises(ValueError):
        obj.set_params(contract_typo_arg=1)
