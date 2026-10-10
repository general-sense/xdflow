from __future__ import annotations

import inspect
from inspect import Parameter, signature
from typing import Any


def _get_declared_init_params(cls: type) -> set[str]:
    """Return explicit __init__ parameter names for a class."""
    params: set[str] = set()
    init = getattr(cls, "__init__", None)
    if init is None:
        return params

    try:
        sig = signature(init)
    except (TypeError, ValueError):
        return params

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        params.add(name)

    extra = getattr(cls, "_cooperative_init_kwarg_names", None)
    if extra:
        params.update(extra)

    return params


def collect_super_init_param_names(child_cls: type[Any], stop_class: type[Any]) -> set[str]:
    """
    Collect the names of constructor parameters consumed by cooperative superclasses.

    Args:
        child_cls: The most-derived class participating in cooperative init.
        stop_class: The class whose direct superclasses should be inspected.

    Returns:
        set[str]: Names of keyword arguments expected by the remaining classes in the MRO.
    """
    mro = child_cls.mro()
    if stop_class not in mro:
        raise ValueError(f"{stop_class.__name__} is not in the MRO of {child_cls.__name__}.")

    collect = False
    params: set[str] = set()
    for cls in mro:
        if cls is stop_class:
            collect = True
            continue
        if not collect or cls is object:
            continue
        params.update(_get_declared_init_params(cls))
    return params


def build_estimator(factory: Any, kwargs: dict[str, Any]) -> Any:
    """Build an estimator from a class or a factory (lambda, ``functools.partial``, xdflow wrapper factory).

    ``clone()`` passes back every current estimator parameter, but a factory may accept fewer arguments than
    the estimator it builds exposes, or accept ``**kwargs`` and ignore them. The factory gets the arguments
    it declares; afterwards, any argument it did not take, or whose value did not come out as requested, is
    set on the built estimator (unknown names raise in ``set_params``).
    """
    if inspect.isclass(factory) or not kwargs or getattr(factory, "_forwards_kwargs_to_base", False):
        # Classes take their params directly; xdflow's wrapper factories build their base model with
        # build_estimator themselves, and their own params share names with the base model's.
        return factory(**kwargs)
    try:
        parameters = signature(factory).parameters
    except (TypeError, ValueError):  # signature not introspectable
        return factory(**kwargs)
    takes_any = any(p.kind is Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = dict(kwargs) if takes_any else {k: v for k, v in kwargs.items() if k in parameters}
    estimator = factory(**accepted)
    get_params = getattr(estimator, "get_params", None)
    current = get_params(deep=False) if callable(get_params) else {}
    to_set = {
        k: v
        for k, v in kwargs.items()
        if k not in accepted or (k in current and current[k] is not v and not _equal_param(current[k], v))
    }
    if to_set:
        estimator.set_params(**to_set)
    return estimator


def _equal_param(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # arrays and objects without a scalar equality
        return False
