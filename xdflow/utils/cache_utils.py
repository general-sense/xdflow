"""
Cache utilities for preprocessed and featurized data.

This module provides functions for caching and retrieving preprocessed and featurized data
based on input parameters. It uses a hash of the input parameters to create unique cache keys.
"""

import functools
import hashlib
import inspect
import json
import os
import pickle
import shutil
import threading
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from xdflow.core.data_container import DataContainer

DEFAULT_CACHE_DIR = Path(os.environ.get("XDFLOW_CACHE_DIR", Path.home() / ".cache" / "xdflow"))

# Default maximum cache size in GB
DEFAULT_MAX_CACHE_SIZE_GB = 50

# Default maximum cache size in bytes
DEFAULT_MAX_CACHE_SIZE = DEFAULT_MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024

# Default maximum age of cache files in days
DEFAULT_MAX_CACHE_AGE = 7


def _get_module_hash_from_obj(obj: Any) -> str | None:
    """Get hash of the module containing the object."""
    if inspect.isclass(obj) or inspect.isroutine(obj):
        module = inspect.getmodule(obj)
    else:
        module = inspect.getmodule(obj.__class__)

    if module is None:
        return None

    try:
        source_file = inspect.getsourcefile(module)
    except TypeError:  # built-in module
        return None
    if source_file is None:
        return None

    with open(source_file, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_all_children(obj: Any) -> list[Any]:
    """Recursively get all children of a composite object."""
    children = []
    if hasattr(obj, "children") and obj.children:
        for child in obj.children:
            children.append(child)
            children.extend(_get_all_children(child))
    return children


def get_module_hashes_for_object(instance: Any) -> dict[str, str]:
    """Return module path -> hash for an object and all its children.

    Args:
        instance: Composite object (e.g., Pipeline) whose module hashes to collect

    Returns:
        Dict mapping module file paths to md5 hashes of their contents
    """
    all_objects = [instance] + _get_all_children(instance)
    module_hashes: dict[str, str] = {}
    for obj in all_objects:
        try:
            module = inspect.getmodule(obj.__class__)
            if module is None:
                continue
            module_path = inspect.getsourcefile(module)
            module_hash = _get_module_hash_from_obj(obj)
            if module_path and module_hash is not None and module_path not in module_hashes:
                module_hashes[module_path] = module_hash
        except TypeError:
            # Happens for some built-in types
            continue
    return module_hashes


def _get_cache_path_for_key(prefix: str, key_dict: dict[str, Any]) -> Path:
    """Compute cache file path for a given key dict."""
    cache_key = hash_dict(key_dict)
    return _get_cache_dir(prefix) / f"{cache_key}.pkl"


def try_load_cached_object(prefix: str, key_dict: dict[str, Any]) -> Any | None:
    """Load a cached Python object by key if present, else return None.

    This uses the same directory structure and hashing as the function result cache.
    """
    # Clean and enforce limits before reading
    _cleanup_old_cache_files(DEFAULT_MAX_CACHE_AGE, prefix)
    _enforce_cache_size_limit(DEFAULT_MAX_CACHE_SIZE, prefix)

    cache_path = _get_cache_path_for_key(prefix, key_dict)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            return None
    return None


def save_cached_object(prefix: str, key_dict: dict[str, Any], obj: Any) -> None:
    """Persist a Python object in the cache under the key derived from key_dict."""
    cache_path = _get_cache_path_for_key(prefix, key_dict)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
    except PermissionError:
        # Skip caching if filesystem forbids writing (e.g., immutable files/attrs)
        return


def get_pipeline_cache_key_dict(func: Callable, instance: Any, *args, **kwargs) -> dict[str, Any]:
    """
    Generate a cache key dictionary for a pipeline.

    This function creates a detailed dictionary that includes:
    - The function's arguments.
    - The configuration of the pipeline instance and all its nested transforms.
    - The code hashes of the modules of the pipeline and all its transforms.

    Args:
        func: The function being called (e.g., fit_transform).
        instance: The pipeline instance.
        *args: Positional arguments to the function.
        **kwargs: Keyword arguments to the function.

    Returns:
        A dictionary to be hashed for the cache key.
    """
    # 1. Get function arguments
    bound_args = inspect.signature(func).bind(instance, *args, **kwargs)
    bound_args.apply_defaults()
    key_dict = {k: _get_object_metadata(v) for k, v in bound_args.arguments.items() if k != "self"}

    # 2. Get instance configuration (recursively)
    key_dict["config"] = _get_object_params(instance)

    # 3. Get module hashes for the instance and all its children
    all_objects = [instance] + _get_all_children(instance)
    module_hashes = {}
    for obj in all_objects:
        try:
            module = inspect.getmodule(obj.__class__)
            if module is None:
                continue
            module_path = inspect.getsourcefile(module)
            module_hash = _get_module_hash_from_obj(obj)
            if module_path and module_hash is not None and module_path not in module_hashes:
                module_hashes[module_path] = module_hash
        except TypeError:
            pass  # Happens for some built-in types

    key_dict["module_hashes"] = module_hashes

    return key_dict


def cache_result(
    prefix: str,
    max_size: int = DEFAULT_MAX_CACHE_SIZE,
    max_age_days: float = DEFAULT_MAX_CACHE_AGE,
    key_gen_func: Callable | None = None,
) -> Callable:
    """
    Decorator that caches function results based on all function and class instance parameters.

    Args:
        prefix: Prefix for the cache directory (e.g., 'preprocess', 'featurize')
        max_size: Maximum total cache size in bytes across all prefixes
        max_age_days: Maximum age of cache files in days
        key_gen_func: Optional function to generate the cache key dictionary.
                      If None, a default key generation logic is used.

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the instance if this is an instance method
            instance = args[0] if args and hasattr(args[0], "__dict__") else None

            # Only outputs are cached, so a hit would skip fitting: stateful instances always run.
            if instance and (not getattr(instance, "use_cache", False) or getattr(instance, "is_stateful", False)):
                return func(*args, **kwargs)

            # Clean up cache files
            _cleanup_old_cache_files(max_age_days, prefix)
            _enforce_cache_size_limit(max_size, prefix)

            # Verbosity does not change results, so it must not change the key
            key_kwargs = {k: v for k, v in kwargs.items() if k != "verbose"}
            if key_gen_func and instance:
                key_dict = key_gen_func(func, *args, **key_kwargs)
            else:
                # Get arguments of function
                bound_args = inspect.signature(func).bind(*args, **key_kwargs)
                bound_args.apply_defaults()
                key_dict = {
                    k: _get_object_metadata(v) for k, v in bound_args.arguments.items() if k != "self" or not instance
                }
                if instance:
                    # Make sure the class instance is the same
                    key_dict["config"] = _get_object_params(instance)
                    key_dict["module_hash"] = _get_module_hash_from_obj(instance)

            cache_key = hash_dict(key_dict)
            cache_path = _get_cache_dir(prefix) / f"{cache_key}.pkl"

            if kwargs.get("verbose", False):
                print(f"Cache {'hit' if cache_path.exists() else 'miss'}: {cache_path}")

            # Return cached result if it exists
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        # print(f"Loading cached result from {cache_path}")
                        return pickle.load(f)
                except (pickle.UnpicklingError, EOFError, FileNotFoundError):
                    # If cache is corrupted or gets deleted mid-read, recompute
                    pass

            # Compute result and cache it
            result = func(*args, **kwargs)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            except PermissionError:
                # Skip caching if filesystem forbids writing (e.g., immutable files/attrs)
                pass

            return result

        return wrapper

    return decorator


def _get_ndarray_metadata(arr: np.ndarray) -> dict[str, Any]:
    """Build a deterministic signature for a numpy array from its full contents."""
    return {"shape": tuple(arr.shape), "dtype": str(arr.dtype), "hash": joblib.hash(arr)}


def _get_datacontainer_metadata(dc: DataContainer) -> dict[str, Any]:
    """Build a deterministic signature for a DataContainer from its full values and coordinates.

    Hashing everything (not a sample) is what keeps a cache from returning results computed on
    different data or labels.
    """
    return {
        "shape": tuple(dc.shape),
        "dims": tuple(dc.dims),
        "data": _get_ndarray_metadata(np.asarray(dc.data.values)),
        "coord_metadata": {
            str(name): {"dims": tuple(coord.dims), **_get_ndarray_metadata(np.asarray(coord.values))}
            for name, coord in dc.data.coords.items()
        },
        # Transforms may read settings from attrs (e.g. a sampling rate). `data_history` only records the
        # steps already applied, which the step configuration in the key covers.
        "attrs": _get_object_metadata({str(k): v for k, v in dc.data.attrs.items() if k != "data_history"}),
    }


def _get_dataframe_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Build a deterministic signature for a pandas DataFrame from its full contents."""
    return {"shape": tuple(df.shape), "columns": tuple(str(c) for c in df.columns), "hash": joblib.hash(df)}


def _get_series_metadata(s: pd.Series) -> dict[str, Any]:
    """Build a deterministic signature for a pandas Series from its full contents."""
    return {"length": int(s.shape[0]), "name": str(s.name), "hash": joblib.hash(s)}


def _get_dict_metadata(d: dict[Any, Any]) -> dict[str, Any]:
    """Build a deterministic signature for a dictionary."""
    keys_sorted = sorted(d.keys(), key=lambda x: str(x))
    return {"length": len(d), "items": tuple((str(k), _get_object_metadata(d[k])) for k in keys_sorted)}


def _get_sequence_metadata(seq: Any) -> dict[str, Any]:
    """Build a deterministic signature for a list/tuple."""
    return {"type": type(seq).__name__, "items": tuple(_get_object_metadata(v) for v in seq)}


def _get_set_metadata(s: set) -> dict[str, Any]:
    """Build a deterministic signature for a set."""
    return {"type": "set", "items": tuple(_get_object_metadata(v) for v in sorted(s, key=lambda x: str(x)))}


def _const_repr(const: Any) -> str:
    if inspect.iscode(const):
        return _code_fingerprint(const)
    if isinstance(const, frozenset):  # set literals: repr order depends on per-process hash randomization
        return repr(sorted(_const_repr(c) for c in const))
    if isinstance(const, tuple):
        return repr(tuple(_const_repr(c) for c in const))
    return repr(const)


def _code_fingerprint(code: Any) -> str:
    """Hash of a code object's bytecode, constants and names (nested code objects hashed recursively)."""
    consts = tuple(_const_repr(c) for c in code.co_consts)
    return hashlib.md5(code.co_code + repr((consts, code.co_names)).encode("utf-8")).hexdigest()


def _is_plain_callable(obj: Any) -> bool:
    """Functions, methods, partials and callable instances (ufuncs, dispatchers), but not classes or
    configurable objects such as transforms and estimators (those are fingerprinted by their params)."""
    return callable(obj) and not isinstance(obj, type) and not hasattr(obj, "get_params")


_callables_in_progress = threading.local()


def _get_callable_metadata(fn: Callable) -> dict[str, Any]:
    """Build a deterministic signature for a callable."""
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__qualname__))
    name = f"{getattr(fn, '__module__', None)}.{qualname}"
    in_progress = _callables_in_progress.__dict__.setdefault("ids", set())
    if id(fn) in in_progress:  # recursive closure referring to itself
        return {"callable": name, "recursive": True}
    in_progress.add(id(fn))
    try:
        if isinstance(fn, functools.partial):
            return {
                "partial": _get_callable_metadata(fn.func),
                "args": _get_object_metadata(list(fn.args)),
                "keywords": _get_object_metadata(dict(fn.keywords)),
            }
        if inspect.ismethod(fn):  # bound method: its receiver's state matters too
            return {"method": _get_callable_metadata(fn.__func__), "self": _get_object_params(fn.__self__)}
        meta: dict[str, Any] = {"callable": name, "module_hash": _get_module_hash_from_obj(fn)}
        code = getattr(fn, "__code__", None)
        if code is not None:
            # Lambdas and closures share a qualname, so also hash the code, defaults and captured values.
            # shortcut: module-level globals the function reads are not hashed; edits are covered by module_hash.
            cells = []
            for cell in fn.__closure__ or ():  # type: ignore[attr-defined]
                try:
                    cells.append(cell.cell_contents)
                except ValueError:  # empty cell
                    cells.append(None)
            meta["code"] = _code_fingerprint(code)
            meta["defaults"] = _get_object_metadata(list(fn.__defaults__ or ()))  # type: ignore[attr-defined]
            meta["kwdefaults"] = _get_object_metadata(dict(fn.__kwdefaults__ or {}))  # type: ignore[attr-defined]
            meta["closure"] = _get_object_metadata(cells)
        elif not isinstance(fn, type):
            # Callable instance (ufunc, operator.methodcaller, object with __call__): hash its pickled state,
            # which covers __slots__ and C-level configuration; fall back to public attributes.
            try:
                meta["state"] = joblib.hash(fn)
            except Exception:  # not picklable (e.g. a locally defined class): use its attributes and slots
                state = dict(getattr(fn, "__dict__", {}))
                for klass in type(fn).__mro__:
                    slots = getattr(klass, "__slots__", ())
                    for slot in (slots,) if isinstance(slots, str) else slots:
                        if hasattr(fn, slot):
                            state.setdefault(slot, getattr(fn, slot))
                meta["state"] = {k: _get_object_metadata(v) for k, v in state.items() if not k.startswith("_")}
        return meta
    finally:
        in_progress.discard(id(fn))


def _get_object_with_dict_metadata(obj: Any) -> dict[str, Any]:
    """Build a deterministic signature for a generic object with __dict__."""
    return {
        "__class__": obj.__class__.__name__,
        "config": _get_object_params(obj),
        "module_hash": _get_module_hash_from_obj(obj),
    }


def _get_object_metadata(obj: Any) -> Any:
    """
    Get metadata for an object to use in cache key creation.
    Handles various data types including numpy arrays, pandas DataFrames, and nested structures.

    Args:
        obj: Object to get metadata from

    Returns:
        Metadata representation of the object suitable for hashing
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, np.ndarray):
        return _get_ndarray_metadata(obj)
    if isinstance(obj, DataContainer):
        return _get_datacontainer_metadata(obj)
    if isinstance(obj, pd.DataFrame):
        return _get_dataframe_metadata(obj)
    if isinstance(obj, pd.Series):
        return _get_series_metadata(obj)
    if isinstance(obj, dict):
        return _get_dict_metadata(obj)
    if isinstance(obj, (list, tuple)):
        return _get_sequence_metadata(obj)
    if isinstance(obj, set):
        return _get_set_metadata(obj)
    if callable(obj):
        return _get_callable_metadata(obj)
    if hasattr(obj, "__dict__"):
        return _get_object_with_dict_metadata(obj)
    return {"__class__": obj.__class__.__name__, "repr": str(obj)[:128]}


def _get_object_params(obj: Any) -> Any:
    """
    Get a dictionary of an object's parameters, handling nested objects.

    Args:
        obj: Object to get parameters from

    Returns:
        dict: Dictionary of parameter names and values
    """

    params_to_exclude = ["scent_encoder", "transform_from_name", "fitted_"]  # TODO: make this more robust/flexible?

    if _is_plain_callable(obj):
        return _get_callable_metadata(obj)

    if not hasattr(obj, "__dict__"):
        if isinstance(obj, (list, tuple)):
            return [_get_object_params(item) for item in obj]
        return str(obj)

    params = {}
    # Get class name for type identification
    params["__class__"] = obj.__class__.__name__

    # Public attributes, plus get_params() for objects that keep configuration elsewhere
    # (e.g. AdaptWrapperTransform's estimator kwargs)
    items = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    get_params = getattr(obj, "get_params", None)
    if callable(get_params) and not isinstance(obj, type):
        try:
            items = {**get_params(deep=False), **items}
        except TypeError:
            pass
    for key, value in items.items():
        if key in params_to_exclude or key.startswith("_"):
            continue
        if _is_plain_callable(value):
            params[key] = _get_callable_metadata(value)
        elif hasattr(value, "__dict__"):
            # Recursively get parameters of nested objects
            params[key] = _get_object_params(value)
        elif isinstance(value, (list, tuple)):
            params[key] = [_get_object_params(item) for item in value]
        elif isinstance(value, np.ndarray):
            params[key] = joblib.hash(value)  # str() truncates large arrays
        else:
            params[key] = str(value)

    return params


def hash_dict(d: dict[str, Any]) -> str:
    """
    Create a deterministic hash of a dictionary using MD5.
    Uses MD5 like tuner_utils for consistency, with robust handling of pandas/numpy objects.

    Args:
        d: Dictionary to hash

    Returns:
        str: MD5 hash of the dictionary
    """
    serialized = json.dumps(_get_object_metadata(d), sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def _get_cache_root() -> Path:
    """Get the root directory for caches, allowing an environment override."""
    override = os.environ.get("XDFLOW_CACHE_DIR")
    return Path(override) if override else DEFAULT_CACHE_DIR


def _get_cache_dir(prefix: str) -> Path:
    """Get the cache directory for a given prefix."""
    cache_dir = _get_cache_root() / prefix
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_size(prefix: str | None = None) -> int:
    """
    Get the total size of the cache in bytes.

    Args:
        prefix: Optional prefix to check specific cache directory

    Returns:
        int: Total size in bytes
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return 0

    total_size = 0
    if prefix:
        cache_dir = cache_root / prefix
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
    else:
        for d in cache_root.iterdir():
            if d.is_dir():
                total_size += sum(f.stat().st_size for f in d.glob("*.pkl"))

    return total_size


def _cleanup_old_cache_files(max_age_days: float = DEFAULT_MAX_CACHE_AGE, prefix: str | None = None) -> None:
    """
    Remove cache files older than max_age_days.

    Args:
        max_age_days: Maximum age of cache files in days
        prefix: Optional prefix to clean specific cache directory
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    if prefix:
        cache_dirs = [cache_root / prefix] if (cache_root / prefix).exists() else []
    else:
        cache_dirs = [d for d in cache_root.iterdir() if d.is_dir()]

    for cache_dir in cache_dirs:
        for f in cache_dir.glob("*.pkl"):
            if (current_time - f.stat().st_mtime) > max_age_seconds:
                try:
                    f.unlink()
                except PermissionError:
                    # Ignore files we cannot remove due to filesystem restrictions
                    continue


def _enforce_cache_size_limit(
    max_size: int = DEFAULT_MAX_CACHE_SIZE, prefix: str | None = None, verbose: bool = False
) -> None:
    """
    Remove least-recently-modified cache files until the whole cache is under max_size.

    The limit applies across all prefixes combined, so separate prefixes cannot each grow to
    max_size. ``prefix`` is accepted for call-site compatibility but does not scope the limit.

    Args:
        max_size: Maximum total cache size in bytes across all prefixes
        prefix: Unused; kept for backwards-compatible call signature
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return

    files = [(f, f.stat().st_mtime) for d in cache_root.iterdir() if d.is_dir() for f in d.glob("*.pkl")]
    if len(files) <= 1:
        return

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[1])

    current_size = _get_cache_size()
    if verbose:
        print(f"Current cache size: {current_size / 1024 / 1024 / 1024:.2f} GB")
    for f, _ in files:
        if current_size <= max_size:
            break
        size = f.stat().st_size
        try:
            f.unlink()
            current_size -= size
        except PermissionError:
            # Ignore files we cannot remove; continue trimming others
            continue


def clear_cache(prefix: str | None = None) -> None:
    """
    Clear the cache for a given prefix or all caches if no prefix is specified.

    Args:
        prefix: Optional prefix to clear specific cache directory
    """
    cache_root = _get_cache_root()
    if prefix:
        cache_dir = cache_root / prefix
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    else:
        if cache_root.exists():
            shutil.rmtree(cache_root)
