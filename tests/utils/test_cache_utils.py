import numpy as np
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.utils.cache_utils import hash_dict


def test_cache_key_sees_every_value_and_label():
    n = 200
    da = xr.DataArray(
        np.zeros((n, 3)),
        dims=("trial", "feature"),
        coords={"trial": np.arange(n), "feature": np.arange(3), "stimulus": ("trial", np.array(["a"] * n))},
    )
    relabelled = da.copy()
    relabelled.coords["stimulus"] = ("trial", np.array(["a"] + ["b"] + ["a"] * (n - 2)))
    changed_value = da.copy(data=da.values.copy())
    changed_value[1, 1] = 1.0

    keys = {hash_dict({"x": DataContainer(d)}) for d in (da, relabelled, changed_value)}

    assert len(keys) == 3


def test_pipeline_output_cache_ignores_verbosity():
    import os
    from pathlib import Path

    from xdflow.composite import Pipeline
    from xdflow.transforms.basic_transforms import IdentityTransform

    data = DataContainer(xr.DataArray(np.zeros((4, 2)), dims=("trial", "feature")))
    pipeline = Pipeline("p", [("id", IdentityTransform())], use_cache=True)

    pipeline.fit_transform(data, verbose=True)
    pipeline.fit_transform(data, verbose=False)

    assert len(list(Path(os.environ["XDFLOW_CACHE_DIR"]).rglob("*.pkl"))) == 1


def test_pipeline_cache_key_distinguishes_functions_and_get_params_only_config():
    from xdflow.composite import Pipeline
    from xdflow.transforms.adapt_wrapper import AdaptWrapperTransform
    from xdflow.transforms.basic_transforms import FunctionTransform

    def function_key(fn):
        return hash_dict({"pipeline": Pipeline("p", [("f", FunctionTransform(fn))])})

    def add(c):
        return lambda x: x + c

    assert function_key(lambda x: x + 1) != function_key(lambda x: x + 2)
    assert function_key(add(1)) != function_key(add(2))
    assert function_key(add(1)) == function_key(add(1))

    class Stub:
        def __init__(self, lambda_=1.0):
            self.lambda_ = lambda_

    adapt = AdaptWrapperTransform(
        adapt_estimator_cls=Stub,
        sample_dim="trial",
        target_coord="y",
        group_coord="session",
        target_group="b",
        lambda_=0.5,
    )
    pipeline = Pipeline("p", [("adapt", adapt)])
    before = hash_dict({"pipeline": pipeline})
    pipeline.set_params(adapt__lambda_=0.9)
    assert hash_dict({"pipeline": pipeline}) != before


def test_callable_fingerprints_cover_callable_instances_methods_and_recursion():
    import os
    import subprocess
    import sys

    from xdflow.composite import Pipeline
    from xdflow.transforms.basic_transforms import FunctionTransform

    def key(fn):
        return hash_dict({"pipeline": Pipeline("p", [("f", FunctionTransform(fn))])})

    assert key(np.sin) != key(np.cos)
    assert key(np.mean) != key(np.sum)

    def shifted(x, *, offset=1):
        return x + offset

    before = key(shifted)
    shifted.__kwdefaults__["offset"] = 2
    assert key(shifted) != before

    class Adder:
        def __init__(self, n):
            self.n = n

        def apply(self, x):
            return x + self.n

    assert key(Adder(1).apply) != key(Adder(2).apply)

    def make_recursive():
        def f(x):
            return f(x[0]) if isinstance(x, list) else x

        return f

    assert key(make_recursive()) == key(make_recursive())

    source = (
        "from xdflow.utils.cache_utils import hash_dict\n"
        "def f(x): return x in {'aa', 'bb', 'cc', 'dd', 'ee'}\n"
        "print(hash_dict({'fn': f}))\n"
    )
    keys = {
        subprocess.check_output(
            [sys.executable, "-c", source], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        ).strip()
        for seed in ("1", "2")
    }
    assert len(keys) == 1


def test_callable_instances_with_slots_or_c_state_get_distinct_keys():
    import operator

    from xdflow.composite import Pipeline
    from xdflow.transforms.basic_transforms import FunctionTransform

    def key(fn):
        return hash_dict({"pipeline": Pipeline("p", [("f", FunctionTransform(fn))])})

    class Scale:
        __slots__ = ("factor",)

        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x):
            return x * self.factor

    assert key(operator.methodcaller("mean")) != key(operator.methodcaller("sum"))
    assert key(Scale(1)) != key(Scale(2))
    assert key(Scale(1)) == key(Scale(1))


def test_cache_key_sees_attrs_but_not_data_history():
    da = xr.DataArray(np.zeros((4, 2)), dims=("trial", "feature"), attrs={"sampling_frequency_hz": 500})
    other_rate = da.copy()
    other_rate.attrs["sampling_frequency_hz"] = 1000
    with_history = da.copy()
    with_history.attrs["data_history"] = [{"class": "IdentityTransform", "params": {}}]

    key = hash_dict({"x": DataContainer(da)})
    assert hash_dict({"x": DataContainer(other_rate)}) != key
    assert hash_dict({"x": DataContainer(with_history)}) == key


def test_unpicklable_fold_pipeline_runs_without_the_fit_cache(monkeypatch):
    """A lambda after the first stateful step cannot be pickled; CV must still run, just uncached."""
    monkeypatch.setenv("XDFLOW_CACHE_USER_CODE", "1")  # user code is not cached otherwise
    import pytest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from xdflow import KFoldValidator, Pipeline
    from xdflow.transforms.basic_transforms import FunctionTransform
    from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer

    rng = np.random.default_rng(0)
    data = DataContainer(
        xr.DataArray(
            rng.normal(size=(30, 4)),
            dims=("trial", "feature"),
            coords={"trial": np.arange(30), "label": ("trial", np.arange(30) % 2)},
        )
    )
    pipeline = Pipeline(
        "p",
        [
            ("scale", SKLearnTransformer(StandardScaler, sample_dim="trial")),
            ("double", FunctionTransform(lambda x: x * 2)),
            ("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label")),
        ],
    )
    validator = KFoldValidator(n_splits=3, verbose=False, use_stateful_fit_cache=True)
    validator.set_pipeline(pipeline)
    with pytest.warns(UserWarning, match="cannot be pickled"):
        validator.cross_validate(data)
    assert len(validator.cv_scores_) == 3


def test_unpicklable_pipeline_output_runs_without_the_output_cache(monkeypatch):
    """A lambda step's output records the lambda in its history; the cached pipeline must still run."""
    monkeypatch.setenv("XDFLOW_CACHE_USER_CODE", "1")  # user code is not cached otherwise
    import pytest

    from xdflow.composite import Pipeline
    from xdflow.transforms.basic_transforms import FunctionTransform

    data = DataContainer(xr.DataArray(np.ones((4, 2)), dims=("trial", "feature")))
    pipeline = Pipeline("p", [("double", FunctionTransform(lambda x: x * 2))], use_cache=True)
    with pytest.warns(UserWarning, match="cannot be pickled"):
        out = pipeline.fit_transform(data)
    np.testing.assert_array_equal(out.data.values, 2.0)


_SCALE = 1.0


def _helper(x):
    return x + 1


def _reads_globals(x):
    return _helper(x) * _SCALE


def test_cache_key_follows_globals_and_helpers_a_function_reads(monkeypatch, tmp_path):
    """Editing a helper, a constant or another module a step calls must change the key (notebooks, imports)."""
    import importlib
    import sys

    from xdflow.utils.cache_utils import _get_callable_metadata

    def key(fn):
        return hash_dict(_get_callable_metadata(fn))

    base = key(_reads_globals)
    monkeypatch.setattr(sys.modules[__name__], "_SCALE", 2.0)
    assert key(_reads_globals) != base
    scaled = key(_reads_globals)
    monkeypatch.setattr(sys.modules[__name__], "_helper", lambda x: x - 1)
    assert key(_reads_globals) != scaled

    (tmp_path / "helpers_mod.py").write_text("def center(x):\n    return x.mean()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    helpers_mod = importlib.import_module("helpers_mod")
    namespace = {"helpers_mod": helpers_mod}
    exec("def fit(x):\n    return helpers_mod.center(x)\n", namespace)
    before = key(namespace["fit"])
    (tmp_path / "helpers_mod.py").write_text("def center(x):\n    return x.max()\n")
    assert key(namespace["fit"]) != before


def test_cache_key_sees_the_contents_of_container_parameters_and_globals():
    """Dict parameters (with large arrays), dict key types and order, frozensets and enums all reach the key."""
    from enum import Enum

    from xdflow.transforms.function_step import step
    from xdflow.utils.cache_utils import _get_object_metadata, _get_object_params

    @step
    def shift(data, config):
        return data + config["offsets"][500]

    weights = np.zeros(2000)
    transform = shift({"offsets": weights})
    before = hash_dict(_get_object_params(transform))
    weights[500] = 9.0
    assert hash_dict(_get_object_params(transform)) != before

    class Mode(Enum):
        ADD = 1
        MULTIPLY = 2

    def key(value):
        return hash_dict({"value": _get_object_metadata(value)})

    assert key({1: 2}) != key({"1": 2})
    assert key({"a": 1, "b": 2}) != key({"b": 2, "a": 1})
    assert key(frozenset(range(100))) != key(frozenset(range(101)))
    assert key(Mode.ADD) != key(Mode.MULTIPLY)
    assert hash_dict(_get_object_params(shift(Mode.ADD))) != hash_dict(_get_object_params(shift(Mode.MULTIPLY)))


_WRAPPED_SCALE = 1


def test_user_code_is_recognised_by_where_its_code_lives(monkeypatch):
    import functools
    import sys
    from pathlib import Path
    from types import ModuleType

    import xdflow
    from xdflow.utils.cache_utils import _get_callable_metadata, _is_user_code

    sibling = ModuleType("helpers")
    sibling.__file__ = str(Path(xdflow.__file__).parent.parent / "xdflow_helpers.py")
    assert _is_user_code(sibling)

    @functools.wraps(np.mean)  # advertises numpy's module, but the code is ours
    def wrapped(x):
        return x * _WRAPPED_SCALE

    before = hash_dict(_get_callable_metadata(wrapped))
    monkeypatch.setattr(sys.modules[__name__], "_WRAPPED_SCALE", 3)
    assert hash_dict(_get_callable_metadata(wrapped)) != before


def test_a_global_that_cannot_be_fingerprinted_never_reuses_a_key():
    from xdflow.utils.cache_utils import _get_callable_metadata

    namespace = {"config": {"offset": 1}}
    namespace["config"]["self"] = namespace["config"]  # cyclic: cannot be hashed by content
    exec("def cyclic(x):\n    return x + config['offset']\n", namespace)
    assert hash_dict(_get_callable_metadata(namespace["cyclic"])) != hash_dict(
        _get_callable_metadata(namespace["cyclic"])
    )


def test_results_of_user_functions_are_not_cached_on_disk_unless_allowed(monkeypatch):
    """xdflow cannot see every code a user's function depends on, so its results are recomputed by default."""
    import os
    from pathlib import Path

    from sklearn.linear_model import LogisticRegression

    from xdflow import KFoldValidator
    from xdflow.composite import Pipeline
    from xdflow.transforms.function_step import step
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    @step
    def double(data):
        return data * 2

    def means(data):
        return data.mean("trial")

    @step(fit=means)
    def center(data, state):
        return data - state

    rng = np.random.default_rng(0)
    data = DataContainer(
        xr.DataArray(
            rng.normal(size=(30, 4)),
            dims=("trial", "feature"),
            coords={"trial": np.arange(30), "label": ("trial", np.arange(30) % 2)},
        )
    )
    cache_dir = Path(os.environ["XDFLOW_CACHE_DIR"])

    def run():
        Pipeline("p", [("double", double())], use_cache=True).fit_transform(data)
        validator = KFoldValidator(n_splits=3, verbose=False, use_stateful_fit_cache=True)
        predictor = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label")
        validator.set_pipeline(Pipeline("q", [("center", center()), ("lr", predictor)]))
        validator.cross_validate(data)
        return len(list(cache_dir.rglob("*.pkl")))

    monkeypatch.delenv("XDFLOW_CACHE_USER_CODE", raising=False)
    assert run() == 0
    monkeypatch.setenv("XDFLOW_CACHE_USER_CODE", "1")
    assert run() > 0
