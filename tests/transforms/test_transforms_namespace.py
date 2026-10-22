"""Every built-in transform is importable from xdflow.transforms, and the list covers every module."""

import importlib
import inspect
from pathlib import Path

import pytest

import xdflow.transforms as transforms
from xdflow.core.base import Transform

_OPTIONAL = {"lightgbm", "mne", "nitime", "adapt", "torch"}


@pytest.mark.parametrize("name", sorted(transforms._MODULE_OF))
def test_listed_names_resolve_to_the_module_object(name):
    module = importlib.import_module(f"xdflow.transforms.{transforms._MODULE_OF[name]}")
    try:
        value = getattr(transforms, name)
    except ModuleNotFoundError as exc:  # an optional extra that is not installed
        if exc.name and exc.name.split(".")[0] in _OPTIONAL:
            pytest.skip(f"{exc.name} not installed")
        raise
    assert value is getattr(module, name)


def test_every_public_concrete_transform_is_listed():
    listed = set(transforms._MODULE_OF)
    missing = []
    for path in sorted(Path(transforms.__file__).parent.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            module = importlib.import_module(f"xdflow.transforms.{path.stem}")
        except ModuleNotFoundError:
            continue
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Transform)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
                and not inspect.isabstract(obj)
                and not name.endswith(("Strategy", "Mixin"))
                and name not in {"SKLearnTransform", "AdaptiveTransform", "SingleTargetAligner", "JointGroupAligner"}
            ):
                if name not in listed:
                    missing.append(f"{path.stem}.{name}")
    assert not missing, f"Add to xdflow.transforms._MODULES: {missing}"


def test_unknown_names_raise_attribute_error():
    with pytest.raises(AttributeError):
        transforms.NotATransform  # noqa: B018
