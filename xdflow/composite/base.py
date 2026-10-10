"""
Pipeline orchestration for the framework.
"""

import builtins
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Self, cast

from xdflow.core.base import Predictor, Transform, _init_param_names
from xdflow.core.data_container import DataContainer


def _configure_transform_for_inference(
    transform: Transform,
    *,
    set_n_jobs_single: bool,
    visited: set[int],
) -> None:
    """Recursively disable training-only flags on transforms for inference usage."""
    if transform is None:
        return

    obj_id = id(transform)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if hasattr(transform, "use_cache"):
        try:
            cast(Any, transform).use_cache = False
        except AttributeError:
            pass

    disable_for_inference = getattr(transform, "_disable_for_inference", None)
    if callable(disable_for_inference):
        disable_for_inference()

    if set_n_jobs_single and hasattr(transform, "n_jobs"):
        try:
            cast(Any, transform).n_jobs = 1
        except AttributeError:
            pass

    children = getattr(transform, "children", None)
    if not children:
        return

    for child in children:
        if isinstance(child, Transform):
            _configure_transform_for_inference(child, set_n_jobs_single=set_n_jobs_single, visited=visited)


@dataclass
class TransformStep:
    """Represents a specifically named step in a processing pipeline.

    Attributes:
        name: The name of the step.
        transform: The transform object to be executed in this step.
    """

    name: str
    transform: Transform


class CompositeTransform(Transform, ABC):
    """
    Abstract base class for transforms that are compositions of other transforms.

    This class provides common functionality for orchestrators like Pipeline and
    PipelineUnion, such as dynamically determining statefulness based on its
    constituent children.

    Cloning semantics
    -----------------
    - CompositeTransform.clone() performs a constructor-filtered recursive clone:
      it reconstructs a new instance using only parameters present in the subclass
      __init__ signature and, for any child Transform(s), calls child.clone().
    - "Recursive" means we clone through the transform hierarchy, but do not
      copy fitted state. Each child must keep fitted state out of __init__ so the
      cloned composite is unfitted.
    - Subclasses should ensure that child collections (e.g., self.steps) are set
      before super().__init__ so is_stateful can be computed from children.
    """

    transform_from_name: dict[str, Transform]

    def __init__(
        self,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
        **kwargs,
    ):
        """
        Initializes the CompositeTransform.

        The `is_stateful` attribute is automatically determined by inspecting
        the children defined in the concrete subclass. This requires that child
        collections (e.g., `self.steps`) are initialized in the subclass's
        `__init__` method *before* calling `super().__init__()`.

        Args:
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            transform_sel: Dictionary of coordinates to select for fitting/transforming.
            transform_drop_sel: Dictionary of coordinates to drop for fitting/transforming.
            **kwargs: Additional keyword arguments.
        """
        # Note: This super().__init__() call must come *after* the subclass
        # has defined the collection that self.children will point to.
        super().__init__(
            sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel, **kwargs
        )

    @property
    def is_stateful(self) -> bool:  # type: ignore[override]
        # Derived on read so a child's set_params (e.g. use_fit=True) is reflected in CV's stateless/stateful split
        return builtins.any(child.is_stateful for child in self.children)

    @property
    @abstractmethod
    def children(self) -> Iterable[Transform]:
        """
        An abstract property that must be implemented by subclasses.

        Returns:
            An iterable collection of the child Transform objects contained
            within this composite.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_predictor(self) -> bool:
        """Returns True if the transform performs prediction."""
        raise NotImplementedError

    @property
    def predictive_transform(self) -> Transform | None:
        """
        Returns the predictive transform if it exists, otherwise None.
        Must be implemented by subclasses if the subclass is a predictor.
        """
        if self.is_predictor:
            raise NotImplementedError
        else:
            return None

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predicts on data.
        Must be implemented by subclasses if the subclass is a predictor but does not inherit from Predictor.
        """
        if isinstance(self, Predictor):
            return super().predict(container, **kwargs)
        if self.is_predictor:
            raise NotImplementedError("Child transform must implement predict.")
        else:
            raise ValueError("CompositeTransform is not a predictor.")

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predicts the probabilities on data.
        Must be implemented by subclasses if the subclass is a predictor but does not inherit from Predictor.
        """
        if isinstance(self, Predictor):
            return super().predict_proba(container, **kwargs)
        if self.is_predictor:
            raise NotImplementedError("Child transform must implement predict_proba.")
        else:
            raise ValueError("CompositeTransform is not a predictor.")

    @abstractmethod
    def _validate_composition(self):
        """
        An abstract method to enforce validation of the internal structure
        of the composite transform on initialization.
        """
        raise NotImplementedError

    def set_params(self, **params: Any) -> "CompositeTransform":
        """
        Set the parameters of this transform and its children.

        This method supports nested parameter setting using the `__` separator,
        similar to scikit-learn's Pipeline.
        """
        # Separate parameters for this transform and its children
        self_params = {}
        nested_params = {}
        for key, value in params.items():
            if "__" in key:
                # Key is for a nested transform
                step_name, param_name = key.split("__", 1)
                if step_name not in nested_params:
                    nested_params[step_name] = {}
                nested_params[step_name][param_name] = value
            else:
                # Key is for this transform itself
                self_params[key] = value

        # Set parameters on this transform
        super().set_params(**self_params)

        # Set parameters on the children
        for step_name, params_to_set in nested_params.items():
            child = self.get_transform_from_name(step_name)
            child.set_params(**params_to_set)

        return self

    def clone(self) -> Self:
        """Return a fresh unfitted instance by recursively cloning constructor-filtered params.

        This default implementation mirrors `Transform.clone` but recursively clones any values
        that are `Transform`s (or collections of them), ensuring children are cloned
        without copying fitted state. Only public constructor parameters are passed
        to the new instance.

        Returns:
            Self: A new, unfitted instance with cloned child transforms.
        """

        def _recursive_clone_constructor_value(value):
            """Recursively clone values that are Transforms or collections thereof.

            - Transform: use `.clone()`
            - TransformStep-like: duck-type objects with `.name` and `.transform`
              where `.transform` is a Transform; reconstruct with the same type
              and a cloned child
            - list/tuple/dict: recurse elementwise, preserving container type
            - everything else: return as-is

            Args:
                value: Any parameter value.

            Returns:
                Any: Cloned value preserving structure where applicable.
            """
            if isinstance(value, Transform):
                return value.clone()

            # Duck-typed TransformStep-like object
            has_transform = hasattr(value, "transform") and hasattr(value, "name")
            if has_transform:
                child = value.transform
                name = value.name
                if isinstance(child, Transform):
                    return type(value)(name, child.clone())

            if isinstance(value, tuple):
                # Special-case (name, Transform) tuples while preserving arbitrary metadata in name
                if len(value) == 2 and isinstance(value[1], Transform):
                    return (value[0], value[1].clone())
                return tuple(_recursive_clone_constructor_value(v) for v in value)

            if isinstance(value, list):
                return [_recursive_clone_constructor_value(v) for v in value]

            if isinstance(value, dict):
                return {k: _recursive_clone_constructor_value(v) for k, v in value.items()}

            return value

        ctor_param_names = _init_param_names(type(self))
        raw_params = self.get_params(deep=False) or {}

        # Ensure all constructor parameters are present in raw params
        for ctor_param_name in ctor_param_names:
            assert ctor_param_name in raw_params, (
                f"Constructor parameter {ctor_param_name} not found as parameter in {self.__class__.__name__}."
            )

        filtered_params = {
            k: _recursive_clone_constructor_value(v) for k, v in raw_params.items() if k in ctor_param_names
        }
        # Same as Predictor._get_clone_kwargs: clones must not share a (fitted) label encoder
        if filtered_params.get("encoder") is not None:
            filtered_params["encoder"] = copy.deepcopy(filtered_params["encoder"])

        return type(self)(**filtered_params)

    def get_transform_from_name(self, name: str) -> Transform:
        """Returns the step with the given name."""
        return self.transform_from_name[name]
