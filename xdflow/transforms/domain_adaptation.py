"""
Domain Adaptation Transforms using a Strategy and Factory Pattern.

This module implements various domain adaptation techniques for neural data. The design
is based on a combination of the Strategy and Factory design patterns to create a
flexible and extensible system.

Core Concepts:
-------------

1.  AdaptiveStrategy: The base class for any adaptation procedure
    (e.g., SingleTargetStrategy). It defines the high-level plan for how to adapt
    data across different groups (domains), but delegates the actual mathematical
    computations back to the AdaptiveTransform.

2.  AdaptiveTransform: The base class for any alignment algorithm (e.g., ProcrustesAligner).
    It uses an AdaptiveStrategy to adapt the data, and the specific strategy is specified
    in the __init__ method.

3.  Abstract classes that inherit AdaptiveTransform (e.g., SingleTargetAligner, JointGroupAligner):
    These abstract classes define the "contract" of methods an AdaptiveTransform must implement to be
    compatible with a specific strategy. This ensures a clear and safe connection between a strategy
    and the aligners it can work with. E.g. SingleTargetStrategy is tied to SingleTargetAligner,
    since you need the methods in SingleTargetAligner to be implemented to use SingleTargetStrategy.

4.  Concrete classes that inherit Abstract classes (e.g., ProcrustesAligner, CoralAligner):
    These concrete classes implement the methods specified in the Abstract classes. Thse methods perform
    the actual mathematical computations for domain adaptation.
    E.g. ProcrustesAligner implements the methods specified in SingleTargetAligner.

The hierarchy is as follows:
Concrete classes inherit from Abstract classes, which inherit from AdaptiveTransform.

Concrete classes may perform different domain adaptation strategies. For each strategy they implement,
they inherit from the Abstract class that defines the "contract" of methods for that strategy. Each
abstract class then inherits from AdaptiveTransform.

E.g. ProcrustesAligner can perform the SingleTargetStrategy, so it inherits from SingleTargetAligner.
SingleTargetAligner inherits from AdaptiveTransform. A class that can perform both SingleTargetStrategy
and JointGroupStrategy would inherit from both SingleTargetAligner and JointGroupAligner.

"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable
from inspect import Parameter, signature
from typing import Any

import numpy as np
import scipy.linalg as linalg
import xarray as xr
from joblib import Parallel, delayed
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, polynomial_kernel, rbf_kernel

from xdflow.core.base import Transform, _init_param_names
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.utils.positions import latest_tag, restore_positions, tag_positions
from xdflow.utils.sampling import get_container_by_conditions, stratified_sample

########################################################################################
# ADAPTIVE STRATEGIES
########################################################################################


# Base class for all domain adaptation strategies


class AdaptiveStrategy(ABC):
    """
    Abstract base class for domain adaptation strategies.

    A strategy encapsulates a high-level procedure for domain adaptation, such as
    a single-target or joint domain approach. It orchestrates the process by
    managing data grouping and calling back to the aligner for specific
    mathematical computations.
    """

    def __init__(self, group_coord: str, n_jobs: int = 1, adapt_sel: dict[str, Any] | None = None):
        """
        Initialize the AdaptiveStrategy.

        Args:
            group_coord: The coordinate to group by.
            n_jobs: The number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
        """
        self.group_coord = group_coord
        self.n_jobs = n_jobs
        self.adapt_sel = adapt_sel
        self.group_dim: str | None = None  # set by the subclass during adapt

    def adapt(self, aligner: "AdaptiveTransform", container: DataContainer, **kwargs):
        """
        Adapt the data using the adaptation strategy.
        """
        if self.adapt_sel:
            container = get_container_by_conditions(container, self.adapt_sel)
            print(f"Adapted container shape: {container.data.shape}")
        self._adapt(aligner, container, **kwargs)

    @abstractmethod
    def _adapt(self, aligner: "AdaptiveTransform", container: DataContainer, **kwargs):
        """
        Abstract method for the fitting logic of the adaptation strategy.

        Args:
            aligner: The AdaptiveTransform instance using this strategy.
            container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def transform(self, aligner: "AdaptiveTransform", container: DataContainer, **kwargs) -> DataContainer:
        """
        Abstract method for the transformation logic of the adaptation strategy.

        Args:
            aligner: The AdaptiveTransform instance using this strategy.
            container: The DataContainer to be transformed.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_group_dim(self, container: DataContainer) -> str:
        """Resolves the dimension that the group_coord indexes."""
        if self.group_coord not in container.data.coords:
            raise ValueError(f"Group coordinate '{self.group_coord}' not found in data coordinates")

        coord_dims = container.data.coords[self.group_coord].dims
        if len(coord_dims) != 1:
            raise ValueError(
                f"Group coordinate '{self.group_coord}' must index exactly one dimension, "
                f"but it indexes {len(coord_dims)}: {coord_dims}"
            )
        return coord_dims[0]

    def _discover_groups(self, container: DataContainer) -> list[Hashable]:
        """Discovers unique group values from the data."""
        group_values = container.data.coords[self.group_coord].values
        return sorted(np.unique(group_values).tolist())

    def _select_group(self, container: DataContainer, group_val: Hashable) -> DataContainer:
        """Selects data for a specific group using boolean indexing."""
        # Positional boolean indexing; `where(drop=True)` would cast int data to float
        group_mask = container.data.coords[self.group_coord].values == group_val
        return DataContainer(container.data.isel({self._get_group_dim(container): group_mask}))

    def _tag_positions(self, container: DataContainer) -> DataContainer:
        """Record each sample's input position so outputs can be reordered even without unique labels."""
        dim = self._get_group_dim(container)
        return DataContainer(tag_positions(container.data, dim)[0])

    def _restore_input_order(self, reassembled: xr.DataArray, container: DataContainer) -> xr.DataArray:
        """Groups are concatenated one after another; put samples back in input order so positional
        consumers (label arrays, ensembles) stay aligned with the input."""
        dim = self._get_group_dim(container)
        tag = latest_tag(container.data)
        restored = restore_positions(reassembled, dim, tag) if tag is not None else None
        if restored is not None:
            return restored
        return reassembled.sel({dim: container.data[dim].values})


# Single source strategy
class SingleTargetStrategy(AdaptiveStrategy):
    """
    An adaptation strategy for a single target group and multiple source groups.
    During adaptation, it fits the target domain and then adapts each source domain to the target domain independently.
    During transformation, it leaves the target domain unchanged and applies the adapted models to the source domains.
    Domains are determined by the group_coord.

    SingleTargetStrategy can be implemented by any aligner that inherits from SingleTargetAligner.
    """

    def __init__(
        self,
        group_coord: str,
        target_group: str | int | float,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
    ):
        """
        Initialize the SingleTargetStrategy.

        Args:
            group_coord: The coordinate to group by. Determines different domains.
            target_group: The target group to adapt to.
            n_jobs: The number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
        """
        super().__init__(group_coord=group_coord, n_jobs=n_jobs, adapt_sel=adapt_sel)
        self.target_group = target_group
        self.target_params = {}
        self.adapted_params = {}
        self.group_dim = None
        self.seen_source_groups_ = []
        self.seen_groups_ = []

    def _adapt(self, aligner: "SingleTargetAligner", container: DataContainer, **kwargs) -> None:
        """
        Adapt the data using a single-target strategy.
        Fit the target domain and then adapt each source domain to the target domain independently.
        Domains are determined by the group_coord.

        Args:
            aligner: The SingleTargetAligner instance using this strategy.
            container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.
        """

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        all_groups = self._discover_groups(container)
        self.seen_source_groups_ = [g for g in all_groups if g != self.target_group]
        self.seen_groups_ = all_groups

        if self.target_group not in all_groups:
            raise ValueError(f"Target group '{self.target_group}' not found in data")

        if not self.seen_source_groups_:
            raise ValueError("No source groups found in the provided container for adaptation.")

        # Reset params
        self.target_params = {}  # dict[param_name: param_value]
        self.adapted_params = {}  # dict[group_val: dict[param_name: param_value]]

        # Handle sampling logic
        min_count = None
        if isinstance(aligner, SamplingMixin) and hasattr(aligner, "sampling_method"):
            aligner._check_class_sets(container, self.group_coord, aligner.target_coord)
            if aligner.sampling_method == "min_count":
                min_count = aligner._calculate_min_count(container, self.group_coord, aligner.target_coord)
            kwargs["min_count"] = min_count

        # Fit the source domain
        target_container = self._select_group(container, self.target_group)
        self.target_params = aligner._fit_target(target_container, **kwargs)

        if self.seen_source_groups_:

            def adapt_group(group_val):
                group_container = self._select_group(container, group_val)
                fitted_params_dict = aligner._adapt_source(group_container, **kwargs)
                return group_val, fitted_params_dict

            # Adapt each source domain
            if self.n_jobs != 1:
                # Parallel fitting
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(adapt_group)(group_val) for group_val in self.seen_source_groups_
                )
                self.adapted_params = dict(results)
            else:
                # Sequential fitting
                for group_val in self.seen_source_groups_:
                    group_val, fitted_params_dict = adapt_group(group_val)
                    self.adapted_params[group_val] = fitted_params_dict
        else:
            warnings.warn("No source groups found in the provided container for adaptation.", stacklevel=2)

    def transform(self, aligner: "SingleTargetAligner", container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data by applying the appropriate target or adapted model to each group.

        Args:
            aligner: The SingleTargetAligner instance using this strategy.
            container: The DataContainer to be transformed.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        container = self._tag_positions(container)
        current_groups = self._discover_groups(container)

        def transform_group(group_val):
            group_container = self._select_group(container, group_val)
            if group_val == self.target_group:
                return group_container
            elif group_val in self.adapted_params:
                return aligner._adapted_transform(
                    group_container, self.adapted_params[group_val], self.target_params, **kwargs
                )
            else:  # Unseen source group
                raise TransformError(
                    f"Source group '{group_val}' was not seen during 'adapt'. "
                    f"Seen source groups: {self.seen_source_groups_}"
                )

        group_outputs = []
        if self.n_jobs != 1:
            transformed_containers = Parallel(n_jobs=self.n_jobs)(
                delayed(transform_group)(group_val) for group_val in current_groups
            )
            group_outputs.extend([output.data for output in transformed_containers])
        else:
            for group_val in current_groups:
                transformed_container = transform_group(group_val)
                group_outputs.append(transformed_container.data)

        # Reassemble outputs #TODO: do we need to reassemble in the same order?
        reassembled = xr.concat(group_outputs, dim=self.group_dim)
        return DataContainer(self._restore_input_order(reassembled, container))


# Joint group strategy
class JointGroupStrategy(AdaptiveStrategy):
    """
    An adaptation strategy for jointly fitting all domains. Domains are determined by the group_coord.
    During adaptation, it fits all domains jointly.
    During transformation, it applies the appropriate adapted model to each domain.

    JointGroupStrategy can be implemented by any aligner that inherits from JointGroupAligner.
    """

    def __init__(self, group_coord: str, n_jobs: int = 1, adapt_sel: dict[str, Any] | None = None):
        """
        Initialize the JointGroupStrategy.

        Args:
            group_coord: The coordinate to group by. Determines different domains.
            n_jobs: The number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
        """
        super().__init__(group_coord=group_coord, n_jobs=n_jobs, adapt_sel=adapt_sel)
        self.adapted_params = {}
        self.group_dim = None
        self.seen_groups_ = []

    def _adapt(self, aligner: "JointGroupAligner", container: DataContainer, **kwargs) -> None:
        """
        Adapt the data using a joint group strategy.
        Fit all domains jointly.
        Domains are determined by the group_coord.

        Args:
            aligner: The JointGroupAligner instance using this strategy.
            container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.
        """

        warnings.warn("JointGroupStrategy used. Groups aligning to new shared subspace.", stacklevel=2)

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        all_groups = self._discover_groups(container)
        self.seen_groups_ = all_groups

        if len(all_groups) < 2:
            raise ValueError("JointGroupStrategy requires at least two groups for adaptation.")

        # Reset params
        self.adapted_params = {}  # dict[group_val: dict[param_name: param_value]]

        # Handle sampling logic
        min_count = None
        if isinstance(aligner, SamplingMixin) and hasattr(aligner, "sampling_method"):
            aligner._check_class_sets(container, self.group_coord, aligner.target_coord)
            if aligner.sampling_method == "min_count":
                min_count = aligner._calculate_min_count(container, self.group_coord, aligner.target_coord)
            kwargs["min_count"] = min_count

        group_containers = {group_val: self._select_group(container, group_val) for group_val in all_groups}

        self.adapted_params = aligner._fit_joint(group_containers, **kwargs)

    def transform(self, aligner: "JointGroupAligner", container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data by applying the appropriate adapted model to each group.

        Args:
            aligner: The JointGroupAligner instance using this strategy.
            container: The DataContainer to be transformed.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        container = self._tag_positions(container)
        current_groups = self._discover_groups(container)

        def transform_group(group_val):
            group_container = self._select_group(container, group_val)
            if group_val in self.adapted_params:
                return aligner._adapted_transform(group_container, self.adapted_params[group_val], **kwargs)
            else:
                raise TransformError(
                    f"Group '{group_val}' was not seen during 'adapt'. Seen groups: {self.seen_groups_}"
                )

        group_outputs = []
        if self.n_jobs != 1:
            transformed_containers = Parallel(n_jobs=self.n_jobs)(
                delayed(transform_group)(group_val) for group_val in current_groups
            )
            group_outputs.extend([output.data for output in transformed_containers])
        else:
            for group_val in current_groups:
                transformed_container = transform_group(group_val)
                group_outputs.append(transformed_container.data)

        # Reassemble outputs
        reassembled = xr.concat(group_outputs, dim=self.group_dim)
        return DataContainer(self._restore_input_order(reassembled, container))


########################################################################################
# ABSTRACT ADAPTIVE TRANSFORMS
########################################################################################


# Base class for all domain adaptation transforms
class AdaptiveTransform(Transform, ABC):
    """
    Abstract base class for domain adaptation aligners.

    This class serves as the main context for the Strategy pattern. It is initialized
    with a strategy (either as a string or a pre-made instance) and delegates the
    fitting and transforming logic to it.
    """

    # This map should be overridden by concrete aligner classes
    _STRATEGY_MAP: dict[str, type[AdaptiveStrategy]] = {}  # dict[strategy_name (str): AdaptiveStrategy]

    # Class attributes required by Transform base classq1
    is_stateful: bool = True
    input_dims: tuple[str, ...] = ()  # Accept any input dimensions
    output_dims: tuple[str, ...] = ()  # Output dimensions same as input

    def __init__(
        self,
        strategy_name: str,
        sample_dim: str,
        **kwargs,
    ):
        """
        Initialize domain adaptation transform.

        Args:
            strategy_name: The adaptation strategy to use.
            sample_dim: The dimension to use for the sample.
            **kwargs: Additional arguments passed to Transform base class or strategy's constructor.
        """
        if strategy_name not in type(self)._STRATEGY_MAP:
            raise ValueError(
                f"Unknown or unsupported strategy: '{strategy_name}' for {type(self).__name__}. "
                f"Available strategies: {list(type(self)._STRATEGY_MAP.keys())}"
            )
        self.strategy_name = strategy_name
        # The strategy's arguments are this transform's hyperparameters: plain attributes that
        # `clone()` and `set_params` treat like any other, and that `_fit` builds the strategy from.
        for name, param in self._strategy_parameters().items():
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            elif param.default is Parameter.empty:
                raise ValueError(f"Required argument '{name}' for strategy {self._strategy_cls().__name__} is missing.")
            else:
                setattr(self, name, param.default)
        unexpected = set(kwargs) - _init_param_names(Transform)
        if unexpected:
            raise TypeError(f"{type(self).__name__} got unexpected keyword argument(s): {sorted(unexpected)}")
        super().__init__(**kwargs)

        self.sample_dim = sample_dim
        self._check_strategy_args()
        self.strategy: AdaptiveStrategy | None = None  # built by `_fit` from the current attributes
        self._is_fitted = False

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Aligners pickled with xdflow < 0.3 kept strategy arguments in `_strategy_kwargs`; restore them as the
        # attributes clone() and fit now read.
        legacy = state.pop("_strategy_kwargs", None)
        state.pop("_strategy_class", None)
        self.__dict__.update(state)
        if legacy is not None:
            for name, param in self._strategy_parameters().items():
                if name not in self.__dict__:
                    self.__dict__[name] = legacy.get(name, param.default)

    def _strategy_cls(self) -> type[AdaptiveStrategy]:
        return type(self)._STRATEGY_MAP[self.strategy_name]

    def _strategy_parameters(self) -> dict[str, Parameter]:
        return {
            name: p
            for name, p in signature(self._strategy_cls().__init__).parameters.items()
            if name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

    def _check_strategy_args(self) -> None:
        for name, param in self._strategy_parameters().items():
            if param.default is Parameter.empty and getattr(self, name) is None:
                raise ValueError(f"Required argument '{name}' for strategy {self._strategy_cls().__name__} is None.")

    def _fit(self, data_container: DataContainer, **kwargs) -> "AdaptiveTransform":
        """
        Internal fit method that delegates adaptation to the strategy.

        Args:
            data_container: DataContainer for adaptation.
            **kwargs: Additional arguments.

        Returns:
            Self (fitted transform).
        """
        self._check_strategy_args()
        self.strategy = self._strategy_cls()(**{name: getattr(self, name) for name in self._strategy_parameters()})
        self.strategy.adapt(self, data_container, **kwargs)
        self._is_fitted = True
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Internal transform method that delegates transformation to the strategy.

        Args:
            data_container: DataContainer to transform.
            **kwargs: Additional arguments.

        Returns:
            Transformed DataContainer.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call 'fit' before 'transform'.")
        return self.strategy.transform(self, data_container, **kwargs)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Domain adaptation transforms preserve input dimensions.

        Args:
            input_dims: Input dimension names

        Returns:
            Same dimensions as input
        """
        return input_dims

    def _prepare_data(self, data: xr.DataArray) -> xr.DataArray:
        """
        Prepare the data to be in the right format for the transform.
        """
        if data.ndim != 2:
            raise ValueError(
                f"AdaptiveTransforms currently requires 2D data. Multi-dimensional data not supported yet"
                f"Received data with {data.ndim} dimensions: {data.dims}"
            )

        if data.dims[0] != self.sample_dim:
            data = data.transpose(self.sample_dim, ...)
        return data


# Abstract base class for aligners that work with a SingleTargetStrategy
class SingleTargetAligner(AdaptiveTransform, ABC):
    """
    Defines the contract for aligners compatible with SingleTargetStrategy.

    This ABC acts as an interface. Any aligner that inherits from it promises
    to implement the methods required by the SingleTargetStrategy, ensuring a
    safe and explicit connection between the two components.
    """

    @abstractmethod
    def _fit_target(self, data_container: DataContainer, /, **kwargs) -> dict[str, Any]:
        """Abstract method for fitting the target params.

        Args:
            data_container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.

        Returns:
            Dict [param_name: param_value] of the target params.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _adapt_source(self, data_container: DataContainer, /, **kwargs) -> dict[str, Any]:
        """Abstract method for fitting the source params for a single group.

        Args:
            data_container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.

        Returns:
            Dict [param_name: param_value] of the source params.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _adapted_transform(
        self, data_container: DataContainer, adapted_params: dict, target_params: dict, /, **kwargs
    ) -> DataContainer:
        """Abstract method for transforming the data for a single group.

        Args:
            data_container: The DataContainer to be transformed.
            adapted_params: The adapted params for the group.
            target_params: The target params for the target domain.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        raise NotImplementedError("Subclasses must implement this method.")


# Abstract base class for aligners that work with a JointGroupStrategy
class JointGroupAligner(AdaptiveTransform, ABC):
    """
    Defines the contract for aligners compatible with JointGroupStrategy.
    This ABC acts as an interface. Any aligner that inherits from it promises
    to implement the methods required by the JointGroupStrategy.
    """

    def __init__(self, strategy_name: str, output_dim_name: str = "component", **kwargs):
        """
        Initialize the JointGroupAligner.

        Args:
            strategy_name: The strategy to use for adaptation.
            output_dim_name: The name of the output dimension/space that all groups are aligned to.
            **kwargs: Additional arguments.
        """
        super().__init__(strategy_name, **kwargs)

        self.output_dim_name = output_dim_name

    @abstractmethod
    def _fit_joint(self, group_containers: dict, /, **kwargs) -> dict[Hashable, dict[str, Any]]:
        """Abstract method for jointly fitting all groups.

        Args:
            group_containers: The DataContainers for all groups.
            **kwargs: Additional arguments.

        Returns:
            Dict [group_val: dict[param_name: param_value]] of the fitted params.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, /, **kwargs) -> DataContainer:
        """Abstract method for transforming the data for a single group.

        Args:
            data_container: The DataContainer to be transformed.
            adapted_params: The adapted params for the group.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions.
        JointGroupAligner aligns data to a common space, so it does not preserve the original dimension names.
        """
        if len(input_dims) != 2:
            raise ValueError(
                f"AdaptiveTransforms currently requires 2D data, but got data with {len(input_dims)} dimensions: {input_dims}"
            )

        return (self.sample_dim, self.output_dim_name)

    def _create_output_container(self, original_da: xr.DataArray, transformed_np: np.ndarray) -> DataContainer:
        """Creates the output DataContainer."""

        # Create new DataContainer with transformed data
        # Note: The output dimensions are now (sample, n_components)
        # The original dimension names are lost. Creating new ones.

        output_dims = (self.sample_dim, self.output_dim_name)
        output_coords = {self.sample_dim: original_da.coords[self.sample_dim].values}

        #  Preserve all coordinates that are associated with the sample dimension
        for coord_name, coord_data in original_da.coords.items():
            if coord_name != self.sample_dim and self.sample_dim in coord_data.dims:
                output_coords[coord_name] = coord_data
        output_coords[self.output_dim_name] = np.arange(transformed_np.shape[1])
        transformed_da = xr.DataArray(transformed_np, dims=output_dims, coords=output_coords, attrs=original_da.attrs)

        return DataContainer(transformed_da)


########################################################################################
# MIXINS FOR ADAPTIVE TRANSFORMS
########################################################################################


class SamplingMixin:
    """
    Mixin class for aligners that support sampling of data before adaptation.
    This provides a standardized way to handle 'mean' and 'min_count' sampling.
    """

    @staticmethod
    def _check_class_sets(container: DataContainer, group_coord: str, target_coord: str) -> None:
        """Sampled class rows are matched across groups by position, so every group needs the same classes."""
        target_values = container.data.coords[target_coord].values
        group_values = container.data.coords[group_coord].values
        class_sets = {
            group: sorted(np.unique(target_values[group_values == group]).tolist()) for group in np.unique(group_values)
        }
        if len({tuple(classes) for classes in class_sets.values()}) > 1:
            raise ValueError(
                f"All groups must contain the same '{target_coord}' classes to be aligned; got {class_sets}."
            )

    @staticmethod
    def _calculate_min_count(container: DataContainer, group_coord: str, target_coord: str) -> int:
        """Calculates the minimum number of samples per class across all data."""
        if target_coord not in container.data.coords:
            raise ValueError(f"Target coordinate '{target_coord}' not found in data.")

        target_values = container.data.coords[target_coord].values
        group_values = container.data.coords[group_coord].values

        min_count = np.inf
        for group_value in np.unique(group_values):
            target_values_group = target_values[group_values == group_value]
            _, counts = np.unique(target_values_group, return_counts=True)
            min_count_group = np.min(counts)
            min_count = min(min_count, min_count_group)

        return int(min_count)

    def _sample_data(
        self,
        data: xr.DataArray,
        target_coord: str,
        sample_dim: str,
        sampling_method: str,
        min_count: int | None = None,
        random_state: int = 0,
    ) -> xr.DataArray:
        """Samples the data according to the specified method."""
        if sampling_method == "mean":
            return data.sortby(target_coord).groupby(target_coord).mean(sample_dim)
        elif sampling_method == "min_count":
            if min_count is None:
                raise ValueError("'min_count' must be provided for 'min_count' sampling method.")
            return stratified_sample(data, target_coord, max_samples_per_class=min_count, random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")


########################################################################################
# CONCRETE ADAPTIVE TRANSFORMS
########################################################################################


# A dummy strategy for a dummy aligner.
class _DummyStrategy(AdaptiveStrategy):
    def __init__(self):
        super().__init__(group_coord="dummy")

    def _adapt(self, aligner, container, **kwargs):
        pass

    def transform(self, aligner, container, **kwargs):
        return container


class DummyAligner(AdaptiveTransform):
    """
    Dummy aligner that does nothing.
    Used for testing.
    """

    _STRATEGY_MAP = {"dummy": _DummyStrategy}

    def __init__(self, sample_dim: str | None = None, sel: dict | None = None, drop_sel: dict | None = None):
        super().__init__(strategy_name="dummy", sample_dim=sample_dim, sel=sel, drop_sel=drop_sel)


class ProcrustesAligner(SingleTargetAligner, SamplingMixin):
    """
    Concrete aligner that performs Procrustes analysis to align datasets.

    This method aligns target class means and finds an optimal rotation and scaling
    transformation to match the distributions between source and target domains.
    """

    _STRATEGY_MAP = {"single_target": SingleTargetStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        scaling: bool = True,
        strategy_name: str = "single_target",
        sampling_method: str = "mean",
        random_state: int = 0,
        group_coord: str | None = None,
        target_group: str | int | float | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize ProcrustesAligner.

        Args:
            target_coord: Target coordinate to adapt to.
            sample_dim: The dimension to average over when calculating class means.
            scaling: Whether to scale the data.
            strategy_name: The adaptation strategy to use. Currently only 'single_target' is supported.
            sampling_method: The method to use for sampling the data for alignment.
            random_state: The random state to use for sampling the data.
            group_coord: Group coordinate to adapt to.
            target_group: Target group to adapt to.
            n_jobs: Number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for adaptation. None means all data is used for adaptation calculations.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            **kwargs: Arguments passed to Transform base class.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            # strategy_kwargs
            group_coord=group_coord,
            target_group=target_group,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            # transform kwargs
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )
        self.target_coord = target_coord
        self.scaling = scaling
        self.sampling_method = sampling_method
        self.random_state = random_state

        assert self.sampling_method in [
            "mean",
            "min_count",
        ], "Sampling method must be either 'mean' or 'min_count'."

    def _fit_target(self, container: DataContainer, min_count: int | None = None, **kwargs) -> dict[str, Any]:
        """
        Fit the target domain.
        """
        target_data = self._prepare_data(container.data)
        target_centroid = np.mean(target_data.values, axis=0)
        sampled_target_data = self._sample_data(
            data=target_data,
            target_coord=self.target_coord,
            sample_dim=self.sample_dim,
            sampling_method=self.sampling_method,
            min_count=min_count,
            random_state=self.random_state,
        )

        # Store the source centroid as the reference
        target_centered = sampled_target_data.values - target_centroid

        # Store the source params
        return {
            "centroid": target_centroid,
            "centered": target_centered,
        }

    def _adapt_source(self, container: DataContainer, min_count: int | None = None, **kwargs) -> dict[str, Any]:
        """
        Adapt a single target domain.
        """
        source_data = self._prepare_data(container.data)

        # Get target centroid and centered data
        source_centroid = np.mean(source_data.values, axis=0)
        sampled_source_data = self._sample_data(
            data=source_data,
            target_coord=self.target_coord,
            sample_dim=self.sample_dim,
            sampling_method=self.sampling_method,
            min_count=min_count,
            random_state=self.random_state,
        )
        source_centered = sampled_source_data.values - source_centroid

        # Get target params
        target_centered = self.strategy.target_params["centered"]
        assert target_centered.shape == source_centered.shape, (
            "Centered target and source data must have the same shape (n_targets, n_features). "
            f"Source shape {source_centered.shape}, target shape {target_centered.shape}"
        )

        rotation_matrix, _ = linalg.orthogonal_procrustes(source_centered, target_centered)

        # Apply rotation to target
        source_rotated = source_centered @ rotation_matrix

        # Compute optimal scaling factor if enabled
        if self.scaling:
            numerator = np.sum(source_rotated * target_centered)
            denominator = np.sum(source_rotated * source_rotated)
            scale = numerator / denominator if denominator > 1e-10 else 1.0
        else:
            scale = 1.0

        return {
            "centroid": source_centroid,
            # "centered": source_centered,
            "rotation_matrix": rotation_matrix,
            "scale": scale,
        }

    def _adapted_transform(
        self,
        data_container: DataContainer,
        adapted_params: dict,
        target_params: dict,
        **kwargs,
    ) -> DataContainer:
        """
        Apply Procrustes transformation to data.

        Args:
            data_container: DataContainer to transform

        Returns:
            Transformed DataContainer
        """
        original_dims = data_container.data.dims
        data = self._prepare_data(data_container.data)

        # Get params
        source_centroid = adapted_params["centroid"]
        rotation_matrix = adapted_params["rotation_matrix"]
        scale = adapted_params["scale"]
        target_centroid = target_params["centroid"]

        # Apply transformation
        transformed_values = (data.values - source_centroid) @ rotation_matrix * scale + target_centroid

        # Create new DataArray with transformed data, transpose to match original dimensions
        transformed_da = xr.DataArray(transformed_values, dims=data.dims, coords=data.coords, attrs=data.attrs)
        transformed_da = transformed_da.transpose(*original_dims)

        return DataContainer(transformed_da)


class CoralAligner(SingleTargetAligner):
    """
    Correlation Alignment (CORAL) for domain adaptation.

    This method aligns the second-order statistics (covariances) of the
    source and target distributions by learning a linear transformation matrix.
    The transformation preserves the original data dimensions.

    The method learns a transformation matrix that aligns covariance structures:
    transform_matrix_ = source_cov^(-1/2) @ target_cov^(1/2)
    """

    _STRATEGY_MAP = {"single_target": SingleTargetStrategy}

    def __init__(
        self,
        sample_dim: str,
        reg: float = 1e-5,
        strategy_name: str = "single_target",
        group_coord: str | None = None,
        target_group: str | int | float | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize CoralAligner.

        Args:
            sample_dim: The dimension to average over when calculating class means.
            reg: Regularization parameter for covariance matrix stability
            strategy_name: The adaptation strategy to use. Currently only 'single_target' is supported.
            group_coord: Group coordinate to adapt to.
            target_group: Target group to adapt to.
            n_jobs: Number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for adaptation. None means all data is used for adaptation calculations.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            **kwargs: Arguments passed to Transform base class.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            # strategy_kwargs
            group_coord=group_coord,
            target_group=target_group,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            # transform kwargs
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.sample_dim = sample_dim
        self.reg = reg  # Regularization parameter

    def _prepare_data(self, data: xr.DataArray) -> xr.DataArray:
        """
        Prepare the data.
        """

        # TODO: support multi-dimensional data and handle reshaping after transformation
        if data.ndim != 2:
            raise ValueError(
                f"CoralAligner requires 2D data. Multi-dimensional data not supported yet. Received data with {data.ndim} dimensions: {data.dims}"
            )

        if data.dims[0] != self.sample_dim:
            data = data.transpose(self.sample_dim, ...)

        return data

    def _fit_target(self, container: DataContainer, **kwargs) -> dict[str, Any]:
        """
        Fit the target domain for CORAL.
        """
        target_data = self._prepare_data(container.data).values
        target_centroid = np.mean(target_data, axis=0)

        # Calculate covariance matrices with regularization
        cov_target = np.cov(target_data, rowvar=False) + self.reg * np.eye(target_data.shape[1])

        # Compute the square root of the target covariance matrix
        target_sqrt = linalg.sqrtm(cov_target)

        if np.iscomplexobj(target_sqrt):
            target_sqrt = target_sqrt.real

        return {"target_sqrt": target_sqrt, "centroid": target_centroid}

    def _adapt_source(self, container: DataContainer, **kwargs) -> dict[str, Any]:
        """
        Adapt a single source domain for CORAL.
        """
        source_data = self._prepare_data(container.data).values
        source_centroid = np.mean(source_data, axis=0)

        # Calculate covariance matrices with regularization
        cov_source = np.cov(source_data, rowvar=False) + self.reg * np.eye(source_data.shape[1])

        # Compute the inverse square root of the source covariance matrix
        source_sqrt_inv = linalg.inv(linalg.sqrtm(cov_source))
        if np.iscomplexobj(source_sqrt_inv):
            source_sqrt_inv = source_sqrt_inv.real

        # Get target params
        target_sqrt = self.strategy.target_params["target_sqrt"]

        # Compute the transformation matrix
        transform_matrix = source_sqrt_inv @ target_sqrt

        return {"transform_matrix": transform_matrix, "centroid": source_centroid}

    def _adapted_transform(
        self, data_container: DataContainer, adapted_params: dict, target_params: dict, **kwargs
    ) -> DataContainer:
        """
        Apply CORAL transformation to data.

        Args:
            data_container: DataContainer to transform

        Returns:
            Transformed DataContainer
        """
        original_dims = data_container.data.dims
        data = self._prepare_data(data_container.data)

        # Get params
        transform_matrix = adapted_params["transform_matrix"]
        source_centroid = adapted_params["centroid"]
        target_centroid = target_params["centroid"]

        # Apply CORAL transformation
        transformed_data = (data.values - source_centroid) @ transform_matrix + target_centroid
        transformed_data = transformed_data.real

        # Create new DataArray with transformed data, transpose to match original dimensions
        transformed_da = xr.DataArray(transformed_data, dims=data.dims, coords=data.coords, attrs=data.attrs)
        transformed_da = transformed_da.transpose(*original_dims)

        return DataContainer(transformed_da)


class SAAligner(SingleTargetAligner):
    """
    Subspace Alignment (SA) for domain adaptation.

    This method aligns the basis vectors (subspaces) of the source and
    target domains, learned via PCA. The transformation projects data
    onto source subspace then aligns it to target subspace.

    The method learns a transformation matrix that aligns PCA subspaces:
    transform_matrix_ = (source_basis @ target_basis)^T
    """

    _STRATEGY_MAP = {"single_target": SingleTargetStrategy}

    def __init__(
        self,
        sample_dim: str,
        n_components: int = 10,
        strategy_name: str = "single_target",
        group_coord: str | None = None,
        target_group: str | int | float | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize SAAligner.

        Args:
            sample_dim: The dimension to average over when calculating class means.
            n_components: Number of principal components to use for alignment
            strategy_name: The adaptation strategy to use. Currently only 'single_target' is supported.
            group_coord: Group coordinate to adapt to.
            target_group: Target group to adapt to.
            n_jobs: Number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for adaptation. None means all data is used for adaptation calculations.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            **kwargs: Arguments passed to Transform base class.
        """

        super().__init__(
            sample_dim=sample_dim,
            strategy_name=strategy_name,
            # strategy_kwargs
            group_coord=group_coord,
            target_group=target_group,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            # transform kwargs
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )
        self.n_components = n_components

    def _fit_target(self, container: DataContainer, **kwargs) -> dict[str, Any]:
        """
        Fit the target domain for Subspace Alignment.
        """
        from sklearn.decomposition import PCA

        target_data = self._prepare_data(container.data).values

        # Get the top principal components (basis vectors) for target domain
        pca_target = PCA(n_components=self.n_components).fit(target_data)
        target_basis = pca_target.components_.T

        return {"target_basis": target_basis, "centroid": pca_target.mean_}

    def _adapt_source(self, container: DataContainer, **kwargs) -> dict[str, Any]:
        """
        Adapt a single source domain for Subspace Alignment.
        """
        from sklearn.decomposition import PCA

        source_data = self._prepare_data(container.data).values

        # Get the top principal components (basis vectors) for each domain
        pca_source = PCA(n_components=self.n_components).fit(source_data)
        source_basis = pca_source.components_.T

        # Get target params
        target_basis = self.strategy.target_params["target_basis"]

        # Compute the transformation matrix that aligns the source basis to the target basis
        transform_matrix = source_basis @ target_basis.T

        return {"transform_matrix": transform_matrix, "centroid": pca_source.mean_}

    def _adapted_transform(
        self, data_container: DataContainer, adapted_params: dict, target_params: dict, **kwargs
    ) -> DataContainer:
        """
        Apply Subspace Alignment transformation to data.

        Args:
            data_container: DataContainer to transform

        Returns:
            Transformed DataContainer
        """
        original_dims = data_container.data.dims
        data = self._prepare_data(data_container.data)

        transform_matrix = adapted_params["transform_matrix"]
        source_centroid = adapted_params["centroid"]
        target_centroid = target_params["centroid"]

        # Center data, apply transformation, and shift to target centroid
        transformed_data = (data.values - source_centroid) @ transform_matrix + target_centroid

        # Create new DataArray with transformed data, transpose to match original dimensions
        transformed_da = xr.DataArray(transformed_data, dims=data.dims, coords=data.coords, attrs=data.attrs)
        transformed_da = transformed_da.transpose(*original_dims)

        return DataContainer(transformed_da)


class CCAAligner(JointGroupAligner, SamplingMixin):
    """
    Canonical Correlation Analysis (CCA) for domain adaptation.

    This method finds a common latent space for two domains by learning a linear transformation matrix
    that maximizes the correlation between the two domains.
    """

    _STRATEGY_MAP = {"joint": JointGroupStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        output_dim_name: str = "component",
        n_components: int = 10,
        strategy_name: str = "joint",
        group_coord: str | None = None,
        n_jobs: int = 1,
        sampling_method: str = "mean",
        random_state: int = 0,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the CCAAligner.

        Args:
            target_coord: The coordinate to use for the target domain.
            sample_dim: The dimension to use for the sample.
            output_dim_name: The name of the output dimension/space that all groups are aligned to.
            n_components: The number of components to learn.
            strategy_name: The strategy to use for adaptation.
            group_coord: The coordinate to group by. Determines different domains.
            n_jobs: The number of jobs to use for parallel processing.
            sampling_method: The method to use for sampling the data for alignment.
            random_state: The random state to use for sampling the data.
            adapt_sel: The selection criteria for adaptation. None means all data is used for adaptation calculations.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            group_coord=group_coord,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.n_components = n_components if n_components is not None else np.inf
        self.target_coord = target_coord
        self.sampling_method = sampling_method
        self.random_state = random_state

        assert self.sampling_method in ["mean", "min_count"], "Sampling method must be either 'mean' or 'min_count'."

    def _fit_joint(
        self, group_containers: dict[Hashable, DataContainer], min_count: int | None = None, **kwargs
    ) -> dict[Hashable, dict[str, Any]]:
        """
        Fit the CCA model for a domain pair.
        """

        # Prepare data
        groups = list(group_containers.keys())
        assert len(groups) == 2, "CCAAligner requires exactly two groups for adaptation."

        adapted_params = {group: {} for group in groups}  # dict[group_val: dict[param_name: param_value]]

        group_datasets = []  # list[np.ndarray]
        for group, container in group_containers.items():
            data = self._prepare_data(container.data)
            adapted_params[group]["mean"] = np.mean(data.values, axis=0)
            adapted_params[group]["std"] = np.std(data.values, axis=0)

            sampled_data = self._sample_data(
                data=data,
                target_coord=self.target_coord,
                sample_dim=self.sample_dim,
                sampling_method=self.sampling_method,
                min_count=min_count,
                random_state=self.random_state,
            )
            group_datasets.append(sampled_data.values)

        if not all(data.shape == group_datasets[0].shape for data in group_datasets):
            raise ValueError("All groups must have the same shape. Some groups are missing targets.")

        X_group = groups[0]
        y_group = groups[1]
        X_group_data = group_datasets[0]
        y_group_data = group_datasets[1]

        # Fit CCA model
        n_components = min(X_group_data.shape[0], X_group_data.shape[1], self.n_components)

        if n_components < self.n_components:
            warnings.warn(
                f"CCAAligner: specified n_components as {self.n_components} but max n_components allowed is {n_components}. Using {n_components}.",
                stacklevel=2,
            )

        cca = CCA(n_components=n_components)
        cca.fit(X_group_data, y_group_data)

        # Store adapted params
        adapted_params[X_group]["rotation_matrix"] = cca.x_rotations_
        adapted_params[y_group]["rotation_matrix"] = cca.y_rotations_

        return adapted_params

    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, **kwargs) -> DataContainer:
        """
        Apply CCA transformation to data.
        """

        data = self._prepare_data(data_container.data)
        data_np = data.values

        mean = adapted_params["mean"]
        std = adapted_params["std"]
        rotation_matrix = adapted_params["rotation_matrix"]

        # Apply normalization
        data_np -= mean
        # avoid division by zero
        std[std == 0] = 1.0
        data_np /= std

        # Apply rotation
        transformed_data = np.dot(data_np, rotation_matrix)

        output_container = self._create_output_container(original_da=data, transformed_np=transformed_data)

        return output_container


class MCCAAligner(JointGroupAligner, SamplingMixin):
    """
    Multiset Canonical Correlation Analysis (MCCA) for domain adaptation.

    This method finds a common latent space for multiple domains (groups) by
    finding projections that maximize the total correlation between the domains.
    It solves a generalized eigenvalue problem to find the canonical components.
    """

    _STRATEGY_MAP = {"joint": JointGroupStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        output_dim_name: str = "component",
        n_components: int = 10,
        reg: float = 1e-5,
        strategy_name: str = "joint",
        sampling_method: str = "mean",
        random_state: int = 0,
        group_coord: str | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize MCCAAligner.

        Args:
            target_coord: Target coordinate to adapt to.
            sample_dim: The dimension to average over when calculating class means.
            output_dim_name: The name of the output dimension/space that all groups are aligned to.
            n_components: Number of canonical components to keep.
            reg: Regularization parameter for covariance matrices.
            strategy_name: The adaptation strategy to use. Currently only 'joint' is supported.
            group_coord: Group coordinate to adapt to.
            n_jobs: Number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            **kwargs: Arguments passed to Transform base class.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            group_coord=group_coord,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.n_components = n_components
        self.target_coord = target_coord
        self.reg = reg
        self.sampling_method = sampling_method
        self.random_state = random_state

        assert self.sampling_method in ["mean", "min_count"], "Sampling method must be either 'mean' or 'min_count'."

    def _fit_joint(
        self, group_containers: dict[Hashable, DataContainer], min_count: int | None = None, **kwargs
    ) -> dict[Hashable, dict[str, Any]]:
        """
        Fit all domains jointly using MCCA.
        """
        # Prepare data
        groups = list(group_containers.keys())
        adapted_params = {group: {} for group in groups}  # dict[group_val: dict[param_name: param_value]]

        group_datasets = []  # list[np.ndarray]
        for container in group_containers.values():
            data = self._prepare_data(container.data)
            sampled_data = self._sample_data(
                data=data,
                target_coord=self.target_coord,
                sample_dim=self.sample_dim,
                sampling_method=self.sampling_method,
                min_count=min_count,
                random_state=self.random_state,
            )
            group_datasets.append(sampled_data.values)

        if not all(data.shape == group_datasets[0].shape for data in group_datasets):
            raise ValueError("All groups must have the same shape. Some groups are missing targets.")

        # Center data
        for i, group in enumerate(groups):
            centroid = np.mean(group_datasets[i], axis=0)  # (n_features)
            centered = group_datasets[i] - centroid
            group_datasets[i] = centered

            adapted_params[group]["centroid"] = centroid

        # Compute covariance matrices
        within_cov, between_cov = self._compute_covariance_matrices(group_datasets)

        # Solve generalized eigenvalue problem
        eigvals, eigvecs = self._solve_generalized_eigenvalue_problem(within_cov, between_cov)

        # Extract canonical weights for each dataset
        start_idx = 0
        for i, data in enumerate(group_datasets):
            n_features = data.shape[1]
            group = groups[i]
            end_idx = start_idx + n_features

            n_components = min(n_features, self.n_components)
            weights = eigvecs[start_idx:end_idx, :n_components]  # (n_features, n_components)
            adapted_params[group]["projection_matrix"] = weights

            start_idx = end_idx

        # Store canonical correlations (square root of eigenvalues)
        self.canonical_correlations_ = np.sqrt(np.maximum(0, eigvals[:n_components]))  # (n_components)

        return adapted_params

    def _compute_covariance_matrices(self, group_datasets: list[np.ndarray]) -> tuple:
        # Compute covariance matrices
        n_groups = len(group_datasets)
        n_samples = group_datasets[0].shape[0]

        # Compute within-set covariance matrices
        within_cov_list = []
        for X in group_datasets:
            cov = (X.T @ X) / (n_samples - 1)
            # Add regularization to diagonal
            cov += self.reg * np.eye(cov.shape[0])
            within_cov_list.append(cov)

        # Compute between-set covariance matrices
        between_blocks = []
        for i in range(n_groups):
            row_blocks = []
            for j in range(n_groups):
                if i == j:
                    row_blocks.append(within_cov_list[i])
                else:
                    # Between-set covariance
                    cov_ij = (group_datasets[i].T @ group_datasets[j]) / (n_samples - 1)
                    row_blocks.append(cov_ij)
            between_blocks.append(np.hstack(row_blocks))

        between_cov = np.vstack(between_blocks)

        # Create block diagonal within-set covariance matrix
        within_cov = linalg.block_diag(*within_cov_list)

        return within_cov, between_cov

    def _solve_generalized_eigenvalue_problem(
        self, within_cov: np.ndarray, between_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem for MCCA.

        The MCCA problem can be formulated as:
        maximize: w^T * S_b * w
        subject to: w^T * S_w * w = I

        where S_b is the between-set covariance and S_w is the block-diagonal
        within-set covariance matrix.
        This is solved via the equivalent problem on the total covariance S_t = S_b + S_w:
        S_t * w = λ * S_w * w
        """
        try:
            # Solve generalized eigenvalue problem: S_t * w = λ * S_w * w
            eigenvalues, eigenvectors = linalg.eigh(between_cov, within_cov)

            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            return eigenvalues, eigenvectors

        except np.linalg.LinAlgError:
            warnings.warn("Numerical issues encountered. Using regularized solution.", stacklevel=2)
            # Fallback: use SVD-based approach
            return self._svd_fallback(within_cov, between_cov)

    def _svd_fallback(self, within_cov: np.ndarray, between_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback SVD-based solution for numerical stability."""
        try:
            within_inv = linalg.inv(within_cov)
        except np.linalg.LinAlgError:
            within_inv = np.linalg.pinv(within_cov)

        # Compute the matrix for eigendecomposition
        M = within_inv @ between_cov

        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(M)

        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, **kwargs) -> DataContainer:
        """
        Apply MCCA transformation to data.
        """

        data = self._prepare_data(data_container.data)
        centroid = adapted_params["centroid"]
        projection_matrix = adapted_params["projection_matrix"]

        # Apply transformation
        centered_data = data.values - centroid
        transformed_data = centered_data @ projection_matrix

        output_container = self._create_output_container(original_da=data, transformed_np=transformed_data)

        return output_container


class GCCAAligner(JointGroupAligner, SamplingMixin):
    """
    Generalized Canonical Correlation Analysis (GCCA) for domain adaptation.

    This method finds a common latent space for multiple domains (groups) by
    finding projections that maximize the agreement between the domains. It is based
    on finding a common subspace that is predictable from all domains.
    This implementation solves the GCCA problem by finding the leading eigenvectors
    of the sum of projection matrices onto each domain's space.
    """

    _STRATEGY_MAP = {"joint": JointGroupStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        output_dim_name: str = "component",
        n_components: int = 10,
        reg: float = 1e-5,
        strategy_name: str = "joint",
        sampling_method: str = "mean",
        random_state: int = 0,
        group_coord: str | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize GCCAAligner.

        Args:
            target_coord: Target coordinate to adapt to.
            sample_dim: The dimension to average over when calculating class means.
            output_dim_name: The name of the output dimension/space that all groups are aligned to.
            n_components: Number of canonical components to keep.
            reg: Regularization parameter for covariance matrices.
            strategy_name: The adaptation strategy to use. Currently only 'joint' is supported.
            group_coord: Group coordinate to adapt to.
            n_jobs: Number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
            **kwargs: Arguments passed to Transform base class.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            group_coord=group_coord,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.n_components = n_components if n_components is not None else np.inf
        self.target_coord = target_coord
        self.reg = reg
        self.sampling_method = sampling_method
        self.random_state = random_state

        assert self.sampling_method in ["mean", "min_count"], "Sampling method must be either 'mean' or 'min_count'."

    def _fit_joint(
        self, group_containers: dict[Hashable, DataContainer], min_count: int | None = None, **kwargs
    ) -> dict[Hashable, dict[str, Any]]:
        """
        Fit all domains jointly using GCCA.
        """
        # Prepare data
        groups = list(group_containers.keys())
        adapted_params = {group: {} for group in groups}  # dict[group_val: dict[param_name: param_value]]

        group_datasets = []  # list[np.ndarray]
        for container in group_containers.values():
            data = self._prepare_data(container.data)
            sampled_data = self._sample_data(
                data=data,
                target_coord=self.target_coord,
                sample_dim=self.sample_dim,
                sampling_method=self.sampling_method,
                min_count=min_count,
                random_state=self.random_state,
            )
            group_datasets.append(sampled_data.values)

        if not all(data.shape == group_datasets[0].shape for data in group_datasets):
            raise ValueError("All groups must have the same shape. Some groups are missing targets.")

        # Center data
        for i, group in enumerate(groups):
            centroid = np.mean(group_datasets[i], axis=0)  # (n_features)
            centered = group_datasets[i] - centroid
            group_datasets[i] = centered

            adapted_params[group]["centroid"] = centroid

        # Compute sum of projection matrices
        n_samples = group_datasets[0].shape[0]
        sum_of_projections = np.zeros((n_samples, n_samples))
        for X in group_datasets:
            cov = X.T @ X
            # Add regularization to avoid singularity
            cov += self.reg * np.eye(cov.shape[1])
            try:
                cov_inv = linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)

            P_i = X @ cov_inv @ X.T
            sum_of_projections += P_i

        # Solve eigenvalue problem for the sum of projections
        eigvals, eigvecs = linalg.eigh(sum_of_projections)

        # Sort by eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Determine number of components and get common subspace basis U
        n_features_min = min(X.shape[1] for X in group_datasets)
        n_components = min(n_samples, n_features_min, self.n_components)

        if n_components < self.n_components:
            warnings.warn(
                f"GCCAAligner: specified n_components as {self.n_components} but max n_components allowed is {n_components}. Using {n_components}.",
                stacklevel=2,
            )

        U = eigvecs[:, :n_components]

        # Store eigenvalues
        self.eigenvalues_ = eigvals[:n_components]

        # Compute projection matrices (canonical weights) for each group
        for i, X in enumerate(group_datasets):
            group = groups[i]
            # W = (X.T @ X + reg*I)^-1 @ X.T @ U
            cov = X.T @ X
            cov += self.reg * np.eye(cov.shape[1])
            try:
                cov_inv = linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)

            weights = cov_inv @ X.T @ U
            adapted_params[group]["projection_matrix"] = weights

        return adapted_params

    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, **kwargs) -> DataContainer:
        """
        Apply GCCA transformation to data.
        """

        data = self._prepare_data(data_container.data)
        centroid = adapted_params["centroid"]
        projection_matrix = adapted_params["projection_matrix"]

        # Apply transformation
        centered_data = data.values - centroid
        transformed_data = centered_data @ projection_matrix

        output_container = self._create_output_container(original_da=data, transformed_np=transformed_data)

        return output_container


class JDAAligner(JointGroupAligner, SamplingMixin):
    """
    Joint Distribution Adaptation (JDA) for domain adaptation.

    This method simultaneously adapts both marginal and conditional distributions
    between domains using Maximum Mean Discrepancy (MMD). JDA learns a
    transformation that projects all groups to a common subspace while minimizing
    distribution discrepancy and preserving discriminative information.

    The method optimizes both:
    1. Marginal distribution alignment: P(X_1) ≈ P(X_2) ≈ ... ≈ P(X_k)
    2. Conditional distribution alignment: P(X_1|Y_1) ≈ P(X_2|Y_2) ≈ ... ≈ P(X_k|Y_k)
    """

    _STRATEGY_MAP = {"joint": JointGroupStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        output_dim_name: str = "component",
        n_components: int = 50,
        kernel: str = "linear",
        gamma: float = 1.0,
        mu: float = 0.5,
        reg: float = 1e-3,
        strategy_name: str = "joint",
        sampling_method: str = "mean",
        random_state: int = 0,
        group_coord: str | None = None,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize JDAAligner.

        Args:
            target_coord: Coordinate name for class labels (e.g., 'stimulus').
            sample_dim: Dimension name for samples (e.g., 'sample').
            output_dim_name: Name for the output component dimension.
            n_components: Number of components for dimensionality reduction.
            kernel: Kernel type for MMD computation ('linear', 'rbf').
            gamma: Kernel bandwidth parameter for RBF kernel.
            mu: Trade-off parameter between marginal (0) and conditional (1) adaptation.
            reg: Regularization parameter for numerical stability.
            strategy_name: Strategy to use ('joint' for JointGroupStrategy).
            sampling_method: Method for sampling data ('mean' or 'min_count').
            random_state: Random seed for reproducibility.
            group_coord: Coordinate name for groups (e.g., 'session').
            n_jobs: Number of parallel jobs.
            adapt_sel: Selection criteria for adaptation data.
            sel: General selection criteria.
            drop_sel: Coordinates to drop.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            group_coord=group_coord,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.target_coord = target_coord
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.mu = mu
        self.reg = reg
        self.sampling_method = sampling_method
        self.random_state = random_state

        # Validation
        assert 0 <= self.mu <= 1, "mu must be between 0 and 1"
        assert self.kernel in ["linear", "rbf"], "kernel must be 'linear' or 'rbf'"
        assert self.sampling_method in ["mean", "min_count"], "sampling_method must be 'mean' or 'min_count'"

    def _fit_joint(
        self, group_containers: dict[Hashable, DataContainer], min_count: int | None = None, **kwargs
    ) -> dict[Hashable, dict[str, Any]]:
        """
        Fit all groups jointly using JDA.

        This method computes a common transformation for all groups based on
        pairwise MMD minimization between all group pairs.
        """
        groups = list(group_containers.keys())
        if len(groups) < 2:
            raise ValueError("JDAAligner requires at least two groups for adaptation.")

        adapted_params = {group: {} for group in groups}

        # Sample and prepare data for all groups
        group_data = {}
        group_labels = {}

        for group, container in group_containers.items():
            data = self._prepare_data(container.data)
            centroid = np.mean(data.values, axis=0)
            adapted_params[group]["centroid"] = centroid

            # Sample data for JDA computation
            sampled_data = self._sample_data(
                data=data,
                target_coord=self.target_coord,
                sample_dim=self.sample_dim,
                sampling_method=self.sampling_method,
                min_count=min_count,
                random_state=self.random_state,
            )

            group_data[group] = sampled_data.values
            group_labels[group] = sampled_data.coords[self.target_coord].values

        # Verify consistent shapes across groups
        reference_shape = next(iter(group_data.values())).shape
        for group, data in group_data.items():
            if data.shape != reference_shape:
                raise ValueError(
                    f"Group '{group}' has shape {data.shape}, expected {reference_shape}. "
                    "All groups must have the same sampled shape."
                )

        # Compute joint transformation using all pairwise comparisons
        transformation_matrix = self._compute_joint_transformation(group_data, group_labels)

        # Store the same transformation for all groups
        for group in groups:
            adapted_params[group]["transformation_matrix"] = transformation_matrix

        return adapted_params

    def _compute_joint_transformation(
        self, group_data: dict[str, np.ndarray], group_labels: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute the joint transformation matrix using pairwise MMD.

        For simplicity, we use the first pair of groups to compute the transformation.
        A full implementation could optimize over all pairs or use a multi-view approach.
        """
        groups = list(group_data.keys())

        # Use first two groups for transformation computation
        group1, group2 = groups[0], groups[1]
        X1, X2 = group_data[group1], group_data[group2]
        labels1, labels2 = group_labels[group1], group_labels[group2]

        # Compute MMD matrices
        K, L = self._compute_mmd_matrix(X1, X2, labels1, labels2)

        # Solve the JDA optimization problem
        return self._solve_jda_optimization(K, L, X1)

    def _compute_mmd_matrix(
        self, X1: np.ndarray, X2: np.ndarray, labels1: np.ndarray, labels2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the MMD kernel and loss matrices.

        Args:
            X1, X2: Data matrices for two groups (n_samples, n_features)
            labels1, labels2: Label arrays for two groups (n_samples,)

        Returns:
            K: Kernel matrix (n_total, n_total)
            L: MMD loss matrix (n_total, n_total)
        """
        n1, n2 = X1.shape[0], X2.shape[0]

        # Combine data
        X_combined = np.vstack([X1, X2])

        # Compute kernel matrix
        if self.kernel == "linear":
            K = X_combined @ X_combined.T
        elif self.kernel == "rbf":
            # RBF kernel with numerical stability
            sq_dists = (
                np.sum(X_combined**2, axis=1, keepdims=True)
                + np.sum(X_combined**2, axis=1)
                - 2 * X_combined @ X_combined.T
            )
            sq_dists = np.maximum(sq_dists, 0.0)  # Numerical stability
            K = np.exp(-self.gamma * sq_dists)

        # Ensure K is finite
        K = np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)

        # Compute MMD loss matrix
        L = self._compute_mmd_loss_matrix(n1, n2, labels1, labels2)

        return K, L

    def _compute_mmd_loss_matrix(self, n1: int, n2: int, labels1: np.ndarray, labels2: np.ndarray) -> np.ndarray:
        """
        Compute the MMD loss matrix combining marginal and conditional terms.
        """
        n_total = n1 + n2

        # Marginal distribution alignment term
        L_marginal = np.zeros((n_total, n_total))
        L_marginal[:n1, :n1] = 1.0 / (n1 * n1)
        L_marginal[n1:, n1:] = 1.0 / (n2 * n2)
        L_marginal[:n1, n1:] = -1.0 / (n1 * n2)
        L_marginal[n1:, :n1] = -1.0 / (n1 * n2)

        # Conditional distribution alignment term
        L_conditional = np.zeros((n_total, n_total))

        # Find common labels
        unique_labels = np.unique(np.concatenate([labels1, labels2]))
        C = len(unique_labels)

        if C > 0:
            for label in unique_labels:
                # Indices for this label in each group
                idx1 = np.where(labels1 == label)[0]
                idx2 = np.where(labels2 == label)[0] + n1  # Offset for second group

                n1_c, n2_c = len(idx1), len(idx2)

                if n1_c > 0 and n2_c > 0:
                    # Within-group terms
                    L_conditional[np.ix_(idx1, idx1)] += 1.0 / (C * n1_c * n1_c)
                    L_conditional[np.ix_(idx2, idx2)] += 1.0 / (C * n2_c * n2_c)

                    # Between-group terms
                    L_conditional[np.ix_(idx1, idx2)] -= 1.0 / (C * n1_c * n2_c)
                    L_conditional[np.ix_(idx2, idx1)] -= 1.0 / (C * n1_c * n2_c)

        # Combine marginal and conditional terms
        L = (1 - self.mu) * L_marginal + self.mu * L_conditional

        # Ensure L is finite
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)

        return L

    def _solve_jda_optimization(self, K: np.ndarray, L: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Solve the JDA optimization problem via generalized eigenvalue decomposition.

        The problem is: min tr(A^T X^T K L K X A) / tr(A^T X^T K X A)
        which reduces to a generalized eigenvalue problem.
        """
        n_total = K.shape[0]
        n_samples = X.shape[0]

        # Centering matrix
        H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total

        # Compute matrices for generalized eigenvalue problem
        K_centered = H @ K @ H
        KLK = K @ L @ K

        # Add regularization
        K_centered_reg = K_centered + self.reg * np.eye(n_total)
        KLK_reg = KLK + self.reg * np.eye(n_total)

        try:
            # Solve generalized eigenvalue problem: KLK @ v = λ @ K_centered @ v
            eigenvals, eigenvecs = linalg.eigh(KLK_reg, K_centered_reg)

            # Filter and sort eigenvalues
            valid_mask = np.isfinite(eigenvals) & np.isreal(eigenvals)
            eigenvals = np.real(eigenvals[valid_mask])
            eigenvecs = np.real(eigenvecs[:, valid_mask])

            # Sort by ascending eigenvalues (we want smallest)
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Select top components
            n_components = min(self.n_components, len(eigenvals), X.shape[1])
            if n_components == 0:
                raise ValueError("No valid eigenvalues found")

            selected_eigenvecs = eigenvecs[:, :n_components]

            # Extract transformation for the first group
            alpha = selected_eigenvecs[:n_samples, :]

            # Compute feature-space transformation matrix
            A = X.T @ alpha  # (n_features, n_components)

            if np.isnan(A).any():
                raise np.linalg.LinAlgError("NaNs in transformation matrix")

            return A

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"JDA optimization failed ({e}). Using PCA fallback.", stacklevel=2)
            return self._pca_fallback(X)

    def _pca_fallback(self, X: np.ndarray) -> np.ndarray:
        """
        Fallback to PCA if JDA optimization fails.
        """
        from sklearn.decomposition import PCA

        n_components = min(self.n_components, X.shape[1], X.shape[0] - 1)
        n_components = max(1, n_components)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(X)

        return pca.components_.T  # (n_features, n_components)

    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, **kwargs) -> DataContainer:
        """
        Apply JDA transformation to data.
        """
        data = self._prepare_data(data_container.data)

        # Get parameters
        centroid = adapted_params["centroid"]
        transformation_matrix = adapted_params["transformation_matrix"]

        # Apply transformation
        centered_data = data.values - centroid
        transformed_data = centered_data @ transformation_matrix

        # Handle any remaining NaNs
        if np.isnan(transformed_data).any():
            warnings.warn("NaNs detected in transformed data. Replacing with zeros.", stacklevel=2)
            transformed_data = np.nan_to_num(transformed_data, nan=0.0)

        # Create output using parent method
        return self._create_output_container(original_da=data, transformed_np=transformed_data)


def _get_kernel(X, Y=None, kernel="linear", gamma=None, degree=3, coef0=1):
    """Compute kernel matrix."""
    if kernel == "linear":
        return linear_kernel(X, Y)
    elif kernel == "poly":
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    elif kernel == "rbf":
        return rbf_kernel(X, Y, gamma=gamma)
    elif kernel == "cosine":
        return cosine_similarity(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


class KCCAAligner(JointGroupAligner, SamplingMixin):
    """
    Kernel Canonical Correlation Analysis (KCCA) for domain adaptation.

    This method extends CCA to nonlinear relationships by using the kernel trick.
    It finds a common latent space for two domains by learning a nonlinear transformation
    that maximizes the correlation between the two domains.
    This is a custom implementation of KCCA.
    """

    _STRATEGY_MAP = {"joint": JointGroupStrategy}

    def __init__(
        self,
        target_coord: str,
        sample_dim: str,
        output_dim_name: str = "component",
        n_components: int = 10,
        strategy_name: str = "joint",
        group_coord: str | None = None,
        n_jobs: int = 1,
        sampling_method: str = "mean",
        random_state: int = 0,
        kernel: str = "linear",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        reg: float = 1e-5,
        adapt_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the KCCAAligner.

        Args:
            target_coord: The coordinate to use for the target domain.
            sample_dim: The dimension to use for the sample.
            output_dim_name: The name of the output dimension/space that all groups are aligned to.
            n_components: The number of components to learn.
            strategy_name: The strategy to use for adaptation.
            group_coord: The coordinate to group by. Determines different domains.
            n_jobs: The number of jobs to use for parallel processing.
            sampling_method: The method to use for sampling the data for alignment.
            random_state: The random state to use for sampling the data.
            kernel: Kernel mapping used internally. One of 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'.
            gamma: Kernel coefficient for rbf, poly and sigmoid.
            degree: Degree for poly kernels.
            coef0: Independent term in poly and sigmoid kernels.
            reg: Regularization parameter.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
            sel: Dictionary of coordinates to select.
            drop_sel: Dictionary of coordinates to drop.
        """

        super().__init__(
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            group_coord=group_coord,
            n_jobs=n_jobs,
            adapt_sel=adapt_sel,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.n_components = n_components if n_components is not None else np.inf
        self.target_coord = target_coord
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.reg = reg

        assert self.sampling_method in ["mean", "min_count"], "Sampling method must be either 'mean' or 'min_count'."

    def _fit_joint(
        self, group_containers: dict[Hashable, DataContainer], min_count: int | None = None, **kwargs
    ) -> dict[Hashable, dict[str, Any]]:
        """
        Fit the KCCA model for a domain pair.
        """
        groups = list(group_containers.keys())
        assert len(groups) == 2, "KCCAAligner requires exactly two groups for adaptation."

        group_datasets = []
        for container in group_containers.values():
            data = self._prepare_data(container.data)
            sampled_data = self._sample_data(
                data=data,
                target_coord=self.target_coord,
                sample_dim=self.sample_dim,
                sampling_method=self.sampling_method,
                min_count=min_count,
                random_state=self.random_state,
            )
            print("sampled_data", sampled_data.shape, self.sampling_method)
            group_datasets.append(sampled_data.values)

        if group_datasets[0].shape != group_datasets[1].shape:
            raise ValueError(
                f"All groups must have the same shape but we have {group_datasets[0].shape} and {group_datasets[1].shape}"
            )

        X, Y = group_datasets
        X_group, Y_group = groups

        # Compute kernel matrices
        Kx_unc = _get_kernel(X, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        Ky_unc = _get_kernel(Y, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)

        # Center kernel matrices
        n = Kx_unc.shape[0]
        N = np.eye(n) - np.ones((n, n)) / n
        Kx = N @ Kx_unc @ N
        Ky = N @ Ky_unc @ N

        # Regularize kernel matrices
        Kx_reg = Kx + self.reg * np.eye(n)
        Ky_reg = Ky + self.reg * np.eye(n)

        # Solve for KCCA coefficients using eigenvalue decomposition for stability
        Sx, Ux = linalg.eigh(Kx_reg)
        Sy, Uy = linalg.eigh(Ky_reg)

        # Filter out small eigenvalues
        Sx = np.where(Sx < 1e-12, 0, Sx)
        Sy = np.where(Sy < 1e-12, 0, Sy)

        # Compute inverse square root of kernel matrices
        Sx_pos_idx = Sx > 0
        Sy_pos_idx = Sy > 0
        Sx_inv_sqrt_diag = 1.0 / np.sqrt(Sx[Sx_pos_idx])
        Sy_inv_sqrt_diag = 1.0 / np.sqrt(Sy[Sy_pos_idx])

        Kx_inv_sqrt = Ux[:, Sx_pos_idx] @ np.diag(Sx_inv_sqrt_diag) @ Ux[:, Sx_pos_idx].T
        Ky_inv_sqrt = Uy[:, Sy_pos_idx] @ np.diag(Sy_inv_sqrt_diag) @ Uy[:, Sy_pos_idx].T

        # Form the matrix for SVD
        M = Kx_inv_sqrt @ Ky @ Ky_inv_sqrt
        U, s, Vh = np.linalg.svd(M)

        # Compute the dual coefficients
        alpha = Kx_inv_sqrt @ U
        beta = Ky_inv_sqrt @ Vh.T

        # Normalize coefficients to ensure unit variance of projections
        # This is crucial for the projections to be comparable.
        alpha_norms = np.diag(alpha.T @ Kx @ alpha)
        beta_norms = np.diag(beta.T @ Ky @ beta)

        # Avoid division by zero for components with zero variance
        alpha_norms = np.where(alpha_norms <= 1e-12, 1.0, alpha_norms)
        beta_norms = np.where(beta_norms <= 1e-12, 1.0, beta_norms)

        alpha = alpha / np.sqrt(alpha_norms)
        beta = beta / np.sqrt(beta_norms)

        n_components = min(self.n_components, alpha.shape[1])
        if n_components < self.n_components:
            warnings.warn(
                f"KCCAAligner: specified n_components as {self.n_components} but max n_components allowed is {n_components}. Using {n_components}.",
                stacklevel=2,
            )

        alpha_ = alpha[:, :n_components]
        beta_ = beta[:, :n_components]

        # Store parameters for transforming test data later
        adapted_params = {}
        adapted_params[X_group] = {
            "train_data": X,
            "K_train_col_mean": np.mean(Kx_unc, axis=0, keepdims=True),
            "K_global_mean": np.mean(Kx_unc),
            "projection_matrix": alpha_,
        }
        adapted_params[Y_group] = {
            "train_data": Y,
            "K_train_col_mean": np.mean(Ky_unc, axis=0, keepdims=True),
            "K_global_mean": np.mean(Ky_unc),
            "projection_matrix": beta_,
        }

        return adapted_params

    def _adapted_transform(self, data_container: DataContainer, adapted_params: dict, **kwargs) -> DataContainer:
        """
        Apply KCCA transformation to data.
        """
        data = self._prepare_data(data_container.data)
        data_np = data.values

        train_data = adapted_params["train_data"]
        K_train_col_mean = adapted_params["K_train_col_mean"]
        K_global_mean = adapted_params["K_global_mean"]
        projection_matrix = adapted_params["projection_matrix"]

        K_new = _get_kernel(
            data_np, train_data, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0
        )
        # Center K_new with respect to training data
        K_new_c = K_new - np.mean(K_new, axis=1, keepdims=True) - K_train_col_mean + K_global_mean
        transformed_data = K_new_c @ projection_matrix

        output_container = self._create_output_container(original_da=data, transformed_np=transformed_data)

        return output_container
