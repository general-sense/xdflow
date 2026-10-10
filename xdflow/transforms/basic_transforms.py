from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xdflow.core.base import Transform, _infer_dict_key_type

# Assuming the base class and DataContainer are in these locations
from xdflow.core.data_container import DataContainer


class TransposeDimsTransform(Transform):
    """Transpose dimensions of the data container."""

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, dims: tuple[str, ...], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        """
        Initializes the TransposeDimsTransform.

        Args:
            dims: The dimensions to transpose to.
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.dims = dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        return container.transpose(*self.dims)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.dims)


class RenameDimsTransform(Transform):
    """Rename xarray dimension names in the data container.

    This transform applies `xarray.DataArray.rename` to change dimension names
    (and matching coordinate names) according to a provided mapping. It preserves
    data and coordinate values.

    Example:
        Rename `feature` to `channel` after a union or feature step:
        transform = RenameDimsTransform(rename_map={"feature": "channel"})

    Args:
        rename_map: Mapping from old dimension names to new names.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, rename_map: dict[str, str], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.rename_map = rename_map

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Apply dimension renaming.

        Args:
            container: Input data container.

        Returns:
            A new `DataContainer` with renamed dimensions.
        """
        data = container.data
        renamed = data.rename(self.rename_map)
        return DataContainer(renamed)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Infer output dims by applying the mapping to the input dims.

        Args:
            input_dims: Tuple of input dim names.

        Returns:
            Tuple of output dim names with mapping applied.
        """
        return tuple(self.rename_map.get(d, d) for d in input_dims)


class IdentityTransform(Transform):
    """A no-op transform that returns the input unchanged.

    This is useful as a selectable option in a `SwitchTransform` when you want
    the step to optionally do nothing while preserving dimension contracts.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None):
        """Initialize an identity transform.

        Args:
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Return the input container unchanged.

        Args:
            container: Input container.

        Returns:
            The same `DataContainer` instance.
        """
        return container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Return the same dims as input.

        Args:
            input_dims: Input dimension names.

        Returns:
            The same dimension names.
        """
        return input_dims


class SampleWeightTransform(Transform):
    """Attach a `sample_weight` coordinate derived from an existing coordinate.

    The transform maps values from a source coordinate (e.g. ``session``) to
    scalar weights using either a ``weight_map`` or a callable ``weight_func``.
    The resulting weights are written as a new coordinate (default
    ``sample_weight``) on the same dimension(s) as the source, so downstream
    predictors can consume them without changing the data layout.

    Partial write-back with ``transform_sel``/``transform_drop_sel`` is not
    supported because this transform changes coordinates.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = False

    def __init__(
        self,
        coord_name: str,
        weight_map: Mapping[Any, float] | None = None,
        weight_func: Callable[[Any], float] | None = None,
        default_weight: float = 1.0,
        target_coord: str = "sample_weight",
        dtype: np.dtype | type = np.float64,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
    ):
        """
        Args:
            coord_name: Name of the coordinate to read values from.
            weight_map: Mapping of coordinate values to weights.
            weight_func: Callable taking a coordinate value and returning a weight.
            default_weight: Weight to use when neither map nor func provide a value.
            target_coord: Name of the coordinate that will store computed weights.
            dtype: Numeric dtype used for the weight array.
        """
        if weight_map is not None and weight_func is not None:
            raise ValueError("Specify either 'weight_map' or 'weight_func', not both.")
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.coord_name = coord_name
        self.weight_map = {key: float(value) for key, value in weight_map.items()} if weight_map is not None else None
        self.weight_func = weight_func
        self.default_weight = float(default_weight)
        self.target_coord = target_coord
        self.dtype = dtype
        self._active_for_inference = True

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        weight_map = self.weight_map
        if weight_map is not None:
            for key, value in weight_map.items():
                params[f"weight_map__{key}"] = value
        return params

    def set_params(self, **params: Any) -> SampleWeightTransform:
        params = dict(params)
        map_updates: dict[Any, float] = {}
        for key in list(params.keys()):
            if key.startswith("weight_map__"):
                if self.weight_func is not None:
                    raise ValueError("Cannot tune 'weight_map__*' parameters when 'weight_func' is set.")
                _, map_key = key.split("__", 1)
                map_updates[map_key] = params.pop(key)

        if "weight_map" in params:
            new_map = params.pop("weight_map")
            if new_map is None:
                self.weight_map = None
            else:
                self.weight_map = {key: float(value) for key, value in dict(new_map).items()}

        if map_updates:
            if self.weight_map is None:
                self.weight_map = {}
            else:
                self.weight_map = dict(self.weight_map)
            for map_key, map_value in map_updates.items():
                typed_key = _infer_dict_key_type(self.weight_map, map_key)
                self.weight_map[typed_key] = float(map_value)

        super().set_params(**params)
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        if not self._active_for_inference:
            return container

        data = container.data
        if self.coord_name not in data.coords:
            raise ValueError(f"Coordinate '{self.coord_name}' not found in DataArray.")

        source_coord = data.coords[self.coord_name]
        coord_values = source_coord.values

        weight_map = self.weight_map
        if weight_map is not None:
            # Convert numpy scalar types to Python native types for dict lookup
            # (numpy.int64 != int in dict.get(), causing all lookups to fail)
            def safe_lookup(value):
                # Convert numpy scalars to Python native types
                if hasattr(value, "item"):
                    value = value.item()
                return weight_map.get(value, self.default_weight)

            vectorized_lookup = np.vectorize(safe_lookup, otypes=[np.float64])
            weights = vectorized_lookup(coord_values)
        elif self.weight_func is not None:
            vectorized_func = np.vectorize(self.weight_func, otypes=[np.float64])
            weights = vectorized_func(coord_values)
        else:
            weights = np.full_like(coord_values, fill_value=self.default_weight, dtype=np.float64)

        try:
            weights_array = np.asarray(weights, dtype=self.dtype)
        except TypeError as exc:
            raise TypeError(f"Could not convert computed weights to dtype '{self.dtype}'.") from exc

        if weights_array.shape != source_coord.shape:
            raise ValueError(
                "Computed weights must match the shape of the source coordinate "
                f"(expected {source_coord.shape}, got {weights_array.shape})."
            )

        weight_coord = source_coord.copy(data=weights_array)
        weight_coord.name = self.target_coord
        updated = data.assign_coords({self.target_coord: weight_coord})

        return DataContainer(updated)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims

    def _disable_for_inference(self) -> None:
        self._active_for_inference = False


class BalanceClassWeightTransform(Transform):
    """Attach a sample-weight coordinate that balances classes, optionally by domain.

    Partial write-back with ``transform_sel``/``transform_drop_sel`` is not
    supported because this transform changes coordinates.
    """

    # Stateful so cross-validation recomputes weights from each training fold; computed once over all
    # data, the class counts of validation/holdout trials would shape the training weights.
    is_stateful: bool = True
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = False

    def __init__(
        self,
        class_coord: str,
        balance_domains: bool = False,
        domain_coord: str | None = None,
        domain_weights: Mapping[Any, float] | None = None,
        normalize_domain_totals: bool = False,
        weight_normalize: str | None = None,
        target_coord: str = "sample_weight",
        dtype: np.dtype | type = np.float64,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
    ):
        if weight_normalize is not None and weight_normalize not in {"mean", "sum"}:
            raise ValueError("weight_normalize must be one of: None, 'mean', or 'sum'.")
        if not balance_domains:
            if domain_coord is not None:
                raise ValueError("Specify 'domain_coord' only when balance_domains=True.")
            if domain_weights is not None:
                raise ValueError("Specify 'domain_weights' only when balance_domains=True.")
            if normalize_domain_totals:
                raise ValueError("Specify 'normalize_domain_totals' only when balance_domains=True.")
        elif domain_coord is None:
            raise ValueError("domain_coord is required when balance_domains=True.")

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.class_coord = class_coord
        self.balance_domains = balance_domains
        self.domain_coord = domain_coord
        self.domain_weights = {key: float(value) for key, value in domain_weights.items()} if domain_weights else None
        self.normalize_domain_totals = normalize_domain_totals
        self.weight_normalize = weight_normalize
        self.target_coord = target_coord
        self.dtype = dtype

    def _fit(self, container: DataContainer, **kwargs) -> BalanceClassWeightTransform:
        # Nothing to learn: weights depend only on the data being transformed (the training fold inside CV).
        return self

    @staticmethod
    def _safe_key(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

    def _resolve_domain_weights(self, domain_values: np.ndarray) -> dict[Any, float]:
        if self.domain_weights is not None:
            weights = dict(self.domain_weights)
        else:
            if len(domain_values) == 0:
                raise ValueError("No domain values found; cannot compute weights.")
            equal_weight = 1.0 / float(len(domain_values))
            weights = {value: equal_weight for value in domain_values}

        weights = {self._safe_key(key): float(value) for key, value in weights.items()}
        domain_keys = [self._safe_key(value) for value in domain_values]
        missing = [value for value in domain_keys if value not in weights]
        if missing:
            raise ValueError(f"Domain weight(s) missing for: {missing}")

        return weights

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        if self.class_coord not in data.coords:
            raise ValueError(f"Coordinate '{self.class_coord}' not found in DataArray.")

        class_coord = data.coords[self.class_coord]
        if class_coord.ndim != 1:
            raise ValueError("Class coordinate must be 1D for sample-weight computation.")

        classes = np.asarray(class_coord.values)
        if pd.isna(classes).any():
            raise ValueError("Class coordinate contains missing values (NaN/None).")

        if not self.balance_domains:
            class_values, class_counts = np.unique(classes, return_counts=True)
            class_count_map = {
                self._safe_key(class_value): int(class_count)
                for class_value, class_count in zip(class_values, class_counts, strict=True)
            }
            weights = np.array([1.0 / class_count_map[self._safe_key(value)] for value in classes], dtype=np.float64)
            weight_coord_template = class_coord
        else:
            if self.domain_coord not in data.coords:
                raise ValueError(f"Coordinate '{self.domain_coord}' not found in DataArray.")

            domain_coord = data.coords[self.domain_coord]
            if class_coord.dims != domain_coord.dims:
                raise ValueError(
                    f"Class coord dims {class_coord.dims} must match domain coord dims {domain_coord.dims}."
                )
            if domain_coord.ndim != 1:
                raise ValueError("Domain coordinate must be 1D for sample-weight computation.")

            domains = np.asarray(domain_coord.values)
            if pd.isna(domains).any():
                raise ValueError("Domain coordinate contains missing values (NaN/None).")

            unique_domains = np.unique(domains)
            domain_weights = self._resolve_domain_weights(unique_domains)
            weights = np.zeros_like(domains, dtype=np.float64)

            for domain in unique_domains:
                domain_key = self._safe_key(domain)
                domain_mask = domains == domain
                domain_count = int(np.sum(domain_mask))
                if domain_count == 0:
                    continue

                domain_classes = classes[domain_mask]
                class_values, class_counts = np.unique(domain_classes, return_counts=True)
                if len(class_values) == 0:
                    continue

                class_count_map = {
                    self._safe_key(class_value): int(class_count)
                    for class_value, class_count in zip(class_values, class_counts, strict=True)
                }

                if self.normalize_domain_totals:
                    base = domain_weights[domain_key] / float(len(class_values))
                else:
                    base = domain_weights[domain_key] / float(domain_count)

                counts = np.array(
                    [class_count_map[self._safe_key(value)] for value in domain_classes], dtype=np.float64
                )
                weights[domain_mask] = base / counts
            weight_coord_template = domain_coord

        try:
            weights_array = np.asarray(weights, dtype=self.dtype)
        except TypeError as exc:
            raise TypeError(f"Could not convert computed weights to dtype '{self.dtype}'.") from exc

        if self.weight_normalize == "mean":
            mean_weight = float(np.mean(weights_array))
            if mean_weight <= 0:
                raise ValueError("Cannot normalize weights by mean when mean <= 0.")
            weights_array = weights_array / mean_weight
        elif self.weight_normalize == "sum":
            total_weight = float(np.sum(weights_array))
            if total_weight <= 0:
                raise ValueError("Cannot normalize weights by sum when sum <= 0.")
            weights_array = weights_array / total_weight

        if weights_array.shape != weight_coord_template.shape:
            raise ValueError(
                "Computed weights must match the shape of the template coordinate "
                f"(expected {weight_coord_template.shape}, got {weights_array.shape})."
            )

        weight_coord = weight_coord_template.copy(data=weights_array)
        weight_coord.name = self.target_coord
        updated = data.assign_coords({self.target_coord: weight_coord})

        return DataContainer(updated)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims


class AverageTransform(Transform):
    """Average data over one or more dimensions.

    The named dimensions are reduced with `xarray.DataArray.mean`, so their
    coordinate values are removed from the result and all other dimensions are
    preserved. Attributes are kept on the resulting array.
    """

    is_stateful: bool = False
    # No constraints on input or output dimensions, as they are dynamic.
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, dims: str | tuple[str, ...], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        """Initialize an averaging transform.

        Args:
            dims: Dimension name or dimension names to average over.
            sel: Label selection applied before averaging.
            drop_sel: Label selection dropped before averaging.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.dims = (dims,) if isinstance(dims, str) else dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the averaging transformation.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with data averaged over the specified dimensions.
        """
        data = container.data

        # Perform the averaging using xarray's mean function
        averaged_data = data.mean(dim=self.dims, keep_attrs=True)

        # Create, log history, and return the new container
        new_container = DataContainer(averaged_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions by removing the averaged dimensions.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        return tuple(dim for dim in input_dims if dim not in self.dims)


class FlattenTransform(Transform):
    """Stack multiple dimensions into one new dimension.

    Flattening uses `xarray.DataArray.stack`, so the new dimension receives a
    pandas `MultiIndex` coordinate containing the original coordinate labels.
    The new dimension is named `flat_<dim1>__<dim2>`.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, dims: tuple[str, ...], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        """Initialize a flattening transform.

        Args:
            dims: At least two dimension names to stack into one.
            sel: Label selection applied before flattening.
            drop_sel: Label selection dropped before flattening.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        if not isinstance(dims, tuple) or len(dims) < 2:
            raise ValueError("`dims` must be a tuple of at least two strings.")
        self.dims = dims

    @property
    def new_dim_name(self) -> str:
        return f"flat_{'__'.join(self.dims)}"

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the flattening transformation using xarray's stack method.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with specified dimensions stacked into a single new one.
        """
        data = container.data

        # Validate that all requested dims exist in the data
        missing_dims = [dim for dim in self.dims if dim not in data.dims]
        if missing_dims:
            raise ValueError(
                f"FlattenTransform: requested dims {missing_dims} not found in data dims {list(data.dims)}"
            )

        # Stack the specified dimensions into a new dimension
        flattened_data = data.stack({self.new_dim_name: self.dims})

        # Create, log history, and return the new container
        new_container = DataContainer(flattened_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after flattening. The new
        flattened dimension is appended at the end.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        # Remove the original dimensions that are being flattened
        remaining_dims = [dim for dim in input_dims if dim not in self.dims]
        # Add the new flattened dimension to the end
        return tuple(remaining_dims) + (self.new_dim_name,)


class FunctionTransform(Transform):
    """
    Applies a function to the whole xarray.DataArray. The function must work on xarray.DataArray/numpy.
    Useful for applying simple mathemtical functions like np.abs, np.log, etc.
    For xarray functions with additional arguments, use partial functions, e.g.

        from functools import partial
        FunctionTransform(func=partial(xr.DataArray.max, dim="time"), expected_output_dims=("trial", "channel", "freq_band"))
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        func: Callable,
        expected_output_dims: tuple[str, ...]
        | None = None,  # needed for functions that change dimensionality (e.g. np.mean)
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        """
        Initializes the FunctionTransform.

        Args:
            func: The function to apply to the data. The function must work on xarray.DataArray/numpy.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.

            It is highly recommended to use a vectorized function from NumPy (e.g., `np.abs`) or from xarray
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.func = func

        # Perform a check to ensure the function is compatible with xarray DataArrays at initialization.
        try:
            # Use a very small, simple DataArray for the check.
            test_data = xr.DataArray(np.array([1.0, 2.0]))
            self.func(test_data)
        except Exception as e:
            raise ValueError(
                f"The provided function '{getattr(self.func, '__name__', 'unknown')}' is not compatible with xarray.DataArray. "
                "Please provide a vectorized function (like those from NumPy). "
            ) from e

        self.expected_output_dims = expected_output_dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        transformed_data = self.func(data)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if self.expected_output_dims is not None:
            warnings.warn(
                f"Expected output dimensions are manually specified for {getattr(self.func, '__name__', 'unknown')}. Might be inaccurate since not computed."
            )
            return self.expected_output_dims

        return input_dims


class UnflattenTransform(Transform):
    """Unstack a flattened dimension back into its source dimensions.

    The input dimension must follow the naming convention produced by
    `FlattenTransform`, such as `flat_trial__time`. If the coordinate is not
    already a pandas `MultiIndex`, XDFlow attempts to build one from tuple-like
    coordinate values before unstacking.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, dim: str, sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None):
        """
        Initializes the UnflattenTransform.

        Args:
            dim: dimension to unflatten, must follow naming output of FlattenTransform (e.g. 'flat_dim1__dim2').
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        if not dim.startswith("flat_"):
            raise ValueError("`dim` must have been flattened before, and must start with 'flat_'.")
        self.dim = dim

    @property
    def new_dim_names(self) -> list[str]:
        return self.dim.removeprefix("flat_").split("__")

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the unflattening transformation using xarray's unstack method.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with specified dimensions stacked into a single new one.
        """
        data = container.data

        # Validate that requested dim exists in the data
        if self.dim not in data.dims:
            raise ValueError(f"UnflattenTransform: requested dim {self.dim} not found in data dims {list(data.dims)}")

        # Check if the coordinate for the dimension is a MultiIndex
        if not isinstance(data.coords[self.dim].to_index(), pd.MultiIndex):
            # If not, try to create one assuming the coordinate values are tuples
            try:
                multi_index = pd.MultiIndex.from_tuples(data.coords[self.dim].values, names=self.new_dim_names)
                data = data.assign_coords({self.dim: multi_index})
            except Exception as e:
                # If that fails, re-raise with a more informative error
                raise ValueError(
                    f"Failed to unstack '{self.dim}'. It is not a MultiIndex, "
                    f"and could not be converted from coordinate values. Original error: {e}"
                ) from e

        # Unstack the specified dimension into multiple dimensions
        unflattened_data = data.unstack(self.dim)

        # When unstacking, coordinates that should only depend on 'trial' can get
        # broadcast across the newly created dimensions. This block reverts them
        # so they only depend on 'trial'.
        if "trial" in self.new_dim_names:
            # Find the other dimensions that came from unstacking
            other_dims = [dim for dim in self.new_dim_names if dim != "trial"]

            coords_to_update = {}
            for coord_name, coord_data in unflattened_data.coords.items():
                # We are looking for coordinates that are not the dimensions themselves
                if coord_name in self.new_dim_names:
                    continue

                # Check if the coordinate depends on 'trial' and at least one of the other new dims
                if "trial" in coord_data.dims and any(d in coord_data.dims for d in other_dims):
                    # Drop the other dims only if the values were merely broadcast along them;
                    # coords that genuinely vary per (trial, other dim) are kept as they are.
                    selector = {dim: 0 for dim in other_dims if dim in coord_data.dims}
                    if selector:
                        new_coord_data = coord_data.isel(**selector, drop=True)
                        if bool((coord_data.variable == new_coord_data.variable).all()):
                            # assign the bare variable: the DataArray carries sliced copies of other coords
                            coords_to_update[coord_name] = new_coord_data.variable

            if coords_to_update:
                unflattened_data = unflattened_data.assign_coords(coords_to_update)

        # Create, log history, and return the new container
        new_container = DataContainer(unflattened_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after unflattening. The new
        unflattened dimensions are appended at the end.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        # Remove the original dimensions that are being unflattened
        remaining_dims = [dim for dim in input_dims if dim != self.dim]
        # Add the new unflattened dimensions to the end
        return tuple(remaining_dims) + tuple(self.new_dim_names)


class TrialSampler(Transform):
    """Select a fixed number of trials from the `trial` dimension.

    The transform samples by integer position. When `shuffle=True`, trial order
    is shuffled with NumPy's default random generator before the first
    `n_trials` positions are selected.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        n_trials: int,
        shuffle: bool = True,
        random_state: int = 0,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        """Initialize trial subsampling.

        Args:
            n_trials: Number of trials to keep.
            shuffle: Whether to shuffle trial indices before sampling.
            random_state: Seed used when `shuffle=True`.
            sel: Label selection applied before sampling.
            drop_sel: Label selection dropped before sampling.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.n_trials = n_trials
        self.shuffle = shuffle
        self.random_state = random_state

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Samples trials from the data.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)
        """
        data = container.data

        num_trials = data.sizes.get("trial", 0)
        if self.n_trials > num_trials:
            raise ValueError(f"Cannot sample {self.n_trials} trials when only {num_trials} are available.")

        trial_indices = np.arange(num_trials)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(trial_indices)

        sampled_indices = trial_indices[: self.n_trials]
        sampled_data = data.isel(trial=sampled_indices)

        new_container = DataContainer(sampled_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after sampling.
        """
        return input_dims


class CropTimeTransform(Transform):
    """
    Crop data to a time window using the time coordinate, inclusive.

    This transform selects a subset of the data along a time-like coordinate using
    label-based slicing, preserving all other dimensions and coordinates.

    The start and end are inclusive and interpreted in the same units as the
    DataArray's time coordinate values.

    Args:
        start: Start of the time window (inclusive), in the time coordinate's units.
        end: End of the time window (inclusive), in the time coordinate's units.
        time_coord: Name of the time coordinate/dimension to slice over. Defaults to "time".
        sel: Optional selection to apply before transforming.
        drop_sel: Optional drop selection to apply before transforming.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        start: float,
        end: float,
        time_coord: str = "time",
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.start = start
        self.end = end
        self.time_coord = time_coord

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Apply inclusive time-window cropping using label-based selection.

        Args:
            container: Input data container.

        Returns:
            A new `DataContainer` cropped to the specified time window.
        """
        data = container.data

        if self.time_coord not in data.coords:
            raise ValueError(f"Time coordinate '{self.time_coord}' not found in DataArray coords: {list(data.coords)}")

        # Inclusive label-based slice on the time coordinate
        sliced = container.sel(**{self.time_coord: slice(self.start, self.end)})
        if sliced.data.sizes.get(self.time_coord, 1) == 0:
            # Most often a unit mismatch, e.g. milliseconds given for a time coordinate in seconds
            values = data.coords[self.time_coord].values
            raise ValueError(
                f"CropTimeTransform window [{self.start}, {self.end}] selects no samples; '{self.time_coord}' spans "
                f"{values.min()} to {values.max()}. start/end must be in the time coordinate's units."
            )
        return sliced

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Cropping preserves dimension names; sizes may shrink.

        Args:
            input_dims: Input dimension names.

        Returns:
            The same dimension names.
        """
        return input_dims
