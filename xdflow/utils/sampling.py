import warnings
from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split

from xdflow.core.data_container import DataContainer


def get_container_by_conditions(container: DataContainer, conditions: dict) -> DataContainer:
    """
    Get a container by conditions.
    """
    return DataContainer(get_da_by_conditions(container.data, conditions))


def get_da_by_conditions(da: xr.DataArray, conditions: dict[str, Any]) -> xr.DataArray:
    """
    Select a DataArray subset based on flexible coordinate conditions.

    Each condition can be:
      - single value → equality
      - list → membership
      - tuple of 2 → range (inclusive)
      - dict with comparison operator → inequalities
        e.g. {'>': 5}, {'<=': 10}

    Example:
    conditions = {
        "latitude": {">": 15},          # latitude > 15
        "time": {"<=": 2},              # time <= 2
        "depth": (10, 30),              # between 10 and 30 (inclusive)
        "channel": [1, 3, 5],           # in [1, 3, 5]
        "animal": 35                    # equals 35
    }
    """
    # Per-dimension boolean masks as 1D NumPy arrays, applied with positional `isel` rather than
    # `where(drop=True)`, which silently casts int/bool data to float64 via its NaN fill.
    dim_masks: dict[Hashable, np.ndarray] = {}
    for key, value in conditions.items():
        coord = da[key]

        if len(coord.dims) != 1:
            raise ValueError(
                f"Coordinate {key} has {len(coord.dims)} dimensions, conditions must be applied to a single dimension"
            )
        coord_dim = coord.dims[0]
        coord_vals = np.asarray(coord.values)

        mask = dim_masks.get(coord_dim)
        if mask is None:
            mask = np.ones(coord_vals.shape[0], dtype=bool)

        # --- Operator-based conditions ---
        if isinstance(value, dict):
            for op, val in value.items():
                if op == ">":
                    mask &= coord_vals > val
                elif op == ">=":
                    mask &= coord_vals >= val
                elif op == "<":
                    mask &= coord_vals < val
                elif op == "<=":
                    mask &= coord_vals <= val
                elif op == "!=":
                    if isinstance(val, (list)):
                        mask &= ~np.isin(coord_vals, val)
                    else:
                        mask &= coord_vals != val
                else:
                    raise ValueError(f"Unsupported operator: {op}")

        # --- Range ---
        elif isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], bool):
            mask &= (coord_vals >= value[0]) & (coord_vals <= value[1])

        # --- Membership ---
        elif isinstance(value, (list, np.ndarray)):
            mask &= np.isin(coord_vals, value)

        # --- Equality ---
        else:
            mask &= coord_vals == value

        dim_masks[coord_dim] = mask

    for dim, mask in dim_masks.items():
        da = da.isel({dim: mask})

    return da


def train_test_split_container(
    container: DataContainer, target_coord: str, test_size: float = 0.2, random_state: int | None = None
) -> tuple[DataContainer, DataContainer]:
    """
    Split a container into train and test sets.
    """
    all_trials = container.data.trial.values

    if test_size is None or test_size == 0:
        # No holdout set - use all data for cross-validation
        train_indices, test_indices = all_trials, np.array([])
    else:
        # Get labels for stratification
        if target_coord not in container.data.coords:
            raise ValueError(f"Target coordinate '{target_coord}' not found in container coords.")
        labels = container.data.coords[target_coord].values

        # Perform stratified split
        train_indices, test_indices = train_test_split(
            all_trials, test_size=test_size, stratify=labels, random_state=random_state
        )

    train_container = DataContainer(container.data.sel(trial=train_indices))
    test_container = DataContainer(container.data.sel(trial=test_indices))

    return train_container, test_container


def stratified_sample(da, coord_name, max_samples_per_class=10, random_state=None) -> xr.DataArray:
    """
    Perform stratified sampling on categorical coordinates.
    #TODO: add support for non-categorical coordinates and balanced classes.

    Parameters:
    -----------
    da : xr.DataArray
        Input data array
    coord_name : str
        Name of categorical coordinate to stratify on
    max_samples_per_class : int
        Maximum number of samples per category/class
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    xr.DataArray
        Stratified sample with max_samples_per_class from each category
    """
    # Local generator: same draws as seeding the global RNG, without resetting it for the caller
    rng = np.random.RandomState(random_state) if random_state is not None else None

    # Get unique categories
    coord_values = da.coords[coord_name].values

    # Sample from each category
    sampled_indices = sample_by_max_count(np.arange(len(coord_values)), coord_values, max_samples_per_class, rng=rng)

    if coord_name not in da.dims:
        dim_names = da.coords[coord_name].dims
        if len(dim_names) > 1:
            raise ValueError(f"Coordinate {coord_name} has multiple dimensions: {dim_names}")
        dim_name = dim_names[0]
    else:
        dim_name = coord_name

    # Return stratified sample using xarray's isel
    return da.isel({dim_name: sampled_indices})


def sample_by_max_count(
    indices: np.ndarray, labels: np.ndarray, max_samples: int, rng: np.random.RandomState | None = None
) -> np.ndarray:
    """Sample up to max_samples from each class."""
    sampled_indices = []

    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_indices = indices[class_mask]
        class_size = len(class_indices)

        n_samples = min(class_size, max_samples)

        if n_samples < max_samples:
            warnings.warn(f"Class {class_label} has only {class_size} samples, requested {max_samples}")

        sampled = (rng or np.random).choice(class_indices, size=n_samples, replace=False)
        sampled_indices.append(sampled)

    return np.concatenate(sampled_indices)


def sample_by_fraction(
    indices: np.ndarray, labels: np.ndarray, all_labels: np.ndarray, sample_fraction: float
) -> np.ndarray:
    """Sample a fraction of each class based on the whole dataset."""
    sampled_indices = []

    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_indices = indices[class_mask]
        class_size = len(class_indices)

        # Base fraction on the whole dataset class size
        total_class_size = np.sum(all_labels == class_label)
        n_samples = max(1, int(total_class_size * sample_fraction))
        # Can't sample more than available
        n_samples = min(n_samples, class_size)

        if n_samples > class_size:
            warnings.warn(f"Class {class_label} has only {class_size} samples in this fold, requested {n_samples}")

        sampled = np.random.choice(class_indices, size=n_samples, replace=False)
        sampled_indices.append(sampled)

    return np.concatenate(sampled_indices)


def get_group_dim(container: DataContainer, group_coord: str) -> str:
    """Resolves the dimension that the group_coord indexes."""
    if group_coord not in container.data.coords:
        raise ValueError(f"Group coordinate '{group_coord}' not found in data coordinates")

    coord_dims = container.data.coords[group_coord].dims
    if len(coord_dims) != 1:
        raise ValueError(
            f"Group coordinate '{group_coord}' must index exactly one dimension, "
            f"but it indexes {len(coord_dims)}: {coord_dims}"
        )
    return coord_dims[0]


def discover_groups(container: DataContainer, group_coord: str) -> list[Hashable]:
    """Discovers unique group values from the data."""
    group_values = container.data.coords[group_coord].values
    return sorted(np.unique(group_values).tolist())


def select_group(container: DataContainer, group_coord: str, group_val: Hashable) -> DataContainer:
    """Selects data for a specific group using boolean indexing."""
    group_mask = container.data.coords[group_coord] == group_val
    group_data = container.data.where(group_mask, drop=True)
    return DataContainer(group_data)
