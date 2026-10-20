"""Converters from other libraries' data objects into xdflow containers."""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xdflow.core.data_container import DataContainer


def from_mne_epochs(epochs: Any, label_coord: str = "label", **trial_coords: Any) -> DataContainer:
    """Convert MNE ``Epochs`` into a ``(trial, channel, time)`` container.

    MNE is not imported here: any object with the ``Epochs`` interface works. Each trial is labelled with its
    event name (the inverse of ``epochs.event_id``), every ``epochs.metadata`` column becomes a trial
    coordinate, and the sampling rate is kept in ``attrs["sfreq"]``.

    Args:
        epochs: An ``mne.Epochs`` (or ``EpochsArray``) object.
        label_coord: Name of the trial coordinate holding the event names.
        **trial_coords: Extra trial coordinates, each a scalar (same for every trial, e.g. ``subject=3``) or a
            sequence with one value per trial.

    Returns:
        A DataContainer with dims ``(trial, channel, time)``; ``time`` is in seconds.

    Example:
        >>> data = from_mne_epochs(epochs, subject=3)
        >>> both = DataContainer(xr.concat([data_s1.data, data_s2.data], dim="trial"))  # then renumber trials
    """
    channel_names = list(epochs.ch_names)
    values = np.asarray(epochs.get_data(picks=channel_names))
    n_trials = values.shape[0]
    if values.shape[1] != len(channel_names):
        raise ValueError(f"Got {values.shape[1]} channels from epochs.get_data() for {len(channel_names)} names.")

    names_by_code = {code: name for name, code in epochs.event_id.items()}
    codes = np.asarray(epochs.events)[:, 2]
    unknown = sorted(set(codes.tolist()) - set(names_by_code))
    if unknown:
        raise ValueError(f"Event codes {unknown} are not in epochs.event_id.")

    coords: dict[str, Any] = {
        "trial": np.arange(n_trials),
        "channel": channel_names,
        "time": ("time", np.asarray(epochs.times), {"units": "s"}),
        label_coord: ("trial", np.array([names_by_code[code] for code in codes])),
    }
    metadata = getattr(epochs, "metadata", None)
    if metadata is not None:
        for column in metadata.columns:
            if column in coords:
                raise ValueError(f"epochs.metadata column '{column}' clashes with the '{column}' coordinate.")
            coords[str(column)] = ("trial", metadata[column].to_numpy())
    _add_trial_coords(coords, trial_coords, n_trials)

    data = xr.DataArray(
        values, dims=("trial", "channel", "time"), coords=coords, attrs={"sfreq": float(epochs.info["sfreq"])}
    )
    return DataContainer(data)


def from_numpy(
    X: Any,
    dims: tuple[str, ...] | None = None,
    *,
    coords: dict[str, Any] | None = None,
    metadata: Any = None,
    **trial_coords: Any,
) -> DataContainer:
    """Wrap a sample-first NumPy array, with per-sample labels and metadata, in a container.

    Args:
        X: Array of shape ``(n_samples, ...)``.
        dims: Names for every axis, starting with ``"trial"`` (the sample axis), e.g.
            ``("trial", "channel", "time")``. Defaults to ``("trial", "dim_1", "dim_2", ...)``.
        coords: Coordinates for the other dims, e.g. ``{"channel": ch_names, "time": times}``.
        metadata: Optional pandas DataFrame with one row per sample; each column becomes a trial coordinate.
        **trial_coords: Per-sample coordinates such as ``label=y`` or ``subject=subjects``, each a sequence with
            one value per sample or a scalar shared by all samples.

    Example:
        >>> data = from_numpy(X, ("trial", "channel", "time"), label=y, subject=subject_ids)
    """
    values = np.asarray(X)
    if values.ndim < 1:
        raise ValueError("X must have a sample axis.")
    if dims is None:
        dims = ("trial", *(f"dim_{axis}" for axis in range(1, values.ndim)))
    dims = tuple(dims)
    if len(dims) != values.ndim:
        raise ValueError(f"Got {len(dims)} dim names {dims} for an array with {values.ndim} axes {values.shape}.")
    if dims[0] != "trial":
        raise ValueError(f"The first axis holds the samples and must be named 'trial'; got dims {dims}.")
    n_trials = values.shape[0]

    all_coords: dict[str, Any] = {"trial": np.arange(n_trials)}
    for name, value in (coords or {}).items():
        if name == "trial" or name in all_coords:
            raise ValueError(
                f"Coordinate '{name}' clashes with an existing coordinate; pass per-sample values as keywords."
            )
        if name in dims and np.ndim(value) == 1 and len(value) != values.shape[dims.index(name)]:
            raise ValueError(
                f"Coordinate '{name}' has {len(value)} values for a dim of size {values.shape[dims.index(name)]}."
            )
        all_coords[name] = value
    if metadata is not None:
        if len(metadata) != n_trials:
            raise ValueError(f"metadata has {len(metadata)} rows for {n_trials} samples.")
        _add_trial_coords(
            all_coords, {str(column): metadata[column].to_numpy() for column in metadata.columns}, n_trials
        )
    _add_trial_coords(all_coords, trial_coords, n_trials)
    return DataContainer(xr.DataArray(values, dims=dims, coords=all_coords))


def _add_trial_coords(coords: dict[str, Any], trial_coords: dict[str, Any], n_trials: int) -> None:
    """Add per-trial coordinates (a scalar is repeated for every trial), refusing clashes and wrong lengths."""
    for name, value in trial_coords.items():
        if name in coords:
            raise ValueError(f"Trial coordinate '{name}' clashes with an existing coordinate.")
        per_trial = np.full(n_trials, value) if np.ndim(value) == 0 else np.asarray(value)
        if per_trial.shape != (n_trials,):
            raise ValueError(f"Trial coordinate '{name}' has shape {per_trial.shape}; expected ({n_trials},).")
        coords[name] = ("trial", per_trial)


def _as_container(
    X: Any, *, dims: tuple[str, ...] | None = None, metadata: Any = None, **trial_coords: Any
) -> DataContainer:
    """Accept a DataContainer, DataArray, MNE Epochs or sample-first NumPy array, plus optional per-trial
    metadata (a DataFrame) and per-trial coordinates. Coordinates given as None are skipped."""
    trial_coords = {name: value for name, value in trial_coords.items() if value is not None}
    if isinstance(X, (DataContainer, xr.DataArray)) or _is_mne_epochs(X):
        if dims is not None:
            raise ValueError(
                "dims only applies to NumPy input; DataArray, DataContainer and Epochs name their own dims."
            )
        data = X.data if isinstance(X, DataContainer) else X if isinstance(X, xr.DataArray) else from_mne_epochs(X).data
        own_trial_labels = True
    else:
        data = from_numpy(X, dims).data
        own_trial_labels = False
    if "trial" not in data.dims:
        raise ValueError(f"Data needs a 'trial' dim holding the samples; got dims {data.dims}.")
    new_coords: dict[str, Any] = {}
    if metadata is not None:
        if len(metadata) != data.sizes["trial"]:
            raise ValueError(f"metadata has {len(metadata)} rows for {data.sizes['trial']} trials.")
        # Rows are matched to trials by position, as in scikit-learn.
        # Metadata indexed by the data's own trial labels in another order could mean rows by label or rows by
        # position: refuse rather than guess.
        trials = data.indexes.get("trial") if own_trial_labels else None
        index = metadata.index
        # Labels must match one to one: a default index (0, 1, 2...) is positions, and intervals match points by
        # containment, so neither names trials.
        default = isinstance(index, pd.RangeIndex) and index.equals(pd.RangeIndex(len(index)))
        containment = isinstance(index, pd.IntervalIndex) and not isinstance(trials, pd.IntervalIndex)
        exact = not default and not containment and index.is_unique
        rows = index.get_indexer(trials) if trials is not None and exact and len(index) == len(trials) else None
        one_to_one = rows is not None and bool((rows >= 0).all()) and np.unique(rows).size == rows.size
        if one_to_one and not (rows == np.arange(rows.size)).all():
            raise ValueError(
                "metadata is indexed by the trial labels, in another order than the data's trials. Reorder it "
                "(metadata.loc[data.trial.values]), or drop its index (metadata.reset_index(drop=True)) if its rows "
                "already follow the data."
            )
        new_coords.update({str(column): metadata[column].to_numpy() for column in metadata.columns})
        given_twice = sorted(set(new_coords) & set(trial_coords))
        if given_twice:
            raise ValueError(f"{given_twice} are given both as metadata columns and as arguments; pass them once.")
    new_coords.update(trial_coords)
    clashes = sorted(set(new_coords) & set(data.coords))
    if clashes:
        raise ValueError(f"Coordinates {clashes} already exist in the data; refer to them by name instead.")
    resolved: dict[str, Any] = {}
    _add_trial_coords(resolved, new_coords, data.sizes["trial"])
    return DataContainer(data.assign_coords(resolved) if resolved else data)


def _is_mne_epochs(obj: Any) -> bool:
    return hasattr(obj, "get_data") and hasattr(obj, "event_id") and hasattr(obj, "ch_names")
