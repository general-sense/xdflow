"""Converters from other libraries' data objects into xdflow containers."""

from typing import Any

import numpy as np
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
    for name, value in trial_coords.items():
        if name in coords:
            raise ValueError(f"Trial coordinate '{name}' clashes with an existing coordinate.")
        per_trial = np.full(n_trials, value) if np.ndim(value) == 0 else np.asarray(value)
        if per_trial.shape != (n_trials,):
            raise ValueError(f"Trial coordinate '{name}' has shape {per_trial.shape}; expected ({n_trials},).")
        coords[name] = ("trial", per_trial)

    data = xr.DataArray(
        values, dims=("trial", "channel", "time"), coords=coords, attrs={"sfreq": float(epochs.info["sfreq"])}
    )
    return DataContainer(data)
