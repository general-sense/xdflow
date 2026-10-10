"""Temporary input-position tags, so grouped outputs can be put back in input order.

Group-wise operations (GroupApplyTransform, domain aligners) concatenate group outputs one group after
another. Tagging each sample with its input position before grouping lets them restore input order even
when sample labels are missing or duplicated. Nested group-wise operations each use their own tag.
"""

import numpy as np
import xarray as xr

POSITION_TAG_PREFIX = "_xdflow_input_position"


def tag_positions(data: xr.DataArray, dim: str) -> tuple[xr.DataArray, str]:
    """Return `data` with a fresh position coordinate along `dim`, and that coordinate's name."""
    # One past the highest existing tag, so latest_tag() always finds the one added here
    latest = latest_tag(data)
    name = f"{POSITION_TAG_PREFIX}{int(latest[len(POSITION_TAG_PREFIX) :]) + 1 if latest is not None else 0}"
    return data.assign_coords({name: (dim, np.arange(data.sizes[dim]))}), name


def restore_positions(data: xr.DataArray, dim: str, name: str) -> xr.DataArray | None:
    """Reorder `data` along `dim` by the tag `name` and drop it; None if a step dropped the tag."""
    if name not in data.coords:
        return None
    order = np.argsort(data[name].values, kind="stable")
    return data.isel({dim: order}).drop_vars(name)


def latest_tag(data: xr.DataArray) -> str | None:
    """The innermost (most recently added) position tag on `data`, if any."""
    tags = [
        str(c)
        for c in data.coords
        if str(c).startswith(POSITION_TAG_PREFIX) and str(c)[len(POSITION_TAG_PREFIX) :].isdigit()
    ]
    return max(tags, key=lambda tag: int(tag[len(POSITION_TAG_PREFIX) :]), default=None)
