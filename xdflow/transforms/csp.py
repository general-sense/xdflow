"""Common spatial patterns (CSP), wrapping MNE's implementation."""

from typing import Any

import numpy as np
import xarray as xr

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer


def _mne_csp() -> Any:
    try:
        from mne.decoding import CSP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("CSPTransform requires MNE: pip install 'xdflow[mne]'") from exc
    return CSP


class CSPTransform(Transform):
    """Common spatial pattern features for two (or more) classes of trials, via ``mne.decoding.CSP``.

    CSP learns spatial filters from labelled trials, so it is stateful: validators refit it inside every
    training fold. The output has dims ``(trial, csp)``; by default each feature is the log-variance of one
    CSP component, the usual input to a linear classifier.
    """

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "csp")

    def __init__(
        self,
        n_components: int = 4,
        target_coord: str = "label",
        log: bool = True,
        reg: float | str | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        """Initialize the transform.

        Args:
            n_components: Number of CSP components to keep.
            target_coord: Trial coordinate holding the class labels used to fit the filters.
            log: Return log-variance features (``True``) as in ``mne.decoding.CSP``.
            reg: Covariance regularization passed to ``mne.decoding.CSP`` (e.g. ``"ledoit_wolf"``).
            sel: Optional selection applied before fitting or transforming.
            drop_sel: Optional drop selection applied before fitting or transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        _mne_csp()  # fail at construction, not mid-pipeline, if MNE is missing
        self.n_components = n_components
        self.target_coord = target_coord
        self.log = log
        self.reg = reg
        self._csp: Any = None

    def _fit(self, container: DataContainer, **kwargs) -> "CSPTransform":
        data = container.data.transpose("trial", "channel", "time")
        if self.target_coord not in data.coords:
            raise ValueError(f"Target coordinate '{self.target_coord}' not found in data coords.")
        self._csp = _mne_csp()(n_components=self.n_components, log=self.log, reg=self.reg)
        self._csp.fit(data.values, data[self.target_coord].values)
        self._fit_channels = data["channel"].values.copy()
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        if self._csp is None:
            raise ValueError("CSPTransform must be fitted before transform().")
        # Spatial filters are applied by position, so order channels as they were at fit (missing ones raise)
        data = container.data.sel(channel=self._fit_channels).transpose("trial", "channel", "time")
        trial_coords = {name: coord for name, coord in data.coords.items() if coord.dims == ("trial",)}
        features = xr.DataArray(
            self._csp.transform(data.values),
            dims=("trial", "csp"),
            coords={**trial_coords, "csp": np.arange(self.n_components)},
            attrs=data.attrs,
        )
        return DataContainer(features)
