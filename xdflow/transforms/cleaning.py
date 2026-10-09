from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.transforms.normalization import _stat_dims, _warn_if_pooling_trials


def _signal_channel_mask(channel_values: np.ndarray, excluded_channels: Sequence[str] | None) -> np.ndarray:
    """Return a boolean mask selecting signal channels (i.e., not excluded)."""
    if not excluded_channels:
        return np.ones(len(channel_values), dtype=bool)
    return ~np.isin(channel_values, np.asarray(excluded_channels, dtype=object))


def _normalize_per_dim(per_dim: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return (per_dim,) if isinstance(per_dim, str) else tuple(per_dim)


class CARTransform(Transform):
    """
    Apply Common Average Referencing (CAR) to the data.

    CAR can be applied to all signal channels or disabled.

    Args:
        car_method: 'all' or 'none'.
        excluded_channels: Channels to leave untouched (e.g., reference sensors).
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")

    def __init__(
        self,
        car_method: str = "all",
        excluded_channels: Sequence[str] | None = None,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.car_method = car_method
        self.excluded_channels = excluded_channels

    @property
    def _excluded_channels(self) -> tuple[str, ...]:
        return tuple(self.excluded_channels or ())

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the CAR transformation.
        """
        # This transform modifies the data in-place, so we need a deep copy.
        data = data_container.data.copy(deep=True)

        channel_values = np.asarray(data.coords["channel"].values, dtype=object)
        lfp_mask = _signal_channel_mask(channel_values, self._excluded_channels)
        lfp_channels = channel_values[lfp_mask]

        if self.car_method is None or self.car_method == "none":
            return DataContainer(data)

        elif self.car_method == "all":
            mean_all = data.sel(channel=lfp_channels).mean(dim="channel")
            data.loc[{"channel": lfp_channels}] -= mean_all
        else:
            raise ValueError(f"CAR method '{self.car_method}' not recognized. Must be one of 'all' or 'none'.")

        return DataContainer(data)


class RegressOutReferenceTransform(Transform):
    """
    Regress out a reference channel from all other channels.

    Uses linear regression to model the relationship between the reference signal
    and each target channel, then subtracts the predicted component.

    By default the regression is fit on the data being transformed, pooling all trials. With
    `use_fit=True` it is learned during `fit` and reused during `transform`, which keeps validation
    trials out of the fit inside cross-validation.
    """

    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")

    def __init__(
        self,
        reference_channel: str,
        excluded_channels: Sequence[str] | None = None,
        use_fit: bool = False,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        if not reference_channel:
            raise ValueError("reference_channel must be provided.")
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.reference_channel = reference_channel
        self.excluded_channels = excluded_channels
        self.use_fit = use_fit

    @property
    def is_stateful(self) -> bool:  # type: ignore[override]
        return self.use_fit

    @property
    def _excluded_channels(self) -> tuple[str, ...]:
        return tuple(set(self.excluded_channels or ()) | {self.reference_channel})

    def _fit_models(self, data: xr.DataArray) -> dict[Any, LinearRegression]:
        """One reference->channel regression per target channel, pooling trials and time."""
        if self.reference_channel not in data.coords["channel"]:
            raise ValueError(f"Reference channel '{self.reference_channel}' not present in data.")
        ref_signal_flat = data.sel(channel=self.reference_channel).values.reshape(-1, 1)
        channel_values = np.asarray(data.coords["channel"].values)
        target_channels = channel_values[_signal_channel_mask(channel_values, self._excluded_channels)]
        return {
            chan: LinearRegression().fit(ref_signal_flat, data.sel(channel=chan).values.ravel())
            for chan in target_channels
        }

    def _fit(self, data_container: DataContainer, **kwargs) -> "RegressOutReferenceTransform":
        self._models = self._fit_models(data_container.data)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """Apply the reference regression."""
        data = data_container.data.copy(deep=True)
        if self.use_fit:
            models = self._models
            channel_values = np.asarray(data.coords["channel"].values)
            targets = channel_values[_signal_channel_mask(channel_values, self._excluded_channels)]
            unfitted = [chan for chan in targets if chan not in models]
            if unfitted:
                raise ValueError(
                    f"No reference regression was fitted for channel(s) {unfitted}; refit on these channels."
                )
        else:
            _warn_if_pooling_trials(self, ["trial"])
            models = self._fit_models(data)

        ref_signal_flat = data.sel(channel=self.reference_channel).values.reshape(-1, 1)
        for chan, model in models.items():
            lfp_signal_da = data.sel(channel=chan)
            data.loc[{"channel": chan}] -= model.predict(ref_signal_flat).reshape(lfp_signal_da.shape)
        return DataContainer(data)


class RemoveOutliersTransform(Transform):
    """
    Remove outliers from the data by clipping independently per selected dimension label.

    Outliers are identified as values exceeding a specified number of standard
    deviations from the mean. They are replaced by the boundary value.

    Args:
        per_dim: Dimension or dimensions to keep distinct while computing stats.
        std_threshold (float): The number of standard deviations to use as the
            threshold. Defaults to 5.0.
        use_fit (bool): Whether to use the fit data to compute the mean and std.
            Defaults to False.
        transform_sel (dict, optional): A dictionary to select a subset of data
            for transformation, leaving the rest untouched.
        transform_drop_sel (dict, optional): A dictionary to select a subset of
            data for transformation by excluding labels, leaving the rest untouched.

    E.g. if your input dims are ("trial", "channel", "time"), and you set per_dim to "channel",
    then the data will be clipped per channel by clipping the data to the std_threshold from the mean of trial and time per channel.
    Unlike xarray's `dim=`, `per_dim` does not name dimensions to reduce.
    When multiple dimensions are provided, clipping bounds are computed separately for each coordinate tuple across those dimensions.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    @property
    def is_stateful(self) -> bool:  # type: ignore[override]
        # Derived from use_fit so set_params(use_fit=...) and clone() stay consistent.
        return self.use_fit

    def __init__(
        self,
        per_dim: str | list[str] | tuple[str, ...],
        std_threshold: float = 5.0,
        use_fit: bool = False,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
        transform_sel: dict[str, object] | None = None,
        transform_drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.std_threshold = std_threshold
        self.per_dim = per_dim
        self.use_fit = use_fit

    def _fit(self, data_container: DataContainer, **kwargs) -> "RemoveOutliersTransform":
        """
        Fit the transform to the data.

        Note: This method receives a pre-sliced DataContainer if transform_sel
        or transform_drop_sel was used, so it only needs to compute stats on
        the data it is given.
        """
        if self.use_fit:
            mean_dims = _stat_dims(self.per_dim, data_container.data.dims)
            self.mean = data_container.data.mean(dim=mean_dims)
            self.std = data_container.data.std(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the outlier removal transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with outliers clipped.
        """
        data = data_container.data
        std_threshold = self.std_threshold

        if self.use_fit:
            mean = self.mean
            std = self.std
        else:
            mean_dims = _stat_dims(self.per_dim, data.dims)
            _warn_if_pooling_trials(self, mean_dims)
            mean = data.mean(dim=mean_dims)
            std = data.std(dim=mean_dims)
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        transformed_data = data.clip(lower_bound, upper_bound)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
