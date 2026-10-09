import warnings
from typing import Any

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.sampling import get_container_by_conditions


def _normalize_per_dim(per_dim: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return (per_dim,) if isinstance(per_dim, str) else tuple(per_dim)


def _stat_dims(per_dim: str | list[str] | tuple[str, ...], dims: tuple) -> list:
    """Dims to compute statistics over: all but `per_dim`, which must exist (a misspelled or renamed dim would
    otherwise silently pool statistics over everything)."""
    per_dims = _normalize_per_dim(per_dim)
    missing = [d for d in per_dims if d not in dims]
    if missing:
        raise ValueError(f"per_dim {missing} not found in data dims {tuple(dims)}.")
    return [d for d in dims if d not in per_dims]


def _warn_if_pooling_trials(transform: object, mean_dims: list) -> None:
    """Statistics pooled across trials without use_fit are computed before CV splits, leaking validation data."""
    if "trial" in mean_dims:
        warnings.warn(
            f"{type(transform).__name__} with use_fit=False computes statistics across trials from the data being "
            "transformed; inside cross-validation this includes validation trials. Set use_fit=True to learn them "
            "from training folds, or include 'trial' in per_dim to normalize each trial independently.",
            stacklevel=3,
        )


class DemeanTransform(Transform):
    """Subtract a mean independently per selected dimension label.

    `per_dim` names the dimensions whose labels remain distinct while the mean
    is computed over all other dimensions. For data with dimensions `("trial",
    "channel", "time")`, `per_dim="channel"` subtracts a separate channel mean
    computed across trials and time. Unlike xarray's `dim=`, `per_dim` does
    not name dimensions to reduce. When multiple dimensions are provided,
    statistics are computed separately for each coordinate tuple across those dimensions.

    By default the mean is computed from the data being transformed. With
    `use_fit=True`, the mean is learned during `fit` and reused during
    `transform`, which is the usual choice inside cross-validation.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        per_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        """
        Initializes the DemeanTransform.

        Args:
            per_dim: Dimension or dimensions to keep distinct while computing the mean.
            use_fit: Whether to use the fit data to compute the mean.
            fit_sel: A dictionary to select a subset of data for fitting.
                Useful if you want to demean in reference to a subset of the data.
                If specified, the transform fits even when use_fit is False.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.per_dim = per_dim
        self.use_fit = use_fit
        self.fit_sel = fit_sel

        if self.fit_sel is not None and not self.use_fit:
            warnings.warn("fit_sel is specified but use_fit is False; fitting on fit_sel anyway.")

    @property
    def is_stateful(self) -> bool:
        """Fits when `use_fit` is set or a `fit_sel` is given; derived on read so `set_params` reaches it."""
        return self.use_fit or self.fit_sel is not None

    def _fit(self, data_container: DataContainer, **kwargs) -> "DemeanTransform":
        """
        Fit the transform to the data.
        """
        if self.is_stateful:
            if self.fit_sel is not None:
                data_container = get_container_by_conditions(data_container, self.fit_sel)

            mean_dims = _stat_dims(self.per_dim, data_container.data.dims)
            self.mean = data_container.data.mean(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the demean transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with the transformed data.
        """
        data = data_container.data
        if self.is_stateful:
            transformed_data = data - self.mean
        else:
            mean_dims = _stat_dims(self.per_dim, data.dims)
            _warn_if_pooling_trials(self, mean_dims)
            transformed_data = data - data.mean(dim=mean_dims)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims


class ZScoreTransform(Transform):
    """Apply z-score normalization independently per selected dimension label.

    `per_dim` names the dimensions whose labels remain distinct while the mean
    and standard deviation are computed over all other dimensions. For data with
    dimensions `("trial", "channel", "time")`, `per_dim="channel"`
    normalizes each channel using statistics computed across trials and time.
    Unlike xarray's `dim=`, `per_dim` does not name dimensions to reduce. When
    multiple dimensions are provided, statistics are computed separately for
    each coordinate tuple across those dimensions.

    By default statistics are computed from the data being transformed. With
    `use_fit=True`, statistics are learned during `fit` and reused during
    `transform`, which avoids validation leakage inside cross-validation.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        per_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        """Initialize a z-score transform.

        Args:
            per_dim: Dimension or dimensions to keep distinct while computing the mean and std.
            use_fit: Whether to use the fit data to compute the mean and std.
            fit_sel: A dictionary to select a subset of data for fitting.
                Useful if you want to zscore in reference to a subset of the data.
                If specified, the transform fits even when use_fit is False.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.per_dim = per_dim
        self.use_fit = use_fit
        self.fit_sel = fit_sel

        if self.fit_sel is not None and not self.use_fit:
            warnings.warn("fit_sel is specified but use_fit is False; fitting on fit_sel anyway.")

    @property
    def is_stateful(self) -> bool:
        """Fits when `use_fit` is set or a `fit_sel` is given; derived on read so `set_params` reaches it."""
        return self.use_fit or self.fit_sel is not None

    def _fit(self, data_container: DataContainer, **kwargs) -> "ZScoreTransform":
        """
        Fit the transform to the data.
        """
        if self.is_stateful:
            if self.fit_sel is not None:
                data_container = get_container_by_conditions(data_container, self.fit_sel)

            mean_dims = _stat_dims(self.per_dim, data_container.data.dims)
            self.mean = data_container.data.mean(dim=mean_dims)
            self.std = data_container.data.std(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the Z-score transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with the transformed data.
        """
        data = data_container.data
        if self.is_stateful:
            mean = self.mean
            std = self.std
        else:
            mean_dims = _stat_dims(self.per_dim, data.dims)
            _warn_if_pooling_trials(self, mean_dims)
            mean = data.mean(dim=mean_dims)
            std = data.std(dim=mean_dims)
        transformed_data = (data - mean) / std
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
