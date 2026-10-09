import inspect
from typing import Any

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from xdflow.core.data_container import DataContainer, TransformError
from xdflow.transforms.domain_adaptation import AdaptiveStrategy, AdaptiveTransform


class AdaptWrapperStrategy(AdaptiveStrategy):
    """
    An adaptation strategy for adapt classes from the adapt package. This is used with the AdaptWrapperTransform.
    During adaptation, it fits the one source domain and one target domain.
    During transformation, it transforms the source and target domains according to the adapt class.
    Domains are determined by the group_coord.
    """

    def __init__(
        self,
        group_coord: str,
        target_group: str | int | float,
        n_jobs: int = 1,
        adapt_sel: dict[str, Any] | None = None,
    ):
        """
        Initialize the AdaptWrapperStrategy.

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
        self.seen_target_groups_ = []
        self.seen_groups_ = []

    def _adapt(self, aligner: "AdaptWrapperTransform", container: DataContainer, **kwargs) -> None:
        """
        Adapt the data using a single-source strategy.
        Fit and adapt the target and source domains according to the adapt class.
        Domains are determined by the group_coord.

        Args:
            aligner: The AdaptWrapperAligner instance using this strategy.
            container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.
        """

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        all_groups = self._discover_groups(container)
        self.seen_source_groups_ = [g for g in all_groups if g != self.target_group]
        self.seen_groups_ = all_groups

        # TODO: handle more domains for certain classes
        if len(all_groups) != 2:
            raise ValueError("AdaptWrapperStrategy requires exactly two groups for adaptation.")

        if self.target_group not in all_groups:
            raise ValueError(f"Target group '{self.target_group}' not found in data")

        if not self.seen_source_groups_:
            raise ValueError("No source group found in the provided container for adaptation.")

        self.source_group = self.seen_source_groups_[0]  # should only be one source group

        # Convert data to right format
        source_container = self._select_group(container, self.source_group)
        target_container = self._select_group(container, self.target_group)

        aligner._fit_adapt(source_container, target_container, **kwargs)

    def transform(self, aligner: "AdaptWrapperTransform", container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data by applying the appropriate source or adapted model to each group.

        Args:
            aligner: The AdaptWrapperAligner instance using this strategy.
            container: The DataContainer to be transformed.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        container = self._tag_positions(container)
        current_groups = self._discover_groups(container)

        def transform_group(group_val):
            group_container = self._select_group(container, group_val)
            if group_val == self.source_group:
                return aligner._adapted_transform(group_container, domain="source", **kwargs)
            elif group_val == self.target_group:
                return aligner._adapted_transform(group_container, domain="target", **kwargs)
            else:  # Unseen target group
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

        # Reassemble outputs in input order
        reassembled = xr.concat(group_outputs, dim=self.group_dim)
        return DataContainer(self._restore_input_order(reassembled, container))


class AdaptWrapperTransform(AdaptiveTransform):
    is_stateful: bool = True
    _supports_transform_sel: bool = False

    _STRATEGY_MAP = {"adapt_wrapper": AdaptWrapperStrategy}

    def __init__(
        self,
        adapt_estimator_cls,
        sample_dim: str,
        target_coord: str,
        group_coord: str,
        target_group: str | int | float,
        random_state: int = 0,
        adapt_sel: dict[str, Any] | None = None,
        n_jobs: int = 1,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            strategy_name="adapt_wrapper",
            sample_dim=sample_dim,
            # the strategy's arguments, stored as attributes by AdaptiveTransform
            group_coord=group_coord,
            target_group=target_group,
            adapt_sel=adapt_sel,
            n_jobs=n_jobs,
            sel=sel,
            drop_sel=drop_sel,
        )

        self.target_coord = target_coord
        self.adapt_estimator_cls = adapt_estimator_cls  # needed for clone
        self.random_state = random_state

        # The remaining keyword arguments are the estimator's; it is built at fit from them.
        self._adapt_estimator_kwargs = dict(kwargs)
        self.adapt_estimator: Any = None  # built by _fit_adapt, from the current arguments

    @property
    def _specifies_transform_domain(self) -> bool:
        """Whether the estimator's `fit` takes a `domain`, read from its class."""
        return "domain" in inspect.signature(self.adapt_estimator_cls.fit).parameters

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Include the adapt estimator's arguments, which clone() passes back to the constructor."""
        return {**super().get_params(deep=deep), **self._adapt_estimator_kwargs}

    def set_params(self, **params: Any) -> "AdaptWrapperTransform":
        """Route adapt-estimator arguments into the dict the estimator is built from at fit.

        A flat name that is not one of this transform's own attributes belongs to the estimator. Writes into
        the fitted `adapt_estimator` are refused: the next fit rebuilds it from the dict, so they would be lost.
        """
        if any(key.split("__", 1)[0] == "adapt_estimator" for key in params):
            raise ValueError("Set adapt estimator arguments by their own name, not through 'adapt_estimator__'.")
        estimator_params = {k: v for k, v in params.items() if "__" not in k and k not in vars(self)}
        self._adapt_estimator_kwargs = {**self._adapt_estimator_kwargs, **estimator_params}
        super().set_params(**{k: v for k, v in params.items() if k not in estimator_params})
        return self

    def _fit_adapt(self, source_container: DataContainer, target_container: DataContainer, **kwargs) -> None:
        # get source data
        source_data = self._prepare_data(source_container.data)
        source_input = source_data.values
        source_output = source_data.coords[self.target_coord].values

        assert source_data.ndim == 2, "AdaptWrapperTransform requires 2D data"

        # get target data
        target_data = self._prepare_data(target_container.data)
        target_input = target_data.values
        target_output = target_data.coords[self.target_coord].values

        # encoder for outputs, only needed for adapt estimators' fit method, does not affect transform
        encoder = LabelEncoder()
        all_outputs = np.concatenate([source_output, target_output])
        encoder.fit(all_outputs)
        source_output = encoder.transform(source_output)
        target_output = encoder.transform(target_output)

        # Built here, from the current arguments, so a set_params since construction is what gets fitted.
        estimator_kwargs = dict(self._adapt_estimator_kwargs)
        if "random_state" in inspect.signature(self.adapt_estimator_cls).parameters:
            estimator_kwargs["random_state"] = self.random_state
        self.adapt_estimator = self.adapt_estimator_cls(**estimator_kwargs)

        self.adapt_estimator.fit(X=source_input, y=source_output, Xt=target_input, yt=target_output)

    def _adapted_transform(self, data_container: DataContainer, domain: str, **kwargs) -> DataContainer:
        original_dims = data_container.data.dims
        data = self._prepare_data(data_container.data)

        data_input = data.values

        if self._specifies_transform_domain:
            assert domain in ["source", "target"], "Domain must be either 'source' or 'target'"
            transformed_values = self.adapt_estimator.transform(X=data_input, domain=domain, **kwargs)
        else:
            transformed_values = self.adapt_estimator.transform(X=data_input, **kwargs)

        output_coords = self._get_output_coords(data)
        output_dim_name = [dim for dim in data.dims if dim != self.sample_dim][0]  # should only be one
        output_coords[output_dim_name] = np.arange(transformed_values.shape[1])

        transformed_da = xr.DataArray(transformed_values, dims=data.dims, coords=output_coords, attrs=data.attrs)
        transformed_da = transformed_da.transpose(*original_dims)

        return DataContainer(transformed_da)

    def _get_clone_kwargs(self) -> dict:
        """Include adapt estimator parameters in clone kwargs."""
        return {**super()._get_clone_kwargs(), **self._adapt_estimator_kwargs}

    def _get_output_coords(self, data: xr.DataArray) -> dict[str, Any]:
        """
        Gets the output coordinates for the predictor.
        """
        output_coords = {self.sample_dim: data.coords[self.sample_dim].values}

        for coord_name, coord_data in data.coords.items():
            if (coord_name != self.sample_dim) and (self.sample_dim in coord_data.dims):
                output_coords[coord_name] = coord_data

        return output_coords
