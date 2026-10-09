import warnings
from collections.abc import Hashable
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from xdflow.composite.base import CompositeTransform
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.utils.positions import latest_tag, restore_positions, tag_positions


def _final_predictor(transform: Transform) -> Predictor | None:
    """Return the predictor that makes `transform`'s predictions: itself, or a composite's predictive step."""
    if isinstance(transform, CompositeTransform):
        predictive = transform.predictive_transform
        return predictive if isinstance(predictive, Predictor) else None
    return transform if isinstance(transform, Predictor) else None


def _supports_proba(predictor: Predictor) -> bool:
    """Whether a fitted predictor can produce class probabilities."""
    estimator = getattr(predictor, "estimator", None)
    if estimator is not None:
        return hasattr(estimator, "predict_proba")
    return type(predictor)._predict_proba is not Predictor._predict_proba


class GroupApplyTransform(CompositeTransform):
    """
    Applies a transform individually to each group defined by a metadata coordinate.

    This transform discovers groups from the data at fit time, creates independent transform
    instances per group by cloning the template, and applies transformations per group.
    The outputs are reassembled along the original grouped axis.

    Use cases:
    - Apply per-animal preprocessing where each animal needs independent fitting
    - Train separate models per session or experimental condition
    - Any scenario where groups should be processed independently

    Args:
        group_coord: Coordinate name to use for grouping (e.g., "animal", "session")
        transform: Template transform to clone per group (unfitted)
        unseen_policy: How to handle groups not seen during fit:
            - "error": raise TransformError (default)
            - "average": uniform average across all fitted group transforms
            - "weighted_average": weighted average by training counts per group
        unequal_output_dims_strategy: How to handle unequal (non-group) output dimensions across groups:
            (unequal output dims lead to NaNs during concatenation)
            - "error": raise TransformError (default)
            - "cut_to_min": use the min size per dimension across groups
        n_jobs: Number of parallel jobs for per-group processing
    """

    def __init__(
        self,
        group_coord: str | list[str],
        transform_template: Transform,  # needs same name as param for cloning
        unseen_policy: Literal["error", "average", "weighted_average"] = "error",
        unequal_output_dims_strategy: Literal["error", "cut_to_min"] = "error",
        n_jobs: int = 1,
    ):
        """Initialize GroupApplyTransform with grouping parameters."""

        self.group_coord = group_coord if isinstance(group_coord, list) else [group_coord]
        self.transform_template = transform_template
        self.unseen_policy = unseen_policy
        self.unequal_output_dims_strategy = unequal_output_dims_strategy
        self.n_jobs = n_jobs

        # State set during fitting
        self.seen_groups: list[Hashable] = []
        self.per_group_fitted: dict[Hashable, Transform] = {}
        self.train_counts: dict[Hashable, int] = {}
        self.group_dim: str | None = None

        # Compute input/output dimensions from template
        self.input_dims = self.transform_template.input_dims
        self.output_dims = self.transform_template.output_dims

        # Compute max size per dimension to avoid nans during concatenation
        self.max_size_per_dim = {}  # used for equalizing output dimensions

        super().__init__()
        self._validate_composition()

    @property
    def transform_from_name(self) -> dict[str, Transform]:
        # Derived on read so set_params(transform_template=...) is followed by nested set_params
        return {"transform_template": self.transform_template}

    @property
    def children(self) -> list[Transform]:
        """Returns fitted per-group transforms after fitting."""
        return list(self.per_group_fitted.values())

    @property
    def is_stateful(self) -> bool:  # type: ignore[override]
        # Per-group children only exist after fitting, so follow the template
        return self.transform_template.is_stateful

    @property
    def combined_group_coord(self) -> str:
        return "_".join(self.group_coord)

    def set_params(self, **params: Any) -> "GroupApplyTransform":
        if isinstance(params.get("group_coord"), str):
            params = {**params, "group_coord": [params["group_coord"]]}
        return cast("GroupApplyTransform", super().set_params(**params))

    @property
    def is_predictor(self) -> bool:
        """Returns True if the template transform performs prediction."""
        template_transform = self.transform_template
        if isinstance(template_transform, CompositeTransform):
            return template_transform.is_predictor
        else:
            return isinstance(template_transform, Predictor)

    @property
    def predictive_transform(self) -> Transform | None:
        """Returns the predictive transform if it exists, otherwise None."""

        if not self.is_predictor:
            return None

        template_transform = self.transform_template
        if isinstance(template_transform, CompositeTransform):
            return template_transform.predictive_transform
        else:
            return template_transform

    def _validate_composition(self):
        """Validates the transform template."""
        # No validation needed since we only have one template transform
        pass

    def _get_group_dim(self, container: DataContainer) -> str:
        """Resolves the dimension that the group_coord indexes."""
        for coord in self.group_coord:
            if coord not in container.data.coords:
                raise ValueError(f"Group coordinate '{coord}' not found in data coordinates")

        group_coord_dims = []
        for coord in self.group_coord:
            coord_dims = container.data.coords[coord].dims
            if len(coord_dims) != 1:
                raise ValueError(
                    f"Group coordinate '{coord}' must index exactly one dimension, "
                    f"but it indexes {len(coord_dims)}: {coord_dims}"
                )
            group_coord_dims.append(coord_dims[0])

        # all group coord dims must be the same
        if len(set(group_coord_dims)) != 1:
            raise ValueError(
                f"Group coordinates {self.group_coord} must index the same dimension, "
                f"but they index different dimensions: {group_coord_dims}"
            )

        return group_coord_dims[0]

    def _set_combined_group_coord_values(self, container: DataContainer) -> DataContainer:
        """Sets the combined group coordinate and tags each sample with its input position."""
        data, _ = tag_positions(container.data, cast(str, self.group_dim))

        if len(self.group_coord) == 1:
            return DataContainer(data)

        # Use zip for robust, element-wise string joining of coordinate values.
        coord_values = [data.coords[c].values for c in self.group_coord]
        rows = list(zip(*coord_values, strict=True))
        combined_coords = ["_".join(map(str, row)) for row in rows]
        if len(set(combined_coords)) != len(set(rows)):
            raise ValueError(
                f"Group coordinates {self.group_coord} produce ambiguous combined keys when joined with '_' "
                "(e.g. ('a_b', 'c') and ('a', 'b_c')); rename coordinate values to avoid merging groups."
            )

        data = data.assign_coords({self.combined_group_coord: (self.group_dim, combined_coords)})

        return DataContainer(data)

    def _discover_groups(self, container: DataContainer) -> list[Hashable]:
        """Discovers unique group values from the data."""
        group_values = container.data.coords[self.combined_group_coord].values
        return sorted(np.unique(group_values).tolist())

    def _select_group(self, container: DataContainer, group_val: Hashable) -> DataContainer:
        """Selects data for a specific group using boolean indexing."""
        # Positional boolean indexing; `where(drop=True)` would cast int data to float
        group_mask = container.data.coords[self.combined_group_coord].values == group_val
        return DataContainer(container.data.isel({self.group_dim: group_mask}))

    def _validate_group_output_preserves_axis(
        self, group_val: Hashable, input_container: DataContainer, output_container: DataContainer
    ):
        """Validates that the transform preserved the grouped axis."""
        if self.group_dim not in output_container.data.dims:
            raise TransformError(
                f"Transform for group '{group_val}' removed the grouped dimension '{self.group_dim}'. "
                f"GroupApplyTransform requires that inner transforms preserve the grouped axis."
            )

        # Check that the group coordinates are still present and have the same values for this group
        for coord in self.group_coord:
            if coord not in output_container.data.coords:
                raise TransformError(f"Transform for group '{group_val}' removed the group coordinate '{coord}'")

    def fit(self, container: DataContainer, **kwargs) -> "GroupApplyTransform":
        """
        Fits per-group transforms after discovering groups from the data.

        Args:
            container: DataContainer to fit on
            **kwargs: Additional context/parameters passed through

        Returns:
            Self (fitted GroupApplyTransform)
        """

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        container = self._set_combined_group_coord_values(container)
        self.seen_groups = self._discover_groups(container)

        # Reset state
        self.per_group_fitted = {}
        self.train_counts = {}

        # Fit each group
        if self.n_jobs != 1:
            # Parallel fitting
            def fit_group(group_val):
                group_container = self._select_group(container, group_val)
                transform = self.transform_template.clone()
                fitted_transform = transform.fit(group_container, **kwargs)
                train_count = group_container.data.sizes[self.group_dim]
                return group_val, fitted_transform, train_count

            results = Parallel(n_jobs=self.n_jobs)(delayed(fit_group)(group_val) for group_val in self.seen_groups)

            for group_val, fitted_transform, train_count in results:
                self.per_group_fitted[group_val] = fitted_transform
                self.train_counts[group_val] = train_count
        else:
            # Sequential fitting
            for group_val in self.seen_groups:
                group_container = self._select_group(container, group_val)
                transform = self.transform_template.clone()
                self.per_group_fitted[group_val] = transform.fit(group_container, **kwargs)
                self.train_counts[group_val] = group_container.data.sizes[self.group_dim]

        return self

    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Fits and transforms in a single pass for efficiency.

        Args:
            container: DataContainer to fit and transform
            **kwargs: Additional context/parameters passed through

        Returns:
            Transformed DataContainer with results reassembled
        """

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        container = self._set_combined_group_coord_values(container)
        self.seen_groups = self._discover_groups(container)

        # Reset state
        self.per_group_fitted = {}
        self.train_counts = {}

        # Fit and transform each group
        if self.n_jobs != 1:
            # Parallel fit_transform
            def fit_transform_group(group_val):
                group_container = self._select_group(container, group_val)
                transform = self.transform_template.clone()
                transformed_container = transform.fit_transform(group_container, **kwargs)
                train_count = group_container.data.sizes[self.group_dim]
                return group_val, transform, transformed_container, train_count

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_transform_group)(group_val) for group_val in self.seen_groups
            )

            group_outputs = []
            for group_val, fitted_transform, transformed_container, train_count in results:
                self.per_group_fitted[group_val] = fitted_transform
                self.train_counts[group_val] = train_count

                # Validate output preserves grouped axis
                group_input = self._select_group(container, group_val)
                self._validate_group_output_preserves_axis(group_val, group_input, transformed_container)

                group_outputs.append(transformed_container.data)
        else:
            # Sequential fit_transform
            group_outputs = []
            for group_val in self.seen_groups:
                group_container = self._select_group(container, group_val)
                transform = self.transform_template.clone()
                transformed_container = transform.fit_transform(group_container, **kwargs)

                # Store fitted transform and count
                self.per_group_fitted[group_val] = transform
                self.train_counts[group_val] = group_container.data.sizes[self.group_dim]

                # Validate output preserves grouped axis
                self._validate_group_output_preserves_axis(group_val, group_container, transformed_container)

                group_outputs.append(transformed_container.data)

        # Reassemble outputs along the grouped dimension
        reassembled = xr.concat(group_outputs, dim=self.group_dim, join="outer")
        reassembled = self._handle_unequal_output_dims(reassembled, group_outputs, fitted=False)

        reassembled = self._restore_input_order(reassembled, container)

        # remove the combined group coord
        if len(self.group_coord) > 1:
            reassembled = reassembled.drop_vars(self.combined_group_coord)

        return DataContainer(reassembled)

    def _apply_to_unseen_group(self, group_container: DataContainer, group_val: Hashable, **kwargs) -> DataContainer:
        """
        Applies the unseen group policy to handle a group not seen during fit.

        Args:
            group_container: DataContainer for the unseen group
            group_val: The unseen group value
            **kwargs: Additional context/parameters

        Returns:
            DataContainer with policy-based result
        """
        if self.unseen_policy == "error":
            raise TransformError(f"Group '{group_val}' was not seen during fit. Seen groups: {self.seen_groups}")

        if not self.per_group_fitted:
            raise TransformError("No fitted transforms available for unseen group averaging")

        # Apply all fitted transforms to this group's data
        if self.n_jobs != 1:
            # Parallel application
            def apply_transform(fitted_transform):
                return fitted_transform.transform(group_container, **kwargs)

            group_outputs = Parallel(n_jobs=self.n_jobs)(
                delayed(apply_transform)(fitted_transform) for fitted_transform in self.per_group_fitted.values()
            )
        else:
            # Sequential application
            group_outputs = []
            for fitted_transform in self.per_group_fitted.values():
                group_outputs.append(fitted_transform.transform(group_container, **kwargs))

        # Validate all outputs have identical dims (sizes along labelled dims, e.g. classes, may differ)
        ref_output = group_outputs[0]
        for i, output in enumerate(group_outputs[1:], 1):
            if output.data.dims != ref_output.data.dims:
                raise TransformError(
                    f"Inconsistent output dimensions for unseen group averaging. "
                    f"Transform {i} has dims {output.data.dims} but reference has {ref_output.data.dims}"
                )

        outputs = {g: output.data for g, output in zip(self.per_group_fitted, group_outputs, strict=True)}
        final_predictor = _final_predictor(next(iter(self.per_group_fitted.values())))
        if (
            final_predictor is not None
            and final_predictor.is_classifier
            and not final_predictor.is_multilabel
            and not final_predictor.proba
        ):
            # Hard class labels cannot be averaged
            return DataContainer(self._vote_unseen(outputs))
        return DataContainer(self._average_unseen(outputs))

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data by applying per-group transforms and reassembling.

        Args:
            container: DataContainer to transform
            **kwargs: Additional context/parameters

        Returns:
            Transformed DataContainer
        """
        if not self.per_group_fitted:
            raise TransformError("GroupApplyTransform must be fitted before transform")

        # set the combined group coord values and discover groups
        container = self._set_combined_group_coord_values(container)
        current_groups = self._discover_groups(container)

        # Separate seen and unseen groups
        seen_in_current = [g for g in current_groups if g in self.seen_groups]
        unseen_in_current = [g for g in current_groups if g not in self.seen_groups]

        group_outputs = []

        # Process seen groups
        if seen_in_current:
            if self.n_jobs != 1:
                # Parallel processing of seen groups
                def transform_seen_group(group_val):
                    group_container = self._select_group(container, group_val)
                    fitted_transform = self.per_group_fitted[group_val]
                    return fitted_transform.transform(group_container, **kwargs)

                seen_outputs = Parallel(n_jobs=self.n_jobs)(
                    delayed(transform_seen_group)(group_val) for group_val in seen_in_current
                )
                group_outputs.extend([output.data for output in seen_outputs])
            else:
                # Sequential processing of seen groups
                for group_val in seen_in_current:
                    group_container = self._select_group(container, group_val)
                    fitted_transform = self.per_group_fitted[group_val]
                    transformed = fitted_transform.transform(group_container, **kwargs)
                    group_outputs.append(transformed.data)

        # Process unseen groups
        for group_val in unseen_in_current:
            group_container = self._select_group(container, group_val)
            averaged_result = self._apply_to_unseen_group(group_container, group_val, **kwargs)
            group_outputs.append(averaged_result.data)

        # Reassemble in the original group order
        if not group_outputs:
            raise TransformError("No groups to process")

        reassembled = xr.concat(group_outputs, dim=self.group_dim, join="outer")
        reassembled = self._handle_unequal_output_dims(reassembled, group_outputs, fitted=True)

        reassembled = self._restore_input_order(reassembled, container)

        # remove the combined group coord
        if len(self.group_coord) > 1:
            reassembled = reassembled.drop_vars(self.combined_group_coord)

        return DataContainer(reassembled)

    def _restore_input_order(self, reassembled: xr.DataArray, container: DataContainer) -> xr.DataArray:
        """Put samples back in input order; groups are concatenated one after another, which would
        otherwise misalign positional consumers (e.g. ensembles, label arrays) with the input."""
        tag = latest_tag(container.data)
        restored = restore_positions(reassembled, cast(str, self.group_dim), tag) if tag is not None else None
        if restored is not None:
            return restored
        # The group transform dropped the position tag; fall back to (unique) sample labels
        input_labels = pd.Index(container.data[self.group_dim].values)
        return reassembled.sel({self.group_dim: input_labels[input_labels.isin(reassembled[self.group_dim].values)]})

    def _average_unseen(self, outputs: dict[Hashable, xr.DataArray]) -> xr.DataArray:
        """Average every group model's output for an unseen group, per `unseen_policy`.

        Class coordinates are outer-joined: a class one group model never saw gets probability 0
        instead of being dropped by xarray's default inner join.
        """
        groups = list(outputs)
        aligned = xr.align(*outputs.values(), join="outer", fill_value=0.0)
        total = sum(self.train_counts.get(g, 0) for g in groups)
        if self.unseen_policy == "weighted_average" and total > 0:
            weights = [self.train_counts.get(g, 0) / total for g in groups]
        else:
            weights = [1.0 / len(groups)] * len(groups)
        return cast(xr.DataArray, sum(w * output for w, output in zip(weights, aligned, strict=True)))

    def _vote_unseen(self, outputs: dict[Hashable, xr.DataArray]) -> xr.DataArray:
        """Weighted majority vote over hard class labels, for group models without predict_proba."""
        groups = list(outputs)
        labels = np.stack([outputs[g].values for g in groups])  # (n_models, n_samples)
        total = sum(self.train_counts.get(g, 0) for g in groups)
        if self.unseen_policy == "weighted_average" and total > 0:
            weights = np.array([self.train_counts.get(g, 0) for g in groups], dtype=float)
        else:
            weights = np.ones(len(groups))
        classes = np.unique(labels)
        model_axis_weights = weights.reshape(-1, *([1] * (labels.ndim - 1)))  # labels: (n_models, *sample_shape)
        votes = np.stack([(model_axis_weights * (labels == c)).sum(axis=0) for c in classes])
        return outputs[groups[0]].copy(data=classes[votes.argmax(axis=0)])

    def _handle_unequal_output_dims(
        self, reassembled: xr.DataArray, group_outputs: list[xr.DataArray], fitted: bool = False
    ) -> xr.DataArray:
        """
        Handles unequal output dimensions for a given dimension.
        """

        # Concatenate the outputs along the group dimension
        reassembled = xr.concat(group_outputs, dim=self.group_dim, join="outer")

        # Check for unequal output dims
        for dim in reassembled.dims:
            if dim == self.group_dim:
                continue

            sizes = [output.sizes[dim] for output in group_outputs]

            # If sizes are not the same, handle according to the unequal_output_dims_strategy
            if not all(size == sizes[0] for size in sizes):
                if self.unequal_output_dims_strategy == "error":
                    raise TransformError(f"Sizes for dimension {dim} are not the same: {sizes}")
                elif self.unequal_output_dims_strategy == "cut_to_min":
                    if not fitted:
                        warnings.warn(
                            f"Sizes for dimension {dim} are not the same: {sizes}. Will only keep the min length",
                            stacklevel=2,
                        )
                        max_allowed_size = min(sizes)
                        self.max_size_per_dim[dim] = max_allowed_size
                        reassembled = reassembled.isel({dim: slice(None, max_allowed_size)})
                    else:
                        max_allowed_size = self.max_size_per_dim[dim]
                        current_size = reassembled.sizes[dim]
                        if current_size > max_allowed_size:
                            warnings.warn(
                                f"Length for dimension {dim} is greater than the allowed max shape: {current_size} > {max_allowed_size}."
                                "Will only keep the max shape",
                                stacklevel=2,
                            )
                            reassembled = reassembled.isel({dim: slice(None, max_allowed_size)})
                else:
                    raise ValueError(f"Invalid unequal_output_dims_strategy: {self.unequal_output_dims_strategy}")

        return reassembled

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Generates predictions using per-group fitted predictors.

        Args:
            container: DataContainer to make predictions on
            **kwargs: Additional context/parameters

        Returns:
            DataContainer with predictions
        """
        if not self.per_group_fitted:
            raise TransformError("GroupApplyTransform must be fitted before predict")

        # Check that all fitted transforms are predictors
        for group_val, fitted_transform in self.per_group_fitted.items():
            if _final_predictor(fitted_transform) is None:
                raise TypeError(
                    f"Transform for group '{group_val}' is not a Predictor or a composite ending in one. "
                    f"predict() requires all fitted transforms to predict."
                )
        fitted_predictors = {
            group_val: cast("Predictor | CompositeTransform", fitted_transform)
            for group_val, fitted_transform in self.per_group_fitted.items()
        }

        # set the combined group coord values and discover groups
        container = self._set_combined_group_coord_values(container)
        current_groups = self._discover_groups(container)

        group_outputs = []

        # Process each group
        for group_val in current_groups:
            group_container = self._select_group(container, group_val)

            if group_val in self.per_group_fitted:
                # Use fitted predictor for seen group
                fitted_predictor = fitted_predictors[group_val]
                prediction = fitted_predictor.predict(group_container, **kwargs)
                group_outputs.append(prediction.data)
            else:
                # Apply unseen group policy
                if self.unseen_policy == "error":
                    raise TransformError(
                        f"Group '{group_val}' was not seen during fit. Seen groups: {self.seen_groups}"
                    )

                # Hard class labels cannot be averaged: vote on averaged probabilities when every group
                # model provides them, otherwise take a weighted majority vote over labels.
                first_predictor = cast(Predictor, _final_predictor(next(iter(fitted_predictors.values()))))
                label_classifier = first_predictor.is_classifier and not getattr(
                    first_predictor, "is_multilabel", False
                )
                vote_on_proba = label_classifier and all(
                    _supports_proba(cast(Predictor, _final_predictor(p))) for p in fitted_predictors.values()
                )

                def predict_with_fitted(fitted_predictor, group_data):
                    if vote_on_proba:
                        return fitted_predictor.predict_proba(group_data, **kwargs)
                    return fitted_predictor.predict(group_data, **kwargs)

                # Apply all fitted predictors
                if self.n_jobs != 1:
                    predictions = Parallel(n_jobs=self.n_jobs)(
                        delayed(predict_with_fitted)(fitted_predictor, group_container)
                        for fitted_predictor in fitted_predictors.values()
                    )
                else:
                    predictions = [predict_with_fitted(p, group_container) for p in fitted_predictors.values()]
                outputs = {g: pred.data for g, pred in zip(fitted_predictors, predictions, strict=True)}

                if vote_on_proba:
                    averaged = self._average_unseen(outputs)
                    # argmax + indexing keeps the label dtype (xarray's idxmax casts integer labels to float)
                    labels = averaged["class"].values[averaged.argmax(dim="class").values]
                    averaged_data = averaged.isel({"class": 0}, drop=True).copy(data=labels).rename("prediction")
                elif label_classifier:
                    averaged_data = self._vote_unseen(outputs)
                else:
                    averaged_data = self._average_unseen(outputs)

                group_outputs.append(averaged_data)

        # Reassemble predictions
        reassembled = xr.concat(group_outputs, dim=self.group_dim)

        reassembled = self._restore_input_order(reassembled, container)

        # remove the combined group coord
        if len(self.group_coord) > 1:
            reassembled = reassembled.drop_vars(self.combined_group_coord)

        return DataContainer(reassembled)

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Generates prediction probabilities using per-group fitted predictors.

        Args:
            container: DataContainer to make predictions on
            **kwargs: Additional context/parameters

        Returns:
            DataContainer with prediction probabilities
        """
        if not self.per_group_fitted:
            raise TransformError("GroupApplyTransform must be fitted before predict_proba")

        # Check that all fitted transforms are classifiers
        for group_val, fitted_transform in self.per_group_fitted.items():
            final_predictor = _final_predictor(fitted_transform)
            if final_predictor is None:
                raise TypeError(
                    f"Transform for group '{group_val}' is not a Predictor or a composite ending in one. "
                    f"predict_proba() requires all fitted transforms to predict."
                )
            if not final_predictor.is_classifier:
                raise TypeError(
                    f"Transform for group '{group_val}' is not a classifier. "
                    f"predict_proba() requires all fitted transforms to be classifiers."
                )
        fitted_predictors = {
            group_val: cast("Predictor | CompositeTransform", fitted_transform)
            for group_val, fitted_transform in self.per_group_fitted.items()
        }

        # set the combined group coord values and discover groups
        container = self._set_combined_group_coord_values(container)
        current_groups = self._discover_groups(container)

        group_outputs = []

        # Process each group
        for group_val in current_groups:
            group_container = self._select_group(container, group_val)

            if group_val in self.per_group_fitted:
                # Use fitted predictor for seen group
                fitted_predictor = fitted_predictors[group_val]
                probabilities = fitted_predictor.predict_proba(group_container, **kwargs)
                group_outputs.append(probabilities.data)
            else:
                # Apply unseen group policy
                if self.unseen_policy == "error":
                    raise TransformError(
                        f"Group '{group_val}' was not seen during fit. Seen groups: {self.seen_groups}"
                    )

                # Apply all fitted predictors and average probabilities
                if self.n_jobs != 1:
                    # Parallel prediction
                    def predict_proba_with_fitted(fitted_predictor, group_data):
                        return fitted_predictor.predict_proba(group_data, **kwargs)

                    probabilities = Parallel(n_jobs=self.n_jobs)(
                        delayed(predict_proba_with_fitted)(fitted_predictor, group_container)
                        for fitted_predictor in fitted_predictors.values()
                    )
                else:
                    # Sequential prediction
                    probabilities = []
                    for fitted_predictor in fitted_predictors.values():
                        probabilities.append(fitted_predictor.predict_proba(group_container, **kwargs))

                averaged_data = self._average_unseen(
                    {g: prob.data for g, prob in zip(fitted_predictors, probabilities, strict=True)}
                )
                group_outputs.append(averaged_data)

        # Reassemble probabilities; a class a group's model never saw has probability 0, not NaN
        reassembled = xr.concat(group_outputs, dim=self.group_dim, join="outer", fill_value=0.0)

        reassembled = self._restore_input_order(reassembled, container)

        # remove the combined group coord
        if len(self.group_coord) > 1:
            reassembled = reassembled.drop_vars(self.combined_group_coord)

        return DataContainer(reassembled)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Returns the expected output dimensions for the GroupApplyTransform.

        Args:
            input_dims: Expected input dimensions

        Returns:
            Expected output dimensions from the reference transform
        """
        # Use template as reference
        return self.transform_template.get_expected_output_dims(input_dims)

    def __repr__(self) -> str:
        return (
            f"GroupApplyTransform(group_coord='{self.group_coord}', "
            f"seen_groups={len(self.seen_groups)}, "
            f"unseen_policy='{self.unseen_policy}', "
            f"n_jobs={self.n_jobs})"
        )
