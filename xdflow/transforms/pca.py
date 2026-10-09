"""Global PCA transform over features preserving (trial, feature, time).

This module defines a Transform that performs PCA across the feature axis by
stacking samples over (trial, time), projecting to the top components, and
reshaping back to the original dimensionality ordering. The resulting data keeps
the same dimension names ("trial", "feature", "time") so it remains compatible
with downstream transforms that expect a feature dimension, but the feature axis
now represents PCA components.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer


def _select_fit_labels(data_array: xr.DataArray, dim: str, fit_labels: np.ndarray | None) -> xr.DataArray:
    """Order `dim` as it was at fit; fitted matrices are applied by position, so a reordered or
    different set of labels would otherwise be silently mis-projected (missing labels raise)."""
    if fit_labels is None or dim not in data_array.indexes:
        return data_array
    return data_array.sel({dim: fit_labels})


def _fit_labels(data_array: xr.DataArray, dim: str) -> np.ndarray | None:
    return data_array[dim].values.copy() if dim in data_array.indexes else None


def _center_mode_for(center_data: bool | str) -> str:
    """Normalize `center_data` to "true", "false" or "false_oldstyle"; raise on anything else."""
    if isinstance(center_data, bool):
        return "true" if center_data else "false"
    if isinstance(center_data, str) and center_data.lower() == "false_oldstyle":
        return "false_oldstyle"
    raise ValueError("center_data must be True, False, or 'false_oldstyle'.")


class GlobalFeaturePCA(Transform):
    """Performs global PCA across features while preserving time structure.

    The transform treats each sample as a concatenation of (trial, time_dim) and fits
    PCA over the feature covariance matrix. It then projects the data onto the top
    principal components and reshapes back to (trial, component, time_dim). The output
    dimension name remains "feature_dim" for downstream compatibility (components
    replace features).
    """

    is_stateful = True
    _supports_transform_sel: bool = False

    def __init__(
        self,
        n_components: float | int | None = None,
        pca_frac_to_keep: float | None = None,
        center_data: bool | str = True,
        whiten: bool = False,
        time_dim: str = "time",
        feature_dim: str = "channel",
        sel: dict | None = None,
        drop_sel: dict | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            n_components: Optional fraction (0, 1] of variance to keep or integer >= 1
                specifying exact number of components (sklearn standard behavior).
                Cannot be specified together with pca_frac_to_keep.
            pca_frac_to_keep: Optional fraction of principal components to keep.
                Cannot be specified together with n_components.
            center_data: Whether to center features across (trial, time) before PCA.
            whiten: Whether to whiten the data before fitting/projection.
            time_dim: The dimension name for the time dimension.
            feature_dim: The dimension name for the feature dimension.
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
            transform_sel: Optional selection applied only during transform.
            transform_drop_sel: Optional drop selection applied only during transform.
        """
        # Use drop_sel to exclude reference channels globally; selective transform is not supported
        super().__init__(
            sel=sel,
            drop_sel=drop_sel,
        )

        # Validate n_components
        if n_components is not None:
            if isinstance(n_components, float):
                if not (0.0 < n_components <= 1.0):
                    raise ValueError("n_components float must be in (0, 1].")
            elif isinstance(n_components, int):
                if n_components < 1:
                    raise ValueError("n_components int must be >= 1.")
            else:
                raise TypeError("n_components must be a float in (0,1] or an int >= 1.")

        # Validate pca_frac_to_keep
        if pca_frac_to_keep is not None:
            if not isinstance(pca_frac_to_keep, (int, float)) or not (0.0 < pca_frac_to_keep <= 1.0):
                raise ValueError("pca_frac_to_keep must be a number greater than 0.0 and less than or equal to 1.0.")

        # Mutual exclusion validation
        if n_components is not None and pca_frac_to_keep is not None:
            raise ValueError("Cannot specify both n_components and pca_frac_to_keep. Use one or the other.")

        # Ensure at least one is specified
        if n_components is None and pca_frac_to_keep is None:
            raise ValueError("Must specify either n_components or pca_frac_to_keep.")

        self.n_components = n_components
        self.pca_frac_to_keep = float(pca_frac_to_keep) if pca_frac_to_keep is not None else None
        _center_mode_for(center_data)  # validate now; `_center_mode` derives it on read
        self.center_data = center_data  # keep original for introspection
        self.whiten = bool(whiten)

        # Learned parameters
        self.components_: np.ndarray | None = None  # shape: (k, n_features)
        self.feature_mean = None  # xarray DataArray of shape (feature,)

        self.time_dim = time_dim
        self.feature_dim = feature_dim

    @property
    def input_dims(self) -> tuple[str, ...]:
        return ("trial", self.feature_dim, self.time_dim)

    @property
    def output_dims(self) -> tuple[str, ...]:
        return ("trial", self.feature_dim, self.time_dim)

    @property
    def _center_mode(self) -> str:
        """`center_data` normalized, derived on read so `set_params(center_data=...)` reaches it."""
        return _center_mode_for(self.center_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return self.output_dims

    def _fit(self, container: DataContainer, **kwargs) -> GlobalFeaturePCA:
        """Fit PCA across feature covariance.

        Args:
            container: Input data container with dims (trial, feature_dim, time_dim).

        Returns:
            Self.
        """
        data_array = container.data
        self._fit_feature_labels = _fit_labels(data_array, self.feature_dim)

        # Save mean if needed (vector of length feature)
        if self._center_mode == "true":
            self.feature_mean = data_array.mean(dim=["trial", self.time_dim])  # (feature,)
        else:
            self.feature_mean = None

        # Prepare (features, samples)
        data_ctt = data_array.transpose(self.feature_dim, "trial", self.time_dim)
        X = data_ctt.values.reshape(data_ctt.sizes[self.feature_dim], -1)

        # Prepare covariance according to centering mode
        n_samples = X.shape[1]
        if self._center_mode == "true":
            if self.feature_mean is None:
                raise RuntimeError("feature_mean was not computed during fit.")
            Xc = X - self.feature_mean.values[:, None]
            cov = (Xc @ Xc.T) / (n_samples - 1)
        elif self._center_mode == "false_oldstyle":
            # Center for covariance only (do not store/apply mean at transform)
            temp_mu = X.mean(axis=1, keepdims=True)
            Xc = X - temp_mu
            cov = (Xc @ Xc.T) / (n_samples - 1)
        else:  # "false"
            cov = (X @ X.T) / n_samples

        # Use eigh for symmetric matrices; returns ascending eigenvalues
        S, U = np.linalg.eigh(cov)
        # Sort in descending order
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        # Clamp tiny negative eigenvalues from numerical error
        S = np.maximum(S, 0.0)

        n_features = X.shape[0]

        # Determine number of components based on sklearn standard behavior
        if self.n_components is not None:
            if isinstance(self.n_components, float):
                # sklearn behavior: n_components is fraction of variance to keep
                cumsum_variance = np.cumsum(S) / np.sum(S)
                k = np.argmax(cumsum_variance >= self.n_components) + 1
                k = max(1, min(k, n_features))  # Ensure at least 1, at most all features
            else:
                # Integer: exact number of components
                k = min(n_features, int(self.n_components))
        else:
            # pca_frac_to_keep specified - use all features as base
            k = n_features

        # Apply pca_frac_to_keep if specified
        if self.pca_frac_to_keep is not None:
            n_to_keep = int(np.ceil(k * self.pca_frac_to_keep))
            k = min(k, n_to_keep)  # Don't exceed the variance-based selection

        # Build projection matrix rows explicitly; whitening optional
        if self.whiten:
            epsilon = 1e-6  # small regularizer for numerical stability
            self.components_ = (U[:, :k] / np.sqrt(S[:k] + epsilon)).T  # (k, n_features)
        else:
            self.components_ = U[:, :k].T  # (k, n_features)

        # Store PCA spectrum for downstream use
        self.explained_variance_ = S[:k]
        total_variance = S.sum() if S.size else 0.0
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance if total_variance > 0 else self.explained_variance_
        )
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Project data onto learned PCA components.

        Args:
            container: Input data container with dims (trial, feature, time).

        Returns:
            DataContainer with projected data of dims (trial, feature, time) where
            feature now indexes PCA components.
        """
        if self.components_ is None:
            raise RuntimeError("The transform has not been fitted yet. Call 'fit' before 'transform'.")

        data_array = container.data
        data_array = _select_fit_labels(data_array, self.feature_dim, getattr(self, "_fit_feature_labels", None))
        data_ctt = data_array.transpose(self.feature_dim, "trial", self.time_dim)
        X = data_ctt.values.reshape(data_ctt.sizes[self.feature_dim], -1)

        if self._center_mode == "true" and self.feature_mean is not None:
            mean_np = self.feature_mean.values.astype(X.dtype, copy=False)
            X = X - mean_np[:, None]

        # (k, n_features) @ (n_features, n_samples) -> (k, n_samples)
        Xproj = self.components_ @ X

        # Reshape back and build a fresh DataArray with updated feature length
        k = self.components_.shape[0]
        Xproj = Xproj.reshape(k, data_ctt.sizes["trial"], data_ctt.sizes[self.time_dim])  # type: ignore[index]

        # Build output coordinates, preserving trial-aligned coords
        base_coords = {
            self.feature_dim: np.arange(k),
            "trial": data_ctt["trial"].values,
            self.time_dim: data_ctt[self.time_dim].values,
        }
        for coord_name, coord_data in data_array.coords.items():
            if coord_name in base_coords:
                continue
            if "trial" in coord_data.dims:
                base_coords[coord_name] = coord_data

        projected = xr.DataArray(
            Xproj,
            dims=(self.feature_dim, "trial", self.time_dim),
            coords=base_coords,
            attrs=data_array.attrs,
            name=getattr(data_array, "name", None),
        ).transpose("trial", self.feature_dim, self.time_dim)
        return DataContainer(projected)
