from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.transforms.pca import _fit_labels, _select_fit_labels
from xdflow.transforms.sklearn_transform import SKLearnTransformer
from xdflow.transforms.spatial import _map_from_grid, _map_to_grid


def _center_mode_for(center_data: bool | str) -> str:
    """Normalize `center_data` to "true", "false" or "false_oldstyle"; raise on anything else."""
    if isinstance(center_data, bool):
        return "true" if center_data else "false"
    if isinstance(center_data, str) and center_data.lower() == "false_oldstyle":
        return "false_oldstyle"
    raise ValueError("center_data must be True, False, or 'false_oldstyle'.")


class LocalZCAWhitening(Transform):
    """
    Performs local ZCA whitening on neural data using spatial electrode neighborhoods.

    This is a stateful transform that uses the electrode array layout to determine
    spatial neighborhoods. The 'fit' method computes a whitening matrix based on
    the covariance of local spatial neighborhoods. The 'transform' method applies
    this pre-computed matrix to the data.

    Attributes:
        grid_layout: 2D numpy array defining the spatial electrode layout, where values
            correspond to channel IDs. Required for spatial mapping and neighborhood
            determination.
        radius (float): The radius in grid units for defining a channel's neighborhood.
        epsilon (float): Regularization parameter for numerical stability.
        whitening_strength (float): Controls the degree of whitening applied.
        n_components: Optional fraction (0, 1] of variance to keep or integer >= 1.
            Cannot be specified together with pca_frac_to_keep.
        pca_frac_to_keep: Optional fraction of principal components to keep.
            Cannot be specified together with n_components.
        center_data (bool): Whether to center data before whitening.
    """

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")
    _supports_transform_sel: bool = True

    def __init__(
        self,
        grid_layout: np.ndarray,
        radius: float = 1.0,
        epsilon: float = 1e-6,
        whitening_strength: float = 1.0,
        n_components: float | int | None = None,
        pca_frac_to_keep: float | None = None,
        center_data: bool | str = True,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.grid_layout = grid_layout

        # Validate inputs
        if not isinstance(radius, (int, float)) or radius < 0:
            raise ValueError("Radius must be a non-negative number.")

        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("Epsilon must be a positive number.")

        if not isinstance(whitening_strength, (int, float)) or not (0.0 <= whitening_strength <= 1.0):
            raise ValueError("whitening_strength must be a number between 0.0 and 1.0.")

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

        self.radius = radius
        self.epsilon = epsilon
        self.whitening_strength = whitening_strength
        self.n_components = n_components
        self.pca_frac_to_keep = float(pca_frac_to_keep) if pca_frac_to_keep is not None else None
        _center_mode_for(center_data)  # validate now; `_center_mode` derives it on read
        self.center_data = center_data
        self.W_local = None  # The whitening matrix, learned during fit
        self.data_mean = None  # Mean for centering, computed during fit

    @property
    def _center_mode(self) -> str:
        """`center_data` normalized, derived on read so `set_params(center_data=...)` reaches it."""
        return _center_mode_for(self.center_data)

    def _calculate_zca_matrix(self, covariance_matrix):
        """
        Helper function to compute the ZCA matrix from a covariance matrix,
        incorporating whitening strength and PCA component selection.

        Args:
            covariance_matrix (np.ndarray): Input covariance matrix.

        Returns:
            np.ndarray: ZCA whitening matrix.
        """
        # Check for numerical issues
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            warnings.warn("Covariance matrix is not symmetric. This may indicate numerical issues.", stacklevel=3)

        # Use eigh for symmetric covariance; numerically preferable
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Clamp tiny negative eigenvalues from numerical error
        eigvals = np.maximum(eigvals, 0.0)

        # Check condition number for numerical stability
        condition_number = eigvals.max() / (eigvals.min() + self.epsilon)
        if condition_number > 1e12:
            warnings.warn(
                f"Covariance matrix is ill-conditioned (condition number: {condition_number:.2e}). "
                f"Consider increasing epsilon (current: {self.epsilon}) or reducing the neighborhood size.",
                stacklevel=3,
            )

        # 1. Determine how many principal components to keep
        n_components = len(eigvals)
        if self.n_components is not None:
            if isinstance(self.n_components, float):
                # sklearn behavior: n_components is fraction of variance to keep
                cumsum_variance = np.cumsum(eigvals) / np.sum(eigvals)
                n_to_keep = np.argmax(cumsum_variance >= self.n_components) + 1
                n_to_keep = max(1, min(n_to_keep, n_components))
            else:
                # Integer: exact number of components
                n_to_keep = min(n_components, int(self.n_components))
        else:
            # pca_frac_to_keep specified - use fraction of all components
            assert self.pca_frac_to_keep is not None
            n_to_keep = int(np.ceil(n_components * self.pca_frac_to_keep))

        # 2. Compute the exponent for whitening based on the strength parameter
        exponent = -self.whitening_strength / 2.0
        whitening_scaler = (eigvals + self.epsilon) ** exponent

        # 3. Zero out the components that are being discarded for denoising
        # This effectively removes their contribution from the final transformation
        whitening_scaler[n_to_keep:] = 0.0

        # 4. Construct the final transformation matrix
        zca_matrix = eigvecs @ np.diag(whitening_scaler) @ eigvecs.T

        return zca_matrix

    def _find_spatial_neighbors(self, row: int, col: int, grid_shape: tuple) -> list:
        """
        Find spatial neighbors within radius on the electrode grid.

        Args:
            row (int): Row position on the grid.
            col (int): Column position on the grid.
            grid_shape (tuple): Shape of the electrode grid (rows, cols).

        Returns:
            list: List of (row, col) tuples for valid neighbors within radius.
        """
        neighbors = []
        rows, cols = grid_shape

        for r in range(max(0, row - int(self.radius)), min(rows, row + int(self.radius) + 1)):
            for c in range(max(0, col - int(self.radius)), min(cols, col + int(self.radius) + 1)):
                # Calculate Euclidean distance
                distance = np.sqrt((r - row) ** 2 + (c - col) ** 2)
                if distance <= self.radius:
                    neighbors.append((r, c))

        return neighbors

    def _fit(self, container: DataContainer, **kwargs) -> LocalZCAWhitening:
        """
        Computes the local spatial whitening matrix from the data.
        """
        data_array = container.data

        # Map data to spatial grid
        gridded_data = _map_to_grid(data_array, self.grid_layout)

        # Center data if requested
        if self._center_mode == "true":
            self.gridded_mean = gridded_data.mean(dim=["trial", "time"])
            gridded_data = gridded_data - self.gridded_mean
        else:
            self.gridded_mean = None

        # Get grid dimensions
        grid_shape = self.grid_layout.shape

        # Initialize spatial whitening filters - one for each grid position
        self.spatial_whitening_filters = {}

        # Process each valid position in the grid
        for row in range(grid_shape[0]):
            for col in range(grid_shape[1]):
                # Skip positions that don't have electrodes (NaN in grid_layout)
                if np.isnan(self.grid_layout[row, col]):
                    continue

                # Find spatial neighbors within radius
                neighbor_positions = self._find_spatial_neighbors(row, col, grid_shape)

                # Filter out positions that don't have electrodes
                valid_neighbors = [(r, c) for r, c in neighbor_positions if not np.isnan(self.grid_layout[r, c])]

                if len(valid_neighbors) == 0:
                    warnings.warn(
                        f"Grid position ({row}, {col}) has no valid neighbors within radius {self.radius}. "
                        f"Skipping whitening for this position.",
                        stacklevel=2,
                    )
                    continue

                # Extract neighborhood data for covariance computation
                neighbor_rows, neighbor_cols = zip(*valid_neighbors)

                # Extract data for each neighbor position
                neighbor_data_list = []
                for r, c in valid_neighbors:
                    # Extract data at this grid position: shape (trials, time)
                    position_data = gridded_data.isel(row=r, col=c)
                    # Flatten to (trials*time,) for this neighbor
                    neighbor_data_list.append(position_data.values.flatten())

                # Stack to create (n_neighbors, n_samples) matrix
                neighborhood_reshaped = np.stack(neighbor_data_list, axis=0)

                # Compute local covariance matrix
                if len(valid_neighbors) == 1:
                    # Single neighbor case - just pass through with no whitening
                    warnings.warn(
                        f"Grid position ({row}, {col}) has only one neighbor (itself). "
                        f"Skipping whitening for this position.",
                        stacklevel=2,
                    )
                    continue

                # Covariance/second-moment honoring center_data
                n_samples_local = neighborhood_reshaped.shape[1]
                if self._center_mode == "true":
                    local_covariance = (neighborhood_reshaped @ neighborhood_reshaped.T) / (n_samples_local - 1)
                elif self._center_mode == "false_oldstyle":
                    temp_mu = neighborhood_reshaped.mean(axis=1, keepdims=True)
                    x_centered_local = neighborhood_reshaped - temp_mu
                    local_covariance = (x_centered_local @ x_centered_local.T) / (n_samples_local - 1)
                else:
                    local_covariance = (neighborhood_reshaped @ neighborhood_reshaped.T) / n_samples_local

                # Check for numerical issues
                if not np.allclose(local_covariance, local_covariance.T, rtol=1e-10):
                    warnings.warn(
                        f"Local covariance matrix at position ({row}, {col}) is not symmetric. "
                        f"This may indicate numerical issues. Skipping whitening for this position.",
                        stacklevel=2,
                    )
                    continue

                # Compute ZCA whitening matrix for this neighborhood
                try:
                    local_zca_matrix = self._calculate_zca_matrix(local_covariance)
                except np.linalg.LinAlgError as e:
                    warnings.warn(
                        f"Failed to compute ZCA matrix at position ({row}, {col}): {e}. "
                        f"Skipping whitening for this position.",
                        stacklevel=2,
                    )
                    continue

                # Find the index of the center position within its neighborhood
                try:
                    center_idx = valid_neighbors.index((row, col))
                    # Store the whitening filter for this position
                    self.spatial_whitening_filters[(row, col)] = {
                        "neighbors": valid_neighbors,
                        "filter": local_zca_matrix[center_idx, :],
                        "neighbor_positions": valid_neighbors,
                    }
                except ValueError:
                    # Center position not in its own neighborhood (shouldn't happen with radius >= 0)
                    warnings.warn(
                        f"Center position ({row}, {col}) not found in its own neighborhood. Skipping.",
                        stacklevel=2,
                    )
                    continue

        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the pre-computed spatial whitening filters to the data.
        """
        if not hasattr(self, "spatial_whitening_filters") or not self.spatial_whitening_filters:
            raise RuntimeError("The transform has not been fitted yet. Call 'fit' before 'transform'.")

        data_array = container.data

        # Map data to spatial grid
        gridded_data = _map_to_grid(data_array, self.grid_layout)

        # Center data if it was centered during fitting
        if self._center_mode == "true" and self.gridded_mean is not None:
            gridded_data = gridded_data - self.gridded_mean

        # Apply spatial whitening filters
        whitened_gridded_data = gridded_data.copy()

        # Process each position that has a whitening filter
        for (row, col), filter_info in self.spatial_whitening_filters.items():
            neighbors = filter_info["neighbors"]
            whitening_filter = filter_info["filter"]

            # Extract neighborhood data for this position
            neighbor_data_list = []
            for r, c in neighbors:
                # Extract data at this grid position: shape (trials, time)
                position_data = gridded_data.isel(row=r, col=c)
                # Flatten to (trials*time,) for this neighbor
                neighbor_data_list.append(position_data.values.flatten())

            # Stack to create (n_neighbors, n_samples) matrix
            neighborhood_reshaped = np.stack(neighbor_data_list, axis=0)

            # Apply whitening filter (1 x n_neighbors) @ (n_neighbors x n_samples)
            whitened_value = whitening_filter @ neighborhood_reshaped

            # Reshape back to (trials, time) and assign to the center position
            n_trials, n_time = gridded_data.isel(row=row, col=col).shape
            whitened_reshaped = whitened_value.reshape(n_trials, n_time)
            whitened_gridded_data.values[:, :, row, col] = whitened_reshaped

        # Map whitened spatial data back to channel space using current data structure
        output_data = _map_from_grid(
            whitened_gridded_data,
            self.grid_layout,
            data_array.dims,
            data_array.coords,
            data_array.sizes,
            data_array.attrs,
        )

        return DataContainer(output_data)


class GlobalZCAWhitening(Transform):
    """
    Performs global ZCA whitening across channels.

    This transform computes a single ZCA whitening matrix over all channels,
    flattening samples across (trial, time). It supports partial whitening via
    `whitening_strength`, optional dimensionality reduction via
    `n_components` or `pca_frac_to_keep`, and optional centering of the data before whitening.

    Attributes:
        epsilon (float): Regularization added to eigenvalues for stability.
        whitening_strength (float): 0.0=no whitening, 1.0=full whitening.
        n_components: Optional fraction (0, 1] of variance to keep or integer >= 1.
            Cannot be specified together with pca_frac_to_keep.
        pca_frac_to_keep: Optional fraction of principal components to retain.
            Cannot be specified together with n_components.
        center_data (bool): Whether to center data across (trial, time).
        shrinkage (str | None): Placeholder to choose covariance shrinkage
            strategy during fitting (e.g., "ledoit-wolf"). Not implemented yet.
    """

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")
    _supports_transform_sel: bool = True

    def __init__(
        self,
        epsilon: float = 1e-6,
        whitening_strength: float = 1.0,
        n_components: float | int | None = None,
        pca_frac_to_keep: float | None = None,
        center_data: bool | str = True,
        keep_in_pc_space: bool = False,
        shrinkage: str | None = None,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(
            sel=sel,
            drop_sel=drop_sel,
        )

        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive number.")
        if not isinstance(whitening_strength, (int, float)) or not (0.0 <= whitening_strength <= 1.0):
            raise ValueError("whitening_strength must be in [0.0, 1.0].")

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
                raise ValueError("pca_frac_to_keep must be in (0.0, 1.0].")

        # Mutual exclusion validation
        if n_components is not None and pca_frac_to_keep is not None:
            raise ValueError("Cannot specify both n_components and pca_frac_to_keep. Use one or the other.")

        # Ensure at least one is specified
        if n_components is None and pca_frac_to_keep is None:
            raise ValueError("Must specify either n_components or pca_frac_to_keep.")

        if shrinkage is not None:
            warnings.warn(
                "Covariance shrinkage is accepted as a parameter but not implemented yet; proceeding without shrinkage.",
                stacklevel=2,
            )

        self.epsilon = float(epsilon)
        self.whitening_strength = float(whitening_strength)
        self.n_components = n_components
        self.pca_frac_to_keep = float(pca_frac_to_keep) if pca_frac_to_keep is not None else None
        _center_mode_for(center_data)  # validate now; `_center_mode` derives it on read
        self.center_data = center_data
        self.keep_in_pc_space = bool(keep_in_pc_space)
        self.shrinkage = shrinkage

        self.W_global: np.ndarray | None = None
        self.channel_mean = None  # xarray DataArray of shape (channel,)

    @property
    def _center_mode(self) -> str:
        """`center_data` normalized, derived on read so `set_params(center_data=...)` reaches it."""
        return _center_mode_for(self.center_data)

    def _calculate_zca_matrix(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute ZCA matrix from covariance with strength and PCA component selection.

        Args:
            covariance_matrix: Symmetric covariance matrix of shape (n_channels, n_channels).

        Returns:
            ZCA transform matrix of shape (n_channels, n_channels).
        """
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            warnings.warn("Covariance matrix is not symmetric; numerical issues possible.", stacklevel=3)

        # Use eigh for symmetric covariance; numerically preferable
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Clamp tiny negative eigenvalues from numerical error
        eigvals = np.maximum(eigvals, 0.0)

        condition_number = eigvals.max() / (eigvals.min() + self.epsilon)
        if condition_number > 1e12:
            warnings.warn(
                f"Covariance matrix is ill-conditioned (cond: {condition_number:.2e}). Consider increasing epsilon.",
                stacklevel=3,
            )

        n_components = len(eigvals)
        if self.n_components is not None:
            if isinstance(self.n_components, float):
                # sklearn behavior: n_components is fraction of variance to keep
                cumsum_variance = np.cumsum(eigvals) / np.sum(eigvals)
                n_to_keep = np.argmax(cumsum_variance >= self.n_components) + 1
                n_to_keep = max(1, min(n_to_keep, n_components))
            else:
                # Integer: exact number of components
                n_to_keep = min(n_components, int(self.n_components))
        else:
            # pca_frac_to_keep specified - use fraction of all components
            assert self.pca_frac_to_keep is not None
            n_to_keep = int(np.ceil(n_components * self.pca_frac_to_keep))

        exponent = -self.whitening_strength / 2.0
        whitening_scaler = (eigvals + self.epsilon) ** exponent

        if self.keep_in_pc_space:
            # Return reduced PC-space projection: (k, n_channels)
            zca_matrix = np.diag(whitening_scaler[:n_to_keep]) @ eigvecs[:, :n_to_keep].T
        else:
            # Zero discarded components when returning to sensor space
            whitening_scaler[n_to_keep:] = 0.0
            zca_matrix = eigvecs @ np.diag(whitening_scaler) @ eigvecs.T
        return zca_matrix

    def _fit(self, container: DataContainer, **kwargs) -> GlobalZCAWhitening:
        """
        Fit the global ZCA whitening matrix.

        Args:
            container: Input data container with dims (trial, channel, time).

        Returns:
            Self.
        """
        data_array = container.data
        self._fit_channel_labels = _fit_labels(data_array, "channel")

        # Optional centering mean saved for use at transform-time
        if self._center_mode == "true":
            self.channel_mean = data_array.mean(dim=["trial", "time"])  # shape: (channel,)
        else:
            self.channel_mean = None

        # Build (channels, samples) matrix
        data_ctt = data_array.transpose("channel", "trial", "time")
        x = data_ctt.values.reshape(data_ctt.sizes["channel"], -1)

        # Center prior to covariance if requested, to mirror LocalZCA behavior
        # Compute covariance/second-moment honoring center mode
        n_samples = x.shape[1]
        if self._center_mode == "true":
            if self.channel_mean is None:
                raise RuntimeError("channel_mean was not computed during fit.")
            x_centered = x - self.channel_mean.values[:, None]
            covariance = (x_centered @ x_centered.T) / (n_samples - 1)
        elif self._center_mode == "false_oldstyle":
            temp_mu = x.mean(axis=1, keepdims=True)
            x_centered = x - temp_mu
            covariance = (x_centered @ x_centered.T) / (n_samples - 1)
        else:
            covariance = (x @ x.T) / n_samples

        self.W_global = self._calculate_zca_matrix(covariance)
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the learned global ZCA whitening matrix to the data.

        Args:
            container: Input data container with dims (trial, channel, time).

        Returns:
            DataContainer with whitened data.
        """
        if self.W_global is None:
            raise RuntimeError("The transform has not been fitted yet. Call 'fit' before 'transform'.")

        data_array = container.data
        data_array = _select_fit_labels(data_array, "channel", getattr(self, "_fit_channel_labels", None))

        # Prepare data as (channels, samples)
        data_ctt = data_array.transpose("channel", "trial", "time")
        x = data_ctt.values.reshape(data_ctt.sizes["channel"], -1)

        # Center if requested and available
        if self._center_mode == "true" and self.channel_mean is not None:
            mean_np = self.channel_mean.values.astype(x.dtype, copy=False)
            x = x - mean_np[:, None]

        # Apply whitening/projection
        x_whitened = self.W_global @ x  # shape: (k_or_n, n_samples)

        if self.keep_in_pc_space:
            # Reduce channel dimension to k
            k = self.W_global.shape[0]
            x_whitened = x_whitened.reshape(k, data_ctt.sizes["trial"], data_ctt.sizes["time"])  # type: ignore[index]

            # Build new coordinates akin to GlobalColoringProjection
            new_coords = {
                "channel": np.arange(k),
                "trial": data_ctt["trial"].values,
                "time": data_ctt["time"].values,
            }
            for coord_name, coord_data in data_array.coords.items():
                if coord_name not in new_coords and all(dim in new_coords for dim in coord_data.dims):
                    new_coords[coord_name] = coord_data

            whitened = xr.DataArray(
                x_whitened,
                dims=("channel", "trial", "time"),
                coords=new_coords,
                attrs=data_array.attrs,
                name=getattr(data_array, "name", None),
            ).transpose("trial", "channel", "time")
        else:
            # Preserve original channel count
            x_whitened = x_whitened.reshape(data_ctt.sizes["channel"], data_ctt.sizes["trial"], data_ctt.sizes["time"])  # type: ignore[index]
            whitened = data_ctt.copy(data=x_whitened).transpose(*data_array.dims)

        return DataContainer(whitened)


class GlobalColoringProjection(Transform):
    """
    Performs a global "coloring" projection across channels.

    This transform learns the covariance structure of the channels and creates a
    "coloring" matrix, which is the inverse of a ZCA whitening matrix. Instead of
    removing correlations, this matrix embodies them.

    The transform then projects the data onto the basis vectors of this coloring
    matrix. The resulting features represent how strongly the neural activity at
    each time point aligns with the dominant, learned patterns of spatial
    covariance.

    This serves as a powerful alternative to GlobalFeaturePCA. While PCA finds
    orthogonal axes of maximum variance, this transform finds axes that represent
    the natural, correlated modes of the system.

    Attributes:
        epsilon (float): Regularization added to eigenvalues for stability.
        n_components: Optional fraction (0, 1] of variance to keep or integer >= 1.
            Cannot be specified together with pca_frac_to_keep.
        pca_frac_to_keep: Optional fraction of principal components to retain,
            analogous to PCA, for dimensionality reduction.
            Cannot be specified together with n_components.
        center_data (bool): Whether to center data across (trial, time).
    """

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")
    _supports_transform_sel: bool = True

    def __init__(
        self,
        epsilon: float = 1e-6,
        n_components: float | int | None = None,
        pca_frac_to_keep: float | None = None,
        center_data: bool | str = True,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        """Initialize the transform.

        Args:
            epsilon: Regularization for numerical stability.
            n_components: Optional fraction (0, 1] of variance to keep or integer >= 1
                specifying exact number of components (sklearn standard behavior).
                Cannot be specified together with pca_frac_to_keep.
            pca_frac_to_keep: Optional fraction of components to keep, controlling output
                dimensionality. Cannot be specified together with n_components.
            center_data: Whether to center channels before fitting/projection.
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)

        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive number.")

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
                raise ValueError("pca_frac_to_keep must be in (0.0, 1.0].")

        # Mutual exclusion validation
        if n_components is not None and pca_frac_to_keep is not None:
            raise ValueError("Cannot specify both n_components and pca_frac_to_keep. Use one or the other.")

        # Ensure at least one is specified
        if n_components is None and pca_frac_to_keep is None:
            raise ValueError("Must specify either n_components or pca_frac_to_keep.")

        self.epsilon = float(epsilon)
        self.n_components = n_components
        self.pca_frac_to_keep = float(pca_frac_to_keep) if pca_frac_to_keep is not None else None
        _center_mode_for(center_data)  # validate now; `_center_mode` derives it on read
        self.center_data = center_data

        # Learned parameters
        self.W_color: np.ndarray | None = None  # The coloring matrix
        self.channel_mean: xr.DataArray | None = None  # Mean for centering

    @property
    def _center_mode(self) -> str:
        """`center_data` normalized, derived on read so `set_params(center_data=...)` reaches it."""
        return _center_mode_for(self.center_data)

    def _calculate_coloring_matrix(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the coloring matrix from covariance.

        This matrix is constructed to re-introduce correlations into decorrelated data,
        making its rows a basis of the dominant covariance patterns.

        Args:
            covariance_matrix: Symmetric covariance matrix of shape (n_channels, n_channels).

        Returns:
            Coloring projection matrix of shape (k, n_channels), where k is the
            number of components to keep.
        """
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            warnings.warn("Covariance matrix is not symmetric; numerical issues possible.", stacklevel=3)

        # Use eigh for symmetric covariance; numerically preferable
        eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Clamp tiny negative eigenvalues from numerical error
        eigvals = np.maximum(eigvals, 0.0)

        condition_number = eigvals.max() / (eigvals.min() + self.epsilon)
        if condition_number > 1e12:
            warnings.warn(
                f"Covariance matrix is ill-conditioned (cond: {condition_number:.2e}). Consider increasing epsilon.",
                stacklevel=3,
            )

        n_components = len(eigvals)
        if self.n_components is not None:
            if isinstance(self.n_components, float):
                # sklearn behavior: n_components is fraction of variance to keep
                cumsum_variance = np.cumsum(eigvals) / np.sum(eigvals)
                n_to_keep = np.argmax(cumsum_variance >= self.n_components) + 1
                n_to_keep = max(1, min(n_to_keep, n_components))
            else:
                # Integer: exact number of components
                n_to_keep = min(n_components, int(self.n_components))
        else:
            # pca_frac_to_keep specified - use fraction of all components
            assert self.pca_frac_to_keep is not None
            n_to_keep = int(np.ceil(n_components * self.pca_frac_to_keep))

        # The "coloring" exponent is positive, opposite of whitening
        exponent = 0.5
        coloring_scaler = (eigvals + self.epsilon) ** exponent

        # Project onto the top 'n_to_keep' eigenvectors, scaled by eigvals^1/2:
        # (eigvecs[:, :n_to_keep] @ diag(eigvals[:n_to_keep]^1/2)).T
        return (eigvecs[:, :n_to_keep] * coloring_scaler[:n_to_keep]).T

    def _fit(self, container: DataContainer, **kwargs) -> GlobalColoringProjection:
        """
        Fit the global coloring projection matrix from the data's covariance.

        Args:
            container: Input data container with dims (trial, channel, time).

        Returns:
            Self.
        """
        data_array = container.data
        self._fit_channel_labels = _fit_labels(data_array, "channel")

        # Optional centering mean saved for use at transform-time
        if self._center_mode == "true":
            self.channel_mean = data_array.mean(dim=["trial", "time"])  # shape: (channel,)
        else:
            self.channel_mean = None

        # Build (channels, samples) matrix
        data_ctt = data_array.transpose("channel", "trial", "time")
        x = data_ctt.values.reshape(data_ctt.sizes["channel"], -1)

        # Compute covariance/second-moment honoring center mode
        n_samples = x.shape[1]
        if self._center_mode == "true":
            if self.channel_mean is None:
                raise RuntimeError("channel_mean was not computed during fit.")
            x_centered = x - self.channel_mean.values[:, None]
            covariance = (x_centered @ x_centered.T) / (n_samples - 1)
        elif self._center_mode == "false_oldstyle":
            temp_mu = x.mean(axis=1, keepdims=True)
            x_centered = x - temp_mu
            covariance = (x_centered @ x_centered.T) / (n_samples - 1)
        else:
            covariance = (x @ x.T) / n_samples

        self.W_color = self._calculate_coloring_matrix(covariance)
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the learned coloring projection to the data.

        Args:
            container: Input data container with dims (trial, channel, time).

        Returns:
            DataContainer with projected data. The 'channel' dimension now
            represents the strength of the covariance patterns.
        """
        if self.W_color is None:
            raise RuntimeError("The transform has not been fitted yet. Call 'fit' before 'transform'.")

        data_array = container.data
        data_array = _select_fit_labels(data_array, "channel", getattr(self, "_fit_channel_labels", None))

        # Prepare data as (channels, samples)
        data_ctt = data_array.transpose("channel", "trial", "time")
        x = data_ctt.values.reshape(data_ctt.sizes["channel"], -1)

        # Center if requested and available
        if self._center_mode == "true" and self.channel_mean is not None:
            mean_np = self.channel_mean.values.astype(x.dtype, copy=False)
            x = x - mean_np[:, None]

        # Apply coloring projection: (k, n_channels) @ (n_channels, n_samples) -> (k, n_samples)
        x_proj = self.W_color @ x

        # Reshape back and build a fresh DataArray with updated channel coordinate
        k = self.W_color.shape[0]
        x_proj_reshaped = x_proj.reshape(k, data_ctt.sizes["trial"], data_ctt.sizes["time"])

        # Create new coordinates for the projected 'channel' dimension
        new_coords = {
            "channel": np.arange(k),
            "trial": data_ctt["trial"].values,
            "time": data_ctt["time"].values,
        }
        # Preserve other coordinates that align with the preserved dimensions
        for coord_name, coord_data in data_array.coords.items():
            if coord_name not in new_coords and all(dim in new_coords for dim in coord_data.dims):
                new_coords[coord_name] = coord_data

        projected_data_array = xr.DataArray(
            x_proj_reshaped,
            dims=("channel", "trial", "time"),
            coords=new_coords,
            attrs=data_array.attrs,
            name=getattr(data_array, "name", None),
        ).transpose("trial", "channel", "time")

        return DataContainer(projected_data_array)


class ZCAWhitening(BaseEstimator, TransformerMixin):
    """
    ZCA (Zero Component Analysis) whitening using sklearn's PCA.

    This estimator performs PCA with whitening and then inverts the PCA transform
    to return to the original space, effectively implementing ZCA whitening.

    Parameters:
        n_components: int, float or None, default=None
            Number of components to keep. If None, all components are kept.
            If float in (0, 1], it represents the fraction of variance to keep.
        random_state: int, RandomState instance or None, default=None
            Random state for reproducibility.
    """

    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, x, y=None):
        """
        Fit the ZCA whitening transform.

        Args:
            x: array-like of shape (n_samples, n_features)
            y: ignored

        Returns:
            self
        """
        # Use PCA with whitening
        self.pca_ = PCA(n_components=self.n_components, whiten=True, random_state=self.random_state)
        self.pca_.fit(x)
        self.zca_version_ = 2  # objects fitted before the whitening fix lack this (see transform)
        return self

    def transform(self, x):
        """
        Apply ZCA whitening transform.

        Args:
            x: array-like of shape (n_samples, n_features)

        Returns:
            array-like of shape (n_samples, n_features)
        """
        if not hasattr(self, "pca_"):
            raise ValueError("ZCAWhitening must be fitted before transform")
        if not hasattr(self, "zca_version_"):
            # Unpickled from xdflow < 0.3, whose transform returned its input; a downstream model fitted on that
            # output would silently get different features now.
            raise ValueError(
                "This ZCAWhitening was fitted with xdflow < 0.3, which did not whiten; refit the pipeline."
            )

        # Whiten in PCA space, then rotate back to feature space (no rescaling): U diag(1/sqrt(l)) U^T (x - mean)
        x_whitened = self.pca_.transform(x)
        return x_whitened @ self.pca_.components_

    def inverse_transform(self, x):
        """
        Apply inverse ZCA transform.

        Args:
            x: array-like of shape (n_samples, n_features)

        Returns:
            array-like of shape (n_samples, n_features)
        """
        if not hasattr(self, "pca_"):
            raise ValueError("ZCAWhitening must be fitted before inverse_transform")

        # Rotate back into PCA space, then un-whiten and restore the mean
        return self.pca_.inverse_transform(np.asarray(x) @ self.pca_.components_.T)


class ZCATransform(SKLearnTransformer):
    """
    ZCA (Zero Component Analysis) whitening transform.

    This transform performs PCA with whitening and then inverts the PCA transform
    to return to the original space, effectively implementing ZCA whitening.

    Parameters:
        n_components: int, float or None, default=None
            Number of components to keep. If None, all components are kept.
            If float in (0, 1], it represents the fraction of variance to keep.
        random_state: int, RandomState instance or None, default=None
            Random state for reproducibility.
        sample_dim: str, default="trial"
            The dimension that corresponds to samples.
        output_dim_name: str, default="channel"
            The name for the output dimension (same as input for ZCA).
        sel: dict, optional
            Selection to apply before transforming.
        drop_sel: dict, optional
            Drop selection to apply before transforming.
    """

    def __init__(
        self,
        n_components=None,
        random_state=None,
        sample_dim="trial",
        output_dim_name="channel",
        sel=None,
        drop_sel=None,
    ):
        # Validate n_components - require it to be specified
        if n_components is None:
            raise ValueError("Must specify n_components.")

        super().__init__(
            estimator_cls=ZCAWhitening,
            sample_dim=sample_dim,
            output_dim_name=output_dim_name,
            sel=sel,
            drop_sel=drop_sel,
            n_components=n_components,
            random_state=random_state,
        )
