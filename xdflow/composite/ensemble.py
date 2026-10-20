"""
Ensemble predictor for combining multiple predictors with weighted averaging.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xdflow.composite.base import CompositeTransform, TransformStep, _configure_transform_for_inference
from xdflow.core.base import Predictor, Transform, _init_param_names
from xdflow.core.data_container import DataContainer


def _identity_transform(x: float) -> float:
    """
    Identity function for score transformation.

    Args:
        x: Input value

    Returns:
        The input value unchanged
    """
    return x


def _entropy_uncertainty_components(probs: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute aleatoric and epistemic entropy-based uncertainty components for an ensemble.

    probs:   (n_models, n_samples, n_classes)
    weights: (n_models,) non-negative, sum to 1

    Returns:
        A: aleatoric component E_w[H(p_i)] with shape (n_samples,)
        B: epistemic component H(E_w[p_i]) - E_w[H(p_i)] with shape (n_samples,)
    """
    eps = 1e-10

    # Entropy per model
    h_ind = -np.sum(probs * np.log(probs + eps), axis=-1)  # (n_models, n_samples)
    aleatoric = np.tensordot(weights, h_ind, axes=(0, 0))  # (n_samples,)

    # Entropy of weighted mean
    mean_probs = np.tensordot(weights, probs, axes=(0, 0))  # (n_samples, n_classes)
    h_mean = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)  # (n_samples,)

    epistemic = h_mean - aleatoric
    return aleatoric, epistemic


def _fit_member(transform, container, **kwargs):
    """
    Helper function for parallel fitting of ensemble members.

    Args:
        transform: The transform to fit
        container: DataContainer to process
        **kwargs: Additional arguments

    Returns:
        Fitted predictor
    """
    return transform.fit(container, **kwargs)


def _predict_member(transform, container, **kwargs):
    """
    Helper function for parallel prediction on ensemble members.

    Args:
        transform: The fitted transform
        container: DataContainer to predict on
        **kwargs: Additional arguments

    Returns:
        Predictions from the member
    """
    return transform.predict(container, **kwargs)


def _predict_proba_member(transform, container, **kwargs):
    """
    Helper function for parallel probability prediction on ensemble members.

    Args:
        transform: The fitted transform
        container: DataContainer to predict on
        **kwargs: Additional arguments

    Returns:
        Probability predictions from the member
    """
    return transform.predict_proba(container, **kwargs)


def _predict_member_with_encoded(
    transform: Transform,
    encoded_container: DataContainer,
    original_container: DataContainer,
    *,
    proba: bool,
    kwargs: dict,
) -> DataContainer:
    """
    Helper that prefers encoded entrypoints when available, otherwise falls back to standard predict APIs.
    """
    predictive_transform = _as_predictive_transform(transform)
    if proba:
        predict_fn = getattr(transform, "_predict_proba_from_encoded", None)
        if callable(predict_fn):
            return predict_fn(encoded_container.copy(deep=True), **kwargs)
        return predictive_transform.predict_proba(original_container.copy(deep=True), **kwargs)

    predict_fn = getattr(transform, "_predict_from_encoded", None)
    if callable(predict_fn):
        return predict_fn(encoded_container.copy(deep=True), **kwargs)
    return predictive_transform.predict(original_container.copy(deep=True), **kwargs)


def _as_predictive_transform(transform: Transform) -> Predictor | CompositeTransform:
    if isinstance(transform, Predictor):
        return transform
    if isinstance(transform, CompositeTransform) and transform.is_predictor:
        return transform
    raise TypeError(f"Transform {type(transform).__name__} does not expose prediction methods.")


@dataclass
class EnsembleMember:
    """Represents a member of an ensemble with its name, predictor, and weight.

    Attributes:
        name: The name of the ensemble member.
        transform: The transform object.
        weight: The weight for this member in the ensemble.
    """

    name: str
    transform: Transform
    weight: float = 1.0

    @property
    def predictor(self) -> Predictor:
        """Return the predictive component for this member's transform."""
        transform = self.transform
        if isinstance(transform, Predictor):
            return transform

        if isinstance(transform, CompositeTransform):
            predictive = transform.predictive_transform
            if isinstance(predictive, Predictor):
                return predictive

        raise ValueError(f"Transform '{self.name}' does not expose a predictive interface (got {type(transform)}).")


class EnsemblePredictor(CompositeTransform, Predictor):
    """
    An ensemble predictor that combines multiple predictors using weighted averaging.

    This predictor applies multiple child predictors to the same input and combines their
    outputs using weighted averaging. It supports various weighting strategies for
    combining predictor outputs.

    Features:
    - Multiple weighting strategies (uniform, score-based, custom)
    - Parallel execution support
    - Score-based weighting with customizable scoring functions
    - Proper validation and error handling
    - Both prediction and probability prediction ensemble

    Args:
        members: List of (name, predictor) tuples, EnsembleMember objects, or TransformStep objects
        sample_dim: Name of the sample dimension
        target_coord: Name of the target coordinate
        encoder: Optional label encoder for the predictor
        weights: Optional explicit weights for the members (overrides weighting_strategy)
        weighting_strategy: Strategy for determining weights ('uniform', 'score_based', 'custom')
        scoring_func: Function to use for score-based weighting (default: accuracy_score)
        scoring_transform_func: Function to transform scores before using as weights
        normalize_weights: Whether to normalize weights to sum to 1
        normalize_outputs: Whether to normalize final ensemble outputs
        n_jobs: Number of parallel jobs for execution
        calibration_container: Optional container for score-based weighting calibration
    """

    def __init__(
        self,
        members: list[tuple[str, Predictor] | EnsembleMember | TransformStep | Transform],
        sample_dim: str,
        target_coord: str,
        encoder: LabelEncoder | None = None,
        weights: list[float] | None = None,
        weighting_strategy: Literal["uniform", "score_based", "custom"] = "uniform",
        scoring_func: Callable = accuracy_score,
        scoring_transform_func: Callable[[float], float] | None = None,
        normalize_weights: bool = True,
        normalize_outputs: bool = True,
        n_jobs: int = 1,
        calibration_container: DataContainer | None = None,
        proba: bool = False,
        sel: dict | None = None,
        drop_sel: dict | None = None,
        **kwargs,
    ):
        """
        Initialize EnsemblePredictor with ensemble members and configuration.

        Args:
            members: List of ensemble members in various formats
            sample_dim: Name of the sample dimension
            target_coord: Name of the target coordinate
            encoder: Optional label encoder for the predictor
            weights: Optional explicit weights (overrides weighting_strategy)
            weighting_strategy: How to determine member weights
            scoring_func: Function for score-based weighting evaluation
            scoring_transform_func: Transform function applied to scores (defaults to identity function)
            normalize_weights: Whether to normalize weights to sum to 1
            normalize_outputs: Whether to normalize final outputs
            n_jobs: Number of parallel jobs to use
            calibration_container: Data for score-based weight calibration
            proba: Whether to return probabilities by default
            sel: Optional selection to apply before predicting
            drop_sel: Optional drop selection to apply before predicting
        """

        unexpected = set(kwargs) - _init_param_names(Predictor)
        if unexpected:
            raise TypeError(f"{type(self).__name__} got unexpected keyword argument(s): {sorted(unexpected)}")

        # Set default for scoring_transform_func
        if scoring_transform_func is None:
            scoring_transform_func = _identity_transform

        # Normalize inputs to EnsembleMember objects
        self.members: list[EnsembleMember] = []
        initial_weights = []
        for i, member in enumerate(members):
            if isinstance(member, EnsembleMember):
                self.members.append(member)
                initial_weights.append(member.weight)
            elif isinstance(member, TransformStep) or isinstance(member, tuple) or isinstance(member, Transform):
                if isinstance(member, tuple):
                    name, transform = member
                elif isinstance(member, TransformStep):
                    name = member.name
                    transform = member.transform
                else:
                    name = f"member_{i}"
                    transform = member

                if isinstance(transform, CompositeTransform):
                    if not transform.is_predictor:
                        raise ValueError(f"Member '{name}' is/contains a CompositeTransform that is not a predictor")
                    self.members.append(EnsembleMember(name=name, transform=transform))
                else:
                    if not isinstance(transform, Predictor):
                        raise ValueError(f"Member '{name}' must be/contain a Predictor, got {type(transform)}")
                    self.members.append(EnsembleMember(name=name, transform=transform))  # predictor same as transform
                initial_weights.append(1.0)
            else:
                raise ValueError(f"Invalid member type: {type(member)}")

        if not self.members:
            raise ValueError("At least one ensemble member must be provided")

        # Store configuration
        self.weighting_strategy = weighting_strategy
        self.scoring_func = scoring_func
        self.scoring_transform_func = scoring_transform_func
        self.normalize_weights = normalize_weights
        self.normalize_outputs = normalize_outputs
        self.n_jobs = n_jobs
        self.calibration_container = calibration_container

        # Set weights
        if weights is not None:
            if len(weights) != len(self.members):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of members ({len(self.members)})"
                )
            self._set_weights(weights)
        else:
            # Use initial weights from members or uniform weights
            self._set_weights(initial_weights)

        # Determine if this is a classifier based on first member
        first_predictor = self.members[0].predictor
        is_classifier = first_predictor.is_classifier

        # Initialize parent Predictor
        super().__init__(
            sample_dim=sample_dim,
            target_coord=target_coord,
            is_classifier=is_classifier,
            encoder=encoder,
            proba=proba,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self._validate_composition()
        self._ensure_shared_encoders()  # Handle pre-fitted models at initialization

        # set is_fitted
        self._is_fitted = self.check_is_fitted()

    def check_is_fitted(self) -> bool:
        """Checks if the ensemble is fitted if all members are fitted."""
        for member in self.members:
            if not getattr(member.predictor, "_is_fitted", False):
                return False
        return True

    @property
    def children(self) -> list[Transform]:
        """Returns the transform objects from the ensemble members."""
        return [member.transform for member in self.members]

    def get_transform_from_name(self, name: str) -> Transform:
        """Returns the member transform with the given name (used for nested set_params)."""
        for member in self.members:
            if member.name == name:
                return member.transform
        raise KeyError(f"No ensemble member named '{name}'. Members: {[m.name for m in self.members]}")

    @property
    def is_predictor(self) -> bool:
        return True

    @property
    def predictive_transform(self) -> Predictor | None:
        return self

    def _validate_composition(self):
        """Validates that all ensemble members have compatible structure."""

        # Check that member names are unique
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("Ensemble member names must be unique")

        # Check predictor compatibility - all members are predictors by construction
        ref_predictor = self.members[0].predictor
        for member in self.members[1:]:
            pred = member.predictor
            if pred.sample_dim != ref_predictor.sample_dim:
                raise ValueError(f"Inconsistent sample_dim: '{ref_predictor.sample_dim}' vs '{pred.sample_dim}'")
            if pred.target_coord != ref_predictor.target_coord:
                raise ValueError(f"Inconsistent target_coord: '{ref_predictor.target_coord}' vs '{pred.target_coord}'")
            if pred.is_classifier != ref_predictor.is_classifier:
                raise ValueError(
                    f"Mixed classifier/regressor predictors: {ref_predictor.is_classifier} vs {pred.is_classifier}"
                )

    def _ensure_shared_encoders(self):
        """Ensures all ensemble members use the same encoder for consistent class handling."""
        if not self.is_classifier:
            return

        # Check for fitted encoders and validate consistency
        fitted_encoders = []
        for member in self.members:
            if member.predictor.encoder is not None and hasattr(member.predictor.encoder, "classes_"):
                fitted_encoders.append((member.name, member.predictor.encoder))

        # If we have fitted encoders, validate they all have the same classes
        if fitted_encoders:
            reference_name, reference_encoder = fitted_encoders[0]
            reference_classes = set(reference_encoder.classes_)

            for member_name, encoder in fitted_encoders[1:]:
                member_classes = set(encoder.classes_)
                if member_classes != reference_classes:
                    raise ValueError(
                        f"Inconsistent encoder classes: member '{reference_name}' has classes "
                        f"{sorted(reference_classes)} but member '{member_name}' has classes "
                        f"{sorted(member_classes)}. All ensemble members must be trained on the same classes."
                    )

            # All encoders have consistent classes - just use the first one as the ensemble encoder
            self.encoder = reference_encoder
            # Propagate the shared encoder reference to all members for consistency
            for member in self.members:
                member.predictor.set_encoder(self.encoder)

    def prepare_for_inference(self, *, set_n_jobs_single: bool = True) -> None:
        """
        Disable training-time options that slow down per-request inference.

        Args:
            set_n_jobs_single: When True, force single-threaded execution for members.
        """
        if set_n_jobs_single and hasattr(self, "n_jobs"):
            try:
                self.n_jobs = 1
            except AttributeError:
                pass

        visited: set[int] = {id(self)}
        for member in self.members:
            _configure_transform_for_inference(
                member.transform,
                set_n_jobs_single=set_n_jobs_single,
                visited=visited,
            )

    def _set_weights(self, weights: list[float]):
        """Sets the weights for ensemble members."""
        if len(weights) != len(self.members):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of members ({len(self.members)})")

        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        if self.normalize_weights:
            weights = [w / total_weight for w in weights]

        for member, weight in zip(self.members, weights, strict=True):
            member.weight = weight

        # Update stored weights to follow cloning policy
        self.weights = [member.weight for member in self.members]

    def _compute_weights(self, calibration_container: DataContainer, **kwargs):
        """Computes weights based on the configured weighting strategy."""

        if self.weighting_strategy == "uniform":
            # Uniform weights (already set in __init__)
            return
        elif self.weighting_strategy == "score_based":
            if calibration_container is None:
                raise ValueError("Calibration container required for score-based weighting")
            self._compute_score_based_weights(calibration_container, **kwargs)
        elif self.weighting_strategy == "custom":
            # Custom weights should already be set via the weights parameter
            return
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

    def _compute_score_based_weights(self, calibration_container: DataContainer, **kwargs):
        """Computes weights based on member performance on calibration data."""

        # Get target data for scoring (all members are predictors by construction)
        first_predictor = self.members[0].predictor
        if (
            not hasattr(calibration_container.data, "coords")
            or first_predictor.target_coord not in calibration_container.data.coords
        ):
            raise ValueError(f"Calibration container must have target coordinate '{first_predictor.target_coord}'")

        # Members' public predict() returns decoded labels, so score against the raw target values
        target_data = calibration_container.data.coords[first_predictor.target_coord].values

        # Compute scores for each member
        scores = []
        for member in self.members:
            try:
                # Get predictions from this member's transform
                predictions = _as_predictive_transform(member.transform).predict(calibration_container, **kwargs)
                pred_values = predictions.data.values

                # Ensure both target and prediction data are compatible for scoring
                if len(target_data) != len(pred_values):
                    warnings.warn(
                        f"Length mismatch for member '{member.name}': target={len(target_data)}, pred={len(pred_values)}",
                        UserWarning,
                        stacklevel=2,
                    )
                    scores.append(0.0)
                    continue

                # Compute score
                score = self.scoring_func(target_data, pred_values)
                transformed_score = self.scoring_transform_func(score)
                scores.append(max(0.0, transformed_score))  # Ensure non-negative scores

            except Exception as e:
                warnings.warn(f"Failed to compute score for member '{member.name}': {e}", UserWarning, stacklevel=2)
                scores.append(0.0)

        # Handle case where all scores are zero
        if all(score == 0.0 for score in scores):
            warnings.warn("All scores are zero, falling back to uniform weighting", UserWarning, stacklevel=2)
            scores = [1.0] * len(scores)

        # Set weights based on scores
        self._set_weights(scores)

    def _fit(self, container: DataContainer, **kwargs) -> "EnsemblePredictor":
        """
        Core fit implementation invoked by Predictor.fit after selection/encoding.

        Args:
            container: Possibly encoded DataContainer produced by Predictor.fit
            **kwargs: Additional context/parameters passed through
        """
        original_container = container
        if self.is_classifier and f"{self.target_coord}_orig" in container.data.coords:
            original_da = self._reset_target_coord(container.data)
            original_container = DataContainer(original_da)

        if self.is_classifier and self.encoder is not None:
            for member in self.members:
                member.predictor.set_encoder(self.encoder)

        if self.n_jobs != 1:
            fitted_transforms = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_member)(member.transform, original_container.copy(deep=True), **kwargs)
                for member in self.members
            )
            for member, fitted in zip(self.members, fitted_transforms, strict=True):
                member.transform = fitted
                # Ensure the fitted transform still exposes a predictor interface
                _ = member.predictor
        else:
            for member in self.members:
                fitted = member.transform.fit(original_container.copy(deep=True), **kwargs)
                member.transform = fitted
                _ = member.predictor

        self._ensure_shared_encoders()

        calibration_data = self.calibration_container or original_container
        if self.weighting_strategy == "score_based":
            self._compute_weights(calibration_data, **kwargs)

        return self

    def transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms the data using all ensemble members.
        """
        warnings.warn(
            "using predict instead of transform. transform should not be needed/used for EnsemblePredictor",
            UserWarning,
            stacklevel=2,
        )
        return self.predict(container, **kwargs)

    def _collect_member_predictions(
        self,
        *,
        encoded_container: DataContainer,
        original_container: DataContainer,
        proba: bool,
        **kwargs,
    ) -> list[DataContainer]:
        """
        Gather per-member prediction outputs, optionally in parallel.
        Prefers encoded entrypoints to avoid duplicate encode/decode cycles.
        """
        if self.n_jobs != 1:
            tasks = [
                delayed(_predict_member_with_encoded)(
                    member.transform,
                    encoded_container.copy(deep=True),
                    original_container.copy(deep=True),
                    proba=proba,
                    kwargs=dict(kwargs),
                )
                for member in self.members
            ]
            return self._in_input_order(Parallel(n_jobs=self.n_jobs)(tasks), original_container)

        outputs: list[DataContainer] = []
        for member in self.members:
            outputs.append(
                _predict_member_with_encoded(
                    member.transform,
                    encoded_container.copy(deep=True),
                    original_container.copy(deep=True),
                    proba=proba,
                    kwargs=dict(kwargs),
                )
            )
        return self._in_input_order(outputs, original_container)

    def _in_input_order(self, outputs: list[DataContainer], container: DataContainer) -> list[DataContainer]:
        """Member outputs in the input's trial order: members are aggregated by position, and a member pipeline may
        reorder trials."""
        trials = container.data[self.sample_dim].values
        return [
            DataContainer(output.data.sel({self.sample_dim: trials}))
            if self.sample_dim in output.data.indexes
            and not np.array_equal(output.data[self.sample_dim].values, trials)
            else output
            for output in outputs
        ]

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predict labels using ensemble, leveraging shared encoding optimization.

        Args:
            container: DataContainer to predict on
            **kwargs: Additional context/parameters passed through

        Returns:
            DataContainer with predictions
        """
        # Apply selection using base helper
        container = self._apply_selection(container)

        # Encode target coordinate if classifier
        if self.is_classifier:
            if self.encoder is None or not hasattr(self.encoder, "classes_"):
                raise ValueError(f"{self.__class__.__name__} requires a fitted encoder before calling predict.")
            encoded_da = self._encode_target_coord(container.data, missing_ok=True)
            encoded_container = DataContainer(encoded_da)
        else:
            encoded_container = container

        # Delegate to encoded entry point for ensemble logic
        return self._predict_from_encoded(encoded_container, original_container=container, **kwargs)

    def _predict_from_encoded(
        self, container: DataContainer, *, original_container: DataContainer | None = None, **kwargs
    ) -> DataContainer:
        """
        Core ensemble prediction logic operating on an encoded container.

        Fans out to members (preferring their _predict_from_encoded fast paths),
        aggregates decoded results, and returns a DataContainer with decoded predictions.

        Args:
            container: Encoded DataContainer
            original_container: Optional decoded container (lazily created if needed)
            **kwargs: Additional context/parameters

        Returns:
            DataContainer with decoded predictions
        """
        encoded_container = container

        # Lazily create decoded view only if needed (for fallback members)
        if original_container is None:
            if self.is_classifier and f"{self.target_coord}_orig" in container.data.coords:
                original_da = self._reset_target_coord(container.data)
                original_container = DataContainer(original_da)
            else:
                original_container = encoded_container

        # Fan out to members (preferring encoded fast paths)
        predictions = self._collect_member_predictions(
            encoded_container=encoded_container,
            original_container=original_container,
            proba=False,
            **kwargs,
        )

        # Aggregate decoded predictions
        ensemble_prediction = self._ensemble_predictions([pred.data for pred in predictions])

        # Build output DataContainer with decoded predictions
        output_coords = self._get_output_coords(original_container.data)
        output_da = xr.DataArray(
            ensemble_prediction.values.reshape(-1),
            dims=[self.sample_dim],
            coords=output_coords,
            attrs=original_container.data.attrs,
            name="prediction",
        )
        return DataContainer(output_da)

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predict class probabilities using ensemble, leveraging shared encoding optimization.

        Args:
            container: DataContainer to predict on
            **kwargs: Additional context/parameters passed through

        Returns:
            DataContainer with class probabilities
        """
        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier "
                "(is_classifier=False) so should not call the 'predict_proba' method."
            )

        # Apply selection using base helper
        container = self._apply_selection(container)

        # Encode target coordinate
        if self.encoder is None or not hasattr(self.encoder, "classes_"):
            raise ValueError(f"{self.__class__.__name__} requires a fitted encoder before calling predict_proba.")
        encoded_da = self._encode_target_coord(container.data, missing_ok=True)
        encoded_container = DataContainer(encoded_da)

        # Delegate to encoded entry point for ensemble logic
        return self._predict_proba_from_encoded(encoded_container, original_container=container, **kwargs)

    def predict_proba_with_uncertainty_components(
        self, container: DataContainer, **kwargs
    ) -> tuple[DataContainer, DataContainer, DataContainer]:
        """
        Predict class probabilities and entropy-based aleatoric/epistemic uncertainty components.

        The two components are:

            A = E_w[H(p_i)]                     (aleatoric)
            B = H(E_w[p_i]) - E_w[H(p_i)]       (epistemic)

        Args:
            container: DataContainer to predict on.
            **kwargs: Additional context/parameters passed through to member predictors.

        Returns:
            Tuple of:
                - DataContainer with class probabilities (same as predict_proba)
                - DataContainer with aleatoric uncertainty (A), one score per sample
                - DataContainer with epistemic uncertainty (B), one score per sample
        """
        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier "
                "(is_classifier=False) so should not call 'predict_proba_with_uncertainty_components'."
            )

        # Apply selection using base helper
        container = self._apply_selection(container)

        # Encode target coordinate
        if self.encoder is None or not hasattr(self.encoder, "classes_"):
            raise ValueError(
                f"{self.__class__.__name__} requires a fitted encoder before calling "
                "predict_proba_with_uncertainty_components."
            )
        encoded_da = self._encode_target_coord(container.data, missing_ok=True)
        encoded_container = DataContainer(encoded_da)

        # Collect per-member probability predictions (aligned via shared encoder)
        prob_predictions = self._collect_member_predictions(
            encoded_container=encoded_container,
            original_container=container,
            proba=True,
            **kwargs,
        )
        prob_data_arrays = [pred.data for pred in prob_predictions]
        if not prob_data_arrays:
            raise ValueError("No probability outputs to compute uncertainty from")

        # Aggregate probability outputs (same as predict_proba)
        ensemble_proba = self._ensemble_proba(prob_data_arrays)

        # Extract class labels and encode them for alignment
        class_coord = ensemble_proba.coords.get("class")
        if class_coord is not None:
            class_labels = np.asarray(class_coord.values)
            if self.encoder is None:
                raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
            encoded_classes = self.encoder.transform(class_labels)
        else:
            encoded_classes = np.arange(ensemble_proba.shape[-1])

        aligned_proba = self._align_proba_to_global(ensemble_proba.values, encoded_classes)

        # Build probability DataContainer
        output_coords = self._get_output_coords(container.data)
        if self.encoder is None:
            raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
        output_coords["class"] = self.encoder.classes_

        proba_da = xr.DataArray(
            aligned_proba,
            dims=(self.sample_dim, "class"),
            coords=output_coords,
            attrs=container.data.attrs,
        )
        proba_container = DataContainer(proba_da)

        # Compute entropy-based aleatoric and epistemic uncertainty components
        sample_dim = self.sample_dim
        class_dim = "class"
        probs = np.stack(
            [da.transpose(sample_dim, class_dim).values for da in prob_data_arrays],
            axis=0,
        )

        weights = np.asarray(self.weights, dtype=float)
        if weights.shape[0] != probs.shape[0]:
            raise ValueError(
                f"Number of weights ({weights.shape[0]}) must match number of members ({probs.shape[0]}) "
                "when computing uncertainty."
            )
        weights = weights / weights.sum()

        aleatoric_scores, epistemic_scores = _entropy_uncertainty_components(probs, weights)

        # Build uncertainty DataContainers with one score per sample
        uncertainty_coords = self._get_output_coords(container.data)
        aleatoric_da = xr.DataArray(
            aleatoric_scores,
            dims=[self.sample_dim],
            coords=uncertainty_coords,
            attrs=container.data.attrs,
            name="aleatoric_uncertainty",
        )
        epistemic_da = xr.DataArray(
            epistemic_scores,
            dims=[self.sample_dim],
            coords=uncertainty_coords,
            attrs=container.data.attrs,
            name="epistemic_uncertainty",
        )

        return proba_container, DataContainer(aleatoric_da), DataContainer(epistemic_da)

    def predict_proba_with_std(
        self, container: DataContainer, *, return_stderr: bool = False, **kwargs
    ) -> tuple[DataContainer, DataContainer]:
        """
        Predict class probabilities along with the standard deviation or standard error across ensemble members.

        Args:
            container: DataContainer to predict on.
            return_stderr: When True, return standard error instead of standard deviation.
            **kwargs: Additional context/parameters passed through to member predictors.

        Returns:
            Tuple of:
                - DataContainer with class probabilities (same as predict_proba)
                - DataContainer with standard deviation (or standard error) per sample/class
        """
        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier "
                "(is_classifier=False) so should not call 'predict_proba_with_std'."
            )

        # Apply selection using base helper
        container = self._apply_selection(container)

        # Encode target coordinate
        if self.encoder is None or not hasattr(self.encoder, "classes_"):
            raise ValueError(f"{self.__class__.__name__} requires a fitted encoder before calling predict_proba.")
        encoded_da = self._encode_target_coord(container.data, missing_ok=True)
        encoded_container = DataContainer(encoded_da)

        # Collect per-member probability predictions (aligned via shared encoder)
        prob_predictions = self._collect_member_predictions(
            encoded_container=encoded_container,
            original_container=container,
            proba=True,
            **kwargs,
        )
        prob_data_arrays = [pred.data for pred in prob_predictions]
        if not prob_data_arrays:
            raise ValueError("No probability outputs to compute uncertainty from")

        # Aggregate probability outputs (same as predict_proba)
        ensemble_proba = self._ensemble_proba(prob_data_arrays)

        # Extract class labels and encode them for alignment
        class_coord = ensemble_proba.coords.get("class")
        if class_coord is not None:
            class_labels = np.asarray(class_coord.values)
            if self.encoder is None:
                raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
            encoded_classes = self.encoder.transform(class_labels)
        else:
            encoded_classes = np.arange(ensemble_proba.shape[-1])

        aligned_proba = self._align_proba_to_global(ensemble_proba.values, encoded_classes)

        # Build probability DataContainer
        output_coords = self._get_output_coords(container.data)
        if self.encoder is None:
            raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
        output_coords["class"] = self.encoder.classes_

        proba_da = xr.DataArray(
            aligned_proba,
            dims=(self.sample_dim, "class"),
            coords=output_coords,
            attrs=container.data.attrs,
        )
        proba_container = DataContainer(proba_da)

        # Compute standard deviation (or standard error) across members
        probs = np.stack([da.values for da in prob_data_arrays], axis=0)
        std = probs.std(axis=0)
        if return_stderr:
            std = std / np.sqrt(probs.shape[0])

        std_da = xr.DataArray(
            std,
            dims=(self.sample_dim, "class"),
            coords=output_coords,
            attrs=container.data.attrs,
            name="proba_std_error" if return_stderr else "proba_std",
        )
        return proba_container, DataContainer(std_da)

    def _predict_proba_from_encoded(
        self, container: DataContainer, *, original_container: DataContainer | None = None, **kwargs
    ) -> DataContainer:
        """
        Core ensemble probability prediction logic operating on an encoded container.

        Fans out to members, aggregates probability outputs, aligns to global encoder classes,
        and returns a DataContainer with properly aligned probabilities.

        Args:
            container: Encoded DataContainer
            original_container: Optional decoded container (lazily created if needed)
            **kwargs: Additional context/parameters

        Returns:
            DataContainer with class probabilities aligned to encoder.classes_
        """
        encoded_container = container

        # Lazily create decoded view only if needed
        if original_container is None:
            if f"{self.target_coord}_orig" in container.data.coords:
                original_da = self._reset_target_coord(container.data)
                original_container = DataContainer(original_da)
            else:
                original_container = encoded_container

        # Fan out to members (preferring encoded fast paths)
        prob_predictions = self._collect_member_predictions(
            encoded_container=encoded_container,
            original_container=original_container,
            proba=True,
            **kwargs,
        )

        # Aggregate probability outputs
        prob_data_arrays = [pred.data for pred in prob_predictions]
        ensemble_proba = self._ensemble_proba(prob_data_arrays)

        # Extract class labels and encode them for alignment
        class_coord = ensemble_proba.coords.get("class")
        if class_coord is not None:
            class_labels = np.asarray(class_coord.values)
            if self.encoder is None:
                raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
            encoded_classes = self.encoder.transform(class_labels)
        else:
            encoded_classes = np.arange(ensemble_proba.shape[-1])

        # Align to global encoder classes using base helper
        aligned_proba = self._align_proba_to_global(ensemble_proba.values, encoded_classes)

        # Build output DataContainer
        output_coords = self._get_output_coords(original_container.data)
        if self.encoder is None:
            raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
        output_coords["class"] = self.encoder.classes_

        output_da = xr.DataArray(
            aligned_proba,
            dims=(self.sample_dim, "class"),
            coords=output_coords,
            attrs=original_container.data.attrs,
        )
        return DataContainer(output_da)

    def _predict(self, data: xr.DataArray, **kwargs) -> np.ndarray:
        """
        Satisfy the abstract method requirement from Predictor base class.

        This method is not used in the normal flow (predict() overrides the full path),
        but must exist to satisfy the ABC contract. Delegates to _predict_from_encoded.

        Args:
            data: Encoded DataArray
            **kwargs: Additional parameters

        Returns:
            Encoded predictions as 1D numpy array
        """
        # Wrap in DataContainer and delegate to encoded path
        result_container = self._predict_from_encoded(DataContainer(data), **kwargs)
        predictions = result_container.data.values.reshape(-1)

        # Return encoded predictions as expected by base class
        if self.is_classifier:
            if self.encoder is None:
                raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier predictions.")
            return self.encoder.transform(predictions)
        return predictions

    def _predict_proba(self, data: xr.DataArray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Satisfy the abstract method requirement from Predictor base class.

        This method is not used in the normal flow (predict_proba() overrides the full path),
        but must exist to satisfy the ABC contract. Delegates to _predict_proba_from_encoded.

        Args:
            data: Encoded DataArray
            **kwargs: Additional parameters

        Returns:
            Tuple of (probabilities, encoded_classes)
        """
        # Wrap in DataContainer and delegate to encoded path
        result_container = self._predict_proba_from_encoded(DataContainer(data), **kwargs)

        # Extract probabilities and class labels
        probabilities = result_container.data.values
        class_labels = result_container.data.coords["class"].values
        if self.encoder is None:
            raise ValueError(f"{self.__class__.__name__} requires an encoder for classifier probabilities.")
        encoded_classes = self.encoder.transform(class_labels)

        return probabilities, encoded_classes

    def _ensemble_proba(self, prob_outputs: list[xr.DataArray]) -> xr.DataArray:
        """
        Core ensemble logic for combining probability outputs using weighted averaging.

        Args:
            prob_outputs: List of probability DataArrays from ensemble members

        Returns:
            Ensemble probability DataArray
        """

        if not prob_outputs:
            raise ValueError("No probability outputs to ensemble")

        if len(prob_outputs) != len(self.members):
            raise ValueError(
                f"Number of outputs ({len(prob_outputs)}) must match number of members ({len(self.members)})"
            )

        # Weighted average of probability outputs
        weights = self.weights
        weighted_outputs = [w * output for w, output in zip(weights, prob_outputs, strict=True)]
        ensemble_output = weighted_outputs[0]
        for weighted_output in weighted_outputs[1:]:
            ensemble_output = ensemble_output + weighted_output

        # Normalize if requested
        if self.normalize_outputs:
            ensemble_output = self._normalize_output(ensemble_output)

        return ensemble_output

    def _ensemble_predictions(self, predictions: list[xr.DataArray]) -> xr.DataArray:
        """
        Ensemble discrete predictions using weighted voting.

        Args:
            predictions: List of prediction DataArrays from ensemble members

        Returns:
            Ensemble prediction DataArray
        """

        if not predictions:
            raise ValueError("No predictions to ensemble")

        weights = self.weights

        if not self.is_classifier:
            # Continuous targets: weighted mean instead of voting
            averaged = np.average(np.stack([pred.values for pred in predictions]), axis=0, weights=weights)
            return predictions[0].copy(data=averaged)

        # For discrete predictions, we need to do weighted voting
        # Get unique classes across all predictions
        all_classes = set()
        for pred in predictions:
            all_classes.update(pred.values.flatten())
        all_classes = sorted(all_classes)

        # Convert predictions to one-hot and weight them
        sample_shape = predictions[0].shape
        n_classes = len(all_classes)

        # Initialize weighted vote matrix
        vote_matrix = np.zeros(sample_shape + (n_classes,))

        for pred, weight in zip(predictions, weights, strict=True):
            for i, class_val in enumerate(all_classes):
                vote_matrix[..., i] += weight * (pred.values == class_val)

        # Get class with highest weighted vote
        final_predictions = np.array([all_classes[idx] for idx in np.argmax(vote_matrix, axis=-1).flatten()])
        final_predictions = final_predictions.reshape(sample_shape)

        # Create output DataArray with same structure as input
        return xr.DataArray(
            final_predictions, dims=predictions[0].dims, coords=predictions[0].coords, attrs=predictions[0].attrs
        )

    def _normalize_output(self, output: xr.DataArray) -> xr.DataArray:
        """
        Normalizes output along the last dimension (typically classes).

        Args:
            output: DataArray to normalize

        Returns:
            Normalized DataArray
        """

        # Normalize along the last dimension
        if output.dims:
            last_dim = output.dims[-1]
            denom = output.sum(dim=[last_dim], keepdims=True)
            denom_data = denom.data

            # Avoid divide-by-zero; zeros would imply all members predicted zero probability.
            normalized = np.divide(
                output.data,
                denom_data,
                out=np.zeros_like(output.data),
                where=denom_data != 0,
            )

            output = xr.DataArray(normalized, dims=output.dims, coords=output.coords, attrs=output.attrs)

        return output

    def __repr__(self) -> str:
        members_str = ", ".join([f"'{member.name}': {member.transform.__class__.__name__}" for member in self.members])
        return f"EnsemblePredictor(members=[{members_str}], strategy='{self.weighting_strategy}')"
