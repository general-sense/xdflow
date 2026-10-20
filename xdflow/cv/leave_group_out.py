import warnings
from collections.abc import Callable, Hashable, Iterator

import numpy as np

from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator


class LeaveGroupOutValidator(CrossValidator):
    """
    Implements cross-validation by leaving one or more groups out at a time with optional holdout groups.

    This validator iterates through each unique group/groups, using it as the
    validation set once, while all other groups are used for training. This is
    critical for assessing how well a model generalizes to new, unseen groups.

    When n_splits is not set, one group is used for validation at a time.
    When n_splits is set, the groups are split into n_splits folds.
    """

    def __init__(
        self,
        group_coord: str,
        test_group_ids: list[Hashable] | None = None,
        validation_group_ids: list[Hashable] | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        n_splits: int | None = None,
        random_state: int = 0,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        scoring_needs_proba: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Leave-One-Group-Out cross-validator.

        Args:
            group_coord: Coordinate to group by.
            test_group_ids: List of group IDs to use as final holdout test set.
                             If None or empty, no holdout set is created.
            validation_group_ids: List of group IDs to use as validation set.
                             If None or empty, no validation set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            n_splits: Total number of splits to perform. If None, all groups are used.
            random_state: Random state for reproducibility. Used for shuffling groups if n_splits is set.
            exclude_intertrial_from_scoring: Whether to drop intertrial segments during evaluation.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            scoring_needs_proba=scoring_needs_proba,
            verbose=verbose,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
        )
        self.group_coord = group_coord
        self.test_group_ids = test_group_ids or []
        self.validation_group_ids = validation_group_ids or []
        self.n_splits = n_splits
        if self.n_splits is not None and self.n_splits < 2:
            raise ValueError("n_splits must be >= 2 if set.")
        self.random_state = random_state

        # validation_group_ids should not be in test_group_ids
        if set(self.validation_group_ids) & set(self.test_group_ids):
            raise ValueError("Validation group IDs and test group IDs must not overlap.")

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data by reserving specified groups as holdout test set.

        Args:
            container: DataContainer to split. Must have a 'group_coord' coordinate.

        Returns:
            Tuple of (train_val_indices, holdout_indices)

        Raises:
            ValueError: If session coordinate is missing
        """
        if self.group_coord not in container.data.coords:
            raise ValueError(f"LeaveGroupOutValidator requires a '{self.group_coord}' coordinate in the DataContainer.")
        # dim of group_coord must be "trial"
        if container.data.coords[self.group_coord].dims != ("trial",):
            raise ValueError(
                f"LeaveGroupOutValidator requires a '{self.group_coord}' coordinate with dimension 'trial' in the DataContainer."
            )

        all_trials = container.data.trial.values

        if not self.test_group_ids:
            # No holdout sessions - use all data for cross-validation
            return all_trials, np.array([])

        # Find trials belonging to holdout sessions
        group_values = container.data.coords[self.group_coord].values
        missing = [group for group in self.test_group_ids if not np.isin(group, group_values)]
        if missing:  # otherwise the trials meant to be held out are silently cross-validated
            raise ValueError(f"test_group_ids {missing} match no '{self.group_coord}' value in the data.")
        holdout_mask = np.isin(group_values, np.asarray(self.test_group_ids))
        holdout_indices = all_trials[holdout_mask]
        train_val_indices = all_trials[~holdout_mask]

        return train_val_indices, holdout_indices

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates splits where each fold uses specific groups as validation and others as training.
        If n_splits is not set, one group is used for validation at a time.
        If n_splits is set, the groups are split into n_splits folds.

        Args:
            container: DataContainer to split. Must have a 'group_coord' coordinate.
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) for each group

        Raises:
            ValueError: If group coordinate is missing
        """
        if self.group_coord not in container.data.coords:
            raise ValueError(f"LeaveGroupOutValidator requires a '{self.group_coord}' coordinate in the DataContainer.")

        # Select the relevant part of the container first
        cv_container = container.data.sel(trial=indices_to_split)

        # Find unique sessions in the subset
        unique_groups = np.unique(cv_container.coords[self.group_coord].values)

        # Determine which groups are candidates for validation
        if self.validation_group_ids:
            # Preserve user-specified order, but filter to those present
            candidate_groups = []
            for gid in self.validation_group_ids:
                if gid in unique_groups:
                    candidate_groups.append(gid)
                else:
                    warnings.warn(
                        f"Validation group {gid} for {self.group_coord} not found in data. Skipping.",
                        stacklevel=2,
                    )
        else:
            candidate_groups = list(unique_groups)

        group_values = cv_container.coords[self.group_coord].values
        trial_values = cv_container.trial.values

        # If n_splits is not set, default to leave-one-group-out
        if not self.n_splits:
            for group_id in candidate_groups:
                validation_mask = group_values == group_id
                validation_indices = trial_values[validation_mask]
                train_indices = trial_values[~validation_mask]
                yield train_indices, validation_indices
            return

        # n_splits is set: split candidate groups into that many folds
        n_splits = int(self.n_splits)
        if n_splits > len(candidate_groups):
            warnings.warn(
                f"n_splits ({n_splits}) is greater than the number of candidate groups ({len(candidate_groups)}). "
                f"Reducing n_splits to {len(candidate_groups)} so each fold has at least one group.",
                stacklevel=2,
            )
            n_splits = len(candidate_groups)

        # shuffle candidate groups
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(candidate_groups)

        # Use numpy to chunk groups as evenly as possible
        group_chunks = np.array_split(np.array(candidate_groups, dtype=object), n_splits)

        for chunk in group_chunks:
            # chunk is a numpy array; ensure list for isin
            chunk_groups = chunk.tolist()
            validation_mask = np.isin(group_values, chunk_groups)
            validation_indices = trial_values[validation_mask]
            train_indices = trial_values[~validation_mask]
            yield train_indices, validation_indices


class LeaveSessionOutValidator(LeaveGroupOutValidator):
    """
    Implements cross-validation by leaving one or more sessions out at a time with optional holdout sessions.

    This validator iterates through each unique session/sessions, using it as the
    validation set once, while all other sessions are used for training. This is
    critical for assessing how well a model generalizes to new, unseen sessions.

    Note: This is a convenience wrapper around LeaveGroupOutValidator with group_coord="session".
    """

    def __init__(
        self,
        test_session_ids: list[Hashable] | None = None,
        validation_session_ids: list[Hashable] | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        n_splits: int | None = None,
        random_state: int = 0,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        scoring_needs_proba: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Leave-Session-Out cross-validator.

        Args:
            test_session_ids: List of session IDs to use as final holdout test set.
                             If None or empty, no holdout set is created.
            validation_session_ids: List of session IDs to use as validation set.
                             If None or empty, no validation set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            n_splits: Total number of splits to perform. If None, all sessions are used.
            random_state: Random state for reproducibility. Used for shuffling sessions if n_splits is set.
            exclude_intertrial_from_scoring: Whether to drop intertrial segments during evaluation.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        # Delegate to LeaveGroupOutValidator with group_coord="session"
        super().__init__(
            group_coord="session",
            test_group_ids=test_session_ids,
            validation_group_ids=validation_session_ids,
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            n_splits=n_splits,
            random_state=random_state,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            scoring_needs_proba=scoring_needs_proba,
            verbose=verbose,
        )
        # Maintain backward-compatible attribute names for introspection
        self.test_session_ids = self.test_group_ids
        self.validation_session_ids = self.validation_group_ids


class LeaveAnimalOutValidator(LeaveGroupOutValidator):
    """
    Implements cross-validation by leaving one or more animals out at a time with optional holdout animals.

    This validator iterates through each unique animal/animals, using it as the
    validation set once, while all other animals are used for training. This is
    critical for assessing how well a model generalizes to new, unseen animals.

    Note: This is a convenience wrapper around LeaveGroupOutValidator with group_coord="animal".
    """

    def __init__(
        self,
        test_animal_ids: list[Hashable] | None = None,
        validation_animal_ids: list[Hashable] | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        n_splits: int | None = None,
        random_state: int = 0,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        scoring_needs_proba: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize Leave-Animal-Out cross-validator.

        Args:
            test_animal_ids: List of animal IDs to use as final holdout test set.
                             If None or empty, no holdout set is created.
            validation_animal_ids: List of animal IDs to use as validation set.
                             If None or empty, no validation set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            n_splits: Total number of splits to perform. If None, all animals are used.
            random_state: Random state for reproducibility. Used for shuffling sessions if n_splits is set.
            exclude_intertrial_from_scoring: Whether to drop intertrial segments during evaluation.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        # Delegate to LeaveGroupOutValidator with group_coord="animal"
        super().__init__(
            group_coord="animal",
            test_group_ids=test_animal_ids,
            validation_group_ids=validation_animal_ids,
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            n_splits=n_splits,
            random_state=random_state,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            scoring_needs_proba=scoring_needs_proba,
            verbose=verbose,
        )
