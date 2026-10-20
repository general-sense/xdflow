import warnings
from collections.abc import Callable, Hashable, Iterator
from typing import cast

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator


def _as_group_list(groups: list[Hashable] | Hashable | None) -> list[Hashable] | None:
    if groups is None:
        return None
    if isinstance(groups, list):
        return cast(list[Hashable], groups)
    return [groups]


def _combine_group_and_label_strings(groups: np.ndarray, labels: np.ndarray) -> np.ndarray | list[str]:
    """Build stratification labels from group and target labels using public NumPy APIs."""
    try:
        group_strings = groups.astype(str)
        label_strings = labels.astype(str)
        return cast(np.ndarray, np.char.add(np.char.add(group_strings, "_"), label_strings))
    except (TypeError, AttributeError):
        return [str(group) + "_" + str(label) for group, label in zip(groups, labels)]


class KFoldValidator(CrossValidator):
    """
    Implements cross-validation using a stratified K-Fold strategy with optional holdout set.

    This provides a concrete implementation of CrossValidator using
    scikit-learn's StratifiedKFold for balanced splits across classes.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        stratify_coord: str | None = None,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        scoring_needs_proba: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize KFold cross-validator.

        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use as holdout test set (0.0-1.0).
                      If None or 0, no holdout set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            stratify_coord: Optional coordinate name to use for stratified splits (train/val/holdout).
            exclude_intertrial_from_scoring: Whether to drop intertrial segments when evaluating folds/holdout.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            scoring_needs_proba=scoring_needs_proba,
            stratify_coord=stratify_coord,
            verbose=verbose,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
        )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into train/validation and holdout sets using stratified sampling.

        Args:
            container: DataContainer to split

        Returns:
            Tuple of (train_val_indices, holdout_indices)
        """
        all_trials = container.data.trial.values

        if self.test_size is None or self.test_size == 0:
            # No holdout set - use all data for cross-validation
            return all_trials, np.array([])

        # Determine if we can do stratified splitting
        final_predictor = self._get_final_predictor()
        stratify_coord = self.stratify_coord

        if stratify_coord is None:
            # Default: single-target classification stratifies on final target coord
            if (
                final_predictor
                and final_predictor.is_classifier
                and not isinstance(self.final_target_coord_, list)
                and not getattr(final_predictor, "is_multilabel", False)
            ):
                stratify_coord = self.final_target_coord_

        group_values = self._resolve_orig_trial_groups(container.data)
        if stratify_coord:
            if stratify_coord not in container.data.coords:
                raise ValueError(
                    f"Stratification coordinate '{stratify_coord}' not found in container coords. "
                    f"Available: {list(container.data.coords.keys())}"
                )
            labels = container.data.coords[stratify_coord].values
            if group_values is not None:
                split_units, split_labels = self._get_unique_groups_and_labels(group_values, labels, "holdout split")
            else:
                split_units, split_labels = all_trials, labels
            if split_labels is not None:
                self._validate_stratify_labels(
                    split_labels, n_splits=None, test_size=self.test_size, context="holdout split"
                )

            # Perform stratified split
            train_units, holdout_units = train_test_split(
                split_units, test_size=self.test_size, stratify=split_labels, random_state=self.random_state
            )
        else:
            # Regression or multi-target: random split without stratification
            split_units = np.unique(group_values) if group_values is not None else all_trials
            train_units, holdout_units = train_test_split(
                split_units, test_size=self.test_size, random_state=self.random_state
            )

        if group_values is not None:
            train_val_indices = all_trials[np.isin(group_values, train_units)]
            holdout_indices = all_trials[np.isin(group_values, holdout_units)]
        else:
            train_val_indices = train_units
            holdout_indices = holdout_units

        return train_val_indices, holdout_indices

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates splits using scikit-learn's StratifiedKFold on the specified indices.

        Args:
            container: DataContainer to split
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) for each fold
        """
        if len(indices_to_split) == 0:
            return

        # Select the relevant part of the container first
        cv_container = container.data.sel(trial=indices_to_split)

        # Detect task type and use appropriate splitter
        final_predictor = self._get_final_predictor()
        stratify_coord = self.stratify_coord

        if (
            stratify_coord is None
            and final_predictor
            and final_predictor.is_classifier
            and not isinstance(self.final_target_coord_, list)
            and not getattr(final_predictor, "is_multilabel", False)
        ):
            stratify_coord = self.final_target_coord_

        group_values = self._resolve_orig_trial_groups(cv_container)
        if stratify_coord:
            from sklearn.model_selection import StratifiedKFold

            if stratify_coord not in cv_container.coords:
                raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
            labels_array = cv_container.coords[stratify_coord].values
            if group_values is not None:
                split_units, split_labels = self._get_unique_groups_and_labels(group_values, labels_array, "CV split")
            else:
                split_units, split_labels = np.arange(len(labels_array)), labels_array
            if split_labels is not None:
                self._validate_stratify_labels(split_labels, n_splits=self.n_splits, test_size=None, context="CV split")

            if split_labels is None:
                from sklearn.model_selection import KFold

                splitter = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
                splits = splitter.split(np.arange(len(split_units)))
            else:
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
                splits = splitter.split(np.arange(len(split_units)), split_labels)
        else:
            from sklearn.model_selection import KFold

            split_units = np.unique(group_values) if group_values is not None else np.arange(len(indices_to_split))
            splitter = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = splitter.split(np.arange(len(split_units)))

        # Yield train/val indices (map positions back to trial IDs)
        for train_pos, val_pos in splits:
            if group_values is not None:
                train_groups = split_units[train_pos]
                val_groups = split_units[val_pos]
                yield (
                    cv_container.trial.values[np.isin(group_values, train_groups)],
                    cv_container.trial.values[np.isin(group_values, val_groups)],
                )
            else:
                yield cv_container.trial.values[train_pos], cv_container.trial.values[val_pos]


class GroupedKFoldValidator(CrossValidator):
    """
    Implements cross-validation using a stratified K-Fold strategy.
    Groups are specified by the group_coord parameter. K-folds are stratified by both the group and target coordinates.
    Specific groups can be specified for training, validation, and testing using the values of the group_coord coordinate.
    If no groups are specified, all groups are used for training/validation/testing.

    E.g. if group_coord = 'animal', train_groups = None, val_groups = [35], and test_groups = [35],
    all data will be used for training, but only animal 35 will be used for validation and testing.

    Useful for testing the performance of a model across different groups, especially for domain adaptation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float | None = None,
        pooling_score_weight: float = 0.0,
        group_coord: str | None = None,
        train_groups: list[Hashable] | Hashable | None = None,
        val_groups: list[Hashable] | Hashable | None = None,
        test_groups: list[Hashable] | Hashable | None = None,
        scoring: str | Callable | None = None,
        stratify_coord: str | None = None,
        stratify_by_group: bool = True,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        scoring_needs_proba: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize GroupedKFoldValidator.

        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use as holdout test set (0.0-1.0).
                      If None or 0, no holdout set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            group_coord: Coordinate to group by.
            train_groups: Groups to use for training. If None, all groups are used.
            val_groups: Groups to use for validation. If None, all groups are used.
            test_groups: Groups to use for testing. If None, all groups are used.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            stratify_coord: Optional coordinate name to use for stratified splits.
            stratify_by_group: Whether to stratify splits by group coordinate in addition to target.
                              If True (default), stratifies by group+target combination.
                              If False, only stratifies by target (or stratify_coord if set).
            exclude_intertrial_from_scoring: Whether to drop intertrial segments during evaluation.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            scoring_needs_proba=scoring_needs_proba,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
            stratify_coord=stratify_coord,
            verbose=verbose,
        )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size
        self.group_coord = group_coord
        self.stratify_by_group = stratify_by_group

        self.train_groups = _as_group_list(train_groups)
        self.val_groups = _as_group_list(val_groups)
        self.test_groups = _as_group_list(test_groups)

    def _check_groups(self, container: DataContainer):
        """Validates the dimension that the group_coord indexes. Checks that the specified groups are available in the data."""

        def _check_avail_groups(group_type: str, groups: list[Hashable], avail_groups: list[Hashable]):
            n_avail_groups = sum(group in avail_groups for group in groups)
            if n_avail_groups == 0:
                raise ValueError(f"No {group_type} groups {groups} found in data. Available groups: {avail_groups}")
            if n_avail_groups != len(groups):
                warnings.warn(
                    f"Not all {group_type} groups {groups} found in data. Available groups: {avail_groups}",
                    stacklevel=2,
                )

        if self.group_coord not in container.data.coords:
            raise ValueError(f"Group coordinate '{self.group_coord}' not found in data coordinates")

        coord_dims = container.data.coords[self.group_coord].dims
        if len(coord_dims) != 1:
            raise ValueError(
                f"Group coordinate '{self.group_coord}' must index exactly one dimension, "
                f"but it indexes {len(coord_dims)}: {coord_dims}"
            )

        unique_groups = self._discover_groups(container)
        if self.train_groups is not None:
            _check_avail_groups("train", self.train_groups, unique_groups)
        if self.val_groups is not None:
            _check_avail_groups("val", self.val_groups, unique_groups)
        if self.test_groups is not None:
            _check_avail_groups("test", self.test_groups, unique_groups)

        return coord_dims[0]

    def _discover_groups(self, container: DataContainer) -> list[Hashable]:
        """Discovers unique group values from the data."""
        group_values = container.data.coords[self.group_coord].values
        return sorted(np.unique(group_values).tolist())

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into train/validation and holdout sets using stratified sampling.
        Stratification is done by target coordinate (and optionally group coordinate if stratify_by_group=True).
        Keep only indices that are relevant to the specified groups for train/val/test.

        Args:
            container: DataContainer to split

        Returns:
            Tuple of (train_val_indices, holdout_indices)
        """
        self._check_groups(container)

        all_trials = container.data.trial.values
        all_groups = container.data.coords[self.group_coord].values

        if self.test_size is None or self.test_size == 0:
            # No holdout set - use all data for cross-validation
            return all_trials, np.array([])

        # Get labels for stratification
        # Determine stratification coordinate (use stratify_coord if set, otherwise use final_target_coord_)
        stratify_coord = self.stratify_coord if self.stratify_coord is not None else self.final_target_coord_
        if stratify_coord not in container.data.coords:
            raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
        labels = container.data.coords[stratify_coord].values

        # Optionally combine group and label for stratification
        if self.stratify_by_group:
            stratify_labels = _combine_group_and_label_strings(all_groups, labels)
        else:
            stratify_labels = labels

        # Split positions (not trial labels) so group lookups below index all_groups correctly
        train_val_pos, holdout_pos = train_test_split(
            np.arange(len(all_trials)),
            test_size=self.test_size,
            stratify=stratify_labels,
            random_state=self.random_state,
        )

        if (
            self.train_groups is not None and self.val_groups is not None
        ):  # if both are specified, otherwise keep all and split later
            # keep only indices that are in either train or val groups
            train_val_groups = set(self.train_groups + self.val_groups)
            train_val_pos = train_val_pos[np.isin(all_groups[train_val_pos], list(train_val_groups))]

        if self.test_groups is not None:
            # keep only indices that are in test_groups
            holdout_pos = holdout_pos[np.isin(all_groups[holdout_pos], np.asarray(self.test_groups))]

        return all_trials[train_val_pos], all_trials[holdout_pos]

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates train/validation splits using scikit-learn's StratifiedKFold on the specified indices.
        Stratification is done by target coordinate (and optionally group coordinate if stratify_by_group=True).
        Keep only indices that are relevant to the specified groups for train/val.

        Args:
            container: DataContainer to split
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) for each fold
        """
        if len(indices_to_split) == 0:
            return

        # Initialize stratified k-fold splitter
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        # Select the relevant part of the container first
        cv_container = container.data.sel(trial=indices_to_split)
        cv_groups = cv_container.coords[self.group_coord].values

        # Now get labels directly from this smaller container
        # Determine stratification coordinate (use stratify_coord if set, otherwise use final_target_coord_)
        stratify_coord = self.stratify_coord if self.stratify_coord is not None else self.final_target_coord_
        if stratify_coord not in cv_container.coords:
            raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
        labels_array = cv_container.coords[stratify_coord].values

        # Optionally combine group and label for stratification
        if self.stratify_by_group:
            stratify_labels = _combine_group_and_label_strings(cv_groups, labels_array)
        else:
            stratify_labels = labels_array

        # skf.split works on integer positions, so we map back to trial IDs
        for train_pos, val_pos in skf.split(np.arange(len(labels_array)), stratify_labels):
            # keep only indices that are relevant to the specified groups
            if self.train_groups is not None:
                train_pos = train_pos[np.isin(cv_groups[train_pos], np.asarray(self.train_groups))]
            if self.val_groups is not None:
                val_pos = val_pos[np.isin(cv_groups[val_pos], np.asarray(self.val_groups))]

            yield cv_container.trial.values[train_pos], cv_container.trial.values[val_pos]
