from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import cast

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from xdflow.core.base import Predictor
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator


class SampledDomainKFoldValidator(CrossValidator):
    """K-fold validation on target domains with sampled target-domain training trials.

    Splits are created on target-domain trials only. For each fold:
    - validation contains one fold of target-domain trials
    - training contains all source-domain trials plus a sampled subset of the remaining target-domain trials

    Target-domain sampling is label-conditional. Use ``label_sample_counts`` for per-label overrides and
    ``default_samples_per_label`` for all other labels. A count of ``0`` means zero-shot for that label;
    ``None`` means use all available target training samples for that label.
    """

    def __init__(
        self,
        *,
        domain_coord: str,
        target_domains: list[Hashable] | Hashable,
        source_domains: list[Hashable] | Hashable | None = None,
        label_coord: str | None = None,
        label_sample_counts: Mapping[Hashable, int | None] | None = None,
        default_samples_per_label: int | None = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        scoring_needs_proba: bool = False,
        stratify_coord: str | None = None,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = True,
        release_fold_memory: bool = False,
        verbose: bool = True,
    ):
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
        self.domain_coord = domain_coord
        self.target_domains = target_domains if isinstance(target_domains, list) else [target_domains]
        self.source_domains = (
            None
            if source_domains is None
            else (source_domains if isinstance(source_domains, list) else [source_domains])
        )
        self.label_coord = label_coord
        self.label_sample_counts = dict(label_sample_counts or {})
        self.default_samples_per_label = default_samples_per_label
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size

        if not self.target_domains:
            raise ValueError("target_domains must be provided and non-empty.")
        for label, count in self.label_sample_counts.items():
            if count is not None and count < 0:
                raise ValueError(f"label_sample_counts[{label!r}] must be >= 0 or None.")
        if self.default_samples_per_label is not None and self.default_samples_per_label < 0:
            raise ValueError("default_samples_per_label must be >= 0 or None.")

    def _check_domain_coord(self, container: DataContainer) -> np.ndarray:
        if self.domain_coord not in container.data.coords:
            raise ValueError(f"Domain coordinate '{self.domain_coord}' not found in data coordinates.")
        coord = container.data.coords[self.domain_coord]
        if coord.dims != ("trial",):
            raise ValueError(
                f"Domain coordinate '{self.domain_coord}' must have dimension ('trial',), got {coord.dims}."
            )
        return np.asarray(coord.values)

    def _resolve_domain_masks(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        domain_values = self._check_domain_coord(container)
        target_mask = np.isin(domain_values, np.asarray(self.target_domains))
        if self.source_domains is None:
            source_mask = ~target_mask
        else:
            source_mask = np.isin(domain_values, np.asarray(self.source_domains))
        if np.any(source_mask & target_mask):
            raise ValueError("source_domains and target_domains must not overlap.")

        if not target_mask.any():
            raise ValueError(
                f"No trials found for target_domains={self.target_domains} in coord '{self.domain_coord}'."
            )
        if self.source_domains is not None and not source_mask.any():
            raise ValueError(
                f"No trials found for source_domains={self.source_domains} in coord '{self.domain_coord}'."
            )

        unique_domains = np.unique(domain_values)
        missing_sources = (
            [] if self.source_domains is None else [d for d in self.source_domains if d not in unique_domains]
        )
        if missing_sources:
            warnings.warn(f"Some source domains not found in data: {missing_sources}", stacklevel=2)
        missing_targets = [d for d in self.target_domains if d not in unique_domains]
        if missing_targets:
            warnings.warn(f"Some target domains not found in data: {missing_targets}", stacklevel=2)

        return domain_values, source_mask, target_mask

    def _resolve_label_coord(self, container: DataContainer) -> str:
        final_predictor = self._get_final_predictor()
        if final_predictor is not None and (
            not final_predictor.is_classifier or getattr(final_predictor, "is_multilabel", False)
        ):
            raise ValueError("SampledDomainKFoldValidator requires a single-label classifier predictor.")
        label_coord = self.label_coord or self.final_target_coord_
        if label_coord is None:
            raise ValueError("label_coord could not be resolved. Provide label_coord explicitly.")
        if isinstance(label_coord, list):
            raise ValueError("SampledDomainKFoldValidator does not support multi-target labels.")
        if label_coord not in container.data.coords:
            raise ValueError(f"Label coordinate '{label_coord}' not found in container coords.")
        return label_coord

    def _sample_indices(self, pool: np.ndarray, n_samples: int | None, rng: np.random.Generator) -> np.ndarray:
        if n_samples is None:
            return pool
        if n_samples <= 0 or pool.size == 0:
            return np.array([], dtype=pool.dtype)
        if n_samples >= pool.size:
            if n_samples > pool.size:
                warnings.warn(
                    f"Requested n_samples={n_samples} but only {pool.size} available; using all available.",
                    stacklevel=2,
                )
            return pool
        return rng.choice(pool, size=n_samples, replace=False)

    def _samples_for_label(self, label: Hashable) -> int | None:
        item = getattr(label, "item", None)
        key = item() if callable(item) else label
        return self.label_sample_counts.get(key, self.default_samples_per_label)

    def _sample_target_train_indices(
        self, target_train_indices: np.ndarray, target_train_labels: np.ndarray, fold_idx: int
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state + fold_idx)
        sampled: list[np.ndarray] = []
        for label in np.unique(target_train_labels):
            label_mask = target_train_labels == label
            label_indices = target_train_indices[label_mask]
            sampled.append(self._sample_indices(label_indices, self._samples_for_label(label), rng))
        if not sampled:
            return np.array([], dtype=target_train_indices.dtype)
        return np.concatenate(sampled)

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        all_trials = container.data.trial.values
        if self.test_size is None or self.test_size == 0:
            return all_trials, np.array([])

        _, _, target_mask = self._resolve_domain_masks(container)
        target_trials = all_trials[target_mask]
        if target_trials.size == 0:
            return all_trials, np.array([])

        stratify_coord = self.stratify_coord
        if stratify_coord is None:
            final_predictor = self._get_final_predictor()
            if (
                final_predictor
                and final_predictor.is_classifier
                and not isinstance(self.final_target_coord_, list)
                and not getattr(final_predictor, "is_multilabel", False)
            ):
                stratify_coord = self._resolve_label_coord(container)

        if stratify_coord:
            labels = container.data.coords[stratify_coord].values[target_mask]
            self._validate_stratify_labels(labels, n_splits=None, test_size=self.test_size, context="holdout split")
            _, holdout_target = train_test_split(
                target_trials, test_size=self.test_size, stratify=labels, random_state=self.random_state
            )
        else:
            _, holdout_target = train_test_split(
                target_trials, test_size=self.test_size, random_state=self.random_state
            )

        train_val_indices = all_trials[~np.isin(all_trials, holdout_target)]
        return train_val_indices, holdout_target

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if len(indices_to_split) == 0:
            return

        cv_container = container.data.sel(trial=indices_to_split)
        _, source_mask, target_mask = self._resolve_domain_masks(DataContainer(cv_container))
        source_trials = cv_container.trial.values[source_mask]
        target_trials = cv_container.trial.values[target_mask]

        if target_trials.size == 0:
            raise ValueError("No target-domain trials available for splitting.")

        label_coord = self._resolve_label_coord(DataContainer(cv_container))
        labels_all = cv_container.coords[label_coord].values

        final_predictor = self._get_final_predictor()
        stratify_coord = self.stratify_coord
        if (
            stratify_coord is None
            and final_predictor
            and final_predictor.is_classifier
            and not isinstance(self.final_target_coord_, list)
            and not getattr(final_predictor, "is_multilabel", False)
        ):
            stratify_coord = label_coord

        if stratify_coord:
            if stratify_coord not in cv_container.coords:
                raise ValueError(f"Stratification coord '{stratify_coord}' not found in container coords.")
            labels_target = cv_container.coords[stratify_coord].values[target_mask]
            self._validate_stratify_labels(labels_target, n_splits=self.n_splits, test_size=None, context="CV split")
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = splitter.split(np.arange(target_trials.size), labels_target)
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = splitter.split(np.arange(target_trials.size))

        for fold_idx, (train_pos, val_pos) in enumerate(splits):
            target_train_indices = target_trials[train_pos]
            target_val_indices = target_trials[val_pos]
            target_train_labels = labels_all[target_mask][train_pos]

            sampled_target = self._sample_target_train_indices(target_train_indices, target_train_labels, fold_idx)
            train_indices = np.concatenate([source_trials, sampled_target])
            if train_indices.size == 0:
                raise ValueError("Training set is empty after sampling. Check source_domains and sampling settings.")
            yield train_indices, target_val_indices

    def score_on_holdout(self, initial_container: DataContainer, verbose: bool = False) -> float:
        """Fit and score on the target-domain holdout using the validator sampling policy.

        Unlike the base ``KFoldValidator`` holdout path, the final training set is not
        all non-holdout trials. It is all source-domain trials plus the same
        label-conditional sampled subset of non-holdout target-domain trials used
        during cross-validation. This keeps holdout scoring aligned with the
        few-shot/zero-shot transfer regime configured for the validator.
        """
        self._find_and_fit_encoders(self.pipeline, initial_container)

        stateless_pipeline, stateful_pipeline = self._auto_detect_pipeline_parts(self.pipeline)
        assert stateful_pipeline is not None, "There must be at least one stateful step in the pipeline, for fitting."

        if stateless_pipeline is not None:
            preprocessed_data = stateless_pipeline.fit_transform(initial_container, verbose=verbose)
        else:
            preprocessed_data = initial_container

        if self.holdout_trial_labels_ is None:
            warnings.warn(
                "cross_validate() not called first so no holdout indices available. Calculating holdout indices now.",
                stacklevel=2,
            )
            _, holdout_indices = self._split_holdout(preprocessed_data)
            self.holdout_trial_labels_ = holdout_indices

        if len(self.holdout_trial_labels_) == 0:
            raise ValueError("No holdout data available for testing.")

        label_coord = self._resolve_label_coord(preprocessed_data)
        labels_all = preprocessed_data.data.coords[label_coord].values
        _, source_mask, target_mask = self._resolve_domain_masks(preprocessed_data)
        target_trials = preprocessed_data.data.trial.values[target_mask]
        source_trials = preprocessed_data.data.trial.values[source_mask]
        target_labels = labels_all[target_mask]

        train_target_mask = ~np.isin(target_trials, self.holdout_trial_labels_)
        sampled_target = self._sample_target_train_indices(
            target_trials[train_target_mask], target_labels[train_target_mask], fold_idx=0
        )
        train_val_indices = np.concatenate([source_trials, sampled_target])
        if train_val_indices.size == 0:
            raise ValueError("Training set is empty after sampling. Check source_domains and sampling settings.")

        train_val_container = DataContainer(preprocessed_data.data.sel(trial=train_val_indices))
        test_container = DataContainer(preprocessed_data.data.sel(trial=self.holdout_trial_labels_))

        stateful_pipeline_fitted = stateful_pipeline.clone()
        stateful_pipeline_fitted.fit(train_val_container, verbose=verbose)

        test_results_container = stateful_pipeline_fitted.predict(test_container, verbose=verbose)
        scoring_func, _, needs_proba = self._get_scoring_func()
        holdout_probabilities = None
        if needs_proba:
            holdout_probabilities = stateful_pipeline_fitted.predict_proba(test_container, verbose=verbose).data.values

        final_predictor = stateful_pipeline_fitted.predictive_transform
        if final_predictor is None:
            raise ValueError("Stateful pipeline must expose a predictive transform.")
        pred_labels = test_results_container.data.values
        true_labels = self._extract_targets(cast("Predictor", final_predictor), test_results_container)

        pred_labels, true_labels, scoring_container, scoring_mask = self._filter_scoring_inputs(
            pred_labels,
            true_labels,
            test_results_container,
            context="holdout",
        )
        scoring_values = pred_labels
        if needs_proba:
            if holdout_probabilities is None:
                raise RuntimeError("Scoring requires probabilities, but none were produced.")
            scoring_values = holdout_probabilities if scoring_mask is None else holdout_probabilities[scoring_mask]

        self.holdout_container_ = scoring_container
        self.holdout_pred_labels_ = pred_labels
        self.holdout_class_labels_ = self._snapshot_class_labels()
        self.holdout_probabilities_ = scoring_values if needs_proba else holdout_probabilities
        self.holdout_true_labels_ = true_labels
        self.holdout_scoring_mask_ = None

        self.holdout_score_ = self._call_scoring_func(
            scoring_func, self.holdout_true_labels_, scoring_values, scoring_container
        )

        return self.holdout_score_
