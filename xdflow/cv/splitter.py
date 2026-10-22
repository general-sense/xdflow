"""Cross-validation with any scikit-learn splitter."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator


class SplitterValidator(CrossValidator):
    """Cross-validate with a scikit-learn splitter, or with explicit ``(train, test)`` index pairs.

    Use it to keep a split you already rely on (``GroupKFold``, ``StratifiedGroupKFold``, ``LeaveOneGroupOut``,
    ``TimeSeriesSplit``, ``PredefinedSplit``...) while xdflow runs the pipeline: stateless steps once, stateful
    steps refit inside each training fold. The splitter receives the trial-level target as ``y`` and, when
    ``group_coord`` is set, that coordinate's values as ``groups``. Index pairs are positions along ``trial`` in
    the data passed to ``cross_validate``.

    Example:
        >>> cv = SplitterValidator(StratifiedGroupKFold(5), group_coord="subject", scoring="accuracy")
    """

    def __init__(
        self,
        splitter: Any,
        group_coord: str | None = None,
        stratify_coord: str | None = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        scoring_needs_proba: bool = False,
        exclude_intertrial_from_scoring: bool = False,
        exclude_offsets_from_scoring: bool = False,
        use_stateful_fit_cache: bool = False,
        release_fold_memory: bool = False,
        verbose: bool = True,
    ):
        """Initialize the validator.

        Args:
            splitter: A scikit-learn splitter (an object with ``split(X, y, groups)``), or a list of
                ``(train_positions, test_positions)`` pairs.
            group_coord: Trial coordinate passed to the splitter as ``groups``.
            stratify_coord: Trial coordinate passed as ``y``. Defaults to the pipeline's target for
                single-target predictors.
            pooling_score_weight: Interpolation between mean fold score (0.0) and pooled OOF score (1.0).
            scoring: Metric name or callable; defaults to the task's default metric.
            scoring_needs_proba: Whether a custom scorer expects probabilities.
            exclude_intertrial_from_scoring: Drop intertrial segments when scoring.
            exclude_offsets_from_scoring: Drop trials with a non-zero time offset when scoring.
            use_stateful_fit_cache: Whether to reuse fitted fold pipelines cached on disk by earlier identical runs.
            release_fold_memory: Whether to release per-fold objects after each fold.
            verbose: Whether to print progress.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            scoring_needs_proba=scoring_needs_proba,
            stratify_coord=stratify_coord,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            exclude_offsets_from_scoring=exclude_offsets_from_scoring,
            use_stateful_fit_cache=use_stateful_fit_cache,
            release_fold_memory=release_fold_memory,
            verbose=verbose,
        )
        if not hasattr(splitter, "split"):
            if isinstance(splitter, (str, bytes)) or not isinstance(splitter, Iterable):
                raise TypeError(
                    f"splitter must have a split() method or be a list of (train, test) index pairs; got {splitter!r}."
                )
            splitter = list(splitter)  # a generator would be exhausted after the first use
        self.splitter = splitter
        self.group_coord = group_coord

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        return container.data.trial.values, np.array([])

    def cross_validate(
        self, initial_container: DataContainer, *args: Any, input_trials: Any = None, **kwargs: Any
    ) -> float:
        """Cross-validate; ``input_trials`` is the trial order positions refer to, when ``initial_container`` is
        the output of steps already run (the Tuner's prefix cache). See ``CrossValidator.cross_validate``."""
        # Positions (and order-sensitive splitters such as TimeSeriesSplit) refer to the trials as given, so
        # remember that order: steps that run before the split may reorder trials.
        self._input_trials = initial_container.data.trial.values if input_trials is None else np.asarray(input_trials)
        try:
            return super().cross_validate(initial_container, *args, **kwargs)
        finally:
            del self._input_trials

    def _split_target(self, data) -> np.ndarray | None:
        coord = self.stratify_coord
        if coord is None and isinstance(self.final_target_coord_, str):
            predictor = self._get_final_predictor()
            if predictor is None or not getattr(predictor, "is_multilabel", False):
                coord = self.final_target_coord_
        if coord is None:
            return None
        if coord not in data.coords:
            raise ValueError(f"Coordinate '{coord}' not found; available: {list(data.coords)}.")
        return data.coords[coord].values

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        data = container.data.sel(trial=indices_to_split)
        input_trials = getattr(self, "_input_trials", None)
        if input_trials is not None and not np.array_equal(input_trials, data.trial.values):
            present = pd.Index(data.trial.values).get_indexer(input_trials) >= 0
            same_trials = present.sum() == data.sizes["trial"]  # no trial created by a step before the split
            if same_trials:
                data = data.sel(trial=input_trials[present])  # back to input order
            if not hasattr(self.splitter, "split") and not (same_trials and present.all()):
                raise ValueError(
                    "Split positions refer to the input trials, but a step before the split added or removed trials."
                )
        trials = data.trial.values
        n_trials = trials.size
        groups = None
        if self.group_coord is not None:
            if self.group_coord not in data.coords:
                raise ValueError(f"Group coordinate '{self.group_coord}' not found; available: {list(data.coords)}.")
            if data.coords[self.group_coord].dims != ("trial",):
                raise ValueError(f"Group coordinate '{self.group_coord}' must be indexed by 'trial'.")
            groups = data.coords[self.group_coord].values

        split = getattr(self.splitter, "split", None)
        if split is not None:
            target = self._split_target(data)
            try:
                splits = list(split(np.zeros((n_trials, 1)), target, groups))
            except (TypeError, ValueError) as exc:
                if groups is None and "groups" in str(exc):
                    raise ValueError(f"{exc} Set group_coord to the trial coordinate holding the groups.") from exc
                if target is None:
                    raise ValueError(
                        f"{type(self.splitter).__name__} failed without a target ({exc}); set stratify_coord to the "
                        "trial coordinate holding it."
                    ) from exc
                raise
        else:
            splits = self.splitter

        for train_pos, test_pos in splits:
            train_pos, test_pos = _check_positions(train_pos, n_trials), _check_positions(test_pos, n_trials)
            if np.intersect1d(train_pos, test_pos).size:
                raise ValueError("A split puts the same trial in both training and test sets.")
            yield trials[train_pos], trials[test_pos]


def _check_positions(positions: Any, n_trials: int) -> np.ndarray:
    positions = np.asarray(positions)
    if positions.dtype == bool:
        if positions.shape != (n_trials,):
            raise ValueError(f"A boolean split mask has shape {positions.shape}; expected ({n_trials},).")
        return np.flatnonzero(positions)
    if positions.size and not np.issubdtype(positions.dtype, np.integer):
        raise ValueError(f"Split indices must be integer positions along 'trial'; got dtype {positions.dtype}.")
    positions = positions.astype(int, copy=False).ravel()
    if positions.size and (positions.min() < 0 or positions.max() >= n_trials):
        raise ValueError(f"Split indices must be in [0, {n_trials}); got {positions.min()}..{positions.max()}.")
    if np.unique(positions).size != positions.size:
        raise ValueError("Split indices contain duplicates.")
    return positions
