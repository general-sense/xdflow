# Writing Custom Cross-Validators

Most projects should use the built-in validators:

- `KFoldValidator` for trial-level folds with optional holdout
- `GroupedKFoldValidator` for grouped trial splits
- `LeaveGroupOutValidator`, `LeaveSessionOutValidator`, and
  `LeaveAnimalOutValidator` for group generalization
- `SampledDomainKFoldValidator` for source/target domain sampling

Write a custom cross-validator only when the split policy itself is new. The
base `CrossValidator` already owns scoring, holdout evaluation, out-of-fold
prediction collection, stateless preprocessing reuse, stateful refitting,
caching, and final pipeline fitting.

## The Validator Contract

A custom validator usually implements only two methods:

```python
def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]: ...


def _get_splits(
    self,
    container: DataContainer,
    indices_to_split: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
```

Both methods should return `trial` coordinate labels, not positional indices.
This matters because `trial` labels may be non-contiguous IDs.

`cross_validate()` calls these methods after fold-invariant stateless pipeline
steps have run. Your split logic should therefore assume that the `trial`
dimension and trial labels have been preserved by preprocessing.

## Minimal Grouped Example

This validator creates folds by session and optionally reserves named sessions
as a final holdout set.

```python
from collections.abc import Hashable, Iterator

import numpy as np
from sklearn.model_selection import KFold

from xdflow.core import DataContainer
from xdflow.cv import CrossValidator


class SessionBlockedKFoldValidator(CrossValidator):
    def __init__(
        self,
        session_coord: str = "session",
        n_splits: int = 5,
        holdout_sessions: list[Hashable] | None = None,
        shuffle: bool = True,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_coord = session_coord
        self.n_splits = n_splits
        self.holdout_sessions = holdout_sessions or []
        self.shuffle = shuffle
        self.random_state = random_state

    def _validate_session_coord(self, container: DataContainer) -> None:
        if "trial" not in container.data.dims:
            raise ValueError("SessionBlockedKFoldValidator requires a 'trial' dimension.")
        if self.session_coord not in container.data.coords:
            raise ValueError(f"Coordinate '{self.session_coord}' not found.")
        if container.data.coords[self.session_coord].dims != ("trial",):
            raise ValueError(f"Coordinate '{self.session_coord}' must index the 'trial' dimension.")

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        self._validate_session_coord(container)

        trial_labels = container.data.trial.values
        sessions = container.data.coords[self.session_coord].values

        if not self.holdout_sessions:
            return trial_labels, np.array([], dtype=trial_labels.dtype)

        holdout_mask = np.isin(sessions, np.asarray(self.holdout_sessions))
        return trial_labels[~holdout_mask], trial_labels[holdout_mask]

    def _get_splits(
        self,
        container: DataContainer,
        indices_to_split: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        self._validate_session_coord(container)

        cv_data = container.data.sel(trial=indices_to_split)
        trial_labels = cv_data.trial.values
        sessions = cv_data.coords[self.session_coord].values
        unique_sessions = np.unique(sessions)

        if self.n_splits > len(unique_sessions):
            raise ValueError("n_splits cannot exceed the number of sessions.")

        splitter = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for train_pos, val_pos in splitter.split(unique_sessions):
            train_sessions = unique_sessions[train_pos]
            val_sessions = unique_sessions[val_pos]

            train_mask = np.isin(sessions, train_sessions)
            val_mask = np.isin(sessions, val_sessions)

            yield trial_labels[train_mask], trial_labels[val_mask]
```

Then use it like any other validator:

```python
cv = SessionBlockedKFoldValidator(
    n_splits=4,
    holdout_sessions=["session_d"],
    scoring="f1_weighted",
    verbose=False,
)
cv.set_pipeline(pipeline)

score = cv.cross_validate(container, verbose=False)
```

## What The Base Class Handles

Do not reimplement the evaluation loop unless the base class cannot express the
workflow. `CrossValidator.cross_validate()` already handles:

- splitting the pipeline into fold-invariant stateless steps and stateful steps
- running stateless preprocessing once before folds
- cloning and fitting stateful pipeline steps per fold
- calling `predict` or `predict_proba` as needed by the scorer
- collecting fold scores, out-of-fold predictions, and probabilities
- optional holdout scoring
- final fitting through `finalize_pipeline`

Custom validators should focus on split policy only.

## Split Invariants

Good split methods should maintain these invariants:

- returned arrays contain `trial` labels present in the container
- train, validation, and holdout labels do not overlap
- holdout labels are excluded from `_get_splits`
- group-level split policies do not split a group across train and validation
- required coordinates are validated early with clear errors
- randomization is controlled by constructor parameters

For grouped validators, validate that the group coordinate is one-dimensional
and attached to `trial`.

## Testing Checklist

Add focused tests for split behavior before testing scoring:

```python
def test_session_blocked_splits_do_not_split_sessions(multi_session_container):
    cv = SessionBlockedKFoldValidator(n_splits=2, session_coord="session", verbose=False)

    train_val, holdout = cv._split_holdout(multi_session_container)
    splits = list(cv._get_splits(multi_session_container, train_val))

    assert len(holdout) == 0
    assert len(splits) == 2

    data = multi_session_container.data
    for train_trials, val_trials in splits:
        train_sessions = set(data.sel(trial=train_trials).coords["session"].values)
        val_sessions = set(data.sel(trial=val_trials).coords["session"].values)

        assert train_sessions.isdisjoint(val_sessions)
        assert set(train_trials).isdisjoint(set(val_trials))
```

Then add one integration test with a real pipeline and a container that has both
the target coordinate and the session coordinate:

```python
def test_session_blocked_cross_validate_runs(session_labeled_container, pipeline):
    cv = SessionBlockedKFoldValidator(n_splits=2, session_coord="session", verbose=False)
    cv.set_pipeline(pipeline)

    score = cv.cross_validate(session_labeled_container, verbose=False)

    assert np.isfinite(score)
    assert len(cv.cv_scores_) == 2
```

## Sklearn Compatibility

If an sklearn estimator needs a `cv=` splitter, wrap an XDFlow validator with
`SklearnCVAdapter` and provide the active `DataContainer` through
`set_cv_container`.

```python
from sklearn.linear_model import LogisticRegressionCV

from xdflow.cv import SklearnCVAdapter, set_cv_container

adapter = SklearnCVAdapter(cv)
estimator = LogisticRegressionCV(cv=adapter)

with set_cv_container(container):
    estimator.fit(X, y)
```

Use this only for estimators that own their own internal CV. For XDFlow
pipelines, prefer `cv.set_pipeline(pipeline)` followed by `cv.cross_validate`.
