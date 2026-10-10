"""Audit a train/test split for shared subjects, sessions, source trials and duplicate samples."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import check_cv
from sklearn.utils.multiclass import type_of_target

from xdflow.cv.base import CrossValidator
from xdflow.cv.splitter import _check_positions
from xdflow.interop import _as_container

# Coordinate names recognised as units that should not be split between training and test sets.
_SUBJECT_WORDS = {"subject", "subj", "participant", "patient", "animal", "mouse", "rat", "monkey", "individual"}
_RECORDING_WORDS = {"session", "sess", "run", "recording", "segment", "day", "block", "visit", "site"}
_SOURCE_NAMES = {"orig_trial", "original_trial", "source_trial", "parent_trial", "epoch_source", "window_source"}
_ID_SUFFIXES = {"id", "ids", "nr", "no", "num", "number", "idx", "index", "code", "name", "label", "key"}


def grouping_coords(data: xr.DataArray, min_values: int = 1) -> list[str]:
    """Trial coordinates whose name identifies a subject, recording or source trial (``subject``,
    ``session_id``, ``orig_trial``...) and whose values repeat across trials, subjects first.

    Args:
        data: The data.
        min_values: Fewest distinct values to report; 2 keeps only coordinates that can be split by.
    """
    found = []
    for name, coord in data.coords.items():
        name = str(name)
        if name == "trial" or coord.dims != ("trial",) or _grouping_kind(name) is None:
            continue
        n_values = len(pd.unique(np.asarray(coord.values).ravel()))
        if min_values <= n_values < data.sizes["trial"]:
            found.append(name)
    order = {"subject": 0, "recording": 1, "source": 2}
    return sorted(found, key=lambda name: order[_grouping_kind(name) or "source"])


def _grouping_kind(name: str) -> str | None:
    """ "subject", "recording" or "source" for identity-style names (``subject``, ``SubjectID``, ``mouse_ids``,
    ``block_nr``, ``recording_session``), None otherwise, including attributes such as ``subject_age``."""
    tokens = [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]
    if len(tokens) > 1 and tokens[-1] in _ID_SUFFIXES:
        tokens = tokens[:-1]
    if not tokens:
        return None
    if "_".join(tokens) in _SOURCE_NAMES or "_".join(tokens[-2:]) in _SOURCE_NAMES:
        return "source"
    word = tokens[-1]
    for candidate in (word, word.removesuffix("ids"), word.removesuffix("id"), word.removesuffix("s")):
        if candidate in _SUBJECT_WORDS:
            return "subject"
        if candidate in _RECORDING_WORDS:
            return "recording"
    return None


@dataclass
class SplitReport:
    """What a set of train/test splits shares between training and test data.

    Attributes:
        n_folds: Number of splits audited.
        table: One row per check: the coordinate (or ``"samples"`` for duplicates), how many values it has,
            in how many folds training and test sets share values, and examples of shared values.
        findings: Plain-language description of every shared unit, one per check that found any.
        shared_coords: Coordinates with values in both training and test sets in at least one fold.
        notes: Context that is not a finding (e.g. which coordinates were audited).
    """

    n_folds: int
    table: pd.DataFrame
    findings: list[str] = field(default_factory=list)
    shared_coords: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no audited unit is shared between training and test data in any fold."""
        return not self.findings

    def __repr__(self) -> str:
        lines = [f"Split check over {self.n_folds} folds: " + ("nothing shared." if self.ok else "shared data found.")]
        lines += [f"  - {finding}" for finding in self.findings]
        lines += [f"  {note}" for note in self.notes]
        if not self.table.empty:
            lines += ["", self.table.to_string(index=False)]
        return "\n".join(lines)


def check_split(
    X: Any,
    cv: Any = 5,
    *,
    y: Any = None,
    groups: Any = None,
    metadata: pd.DataFrame | None = None,
    dims: tuple[str, ...] | None = None,
) -> SplitReport:
    """Check what a cross-validation split shares between training and test sets, without running a model.

    Reports, per fold, subjects, sessions and other recognised grouping coordinates (and ``groups``) that
    appear on both sides, windows cut from the same source trial (``orig_trial``), and test samples with an
    exact copy in the training set. Shared subjects are not wrong in themselves: they mean the score
    estimates performance on new trials from known subjects rather than on new subjects.

    Args:
        X: NumPy array (samples first), ``xr.DataArray``, ``DataContainer`` or MNE ``Epochs``.
        cv: What scikit-learn's ``cross_val_score`` accepts: an int (stratified K-fold for class labels, as
            in scikit-learn), a splitter, or ``(train, test)`` index pairs. Also accepts an xdflow validator
            that has already run ``cross_validate`` (its recorded splits are audited).
        y: Target, as a trial coordinate name or one value per sample; used by stratified splitters.
        groups: Groups passed to the splitter, as a coordinate name or one value per sample. Always audited.
        metadata: DataFrame with one row per sample; its columns are audited when their names are recognised
            (subject, session, run, ...) and can be referenced by name in ``y`` and ``groups``.
        dims: Dim names for NumPy input, starting with ``"trial"``.

    Example:
        >>> check_split(X, StratifiedKFold(5, shuffle=True), y=y, metadata=info)  # info has a "subject" column
    """
    container = _as_container(
        X,
        dims=dims,
        metadata=metadata,
        y=None if isinstance(y, str) else y,
        group=None if isinstance(groups, str) else groups,
    )
    data = container.data
    y_name = y if isinstance(y, str) else ("y" if y is not None else None)
    groups_name = groups if isinstance(groups, str) else ("group" if groups is not None else None)
    for name in (y_name, groups_name):
        if name is not None and name not in data.coords:
            raise ValueError(f"Coordinate '{name}' not found; available: {list(data.coords)}.")

    trials = data.trial.values
    if isinstance(cv, CrossValidator):
        if y is not None or groups is not None:
            raise ValueError("y and groups are taken from the validator's own configuration; do not pass them.")
        if not cv.split_trials_:
            raise ValueError("This validator has not run yet; call cross_validate() first, or pass a splitter.")
        split_trials = cv.split_trials_
    else:
        y_values = data.coords[y_name].values if y_name else None
        group_values = data.coords[groups_name].values if groups_name else None
        is_classification = y_values is not None and type_of_target(y_values) in ("binary", "multiclass")
        splitter = check_cv(cv, y_values, classifier=is_classification)  # also wraps (train, test) index pairs
        split_trials = []
        for train_pos, test_pos in splitter.split(np.zeros((trials.size, 1)), y_values, group_values):
            train_pos, test_pos = _check_positions(train_pos, trials.size), _check_positions(test_pos, trials.size)
            split_trials.append((trials[train_pos], trials[test_pos]))

    audited = grouping_coords(data)
    if groups_name and groups_name not in audited:
        audited.append(groups_name)
    samples = data.transpose("trial", ...).values.reshape(trials.size, -1)
    return audit_splits(data, split_trials, audited, samples=samples)


def audit_splits(
    data: xr.DataArray,
    split_trials: Sequence[tuple[np.ndarray, np.ndarray]],
    coords: Sequence[str],
    samples: np.ndarray | None = None,
) -> SplitReport:
    """Audit splits given as ``(train_trials, test_trials)`` trial labels against trial coordinates."""
    trial_position = pd.Index(data.trial.values)
    for train_trials, test_trials in split_trials:
        if (trial_position.get_indexer(np.concatenate([train_trials, test_trials])) < 0).any():
            raise ValueError("The splits refer to trials that are not in the data; audit the data that was split.")
    rows, findings, shared_coords = [], [], []
    subject_coord = next((name for name in coords if _grouping_kind(name) == "subject"), None)

    for name in coords:
        values = np.asarray(data.coords[name].values)
        kind = _grouping_kind(name)
        scoped = kind == "recording" and subject_coord is not None
        if scoped:  # session/run ids are often reused across subjects (session 1 of every subject)
            subjects = np.asarray(data.coords[subject_coord].values)
            values = np.array([f"{s}/{v}" for s, v in zip(subjects, values)], dtype=object)
        folds_shared, examples = 0, []
        for train_trials, test_trials in split_trials:
            shared = set(values[trial_position.get_indexer(train_trials)]) & set(
                values[trial_position.get_indexer(test_trials)]
            )
            if shared:
                folds_shared += 1
                examples.extend(sorted(map(str, shared))[: 3 - len(examples)])
        label = f"{name} (within {subject_coord})" if scoped else name
        rows.append(
            {
                "check": label,
                "values": len(pd.unique(values)),
                "folds shared": f"{folds_shared}/{len(split_trials)}",
                "examples": ", ".join(examples[:3]),
            }
        )
        if folds_shared:
            findings.append(_describe_shared(name, kind, folds_shared, len(split_trials)))
            shared_coords.append(name)

    if samples is not None:
        digests = np.array(
            [hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest() for row in samples]
        )
        folds_dup, n_dup = 0, 0
        for train_trials, test_trials in split_trials:
            train_digests = set(digests[trial_position.get_indexer(train_trials)])
            n_fold = sum(digest in train_digests for digest in digests[trial_position.get_indexer(test_trials)])
            folds_dup += bool(n_fold)
            n_dup += n_fold
        rows.append(
            {
                "check": "identical samples",
                "values": len(set(digests)),
                "folds shared": f"{folds_dup}/{len(split_trials)}",
                "examples": f"{n_dup} test samples" if n_dup else "",
            }
        )
        if n_dup:
            findings.append(
                f"{n_dup} test samples have an exact copy in the training set ({folds_dup} of {len(split_trials)} "
                "folds), so the model is partly scored on data it was trained on."
            )

    notes = []
    if not coords:
        notes.append(
            "No subject, session or source-trial coordinate recognised; pass groups= to audit another coordinate."
        )
    return SplitReport(
        n_folds=len(split_trials),
        table=pd.DataFrame(rows),
        findings=findings,
        shared_coords=shared_coords,
        notes=notes,
    )


def _describe_shared(name: str, kind: str | None, folds_shared: int, n_folds: int) -> str:
    where = f"in {folds_shared} of {n_folds} folds"
    if kind == "source":
        return (
            f"Samples cut from the same '{name}' are in both training and test sets {where}; near-copies of "
            "training samples make the score optimistic."
        )
    unit = name.removesuffix("_id").removesuffix("_ids")
    return (
        f"The same '{name}' values appear in training and test sets {where}: the score measures performance on "
        f"new trials from a {unit} seen in training, not on a new {unit}."
    )
