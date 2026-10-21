"""check_split reports subjects, sessions, source trials and duplicate samples shared across a split."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

from xdflow import KFoldValidator, Pipeline
from xdflow.diagnostics import _grouping_kind, check_split
from xdflow.interop import from_numpy
from xdflow.transforms.sklearn_transform import SKLearnPredictor


@pytest.fixture
def X():  # noqa: N802 - scikit-learn's name for the feature array
    return np.random.default_rng(0).normal(size=(60, 4))


@pytest.fixture
def y():
    return np.array(["a", "b"])[np.arange(60) % 2]


@pytest.fixture
def info():
    # Session numbers restart for every subject, as they usually do.
    return pd.DataFrame({"subject": np.repeat(np.arange(6), 10), "session": np.tile(np.repeat([1, 2], 5), 6)})


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("subject", "subject"),
        ("subject_id", "subject"),
        ("SubjectID", "subject"),
        ("participant", "subject"),
        ("mouse_ids", "subject"),
        ("session", "recording"),
        ("run", "run"),
        ("block_nr", "run"),
        ("days", "recording"),
        ("orig_trial", "source"),
        ("participant_code", "subject"),
        ("recording_session", "recording"),
        ("orig_trial_id", "source"),
        ("epoch_id", "source"),
        ("epoch", "source"),
        ("trial_id", "source"),
        ("trial_nr", "source"),
        ("subject_age", None),
        ("session_duration", None),
        ("run_length", None),
        ("runtime", None),
        ("label", None),
        ("condition", None),
        ("trial", None),
    ],
)
def test_recognised_coordinate_names(name, kind):
    assert _grouping_kind(name) == kind


def test_shuffled_kfold_shares_subjects(X, y, info):
    report = check_split(X, StratifiedKFold(5, shuffle=True, random_state=0), y=y, metadata=info)
    assert not report.ok
    assert report.n_folds == 5
    assert any("'subject'" in finding and "5 of 5 folds" in finding for finding in report.findings)
    assert report.table.set_index("check").loc["subject", "folds shared"] == "5/5"


def test_grouped_split_is_clean_and_sessions_are_scoped_by_subject(X, y, info):
    report = check_split(X, GroupKFold(3), y=y, groups="subject", metadata=info)
    assert report.ok, report
    # Unscoped, session "1" would look shared in every fold; within subject it is not.
    assert report.table.set_index("check").loc["session (within subject)", "folds shared"] == "0/3"


def test_duplicate_samples_are_found(X, y, info):
    X = X.copy()
    X[5] = X[50]
    report = check_split(X, GroupKFold(3), y=y, groups="subject", metadata=info)
    assert not report.ok
    assert "exact copy" in report.findings[0]


def test_windows_from_one_source_trial(X, y):
    data = from_numpy(X, ("trial", "feature"), label=y, orig_trial=np.repeat(np.arange(20), 3))
    report = check_split(data, StratifiedKFold(5, shuffle=True, random_state=0), y="label")
    assert any("same 'orig_trial'" in finding for finding in report.findings)


def test_int_cv_matches_scikit_learn(X, y, info):
    report = check_split(X, 5, y=y, metadata=info)
    expected = sum(
        bool(set(info.subject[train]) & set(info.subject[test])) for train, test in StratifiedKFold(5).split(X, y)
    )
    assert report.table.set_index("check").loc["subject", "folds shared"] == f"{expected}/5"


def test_index_pairs_and_unrecognised_columns(X):
    report = check_split(X, [(np.arange(30), np.arange(30, 60))])
    assert report.ok and report.n_folds == 1
    assert "pass groups=" in report.notes[0]
    with pytest.raises(ValueError, match=r"in \[0, 60\)"):
        check_split(X, [(np.arange(30), np.arange(30, 61))])


def test_validator_splits_are_audited_after_cross_validation(X, y, info):
    data = from_numpy(X, ("trial", "feature"), metadata=info, label=y)
    validator = KFoldValidator(n_splits=3, verbose=False)
    with pytest.raises(ValueError, match="not run yet"):
        check_split(data, validator)
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    validator.cross_validate(data)
    assert not check_split(data, validator).ok
    with pytest.raises(ValueError, match="not in the data"):
        check_split(data.data.assign_coords(trial=np.arange(60) + 1000), validator)


def test_a_single_subject_is_reported_as_shared(X, y):
    info = pd.DataFrame({"subject": "S1", "session": np.repeat([1, 2], 30)})
    report = check_split(X, GroupKFold(2), y=y, groups="session", metadata=info)
    assert report.shared_coords == ["subject"]
    assert report.table.set_index("check").loc["session (within subject)", "folds shared"] == "0/2"


def test_runs_are_scoped_by_session(X, y):
    # MOABB-style: run ids restart in every session; holding out a session shares no run.
    info = pd.DataFrame(
        {"subject": 1, "session": np.repeat(["0train", "1test"], 30), "run": np.tile(np.repeat([0, 1, 2], 10), 2)}
    )
    report = check_split(X, GroupKFold(2), y=y, groups="session", metadata=info)
    assert report.shared_coords == ["subject"]
    assert report.table.set_index("check").loc["run (within subject, session)", "folds shared"] == "0/2"


def test_sessions_are_scoped_by_subject_ids_that_never_repeat():
    info = pd.DataFrame({"subject": ["a", "b"], "session": [1, 1]})
    assert check_split(np.array([[0.0], [1.0]]), [([0], [1])], metadata=info).ok


def test_a_validators_group_coordinate_is_audited(X, y):
    from sklearn.model_selection import KFold

    from xdflow.cv import SplitterValidator

    data = from_numpy(X, ("trial", "feature"), label=y, device=np.arange(60) % 3)
    validator = SplitterValidator(KFold(2), group_coord="device", verbose=False)
    validator.set_pipeline(
        Pipeline("p", [("lr", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label"))])
    )
    validator.cross_validate(data)
    assert check_split(data, validator).shared_coords == ["device"]


def test_duplicates_are_found_in_object_arrays():
    samples = np.empty((2, 1), dtype=object)
    samples[0, 0], samples[1, 0] = float("1.5"), float("1.0") + 0.5  # equal values, distinct objects
    assert not check_split(samples, [([0], [1])]).ok


def test_a_stratified_split_without_y_explains(X):
    with pytest.raises(ValueError, match="y="):
        check_split(X, StratifiedKFold(3))


def test_runs_are_not_scoped_by_other_recording_coordinates():
    info = pd.DataFrame({"session": 1, "run": 1, "day": [1, 1, 2, 2]})  # one run spanning midnight
    report = check_split(np.arange(4.0)[:, None], [([0, 1], [2, 3])], metadata=info)
    assert "run" in report.shared_coords


def test_windows_cut_from_one_epoch_are_found_under_common_names(X, y):
    data = from_numpy(X, ("trial", "feature"), label=y, epoch_id=np.repeat(np.arange(20), 3))
    report = check_split(data, StratifiedKFold(3, shuffle=True, random_state=0), y="label")
    assert "near-copies" in report.findings[0]


def test_a_report_with_nothing_to_audit_says_so(X):
    report = check_split(X, 3)
    assert report.ok
    assert "nothing shared" not in repr(report)
    assert "no subject, session or source-trial coordinate" in repr(report)


def test_findings_use_the_right_article(X, y):
    data = from_numpy(X, ("trial", "feature"), label=y, animal=np.repeat(np.arange(6), 10))
    finding = check_split(data, StratifiedKFold(3, shuffle=True, random_state=0), y="label").findings[0]
    assert "from an animal seen in training" in finding


def test_missing_ids_are_reported_rather_than_certified_apart():
    data = from_numpy(np.arange(8.0).reshape(4, 2), ("trial", "feature"), subject=np.full(4, np.nan))
    report = check_split(data, [([0, 1], [2, 3])])
    assert not report.ok
    assert "no 'subject'" in report.findings[0]


def test_signed_zeros_are_duplicates():
    assert not check_split(np.array([[0.0, 1.0], [-0.0, 1.0]]), [([0], [1])]).ok


def test_mixed_type_trial_labels_are_audited():
    data = from_numpy(np.arange(8.0).reshape(4, 2), ("trial", "feature"), subject=[0, 0, 1, 1]).data
    data = data.assign_coords(trial=np.array([1, "a", 2, "b"], dtype=object))
    assert check_split(data, [([0, 1], [2, 3])]).ok


def test_source_trials_are_scoped_by_session():
    # orig_trial restarts in every session: (session, orig_trial) is the source trial
    info = pd.DataFrame({"session": np.repeat([1, 2], 30), "orig_trial": np.tile(np.repeat(np.arange(10), 3), 2)})
    sessions_apart = [(np.arange(30), np.arange(30, 60))]
    report = check_split(np.random.default_rng(0).normal(size=(60, 2)), sessions_apart, metadata=info)
    assert "orig_trial" not in report.shared_coords
