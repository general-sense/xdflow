"""EEG motor imagery decoding with xdflow, next to the plain scikit-learn version.

Decodes imagined left- vs right-fist movement from the PhysioNet EEG Motor
Movement/Imagery dataset (runs 4, 8 and 12) with a classic band-pass -> CSP -> LDA
model, and asks two different questions of it:

1. Within subject: does a model calibrated on a person's own trials decode their new trials?
2. Across subjects: does a model trained on other people decode a person it has never seen?

Run from the repository root with:

    uv run --extra mne python examples/eeg_motor_imagery.py

The first run downloads ~39 MB of EDF files (5 subjects x 3 runs) from PhysioNet into
MNE's data directory (~/mne_data unless MNE_DATA is set); use --data-dir to choose another.
MNE reads and epochs the recordings and provides CSP (`pip install "xdflow[mne]"`); `from_mne_epochs`
turns each run's epochs into an xdflow container.
"""

from __future__ import annotations

import argparse
import time

import mne
import numpy as np
import xarray as xr
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from xdflow import from_mne_epochs
from xdflow.composite import GroupApplyTransform, Pipeline
from xdflow.core import DataContainer, Transform
from xdflow.cv import GroupedKFoldValidator, LeaveGroupOutValidator
from xdflow.transforms.basic_transforms import CropTimeTransform
from xdflow.transforms.csp import CSPTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.transforms.spectral import BandpassFilterTransform

RUNS = (4, 8, 12)  # imagery runs; annotation T1 = left fist, T2 = right fist
FS = 160  # Hz; load_motor_imagery rejects recordings at any other rate
EPOCH = (-1.0, 4.5)  # s around the cue; margins absorb filter edge effects (runs end ~4.6 s after the last cue)
WINDOW = (0.5, 4.0)  # s after the cue that the model sees
BAND = (8.0, 30.0)  # Hz, mu + beta rhythms, which desynchronize over the contralateral motor cortex
N_CSP = 6
SEED = 0


# --- 1. Load: EDF files -> one DataArray with trial-level metadata ---------------------------------
def load_motor_imagery(subjects: list[int], data_dir: str | None = None) -> xr.DataArray:
    """Epoch the imagery runs into (trial, channel, time) with label/subject/run coordinates."""
    pieces = []
    for subject in subjects:
        for run in RUNS:
            [path] = eegbci.load_data(subject, [run], path=data_dir, update_path=False)
            raw = mne.io.read_raw_edf(path, preload=True)
            eegbci.standardize(raw)  # canonical 10-20 channel names
            if raw.info["sfreq"] != FS:
                raise ValueError(f"S{subject:03d}R{run:02d}: expected {FS} Hz, got {raw.info['sfreq']} Hz")
            events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
            epochs = mne.Epochs(raw, events, {"left": 1, "right": 2}, *EPOCH, baseline=None, preload=True)
            # labels come from the event names; subject and run become trial coordinates
            pieces.append(from_mne_epochs(epochs, subject=subject, run=run).data)
    # join="exact" fails loudly if channel names or time stamps differ between recordings.
    data = xr.concat(pieces, dim="trial", join="exact")
    return data.assign_coords(trial=np.arange(data.sizes["trial"]))


# --- 2. xdflow: one pipeline from epochs to predictions; validators pick the split boundary ---------
def build_pipeline(per_subject: bool = False) -> Pipeline:
    model: Transform = Pipeline(
        name="csp_lda",
        steps=[
            ("csp", CSPTransform(n_components=N_CSP)),  # wraps mne.decoding.CSP; stateful: refit on every fold
            ("lda", SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label")),
        ],
    )
    if per_subject:  # a separate CSP+LDA for each value of the `subject` coordinate
        model = GroupApplyTransform(group_coord="subject", transform_template=model)
    return Pipeline(
        name="motor_imagery",
        steps=[
            ("bandpass", BandpassFilterTransform(*BAND, fs=FS)),  # stateless: runs once, before the folds
            ("crop", CropTimeTransform(*WINDOW)),  # stateless: label-based slice of the time coord
            ("model", model),
        ],
    )


def run_xdflow(container: DataContainer, holdout_subject: int) -> dict:
    # (a) Within subject: per-subject models, folds stratified by subject x label,
    #     and 20% of every subject's trials held out until the end.
    within = GroupedKFoldValidator(
        n_splits=5, test_size=0.2, group_coord="subject", scoring="accuracy", random_state=SEED, verbose=False
    )
    within.set_pipeline(build_pipeline(per_subject=True))
    within_cv = within.cross_validate(container)
    within_holdout = within.score_on_holdout(container)

    # (b) Across subjects: leave-one-subject-out on the others, then score the never-seen holdout subject.
    across = LeaveGroupOutValidator(
        group_coord="subject", test_group_ids=[holdout_subject], scoring="accuracy", verbose=False
    )
    across.set_pipeline(build_pipeline())
    across_cv = across.cross_validate(container)
    across_holdout = across.score_on_holdout(container)

    # Out-of-fold predictions keep their trial coordinates, so per-subject CV accuracy is a groupby away.
    predicted = within.oof_predictions_container_.data
    per_subject_cv = (predicted == predicted["label"]).groupby("subject").mean().to_series()
    return {
        "within_cv": within_cv,
        "within_holdout": within_holdout,
        "across_cv": across_cv,
        "across_holdout": across_holdout,
        "per_subject_cv": per_subject_cv,
        "cv_counts": predicted["subject"].to_series().value_counts().sort_index(),
    }


# --- 3. The same analysis in plain NumPy + scikit-learn ---------------------------------------------
def run_sklearn(data: xr.DataArray, holdout_subject: int) -> dict:
    # A plain workflow keeps aligned positional arrays (taken here from the same DataArray).
    X, y, subjects, times = data.values, data["label"].values, data["subject"].values, data["time"].values
    b, a = butter(4, BAND, btype="bandpass", fs=FS)
    X = filtfilt(b, a, X, axis=-1)  # axis -1 must be time
    X = X[..., (times >= WINDOW[0]) & (times <= WINDOW[1])]

    def model():
        return make_pipeline(CSP(n_components=N_CSP, log=True), LinearDiscriminantAnalysis())

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    within = [
        cross_val_score(model(), X[subjects == s], y[subjects == s], cv=kfold).mean() for s in np.unique(subjects)
    ]

    train = subjects != holdout_subject
    across = cross_val_score(model(), X[train], y[train], groups=subjects[train], cv=LeaveOneGroupOut())
    across_holdout = model().fit(X[train], y[train]).score(X[~train], y[~train])

    # Common shortcut: compute CSP features once on all trials "to save time", then cross-validate
    # only the classifier. CSP has then seen the validation trials and their labels.
    leaky_within = []
    for s in np.unique(subjects):
        features = CSP(n_components=N_CSP, log=True).fit_transform(X[subjects == s], y[subjects == s])
        leaky_within.append(cross_val_score(LinearDiscriminantAnalysis(), features, y[subjects == s], cv=kfold).mean())
    features = CSP(n_components=N_CSP, log=True).fit_transform(X[train], y[train])
    leaky_across = cross_val_score(
        LinearDiscriminantAnalysis(), features, y[train], groups=subjects[train], cv=LeaveOneGroupOut()
    )
    return {
        "within_cv": np.mean(within),
        "across_cv": across.mean(),
        "across_holdout": across_holdout,
        "leaky_within_cv": np.mean(leaky_within),
        "leaky_across_cv": leaky_across.mean(),
    }


# --- 4. Run both and print one table ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="the last one is held out")
    parser.add_argument("--data-dir", default=None, help="download/cache directory (default: MNE's data dir)")
    args = parser.parse_args()
    if len(args.subjects) < 3:
        raise ValueError("Need at least 3 subjects: leave-one-subject-out on >= 2, plus one holdout subject.")
    mne.set_log_level("ERROR")
    holdout_subject = args.subjects[-1]

    start = time.perf_counter()
    data = load_motor_imagery(args.subjects, args.data_dir)
    print(f"Loaded {dict(data.sizes)} in {time.perf_counter() - start:.1f}s")
    counts = data["label"].to_series().groupby(data["subject"].values).value_counts().unstack()
    print(f"Trials per subject:\n{counts.to_string()}\n")

    xdf = run_xdflow(DataContainer(data), holdout_subject)
    skl = run_sklearn(data, holdout_subject)

    rows = [
        ("Within subject, 5-fold CV", xdf["within_cv"], skl["within_cv"], skl["leaky_within_cv"]),
        ("Within subject, 20% holdout", xdf["within_holdout"], None, None),
        (
            f"Across subjects, LOSO CV (excl. S{holdout_subject})",
            xdf["across_cv"],
            skl["across_cv"],
            skl["leaky_across_cv"],
        ),
        (f"Across subjects, holdout S{holdout_subject}", xdf["across_holdout"], skl["across_holdout"], None),
    ]
    print(f"{'Accuracy (chance = 0.50)':<45}{'xdflow':>8}{'sklearn':>9}{'sklearn, CSP fit before CV':>28}")
    for name, *scores in rows:
        cells = ["-" if score is None else f"{score:.3f}" for score in scores]
        print(f"{name:<45}{cells[0]:>8}{cells[1]:>9}{cells[2]:>28}")

    by_subject = ", ".join(f"S{s} {acc:.2f} (n={xdf['cv_counts'][s]})" for s, acc in xdf["per_subject_cv"].items())
    print(f"\nWithin-subject CV accuracy by subject: {by_subject}")


if __name__ == "__main__":
    main()
