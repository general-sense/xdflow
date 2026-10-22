"""Evaluate an existing scikit-learn model with one call, and check a split you already use.

Uses the PhysioNet EEG motor imagery recordings from eeg_motor_imagery.py (imagined left vs right fist) and
the classic MNE model, CSP + LDA, exactly as an MNE user would write it.

1. `xdflow.evaluate` cross-validates it across subjects: band-pass filtering and cropping run once on all
   trials, the model is refit inside each fold, and the result is broken down by subject.
2. `xdflow.check_split` audits the common shuffled K-fold split of the same trials, without running a model.
3. `xdflow.to_sklearn` goes the other way: the xdflow pipeline inside scikit-learn's own cross_val_score.

Run from the repository root (downloads ~39 MB on the first run):

    uv run --extra mne python examples/evaluate_existing_model.py
"""

import mne
from eeg_motor_imagery import BAND, FS, SEED, WINDOW, load_motor_imagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

import xdflow
from xdflow.transforms import BandpassFilterTransform, CropTimeTransform

SUBJECTS = [1, 2, 3, 4, 5]


def main() -> None:
    mne.set_log_level("ERROR")
    data = load_motor_imagery(SUBJECTS)  # (trial, channel, time) with label, subject and run coordinates

    # 1. The model as you would write it for MNE + scikit-learn, unchanged.
    existing = make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())
    result = xdflow.evaluate(
        [BandpassFilterTransform(*BAND, fs=FS), CropTimeTransform(*WINDOW), existing],
        data,
        groups="subject",
    )
    print(result, end="\n\n")

    # 2. What does the usual shuffled, stratified 5-fold split share between training and test sets?
    print(xdflow.check_split(data, StratifiedKFold(5, shuffle=True, random_state=SEED), y="label"), end="\n\n")

    # 3. The same xdflow pipeline inside scikit-learn: same splits, same scores.
    estimator = xdflow.to_sklearn(
        result.pipeline,
        dims=("trial", "channel", "time"),
        coords={"channel": data.channel.values, "time": data.time.values},
    )
    scores = cross_val_score(
        estimator, data.values, data.label.values, groups=data.subject.values, cv=GroupKFold(len(SUBJECTS))
    )
    print(f"scikit-learn cross_val_score, GroupKFold by subject: {scores.mean():.3f}")


if __name__ == "__main__":
    main()
