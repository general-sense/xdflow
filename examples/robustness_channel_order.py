"""What happens when the same channels arrive in a different order?

Recordings from another session, headset or export tool often list the same channels in a different order.
Positional code (NumPy arrays into a scikit-learn pipeline) cannot tell: a CSP model applies its spatial filters
to whatever is in each column, with no error. xdflow addresses channels by name, so the same model gives the
same predictions whatever the order (and raises if a channel it was fitted on is missing).

Same PhysioNet motor-imagery data and band-pass -> crop -> CSP -> LDA model as eeg_motor_imagery.py; within-subject
5-fold CV for subjects 1-5, scoring each fold's test trials twice: in the original channel order, and with the
channels in a shuffled order.

Run from the repository root (downloads ~39 MB on the first run):

    uv run --extra mne python examples/robustness_channel_order.py
"""

import warnings

import numpy as np
from eeg_motor_imagery import BAND, FS, N_CSP, SEED, WINDOW, load_motor_imagery
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from xdflow import DataContainer, Pipeline
from xdflow.transforms.basic_transforms import CropTimeTransform
from xdflow.transforms.csp import CSPTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.transforms.spectral import BandpassFilterTransform

SUBJECTS = [1, 2, 3, 4, 5]


def xdflow_model() -> Pipeline:
    return Pipeline(
        "csp_lda",
        [
            ("bandpass", BandpassFilterTransform(*BAND, fs=FS)),
            ("crop", CropTimeTransform(*WINDOW)),
            ("csp", CSPTransform(n_components=N_CSP)),
            ("lda", SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label")),
        ],
    )


def main() -> None:
    warnings.simplefilter("ignore")
    data = load_motor_imagery(SUBJECTS)
    shuffled_channels = np.random.default_rng(SEED).permutation(data.sizes["channel"])
    b, a = butter(4, BAND, btype="bandpass", fs=FS)
    accuracy = {"sklearn": ([], []), "xdflow": ([], [])}  # (original order, shuffled order) per fold

    for subject in SUBJECTS:
        subject_data = data.isel(trial=data["subject"].values == subject)
        y, trials = subject_data["label"].values, subject_data["trial"].values
        in_window = (subject_data["time"].values >= WINDOW[0]) & (subject_data["time"].values <= WINDOW[1])
        X = filtfilt(b, a, subject_data.values, axis=-1)[..., in_window]

        for train, test in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            sklearn_model = make_pipeline(CSP(n_components=N_CSP, log=True), LinearDiscriminantAnalysis())
            sklearn_model.fit(X[train], y[train])
            accuracy["sklearn"][0].append(np.mean(sklearn_model.predict(X[test]) == y[test]))
            accuracy["sklearn"][1].append(np.mean(sklearn_model.predict(X[test][:, shuffled_channels]) == y[test]))

            model = xdflow_model().fit(DataContainer(subject_data.sel(trial=trials[train])))
            test_data = subject_data.sel(trial=trials[test])
            reordered = test_data.isel(channel=shuffled_channels)
            accuracy["xdflow"][0].append(np.mean(model.predict(DataContainer(test_data)).data.values == y[test]))
            accuracy["xdflow"][1].append(np.mean(model.predict(DataContainer(reordered)).data.values == y[test]))

    print(f"Within-subject 5-fold accuracy, subjects {SUBJECTS} (chance = 0.50)")
    print(f"{'':10}{'original channel order':>25}{'shuffled channel order':>25}")
    for name, (original, shuffled) in accuracy.items():
        print(f"{name:10}{np.mean(original):>25.3f}{np.mean(shuffled):>25.3f}")


if __name__ == "__main__":
    main()
