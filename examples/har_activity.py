"""Human activity recognition from smartphone sensors with xdflow, next to the plain scikit-learn version.

Classifies six activities (walking, walking upstairs, walking downstairs, sitting, standing,
lying down) from 2.56 s windows of a waist-worn phone's accelerometer and gyroscope, using the
UCI Human Activity Recognition Using Smartphones dataset (30 subjects). It asks how well the
model works on people it has never seen, and shows how much a random split over windows
overstates that.

Run from the repository root with:

    uv run python examples/har_activity.py

The first run downloads the 61 MB UCI zip and keeps the 61 MB inner archive in ~/xdflow_data
(set XDFLOW_DATA_DIR or pass --data-dir to choose another directory). Signals are read straight
from that archive; nothing else is extracted. Only base xdflow dependencies are needed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xdflow.composite import Pipeline
from xdflow.core import DataContainer, Transform
from xdflow.cv import KFoldValidator, LeaveGroupOutValidator
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ROOT = "UCI HAR Dataset"
CHANNELS = [f"{sensor}_{axis}" for sensor in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]
FS = 50  # Hz; every window is 128 samples = 2.56 s, consecutive windows overlap by 50%
BANDS = ((0, 1), (1, 3), (3, 6), (6, 12), (12, 26))  # Hz; the last band includes the 25 Hz Nyquist bin
FEATURES = ["mean", "std", "min", "max", "p25", "p75"] + [f"logpow_{lo}_{hi}hz" for lo, hi in BANDS]
SEED = 0


# --- 1. Load: UCI text files -> one DataArray with trial-level metadata ---------------------------
def check_archive(archive: Path) -> Path:
    """Fail loudly unless the dataset archive has every file the loader reads."""
    expected = {f"{ROOT}/activity_labels.txt"} | {
        f"{ROOT}/{split}/{name}_{split}.txt"
        for split in ("train", "test")
        for name in ("y", "subject", *(f"Inertial Signals/{channel}" for channel in CHANNELS))
    }
    with zipfile.ZipFile(archive) as z:
        missing = expected - set(z.namelist())
    if missing:
        raise RuntimeError(f"{archive} lacks {sorted(missing)}; delete it to download again.")
    return archive


def fetch_har(data_dir: Path) -> Path:
    """Download once and return the checked inner dataset archive."""
    archive = data_dir / f"{ROOT}.zip"
    if not archive.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} (61 MB) to {data_dir} ...")
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:  # only a complete, checked archive enters the cache
            outer = Path(tmp) / "download.zip"
            urllib.request.urlretrieve(URL, outer)
            with zipfile.ZipFile(outer) as z:
                z.extract(archive.name, tmp)
            os.replace(check_archive(Path(tmp) / archive.name), archive)
    return check_archive(archive)


def load_har(archive: Path) -> xr.DataArray:
    """(trial, channel, time) windows with activity, subject, split and segment coordinates."""
    pieces = []
    with zipfile.ZipFile(archive) as z:
        codes = dict(line.split() for line in z.read(f"{ROOT}/activity_labels.txt").decode().splitlines() if line)
        for split in ("train", "test"):
            signals = np.stack(
                [np.loadtxt(z.open(f"{ROOT}/{split}/Inertial Signals/{ch}_{split}.txt")) for ch in CHANNELS], axis=1
            )
            activity = np.loadtxt(z.open(f"{ROOT}/{split}/y_{split}.txt"), dtype=int)
            subject = np.loadtxt(z.open(f"{ROOT}/{split}/subject_{split}.txt"), dtype=int)
            if signals.shape != (len(activity), len(CHANNELS), 128) or len(subject) != len(activity):
                raise ValueError(f"{split}: inconsistent shapes {signals.shape}, {activity.shape}, {subject.shape}")
            pieces.append(
                xr.DataArray(
                    signals,
                    dims=("trial", "channel", "time"),
                    coords={
                        "channel": CHANNELS,
                        "time": np.arange(128) / FS,
                        "activity": ("trial", [codes[str(code)].lower() for code in activity]),
                        "subject": ("trial", subject),
                        "split": ("trial", np.full(len(activity), split)),
                    },
                )
            )
    data = xr.concat(pieces, dim="trial", join="exact")  # fails loudly if channels or times differ
    # A segment is a run of consecutive windows (file order) from one subject doing one activity.
    activity, subject = data["activity"].values, data["subject"].values
    new_segment = np.r_[True, (activity[1:] != activity[:-1]) | (subject[1:] != subject[:-1])]
    return data.assign_coords(trial=np.arange(data.sizes["trial"]), segment=("trial", np.cumsum(new_segment)))


# --- 2. Features: one NumPy function, wrapped in a small stateless transform -----------------------
def window_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Summary statistics and log band power over the last axis (time); returns (..., len(FEATURES))."""
    power = np.abs(np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    band_power = [power[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) for lo, hi in BANDS]
    stats = [x.mean(-1), x.std(-1), x.min(-1), x.max(-1), *np.percentile(x, [25, 75], axis=-1)]
    return np.stack(stats + [np.log(p + 1e-10) for p in band_power], axis=-1)


class WindowFeaturesTransform(Transform):
    """Per-channel window features. Nothing is learned from the data, so it is declared stateless."""

    is_stateful = False
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "feature")

    def __init__(self, fs: float):
        super().__init__()
        self.fs = fs

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data.transpose("trial", "channel", "time")
        if not np.allclose(np.diff(data["time"].values), 1 / self.fs):
            raise ValueError(f"The time coordinate is not sampled at fs={self.fs} Hz.")
        trial_coords = {name: coord for name, coord in data.coords.items() if coord.dims == ("trial",)}
        features = xr.DataArray(
            window_features(data.values, self.fs),
            dims=("trial", "channel", "feature"),
            coords={**trial_coords, "channel": data["channel"].values, "feature": FEATURES},
            attrs=data.attrs,
        )
        return DataContainer(features)


# --- 3. xdflow: one pipeline; the validator's group_coord sets the split boundary -------------------
def build_pipeline() -> Pipeline:
    return Pipeline(
        name="har",
        steps=[
            ("features", WindowFeaturesTransform(fs=FS)),  # stateless: computed once, before the folds
            ("flatten", FlattenTransform(dims=("channel", "feature"))),
            ("scale", SKLearnTransformer(StandardScaler, sample_dim="trial")),  # stateful: refit per fold
            (
                "classifier",
                SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="activity", max_iter=2000),
            ),
        ],
    )


def run_xdflow(data: xr.DataArray) -> dict:
    common = {"scoring": "accuracy", "verbose": False}
    train = DataContainer(data.isel(trial=data["split"].values == "train"))  # the 21 official training subjects
    validators = {
        "windows": KFoldValidator(n_splits=5, shuffle=True, random_state=SEED, **common),
        "segments": LeaveGroupOutValidator(group_coord="segment", n_splits=5, random_state=SEED, **common),
        "subjects": LeaveGroupOutValidator(group_coord="subject", n_splits=5, random_state=SEED, **common),
    }
    scores = {}
    for name, cv in validators.items():
        cv.set_pipeline(build_pipeline())
        scores[name] = cv.cross_validate(train)

    # Leave one subject out on the training subjects, then score the 9 official test subjects, never seen.
    test_subjects = np.unique(data["subject"].values[data["split"].values == "test"]).tolist()
    loso = LeaveGroupOutValidator(group_coord="subject", test_group_ids=test_subjects, **common)
    loso.set_pipeline(build_pipeline())
    scores["loso"] = loso.cross_validate(DataContainer(data))
    scores["holdout"] = loso.score_on_holdout(DataContainer(data))

    # Predictions keep their trial coordinates, so breakdowns by subject or activity are a groupby away.
    oof, held = loso.oof_predictions_container_.data, loso.holdout_container_.data
    per_subject_cv = (oof == oof["activity"]).groupby("subject").mean().to_series()
    if not np.isclose(per_subject_cv.mean(), scores["loso"]):  # one subject per fold, so these must agree
        raise AssertionError("LOSO score should be the mean of per-subject out-of-fold accuracy.")
    per_subject_holdout = (held == held["activity"]).groupby("subject").mean().to_series()
    scores["per_subject"] = pd.concat([per_subject_cv, per_subject_holdout]).sort_index()
    scores["confusion"] = pd.crosstab(
        oof["activity"].values, oof.values, normalize="index", rownames=["true"], colnames=["predicted"]
    )
    scores["test_subjects"] = test_subjects
    return scores


# --- 4. The same analysis in plain NumPy + scikit-learn ---------------------------------------------
SKLEARN_MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "k-nearest neighbours (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "random forest (100 trees)": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
}


def run_sklearn(data: xr.DataArray, model_name: str, split_names: tuple[str, ...]) -> dict:
    # A plain workflow keeps aligned positional arrays (taken here from the same DataArray).
    X = window_features(data.transpose("trial", "channel", "time").values, FS).reshape(data.sizes["trial"], -1)
    y, subjects, segments = data["activity"].values, data["subject"].values, data["segment"].values
    train = data["split"].values == "train"
    splits = {
        "windows": (StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), None),
        "segments": (GroupKFold(n_splits=5), segments[train]),
        "subjects": (GroupKFold(n_splits=5), subjects[train]),
        "loso": (LeaveOneGroupOut(), subjects[train]),
    }
    model = SKLEARN_MODELS[model_name]
    scores = {
        name: cross_val_score(model(), X[train], y[train], cv=splits[name][0], groups=splits[name][1]).mean()
        for name in split_names
    }
    scores["holdout"] = model().fit(X[train], y[train]).score(X[~train], y[~train])
    return scores


# --- 5. Run both and print the tables ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = os.environ.get("XDFLOW_DATA_DIR", str(Path.home() / "xdflow_data"))
    parser.add_argument("--data-dir", default=default_dir, help=f"download/cache directory (default: {default_dir})")
    args = parser.parse_args()

    start = time.perf_counter()
    data = load_har(fetch_har(Path(args.data_dir).expanduser()))
    print(f"Loaded {dict(data.sizes)} in {time.perf_counter() - start:.1f}s")
    counts = pd.crosstab(data["activity"].values, data["split"].values, rownames=["activity"], colnames=["split"])
    subjects = data["subject"].to_series().groupby(data["split"].values).nunique()
    print(f"Windows per activity ({subjects['train']} train / {subjects['test']} test subjects):\n{counts.to_string()}")
    print(f"Majority-class accuracy: {(counts.max() / counts.sum()).round(3).to_dict()}")

    xdf = run_xdflow(data)
    skl = run_sklearn(data, "logistic regression", ("windows", "segments", "subjects", "loso"))
    # Does the inflation depend on the model? Same features and splits, two more flexible classifiers.
    by_model = {
        name: run_sklearn(data, name, ("windows", "subjects"))
        for name in SKLEARN_MODELS
        if name != "logistic regression"
    }

    print(f"\n{'Accuracy, logistic regression':<40}{'xdflow':>8}{'sklearn':>9}")
    for key, name in [
        ("windows", "5-fold, windows shuffled"),
        ("segments", "5-fold, segments kept apart"),
        ("subjects", "5-fold, subjects kept apart"),
        ("loso", "Leave one subject out (21 folds)"),
        ("holdout", "Holdout: 9 never-seen test subjects"),
    ]:
        print(f"{name:<40}{xdf[key]:>8.3f}{skl[key]:>9.3f}")
    print(f"\n{'Accuracy by model (sklearn)':<30}{'windows shuffled':>18}{'subjects apart':>16}{'never-seen':>12}")
    for name, scores in {"logistic regression": skl, **by_model}.items():
        print(f"{name:<30}{scores['windows']:>18.3f}{scores['subjects']:>16.3f}{scores['holdout']:>12.3f}")

    print("\nAccuracy by subject (LOSO out-of-fold; * = test subject, scored by the model fit on all 21 others):")
    per_subject, test_subjects = xdf["per_subject"], xdf["test_subjects"]
    print(", ".join(f"{s}{'*' if s in test_subjects else ''} {acc:.2f}" for s, acc in per_subject.items()))
    for name, part in [("training", per_subject.drop(test_subjects)), ("test", per_subject.loc[test_subjects])]:
        print(f"  {name} subjects: mean {part.mean():.3f}, SD {part.std():.3f}, min {part.min():.2f}")
    print(f"\nLOSO out-of-fold confusion, % of each true activity:\n{(100 * xdf['confusion']).round(1).to_string()}")


if __name__ == "__main__":
    main()
