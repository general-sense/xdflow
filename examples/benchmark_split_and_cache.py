"""Leak-free tuning, three ways: xdflow's automatic split and cache, scikit-learn with its own caching,
and refitting every step in every fold.

All runs tune the same pipeline (multitaper band power -> flatten -> z-score -> logistic regression) over
the same C values, with the same folds and the same accuracy metric, and all are leak-free: every learned
step is fit inside the training folds only. They differ only in where the work happens.

- "xdflow split + cache": the spectral features learn nothing, so validators compute them once on all
  trials and cache them; each tuning trial only refits z-score + classifier per fold.
- "sklearn Pipeline(memory=...)": the idiomatic careful scikit-learn version, GridSearchCV over a cached
  Pipeline. Fitted feature steps are reused across C values, but features are still computed per training
  fold, and validation-fold features are recomputed for every C value.
- "refit everything": a no-op step marked stateful is placed first, so every step, spectra included, is
  recomputed inside every fold of every trial. This is what a conservative hand-written loop does.

Run from the repository root (needs the `tuning` and `spectral` extras):

    uv run --extra tuning --extra spectral python examples/benchmark_split_and_cache.py --trials 10
"""

import argparse
import os
import tempfile
import time
import warnings

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from xdflow import DataContainer, KFoldValidator, Pipeline, Transform
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.transforms.spectral import MultiTaperTransform
from xdflow.tuning import Tuner

FS = 500
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0), "gamma": (30.0, 80.0)}


class RefitEverythingPerFold(Transform):
    """A no-op marked stateful. Placed first, it makes validators refit every later step inside each fold."""

    is_stateful = True

    def _fit(self, container, **kwargs):
        return self

    def _transform(self, container, **kwargs):
        return container

    def get_expected_output_dims(self, input_dims):
        return input_dims


def make_data(n_trials: int = 240, n_channels: int = 16, n_times: int = 1000, seed: int = 0) -> DataContainer:
    """Two classes; class "b" carries a weak 20 Hz rhythm on the first four channels."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(["a", "b"], n_trials // 2)
    values = rng.normal(size=(n_trials, n_channels, n_times))
    values[labels == "b", :4] += 0.1 * np.sin(2 * np.pi * 20 * np.arange(n_times) / FS)
    return DataContainer(
        xr.DataArray(
            values,
            dims=("trial", "channel", "time"),
            coords={
                "trial": np.arange(n_trials),
                "channel": np.arange(n_channels),
                "time": np.arange(n_times) / FS,
                "y": ("trial", labels),
            },
        )
    )


def make_pipeline(refit_everything: bool) -> Pipeline:
    steps = [
        (
            "spectra",
            MultiTaperTransform(fs=FS, avg_over_time_windows=True, avg_within_freq_bands=True, freq_ranges=BANDS),
        ),
        ("flatten", FlattenTransform(dims=("channel", "freq_band"))),
        ("zscore", ZScoreTransform(per_dim="flat_channel__freq_band", use_fit=True)),  # per feature
        ("clf", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="y", max_iter=500)),
    ]
    if refit_everything:
        steps.insert(0, ("refit_everything", RefitEverythingPerFold()))
    return Pipeline("bench", steps)


def make_validator() -> KFoldValidator:
    return KFoldValidator(n_splits=5, random_state=0, scoring="accuracy", verbose=False)


def run_xdflow(data: DataContainer, refit_everything: bool, n_trials: int) -> tuple[float, list[float], list[float]]:
    os.environ["XDFLOW_CACHE_DIR"] = tempfile.mkdtemp(prefix="xdflow_bench_")  # cold cache for each run
    tuner = Tuner(
        make_pipeline(refit_everything),
        make_validator(),
        {"bench": {"clf": {"C": ("log", 1e-3, 10.0)}}},
        data,
        random_seed=0,
        verbose=0,
    )
    start = time.perf_counter()
    tuner.tune(n_trials=n_trials)
    elapsed = time.perf_counter() - start
    trials = tuner.study.trials
    return elapsed, [t.value for t in trials], [t.params["bench__clf__C"] for t in trials]


def spectral_features(values: np.ndarray) -> np.ndarray:
    """The same multitaper band power + flatten as the xdflow pipeline, on a (trial, channel, time) array."""
    n_trials, n_channels, n_times = values.shape
    array = xr.DataArray(
        values,
        dims=("trial", "channel", "time"),
        coords={"trial": np.arange(n_trials), "channel": np.arange(n_channels), "time": np.arange(n_times) / FS},
    )
    steps = make_pipeline(refit_everything=False).steps[:2]  # spectra, flatten (both stateless)
    container = DataContainer(array)
    for step in steps:
        container = step.transform.transform(container)
    return container.data.transpose("trial", ...).values


def run_sklearn(data: DataContainer, c_values: list[float]) -> tuple[float, list[float]]:
    """GridSearchCV over a memory-cached sklearn Pipeline, on exactly the folds xdflow used."""
    validator = make_validator()
    validator.set_pipeline(make_pipeline(refit_everything=False))
    train_val, _ = validator._split_holdout(data)
    position = {trial: i for i, trial in enumerate(data.data["trial"].values)}
    folds = [
        ([position[t] for t in train], [position[t] for t in val])
        for train, val in validator._get_splits(data, train_val)
    ]
    X = data.data.transpose("trial", "channel", "time").values
    y = data.data["y"].values
    pipeline = SkPipeline(
        [
            ("spectra", FunctionTransformer(spectral_features)),
            ("zscore", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ],
        memory=tempfile.mkdtemp(prefix="sklearn_bench_"),
    )
    search = GridSearchCV(pipeline, {"clf__C": list(c_values)}, cv=folds, scoring="accuracy", refit=False)
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    scores = dict(zip([p["clf__C"] for p in search.cv_results_["params"]], search.cv_results_["mean_test_score"]))
    return elapsed, [scores[c] for c in c_values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    data = make_data()
    split_seconds, split_scores, c_values = run_xdflow(data, refit_everything=False, n_trials=args.trials)
    sklearn_seconds, sklearn_scores = run_sklearn(data, c_values)
    refit_seconds, refit_scores, _ = run_xdflow(data, refit_everything=True, n_trials=args.trials)

    print(
        f"{args.trials} C values x 5 folds, {data.data.sizes['trial']} trials x {data.data.sizes['channel']} channels"
    )
    print(f"  xdflow split + cache          : {split_seconds:6.1f} s")
    print(
        f"  sklearn Pipeline(memory=...)  : {sklearn_seconds:6.1f} s  ({sklearn_seconds / split_seconds:.1f}x slower)"
    )
    print(f"  refit everything in each fold : {refit_seconds:6.1f} s  ({refit_seconds / split_seconds:.1f}x slower)")
    print(f"  accuracy per C value: {np.round(split_scores, 3).tolist()}")
    print(
        f"  same scores in all three: {np.allclose(split_scores, sklearn_scores) and np.allclose(split_scores, refit_scores)}"
    )


if __name__ == "__main__":
    main()
