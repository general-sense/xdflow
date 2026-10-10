# XDFlow

**ML pipelines for structured experimental data: correct by construction, fast, and hard to break.**

[![PyPI](https://img.shields.io/pypi/v/xdflow.svg)](https://pypi.org/project/xdflow/)
[![CI](https://github.com/general-sense/xdflow/actions/workflows/ci.yml/badge.svg)](https://github.com/general-sense/xdflow/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/xdflow/badge/?version=latest)](https://xdflow.readthedocs.io/en/latest/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Experimental data has structure, with trials recorded from subjects, across sessions, over channels and time.
`xdflow` keeps that structure in the data. It is built on [xarray](https://xarray.dev/), so dimensions and
coordinates such as `trial`, `channel`, `time`, `subject` and `session` travel with every array through
preprocessing, fitting, cross-validation, tuning and prediction, and your scikit-learn estimators plug
straight in. Building on structure instead of bare arrays makes validation **correct** and **fast**, and
analysis code **hard to break**.

## Getting started

```bash
pip install xdflow
```

Cross-validate the model you already have with one call. `groups` keeps each person's trials on one side of
every split:

```python
import xdflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))  # any scikit-learn model
result = xdflow.evaluate(model, X, y, groups=subject_ids)  # NumPy arrays, an xarray DataArray or MNE Epochs
print(result)
```

```text
accuracy 0.924 ± 0.029 (mean ± sd over 5 folds, 10299 trials)
Split: 5-fold, grouped (30 groups), stratified by class
Computed once: nothing. Refit in each fold: pipeline (standardscaler, logisticregression).

By group:
       accuracy  n_trials
group
1         0.988       347
2         0.977       302
...
```

(Activity recognition from smartphone sensors, 30 people; [walkthrough](docs/tutorials/har-activity.md).)
`result.predictions` holds every out-of-fold prediction with its trial metadata, `result.by("session")` breaks
the score down by any coordinate, and `result.fit_final()` refits on all the data for new predictions.

Check a split you already use, without running a model:

```python
from sklearn.model_selection import StratifiedKFold

xdflow.check_split(X, StratifiedKFold(5, shuffle=True), y=y, metadata=info)  # info: one row per sample
```

```text
Split check over 5 folds: shared data found.
  - The same 'subject' values appear in training and test sets in 5 of 5 folds: the score measures performance on new trials from a subject seen in training, not on a new subject.
```

Your own preprocessing is a plain function. xdflow computes it once for all folds, after checking that each
trial's output depends on that trial alone:

```python
import numpy as np


@xdflow.step
def log_power(data, band=(8.0, 30.0)):  # data: an xarray DataArray with named dims
    return np.log((data.sel(freq=slice(*band)) ** 2).mean("freq"))


xdflow.evaluate([log_power(), model], data, groups="subject")  # band is tunable as "log_power__band"
```

## Structure is the data

In a NumPy/scikit-learn workflow, structure lives outside the data. `X` is an anonymous array whose axes you
remember by position, and labels, subject IDs, channel names and time stamps are side arrays you keep aligned
by hand through every reshape, filter, crop and split. That bookkeeping is where most decoding bugs live, and
they don't crash. They just give you a wrong number.

In xdflow, every step addresses the data by name:

- **Time by value:** `CropTimeTransform(0.5, 4.0)` selects 0.5–4.0 s from the time coordinate, not an index
  range you computed by hand. A window that selects nothing raises.
- **Dimensions by name:** `per_dim="trial"`, `dims=("channel", "time")`, never `axis=2`. A misspelled
  dimension raises instead of silently pooling over everything.
- **Channels and features by name:** every fitted step (spatial filters, PCA, whitening, your scikit-learn
  model) sees channels and features by label, whatever order a new recording lists them in. A missing channel
  raises.
- **Metadata travels with every trial:** labels, subjects and sessions follow each trial through filtering,
  reordering, splitting and prediction, so there is no side array to fall out of sync.
- **Results keep their context:** out-of-fold and holdout predictions carry their trial coordinates, so
  per-subject or per-condition results are a `groupby`.

## Correct and fast

Every step in a decoding pipeline either learns from data or it doesn't. Scalers, CSP, PCA, feature
selection and the model learn. Filtering, cropping and per-trial spectral features don't. To validate the
pipeline, you have to decide where each step runs:

- **Refit everything inside every fold.** This keeps preprocessing from leaking, but expensive feature
  extraction is recomputed for every fold, and again for every hyperparameter trial.
- **Compute features once, then cross-validate the model.** This is fast, but the moment a learned step
  ends up in the precomputed part, validation and holdout data leak into training and the scores inflate.

Hand-written code fixes a point on that line per project, and it has to be re-checked whenever the pipeline
changes. **`xdflow` finds the optimal point for you, automatically, at every level:**

1. **The validation split point.** The pipeline is cut before its first step that learns. Everything before
   the cut runs once on all trials. Everything after it is cloned and refit inside each training fold, and
   holdout trials are left out of every learned fit and every CV score. The cut follows the configuration:
   a composite step counts as learning if any part of it does, and parameters such as
   `ZScoreTransform(use_fit=True)` turn a step into a learning one. Editing or tuning the pipeline
   therefore can't strand a learned step before the cut. Every step declares whether it learns: built-in
   transforms do it for you, and a custom transform does it in one line.
2. **Reuse across folds, tuning trials and runs.** The shared part runs once per validation run. With
   caching on (the default when tuning; `Pipeline(use_cache=True)` otherwise), its output is stored under
   a hash of the data values, coordinates and attrs, every step's configuration, and the code. Tuning
   trials that only change later steps reuse it instead of recomputing features. Fitted fold models are
   cached under their exact training set.
3. **Splits by name.** Folds and holdouts are defined by coordinates such as `subject` or `session`, so
   keeping correlated samples apart is one argument rather than a `groups` array kept in sync by hand.
   You choose what to keep apart; xdflow keeps it in sync, everywhere.

### Measured

**Safety: leakage makes a decoder look far better than it is.** Same data, same model, validated two ways.
The higher number is wrong, and xdflow gives you the right one.

| Same data, same model | Validated with leakage ✗ | Validated correctly ✓ |
|---|---|---|
| EEG motor imagery, within subject: CSP fit before vs inside CV ([walkthrough](docs/tutorials/eeg-motor-imagery.md)) | 0.96, inflated | **0.64** |
| EEG motor imagery, across subjects: same shortcut | 0.78, inflated | **0.56** (≈ chance) |
| Activity recognition, 30 people: shuffled windows vs subjects kept apart ([walkthrough](docs/tutorials/har-activity.md)) | 0.97, inflated (and picks the wrong model) | **0.91** (9 unseen test people: 0.92) |

Learned steps are refit inside each fold automatically (EEG rows); keeping subjects apart is a single
argument (HAR row). Full details in each walkthrough.

**Robustness: positional code breaks silently; named data doesn't.** The same model, tested on the same
trials, with the channels listed in a different order (as another session or export tool might):

| Same CSP + LDA model, EEG within subject | Original channel order | Shuffled channel order |
|---|---|---|
| Plain NumPy + scikit-learn | 0.64 | **0.52**, chance, with no error ✗ |
| xdflow | 0.64 | **0.64** ✓ |

([`examples/robustness_channel_order.py`](examples/robustness_channel_order.py))

**Speed: correct *and* fast.** Tuning a leak-free spectral pipeline (multitaper spectra → z-score →
logistic regression) over 10 values of C with 5-fold CV. Same folds, same features, identical scores in all
three:

| Every step recomputed in every fold | scikit-learn `Pipeline(memory=...)` + `GridSearchCV` | xdflow |
|---|---|---|
| 91 s | 29 s | **2.9 s** |

**8–10× faster than scikit-learn's cached pipeline, ~30× faster than refitting everything** (two runs,
M3 Max). scikit-learn's cache still recomputes features for every training fold and every validation fold;
xdflow computes them once. The more expensive your feature extraction, the bigger the win.
([benchmark script](examples/benchmark_split_and_cache.py))

## Bring your own MNE epochs

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from xdflow import LeaveGroupOutValidator, Pipeline, from_mne_epochs
from xdflow.transforms.csp import CSPTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.transforms.spectral import BandpassFilterTransform

data = from_mne_epochs(epochs)  # event names -> "label"; metadata columns (e.g. "subject") -> coordinates

pipeline = Pipeline(
    "csp_lda",
    [
        ("bandpass", BandpassFilterTransform(8, 30, fs=epochs.info["sfreq"])),  # learns nothing: runs once
        ("csp", CSPTransform(n_components=6)),  # learns: refit inside every fold
        ("lda", SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label")),
    ],
)

cv = LeaveGroupOutValidator(group_coord="subject", scoring="accuracy")  # leave one subject out
cv.set_pipeline(pipeline)
cv.cross_validate(data)

oof = cv.oof_predictions_container_.data  # predictions keep trial, subject and every other coordinate
(oof == oof["label"]).groupby("subject").mean()  # accuracy per subject, no index arrays
model = cv.finalize_pipeline(data)  # refit on all data for inference: model.predict(new_data)
```

`pip install "xdflow[mne]"` for `CSPTransform`; `from_mne_epochs` itself needs nothing beyond your `Epochs`.

## Using an existing scikit-learn pipeline

Pass it to `evaluate` as is, or put xdflow steps in front of it. Steps that learn nothing from the data run
once on all trials, and your pipeline is refit inside every training fold:

```python
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline

import xdflow
from xdflow.transforms import BandpassFilterTransform, CropTimeTransform

existing = make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())  # unchanged
result = xdflow.evaluate(
    [BandpassFilterTransform(8, 30, fs=160), CropTimeTransform(0.5, 4.0), existing],  # filter and crop: once
    epochs,  # labels from the event names; metadata columns such as "subject" become coordinates
    groups="subject",
)
```

Multi-dimensional input such as `trial × channel × time` is passed straight through, and its parameters stay
tunable by their usual names (`csp__n_components`). In the other direction, `xdflow.to_sklearn(pipeline, ...)`
makes any xdflow pipeline a scikit-learn estimator for `cross_val_score`, `GridSearchCV` or MOABB.

## Try it

| Example | What it shows | |
|---|---|---|
| [Quickstart](docs/tutorials/quickstart.md) | The core API on synthetic data, in seconds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/quickstart.ipynb) |
| [EEG motor imagery](docs/tutorials/eeg-motor-imagery.md) | Within- and cross-subject decoding; the CSP leakage shortcut | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/eeg_motor_imagery.ipynb) |
| [Human activity recognition](docs/tutorials/har-activity.md) | Wearable sensors, 30 subjects; subject leakage in random K-fold | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/har_activity.ipynb) |
| [Evaluate an existing model](examples/evaluate_existing_model.py) | `evaluate`, `check_split` and `to_sklearn` on EEG with an MNE + scikit-learn model | |
| [MOABB benchmark](examples/moabb_evaluation.py) | An xdflow pipeline evaluated by MOABB next to the same model in scikit-learn | |

## Installation

```bash
pip install xdflow            # core
pip install "xdflow[tuning]"  # + Optuna hyperparameter search
pip install "xdflow[mne]"     # + MNE-Python interop (CSPTransform)
pip install "xdflow[all]"     # + LightGBM, MLflow, MNE, plotting, multitaper spectra, domain adaptation
```

Python 3.11+.

## Quick example

Runs with the core install; the full version is [`examples/quickstart.py`](examples/quickstart.py).

<details>
<summary>Show the code</summary>

```python
import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow import DataContainer, KFoldValidator, Pipeline
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

rng = np.random.default_rng(0)
stimuli = np.repeat(["rest", "tone", "odor"], 60)
rng.shuffle(stimuli)
values = rng.normal(0.0, 1.2, size=(stimuli.size, 4, 25))
values[stimuli == "tone", 1, 8:15] += 1.5
values[stimuli == "odor", 2, 14:22] += 1.5

data = DataContainer(
    xr.DataArray(
        values,
        dims=("trial", "channel", "time"),
        coords={
            "trial": np.arange(stimuli.size),
            "channel": [f"ch{i}" for i in range(4)],
            "time": np.linspace(-0.2, 0.8, 25),
            "stimulus": ("trial", stimuli),
        },
    )
)

pipeline = Pipeline(
    name="core_quickstart",
    steps=[
        ("zscore", ZScoreTransform(per_dim="trial")),  # each trial over its own channels and time
        ("flatten", FlattenTransform(dims=("channel", "time"))),
        ("classifier", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus", max_iter=500)),
    ],
)

cv = KFoldValidator(n_splits=5, shuffle=True, random_state=0, stratify_coord="stimulus", scoring="f1_weighted")
cv.set_pipeline(pipeline)
print(f"Weighted F1: {cv.cross_validate(data):.3f}")
```

</details>

## What you get

- **Metadata that survives.** Out-of-fold and holdout predictions keep their trial coordinates, so results
  can be broken down by subject, session, or condition with a `groupby`.
- **Composition.** `Pipeline`, `TransformUnion`, `SwitchTransform`/`OptionalTransform` for architecture
  search, `GroupApplyTransform` for per-subject or per-session models, `EnsemblePredictor`.
- **Tuning.** Optuna search over parameters *and* pipeline structure, with seeds threaded through every
  step, pruning, and optional MLflow tracking.
- **Your estimators.** scikit-learn, LightGBM, XGBoost, or anything with the sklearn API, via
  `SKLearnPredictor`/`SKLearnTransformer`, or wrap an existing estimator or `Pipeline` as-is with
  `SKLearnPredictor.from_estimator`. Multi-target regression, multilabel classification, and sample
  weights are supported.
- **Custom steps are functions.** `@xdflow.step` turns a function of an `xarray.DataArray` into a step that
  caches, clones and tunes like a built-in one; steps that learn from training data take a `fit` function.
  For full control, a transform is a small class with `_fit`/`_transform` and declared dimensions, easy to
  write, test, review, and generate with LLM coding tools ([guide](docs/guides/writing-transforms.md)).

## Works with the tools you already use

- **scikit-learn.** `xdflow` builds on it: scikit-learn estimators, metric names and splitters plug in, an
  existing `Pipeline` runs as is in `evaluate`, and `to_sklearn` turns an xdflow pipeline into a scikit-learn
  estimator. On top of that, metadata lives in
  the data, splits are named, and stateless features are computed once for all folds and tuning trials,
  where `Pipeline(memory=...)` recomputes them for every fold.
- **MNE-Python.** `from_mne_epochs` brings `Epochs` in with their labels and metadata, and `CSPTransform`
  turns `mne.decoding.CSP` into a fold-safe pipeline step.
- **sklearn-xarray** brings xarray to scikit-learn estimators. `xdflow` adds a learned-or-not declaration
  on every step, which drives the split point and the caching above, plus holdouts, per-group models and
  Optuna tuning over pipeline structure.

## Status

Actively developed and used in research code. Changes are tracked in the [changelog](CHANGELOG.md). Feedback
and use cases are very welcome in [issues](https://github.com/general-sense/xdflow/issues).

## Documentation

**[xdflow.readthedocs.io](https://xdflow.readthedocs.io/en/latest/)**: [concepts](docs/concepts/index.md),
[tutorials](docs/tutorials/index.md), [writing custom transforms](docs/guides/writing-transforms.md) and
[cross-validators](docs/guides/writing-cross-validators.md), and the [API reference](docs/api/index.md).

## Contributing

Contributions are welcome: new transforms, validators, examples on your own kind of data, bug reports, and
documentation fixes. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/general-sense/xdflow.git
cd xdflow
uv sync --all-extras   # creates .venv with everything
uv run pytest
uv run ruff check .
```

## Citing

If you use `xdflow` in published work, please cite it using the metadata in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button).

## License

MIT; see [LICENSE](LICENSE).

## Contact

- **Lead maintainer**: [Julien Bloch](https://github.com/julien-bloch)
- **Issues**: [bugs and feature requests](https://github.com/general-sense/xdflow/issues)

