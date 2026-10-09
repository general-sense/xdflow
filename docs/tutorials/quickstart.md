# 5-Minute Core Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/quickstart.ipynb) Run this tutorial as a notebook in your browser.

This quickstart uses only the base `xdflow` install. It creates a small labeled `xarray.DataArray`, wraps it in a `DataContainer`, builds a pipeline, and lets `KFoldValidator` run the evaluation loop.

The example shows the mechanics that matter in larger experiments: labels live as coordinates, the classifier reads its target from a coordinate, the validator stratifies by that coordinate, stateful steps refit per fold, and predictions stay aligned with the trial axis.

From a repository checkout, the same example is available as a runnable script:

```bash
python examples/quickstart.py
```

## 1. Create Structured Data

`xdflow` starts from named dimensions and sample-level coordinates. Here each trial has `channel` and `time` dimensions, plus `stimulus` and `session` metadata.

```python
import numpy as np
import xarray as xr

from xdflow.core import DataContainer

rng = np.random.default_rng(0)

stimuli = np.repeat(["rest", "tone", "odor"], 60)
rng.shuffle(stimuli)

values = rng.normal(0.0, 0.8, size=(stimuli.size, 4, 25))
values[stimuli == "tone", 1, 8:15] += 2.0
values[stimuli == "odor", 2, 14:22] += 2.0

trial_ids = np.arange(stimuli.size)

data = xr.DataArray(
    values,
    dims=("trial", "channel", "time"),
    coords={
        "trial": trial_ids,
        "channel": [f"ch{i}" for i in range(4)],
        "time": np.linspace(-0.2, 0.8, 25),
        "stimulus": ("trial", stimuli),
        "session": ("trial", np.where(trial_ids < stimuli.size // 2, "session_a", "session_b")),
    },
)
container = DataContainer(data)
```

## 2. Build A Pipeline

The pipeline keeps labels and coordinates attached while each step transforms the data. The classifier reads its targets from the `stimulus` coordinate.

```python
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

pipeline = Pipeline(
    name="core_quickstart",
    steps=[
        ("zscore", ZScoreTransform(per_dim="trial")),
        ("flatten", FlattenTransform(dims=("channel", "time"))),
        (
            "classifier",
            SKLearnPredictor(
                LogisticRegression,
                sample_dim="trial",
                target_coord="stimulus",
                max_iter=500,
            ),
        ),
    ],
)
```

## 3. Cross-Validate

`KFoldValidator` owns the split loop, scoring, prediction collection, and stateful refits. The `stratify_coord` argument keeps class proportions balanced across folds using the named `stimulus` coordinate.

In this pipeline, z-scoring is per trial and flattening is structural, so that fold-invariant preprocessing can run before the stateful classifier is cloned and refit on each training fold.

```python
from xdflow.cv import KFoldValidator

cv = KFoldValidator(
    n_splits=5,
    shuffle=True,
    random_state=0,
    stratify_coord="stimulus",
    scoring="f1_weighted",
    verbose=False,
)
cv.set_pipeline(pipeline)

score = cv.cross_validate(container, verbose=False)
print(f"Weighted F1: {score:.3f}")
```

## 4. Fit And Predict

After choosing a pipeline, fit it on the data you want to use for the final model and call `predict`.

```python
pipeline.fit(container)
predictions = pipeline.predict(container)

print(predictions.data.dims)
```

The prediction container still carries the sample dimension, so predictions can be aligned back to trial-level metadata.

## Next Steps

- Read [Data Contract](../concepts/data_contract.md) before adapting your own arrays.
- Use [Writing Custom Transforms](../guides/writing-transforms.md) when your preprocessing should become a reusable pipeline step.
- Use [Hyperparameter Tuning](tuning.md) to search over the same kind of pipeline with Optuna.
- Use [Spectral Pipeline Walkthrough](basic-pipeline.md) for a richer signal-processing example.
- Use [Reusable ML Patterns](reusable-ml-patterns.md) for multilabel, sample-weighting, and domain-transfer workflows.
