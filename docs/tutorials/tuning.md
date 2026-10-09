# Hyperparameter Tuning

This tutorial continues the core XDFlow workflow: data lives in a `DataContainer`, model structure lives in a `Pipeline`, evaluation lives in a `CrossValidator`, and search lives in a `Tuner`.

Tuning is a first-class workflow in XDFlow. It is installed as an extra only because it depends on Optuna:

```bash
pip install "xdflow[tuning]"
```

MLflow tracking is off by default; pass `use_mlflow=True` (and install `xdflow[mlflow]`) to log trials.

## 1. Build A Tunable Pipeline

The pipeline can be the same kind of pipeline used for ordinary cross-validation. The important part is that each step has a stable name, because the tuning grid refers to those names.

```python
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

pipeline = Pipeline(
    name="logistic_pipeline",
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

The classifier reads targets from the `stimulus` coordinate. There is no separate `y` array to keep aligned during tuning.

## 2. Choose The Evaluation Policy

The validator defines how each trial is scored. This is the same object you would use without tuning.
Use a built-in validator when possible; custom split policies are covered in
[Writing Custom Cross-Validators](../guides/writing-cross-validators.md).

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
```

## 3. Define The Search Space

The tuning grid mirrors the pipeline:

```python
param_grid = {
    "logistic_pipeline": {
        "classifier": {
            "C": ("log", 1e-3, 10.0),
            "class_weight": [None, "balanced"],
        },
    },
}
```

Search space formats:

- list: categorical choice
- two-int tuple: integer range
- two-number tuple with a float: float range
- `("log", low, high)`: log-scaled float range

## 4. Run The Tuner

```python
from xdflow.tuning import Tuner

tuner = Tuner(
    pipelines_to_tune=pipeline,
    cv_strategy=cv,
    param_grid=param_grid,
    initial_data_container=container,
    random_seed=0,
    use_cache=True,
    use_mlflow=False,
    verbose=1,
)

best_params, best_score = tuner.tune(n_trials=20)
print(best_params)
print(best_score)
```

Each trial clones the pipeline, samples parameters, evaluates through the cross-validator, and reports the score back to Optuna. With `use_cache=True`, XDFlow can reuse fold-invariant stateless preprocessing before the first tunable or stateful step.

## 5. Use The Best Pipeline

```python
best_pipeline = tuner.get_best_pipeline()
final_pipeline = tuner.finalize_best_pipeline()

predictions = final_pipeline.predict(container)
```

`get_best_pipeline()` returns the selected configuration. `finalize_best_pipeline()` fits that configuration on the final data through the validator's finalization path, so the same split/refit assumptions are used consistently.

## Architecture Search With Optional Steps

Tuning can search over pipeline structure, not just scalar parameters. `OptionalTransform` and `SwitchTransform` make architecture choices part of the pipeline.

```python
from xdflow.composite import OptionalTransform, Pipeline
from xdflow.transforms.normalization import ZScoreTransform

pipeline = Pipeline(
    name="optional_preprocessing",
    steps=[
        (
            "maybe_zscore",
            OptionalTransform(
                ZScoreTransform(per_dim="trial"),
                use=True,
            ),
        ),
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

param_grid = {
    "optional_preprocessing": {
        "maybe_zscore": {
            "zscoretransform": {
                "per_dim": ["trial", "channel"],
            },
        },
        "classifier": {
            "C": ("log", 1e-3, 10.0),
        },
    },
}
```

The tuner samples the `maybe_zscore` branch choice during trials. The `zscoretransform` key is the name of the non-identity choice created by `OptionalTransform`; its nested values are only applied when that branch is selected.

If a validator defines a holdout split, score the selected configuration before final fitting:

```python
holdout_score = tuner.score_best_pipeline_on_holdout()
print(holdout_score)
```

## Multiple Pipelines

Pass a list of named pipelines to compare complete architectures:

```python
tuner = Tuner(
    pipelines_to_tune=[pipeline_a, pipeline_b],
    cv_strategy=cv,
    param_grid={
        "pipeline_a": {"classifier": {"C": ("log", 1e-3, 10.0)}},
        "pipeline_b": {"classifier": {"alpha": ("log", 1e-4, 1.0)}},
    },
    initial_data_container=container,
    use_mlflow=False,
)
```

The Optuna trial records which pipeline was selected through the `"pipeline"` parameter, then applies only that pipeline's search space.

## Next Steps

- Read [Tuning](../concepts/tuning.md) for the conceptual model.
- Read [Composing Pipelines](../concepts/composition.md) for switches, optional steps, unions, and grouped transforms.
- See [Tuning API](../api/tuning.md) for `Tuner` and helper function details.
