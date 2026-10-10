# Core API

Core APIs define the data container and transform contracts used throughout XDFlow.

For extension-oriented examples, see [Writing Custom Transforms](../guides/writing-transforms.md) and [Writing Custom Cross-Validators](../guides/writing-cross-validators.md).

## Top-Level Exports

The root package exposes the main workflow primitives:

```python
from xdflow import DataContainer, Transform, Predictor, Pipeline, CrossValidator
from xdflow import evaluate, check_split, make_pipeline, step, from_numpy, from_mne_epochs, to_sklearn
```

`Tuner` is conditionally exported when the tuning extra is installed.

## Evaluation and Split Checks

::: xdflow.evaluate.evaluate

::: xdflow.evaluate.EvaluationResult

::: xdflow.diagnostics.check_split

::: xdflow.diagnostics.SplitReport

## Converters and scikit-learn Interop

::: xdflow.interop.from_numpy

::: xdflow.interop.from_mne_epochs

::: xdflow.sklearn_api.to_sklearn

## Data Containers

::: xdflow.core.data_container.DataContainer

::: xdflow.core.data_container.TransformError

## Transform Contracts

::: xdflow.core.base.Transform

::: xdflow.core.base.Predictor

::: xdflow.core.base.SampleWeightMixin
