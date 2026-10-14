# XDFlow

`xdflow` is a machine learning framework that leverages metadata in labeled structured data. Built on `xarray` objects, it uses dimensions and coordinates such as `trial`, `channel`, `time`, `stimulus`, `session`, and `subject` as pipeline context for transforms, prediction, cross-validation, tuning, and scoring.

The metadata is part of the workflow, not just something carried along for inspection. Named dimensions tell transforms what to operate on. Target, group, session, and subject coordinates drive validation. Transform state tells validators and tuners what can be reused across folds and what must be refit. Predictions and scores stay tied to the coordinates that produced them.

## Why use it

Use `xdflow` when your pipeline should understand the structure of the data:

- **Coordinate-aware transforms**: operate on `trial`, `channel`, or `time` by name instead of by axis index.
- **Faster CV and tuning**: fold-invariant stateless transforms can run once and be reused across folds, tuning trials, and pipeline comparisons.
- **Better leakage safety**: stateful or split-dependent steps are cloned and refit inside each training fold.
- **Named-coordinate validation**: validators can split, group, or stratify by coordinates such as `stimulus`, `session`, `subject`, or `animal`.
- **Aligned outputs**: targets, groups, sessions, subjects, channels, timestamps, and predictions stay attached to the data.
- **Modular pipelines**: transforms, predictors, unions, switches, and per-group steps compose without handwritten split loops.
- **Automatic split, cache, and refit planning**: validators and tuners use dimension contracts, coordinates, split settings, and transform state to decide what can be reused and what must be refit.

These patterns show up in neural recordings, biosignals, sensor arrays, medical time series, geophysical data, and other labeled datasets where the metadata is part of the experiment.

## Where it fits

`xdflow` builds on the scientific Python tools that already work well:

- `xarray` is the labeled data representation
- scikit-learn-style estimators remain the model interface when they fit the task
- Optuna handles hyperparameter and architecture search
- MLflow can record experiment runs

XDFlow is the layer around those tools that keeps dimensions, coordinates, transform state, validation, tuning, caching, and predictions in one pipeline contract. It is closest in spirit to `sklearn.Pipeline`, but it is designed for data whose axes and coordinates carry experimental meaning.

The library is built around a small set of abstractions:

- `DataContainer` carries labeled xarray data through pipeline APIs
- `Transform` defines immutable preprocessing steps with explicit dimension and state contracts
- `Pipeline` composes transforms and predictors into reusable workflows
- `CrossValidator` runs structure-aware evaluation while separating fold-invariant stateless work from stateful or split-dependent steps
- `Tuner` searches over pipeline parameters and architecture choices through the same validator and data contract

## Current scope

XDFlow currently operates on in-memory `xarray.DataArray` inputs. Out-of-core and Dask-backed execution are future-facing areas rather than part of the current core runtime.

## Package layout

- `xdflow.core`: base container and transform interfaces
- `xdflow.composite`: sequential, grouped, conditional, and ensemble composition
- `xdflow.cv`: cross-validation orchestration and splitting strategies
- `xdflow.tuning`: first-class hyperparameter and architecture search built on Optuna
- `xdflow.transforms`: preprocessing, sklearn adapters, spectral transforms, and predictors
- `xdflow.utils`: sampling, caching, target resolution, and plotting helpers
- `examples/`: runnable scripts for public workflows

## Start here

1. Read [Installation](installation.md) to set up the package and optional extras.
2. Try [Evaluate an Existing Model](tutorials/evaluate.md): one call with your data and scikit-learn model.
3. Read [Concepts](concepts/index.md) for the runtime model and transform rules.
4. Run the [5-Minute Core Quickstart](tutorials/quickstart.md) for a base-install example.
5. Read [Execution Model](concepts/execution.md) for the split, refit, and caching rules.
6. Use [Hyperparameter Tuning](tutorials/tuning.md) to search over pipeline parameters and architecture choices.
7. Follow the [Spectral Pipeline Walkthrough](tutorials/basic-pipeline.md) for an end-to-end signal-processing example.
8. Read [Reusable ML Patterns](tutorials/reusable-ml-patterns.md) for multilabel, weighting, and domain-transfer examples.
9. Use [Writing Custom Transforms](guides/writing-transforms.md) when adding reusable preprocessing or feature extraction.
10. Use [Writing Custom Cross-Validators](guides/writing-cross-validators.md) only when built-in split policies do not fit.
11. Use [XDFlow With LLMs](guides/llm.md) when asking an LLM to implement against the framework.
12. Use the [API Reference](api/index.md) for class and function details.
