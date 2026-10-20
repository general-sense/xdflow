# Tuning

Tuning is a core XDFlow workflow. The optional `tuning` install extra exists to keep the base dependency set light; it does not mean hyperparameter search is an add-on conceptually.

XDFlow tuning uses the same objects as the rest of the framework:

- a `Pipeline` defines the model architecture
- a `CrossValidator` defines the evaluation policy
- a `DataContainer` keeps targets, groups, and metadata attached to the data
- a `Tuner` searches over pipeline choices and nested transform parameters

The point is to avoid rewriting experiment loops. The tuner asks Optuna for a candidate configuration, applies it to a cloned pipeline, evaluates it through the validator, and records the score. Cross-validation still owns splitting, refitting, scoring, and prediction alignment.

For custom evaluation policies, prefer configuring the built-in validators first. If the split policy itself is new, see [Writing Custom Cross-Validators](../guides/writing-cross-validators.md).

## Install model

Use the tuning extra when you want to run optimization:

```bash
pip install "xdflow[tuning]"
```

The extra installs Optuna and related tuning dependencies. The workflow remains first-class in XDFlow; only the optimizer dependency is optional.

MLflow tracking is separate:

```bash
pip install "xdflow[mlflow]"
```

Tracking is off by default; pass `use_mlflow=True` to log trials to MLflow.

## Search space structure

The search space mirrors the pipeline structure:

```python
param_grid = {
    "pipeline_name": {
        "step_name": {
            "parameter_name": search_space,
        },
    },
}
```

Supported search spaces are intentionally simple:

- `["a", "b", "c"]` for categorical choices
- `(1, 10)` for integer ranges
- `(0.0, 1.0)` for float ranges
- `("log", 1e-4, 10.0)` for log-scaled float ranges

The tuner converts those definitions into Optuna suggestions and applies them with the same nested parameter mechanics used by composite transforms.

## Pipeline and architecture search

Tuning is not limited to scalar estimator parameters. Because switches and optional steps are transforms, the tuner can search over architecture choices too.

`SwitchTransform` exposes a pipeline position with several possible implementations. During tuning, XDFlow treats the switch as dynamic and samples the active choice per trial.

```python
from xdflow.composite import OptionalTransform
from xdflow.transforms.normalization import ZScoreTransform

maybe_zscore = OptionalTransform(
    ZScoreTransform(per_dim="channel", use_fit=True),
    use=True,
)
```

A tuning grid can then tune the branch choice and any parameters on the selected branch. This is how feature extraction choices, preprocessing toggles, and model families can stay inside one pipeline contract.

## Caching during tuning

Tuning repeatedly evaluates related pipelines. XDFlow uses transform state metadata to avoid recomputing fold-invariant work when possible.

Before each trial, the tuner can split a pipeline into:

- a static prefix that has no tunable or stateful steps
- a dynamic suffix that contains tunable choices, stateful transforms, or predictors

The static prefix is fit/transformed once and reused as the trial input. The dynamic suffix is cloned and evaluated by the cross-validator. This is especially important for expensive deterministic preprocessing such as filtering or feature extraction.

If a step learns from samples or depends on the split boundary, it should be marked stateful so it is not cached across folds or trials incorrectly.

The prefix is reused through the disk cache. Results that depend on your own functions are recomputed rather than reused from the disk cache, since xdflow cannot see every code they depend on; set `XDFLOW_CACHE_USER_CODE=1` to cache them when they depend only on their code, parameters and the modules they read. Without it, a prefix that contains your own functions is recomputed for each trial.

A cache key covers the data (values, coordinates, attrs), every step's parameters, and the code of xdflow and of the installed packages (reinstalling or upgrading one invalidates the cache). It does not cover files or environment variables a step reads, packages edited in place without reinstalling, or the global NumPy random state: an estimator without `random_state` reuses its first cached fit, so for a seed study set `random_state` or pass `use_stateful_fit_cache=False` to the validator.

## Final model and holdout score

After tuning, the common workflow is:

1. `tuner.tune(...)`
2. `tuner.get_best_pipeline()` to inspect the selected configuration
3. `tuner.score_best_pipeline_on_holdout()` if the validator has a holdout split
4. `tuner.finalize_best_pipeline()` to fit the selected pipeline on the final training data

This keeps optimization, validation, holdout scoring, and final fitting aligned with the same data contract.

## Where tuning fits

Use cross-validation directly when you already know the pipeline configuration and want an evaluation score.

Use tuning when you want XDFlow to search over parameters, optional steps, switch branches, or multiple named pipelines while preserving the same split and refit rules.

In practice, tuning is the layer that turns XDFlow's composable pipelines into repeatable experiment search.
