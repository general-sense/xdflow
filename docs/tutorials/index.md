# Tutorials

The tutorials show how XDFlow uses metadata as pipeline context: coordinate targets, named-dimension transforms, leakage-safe validation, fold-invariant reuse, modular pipelines, and specialized split policies.

- [Evaluate an Existing Model](evaluate.md): one call from your data and scikit-learn model to per-fold, per-subject and per-trial results; auditing a split you already use; xdflow pipelines inside scikit-learn and MOABB
- [5-Minute Core Quickstart](quickstart.md): runnable base-install pipeline with coordinate targets, stratified CV, stateful refits, and prediction alignment
- [Hyperparameter Tuning](tuning.md): Optuna-backed search over pipeline parameters, optional steps, switch choices, and multiple pipelines
- [Spectral Pipeline Walkthrough](basic-pipeline.md): signal-processing pipeline where expensive fold-invariant feature extraction can be reused across folds
- [Reusable ML Patterns](reusable-ml-patterns.md): multilabel prediction, class/domain weighting, and few-shot domain transfer without side-channel label or split bookkeeping
- [EEG Motor Imagery: xdflow vs. sklearn](eeg-motor-imagery.md): public EEG data, within- vs. cross-subject validation, per-subject models, and the same analysis in plain scikit-learn with a quantified leakage pitfall
- [Activity Recognition: Splitting by Subject](har-activity.md): smartphone sensor data outside neuroscience, subject-grouped vs. shuffled-window validation, per-subject accuracy from out-of-fold predictions, and the same analysis in plain scikit-learn (also as a Colab notebook)

These are also Colab notebooks you can run in the browser: [evaluate](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/evaluate.ipynb), [quickstart](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/quickstart.ipynb), [EEG motor imagery](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/eeg_motor_imagery.ipynb) and [activity recognition](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/har_activity.ipynb) (sources in `examples/notebooks/`).

The original notebook and dataset helper are still kept under `docs/tutorials/` for local exploration, but the Markdown walkthroughs are the canonical versions for the published docs site.
