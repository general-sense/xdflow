# Evaluate an Existing Model

`xdflow.evaluate` cross-validates a model in one call and reports the score per fold, per group and per
trial. It takes the data and model you already have: NumPy arrays, an `xarray.DataArray` or MNE `Epochs`, and a
scikit-learn estimator or `Pipeline`. This page also covers `check_split`, which audits a split without running
a model, and `to_sklearn`, which goes the other way and turns an xdflow pipeline into a scikit-learn estimator.

The runnable version, on public EEG data with an MNE + scikit-learn model, is
[`examples/evaluate_existing_model.py`](https://github.com/general-sense/xdflow/blob/main/examples/evaluate_existing_model.py).

## One call

```python
import xdflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
result = xdflow.evaluate(model, X, y, groups=subject_ids)
print(result)
```

```text
accuracy 0.924 ± 0.029 (mean ± sd over 5 folds, 10299 trials)
Split: 5-fold, grouped (30 groups), stratified by class
Computed once: nothing. Refit in each fold: pipeline (standardscaler, logisticregression).
...
```

`X` holds one sample per row (or per first-axis entry, for `trial × channel × time` arrays). `y` and `groups`
are either per-sample arrays or names of trial coordinates:

| Data | Call |
|---|---|
| NumPy arrays | `evaluate(model, X, y, groups=subject_ids)` |
| NumPy arrays plus a metadata table | `evaluate(model, X, y, groups="subject", metadata=info)` |
| `xarray.DataArray` with a `trial` dim | `evaluate(model, data, "label", groups="subject")` |
| MNE `Epochs` | `evaluate(model, epochs, groups="subject")`: labels come from the event names |

### Choosing the split

- **`groups`** keeps every trial of a group (subject, session, recording) on one side of each split, so the
  score is for new groups. With no more groups than folds, each group is held out in turn; otherwise groups are
  split into `cv` folds, stratified by class.
- **Without `groups`**, folds are stratified by class and trials are shuffled. If the data has a subject,
  session or similar coordinate, `evaluate` warns that the score is for new trials from known subjects, and
  names the `groups=` that would score new subjects instead.
- **`cv`** also accepts any scikit-learn splitter (`GroupKFold(5)`, `TimeSeriesSplit(4)`...), explicit
  `(train, test)` index pairs, or an xdflow validator for holdouts and custom policies.

With the same split, `evaluate` gives the same fold scores as scikit-learn's `cross_val_score`.

### Metrics

`scoring` takes scikit-learn's metric names with their scikit-learn meaning (`"accuracy"`,
`"balanced_accuracy"`, `"f1_macro"`, `"r2"`, `"neg_mean_absolute_error"`...) or a function
`(y_true, y_pred) -> float`. The default is accuracy for classifiers and R² for regressors.

## Reading the result

```python
result.score                 # mean over folds
result.fold_scores           # one per fold
result.by("subject")         # score and trial count per subject (or session, condition...)
result.predictions           # one row per trial: its coordinates, y_true, y_pred and fold
result.split                 # what the folds share between training and test sets
model = result.fit_final()   # refit on all trials; model.predict(new_data) needs no labels
```

## xdflow steps in front of your model

A list puts xdflow steps before a final scikit-learn model. Steps that learn nothing from the data, such as
filters and crops, run once on all trials; the model is refit inside each training fold:

```python
from xdflow.transforms import BandpassFilterTransform, CropTimeTransform

result = xdflow.evaluate(
    [BandpassFilterTransform(8, 30, fs=160), CropTimeTransform(0.5, 4.0), existing_model],
    epochs,
    groups="subject",
)
```

Your own preprocessing can be a plain function with `@xdflow.step`; see
[Writing Custom Transforms](../guides/writing-transforms.md#functions-as-steps).

## Checking a split you already use

`check_split` takes what `cross_val_score` takes and reports, per fold, which subjects, sessions, runs and
source trials appear in both training and test sets, and which test samples have an exact copy in the
training set:

```python
from sklearn.model_selection import StratifiedKFold

report = xdflow.check_split(X, StratifiedKFold(5, shuffle=True), y=y, metadata=info)
print(report)
```

```text
Split check over 5 folds: shared data found.
  - The same 'subject' values appear in training and test sets in 5 of 5 folds: the score measures performance on new trials from a subject seen in training, not on a new subject.

            check  values folds shared  examples
          subject      30          5/5 1, 10, 11
identical samples   10299          0/5
```

Coordinates are recognised by name (`subject`, `subject_id`, `participant`, `session`, `run`, `block`,
`segment`, `orig_trial`...); pass `groups=` to audit any other. Session-like coordinates are compared within
subject, since session numbers usually restart for every subject. A validator that has run
`cross_validate` can be audited too: `check_split(data, validator)`.

## Using xdflow inside scikit-learn

`to_sklearn` wraps an xdflow pipeline as a scikit-learn estimator that takes sample-first NumPy arrays. It
works with `cross_val_score` (including `n_jobs`), `GridSearchCV` (parameters are
`"pipeline__<step>__<param>"`), probability scorers, and frameworks built on scikit-learn's estimator API
such as MOABB:

```python
from sklearn.model_selection import GroupKFold, cross_val_score

estimator = xdflow.to_sklearn(
    result.pipeline,
    dims=("trial", "channel", "time"),
    coords={"channel": channel_names, "time": times},  # for steps that select by label
)
cross_val_score(estimator, X, y, groups=subject_ids, cv=GroupKFold(5))
```
