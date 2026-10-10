# Reusable ML Patterns

This page covers reusable patterns for keeping labels, weights, and domain information in the data. Predictors and validators read those coordinates directly, so multilabel prediction, sample weighting, and few-shot domain transfer use the same pipeline machinery as simpler workflows.

## Multilabel Classification

Use `SKLearnPredictor` with `is_multilabel=True` when each target is a binary coordinate. The targets remain named coordinates instead of a separate `y` matrix.

```python
from sklearn.linear_model import LogisticRegression

from xdflow.transforms.sklearn_transform import SKLearnPredictor

predictor = SKLearnPredictor(
    LogisticRegression,
    sample_dim="trial",
    target_coord=["target__ethanol", "target__hexanal", "target__limonene"],
    is_classifier=True,
    is_multilabel=True,
    max_iter=500,
)

predictor.fit(train_container)
predictions = predictor.predict(test_container)  # dims: ("trial", "target")
probabilities = predictor.predict_proba(test_container)  # dims: ("trial", "target")
```

For wildcard target coordinates, pass a pattern such as `target_coord="target__*"`. XDFlow resolves the target coordinates at fit time and wraps compatible sklearn classifiers with `MultiOutputClassifier`.

## Class and Domain Weighting

Use `BalanceClassWeightTransform` to attach a `sample_weight` coordinate that downstream sklearn-compatible estimators can consume. Weighting stays in the container, so it can move with the same trial axis as the labels and metadata.

```python
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.transforms.basic_transforms import BalanceClassWeightTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

pipeline = Pipeline(
    name="balanced_classifier",
    steps=[
        ("weights", BalanceClassWeightTransform(class_coord="stimulus")),
        ("clf", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus")),
    ],
)
```

For transfer settings, balance within domains while assigning domain-level totals:

```python
weights = BalanceClassWeightTransform(
    class_coord="stimulus",
    balance_domains=True,
    domain_coord="animal",
    domain_weights={"source": 0.25, "target": 0.75},
    normalize_domain_totals=True,
    weight_normalize="sum",
)
```

The default sklearn wrappers look for a `sample_weight` coordinate. Pass `sample_weight_coord=None` to disable this behavior.

## Few-Shot Domain Transfer

Use `SampledDomainKFoldValidator` when validation should occur only on target-domain folds, while training uses all source-domain samples plus a controlled number of target-domain samples. The domain policy is part of the validator configuration instead of a custom split script.

```python
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.cv import SampledDomainKFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor

pipeline = Pipeline(
    name="few_shot_transfer",
    steps=[
        ("clf", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus")),
    ],
)

cv = SampledDomainKFoldValidator(
    domain_coord="animal",
    source_domains=["animal_a", "animal_b"],
    target_domains="animal_c",
    label_coord="stimulus",
    default_samples_per_label=2,
    label_sample_counts={"control": None},
    n_splits=5,
    random_state=0,
    verbose=False,
)
cv.set_pipeline(pipeline)

score = cv.cross_validate(data_container, verbose=False)
```

Sampling semantics:

- `default_samples_per_label=0`: zero-shot target training for labels not otherwise specified
- `default_samples_per_label=2`: two target-domain samples per label per fold
- `label_sample_counts={"control": None}`: use all available target-domain training samples for `control`
- source-domain trials are included in every fold unless `source_domains` excludes them

This validator is intentionally domain-agnostic. Domain labels can represent animals, subjects, sessions, instruments, sites, or any other sample-level grouping.
