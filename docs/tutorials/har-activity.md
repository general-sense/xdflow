# Activity Recognition from Phone Sensors: Splitting by Subject

This tutorial classifies what a person is doing (walking, climbing stairs, sitting, standing, lying down) from a smartphone's accelerometer and gyroscope. It runs the same model with xdflow and with plain NumPy and scikit-learn, on the same data. Nothing here is specific to neuroscience. The data is a labeled array of windows with a few pieces of metadata per window, which is the shape xdflow is built for.

The short version:

- **The split decides the answer.** Shuffling windows into 5 folds gives 0.972 accuracy. Keeping each subject's windows on one side of the split gives 0.914-0.916, and the 9 official test subjects, never seen in training, give 0.922. The shuffled estimate understates the error rate about threefold (2.8% vs. 8.4%).
- **It also picks the wrong model.** With shuffled folds, a random forest looks best (0.981). Evaluated on unseen subjects, logistic regression is best (0.922 vs. 0.893).
- **Like for like, xdflow and scikit-learn agree exactly.** With identical splits (shuffled 5-fold, leave-one-subject-out, the official holdout), the scores match to three decimals, so every difference below comes from the split, not the library.
- **What xdflow adds.** The split boundary is a coordinate name (`group_coord="subject"`), and predictions come back with `subject` and `activity` still attached, so per-subject accuracy and a confusion table are a `groupby` away.

The full script is `examples/har_activity.py`, and the same analysis is in a Colab-ready notebook, `examples/notebooks/har_activity.ipynb`. All numbers below come from running the script.

## Run it

```bash
uv run python examples/har_activity.py
```

- **Dependencies:** only the base xdflow install (xarray, NumPy, pandas, scikit-learn).
- **Download:** the first run downloads the 61 MB UCI zip and keeps the 61 MB dataset archive inside it in `~/xdflow_data`. Set `XDFLOW_DATA_DIR` or pass `--data-dir PATH` to use another directory. The signals are read straight from that archive, so nothing else is unpacked. The download took a few seconds here.
- **Runtime:** after the download, the script runs in under a minute on a laptop.
- **Fit cache:** with `use_stateful_fit_cache=True`, xdflow validators cache fitted fold pipelines on disk (`~/.cache/xdflow`, or `XDFLOW_CACHE_DIR`), and a second run reuses them. It is off by default, and in this script; the timings above are without it.

## The data

The [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset has 30 volunteers aged 19-48. Each wore a Samsung Galaxy S II on the waist and performed six scripted activities. The phone recorded 3-axis acceleration and 3-axis angular velocity at 50 Hz. The dataset authors filtered the signals, split acceleration into body motion and gravity, and cut everything into 2.56 s windows (128 samples) that **overlap by 50%**. They also split the volunteers 70/30 into 21 training and 9 test subjects.

The loader stacks the nine raw signal channels (body acceleration, gyroscope and total acceleration, each x/y/z) into one `DataArray` with `trial`, `channel` and `time` dimensions. Each window gets four trial-level coordinates:

- `activity`: the label.
- `subject`: who was wearing the phone (1-30).
- `split`: the official `train`/`test` assignment.
- `segment`: a run of consecutive windows from one subject doing one activity. Windows are stored in recording order, so segments follow from consecutive rows.

```python
for split in ("train", "test"):
    signals = np.stack(
        [np.loadtxt(z.open(f"{ROOT}/{split}/Inertial Signals/{ch}_{split}.txt")) for ch in CHANNELS], axis=1
    )
    activity = np.loadtxt(z.open(f"{ROOT}/{split}/y_{split}.txt"), dtype=int)
    subject = np.loadtxt(z.open(f"{ROOT}/{split}/subject_{split}.txt"), dtype=int)
    if signals.shape != (len(activity), len(CHANNELS), 128) or len(subject) != len(activity):
        raise ValueError(f"{split}: inconsistent shapes {signals.shape}, {activity.shape}, {subject.shape}")
    pieces.append(
        xr.DataArray(
            signals,
            dims=("trial", "channel", "time"),
            coords={
                "channel": CHANNELS,
                "time": np.arange(128) / FS,
                "activity": ("trial", [codes[str(code)].lower() for code in activity]),
                "subject": ("trial", subject),
                "split": ("trial", np.full(len(activity), split)),
            },
        )
    )
data = xr.concat(pieces, dim="trial", join="exact")  # fails loudly if channels or times differ
activity, subject = data["activity"].values, data["subject"].values
new_segment = np.r_[True, (activity[1:] != activity[:-1]) | (subject[1:] != subject[:-1])]
data = data.assign_coords(trial=np.arange(data.sizes["trial"]), segment=("trial", np.cumsum(new_segment)))
```

The result has 10,299 windows: 7,352 from 21 training subjects (280 segments) and 2,947 from 9 test subjects (120 segments). Classes are roughly balanced. `laying` is the most common, so always predicting it scores 0.191 on the training subjects and 0.182 on the test subjects.

## The model

The model is deliberately simple. For each channel and window, it computes six summary statistics (mean, standard deviation, min, max, 25th and 75th percentiles) and the log power in five frequency bands (0-1, 1-3, 3-6, 6-12 and 12-25 Hz). That gives 9 channels × 11 features = 99 features. A standardization step and a multinomial logistic regression follow.

The features are a plain NumPy function. Both versions call it, so any difference between them comes from the splits, not the features:

```python
def window_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Summary statistics and log band power over the last axis (time); returns (..., len(FEATURES))."""
    power = np.abs(np.fft.rfft(x - x.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
    band_power = [power[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) for lo, hi in BANDS]
    stats = [x.mean(-1), x.std(-1), x.min(-1), x.max(-1), *np.percentile(x, [25, 75], axis=-1)]
    return np.stack(stats + [np.log(p + 1e-10) for p in band_power], axis=-1)
```

In xdflow, the same function becomes a pipeline step with `@step`. `xr.apply_ufunc` swaps the `time` dimension for a `feature` dimension and keeps every trial and channel coordinate. The step also checks the sampling rate against the `time` coordinate:

```python
@step
def features(data: xr.DataArray, fs: float = 50) -> xr.DataArray:
    """Per-channel window features. Each window's features depend on that window only, so cross-validation
    computes them once for all folds."""
    if not np.allclose(np.diff(data["time"].values), 1 / fs):
        raise ValueError(f"The time coordinate is not sampled at fs={fs} Hz.")
    return xr.apply_ufunc(
        window_features, data, input_core_dims=[["time"]], output_core_dims=[["feature"]], kwargs={"fs": fs}
    ).assign_coords(feature=FEATURES)


def build_pipeline() -> Pipeline:
    return Pipeline(
        name="har",
        steps=[
            ("features", features(fs=50)),  # stateless: computed once, before the folds
            ("flatten", FlattenTransform(dims=("channel", "feature"))),
            ("scale", SKLearnTransformer(StandardScaler, sample_dim="trial")),  # stateful: refit per fold
            (
                "classifier",
                SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="activity", max_iter=2000),
            ),
        ],
    )
```

Validators split this pipeline at its first stateful step. The features and the flatten run once on all windows. The scaler and the classifier are cloned and refit on each training fold.

## The split is a coordinate name

All cross-validation below uses only the 21 official training subjects. The three 5-fold validators differ only in which coordinate they keep apart:

```python
common = {"scoring": "accuracy", "verbose": False}
train = DataContainer(data.isel(trial=data["split"].values == "train"))
validators = {
    "windows": KFoldValidator(n_splits=5, shuffle=True, random_state=0, **common),  # stratified by activity
    "segments": LeaveGroupOutValidator(group_coord="segment", n_splits=5, random_state=0, **common),
    "subjects": LeaveGroupOutValidator(group_coord="subject", n_splits=5, random_state=0, **common),
}
for name, cv in validators.items():
    cv.set_pipeline(build_pipeline())
    scores[name] = cv.cross_validate(train)
```

Then leave-one-subject-out on the training subjects, and a final score on the 9 test subjects, which play no part in fitting or model choice inside this script:

```python
test_subjects = np.unique(data["subject"].values[data["split"].values == "test"]).tolist()
loso = LeaveGroupOutValidator(group_coord="subject", test_group_ids=test_subjects, **common)
loso.set_pipeline(build_pipeline())
scores["loso"] = loso.cross_validate(DataContainer(data))
scores["holdout"] = loso.score_on_holdout(DataContainer(data))
```

!!! note "Which xdflow validators keep groups apart"
    `LeaveGroupOutValidator(group_coord=...)` puts all of a group's trials on the same side of every split, like scikit-learn's `LeaveOneGroupOut` (or `GroupKFold` when `n_splits` is set): the tool for questions about unseen subjects. For per-subject models, `GroupedKFoldValidator(group_coord=...)` puts every subject in every training fold.

Out-of-fold predictions keep their trial coordinates, plus a `fold` coordinate. Per-subject accuracy and a confusion table need no index bookkeeping. `holdout_container_` keeps the trial coordinates of the test-subject predictions in the same way (without `fold`):

```python
oof = loso.oof_predictions_container_.data  # dims: (trial,), coords: activity, subject, split, segment, fold
per_subject = (oof == oof["activity"]).groupby("subject").mean()
confusion = pd.crosstab(oof["activity"].values, oof.values, normalize="index")
```

With one subject per fold, the mean of `per_subject` is the leave-one-subject-out score. The script checks this.

## The plain scikit-learn version

The plain version takes `X`, `y` and the group arrays from the same `DataArray`, so both versions see exactly the same windows in the same order:

```python
X = window_features(data.transpose("trial", "channel", "time").values, 50).reshape(data.sizes["trial"], -1)
y, subjects, segments = data["activity"].values, data["subject"].values, data["segment"].values
train = data["split"].values == "train"


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


splits = {
    "windows": (StratifiedKFold(n_splits=5, shuffle=True, random_state=0), None),
    "segments": (GroupKFold(n_splits=5), segments[train]),
    "subjects": (GroupKFold(n_splits=5), subjects[train]),
    "loso": (LeaveOneGroupOut(), subjects[train]),
}
scores = {
    name: cross_val_score(model(), X[train], y[train], cv=cv, groups=g).mean() for name, (cv, g) in splits.items()
}
scores["holdout"] = model().fit(X[train], y[train]).score(X[~train], y[~train])
```

The scaler sits inside the sklearn `Pipeline`, so it is refit per fold. The catch is that row-level splitters split rows, and on this data a row is a window, not a person: nothing in `X` tells the splitter that consecutive rows overlap or come from the same person. The `groups` array has to be built and kept aligned with `X` by hand.

## Results

Output of `examples/har_activity.py` (seed 0):

```text
Loaded {'trial': 10299, 'channel': 9, 'time': 128} in 1.2s
Windows per activity (21 train / 9 test subjects):
split               test  train
activity
laying               537   1407
sitting              491   1286
standing             532   1374
walking              496   1226
walking_downstairs   420    986
walking_upstairs     471   1073
Majority-class accuracy: {'test': 0.182, 'train': 0.191}

Accuracy, logistic regression             xdflow  sklearn
5-fold, windows shuffled                   0.972    0.972
5-fold, segments kept apart                0.949    0.943
5-fold, subjects kept apart                0.914    0.916
Leave one subject out (21 folds)           0.912    0.912
Holdout: 9 never-seen test subjects        0.922    0.922

Accuracy by model (sklearn)     windows shuffled  subjects apart  never-seen
logistic regression                        0.972           0.916       0.922
k-nearest neighbours (k=5)                 0.966           0.869       0.875
random forest (100 trees)                  0.981           0.895       0.893

Accuracy by subject (LOSO out-of-fold; * = test subject, scored by the model fit on all 21 others):
1 0.99, 2* 0.96, 3 0.90, 4* 0.96, 5 0.88, 6 0.87, 7 0.92, 8 0.89, 9* 0.82, 10* 0.75, 11 0.99, 12* 0.92, 13* 0.94, 14 0.66, 15 0.99, 16 0.73, 17 0.97, 18* 0.95, 19 0.87, 20* 0.95, 21 0.96, 22 0.92, 23 0.95, 24* 0.99, 25 0.89, 26 0.95, 27 0.99, 28 0.90, 29 0.96, 30 0.98
  training subjects: mean 0.912, SD 0.085, min 0.66
  test subjects: mean 0.917, SD 0.078, min 0.75

LOSO out-of-fold confusion, % of each true activity:
predicted           laying  sitting  standing  walking  walking_downstairs  walking_upstairs
true
laying                98.0      0.0       0.2      0.0                 0.1               1.6
sitting                0.9     92.0       7.1      0.0                 0.0               0.1
standing               0.0      6.9      92.9      0.1                 0.0               0.0
walking                0.0      0.0       0.0     88.0                 6.0               6.0
walking_downstairs     0.0      0.1       0.0      6.3                85.0               8.6
walking_upstairs       0.0      0.0       0.0      3.4                 7.3              89.4

```

### What this shows

- **Same model, same numbers.** Where the splits are identical (shuffled 5-fold, leave-one-subject-out, the official holdout), xdflow and scikit-learn give identical accuracy; the grouped 5-fold rows agree within 0.006.
- **The split decides the answer.** Shuffled windows report 0.972. Keeping subjects apart gives 0.914, in line with the 0.922 on never-seen people. Shuffled CV also ranks the random forest first, while subject-grouped CV and the unseen people both rank logistic regression first. In xdflow, these are the same pipeline with a different `group_coord`.
- **Per-subject accuracy and the confusion table come straight from the data.** Both are a `groupby` or `crosstab` on the coordinates the out-of-fold predictions carry.

## What xdflow buys you here

- **The split boundary is a name.** `group_coord="subject"`, `"segment"`, or plain windows with `KFoldValidator`: the same pipeline, three questions. With scikit-learn the same choice needs a separate `groups` array built and kept aligned with `X`.
- **Metadata stays attached to predictions.** `oof_predictions_container_` and `holdout_container_` carry `subject`, `activity` and `segment` (out-of-fold predictions also carry `fold`), so per-subject accuracy and confusion tables are a `groupby` or `crosstab`, not an index lookup.
- **The holdout is defined by metadata.** `test_group_ids` comes straight from the `subject` and `split` coordinates.
- **Preprocessing and the model are one object.** The window features run once before the folds, and the scaler and classifier refit inside them. The fitted pipeline goes from raw windows to predictions.
- **You choose the question; xdflow keeps it consistent.** Windows, segments or people are one argument apart, and the grouping you pick flows through cross-validation, the holdout and the per-subject results, with no array to keep in sync.
- **Your model stays your model.** With identical splits, xdflow and scikit-learn give identical scores, and you can wrap an existing scikit-learn pipeline with `SKLearnPredictor.from_estimator` to keep it.

## Next steps

- [EEG Motor Imagery: xdflow vs. sklearn](eeg-motor-imagery.md) applies the same comparison to EEG, where the pitfall is fitting a supervised step before the split.
- [Writing Custom Transforms](../guides/writing-transforms.md) covers `@step`, including steps that learn from training data.
- [Execution Model](../concepts/execution.md) describes the stateless/stateful split boundary.
- [Hyperparameter Tuning](tuning.md) shows how to choose features or models inside the validation loop instead of by hand.
