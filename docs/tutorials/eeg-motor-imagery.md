# EEG Motor Imagery: xdflow vs. Plain scikit-learn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-sense/xdflow/blob/main/examples/notebooks/eeg_motor_imagery.ipynb) Run this tutorial as a notebook in your browser.

This tutorial decodes imagined left- vs right-hand movement from a public EEG dataset, once with xdflow and once with plain NumPy and scikit-learn, and compares the two on the same data.

The short version:

- With identical preprocessing, model and folds, the two versions produce **identical scores**: xdflow changes how the model is validated, not the model itself.
- What xdflow adds: labels and subjects travel with the trials as coordinates, splits are requested by coordinate name, per-subject models are one wrapper, and the validator decides which steps run once and which are refit per fold.
- The last point matters on this dataset. A common shortcut in the plain version, fitting CSP on all trials before cross-validating the classifier, reports 0.96 within-subject accuracy. That number is inflated by leakage: the correct estimate, with CSP refit inside each training fold, is **0.64**. In the xdflow version, CSP is a stateful step, so validators always refit it inside each training fold.

The full script is `examples/eeg_motor_imagery.py`. All accuracies below come from running it; the chance thresholds are binomial 97.5th percentiles (`scipy.stats.binom.ppf(0.975, n, 0.5) / n`), and the configuration sweep mentioned at the end was run separately while writing the example.

## Run it

```bash
uv run --extra mne python examples/eeg_motor_imagery.py
```

- **Dependencies:** `pip install "xdflow[mne]"`. MNE reads the EDF files and provides the CSP implementation behind `CSPTransform`; everything else is core xdflow.
- **Download:** the first run downloads 38.6 MB (15 EDF files: subjects 1-5, runs 4, 8 and 12) from PhysioNet. They go into MNE's data directory (`~/mne_data` unless `MNE_DATA` is set). Use `--data-dir PATH` to choose another directory. PhysioNet can be slow: the download took over 10 minutes here.
- **Runtime:** after the download, the script runs in about 15 s on a laptop.
- **Options:** `--subjects 1 2 3 4 5 6 7` uses more subjects. The last subject listed is the cross-subject holdout.

## The data

The [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/) has 64-channel EEG sampled at 160 Hz. In runs 4, 8 and 12, a cue tells the subject to imagine opening and closing the left fist (`T1`) or the right fist (`T2`) for about 4 s. Each run has about 15 cues, so each subject has 42 to 45 trials.

The loader epochs each run with MNE and converts it with `from_mne_epochs`, which turns the event names
into a `label` coordinate and adds `subject` and `run` as trial coordinates. The runs are then stacked into
one `DataArray` with `trial`, `channel` and `time` dimensions.

```python
epochs = mne.Epochs(raw, events, {"left": 1, "right": 2}, *EPOCH, baseline=None, preload=True)
pieces.append(from_mne_epochs(epochs, subject=subject, run=run).data)
...
data = xr.concat(pieces, dim="trial", join="exact")  # fail loudly if channels or times differ
data = data.assign_coords(trial=np.arange(data.sizes["trial"]))  # unique trial ids across runs
```

The result has 216 trials, 64 channels and 881 time samples (-1.0 to 4.5 s around the cue). Classes are balanced within each subject (21-23 trials per class). The loader also raises an error if a recording is not sampled at 160 Hz.

## The model

The model is the classic motor-imagery pipeline:

1. Band-pass filter to 8-30 Hz. This keeps the mu and beta rhythms, whose power drops over the motor cortex opposite the imagined hand.
2. Crop to 0.5-4.0 s after the cue.
3. Apply [common spatial patterns](https://en.wikipedia.org/wiki/Common_spatial_pattern) (CSP), which learns spatial filters that separate the two classes by variance.
4. Classify with linear discriminant analysis (LDA).

CSP uses the labels to learn its filters, so it must only ever see training trials.

`CSPTransform` (in `xdflow.transforms.csp`, installed with `pip install "xdflow[mne]"`) wraps MNE's
`mne.decoding.CSP`. It declares `is_stateful = True`, reads the labels from the `label` coordinate, keeps the
trial-level coordinates on its output, and applies its filters by channel name. Writing a transform like it
yourself takes about 30 lines; see [Writing Custom Transforms](../guides/writing-transforms.md).

The pipeline goes from raw epochs to predictions. For within-subject decoding, `GroupApplyTransform` wraps CSP+LDA, so each value of the `subject` coordinate gets its own CSP+LDA:

```python
def build_pipeline(per_subject: bool = False) -> Pipeline:
    model = Pipeline(
        name="csp_lda",
        steps=[
            ("csp", CSPTransform(n_components=6)),
            ("lda", SKLearnPredictor(LinearDiscriminantAnalysis, sample_dim="trial", target_coord="label")),
        ],
    )
    if per_subject:
        model = GroupApplyTransform(group_coord="subject", transform_template=model)
    return Pipeline(
        name="motor_imagery",
        steps=[
            ("bandpass", BandpassFilterTransform(8.0, 30.0, fs=160)),  # stateless
            ("crop", CropTimeTransform(0.5, 4.0)),  # stateless, label-based slice of `time`
            ("model", model),  # stateful: CSP + LDA
        ],
    )
```

The validator splits this pipeline at its first stateful step:

- The band-pass filter and the crop run once on all trials.
- CSP and LDA are cloned and refit on each training fold.

With a call counter on the example's first three subjects, a 5-fold within-subject run filtered the data once and fit CSP 15 times (5 folds × 3 subjects).

## Within-subject and cross-subject validation

The two evaluations answer different scientific questions, so they use different split policies:

```python
# (a) Within subject: a person's own trials, folds stratified by subject x label,
#     plus 20% of each subject's trials held out until the end.
within = GroupedKFoldValidator(
    n_splits=5, test_size=0.2, group_coord="subject", scoring="accuracy", random_state=0, verbose=False
)
within.set_pipeline(build_pipeline(per_subject=True))
within_cv = within.cross_validate(container)
within_holdout = within.score_on_holdout(container)

# (b) Across subjects: leave one subject out on subjects 1-4, then test on subject 5, never seen in training.
across = LeaveGroupOutValidator(group_coord="subject", test_group_ids=[5], scoring="accuracy", verbose=False)
across.set_pipeline(build_pipeline())
across_cv = across.cross_validate(container)
across_holdout = across.score_on_holdout(container)
```

!!! note "Two kinds of grouped validation"
    `GroupedKFoldValidator(group_coord="subject")` puts every subject in every training fold, stratified by subject × label: the right design for within-subject models. `LeaveGroupOutValidator` keeps each subject on one side of the split, like scikit-learn's `GroupKFold`: the tool for generalizing to new people.

Out-of-fold predictions keep their trial coordinates. Per-subject cross-validation accuracy is therefore an xarray `groupby`, with no index arrays to carry around (`holdout_container_` does the same for the holdout):

```python
predicted = within.oof_predictions_container_.data  # dims: (trial,), coords: label, subject, run, fold
(predicted == predicted["label"]).groupby("subject").mean()
```

## The plain scikit-learn version

The same analysis without xdflow. A plain workflow builds `X`, `y` and `subjects` in the loading loop. Here they are taken from the same `DataArray`, so both versions see exactly the same trials:

```python
X, y, subjects, times = data.values, data["label"].values, data["subject"].values, data["time"].values
b, a = butter(4, (8.0, 30.0), btype="bandpass", fs=160)
X = filtfilt(b, a, X, axis=-1)  # axis -1 must be time
X = X[..., (times >= 0.5) & (times <= 4.0)]


def model():
    return make_pipeline(CSP(n_components=6, log=True), LinearDiscriminantAnalysis())


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
within = [cross_val_score(model(), X[subjects == s], y[subjects == s], cv=kfold).mean() for s in np.unique(subjects)]

train = subjects != 5
across = cross_val_score(model(), X[train], y[train], groups=subjects[train], cv=LeaveOneGroupOut())
across_holdout = model().fit(X[train], y[train]).score(X[~train], y[~train])
```

CSP sits inside the sklearn `Pipeline`, so `cross_val_score` refits it per fold. The bookkeeping is all positional:

- `X`, `y` and `subjects` must stay in the same order through every mask.
- `axis=-1` must be time.
- The crop is a boolean mask over a separate `times` array.
- A per-subject model is a Python loop.
- The filter and crop live outside the sklearn `Pipeline`. To deploy the model, you have to repeat them by hand before calling `predict`, or wrap them in a `FunctionTransformer` that then reruns on every fold.

### A shortcut with a hidden trap

CSP is the expensive, label-dependent step. A tempting shortcut is to compute CSP features once on all trials, "to save time", and then cross-validate only the classifier:

```python
# The mistake: CSP is fit on all trials, including every validation fold.
features = CSP(n_components=6, log=True).fit_transform(X_subject, y_subject)
cross_val_score(LinearDiscriminantAnalysis(), features, y_subject, cv=kfold)
```

The code runs cleanly. However, CSP has already seen the validation trials and their labels, so the spatial filters are tuned to the trials being scored. The last column of the results table shows the effect.

## Results

Output of `examples/eeg_motor_imagery.py` (subjects 1-5, seed 0):

```text
Loaded {'trial': 216, 'channel': 64, 'time': 881} in 1.1s
Trials per subject:
label  left  right
1        23     22
2        21     21
3        23     22
4        21     21
5        21     21

Accuracy (chance = 0.50)                       xdflow  sklearn  sklearn, CSP fit before CV
Within subject, 5-fold CV                       0.645    0.636                       0.962
Within subject, 20% holdout                     0.614        -                           -
Across subjects, LOSO CV (excl. S5)             0.563    0.563                       0.775
Across subjects, holdout S5                     0.500    0.500                           -

Within-subject CV accuracy by subject: S1 0.77 (n=35), S2 0.91 (n=34), S3 0.57 (n=35), S4 0.50 (n=34), S5 0.47 (n=34)
```

### What this shows

- **Same model, same numbers.** Like for like, the two versions agree: identical across subjects (0.563 and 0.500), and within 0.01 within subject (0.645 and 0.636, where xdflow cross-validates the 80% of trials left after its holdout).
- **The shortcut inflates accuracy dramatically.** Fitting CSP on all trials before cross-validating reports 0.962 within subject and 0.775 across subjects, against 0.64 and 0.56 when CSP is fit inside each fold. In xdflow, CSP is a stateful step, so it is always refit inside each fold. The inflation held in every configuration tried (windows, bands, number of components, classifiers).
- **Per-subject results come straight from the data.** The per-subject line in the output is a `groupby` on the `subject` coordinate of the out-of-fold predictions.

## What xdflow buys you here

- **Metadata stays attached.** `label`, `subject` and `run` are coordinates. Filtering, cropping, CSP and prediction all keep them, so nothing has to be re-aligned by position. `CropTimeTransform(0.5, 4.0)` slices the `time` coordinate by value, and CSP selects dimensions by name.
- **Splits are named.** You ask for `group_coord="subject"` and `test_group_ids=[5]` instead of passing a `groups` array that must stay aligned with `X`.
- **Per-group models are one wrapper.** `GroupApplyTransform(group_coord="subject", ...)` replaces the per-subject loop and stays inside the same cross-validation and holdout machinery.
- **The split boundary comes from the steps.** The validator runs stateless steps once and refits stateful steps per fold, so the whole raw-to-prediction pipeline lives in one object without recomputing preprocessing per fold. The safe version is also the fast one, which removes the reason for the precompute shortcut above.
- **Leak safety is declared, not remembered.** Every built-in transform declares whether it learns, and the ones that can pool statistics across trials warn when they do. Your own transforms declare it in one line, `is_stateful = True`, and every validator, cache and tuner respects it from then on.
- **Your model stays your model.** Like for like, xdflow and the plain scikit-learn version give the same scores: xdflow changes how the model is validated, not what it computes. Already have a scikit-learn pipeline? Wrap it with `SKLearnPredictor.from_estimator` and keep it.

## Next steps

- [Writing Custom Transforms](../guides/writing-transforms.md) covers the contract that `CSPTransform` and your own transforms follow.
- [Execution Model](../concepts/execution.md) describes the stateless/stateful split boundary in detail.
- [Hyperparameter Tuning](tuning.md) shows how to select the window, band or number of CSP components inside the validation loop instead of by hand.
