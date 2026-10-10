# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-10-10

- Prepared initial open-source release.
- Added reusable ML primitives for multi-output prediction, sample weighting, and sampled-domain validation.
- Updated public tutorials, runnable examples, API notes, and LLM coding-agent guidance.
- Added an EEG motor imagery example and tutorial that compares xdflow with plain scikit-learn on PhysioNet data.
- Added a human activity recognition example, tutorial and Colab notebook (UCI HAR smartphone data) on subject-grouped vs. shuffled-window validation.

### Added

- `CrossValidator.oof_predictions_container_`: out-of-fold predictions with their trial coordinates and a
  `fold` coordinate, so CV results can be broken down by subject, session or any other metadata.
- MNE-Python interop: `from_mne_epochs(epochs, subject=...)` converts `Epochs` to a `(trial, channel, time)`
  container with event labels, metadata columns and extra trial coordinates; `CSPTransform` (new `mne` extra)
  wraps `mne.decoding.CSP` as a stateful step.
- scikit-learn wrappers align new data to the feature/channel labels seen at fit before predicting, so input
  whose channels arrive in another order gives the same predictions (missing labels raise).
- `SKLearnPredictor.from_estimator(...)` wraps an existing scikit-learn estimator or `Pipeline` instance, so
  existing models can use xdflow validation, caching and tuning unchanged. Predictors accept N-D input
  sample-first (e.g. `trial × channel × time` for an MNE CSP pipeline).

### Fixed

- Hyperparameters set after construction (`set_params`, the Tuner, seeds) now reach every CV fold and
  clone. Previously sklearn estimator params were rebuilt from constructor values, so tuning had no effect.
- Cache keys hash full data and labels instead of a 32-value sample, so `use_cache` and the CV fold
  cache can no longer return results computed on different data. A cache hit never skips fitting a
  stateful pipeline.
- Clones get their own copy of a fitted `LabelEncoder`, so finalizing several pipelines from one
  validator no longer relabels earlier models.
- `GroupedKFoldValidator` holdout groups were chosen by trial label as if it were a position.
- `GroupApplyTransform` with a `Pipeline` template ending in a predictor can predict and be
  cross-validated. Previously `predict()` raised `TypeError` and validators could not resolve scoring.
- `GroupApplyTransform` and domain aligners return samples in input order. GroupApply's unseen-group
  averaging votes on probabilities rather than hard labels.
- Regression ensembles average member predictions; score-based ensemble weights compare decoded labels.
- `ZCAWhitening` whitens (it returned its input); `GlobalColoringProjection` keeps the top components.
- PCA/ZCA apply fitted matrices by channel label, not position.
- Confusion matrices keep rows for classes absent from a split; scoring is re-resolved after changes.
- sklearn metric functions (e.g. `accuracy_score`) work as `scoring`; `"accuracy"` is a string scorer.
- Pipelines ending in `EnsemblePredictor` can be cross-validated. Tuner pruners receive fold scores.
- `HilbertPhaseTransform` keeps float output for integer input, and `relative_average` uses a circular mean.
- `UnflattenTransform` keeps coords that vary along unstacked dims.
- `SampleWeightTransform` keeps fractional weights; `stratified_sample` no longer reseeds the global RNG.
- `BalanceClassWeightTransform` computes weights per training fold inside CV.
- `LGBMPredictor` early stopping works with multiple targets.
- `SklearnCVAdapter` recounts splits for each container.
- Cache keys distinguish different functions/closures, ignore `verbose`, and include config exposed only via
  `get_params` (e.g. `AdaptWrapperTransform` estimator kwargs).
- GroupApply and domain aligners restore input order without needing unique trial labels. GroupApply's
  unseen-group averaging keeps classes only some group models saw, and falls back to a label vote for
  models without `predict_proba`.
- Confusion matrices use the class order from evaluation time, even after `finalize_pipeline`.
- Estimator factories (lambdas, `functools.partial`) clone; Tuner seeds reach GroupApply templates.
- Cache keys cover ufuncs and other callable instances, bound methods, keyword-only defaults and recursive
  closures, and are stable across processes. Nested GroupApply/aligner steps keep input order.
- `LGBMPredictor` early stopping works with target patterns resolving to several targets.
- `RegressOutReferenceTransform(use_fit=True)` refuses channels it was not fitted on.

### Changed

- `set_params` raises for names that are not constructor arguments. Aligners and `EnsemblePredictor`
  raise for unknown constructor kwargs. Writes to an automatic MultiOutput wrapper's own params raise.
  All of these were previously ignored silently.
- Domain aligners raise when groups have different class sets. Cross-validation raises on duplicate
  trial labels.
- Pass-through transforms (e.g. `IdentityTransform`) return a new container instead of the input.
- `Tuner` defaults to `use_mlflow=False`.
- Stateless `ZScoreTransform`/`DemeanTransform`/`RemoveOutliersTransform` warn when they pool statistics
  across trials (which leaks validation trials inside CV); use `use_fit=True` or include `trial` in `per_dim`.
- Cache diagnostics print only when called with `verbose=True`.
- Domain aligners build their strategy at fit; `get_strategy_and_kwargs` was removed. Aligners pickled with
  older versions still load and clone.
- `RegressOutReferenceTransform` gains `use_fit` (default behaviour unchanged) and warns when pooling trials.
- `CropTimeTransform` arguments renamed `time_window_start_ms`/`time_window_end_ms` -> `start`/`end` (they are in
  the time coordinate's units, not milliseconds), and a window that selects no samples raises.
- `BandpassFilterTransform` requires `fs` (the old default of 500 Hz silently mis-filtered other sampling
  rates) and no longer needs the `spectral` extra; only `MultiTaperTransform` does.

### Refit required for saved models

- Models containing `ZCAWhitening`/`ZCATransform`: the old transform returned its input. Unpickled old
  instances now raise instead of silently producing different features.
- Models using `HilbertPhaseTransform(mode="relative_average")`: features change (circular mean). This
  transform is stateless, so old pickles cannot detect it; refit them.
