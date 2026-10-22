# Transforms API

Transforms are the reusable preprocessing, feature extraction, and model-adapter units that move `DataContainer` objects through a pipeline. Validators and tuners use transform metadata during execution: dimension declarations define valid handoffs between steps, and `is_stateful` controls which steps are refit inside each fold versus reused as fold-invariant work.

For authoring guidance, examples, and test patterns, see [Writing Custom Transforms](../guides/writing-transforms.md).

## Basic Transforms

::: xdflow.transforms.basic_transforms.TransposeDimsTransform

::: xdflow.transforms.basic_transforms.RenameDimsTransform

::: xdflow.transforms.basic_transforms.IdentityTransform

::: xdflow.transforms.basic_transforms.SampleWeightTransform

::: xdflow.transforms.basic_transforms.BalanceClassWeightTransform

::: xdflow.transforms.basic_transforms.AverageTransform

::: xdflow.transforms.basic_transforms.FlattenTransform

::: xdflow.transforms.function_step.step

::: xdflow.transforms.basic_transforms.FunctionTransform

::: xdflow.transforms.basic_transforms.UnflattenTransform

::: xdflow.transforms.basic_transforms.TrialSampler

::: xdflow.transforms.basic_transforms.CropTimeTransform

## Cleaning And Normalization

::: xdflow.transforms.cleaning.CARTransform

::: xdflow.transforms.cleaning.RegressOutReferenceTransform

::: xdflow.transforms.cleaning.RemoveOutliersTransform

::: xdflow.transforms.normalization.DemeanTransform

::: xdflow.transforms.normalization.ZScoreTransform

## Sklearn Adapters

::: xdflow.transforms.sklearn_transform.SKLearnTransform

::: xdflow.transforms.sklearn_transform.SKLearnTransformer

::: xdflow.transforms.sklearn_transform.SKLearnPredictor

::: xdflow.transforms.multi_output_wrapper.MultiOutputRegressorFactory

::: xdflow.transforms.multi_output_wrapper.MultiOutputClassifierFactory

::: xdflow.transforms.multi_output_wrapper.make_multi_output

## Estimators And Predictors

::: xdflow.transforms.nearestcentroid.NearestCentroidTransform

::: xdflow.transforms.nearestcentroid.NearestCentroid

::: xdflow.transforms.lda.CholeskyLDA

::: xdflow.transforms.lda.CholeskyLDATransformer

::: xdflow.transforms.lgbm_predictor.LGBMPredictor

## Time-Series And Spatial Transforms

::: xdflow.transforms.phase.HilbertPhaseTransform

::: xdflow.transforms.pca.GlobalFeaturePCA

::: xdflow.transforms.spatial.LaplacianCSDTransform

::: xdflow.transforms.spatial.WindowMeanPyramidTransform

::: xdflow.transforms.spatial.GaussianPyramidTransform

::: xdflow.transforms.zca.LocalZCAWhitening

::: xdflow.transforms.zca.GlobalZCAWhitening

::: xdflow.transforms.zca.GlobalColoringProjection

::: xdflow.transforms.zca.ZCAWhitening

::: xdflow.transforms.zca.ZCATransform

## Domain Adaptation

::: xdflow.transforms.domain_adaptation.AdaptiveStrategy

::: xdflow.transforms.domain_adaptation.SingleTargetStrategy

::: xdflow.transforms.domain_adaptation.JointGroupStrategy

::: xdflow.transforms.domain_adaptation.AdaptiveTransform

::: xdflow.transforms.domain_adaptation.SingleTargetAligner

::: xdflow.transforms.domain_adaptation.JointGroupAligner

::: xdflow.transforms.domain_adaptation.ProcrustesAligner

::: xdflow.transforms.domain_adaptation.CoralAligner

::: xdflow.transforms.domain_adaptation.SAAligner

::: xdflow.transforms.domain_adaptation.CCAAligner

::: xdflow.transforms.domain_adaptation.MCCAAligner

::: xdflow.transforms.domain_adaptation.GCCAAligner

::: xdflow.transforms.domain_adaptation.JDAAligner

::: xdflow.transforms.domain_adaptation.KCCAAligner

::: xdflow.transforms.adapt_wrapper.AdaptWrapperStrategy

::: xdflow.transforms.adapt_wrapper.AdaptWrapperTransform

## Optional Spectral Module

`xdflow.transforms.spectral` depends on `spectral-connectivity`. Install `xdflow[spectral]` or `xdflow[all]` before importing:

- `MultiTaperTransform`
- `BandpassFilterTransform`

The published docs avoid importing that module during the standard docs build so Read the Docs does not need optional spectral dependencies.
