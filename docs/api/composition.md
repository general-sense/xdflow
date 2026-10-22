# Composition API

Composition APIs define how transforms are combined while keeping the pipeline visible to validators and tuners. Use them for sequential pipelines, branching, per-group fitting, optional steps, and ensembles when those choices should participate in split, refit, and cache planning.

## Base Composition Types

::: xdflow.composite.base.TransformStep

::: xdflow.composite.base.CompositeTransform

## Pipelines

::: xdflow.composite.pipeline.Pipeline

::: xdflow.composite.pipeline.make_pipeline

## Grouped Application

::: xdflow.composite.group_apply.GroupApplyTransform

## Parallel Branches

::: xdflow.composite.transform_union.TransformUnion

::: xdflow.composite.transform_union.UnionWithInput

## Conditional Branches

::: xdflow.composite.switch_transform.SwitchTransform

::: xdflow.composite.switch_transform.OptionalTransform

## Ensembles

::: xdflow.composite.ensemble.EnsembleMember

::: xdflow.composite.ensemble.EnsemblePredictor
