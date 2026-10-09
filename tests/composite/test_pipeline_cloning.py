"""Test that GroupApplyTransform can clone Pipeline templates."""

from xdflow.composite import GroupApplyTransform, Pipeline
from xdflow.core.base import Transform


class DummyTransform(Transform):
    def __init__(self, multiplier=1.0):
        super().__init__()
        self.multiplier = multiplier

    def _transform(self, container, **kwargs):
        return container

    def get_expected_output_dims(self, input_dims):
        return input_dims


def test_pipeline_cloning():
    pipeline = Pipeline(name="test_pipeline", steps=[("step1", DummyTransform(multiplier=2.0))])

    group_apply = GroupApplyTransform(group_coord="group", transform_template=pipeline)

    cloned = group_apply.transform_template.clone()
    assert cloned is not pipeline
    assert cloned.name == pipeline.name


def test_composite_set_params_reach_live_object_and_clone():
    from sklearn.linear_model import LogisticRegression

    from xdflow.composite import EnsemblePredictor, GroupApplyTransform, OptionalTransform
    from xdflow.transforms.normalization import ZScoreTransform
    from xdflow.transforms.sklearn_transform import SKLearnPredictor

    def lr():
        return SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus")

    optional = OptionalTransform(ZScoreTransform(per_dim="feature"), use=True)
    optional.set_params(use=False)
    assert optional.choose == optional.clone().choose == "identity"

    group_apply = GroupApplyTransform(group_coord="session", transform_template=lr())
    group_apply.set_params(group_coord="animal")
    assert group_apply.combined_group_coord == "animal"

    ensemble = EnsemblePredictor(members=[("p1", lr()), ("p2", lr())], sample_dim="trial", target_coord="stimulus")
    ensemble.set_params(p1__C=0.02)
    assert ensemble.clone().members[0].predictor.estimator.C == 0.02

    inner = Pipeline(name="inner", steps=[("z", ZScoreTransform(per_dim="feature"))])
    outer = Pipeline(name="outer", steps=[("inner", inner), ("clf", lr())])
    assert not inner.is_stateful
    outer.set_params(inner__z__use_fit=True)
    assert inner.is_stateful
