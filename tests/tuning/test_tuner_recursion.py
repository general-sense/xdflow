from unittest.mock import MagicMock

import optuna
import pytest

from xdflow.composite.base import CompositeTransform
from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator
from xdflow.tuning.base import Tuner


class MockSimpleTransform(Transform):
    is_stateful = False

    def __init__(self, param_a=1, param_b="x", **kwargs):
        super().__init__(**kwargs)
        self.param_a = param_a
        self.param_b = param_b

    def _transform(self, data, **kwargs):
        return data  # pragma: no cover

    def get_expected_output_dims(self, input_dims):
        return input_dims


class MockTransformWithState(Transform):
    is_stateful = True

    def __init__(self, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def _transform(self, data, **kwargs):
        return data  # pragma: no cover

    def _fit(self, data, **kwargs):
        return self  # pragma: no cover

    def get_expected_output_dims(self, input_dims):
        return input_dims


class MockComposite(CompositeTransform):
    def __init__(self, name: str, steps: list, **kwargs):
        self.name = name
        self.steps = steps
        self.transform_from_name = {step[0]: step[1] for step in steps}
        super().__init__(**kwargs)

    @property
    def children(self):
        return [step[1] for step in self.steps]

    @property
    def is_predictor(self) -> bool:
        return False

    def _validate_composition(self):
        pass

    def _transform(self, data, **kwargs):
        for _, transform in self.steps:  # pragma: no cover
            data = transform.transform(data, **kwargs)  # pragma: no cover
        return data  # pragma: no cover

    def clone(self):
        cloned_steps = [(name, transform.clone()) for name, transform in self.steps]
        return MockComposite(name=self.name, steps=cloned_steps)

    def get_expected_output_dims(self, input_dims):
        return input_dims


MockPipeline = MockComposite
MockSwitchTransform = MockComposite


@pytest.fixture
def nested_pipeline():
    inner_pipeline = MockPipeline(
        name="inner_pipe",
        steps=[
            ("inner_step1", MockSimpleTransform(param_a=10)),
            ("inner_step2", MockTransformWithState(random_state=1)),
        ],
    )

    switch = MockSwitchTransform(
        name="switch",
        steps=[
            ("inner_pipe", inner_pipeline),
            ("switch_option2", MockSimpleTransform()),
        ],
    )

    main_pipeline = MockPipeline(
        name="main_pipe",
        steps=[
            ("main_step1", MockSimpleTransform(param_a=100)),
            ("main_step2", switch),
            ("main_step3", MockTransformWithState(random_state=42)),
        ],
    )
    return main_pipeline


@pytest.fixture
def param_grid(nested_pipeline):
    return {
        nested_pipeline.name: {
            "main_step1": {"param_a": [2]},
            "main_step2": {
                "inner_pipe": {
                    "inner_step1": {"param_a": (10, 20)},
                    "inner_step2": {"random_state": [1, 2, 123]},
                },
                "switch_option2": {"param_b": ["a"]},
            },
            "main_step3": {"random_state": (100, 200.0)},
        }
    }


@pytest.fixture
def mock_cv():
    cv = MagicMock(spec=CrossValidator)
    cv.cross_validate.return_value = 0.9
    return cv


@pytest.fixture
def mock_data_container():
    return MagicMock(spec=DataContainer)


@pytest.fixture
def tuner_instance(nested_pipeline, mock_cv, param_grid, mock_data_container):
    return Tuner(
        pipelines_to_tune=[nested_pipeline],
        cv_strategy=mock_cv,
        param_grid=param_grid,
        initial_data_container=mock_data_container,
        random_seed=123,
    )


def test_inject_random_seeds_into_pipelines(tuner_instance):
    main_pipeline = tuner_instance.pipelines_to_tune["main_pipe"]
    switch = main_pipeline.get_transform_from_name("main_step2")
    inner_pipe = switch.get_transform_from_name("inner_pipe")
    inner_step2 = inner_pipe.get_transform_from_name("inner_step2")
    main_step3 = main_pipeline.get_transform_from_name("main_step3")

    assert inner_step2.random_state == 123
    assert main_step3.random_state == 123


def test_enqueue_initial_trials_invalid_init_raises(tuner_instance, nested_pipeline):
    nested_pipeline.get_transform_from_name("main_step1").param_a = 999

    tuner_instance.study = optuna.create_study()
    with pytest.raises(ValueError, match="has initialized value"):
        tuner_instance._enqueue_initial_trials()


def test_suggest_and_set_params(tuner_instance, nested_pipeline, param_grid):
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

    trial.suggest_int = MagicMock(return_value=15)
    trial.suggest_categorical = MagicMock(return_value=2)
    trial.suggest_float = MagicMock(return_value=150.5)

    pipeline_param_grid = param_grid[nested_pipeline.name]

    tuner_instance._suggest_and_set_params(nested_pipeline, pipeline_param_grid, nested_pipeline.name, trial)

    main_step1 = nested_pipeline.get_transform_from_name("main_step1")
    switch = nested_pipeline.get_transform_from_name("main_step2")
    inner_pipe = switch.get_transform_from_name("inner_pipe")
    inner_step1 = inner_pipe.get_transform_from_name("inner_step1")
    inner_step2 = inner_pipe.get_transform_from_name("inner_step2")
    main_step3 = nested_pipeline.get_transform_from_name("main_step3")

    assert main_step1.param_a == 2
    assert inner_step1.param_a == 15
    assert inner_step2.random_state == 2
    assert main_step3.random_state == 150.5


def test_enqueue_initial_trials_includes_selected_switch_params(mock_cv, mock_data_container):
    """The baseline trial must use the configured params of the selected switch branch, not sample them."""
    from xdflow.composite import SwitchTransform

    switch = SwitchTransform(
        choices=[
            ("configured", MockSimpleTransform(param_a=10, param_b="x")),
            ("passthrough", MockSimpleTransform()),
        ],
        choose="configured",
    )
    pipeline = MockPipeline(name="switch_init", steps=[("switch", switch)])
    param_grid = {pipeline.name: {"switch": {"configured": {"param_a": (5, 15), "param_b": ["x", "y"]}}}}
    tuner = Tuner(
        pipelines_to_tune=[pipeline],
        cv_strategy=mock_cv,
        param_grid=param_grid,
        initial_data_container=mock_data_container,
        use_mlflow=False,
    )
    tuner.study = optuna.create_study()

    tuner._enqueue_initial_trials()

    assert tuner.study.get_trials()[0].system_attrs["fixed_params"] == {
        "pipeline": "switch_init",
        "switch_init__switch__choose": "configured",
        "switch_init__switch__configured__param_a": 10,
        "switch_init__switch__configured__param_b": "x",
    }


def test_implicit_optional_params_rebuild_the_evaluated_pipeline():
    """Params given at an OptionalTransform's level tune its branch; the recorded choice must match."""
    import numpy as np
    import xarray as xr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    from xdflow.composite import Pipeline
    from xdflow.composite.switch_transform import OptionalTransform
    from xdflow.core.data_container import DataContainer
    from xdflow.cv import KFoldValidator
    from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer

    rng = np.random.default_rng(0)
    data = DataContainer(
        xr.DataArray(
            rng.normal(size=(60, 6)),
            dims=("trial", "feature"),
            coords={"trial": np.arange(60), "feature": np.arange(6), "y": ("trial", np.array(["a", "b"] * 30))},
        )
    )
    optional = OptionalTransform(SKLearnTransformer(PCA, sample_dim="trial", n_components=2), name="pca")
    clf = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="y")
    pipeline = Pipeline("pipe", [("opt", optional), ("clf", clf)])
    tuner = Tuner(
        pipelines_to_tune=pipeline,
        cv_strategy=KFoldValidator(n_splits=3, use_stateful_fit_cache=False, verbose=False),
        param_grid={"pipe": {"opt": {"n_components": [2, 4]}}},
        initial_data_container=data,
        use_mlflow=False,
        use_cache=False,
        verbose=0,
    )
    tuner.tune(n_trials=4)

    assert {t.params["pipe__opt__choose"] for t in tuner.study.trials} == {"pca"}
    # the baseline trial is enqueued with the configured value (not merely sampled to it)
    assert tuner.study.trials[0].system_attrs["fixed_params"]["pipe__opt__pca__n_components"] == 2
    best = tuner.get_best_pipeline().get_transform_from_name("opt")
    assert best.choose == "pca"
    assert (
        best.get_transform_from_name("pca").estimator.n_components
        == tuner.study.best_params["pipe__opt__pca__n_components"]
    )
