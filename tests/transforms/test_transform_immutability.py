"""
Comprehensive immutability tests for xdflow transforms.

This module tests that all transforms follow the Transform-Responsibility Model:
transforms must not mutate their input DataContainer or its underlying data.
"""

import numpy as np
import pytest

from xdflow.composite import (
    GroupApplyTransform,
    OptionalTransform,
    Pipeline,
    SwitchTransform,
    TransformUnion,
)
from xdflow.transforms.basic_transforms import (
    AverageTransform,
    CropTimeTransform,
    FlattenTransform,
    FunctionTransform,
    IdentityTransform,
    RenameDimsTransform,
    TransposeDimsTransform,
    TrialSampler,
    UnflattenTransform,
)
from xdflow.transforms.normalization import (
    DemeanTransform,
    ZScoreTransform,
)
from xdflow.transforms.pca import GlobalFeaturePCA

# ====================
# Basic Transforms
# ====================


def test_average_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test AverageTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = AverageTransform(dims="time")
    assert_transform_immutability(transform, container)


def test_flatten_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test FlattenTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = FlattenTransform(dims=("channel", "time"))
    assert_transform_immutability(transform, container)


def test_trial_sampler_immutability(data_container_factory, assert_transform_immutability):
    """Test TrialSampler immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = TrialSampler(n_trials=5, shuffle=True)
    assert_transform_immutability(transform, container)


def test_function_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test FunctionTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = FunctionTransform(func=np.abs, expected_output_dims=("trial", "channel", "time"))
    assert_transform_immutability(transform, container)


def test_crop_time_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test CropTimeTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = CropTimeTransform(start=0.1, end=0.5)
    assert_transform_immutability(transform, container)


def test_identity_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test IdentityTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = IdentityTransform()

    # returns a new container sharing the input values, so should error as a view
    with pytest.raises(AssertionError, match="returned a view of the original data!"):
        assert_transform_immutability(transform, container)


def test_rename_dims_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test RenameDimsTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = RenameDimsTransform(rename_map={"channel": "sensor"})

    # should return a view, so should error
    with pytest.raises(AssertionError, match="returned a view of the original data!"):
        assert_transform_immutability(transform, container)


def test_transpose_dims_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test TransposeDimsTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    transform = TransposeDimsTransform(dims=("channel", "time", "trial"))
    assert_transform_immutability(transform, container)


def test_unflatten_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test UnflattenTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
    flat_transform = FlattenTransform(dims=("channel", "time"))
    container_flat = flat_transform.transform(container)
    transform = UnflattenTransform(dim="flat_channel__time")
    assert_transform_immutability(transform, container_flat)


# ====================
# Composite Transforms
# ====================


def test_group_apply_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test GroupApplyTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=5, n_time=10)
    container.data.coords["stimulus"] = ("trial", np.tile(["a", "b"], 5))
    transform = GroupApplyTransform(
        group_coord="stimulus", transform_template=DemeanTransform(per_dim="time", use_fit=True)
    )
    assert_transform_immutability(transform, container)


def test_optional_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test OptionalTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=5, n_time=10)
    transform = OptionalTransform(transform_template=DemeanTransform(per_dim="time"), use=True)
    assert_transform_immutability(transform, container)


def test_pipeline_immutability(data_container_factory, assert_transform_immutability):
    """Test Pipeline immutability."""
    container = data_container_factory(n_trials=10, n_channels=5, n_time=10)
    transform = Pipeline(
        name="test_pipeline",
        steps=[
            ("avg", AverageTransform(dims="time")),
            ("flatten", FlattenTransform(dims=("trial", "channel"))),
        ],
    )
    assert_transform_immutability(transform, container)


def test_switch_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test SwitchTransform immutability."""
    container = data_container_factory(n_trials=10, n_channels=5, n_time=10)
    transform = SwitchTransform(
        choices=[
            ("avg", AverageTransform(dims="time")),
            ("identity", IdentityTransform()),
        ],
        choose="avg",
    )
    assert_transform_immutability(transform, container)


def test_transform_union_immutability(data_container_factory, assert_transform_immutability):
    """Test TransformUnion immutability."""
    container = data_container_factory(n_trials=10, n_channels=5, n_time=10)
    transform = TransformUnion(
        transforms_list=[
            ("avg_time", AverageTransform(dims="time")),
            ("avg_channel", AverageTransform(dims="channel")),
        ],
        from_dims=["channel", "time"],
    )
    assert_transform_immutability(transform, container)


# ====================
# Normalization Transforms
# ====================


def test_demean_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test DemeanTransform immutability."""
    container = data_container_factory(n_trials=5, n_channels=10, n_time=100)
    transform = DemeanTransform(per_dim="channel")
    assert_transform_immutability(transform, container)


def test_zscore_transform_immutability(data_container_factory, assert_transform_immutability):
    """Test ZScoreTransform immutability."""
    container = data_container_factory(n_trials=5, n_channels=10, n_time=100)
    transform = ZScoreTransform(per_dim="channel")
    assert_transform_immutability(transform, container)


# ====================
# PCA Transforms
# ====================


def test_global_feature_pca_immutability(data_container_factory, assert_transform_immutability):
    """Test GlobalFeaturePCA immutability."""
    container = data_container_factory(n_trials=10, n_channels=10, n_time=10)
    transform = GlobalFeaturePCA(n_components=5)
    assert_transform_immutability(transform, container)
