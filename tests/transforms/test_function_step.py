"""Plain functions become pipeline steps with @step: hyperparameters, cloning, caching, safety checks."""

import inspect
import pickle

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow import DataContainer
from xdflow.composite.pipeline import make_pipeline
from xdflow.cv import KFoldValidator
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.function_step import step
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.utils.cache_utils import _get_object_params


@step
def scale(data, factor=2.0, *, offset=0.0):
    """Multiply, then shift."""
    return data * factor + offset


@step(output_dims=("trial", "channel"))
def time_mean(data):
    return data.mean("time")


@step
def as_numpy(data):
    return np.abs(data.values)


def _trial_means(data, dim="trial"):
    return data.mean(dim)


@step(fit=_trial_means)
def recenter(data, state, dim="trial"):
    return data - state


@pytest.fixture
def container():
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.normal(size=(12, 3, 5)),
        dims=("trial", "channel", "time"),
        coords={
            "trial": np.arange(12) + 100,
            "channel": ["c0", "c1", "c2"],
            "time": np.linspace(0, 1, 5),
            "label": ("trial", np.array(["a", "b"])[np.arange(12) % 2]),
        },
    )
    return DataContainer(data)


def test_hyperparameters_are_constructor_arguments(container):
    assert str(inspect.signature(scale)) == "(factor=2.0, *, offset=0.0, sel=None, drop_sel=None)"
    assert scale.__doc__ == "Multiply, then shift."
    transformed = scale(3.0, offset=1.0).transform(container)
    np.testing.assert_allclose(transformed.data.values, container.data.values * 3.0 + 1.0)
    assert transformed.data.dims == container.data.dims
    np.testing.assert_array_equal(transformed.data.trial.values, container.data.trial.values)
    np.testing.assert_allclose(scale.func(container.data, 2.0), container.data * 2.0)


def test_clone_and_set_params(container):
    original = scale(factor=3.0)
    original.set_params(offset=5.0)
    clone = original.clone()
    assert (clone.factor, clone.offset) == (3.0, 5.0)
    assert clone is not original
    with pytest.raises(ValueError, match="no parameter"):
        original.set_params(nope=1)


def test_tunable_through_pipeline_parameter_paths(container):
    pipeline = make_pipeline(scale(), FlattenTransform(dims=("channel", "time")), _predictor())
    pipeline.set_params(scale__factor=7.0)
    assert pipeline.clone().steps[0].transform.factor == 7.0


def test_pickle_and_fitted_state_round_trip(container):
    fitted = recenter().fit(container)
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.func is recenter.func and restored.fit_func is _trial_means
    np.testing.assert_allclose(restored.transform(container).data.values, fitted.transform(container).data.values)
    assert not hasattr(fitted.clone(), "_state")


def test_numpy_return_keeps_dims_and_coords(container):
    out = as_numpy().transform(container).data
    assert out.dims == container.data.dims
    np.testing.assert_array_equal(out.coords["label"].values, container.data.coords["label"].values)


def test_numpy_return_with_new_shape_raises(container):
    @step
    def collapse(data):
        return data.values.mean(axis=-1)

    with pytest.raises(ValueError, match="Return an xr.DataArray"):
        collapse().transform(container)


def test_in_place_edits_raise_and_leave_input_untouched(container):
    @step
    def zero_first(data):
        data.values[0] = 0.0
        return data

    before = container.data.values.copy()
    with pytest.raises(ValueError, match="modified its input in place"):
        zero_first().transform(container)
    np.testing.assert_array_equal(container.data.values, before)
    assert container.data.values.flags.writeable


def test_returned_input_is_writable_downstream(container):
    @step
    def identity(data):
        return data

    out = identity().transform(container)
    assert out.data.values.flags.writeable
    assert not np.shares_memory(out.data.values, container.data.values)


def test_cross_trial_function_is_rejected(container):
    @step
    def leaky(data):
        return data - data.mean("trial")

    with pytest.raises(ValueError, match="fit function"):
        leaky().transform(container)

    @step(check_independence=False)
    def unchecked(data):
        return data - data.mean("trial")

    unchecked().transform(container)


def test_trial_selection_is_not_misreported(container):
    @step
    def first_half(data):
        return data.isel(trial=slice(0, 6))

    assert first_half().transform(container).data.sizes["trial"] == 6


def test_fit_runs_on_training_folds_only(container):
    seen = []

    def fit_means(data):
        seen.append(set(data.trial.values.tolist()))
        return data.mean("trial")

    @step(fit=fit_means)
    def center(data, state):
        return data - state

    pipeline = make_pipeline(center(), FlattenTransform(dims=("channel", "time")), _predictor())
    validator = KFoldValidator(n_splits=3, verbose=False)
    validator.set_pipeline(pipeline)
    validator.cross_validate(container)
    assert seen == [set(train.tolist()) for train, _ in validator.split_trials_]


def test_declared_output_dims(container):
    assert time_mean().get_expected_output_dims(("trial", "channel", "time")) == ("trial", "channel")
    with pytest.raises(NotImplementedError, match="output_dims"):
        scale().get_expected_output_dims(("trial",))


def test_cache_key_follows_function_code():
    def build(body: str):
        namespace: dict = {}
        exec(f"def shift(data):\n    return {body}", namespace)  # same name and module, different code
        namespace["shift"].__module__ = __name__
        return step(namespace["shift"])()

    assert _get_object_params(build("data + 1")) != _get_object_params(build("data + 2"))
    assert _get_object_params(build("data + 1")) == _get_object_params(build("data + 1"))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def f(*, data): return data", "positionally"),
        ("def f(data, *args): return data", "named"),
        ("def f(data, sel=None): return data", "clashes"),
        ("def f(data, state): return data", "clashes"),
    ],
)
def test_invalid_signatures(source, message):
    namespace: dict = {}
    exec(source, namespace)
    with pytest.raises(TypeError, match=message):
        step(namespace["f"])


def test_fit_parameters_must_be_step_parameters():
    def fit_fn(data, window=3):
        return 0

    with pytest.raises(TypeError, match="window"):

        @step(fit=fit_fn)
        def apply(data, state):
            return data

    with pytest.raises(TypeError, match="state"):

        @step(fit=_trial_means)
        def no_state(data):
            return data


def _predictor():
    return SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="label")


@pytest.mark.parametrize(
    "leaky",
    [
        lambda d: (d - d.mean("trial")).isel(trial=slice(None, None, -1)),  # reorders trials
        lambda d: d.assign_coords(mean=("trial", np.full(d.sizes["trial"], float(d.mean())))),  # stats in coords
        lambda d: d * 0 + 1e9 + d.mean("trial"),  # small difference on a large offset
        lambda d: d.mean("trial"),  # trial dim gone
    ],
)
def test_cross_trial_functions_are_rejected_however_they_hide(container, leaky):
    with pytest.raises(ValueError, match="fit"):
        step(leaky)().transform(container)


def test_coordinates_are_read_only_too(container):
    @step
    def relabel(data):
        data.coords["label"].values[:] = "z"
        return data

    with pytest.raises(ValueError, match="in place"):
        relabel().transform(container)
    assert set(container.data.coords["label"].values) == {"a", "b"}


def test_index_buffers_are_not_shared(container):
    @step
    def shift_time_index(data):
        values = data.indexes["time"].values
        if values.flags.writeable:
            values *= 10
        return data

    shift_time_index().transform(container)
    np.testing.assert_allclose(container.data.time.values, np.linspace(0, 1, 5))
    assert container.data.sel(time=1.0).sizes["trial"] == 12


def test_a_large_unrelated_feature_does_not_hide_leakage(container):
    big = container.data.astype(np.float32).copy()
    big[:, 0, :] = 1e6  # one channel with a large constant value

    @step
    def center_other_channels(data):
        return xr.concat(
            [data.isel(channel=[0]), data.isel(channel=[1, 2]) - data.isel(channel=[1, 2]).mean("trial")], "channel"
        )

    with pytest.raises(ValueError, match="fit"):
        center_other_channels().transform(DataContainer(big))


def test_min_max_normalisation_over_trials_is_rejected_on_ordered_data(container):
    ordered = container.data.copy(data=np.sort(container.data.values, axis=0))  # each value rises with trial

    @step
    def min_max(data):
        return (data - data.min("trial")) / (data.max("trial") - data.min("trial"))

    with pytest.raises(ValueError, match="fit"):
        min_max().transform(DataContainer(ordered))


def test_object_coordinates_with_missing_values_pass(container):
    notes = np.array(["ok", np.nan] * 6, dtype=object)
    data = DataContainer(container.data.assign_coords(note=("trial", notes)))

    @step
    def double(data):
        return data * 2

    double().transform(data)


def test_data_without_trial_labels_passes_the_check(container):
    unlabeled = DataContainer(container.data.drop_vars("trial"))
    out = scale(factor=3.0).transform(unlabeled)
    np.testing.assert_allclose(out.data.values, container.data.values * 3.0)


def test_repeated_trial_labels_get_a_clear_error(container):
    repeated = DataContainer(container.data.assign_coords(trial=np.arange(12) // 2))
    with pytest.raises(ValueError, match="unique"):
        scale().transform(repeated)


def test_outputs_can_be_pickled_for_the_cache(container):
    out = make_pipeline(scale(), recenter()).fit_transform(container)
    assert pickle.loads(pickle.dumps(out)).data.equals(out.data)


def test_statistics_over_trials_in_other_coordinates_are_rejected(container):
    @step
    def attach_channel_means(data):
        return data.assign_coords(mu=("channel", data.mean(("trial", "time")).values))

    with pytest.raises(ValueError, match="depending on which other trials"):
        attach_channel_means().transform(container)


def test_unlabeled_data_reaches_the_rerun_unlabeled(container):
    @step
    def rebuilt(data):
        return xr.DataArray(data.values * 2, dims=data.dims)  # drops every coordinate

    unlabeled = DataContainer(container.data.drop_vars("trial"))
    np.testing.assert_allclose(rebuilt().transform(unlabeled).data.values, container.data.values * 2)


def test_coordinate_attrs_survive_a_step(container):
    data = container.data.assign_coords(time=("time", container.data.time.values, {"units": "s"}))
    out = scale().transform(DataContainer(data))
    assert out.data.time.attrs == {"units": "s"}


def test_nested_coordinate_attrs_cannot_change_the_callers_data(container):
    data = container.data.assign_coords(time=("time", container.data.time.values, {"calibration": {"scale": 1}}))

    @step
    def bump(d):
        d.time.attrs["calibration"]["scale"] += 1
        return d

    bump().transform(DataContainer(data))
    assert data.time.attrs["calibration"]["scale"] == 1


def test_nested_array_and_auxiliary_coordinate_attrs_cannot_change_the_callers_data(container):
    data = container.data.assign_coords(gain=("trial", np.ones(container.data.sizes["trial"]), {"cal": {"x": 1}}))
    data.attrs["cal"] = {"x": 1}

    @step
    def bump(d):
        d.attrs["cal"]["x"] += 1
        d.gain.attrs["cal"]["x"] += 1
        return d

    bump().transform(DataContainer(data))
    assert data.attrs["cal"]["x"] == 1 and data.gain.attrs["cal"]["x"] == 1
