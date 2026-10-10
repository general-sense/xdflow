# Writing Custom Transforms

Transforms are the basic extension point in XDFlow. A transform takes a
`DataContainer`, operates on the wrapped `xarray.DataArray`, and returns a new
`DataContainer`.

Write a custom transform when the operation belongs inside a reusable pipeline:
preprocessing, feature extraction, denoising, reshaping, metadata annotation, or
a model adapter. Prefer existing transforms first, especially `FunctionTransform`
for simple array functions and `SKLearnTransformer` or `SKLearnPredictor` for
sklearn-compatible estimators.

## The Transform Contract

Every transform should answer these questions:

- what dimensions it requires
- what dimensions it produces
- whether it learns state during `fit`
- which coordinates it preserves or intentionally changes
- whether it can safely transform only a selected subset and write it back

The base class handles the public `fit`, `transform`, and `fit_transform`
methods. New transforms usually implement `_transform`, and stateful transforms
also implement `_fit`.

The base `Transform` also logs completed transforms to
`container.data.attrs["data_history"]`. Custom transforms normally do not need
to manage history themselves.

```python
from typing import Any

from xdflow.core import DataContainer, Transform


class MyTransform(Transform):
    is_stateful = False
    input_dims = ()
    output_dims = ()

    def __init__(self, scale: float = 1.0, sel: dict[str, Any] | None = None):
        super().__init__(sel=sel)
        self.scale = scale

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        transformed = container.data * self.scale
        return DataContainer(transformed)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
```

Constructor hyperparameters should be explicit arguments and public attributes
with matching names. Learned state should be private, such as `self._mean` or
`self._estimator`, so `clone()` creates a fresh unfitted transform.

## Dimension Declarations

Use class attributes when dimensions are fixed:

```python
class TrialChannelTimeTransform(Transform):
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")
```

Use dynamic dimensions when the transform depends on constructor arguments or
input shape:

```python
class PeakToPeakTransform(Transform):
    """Compute peak-to-peak amplitude over one dimension."""

    is_stateful = False
    input_dims = ()
    output_dims = ()

    def __init__(self, dim: str = "time"):
        super().__init__()
        self.dim = dim

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        if self.dim not in data.dims:
            raise ValueError(f"Dimension '{self.dim}' not found in input dims {data.dims}.")

        transformed = data.max(dim=self.dim) - data.min(dim=self.dim)
        transformed.name = "peak_to_peak"
        return DataContainer(transformed)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if self.dim not in input_dims:
            raise ValueError(f"Dimension '{self.dim}' not found in input dims {input_dims}.")
        return tuple(dim for dim in input_dims if dim != self.dim)
```

If `output_dims` is non-empty, the inherited `get_expected_output_dims()` returns
it automatically. Implement `get_expected_output_dims()` only when the output
depends on the input dims or constructor settings.

## What XDFlow Checks

Dimension metadata is used in several places:

- `Pipeline` checks adjacent declared `output_dims` and `input_dims` during
  construction when both are known statically.
- `Pipeline(expected_input_dims=...)` validates the expected input dims for every
  step and checks dynamic handoffs through `get_expected_output_dims()`.
- During execution, `expected_input_dims` checks the actual dims before each
  step runs. A step's output is therefore checked when the next step starts.
- `pipeline.get_expected_output_dims(input_dims, print_steps=True)` prints the
  expected dimension evolution without running data.
- `transform_sel` write-back checks dims, sizes, and dimension coordinates before
  replacing a selected subset.

The framework does not infer every semantic requirement. If your transform needs
a coordinate such as `stimulus` or `session`, or an attr such as
`sampling_frequency_hz`, validate that requirement inside the transform and
raise a clear error.

```python
from xdflow.composite import Pipeline
from xdflow.transforms.basic_transforms import RenameDimsTransform

pipeline = Pipeline(
    name="features",
    steps=[
        ("ptp", PeakToPeakTransform(dim="time")),
        ("rename", RenameDimsTransform(rename_map={"channel": "feature"})),
    ],
    expected_input_dims={
        "ptp": ("trial", "channel", "time"),
        "rename": ("trial", "channel"),
    },
)
```

## Preserving Coordinates

Prefer xarray operations such as `.mean(dim=...)`, `.stack(...)`, `.rename(...)`,
and arithmetic on `DataArray` objects. They usually preserve compatible
coordinates and attrs.

When constructing a new `DataArray` from NumPy output, explicitly rebuild the
coords that still make sense:

```python
import xarray as xr


def make_output(data, values, output_dims):
    coords = {name: coord for name, coord in data.coords.items() if set(coord.dims).issubset(output_dims)}
    return xr.DataArray(values, dims=output_dims, coords=coords, attrs=data.attrs)
```

This pattern keeps trial-level coordinates, channel labels, and other compatible
metadata attached while dropping coordinates whose dimensions were removed.

## Stateless Dim-Preserving Transform

If a transform preserves dims, sizes, and coordinates, it can opt into
`transform_sel` and `transform_drop_sel` write-back by setting
`_supports_transform_sel = True`.

```python
from typing import Any

from xdflow.core import DataContainer, Transform


class ClipTransform(Transform):
    is_stateful = False
    input_dims = ()
    output_dims = ()
    _supports_transform_sel = True

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        super().__init__(
            sel=sel,
            drop_sel=drop_sel,
            transform_sel=transform_sel,
            transform_drop_sel=transform_drop_sel,
        )
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        clipped = container.data.clip(min=self.min_value, max=self.max_value)
        return DataContainer(clipped)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
```

Only set `_supports_transform_sel = True` when selected output can be written
back into the original array without changing dims, sizes, or dimension
coordinates.

## Stateful Transform

Stateful transforms learn something in `_fit` and reuse it in `_transform`.
Set `is_stateful = True` so cross-validation clones and refits the transform
inside each training fold.

```python
from typing import Any

import xarray as xr

from xdflow.core import DataContainer, Transform


class ChannelCenterTransform(Transform):
    """Subtract a fitted per-channel mean."""

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "channel", "time")

    def __init__(self, center_dims: tuple[str, ...] = ("trial", "time"), sel: dict[str, Any] | None = None):
        super().__init__(sel=sel)
        self.center_dims = center_dims
        self._mean: xr.DataArray | None = None

    def _fit(self, container: DataContainer, **kwargs) -> "ChannelCenterTransform":
        missing = [dim for dim in self.center_dims if dim not in container.data.dims]
        if missing:
            raise ValueError(f"center_dims not found in input dims: {missing}")

        self._mean = container.data.mean(dim=list(self.center_dims))
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        if self._mean is None:
            raise ValueError("ChannelCenterTransform must be fitted before transform().")
        return DataContainer(container.data - self._mean)
```

Do not put fitted arrays, estimators, encoders, or lookup tables in constructor
arguments. They should be private attributes initialized empty and populated by
`_fit`.

## Selection Arguments

All transforms inherit:

- `sel`: subset the whole input before transforming
- `drop_sel`: drop labels before transforming
- `transform_sel`: transform only a selected subset, then write it back
- `transform_drop_sel`: inverse form of `transform_sel`

`sel` and `drop_sel` change the whole output and are safe for any transform.
`transform_sel` and `transform_drop_sel` require `_supports_transform_sel = True`
because the transformed subset must fit back into the original structure.

## Testing Checklist

Add focused tests when introducing a transform:

```python
def test_peak_to_peak_dims(data_container_factory):
    container = data_container_factory(n_trials=5, n_channels=3, n_time=20)
    transform = PeakToPeakTransform(dim="time")

    result = transform.transform(container)

    assert result.data.dims == ("trial", "channel")
    assert transform.get_expected_output_dims(container.dims) == ("trial", "channel")
    assert "trial" in result.data.coords
    assert "channel" in result.data.coords


def test_channel_center_clone_is_unfitted(data_container_factory):
    container = data_container_factory()
    transform = ChannelCenterTransform()

    transform.fit(container)
    cloned = transform.clone()

    assert cloned.center_dims == transform.center_dims
    assert cloned._mean is None


def test_channel_center_immutability(data_container_factory, assert_transform_immutability):
    container = data_container_factory()
    transform = ChannelCenterTransform()

    assert_transform_immutability(transform, container)
```

For higher-risk transforms, also test:

- missing required dims or coords raise clear errors
- output sizes and coordinates are preserved or intentionally changed
- `expected_input_dims` catches invalid pipeline handoffs
- `transform_sel` works only when the transform truly preserves structure
- stateful transforms produce the same result through `fit_transform` and
  `fit` followed by `transform`

## Common Mistakes

Avoid these patterns:

- using positional axes when a dimension name is available
- mutating `container.data` in place
- returning a bare `xarray.DataArray` instead of `DataContainer`
- storing learned state in public constructor attributes
- marking a split-dependent transform as stateless
- constructing a new `DataArray` without rebuilding compatible coords
- declaring `_supports_transform_sel = True` for a transform that changes dims,
  sizes, or dimension coordinates
