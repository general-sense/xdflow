from typing import Any, cast

from xdflow.composite.base import CompositeTransform, TransformStep
from xdflow.composite.pipeline import Pipeline
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.transforms.basic_transforms import IdentityTransform, RenameDimsTransform


class SwitchTransform(CompositeTransform):
    """
    A conditional transform that selects one of several child transforms to execute.

    This acts as a placeholder in a pipeline for a step that has multiple
    possible implementations. The choice of which transform to run is determined
    at runtime by the `choose` keyword argument passed to `fit` or `transform`.

    Args:
        choices: Preferred style is a list of `(name, transform)` tuples, `TransformStep`s,
            or `Pipeline`s, mirroring how `Pipeline` is declared. For backward compatibility,
            a `dict[str, Transform]` is also accepted.
        choose: Optional explicit selection for the switch. If provided, it must match one
            of the choice names. If not provided, the user must supply `choose` at
            fit/transform time.
    """

    def __init__(
        self,
        choices: list[tuple[str, Transform] | TransformStep | Pipeline] | dict[str, Transform],
        choose: str | None = None,
        from_dim: str | None = None,
        to_dim: str | None = None,
    ):
        """Initialize SwitchTransform with multiple choice transforms."""

        # Handle dict input for backward compatibility
        if isinstance(choices, dict):
            choices = list(choices.items())

        # Normalize inputs to a list of TransformStep objects
        normalized_steps: list[TransformStep] = []
        for item in choices:
            if isinstance(item, Pipeline):
                normalized_steps.append(TransformStep(item.name, item))
            elif isinstance(item, TransformStep):
                normalized_steps.append(item)
            elif isinstance(item, tuple):
                name, transform = item
                if isinstance(transform, Pipeline) and (name != transform.name):
                    raise ValueError(f"Pipeline name ({transform.name}) must match the step name ({name})")
                normalized_steps.append(TransformStep(name, transform))
            else:
                raise ValueError(f"Invalid choice type: {type(item)}")

        if not normalized_steps:
            raise ValueError("At least one choice must be provided")

        # Ensure unique choice names
        if len(normalized_steps) != len({step.name for step in normalized_steps}):
            raise ValueError("Choice names must be unique")

        self.choices: list[TransformStep] = normalized_steps
        self.transform_from_name = {step.name: step.transform for step in self.choices}
        self.from_dim = from_dim
        self.to_dim = to_dim

        # Validate provided 'choose' if given
        if choose is not None:
            if choose not in self.transform_from_name:
                raise ValueError(
                    f"Invalid choose='{choose}'. Available choices: {list(self.transform_from_name.keys())}"
                )
        self.choose = choose

        # Call parent after children are established
        super().__init__()

        # Compute input/output dims from the first choice; assert consistency later
        first_transform = self.choices[0].transform
        self.input_dims = first_transform.input_dims
        self.output_dims = first_transform.output_dims

        self._validate_composition()

    @property
    def children(self) -> list[Transform]:
        """Returns the transform objects from the choices list."""
        return [step.transform for step in self.choices]

    def _validate_composition(self):
        """Validates the composition structure (validation already done in __init__)."""
        # Additional validation could go here if needed
        pass

    def _get_selected_transform(self, **kwargs) -> Transform:
        """
        Determines which transform to use based on the 'choose' keyword argument.
        """
        choice_key = kwargs.get("choose", getattr(self, "choose", None))
        if choice_key is None:
            raise TransformError(
                f"'choose' keyword argument must be provided. Available choices: {list(self.transform_from_name.keys())}"
            )

        if choice_key not in self.transform_from_name:
            raise TransformError(
                f"Selected choice '{choice_key}' is not a valid option. Available choices: {list(self.transform_from_name.keys())}"
            )
        return self.transform_from_name[choice_key]

    @property
    def is_predictor(self) -> bool:
        """Returns True if the selected transform performs prediction."""
        selected_transform = self._get_selected_transform()
        if isinstance(selected_transform, CompositeTransform):
            return selected_transform.is_predictor
        else:
            return isinstance(selected_transform, Predictor)

    @property
    def predictive_transform(self) -> Transform | None:
        """Returns the predictive transform if it exists, otherwise None."""

        if not self.is_predictor:
            return None

        transform = self._get_selected_transform()
        if isinstance(transform, CompositeTransform):
            return transform.predictive_transform
        else:
            return transform

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """Predicts the data using the selected child transform."""
        if not self.is_predictor:
            raise ValueError("SwitchTransform is not a predictor.")
        selected_transform = self._get_selected_transform(**kwargs)
        if isinstance(selected_transform, CompositeTransform):
            return self._rename_output_dim(selected_transform.predict(container, **kwargs))
        if isinstance(selected_transform, Predictor):
            return self._rename_output_dim(selected_transform.predict(container, **kwargs))
        raise TypeError(f"Selected transform '{type(selected_transform).__name__}' is not a predictor.")

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """Predicts the probabilities using the selected child transform."""
        if not self.is_predictor:
            raise ValueError("SwitchTransform is not a predictor.")
        selected_transform = self._get_selected_transform(**kwargs)
        if isinstance(selected_transform, CompositeTransform):
            return self._rename_output_dim(selected_transform.predict_proba(container, **kwargs))
        if isinstance(selected_transform, Predictor):
            return self._rename_output_dim(selected_transform.predict_proba(container, **kwargs))
        raise TypeError(f"Selected transform '{type(selected_transform).__name__}' is not a predictor.")

    def fit(self, container: DataContainer, **kwargs) -> "SwitchTransform":
        """Fits the selected child transform."""
        selected_transform = self._get_selected_transform(**kwargs)
        selected_transform.fit(container, **kwargs)
        return self

    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Fit/transform by delegating to the selected child.

        If the selected child is stateful, call its fit_transform; otherwise,
        call its transform. This allows mixing stateful and stateless choices
        without requiring the switch wrapper itself to implement _fit.
        """
        selected_transform = self._get_selected_transform(**kwargs)
        if getattr(selected_transform, "is_stateful", False):
            result = selected_transform.fit_transform(container, **kwargs)
        else:
            result = selected_transform.transform(container, **kwargs)
        return self._rename_output_dim(result)

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Transforms the data using the selected child transform."""
        selected_transform = self._get_selected_transform(**kwargs)
        result = selected_transform.transform(container, **kwargs)
        return self._rename_output_dim(result)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions.

        For consistency, this implementation requires that all possible choices
        produce the same output dimensions for a given input. It validates
        this by checking the first choice and then asserting all others match.
        """
        if not self.choices:
            return input_dims

        # Get the expected output from the first choice as a reference
        ref_name = self.choices[0].name
        ref_transform = self.choices[0].transform
        expected_dims = self._rename_dims_tuple(ref_transform.get_expected_output_dims(input_dims))

        # Verify that all other choices produce the same output dimensions
        for step in self.choices[1:]:
            name = step.name
            current_dims = self._rename_dims_tuple(step.transform.get_expected_output_dims(input_dims))
            if current_dims != expected_dims:
                raise TransformError(
                    f"Inconsistent output dimensions in SwitchTransform. "
                    f"Choice '{ref_name}' produces {expected_dims}, but "
                    f"choice '{name}' produces {current_dims}. All choices must have "
                    "the same output dimension signature."
                )

        return expected_dims

    def _rename_dims_tuple(self, dims: tuple[str, ...]) -> tuple[str, ...]:
        if self.from_dim and self.to_dim and self.from_dim in dims:
            return tuple(self.to_dim if d == self.from_dim else d for d in dims)
        return dims

    def _rename_output_dim(self, container: DataContainer) -> DataContainer:
        if self.from_dim and self.to_dim and self.from_dim in container.data.dims:
            renamed = container.data.rename({self.from_dim: self.to_dim})
            return DataContainer(renamed)
        return container

    def __repr__(self) -> str:
        choices_str = ", ".join(
            f"'{name}': {transform.__class__.__name__}" for name, transform in self.transform_from_name.items()
        )
        return f"SwitchTransform(choices=[{choices_str}])"


class OptionalTransform(SwitchTransform):
    """
    Optionally apply a transform or skip it entirely (identity behavior).

    This is a convenience wrapper over `SwitchTransform` that defines two choices:
    - "use": apply the provided transform
    - "skip": apply `IdentityTransform` (no-op)

    You can control selection by either:
    - `use=True|False` boolean, or
    - `choose` set to either "use" or "skip".

    Note:
        For this to be valid within a statically validated pipeline, the wrapped
        transform should preserve the dimension signature. Otherwise, the two
        choices would yield different output dims and violate the validation
        requirement that choices share the same output dims.

    Args:
        transform_template: The transform or pipeline to optionally apply.
        choose: Optional explicit selection ("use" or "skip").
        use: Optional boolean shorthand for `choose`.
    """

    def __init__(
        self,
        transform_template: Transform,
        choose: str | None = None,
        use: bool | None = None,
        name: str | None = None,
        skip_name: str = "identity",
        identity_rename: dict[str, str] | None = None,
    ):
        """
        Initialize an OptionalTransform.

        Args:
            transform_template: The wrapped transform/pipeline to optionally apply.
            choose: Explicit choice label to select the branch. If provided, must
                be either the transform branch name or `skip_name`.
            use: Boolean shorthand; if provided, maps to `choose` with
                transform branch name when True and `skip_name` when False.
            name: Optional label for the transform branch. Defaults to the
                lowercased class name of `transform`.
            skip_name: Label for the identity branch. Defaults to "identity".
            identity_rename: Optional mapping of coordinate names to rename ONLY when
                the identity branch is selected, e.g., {"old_coord": "new_coord"}.
                This renames coordinates without altering dimension names.
        """
        # Store original constructor args for easy cloning
        self.transform_template = transform_template
        self.choose = choose
        self.use = use
        self.name = name
        self.skip_name = skip_name
        self.identity_rename = identity_rename

        # Prefer an explicit name; otherwise prefer the wrapped transform's own name
        # (e.g., Pipeline(name=...)); finally fall back to the class name.
        explicit_name = name
        auto_name_from_transform = getattr(transform_template, "name", None)
        transform_choice_name = (
            explicit_name or auto_name_from_transform or transform_template.__class__.__name__.lower()
        )

        if choose is None and use is not None:
            resolved_choose = transform_choice_name if use else self.skip_name
        elif choose is None and use is None:
            # Default to applying the transform when not specified explicitly
            resolved_choose = transform_choice_name
        else:
            resolved_choose = choose

        # Create identity branch - either plain IdentityTransform or with renaming
        if identity_rename:
            identity_transform = RenameDimsTransform(rename_map=identity_rename)
        else:
            identity_transform = IdentityTransform()

        super().__init__(
            choices=[(transform_choice_name, transform_template), (self.skip_name, identity_transform)],
            choose=resolved_choose,
        )

    def set_params(self, **params: Any) -> "OptionalTransform":
        # `use` is shorthand for `choose`; map it so the live object and its clone follow it
        if params.get("use") is not None and "choose" not in params:
            params = {**params, "choose": self.choices[0].name if params["use"] else self.skip_name}
        return cast("OptionalTransform", super().set_params(**params))
