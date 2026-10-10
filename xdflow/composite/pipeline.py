from typing import Any

from xdflow.composite.base import CompositeTransform, TransformStep, _configure_transform_for_inference
from xdflow.core import DataContainer, TransformError
from xdflow.core.base import Predictor, Transform
from xdflow.utils.cache_utils import cache_result, get_pipeline_cache_key_dict


class Pipeline(CompositeTransform):
    """Run named transforms in sequence.

    A pipeline is itself a transform, so it can be nested inside other
    composites or passed directly to `CrossValidator`. Each step receives the
    `DataContainer` produced by the previous step. `fit_transform` fits stateful
    steps as the data flows forward; `transform` assumes stateful steps have
    already been fitted.

    If the final step is a `Predictor`, the pipeline also exposes `predict`,
    `predict_proba`, and `get_labels`. In that case all steps before the final
    predictor are applied first, then prediction is delegated to the predictor.

    Step names must be unique. Optional `expected_input_dims` can be used to
    validate the dimensions seen by each step at runtime.
    """

    def __init__(
        self,
        name: str,
        steps: list[tuple[str, Transform]] | list[TransformStep],
        expected_input_dims: dict[str, tuple[str, ...]] = None,
        use_cache: bool = False,
    ):
        """Create a named pipeline from transform steps.

        Args:
            name: Human-readable pipeline name.
            steps: Ordered `(step_name, transform)` pairs or `TransformStep`
                objects.
            expected_input_dims: Optional mapping from each step name to the
                dimensions expected immediately before that step runs.
            use_cache: Whether to cache `fit_transform` output for reuse.
        """
        self.name = name

        if steps and not isinstance(steps[0], TransformStep):
            steps = [TransformStep(name, transform) for name, transform in steps]
        self.steps: list[TransformStep] = steps
        self.transform_from_name = {step.name: step.transform for step in steps}
        self.expected_input_dims = expected_input_dims or {}
        self.use_cache = use_cache

        # The super().__init__() call will now handle is_stateful
        super().__init__()

        # Compute input/output dimensions
        self.input_dims = self.steps[0].transform.input_dims if self.steps else ()
        self.output_dims = ()  # Dynamic based on step composition

        # Validate pipeline structure at initialization
        self._validate_composition()

    @property
    def children(self) -> list[Transform]:
        """Returns the transform objects from the steps."""
        return [step.transform for step in self.steps]

    @property
    def is_predictor(self) -> bool:
        """Returns True if the last step is a Predictor."""
        last_step = self.steps[-1]
        if isinstance(last_step.transform, CompositeTransform):
            return last_step.transform.is_predictor
        else:
            return isinstance(last_step.transform, Predictor)

    @property
    def predictive_transform(self) -> Transform | None:
        """Returns the predictive transform if it exists, otherwise None."""
        if not self.is_predictor:
            return None

        last_step = self.steps[-1]
        if isinstance(last_step.transform, CompositeTransform):
            return last_step.transform.predictive_transform
        else:
            return last_step.transform

    def _validate_composition(self):
        """Checks that the pipeline is valid."""

        # check that step names are unique
        assert len(self.steps) == len({step.name for step in self.steps}), "Step names must be unique"

        # check expected_dims correspond to steps
        if self.expected_input_dims:
            assert set(self.expected_input_dims.keys()) == {step.name for step in self.steps}, (
                "If expected_input_dims is provided, each step must have a corresponding expected dimension"
            )

        # check dimension mismatch between steps
        for i in range(len(self.steps) - 1):
            p_step1 = self.steps[i]
            p_step2 = self.steps[i + 1]
            name1, transform1 = p_step1.name, p_step1.transform
            name2, transform2 = p_step2.name, p_step2.transform

            # check automatic dimensions
            if transform1.output_dims and transform2.input_dims:
                assert transform1.output_dims == transform2.input_dims, (
                    f"Mismatch between step '{name1}' and step '{name2}'.\n"
                    f"Expected automatic input: {transform2.input_dims}, but got automatic output: {transform1.output_dims}"
                )

            # check manual dimensions
            if self.expected_input_dims:
                expected_input_dims1 = self.expected_input_dims[name1]
                expected_input_dims2 = self.expected_input_dims[name2]
                expected_output_dims1 = transform1.get_expected_output_dims(expected_input_dims1)

                if transform1.input_dims:
                    assert transform1.input_dims == expected_input_dims1, (
                        f"Step {name1} automatically requires {transform1.input_dims} but manual input is {expected_input_dims1}"
                    )

                assert expected_output_dims1 == expected_input_dims2, (
                    f"Mismatch between step '{name1}' and step '{name2}'.\n"
                    f"Expected manual input: {expected_input_dims2}, \n but got expected output: {expected_output_dims1} from {expected_input_dims1}"
                )
        # check final step
        final_step = self.steps[-1]
        name, transform = final_step.name, final_step.transform
        if self.expected_input_dims:
            if transform.input_dims:
                assert transform.input_dims == self.expected_input_dims[name], (
                    f"Step {name} automatically requires {transform.input_dims} but manual input is {self.expected_input_dims[name]}"
                )
            transform.get_expected_output_dims(self.expected_input_dims[name])  # will raise error if mismatch

    def fit(self, container: DataContainer, **kwargs) -> "Pipeline":
        """
        Fits all stateful transforms in the pipeline using recursive delegation.

        This method fits all the transformers in sequence. The data is transformed
        by each step and passed to the next. The final transformed data is discarded.
        The primary purpose is to prepare the pipeline for future transform() calls.

        Args:
            container: DataContainer to fit on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            Self (fitted pipeline)
        """
        self.fit_transform(container, **kwargs)
        # Discard the transformed data and return self for method chaining
        return self

    @cache_result(prefix="fit_transform", key_gen_func=get_pipeline_cache_key_dict)
    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Fits and transforms the data in a single, efficient pass.

        This is the preferred method when you need to both fit the pipeline
        and get the transformed training data back. It performs the exact same
        fitting logic as fit() but returns the final transformed result instead
        of discarding it.

        Args:
            container: DataContainer to fit and transform.
            **kwargs: Additional context/parameters passed through the pipeline.

        Returns:
            The transformed DataContainer.
        """

        temp_container = container
        for step in self.steps:
            try:
                # Runtime validation using the internal expected_input_dims map
                if self.expected_input_dims:
                    expected = self.expected_input_dims[step.name]
                    actual = temp_container.dims
                    if actual != expected:
                        raise RuntimeError(
                            f"Dimension mismatch before step '{step.name}': Expected {expected}, got {actual}"
                        )
                # Fit and transform each step to provide correct input for next step
                temp_container = step.transform.fit_transform(temp_container, **kwargs)

            except Exception as e:
                raise TransformError(f"Error in step '{step.name}' ({step.__class__.__name__}): {e}") from e

        return temp_container

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies all transformations in sequence using recursive delegation.

        Args:
            container: DataContainer to transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            Transformed DataContainer
        """

        temp_container = container
        for step in self.steps:
            try:
                # Runtime validation using the internal expected_input_dims map
                if self.expected_input_dims:
                    expected = self.expected_input_dims[step.name]
                    actual = temp_container.dims
                    if actual != expected:
                        raise RuntimeError(
                            f"Dimension mismatch before step '{step.name}': Expected {expected}, got {actual}"
                        )

                # Apply the transform function of this step
                temp_container = step.transform.transform(temp_container, **kwargs)

            except Exception as e:
                raise TransformError(f"Error in step '{step.name}' ({step.__class__.__name__}): {e}") from e

        return temp_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...], print_steps: bool = False) -> tuple[str, ...]:
        """Returns the expected output dimensions for the pipeline."""

        # run through steps and get expected output dims
        curr_input_dims = input_dims
        for step in self.steps:
            if print_steps:
                print(f"Step {step.name} \ninput dims: {curr_input_dims}")
            curr_input_dims = step.transform.get_expected_output_dims(curr_input_dims)
            if print_steps:
                print(f"output dims: {curr_input_dims}")

        expected_output_dims = curr_input_dims
        return expected_output_dims

    def _transform_to_final_step(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data through all steps except the last (the predictor).

        Args:
            container: DataContainer to transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer transformed up to the final step
        """

        transformed_container = container
        if len(self.steps) > 1:
            for step in self.steps[:-1]:
                transformed_container = step.transform.transform(transformed_container, **kwargs)
        return transformed_container

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Generates predictions using the final predictor in the pipeline.

        Args:
            container: DataContainer to make predictions on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with predictions as the primary data
        """
        if not self.steps:
            raise ValueError("Cannot predict with an empty pipeline.")

        final_step = self.steps[-1]
        if not self.is_predictor:
            raise TypeError("The last step of the pipeline must be a Predictor for prediction.")

        transformed_container = self._transform_to_final_step(container, **kwargs)
        return final_step.transform.predict(transformed_container, **kwargs)

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Generates prediction probabilities using the final Predictor in the pipeline.

        Args:
            container: DataContainer to make predictions on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with prediction probabilities
        """
        if not self.steps:
            raise ValueError("Cannot predict with an empty pipeline.")

        final_step = self.steps[-1]
        if not self.is_predictor:
            raise TypeError("The last step of the pipeline must be a Predictor for prediction.")

        transformed_container = self._transform_to_final_step(container, **kwargs)
        return final_step.transform.predict_proba(transformed_container, **kwargs)

    def prepare_for_inference(self, *, set_n_jobs_single: bool = True) -> None:
        """
        Disable training-time optimizations that are undesirable at inference.

        Args:
            set_n_jobs_single: When True, force transforms that expose an `n_jobs`
                attribute to run single-threaded for request/response latency.
        """
        self.use_cache = False

        if set_n_jobs_single and hasattr(self, "n_jobs"):
            try:
                self.n_jobs = 1
            except AttributeError:
                pass

        visited: set[int] = {id(self)}
        for step in self.steps:
            _configure_transform_for_inference(
                step.transform,
                set_n_jobs_single=set_n_jobs_single,
                visited=visited,
            )

    def __repr__(self) -> str:
        steps_str = ", ".join([f"'{p_step.name}': {p_step.transform.__class__.__name__}" for p_step in self.steps])
        return f"Pipeline(name='{self.name}', steps=[{steps_str}])"

    @property
    def final_target_coord(self) -> str | None:
        """Convenience: expose the final predictor's target coordinate, if any.

        Returns:
            The `target_coord` of the last step when it is a `Predictor`, else None.
        """
        if (not self.steps) or (not self.is_predictor):
            return None
        return getattr(self.predictive_transform, "target_coord", None)

    def get_labels(self) -> list[Any]:
        """
        Return the label ordering from the final predictor.

        Relies on the predictor implementing `get_labels`; raises when the pipeline
        cannot provide labels unambiguously.
        """
        if not self.steps:
            raise ValueError("Cannot get labels from an empty pipeline.")

        if not self.is_predictor:
            raise TypeError("Pipeline does not terminate in a Predictor, so labels are undefined.")

        predictive_transform = self.predictive_transform
        if predictive_transform is None:
            raise RuntimeError("Pipeline.predictive_transform resolved to None; cannot determine labels.")

        labels = predictive_transform.get_labels()
        if labels is None:
            raise ValueError(
                f"Predictive transform {predictive_transform.__class__.__name__} returned no labels. "
                "Ensure it is fitted and exposes label metadata."
            )

        return labels


def make_pipeline(*steps: Transform, name: str = "pipeline", use_cache: bool = False) -> Pipeline:
    """Build a ``Pipeline`` from transforms, naming each step after its class (like scikit-learn's).

    Step names are the lowercased class names without a ``Transform`` suffix (``"zscore"`` for
    ``ZScoreTransform``), the estimator name for scikit-learn wrappers (``"logisticregression"``), or the
    function name for ``@step`` functions, with ``-1``, ``-2``... appended when a name repeats. Use them in
    parameter paths such as ``"zscore__per_dim"``.

    Example:
        >>> pipeline = make_pipeline(BandpassFilterTransform(8, 30, fs=160), CSPTransform(), predictor)
    """
    for transform in steps:
        if isinstance(transform, type):
            raise TypeError(
                f"make_pipeline got the class {transform.__name__}; pass an instance: {transform.__name__}()."
            )
        if not isinstance(transform, Transform):
            raise TypeError(
                f"make_pipeline got {type(transform).__name__}, which is not an xdflow Transform. Wrap scikit-learn "
                "estimators with SKLearnPredictor.from_estimator(...) or SKLearnTransformer(...)."
            )
    # Wrapped scikit-learn estimators are named after the estimator ("logisticregression"), not the wrapper.
    base_names = [_step_name(transform) for transform in steps]
    counts = {base: base_names.count(base) for base in base_names}
    seen: dict[str, int] = {}
    named = []
    for base, transform in zip(base_names, steps):
        if counts[base] > 1:
            seen[base] = seen.get(base, 0) + 1
            named.append((f"{base}-{seen[base]}", transform))
        else:
            named.append((base, transform))
    return Pipeline(name, named, use_cache=use_cache)


def _step_name(transform: Transform) -> str:
    """Lowercased class name without a "Transform" suffix; scikit-learn wrappers use the estimator's name."""
    estimator = getattr(transform, "_base_estimator_cls", None) or getattr(transform, "estimator_cls", None)
    name = str(getattr(estimator, "__name__", type(transform).__name__)).lower()
    return name.removesuffix("transform") or name
