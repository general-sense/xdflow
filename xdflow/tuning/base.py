"""
Hyperparameter tuning framework using Optuna.
"""

import copy

# Import for setting global random seeds
import random
import time
from typing import Any, cast

import numpy as np
import optuna
from optuna.samplers import BaseSampler, TPESampler

# Optional mlflow integration
try:
    import mlflow
    from optuna.integration.mlflow import MLflowCallback

    HAS_MLFLOW = True
except (ImportError, ModuleNotFoundError):
    HAS_MLFLOW = False
    mlflow = None
    MLflowCallback = None

from xdflow.composite import Pipeline, SwitchTransform
from xdflow.composite.base import CompositeTransform
from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.cv.base import CrossValidator
from xdflow.transforms.sklearn_transform import SKLearnTransform
from xdflow.utils.cache_utils import _get_object_metadata, _get_object_params, hash_dict

# Try to import torch for PyTorch seed setting
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Tuner:
    """
    Orchestrates hyperparameter tuning for pipelines and cross-validators.

    The Tuner class orchestrates hyperparameter optimization using a
    "configuration-driven" approach. Users specify which pipelines and
    parameters to test, and the Tuner translates this into an Optuna study.

    The underlying CrossValidator automatically optimizes pipeline execution
    by detecting and separating stateless/stateful components, providing both
    a simple API and efficient execution.

    Features:
    - High-level wrapper around Optuna optimization
    - Support for multiple pipeline architectures
    - Flexible parameter space specification
    - Integration with CrossValidator for evaluation
    - Automatic pipeline optimization via CrossValidator
    - Comprehensive random seed management for reproducibility
    """

    def __init__(
        self,
        pipelines_to_tune: Pipeline | list[Pipeline],
        cv_strategy: CrossValidator,
        param_grid: dict[str, dict[str, dict[str, Any]]],
        initial_data_container: DataContainer,
        sampler: BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        direction: str = "maximize",
        verbose: int = 1,
        use_mlflow: bool = False,
        mlflow_experiment_name: str = "",
        mlflow_metadata: dict[str, Any] | None = None,
        log_artifacts: bool = True,
        random_seed: int = 0,
        use_cache: bool = True,
        verbose_transforms: bool = False,
        exclude_intertrial_from_scoring: bool | None = None,
    ):
        """
        Initialize Tuner with pipelines and search space.

        Args:
            pipelines_to_tune: List of Pipeline objects to compare/tune.
                              Initial parameter values from these instances will be used for the initial trials.
                              Parameter values will be kept if not overwritten by the param_grid.
            cv_strategy: CrossValidator instance for evaluation
            param_grid: Nested dict defining parameter search spaces
                       Format: {pipeline_name: {step_name: {param_name: space}}}
            initial_data_container: DataContainer with data for optimization
            sampler: Optuna sampler for hyperparameter optimization
            direction: Direction to optimize ("maximize" or "minimize")
            verbose: Verbosity level (0, 1, or 2)
            use_mlflow: Whether to use MLflow for experiment tracking
            mlflow_experiment_name: Name of the MLflow experiment
            mlflow_metadata: Additional metadata to log to MLflow
            random_seed: Global random seed for reproducibility.
            use_cache: Whether to use caching for the static part of the pipeline.
            verbose_transforms: Whether to enable verbose logging in transforms during tuning
            exclude_intertrial_from_scoring: Optional toggle that, when set, overrides the validator's
                exclude_intertrial_from_scoring flag so tuning scores ignore synthetic blanks.
        """
        if isinstance(pipelines_to_tune, Pipeline):
            pipelines_to_tune = [pipelines_to_tune]

        self.pipelines_to_tune = {p.name: p for p in pipelines_to_tune}
        self.cv_strategy = cv_strategy
        self.param_grid = param_grid
        self.initial_data = initial_data_container
        self.sampler = sampler or TPESampler(seed=random_seed)
        # Disable pruning by default; users can still supply a custom pruner
        self.pruner = pruner or optuna.pruners.NopPruner()
        self.direction = direction
        self.use_mlflow = use_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_metadata = mlflow_metadata or {}
        self.log_artifacts = log_artifacts
        self.verbose = verbose
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.verbose_transforms = verbose_transforms
        self.exclude_intertrial_from_scoring = exclude_intertrial_from_scoring

        if self.use_cache:
            for pipeline in self.pipelines_to_tune.values():
                pipeline.use_cache = True

        if self.exclude_intertrial_from_scoring is not None:
            if not hasattr(self.cv_strategy, "exclude_intertrial_from_scoring"):
                raise AttributeError(
                    f"{type(self.cv_strategy).__name__} does not expose 'exclude_intertrial_from_scoring'."
                )
            self.cv_strategy.exclude_intertrial_from_scoring = self.exclude_intertrial_from_scoring

        self._validate_param_grid()

        # MLflow run tracking
        self.current_run_id = None

        # set random seeds
        self._set_random_seeds()

        # set up mlflow
        if self.use_mlflow:
            self._setup_mlflow()

        if self.verbose == 0:
            self.set_low_verbosity()

    def _set_random_seeds(self):
        """
        Set random seeds for reproducibility.
        """
        if self.random_seed is not None:
            # Set global random seeds
            self._set_global_random_seeds()

            # Inject seeds into all pipeline components
            self._inject_random_seeds_into_pipelines()

            # Inject seed into cross-validator
            if hasattr(self.cv_strategy, "random_state"):
                cast(Any, self.cv_strategy).random_state = self.random_seed

    def _validate_param_grid(self):
        """
        Check the param grid is valid.
        """

        # Check each item in param grid is valid
        for pipeline_name, pipeline_params_grid in self.param_grid.items():
            self._validate_grid(self.pipelines_to_tune[pipeline_name], pipeline_params_grid, pipeline_name)

        # Check param grids exist for each pipeline (not mandatory, but raise
        # warning if not)
        pipeline_names = [pipeline.name for pipeline in self.pipelines_to_tune.values()]
        for pipeline_name in pipeline_names:
            if pipeline_name not in self.param_grid:
                print(f"Warning: No param grid found for pipeline {pipeline_name}")

    def _validate_grid(
        self, composite_transform: CompositeTransform, param_grid: dict[str, Any], composite_transform_name: str
    ):
        transform_names = composite_transform.transform_from_name.keys()

        for param_name, param_values in param_grid.items():
            assert param_name in transform_names, (
                f"Transform {param_name} in param grid not found in {composite_transform_name}"
            )
            transform = composite_transform.get_transform_from_name(param_name)

            # Handle SwitchTransform with conditional parameters
            if self._is_switch_transform_with_conditionals(transform, param_values):
                self._validate_switch_conditional_grid(transform, param_values, param_name)
            elif isinstance(transform, CompositeTransform):
                self._validate_grid(transform, param_values, param_name)
            else:
                self._validate_param_space_dict(param_values, f"{composite_transform_name}__{param_name}")

    def _validate_param_space_dict(self, param_config: dict[str, Any], namespace: str) -> None:
        """
        Ensure every parameter entry in a leaf transform has a supported space definition.

        Args:
            param_config: Dict mapping parameter names to space specs
            namespace: Fully-qualified name for error reporting
        """

        def _validate_param_space(full_name: str, space: Any) -> None:
            """
            Validate a single parameter space using the same rules as _suggest_param.

            Args:
                full_name: Fully-qualified parameter name (for error messages)
                space: Parameter space specification
            """
            if isinstance(space, tuple):
                if len(space) == 2 and all(isinstance(v, int) for v in space):
                    return
                if len(space) == 2 and any(isinstance(v, float) for v in space):
                    return
                if len(space) == 3 and space[0] == "log":
                    return
            elif isinstance(space, list):
                return

            raise ValueError(f"Unsupported parameter space for {full_name}: {space}")

        if not isinstance(param_config, dict):
            raise ValueError(f"Parameter grid for {namespace} must be a dict, got {type(param_config)}")

        for param_name, param_space in param_config.items():
            if isinstance(param_space, dict):
                raise ValueError(f"Unsupported nested parameter space for {namespace}__{param_name}: {param_space}")
            _validate_param_space(f"{namespace}__{param_name}", param_space)

    @staticmethod
    def set_low_verbosity():
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _setup_mlflow(self):
        """Set up MLflow."""
        if not HAS_MLFLOW:
            raise ImportError(
                "MLflow support requires additional dependencies. "
                "Install with: pip install xdflow[mlflow] or pip install mlflow optuna-integration[mlflow]"
            )

        # Default to SQLite backend if no tracking URI is configured
        if mlflow.get_tracking_uri().startswith("file:") or mlflow.get_tracking_uri().startswith("./"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

        # set up experiment
        today_date = time.strftime("%Y-%m-%d")
        experiment_name = f"{self.mlflow_experiment_name} {today_date}"
        mlflow.set_experiment(experiment_name)

        # Set callback to None initially - will be created when needed
        self.mlflow_callbacks = None

    @staticmethod
    def _create_mlflow_callback():
        """Create MLflow callback when parent run is already active."""
        if not HAS_MLFLOW:
            raise ImportError(
                "MLflow support requires additional dependencies. "
                "Install with: pip install xdflow[mlflow] or pip install mlflow optuna-integration[mlflow]"
            )
        return MLflowCallback(
            tracking_uri=None,
            metric_name="f1_score",
            create_experiment=False,
            tag_study_user_attrs=False,
            tag_trial_user_attrs=True,
            mlflow_kwargs={"nested": True},
        )

    def _log_initial_params(self):
        """Log initial parameters and metadata to MLflow."""
        if not self.use_mlflow:
            return

        # Log tuning configuration
        mlflow.log_param("tuning__param_grid", str(self.param_grid))
        mlflow.log_param("tuning__cv_strategy", self.cv_strategy.__class__.__name__)
        mlflow.log_param("tuning__cv_strategy_params", _get_object_params(self.cv_strategy))
        mlflow.log_param("tuning__random_seed", self.random_seed)

        # Log starting pipeline params
        pipeline_params = {}
        for pipeline_name, pipeline in self.pipelines_to_tune.items():
            pipeline_params[pipeline_name] = _get_object_params(pipeline)
        mlflow.log_param("tuning__pipelines", pipeline_params)

        # Log data
        mlflow.log_param("data__hash", hash_dict(_get_object_metadata(self.initial_data)))

        # Log any additional metadata
        for key, value in self.mlflow_metadata.items():
            mlflow.log_param(key, value)

    def _set_global_random_seeds(self) -> None:
        """
        Set global random seeds for reproducibility across all libraries.

        Args:
            seed: Random seed to set globally
        """
        seed = self.random_seed

        # Set Python's built-in random module
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seeds if available
        if HAS_TORCH:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _inject_random_seeds_into_pipelines(self):
        """
        Inject random seeds into pipeline steps that support randomness.

        This method identifies transforms that have random_state parameters and sets them.
        It modifies self.pipelines_to_tune in place.
        """

        def _inject_random_seeds(transform: Any, random_seed: int):
            """Recursively inject seeds into a transform and its children."""
            # First, set the seed on the current transform if it's supported
            if hasattr(transform, "random_state"):
                transform.random_state = random_seed
            elif hasattr(transform, "seed"):
                transform.seed = random_seed

            # If it's a SKLearnTransform, seed the model through set_params: that looks through an
            # automatic MultiOutput wrapper (which has no random_state of its own), and
            # `estimator__random_state` reaches the model inside an explicit MultiOutput estimator_cls.
            if isinstance(transform, SKLearnTransform):
                params = transform.get_params(deep=True)
                for name in ("random_state", "seed", "estimator__random_state", "estimator__seed"):
                    if name in params:
                        transform.set_params(**{name: random_seed})
                        break

            # Then, if it's a composite, recurse into its children
            if isinstance(transform, CompositeTransform):
                children = list(transform.children)
                # GroupApplyTransform's children exist only after fit; seed the template its groups are cloned from
                template = getattr(transform, "transform_template", None)
                if isinstance(template, Transform) and not any(child is template for child in children):
                    children.append(template)
                for child_transform in children:
                    _inject_random_seeds(child_transform, random_seed)

        for pipeline in self.pipelines_to_tune.values():
            _inject_random_seeds(pipeline, self.random_seed)

    def _objective(self, trial: optuna.Trial) -> float:
        """
        The objective function called by Optuna for each trial.

        This method works with single, complete pipelines. The CrossValidator
        automatically handles the optimization of stateless/stateful execution.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (cross-validation score) to maximize/minimize

        Raises:
            optuna.TrialPruned: If the trial fails due to numerical errors or other issues
        """
        try:
            # 1. Suggest which pipeline to use for this trial
            pipeline_name = trial.suggest_categorical("pipeline", list(self.pipelines_to_tune.keys()))

            if self.use_cache:
                pipeline_template, initial_data = self.prepare_pipeline_with_caching(
                    self.pipelines_to_tune[pipeline_name]
                )
                print("pipeline_template", pipeline_template)
            else:
                pipeline_template = self.pipelines_to_tune[pipeline_name]
                initial_data = self.initial_data

            trial_pipeline = pipeline_template.clone()

            # 2. Suggest hyperparameters for the chosen pipeline
            # Only process steps and parameters that are specified in the
            # param_grid
            pipeline_param_grid = self.param_grid.get(pipeline_name, {})
            self._suggest_and_set_params(
                composite_transform=trial_pipeline,
                param_grid=pipeline_param_grid,
                namespace_prefix=pipeline_name,
                trial=trial,
            )

            # 3. Use the fixed CrossValidator strategy with the complete pipeline
            trial_validator = copy.deepcopy(self.cv_strategy)
            # Assign the complete tuned pipeline
            trial_validator.set_pipeline(trial_pipeline)

            # 4. Run the cross-validation - CrossValidator handles stateless/stateful optimization automatically.
            # Each fold score is reported so the configured pruner can stop unpromising trials.
            def pruning_callback(step_idx: int, fold_score: float) -> None:
                trial.report(fold_score, step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at fold {step_idx} (score {fold_score:.4f})")

            score = trial_validator.cross_validate(
                initial_data, verbose=self.verbose_transforms, pruning_callback=pruning_callback
            )
            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Check if this is a numerical error we can recover from
            error_msg = str(e).lower()
            is_numerical_error = isinstance(
                e, (np.linalg.LinAlgError, ValueError, RuntimeError, TransformError)
            ) and any(
                keyword in error_msg
                for keyword in ["svd", "convergence", "singular", "decomposition", "linalg", "not converge"]
            )

            if is_numerical_error:
                # Prune trial for expected numerical failures (SVD convergence, etc.)
                if self.verbose >= 1:
                    print(f"Trial {trial.number} failed due to numerical error: {e}")
                    print("Pruning trial and continuing optimization...")
                # Recorded on the trial so `tune()` can report it if no trial completes.
                trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                raise optuna.TrialPruned(f"Numerical error: {e}") from e
            else:
                # Re-raise all other errors (unexpected errors or non-numerical errors)
                if self.verbose >= 1:
                    print(f"Trial {trial.number} failed with error: {type(e).__name__}: {e}")
                raise

    def _suggest_and_set_params(
        self,
        composite_transform: CompositeTransform,
        param_grid: dict[str, Any],
        namespace_prefix: str,
        trial: optuna.Trial,
    ):
        """
        Suggest and set parameters for a composite transform based on param_grid structure.
        Only processes steps that are actually specified in the param_grid.

        Supports conditional parameters for SwitchTransform via 'conditional_params' key.

        Args:
            composite_transform: Pipeline, TransformUnion, or SwitchTransform to set parameters on
            param_grid: Parameter grid specifying which parameters to tune for which steps
            namespace_prefix: Prefix for parameter names in Optuna (e.g., "pipeline_name")
            trial: Optuna trial object for parameter suggestion
        """
        for step_name, step_param_grid in param_grid.items():
            target_transform = composite_transform.get_transform_from_name(step_name)
            if target_transform is None:
                print(f"Warning: Step '{step_name}' not found in {namespace_prefix}")
                continue

            # Check if this is a SwitchTransform with conditional parameters
            if self._is_switch_transform_with_conditionals(target_transform, step_param_grid):
                self._handle_switch_transform_conditionals(
                    target_transform, step_param_grid, namespace_prefix, step_name, trial
                )
            elif isinstance(target_transform, CompositeTransform):
                nested_namespace = f"{namespace_prefix}__{step_name}"
                self._suggest_and_set_params(target_transform, step_param_grid, nested_namespace, trial)
            else:
                params_to_set = {}
                for param_name, param_space in step_param_grid.items():
                    optuna_param_name = f"{namespace_prefix}__{step_name}__{param_name}"
                    suggested_value = self._suggest_param(trial, optuna_param_name, param_space)
                    params_to_set[param_name] = suggested_value
                target_transform.set_params(**params_to_set)

        # Second pass: ensure all SwitchTransforms (even those not in param_grid)
        # have a 'choose' suggested so each trial explores a branch.
        self._suggest_choose_for_all_switches(
            composite_transform=composite_transform,
            namespace_prefix=namespace_prefix,
            trial=trial,
            processed_step_names=set(param_grid.keys()),
        )

    def _suggest_choose_for_all_switches(
        self,
        composite_transform: CompositeTransform,
        namespace_prefix: str,
        trial: optuna.Trial,
        processed_step_names: set,
    ) -> None:
        """Walk the composite and suggest 'choose' for any SwitchTransform not in param_grid.

        This guarantees Switch/Optional branches are selected per trial even when
        not explicitly present in the grid.
        """
        # Iterate available named children
        for child_name, child in composite_transform.transform_from_name.items():
            if isinstance(child, SwitchTransform):
                if child_name not in processed_step_names:
                    # Suggest a choice without any choice-specific params
                    choose_param_name = f"{namespace_prefix}__{child_name}__choose"
                    selected_choice = trial.suggest_categorical(
                        choose_param_name, list(child.transform_from_name.keys())
                    )
                    child.set_params(choose=selected_choice)

                    # If the selected choice is itself a composite, recurse to allow nested switches
                    chosen_transform = child.get_transform_from_name(selected_choice)
                    if isinstance(chosen_transform, CompositeTransform):
                        nested_namespace = f"{namespace_prefix}__{child_name}__{selected_choice}"
                        self._suggest_choose_for_all_switches(
                            composite_transform=chosen_transform,
                            namespace_prefix=nested_namespace,
                            trial=trial,
                            processed_step_names=set(),
                        )
            elif isinstance(child, CompositeTransform):
                nested_namespace = f"{namespace_prefix}__{child_name}"
                self._suggest_choose_for_all_switches(
                    composite_transform=child,
                    namespace_prefix=nested_namespace,
                    trial=trial,
                    processed_step_names=set(),
                )

    def _is_switch_transform_with_conditionals(self, transform: Any, param_grid: dict[str, Any]) -> bool:
        """
        Check if this is a SwitchTransform we should handle specially.

        We always handle SwitchTransform here to allow suggesting the 'choose'
        parameter even when some choices have no parameters in the grid. Choice-
        specific parameters will only be set for choices that appear in the grid.

        Args:
            transform: The transform to check
            param_grid: The parameter grid for this transform (unused for detection)

        Returns:
            True if this is a SwitchTransform
        """
        return isinstance(transform, SwitchTransform)

    def _handle_switch_transform_conditionals(
        self,
        switch_transform: SwitchTransform,
        step_param_grid: dict[str, Any],
        namespace_prefix: str,
        step_name: str,
        trial: optuna.Trial,
    ):
        """
        Handle conditional parameter tuning for SwitchTransform.

        Args:
            switch_transform: The SwitchTransform instance
            step_param_grid: Parameter grid for this switch step
            namespace_prefix: Namespace prefix for parameter names
            step_name: Name of the switch step
            trial: Optuna trial object
        """
        params_to_set = {}

        # Determine choice candidates: always allow all available choices. If the
        # selected choice has parameters in the grid, we will suggest them below;
        # otherwise, the constructor/default params are used.
        available_choices = list(switch_transform.transform_from_name.keys())
        choice_candidates = available_choices

        implicit_use_choice = self._implicit_switch_choice(switch_transform, step_param_grid)

        # 1. Always suggest the 'choose' parameter from the available choices
        choose_param_name = f"{namespace_prefix}__{step_name}__choose"
        if implicit_use_choice is not None:
            # Params given at the switch level target the non-identity branch; record the choice actually
            # evaluated so the best trial's params rebuild the same pipeline
            selected_choice = trial.suggest_categorical(choose_param_name, [implicit_use_choice])
        else:
            selected_choice = trial.suggest_categorical(choose_param_name, choice_candidates)
        params_to_set["choose"] = selected_choice

        # 2. Handle any non-choice parameters in the param grid (if any)
        for param_name, param_space in step_param_grid.items():
            if param_name in available_choices or implicit_use_choice is not None:
                continue  # Choice parameters (and implicit branch params) are applied to the branch below
            # If this is a nested dict (likely composite child params), skip here;
            # they will be applied when handling choice_param_grid below.
            if isinstance(param_space, dict):
                continue

            optuna_param_name = f"{namespace_prefix}__{step_name}__{param_name}"
            suggested_value = self._suggest_param(trial, optuna_param_name, param_space)
            params_to_set[param_name] = suggested_value

        # Set the basic parameters (including auto-suggested 'choose')
        switch_transform.set_params(**params_to_set)

        # 3. Apply conditional parameters to the selected choice transform
        if selected_choice in step_param_grid:
            choice_param_grid = step_param_grid[selected_choice]
        elif implicit_use_choice is not None and selected_choice == implicit_use_choice:
            # Treat non-choice keys as the grid for the non-identity branch
            choice_param_grid = {k: v for k, v in step_param_grid.items() if k not in available_choices}
        else:
            choice_param_grid = None

        if choice_param_grid:
            choice_transform = switch_transform.get_transform_from_name(selected_choice)
            # If the selected choice is itself a composite, recurse into it so nested params apply
            if isinstance(choice_transform, CompositeTransform):
                nested_namespace = f"{namespace_prefix}__{step_name}__{selected_choice}"
                self._suggest_and_set_params(
                    composite_transform=choice_transform,
                    param_grid=choice_param_grid,
                    namespace_prefix=nested_namespace,
                    trial=trial,
                )
            else:
                choice_params_to_set = {}
                for param_name, param_space in choice_param_grid.items():
                    optuna_param_name = f"{namespace_prefix}__{step_name}__{selected_choice}__{param_name}"
                    suggested_value = self._suggest_param(trial, optuna_param_name, param_space)
                    choice_params_to_set[param_name] = suggested_value
                choice_transform.set_params(**choice_params_to_set)

    def _validate_switch_conditional_grid(
        self, switch_transform: SwitchTransform, param_grid: dict[str, Any], switch_name: str
    ):
        """
        Validate conditional parameters for a SwitchTransform.

        Args:
            switch_transform: The SwitchTransform instance
            param_grid: Parameter grid for this switch
            switch_name: Name of the switch for error messages
        """
        # Find choice parameters in the param grid
        choice_names = set(switch_transform.transform_from_name.keys())
        choice_params_in_grid = choice_names.intersection(param_grid.keys())

        # Validate that all choice parameter keys are valid choice names
        for choice_name in choice_params_in_grid:
            if choice_name not in choice_names:
                raise ValueError(
                    f"Parameter choice '{choice_name}' not found in SwitchTransform '{switch_name}'. "
                    f"Available choices: {list(choice_names)}"
                )

        # Note: We no longer validate a manual 'choose' parameter since choices are auto-detected

    @staticmethod
    def _implicit_switch_choice(switch_transform: SwitchTransform, step_param_grid: dict[str, Any]) -> str | None:
        """The non-identity branch of a two-way (identity / transform) switch whose grid gives params at the
        switch level; those params tune that branch, which every trial then uses. None otherwise."""
        available_choices = list(switch_transform.transform_from_name.keys())
        non_choice_keys = [k for k in step_param_grid if k not in available_choices]
        if non_choice_keys and "identity" in available_choices and len(available_choices) == 2:
            return next(c for c in available_choices if c != "identity")
        return None

    def _get_initial_params_for_switch(
        self,
        switch_transform: SwitchTransform,
        step_param_grid: dict[str, Any],
        namespace_prefix: str,
        step_name: str,
        initial_params: dict[str, Any],
    ):
        """
        Get initial parameters for a SwitchTransform with conditional parameters.

        Args:
            switch_transform: The SwitchTransform instance
            step_param_grid: Parameter grid for this switch step
            namespace_prefix: Namespace prefix for parameter names
            step_name: Name of the switch step
            initial_params: Dictionary to add initial parameters to
        """
        # Use the transform's configured 'choose' as the initial trial seed,
        # so the first trial evaluates the provided default choice. Subsequent
        # trials will still explore choices via suggest_categorical.

        # With implicit branch params every trial uses that branch, so the baseline must too
        initial_choice = self._implicit_switch_choice(switch_transform, step_param_grid) or getattr(
            switch_transform, "choose", None
        )
        if initial_choice is None:
            return

        available_choices = list(switch_transform.transform_from_name.keys())
        if initial_choice not in available_choices:
            return

        choose_param_name = f"{namespace_prefix}__{step_name}__choose"
        initial_params[choose_param_name] = initial_choice

    def _suggest_param(self, trial: optuna.Trial, name: str, space: Any) -> Any:
        """
        (Private) Translates a simple parameter space config into an Optuna call.

        Args:
            trial: Optuna trial object
            name: Parameter name for Optuna
            space: Parameter space specification

        Returns:
            Suggested parameter value

        Raises:
            ValueError: If parameter space format is unsupported
        """
        if isinstance(space, tuple):
            if len(space) == 2 and all(isinstance(v, int) for v in space):
                # Integer range: (min, max)
                return trial.suggest_int(name, space[0], space[1])
            if len(space) == 2 and any(isinstance(v, float) for v in space):
                # Float range: (min, max)
                return trial.suggest_float(name, space[0], space[1])
            if len(space) == 3 and space[0] == "log":
                # Log scale float: ('log', min, max)
                return trial.suggest_float(name, space[1], space[2], log=True)
        if isinstance(space, list):
            # Categorical choices
            return trial.suggest_categorical(name, space)

        raise ValueError(f"Unsupported parameter space for {name}: {space}")

    def tune(
        self,
        n_trials: int = 50,
        show_progress_bar: bool = False,
        run_name: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """
        Runs the hyperparameter tuning study.

        The first trial will always use the original parameter values from the
        pipeline instances as the starting point, ensuring that the optimization
        can evaluate the baseline configuration.

        Args:
            n_trials: Number of optimization trials to run
            show_progress_bar: Whether to show progress bar during optimization
            run_name: Custom name for the MLflow run

        Returns:
            Tuple of (best_parameters, best_score)
        """

        run_name = f"seed_{self.random_seed}__{run_name}" if run_name else f"seed_{self.random_seed}"

        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name) as parent_run:
                self._log_initial_params()

                # Create callback after parent run is established
                mlflow_callback = self._create_mlflow_callback()
                self._tune(n_trials, show_progress_bar, callbacks=[mlflow_callback])
                self.current_run_id = parent_run.info.run_id
        else:
            self._tune(n_trials, show_progress_bar, callbacks=None)

        # Optuna's own "No trials are completed yet" hides why every trial was pruned, so name the errors.
        if not self.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            errors = sorted({t.user_attrs["error"] for t in self.study.trials if "error" in t.user_attrs})
            raise ValueError("No tuning trial completed." + "".join(f"\n  {error}" for error in errors))

        best_params = self.study.best_params
        best_score = self.study.best_value

        print("Best Parameters:", best_params)
        print("Best Score:", best_score)

        if self.use_mlflow:
            # Log additional metrics if there's an active run
            try:
                with mlflow.start_run(run_id=self.current_run_id):
                    mlflow.log_params(best_params)
                    mlflow.log_metric("best_val_score", best_score)
            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")

        return best_params, best_score

    def _tune(self, n_trials: int, show_progress_bar: bool, callbacks: list[Any] | None = None):
        """Run tuning."""
        self.study = optuna.create_study(direction=self.direction, sampler=self.sampler, pruner=self.pruner)

        # Enqueue initial trials with original parameter values
        self._enqueue_initial_trials()

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=1,  # if we change this, we have to be careful about mlflow callbacks
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
        )

    def prepare_pipeline_with_caching(self, pipeline: Pipeline) -> tuple[Pipeline, DataContainer]:
        """
        Prepares for tuning by splitting pipelines and caching static parts.
        """
        pipeline_param_grid = self.param_grid.get(pipeline.name, {})
        split_index = self._find_split_index(pipeline, pipeline_param_grid)

        if split_index > 0:
            static_pipeline = Pipeline(
                name=f"{pipeline.name}_static", steps=pipeline.steps[:split_index], use_cache=True
            )
            # Execute the static part and cache the data
            data = static_pipeline.fit_transform(self.initial_data, verbose=self.verbose_transforms)
        else:
            # No static part, use initial data
            data = self.initial_data

        # The dynamic part of the pipeline
        dynamic_pipeline = Pipeline(name=f"{pipeline.name}_dynamic", steps=pipeline.steps[split_index:])
        # Mark this pipeline as the post-cache pipeline for clearer logging downstream
        dynamic_pipeline.is_post_cache_pipeline = True
        try:
            dynamic_pipeline.cached_prefix_step_names = [
                (step.name if hasattr(step, "name") else step[0]) for step in pipeline.steps[:split_index]
            ]
        except Exception:
            dynamic_pipeline.cached_prefix_step_names = []
        dynamic_pipeline.origin_pipeline_name = pipeline.name

        return dynamic_pipeline, data

    def _find_split_index(self, pipeline, param_grid):
        """
        Finds the index of the first step in the pipeline that has tunable parameters within stateless part
        """
        for i, step in enumerate(pipeline.steps):
            # Handle both real Pipeline TransformStep objects and test doubles
            name = step.name if hasattr(step, "name") else step[0]
            transform = step.transform if hasattr(step, "transform") else step[1]
            # Always treat Switch/Optional transforms as dynamic so tuner can decide per trial
            if isinstance(transform, SwitchTransform):
                return i
            if (getattr(transform, "is_stateful", False)) or (name in param_grid):
                return i

        raise ValueError(f"No tunable parameters found in pipeline {pipeline.name} and all steps are stateless")

    def _enqueue_initial_trials(self):
        """
        Enqueues initial trials with the original parameter values from each pipeline instance.
        Only enqueues parameters that are valid within the defined parameter grid.
        """

        def _is_value_in_space(value: Any, space: Any) -> bool:
            """Checks if a given value is valid within a given search space."""
            if value is None:
                return False
            if isinstance(space, list):  # Categorical
                return value in space
            if isinstance(space, tuple):
                # Handle log-scale ranges: ('log', min, max)
                if len(space) == 3 and space[0] == "log":
                    low, high = space[1], space[2]
                    if not isinstance(value, (int, float)):
                        return False
                    return float(low) <= float(value) <= float(high)

                # This check is designed to handle tuples of numbers, like ranges.
                # It's not intended for tuples of strings or other types.
                if not all(isinstance(v, (int, float)) for v in space[:2]):
                    return False

                low, high = space[:2]

                # Promote to float comparison if either bound or the value
                # itself is a float.
                is_float_comparison = any(isinstance(v, float) for v in space) or isinstance(value, float)

                if is_float_comparison:
                    return float(low) <= float(value) <= float(high)
                # Otherwise, it's an integer comparison.
                return int(low) <= int(value) <= int(high)

            return False

        def _get_initial_params(
            composite_transform: CompositeTransform,
            param_grid: dict[str, Any],
            namespace_prefix: str,
            initial_params: dict[str, Any],
        ):
            """
            Recursively get initial parameters for a composite transform, but only if they are
            valid within the defined parameter grid.
            """
            for step_name, step_param_grid in param_grid.items():
                target_transform = composite_transform.get_transform_from_name(step_name)

                # Handle SwitchTransform with conditional parameters
                if self._is_switch_transform_with_conditionals(target_transform, step_param_grid):
                    self._get_initial_params_for_switch(
                        target_transform, step_param_grid, namespace_prefix, step_name, initial_params
                    )

                    # Also seed the selected branch's own params, so the baseline trial is the configured pipeline.
                    implicit_choice = self._implicit_switch_choice(target_transform, step_param_grid)
                    if implicit_choice is not None:
                        initial_choice = implicit_choice
                        branch_names = target_transform.transform_from_name.keys()
                        choice_grid = {k: v for k, v in step_param_grid.items() if k not in branch_names}
                    else:
                        initial_choice = getattr(target_transform, "choose", None)
                        if initial_choice not in step_param_grid:
                            continue
                        choice_grid = step_param_grid[initial_choice]
                    choice_transform = target_transform.get_transform_from_name(initial_choice)
                    choice_namespace = f"{namespace_prefix}__{step_name}__{initial_choice}"
                    if isinstance(choice_transform, CompositeTransform):
                        _get_initial_params(choice_transform, choice_grid, choice_namespace, initial_params)
                    else:
                        _get_leaf_initial_params(choice_transform, choice_grid, choice_namespace, initial_params)
                elif isinstance(target_transform, CompositeTransform):
                    nested_namespace = f"{namespace_prefix}__{step_name}"
                    _get_initial_params(target_transform, step_param_grid, nested_namespace, initial_params)
                else:
                    _get_leaf_initial_params(
                        target_transform, step_param_grid, f"{namespace_prefix}__{step_name}", initial_params
                    )

        def _get_leaf_initial_params(
            target_transform: Any, param_grid: dict[str, Any], namespace: str, initial_params: dict[str, Any]
        ):
            for param_name, space in param_grid.items():
                current_value = target_transform.get_params(deep=False).get(param_name)

                # If current_value is None, it means the parameter wasn't explicitly initialized.
                # In this case, skip enqueueing it and let Optuna sample it.
                if current_value is None:
                    continue

                if not _is_value_in_space(current_value, space):
                    raise ValueError(
                        f"Parameter '{namespace}.{param_name}' has initialized value {current_value!r} "
                        f"which is not within the param_grid space {space}. "
                        f"Either update the initialization to use a value within the allowed range/choices, "
                        f"or modify the param_grid to include the initialized value."
                    )

                initial_params[f"{namespace}__{param_name}"] = current_value

        for pipeline_name, pipeline in self.pipelines_to_tune.items():
            # Only enqueue if there's a param grid for this pipeline
            param_grid = self.param_grid.get(pipeline_name)
            if not param_grid:
                continue

            initial_params = {"pipeline": pipeline_name}

            # This function modifies initial_params in-place
            _get_initial_params(pipeline, param_grid, pipeline_name, initial_params)

            # Enqueue the initial trial with the collected parameters, if any
            # were found
            if len(initial_params) > 1:
                self.study.enqueue_trial(initial_params)
                print(f"Enqueued initial trial with params: {initial_params}")

    def get_best_pipeline(self) -> Pipeline:
        """
        Reconstructs the best pipeline from the optimization results.

        Returns:
            Complete Pipeline object configured with best parameters

        Raises:
            ValueError: If tuning hasn't been run yet
        """
        if self.study is None:
            raise ValueError("Must run tune() before getting best pipeline")

        best_params = self.study.best_trial.params

        # Get the best pipeline name
        pipeline_name = best_params["pipeline"]
        best_pipeline = self.pipelines_to_tune[pipeline_name].clone()

        # Apply the best parameters to the complete pipeline
        # Strip the pipeline name prefix from parameter names and normalize legacy
        # 'choice_' segments introduced by older SwitchTransform naming.
        params_to_set = {}

        def _normalize_choice_segments(name: str) -> str:
            parts = name.split("__")
            normalized_parts = [p[7:] if p.startswith("choice_") else p for p in parts]
            return "__".join(normalized_parts)

        for param_name, param_value in best_params.items():
            if param_name == "pipeline":
                continue
            # Remove pipeline name prefix: "pipeline_name__step__param" ->
            # "step__param"
            if param_name.startswith(f"{pipeline_name}__"):
                stripped_name = param_name[len(f"{pipeline_name}__") :]
                normalized_name = _normalize_choice_segments(stripped_name)
                params_to_set[normalized_name] = param_value
            else:
                params_to_set[_normalize_choice_segments(param_name)] = param_value

        best_pipeline.set_params(**params_to_set)

        return best_pipeline

    def finalize_best_pipeline(
        self,
        data_container: DataContainer | None = None,
        verbose: bool = False,
    ) -> Pipeline:
        """
        Fit the best pipeline on the full dataset to produce a finalized model.

        Args:
            data_container: DataContainer to use for the final fit. Defaults to the initial data.
            verbose: Whether to enable verbose logging in transforms during the final fit.

        Returns:
            A fitted Pipeline ready for inference.
        """
        if self.study is None:
            raise ValueError("Must run tune() before finalizing the best pipeline")

        container = data_container or self.initial_data
        best_pipeline = self.get_best_pipeline()

        validator = copy.deepcopy(self.cv_strategy)
        validator.set_pipeline(best_pipeline)

        return validator.finalize_pipeline(container, verbose=verbose)

    def score_best_pipeline_on_holdout(self, return_validator: bool = False):
        """
        Score the best pipeline on the holdout set.

        Returns:
            Holdout test score for the best pipeline
        """
        best_pipeline = self.get_best_pipeline()

        if self.use_cache:
            best_pipeline, initial_data = self.prepare_pipeline_with_caching(best_pipeline)
        else:
            initial_data = self.initial_data

        validator = copy.deepcopy(self.cv_strategy)
        validator.set_pipeline(best_pipeline)  # Assign the best complete pipeline
        score = validator.score_on_holdout(initial_data, verbose=self.verbose_transforms)

        if self.use_mlflow and self.current_run_id is not None:
            # Reopen the run to log holdout metrics
            with mlflow.start_run(run_id=self.current_run_id):
                mlflow.log_metric("test_score", score)

                # Log confusion matrix (single-label classification only; it is undefined otherwise)
                final_predictor = validator.pipeline.predictive_transform
                if (
                    self.log_artifacts
                    and final_predictor is not None
                    and final_predictor.is_classifier
                    and not getattr(final_predictor, "is_multilabel", False)
                ):
                    artifact_path = "confusion_matrix.png"
                    validator.plot_confusion_matrix(
                        use_holdout=True, normalize=True, save_as=artifact_path, show_plot=False
                    )
                    mlflow.log_artifact(artifact_path, "plots")

        if return_validator:
            return score, validator
        else:
            return score
