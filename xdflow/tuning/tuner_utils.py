# # import copy
# # import importlib.util
# # import os
# # import sys
# # import time
# # from typing import Dict, List

# # import matplotlib.pyplot as plt
# # import mlflow
# # import numpy as np
# # from lightgbm import LGBMClassifier
# # from optuna_integration import MLflowCallback

# # from xdflow import Tuner
# # from xdflow.config.config_mapping import CLASSNAME_CONFIG_MAP
# # from xdflow.config.config_utils import is_hydra_class
# # from xdflow.config.constants import CONF_DIR, PARAM_GRID_DIR
# # from xdflow.cross_validators.base import AcrossSessionCV, KFoldCV
# # from xdflow.functions.visualization import plot_combined_confusion_matrix, plot_tune_importances
# # from xdflow.utils.cache_utils import hash_dict

import copy
import gc
import random
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator
from xdflow.tuning.base import Tuner
from xdflow.utils.visualizations import (
    plot_combined_confusion_matrix,
    plot_confusion_matrix,
    plot_tune_importances,
)

# # def create_param_grids(
# #     classes: List,
# #     grid_changes: Dict,
# #     base_config_name: str = "default_param_grids.py",
# #     config_dir: str = PARAM_GRID_DIR,
# # ) -> Dict:
# #     """
# #     Loads the param_grids dict from a Python config file and applies changes.

# #     Args:
# #         classes: List of classes to get param_grids for.
# #         grid_changes: Dict of grid changes to apply to param_grids, keys are class names, values are dicts of grid changes, will replace the entire grid for that class.
# #         base_config_name: Name of the base config file. Defaults to "default_param_grids.py".
# #         config_dir: Directory of the base config file. Defaults to PARAM_GRID_DIR.

# #     Returns:
# #         param_grids: Dict of param_grids for classes.

# #     Raises:
# #         KeyError: If a class is not found in the base param_grids.
# #     """

# #     # get param_grids from config file
# #     base_param_grids = get_param_grids(base_config_name, config_dir)

# #     # get param_grids for classes
# #     param_grids = {}
# #     for class_name in classes:
# #         if class_name in base_param_grids:
# #             param_grids[class_name] = base_param_grids[class_name]
# #         else:
# #             raise KeyError(f"Class {class_name} not found in param_grids")

# #     # apply grid_changes to param_grids
# #     param_grids = update_param_grids(param_grids, grid_changes=grid_changes)

# #     # TODO: add ways to change param grids based on predefined sub-grids

# #     return param_grids


# # def get_param_grids(config_name: str, config_dir: str = PARAM_GRID_DIR):
# #     """
# #     Loads the param_grids dictionary from a Python config file.

# #     Args:
# #         config_name: Name of the config file.
# #         config_dir: Directory of the config file. Defaults to PARAM_GRID_DIR.

# #     Returns:
# #         param_grids: The param_grids dictionary loaded from the config file.

# #     Raises:
# #         FileNotFoundError: If the config file does not exist.
# #         ValueError: If the config file is not a Python file.
# #     """
# #     # get path to config file
# #     config_path = os.path.join(config_dir, config_name)
# #     if not os.path.exists(config_path):
# #         raise FileNotFoundError(f"Config file {config_path} not found")
# #     if not config_path.endswith(".py"):
# #         raise ValueError(f"Config file {config_path} is not a python file")

# #     # get param_grids from config file
# #     spec = importlib.util.spec_from_file_location("param_grid_module", config_path)
# #     module = importlib.util.module_from_spec(spec)
# #     sys.modules["param_grid_module"] = module
# #     spec.loader.exec_module(module)
# #     param_grids = module.param_grids

# #     return param_grids


# # def update_param_grids(
# #     param_grids: Dict,
# #     grid_changes: Dict = None,
# #     changes_config_name: str = None,
# #     config_dir: str = PARAM_GRID_DIR,
# # ) -> Dict:
# #     """
# #     Update the param_grids dict with the grid_changes dict or another config file.

# #     Args:
# #         param_grids: The original param_grids dictionary.
# #         grid_changes: Dictionary of changes to apply. Defaults to None.
# #         changes_config_name: Name of another config file to pull changes from. Defaults to None.
# #         config_dir: Directory of the config file. Defaults to PARAM_GRID_DIR.

# #     Returns:
# #         param_grids: The updated param_grids dictionary.

# #     Raises:
# #         KeyError: If a key in grid_changes or the new config is not found in param_grids.
# #     """
# #     param_grids = copy.deepcopy(param_grids)

# #     if grid_changes is not None:
# #         for key, value in grid_changes.items():
# #             if key in param_grids:
# #                 param_grids[key] = value
# #             else:
# #                 raise KeyError(f"Key {key} not found in param_grids")

# #     if changes_config_name is not None:
# #         new_param_grids = get_param_grids(changes_config_name, config_dir)
# #         for key, value in new_param_grids.items():
# #             if key in param_grids:
# #                 param_grids[key] = value
# #             else:
# #                 raise KeyError(f"Key {key} not found in param_grids")

# #     return param_grids


# # def _update_param_grids_within_tuning(param_grids: Dict, seed: int, test_session: str) -> Dict:
# #     """
# #     Update the param_grids dict with the seed and test_session name for specific components during tuning.

# #     Args:
# #         param_grids: The param_grids dictionary to update.
# #         seed: The random seed to set.
# #         test_session: The test session name to set.

# #     Returns:
# #         param_grids: The updated param_grids dictionary.
# #     """
# #     # update param grids for specific components based on tuning loop info

# #     if KFoldCV in param_grids:
# #         param_grids[KFoldCV]["seed"] = [seed]

# #     if AcrossSessionCV in param_grids:
# #         param_grids[AcrossSessionCV]["test_session_name"] = [test_session]
# #         param_grids[AcrossSessionCV]["seed"] = [seed]

# #     if LGBMClassifier in param_grids:
# #         param_grids[LGBMClassifier]["random_state"] = [seed]

# #     for key in param_grids.keys():
# #         if is_hydra_class(key):  # TODO: make sure random seeds are handled properly in tuning
# #             param_grids[key]["seed"] = [seed]

# #     return param_grids


def run_tuning_pipeline(
    pipelines_to_tune: Pipeline | list[Pipeline],
    cv_strategy: CrossValidator,
    param_grid: dict[str, dict[str, dict[str, Any]]],
    initial_data_container: DataContainer,
    experiment_name: str | None = None,
    mlflow_metadata: dict[str, Any] | None = None,
    n_seeds: int = 1,
    n_trials: int = 10,
    plot_importances: bool = False,
    plot_combined_conf_matrix: bool = True,
    plot_each_seed_conf_matrix: bool = False,
    scoring_mask_func: Any | None = None,
    exclude_intertrial_from_scoring: bool | None = None,
    holdout_ids: Sequence[Any] | None = None,
    n_holdouts: int | None = 1,
    holdout_chunk_seed: int | None = 0,
    return_pipelines: bool = True,
    score_on_holdout: bool = True,
    log_trial_params: bool = False,
    log_artifacts: bool = True,
    **kwargs,
):
    """
    Run a tuning pipeline and return the finalized pipelines.

    Args:
        pipelines_to_tune: The pipelines to tune.
        cv_strategy: The cross-validator to use.
        param_grid: The parameter grid to use.
        initial_data_container: The initial data container to use.
        experiment_name: The name of the experiment.
        mlflow_metadata: The metadata to use for mlflow.
        n_seeds: The number of seeds to use.
        n_trials: The number of trials to use.
        plot_importances: Whether to plot the importances.
        plot_combined_conf_matrix: Whether to plot the combined confusion matrix.
        plot_each_seed_conf_matrix: Whether to plot the confusion matrix for each seed.
        scoring_mask_func: Optional function to compute scoring mask for container-aware scorers.
                          Should have signature: (DataContainer) -> np.ndarray (boolean mask).
                          Used to filter confusion matrices to match the scorer's logic.
        exclude_intertrial_from_scoring: Optional toggle propagated to each validator so tuning ignores
            synthetic intertrial segments when scoring folds/holdouts.
        holdout_ids: Optional sequence of IDs eligible for holdout (e.g., session IDs).
            When provided alongside n_holdouts, each seed samples `n_holdouts`
            unique IDs from this pool (without replacement within the sample) and
            assigns them to the validator's holdout attribute.
        n_holdouts: Number of holdout IDs to sample per seed. Required if holdout_ids is set.
        holdout_chunk_seed: Optional RNG seed controlling holdout sampling.
        return_pipelines: Whether to return the finalized pipelines.
        score_on_holdout: When False, skip the holdout eval step and rely on cross-validation
            scores instead (useful for CV-only tuning flows).
        log_trial_params: When True, print every Optuna trial's parameter dictionary and score
            after tuning completes. Helpful for debugging the search space.
        **kwargs: Additional keyword arguments to pass to the Tuner.

    Returns:
        The finalized pipelines.
    """
    cms = []
    test_scores = []
    test_trues = []
    finalized_pipelines = []
    class_labels: list[Any] | None = None
    is_classification = None  # Will be set in first iteration
    metric_name = None  # For regression tasks

    holdout_pool: list[Any] | None = None
    holdout_sampler: random.Random | None = None
    holdout_assignment: list[Any] | None = None
    holdout_pointer = 0
    if holdout_ids is not None:
        if n_holdouts is None:
            raise ValueError("n_holdouts must be provided when holdout_ids is set.")
        holdout_pool = list(holdout_ids)
        if len(holdout_pool) < n_holdouts:
            raise ValueError(f"Requested n_holdouts={n_holdouts} but only {len(holdout_pool)} holdout_ids provided.")
        holdout_sampler = random.Random(holdout_chunk_seed)
        holdout_assignment = holdout_pool[:]
        holdout_sampler.shuffle(holdout_assignment)
        total_needed = n_seeds * n_holdouts
        if len(holdout_assignment) < total_needed:
            raise ValueError(
                f"Need {total_needed} unique holdout IDs for {n_seeds} seeds "
                f"(n_holdouts={n_holdouts}) but only {len(holdout_assignment)} provided."
            )
        warnings.warn(
            f"Randomly assigning {n_holdouts} holdout IDs for each seed. Current holdout_ids will be overwritten."
        )

    def _assign_holdout_ids(validator: CrossValidator, holdout_ids: list[Any]) -> None:
        """Assign holdout IDs to the validator using common attribute names."""
        if hasattr(validator, "test_group_ids"):
            cast(Any, validator).test_group_ids = holdout_ids
        elif hasattr(validator, "test_session_ids"):
            cast(Any, validator).test_session_ids = holdout_ids
        elif hasattr(validator, "test_animal_ids"):
            cast(Any, validator).test_animal_ids = holdout_ids
        else:
            raise AttributeError("Validator does not expose a test_* attribute to assign holdout IDs.")

    def _resolve_class_labels(final_predictor: Any) -> list[Any]:
        """Resolve class labels from a predictor or its underlying estimator."""
        if final_predictor is None:
            raise ValueError("No predictive_transform found on the pipeline; cannot resolve class labels.")
        if getattr(final_predictor, "encoder", None) is not None:
            return list(final_predictor.encoder.classes_)
        estimator = getattr(final_predictor, "estimator", None)
        if estimator is None or not hasattr(estimator, "classes_"):
            raise ValueError(
                "Class labels are unavailable. Provide an encoder on the final predictor or use an estimator "
                "exposing classes_."
            )
        return list(estimator.classes_)

    for seed in range(n_seeds):
        print(f"Tuning with seed {seed}")
        cv_for_seed = copy.deepcopy(cv_strategy)
        validator = None
        if holdout_pool is not None:
            if holdout_assignment is None:
                raise RuntimeError("holdout_assignment was not initialized.")
            if n_holdouts is None:
                raise RuntimeError("n_holdouts was not initialized.")
            sampled_ids = holdout_assignment[holdout_pointer : holdout_pointer + n_holdouts]
            holdout_pointer += n_holdouts
            _assign_holdout_ids(cv_for_seed, sampled_ids)

        tuner = Tuner(
            pipelines_to_tune=pipelines_to_tune,
            cv_strategy=cv_for_seed,
            param_grid=param_grid,
            initial_data_container=initial_data_container,
            mlflow_experiment_name=experiment_name,
            mlflow_metadata=mlflow_metadata,
            random_seed=seed,
            log_artifacts=log_artifacts,
            **kwargs,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
        )
        tuner.tune(n_trials=n_trials, show_progress_bar=True)

        if log_trial_params:
            print("Optuna trial parameters:")
            for trial in tuner.study.trials:
                status = getattr(trial.state, "name", str(trial.state))
                value = trial.value
                print(f"  Trial {trial.number} [{status}] value={value}: {trial.params}")

        # Get best pipeline and score
        if score_on_holdout:
            holdout_score, validator = tuner.score_best_pipeline_on_holdout(return_validator=True)
            holdout_trues = validator.holdout_true_labels_

            # Compute scoring mask if provided (for container-aware scorers)
            if scoring_mask_func is not None:
                validator.compute_holdout_scoring_mask(scoring_mask_func)
                # Filter holdout_trues to match the scoring mask
                if validator.holdout_scoring_mask_ is not None:
                    holdout_trues = holdout_trues[validator.holdout_scoring_mask_]

            final_predictor = validator.pipeline.predictive_transform
            if final_predictor is None:
                raise ValueError("No predictive_transform found on the pipeline; cannot score on holdout.")
            is_classification = final_predictor.is_classifier

            if is_classification:
                # Classification: get confusion matrix and labels
                conf_matrix = validator.holdout_confusion_matrix_normalized_

                class_labels = _resolve_class_labels(final_predictor)

                cms.append(conf_matrix)
                print(f"Holdout F1 score: {holdout_score}")
            else:
                # Regression: no confusion matrix
                conf_matrix = None
                metric_name = validator.metric_name_
                print(f"Holdout {metric_name.upper()}: {holdout_score:.4f}")

            score = holdout_score
            trues = holdout_trues
        else:
            best_pipeline = tuner.get_best_pipeline()
            final_predictor = best_pipeline.predictive_transform
            if final_predictor is None:
                raise ValueError("No predictive_transform found on the pipeline; cannot score cross-validation.")
            is_classification = final_predictor.is_classifier
            # The Tuner's own validator: it carries the overrides and seeds the Tuner applied to its copy.
            validator = copy.deepcopy(tuner.cv_strategy)
            if hasattr(validator, "set_pipeline"):  # metric_name_ needs the pipeline's predictor
                validator.set_pipeline(best_pipeline)
            metric_name = validator.metric_name_

            needs_cv_eval = is_classification and (plot_each_seed_conf_matrix or plot_combined_conf_matrix)
            if needs_cv_eval:
                cv_score = validator.cross_validate(initial_data_container, verbose=validator.verbose)
                cv_trues = np.concatenate(validator.true_labels_) if validator.true_labels_ else np.array([])
                conf_matrix = validator.oof_confusion_matrix_normalized_
                class_labels = _resolve_class_labels(final_predictor)
                cms.append(conf_matrix)
            else:
                cv_score = tuner.study.best_value
                cv_trues = np.array([])
                conf_matrix = None
            score = cv_score
            trues = cv_trues
            print(f"Cross-validation {metric_name.upper()} (no holdout): {cv_score:.4f}")

        # delete validator
        if validator is not None:
            del validator
            gc.collect()

        # log
        test_scores.append(score)  # Name kept for backward compat, but contains regression scores too
        test_trues.append(trues)

        if plot_each_seed_conf_matrix and is_classification and conf_matrix is not None:
            if class_labels is None:
                raise RuntimeError("Class labels were not resolved for classification plotting.")
            labels = class_labels
            source = "holdout" if score_on_holdout else "cv"
            plot_confusion_matrix(
                conf_matrix,
                labels,
                title=f"Confusion matrix ({source}) for seed {seed}, F1 score: {score:.4f}",
                test_trues=trues,
                ylabels=labels,
                xlabels=labels,
            )

        if plot_importances:
            plot_tune_importances(tuner.study)

        if return_pipelines:
            finalized_pipeline = tuner.finalize_best_pipeline(verbose=tuner.verbose_transforms)
            finalized_pipelines.append(finalized_pipeline)

    # Print average score and standard error across all seeds
    if len(test_scores) > 0:
        mean_score = np.mean(test_scores)
        std_error_score = np.std(test_scores, ddof=1) / np.sqrt(len(test_scores)) if len(test_scores) > 1 else 0.0

        # Use appropriate metric name (F1 for classification, or detected metric for regression)
        if is_classification:
            print(f"\nAverage F1 score across {len(test_scores)} seed(s): {mean_score:.4f} ± {std_error_score:.4f}")
        else:
            # metric_name from last validator (all should be same)
            metric_label = metric_name or "score"
            print(
                f"\nAverage {metric_label.upper()} across {len(test_scores)} seed(s): {mean_score:.4f} ± {std_error_score:.4f}"
            )

    # Only plot confusion matrix for classification tasks
    if plot_combined_conf_matrix and class_labels is not None and is_classification and len(cms) > 0:
        source = "holdout" if score_on_holdout else "cv"
        plot_combined_confusion_matrix(
            cms,
            class_labels,
            test_scores,
            test_trues=test_trues,
            title=f"{experiment_name} ({source}), average score: {mean_score:.4f}",
            want_plot=True,
        )

    if return_pipelines:
        return finalized_pipelines
    else:
        return None
