"""Metadata-driven ML pipelines for labeled structured data.

XDFlow works with structured data stored as labeled xarray objects. It keeps
dimensions, coordinates, targets, groups, split policies, and transform state in
the pipeline contract so validators and tuners can operate by name instead of
by positional side arrays.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xdflow")
except PackageNotFoundError:  # pragma: no cover - direct source-tree import fallback
    __version__ = "0+unknown"

from xdflow.composite import (
    EnsemblePredictor,
    GroupApplyTransform,
    OptionalTransform,
    Pipeline,
    SwitchTransform,
    TransformUnion,
    UnionWithInput,
)
from xdflow.composite.pipeline import make_pipeline
from xdflow.core import DataContainer, Predictor, Transform, TransformError
from xdflow.cv import (
    CrossValidator,
    GroupedKFoldValidator,
    KFoldValidator,
    LeaveAnimalOutValidator,
    LeaveGroupOutValidator,
    LeaveSessionOutValidator,
    SampledDomainKFoldValidator,
    SplitterValidator,
)
from xdflow.diagnostics import check_split
from xdflow.evaluate import evaluate
from xdflow.interop import from_mne_epochs, from_numpy
from xdflow.sklearn_api import to_sklearn
from xdflow.transforms.function_step import step
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer

# Hyperparameter tuning (optional dependency: optuna)
try:
    from xdflow.tuning import Tuner, run_tuning_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "optuna":
        raise
    Tuner = None
    run_tuning_pipeline = None

__all__ = [
    "__version__",
    # Getting started
    "evaluate",
    "check_split",
    "make_pipeline",
    "step",
    "from_numpy",
    "from_mne_epochs",
    "to_sklearn",
    "SKLearnPredictor",
    "SKLearnTransformer",
    # Core
    "Transform",
    "Predictor",
    "DataContainer",
    "TransformError",
    # Composite
    "Pipeline",
    "TransformUnion",
    "UnionWithInput",
    "GroupApplyTransform",
    "SwitchTransform",
    "OptionalTransform",
    "EnsemblePredictor",
    # Cross-validation
    "CrossValidator",
    "KFoldValidator",
    "GroupedKFoldValidator",
    "SampledDomainKFoldValidator",
    "LeaveGroupOutValidator",
    "LeaveSessionOutValidator",
    "LeaveAnimalOutValidator",
    "SplitterValidator",
    # Tuning
    "Tuner",
    "run_tuning_pipeline",
]
