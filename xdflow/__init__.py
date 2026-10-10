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
from xdflow.core import DataContainer, Predictor, Transform, TransformError
from xdflow.cv import (
    CrossValidator,
    GroupedKFoldValidator,
    KFoldValidator,
    LeaveAnimalOutValidator,
    LeaveGroupOutValidator,
    LeaveSessionOutValidator,
    SampledDomainKFoldValidator,
)
from xdflow.interop import from_mne_epochs

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
    # Core
    "Transform",
    "Predictor",
    "DataContainer",
    "TransformError",
    "from_mne_epochs",
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
    # Tuning
    "Tuner",
    "run_tuning_pipeline",
]
