"""Transforms module: the base classes, plus every built-in transform by name.

``from xdflow.transforms import CropTimeTransform, ZScoreTransform`` works for all built-in transforms. Each is
imported from its module on first use, so optional dependencies (MNE, LightGBM...) load only when needed.
"""

import importlib
from typing import TYPE_CHECKING, Any

from xdflow.core.base import Predictor, Transform

_MODULES = {
    "adapt_wrapper": ["AdaptWrapperTransform"],
    "basic_transforms": [
        "AverageTransform",
        "BalanceClassWeightTransform",
        "CropTimeTransform",
        "FlattenTransform",
        "FunctionTransform",
        "IdentityTransform",
        "RenameDimsTransform",
        "SampleWeightTransform",
        "TransposeDimsTransform",
        "TrialSampler",
        "UnflattenTransform",
    ],
    "cleaning": ["CARTransform", "RegressOutReferenceTransform", "RemoveOutliersTransform"],
    "csp": ["CSPTransform"],
    "domain_adaptation": [
        "CCAAligner",
        "CoralAligner",
        "DummyAligner",
        "GCCAAligner",
        "JDAAligner",
        "KCCAAligner",
        "MCCAAligner",
        "ProcrustesAligner",
        "SAAligner",
    ],
    "function_step": ["step"],
    "lda": ["CholeskyLDA", "CholeskyLDATransformer"],
    "lgbm_predictor": ["LGBMPredictor"],
    "nearestcentroid": ["NearestCentroid", "NearestCentroidTransform"],
    "normalization": ["DemeanTransform", "ZScoreTransform"],
    "pca": ["GlobalFeaturePCA"],
    "phase": ["HilbertPhaseTransform"],
    "sklearn_transform": ["SKLearnPredictor", "SKLearnTransformer"],
    "spatial": ["GaussianPyramidTransform", "LaplacianCSDTransform", "WindowMeanPyramidTransform"],
    "spectral": ["BandpassFilterTransform", "MultiTaperTransform"],
    "zca": ["GlobalColoringProjection", "GlobalZCAWhitening", "LocalZCAWhitening", "ZCATransform", "ZCAWhitening"],
}
_MODULE_OF = {name: module for module, names in _MODULES.items() for name in names}

__all__ = ["Transform", "Predictor", *sorted(_MODULE_OF)]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module 'xdflow.transforms' has no attribute '{name}'")
    return getattr(importlib.import_module(f"xdflow.transforms.{module}"), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:  # names for type checkers and IDEs; at runtime they load on first use
    from xdflow.transforms.adapt_wrapper import AdaptWrapperTransform
    from xdflow.transforms.basic_transforms import (
        AverageTransform,
        BalanceClassWeightTransform,
        CropTimeTransform,
        FlattenTransform,
        FunctionTransform,
        IdentityTransform,
        RenameDimsTransform,
        SampleWeightTransform,
        TransposeDimsTransform,
        TrialSampler,
        UnflattenTransform,
    )
    from xdflow.transforms.cleaning import CARTransform, RegressOutReferenceTransform, RemoveOutliersTransform
    from xdflow.transforms.csp import CSPTransform
    from xdflow.transforms.domain_adaptation import (
        CCAAligner,
        CoralAligner,
        DummyAligner,
        GCCAAligner,
        JDAAligner,
        KCCAAligner,
        MCCAAligner,
        ProcrustesAligner,
        SAAligner,
    )
    from xdflow.transforms.function_step import step
    from xdflow.transforms.lda import CholeskyLDA, CholeskyLDATransformer
    from xdflow.transforms.lgbm_predictor import LGBMPredictor
    from xdflow.transforms.nearestcentroid import NearestCentroid, NearestCentroidTransform
    from xdflow.transforms.normalization import DemeanTransform, ZScoreTransform
    from xdflow.transforms.pca import GlobalFeaturePCA
    from xdflow.transforms.phase import HilbertPhaseTransform
    from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer
    from xdflow.transforms.spatial import GaussianPyramidTransform, LaplacianCSDTransform, WindowMeanPyramidTransform
    from xdflow.transforms.spectral import BandpassFilterTransform, MultiTaperTransform
    from xdflow.transforms.zca import (
        GlobalColoringProjection,
        GlobalZCAWhitening,
        LocalZCAWhitening,
        ZCATransform,
        ZCAWhitening,
    )
