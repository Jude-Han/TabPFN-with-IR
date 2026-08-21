"""Prediction adapters."""

from tabpfn_ir.models.configuration import (
    TABPFN_FINETUNING_MODEL_VERSIONS,
    TABPFN_INFERENCE_PROFILES,
    TABPFN_MODEL_VERSIONS,
    build_tabpfn_classifier_kwargs,
    resolve_n_estimators,
    validate_model_version,
)
from tabpfn_ir.models.local_finetuning import (
    LocalFinetunedTabPFNClassifier,
    LocalPredictionResult,
)
from tabpfn_ir.models.tabpfn_adapter import (
    ContextInferenceStats,
    ContextualTabPFNClassifier,
)
from tabpfn_ir.models.tabpfn_v1 import LEGACY_TABPFN_VERSION

__all__ = [
    "LEGACY_TABPFN_VERSION",
    "TABPFN_FINETUNING_MODEL_VERSIONS",
    "TABPFN_INFERENCE_PROFILES",
    "TABPFN_MODEL_VERSIONS",
    "ContextInferenceStats",
    "ContextualTabPFNClassifier",
    "LocalFinetunedTabPFNClassifier",
    "LocalPredictionResult",
    "build_tabpfn_classifier_kwargs",
    "resolve_n_estimators",
    "validate_model_version",
]
