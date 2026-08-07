"""Prediction adapters."""

from tabpfn_ir.models.configuration import (
    TABPFN_INFERENCE_PROFILES,
    build_tabpfn_classifier_kwargs,
    resolve_n_estimators,
)
from tabpfn_ir.models.tabpfn_adapter import (
    ContextInferenceStats,
    ContextualTabPFNClassifier,
)

__all__ = [
    "ContextInferenceStats",
    "ContextualTabPFNClassifier",
    "TABPFN_INFERENCE_PROFILES",
    "build_tabpfn_classifier_kwargs",
    "resolve_n_estimators",
]
