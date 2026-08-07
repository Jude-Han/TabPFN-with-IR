"""Prediction adapters."""

from tabpfn_ir.models.configuration import build_tabpfn_classifier_kwargs
from tabpfn_ir.models.tabpfn_adapter import (
    ContextInferenceStats,
    ContextualTabPFNClassifier,
)

__all__ = [
    "ContextInferenceStats",
    "ContextualTabPFNClassifier",
    "build_tabpfn_classifier_kwargs",
]
