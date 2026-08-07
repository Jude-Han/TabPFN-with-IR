"""Prediction adapters."""

from tabpfn_ir.models.configuration import build_tabpfn_classifier_kwargs
from tabpfn_ir.models.tabpfn_adapter import ContextualTabPFNClassifier

__all__ = ["ContextualTabPFNClassifier", "build_tabpfn_classifier_kwargs"]
