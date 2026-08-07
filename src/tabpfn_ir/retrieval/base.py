"""Common retrieval interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    """Training-row indices and retrieval scores for each query."""

    indices: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.scores.ndim != 2:
            raise ValueError("indices and scores must both have shape [n_query, k].")
        if self.indices.shape != self.scores.shape:
            raise ValueError("indices and scores must have identical shapes.")


class Retriever(ABC):
    """Base class for context-row selection."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Retriever":
        """Index the training fold."""

    @abstractmethod
    def retrieve(self, X_query: np.ndarray, k: int | None) -> RetrievalResult:
        """Return training-row indices and higher-is-better scores."""


def validate_training_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize arrays shared by retriever implementations."""

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be two-dimensional, got {X_train.shape}.")
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be one-dimensional, got {y_train.shape}.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if X_train.shape[0] == 0:
        raise ValueError("The training fold cannot be empty.")
    return X_train, y_train


def validate_query_array(X_query: np.ndarray, n_features: int) -> np.ndarray:
    """Validate a query matrix against the indexed feature dimension."""

    X_query = np.asarray(X_query)
    if X_query.ndim != 2:
        raise ValueError(f"X_query must be two-dimensional, got {X_query.shape}.")
    if X_query.shape[1] != n_features:
        raise ValueError(
            f"X_query has {X_query.shape[1]} features; expected {n_features}."
        )
    return X_query


def resolve_context_size(k: int | None, n_train: int) -> int:
    """Validate a requested context budget and cap it at the fold size."""

    if k is None:
        return n_train
    if k <= 0:
        raise ValueError("k must be positive.")
    return min(k, n_train)
