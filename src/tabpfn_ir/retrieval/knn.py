"""Exact feature-space k-nearest-neighbor retrieval."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from tabpfn_ir.retrieval.base import (
    RetrievalResult,
    Retriever,
    resolve_context_size,
    validate_query_array,
    validate_training_arrays,
)


def localpfn_context_size(n_train: int, *, maximum: int = 1000) -> int:
    """Return floor(min(10 * sqrt(n_train), maximum, n_train))."""

    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    return min(int(10 * np.sqrt(n_train)), maximum, n_train)


def resolve_context_specification(specification: str, n_train: int) -> int:
    """Resolve a fixed integer or the LoCalPFN context-size heuristic."""

    if specification.lower() == "localpfn":
        return localpfn_context_size(n_train)
    try:
        context_size = int(specification)
    except ValueError as exc:
        raise ValueError("k must be a positive integer or 'localpfn'.") from exc
    if context_size <= 0:
        raise ValueError("k must be positive.")
    return min(context_size, n_train)


class KNNRetriever(Retriever):
    """Retrieve a query-specific local context in the retrieval feature space."""

    def __init__(self, *, metric: str = "euclidean", algorithm: str = "brute") -> None:
        self.metric = metric
        self.algorithm = algorithm
        self._index: NearestNeighbors | None = None
        self._n_train: int | None = None
        self._n_features: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNRetriever":
        X_train, _ = validate_training_arrays(X_train, y_train)
        self._n_train, self._n_features = X_train.shape
        self._index = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        self._index.fit(X_train)
        return self

    def retrieve(self, X_query: np.ndarray, k: int | None) -> RetrievalResult:
        if self._index is None or self._n_train is None or self._n_features is None:
            raise RuntimeError("Call fit before retrieve.")
        X_query = validate_query_array(X_query, self._n_features)
        context_size = resolve_context_size(k, self._n_train)
        distances, indices = self._index.kneighbors(
            X_query,
            n_neighbors=context_size,
            return_distance=True,
        )
        return RetrievalResult(
            indices=indices.astype(np.int64, copy=False),
            scores=(-distances).astype(np.float32, copy=False),
        )
