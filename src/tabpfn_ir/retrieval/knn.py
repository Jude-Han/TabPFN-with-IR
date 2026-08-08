"""Exact LoCalPFN-style feature-space retrieval with FAISS."""

from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised only in broken environments
    raise ImportError(
        "FAISS is required for kNN retrieval. Install the project with `pip install -e .`."
    ) from exc

from tabpfn_ir.retrieval.base import (
    RetrievalResult,
    Retriever,
    resolve_context_size,
    validate_query_array,
    validate_training_arrays,
)


def localpfn_context_size(n_train: int, *, maximum: int = 1000) -> int:
    """Return the LoCalPFN paper budget ``min(int(10 * sqrt(n_train)), maximum)``."""

    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    return min(int(10 * np.sqrt(n_train)), maximum)


def resolve_context_specification(
    specification: str,
    n_train: int,
    *,
    maximum: int = 1000,
) -> int:
    """Resolve a fixed integer or the LoCalPFN context-size heuristic."""

    if specification.lower() == "localpfn":
        requested = localpfn_context_size(n_train, maximum=maximum)
        return resolve_context_size(requested, n_train)
    try:
        context_size = int(specification)
    except ValueError as exc:
        raise ValueError("k must be a positive integer or 'localpfn'.") from exc
    if context_size <= 0:
        raise ValueError("k must be positive.")
    return min(context_size, n_train)


class KNNRetriever(Retriever):
    """Retrieve an exact L2 local context with ``faiss.IndexFlatL2``.

    FAISS reports squared L2 distances. The returned score is their negative so
    that the common retrieval interface remains higher-is-better.
    """

    def __init__(self, *, query_batch_size: int = 512) -> None:
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive.")
        self.query_batch_size = query_batch_size
        self._index: faiss.IndexFlatL2 | None = None
        self._n_train: int | None = None
        self._n_features: int | None = None

    @staticmethod
    def _as_faiss_array(X: np.ndarray, *, name: str) -> np.ndarray:
        values = np.ascontiguousarray(X, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values for FAISS search.")
        return values

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNRetriever":
        X_train, _ = validate_training_arrays(X_train, y_train)
        X_train = self._as_faiss_array(X_train, name="X_train")
        self._n_train, self._n_features = X_train.shape
        self._index = faiss.IndexFlatL2(self._n_features)
        self._index.add(X_train)
        return self

    def retrieve(self, X_query: np.ndarray, k: int | None) -> RetrievalResult:
        if self._index is None or self._n_train is None or self._n_features is None:
            raise RuntimeError("Call fit before retrieve.")
        X_query = validate_query_array(X_query, self._n_features)
        X_query = self._as_faiss_array(X_query, name="X_query")
        context_size = resolve_context_size(k, self._n_train)
        if X_query.shape[0] == 0:
            empty_shape = (0, context_size)
            return RetrievalResult(
                indices=np.empty(empty_shape, dtype=np.int64),
                scores=np.empty(empty_shape, dtype=np.float32),
            )
        distance_batches = []
        index_batches = []
        for start in range(0, X_query.shape[0], self.query_batch_size):
            distances, indices = self._index.search(
                X_query[start : start + self.query_batch_size],
                context_size,
            )
            distance_batches.append(distances)
            index_batches.append(indices)
        distances = np.concatenate(distance_batches, axis=0)
        indices = np.concatenate(index_batches, axis=0)
        return RetrievalResult(
            indices=indices.astype(np.int64, copy=False),
            scores=(-distances).astype(np.float32, copy=False),
        )
