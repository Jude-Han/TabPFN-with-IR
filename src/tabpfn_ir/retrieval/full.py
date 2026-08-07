"""Full training-fold context baseline."""

from __future__ import annotations

import numpy as np

from tabpfn_ir.retrieval.base import (
    RetrievalResult,
    Retriever,
    validate_query_array,
    validate_training_arrays,
)


class FullContextRetriever(Retriever):
    """Return every indexed training row for every query."""

    def __init__(self) -> None:
        self._n_train: int | None = None
        self._n_features: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FullContextRetriever":
        X_train, _ = validate_training_arrays(X_train, y_train)
        self._n_train, self._n_features = X_train.shape
        return self

    def retrieve(self, X_query: np.ndarray, k: int | None = None) -> RetrievalResult:
        if self._n_train is None or self._n_features is None:
            raise RuntimeError("Call fit before retrieve.")
        X_query = validate_query_array(X_query, self._n_features)
        if k is not None and k < self._n_train:
            raise ValueError(
                "FullContextRetriever always returns all training rows; pass k=None "
                "or use another retriever for a bounded context."
            )
        indices = np.broadcast_to(
            np.arange(self._n_train, dtype=np.int64),
            (X_query.shape[0], self._n_train),
        ).copy()
        scores = np.zeros_like(indices, dtype=np.float32)
        return RetrievalResult(indices=indices, scores=scores)
