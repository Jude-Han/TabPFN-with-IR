"""Uniform random context baseline."""

from __future__ import annotations

import numpy as np

from tabpfn_ir.retrieval.base import (
    RetrievalResult,
    Retriever,
    resolve_context_size,
    validate_query_array,
    validate_training_arrays,
)


class RandomRetriever(Retriever):
    """Uniformly sample training rows without replacement.

    With ``global_context=True``, all queries share one sampled context. This is
    the primary random baseline. Otherwise each query receives an independent
    random context. Calls are deterministic for a fixed seed and query count.
    """

    def __init__(self, *, seed: int = 42, global_context: bool = True) -> None:
        self.seed = seed
        self.global_context = global_context
        self._n_train: int | None = None
        self._n_features: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomRetriever":
        X_train, _ = validate_training_arrays(X_train, y_train)
        self._n_train, self._n_features = X_train.shape
        return self

    def retrieve(self, X_query: np.ndarray, k: int | None) -> RetrievalResult:
        if self._n_train is None or self._n_features is None:
            raise RuntimeError("Call fit before retrieve.")
        X_query = validate_query_array(X_query, self._n_features)
        context_size = resolve_context_size(k, self._n_train)
        rng = np.random.default_rng(self.seed)

        if self.global_context:
            context = rng.choice(self._n_train, size=context_size, replace=False)
            indices = np.broadcast_to(context, (X_query.shape[0], context_size)).copy()
        else:
            indices = np.vstack(
                [
                    rng.choice(self._n_train, size=context_size, replace=False)
                    for _ in range(X_query.shape[0])
                ]
            ).astype(np.int64, copy=False)

        scores = np.zeros_like(indices, dtype=np.float32)
        return RetrievalResult(indices=indices, scores=scores)
