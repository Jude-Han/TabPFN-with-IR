"""Adapter for running frozen TabPFN with query-specific contexts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class ProbabilisticClassifier(Protocol):
    """The minimal estimator API used by the adapter."""

    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class ContextualTabPFNClassifier:
    """Run one TabPFN fit per unique retrieved context.

    Full and global-random contexts are fitted once and shared across their query
    batches. Query-specific kNN contexts are fitted independently. TabPFN itself
    remains frozen: ``fit`` supplies ICL context and does not perform supervised
    gradient updates.
    """

    def __init__(
        self,
        *,
        estimator_factory: Callable[[], ProbabilisticClassifier] | None = None,
        tabpfn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._estimator_factory = estimator_factory
        self._tabpfn_kwargs = dict(tabpfn_kwargs or {})

    def _new_estimator(self) -> ProbabilisticClassifier:
        if self._estimator_factory is not None:
            return self._estimator_factory()
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TabPFN support is optional. Install the project with "
                "`pip install -e '.[benchmark]'`."
            ) from exc
        return TabPFNClassifier(**self._tabpfn_kwargs)

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        local_classes: np.ndarray,
        global_classes: np.ndarray,
    ) -> np.ndarray:
        aligned = np.zeros((probabilities.shape[0], global_classes.shape[0]), dtype=float)
        global_positions = {label: position for position, label in enumerate(global_classes.tolist())}
        for local_position, label in enumerate(np.asarray(local_classes).tolist()):
            aligned[:, global_positions[label]] = probabilities[:, local_position]
        return aligned

    def predict_proba_with_contexts(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_query: np.ndarray,
        context_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict probabilities for query-specific retrieved row indices."""

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_query = np.asarray(X_query)
        context_indices = np.asarray(context_indices)
        if context_indices.ndim != 2 or context_indices.shape[0] != X_query.shape[0]:
            raise ValueError("context_indices must have shape [n_query, k].")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if context_indices.size and (
            context_indices.min() < 0 or context_indices.max() >= X_train.shape[0]
        ):
            raise IndexError("context_indices contains a row outside the training fold.")

        global_classes = np.unique(y_train)
        output = np.zeros((X_query.shape[0], global_classes.shape[0]), dtype=float)
        grouped_queries: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for query_position, row_indices in enumerate(context_indices):
            grouped_queries[tuple(int(index) for index in row_indices)].append(query_position)

        estimator = self._new_estimator()
        for context, query_positions in grouped_queries.items():
            selected = np.asarray(context, dtype=np.int64)
            estimator.fit(X_train[selected], y_train[selected])
            local_probabilities = np.asarray(
                estimator.predict_proba(X_query[query_positions]),
                dtype=float,
            )
            local_classes = np.asarray(estimator.classes_)
            output[query_positions] = self._align_probabilities(
                local_probabilities,
                local_classes,
                global_classes,
            )
        return output, global_classes
