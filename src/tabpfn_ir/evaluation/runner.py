"""One-fold retrieval experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from tabpfn_ir.evaluation.metrics import classification_metrics
from tabpfn_ir.models import ContextualTabPFNClassifier
from tabpfn_ir.retrieval import Retriever


@dataclass(frozen=True)
class ExperimentResult:
    """Metrics and timing information for one method, fold, and context budget."""

    method: str
    requested_k: int | None
    actual_k: int
    n_train: int
    n_query: int
    index_seconds: float
    retrieval_seconds: float
    prediction_seconds: float
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""

        return asdict(self)


def run_retrieval_experiment(
    *,
    retriever: Retriever,
    predictor: ContextualTabPFNClassifier,
    X_train_model: np.ndarray,
    X_train_retrieval: np.ndarray,
    y_train: np.ndarray,
    X_query_model: np.ndarray,
    X_query_retrieval: np.ndarray,
    y_query: np.ndarray,
    k: int | None,
) -> ExperimentResult:
    """Fit a retriever, build contexts, predict, and evaluate one query split."""

    started = perf_counter()
    retriever.fit(X_train_retrieval, y_train)
    index_seconds = perf_counter() - started

    started = perf_counter()
    retrieval = retriever.retrieve(X_query_retrieval, k)
    retrieval_seconds = perf_counter() - started

    started = perf_counter()
    probabilities, classes = predictor.predict_proba_with_contexts(
        X_train_model,
        y_train,
        X_query_model,
        retrieval.indices,
    )
    prediction_seconds = perf_counter() - started

    return ExperimentResult(
        method=type(retriever).__name__,
        requested_k=k,
        actual_k=retrieval.indices.shape[1],
        n_train=X_train_model.shape[0],
        n_query=X_query_model.shape[0],
        index_seconds=index_seconds,
        retrieval_seconds=retrieval_seconds,
        prediction_seconds=prediction_seconds,
        metrics=classification_metrics(y_query, probabilities, classes),
    )
