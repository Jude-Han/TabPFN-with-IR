"""Full-context imitation diagnostics for nested query-specific contexts."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from tabpfn_ir.evaluation.metrics import classification_metrics
from tabpfn_ir.models import ContextualTabPFNClassifier


def resolve_context_grid(
    requested_context_sizes: list[int] | tuple[int, ...],
    n_train: int,
    *,
    include_full: bool = True,
) -> tuple[int, ...]:
    """Validate, cap, deduplicate, and sort a context-size grid.

    The full training-fold size is included by default so the final curve point
    is the exact reference prediction rather than another retrieval run.
    """

    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if not requested_context_sizes and not include_full:
        raise ValueError("At least one context size is required.")
    resolved: set[int] = set()
    for requested in requested_context_sizes:
        if requested <= 0:
            raise ValueError("Every context size must be positive.")
        resolved.add(min(int(requested), n_train))
    if include_full:
        resolved.add(n_train)
    return tuple(sorted(resolved))


def predictive_total_variation(
    probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
) -> np.ndarray:
    """Return per-query TV distance between two categorical predictions."""

    probabilities = _normalize_probabilities(probabilities, name="probabilities")
    reference_probabilities = _normalize_probabilities(
        reference_probabilities,
        name="reference_probabilities",
    )
    if probabilities.shape != reference_probabilities.shape:
        raise ValueError("probabilities and reference_probabilities must have equal shape.")
    return 0.5 * np.abs(probabilities - reference_probabilities).sum(axis=1)


def minimum_stable_context_sizes(
    context_sizes: list[int] | tuple[int, ...],
    tv_by_context: np.ndarray,
    *,
    tolerance: float,
    class_agreement_by_context: np.ndarray | None = None,
) -> list[int | None]:
    """Find the first acceptable context whose larger grid points also pass.

    Requiring every larger evaluated context to pass avoids declaring success at
    a single accidental crossing of the tolerance threshold.
    """

    sizes = np.asarray(context_sizes, dtype=np.int64)
    tv_by_context = np.asarray(tv_by_context, dtype=float)
    if sizes.ndim != 1 or sizes.size == 0:
        raise ValueError("context_sizes must be a non-empty one-dimensional sequence.")
    if np.any(sizes <= 0) or np.any(np.diff(sizes) <= 0):
        raise ValueError("context_sizes must contain unique increasing positive values.")
    if tv_by_context.ndim != 2 or tv_by_context.shape[0] != sizes.size:
        raise ValueError("tv_by_context must have shape [n_context_sizes, n_queries].")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    acceptable = tv_by_context <= tolerance
    if class_agreement_by_context is not None:
        agreement = np.asarray(class_agreement_by_context, dtype=bool)
        if agreement.shape != tv_by_context.shape:
            raise ValueError("class_agreement_by_context must match tv_by_context.")
        acceptable &= agreement

    stable_from_here = np.logical_and.accumulate(acceptable[::-1], axis=0)[::-1]
    minimum_sizes: list[int | None] = []
    for query_position in range(stable_from_here.shape[1]):
        passing = np.flatnonzero(stable_from_here[:, query_position])
        minimum_sizes.append(int(sizes[passing[0]]) if passing.size else None)
    return minimum_sizes


def run_full_context_imitation_curve(
    *,
    predictor: ContextualTabPFNClassifier,
    X_train_model: np.ndarray,
    y_train: np.ndarray,
    X_query_model: np.ndarray,
    y_query: np.ndarray,
    ranked_context_indices: np.ndarray,
    context_sizes: list[int] | tuple[int, ...],
    tolerance: float = 0.05,
    require_class_agreement: bool = True,
    query_ids: np.ndarray | None = None,
    auc_mode: str = "ovo",
) -> dict[str, Any]:
    """Measure how nested retrieved contexts imitate full-context TabPFN.

    ``ranked_context_indices`` must contain the retrieval result up to the
    largest requested context smaller than the full training fold. Each smaller
    context is obtained by taking a prefix, making the curve exactly nested.
    """

    X_train_model = np.asarray(X_train_model)
    y_train = np.asarray(y_train)
    X_query_model = np.asarray(X_query_model)
    y_query = np.asarray(y_query)
    ranked_context_indices = np.asarray(ranked_context_indices, dtype=np.int64)
    n_train = X_train_model.shape[0]
    n_query = X_query_model.shape[0]
    sizes = resolve_context_grid(list(context_sizes), n_train, include_full=True)
    if y_train.shape != (n_train,):
        raise ValueError("y_train must have one label per training row.")
    if y_query.shape != (n_query,):
        raise ValueError("y_query must have one label per query row.")
    if ranked_context_indices.ndim != 2 or ranked_context_indices.shape[0] != n_query:
        raise ValueError("ranked_context_indices must have shape [n_query, max_retrieved_k].")
    largest_retrieved = max((size for size in sizes if size < n_train), default=0)
    if ranked_context_indices.shape[1] < largest_retrieved:
        raise ValueError(
            "ranked_context_indices does not contain the largest requested context."
        )
    if ranked_context_indices.size and (
        ranked_context_indices.min() < 0 or ranked_context_indices.max() >= n_train
    ):
        raise IndexError("ranked_context_indices contains a row outside the training fold.")
    if query_ids is None:
        query_ids = np.arange(n_query)
    query_ids = np.asarray(query_ids)
    if query_ids.shape != (n_query,):
        raise ValueError("query_ids must contain one value per query row.")

    full_context_indices = np.broadcast_to(
        np.arange(n_train, dtype=np.int64),
        (n_query, n_train),
    )
    started = perf_counter()
    full_probabilities, classes = predictor.predict_proba_with_contexts(
        X_train_model,
        y_train,
        X_query_model,
        full_context_indices,
    )
    full_prediction_seconds = perf_counter() - started
    full_inference_stats = asdict(predictor.last_inference_stats)
    full_metrics = classification_metrics(
        y_query,
        full_probabilities,
        classes,
        auc_mode=auc_mode,
    )
    full_predictions = classes[np.argmax(full_probabilities, axis=1)]

    tv_rows: list[np.ndarray] = []
    agreement_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    curve: list[dict[str, Any]] = []
    for context_size in sizes:
        is_full_reference = context_size == n_train
        if is_full_reference:
            probabilities = full_probabilities
            prediction_seconds = full_prediction_seconds
            inference_stats = full_inference_stats
        else:
            started = perf_counter()
            probabilities, local_classes = predictor.predict_proba_with_contexts(
                X_train_model,
                y_train,
                X_query_model,
                ranked_context_indices[:, :context_size],
            )
            prediction_seconds = perf_counter() - started
            if not np.array_equal(local_classes, classes):
                raise RuntimeError("The global class order changed across context sizes.")
            inference_stats = asdict(predictor.last_inference_stats)

        tv = predictive_total_variation(probabilities, full_probabilities)
        predictions = classes[np.argmax(probabilities, axis=1)]
        agreement = predictions == full_predictions
        metrics = classification_metrics(
            y_query,
            probabilities,
            classes,
            auc_mode=auc_mode,
        )
        tv_rows.append(tv)
        agreement_rows.append(agreement)
        prediction_rows.append(predictions)
        curve.append(
            {
                "context_size": context_size,
                "is_full_reference": is_full_reference,
                "prediction_seconds": prediction_seconds,
                "prediction_seconds_per_query": prediction_seconds / max(n_query, 1),
                "predictive_tv": _summarize_values(tv),
                "class_agreement": float(np.mean(agreement)),
                "classification_metrics": metrics,
                "metric_delta_from_full": {
                    metric: value - full_metrics[metric] for metric, value in metrics.items()
                },
                "inference_stats": inference_stats,
            }
        )

    tv_matrix = np.stack(tv_rows)
    agreement_matrix = np.stack(agreement_rows)
    minimum_sizes = minimum_stable_context_sizes(
        sizes,
        tv_matrix,
        tolerance=tolerance,
        class_agreement_by_context=(
            agreement_matrix if require_class_agreement else None
        ),
    )
    for point in curve:
        context_size = int(point["context_size"])
        point["stable_query_coverage"] = float(
            np.mean(
                [
                    minimum is not None and minimum <= context_size
                    for minimum in minimum_sizes
                ]
            )
        )

    per_query = []
    for query_position in range(n_query):
        per_query.append(
            {
                "query_position": query_position,
                "query_id": _to_python_scalar(query_ids[query_position]),
                "true_label": _to_python_scalar(y_query[query_position]),
                "full_predicted_label": _to_python_scalar(
                    full_predictions[query_position]
                ),
                "full_probabilities": [
                    float(value) for value in full_probabilities[query_position]
                ],
                "minimum_stable_context_size": minimum_sizes[query_position],
                "tv_by_context": {
                    str(size): float(tv_matrix[index, query_position])
                    for index, size in enumerate(sizes)
                },
                "class_agreement_by_context": {
                    str(size): bool(agreement_matrix[index, query_position])
                    for index, size in enumerate(sizes)
                },
                "predicted_label_by_context": {
                    str(size): _to_python_scalar(
                        prediction_rows[index][query_position]
                    )
                    for index, size in enumerate(sizes)
                },
            }
        )

    return {
        "n_train": n_train,
        "n_query": n_query,
        "classes": [_to_python_scalar(label) for label in classes],
        "context_sizes": list(sizes),
        "criterion": {
            "predictive_tv_tolerance": tolerance,
            "require_class_agreement": require_class_agreement,
            "stability_rule": (
                "The selected k and every larger evaluated context must pass."
            ),
        },
        "full_reference": {
            "context_size": n_train,
            "prediction_seconds": full_prediction_seconds,
            "classification_metrics": full_metrics,
            "inference_stats": full_inference_stats,
        },
        "curve": curve,
        "per_query": per_query,
    }


def _normalize_probabilities(probabilities: np.ndarray, *, name: str) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError(f"{name} must have shape [n_queries, n_classes].")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must contain only finite values.")
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"Every row in {name} must have a positive sum.")
    return probabilities / row_sums


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _to_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
