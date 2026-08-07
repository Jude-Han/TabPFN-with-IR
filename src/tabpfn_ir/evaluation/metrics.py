"""Classification metrics with explicit class ordering."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    auc_mode: str = "ovr",
) -> dict[str, float]:
    """Compute classification metrics using an explicit global class order.

    ``auc_mode="ovo"`` reproduces the multiclass convention reported in the
    TabPFN v1 and LoCalPFN papers. Binary AUC is unchanged by this option.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.shape != (y_true.shape[0], classes.shape[0]):
        raise ValueError("probabilities must have shape [n_examples, n_classes].")
    if auc_mode not in {"ovo", "ovr"}:
        raise ValueError("auc_mode must be either 'ovo' or 'ovr'.")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values.")
    # Aligned local-context outputs can differ from one by floating-point noise.
    # Normalizing here avoids misleading sklearn warnings without changing ranks.
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every probability row must have a positive sum.")
    probabilities = probabilities / row_sums

    predicted = classes[np.argmax(probabilities, axis=1)]
    metrics = {
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
    }
    if classes.shape[0] == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities[:, 1], labels=classes)
        )
    else:
        auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class=auc_mode,
                average="macro",
            )
        )
        metrics["roc_auc"] = auc
        metrics[f"roc_auc_{auc_mode}"] = auc
    return metrics
