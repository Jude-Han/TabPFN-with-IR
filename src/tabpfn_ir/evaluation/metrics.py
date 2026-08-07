"""Classification metrics with explicit class ordering."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Compute the baseline classification metrics used by this project."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    if probabilities.shape != (y_true.shape[0], classes.shape[0]):
        raise ValueError("probabilities must have shape [n_examples, n_classes].")

    predicted = classes[np.argmax(probabilities, axis=1)]
    metrics = {
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }
    if classes.shape[0] == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities[:, 1], labels=classes)
        )
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    return metrics
