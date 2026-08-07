import numpy as np

from tabpfn_ir.evaluation import classification_metrics


def test_multiclass_metrics_support_paper_ovo_auc_and_normalize_rows():
    y = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.asarray(
        [
            [8, 1, 1],
            [1, 8, 1],
            [1, 1, 8],
            [7, 2, 1],
            [2, 7, 1],
            [1, 2, 7],
        ],
        dtype=float,
    )

    metrics = classification_metrics(y, probabilities, np.arange(3), auc_mode="ovo")

    assert metrics["roc_auc"] == 1.0
    assert metrics["roc_auc_ovo"] == 1.0
    assert metrics["weighted_f1"] == 1.0
