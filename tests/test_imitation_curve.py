import numpy as np
import pytest

from scripts.run_context_imitation_curve import _write_query_csv, _write_summary_csv
from tabpfn_ir.evaluation import (
    minimum_stable_context_sizes,
    predictive_total_variation,
    resolve_context_grid,
    run_full_context_imitation_curve,
)
from tabpfn_ir.models import ContextualTabPFNClassifier


class NearestCentroidClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict_proba(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        scores = np.exp(-distances)
        return scores / scores.sum(axis=1, keepdims=True)


def test_resolve_context_grid_caps_deduplicates_and_appends_full():
    assert resolve_context_grid([64, 16, 64, 1_000], n_train=100) == (16, 64, 100)

    with pytest.raises(ValueError, match="positive"):
        resolve_context_grid([0, 16], n_train=100)


def test_predictive_total_variation_is_computed_per_query_after_normalizing():
    actual = predictive_total_variation(
        np.asarray([[8.0, 2.0], [1.0, 3.0]]),
        np.asarray([[5.0, 5.0], [1.0, 3.0]]),
    )

    np.testing.assert_allclose(actual, [0.3, 0.0])


def test_minimum_stable_context_requires_all_larger_points_to_pass():
    minimum = minimum_stable_context_sizes(
        [16, 32, 64],
        np.asarray(
            [
                [0.01, 0.08, 0.08],
                [0.08, 0.04, 0.03],
                [0.02, 0.02, 0.01],
            ]
        ),
        tolerance=0.05,
        class_agreement_by_context=np.asarray(
            [
                [True, True, True],
                [True, True, False],
                [True, True, True],
            ]
        ),
    )

    assert minimum == [64, 32, 64]


def test_full_context_imitation_curve_returns_reference_and_query_level_results(tmp_path):
    X_train = np.asarray([[0.0], [0.2], [0.4], [9.6], [9.8], [10.0]])
    y_train = np.asarray([0, 0, 0, 1, 1, 1])
    X_query = np.asarray([[0.1], [0.3], [9.7], [9.9]])
    y_query = np.asarray([0, 0, 1, 1])
    ranked = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 0, 3],
            [3, 4, 5, 2],
            [5, 4, 3, 2],
        ]
    )

    result = run_full_context_imitation_curve(
        predictor=ContextualTabPFNClassifier(
            estimator_factory=NearestCentroidClassifier,
            use_batched_contexts=False,
        ),
        X_train_model=X_train,
        y_train=y_train,
        X_query_model=X_query,
        y_query=y_query,
        ranked_context_indices=ranked,
        context_sizes=[2, 4],
        tolerance=0.05,
        query_ids=np.asarray([10, 11, 12, 13]),
    )

    assert result["context_sizes"] == [2, 4, 6]
    assert result["curve"][-1]["is_full_reference"]
    assert result["curve"][-1]["predictive_tv"]["max"] == 0.0
    assert result["curve"][-1]["class_agreement"] == 1.0
    assert result["curve"][-1]["stable_query_coverage"] == 1.0
    assert result["per_query"][0]["query_id"] == 10
    assert len(result["per_query"]) == len(X_query)

    summary_path = tmp_path / "curve.summary.csv"
    query_path = tmp_path / "curve.queries.csv"
    payload = {"result": result}
    _write_summary_csv(payload, summary_path)
    _write_query_csv(payload, query_path)

    assert "predictive_tv_p95" in summary_path.read_text(encoding="utf-8")
    assert "minimum_stable_context_size" in query_path.read_text(encoding="utf-8")
