import numpy as np

from tabpfn_ir.evaluation import run_retrieval_experiment
from tabpfn_ir.models import ContextualTabPFNClassifier
from tabpfn_ir.retrieval import RandomRetriever


class NearestCentroidClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict_proba(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        scores = np.exp(-distances)
        return scores / scores.sum(axis=1, keepdims=True)


def test_adapter_groups_shared_context_and_runner_returns_metrics():
    X_train = np.asarray([[0.0], [0.2], [9.8], [10.0]], dtype=np.float32)
    y_train = np.asarray([0, 0, 1, 1])
    X_query = np.asarray([[0.1], [9.9]], dtype=np.float32)
    y_query = np.asarray([0, 1])
    predictor = ContextualTabPFNClassifier(estimator_factory=NearestCentroidClassifier)

    result = run_retrieval_experiment(
        retriever=RandomRetriever(seed=2, global_context=True),
        predictor=predictor,
        X_train_model=X_train,
        X_train_retrieval=X_train,
        y_train=y_train,
        X_query_model=X_query,
        X_query_retrieval=X_query,
        y_query=y_query,
        k=4,
    )

    assert result.actual_k == 4
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["roc_auc"] == 1.0
    assert result.index_seconds >= 0
    assert result.retrieval_seconds >= 0
    assert result.prediction_seconds >= 0


def test_adapter_handles_a_single_class_context_without_fitting_tabpfn():
    predictor = ContextualTabPFNClassifier(
        estimator_factory=lambda: (_ for _ in ()).throw(AssertionError("must not fit"))
    )

    probabilities, classes = predictor.predict_proba_with_contexts(
        X_train=np.asarray([[0.0], [1.0], [10.0], [11.0]]),
        y_train=np.asarray([0, 0, 1, 1]),
        X_query=np.asarray([[0.1], [10.1]]),
        context_indices=np.asarray([[0, 1], [2, 3]]),
    )

    assert classes.tolist() == [0, 1]
    assert probabilities.tolist() == [[1.0, 0.0], [0.0, 1.0]]
