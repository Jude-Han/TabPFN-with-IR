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


class BatchedNearestCentroidClassifier(NearestCentroidClassifier):
    def __init__(self):
        self.batch_sizes = []

    def predict_proba_batched(self, X_train_list, y_train_list, X_test_list):
        self.batch_sizes.append(len(X_train_list))
        assert len({np.asarray(X).shape for X in X_train_list}) == 1
        assert len({np.asarray(X).shape for X in X_test_list}) == 1
        assert len({tuple(np.unique(y).tolist()) for y in y_train_list}) == 1
        outputs = []
        for X_train, y_train, X_test in zip(
            X_train_list,
            y_train_list,
            X_test_list,
            strict=True,
        ):
            local_estimator = NearestCentroidClassifier().fit(X_train, y_train)
            outputs.append(local_estimator.predict_proba(X_test))
        return np.stack(outputs)


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
    assert result.unique_contexts == 1
    assert result.sequential_contexts == 1
    assert not result.used_batched_inference


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


def test_adapter_fuses_query_specific_contexts_in_bounded_batches():
    estimator = BatchedNearestCentroidClassifier()
    predictor = ContextualTabPFNClassifier(
        estimator_factory=lambda: estimator,
        context_batch_size=2,
    )
    X_train = np.asarray([[0.0], [10.0], [0.2], [9.8], [0.4], [9.6]])
    y_train = np.asarray([0, 1, 0, 1, 0, 1])

    probabilities, classes = predictor.predict_proba_with_contexts(
        X_train=X_train,
        y_train=y_train,
        X_query=np.asarray([[0.1], [9.9], [0.3], [9.7], [5.0]]),
        context_indices=np.asarray(
            [
                [0, 1],
                [2, 3],
                [4, 5],
                [0, 3],
                [2, 5],
            ]
        ),
    )

    assert classes.tolist() == [0, 1]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.batch_sizes == [2, 2, 1]
    assert predictor.last_inference_stats.unique_contexts == 5
    assert predictor.last_inference_stats.batched_contexts == 5
    assert predictor.last_inference_stats.sequential_contexts == 0
    assert predictor.last_inference_stats.context_batches == 3
    assert predictor.last_inference_stats.used_batched_inference


def test_adapter_groups_batched_contexts_by_local_class_set():
    estimator = BatchedNearestCentroidClassifier()
    predictor = ContextualTabPFNClassifier(
        estimator_factory=lambda: estimator,
        context_batch_size=8,
    )

    probabilities, classes = predictor.predict_proba_with_contexts(
        X_train=np.asarray([[0.0], [5.0], [0.2], [5.2], [10.0], [10.2]]),
        y_train=np.asarray([0, 1, 0, 1, 2, 2]),
        X_query=np.asarray([[0.1], [5.1], [7.5], [9.9]]),
        context_indices=np.asarray([[0, 1], [2, 3], [1, 4], [3, 5]]),
    )

    assert classes.tolist() == [0, 1, 2]
    assert estimator.batch_sizes == [2, 2]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[:2, 2], 0.0)
    np.testing.assert_allclose(probabilities[2:, 0], 0.0)


def test_adapter_can_disable_batched_context_inference():
    estimator = BatchedNearestCentroidClassifier()
    predictor = ContextualTabPFNClassifier(
        estimator_factory=lambda: estimator,
        use_batched_contexts=False,
    )

    predictor.predict_proba_with_contexts(
        X_train=np.asarray([[0.0], [10.0], [0.2], [9.8]]),
        y_train=np.asarray([0, 1, 0, 1]),
        X_query=np.asarray([[0.1], [9.9]]),
        context_indices=np.asarray([[0, 1], [2, 3]]),
    )

    assert estimator.batch_sizes == []
    assert predictor.last_inference_stats.sequential_contexts == 2
    assert not predictor.last_inference_stats.used_batched_inference
