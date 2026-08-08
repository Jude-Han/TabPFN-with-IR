import numpy as np
import pytest

from tabpfn_ir.retrieval import (
    FullContextRetriever,
    KNNRetriever,
    RandomRetriever,
    context_size_from_ratio,
    localpfn_context_size,
)


@pytest.fixture
def training_data():
    X = np.asarray([[0.0], [1.0], [5.0], [10.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1])
    return X, y


def test_full_context_returns_every_training_row(training_data):
    X, y = training_data
    result = FullContextRetriever().fit(X, y).retrieve(np.asarray([[2.0], [3.0]]), None)
    np.testing.assert_array_equal(result.indices, [[0, 1, 2, 3], [0, 1, 2, 3]])


def test_random_global_context_is_shared_and_reproducible(training_data):
    X, y = training_data
    query = np.asarray([[2.0], [3.0]])
    retriever = RandomRetriever(seed=7, global_context=True).fit(X, y)
    first = retriever.retrieve(query, 2)
    second = retriever.retrieve(query, 2)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.indices[0], first.indices[1])
    assert np.unique(first.indices[0]).shape[0] == 2


def test_knn_returns_query_specific_nearest_rows(training_data):
    X, y = training_data
    retriever = KNNRetriever(query_batch_size=1).fit(X, y)
    result = retriever.retrieve(np.asarray([[0.2], [8.0]], dtype=np.float64), 2)
    np.testing.assert_array_equal(result.indices[0], [0, 1])
    np.testing.assert_array_equal(result.indices[1], [3, 2])
    assert type(retriever._index).__name__ == "IndexFlatL2"
    np.testing.assert_allclose(result.scores[0], [-0.04, -0.64], atol=1e-6)
    assert np.all(result.scores <= 0)


def test_context_budget_is_capped_at_training_size(training_data):
    X, y = training_data
    result = RandomRetriever(seed=1).fit(X, y).retrieve(np.asarray([[0.0]]), 100)
    assert result.indices.shape == (1, X.shape[0])


def test_localpfn_context_size_implements_paper_heuristic():
    assert localpfn_context_size(25) == 50
    assert localpfn_context_size(10_000) == 1000
    assert localpfn_context_size(1_000_000) == 1000


def test_context_ratio_uses_ceiling_and_validates_range():
    assert context_size_from_ratio(0.10, 101) == 11
    assert context_size_from_ratio(0.001, 10) == 1
    assert context_size_from_ratio(1.0, 10) == 10
    with pytest.raises(ValueError):
        context_size_from_ratio(0, 10)
    with pytest.raises(ValueError):
        context_size_from_ratio(1.01, 10)
