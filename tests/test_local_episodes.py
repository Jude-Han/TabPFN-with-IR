import numpy as np
import pytest

from tabpfn_ir.training import LocalEpisodeSampler, resolve_episode_sizes


def test_episode_sizes_keep_context_and_shorten_query_on_small_folds():
    sizes = resolve_episode_sizes(
        n_train=20,
        requested_context_size=15,
        requested_query_size=1000,
        n_classes=3,
    )

    assert sizes.context_size == 15
    assert sizes.query_size == 5
    assert sizes.total_size == 20


def test_episode_sizes_require_space_for_every_class():
    with pytest.raises(ValueError, match="one row per training class"):
        resolve_episode_sizes(
            n_train=20,
            requested_context_size=2,
            requested_query_size=5,
            n_classes=3,
        )


def test_local_episode_sampling_is_deterministic_and_context_covers_query_labels():
    X = np.asarray([[float(i)] for i in range(12)], dtype=np.float32)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    sampler = LocalEpisodeSampler(
        context_size=5,
        query_size=3,
        seed=17,
        retrieval_batch_size=2,
    ).fit(X, y)

    first = sampler.sample_epoch(epoch=2, n_episodes=4)
    second = sampler.sample_epoch(epoch=2, n_episodes=4)

    assert [episode.anchor_index for episode in first] == [
        episode.anchor_index for episode in second
    ]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.context_indices, right.context_indices)
        np.testing.assert_array_equal(left.query_indices, right.query_indices)
        assert left.context_indices.shape == (5,)
        assert left.query_indices.shape == (3,)
        assert np.unique(left.all_indices).shape[0] == 8
        assert set(np.unique(y[left.query_indices])).issubset(
            set(np.unique(y[left.context_indices]))
        )
        assert set(np.unique(y[left.context_indices])) == {0, 1, 2}


def test_missing_faraway_class_is_added_to_the_training_context():
    X = np.asarray([[0.0], [0.1], [0.2], [0.3], [100.0], [100.1]], dtype=np.float32)
    y = np.asarray([0, 0, 0, 0, 1, 1])
    sampler = LocalEpisodeSampler(
        context_size=3,
        query_size=1,
        seed=3,
    ).fit(X, y)

    episodes = sampler.sample_epoch(epoch=0, n_episodes=6)

    for episode in episodes:
        assert set(y[episode.context_indices].tolist()) == {0, 1}


def test_sampling_rejects_unresolved_sizes():
    X = np.arange(10, dtype=np.float32).reshape(-1, 1)
    y = np.asarray([0, 1] * 5)
    with pytest.raises(ValueError, match="already-resolved sizes"):
        LocalEpisodeSampler(context_size=9, query_size=5).fit(X, y)
