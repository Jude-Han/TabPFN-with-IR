"""LoCalPFN-style local context/query episode construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabpfn_ir.retrieval import KNNRetriever


@dataclass(frozen=True)
class ResolvedEpisodeSizes:
    """Effective episode sizes after respecting the training-fold size."""

    context_size: int
    query_size: int

    @property
    def total_size(self) -> int:
        return self.context_size + self.query_size


@dataclass(frozen=True)
class LocalEpisode:
    """Indices for one local meta-dataset used by the fine-tuning loop."""

    anchor_index: int
    context_indices: np.ndarray
    query_indices: np.ndarray

    @property
    def all_indices(self) -> np.ndarray:
        """Return context first and query second, as expected by the splitter."""

        return np.concatenate((self.context_indices, self.query_indices))


def resolve_episode_sizes(
    *,
    n_train: int,
    requested_context_size: int,
    requested_query_size: int,
    n_classes: int,
) -> ResolvedEpisodeSizes:
    """Resolve a feasible local episode without silently dropping the context.

    The inference context budget has priority. On a small fold, the query portion
    is shortened before the context is shortened. One row is always retained for
    the supervised query loss. Every class must fit in the context because the
    pinned TabPFN v2.6 preprocessing builds a class encoding per meta-dataset.
    """

    if n_train < 2:
        raise ValueError("Local fine-tuning requires at least two training rows.")
    if requested_context_size <= 0:
        raise ValueError("requested_context_size must be positive.")
    if requested_query_size <= 0:
        raise ValueError("requested_query_size must be positive.")
    if n_classes <= 1:
        raise ValueError("Local classification fine-tuning requires at least two classes.")

    context_size = min(requested_context_size, n_train - 1)
    if context_size < n_classes:
        raise ValueError(
            "The local context must contain at least one row per training class: "
            f"context_size={context_size}, n_classes={n_classes}. Increase --k."
        )
    query_size = min(requested_query_size, n_train - context_size)
    return ResolvedEpisodeSizes(context_size=context_size, query_size=query_size)


class LocalEpisodeSampler:
    """Sample deterministic, query-supervised neighborhoods around random anchors.

    A global exact-L2 FAISS index supplies the nearest rows. Each retrieved set is
    shuffled and split into a context and a query set. The context is guaranteed
    to contain every class in the training fold. This small compatibility step is
    needed because TabPFN 8.2 creates an episode-local label encoder; it also makes
    every query label representable by its context.
    """

    def __init__(
        self,
        *,
        context_size: int,
        query_size: int,
        seed: int = 0,
        retrieval_batch_size: int = 512,
    ) -> None:
        if context_size <= 0:
            raise ValueError("context_size must be positive.")
        if query_size <= 0:
            raise ValueError("query_size must be positive.")
        if retrieval_batch_size <= 0:
            raise ValueError("retrieval_batch_size must be positive.")
        self.context_size = context_size
        self.query_size = query_size
        self.seed = seed
        self.retrieval_batch_size = retrieval_batch_size
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._retriever: KNNRetriever | None = None
        self._class_indices: list[np.ndarray] | None = None
        self._class_retrievers: list[KNNRetriever] | None = None

    def fit(self, X_retrieval: np.ndarray, y: np.ndarray) -> LocalEpisodeSampler:
        """Build the global and per-class exact FAISS indexes once."""

        X_retrieval = np.asarray(X_retrieval)
        y = np.asarray(y)
        if X_retrieval.ndim != 2:
            raise ValueError("X_retrieval must be a two-dimensional array.")
        if y.ndim != 1 or y.shape[0] != X_retrieval.shape[0]:
            raise ValueError("y must be one-dimensional and aligned with X_retrieval.")
        classes = np.unique(y)
        sizes = resolve_episode_sizes(
            n_train=y.shape[0],
            requested_context_size=self.context_size,
            requested_query_size=self.query_size,
            n_classes=classes.shape[0],
        )
        if sizes.context_size != self.context_size or sizes.query_size != self.query_size:
            raise ValueError(
                "LocalEpisodeSampler expects already-resolved sizes; call "
                "resolve_episode_sizes before construction."
            )

        self._X = np.ascontiguousarray(X_retrieval, dtype=np.float32)
        self._y = y.copy()
        self._classes = classes
        self._retriever = KNNRetriever(query_batch_size=self.retrieval_batch_size).fit(
            self._X,
            self._y,
        )
        self._class_indices = []
        self._class_retrievers = []
        for label in classes:
            global_indices = np.flatnonzero(self._y == label)
            self._class_indices.append(global_indices)
            self._class_retrievers.append(
                KNNRetriever(query_batch_size=self.retrieval_batch_size).fit(
                    self._X[global_indices],
                    self._y[global_indices],
                )
            )
        return self

    def _require_fitted(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, KNNRetriever]:
        if (
            self._X is None
            or self._y is None
            or self._classes is None
            or self._retriever is None
            or self._class_indices is None
            or self._class_retrievers is None
        ):
            raise RuntimeError("Call fit before sampling local episodes.")
        return self._X, self._y, self._classes, self._retriever

    def _add_missing_classes(
        self,
        *,
        anchor_features: np.ndarray,
        neighbor_indices: np.ndarray,
    ) -> np.ndarray:
        """Replace far neighbors with the nearest row of each missing class."""

        _, y, classes, _ = self._require_fitted()
        assert self._class_indices is not None
        assert self._class_retrievers is not None

        selected = np.asarray(neighbor_indices, dtype=np.int64).copy()
        selected_labels = y[selected]
        counts = {label: int(np.count_nonzero(selected_labels == label)) for label in classes}
        missing_positions = [
            position for position, label in enumerate(classes) if counts[label] == 0
        ]

        for class_position in missing_positions:
            class_retrieval = self._class_retrievers[class_position].retrieve(
                anchor_features.reshape(1, -1),
                1,
            )
            replacement = int(self._class_indices[class_position][class_retrieval.indices[0, 0]])

            replace_position = None
            for candidate_position in range(selected.shape[0] - 1, -1, -1):
                candidate_label = y[selected[candidate_position]]
                if counts[candidate_label] > 1:
                    replace_position = candidate_position
                    break
            if replace_position is None:  # guarded by total_size >= n_classes
                raise RuntimeError("Could not create an all-class local episode.")

            previous_label = y[selected[replace_position]]
            counts[previous_label] -= 1
            selected[replace_position] = replacement
            counts[classes[class_position]] += 1

        if np.unique(selected).shape[0] != selected.shape[0]:
            raise RuntimeError("Local episode construction produced duplicate rows.")
        return selected

    def _split_context_query(
        self,
        neighbor_indices: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle an episode while reserving one example per class for context."""

        _, y, classes, _ = self._require_fitted()
        shuffled = np.asarray(neighbor_indices, dtype=np.int64)[
            rng.permutation(neighbor_indices.shape[0])
        ]
        reserved_positions = []
        for label in classes:
            label_positions = np.flatnonzero(y[shuffled] == label)
            if label_positions.size == 0:
                raise RuntimeError("Every local episode must contain every training class.")
            reserved_positions.append(int(label_positions[0]))

        reserved_mask = np.zeros(shuffled.shape[0], dtype=bool)
        reserved_mask[reserved_positions] = True
        reserved = shuffled[reserved_mask]
        remaining = shuffled[~reserved_mask]
        n_fill = self.context_size - reserved.shape[0]
        context = np.concatenate((reserved, remaining[:n_fill]))
        context = context[rng.permutation(context.shape[0])]
        query = remaining[n_fill:]

        if context.shape[0] != self.context_size or query.shape[0] != self.query_size:
            raise RuntimeError("Local episode split has an unexpected size.")
        if not np.isin(np.unique(y[query]), np.unique(y[context])).all():
            raise RuntimeError("A local query label is missing from its context.")
        return context, query

    def sample_epoch(self, *, epoch: int, n_episodes: int) -> list[LocalEpisode]:
        """Create all local episodes for one epoch deterministically."""

        X, _, _, retriever = self._require_fitted()
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        if n_episodes <= 0:
            raise ValueError("n_episodes must be positive.")

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
        anchors = rng.choice(
            X.shape[0],
            size=n_episodes,
            replace=n_episodes > X.shape[0],
        ).astype(np.int64, copy=False)
        total_size = self.context_size + self.query_size
        retrieved = retriever.retrieve(X[anchors], total_size)

        episodes = []
        for anchor, neighbors in zip(anchors, retrieved.indices, strict=True):
            complete_neighbors = self._add_missing_classes(
                anchor_features=X[anchor],
                neighbor_indices=neighbors,
            )
            context, query = self._split_context_query(complete_neighbors, rng=rng)
            episodes.append(
                LocalEpisode(
                    anchor_index=int(anchor),
                    context_indices=context,
                    query_indices=query,
                )
            )
        return episodes
