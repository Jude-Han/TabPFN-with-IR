"""Adapter for running frozen TabPFN with query-specific contexts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from tabpfn_ir.environment import load_project_dotenv
from tabpfn_ir.models.configuration import validate_model_version


class ProbabilisticClassifier(Protocol):
    """The minimal estimator API used by the adapter."""

    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ContextInferenceStats:
    """Diagnostics for one query-specific context prediction call."""

    unique_contexts: int = 0
    single_class_contexts: int = 0
    batched_contexts: int = 0
    sequential_contexts: int = 0
    context_batches: int = 0
    used_batched_inference: bool = False


class ContextualTabPFNClassifier:
    """Run frozen TabPFN inference over retrieved, query-specific contexts.

    Full and global-random contexts are fitted once and shared across their query
    batches. Modern TabPFN versions fuse compatible query-specific contexts with
    ``predict_proba_batched``. The isolated v1 backend automatically falls back
    to sequential context inference. TabPFN itself remains frozen: context
    fitting supplies ICL examples and does not perform supervised gradient updates.
    """

    def __init__(
        self,
        *,
        estimator_factory: Callable[[], ProbabilisticClassifier] | None = None,
        tabpfn_kwargs: dict[str, Any] | None = None,
        model_version: str = "v2.6",
        v1_runtime_path: str | None = None,
        v1_checkpoint_path: str | None = None,
        context_batch_size: int = 32,
        use_batched_contexts: bool = True,
    ) -> None:
        validate_model_version(model_version)
        if context_batch_size <= 0:
            raise ValueError("context_batch_size must be positive.")
        self._estimator_factory = estimator_factory
        self._tabpfn_kwargs = dict(tabpfn_kwargs or {})
        if "model_path" in self._tabpfn_kwargs:
            raise ValueError(
                "model_path cannot override the selected official TabPFN checkpoint. "
                "Use v1_checkpoint_path for an original-v1 .cpkt file."
            )
        self.model_version = model_version
        self.v1_runtime_path = v1_runtime_path
        self.v1_checkpoint_path = v1_checkpoint_path
        self.context_batch_size = context_batch_size
        self.use_batched_contexts = use_batched_contexts
        self.last_inference_stats = ContextInferenceStats()

    def _new_estimator(self) -> ProbabilisticClassifier:
        if self._estimator_factory is not None:
            return self._estimator_factory()
        # Also support package users who instantiate the adapter without a CLI.
        # Exported variables remain authoritative because override=False.
        load_project_dotenv(override=False)
        if self.model_version == "v1":
            from tabpfn_ir.models.tabpfn_v1 import LegacyTabPFNClassifier

            return LegacyTabPFNClassifier(
                runtime_path=self.v1_runtime_path,
                checkpoint_path=self.v1_checkpoint_path,
                **self._tabpfn_kwargs,
            )
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Modern TabPFN support is optional. Install the project with "
                "`pip install -e '.[benchmark]'`."
            ) from exc
        selected_devices = self._tabpfn_kwargs.get("device")
        if isinstance(selected_devices, (list, tuple)) and len(selected_devices) > 1:
            try:
                from tabpfn.utils import infer_devices
            except ImportError as exc:  # pragma: no cover - depends on TabPFN version
                raise RuntimeError(
                    "The installed TabPFN does not expose its multi-GPU device API. "
                    "Install the pinned release with "
                    "`pip install --upgrade 'tabpfn==8.2.0'`."
                ) from exc
            infer_devices(selected_devices)
        selected_version = {
            "v2.6": ModelVersion.V2_6,
            "v3": ModelVersion.V3,
        }[self.model_version]
        return TabPFNClassifier.create_default_for_version(
            selected_version,
            **self._tabpfn_kwargs,
        )

    @staticmethod
    def _close_estimator(estimator: ProbabilisticClassifier) -> None:
        close = getattr(estimator, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _align_probabilities(
        probabilities: np.ndarray,
        local_classes: np.ndarray,
        global_classes: np.ndarray,
    ) -> np.ndarray:
        aligned = np.zeros((probabilities.shape[0], global_classes.shape[0]), dtype=float)
        global_positions = {
            label: position for position, label in enumerate(global_classes.tolist())
        }
        for local_position, label in enumerate(np.asarray(local_classes).tolist()):
            aligned[:, global_positions[label]] = probabilities[:, local_position]
        return aligned

    def _predict_contexts_sequentially(
        self,
        *,
        estimator: ProbabilisticClassifier,
        contexts: list[tuple[np.ndarray, list[int], np.ndarray]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_query: np.ndarray,
        global_classes: np.ndarray,
        output: np.ndarray,
    ) -> int:
        for selected, query_positions, _ in contexts:
            estimator.fit(X_train[selected], y_train[selected])
            local_probabilities = np.asarray(
                estimator.predict_proba(X_query[query_positions]),
                dtype=float,
            )
            output[query_positions] = self._align_probabilities(
                local_probabilities,
                np.asarray(estimator.classes_),
                global_classes,
            )
        return len(contexts)

    def _predict_contexts_batched(
        self,
        *,
        estimator: ProbabilisticClassifier,
        contexts: list[tuple[np.ndarray, list[int], np.ndarray]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_query: np.ndarray,
        global_classes: np.ndarray,
        output: np.ndarray,
    ) -> int:
        """Fuse shape- and class-compatible contexts in bounded batches."""

        predict_batched = estimator.predict_proba_batched
        compatible_groups: dict[
            tuple[int, int, tuple[object, ...]],
            list[tuple[np.ndarray, list[int], np.ndarray]],
        ] = defaultdict(list)
        for selected, query_positions, context_classes in contexts:
            key = (
                selected.shape[0],
                len(query_positions),
                tuple(context_classes.tolist()),
            )
            compatible_groups[key].append((selected, query_positions, context_classes))

        n_batches = 0
        for compatible_contexts in compatible_groups.values():
            for start in range(0, len(compatible_contexts), self.context_batch_size):
                chunk = compatible_contexts[start : start + self.context_batch_size]
                probabilities = np.asarray(
                    predict_batched(
                        [X_train[selected] for selected, _, _ in chunk],
                        [y_train[selected] for selected, _, _ in chunk],
                        [X_query[query_positions] for _, query_positions, _ in chunk],
                    ),
                    dtype=float,
                )
                expected_shape = (
                    len(chunk),
                    len(chunk[0][1]),
                    chunk[0][2].shape[0],
                )
                if probabilities.shape != expected_shape:
                    raise RuntimeError(
                        "TabPFN predict_proba_batched returned shape "
                        f"{probabilities.shape}, expected {expected_shape}."
                    )
                for local_probabilities, (_, query_positions, context_classes) in zip(
                    probabilities,
                    chunk,
                    strict=True,
                ):
                    output[query_positions] = self._align_probabilities(
                        local_probabilities,
                        context_classes,
                        global_classes,
                    )
                n_batches += 1
        return n_batches

    def predict_proba_with_contexts(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_query: np.ndarray,
        context_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict probabilities for query-specific retrieved row indices."""

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_query = np.asarray(X_query)
        context_indices = np.asarray(context_indices)
        if context_indices.ndim != 2 or context_indices.shape[0] != X_query.shape[0]:
            raise ValueError("context_indices must have shape [n_query, k].")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if context_indices.size and (
            context_indices.min() < 0 or context_indices.max() >= X_train.shape[0]
        ):
            raise IndexError("context_indices contains a row outside the training fold.")

        global_classes = np.unique(y_train)
        output = np.zeros((X_query.shape[0], global_classes.shape[0]), dtype=float)
        grouped_queries: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for query_position, row_indices in enumerate(context_indices):
            grouped_queries[tuple(int(index) for index in row_indices)].append(query_position)

        global_positions = {
            label: position for position, label in enumerate(global_classes.tolist())
        }
        multi_class_contexts: list[tuple[np.ndarray, list[int], np.ndarray]] = []
        single_class_contexts = 0
        for context, query_positions in grouped_queries.items():
            selected = np.asarray(context, dtype=np.int64)
            context_classes = np.unique(y_train[selected])
            if context_classes.shape[0] == 1:
                output[query_positions, global_positions[context_classes[0]]] = 1.0
                single_class_contexts += 1
                continue
            multi_class_contexts.append((selected, query_positions, context_classes))

        batched_contexts = 0
        sequential_contexts = 0
        context_batches = 0
        used_batched_inference = False
        if multi_class_contexts:
            estimator = self._new_estimator()
            try:
                predict_batched = getattr(estimator, "predict_proba_batched", None)
                can_batch = (
                    self.use_batched_contexts
                    and len(multi_class_contexts) > 1
                    and callable(predict_batched)
                )
                if can_batch:
                    context_batches = self._predict_contexts_batched(
                        estimator=estimator,
                        contexts=multi_class_contexts,
                        X_train=X_train,
                        y_train=y_train,
                        X_query=X_query,
                        global_classes=global_classes,
                        output=output,
                    )
                    batched_contexts = len(multi_class_contexts)
                    used_batched_inference = True
                else:
                    sequential_contexts = self._predict_contexts_sequentially(
                        estimator=estimator,
                        contexts=multi_class_contexts,
                        X_train=X_train,
                        y_train=y_train,
                        X_query=X_query,
                        global_classes=global_classes,
                        output=output,
                    )
            finally:
                self._close_estimator(estimator)

        self.last_inference_stats = ContextInferenceStats(
            unique_contexts=len(grouped_queries),
            single_class_contexts=single_class_contexts,
            batched_contexts=batched_contexts,
            sequential_contexts=sequential_contexts,
            context_batches=context_batches,
            used_batched_inference=used_batched_inference,
        )
        return output, global_classes
