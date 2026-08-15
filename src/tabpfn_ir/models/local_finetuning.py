"""LoCalPFN-style task fine-tuning on the pinned TabPFN v2.6 backbone.

The public TabPFN 8.2 fine-tuner already owns the optimizer, mixed precision,
checkpoint, resume, DDP, and early-stopping machinery. This module supplies a
narrow adapter that replaces its global random chunks with local kNN episodes
and evaluates the changing weights with exact query-specific kNN contexts.
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any, Literal

import numpy as np

from tabpfn_ir.models.tabpfn_adapter import (
    ContextInferenceStats,
    ContextualTabPFNClassifier,
)
from tabpfn_ir.retrieval import KNNRetriever
from tabpfn_ir.training import LocalEpisodeSampler, resolve_episode_sizes

SUPPORTED_TABPFN_VERSION = "8.2.0"
_DATASET_BUILDER_LOCK = threading.RLock()


@dataclass(frozen=True)
class LocalPredictionResult:
    """Probabilities plus retrieval and inference diagnostics."""

    probabilities: np.ndarray
    classes: np.ndarray
    actual_k: int
    index_seconds: float
    retrieval_seconds: float
    prediction_seconds: float
    inference_stats: ContextInferenceStats


def _require_supported_tabpfn() -> None:
    try:
        installed = version("tabpfn")
    except PackageNotFoundError as exc:
        raise ImportError(
            "Local fine-tuning requires the benchmark dependencies. Install them with "
            "`pip install -e '.[benchmark]'`."
        ) from exc
    if installed != SUPPORTED_TABPFN_VERSION:
        raise RuntimeError(
            "The local fine-tuning adapter targets TabPFN's private training hook in "
            f"version {SUPPORTED_TABPFN_VERSION}, but version {installed} is installed. "
            "Install the pinned project dependencies before training."
        )


def _ordered_episode_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    context_size: int,
    stratify: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Split an already ordered [context, query] local meta-dataset."""

    del stratify
    return [
        X[:context_size],
        X[context_size:],
        y[:context_size],
        y[context_size:],
    ]


@contextmanager
def _replace_tabpfn_dataset_builder(builder: Any) -> Iterator[None]:
    """Temporarily install the local episode builder in TabPFN's v8.2 loop."""

    from tabpfn.finetuning import finetuned_base

    with _DATASET_BUILDER_LOCK:
        original = finetuned_base.get_preprocessed_dataset_chunks
        finetuned_base.get_preprocessed_dataset_chunks = builder
        try:
            yield
        finally:
            finetuned_base.get_preprocessed_dataset_chunks = original


@contextmanager
def _backport_v26_activation_checkpointing(block_class: type[Any] | None = None) -> Iterator[None]:
    """Keep v8.2.0 checkpoint inputs reusable during backward recomputation.

    TabPFN 8.2.0 passes a length-one list into ``torch.utils.checkpoint`` and
    ``TabPFNBlock.forward`` consumes that list with ``pop(0)``. The non-reentrant
    checkpoint implementation invokes the block again during backward with the
    same, now-empty list. Upstream fixed this by passing the tensor directly.

    This scoped backport preserves the external list and lets the original block
    consume a fresh shallow copy on each forward/recompute call. It is equivalent
    for the contained tensor and avoids modifying the installed TabPFN package.
    """

    if block_class is None:
        from tabpfn.architectures.tabpfn_v2_6 import TabPFNBlock

        block_class = TabPFNBlock

    original_forward = block_class.forward

    @wraps(original_forward)
    def forward_without_consuming_checkpoint_input(
        block: Any,
        state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        reusable_state = list(state) if isinstance(state, list) else state
        return original_forward(block, reusable_state, *args, **kwargs)

    with _DATASET_BUILDER_LOCK:
        block_class.forward = forward_without_consuming_checkpoint_input
        try:
            yield
        finally:
            block_class.forward = original_forward


class LocalFinetunedTabPFNClassifier:
    """Fine-tune every TabPFN v2.6 weight on LoCalPFN-style local episodes.

    This is a task-specific classifier, not a frozen ICL adapter. Its delegate is
    the official ``FinetunedTabPFNClassifier`` from TabPFN 8.2.0. Only dataset
    construction, batched episode loss, local validation, and local inference are
    adapted here.
    """

    def __init__(
        self,
        *,
        context_size: int,
        train_query_size: int = 1000,
        steps_per_epoch: int = 30,
        episode_batch_size: int = 2,
        retrieval_batch_size: int = 512,
        context_batch_size: int = 32,
        device: str = "cuda",
        epochs: int = 30,
        time_limit: int | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 2,
        use_activation_checkpointing: bool = True,
        save_checkpoint_interval: int | None = 10,
        use_fixed_preprocessing_seed: bool = True,
        eval_metric: Literal["roc_auc", "log_loss"] = "roc_auc",
        experiment_logger: Any | None = None,
        extra_classifier_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _require_supported_tabpfn()
        if context_size <= 0:
            raise ValueError("context_size must be positive.")
        if train_query_size <= 0:
            raise ValueError("train_query_size must be positive.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if episode_batch_size <= 0:
            raise ValueError("episode_batch_size must be positive.")
        if retrieval_batch_size <= 0 or context_batch_size <= 0:
            raise ValueError("Retrieval and context batch sizes must be positive.")
        if random_state < 0:
            raise ValueError("random_state must be non-negative.")
        if extra_classifier_kwargs and "model_path" in extra_classifier_kwargs:
            raise ValueError(
                "model_path cannot override the pinned Hugging Face TabPFN v2.6 checkpoint."
            )

        from tabpfn.constants import ModelVersion
        from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

        self.requested_context_size = context_size
        self.requested_train_query_size = train_query_size
        self.steps_per_epoch = steps_per_epoch
        self.episode_batch_size = episode_batch_size
        self.retrieval_batch_size = retrieval_batch_size
        self.context_batch_size = context_batch_size
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.experiment_logger = experiment_logger
        self.use_activation_checkpointing = use_activation_checkpointing
        self.context_size_: int | None = None
        self.train_query_size_: int | None = None
        self._sampler: LocalEpisodeSampler | None = None
        self._X_train_model: np.ndarray | None = None
        self._X_train_retrieval: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._validation_context_indices: np.ndarray | None = None
        self._final_inference_config: dict[str, Any] | None = None

        self.delegate = FinetunedTabPFNClassifier(
            device=device,
            epochs=epochs,
            time_limit=time_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split_ratio=None,
            n_finetune_ctx_plus_query_samples=context_size + train_query_size,
            finetune_ctx_query_split_ratio=(train_query_size / (context_size + train_query_size)),
            n_inference_subsample_samples=None,
            random_state=random_state,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            grad_clip_value=grad_clip_value,
            use_lr_scheduler=use_lr_scheduler,
            lr_warmup_only=lr_warmup_only,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            use_activation_checkpointing=use_activation_checkpointing,
            save_checkpoint_interval=save_checkpoint_interval,
            use_fixed_preprocessing_seed=use_fixed_preprocessing_seed,
            extra_classifier_kwargs=copy.deepcopy(extra_classifier_kwargs),
            eval_metric=eval_metric,
            experiment_logger=experiment_logger,
            model_version=ModelVersion.V2_6,
        )
        # TabPFN 8.2 hard-codes one meta-dataset per step in the stock wrapper.
        # The batched executor itself supports independent, equal-shaped datasets,
        # so the generalized loss below enables the paper's B=2 episode batch.
        self.delegate.meta_batch_size = episode_batch_size
        self.delegate._forward_with_loss = MethodType(  # type: ignore[method-assign]
            self._forward_with_local_batch_loss,
            self.delegate,
        )
        self.delegate._evaluate_model = MethodType(  # type: ignore[method-assign]
            self._evaluate_with_local_contexts,
            self.delegate,
        )
        self._delegate_log_epoch_evaluation = self.delegate._log_epoch_evaluation
        self.delegate._log_epoch_evaluation = MethodType(  # type: ignore[method-assign]
            self._log_epoch_evaluation_with_initial_metric,
            self.delegate,
        )
        self.delegate._setup_inference_model = MethodType(  # type: ignore[method-assign]
            self._capture_final_inference_config,
            self.delegate,
        )

    def _log_epoch_evaluation_with_initial_metric(
        self,
        delegate: Any,
        epoch: int,
        eval_result: Any,
        mean_train_loss: float | None,
    ) -> None:
        """Preserve TabPFN logging and expose its otherwise-unlogged initial metric."""

        del delegate
        self._delegate_log_epoch_evaluation(epoch, eval_result, mean_train_loss)
        if epoch == -1 and self.experiment_logger is not None:
            metrics: dict[str, float] = {
                "train/epoch": -1.0,
                "val/primary_metric": float(eval_result.primary),
            }
            for name, value in eval_result.secondary.items():
                metrics[f"val/{name}"] = float(value)
            self.experiment_logger.log_epoch(metrics, step=0)

    def _forward_with_local_batch_loss(self, delegate: Any, batch: Any) -> Any:
        import torch

        logits_QBEL = delegate._training_forward(
            batch.X_query,
            return_raw_logits=True,
        )
        n_query, batch_size, n_estimators, n_classes = logits_QBEL.shape
        if tuple(batch.y_query.shape) != (batch_size, n_query):
            raise RuntimeError(
                "Unexpected local query target shape: "
                f"{tuple(batch.y_query.shape)} versus {(batch_size, n_query)}."
            )
        if n_estimators != delegate.n_estimators_finetune:
            raise RuntimeError("TabPFN returned an unexpected estimator dimension.")
        if n_classes != delegate.finetuned_estimator_.n_classes_:
            raise RuntimeError("TabPFN returned an unexpected class dimension.")

        logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(
            batch_size * n_estimators,
            n_classes,
            n_query,
        )
        targets_BQ = (
            batch.y_query[:, None, :]
            .expand(batch_size, n_estimators, n_query)
            .reshape(batch_size * n_estimators, n_query)
            .to(delegate.device)
        )
        return torch.nn.functional.cross_entropy(logits_BLQ, targets_BQ)

    def _capture_final_inference_config(
        self,
        delegate: Any,
        final_inference_eval_config: dict[str, Any],
    ) -> None:
        del delegate
        self._final_inference_config = copy.deepcopy(final_inference_eval_config)

    def _new_context_estimator(self, config: dict[str, Any]) -> Any:
        from tabpfn import TabPFNClassifier
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        if not hasattr(self.delegate, "finetuned_estimator_"):
            raise RuntimeError("Call fit before constructing a fine-tuned estimator.")
        return clone_model_for_evaluation(
            self.delegate.finetuned_estimator_,
            copy.deepcopy(config),
            TabPFNClassifier,
        )

    def _predict_from_context_indices(
        self,
        *,
        X_train_model: np.ndarray,
        y_train: np.ndarray,
        X_query_model: np.ndarray,
        context_indices: np.ndarray,
        estimator_config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, ContextInferenceStats]:
        predictor = ContextualTabPFNClassifier(
            estimator_factory=lambda: self._new_context_estimator(estimator_config),
            model_version="v2.6",
            context_batch_size=self.context_batch_size,
            use_batched_contexts=True,
        )
        probabilities, classes = predictor.predict_proba_with_contexts(
            X_train_model,
            y_train,
            X_query_model,
            context_indices,
        )
        return probabilities, classes, predictor.last_inference_stats

    def _evaluate_with_local_contexts(
        self,
        delegate: Any,
        eval_config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        from tabpfn.finetuning.finetuned_base import EvalResult

        from tabpfn_ir.evaluation.metrics import classification_metrics

        del delegate
        if self._validation_context_indices is None:
            raise RuntimeError("Validation retrieval contexts were not initialized.")
        probabilities, classes, _ = self._predict_from_context_indices(
            X_train_model=np.asarray(X_train),
            y_train=np.asarray(y_train),
            X_query_model=np.asarray(X_val),
            context_indices=self._validation_context_indices,
            estimator_config=eval_config,
        )
        metrics = classification_metrics(
            np.asarray(y_val),
            probabilities,
            classes,
            auc_mode="ovo",
        )
        primary = metrics["roc_auc"] if self.eval_metric == "roc_auc" else -metrics["log_loss"]
        return EvalResult(
            primary=primary,
            secondary={
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc"],
            },
        )

    def _make_local_dataset_builder(self, original_builder: Any) -> Any:
        if self._sampler is None or self.context_size_ is None:
            raise RuntimeError("Local episode sampling was not initialized.")

        def build_local_datasets(
            calling_instance: Any,
            X_raw: Any,
            y_raw: Any,
            split_fn: Any,
            max_data_size: int | None,
            model_type: str,
            *,
            equal_split_size: bool,
            data_shuffle_seed: int,
            preprocessing_random_state: Any,
            shuffle: bool = True,
            force_no_stratify: bool = False,
        ) -> Any:
            del split_fn, max_data_size, equal_split_size, shuffle, force_no_stratify
            if model_type != "classifier":
                raise ValueError("Local fine-tuning currently supports classification only.")
            X_values = np.asarray(X_raw)
            y_values = np.asarray(y_raw)
            if self._X_train_retrieval is None or X_values.shape[0] != len(self._X_train_retrieval):
                raise RuntimeError("The local retrieval view is not aligned with training rows.")

            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            n_episodes = self.steps_per_epoch * self.episode_batch_size * world_size
            episodes = self._sampler.sample_epoch(
                epoch=int(data_shuffle_seed),
                n_episodes=n_episodes,
            )
            ordered_indices = [episode.all_indices for episode in episodes]
            local_X = [X_values[indices] for indices in ordered_indices]
            local_y = [y_values[indices] for indices in ordered_indices]
            return original_builder(
                calling_instance=calling_instance,
                X_raw=local_X,
                y_raw=local_y,
                split_fn=lambda X, y, stratify=None: _ordered_episode_split(
                    X,
                    y,
                    context_size=self.context_size_,
                    stratify=stratify,
                ),
                max_data_size=None,
                model_type="classifier",
                equal_split_size=False,
                data_shuffle_seed=data_shuffle_seed,
                preprocessing_random_state=preprocessing_random_state,
                shuffle=False,
                force_no_stratify=True,
            )

        return build_local_datasets

    def fit(
        self,
        X_train_model: np.ndarray,
        y_train: np.ndarray,
        *,
        X_train_retrieval: np.ndarray,
        X_val_model: np.ndarray,
        y_val: np.ndarray,
        X_val_retrieval: np.ndarray,
        output_dir: Path | None = None,
    ) -> LocalFinetunedTabPFNClassifier:
        """Fine-tune on train-only local episodes and early-stop on validation."""

        X_train_model = np.asarray(X_train_model)
        X_train_retrieval = np.asarray(X_train_retrieval)
        X_val_model = np.asarray(X_val_model)
        X_val_retrieval = np.asarray(X_val_retrieval)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        if y_train.ndim != 1 or y_val.ndim != 1:
            raise ValueError("Training and validation labels must be one-dimensional.")
        if X_train_model.ndim != 2 or X_train_retrieval.ndim != 2:
            raise ValueError("Training model and retrieval views must be two-dimensional.")
        if X_val_model.ndim != 2 or X_val_retrieval.ndim != 2:
            raise ValueError("Validation model and retrieval views must be two-dimensional.")
        if X_train_model.shape[0] != y_train.shape[0]:
            raise ValueError("Training model rows and labels are not aligned.")
        if X_train_retrieval.shape[0] != y_train.shape[0]:
            raise ValueError("Training retrieval rows and labels are not aligned.")
        if X_val_model.shape[0] != y_val.shape[0]:
            raise ValueError("Validation model rows and labels are not aligned.")
        if X_val_retrieval.shape[0] != y_val.shape[0]:
            raise ValueError("Validation retrieval rows and labels are not aligned.")
        if X_val_model.shape[0] == 0:
            raise ValueError("An explicit non-empty validation fold is required.")
        unknown_validation_labels = np.setdiff1d(np.unique(y_val), np.unique(y_train))
        if unknown_validation_labels.size:
            raise ValueError(
                "Validation contains labels absent from training: "
                f"{unknown_validation_labels.tolist()}."
            )

        sizes = resolve_episode_sizes(
            n_train=y_train.shape[0],
            requested_context_size=self.requested_context_size,
            requested_query_size=self.requested_train_query_size,
            n_classes=np.unique(y_train).shape[0],
        )
        self.context_size_ = sizes.context_size
        self.train_query_size_ = sizes.query_size
        self._X_train_model = X_train_model
        self._X_train_retrieval = X_train_retrieval
        self._y_train = y_train
        self._sampler = LocalEpisodeSampler(
            context_size=sizes.context_size,
            query_size=sizes.query_size,
            seed=self.random_state,
            retrieval_batch_size=self.retrieval_batch_size,
        ).fit(X_train_retrieval, y_train)

        validation_retriever = KNNRetriever(query_batch_size=self.retrieval_batch_size).fit(
            X_train_retrieval, y_train
        )
        self._validation_context_indices = validation_retriever.retrieve(
            X_val_retrieval,
            sizes.context_size,
        ).indices

        from tabpfn.finetuning import finetuned_base

        local_builder = self._make_local_dataset_builder(
            finetuned_base.get_preprocessed_dataset_chunks
        )
        with _replace_tabpfn_dataset_builder(local_builder):
            if self.use_activation_checkpointing:
                with _backport_v26_activation_checkpointing():
                    self.delegate.fit(
                        X_train_model,
                        y_train,
                        X_val=X_val_model,
                        y_val=y_val,
                        output_dir=output_dir,
                    )
            else:
                self.delegate.fit(
                    X_train_model,
                    y_train,
                    X_val=X_val_model,
                    y_val=y_val,
                    output_dir=output_dir,
                )
        return self

    def predict_proba_local(
        self,
        X_query_model: np.ndarray,
        X_query_retrieval: np.ndarray,
        *,
        context_size: int | None = None,
    ) -> LocalPredictionResult:
        """Predict with exact train-only kNN contexts and fine-tuned weights."""

        if (
            self._X_train_model is None
            or self._X_train_retrieval is None
            or self._y_train is None
            or self.context_size_ is None
            or self._final_inference_config is None
        ):
            raise RuntimeError("Call fit before predict_proba_local.")
        X_query_model = np.asarray(X_query_model)
        X_query_retrieval = np.asarray(X_query_retrieval)
        if X_query_model.shape[0] != X_query_retrieval.shape[0]:
            raise ValueError("Query model and retrieval views must remain row-aligned.")
        resolved_k = self.context_size_ if context_size is None else context_size
        if resolved_k <= 0:
            raise ValueError("context_size must be positive.")

        started = perf_counter()
        retriever = KNNRetriever(query_batch_size=self.retrieval_batch_size).fit(
            self._X_train_retrieval,
            self._y_train,
        )
        index_seconds = perf_counter() - started
        started = perf_counter()
        contexts = retriever.retrieve(X_query_retrieval, resolved_k).indices
        retrieval_seconds = perf_counter() - started
        started = perf_counter()
        probabilities, classes, stats = self._predict_from_context_indices(
            X_train_model=self._X_train_model,
            y_train=self._y_train,
            X_query_model=X_query_model,
            context_indices=contexts,
            estimator_config=self._final_inference_config,
        )
        prediction_seconds = perf_counter() - started
        return LocalPredictionResult(
            probabilities=probabilities,
            classes=classes,
            actual_k=contexts.shape[1],
            index_seconds=index_seconds,
            retrieval_seconds=retrieval_seconds,
            prediction_seconds=prediction_seconds,
            inference_stats=stats,
        )
