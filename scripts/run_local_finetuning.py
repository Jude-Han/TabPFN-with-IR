#!/usr/bin/env python3
"""Fine-tune TabPFN v2.6 with LoCalPFN-style local kNN episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from run_benchmark import (
    DEFAULT_CC18_MANIFEST,
    FoldInput,
    _subsample_query_indices,
    iter_localpfn_folds,
    iter_openml_cc18_folds,
    software_versions,
)

from tabpfn_ir.data import TabularPreprocessor
from tabpfn_ir.environment import load_project_dotenv
from tabpfn_ir.evaluation import classification_metrics
from tabpfn_ir.models import LocalFinetunedTabPFNClassifier
from tabpfn_ir.retrieval import resolve_context_specification

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["openml-cc18", "localpfn"],
        required=True,
        help="Both supported protocols provide an explicit validation fold.",
    )
    parser.add_argument(
        "--k",
        default="localpfn",
        help="Inference/training context size or the dynamic value 'localpfn'.",
    )
    parser.add_argument("--maximum-context-size", type=int, default=1000)
    parser.add_argument("--train-query-size", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--episode-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-value", type=float, default=1.0)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "warmup-only", "none"],
        default="cosine",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--eval-metric",
        choices=["roc_auc", "log_loss"],
        default="roc_auc",
    )
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--n-estimators-finetune", type=int, default=2)
    parser.add_argument("--n-estimators-validation", type=int, default=2)
    parser.add_argument("--n-estimators-final", type=int, default=2)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fixed-preprocessing-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-checkpoint-interval",
        type=int,
        default=10,
        help="Epoch interval; use 0 to save only improving best checkpoints.",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=512)
    parser.add_argument("--context-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--dataset-ids", type=int, nargs="+")
    parser.add_argument("--dataset-names", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-query-samples", type=int)
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        help="Smoke-test option only; final experiments should use the full validation fold.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CC18_MANIFEST)
    parser.add_argument("--tabzilla-root", type=Path)
    parser.add_argument("--expected-localpfn-count", type=int, default=95)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/local-finetuning/checkpoints"),
    )
    parser.add_argument(
        "--checkpoint-tag",
        default="default",
        help="Change this tag to intentionally start a separate checkpoint lineage.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "maximum_context_size": args.maximum_context_size,
        "train_query_size": args.train_query_size,
        "steps_per_epoch": args.steps_per_epoch,
        "episode_batch_size": args.episode_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "n_estimators_finetune": args.n_estimators_finetune,
        "n_estimators_validation": args.n_estimators_validation,
        "n_estimators_final": args.n_estimators_final,
        "retrieval_batch_size": args.retrieval_batch_size,
        "context_batch_size": args.context_batch_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {', '.join(invalid)}.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.grad_clip_value < 0:
        raise ValueError("--grad-clip-value must be non-negative; use 0 to disable it.")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative.")
    if args.save_checkpoint_interval < 0:
        raise ValueError("--save-checkpoint-interval must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("--seed and --split-seed must be non-negative.")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if args.benchmark == "localpfn" and args.tabzilla_root is None:
        raise ValueError("--tabzilla-root is required for --benchmark localpfn.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and (
        args.folds is None or len(set(args.folds)) != 1
    ):
        raise ValueError(
            "A torchrun fine-tuning invocation must select exactly one --folds value. "
            "Launch separate jobs for additional folds."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        one_id = args.dataset_ids is not None and len(set(args.dataset_ids)) == 1
        one_name = args.dataset_names is not None and len(set(args.dataset_names)) == 1
        if not (one_id or one_name or args.limit == 1):
            raise ValueError(
                "A torchrun invocation must select exactly one dataset using one "
                "--dataset-ids value, one --dataset-names value, or --limit 1."
            )


def _fine_tuning_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_version": "v2.6",
        "tabpfn_package_version": "8.2.0",
        "context_specification": args.k,
        "maximum_context_size": args.maximum_context_size,
        "train_query_size": args.train_query_size,
        "steps_per_epoch": args.steps_per_epoch,
        "episode_batch_size": args.episode_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_value": None if args.grad_clip_value == 0 else args.grad_clip_value,
        "scheduler": args.scheduler,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "eval_metric": args.eval_metric,
        "time_limit": args.time_limit,
        "n_estimators_finetune": args.n_estimators_finetune,
        "n_estimators_validation": args.n_estimators_validation,
        "n_estimators_final": args.n_estimators_final,
        "activation_checkpointing": args.activation_checkpointing,
        "fixed_preprocessing_seed": args.fixed_preprocessing_seed,
        "save_checkpoint_interval": (
            None if args.save_checkpoint_interval == 0 else args.save_checkpoint_interval
        ),
        "retrieval_batch_size": args.retrieval_batch_size,
        "context_batch_size": args.context_batch_size,
        "max_validation_samples": args.max_validation_samples,
        "max_query_samples": args.max_query_samples,
        "device": args.device,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "seed": args.seed,
    }


def _record_key(record: dict[str, object]) -> tuple[object, ...]:
    return (
        record.get("benchmark"),
        record.get("dataset_key"),
        record.get("dataset_version"),
        record.get("target"),
        record.get("fold"),
        record.get("split_protocol"),
        record.get("split_seed"),
        json.dumps(record.get("fine_tuning_configuration"), sort_keys=True),
        record.get("checkpoint_tag"),
    )


def _completed_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    keys = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
        if record.get("status") == "ok":
            keys.add(_record_key(record))
    return keys


def _checkpoint_directory(
    *,
    fold_input: FoldInput,
    args: argparse.Namespace,
    configuration: dict[str, object],
) -> Path:
    checkpoint_identity = {
        "configuration": configuration,
        "dataset_version": fold_input.dataset_version,
        "target": fold_input.target,
        "split_protocol": fold_input.split_protocol,
        "split_seed": fold_input.split_seed,
    }
    signature = hashlib.sha256(
        json.dumps(checkpoint_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", fold_input.dataset_key).strip("-")
    tag_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.checkpoint_tag).strip("-")
    if not tag_slug:
        raise ValueError("--checkpoint-tag must contain a filename-safe character.")
    return (
        args.checkpoint_root
        / fold_input.benchmark
        / dataset_slug
        / f"fold-{fold_input.fold}"
        / f"{tag_slug}-{signature}"
    )


def run_fold(
    fold_input: FoldInput,
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
) -> dict[str, object] | None:
    validation_indices = _subsample_query_indices(
        fold_input.validation,
        args.max_validation_samples,
        seed=args.seed + 10_000 + fold_input.fold,
        y=fold_input.y,
    )
    test_indices = _subsample_query_indices(
        fold_input.test,
        args.max_query_samples,
        seed=args.seed + fold_input.fold,
        y=fold_input.y,
    )
    X_train = fold_input.X.iloc[fold_input.train]
    X_validation = fold_input.X.iloc[validation_indices]
    X_test = fold_input.X.iloc[test_indices]
    y_train = fold_input.y[fold_input.train]
    y_validation = fold_input.y[validation_indices]
    y_test = fold_input.y[test_indices]

    started = perf_counter()
    preprocessor = TabularPreprocessor(fold_input.categorical_columns)
    train_views = preprocessor.fit_transform(X_train)
    validation_views = preprocessor.transform(X_validation)
    test_views = preprocessor.transform(X_test)
    preprocessing_seconds = perf_counter() - started
    context_size = resolve_context_specification(
        args.k,
        train_views.model.shape[0],
        maximum=args.maximum_context_size,
    )

    finetuner = LocalFinetunedTabPFNClassifier(
        context_size=context_size,
        train_query_size=args.train_query_size,
        steps_per_epoch=args.steps_per_epoch,
        episode_batch_size=args.episode_batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        context_batch_size=args.context_batch_size,
        device=args.device,
        epochs=args.epochs,
        time_limit=args.time_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        random_state=args.seed,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        grad_clip_value=None if args.grad_clip_value == 0 else args.grad_clip_value,
        use_lr_scheduler=args.scheduler != "none",
        lr_warmup_only=args.scheduler == "warmup-only",
        n_estimators_finetune=args.n_estimators_finetune,
        n_estimators_validation=args.n_estimators_validation,
        n_estimators_final_inference=args.n_estimators_final,
        use_activation_checkpointing=args.activation_checkpointing,
        save_checkpoint_interval=(
            None if args.save_checkpoint_interval == 0 else args.save_checkpoint_interval
        ),
        use_fixed_preprocessing_seed=args.fixed_preprocessing_seed,
        eval_metric=args.eval_metric,
    )
    started = perf_counter()
    finetuner.fit(
        train_views.model,
        y_train,
        X_train_retrieval=train_views.retrieval,
        X_val_model=validation_views.model,
        y_val=y_validation,
        X_val_retrieval=validation_views.retrieval,
        output_dir=checkpoint_dir,
    )
    fine_tuning_seconds = perf_counter() - started

    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return None

    prediction = finetuner.predict_proba_local(
        test_views.model,
        test_views.retrieval,
    )
    metrics = classification_metrics(
        y_test,
        prediction.probabilities,
        prediction.classes,
        auc_mode="ovo",
    )
    return {
        "method": "LocalFinetunedTabPFNClassifier",
        "n_train": int(train_views.model.shape[0]),
        "n_validation": int(validation_views.model.shape[0]),
        "n_test": int(test_views.model.shape[0]),
        "requested_k": int(context_size),
        "actual_k": prediction.actual_k,
        "actual_train_query_size": finetuner.train_query_size_,
        "preprocessing_seconds": preprocessing_seconds,
        "fine_tuning_seconds": fine_tuning_seconds,
        "index_seconds": prediction.index_seconds,
        "retrieval_seconds": prediction.retrieval_seconds,
        "prediction_seconds": prediction.prediction_seconds,
        **asdict(prediction.inference_stats),
        "metrics": metrics,
    }


def main() -> None:
    load_project_dotenv(REPOSITORY_ROOT / ".env", override=False)
    args = parse_args()
    _validate_args(args)
    configuration = _fine_tuning_configuration(args)
    is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(args.output) if args.resume else set()
    versions = software_versions()
    fold_iterator = (
        iter_openml_cc18_folds(args)
        if args.benchmark == "openml-cc18"
        else iter_localpfn_folds(args)
    )

    for fold_input in fold_iterator:
        checkpoint_dir = _checkpoint_directory(
            fold_input=fold_input,
            args=args,
            configuration=configuration,
        )
        base_record: dict[str, object] = {
            "benchmark": fold_input.benchmark,
            "dataset_key": fold_input.dataset_key,
            "task_id": fold_input.task_id,
            "dataset_id": fold_input.dataset_id,
            "dataset_version": fold_input.dataset_version,
            "dataset_name": fold_input.dataset_name,
            "target": fold_input.target,
            "fold": fold_input.fold,
            "split_protocol": fold_input.split_protocol,
            "split_seed": fold_input.split_seed,
            "method": "local-ft",
            "fine_tuning_configuration": configuration,
            "checkpoint_tag": args.checkpoint_tag,
            "checkpoint_directory": str(checkpoint_dir),
            "software_versions": versions,
        }
        if _record_key(base_record) in completed:
            if is_main_process:
                print(f"skip {fold_input.dataset_key} fold={fold_input.fold} (complete)")
            continue
        try:
            result = run_fold(
                fold_input,
                args,
                checkpoint_dir=checkpoint_dir,
            )
            if not is_main_process:
                continue
            assert result is not None
            record = {**base_record, "status": "ok", "result": result}
            print(
                f"ok {fold_input.dataset_key} fold={fold_input.fold} "
                f"roc_auc={result['metrics']['roc_auc']:.6f}"
            )
        except Exception as exc:  # preserve long-run failures as data
            if not is_main_process:
                raise
            record = {
                **base_record,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                f"error {fold_input.dataset_key} fold={fold_input.fold}: "
                f"{type(exc).__name__}: {exc}"
            )
            if args.fail_fast:
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                raise
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
