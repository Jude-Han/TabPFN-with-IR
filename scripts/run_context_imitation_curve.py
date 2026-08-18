#!/usr/bin/env python3
"""Measure how nested kNN contexts imitate full-context TabPFN v2.6."""

from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

from tabpfn_ir.data import TabularPreprocessor, load_openml_dataset, localpfn_split_indices
from tabpfn_ir.environment import load_project_dotenv
from tabpfn_ir.evaluation import resolve_context_grid, run_full_context_imitation_curve
from tabpfn_ir.models import (
    TABPFN_INFERENCE_PROFILES,
    ContextualTabPFNClassifier,
    build_tabpfn_classifier_kwargs,
    resolve_n_estimators,
)
from tabpfn_ir.retrieval import KNNRetriever

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_SIZES = (16, 32, 64, 128, 256, 512, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--target")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_SIZES),
        help=(
            "Nested kNN context sizes. Values above the training-fold size are capped, "
            "and the exact full-context point is always appended."
        ),
    )
    parser.add_argument(
        "--tv-tolerance",
        type=float,
        default=0.05,
        help="Per-query predictive TV threshold used to define the minimum stable k.",
    )
    parser.add_argument(
        "--allow-class-change",
        action="store_true",
        help="Do not require the predicted class to match full-context TabPFN.",
    )
    parser.add_argument("--fold", type=int, default=0, help="LoCalPFN fold in 0..9.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="validation",
        help="Use validation for choosing k; reserve test for final confirmation.",
    )
    parser.add_argument(
        "--max-query-samples",
        type=int,
        default=256,
        help="Stratified pilot-query cap; pass 0 to use the complete split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        nargs="+",
        help="Explicit device list, for example cuda:0 cuda:1 cuda:2 cuda:3.",
    )
    parser.add_argument("--ignore-pretraining-limits", action="store_true")
    parser.add_argument(
        "--fit-mode",
        choices=["fit_preprocessors", "low_memory", "fit_with_cache"],
        default="fit_preprocessors",
    )
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument(
        "--inference-profile",
        choices=sorted(TABPFN_INFERENCE_PROFILES),
        default="single-estimator",
        help="The single-estimator default keeps this first diagnostic inexpensive.",
    )
    parser.add_argument(
        "--model-version",
        choices=["v2.6"],
        default="v2.6",
        help="Pinned checkpoint family; other TabPFN versions are intentionally rejected.",
    )
    parser.add_argument("--context-batch-size", type=int, default=32)
    parser.add_argument("--disable-batched-contexts", action="store_true")
    parser.add_argument("--retrieval-query-batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output. Matching .summary.csv and .queries.csv files are also written.",
    )
    return parser.parse_args()


def software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("tabpfn-ir", "tabpfn", "torch", "scikit-learn", "faiss-cpu"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _subsample_query_indices(
    indices: np.ndarray,
    maximum: int | None,
    *,
    seed: int,
    y: np.ndarray,
) -> np.ndarray:
    if maximum is None or maximum == 0 or indices.size <= maximum:
        return np.asarray(indices)
    if maximum < 0:
        raise ValueError("--max-query-samples must be non-negative.")
    generator = np.random.default_rng(seed)
    classes = np.unique(y[indices])
    if maximum < classes.size:
        raise ValueError(
            "--max-query-samples must be at least the number of query classes."
        )
    selected = [int(generator.choice(indices[y[indices] == label])) for label in classes]
    remaining = np.setdiff1d(indices, np.asarray(selected), assume_unique=False)
    additional = generator.choice(
        remaining,
        size=maximum - len(selected),
        replace=False,
    )
    return np.sort(np.concatenate([np.asarray(selected), additional]))


def _companion_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{suffix}")


def _write_summary_csv(payload: dict[str, object], path: Path) -> None:
    rows = []
    for point in payload["result"]["curve"]:
        row = {
            "context_size": point["context_size"],
            "is_full_reference": point["is_full_reference"],
            "prediction_seconds": point["prediction_seconds"],
            "prediction_seconds_per_query": point["prediction_seconds_per_query"],
            "class_agreement": point["class_agreement"],
            "stable_query_coverage": point["stable_query_coverage"],
        }
        row.update(
            {f"predictive_tv_{key}": value for key, value in point["predictive_tv"].items()}
        )
        row.update(
            {
                f"metric_{key}": value
                for key, value in point["classification_metrics"].items()
            }
        )
        row.update(
            {
                f"metric_delta_from_full_{key}": value
                for key, value in point["metric_delta_from_full"].items()
            }
        )
        rows.append(row)
    _write_csv(rows, path)


def _write_query_csv(payload: dict[str, object], path: Path) -> None:
    rows = []
    for query in payload["result"]["per_query"]:
        row = {
            "query_position": query["query_position"],
            "query_id": query["query_id"],
            "true_label": query["true_label"],
            "full_predicted_label": query["full_predicted_label"],
            "minimum_stable_context_size": query["minimum_stable_context_size"],
        }
        row.update({f"tv_k_{key}": value for key, value in query["tv_by_context"].items()})
        row.update(
            {
                f"class_agreement_k_{key}": value
                for key, value in query["class_agreement_by_context"].items()
            }
        )
        row.update(
            {
                f"predicted_label_k_{key}": value
                for key, value in query["predicted_label_by_context"].items()
            }
        )
        rows.append(row)
    _write_csv(rows, path)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV to {path}.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_curve(payload: dict[str, object]) -> None:
    print("k\tTV-p95\tagreement\tstable-coverage\tROC-AUC\tseconds")
    for point in payload["result"]["curve"]:
        print(
            f"{point['context_size']}\t"
            f"{point['predictive_tv']['p95']:.6f}\t"
            f"{point['class_agreement']:.4f}\t"
            f"{point['stable_query_coverage']:.4f}\t"
            f"{point['classification_metrics']['roc_auc']:.6f}\t"
            f"{point['prediction_seconds']:.3f}"
        )


def main() -> None:
    load_project_dotenv(REPOSITORY_ROOT / ".env", override=False)
    args = parse_args()
    if args.fold < 0 or args.fold >= 10:
        raise ValueError("--fold must be in 0..9.")
    if args.tv_tolerance < 0:
        raise ValueError("--tv-tolerance must be non-negative.")
    if args.context_batch_size <= 0:
        raise ValueError("--context-batch-size must be positive.")
    if args.retrieval_query_batch_size <= 0:
        raise ValueError("--retrieval-query-batch-size must be positive.")
    if args.max_query_samples < 0:
        raise ValueError("--max-query-samples must be non-negative.")
    args.n_estimators = resolve_n_estimators(
        inference_profile=args.inference_profile,
        n_estimators=args.n_estimators,
    )

    dataset = load_openml_dataset(
        args.dataset_id,
        version=args.dataset_version,
        target=args.target,
    )
    split = localpfn_split_indices(
        dataset.y,
        n_splits=10,
        random_state=args.split_seed,
    )[args.fold]
    evaluation_indices = (
        split.validation if args.evaluation_split == "validation" else split.test
    )
    query_indices = _subsample_query_indices(
        evaluation_indices,
        args.max_query_samples,
        seed=args.seed + args.fold,
        y=dataset.y,
    )
    X_train = dataset.X.iloc[split.train]
    X_query = dataset.X.iloc[query_indices]
    y_train = dataset.y[split.train]
    y_query = dataset.y[query_indices]

    preprocessor = TabularPreprocessor(dataset.categorical_columns)
    train_views = preprocessor.fit_transform(X_train)
    query_views = preprocessor.transform(X_query)
    n_train = train_views.model.shape[0]
    context_sizes = resolve_context_grid(args.k_values, n_train, include_full=True)
    retrieval_sizes = [size for size in context_sizes if size < n_train]

    retriever = KNNRetriever(query_batch_size=args.retrieval_query_batch_size)
    started = perf_counter()
    retriever.fit(train_views.retrieval, y_train)
    index_seconds = perf_counter() - started
    if retrieval_sizes:
        started = perf_counter()
        retrieval = retriever.retrieve(query_views.retrieval, max(retrieval_sizes))
        retrieval_seconds = perf_counter() - started
        ranked_context_indices = retrieval.indices
    else:
        retrieval_seconds = 0.0
        ranked_context_indices = np.empty((len(query_indices), 0), dtype=np.int64)

    tabpfn_kwargs = build_tabpfn_classifier_kwargs(
        device=args.device,
        devices=args.devices,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        fit_mode=args.fit_mode,
        n_estimators=args.n_estimators,
    )
    result = run_full_context_imitation_curve(
        predictor=ContextualTabPFNClassifier(
            tabpfn_kwargs=tabpfn_kwargs,
            model_version=args.model_version,
            context_batch_size=args.context_batch_size,
            use_batched_contexts=not args.disable_batched_contexts,
        ),
        X_train_model=train_views.model,
        y_train=y_train,
        X_query_model=query_views.model,
        y_query=y_query,
        ranked_context_indices=ranked_context_indices,
        context_sizes=context_sizes,
        tolerance=args.tv_tolerance,
        require_class_agreement=not args.allow_class_change,
        query_ids=query_indices,
        auc_mode="ovo",
    )
    payload = {
        "experiment": "tabpfn-v2.6-full-context-imitation-curve",
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.version,
            "name": dataset.name,
            "target": dataset.target_name,
        },
        "fold": args.fold,
        "split_seed": args.split_seed,
        "split_protocol": "localpfn-stratified-8:1:1",
        "evaluation_split": args.evaluation_split,
        "query_sampling": {
            "seed": args.seed + args.fold,
            "maximum": args.max_query_samples or None,
            "selected": len(query_indices),
            "available": len(evaluation_indices),
        },
        "retrieval": {
            "method": "exact-l2-knn",
            "nested_prefixes": True,
            "maximum_retrieved_k": max(retrieval_sizes, default=0),
            "index_seconds": index_seconds,
            "retrieval_seconds": retrieval_seconds,
        },
        "tabpfn_configuration": {
            **tabpfn_kwargs,
            "model_version": args.model_version,
            "inference_profile": args.inference_profile,
            "context_batch_size": args.context_batch_size,
            "batched_contexts": not args.disable_batched_contexts,
        },
        "software_versions": software_versions(),
        "result": result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = _companion_path(args.output, "summary.csv")
    query_path = _companion_path(args.output, "queries.csv")
    _write_summary_csv(payload, summary_path)
    _write_query_csv(payload, query_path)
    _print_curve(payload)
    print(f"JSON: {args.output}")
    print(f"summary CSV: {summary_path}")
    print(f"query CSV: {query_path}")


if __name__ == "__main__":
    main()
