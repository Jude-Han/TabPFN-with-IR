#!/usr/bin/env python3
"""Run the TabPFN v1 or LoCalPFN paper benchmark with one retrieval method."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np

from tabpfn_ir.data import (
    TabularPreprocessor,
    discover_localpfn_dataset_directories,
    load_openml_dataset,
    load_openml_manifest,
    load_tabzilla_dataset,
    tabpfn_v1_split_indices,
)
from tabpfn_ir.evaluation import run_retrieval_experiment
from tabpfn_ir.environment import load_project_dotenv
from tabpfn_ir.models import ContextualTabPFNClassifier, build_tabpfn_classifier_kwargs
from tabpfn_ir.retrieval import (
    FullContextRetriever,
    KNNRetriever,
    RandomRetriever,
    context_size_from_ratio,
    resolve_context_specification,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_MANIFEST = REPOSITORY_ROOT / "data/manifests/tabpfn_v1_30.json"


@dataclass(frozen=True)
class FoldInput:
    """Normalized input for one dataset fold, independent of its source."""

    benchmark: str
    dataset_name: str
    dataset_key: str
    dataset_id: int | None
    dataset_version: int | None
    target: str | None
    X: object
    y: np.ndarray
    categorical_columns: tuple[str, ...]
    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def software_versions() -> dict[str, str | None]:
    """Record the installed predictor/runtime versions without importing CUDA."""

    versions: dict[str, str | None] = {}
    for package in ("tabpfn-ir", "tabpfn", "torch", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["tabpfn-v1", "localpfn"],
        required=True,
    )
    parser.add_argument("--method", choices=["full", "random", "knn"], required=True)
    parser.add_argument(
        "--k",
        default="localpfn",
        help="Positive context size or 'localpfn'; ignored by the full method.",
    )
    parser.add_argument(
        "--random-ratio",
        type=float,
        help=(
            "For random retrieval, sample ceil(ratio * n_train) rows. "
            "Must be in (0, 1] and takes precedence over --k."
        ),
    )
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
        "--model-version",
        choices=["v2.6"],
        default="v2.6",
        help="Pinned TabPFN checkpoint family. TabPFN 8.x otherwise defaults to v3.",
    )
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=32,
        help="Maximum compatible query-specific contexts per fused TabPFN forward.",
    )
    parser.add_argument(
        "--disable-batched-contexts",
        action="store_true",
        help="Use the legacy one-fit-and-predict-per-context path for comparison.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
        help="LoCalPFN can use validation for selection; TabPFN v1 defines test only.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="Zero-based fold numbers. By default, run every paper fold.",
    )
    parser.add_argument("--dataset-ids", type=int, nargs="+")
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        help="Exact manifest names or TabZilla directory/metadata names.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected datasets.")
    parser.add_argument(
        "--max-query-samples",
        type=int,
        help="Deterministically subsample each test fold for smoke tests.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    parser.add_argument(
        "--tabzilla-root",
        type=Path,
        help="Directory containing preprocessed TabZilla dataset directories.",
    )
    parser.add_argument(
        "--expected-localpfn-count",
        type=int,
        default=95,
        help="Expected number after official LoCalPFN filters.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Run even if the discovered LoCalPFN dataset count differs from expected.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Append-only JSONL output.")
    parser.add_argument("--resume", action="store_true", help="Skip completed output keys.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def build_retriever(method: str, seed: int):
    if method == "full":
        return FullContextRetriever()
    if method == "random":
        return RandomRetriever(seed=seed, global_context=True)
    return KNNRetriever(metric="euclidean", algorithm="brute")


def _selected_fold_numbers(requested: list[int] | None, available: int) -> list[int]:
    folds = list(range(available)) if requested is None else requested
    invalid = sorted({fold for fold in folds if fold < 0 or fold >= available})
    if invalid:
        raise ValueError(f"Invalid folds {invalid}; available folds are 0..{available - 1}.")
    return list(dict.fromkeys(folds))


def _matches_filters(
    *,
    dataset_id: int | None,
    names: Iterable[str],
    requested_ids: list[int] | None,
    requested_names: list[str] | None,
) -> bool:
    if requested_ids is not None and dataset_id not in requested_ids:
        return False
    if requested_names is not None and not set(names).intersection(requested_names):
        return False
    return True


def iter_tabpfn_v1_folds(args: argparse.Namespace) -> Iterable[FoldInput]:
    if args.evaluation_split != "test":
        raise ValueError("The reconstructed TabPFN v1 protocol has no validation split.")
    manifest = load_openml_manifest(args.manifest)
    selected = [
        entry
        for entry in manifest.datasets
        if _matches_filters(
            dataset_id=entry.dataset_id,
            names=(entry.name,),
            requested_ids=args.dataset_ids,
            requested_names=args.dataset_names,
        )
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No TabPFN v1 datasets matched the requested filters.")

    for entry in selected:
        dataset = load_openml_dataset(
            entry.dataset_id,
            version=entry.version,
            target=entry.target,
        )
        splits = tabpfn_v1_split_indices(dataset.y, n_splits=5, random_state=args.seed)
        for fold in _selected_fold_numbers(args.folds, len(splits)):
            split = splits[fold]
            yield FoldInput(
                benchmark=manifest.benchmark,
                dataset_name=dataset.name,
                dataset_key=str(dataset.dataset_id),
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.version,
                target=dataset.target_name,
                X=dataset.X,
                y=dataset.y,
                categorical_columns=dataset.categorical_columns,
                fold=fold,
                train=split.train,
                validation=split.validation,
                test=split.test,
            )


def iter_localpfn_folds(args: argparse.Namespace) -> Iterable[FoldInput]:
    if args.tabzilla_root is None:
        raise ValueError("--tabzilla-root is required for the LoCalPFN benchmark.")
    directories = discover_localpfn_dataset_directories(args.tabzilla_root)
    if (
        len(directories) != args.expected_localpfn_count
        and not args.allow_count_mismatch
    ):
        raise ValueError(
            "The official LoCalPFN filters selected "
            f"{len(directories)} datasets, expected {args.expected_localpfn_count}. "
            "Check the TabZilla preprocessing output or pass --allow-count-mismatch "
            "for an intentional subset."
        )

    selected = []
    for path in directories:
        try:
            task_id = int(path.name.rsplit("__", maxsplit=1)[1])
        except (IndexError, ValueError):
            task_id = None
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        metadata_name = str(metadata.get("name", path.name))
        if _matches_filters(
            dataset_id=task_id,
            names=(path.name, metadata_name),
            requested_ids=args.dataset_ids,
            requested_names=args.dataset_names,
        ):
            selected.append(path)
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No LoCalPFN datasets matched the requested filters.")

    for path in selected:
        dataset = load_tabzilla_dataset(path)
        for fold in _selected_fold_numbers(args.folds, len(dataset.splits)):
            split = dataset.splits[fold]
            yield FoldInput(
                benchmark="localpfn-tabzilla",
                dataset_name=dataset.name,
                dataset_key=dataset.directory_name,
                dataset_id=dataset.task_id,
                dataset_version=None,
                target=None,
                X=dataset.X,
                y=dataset.y,
                categorical_columns=dataset.categorical_columns,
                fold=fold,
                train=split.train,
                validation=split.validation,
                test=split.test,
            )


def _subsample_query_indices(
    indices: np.ndarray,
    maximum: int | None,
    *,
    seed: int,
    y: np.ndarray,
) -> np.ndarray:
    if maximum is None or indices.size <= maximum:
        return indices
    if maximum <= 0:
        raise ValueError("--max-query-samples must be positive.")
    generator = np.random.default_rng(seed)
    classes = np.unique(y[indices])
    if maximum < classes.size:
        raise ValueError(
            "--max-query-samples must be at least the number of classes so AUC is defined."
        )
    selected = [int(generator.choice(indices[y[indices] == label])) for label in classes]
    remaining = np.setdiff1d(indices, np.asarray(selected), assume_unique=False)
    additional = generator.choice(
        remaining,
        size=maximum - len(selected),
        replace=False,
    )
    return np.sort(np.concatenate([np.asarray(selected), additional]))


def _result_key(record: dict[str, object]) -> tuple[object, ...]:
    versions = record.get("software_versions")
    tabpfn_version = versions.get("tabpfn") if isinstance(versions, dict) else None
    return (
        record.get("benchmark"),
        record.get("dataset_key"),
        record.get("fold"),
        record.get("evaluation_split"),
        record.get("method"),
        record.get("context_specification"),
        record.get("seed"),
        json.dumps(record.get("tabpfn_configuration"), sort_keys=True),
        tabpfn_version,
    )


def _completed_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    keys = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
        if record.get("status") == "ok":
            keys.add(_result_key(record))
    return keys


def run_fold(fold_input: FoldInput, args: argparse.Namespace) -> dict[str, object]:
    evaluation_indices = (
        fold_input.validation if args.evaluation_split == "validation" else fold_input.test
    )
    query_indices = _subsample_query_indices(
        evaluation_indices,
        args.max_query_samples,
        seed=args.seed + fold_input.fold,
        y=fold_input.y,
    )
    X_train = fold_input.X.iloc[fold_input.train]
    X_query = fold_input.X.iloc[query_indices]
    y_train = fold_input.y[fold_input.train]
    y_query = fold_input.y[query_indices]

    preprocessor = TabularPreprocessor(fold_input.categorical_columns)
    train_views = preprocessor.fit_transform(X_train)
    query_views = preprocessor.transform(X_query)
    if args.method == "full":
        context_size = None
    elif args.random_ratio is not None:
        context_size = context_size_from_ratio(
            args.random_ratio,
            train_views.model.shape[0],
        )
    else:
        context_size = resolve_context_specification(args.k, train_views.model.shape[0])
    retrieval_seed = args.seed + fold_input.fold
    tabpfn_kwargs = build_tabpfn_classifier_kwargs(
        device=args.device,
        devices=args.devices,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        fit_mode=args.fit_mode,
        n_estimators=args.n_estimators,
    )
    result = run_retrieval_experiment(
        retriever=build_retriever(args.method, retrieval_seed),
        predictor=ContextualTabPFNClassifier(
            tabpfn_kwargs=tabpfn_kwargs,
            model_version=args.model_version,
            context_batch_size=args.context_batch_size,
            use_batched_contexts=not args.disable_batched_contexts,
        ),
        X_train_model=train_views.model,
        X_train_retrieval=train_views.retrieval,
        y_train=y_train,
        X_query_model=query_views.model,
        X_query_retrieval=query_views.retrieval,
        y_query=y_query,
        k=context_size,
        auc_mode="ovo",
    )
    return result.to_dict()


def main() -> None:
    load_project_dotenv(REPOSITORY_ROOT / ".env", override=False)
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.context_batch_size <= 0:
        raise ValueError("--context-batch-size must be positive.")
    if args.random_ratio is not None:
        if args.method != "random":
            raise ValueError("--random-ratio can only be used with --method random.")
        context_size_from_ratio(args.random_ratio, 1)
    tabpfn_kwargs = build_tabpfn_classifier_kwargs(
        device=args.device,
        devices=args.devices,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        fit_mode=args.fit_mode,
        n_estimators=args.n_estimators,
    )
    tabpfn_configuration = {
        **tabpfn_kwargs,
        "model_version": args.model_version,
        "context_batch_size": args.context_batch_size,
        "batched_contexts": not args.disable_batched_contexts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(args.output) if args.resume else set()
    versions = software_versions()
    fold_iterator = (
        iter_tabpfn_v1_folds(args)
        if args.benchmark == "tabpfn-v1"
        else iter_localpfn_folds(args)
    )

    for fold_input in fold_iterator:
        base_record: dict[str, object] = {
            "benchmark": fold_input.benchmark,
            "dataset_key": fold_input.dataset_key,
            "dataset_id": fold_input.dataset_id,
            "dataset_version": fold_input.dataset_version,
            "dataset_name": fold_input.dataset_name,
            "target": fold_input.target,
            "fold": fold_input.fold,
            "evaluation_split": args.evaluation_split,
            "method": args.method,
            "context_specification": (
                None
                if args.method == "full"
                else (
                    f"ratio:{args.random_ratio:g}"
                    if args.random_ratio is not None
                    else args.k
                )
            ),
            "random_ratio": args.random_ratio,
            "seed": args.seed,
            "device": tabpfn_kwargs["device"],
            "tabpfn_configuration": tabpfn_configuration,
            "software_versions": versions,
        }
        if _result_key(base_record) in completed:
            print(f"skip {fold_input.dataset_key} fold={fold_input.fold} (already complete)")
            continue
        try:
            result = run_fold(fold_input, args)
            record = {**base_record, "status": "ok", "result": result}
            print(
                f"ok {fold_input.dataset_key} fold={fold_input.fold} "
                f"roc_auc={result['metrics']['roc_auc']:.6f}"
            )
        except Exception as exc:  # keep a long benchmark recoverable and auditable
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
