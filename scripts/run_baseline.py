#!/usr/bin/env python3
"""Run one OpenML baseline experiment and write a JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabpfn_ir.data import (
    TabularPreprocessor,
    load_openml_dataset,
    stratified_train_validation_test_split,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--target")
    parser.add_argument("--method", choices=["full", "random", "knn"], required=True)
    parser.add_argument(
        "--k",
        default="128",
        help="Positive context size or 'localpfn' for min(10 * sqrt(n_train), 1000).",
    )
    parser.add_argument(
        "--random-ratio",
        type=float,
        help=(
            "For random retrieval, sample ceil(ratio * n_train) rows. "
            "Must be in (0, 1] and takes precedence over --k."
        ),
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
        "--evaluation-split",
        choices=["validation", "test"],
        default="test",
        help="Use validation for hyperparameter selection and test only for final reporting.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    return parser.parse_args()


def build_retriever(method: str, seed: int):
    if method == "full":
        return FullContextRetriever()
    if method == "random":
        return RandomRetriever(seed=seed, global_context=True)
    return KNNRetriever(metric="euclidean", algorithm="brute")


def resolve_context_size(specification: str, n_train: int) -> int:
    """Backward-compatible wrapper used by existing commands and tests."""

    return resolve_context_specification(specification, n_train)


def main() -> None:
    load_project_dotenv(REPOSITORY_ROOT / ".env", override=False)
    args = parse_args()
    if args.random_ratio is not None and args.method != "random":
        raise ValueError("--random-ratio can only be used with --method random.")
    dataset = load_openml_dataset(
        args.dataset_id,
        version=args.dataset_version,
        target=args.target,
    )
    split = stratified_train_validation_test_split(dataset.y, random_state=args.seed)
    X_train = dataset.X.iloc[split.train]
    y_train = dataset.y[split.train]
    query_indices = split.validation if args.evaluation_split == "validation" else split.test
    X_query = dataset.X.iloc[query_indices]
    y_query = dataset.y[query_indices]

    preprocessor = TabularPreprocessor(dataset.categorical_columns)
    train_views = preprocessor.fit_transform(X_train)
    query_views = preprocessor.transform(X_query)

    retriever = build_retriever(args.method, args.seed)
    tabpfn_kwargs = build_tabpfn_classifier_kwargs(
        device=args.device,
        devices=args.devices,
        ignore_pretraining_limits=args.ignore_pretraining_limits,
        fit_mode=args.fit_mode,
        n_estimators=args.n_estimators,
    )
    predictor = ContextualTabPFNClassifier(tabpfn_kwargs=tabpfn_kwargs)
    if args.method == "full":
        context_size = None
        context_specification = None
    elif args.random_ratio is not None:
        context_size = context_size_from_ratio(
            args.random_ratio,
            train_views.model.shape[0],
        )
        context_specification = f"ratio:{args.random_ratio:g}"
    else:
        context_size = resolve_context_size(args.k, n_train=train_views.model.shape[0])
        context_specification = args.k
    result = run_retrieval_experiment(
        retriever=retriever,
        predictor=predictor,
        X_train_model=train_views.model,
        X_train_retrieval=train_views.retrieval,
        y_train=y_train,
        X_query_model=query_views.model,
        X_query_retrieval=query_views.retrieval,
        y_query=y_query,
        k=context_size,
    )

    payload = {
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.version,
            "name": dataset.name,
            "target": dataset.target_name,
        },
        "seed": args.seed,
        "tabpfn_configuration": tabpfn_kwargs,
        "evaluation_split": args.evaluation_split,
        "context_specification": context_specification,
        "random_ratio": args.random_ratio,
        "result": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
