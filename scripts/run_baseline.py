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
from tabpfn_ir.models import ContextualTabPFNClassifier
from tabpfn_ir.retrieval import (
    FullContextRetriever,
    KNNRetriever,
    RandomRetriever,
    localpfn_context_size,
)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
    """Resolve an integer or LoCalPFN context-size specification."""

    if specification.lower() == "localpfn":
        return localpfn_context_size(n_train)
    try:
        context_size = int(specification)
    except ValueError as exc:
        raise ValueError("--k must be a positive integer or 'localpfn'.") from exc
    if context_size <= 0:
        raise ValueError("--k must be positive.")
    return min(context_size, n_train)


def main() -> None:
    args = parse_args()
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
    predictor = ContextualTabPFNClassifier(tabpfn_kwargs={"device": args.device})
    context_size = (
        None
        if args.method == "full"
        else resolve_context_size(args.k, n_train=train_views.model.shape[0])
    )
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
        "evaluation_split": args.evaluation_split,
        "context_specification": None if args.method == "full" else args.k,
        "result": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
