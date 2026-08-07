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
from tabpfn_ir.retrieval import FullContextRetriever, KNNRetriever, RandomRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--target")
    parser.add_argument("--method", choices=["full", "random", "knn"], required=True)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/result.json"))
    return parser.parse_args()


def build_retriever(method: str, seed: int):
    if method == "full":
        return FullContextRetriever()
    if method == "random":
        return RandomRetriever(seed=seed, global_context=True)
    return KNNRetriever(metric="euclidean", algorithm="brute")


def main() -> None:
    args = parse_args()
    dataset = load_openml_dataset(
        args.dataset_id,
        version=args.dataset_version,
        target=args.target,
    )
    split = stratified_train_validation_test_split(dataset.y, random_state=args.seed)
    X_train = dataset.X.iloc[split.train]
    X_test = dataset.X.iloc[split.test]
    y_train = dataset.y[split.train]
    y_test = dataset.y[split.test]

    preprocessor = TabularPreprocessor(dataset.categorical_columns)
    train_views = preprocessor.fit_transform(X_train)
    test_views = preprocessor.transform(X_test)

    retriever = build_retriever(args.method, args.seed)
    predictor = ContextualTabPFNClassifier(tabpfn_kwargs={"device": args.device})
    result = run_retrieval_experiment(
        retriever=retriever,
        predictor=predictor,
        X_train_model=train_views.model,
        X_train_retrieval=train_views.retrieval,
        y_train=y_train,
        X_query_model=test_views.model,
        X_query_retrieval=test_views.retrieval,
        y_query=y_test,
        k=None if args.method == "full" else args.k,
    )

    payload = {
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.version,
            "name": dataset.name,
            "target": dataset.target_name,
        },
        "seed": args.seed,
        "result": result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
