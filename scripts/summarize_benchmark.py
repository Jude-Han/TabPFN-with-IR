#!/usr/bin/env python3
"""Summarize benchmark JSONL files into fold, dataset, and rank CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--primary-metric", default="roc_auc")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Use ascending ranks (for example, for log loss).",
    )
    return parser.parse_args()


def load_records(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    successful: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            base = {
                key: record.get(key)
                for key in (
                    "benchmark",
                    "dataset_key",
                    "dataset_id",
                    "dataset_name",
                    "fold",
                    "evaluation_split",
                    "method",
                    "context_specification",
                    "random_ratio",
                    "seed",
                    "device",
                )
            }
            base["source"] = str(path)
            if record.get("status") != "ok":
                errors.append(
                    {
                        **base,
                        "error_type": record.get("error_type"),
                        "error": record.get("error"),
                        "line": line_number,
                    }
                )
                continue
            result = record["result"]
            metrics = result["metrics"]
            successful.append(
                {
                    **base,
                    "actual_k": result["actual_k"],
                    "n_train": result["n_train"],
                    "n_query": result["n_query"],
                    "index_seconds": result["index_seconds"],
                    "retrieval_seconds": result["retrieval_seconds"],
                    "prediction_seconds": result["prediction_seconds"],
                    **metrics,
                }
            )
    return pd.DataFrame(successful), pd.DataFrame(errors)


def main() -> None:
    args = parse_args()
    folds, errors = load_records(args.inputs)
    if folds.empty:
        raise ValueError("No successful benchmark records were found.")
    if args.primary_metric not in folds.columns:
        raise ValueError(f"Metric {args.primary_metric!r} is not present in the records.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "fold_results.csv", index=False)
    if not errors.empty:
        errors.to_csv(args.output_dir / "errors.csv", index=False)

    metric_columns = [
        column
        for column in (
            "roc_auc",
            "roc_auc_ovo",
            "roc_auc_ovr",
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "weighted_f1",
            "index_seconds",
            "retrieval_seconds",
            "prediction_seconds",
        )
        if column in folds.columns
    ]
    group_columns = [
        "benchmark",
        "dataset_key",
        "dataset_id",
        "dataset_name",
        "evaluation_split",
        "method",
        "context_specification",
    ]
    per_dataset = (
        folds.groupby(group_columns, dropna=False)[metric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    per_dataset.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in per_dataset.columns
    ]
    per_dataset.to_csv(args.output_dir / "dataset_results.csv", index=False)

    score_column = f"{args.primary_metric}_mean"
    per_dataset["configuration"] = per_dataset.apply(
        lambda row: (
            str(row["method"])
            if pd.isna(row["context_specification"])
            else f"{row['method']}[k={row['context_specification']}]"
        ),
        axis=1,
    )
    comparable_frames = []
    comparison_groups = ["benchmark", "evaluation_split"]
    for _, benchmark_rows in per_dataset.groupby(comparison_groups, dropna=False):
        dataset_sets = [
            set(configuration_rows["dataset_key"])
            for _, configuration_rows in benchmark_rows.groupby("configuration")
        ]
        common_datasets = set.intersection(*dataset_sets)
        comparable_frames.append(
            benchmark_rows[benchmark_rows["dataset_key"].isin(common_datasets)].copy()
        )
    comparable = pd.concat(comparable_frames, ignore_index=True)
    comparable["rank"] = comparable.groupby(
        [*comparison_groups, "dataset_key"], dropna=False
    )[score_column].rank(method="average", ascending=args.lower_is_better)
    ranks = (
        comparable.groupby([*comparison_groups, "configuration"], dropna=False)
        .agg(
            mean_rank=("rank", "mean"),
            common_datasets=("dataset_key", "nunique"),
            mean_score=(score_column, "mean"),
        )
        .reset_index()
        .sort_values([*comparison_groups, "mean_rank"])
    )
    ranks.to_csv(args.output_dir / "average_ranks.csv", index=False)
    print(ranks.to_string(index=False))
    if not errors.empty:
        print(f"\n{len(errors)} failed fold(s); see {args.output_dir / 'errors.csv'}")


if __name__ == "__main__":
    main()
