#!/usr/bin/env python3
"""List fixed OpenML-CC18 tasks and LoCalPFN-style test-query counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabpfn_ir.data import load_openml_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/manifests/openml_cc18.json"


def query_fold_sizes(n_instances: int, *, n_splits: int = 10) -> tuple[int, ...]:
    """Return the test sizes produced by a balanced ``n_splits`` partition."""

    if n_instances <= 0:
        raise ValueError("n_instances must be positive.")
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than one.")
    quotient, remainder = divmod(n_instances, n_splits)
    return (quotient + 1,) * remainder + (quotient,) * (n_splits - remainder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--format",
        choices=["tsv", "markdown"],
        default="tsv",
        help="Output table format.",
    )
    return parser.parse_args()


def _row(entry) -> list[str]:
    if entry.n_instances is None:
        raise ValueError(f"Dataset {entry.dataset_id} has no n_instances in the manifest.")
    sizes = query_fold_sizes(entry.n_instances)
    return [
        str(entry.task_id),
        str(entry.dataset_id),
        str(entry.version),
        str(entry.target),
        entry.name,
        str(entry.n_instances),
        ",".join(map(str, sizes)),
        str(sum(sizes)),
    ]


def main() -> None:
    args = parse_args()
    manifest = load_openml_manifest(args.manifest)
    headings = [
        "task_id",
        "dataset_id",
        "version",
        "target",
        "dataset_name",
        "n_instances",
        "test_queries_folds_0_to_9",
        "test_queries_all_folds",
    ]
    rows = [_row(entry) for entry in manifest.datasets]

    if args.format == "markdown":
        print("| " + " | ".join(headings) + " |")
        print("| " + " | ".join(["---"] * len(headings)) + " |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        print("\t".join(headings))
        for row in rows:
            print("\t".join(row))

    aggregate_fold_sizes = [
        sum(query_fold_sizes(entry.n_instances)[fold] for entry in manifest.datasets)
        for fold in range(10)
    ]
    print(
        f"# datasets={len(rows)} "
        f"queries_per_complete_10_fold_benchmark={sum(int(row[-1]) for row in rows)} "
        f"aggregate_test_queries_folds_0_to_9="
        f"{','.join(map(str, aggregate_fold_sizes))}"
    )


if __name__ == "__main__":
    main()
