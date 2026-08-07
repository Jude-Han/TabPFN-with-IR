#!/usr/bin/env python3
"""Print stable dataset identifiers for benchmark process sharding."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabpfn_ir.data import (
    discover_localpfn_dataset_directories,
    load_openml_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_MANIFEST = REPOSITORY_ROOT / "data/manifests/tabpfn_v1_30.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["tabpfn-v1", "localpfn"], required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_V1_MANIFEST)
    parser.add_argument("--tabzilla-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark == "tabpfn-v1":
        for entry in load_openml_manifest(args.manifest).datasets:
            print(entry.dataset_id)
        return

    if args.tabzilla_root is None:
        raise ValueError("--tabzilla-root is required for the LoCalPFN benchmark.")
    for path in discover_localpfn_dataset_directories(args.tabzilla_root):
        print(path.name)


if __name__ == "__main__":
    main()
