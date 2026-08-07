#!/usr/bin/env python3
"""List OpenML dataset IDs referenced by the OpenML-CC18 benchmark suite."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-id",
        type=int,
        default=99,
        help="OpenML benchmark suite ID; 99 is OpenML-CC18.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import openml
    except ImportError as exc:
        raise SystemExit(
            "OpenML is not installed. Run `python -m pip install -e '.[benchmark]'`."
        ) from exc

    suite = openml.study.get_suite(args.suite_id)
    print("task_id\tdataset_id\tdataset_version\ttarget\tdataset_name")
    for task_id in suite.tasks:
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        print(
            f"{task_id}\t{task.dataset_id}\t{dataset.version}\t"
            f"{task.target_name}\t{dataset.name}"
        )


if __name__ == "__main__":
    main()
