#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/inspect_local_finetuning_best.sh RESULTS.jsonl

Optional environment filters:
  DATASET_ID=31
  TRAIN_QUERY_SIZE=256
  CHECKPOINT_TAG=creditg-fold7-q256-lr1e-5
  PYTHON_COMMAND=python

Run this command from the same working directory used for fine-tuning if the
JSONL file contains relative checkpoint paths.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

results_path="$1"
if [[ ! -f "${results_path}" ]]; then
  echo "Results file not found: ${results_path}" >&2
  exit 2
fi

python_command="${PYTHON_COMMAND:-python}"

"${python_command}" - "${results_path}" <<'PY'
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


def optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def format_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def resolve_checkpoint_directory(raw_path: str, results_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        (Path.cwd() / path).resolve(),
        (results_path.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:  # compatibility with older torch releases
        return torch.load(path, **kwargs)


results_path = Path(sys.argv[1]).expanduser().resolve()
dataset_id = optional_int("DATASET_ID")
train_query_size = optional_int("TRAIN_QUERY_SIZE")
checkpoint_tag = os.environ.get("CHECKPOINT_TAG") or None

records_by_fold: dict[int, dict[str, Any]] = {}
errors: list[str] = []
for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON at {results_path}:{line_number}: {exc}") from exc

    if record.get("status") != "ok":
        errors.append(
            f"line={line_number} fold={record.get('fold')} "
            f"{record.get('error_type')}: {record.get('error')}"
        )
        continue
    if dataset_id is not None and record.get("dataset_id") != dataset_id:
        continue
    if checkpoint_tag is not None and record.get("checkpoint_tag") != checkpoint_tag:
        continue
    configuration = record.get("fine_tuning_configuration") or {}
    if train_query_size is not None and configuration.get("train_query_size") != train_query_size:
        continue

    fold = record.get("fold")
    if not isinstance(fold, int):
        continue
    records_by_fold[fold] = record

if not records_by_fold:
    filters = {
        "DATASET_ID": dataset_id,
        "TRAIN_QUERY_SIZE": train_query_size,
        "CHECKPOINT_TAG": checkpoint_tag,
    }
    raise SystemExit(f"No successful records matched the filters: {filters}")

print("Matched filters:")
print(f"  DATASET_ID={dataset_id if dataset_id is not None else '*'}")
print(f"  TRAIN_QUERY_SIZE={train_query_size if train_query_size is not None else '*'}")
print(f"  CHECKPOINT_TAG={checkpoint_tag if checkpoint_tag is not None else '*'}")
print(f"  folds={sorted(records_by_fold)}")
print()

header = (
    f"{'fold':>4}  {'epoch':>5}  {'val_auc':>9}  {'val_logloss':>11}  "
    f"{'test_auc':>9}  {'test_logloss':>12}  {'test_balacc':>11}  checkpoint"
)
print(header)
print("-" * len(header))

rows: list[dict[str, Any]] = []
for fold in sorted(records_by_fold):
    record = records_by_fold[fold]
    result = record.get("result") or {}
    test_metrics = result.get("metrics") or {}
    checkpoint_directory = resolve_checkpoint_directory(
        str(record.get("checkpoint_directory", "")),
        results_path,
    )
    best_candidates = sorted(checkpoint_directory.glob("checkpoint_*_best.pth"))

    best_epoch: object = None
    validation_auc: object = None
    validation_log_loss: object = None
    checkpoint_label = "MISSING"
    if best_candidates:
        best_path = best_candidates[-1]
        checkpoint = load_checkpoint(best_path)
        best_epoch = checkpoint.get("epoch")
        validation_auc = checkpoint.get("roc_auc")
        validation_log_loss = checkpoint.get("log_loss")
        checkpoint_label = str(best_path)
        del checkpoint
        gc.collect()
    elif checkpoint_directory.exists():
        checkpoint_label = "NO_SAVED_BEST"

    row = {
        "fold": fold,
        "best_epoch": best_epoch,
        "validation_auc": validation_auc,
        "validation_log_loss": validation_log_loss,
        "test_auc": test_metrics.get("roc_auc"),
        "test_log_loss": test_metrics.get("log_loss"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
    }
    rows.append(row)
    epoch_text = "-" if best_epoch is None else str(best_epoch)
    print(
        f"{fold:>4}  {epoch_text:>5}  {format_number(validation_auc):>9}  "
        f"{format_number(validation_log_loss):>11}  "
        f"{format_number(row['test_auc']):>9}  "
        f"{format_number(row['test_log_loss']):>12}  "
        f"{format_number(row['test_balanced_accuracy']):>11}  "
        f"{checkpoint_label}"
    )

print("\nCross-validation test summary (do not select a fold by its test score):")
for label, key in (
    ("ROC AUC", "test_auc"),
    ("LogLoss", "test_log_loss"),
    ("Balanced accuracy", "test_balanced_accuracy"),
):
    values = [float(row[key]) for row in rows if row[key] is not None]
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else float("nan")
    print(
        f"  {label:18s} mean={mean:.6f} sd={standard_deviation:.6f} "
        f"count={len(values)}"
    )

saved_best_count = sum(row["best_epoch"] is not None for row in rows)
print(f"\nSaved best checkpoints: {saved_best_count}/{len(rows)}")
if saved_best_count != len(rows):
    print(
        "NO_SAVED_BEST means that no post-training epoch beat the initial validation "
        "model sufficiently to create a best checkpoint. The JSONL test result can "
        "still be valid because the in-memory initial/best weights were restored."
    )

missing_folds = sorted(set(range(10)) - set(records_by_fold))
if missing_folds:
    print(f"WARNING: missing successful folds: {missing_folds}", file=sys.stderr)
if errors:
    print(f"Ignored error records: {len(errors)}", file=sys.stderr)
PY
