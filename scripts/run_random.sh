#!/usr/bin/env bash
set -euo pipefail

dataset_id="${1:?Usage: scripts/run_random.sh DATASET_ID [TARGET]}"
target_name="${2:-}"
context_size="${CONTEXT_SIZE:-128}"
random_seeds="${RANDOM_SEEDS:-0 1 2 3 4}"
evaluation_split="${EVALUATION_SPLIT:-test}"
python_command="${PYTHON_COMMAND:-python}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${OUTPUT_DIR:-${repository_dir}/outputs/dataset-${dataset_id}}"

for experiment_seed in ${random_seeds}; do
  common_args=(
    --dataset-id "${dataset_id}"
    --method random
    --k "${context_size}"
    --seed "${experiment_seed}"
    --evaluation-split "${evaluation_split}"
    --output "${output_dir}/random-k${context_size}-seed-${experiment_seed}-${evaluation_split}.json"
  )
  if [[ -n "${target_name}" ]]; then
    common_args+=(--target "${target_name}")
  fi
  if [[ -n "${DATASET_VERSION:-}" ]]; then
    common_args+=(--dataset-version "${DATASET_VERSION}")
  fi

  cd "${repository_dir}"
  "${python_command}" scripts/run_baseline.py "${common_args[@]}"
done
