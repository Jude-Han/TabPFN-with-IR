#!/usr/bin/env bash
set -euo pipefail

dataset_id="${1:?Usage: scripts/run_knn_sweep.sh DATASET_ID [TARGET]}"
target_name="${2:-}"
context_sizes="${K_VALUES:-32 64 128 256 512 1000 localpfn}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for context_size in ${context_sizes}; do
  CONTEXT_SIZE="${context_size}" \
  EVALUATION_SPLIT="${EVALUATION_SPLIT:-validation}" \
    "${repository_dir}/scripts/run_knn.sh" "${dataset_id}" "${target_name}"
done
