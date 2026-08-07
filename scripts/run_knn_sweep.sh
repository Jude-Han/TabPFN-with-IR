#!/usr/bin/env bash
set -euo pipefail

dataset_id="${1:?Usage: scripts/run_knn_sweep.sh DATASET_ID [TARGET]}"
target_name="${2:-}"
context_sizes="${K_VALUES:-32 64 128 256 512 1000 localpfn}"
experiment_seed="${EXPERIMENT_SEED:-42}"
evaluation_split="${EVALUATION_SPLIT:-validation}"
python_command="${PYTHON_COMMAND:-python}"
device="${DEVICE:-auto}"
fit_mode="${FIT_MODE:-fit_preprocessors}"
model_version="${MODEL_VERSION:-v2.6}"
context_batch_size="${CONTEXT_BATCH_SIZE:-32}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${OUTPUT_DIR:-${repository_dir}/outputs/dataset-${dataset_id}}"

for context_size in ${context_sizes}; do
  common_args=(
    --dataset-id "${dataset_id}"
    --method knn
    --k "${context_size}"
    --seed "${experiment_seed}"
    --device "${device}"
    --fit-mode "${fit_mode}"
    --model-version "${model_version}"
    --context-batch-size "${context_batch_size}"
    --evaluation-split "${evaluation_split}"
    --output "${output_dir}/knn-k${context_size}-seed-${experiment_seed}-${evaluation_split}.json"
  )
  if [[ -n "${DEVICES:-}" ]]; then
    read -r -a selected_devices <<< "${DEVICES}"
    common_args+=(--devices "${selected_devices[@]}")
  fi
  if [[ "${IGNORE_PRETRAINING_LIMITS:-0}" == "1" ]]; then
    common_args+=(--ignore-pretraining-limits)
  fi
  if [[ -n "${N_ESTIMATORS:-}" ]]; then
    common_args+=(--n-estimators "${N_ESTIMATORS}")
  fi
  if [[ "${DISABLE_BATCHED_CONTEXTS:-0}" == "1" ]]; then
    common_args+=(--disable-batched-contexts)
  fi
  if [[ -n "${target_name}" ]]; then
    common_args+=(--target "${target_name}")
  fi
  if [[ -n "${DATASET_VERSION:-}" ]]; then
    common_args+=(--dataset-version "${DATASET_VERSION}")
  fi

  cd "${repository_dir}"
  "${python_command}" scripts/run_baseline.py "${common_args[@]}"
done
