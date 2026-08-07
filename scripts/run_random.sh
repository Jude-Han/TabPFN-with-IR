#!/usr/bin/env bash
set -euo pipefail

dataset_id="${1:?Usage: scripts/run_random.sh DATASET_ID [TARGET]}"
target_name="${2:-}"
context_size="${CONTEXT_SIZE:-128}"
context_ratio="${CONTEXT_RATIO:-}"
random_seeds="${RANDOM_SEEDS:-0 1 2 3 4}"
evaluation_split="${EVALUATION_SPLIT:-test}"
python_command="${PYTHON_COMMAND:-python}"
device="${DEVICE:-auto}"
fit_mode="${FIT_MODE:-fit_preprocessors}"
model_version="${MODEL_VERSION:-v2.6}"
context_batch_size="${CONTEXT_BATCH_SIZE:-32}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${OUTPUT_DIR:-${repository_dir}/outputs/dataset-${dataset_id}}"

for experiment_seed in ${random_seeds}; do
  if [[ -n "${context_ratio}" ]]; then
    context_args=(--random-ratio "${context_ratio}")
    context_label="ratio${context_ratio}"
  else
    context_args=(--k "${context_size}")
    context_label="k${context_size}"
  fi
  common_args=(
    --dataset-id "${dataset_id}"
    --method random
    "${context_args[@]}"
    --seed "${experiment_seed}"
    --device "${device}"
    --fit-mode "${fit_mode}"
    --model-version "${model_version}"
    --context-batch-size "${context_batch_size}"
    --evaluation-split "${evaluation_split}"
    --output "${output_dir}/random-${context_label}-seed-${experiment_seed}-${evaluation_split}.json"
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
