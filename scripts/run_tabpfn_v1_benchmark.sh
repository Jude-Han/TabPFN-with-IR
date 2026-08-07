#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_COMMAND:-python}"
device="${DEVICE:-cuda}"
seed="${EXPERIMENT_SEED:-42}"
evaluation_split="${EVALUATION_SPLIT:-test}"
methods="${METHODS:-full random knn}"
k_values="${K_VALUES:-localpfn}"
random_ratios="${RANDOM_RATIOS:-}"
output="${OUTPUT:-${repository_dir}/outputs/tabpfn-v1/results.jsonl}"
model_version="${MODEL_VERSION:-v2.6}"
context_batch_size="${CONTEXT_BATCH_SIZE:-32}"
extra_args=(
  --model-version "${model_version}"
  --context-batch-size "${context_batch_size}"
)

if [[ -n "${DEVICES:-}" ]]; then
  read -r -a selected_devices <<< "${DEVICES}"
  extra_args+=(--devices "${selected_devices[@]}")
fi
if [[ "${IGNORE_PRETRAINING_LIMITS:-0}" == "1" ]]; then
  extra_args+=(--ignore-pretraining-limits)
fi
if [[ -n "${FIT_MODE:-}" ]]; then
  extra_args+=(--fit-mode "${FIT_MODE}")
fi
if [[ -n "${N_ESTIMATORS:-}" ]]; then
  extra_args+=(--n-estimators "${N_ESTIMATORS}")
fi
if [[ "${DISABLE_BATCHED_CONTEXTS:-0}" == "1" ]]; then
  extra_args+=(--disable-batched-contexts)
fi

if [[ -n "${FOLDS:-}" ]]; then
  read -r -a fold_values <<< "${FOLDS}"
  extra_args+=(--folds "${fold_values[@]}")
fi
if [[ -n "${DATASET_IDS:-}" ]]; then
  read -r -a dataset_ids <<< "${DATASET_IDS}"
  extra_args+=(--dataset-ids "${dataset_ids[@]}")
fi
if [[ -n "${DATASET_NAMES:-}" ]]; then
  read -r -a dataset_names <<< "${DATASET_NAMES}"
  extra_args+=(--dataset-names "${dataset_names[@]}")
fi

cd "${repository_dir}"
for method in ${methods}; do
  if [[ "${method}" == "full" ]]; then
    "${python_command}" scripts/run_benchmark.py \
      --benchmark tabpfn-v1 --method full --device "${device}" --seed "${seed}" \
      --evaluation-split "${evaluation_split}" --output "${output}" --resume \
      "${extra_args[@]}"
    continue
  fi
  if [[ "${method}" == "random" && -n "${random_ratios}" ]]; then
    for ratio in ${random_ratios}; do
      "${python_command}" scripts/run_benchmark.py \
        --benchmark tabpfn-v1 --method random --random-ratio "${ratio}" \
        --device "${device}" --seed "${seed}" --evaluation-split "${evaluation_split}" \
        --output "${output}" --resume \
        "${extra_args[@]}"
    done
    continue
  fi
  for k in ${k_values}; do
    "${python_command}" scripts/run_benchmark.py \
      --benchmark tabpfn-v1 --method "${method}" --k "${k}" \
      --device "${device}" --seed "${seed}" --evaluation-split "${evaluation_split}" \
      --output "${output}" --resume \
      "${extra_args[@]}"
  done
done
