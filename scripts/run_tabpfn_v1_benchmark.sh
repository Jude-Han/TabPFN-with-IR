#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_COMMAND:-python}"
benchmark="${BENCHMARK:-tabpfn-v1}"
device="${DEVICE:-cuda}"
seed="${EXPERIMENT_SEED:-42}"
split_seed="${SPLIT_SEED:-0}"
evaluation_split="${EVALUATION_SPLIT:-test}"
methods="${METHODS:-full random knn}"
k_values="${K_VALUES:-localpfn}"
maximum_context_size="${MAXIMUM_CONTEXT_SIZE:-1000}"
random_ratios="${RANDOM_RATIOS:-}"
case "${benchmark}" in
  tabpfn-v1|openml-cc18) ;;
  *)
    echo "BENCHMARK must be tabpfn-v1 or openml-cc18." >&2
    exit 2
    ;;
esac
output="${OUTPUT:-${repository_dir}/outputs/${benchmark}/results.jsonl}"
model_version="${MODEL_VERSION:-v2.6}"
context_batch_size="${CONTEXT_BATCH_SIZE:-32}"
inference_profile="${INFERENCE_PROFILE:-single-estimator}"
extra_args=(
  --model-version "${model_version}"
  --context-batch-size "${context_batch_size}"
  --inference-profile "${inference_profile}"
  --maximum-context-size "${maximum_context_size}"
  --split-seed "${split_seed}"
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
if [[ "${inference_profile}" == "single-estimator" ]]; then
  # This benchmark profile intentionally takes precedence over a global shell value.
  extra_args+=(--n-estimators 1)
elif [[ -n "${N_ESTIMATORS:-}" ]]; then
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
if [[ -n "${MANIFEST:-}" ]]; then
  extra_args+=(--manifest "${MANIFEST}")
fi

cd "${repository_dir}"
for method in ${methods}; do
  if [[ "${method}" == "full" ]]; then
    "${python_command}" scripts/run_benchmark.py \
      --benchmark "${benchmark}" --method full --device "${device}" --seed "${seed}" \
      --evaluation-split "${evaluation_split}" --output "${output}" --resume \
      "${extra_args[@]}"
    continue
  fi
  if [[ "${method}" == "random" && -n "${random_ratios}" ]]; then
    for ratio in ${random_ratios}; do
      "${python_command}" scripts/run_benchmark.py \
        --benchmark "${benchmark}" --method random --random-ratio "${ratio}" \
        --device "${device}" --seed "${seed}" --evaluation-split "${evaluation_split}" \
        --output "${output}" --resume \
        "${extra_args[@]}"
    done
    continue
  fi
  for k in ${k_values}; do
    "${python_command}" scripts/run_benchmark.py \
      --benchmark "${benchmark}" --method "${method}" --k "${k}" \
      --device "${device}" --seed "${seed}" --evaluation-split "${evaluation_split}" \
      --output "${output}" --resume \
      "${extra_args[@]}"
  done
done
