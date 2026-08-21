#!/usr/bin/env bash
set -euo pipefail

# Run independent dataset shards on separate GPUs. This is intentionally
# process-level parallelism: each dataset/fold owns one TabPFN fine-tuning job.

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_COMMAND:-}" ]]; then
  python_command="${PYTHON_COMMAND}"
elif [[ -x "${repository_dir}/.venv/bin/python" ]]; then
  python_command="${repository_dir}/.venv/bin/python"
else
  python_command="python"
fi
benchmark="${BENCHMARK:-openml-cc18}"
model_version="${MODEL_VERSION:-v2.6}"
model_version_slug="${model_version//./}"
manifest="${MANIFEST:-${repository_dir}/data/manifests/tabpfn_v1_30.json}"
tabzilla_root="${TABZILLA_ROOT:-}"
gpu_ids_text="${GPU_IDS:-0 1 2 3}"
output_root="${OUTPUT_ROOT:-${repository_dir}/outputs/local-finetuning/parallel-${model_version_slug}}"
checkpoint_root="${CHECKPOINT_ROOT:-${output_root}/checkpoints}"
checkpoint_tag="${CHECKPOINT_TAG:-parallel-${model_version_slug}}"
dry_run="${DRY_RUN:-0}"

read -r -a gpu_ids <<< "${gpu_ids_text}"
if (( ${#gpu_ids[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU identifier." >&2
  exit 2
fi

parallel_shards="${PARALLEL_SHARDS:-${#gpu_ids[@]}}"
if ! [[ "${parallel_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_SHARDS must be a positive integer." >&2
  exit 2
fi
if (( parallel_shards > ${#gpu_ids[@]} )); then
  echo "PARALLEL_SHARDS=${parallel_shards} exceeds GPU_IDS count ${#gpu_ids[@]}." >&2
  exit 2
fi
if [[ "${benchmark}" != "openml-cc18" && "${benchmark}" != "localpfn" ]]; then
  echo "BENCHMARK must be openml-cc18 or localpfn." >&2
  exit 2
fi
if [[ "${benchmark}" == "localpfn" && -z "${tabzilla_root}" ]]; then
  echo "TABZILLA_ROOT is required for BENCHMARK=localpfn." >&2
  exit 2
fi

cd "${repository_dir}"
identifiers=()
if [[ "${benchmark}" == "openml-cc18" ]]; then
  if [[ -n "${DATASET_IDS:-}" ]]; then
    read -r -a identifiers <<< "${DATASET_IDS}"
  else
    while IFS= read -r identifier; do
      if [[ -n "${identifier}" ]]; then
        identifiers+=("${identifier}")
      fi
    done < <(
      "${python_command}" scripts/list_benchmark_datasets.py \
        --benchmark openml-cc18 \
        --manifest "${manifest}"
    )
  fi
else
  if [[ -n "${DATASET_NAMES:-}" ]]; then
    read -r -a identifiers <<< "${DATASET_NAMES}"
  else
    while IFS= read -r identifier; do
      if [[ -n "${identifier}" ]]; then
        identifiers+=("${identifier}")
      fi
    done < <(
      "${python_command}" scripts/list_benchmark_datasets.py \
        --benchmark localpfn \
        --tabzilla-root "${tabzilla_root}"
    )
  fi
fi

if (( ${#identifiers[@]} == 0 )); then
  echo "No datasets were selected." >&2
  exit 2
fi

shard_filters=()
for ((shard_index = 0; shard_index < parallel_shards; shard_index++)); do
  shard_filters[shard_index]=""
done
for position in "${!identifiers[@]}"; do
  shard_index=$((position % parallel_shards))
  shard_filters[shard_index]="${shard_filters[shard_index]} ${identifiers[position]}"
done

mkdir -p "${output_root}/logs" "${checkpoint_root}"
echo "Fine-tuning benchmark: ${benchmark}"
echo "Model version: ${model_version}"
echo "Dataset count: ${#identifiers[@]}"
echo "Parallel GPU workers: ${parallel_shards}"
echo "Checkpoint root: ${checkpoint_root}"

pids=()
labels=()
cleanup_workers() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_workers INT TERM

for ((shard_index = 0; shard_index < parallel_shards; shard_index++)); do
  filter="${shard_filters[shard_index]# }"
  if [[ -z "${filter}" ]]; then
    continue
  fi
  gpu_id="${gpu_ids[shard_index]}"
  output="${output_root}/results-shard-${shard_index}-gpu-${gpu_id}.jsonl"
  log="${output_root}/logs/shard-${shard_index}-gpu-${gpu_id}.log"
  echo "Shard ${shard_index} -> physical GPU ${gpu_id}: ${filter}"
  echo "  result: ${output}"
  echo "  log:    ${log}"

  if [[ "${dry_run}" == "1" ]]; then
    continue
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export BENCHMARK="${benchmark}"
    export MODEL_VERSION="${model_version}"
    export DEVICE="cuda"
    export OUTPUT="${output}"
    export CHECKPOINT_ROOT="${checkpoint_root}"
    export CHECKPOINT_TAG="${checkpoint_tag}"
    export MANIFEST="${manifest}"
    export TABZILLA_ROOT="${tabzilla_root}"
    export PYTHON_COMMAND="${python_command}"
    if [[ "${benchmark}" == "openml-cc18" ]]; then
      export DATASET_IDS="${filter}"
      export DATASET_NAMES=""
    else
      export DATASET_IDS=""
      export DATASET_NAMES="${filter}"
    fi
    "${repository_dir}/scripts/run_local_finetuning.sh"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  labels+=("shard-${shard_index}/gpu-${gpu_id}")
done

if [[ "${dry_run}" == "1" ]]; then
  echo "DRY_RUN=1: no workers were launched."
  exit 0
fi

failed=0
for position in "${!pids[@]}"; do
  if wait "${pids[position]}"; then
    echo "Completed ${labels[position]}"
  else
    echo "Failed ${labels[position]}; inspect its shard log." >&2
    failed=1
  fi
done
trap - INT TERM

if (( failed != 0 )); then
  exit 1
fi
echo "All fine-tuning shards completed successfully."
