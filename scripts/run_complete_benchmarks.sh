#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_COMMAND:-python}"
benchmarks="${BENCHMARKS:-tabpfn-v1}"
methods="${METHODS:-full random knn}"
context_sizes="${K_VALUES:-128 256 512 1000 localpfn}"
random_ratios="${RANDOM_RATIOS:-}"
tabzilla_root="${TABZILLA_ROOT:-${1:-}}"
output_root="${OUTPUT_ROOT:-${repository_dir}/outputs/complete}"
summary_dir="${SUMMARY_DIR:-${output_root}/summary}"
parallel_shards="${PARALLEL_SHARDS:-1}"
gpu_ids_text="${GPU_IDS:-0 1 2 3}"
device="${DEVICE:-cuda:0}"
devices="${DEVICES:-}"
evaluation_split="${EVALUATION_SPLIT:-test}"

if ! [[ "${parallel_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_SHARDS must be a positive integer." >&2
  exit 2
fi

read -r -a gpu_ids <<< "${gpu_ids_text}"
if (( parallel_shards > ${#gpu_ids[@]} )); then
  echo "PARALLEL_SHARDS=${parallel_shards} exceeds GPU_IDS count ${#gpu_ids[@]}." >&2
  exit 2
fi

mkdir -p "${output_root}" "${summary_dir}"
result_files=()

list_dataset_identifiers() {
  local benchmark="$1"
  if [[ "${benchmark}" == "tabpfn-v1" ]]; then
    "${python_command}" scripts/list_benchmark_datasets.py --benchmark tabpfn-v1
  else
    "${python_command}" scripts/list_benchmark_datasets.py \
      --benchmark localpfn --tabzilla-root "${tabzilla_root}"
  fi
}

run_one_worker() {
  local benchmark="$1"
  local worker_device="$2"
  local filter_value="$3"
  local output="$4"

  if [[ "${benchmark}" == "tabpfn-v1" ]]; then
    DEVICE="${worker_device}" \
    DEVICES="" \
    DATASET_IDS="${filter_value}" \
    DATASET_NAMES="" \
    METHODS="${methods}" \
    K_VALUES="${context_sizes}" \
    RANDOM_RATIOS="${random_ratios}" \
    EVALUATION_SPLIT="${evaluation_split}" \
    OUTPUT="${output}" \
      "${repository_dir}/scripts/run_tabpfn_v1_benchmark.sh"
  else
    DEVICE="${worker_device}" \
    DEVICES="" \
    DATASET_IDS="" \
    DATASET_NAMES="${filter_value}" \
    METHODS="${methods}" \
    K_VALUES="${context_sizes}" \
    RANDOM_RATIOS="${random_ratios}" \
    EVALUATION_SPLIT="${evaluation_split}" \
    OUTPUT="${output}" \
      "${repository_dir}/scripts/run_localpfn_benchmark.sh" "${tabzilla_root}"
  fi
}

run_one_benchmark() {
  local benchmark="$1"
  local benchmark_output_dir="${output_root}/${benchmark}"
  mkdir -p "${benchmark_output_dir}/logs"

  if (( parallel_shards == 1 )); then
    local output="${benchmark_output_dir}/results.jsonl"
    result_files+=("${output}")
    if [[ "${benchmark}" == "tabpfn-v1" ]]; then
      DEVICE="${device}" \
      DEVICES="${devices}" \
      DATASET_IDS="" \
      DATASET_NAMES="" \
      METHODS="${methods}" \
      K_VALUES="${context_sizes}" \
      RANDOM_RATIOS="${random_ratios}" \
      EVALUATION_SPLIT="${evaluation_split}" \
      OUTPUT="${output}" \
        "${repository_dir}/scripts/run_tabpfn_v1_benchmark.sh"
    else
      DEVICE="${device}" \
      DEVICES="${devices}" \
      DATASET_IDS="" \
      DATASET_NAMES="" \
      METHODS="${methods}" \
      K_VALUES="${context_sizes}" \
      RANDOM_RATIOS="${random_ratios}" \
      EVALUATION_SPLIT="${evaluation_split}" \
      OUTPUT="${output}" \
        "${repository_dir}/scripts/run_localpfn_benchmark.sh" "${tabzilla_root}"
    fi
    return
  fi

  local listing
  listing="$(list_dataset_identifiers "${benchmark}")"
  local identifiers=()
  while IFS= read -r identifier; do
    if [[ -n "${identifier}" ]]; then
      identifiers+=("${identifier}")
    fi
  done <<< "${listing}"
  if (( ${#identifiers[@]} == 0 )); then
    echo "No datasets were found for ${benchmark}." >&2
    return 1
  fi

  local shard_filters=()
  local shard_index
  for ((shard_index = 0; shard_index < parallel_shards; shard_index++)); do
    shard_filters[shard_index]=""
  done
  local position
  for position in "${!identifiers[@]}"; do
    shard_index=$((position % parallel_shards))
    shard_filters[shard_index]="${shard_filters[shard_index]} ${identifiers[position]}"
  done

  local pids=()
  local labels=()
  for ((shard_index = 0; shard_index < parallel_shards; shard_index++)); do
    local gpu_id="${gpu_ids[shard_index]}"
    local worker_device="cuda:${gpu_id}"
    local output="${benchmark_output_dir}/shard-cuda${gpu_id}.jsonl"
    local log="${benchmark_output_dir}/logs/shard-cuda${gpu_id}.log"
    result_files+=("${output}")
    echo "Launching ${benchmark} shard ${shard_index} on ${worker_device}; log: ${log}"
    (
      run_one_worker \
        "${benchmark}" "${worker_device}" "${shard_filters[shard_index]}" "${output}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    labels+=("${benchmark}/cuda:${gpu_id}")
  done

  local failed=0
  for position in "${!pids[@]}"; do
    if wait "${pids[position]}"; then
      echo "Completed ${labels[position]}"
    else
      echo "Failed ${labels[position]}; inspect its log file." >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    return 1
  fi
}

cd "${repository_dir}"
echo "Benchmarks: ${benchmarks}"
echo "Methods: ${methods}"
echo "Context sizes: ${context_sizes}"
echo "Parallel dataset shards: ${parallel_shards}"

for benchmark in ${benchmarks}; do
  case "${benchmark}" in
    tabpfn-v1)
      run_one_benchmark "${benchmark}"
      ;;
    localpfn)
      if [[ -z "${tabzilla_root}" ]]; then
        echo "TABZILLA_ROOT or the first positional argument is required for localpfn." >&2
        exit 2
      fi
      run_one_benchmark "${benchmark}"
      ;;
    *)
      echo "Unknown benchmark '${benchmark}'; use tabpfn-v1 and/or localpfn." >&2
      exit 2
      ;;
  esac
done

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  "${python_command}" scripts/summarize_benchmark.py \
    "${result_files[@]}" --output-dir "${summary_dir}"
  echo "Summary written to ${summary_dir}"
fi
