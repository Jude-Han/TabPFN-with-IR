#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_COMMAND:-python}"
benchmark="${BENCHMARK:-openml-cc18}"
output="${OUTPUT:-${repository_dir}/outputs/local-finetuning/results.jsonl}"
checkpoint_root="${CHECKPOINT_ROOT:-${repository_dir}/outputs/local-finetuning/checkpoints}"

args=(
  --benchmark "${benchmark}"
  --model-version "${MODEL_VERSION:-v2.6}"
  --k "${CONTEXT_SIZE:-localpfn}"
  --maximum-context-size "${MAXIMUM_CONTEXT_SIZE:-1000}"
  --train-query-size "${TRAIN_QUERY_SIZE:-1000}"
  --steps-per-epoch "${STEPS_PER_EPOCH:-30}"
  --episode-batch-size "${EPISODE_BATCH_SIZE:-2}"
  --epochs "${EPOCHS:-30}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --grad-clip-value "${GRAD_CLIP_VALUE:-1.0}"
  --scheduler "${SCHEDULER:-cosine}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-8}"
  --min-delta "${MIN_DELTA:-1e-4}"
  --eval-metric "${EVAL_METRIC:-roc_auc}"
  --n-estimators-finetune "${N_ESTIMATORS_FINETUNE:-2}"
  --n-estimators-validation "${N_ESTIMATORS_VALIDATION:-2}"
  --n-estimators-final "${N_ESTIMATORS_FINAL:-2}"
  --save-checkpoint-interval "${SAVE_CHECKPOINT_INTERVAL:-10}"
  --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE:-512}"
  --context-batch-size "${CONTEXT_BATCH_SIZE:-32}"
  --device "${DEVICE:-cuda}"
  --seed "${EXPERIMENT_SEED:-0}"
  --split-seed "${SPLIT_SEED:-0}"
  --checkpoint-tag "${CHECKPOINT_TAG:-default}"
  --checkpoint-root "${checkpoint_root}"
  --output "${output}"
  --resume
)

if [[ "${EARLY_STOPPING:-1}" == "0" ]]; then
  args+=(--no-early-stopping)
fi
if [[ "${ACTIVATION_CHECKPOINTING:-1}" == "0" ]]; then
  args+=(--no-activation-checkpointing)
fi
if [[ "${FIXED_PREPROCESSING_SEED:-1}" == "0" ]]; then
  args+=(--no-fixed-preprocessing-seed)
fi
if [[ "${TRAINING_HISTORY:-1}" == "0" ]]; then
  args+=(--no-training-history)
fi
if [[ "${TENSORBOARD:-0}" == "1" ]]; then
  args+=(--tensorboard)
fi
if [[ -n "${TIME_LIMIT:-}" ]]; then
  args+=(--time-limit "${TIME_LIMIT}")
fi
if [[ -n "${MAX_QUERY_SAMPLES:-}" ]]; then
  args+=(--max-query-samples "${MAX_QUERY_SAMPLES}")
fi
if [[ -n "${MAX_VALIDATION_SAMPLES:-}" ]]; then
  args+=(--max-validation-samples "${MAX_VALIDATION_SAMPLES}")
fi
if [[ -n "${FOLDS:-}" ]]; then
  read -r -a fold_values <<< "${FOLDS}"
  args+=(--folds "${fold_values[@]}")
fi
if [[ -n "${DATASET_IDS:-}" ]]; then
  read -r -a dataset_ids <<< "${DATASET_IDS}"
  args+=(--dataset-ids "${dataset_ids[@]}")
fi
if [[ -n "${DATASET_NAMES:-}" ]]; then
  read -r -a dataset_names <<< "${DATASET_NAMES}"
  args+=(--dataset-names "${dataset_names[@]}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  args+=(--fail-fast)
fi

if [[ "${benchmark}" == "localpfn" ]]; then
  tabzilla_root="${1:-${TABZILLA_ROOT:-}}"
  if [[ -z "${tabzilla_root}" ]]; then
    echo "Set TABZILLA_ROOT or pass it as the first argument for BENCHMARK=localpfn." >&2
    exit 2
  fi
  args+=(--tabzilla-root "${tabzilla_root}")
  if [[ "${ALLOW_COUNT_MISMATCH:-0}" == "1" ]]; then
    args+=(--allow-count-mismatch)
  fi
elif [[ -n "${MANIFEST:-}" ]]; then
  args+=(--manifest "${MANIFEST}")
fi

cd "${repository_dir}"
"${python_command}" scripts/run_local_finetuning.py "${args[@]}"
