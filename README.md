# TabPFN with Information Retrieval

This repository studies whether a plug-and-play information retrieval (IR) module can make tabular foundation models such as TabPFN effective on datasets that are larger than their practical in-context learning (ICL) budget.

Instead of passing an entire training set to TabPFN, the retriever selects a small, query-specific set of rows. These rows and their labels are then used as the ICL examples for TabPFN. The `full`, `random`, and `knn` baselines keep the backbone frozen. The implemented `local-ft` workflow additionally fine-tunes all TabPFN v2.6 weights on LoCalPFN-style local context/query episodes. The long-term goal is to learn a retrieval policy that selects examples for their usefulness to TabPFN, rather than relying only on geometric proximity in the original feature space.

## Research question

Given a training dataset

```text
D_train = {(x_i, y_i)}
```

and a test query `x_q`, we retrieve a context `C_q` containing at most `k` labeled training rows and predict with

```text
p(y_q | x_q, C_q) = TabPFN(x_q, C_q).
```

The central question is:

> Which training rows form the most useful context for TabPFN when the complete dataset cannot be used as context?

The retrieval component is intended to be model-agnostic and plug-and-play: the TabPFN backbone remains frozen, while only the method used to construct its context changes.

## Datasets

The initial benchmark will combine:

- classification datasets from OpenML-CC18 that satisfy the selected TabPFN constraints; and
- the datasets used in the LoCalPFN experiments, based on the TabZilla/OpenML benchmark.

Dataset identifiers, preprocessing decisions, excluded datasets, and exclusion reasons will be recorded in a versioned manifest. Duplicate datasets across the two collections will be resolved by OpenML dataset ID rather than by name.

The first phase focuses on classification. Regression can be added later as a separate benchmark because it requires different metrics and may require a different TabPFN checkpoint or interface.

### Choosing an OpenML dataset ID

The command-line runner expects an OpenML **dataset ID**, not an OpenML task ID. The integer in an
OpenML dataset URL such as `https://www.openml.org/d/31` is the dataset ID; in this example, `31`
identifies the `credit-g` dataset. A dataset ID resolves to a particular uploaded dataset version. The
optional `--dataset-version` argument is therefore used as a validation check rather than to select a
different version behind the same ID.

OpenML-CC18 is benchmark suite `99`. Its 72 fixed tasks are recorded in
`data/manifests/openml_cc18.json`. List task IDs, dataset IDs, versions, targets, and the exact
test-query count of every LoCalPFN-style fold without downloading the datasets:

```bash
python scripts/list_openml_cc18.py
```

CC18 task IDs describe benchmark tasks and predefined evaluation settings. The `openml-cc18` runner
uses each task's dataset and target, then reproduces TabZilla/LoCalPFN's split construction:
`StratifiedKFold(n_splits=10, shuffle=True, random_state=0)`, with fold `i` as test, fold `i+1` as
validation, and the remaining eight folds as training data. These are intentionally not OpenML's
official 90/10 task folds.

Each fold should maintain two aligned views of its rows:

- `X_model`: the representation expected by TabPFN; and
- `X_retrieval`: the fold-fitted representation used only to compute retrieval scores.

Retrievers return row indices, not transformed rows. The selected indices are always applied to `X_model` before TabPFN inference. This separation ensures that a retrieval method is not accidentally credited for changing the predictor's input preprocessing.

## Experimental baselines

All methods use the same data splits, preprocessing, TabPFN backbone, context budget, and prediction settings. They differ only in how the labeled context rows are selected.

### 1. Full-context TabPFN

Use the complete training split as TabPFN context whenever the model and available memory permit it.

This experiment measures the performance of vanilla TabPFN with all available rows. Because full-context inference may be infeasible for large datasets, any context truncation, model-imposed limit, or out-of-memory result must be reported explicitly rather than silently converted into another sampling strategy.

### 2. Random retrieval

Uniformly sample `k` rows without replacement from the training split and use the sampled rows as context.

Random retrieval is the main context-budget-matched baseline. Results should be averaged over multiple retrieval seeds because the sampled context can have high variance. The same sampled global context may be shared by all test queries in the primary baseline; an optional query-wise random variant can be reported separately.

### 3. LoCalPFN-style kNN retrieval (without fine-tuning)

For every query row, retrieve its `k` nearest training rows using a distance that decomposes over features:

```text
d(x_q, x_i) = sum_j d_j(x_qj, x_ij).
```

The initial implementation will use standardized numerical features and an explicit encoding for categorical features. The exact feature-wise distance, encoding, missing-value policy, and scaling method will be treated as part of the experimental configuration.

Only the retrieval/sampling idea is used in this baseline. Unlike the complete LoCalPFN method, this project does **not** perform supervised fine-tuning of TabPFN for this experiment. The frozen TabPFN model receives a different local context for each query.

The paper's heuristic

```text
k = min(10 * sqrt(n_train), 1000)
```

will be evaluated alongside fixed context budgets so that methods can also be compared at exactly the same `k`.

### 4. LoCalPFN-style kNN retrieval with task fine-tuning

The `local-ft` workflow starts from the Hugging Face TabPFN v2.6 classifier checkpoint and performs
end-to-end supervised fine-tuning for one dataset fold. Training examples are local meta-datasets:
random anchor rows define exact-FAISS neighborhoods, each neighborhood is shuffled into a labeled
context and a supervised query set, and cross-entropy is computed only on the query labels. AdamW
updates the complete TabPFN transformer, not merely a prediction head or an added adapter.

Validation and test inference use the same exact query-specific kNN path as the frozen `knn`
baseline. The validation fold is used for early stopping; test labels are used only after the best
weights have been selected. This implementation is LoCalPFN-style rather than a bit-for-bit
reproduction: the original paper fine-tuned the older TabPFN architecture, whereas this repository
deliberately adapts the pinned v2.6 checkpoint through TabPFN 8.2.0's official fine-tuning engine.

## Proposed IR extension

The next phase will replace hand-designed nearest-neighbor retrieval with an IR model that assigns a relevance score to every candidate row:

```text
s_phi(x_q, x_i, y_i, dataset_metadata) -> relevance score.
```

The top-`k` candidates under this score become the context for the frozen TabPFN predictor. Candidate generation and reranking can be separated for scalability:

1. A fast index retrieves a larger candidate pool of `M` rows, where `M > k`.
2. A learned reranker scores query-candidate pairs.
3. A selector constructs a context of `k` rows while optionally enforcing class coverage and diversity.
4. TabPFN predicts from the selected labeled context.

Possible retriever objectives include:

- **metric learning:** learn an embedding in which useful query-context pairs are close;
- **pairwise ranking:** rank a helpful row above a less helpful row for the same query;
- **listwise ranking:** optimize the ordering or composition of a complete candidate set; and
- **downstream utility learning:** define relevance by the improvement in frozen TabPFN validation loss when a row or subset is included in the context.

The primary research target is not to imitate feature-space kNN, but to retrieve rows that maximize downstream TabPFN performance. Labels may be used by the selector only for training rows that are already available in the retrieval corpus; query and validation/test labels must never be exposed to retrieval or context construction at inference time.

### Learned-retriever evaluation regimes

Two settings should be reported separately:

- **Within-dataset adaptation:** train or tune the retriever using only the current dataset's training fold, select hyperparameters on its validation fold, and evaluate on its test fold.
- **Cross-dataset generalization:** train the retriever on a collection of source datasets and evaluate it without retriever training on entirely held-out target datasets.

The second setting is the stronger test of a foundation-style plug-and-play retriever. Dataset-level train/development/test partitions must be fixed before producing retrieval supervision so that queries derived from a target dataset cannot leak into retriever training.

## Evaluation protocol

### Data splitting and leakage prevention

- Use a fixed train/validation/test split, or a documented cross-validation protocol, shared by every method.
- Fit imputers, categorical encoders, scalers, retrieval models, and indexes on the training fold only.
- Tune `k`, preprocessing choices, and retrieval hyperparameters using the validation fold only.
- Build every test-query context exclusively from labeled rows in the corresponding training fold.
- Detect duplicate or near-duplicate rows across splits and report the policy used to handle them.

### Metrics

The primary classification metrics are ROC AUC (binary or macro one-vs-one for multiclass tasks) and log loss. The paper-benchmark runner uses OVO to match TabPFN v1 and LoCalPFN. Accuracy, balanced accuracy, and weighted F1 are included as secondary metrics. If a dataset does not support a metric in a particular fold, that dataset-fold result is saved as a documented failure rather than silently omitted.

Results will be reported both per dataset and in aggregate. Aggregate summaries should include average rank and a robust statistic such as the interquartile mean, together with confidence intervals computed across datasets or by an explicitly documented stratified bootstrap.

### Efficiency

In addition to predictive performance, the benchmark will record:

- index construction time;
- retrieval latency per query;
- TabPFN inference time;
- peak CPU and GPU memory;
- index size; and
- the number of context rows actually used.

### Required ablations

- context size `k`;
- training-set size;
- global versus query-specific random contexts;
- raw standardized, one-hot, and learned retrieval representations;
- exact versus approximate nearest-neighbor search;
- class-balanced and diversity-aware selection; and
- candidate-pool size `M` for learned reranking.

## Repository structure

```text
configs/                 Experiment and dataset configurations
data/manifests/          Versioned OpenML IDs and dataset metadata
src/tabpfn_ir/data/      Downloading, splitting, and preprocessing
src/tabpfn_ir/retrieval/ Full, random, and kNN retrievers
src/tabpfn_ir/models/    Frozen prediction and v2.6 local fine-tuning adapters
src/tabpfn_ir/training/  Local context/query episode sampling
src/tabpfn_ir/evaluation/ Metrics and one-fold experiment runner
scripts/                 Benchmark entry points
tests/                   Leakage, determinism, and retriever tests
outputs/                 Local experiment artifacts (not committed)
```

Every retriever should implement one common interface so that the predictor and evaluator do not contain method-specific logic:

```python
class Retriever:
    def fit(self, X_train, y_train) -> "Retriever":
        ...

    def retrieve(self, X_query, k: int):
        """Return training-row indices and scores with shape [n_query, k]."""
        ...
```

## Quick start

Use Python 3.10-3.12 for the benchmark environment because the TabPFN/PyTorch stack may not yet
support newer Python releases.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[benchmark,dev]"
pytest
```

Run one baseline on an OpenML dataset:

```bash
python scripts/run_baseline.py \
  --dataset-id 31 \
  --target class \
  --method knn \
  --k 128 \
  --output outputs/credit-g-knn-k128.json
```

Choose `full`, `random`, or `knn` with `--method`. Full-context inference uses all training rows.
The initial runner executes one fixed split; manifest-wide sweeps over folds, context budgets, and
random repetitions will be added on top of the same components.

### Baseline command scripts

The convenience scripts take `DATASET_ID` as the first argument and an optional target name as the
second argument. If the target is omitted, the OpenML default target is used.

Full-context TabPFN on the final test split:

```bash
scripts/run_full.sh 31 class
```

Random retrieval with the same context budget over five retrieval seeds (all on split seed 0):

```bash
CONTEXT_SIZE=128 RANDOM_SEEDS="0 1 2 3 4" scripts/run_random.sh 31 class
```

Random retrieval can instead use a fraction of the current **training fold**. For example, 10% uses
`ceil(0.10 * n_train)` rows and always selects at least one row:

```bash
CONTEXT_RATIO=0.10 RANDOM_SEEDS="0 1 2 3 4" scripts/run_random.sh 31 class

# Equivalent direct CLI option
python scripts/run_baseline.py \
  --dataset-id 31 --method random --random-ratio 0.10 \
  --output outputs/credit-g-random-ratio-010.json
```

`CONTEXT_RATIO` takes precedence over `CONTEXT_SIZE` and must satisfy `0 < ratio <= 1`.

One FAISS exact-L2 kNN run with the paper's dynamic context size on fold 0's test split:

```bash
scripts/run_knn.sh 31 class
```

kNN context-size selection over several values on the validation split:

```bash
K_VALUES="32 64 128 256 512 1000 localpfn" scripts/run_knn_sweep.sh 31 class
```

`localpfn` resolves to `min(int(10 * sqrt(n_train)), MAXIMUM_CONTEXT_SIZE)`, where the default
maximum is `1000`. A budget larger than the training fold is capped at `n_train`. Retrieval uses
`faiss.IndexFlatL2` and processes query searches in batches of 512.

Select `k` using validation results only, with one metric chosen before looking at the test set. ROC
AUC is the recommended selection metric for consistency with the main benchmark; use log loss when
probability calibration is the primary target. In a near tie, prefer the smaller `k` because it reduces
memory and inference cost. Then run the selected budget once on the test split for both kNN and
random retrieval:

```bash
CONTEXT_SIZE=128 EVALUATION_SPLIT=test scripts/run_knn.sh 31 class
CONTEXT_SIZE=128 EVALUATION_SPLIT=test scripts/run_random.sh 31 class
```

The environment variables `FOLD`, `SPLIT_SEED`, `MAXIMUM_CONTEXT_SIZE`, `EXPERIMENT_SEED`,
`DATASET_VERSION`, `PYTHON_COMMAND`, and `OUTPUT_DIR` can override the remaining defaults.

### TabPFN v2.6 installation and authentication

The benchmark environment is pinned to `tabpfn==8.2.0`. Upgrade the existing environment after
installing a CUDA-compatible PyTorch build:

```bash
python -m pip install --upgrade -e ".[benchmark,dev]"
python -c "from importlib.metadata import version; print(version('tabpfn'))"
```

The second command must print `8.2.0`. TabPFN 8.x defaults to the v3 checkpoint, but this project
deliberately constructs the classifier with
`TabPFNClassifier.create_default_for_version(ModelVersion.V2_6, ...)`. Therefore the expected model
download remains `tabpfn-v2.6-classifier-v2.6_default.ckpt`. The CLI also records
`"model_version": "v2.6"` in every result, and currently rejects any other value to prevent an
accidental backbone change during the retrieval comparison.

For headless servers, store the Prior Labs API key in the repository-root `.env`. The populated file
is ignored by Git:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` so it contains exactly the following assignment, without shell commands around it:

```dotenv
TABPFN_TOKEN=your_actual_prior_labs_api_key
```

`run_baseline.py` and `run_benchmark.py` call `load_dotenv(..., override=False)` before constructing
TabPFN. Therefore a token exported by the job scheduler or shell takes precedence over `.env`. Check
that a token is visible without printing the secret:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('TABPFN_TOKEN configured:', bool(os.getenv('TABPFN_TOKEN')))"
```

The API key does not itself accept a model license. While logged into the same Prior Labs account,
accept the `tabpfn_2_6` license at the URL reported by TabPFN. If the program still raises
`TabPFNLicenseError: License not yet accepted` while the check above prints `True`, the account-side
license acceptance is the remaining step rather than `.env` loading.

The downloaded checkpoint and fine-tuned derivatives remain subject to the
[TabPFN v2.6 model license](https://huggingface.co/Prior-Labs/tabpfn_2_6). Review that license before
sharing checkpoints or using them outside the permitted research/internal-evaluation scope; the
repository's source-code license does not replace the model license.

## Local fine-tuning on TabPFN v2.6

### What is implemented

`scripts/run_local_finetuning.py` implements Method A: it retains the official TabPFN 8.2.0
fine-tuning engine and changes the construction of its training meta-datasets. Consequently the
official implementation still controls model loading, full-parameter AdamW optimization, automatic
mixed precision on CUDA, gradient clipping, activation checkpointing, LR scheduling, DDP,
checkpoint saving/resume, and best-weight restoration. The project-specific adapter controls local
episode sampling, the batched episode loss, exact-kNN validation, and exact-kNN test inference.

The workflow is version-gated to `tabpfn==8.2.0` because it uses a narrow private data-construction
hook that is not part of TabPFN's stable public API. It also passes `ModelVersion.V2_6` explicitly and
rejects `model_path` overrides. This prevents a future TabPFN default or a different checkpoint from
silently changing an experiment.

TabPFN 8.2.0 also has an activation-checkpointing defect specific to the v2.6 architecture: its
forward pass removes a tensor from a mutable list, so backward recomputation receives an empty list
and raises `IndexError: pop from empty list`. The
[upstream v8.3.0 fix](https://github.com/PriorLabs/TabPFN/commit/fa9b33344bc37b8a896f40efaf9ec5f331057615)
changed the checkpoint path to pass the tensor directly. While activation checkpointing is enabled,
this project applies a scoped compatibility backport with the same effect: the installed package is
not edited, the input list remains reusable during recomputation, and the original method is restored
after `fit` exits.
The applied backport is recorded in every JSONL fine-tuning configuration.

An ordinary `TabPFNClassifier.fit(X, y)` call is still ICL context registration and does **not**
update weights. Supervised updates occur only through `LocalFinetunedTabPFNClassifier.fit(...)` or
the local fine-tuning runner.

### Training and inference flow

For every train/validation/test fold, the implementation performs the following operations:

1. Fit the ordinal model view and standardized numerical plus one-hot categorical retrieval view on
   the training fold only. Validation and test rows are transformed without refitting.
2. Resolve the local context budget `k`. `localpfn` means
   `min(int(10 * sqrt(n_train)), maximum_context_size)`.
3. Build one exact `faiss.IndexFlatL2` index over the training retrieval view and one small exact
   index per class.
4. At every epoch, sample `steps_per_epoch * episode_batch_size` random training anchors. For each
   anchor, retrieve `k + train_query_size` nearby training rows.
5. If a neighborhood omits a training class, replace one of its farthest redundant-class rows with
   the nearest row from the missing class. Then reserve one row per class for context, shuffle the
   remaining rows, fill a context of size `k`, and use the rest as supervised queries. This guarantees
   that every query label is representable by its context and that TabPFN 8.2's episode-local label
   encoder has a stable class dimension. It is a documented v2.6 compatibility adjustment; it can
   add a small number of non-global-kNN rows on extremely class-local or imbalanced data.
6. Fuse `episode_batch_size` independent, equal-shaped local datasets into one TabPFN forward.
   Cross-entropy is averaged over all query rows and fine-tuning ensemble members. Backpropagation
   updates every parameter in the v2.6 backbone.
7. After each epoch, evaluate the changing weights on the held-out validation fold. Every validation
   row receives its own exact `k`-neighbor training context. Macro OVO ROC AUC or negative log loss
   drives early stopping, and the best weights are retained.
8. Run final test inference once, again using exact train-only kNN contexts. Test labels never enter
   retrieval, optimization, checkpoint selection, or early stopping.

With the defaults `steps_per_epoch=30` and `episode_batch_size=2`, validation happens after 30
optimizer steps and 60 local neighborhoods. This matches the paper's evaluation cadence of every 30
gradient steps and its local batch size of two. On a small fold, the context budget is preserved and
`train_query_size` is shortened so at least one query row remains.

### Smoke test

Use one dataset, one fold, smaller episodes, and a single epoch before a real run:

```bash
python scripts/run_local_finetuning.py \
  --benchmark openml-cc18 \
  --dataset-ids 31 \
  --folds 0 \
  --k 128 \
  --train-query-size 32 \
  --steps-per-epoch 1 \
  --episode-batch-size 1 \
  --epochs 1 \
  --n-estimators-finetune 1 \
  --n-estimators-validation 1 \
  --n-estimators-final 1 \
  --max-validation-samples 32 \
  --max-query-samples 32 \
  --device cuda \
  --output outputs/local-finetuning/smoke.jsonl \
  --checkpoint-root outputs/local-finetuning/checkpoints \
  --fail-fast
```

`--max-validation-samples` and `--max-query-samples` are smoke-test controls. Do not use them for
final benchmark numbers.

Run a normal OpenML-CC18 fold with the convenience wrapper:

```bash
DATASET_IDS=31 \
FOLDS=0 \
DEVICE=cuda \
CONTEXT_SIZE=localpfn \
EPOCHS=30 \
scripts/run_local_finetuning.sh
```

Run a stored TabZilla/LoCalPFN fold:

```bash
BENCHMARK=localpfn \
DATASET_NAMES="<tabzilla-directory-name>" \
FOLDS=0 \
DEVICE=cuda \
scripts/run_local_finetuning.sh /path/to/tabzilla/TabZilla/datasets
```

The precise directory name depends on the local TabZilla metadata. Inspect available names with:

```bash
python scripts/list_benchmark_datasets.py \
  --benchmark localpfn \
  --tabzilla-root /path/to/tabzilla/TabZilla/datasets
```

### Multi-GPU fine-tuning

TabPFN's official fine-tuning example recommends a CUDA GPU with 80 GB of VRAM. Actual memory here
depends strongly on `k`, `train_query_size`, the episode batch, and the number of fine-tuning
estimators. Activation checkpointing is enabled by default so smaller GPUs can often run reduced
configurations.

When several datasets must be fine-tuned, prefer dataset-level process sharding: one independent
worker is pinned to each physical GPU, and the selected dataset identifiers are assigned
round-robin. Each worker processes all of its dataset folds sequentially and writes a separate
resumable JSONL file and log. This keeps unrelated dataset/fold models isolated and normally gives
better throughput than using four GPUs for one small fold.

The following command distributes the 30 datasets in the TabPFN-v1 manifest across four GPUs and
runs all ten LoCalPFN-style folds. It uses the validation-log-loss setting demonstrated on
MiceProtein; choose the global learning rate using validation experiments on several representative
datasets before treating this as a final benchmark configuration.

```bash
GPU_IDS="0 1 2 3" \
PARALLEL_SHARDS=4 \
BENCHMARK=openml-cc18 \
MANIFEST=data/manifests/tabpfn_v1_30.json \
FOLDS="0 1 2 3 4 5 6 7 8 9" \
CONTEXT_SIZE=localpfn \
TRAIN_QUERY_SIZE=500 \
LEARNING_RATE=3e-5 \
WEIGHT_DECAY=0.01 \
SCHEDULER=cosine \
GRAD_CLIP_VALUE=1.0 \
EVAL_METRIC=log_loss \
EPOCHS=30 \
EARLY_STOPPING_PATIENCE=8 \
MIN_DELTA=1e-8 \
N_ESTIMATORS_FINETUNE=2 \
N_ESTIMATORS_VALIDATION=2 \
N_ESTIMATORS_FINAL=2 \
SAVE_CHECKPOINT_INTERVAL=5 \
CHECKPOINT_TAG=v26-logloss-lr3e-5 \
OUTPUT_ROOT=outputs/local-finetuning/v26-logloss-30datasets \
CHECKPOINT_ROOT=outputs/local-finetuning/checkpoints-v26-logloss-30datasets \
scripts/run_parallel_local_finetuning.sh
```

Omit `DATASET_IDS` to use every dataset in `MANIFEST`, as above. To run a selected subset, add a
space-separated list such as `DATASET_IDS="31 40966 40975"`. Preview the exact allocation without
launching CUDA processes:

```bash
DRY_RUN=1 \
GPU_IDS="0 1 2 3" \
PARALLEL_SHARDS=4 \
DATASET_IDS="31 40966 40975 40982 40994" \
scripts/run_parallel_local_finetuning.sh
```

Every worker sees only its assigned physical GPU through `CUDA_VISIBLE_DEVICES` and therefore uses
`DEVICE=cuda` internally. Result and log files have names such as
`results-shard-0-gpu-0.jsonl` and `logs/shard-0-gpu-0.log`; checkpoint paths remain collision-safe
because they include benchmark, dataset, fold, tag, and configuration hash. Re-running the same
command resumes interrupted numbered checkpoints and skips successful JSONL records within each
shard.

Before the first four-worker launch, cache the TabPFN checkpoint and OpenML datasets with a
single-process smoke run. Concurrent first-time downloads by four workers can otherwise contend for
the same cache. Monitor all workers with:

```bash
tail -f outputs/local-finetuning/v26-logloss-30datasets/logs/*.log
nvidia-smi -l 2
```

`SAVE_CHECKPOINT_INTERVAL=5` is a compromise for a 300-fold run. Use `1` for a short diagnostic,
but it can consume substantial disk space. `TENSORBOARD=1` may also be supplied; each dataset/fold
writes events under its own checkpoint directory.

Fine-tuning uses data-distributed training rather than the inference-only `--devices` option. Launch
one fold with `torchrun`:

```bash
torchrun --nproc-per-node=4 scripts/run_local_finetuning.py \
  --benchmark openml-cc18 \
  --dataset-ids 31 \
  --folds 0 \
  --device cuda \
  --k localpfn \
  --output outputs/local-finetuning/credit-g-fold0.jsonl
```

Every rank constructs the same deterministic episode pool, and TabPFN's distributed sampler assigns
different episodes to ranks. Only rank 0 validates, writes checkpoints, performs final inference,
and appends the JSONL result. A `torchrun` invocation must select exactly one dataset and one fold;
submit separate jobs for additional dataset-fold pairs. Pre-cache OpenML data before a multi-process
run if the local OpenML cache does not support concurrent first downloads safely.

### Hyperparameters to set deliberately

The defaults are safe starting points for v2.6, not universal optima. Select them with the validation
fold only, then run the test fold once.

| CLI option | Default | What it controls | How to tune it |
| --- | ---: | --- | --- |
| `--k` | `localpfn` | Context rows used in every training episode and every validation/test query. This is the most important locality/bias/variance parameter. | Try `128, 256, 512, 1000, localpfn`. Smaller values are cheaper and more local; larger values improve coverage but can blur local boundaries and increase memory. It must be at least the number of training classes. |
| `--maximum-context-size` | `1000` | Cap in the dynamic LoCalPFN formula. | Lower it first when the dynamic context does not fit memory. Keep it fixed across compared methods. |
| `--train-query-size` | `1000` | Supervised query rows inside each local neighborhood. More rows reduce gradient noise but lengthen attention and consume memory. | Start with `1000` when the fold and GPU allow it. Try `128, 256, 512` on 24 GB-class GPUs. Small folds automatically shorten it after preserving `k`. |
| `--episode-batch-size` | `2` | Independent anchor neighborhoods fused into one optimizer step; this is the paper's `B`. | Keep `2` for the closest paper-style run. Reduce to `1` first after OOM. Larger values require careful LR retuning and are not a default recommendation. |
| `--steps-per-epoch` | `30` | Optimizer updates between validations. Together with epochs, it defines the training budget. | Increase to `60` or `100` only if validation is still improving and validation overhead is acceptable. Lower it for rapid feedback or expensive validation folds. |
| `--epochs` | `30` | Maximum validation cycles. Early stopping may finish earlier. | Use `30-100` with early stopping. Compare runs by optimizer steps (`epochs * steps_per_epoch`), not epochs alone. |
| `--learning-rate` | `1e-5` | AdamW step size for every v2.6 weight. This is the highest-risk parameter. | Start at `1e-5`; validate `3e-6, 1e-5, 3e-5`. Reduce it when loss spikes or validation immediately degrades. The paper's `0.01` was for the older TabPFN code and is intentionally not the v2.6 default. |
| `--weight-decay` | `0.01` | AdamW regularization. | Usually keep `0.01`; try `0` or `0.001` if the task is tiny and underfitting, or a slightly larger value only with clear overfitting. |
| `--scheduler` | `cosine` | `cosine` uses 10% linear warmup followed by cosine decay; `warmup-only` stays at the base LR after warmup; `none` keeps a constant LR. | Use `cosine` first. `none` is the closest paper setting but should be combined with a conservative v2.6 LR. |
| `--grad-clip-value` | `1.0` | Maximum global gradient norm; `0` disables clipping. | Keep `1.0` unless measuring proves it unnecessary. Lower it if gradients/loss are unstable; disabling it is not recommended for an initial run. |
| `--eval-metric` | `roc_auc` | Validation objective for best-checkpoint selection. `log_loss` is internally maximized as negative log loss. | Use `roc_auc` for paper comparison and `log_loss` when calibrated probabilities are the primary goal. Choose before viewing test results. |
| `--early-stopping-patience` | `8` | Validation cycles allowed without a `min_delta` improvement. | Use `5-15`. Increase it for noisy small validation folds; decrease it for expensive, clearly saturated runs. |
| `--min-delta` | `1e-4` | Smallest validation change counted as an improvement. | Keep `1e-4` initially. Raise it to stop earlier when metric noise creates meaningless checkpoint churn. |
| `--n-estimators-finetune` | `2` | TabPFN preprocessing/ensemble members included in each training loss. | Reduce to `1` after OOM. Increasing it improves augmentation/ensemble exposure but roughly multiplies work. |
| `--n-estimators-validation` | `2` | Ensemble members used for every early-stopping evaluation. | Keep equal to the fine-tuning value for comparable preprocessing. Reduce only when validation dominates runtime. |
| `--n-estimators-final` | `2` | Ensemble members used for final exact-kNN inference. | Increase to `4` or `8` only after the training configuration is selected and memory/runtime allow it. Record it because it changes predictions. |
| `--activation-checkpointing` | enabled | Recomputes transformer activations during backward to reduce peak memory. The adapter automatically applies the scoped TabPFN 8.2.0 v2.6 empty-list compatibility backport. | Leave enabled unless profiling shows abundant memory and recomputation is the runtime bottleneck. `--no-activation-checkpointing` is also a quick diagnostic for any future checkpoint-related incompatibility. |
| `--fixed-preprocessing-seed` | enabled | Keeps TabPFN's feature/preprocessing randomization stable across episodes. | Leave enabled for lower training noise and reproducibility. Keep all three estimator counts equal when possible. |
| `--retrieval-batch-size` | `512` | CPU FAISS query chunking during episode, validation, and test retrieval. | It normally affects speed, not neighbors. Lower it only for host-memory pressure. |
| `--context-batch-size` | `32` | Number of shape/class-compatible query contexts fused during local validation/test inference. | Try `8, 16, 32, 64`. Lower after inference OOM; changing it should not change intended predictions. |
| `--save-checkpoint-interval` | `10` | Interval checkpoint cadence. `0` disables interval checkpoints, while improving best checkpoints remain enabled when early stopping is enabled. | Use `1` for maximum interruption recovery, at substantial disk cost. The official resume path resumes numbered interval checkpoints, not a best-only checkpoint. |
| `--training-history` | enabled | Writes every optimizer-step loss/LR, the true epoch mean loss, and validation metrics to `training_history.jsonl`. | Leave enabled. Use `--no-training-history` only when per-step audit data is explicitly unwanted. |
| `--tensorboard` | disabled | Mirrors the same scalars into TensorBoard event files. | Enable with `TENSORBOARD=1` after installing the `tracking` extra. JSONL logging remains the dependency-free source of record. |
| `--time-limit` | none | Wall-clock training limit in seconds. | Use for managed clusters. The loop stops before starting an epoch that is unlikely to finish within the remaining budget. |
| `--seed` | `0` | Anchor sampling, TabPFN initialization/preprocessing, and data-loader order. | Run at least three seeds for a stability study after choosing hyperparameters. Do not use test performance to select a seed. |

When memory is insufficient, reduce settings in this order: set
`episode_batch_size=1`, set all estimator counts to `1`, reduce `train_query_size`, then reduce `k`.
Keep activation checkpointing enabled. `context_batch_size` primarily controls validation/test
inference memory and does not solve backward-pass OOM.

A practical validation search is:

```text
k:             128, 256, 512, localpfn
learning rate: 3e-6, 1e-5, 3e-5
query size:    256, 512, 1000 (subject to memory)
```

First choose a feasible `k` and query size, then tune the learning rate. Avoid a full Cartesian grid
unless compute permits it. Keep split, seed set, training-step budget, and inference estimator count
fixed while comparing configurations.

### Paper settings versus v2.6 defaults

The LoCalPFN paper used `k=min(10*sqrt(n_train),1000)`, 1,000 training query rows, local batch size
two, AdamW with learning rate `0.01` and weight decay `0.01`, no warmup/scheduler, validation AUC
evaluation every 30 gradient steps, and exact-kNN inference batches of 512. This implementation maps
`k`, query size, batch size, weight decay, evaluation cadence, metric, and retrieval batch size
directly. It intentionally changes the starting LR to `1e-5`, enables gradient clipping, activation
checkpointing, and a warmup/cosine schedule because the fine-tuned backbone is TabPFN v2.6 rather
than the paper's old checkpoint. Use `--scheduler none` for the scheduler ablation, but do not copy
the old `0.01` LR without a separate stability study.

### Outputs, checkpoints, and resume behavior

Each successful JSONL row records the complete fine-tuning configuration, package versions,
effective context/query sizes, preprocessing/training/retrieval/prediction timings, context batching
diagnostics, classification metrics, and checkpoint directory. Failures are appended with their
exception type and message.

Checkpoint paths contain benchmark, dataset, fold, user tag, and a hash of all fine-tuning
hyperparameters. Re-running the same path allows TabPFN's official loop to resume from its latest
numbered interval checkpoint. `--resume` additionally skips configurations already marked successful
in the JSONL output. Change `--checkpoint-tag` to create an intentionally separate run lineage; do
not manually mix checkpoints produced by different hyperparameters.

On a first run, TabPFN may warn that the output directory exists but contains no checkpoint. This is
expected: the runner created the directory, TabPFN found nothing to resume, and training starts from
the original v2.6 checkpoint at epoch zero. It is unrelated to the activation-checkpointing error.

Each checkpoint directory now also contains `training_history.jsonl` and
`training_summary.json`. The history separates the noisy loss of each optimizer step from the true
mean loss over an epoch. The summary reports the initial validation metric, best validation metric,
best epoch, and whether training ever improved over the untouched v2.6 model. The final result JSONL
also records the number and names of `.pth` files and the best-checkpoint path, if one exists.

#### Diagnosing flat loss and empty checkpoint directories

The `loss=...` value at the right edge of tqdm is the most recent local episode batch, not an epoch
average. Different anchors generate different kNN neighborhoods, so this number is inherently noisy
and need not decrease monotonically. Inspect `train/mean_loss` in the history before concluding that
optimization is stalled.

There is a more serious and recognizable failure mode with the literal LoCalPFN learning rate. The
paper's `learning_rate=0.01` belonged to an older TabPFN checkpoint/training stack. When all weights
of the already-strong v2.6 checkpoint are updated at that rate, the first few steps can erase the
pretrained solution. Cross-entropy then settles near `log(C)`, where `C` is the class count: this is
the loss of uniform class probabilities. For example, an eight-class loss near `log(8)=2.079` is
model collapse, not evidence of successful convergence. The runner now emits a warning for learning
rates at or above `1e-3`.

An empty checkpoint directory is also possible by design:

1. `SAVE_CHECKPOINT_INTERVAL=0` maps to `save_checkpoint_interval=None`, so periodic `.pth` files are
   disabled.
2. TabPFN writes `checkpoint_<train-size>_best.pth` only when a post-update validation epoch improves
   over the initial, untouched v2.6 model.
3. If no epoch improves, no best file is written and early stopping restores the initial weights in
   memory for final inference.

Consequently, a prior run with interval saving disabled and no `_best.pth` did not produce reusable
fine-tuned weights. Its final test score can still be valid for local-kNN-context inference, but the
weights used for that score are the restored pretrained weights. A numbered interval checkpoint
proves that optimizer state was saved; it does **not** prove that the checkpoint is better. Use the
validation-selected `_best.pth` and `training_summary.json` for that conclusion.

For a fresh v2.6 stability run, use a new output file and checkpoint tag so `--resume` does not skip
the old completed record:

```bash
DATASET_IDS=40966 \
FOLDS=0 \
DEVICE=cuda \
CONTEXT_SIZE=localpfn \
TRAIN_QUERY_SIZE=500 \
LEARNING_RATE=1e-5 \
SCHEDULER=cosine \
GRAD_CLIP_VALUE=1.0 \
EPOCHS=30 \
EARLY_STOPPING_PATIENCE=8 \
MIN_DELTA=1e-4 \
SAVE_CHECKPOINT_INTERVAL=1 \
CHECKPOINT_TAG=miceprotein-v26-stable-lr1e-5 \
OUTPUT=outputs/local-finetuning/miceprotein-v26-stable-lr1e-5.jsonl \
scripts/run_local_finetuning.sh
```

Use interval `1` for this diagnosis so every epoch is recoverable. After stability is established,
use `5` or `10` to reduce disk consumption. Select the learning rate on validation data from
`3e-6`, `1e-5`, and `3e-5`; do not select it by the test score.

The dependency-free plotter creates an SVG from the new history:

```bash
python scripts/plot_finetuning_history.py \
  outputs/local-finetuning/checkpoints/<benchmark>/<dataset>/fold-0/<run>/training_history.jsonl \
  --output outputs/local-finetuning/fold-0-learning-curve.svg
```

It can also diagnose a legacy terminal log, with the limitation that only tqdm's displayed batch
loss—not the true historical epoch mean—is available:

```bash
python scripts/plot_finetuning_history.py \
  outputs/local-finetuning/paper-literal-resume-shard-2.log \
  --dataset-id 40966 \
  --fold 5 \
  --output outputs/local-finetuning/miceprotein-fold5-paper-literal-loss.svg
```

For live monitoring, install TensorBoard and enable its logger:

```bash
pip install -e '.[benchmark,tracking]'
TENSORBOARD=1 SAVE_CHECKPOINT_INTERVAL=1 scripts/run_local_finetuning.sh
tensorboard --logdir outputs/local-finetuning/checkpoints
```

Open the URL printed by TensorBoard and compare `train/loss`, `train/mean_loss`, `train/lr`, and the
validation curves. JSONL history continues to be written even if the optional TensorBoard backend
cannot start.

Programmatic use is available through `LocalFinetunedTabPFNClassifier`. It deliberately requires
both aligned feature views and an explicit validation fold:

```python
from pathlib import Path

from tabpfn_ir.models import LocalFinetunedTabPFNClassifier

model = LocalFinetunedTabPFNClassifier(
    context_size=512,
    train_query_size=1000,
    steps_per_epoch=30,
    episode_batch_size=2,
    learning_rate=1e-5,
    epochs=30,
    device="cuda",
)
model.fit(
    X_train_model,
    y_train,
    X_train_retrieval=X_train_retrieval,
    X_val_model=X_validation_model,
    y_val=y_validation,
    X_val_retrieval=X_validation_retrieval,
    output_dir=Path("outputs/checkpoints/my-fold"),
)
prediction = model.predict_proba_local(X_test_model, X_test_retrieval)
probabilities = prediction.probabilities
classes = prediction.classes
```

The explicit `predict_proba_local` name is intentional: final prediction must receive the retrieval
view and must not accidentally fall back to a global full-context prediction.

## Multi-GPU TabPFN inference

Run full-context inference using all four GPUs explicitly:

```bash
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" \
FIT_MODE=fit_preprocessors \
N_ESTIMATORS=8 \
scripts/run_full.sh 6
```

On TabPFN 8.2, `DEVICE=auto` also selects all visible CUDA GPUs. An explicit `DEVICES` list is useful
for reproducibility and GPU allocation. Multi-GPU inference parallelizes ensemble estimators; it does
not partition one training context's rows across GPU memory. It is supported for
`FIT_MODE=fit_preprocessors` and `FIT_MODE=low_memory`, but not `fit_with_cache`. Choose at least as
many estimators as GPUs; a multiple such as 8 or 16 generally distributes work more evenly.

For kNN, TabPFN 8.2's `predict_proba_batched` API also fuses several independent query contexts into
one forward pass. The adapter groups contexts only when their context shape, query shape, and class
set are compatible, then limits each fused call with `CONTEXT_BATCH_SIZE`. This removes the previous
one-GPU-launch-per-query bottleneck while the explicit device list distributes ensemble work over all
four GPUs:

```bash
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" \
FIT_MODE=fit_preprocessors \
N_ESTIMATORS=8 \
MODEL_VERSION=v2.6 \
CONTEXT_BATCH_SIZE=32 \
CONTEXT_SIZE=128 \
EVALUATION_SPLIT=test \
scripts/run_knn.sh 31 class
```

Start with `CONTEXT_BATCH_SIZE=32` on four RTX 3090 GPUs. Try `64` if memory headroom remains and GPU
utilization is low; reduce it to `16` or `8` after an out-of-memory error. This value changes only
execution chunking, not the retrieved rows or intended predictions. Set
`DISABLE_BATCHED_CONTEXTS=1` to reproduce the legacy sequential path for a runtime/equivalence check.
Each result reports `unique_contexts`, `batched_contexts`, `sequential_contexts`, `context_batches`,
and `used_batched_inference`, while `prediction_seconds` still measures the complete TabPFN stage.

The input-size check is independent of GPU count. If an intentionally pinned model still reports a
pretraining-limit error, it can be overridden explicitly:

```bash
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" \
FIT_MODE=low_memory \
N_ESTIMATORS=8 \
IGNORE_PRETRAINING_LIMITS=1 \
scripts/run_full.sh 6
```

`IGNORE_PRETRAINING_LIMITS=1` only disables the guard. It does not make the model pretrained for that
data scale, reduce memory use, or guarantee valid benchmark quality. Record this setting separately
from runs that stay within the active checkpoint's limits. The same `DEVICES`, `FIT_MODE`,
`N_ESTIMATORS`, `MODEL_VERSION`, `CONTEXT_BATCH_SIZE`, `DISABLE_BATCHED_CONTEXTS`, and
`IGNORE_PRETRAINING_LIMITS` variables are supported by the random, kNN, TabPFN v1, and LoCalPFN
scripts. The TabPFN-v1 benchmark wrapper is the exception to the normal `N_ESTIMATORS` precedence:
its default `INFERENCE_PROFILE=single-estimator` forces one estimator for paper-like evaluation.

In particular, OpenML dataset ID 6 is `letter`: it has 20,000 rows and 26 target classes. The default
80/10/10 runner therefore sends 16,000 training rows to TabPFN, which explains the reported error.
It may subsequently exceed the active checkpoint's built-in class limit as well. Multi-GPU inference
does not remove either semantic limit. Dataset 6 is intentionally absent from the TabPFN v1 30-dataset
manifest and from the LoCalPFN benchmark, whose selected tasks have at most 10 classes. Supporting it
through a many-class decomposition would be a separate experimental method and should not be mixed
into the three current retrieval baselines without being reported separately.

## Paper benchmark runners

The repository includes three paper-oriented benchmark sources:

- **TabPFN v1:** the fixed 30 OpenML dataset IDs in Table 7, stored in
  `data/manifests/tabpfn_v1_30.json`. The runner creates five deterministic stratified 50/50
  train/test splits and reports macro OVO ROC AUC. The paper did not publish its exact random split
  seeds, so these are protocol-compatible reconstructed splits, not the authors' original indices.
- **OpenML-CC18 with LoCalPFN splits:** all 72 tasks in suite 99, fixed in
  `data/manifests/openml_cc18.json`, with the exact TabZilla 10-fold 80/10/10 construction. Passing
  `data/manifests/tabpfn_v1_30.json` as `--manifest` applies the same split protocol only to the
  TabPFN-v1 paper's 30-dataset subset.
- **LoCalPFN:** the classification datasets discovered from a locally preprocessed TabZilla copy,
  using the filters in the public LoCalPFN code: at most 100 features and 10 classes, no regression,
  and exclusion of the four named datasets known to contain missing values. The original stored ten
  TabZilla train/validation/test folds are used without resplitting.

Here “TabPFN v1” names the v1 paper's **dataset and split benchmark**. Prediction still uses the
`TabPFNClassifier` implementation from package version 8.2.0 with the v2.6 classifier checkpoint, so
the resulting numbers are not intended to reproduce the historical v1 checkpoint exactly. Both the
library and checkpoint version are recorded alongside final results.

`scripts/run_tabpfn_v1_benchmark.sh` defaults to the explicit
`INFERENCE_PROFILE=single-estimator` profile. It sets `n_estimators=1` for full, random, and kNN runs,
even when a login shell exports another value such as `N_ESTIMATORS=8`. The resolved value and profile
are stored in every JSONL record and are part of the resume key. To intentionally run a modern
ensemble comparison instead, use `INFERENCE_PROFILE=default N_ESTIMATORS=8`; keep those results
separate from the single-estimator benchmark.

The LoCalPFN paper text states 47 small plus 48 medium/large datasets (95 total), while the dataset
rows visible in its appendix tables do not enumerate all 95. For this reason, benchmark membership
is derived from the official public code and local TabZilla metadata instead of inventing a static
list from the incomplete table. By default, the runner checks that exactly 95 datasets are found.

### Prepare the LoCalPFN/TabZilla data

TabZilla preprocessing is separate from this project's environment and can take substantial time and
disk space:

```bash
git clone https://github.com/naszilla/tabzilla.git
cd tabzilla/TabZilla
python tabzilla_data_preprocessing.py --process_all
```

The path passed to this project is the resulting `tabzilla/TabZilla/datasets` directory. OpenML may
require an API key on servers with download limits.

### Smoke tests

Run one dataset and one fold before starting a complete sweep:

```bash
python scripts/run_benchmark.py \
  --benchmark tabpfn-v1 \
  --dataset-ids 31 \
  --folds 0 \
  --method knn \
  --k localpfn \
  --device cuda:0 \
  --output outputs/smoke-v1.jsonl

python scripts/run_benchmark.py \
  --benchmark openml-cc18 \
  --manifest data/manifests/tabpfn_v1_30.json \
  --dataset-ids 31 \
  --folds 0 \
  --method knn \
  --k localpfn \
  --device cuda:0 \
  --output outputs/smoke-cc18-localpfn-split.jsonl

python scripts/run_benchmark.py \
  --benchmark localpfn \
  --tabzilla-root /path/to/tabzilla/TabZilla/datasets \
  --limit 1 \
  --folds 0 \
  --method random \
  --k 128 \
  --device cuda:0 \
  --allow-count-mismatch \
  --output outputs/smoke-localpfn.jsonl
```

`--max-query-samples N` is available for an even smaller inference smoke test. It must not be used
for final paper numbers. `--dataset-ids` means OpenML **dataset IDs** for the two manifest-backed
benchmarks, but OpenML **task IDs** parsed from TabZilla directory names for LoCalPFN.

The complete 72-dataset, 10-fold CC18 run evaluates 874,726 test queries in total; the TabPFN-v1
30-dataset subset evaluates 32,741. Per-dataset and per-fold counts, plus aggregate fold totals, are
printed with:

```bash
python scripts/list_openml_cc18.py

# The TabPFN-v1 30-dataset subset under the same LoCalPFN split protocol
python scripts/list_openml_cc18.py --manifest data/manifests/tabpfn_v1_30.json
```

The full 72-task suite also contains datasets outside the pinned TabPFN-v2.6 architectural limits,
including tasks with more than 10 classes or more than 100 features. Use the 30-dataset manifest for
the directly compatible TabPFN paper subset; the full-suite runner keeps incompatible outcomes as
explicit error records instead of silently dropping tasks.

### Complete three-baseline sweeps

The convenience scripts run full context, global random sampling, and query-specific kNN. Their
default context budget for random and kNN is the LoCalPFN heuristic; fixed budgets can be supplied
through `K_VALUES`:

```bash
DEVICE=cuda:0 K_VALUES="128 256 512 1000 localpfn" \
  scripts/run_tabpfn_v1_benchmark.sh

BENCHMARK=openml-cc18 \
MANIFEST=data/manifests/tabpfn_v1_30.json \
METHODS=knn \
K_VALUES=localpfn \
DEVICE=cuda:0 \
  scripts/run_tabpfn_v1_benchmark.sh

DEVICE=cuda:0 K_VALUES="128 256 512 1000 localpfn" \
  scripts/run_localpfn_benchmark.sh /path/to/tabzilla/TabZilla/datasets
```

To run the complete TabPFN-v1 benchmark with all three methods while using four
GPUs at the dataset level:

```bash
PARALLEL_SHARDS=4 \
GPU_IDS="0 1 2 3" \
BENCHMARKS="tabpfn-v1" \
METHODS="full random knn" \
K_VALUES="128 256 512 1000 localpfn" \
INFERENCE_PROFILE=single-estimator \
CONTEXT_BATCH_SIZE=32 \
MODEL_VERSION=v2.6 \
scripts/run_complete_benchmarks.sh
```

Run OpenML-CC18 together with the TabPFN-v1 and LoCalPFN suites by also supplying the
preprocessed TabZilla directory:

```bash
TABZILLA_ROOT=/path/to/tabzilla/TabZilla/datasets \
PARALLEL_SHARDS=4 \
GPU_IDS="0 1 2 3" \
BENCHMARKS="tabpfn-v1 openml-cc18 localpfn" \
METHODS="full random knn" \
K_VALUES="128 256 512 1000 localpfn" \
CONTEXT_BATCH_SIZE=32 \
scripts/run_complete_benchmarks.sh
```

In the combined command, the TabPFN-v1 wrapper still forces one estimator by default. The LoCalPFN
wrapper retains its normal behavior and reads `N_ESTIMATORS` when that variable is defined.

With `PARALLEL_SHARDS=4`, dataset identifiers are assigned round-robin to four
independent processes, each pinned to one GPU. Every shard writes its own resumable
JSONL file and log under `outputs/complete/<benchmark>/`; after all workers finish,
the aggregate CSV summaries are written to `outputs/complete/summary/`. This
process-level sharding is intentional: TabPFN 8.2's batched context engine uses only
the first configured device, so one batched kNN process cannot itself spread its
context batches over four GPUs.

The default complete script runs only `tabpfn-v1`; add `openml-cc18` as needed, and include
`localpfn` only when `TABZILLA_ROOT` is available. `PARALLEL_SHARDS=1` runs in the
foreground without dataset sharding. All underlying benchmark calls use `--resume`,
so rerunning the same command skips successful configurations already present in the
same output files.

To compare random sampling at dataset-relative budgets, set `RANDOM_RATIOS`. When it is present,
these values replace `K_VALUES` for the random method only; kNN continues using `K_VALUES`:

```bash
DEVICE=cuda:0 \
METHODS="random knn" \
RANDOM_RATIOS="0.01 0.05 0.10 0.20" \
K_VALUES="128 256 512 1000 localpfn" \
scripts/run_tabpfn_v1_benchmark.sh
```

The JSONL record stores both `random_ratio` and a `context_specification` such as `ratio:0.1`; the
fold result's `actual_k` contains the resulting integer number of sampled rows.

Useful overrides are `METHODS="full random knn"`, `FOLDS="0 1 ..."`, `DATASET_IDS="..."`,
`RANDOM_RATIOS`, `EVALUATION_SPLIT`, `EXPERIMENT_SEED`, `OUTPUT`, and `PYTHON_COMMAND`. For LoCalPFN, use
`EVALUATION_SPLIT=validation` when choosing `k`, then rerun only the chosen configuration with
`EVALUATION_SPLIT=test`. The reconstructed TabPFN v1 protocol has no validation partition. Every
invocation appends one JSON object per fold and uses `--resume`; completed configurations are skipped
after interruption. Errors, including an out-of-memory full-context run, remain in the JSONL result
with their exception type and message.

For four GPUs, regular full and random prediction can distribute ensemble members by setting
`DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"` and using multiple estimators. TabPFN 8.2's batched context
engine, used by the optimized kNN path, runs on only the first configured device. Use
`run_complete_benchmarks.sh` with `PARALLEL_SHARDS=4` to distribute whole datasets over four
single-GPU processes while preserving batched kNN inference within each process.

Summarize any number of result files into fold-level, dataset-level, error, and average-rank tables:

```bash
python scripts/summarize_benchmark.py \
  outputs/tabpfn-v1/results.jsonl \
  outputs/localpfn/results.jsonl \
  --output-dir outputs/summary
```

The kNN adapter uses bounded fused context batches on TabPFN 8.2 and falls back to an independent
frozen context fit only when batching is disabled, unavailable, or unnecessary for a single shared
context. Start with the smoke commands and estimate runtime before launching all datasets.

## Implementation roadmap

1. Run context-budget-matched baselines on the implemented TabPFN v1 and LoCalPFN suites.
2. Validate the implemented v2.6 local fine-tuning workflow on smoke folds and establish GPU-safe
   context/query budgets.
3. Compare frozen kNN and local fine-tuning under identical splits, `k`, retrieval features, and
   inference estimator counts.
4. Scale the benchmark to all eligible datasets and record predictive and efficiency metrics.
5. Build an oracle or leave-one-out utility analysis to study which rows actually help TabPFN.
6. Train a retrieval model from the resulting relevance signal and compare it with random and kNN retrieval.
7. Add candidate generation, reranking, diversity constraints, and retrieval caching for large datasets.

## Reproducibility principles

- Pin dataset versions by OpenML ID and version.
- Store every split, preprocessing configuration, random seed, context budget, and model version.
- Cache row indices selected by each retriever so predictions can be reproduced independently.
- Save failures and resource-limit outcomes instead of dropping them from aggregate results.
- Keep retrieval randomness separate from model and split randomness.
- Report both equal-budget comparisons and the best validation-selected configuration for each method.

## Project status

The baseline implementation now supports the TabPFN v1 30-dataset protocol and the official
TabZilla folds selected by the LoCalPFN filters. Frozen full-context, random-row, and exact-kNN
baselines are implemented. LoCalPFN-style supervised task fine-tuning is also implemented for
OpenML-CC18 and stored TabZilla folds with the pinned TabPFN v2.6 checkpoint; the immediate milestone
is GPU smoke validation followed by matched frozen-kNN versus fine-tuned-kNN experiments.

## Reference

- [*Retrieval & Fine-Tuning for In-Context Tabular Models*](https://github.com/layer6ai-labs/LoCalPFN)
  (Thomas et al. 2024)
- [TabPFN official fine-tuning implementation](https://github.com/PriorLabs/TabPFN/tree/v8.2.0/src/tabpfn/finetuning)
- [*TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*](https://arxiv.org/abs/2207.01848)
  (Hollmann et al. 2022)
- [TabZilla official repository](https://github.com/naszilla/tabzilla)
