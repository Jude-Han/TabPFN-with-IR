# TabPFN with Information Retrieval

This repository studies whether a plug-and-play information retrieval (IR) module can make tabular foundation models such as TabPFN effective on datasets that are larger than their practical in-context learning (ICL) budget.

Instead of passing an entire training set to TabPFN, the retriever selects a small, query-specific set of rows. These rows and their labels are then used as the ICL examples for the frozen TabPFN model. The long-term goal is to learn a retrieval policy that selects examples for their usefulness to TabPFN, rather than relying only on geometric proximity in the original feature space.

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

OpenML-CC18 is benchmark suite `99`. After installing the benchmark dependencies, list its task IDs,
dataset IDs, versions, targets, and names with:

```bash
python scripts/list_openml_cc18.py
```

CC18 task IDs describe benchmark tasks and predefined evaluation settings. The current runner uses
the associated dataset ID and creates its own deterministic 80/10/10 split; it does not yet reproduce
the official OpenML task folds.

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
src/tabpfn_ir/models/    Frozen TabPFN prediction adapter
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

Random retrieval with the same context budget over five retrieval/split seeds:

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

One kNN run with a fixed context size on the final test split:

```bash
CONTEXT_SIZE=128 scripts/run_knn.sh 31 class
```

kNN context-size selection over several values on the validation split:

```bash
K_VALUES="32 64 128 256 512 1000 localpfn" scripts/run_knn_sweep.sh 31 class
```

`localpfn` resolves to `min(10 * sqrt(n_train), 1000)`. Requested values larger than the training
fold are automatically capped at `n_train`.

Select `k` using validation results only, with one metric chosen before looking at the test set. ROC
AUC is the recommended selection metric for consistency with the main benchmark; use log loss when
probability calibration is the primary target. In a near tie, prefer the smaller `k` because it reduces
memory and inference cost. Then run the selected budget once on the test split for both kNN and
random retrieval:

```bash
CONTEXT_SIZE=128 EVALUATION_SPLIT=test scripts/run_knn.sh 31 class
CONTEXT_SIZE=128 EVALUATION_SPLIT=test scripts/run_random.sh 31 class
```

The environment variables `EXPERIMENT_SEED`, `DATASET_VERSION`, `PYTHON_COMMAND`, and
`OUTPUT_DIR` can be used to override the remaining defaults.

### Multi-GPU TabPFN inference

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
scripts.

In particular, OpenML dataset ID 6 is `letter`: it has 20,000 rows and 26 target classes. The default
80/10/10 runner therefore sends 16,000 training rows to TabPFN, which explains the reported error.
It may subsequently exceed the active checkpoint's built-in class limit as well. Multi-GPU inference
does not remove either semantic limit. Dataset 6 is intentionally absent from the TabPFN v1 30-dataset
manifest and from the LoCalPFN benchmark, whose selected tasks have at most 10 classes. Supporting it
through a many-class decomposition would be a separate experimental method and should not be mixed
into the three current retrieval baselines without being reported separately.

## Paper benchmark runners

The repository includes two paper-oriented benchmark sources:

- **TabPFN v1:** the fixed 30 OpenML dataset IDs in Table 7, stored in
  `data/manifests/tabpfn_v1_30.json`. The runner creates five deterministic stratified 50/50
  train/test splits and reports macro OVO ROC AUC. The paper did not publish its exact random split
  seeds, so these are protocol-compatible reconstructed splits, not the authors' original indices.
- **LoCalPFN:** the classification datasets discovered from a locally preprocessed TabZilla copy,
  using the filters in the public LoCalPFN code: at most 100 features and 10 classes, no regression,
  and exclusion of the four named datasets known to contain missing values. The original stored ten
  TabZilla train/validation/test folds are used without resplitting.

Here “TabPFN v1” names the v1 paper's **dataset and split benchmark**. Prediction still uses the
`TabPFNClassifier` implementation from package version 8.2.0 with the v2.6 classifier checkpoint, so
the resulting numbers are not intended to reproduce the historical v1 checkpoint exactly. Both the
library and checkpoint version are recorded alongside final results.

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
for final paper numbers. `--dataset-ids` means OpenML **dataset IDs** for TabPFN v1, but OpenML
**task IDs** parsed from TabZilla directory names for LoCalPFN.

### Complete three-baseline sweeps

The convenience scripts run full context, global random sampling, and query-specific kNN. Their
default context budget for random and kNN is the LoCalPFN heuristic; fixed budgets can be supplied
through `K_VALUES`:

```bash
DEVICE=cuda:0 K_VALUES="128 256 512 1000 localpfn" \
  scripts/run_tabpfn_v1_benchmark.sh

DEVICE=cuda:0 K_VALUES="128 256 512 1000 localpfn" \
  scripts/run_localpfn_benchmark.sh /path/to/tabzilla/TabZilla/datasets
```

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

For four GPUs, one process can use `DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"`: batched context inference
fuses compatible query contexts, while TabPFN distributes ensemble estimators across the devices.
Dataset-level sharding into four single-GPU processes remains an alternative when independent jobs
provide better cluster utilization.

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
2. Validate benchmark counts, runtime, and result aggregation on a small dataset subset.
3. Scale the benchmark to all eligible datasets and record predictive and efficiency metrics.
4. Build an oracle or leave-one-out utility analysis to study which rows actually help TabPFN.
5. Train a retrieval model from the resulting relevance signal and compare it with random and kNN retrieval.
6. Add candidate generation, reranking, diversity constraints, and retrieval caching for large datasets.

## Reproducibility principles

- Pin dataset versions by OpenML ID and version.
- Store every split, preprocessing configuration, random seed, context budget, and model version.
- Cache row indices selected by each retriever so predictions can be reproduced independently.
- Save failures and resource-limit outcomes instead of dropping them from aggregate results.
- Keep retrieval randomness separate from model and split randomness.
- Report both equal-budget comparisons and the best validation-selected configuration for each method.

## Project status

The baseline implementation now supports the TabPFN v1 30-dataset protocol and the official
TabZilla folds selected by the LoCalPFN filters. The immediate milestone is to run and validate the
full-context, random-row, and LoCalPFN-style kNN comparison without supervised fine-tuning.

## Reference

- [*Retrieval & Fine-Tuning for In-Context Tabular Models*](https://github.com/layer6ai-labs/LoCalPFN)
  (Thomas et al. 2024)
- [*TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*](https://arxiv.org/abs/2207.01848)
  (Hollmann et al. 2022)
- [TabZilla official repository](https://github.com/naszilla/tabzilla)
