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

The primary classification metrics are ROC AUC (binary or macro one-vs-rest for multiclass tasks) and log loss. Accuracy and balanced accuracy will be included as secondary metrics. If a dataset does not support a metric in a particular fold, that dataset-fold result will be marked missing with a documented reason.

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
src/data/                Downloading, splitting, and preprocessing
src/retrieval/           Full, random, kNN, and learned retrievers
src/models/              Frozen TabPFN prediction adapter
src/evaluation/          Metrics, aggregation, and statistical tests
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

## Implementation roadmap

1. Create a dataset manifest for the union of OpenML-CC18 and the LoCalPFN benchmark.
2. Implement deterministic dataset loading, split handling, and fold-safe preprocessing.
3. Add the full-context, random, and per-query kNN retrievers behind the common interface.
4. Add a frozen TabPFN adapter that supports query-specific contexts and batched inference.
5. Run context-budget-matched baseline experiments and validate them on a small dataset subset.
6. Scale the benchmark to all eligible datasets and record predictive and efficiency metrics.
7. Build an oracle or leave-one-out utility analysis to study which rows actually help TabPFN.
8. Train a retrieval model from the resulting relevance signal and compare it with random and kNN retrieval.
9. Add candidate generation, reranking, diversity constraints, and retrieval caching for large datasets.

## Reproducibility principles

- Pin dataset versions by OpenML ID and version.
- Store every split, preprocessing configuration, random seed, context budget, and model version.
- Cache row indices selected by each retriever so predictions can be reproduced independently.
- Save failures and resource-limit outcomes instead of dropping them from aggregate results.
- Keep retrieval randomness separate from model and split randomness.
- Report both equal-budget comparisons and the best validation-selected configuration for each method.

## Project status

This repository is in the research-design and baseline-implementation stage. The immediate milestone is a reproducible comparison of full-context TabPFN, random row retrieval, and LoCalPFN-style kNN row retrieval without supervised fine-tuning.

## Reference

- *Retrieval & Fine-Tuning for In-Context Tabular Models. (Thomas et al. 2024)*
