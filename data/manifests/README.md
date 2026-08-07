# Dataset manifests

Dataset manifests are the source of truth for benchmark membership. Each entry should contain an
OpenML dataset ID, version, target column, source collection, and inclusion status. Do not identify a
dataset by name alone because names can be duplicated or point to different OpenML versions.

Split indices produced for an experiment should be stored separately from the manifest and keyed by
dataset ID, version, fold, and seed.

`tabpfn_v1_30.json` is the fixed Table 7 list. LoCalPFN membership is intentionally not duplicated
as a static manifest: `scripts/run_benchmark.py` applies the public LoCalPFN filters to the official
TabZilla metadata and consumes the stored TabZilla folds.
