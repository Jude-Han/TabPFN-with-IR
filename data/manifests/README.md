# Dataset manifests

Dataset manifests are the source of truth for benchmark membership. Each entry should contain an
OpenML dataset ID, version, target column, source collection, and inclusion status. Do not identify a
dataset by name alone because names can be duplicated or point to different OpenML versions.

Split indices produced for an experiment should be stored separately from the manifest and keyed by
dataset ID, version, fold, and seed.

`tabpfn_v1_30.json` is the fixed Table 7 list. `openml_cc18.json` fixes the 72 tasks in OpenML suite
99 together with their targets and row counts; the runner applies TabZilla/LoCalPFN's deterministic
10-fold 80/10/10 construction. LoCalPFN membership is intentionally not duplicated as a static
manifest: `scripts/run_benchmark.py` applies the public LoCalPFN filters to the official TabZilla
metadata and consumes the stored TabZilla folds.
