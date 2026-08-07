"""Dataset loading, splitting, and fold-safe preprocessing."""

from tabpfn_ir.data.openml import OpenMLDataset, load_openml_dataset
from tabpfn_ir.data.manifests import (
    OpenMLBenchmarkManifest,
    OpenMLManifestEntry,
    load_openml_manifest,
)
from tabpfn_ir.data.preprocessing import ProcessedViews, TabularPreprocessor
from tabpfn_ir.data.splitting import (
    SplitIndices,
    stratified_train_validation_test_split,
    tabpfn_v1_split_indices,
)
from tabpfn_ir.data.tabzilla import (
    TabZillaDataset,
    TabZillaSplit,
    discover_localpfn_dataset_directories,
    load_tabzilla_dataset,
)

__all__ = [
    "OpenMLDataset",
    "OpenMLBenchmarkManifest",
    "OpenMLManifestEntry",
    "ProcessedViews",
    "SplitIndices",
    "TabularPreprocessor",
    "TabZillaDataset",
    "TabZillaSplit",
    "discover_localpfn_dataset_directories",
    "load_openml_manifest",
    "load_openml_dataset",
    "load_tabzilla_dataset",
    "stratified_train_validation_test_split",
    "tabpfn_v1_split_indices",
]
