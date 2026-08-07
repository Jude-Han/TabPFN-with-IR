"""Dataset loading, splitting, and fold-safe preprocessing."""

from tabpfn_ir.data.openml import OpenMLDataset, load_openml_dataset
from tabpfn_ir.data.preprocessing import ProcessedViews, TabularPreprocessor
from tabpfn_ir.data.splitting import SplitIndices, stratified_train_validation_test_split

__all__ = [
    "OpenMLDataset",
    "ProcessedViews",
    "SplitIndices",
    "TabularPreprocessor",
    "load_openml_dataset",
    "stratified_train_validation_test_split",
]
