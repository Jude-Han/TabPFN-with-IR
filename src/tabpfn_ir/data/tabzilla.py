"""Load the preprocessed TabZilla datasets and splits used by LoCalPFN."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOCALPFN_EXCLUDED_NAN_DATASETS = frozenset(
    {
        "openml__cjs__14967",
        "openml__higgs__146606",
        "openml__jm1__3904",
        "openml__sick__3021",
    }
)


@dataclass(frozen=True)
class TabZillaSplit:
    """One stored TabZilla 80/10/10 split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class TabZillaDataset:
    """A preprocessed TabZilla dataset with its ten stored splits."""

    directory_name: str
    name: str
    task_id: int | None
    X: pd.DataFrame
    y: np.ndarray
    categorical_columns: tuple[str, ...]
    splits: tuple[TabZillaSplit, ...]
    metadata: dict[str, object]


def _load_gzip_numpy(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    with gzip.GzipFile(path, "rb") as stream:
        return np.load(stream, allow_pickle=allow_pickle)


def _parse_task_id(directory_name: str) -> int | None:
    try:
        return int(directory_name.rsplit("__", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def _validate_split(split: TabZillaSplit, n_rows: int, fold: int) -> None:
    parts = (split.train, split.validation, split.test)
    for values in parts:
        if values.ndim != 1:
            raise ValueError(f"TabZilla fold {fold} indices must be one-dimensional.")
        if values.size and (values.min() < 0 or values.max() >= n_rows):
            raise IndexError(f"TabZilla fold {fold} contains an out-of-range row index.")
    pairs = ((parts[0], parts[1]), (parts[0], parts[2]), (parts[1], parts[2]))
    if any(np.intersect1d(left, right).size for left, right in pairs):
        raise ValueError(f"TabZilla fold {fold} train/validation/test indices overlap.")


def load_tabzilla_dataset(path: str | Path) -> TabZillaDataset:
    """Load one directory created by TabZilla's preprocessing script."""

    path = Path(path)
    required = ("X.npy.gz", "y.npy.gz", "metadata.json", "split_indeces.npy.gz")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"TabZilla dataset {path} is missing: {missing}")

    X_array = _load_gzip_numpy(path / "X.npy.gz", allow_pickle=True)
    y = np.asarray(_load_gzip_numpy(path / "y.npy.gz"))
    stored_splits = _load_gzip_numpy(path / "split_indeces.npy.gz", allow_pickle=True)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    if X_array.ndim != 2 or y.ndim != 1 or X_array.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid TabZilla array shapes: X={X_array.shape}, y={y.shape}.")
    try:
        numeric_X = X_array.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TabZilla dataset {path.name} is not numerically encoded.") from exc
    if not np.isfinite(numeric_X).all():
        raise ValueError(f"TabZilla dataset {path.name} contains NaN or infinite values.")

    columns = [f"feature_{index}" for index in range(X_array.shape[1])]
    categorical_indices = tuple(int(index) for index in metadata.get("cat_idx", []))
    if categorical_indices and max(categorical_indices) >= X_array.shape[1]:
        raise ValueError(f"TabZilla categorical index exceeds feature count in {path.name}.")
    categorical_columns = tuple(columns[index] for index in categorical_indices)

    splits = []
    for fold, stored in enumerate(stored_splits):
        split = TabZillaSplit(
            train=np.asarray(stored["train"], dtype=np.int64),
            validation=np.asarray(stored["val"], dtype=np.int64),
            test=np.asarray(stored["test"], dtype=np.int64),
        )
        _validate_split(split, X_array.shape[0], fold)
        splits.append(split)

    return TabZillaDataset(
        directory_name=path.name,
        name=str(metadata.get("name", path.name)),
        task_id=_parse_task_id(path.name),
        X=pd.DataFrame(X_array, columns=columns),
        y=y,
        categorical_columns=categorical_columns,
        splits=tuple(splits),
        metadata=metadata,
    )


def discover_localpfn_dataset_directories(root: str | Path) -> list[Path]:
    """Apply the public LoCalPFN paper filters to a TabZilla dataset directory."""

    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"TabZilla dataset root does not exist: {root}")

    selected = []
    for path in sorted((candidate for candidate in root.iterdir() if candidate.is_dir())):
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("target_type") == "regression":
            continue
        if int(metadata.get("num_features", 10**9)) > 100:
            continue
        if int(metadata.get("num_classes", 10**9)) > 10:
            continue
        if str(metadata.get("name", path.name)) in LOCALPFN_EXCLUDED_NAN_DATASETS:
            continue
        selected.append(path)
    return selected
