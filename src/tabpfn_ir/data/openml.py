"""OpenML dataset loading with an optional dependency boundary."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpenMLDataset:
    """A loaded classification dataset and its OpenML metadata."""

    dataset_id: int
    name: str
    version: int | None
    target_name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical_columns: tuple[str, ...]


def load_openml_dataset(
    dataset_id: int,
    *,
    target: str | None = None,
    version: int | None = None,
) -> OpenMLDataset:
    """Download one OpenML dataset as a pandas-backed classification dataset.

    The import is intentionally lazy so unit tests and retrieval-only workflows do
    not require the OpenML client.
    """

    try:
        import openml
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "OpenML support is optional. Install the project with "
            "`pip install -e '.[benchmark]'`."
        ) from exc

    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    downloaded_version = getattr(dataset, "version", None)
    if version is not None and downloaded_version != version:
        raise ValueError(
            f"OpenML dataset ID {dataset_id} resolved to version {downloaded_version}, "
            f"not requested version {version}."
        )
    target_name = target or dataset.default_target_attribute
    if not target_name:
        raise ValueError(
            f"OpenML dataset {dataset_id} has no default target; pass target explicitly."
        )

    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=target_name,
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    categorical_columns = tuple(
        name for name, is_categorical in zip(feature_names, categorical_indicator, strict=True)
        if is_categorical
    )

    return OpenMLDataset(
        dataset_id=dataset_id,
        name=dataset.name,
        version=downloaded_version,
        target_name=target_name,
        X=X.reset_index(drop=True),
        y=np.asarray(y),
        categorical_columns=categorical_columns,
    )
