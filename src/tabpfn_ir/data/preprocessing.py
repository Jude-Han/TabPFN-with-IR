"""Separate, aligned feature views for prediction and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class ProcessedViews:
    """Aligned model and retrieval representations for the same rows."""

    model: np.ndarray
    retrieval: np.ndarray


class TabularPreprocessor:
    """Fit fold-safe transformations for TabPFN and the row retriever.

    The model view uses ordinal categorical encoding. The retrieval view uses
    standardized numerical features and one-hot categorical features. Both views
    are fit on the training fold only and preserve row order.
    """

    def __init__(self, categorical_columns: Iterable[str] | None = None) -> None:
        self._categorical_columns = (
            tuple(categorical_columns) if categorical_columns is not None else None
        )
        self._model_transformer: ColumnTransformer | None = None
        self._retrieval_transformer: ColumnTransformer | None = None

    def _resolve_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        if self._categorical_columns is None:
            categorical = X.select_dtypes(
                include=["category", "object", "bool", "string"]
            ).columns.tolist()
        else:
            missing = set(self._categorical_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Categorical columns not found: {sorted(missing)}")
            categorical = list(self._categorical_columns)
        numerical = [column for column in X.columns if column not in categorical]
        return numerical, categorical

    def fit(self, X_train: pd.DataFrame) -> "TabularPreprocessor":
        """Fit both representations using training rows only."""

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame so column types remain explicit.")
        numerical, categorical = self._resolve_columns(X_train)
        if not numerical and not categorical:
            raise ValueError("X_train must contain at least one feature column.")

        model_numerical = Pipeline(
            [("imputer", SimpleImputer(strategy="median"))]
        )
        model_categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        retrieval_numerical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        retrieval_categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        self._model_transformer = ColumnTransformer(
            [
                ("numerical", model_numerical, numerical),
                ("categorical", model_categorical, categorical),
            ],
            sparse_threshold=0,
        )
        self._retrieval_transformer = ColumnTransformer(
            [
                ("numerical", retrieval_numerical, numerical),
                ("categorical", retrieval_categorical, categorical),
            ],
            sparse_threshold=0,
        )
        self._model_transformer.fit(X_train)
        self._retrieval_transformer.fit(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> ProcessedViews:
        """Transform rows into aligned model and retrieval arrays."""

        if self._model_transformer is None or self._retrieval_transformer is None:
            raise RuntimeError("Call fit before transform.")
        model = np.asarray(self._model_transformer.transform(X), dtype=np.float32)
        retrieval = np.asarray(self._retrieval_transformer.transform(X), dtype=np.float32)
        return ProcessedViews(model=model, retrieval=retrieval)

    def fit_transform(self, X_train: pd.DataFrame) -> ProcessedViews:
        """Fit on and transform the training fold."""

        return self.fit(X_train).transform(X_train)
