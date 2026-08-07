"""Deterministic, stratified train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Row indices for one experimental split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def stratified_train_validation_test_split(
    y: np.ndarray,
    *,
    validation_size: float = 0.10,
    test_size: float = 0.10,
    random_state: int = 42,
) -> SplitIndices:
    """Create a disjoint split while preserving class proportions where possible."""

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    indices = np.arange(y.shape[0])
    holdout_size = validation_size + test_size
    train, holdout = train_test_split(
        indices,
        test_size=holdout_size,
        random_state=random_state,
        stratify=y,
    )
    relative_test_size = test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y[holdout],
    )
    return SplitIndices(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
    )


def tabpfn_v1_split_indices(
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[SplitIndices, ...]:
    """Create the five stratified 50/50 splits described by the TabPFN v1 paper.

    The paper does not publish its exact split seeds. This function implements a
    deterministic protocol-compatible reconstruction and leaves validation empty.
    """

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        train_size=0.5,
        test_size=0.5,
        random_state=random_state,
    )
    placeholder = np.zeros((y.shape[0], 1), dtype=np.uint8)
    return tuple(
        SplitIndices(
            train=np.sort(train),
            validation=np.empty(0, dtype=np.int64),
            test=np.sort(test),
        )
        for train, test in splitter.split(placeholder, y)
    )
