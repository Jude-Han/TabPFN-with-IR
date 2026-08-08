"""Deterministic, stratified train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split


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


def localpfn_split_indices(
    y: np.ndarray,
    *,
    n_splits: int = 10,
    random_state: int = 0,
) -> tuple[SplitIndices, ...]:
    """Reproduce TabZilla/LoCalPFN's stratified 80/10/10 fold construction.

    Fold ``i`` is the test set, fold ``(i + 1) % n_splits`` is validation,
    and all remaining folds are training data. This mirrors TabZilla's
    ``split_dataset`` implementation used by LoCalPFN.
    """

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}.")
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder = np.zeros((y.shape[0], 1), dtype=np.uint8)
    test_folds = [np.sort(test) for _, test in splitter.split(placeholder, y)]
    all_indices = np.arange(y.shape[0])
    splits = []
    for fold, test in enumerate(test_folds):
        validation = test_folds[(fold + 1) % n_splits]
        holdout = np.concatenate([test, validation])
        train = np.setdiff1d(all_indices, holdout, assume_unique=True)
        splits.append(
            SplitIndices(
                train=train,
                validation=validation,
                test=test,
            )
        )
    return tuple(splits)
