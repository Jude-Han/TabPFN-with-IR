"""Validated construction of keyword arguments for TabPFN classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


MULTI_GPU_FIT_MODES = frozenset({"fit_preprocessors", "low_memory"})


def build_tabpfn_classifier_kwargs(
    *,
    device: str,
    devices: Sequence[str] | None = None,
    ignore_pretraining_limits: bool = False,
    fit_mode: str = "fit_preprocessors",
    n_estimators: int | None = None,
) -> dict[str, Any]:
    """Build a version-forward TabPFN configuration for one or many GPUs."""

    selected_devices = tuple(devices or ())
    if selected_devices and len(set(selected_devices)) != len(selected_devices):
        raise ValueError("--devices cannot contain duplicate device names.")
    if len(selected_devices) > 1 and fit_mode not in MULTI_GPU_FIT_MODES:
        raise ValueError(
            "Multiple GPUs require --fit-mode fit_preprocessors or low_memory."
        )
    if n_estimators is not None and n_estimators <= 0:
        raise ValueError("--n-estimators must be positive.")

    kwargs: dict[str, Any] = {
        "device": list(selected_devices) if selected_devices else device,
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "fit_mode": fit_mode,
    }
    if n_estimators is not None:
        kwargs["n_estimators"] = n_estimators
    return kwargs
