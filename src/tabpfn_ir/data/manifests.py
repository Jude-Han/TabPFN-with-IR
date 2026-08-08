"""Versioned benchmark manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenMLManifestEntry:
    """One fixed OpenML dataset in a paper benchmark."""

    dataset_id: int
    name: str
    task_id: int | None = None
    target: str | None = None
    version: int | None = None
    n_instances: int | None = None
    n_features: int | None = None
    n_classes: int | None = None


@dataclass(frozen=True)
class OpenMLBenchmarkManifest:
    """A paper benchmark backed by fixed OpenML dataset IDs."""

    benchmark: str
    description: str
    protocol: dict[str, object]
    datasets: tuple[OpenMLManifestEntry, ...]


def load_openml_manifest(path: str | Path) -> OpenMLBenchmarkManifest:
    """Read and validate a JSON OpenML benchmark manifest."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = tuple(OpenMLManifestEntry(**entry) for entry in payload["datasets"])
    dataset_ids = [entry.dataset_id for entry in entries]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Manifest {path} contains duplicate OpenML dataset IDs.")
    if not entries:
        raise ValueError(f"Manifest {path} contains no datasets.")
    return OpenMLBenchmarkManifest(
        benchmark=str(payload["benchmark"]),
        description=str(payload.get("description", "")),
        protocol=dict(payload.get("protocol", {})),
        datasets=entries,
    )
