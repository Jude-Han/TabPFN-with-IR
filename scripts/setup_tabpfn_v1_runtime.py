#!/usr/bin/env python3
"""Install the isolated original TabPFN runtime and its default checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = REPOSITORY_ROOT / ".tabpfn-v1-runtime"
LEGACY_PACKAGE = "tabpfn==0.1.10"
CHECKPOINT_NAME = "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
# The original GitHub URL embedded in v0.1.10 was removed. This preserved
# TabPFN-owned Hugging Face Space snapshot contains the same 103 MB checkpoint.
CHECKPOINT_URL = (
    "https://huggingface.co/spaces/TabPFN/TabPFNPrediction/resolve/main/"
    "TabPFN/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt?download=true"
)
CHECKPOINT_SHA256 = "3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-path", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Install only the legacy package; provide a local checkpoint before a v1 run.",
    )
    parser.add_argument("--force-checkpoint-download", action="store_true")
    return parser.parse_args()


def install_runtime(runtime_path: Path) -> None:
    runtime_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--target",
            str(runtime_path),
            LEGACY_PACKAGE,
        ],
        check=True,
    )


def download_checkpoint(runtime_path: Path, *, force: bool) -> Path:
    destination = runtime_path / "tabpfn" / "models_diff" / CHECKPOINT_NAME
    if destination.is_file() and not force:
        print(f"Checkpoint already present: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(
        CHECKPOINT_URL,
        headers={"User-Agent": "tabpfn-ir-v1-runtime-setup"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
            expected = response.headers.get("Content-Length")
            written = 0
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if expected:
                    print(
                        f"Downloading v1 checkpoint: {written / 1024**2:.1f}/"
                        f"{int(expected) / 1024**2:.1f} MiB",
                        end="\r",
                        flush=True,
                    )
        if expected and written != int(expected):
            raise OSError(
                f"Incomplete checkpoint download: expected {expected} bytes, got {written}."
            )
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise OSError(
                "Downloaded checkpoint checksum mismatch: "
                f"expected {CHECKPOINT_SHA256}, got {digest.hexdigest()}."
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"\nCheckpoint installed: {destination}")
    return destination


def main() -> None:
    args = parse_args()
    runtime_path = args.runtime_path.expanduser().resolve()
    install_runtime(runtime_path)
    checkpoint = None
    if not args.skip_checkpoint:
        checkpoint = download_checkpoint(
            runtime_path,
            force=args.force_checkpoint_download,
        )
    print(f"TabPFN v1 runtime: {runtime_path}")
    if checkpoint is None:
        print("Checkpoint download skipped.")


if __name__ == "__main__":
    main()
