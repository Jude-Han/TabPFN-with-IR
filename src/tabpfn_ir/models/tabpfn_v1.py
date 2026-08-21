"""Isolated adapter for the original TabPFN v1 classifier.

The historical ``tabpfn==0.1.10`` distribution and the modern TabPFN package
both install a top-level package named ``tabpfn``.  They therefore cannot be
imported safely in the same interpreter.  This module keeps v1 in a separate
target directory and talks to a persistent worker process, so the v1 model is
loaded once and can be refit with many retrieved contexts.
"""

from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np

LEGACY_TABPFN_VERSION = "0.1.10"
V1_CHECKPOINT_NAME = "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
V1_RUNTIME_ENV = "TABPFN_V1_RUNTIME"
V1_CHECKPOINT_ENV = "TABPFN_V1_CHECKPOINT"
_FRAME_HEADER = struct.Struct("!Q")


def default_v1_runtime_path() -> Path:
    """Return the repository-local legacy runtime installed by the setup script."""

    configured = os.environ.get(V1_RUNTIME_ENV)
    if configured:
        return Path(configured).expanduser()
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / ".tabpfn-v1-runtime"


def resolve_v1_checkpoint_path(path: str | Path | None = None) -> Path | None:
    """Resolve an explicit v1 checkpoint or the environment override."""

    configured = path if path is not None else os.environ.get(V1_CHECKPOINT_ENV)
    if configured is None:
        return None
    return Path(configured).expanduser().resolve()


def default_v1_checkpoint_path(runtime_path: Path) -> Path:
    """Return the checkpoint location created by the v1 setup command."""

    return runtime_path / "tabpfn" / "models_diff" / V1_CHECKPOINT_NAME


def _write_frame(stream: BinaryIO, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_FRAME_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("The TabPFN v1 worker closed its protocol stream.")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    return pickle.loads(_read_exact(stream, size))


class LegacyTabPFNClassifier:
    """Scikit-learn-shaped proxy backed by an isolated TabPFN v1 process."""

    def __init__(
        self,
        *,
        runtime_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | list[str] = "auto",
        n_estimators: int | None = None,
        ignore_pretraining_limits: bool = False,
        fit_mode: str | None = None,
    ) -> None:
        del fit_mode  # v1 owns its preprocessing and has no modern fit-mode setting.
        self.runtime_path = Path(runtime_path or default_v1_runtime_path()).expanduser()
        self.checkpoint_path = resolve_v1_checkpoint_path(checkpoint_path)
        if isinstance(device, list):
            if len(device) != 1:
                raise ValueError("TabPFN v1 supports exactly one device per worker.")
            device = device[0]
        if n_estimators is not None and n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        self.device = device
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.classes_: np.ndarray | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def _runtime_package(self) -> Path:
        return self.runtime_path / "tabpfn" / "__init__.py"

    def _start(self) -> None:
        if self._process is not None:
            return
        if not self._runtime_package().is_file():
            raise RuntimeError(
                f"TabPFN v1 runtime not found at {self.runtime_path}. Run "
                "`python scripts/setup_tabpfn_v1_runtime.py` first or set "
                f"{V1_RUNTIME_ENV}."
            )
        if self.checkpoint_path is None and not default_v1_checkpoint_path(
            self.runtime_path
        ).is_file():
            raise RuntimeError(
                "The default TabPFN v1 checkpoint is missing. Run "
                "`python scripts/setup_tabpfn_v1_runtime.py` without "
                "--skip-checkpoint, or set TABPFN_V1_CHECKPOINT."
            )
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"TabPFN v1 checkpoint not found: {self.checkpoint_path}")

        worker = Path(__file__).with_name("_tabpfn_v1_worker.py")
        environment = os.environ.copy()
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.runtime_path.resolve()), current_pythonpath) if part
        )
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=environment,
            bufsize=0,
        )
        try:
            self._request(
                {
                    "operation": "initialize",
                    "checkpoint_path": (
                        str(self.checkpoint_path) if self.checkpoint_path is not None else None
                    ),
                    "device": self.device,
                    "n_estimators": self.n_estimators,
                }
            )
        except Exception:
            self.close()
            raise

    def _request(self, message: dict[str, Any]) -> Any:
        if self._process is None:
            raise RuntimeError("The TabPFN v1 worker has not been started.")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("The TabPFN v1 worker protocol pipes are unavailable.")
        try:
            _write_frame(self._process.stdin, message)
            response = _read_frame(self._process.stdout)
        except (BrokenPipeError, EOFError) as exc:
            return_code = self._process.poll()
            raise RuntimeError(
                "The TabPFN v1 worker exited unexpectedly"
                + (f" with status {return_code}." if return_code is not None else ".")
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(
                "TabPFN v1 worker failed: "
                f"{response.get('error_type', 'Error')}: {response.get('error', '')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("result")

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self._start()
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._request(
            {
                "operation": "fit",
                "X": X,
                "y": y,
                "overwrite_warning": self.ignore_pretraining_limits,
            }
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit before predict_proba.")
        probabilities = self._request({"operation": "predict_proba", "X": np.asarray(X)})
        return np.asarray(probabilities, dtype=float)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None and process.stdout is not None:
            try:
                _write_frame(process.stdin, {"operation": "close"})
                _read_frame(process.stdout)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort during interpreter exit
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - interpreter shutdown is best effort
            pass
