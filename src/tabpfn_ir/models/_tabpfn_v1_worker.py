"""Worker process for the original TabPFN v1 package.

This file is launched directly, with the legacy target directory first on
``PYTHONPATH``.  Protocol data uses the original stdout buffer while ordinary
legacy prints are redirected to stderr.
"""

from __future__ import annotations

import pickle
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, BinaryIO

_FRAME_HEADER = struct.Struct("!Q")


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
            raise EOFError
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    return pickle.loads(_read_exact(stream, size))


def _install_dependency_compatibility() -> None:
    """Backport small import/call aliases removed after the v1 release."""

    import typing

    import sklearn.utils.validation as sklearn_validation
    import torch
    import torch.nn.modules.transformer as torch_transformer

    if not hasattr(torch_transformer, "Optional"):
        torch_transformer.Optional = typing.Optional

    original_torch_load = torch.load

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        # PyTorch 2.6 changed the default to weights_only=True, while the v1
        # checkpoint is a trusted tuple containing configuration metadata.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load

    original_check_X_y = sklearn_validation.check_X_y
    original_check_array = sklearn_validation.check_array

    def compatible_check_X_y(*args: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return original_check_X_y(*args, **kwargs)

    def compatible_check_array(*args: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return original_check_array(*args, **kwargs)

    sklearn_validation.check_X_y = compatible_check_X_y
    sklearn_validation.check_array = compatible_check_array


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu:0"
    if requested == "cuda":
        return "cuda:0"
    if requested == "cpu":
        return "cpu:0"
    if requested.startswith("mps"):
        # The historical loader only distinguishes CUDA from CPU.
        return "cpu:0"
    return requested


def _build_classifier(message: dict[str, Any]) -> tuple[Any, tempfile.TemporaryDirectory | None]:
    import tabpfn
    from tabpfn import TabPFNClassifier

    checkpoint_path = message.get("checkpoint_path")
    temporary_directory = None
    if checkpoint_path is None:
        base_path = Path(tabpfn.__file__).resolve().parent
    else:
        source = Path(checkpoint_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"TabPFN v1 checkpoint not found: {source}")
        temporary_directory = tempfile.TemporaryDirectory(prefix="tabpfn-v1-checkpoint-")
        base_path = Path(temporary_directory.name)
        models_directory = base_path / "models_diff"
        models_directory.mkdir(parents=True)
        expected = models_directory / "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
        expected.symlink_to(source)

    kwargs: dict[str, Any] = {
        "device": _resolve_device(str(message.get("device", "auto"))),
        "base_path": base_path,
    }
    if message.get("n_estimators") is not None:
        kwargs["N_ensemble_configurations"] = int(message["n_estimators"])
    return TabPFNClassifier(**kwargs), temporary_directory


def main() -> None:
    protocol_input = sys.stdin.buffer
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    _install_dependency_compatibility()

    classifier = None
    checkpoint_directory = None
    while True:
        try:
            message = _read_frame(protocol_input)
        except EOFError:
            break
        operation = message.get("operation")
        try:
            if operation == "initialize":
                classifier, checkpoint_directory = _build_classifier(message)
                result = None
            elif operation == "fit":
                if classifier is None:
                    raise RuntimeError("Worker was not initialized.")
                classifier.fit(
                    message["X"],
                    message["y"],
                    overwrite_warning=bool(message.get("overwrite_warning", False)),
                )
                result = None
            elif operation == "predict_proba":
                if classifier is None:
                    raise RuntimeError("Worker was not initialized.")
                result = classifier.predict_proba(message["X"])
            elif operation == "close":
                _write_frame(protocol_output, {"ok": True, "result": None})
                break
            else:
                raise ValueError(f"Unknown worker operation: {operation!r}")
        except Exception as exc:  # noqa: BLE001 - return worker failures to the parent
            _write_frame(
                protocol_output,
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            _write_frame(protocol_output, {"ok": True, "result": result})

    del classifier
    if checkpoint_directory is not None:
        checkpoint_directory.cleanup()


if __name__ == "__main__":
    main()
