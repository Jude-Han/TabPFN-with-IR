"""Local experiment loggers for TabPFN fine-tuning.

The TabPFN package accepts a small logger protocol but defaults to a no-op logger.
These implementations keep the dependency-free JSONL audit trail enabled while
optionally mirroring the same scalars to TensorBoard.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlFinetuningLogger:
    """Append step and epoch metrics to a restart-safe JSONL history."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.summary_path = self.path.with_name("training_summary.json")
        self.metadata = dict(metadata or {})
        self.run_id = uuid.uuid4().hex
        self._stream: Any | None = None
        self._epoch_records: list[dict[str, Any]] = []
        self._config: dict[str, Any] = {}

    def _write(self, payload: dict[str, Any]) -> None:
        if self._stream is None:
            return
        record = {
            "run_id": self.run_id,
            "timestamp": _timestamp(),
            **payload,
        }
        self._stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._stream.flush()

    def setup(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._write(
            {
                "event": "run_start",
                "config": config,
                "metadata": self.metadata,
            }
        )

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        self._write({"event": "step", "step": int(step), "metrics": metrics})

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        record = {"event": "epoch", "step": int(step), "metrics": dict(metrics)}
        self._epoch_records.append(record)
        self._write(record)

    def _summary(self) -> dict[str, Any]:
        epoch_rows = [row["metrics"] for row in self._epoch_records]
        initial_rows = [row for row in epoch_rows if float(row.get("train/epoch", 0)) < 0]
        trained_rows = [row for row in epoch_rows if float(row.get("train/epoch", -1)) >= 0]
        eval_metric = str(self._config.get("eval_metric", "roc_auc"))

        def primary_value(row: dict[str, Any]) -> float | None:
            explicit = row.get("val/primary_metric")
            if isinstance(explicit, (int, float)):
                return float(explicit)
            raw = row.get(f"val/{eval_metric}")
            if not isinstance(raw, (int, float)):
                return None
            # TabPFN minimizes log loss by maximizing its negative value. Its
            # epoch payload also contains the human-readable positive log loss.
            return -float(raw) if eval_metric == "log_loss" else float(raw)

        primary_rows = [(row, primary_value(row)) for row in epoch_rows]
        primary_rows = [(row, value) for row, value in primary_rows if value is not None]
        best_primary_pair = max(primary_rows, key=lambda pair: pair[1]) if primary_rows else None
        train_losses = [
            float(row["train/mean_loss"])
            for row in trained_rows
            if isinstance(row.get("train/mean_loss"), (int, float))
        ]
        initial_primary = primary_value(initial_rows[-1]) if initial_rows else None
        best_primary = best_primary_pair[1] if best_primary_pair is not None else None
        best_epoch = (
            int(float(best_primary_pair[0]["train/epoch"]))
            if best_primary_pair is not None
            else None
        )
        min_delta = float(self._config.get("min_delta", 0.0))
        return {
            "run_id": self.run_id,
            "history_path": str(self.path),
            "metadata": self.metadata,
            "epochs_recorded": len(trained_rows),
            "initial_primary_metric": initial_primary,
            "best_primary_metric": best_primary,
            "best_epoch_zero_based": best_epoch,
            "improved_over_initial": (
                best_primary is not None
                and initial_primary is not None
                and best_primary > initial_primary + min_delta
            ),
            "minimum_epoch_mean_train_loss": min(train_losses) if train_losses else None,
            "final_epoch_mean_train_loss": train_losses[-1] if train_losses else None,
        }

    def finish(self) -> None:
        summary = self._summary()
        self._write({"event": "run_end", "summary": summary})
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class TensorBoardFinetuningLogger:
    """Write fine-tuning scalars to TensorBoard when its optional package exists."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self._writer: Any | None = None

    def setup(self, config: dict[str, Any]) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorBoard logging requires the optional 'tensorboard' package. "
                "Install it with `pip install -e '.[benchmark,tracking]'`."
            ) from exc
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self._writer.add_text(
            "configuration/json",
            json.dumps(config, indent=2, sort_keys=True, default=str),
            global_step=0,
        )

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(name, float(value), global_step=int(step))
        self._writer.flush()

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        self._log(metrics, step)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        self._log(metrics, step)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class CompositeFinetuningLogger:
    """Fan out metrics while keeping a failed optional backend non-fatal."""

    def __init__(self, loggers: list[Any]) -> None:
        self.loggers = list(loggers)
        self._active: list[Any] = []

    def setup(self, config: dict[str, Any]) -> None:
        for child in self.loggers:
            try:
                child.setup(config)
            except (ModuleNotFoundError, OSError):
                logger.warning(
                    "Fine-tuning logger %s could not be initialized.",
                    type(child).__name__,
                    exc_info=True,
                )
            else:
                self._active.append(child)

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        for child in self._active:
            child.log_step(metrics, step)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        for child in self._active:
            child.log_epoch(metrics, step)

    def finish(self) -> None:
        for child in self._active:
            child.finish()
        self._active = []
