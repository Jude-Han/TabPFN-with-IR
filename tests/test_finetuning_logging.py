from __future__ import annotations

import json

from tabpfn_ir.finetuning_logging import JsonlFinetuningLogger


def test_jsonl_logger_records_initial_step_and_epoch_metrics(tmp_path) -> None:
    path = tmp_path / "training_history.jsonl"
    logger = JsonlFinetuningLogger(path, metadata={"dataset_key": "example", "fold": 0})

    logger.setup({"eval_metric": "roc_auc", "learning_rate": 1e-5})
    logger.log_epoch(
        {"train/epoch": -1.0, "val/primary_metric": 0.80, "val/roc_auc": 0.80},
        step=0,
    )
    logger.log_step(
        {"train/loss": 0.4, "train/lr": 1e-5, "train/epoch": 0.0},
        step=1,
    )
    logger.log_epoch(
        {"train/epoch": 0.0, "train/mean_loss": 0.35, "val/roc_auc": 0.82},
        step=30,
    )
    logger.finish()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "run_start",
        "epoch",
        "step",
        "epoch",
        "run_end",
    ]
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["initial_primary_metric"] == 0.80
    assert summary["best_primary_metric"] == 0.82
    assert summary["best_epoch_zero_based"] == 0
    assert summary["improved_over_initial"] is True
    assert summary["minimum_epoch_mean_train_loss"] == 0.35


def test_jsonl_logger_treats_lower_log_loss_as_better(tmp_path) -> None:
    logger = JsonlFinetuningLogger(tmp_path / "training_history.jsonl")
    logger.setup({"eval_metric": "log_loss"})
    logger.log_epoch(
        {"train/epoch": -1.0, "val/primary_metric": -0.50, "val/log_loss": 0.50},
        step=0,
    )
    logger.log_epoch(
        {"train/epoch": 0.0, "train/mean_loss": 0.4, "val/log_loss": 0.45},
        step=30,
    )
    logger.finish()

    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["initial_primary_metric"] == -0.50
    assert summary["best_primary_metric"] == -0.45
    assert summary["improved_over_initial"] is True
