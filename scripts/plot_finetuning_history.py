#!/usr/bin/env python3
"""Plot TabPFN fine-tuning history as a dependency-free SVG.

New ``training_history.jsonl`` files contain true optimizer-step losses and epoch
means. Legacy terminal logs are also accepted, but they only expose tqdm's
displayed batch loss and therefore cannot reconstruct an exact epoch mean.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Series:
    label: str
    source_kind: str
    step_loss: list[tuple[float, float]] = field(default_factory=list)
    epoch_loss: list[tuple[float, float]] = field(default_factory=list)
    validation: list[tuple[float, float]] = field(default_factory=list)
    validation_name: str = "validation ROC AUC"


def _parse_jsonl(path: Path, run_id: str | None) -> Series:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    starts = [record for record in records if record.get("event") == "run_start"]
    if not starts:
        raise ValueError(f"No run_start event found in {path}.")
    selected_run = run_id or str(starts[-1]["run_id"])
    selected = [record for record in records if str(record.get("run_id")) == selected_run]
    if not selected:
        raise ValueError(f"Run id {selected_run!r} was not found in {path}.")

    metadata = starts[-1].get("metadata", {})
    for start in starts:
        if str(start.get("run_id")) == selected_run:
            metadata = start.get("metadata", {})
            break
    label = f"{metadata.get('dataset_key', path.stem)} fold={metadata.get('fold', '?')}"
    series = Series(label=label, source_kind="jsonl")
    for record in selected:
        metrics = record.get("metrics", {})
        if record.get("event") == "step" and isinstance(metrics.get("train/loss"), (int, float)):
            series.step_loss.append((float(record["step"]), float(metrics["train/loss"])))
        if record.get("event") != "epoch":
            continue
        epoch = float(metrics.get("train/epoch", -1)) + 1.0
        if isinstance(metrics.get("train/mean_loss"), (int, float)):
            series.epoch_loss.append((epoch, float(metrics["train/mean_loss"])))
        if isinstance(metrics.get("val/roc_auc"), (int, float)):
            series.validation.append((epoch, float(metrics["val/roc_auc"])))
        elif isinstance(metrics.get("val/log_loss"), (int, float)):
            series.validation.append((epoch, float(metrics["val/log_loss"])))
            series.validation_name = "validation log loss"
    return series


def _parse_legacy_log(path: Path, dataset_id: str | None, fold: int | None) -> Series:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    completed = list(re.finditer(r"(?m)^ok\s+(\S+)\s+fold=(\d+)\b", text))
    if not completed:
        raise ValueError(f"No completed 'ok DATASET fold=N' run found in {path}.")

    candidates: list[tuple[str, int, str]] = []
    start = 0
    for match in completed:
        candidates.append((match.group(1), int(match.group(2)), text[start : match.start()]))
        start = match.end()
    matching = [
        item
        for item in candidates
        if (dataset_id is None or item[0] == dataset_id) and (fold is None or item[1] == fold)
    ]
    if not matching:
        raise ValueError(f"No completed legacy run matched dataset={dataset_id!r}, fold={fold!r}.")
    selected_dataset, selected_fold, chunk = matching[-1]
    series = Series(
        label=f"{selected_dataset} fold={selected_fold}",
        source_kind="legacy",
    )
    # tqdm renders twice around each update: once when set_postfix changes and
    # once when the progress counter advances. Keeping the last value for each
    # (epoch, percentage) removes that display duplication.
    progress_values: dict[tuple[int, int], float] = {}
    for match in re.finditer(
        r"Finetuning Epoch\s+(\d+)/(\d+):\s+(\d+)%[^\n]*?loss=([0-9.eE+\-]+)",
        chunk,
    ):
        progress_values[(int(match.group(1)), int(match.group(3)))] = float(match.group(4))
    ordered_step_values: list[float] = []
    for epoch in sorted({key[0] for key in progress_values}):
        epoch_points = [
            (percentage, value)
            for (point_epoch, percentage), value in progress_values.items()
            if point_epoch == epoch
        ]
        epoch_points.sort()
        if len(epoch_points) > 1 and epoch_points[-1][0] == 100:
            epoch_points.pop()
        ordered_step_values.extend(value for _, value in epoch_points)
    series.step_loss = [(float(index), value) for index, value in enumerate(ordered_step_values, 1)]
    epoch_values: dict[int, float] = {}
    for match in re.finditer(r"Finetuning Epoch\s+(\d+)/(\d+):[^\n]*?loss=([0-9.eE+\-]+)", chunk):
        epoch_values[int(match.group(1))] = float(match.group(3))
    series.epoch_loss = [(float(epoch), value) for epoch, value in sorted(epoch_values.items())]
    return series


def load_series(
    path: Path,
    *,
    run_id: str | None = None,
    dataset_id: str | None = None,
    fold: int | None = None,
) -> Series:
    first_nonempty = next(
        (
            line.lstrip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ),
        "",
    )
    if first_nonempty.startswith("{"):
        return _parse_jsonl(path, run_id)
    return _parse_legacy_log(path, dataset_id, fold)


def _format_tick(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.1e}"
    return f"{value:.3g}"


def _panel(
    points: list[tuple[float, float]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    color: str,
    log_y: bool = False,
) -> str:
    if not points:
        return ""
    left, top, right, bottom = x + 82, y + 46, x + width - 24, y + height - 58
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    transformed = [math.log10(max(value, 1e-12)) for value in y_values] if log_y else y_values
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(transformed), max(transformed)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.05)
        y_min, y_max = y_min - padding, y_max + padding
    y_padding = (y_max - y_min) * 0.08
    y_min, y_max = y_min - y_padding, y_max + y_padding
    sx: Callable[[float], float] = lambda value: (
        left + (value - x_min) / (x_max - x_min) * (right - left)
    )
    sy: Callable[[float], float] = lambda value: (
        bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    )

    parts = [
        (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" '
            'fill="#ffffff" stroke="#d9e0ea"/>'
        ),
        f'<text x="{x + 24}" y="{y + 29}" class="panel-title">{html.escape(title)}</text>',
    ]
    for index in range(5):
        ratio = index / 4
        tick_y = bottom - ratio * (bottom - top)
        transformed_value = y_min + ratio * (y_max - y_min)
        shown_value = 10**transformed_value if log_y else transformed_value
        parts.extend(
            [
                (
                    f'<line x1="{left}" y1="{tick_y:.2f}" x2="{right}" '
                    f'y2="{tick_y:.2f}" stroke="#e8edf3"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" '
                    f'class="tick">{_format_tick(shown_value)}</text>'
                ),
            ]
        )
    for index in range(5):
        ratio = index / 4
        tick_x = left + ratio * (right - left)
        shown_value = x_min + ratio * (x_max - x_min)
        parts.append(
            f'<text x="{tick_x:.2f}" y="{bottom + 23}" text-anchor="middle" '
            f'class="tick">{_format_tick(shown_value)}</text>'
        )
    polyline = " ".join(
        f"{sx(px):.2f},{sy(math.log10(max(py, 1e-12)) if log_y else py):.2f}" for px, py in points
    )
    parts.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#738096"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#738096"/>',
            (
                f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                'stroke-width="2.5" stroke-linejoin="round"/>'
            ),
            (
                f'<text x="{(left + right) / 2}" y="{y + height - 17}" '
                'text-anchor="middle" class="axis">step / epoch</text>'
            ),
            (
                f'<text x="{x + 18}" y="{(top + bottom) / 2}" text-anchor="middle" '
                f'class="axis" transform="rotate(-90 {x + 18} {(top + bottom) / 2})">'
                f"{html.escape(y_label)}</text>"
            ),
        ]
    )
    return "\n".join(parts)


def render_svg(series: Series, output: Path) -> None:
    panels: list[tuple[list[tuple[float, float]], str, str, str, bool]] = [
        (series.step_loss, "Optimizer-step loss", "cross-entropy (log scale)", "#d44e5c", True),
        (
            series.epoch_loss,
            (
                "Epoch mean loss"
                if series.source_kind == "jsonl"
                else "Epoch-ending tqdm loss (legacy)"
            ),
            "cross-entropy",
            "#4f6fdc",
            False,
        ),
    ]
    if series.validation:
        panels.append(
            (series.validation, series.validation_name, series.validation_name, "#2c9a73", False)
        )
    panels = [panel for panel in panels if panel[0]]
    if not panels:
        raise ValueError("The selected run contains no plottable loss values.")
    width = 1200
    panel_height = 330
    height = 115 + panel_height * len(panels) + 35
    warning = (
        "Legacy terminal data: each tqdm value is a displayed batch loss, not an epoch average."
        if series.source_kind == "legacy"
        else (
            "JSONL data: step loss, true epoch mean, and validation metrics are "
            "recorded separately."
        )
    )
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        (
            "<style>text{font-family:Inter,Arial,sans-serif;fill:#202938}"
            ".title{font-size:25px;font-weight:700}"
            ".subtitle{font-size:14px;fill:#5f6b7c}"
            ".panel-title{font-size:17px;font-weight:650}"
            ".tick{font-size:12px;fill:#68758a}"
            ".axis{font-size:12px;fill:#526076}</style>"
        ),
        f'<rect width="{width}" height="{height}" fill="#f5f7fa"/>',
        (
            '<text x="28" y="39" class="title">Fine-tuning diagnostics — '
            f"{html.escape(series.label)}</text>"
        ),
        f'<text x="28" y="66" class="subtitle">{html.escape(warning)}</text>',
    ]
    for index, (points, title, label, color, log_y) in enumerate(panels):
        body.append(
            _panel(
                points,
                x=20,
                y=90 + index * panel_height,
                width=1160,
                height=305,
                title=title,
                y_label=label,
                color=color,
                log_y=log_y,
            )
        )
    body.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", help="JSONL run id; defaults to the latest run.")
    parser.add_argument("--dataset-id", help="Dataset key when selecting from a legacy log.")
    parser.add_argument("--fold", type=int, help="Fold when selecting from a legacy log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".svg")
    series = load_series(
        args.input,
        run_id=args.run_id,
        dataset_id=args.dataset_id,
        fold=args.fold,
    )
    render_svg(series, output)
    print(
        f"wrote {output} ({len(series.step_loss)} step points, "
        f"{len(series.epoch_loss)} epoch points, {len(series.validation)} validation points)"
    )


if __name__ == "__main__":
    main()
