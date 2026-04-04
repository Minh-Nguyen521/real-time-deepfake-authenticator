from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual summaries from a training checkpoint and optional dataset predictions CSV."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_history(history_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"No epochs found in {history_path}")
    return epochs


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_metric_panel(
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    epochs: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = panel_box
    draw.rounded_rectangle(
        panel_box, radius=18, fill=(255, 255, 255), outline=(220, 224, 230)
    )
    draw.text((left + 20, top + 16), title, fill=(25, 31, 38), font=font)

    legend_x = left + 20
    legend_y = top + 44
    for label, _values, color in series:
        draw.rounded_rectangle(
            (legend_x, legend_y + 2, legend_x + 14, legend_y + 14),
            radius=4,
            fill=color,
        )
        draw.text((legend_x + 22, legend_y), label, fill=(60, 66, 74), font=font)
        legend_x += 140

    chart_left = left + 56
    chart_top = top + 84
    chart_right = right - 24
    chart_bottom = bottom - 42
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top

    all_values = [value for _label, values, _color in series for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        padding = 0.1 if max_value == 0 else abs(max_value) * 0.1
        min_value -= padding
        max_value += padding

    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom),
        fill=(125, 133, 143),
        width=2,
    )
    draw.line(
        (chart_left, chart_bottom, chart_left, chart_top),
        fill=(125, 133, 143),
        width=2,
    )

    tick_count = 5
    for tick in range(tick_count + 1):
        ratio = tick / tick_count
        y = chart_bottom - ratio * chart_height
        value = min_value + ratio * (max_value - min_value)
        draw.line((chart_left - 6, y, chart_left, y), fill=(125, 133, 143), width=1)
        draw.line((chart_left, y, chart_right, y), fill=(233, 236, 240), width=1)
        draw.text((left + 8, y - 8), f"{value:.3f}", fill=(90, 97, 105), font=font)

    if len(epochs) == 1:
        epoch_positions = [chart_left + chart_width / 2]
    else:
        step = chart_width / (len(epochs) - 1)
        epoch_positions = [chart_left + index * step for index in range(len(epochs))]

    for index, epoch in enumerate(epochs):
        x = epoch_positions[index]
        draw.line((x, chart_bottom, x, chart_bottom + 6), fill=(125, 133, 143), width=1)
        draw.text((x - 8, chart_bottom + 10), str(epoch), fill=(90, 97, 105), font=font)

    value_range = max_value - min_value
    for _label, values, color in series:
        points = []
        for index, value in enumerate(values):
            x = epoch_positions[index]
            normalized = (value - min_value) / value_range if value_range else 0.5
            y = chart_bottom - normalized * chart_height
            points.append((x, y))

        for index, point in enumerate(points):
            if index > 0:
                draw.line((*points[index - 1], *point), fill=color, width=3)
            draw.ellipse(
                (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
                fill=color,
                outline=(255, 255, 255),
            )


def save_training_visualization(
    history: list[dict[str, Any]],
    checkpoint_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    width = 1200
    height = 900
    padding = 36
    background = Image.new("RGB", (width, height), color=(244, 241, 235))
    draw = ImageDraw.Draw(background)
    font = ImageFont.load_default()

    draw.rounded_rectangle(
        (padding, padding, width - padding, height - padding),
        radius=24,
        fill=(250, 248, 244),
        outline=(226, 221, 214),
    )
    draw.text(
        (padding + 20, padding + 18),
        "RLNet training summary",
        fill=(28, 35, 43),
        font=font,
    )

    epochs = [int(epoch["epoch"]) for epoch in history]
    loss_series = [
        ("train loss", [float(epoch["train_loss"]) for epoch in history], (57, 106, 177)),
        ("val loss", [float(epoch["val_loss"]) for epoch in history], (214, 85, 65)),
    ]
    f1_series = [
        ("train f1", [float(epoch["train_f1"]) for epoch in history], (52, 143, 80)),
        ("val f1", [float(epoch["val_f1"]) for epoch in history], (141, 72, 179)),
    ]

    panel_gap = 28
    panel_top = padding + 64
    panel_bottom = height - padding - 90
    panel_width = (width - padding * 2 - panel_gap) // 2
    left_panel = (padding + 20, panel_top, padding + 20 + panel_width, panel_bottom)
    right_panel = (
        padding + 20 + panel_width + panel_gap,
        panel_top,
        width - padding - 20,
        panel_bottom,
    )

    draw_metric_panel(draw, left_panel, epochs, loss_series, "Loss", font)
    draw_metric_panel(draw, right_panel, epochs, f1_series, "F1 score", font)

    metric_text = (
        f"best val f1={float(checkpoint_metrics.get('f1', 0.0)):.4f}   "
        f"val acc={float(checkpoint_metrics.get('accuracy', 0.0)):.4f}   "
        f"val precision={float(checkpoint_metrics.get('precision', 0.0)):.4f}   "
        f"val recall={float(checkpoint_metrics.get('recall', 0.0)):.4f}"
    )
    draw.text((padding + 24, height - padding - 42), metric_text, fill=(60, 66, 74), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path)


def save_predictions_visualization(
    rows: list[dict[str, str]],
    output_path: Path,
) -> dict[str, float]:
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    probabilities = {"real": [], "fake": []}
    correct = 0

    for row in rows:
        label = int(row["label"])
        predicted_label = 1 if row["predicted_label"] == "fake" else 0
        probability = float(row["prob_fake"])
        probabilities["fake" if label == 1 else "real"].append(probability)
        correct += int(predicted_label == label)

        if label == 0 and predicted_label == 0:
            counts["tn"] += 1
        elif label == 0 and predicted_label == 1:
            counts["fp"] += 1
        elif label == 1 and predicted_label == 0:
            counts["fn"] += 1
        else:
            counts["tp"] += 1

    total = max(len(rows), 1)
    accuracy = correct / total

    width = 1200
    height = 760
    padding = 36
    background = Image.new("RGB", (width, height), color=(241, 244, 240))
    draw = ImageDraw.Draw(background)
    font = ImageFont.load_default()

    draw.rounded_rectangle(
        (padding, padding, width - padding, height - padding),
        radius=24,
        fill=(248, 250, 247),
        outline=(218, 224, 217),
    )
    draw.text(
        (padding + 20, padding + 18),
        "Dataset prediction summary",
        fill=(28, 35, 43),
        font=font,
    )

    matrix_box = (padding + 30, padding + 70, 520, 470)
    left, top, right, bottom = matrix_box
    draw.rounded_rectangle(matrix_box, radius=18, fill=(255, 255, 255), outline=(220, 224, 230))
    draw.text((left + 20, top + 16), "Confusion matrix", fill=(25, 31, 38), font=font)

    cell_w = 150
    cell_h = 110
    start_x = left + 80
    start_y = top + 70
    cells = [
        ("TN", counts["tn"], (215, 238, 221), start_x, start_y),
        ("FP", counts["fp"], (250, 223, 214), start_x + cell_w, start_y),
        ("FN", counts["fn"], (251, 232, 205), start_x, start_y + cell_h),
        ("TP", counts["tp"], (225, 234, 250), start_x + cell_w, start_y + cell_h),
    ]
    draw.text((start_x + 40, start_y - 24), "pred real", fill=(60, 66, 74), font=font)
    draw.text((start_x + cell_w + 32, start_y - 24), "pred fake", fill=(60, 66, 74), font=font)
    draw.text((start_x - 60, start_y + 42), "real", fill=(60, 66, 74), font=font)
    draw.text((start_x - 60, start_y + cell_h + 42), "fake", fill=(60, 66, 74), font=font)

    for label, value, color, x, y in cells:
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=14, fill=color)
        draw.text((x + 16, y + 18), label, fill=(40, 47, 55), font=font)
        draw.text((x + 16, y + 56), str(value), fill=(40, 47, 55), font=font)

    hist_box = (560, padding + 70, width - padding - 30, 470)
    left, top, right, bottom = hist_box
    draw.rounded_rectangle(hist_box, radius=18, fill=(255, 255, 255), outline=(220, 224, 230))
    draw.text((left + 20, top + 16), "Fake-probability distribution", fill=(25, 31, 38), font=font)

    chart_left = left + 50
    chart_top = top + 60
    chart_right = right - 24
    chart_bottom = bottom - 44
    chart_width = chart_right - chart_left
    chart_height = chart_bottom - chart_top
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(125, 133, 143), width=2)
    draw.line((chart_left, chart_bottom, chart_left, chart_top), fill=(125, 133, 143), width=2)

    bins = 10
    bucket_counts = {"real": [0] * bins, "fake": [0] * bins}
    for label_name, scores in probabilities.items():
        for score in scores:
            index = min(int(score * bins), bins - 1)
            bucket_counts[label_name][index] += 1

    max_bucket = max(max(bucket_counts["real"]), max(bucket_counts["fake"]), 1)
    bar_group_width = chart_width / bins
    bar_width = (bar_group_width - 10) / 2
    colors = {"real": (75, 123, 185), "fake": (209, 92, 73)}

    for bin_index in range(bins):
        base_x = chart_left + bin_index * bar_group_width + 5
        for offset, label_name in enumerate(("real", "fake")):
            count = bucket_counts[label_name][bin_index]
            bar_height = (count / max_bucket) * (chart_height - 20)
            x0 = base_x + offset * bar_width
            y0 = chart_bottom - bar_height
            draw.rounded_rectangle(
                (x0, y0, x0 + bar_width - 4, chart_bottom),
                radius=6,
                fill=colors[label_name],
            )
        draw.text((base_x, chart_bottom + 8), f"{bin_index / bins:.1f}", fill=(90, 97, 105), font=font)

    draw.rounded_rectangle((chart_right - 170, chart_top + 6, chart_right - 158, chart_top + 18), radius=4, fill=colors["real"])
    draw.text((chart_right - 150, chart_top + 4), "real", fill=(60, 66, 74), font=font)
    draw.rounded_rectangle((chart_right - 100, chart_top + 6, chart_right - 88, chart_top + 18), radius=4, fill=colors["fake"])
    draw.text((chart_right - 80, chart_top + 4), "fake", fill=(60, 66, 74), font=font)

    summary_text = (
        f"samples={len(rows)}   accuracy={accuracy:.4f}   "
        f"real mean p_fake={mean(probabilities['real']):.4f}   "
        f"fake mean p_fake={mean(probabilities['fake']):.4f}"
    )
    draw.text((padding + 24, height - padding - 42), summary_text, fill=(60, 66, 74), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path)
    return {
        "accuracy": accuracy,
        "real_mean_prob_fake": mean(probabilities["real"]),
        "fake_mean_prob_fake": mean(probabilities["fake"]),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_dir = args.checkpoint.parent
    output_dir = args.output_dir or run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.json"
    if history_path.exists():
        history = load_history(history_path)
        training_output = output_dir / "training_curves.png"
        save_training_visualization(
            history,
            checkpoint_metrics=checkpoint.get("val_metrics", {}),
            output_path=training_output,
        )
        print(f"training_visualization={training_output}")
    else:
        print(f"training_visualization_skipped=no_history_json_found_at_{history_path}")

    if args.predictions_csv is not None:
        rows = load_predictions(args.predictions_csv)
        predictions_output = output_dir / "dataset_predictions_summary.png"
        summary = save_predictions_visualization(rows, predictions_output)
        print(f"predictions_visualization={predictions_output}")
        print(f"dataset_accuracy={summary['accuracy']:.4f}")
        print(f"real_mean_prob_fake={summary['real_mean_prob_fake']:.4f}")
        print(f"fake_mean_prob_fake={summary['fake_mean_prob_fake']:.4f}")


if __name__ == "__main__":
    main()
