from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from rlnet.data import default_image_transform, list_frame_paths, sample_frame_indices
from rlnet.model import RLNet
from rlnet.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RLNet predictions for a frame directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RLNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RLNet(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", True),
        pretrained_backbone=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def score_sequence(model: RLNet, sequence: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad():
        logits = model(sequence.unsqueeze(0).to(device))
        return torch.sigmoid(logits).item()


def prefix_probabilities(model: RLNet, sequence: torch.Tensor, device: torch.device) -> list[float]:
    probabilities: list[float] = []
    sequence_length = sequence.size(0)
    for end_idx in range(sequence_length):
        prefix = sequence[: end_idx + 1]
        if prefix.size(0) < sequence_length:
            pad = prefix[-1:].repeat(sequence_length - prefix.size(0), 1, 1, 1)
            prefix = torch.cat([prefix, pad], dim=0)
        probabilities.append(score_sequence(model, prefix, device))
    return probabilities


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def draw_contact_sheet(
    sampled_paths: list[Path],
    prefix_scores: list[float],
    final_score: float,
    output_path: Path,
    tile_size: int,
    columns: int,
) -> None:
    font = ImageFont.load_default()
    banner_height = 38
    footer_height = 18
    gutter = 12
    rows = math.ceil(len(sampled_paths) / columns)
    width = columns * tile_size + (columns + 1) * gutter
    height = rows * (tile_size + banner_height + footer_height) + (rows + 1) * gutter
    sheet = Image.new("RGB", (width, height), color=(247, 245, 239))
    draw = ImageDraw.Draw(sheet)

    for idx, frame_path in enumerate(sampled_paths):
        row = idx // columns
        col = idx % columns
        x = gutter + col * (tile_size + gutter)
        y = gutter + row * (tile_size + banner_height + footer_height + gutter)

        frame = open_rgb(frame_path).resize((tile_size, tile_size))
        sheet.paste(frame, (x, y + banner_height))

        header_fill = score_to_color(prefix_scores[idx])
        draw.rounded_rectangle((x, y, x + tile_size, y + banner_height - 4), radius=8, fill=header_fill)
        draw.text((x + 8, y + 6), f"t={idx + 1:02d}", fill="white", font=font)
        draw.text((x + 8, y + 20), f"prefix p_fake={prefix_scores[idx]:.3f}", fill="white", font=font)

        footer_y = y + banner_height + tile_size
        draw.rounded_rectangle(
            (x, footer_y, x + tile_size, footer_y + footer_height),
            radius=6,
            fill=(33, 37, 41),
        )
        draw.text((x + 8, footer_y + 4), f"final p_fake={final_score:.3f}", fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def draw_probability_plot(prefix_scores: list[float], output_path: Path) -> None:
    width = 960
    height = 360
    margin_left = 64
    margin_right = 24
    margin_top = 24
    margin_bottom = 48
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x0 = margin_left
    y0 = margin_top + plot_height
    x1 = margin_left + plot_width
    y1 = margin_top

    draw.line((x0, y0, x1, y0), fill=(80, 80, 80), width=2)
    draw.line((x0, y0, x0, y1), fill=(80, 80, 80), width=2)

    for tick in range(6):
        value = tick / 5
        y = y0 - value * plot_height
        draw.line((x0 - 6, y, x0, y), fill=(80, 80, 80), width=1)
        draw.text((8, y - 7), f"{value:.1f}", fill=(40, 40, 40), font=font)
        if tick < 5:
            draw.line((x0, y, x1, y), fill=(226, 229, 233), width=1)

    if len(prefix_scores) == 1:
        points = [(x0, y0 - prefix_scores[0] * plot_height)]
    else:
        step = plot_width / (len(prefix_scores) - 1)
        points = [
            (x0 + idx * step, y0 - score * plot_height)
            for idx, score in enumerate(prefix_scores)
        ]

    for idx, point in enumerate(points):
        if idx > 0:
            draw.line((*points[idx - 1], *point), fill=(202, 72, 45), width=3)
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(202, 72, 45))
        draw.text((point[0] - 6, y0 + 8), str(idx + 1), fill=(40, 40, 40), font=font)

    draw.text((margin_left, 4), "RLNet prefix fake probability", fill=(20, 20, 20), font=font)
    draw.text((width - 170, height - 20), "sampled frame index", fill=(40, 40, 40), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def score_to_color(score: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(score, 1.0))
    start = (46, 125, 50)
    end = (183, 28, 28)
    return tuple(int(start[idx] + (end[idx] - start[idx]) * clamped) for idx in range(3))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)

    frame_paths = list_frame_paths(args.frames_dir)
    sequence_length = int(config["sequence_length"])
    image_size = int(config["image_size"])
    sampled_indices = sample_frame_indices(len(frame_paths), sequence_length)
    sampled_paths = [frame_paths[idx] for idx in sampled_indices]

    transform = default_image_transform(image_size=image_size, train=False)
    frames = [transform(open_rgb(frame_path)) for frame_path in sampled_paths]
    sequence = torch.stack(frames, dim=0)

    final_score = score_sequence(model, sequence, device)
    prefix_scores = prefix_probabilities(model, sequence, device)
    predicted_label = "fake" if final_score >= 0.5 else "real"

    output_dir = args.output_dir or Path("artifacts") / "visualizations" / args.frames_dir.name
    contact_sheet_path = output_dir / "contact_sheet.png"
    plot_path = output_dir / "temporal_probabilities.png"
    summary_path = output_dir / "summary.json"

    draw_contact_sheet(
        sampled_paths=sampled_paths,
        prefix_scores=prefix_scores,
        final_score=final_score,
        output_path=contact_sheet_path,
        tile_size=args.tile_size,
        columns=max(args.columns, 1),
    )
    draw_probability_plot(prefix_scores, plot_path)
    save_json(
        summary_path,
        {
            "frames_dir": str(args.frames_dir),
            "checkpoint": str(args.checkpoint),
            "predicted_label": predicted_label,
            "prob_fake": final_score,
            "sampled_frames": [str(path) for path in sampled_paths],
            "prefix_prob_fake": prefix_scores,
        },
    )

    print(f"frames_dir={args.frames_dir}")
    print(f"predicted_label={predicted_label}")
    print(f"prob_fake={final_score:.4f}")
    print(f"contact_sheet={contact_sheet_path}")
    print(f"temporal_plot={plot_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
