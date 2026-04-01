from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlnet.data import FrameSequenceDataset, VideoRecord, discover_records
from rlnet.metrics import classification_metrics
from rlnet.model import RLNet

LABEL_NAMES = {
    0: "real",
    1: "fake",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint over the whole dataset and export predictions to CSV."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("UADFV"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("artifacts/dataset_predictions.csv")
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
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


def build_loader(
    records: list[VideoRecord], config: dict, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = FrameSequenceDataset(
        records,
        sequence_length=int(config["sequence_length"]),
        image_size=int(config["image_size"]),
        train=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "label",
        "label_name",
        "predicted_label",
        "prob_fake",
        "correct",
        "frame_dir",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    records = discover_records(args.dataset_root)
    loader = build_loader(records, config, args.batch_size, args.num_workers)

    record_by_id = {record.video_id: record for record in records}
    csv_rows: list[dict[str, str | int | float]] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    progress = tqdm(loader, desc="dataset inference", leave=False)
    for frames, labels, video_ids in progress:
        with torch.no_grad():
            logits = model(frames.to(device))
        probabilities = torch.sigmoid(logits).cpu()
        predictions = (probabilities >= 0.5).to(torch.int64)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        for index, video_id in enumerate(video_ids):
            record = record_by_id[video_id]
            label = int(labels[index].item())
            prediction = int(predictions[index].item())
            probability = float(probabilities[index].item())
            csv_rows.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "predicted_label": LABEL_NAMES[prediction],
                    "prob_fake": round(probability, 6),
                    "correct": int(prediction == label),
                    "frame_dir": str(record.frame_dir),
                    "video_path": ""
                    if record.video_path is None
                    else str(record.video_path),
                }
            )

    write_csv(args.output_csv, csv_rows)

    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    metrics = classification_metrics(logits_tensor, labels_tensor)

    print(f"samples={len(csv_rows)}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
