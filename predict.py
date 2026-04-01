from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlnet.data import FrameSequenceDataset, VideoRecord
from rlnet.model import RLNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLNet inference on a frame directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    record = VideoRecord(video_id=args.frames_dir.name, label=0, frame_dir=args.frames_dir)
    dataset = FrameSequenceDataset(
        [record],
        sequence_length=config["sequence_length"],
        image_size=config["image_size"],
        train=False,
    )
    frames, _label, video_id = dataset[0]

    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))
        probability = torch.sigmoid(logits).item()

    predicted_label = "fake" if probability >= 0.5 else "real"
    print(f"video_id={video_id}")
    print(f"prob_fake={probability:.4f}")
    print(f"predicted_label={predicted_label}")


if __name__ == "__main__":
    main()
