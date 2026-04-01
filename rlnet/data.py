from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    label: int
    frame_dir: Path
    video_path: Path | None = None


def discover_records(dataset_root: str | Path) -> list[VideoRecord]:
    dataset_root = Path(dataset_root)
    layout = {
        "real": 0,
        "fake": 1,
    }
    records: list[VideoRecord] = []

    for split_name, label in layout.items():
        frame_root = dataset_root / split_name / "frames"
        video_root = dataset_root / split_name
        if not frame_root.exists():
            continue

        for frame_dir in sorted(path for path in frame_root.iterdir() if path.is_dir()):
            video_id = frame_dir.name
            mp4_name = f"{video_id}.mp4" if label == 0 else f"{video_id}.mp4"
            video_path = video_root / mp4_name
            records.append(
                VideoRecord(
                    video_id=video_id,
                    label=label,
                    frame_dir=frame_dir,
                    video_path=video_path if video_path.exists() else None,
                )
            )

    if not records:
        raise FileNotFoundError(f"No frame directories found under {dataset_root}")

    return records


def split_records(
    records: list[VideoRecord],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    labels = [record.label for record in records]
    train_records, val_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_records, val_records


def default_image_transform(image_size: int = 224, train: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    from torchvision import transforms

    ops: list[Callable]
    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.RandomRotation(degrees=5),
        ]
    else:
        ops = [
            transforms.Resize((image_size, image_size)),
        ]
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(ops)


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    frame_dir = Path(frame_dir)
    frame_paths = sorted(path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    return frame_paths


def sample_frame_indices(frame_count: int, sequence_length: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if frame_count == 1:
        return [0] * sequence_length

    if frame_count >= sequence_length:
        positions = torch.linspace(0, frame_count - 1, steps=sequence_length)
        return [int(round(pos.item())) for pos in positions]

    indices = list(range(frame_count))
    while len(indices) < sequence_length:
        indices.append(indices[-1])
    return indices


class FrameSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        records: list[VideoRecord],
        sequence_length: int = 16,
        image_size: int = 224,
        train: bool = False,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.records = records
        self.sequence_length = sequence_length
        self.transform = transform or default_image_transform(image_size=image_size, train=train)
        self._frame_paths_cache: dict[Path, list[Path]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        frame_paths = self._get_frame_paths(record.frame_dir)
        sampled_paths = [frame_paths[idx] for idx in sample_frame_indices(len(frame_paths), self.sequence_length)]
        frames = [self.transform(Image.open(frame_path).convert("RGB")) for frame_path in sampled_paths]
        sequence = torch.stack(frames, dim=0)
        label = torch.tensor(record.label, dtype=torch.float32)
        return sequence, label, record.video_id

    def _get_frame_paths(self, frame_dir: Path) -> list[Path]:
        cached = self._frame_paths_cache.get(frame_dir)
        if cached is not None:
            return cached

        frame_paths = list_frame_paths(frame_dir)
        self._frame_paths_cache[frame_dir] = frame_paths
        return frame_paths

    def _sample_indices(self, frame_count: int) -> list[int]:
        return sample_frame_indices(frame_count, self.sequence_length)
