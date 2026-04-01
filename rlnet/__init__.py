"""RLNet baseline package for deepfake video detection."""

from .data import FrameSequenceDataset, discover_records, split_records
from .model import RLNet

__all__ = ["FrameSequenceDataset", "discover_records", "split_records", "RLNet"]
