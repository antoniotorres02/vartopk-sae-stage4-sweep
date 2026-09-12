"""Frozen feature-cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FeatureSplits:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor


def load_feature_cache(path: str | Path) -> FeatureSplits:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return FeatureSplits(
        train_features=data["train_features"],
        train_labels=data["train_labels"],
        val_features=data["val_features"],
        val_labels=data["val_labels"],
        test_features=data["test_features"],
        test_labels=data["test_labels"],
    )


def make_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device, shuffle: bool):
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
