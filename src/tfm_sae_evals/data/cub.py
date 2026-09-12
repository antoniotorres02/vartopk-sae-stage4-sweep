"""CUB processed-PKL dataset utilities."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_processed_image_path(image_path: str | Path, raw_data_dir: str | Path) -> Path:
    image_path = Path(image_path)
    raw_data_dir = Path(raw_data_dir)
    if image_path.exists():
        return image_path

    normalized = str(image_path).replace("\\", "/")
    if "/images/" in normalized:
        candidate = raw_data_dir / "images" / normalized.split("/images/", 1)[1]
        if candidate.exists():
            return candidate

    marker = "CUB_200_2011/"
    if marker in normalized:
        candidate = raw_data_dir / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate

    parts = normalized.split("/")
    if len(parts) >= 2:
        candidate = raw_data_dir / "images" / parts[-2] / parts[-1]
        if candidate.exists():
            return candidate

    return raw_data_dir / "images" / image_path.name


def make_cub_transforms(img_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


class CUBProcessedPKLDataset(Dataset):
    """Dataset backed by official `train/val/test.pkl` CUB processed files."""

    def __init__(self, processed_data_dir: str | Path, raw_data_dir: str | Path, split: str, transform=None):
        self.processed_data_dir = Path(processed_data_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.split = split
        self.transform = transform

        split_path = self.processed_data_dir / f"{split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing processed split: {split_path}")

        with split_path.open("rb") as handle:
            self.records = pickle.load(handle)

        self.image_paths = [
            resolve_processed_image_path(record["img_path"], self.raw_data_dir)
            for record in self.records
        ]
        self.labels = np.array([int(record["class_label"]) for record in self.records], dtype=np.int64)
        self.attr_values = np.array(
            [np.asarray(record["attribute_label"], dtype=np.float32) for record in self.records],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except OSError:
            image = Image.new("RGB", (299, 299))
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.attr_values[idx], dtype=torch.float),
        )
