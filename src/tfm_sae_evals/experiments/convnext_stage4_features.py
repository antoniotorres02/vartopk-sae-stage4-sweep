"""ConvNeXt Stage4 feature extraction helpers for SAE experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import convnext_tiny

from tfm_sae_evals.config import ExternalPaths
from tfm_sae_evals.data.cub import CUBProcessedPKLDataset, make_cub_transforms


STAGE_NAME = "stage4"
STAGE_FEATURE_IDX = 7
STAGE_DIM = 768
STAGE_HW = 7
NUM_CLASSES = 200


@dataclass(frozen=True)
class StageData:
    train_maps: torch.Tensor
    train_labels: torch.Tensor
    val_maps: torch.Tensor
    val_labels: torch.Tensor
    test_maps: torch.Tensor
    test_labels: torch.Tensor
    suffix_max_abs_diff: float


class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        self.feature_dim = int(self.backbone.classifier[2].in_features)
        self.num_classes = int(num_classes)
        self.backbone.classifier[2] = nn.Linear(self.feature_dim, self.num_classes)

    def forward_features(self, x):
        conv_maps = self.backbone.features(x)
        pooled = self.backbone.avgpool(conv_maps)
        normalized = self.backbone.classifier[0](pooled)
        return self.backbone.classifier[1](normalized)

    def classify_from_features(self, features):
        return self.backbone.classifier[2](features)

    def forward(self, x):
        features = self.forward_features(x)
        return {"features": features, "logits": self.classify_from_features(features)}


def load_convnext_checkpoint(path: Path, device: torch.device) -> ConvNeXtTinyClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Missing ConvNeXt checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(checkpoint.get("config", {}).get("num_classes", NUM_CLASSES))
    model = ConvNeXtTinyClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_eval_loader(
    paths: ExternalPaths,
    split: str,
    subset_size: int | None,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    _train_transform, eval_transform = make_cub_transforms(img_size=224)
    dataset = CUBProcessedPKLDataset(paths.cub_processed_dir, paths.cub_raw_dir, split, transform=eval_transform)
    if subset_size is not None:
        dataset = Subset(dataset, range(min(int(subset_size), len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


@torch.inference_mode()
def forward_until_stage(cnn_model: ConvNeXtTinyClassifier, images: torch.Tensor) -> torch.Tensor:
    x = images
    for idx, module in enumerate(cnn_model.backbone.features):
        x = module(x)
        if idx == STAGE_FEATURE_IDX:
            return x
    raise ValueError(f"Invalid ConvNeXt feature index: {STAGE_FEATURE_IDX}")


@torch.inference_mode()
def classify_from_stage_map(cnn_model: ConvNeXtTinyClassifier, stage_map: torch.Tensor) -> torch.Tensor:
    x = stage_map
    for idx in range(STAGE_FEATURE_IDX + 1, len(cnn_model.backbone.features)):
        x = cnn_model.backbone.features[idx](x)
    pooled = cnn_model.backbone.avgpool(x)
    normalized = cnn_model.backbone.classifier[0](pooled)
    features = cnn_model.backbone.classifier[1](normalized)
    return cnn_model.classify_from_features(features)


@torch.inference_mode()
def extract_stage_split(
    cnn_model: ConvNeXtTinyClassifier,
    paths: ExternalPaths,
    split: str,
    subset_size: int | None,
    image_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = make_eval_loader(paths, split, subset_size=subset_size, batch_size=image_batch_size, device=device)
    map_chunks = []
    label_chunks = []
    for batch in loader:
        images, labels = batch[0].to(device, non_blocking=True), batch[1]
        maps = forward_until_stage(cnn_model, images)
        map_chunks.append(maps.detach().cpu().to(torch.float16))
        label_chunks.append(labels.detach().cpu())
    return torch.cat(map_chunks, dim=0), torch.cat(label_chunks, dim=0)


def cache_path_for(output_dir: Path, train_images: int | None, val_images: int | None, test_images: int | None) -> Path:
    def suffix(value: int | None) -> str:
        return "all" if value is None else str(int(value))

    cache_name = f"{STAGE_NAME}_maps_train{suffix(train_images)}_val{suffix(val_images)}_test{suffix(test_images)}.pt"
    return output_dir / "features" / cache_name


def get_stage_data(
    cnn_model: ConvNeXtTinyClassifier,
    paths: ExternalPaths,
    args: argparse.Namespace,
    device: torch.device,
) -> StageData:
    cache_path = cache_path_for(args.output_dir, args.train_images, args.val_images, args.test_images)
    split_sizes = {"train": args.train_images, "val": args.val_images, "test": args.test_images}
    if cache_path.exists() and not args.fresh_cache:
        print(f"[cache] Loading {STAGE_NAME} maps from {cache_path}", flush=True)
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        print(f"[cache] Extracting {STAGE_NAME} maps", flush=True)
        data = {}
        for split, subset_size in split_sizes.items():
            maps, labels = extract_stage_split(
                cnn_model,
                paths,
                split,
                subset_size=subset_size,
                image_batch_size=args.image_batch_size,
                device=device,
            )
            data[f"{split}_maps"] = maps
            data[f"{split}_labels"] = labels
            print(f"  {split}: maps={tuple(maps.shape)} labels={tuple(labels.shape)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache_path)
        print(f"[cache] Saved {cache_path}", flush=True)

    suffix_max_abs_diff = verify_suffix_replay(
        cnn_model,
        paths,
        data["test_maps"],
        max_images=min(args.suffix_check_images, data["test_maps"].shape[0]),
        device=device,
    )
    return StageData(
        train_maps=data["train_maps"],
        train_labels=data["train_labels"],
        val_maps=data["val_maps"],
        val_labels=data["val_labels"],
        test_maps=data["test_maps"],
        test_labels=data["test_labels"],
        suffix_max_abs_diff=suffix_max_abs_diff,
    )


@torch.inference_mode()
def verify_suffix_replay(
    cnn_model: ConvNeXtTinyClassifier,
    paths: ExternalPaths,
    maps: torch.Tensor,
    max_images: int,
    device: torch.device,
) -> float:
    if max_images <= 0:
        return 0.0
    replay_logits = []
    original_logits = []
    loader = make_eval_loader(paths, "test", subset_size=max_images, batch_size=max(1, min(16, max_images)), device=device)
    offset = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        batch_maps = maps[offset : offset + images.shape[0]].to(device, dtype=torch.float32)
        offset += images.shape[0]
        original_logits.append(cnn_model(images)["logits"].detach().cpu())
        replay_logits.append(classify_from_stage_map(cnn_model, batch_maps).detach().cpu())
    max_abs = (torch.cat(original_logits) - torch.cat(replay_logits)).abs().max().item()
    print(f"[check] {STAGE_NAME} suffix replay max abs logit diff: {max_abs:.3e}", flush=True)
    return float(max_abs)


def maps_to_vectors(maps: torch.Tensor) -> torch.Tensor:
    return maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1]).contiguous().float()


def vectors_to_maps(vectors: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
    n, c, h, w = shape
    return vectors.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
