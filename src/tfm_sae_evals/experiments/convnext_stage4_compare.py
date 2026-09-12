"""ConvNeXt Stage4 SAE architecture comparison.

Examples
--------
Run the default architecture comparison from the repository root:

    PYTHONPATH=src .venv/bin/python -m tfm_sae_evals.cli.main \
      convnext-stage4-compare \
      --config config/external_paths.example.toml

Run the quick reduced-data pilot used before scaling the experiment:

    PYTHONPATH=src .venv/bin/python -m tfm_sae_evals.cli.main \
      convnext-stage4-compare \
      --config config/external_paths.example.toml \
      --fresh \
      --subset-size 256 \
      --epochs 80 \
      --patience 8

Run the requested step-8 TopK-vs-original-Variable-TopK sweep:

    PYTHONPATH=src .venv/bin/python -m tfm_sae_evals.cli.main \
      convnext-stage4-compare \
      --config config/external_paths.example.toml \
      --fresh \
      --methods topk_nonorm variable_topk_original \
      --subset-size 512 \
      --epochs 300 \
      --sweep-logspace \
      --sweep-min-k 8 \
      --sweep-max-k 256 \
      --sweep-step-k 8 \
      --vtk-kmax-factor 1 \
      --vtk-budget-weight 0 \
      --vtk-hard-weight 0

The sweep writes `stage4_arch_compare_mse_l0.png` with only the points that
have completed in the current results CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import convnext_tiny

from tfm_sae_evals.config import ExternalPaths
from tfm_sae_evals.data.cub import CUBProcessedPKLDataset, make_cub_transforms
from tfm_sae_evals.models import (
    JumpReLUSparseAutoencoder,
    L1SparseAutoencoder,
    OriginalVariableTopKSparseAutoencoder,
    TopKSparseAutoencoder,
    TopValsMLPNoNormVariableTopKSparseAutoencoder,
)
from tfm_sae_evals.models.sae import unwrap_model
from tfm_sae_evals.models.variable_topk_loss import compute_discrete_kl, compute_prefix_expected_mse


SEED = 42
STAGE_NAME = "stage4"
STAGE_FEATURE_IDX = 7
STAGE_DIM = 768
STAGE_HW = 7
NUM_CLASSES = 200
DEFAULT_RESULTS_DIR = Path("results/convnext_stage4_arch_compare")
DEFAULT_OUTPUT_DIR = Path("outputs/convnext_stage4_arch_compare")
RESULTS_CSV = "stage4_arch_compare_results.csv"
REPORT_MD = "stage4_arch_compare_report.md"
PARETO_PNG = "stage4_arch_compare_mse_l0.png"
RELATIVE_IMPROVEMENT_PNG = "stage4_arch_compare_vartopk_relative_improvement.png"
RELATIVE_IMPROVEMENT_CSV = "stage4_arch_compare_vartopk_relative_improvement.csv"
SPARSE_PARETO_PNG = "stage4_arch_compare_sparse_mse_l0.png"
SPARSE_AGREEMENT_PNG = "stage4_arch_compare_sparse_top1_agreement_l0.png"
SPARSE_ACCURACY_PNG = "stage4_arch_compare_sparse_top1_accuracy_l0.png"
SPARSE_PARETO_PDF = "stage4_arch_compare_sparse_mse_l0.pdf"
SPARSE_AGREEMENT_PDF = "stage4_arch_compare_sparse_top1_agreement_l0.pdf"
SPARSE_ACCURACY_PDF = "stage4_arch_compare_sparse_top1_accuracy_l0.pdf"
DECODER_NORM_ABLATION_MSE_PNG = "stage4_arch_compare_decoder_norm_ablation_mse_l0.png"
DECODER_NORM_ABLATION_AGREEMENT_PNG = "stage4_arch_compare_decoder_norm_ablation_top1_agreement_l0.png"
DECODER_NORM_ABLATION_ACCURACY_PNG = "stage4_arch_compare_decoder_norm_ablation_top1_accuracy_l0.png"
DECODER_NORM_ABLATION_MSE_PDF = "stage4_arch_compare_decoder_norm_ablation_mse_l0.pdf"
DECODER_NORM_ABLATION_AGREEMENT_PDF = "stage4_arch_compare_decoder_norm_ablation_top1_agreement_l0.pdf"
DECODER_NORM_ABLATION_ACCURACY_PDF = "stage4_arch_compare_decoder_norm_ablation_top1_accuracy_l0.pdf"
DECODER_NORM_ABLATION_CSV = "stage4_arch_compare_decoder_norm_ablation_summary.csv"
PLOT_FIGSIZE = (10.5, 6.3)
PLOT_DPI = 220
PLOT_ANNOTATION_SIZE = 14

L1_METHODS = {"l1_sae", "l1_sae_no_decoder_norm"}
JUMPRELU_METHODS = {"jumprelu_sae", "jumprelu_sae_no_decoder_norm"}
SPARSE_PENALTY_METHODS = L1_METHODS | JUMPRELU_METHODS

METHODS = (
    "topk_nonorm",
    "variable_topk_original",
    "variable_topk_topvals_mlp_nonorm",
    "l1_sae",
    "jumprelu_sae",
    "l1_sae_no_decoder_norm",
    "jumprelu_sae_no_decoder_norm",
)
CSV_FIELDNAMES = [
    "stage",
    "method",
    "status",
    "error",
    "d",
    "hidden_dim",
    "hidden_multiplier",
    "k",
    "k_max",
    "target_l0",
    "lambda_prior",
    "beta",
    "best_epoch",
    "epochs_trained",
    "stopped_reason",
    "runtime_seconds",
    "peak_memory_mb",
    "test_nmse",
    "test_mse",
    "l0",
    "dead_pct",
    "top1_agreement",
    "original_top1_accuracy",
    "reconstructed_top1_accuracy",
    "logit_kl",
    "suffix_max_abs_diff",
    "checkpoint_path",
    "sparsity_lambda",
    "sparsity_target_l0",
    "jump_init_threshold",
    "jump_bandwidth",
    "decoder_normalization",
]


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


class StageVectorDataset(torch.utils.data.Dataset):
    def __init__(self, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        self.vectors = vectors
        self.mean = mean
        self.std = std

    def __len__(self):
        return int(self.vectors.shape[0])

    def __getitem__(self, idx):
        vector = self.vectors[idx].float()
        return (vector - self.mean.squeeze(0)) / self.std.squeeze(0), torch.tensor(0, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_convnext_checkpoint(path: Path, device: torch.device):
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


def make_eval_loader(paths: ExternalPaths, split: str, subset_size: int | None, batch_size: int, device: torch.device):
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
def extract_stage_split(cnn_model, paths: ExternalPaths, split: str, subset_size: int | None, image_batch_size: int, device):
    loader = make_eval_loader(paths, split, subset_size=subset_size, batch_size=image_batch_size, device=device)
    map_chunks = []
    label_chunks = []
    for batch in loader:
        images, labels = batch[0].to(device, non_blocking=True), batch[1]
        maps = forward_until_stage(cnn_model, images)
        map_chunks.append(maps.detach().cpu().to(torch.float16))
        label_chunks.append(labels.detach().cpu())
    return torch.cat(map_chunks, dim=0), torch.cat(label_chunks, dim=0)


def cache_path_for(output_dir: Path, subset_size: int | None) -> Path:
    suffix = "" if subset_size is None else f"_subset{subset_size}"
    return output_dir / "features" / f"{STAGE_NAME}_maps{suffix}.pt"


def get_stage_data(cnn_model, paths: ExternalPaths, args: argparse.Namespace, device: torch.device) -> StageData:
    cache_path = cache_path_for(args.output_dir, args.subset_size)
    if cache_path.exists() and not args.fresh_cache:
        print(f"[cache] Loading {STAGE_NAME} maps from {cache_path}", flush=True)
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        print(f"[cache] Extracting {STAGE_NAME} maps", flush=True)
        data = {}
        for split in ("train", "val", "test"):
            maps, labels = extract_stage_split(
                cnn_model,
                paths,
                split,
                subset_size=args.subset_size,
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
def verify_suffix_replay(cnn_model, paths: ExternalPaths, maps: torch.Tensor, max_images: int, device: torch.device) -> float:
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


def make_vector_loader(vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, shuffle: bool):
    dataset = StageVectorDataset(vectors, mean=mean, std=std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def forward_sae(model, batch: torch.Tensor):
    out = model(batch, return_aux=True)
    missing = {"encoded", "reconstruction", "k_eval"}.difference(out)
    if missing:
        raise ValueError(f"SAE output missing keys: {sorted(missing)}")
    return out


def evaluate_reconstruction_mse(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    elements = 0
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = forward_sae(model, xb)
            mse_sum += F.mse_loss(out["reconstruction"], xb, reduction="sum").item()
            elements += xb.numel()
    return {"mse": mse_sum / max(elements, 1)}


def evaluate_validation_loss(model, method: str, loader: DataLoader, args, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            loss, _out = loss_for_batch(model, xb, method, args)
            loss_sum += loss.item() * xb.shape[0]
            n_samples += xb.shape[0]
    return loss_sum / max(n_samples, 1)


def loss_for_batch(model, xb: torch.Tensor, method: str, args: argparse.Namespace):
    out = forward_sae(model, xb)
    hard = F.mse_loss(out["reconstruction"], xb)
    if method == "topk_nonorm":
        return hard, out
    if method in L1_METHODS:
        sparsity = out["l1_penalty"].mean()
        return hard + args.l1_lambda * sparsity, out
    if method in JUMPRELU_METHODS:
        sparsity = out["l0_surrogate"].mean()
        return hard + args.jump_lambda * sparsity, out
    expected_mse, _ = compute_prefix_expected_mse(out["prefix_reconstruction"], xb, out["k_probs"])
    expected = expected_mse.mean()
    kl = compute_discrete_kl(out["k_probs"], out["k_log_probs"], args.vtk_lambda_prior).mean()


    budget = ((out["expected_k"].mean() - args.vtk_target_l0) / max(args.vtk_target_l0, 1.0)).pow(2) # Push through desired K, maybe test without this and see if it worsen results
    loss = (
        args.vtk_expected_weight * expected
        + args.vtk_hard_weight * hard
        + args.vtk_beta * kl
        + args.vtk_budget_weight * budget
    )
    return loss, out


def train_with_early_stopping(model, method: str, train_loader: DataLoader, val_loader: DataLoader, args, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(unwrap_model(model).state_dict())
    best_val = float("inf")
    best_ema = float("inf")
    best_epoch = 0
    no_improve = 0
    stop_reason = "max_epochs"

    for epoch_idx in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _out = loss_for_batch(model, xb, method, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.shape[0]
            n_samples += xb.shape[0]

        val_mse = evaluate_reconstruction_mse(model, val_loader, device)["mse"]
        val_objective = evaluate_validation_loss(model, method, val_loader, args, device)
        ema = val_objective if epoch_idx == 0 else args.ema_alpha * val_objective + (1.0 - args.ema_alpha) * ema
        scheduler.step(val_objective)
        if val_objective < best_val:
            best_val = val_objective
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
            best_epoch = epoch_idx + 1
        if ema < best_ema - args.min_delta:
            best_ema = ema
            no_improve = 0
        else:
            no_improve += 1

        log_this = epoch_idx == 0 or (epoch_idx + 1) % args.log_every == 0 or no_improve >= args.patience
        if log_this:
            print(
                f"    ep {epoch_idx + 1:03d}/{args.epochs} "
                f"loss={running_loss / max(n_samples, 1):.6f} "
                f"val_mse={val_mse:.6f} val_obj={val_objective:.6f} ema={ema:.6f} best_ep={best_epoch} "
                f"pat={no_improve}/{args.patience}",
                flush=True,
            )
        if no_improve >= args.patience:
            stop_reason = f"early_stop_patience_{args.patience}"
            break

    unwrap_model(model).load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_trained": epoch_idx + 1,
        "stopped_reason": stop_reason,
        "best_val_mse": best_val,
    }


@torch.inference_mode()
def compute_dead_pct(model, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, device) -> float:
    hidden_dim = int(getattr(unwrap_model(model), "hidden_dim"))
    fire = torch.zeros(hidden_dim, dtype=torch.bool)
    model.eval()
    for start in range(0, vectors.shape[0], batch_size):
        xb = ((vectors[start : start + batch_size].float() - mean) / std).to(device, non_blocking=True)
        out = forward_sae(model, xb)
        fire |= (out["encoded"].detach().cpu() > 0).any(dim=0)
    return 100.0 * (1.0 - fire.float().mean().item())


@torch.inference_mode()
def reconstruct_vectors(model, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, device):
    chunks = []
    l0_sum = 0.0
    n = 0
    for start in range(0, vectors.shape[0], batch_size):
        raw = vectors[start : start + batch_size].float()
        xb = ((raw - mean) / std).to(device, non_blocking=True)
        out = forward_sae(model, xb)
        rec = out["reconstruction"].detach().cpu() * std + mean
        chunks.append(rec)
        l0_sum += out["k_eval"].detach().float().sum().item()
        n += xb.shape[0]
    return torch.cat(chunks, dim=0), l0_sum, n


@torch.inference_mode()
def evaluate_downstream(cnn_model, model, maps, labels, mean, std, args, device):
    model.eval()
    feature_idx = STAGE_FEATURE_IDX
    del feature_idx
    mse_sum = 0.0
    total_elements = 0
    l0_sum = 0.0
    l0_n = 0
    agree = 0
    correct_orig = 0
    correct_rec = 0
    total_images = 0
    kl_sum = 0.0
    var = maps.float().var(unbiased=False).item()
    for start in range(0, maps.shape[0], args.eval_image_batch_size):
        batch_maps = maps[start : start + args.eval_image_batch_size].float()
        batch_labels = labels[start : start + batch_maps.shape[0]]
        vectors = maps_to_vectors(batch_maps)
        rec_vectors, batch_l0_sum, batch_l0_n = reconstruct_vectors(
            model,
            vectors,
            mean,
            std,
            batch_size=args.eval_vector_batch_size,
            device=device,
        )
        rec_maps = vectors_to_maps(rec_vectors, tuple(batch_maps.shape))
        mse_sum += F.mse_loss(rec_maps, batch_maps, reduction="sum").item()
        total_elements += batch_maps.numel()
        l0_sum += batch_l0_sum
        l0_n += batch_l0_n
        logits_orig = classify_from_stage_map(cnn_model, batch_maps.to(device))
        logits_rec = classify_from_stage_map(cnn_model, rec_maps.to(device))
        labels_device = batch_labels.to(device)
        pred_orig = logits_orig.argmax(dim=1)
        pred_rec = logits_rec.argmax(dim=1)
        correct_orig += (pred_orig == labels_device).sum().item()
        correct_rec += (pred_rec == labels_device).sum().item()
        agree += (pred_orig == pred_rec).sum().item()
        total_images += labels_device.shape[0]
        kl = F.kl_div(F.log_softmax(logits_rec, dim=1), F.softmax(logits_orig, dim=1), reduction="batchmean")
        kl_sum += kl.item() * labels_device.shape[0]
    mse = mse_sum / max(total_elements, 1)
    return {
        "test_mse": mse,
        "test_nmse": mse / max(var, 1e-12),
        "l0": l0_sum / max(l0_n, 1),
        "top1_agreement": agree / max(total_images, 1),
        "original_top1_accuracy": correct_orig / max(total_images, 1),
        "reconstructed_top1_accuracy": correct_rec / max(total_images, 1),
        "logit_kl": kl_sum / max(total_images, 1),
    }


def build_model(method: str, args: argparse.Namespace, device: torch.device):
    hidden_dim = int(args.hidden_multiplier * STAGE_DIM)
    if method == "topk_nonorm":
        return TopKSparseAutoencoder(STAGE_DIM, hidden_dim, args.k, normalize_decoder=False).to(device)
    if method in L1_METHODS:
        return L1SparseAutoencoder(
            STAGE_DIM,
            hidden_dim,
            normalize_decoder=decoder_normalization_enabled(method),
        ).to(device)
    if method in JUMPRELU_METHODS:
        return JumpReLUSparseAutoencoder(
            STAGE_DIM,
            hidden_dim,
            init_threshold=args.jump_init_threshold,
            bandwidth=args.jump_bandwidth,
            normalize_decoder=decoder_normalization_enabled(method),
        ).to(device)
    k_max = max(1, min(hidden_dim, int(round(args.vtk_kmax_factor * args.vtk_target_l0))))
    if method == "variable_topk_original":
        return OriginalVariableTopKSparseAutoencoder(STAGE_DIM, hidden_dim, k_max=k_max).to(device)
    if method == "variable_topk_topvals_mlp_nonorm":
        return TopValsMLPNoNormVariableTopKSparseAutoencoder(
            STAGE_DIM,
            hidden_dim,
            k_max=k_max,
            selector_hidden=args.vtk_selector_hidden,
        ).to(device)
    raise ValueError(f"Unsupported method: {method}")


def format_k_value(value: float | int) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def decoder_normalization_enabled(method: str) -> bool | None:
    if method in {"l1_sae", "jumprelu_sae"}:
        return True
    if method in {"l1_sae_no_decoder_norm", "jumprelu_sae_no_decoder_norm"}:
        return False
    return None


def sparse_family_for_method(method: str) -> str | None:
    if method in L1_METHODS:
        return "l1_sae"
    if method in JUMPRELU_METHODS:
        return "jumprelu_sae"
    return None


def current_target_for_method(method: str, args: argparse.Namespace) -> float:
    return float(args.k if method == "topk_nonorm" else args.vtk_target_l0)


def result_key(method: str, args: argparse.Namespace) -> str:
    return f"{method}:target={format_k_value(current_target_for_method(method, args))}"


def result_key_from_row(row: dict[str, object]) -> str:
    method = str(row.get("method", ""))
    raw_target = row.get("k") if method == "topk_nonorm" else row.get("target_l0")
    return f"{method}:target={format_k_value(float(raw_target or 0))}"


def checkpoint_name(method: str, args: argparse.Namespace) -> str:
    return f"{STAGE_NAME}_{method}_k{format_k_value(current_target_for_method(method, args))}.pt"


def logspace_k_values(min_k: float, max_k: float, points: int) -> list[int]:
    if min_k <= 0 or max_k <= 0:
        raise ValueError("Geometric sweep bounds must be positive.")
    if points < 1:
        raise ValueError("--sweep-points must be at least 1.")
    values = np.round(np.geomspace(float(min_k), float(max_k), num=int(points))).astype(int).tolist()
    if len(set(values)) != len(values):
        raise ValueError(f"Rounded geometric k values contain duplicates: {values}")
    return values


def step_k_values(min_k: float, max_k: float, step_k: int) -> list[int]:
    if min_k <= 0 or max_k <= 0:
        raise ValueError("Step sweep bounds must be positive.")
    if step_k < 1:
        raise ValueError("--sweep-step-k must be at least 1.")
    start = int(round(min_k))
    stop = int(round(max_k))
    values = [start]
    values.extend(range(start + int(step_k), stop + 1, int(step_k)))
    if values[-1] != stop:
        values.append(stop)
    values = sorted(set(values))
    if len(values) < 1:
        raise ValueError("Step sweep generated no k values.")
    return values


def completed_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    ensure_results_csv_fieldnames(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return {result_key_from_row(row) for row in csv.DictReader(handle) if row.get("status") == "ok"}


def load_sparse_hyperparams(csv_path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sparse hyperparameter CSV not found: {csv_path}")
    out: dict[str, dict[str, dict[str, float]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            method = str(row.get("method", ""))
            family = sparse_family_for_method(method)
            if family is None:
                continue
            target = format_k_value(float(row.get("target_l0") or 0))
            sparsity_lambda = finite_float(row.get("sparsity_lambda"))
            if not math.isfinite(sparsity_lambda):
                continue
            params = {"sparsity_lambda": sparsity_lambda}
            jump_init_threshold = finite_float(row.get("jump_init_threshold"))
            jump_bandwidth = finite_float(row.get("jump_bandwidth"))
            if math.isfinite(jump_init_threshold):
                params["jump_init_threshold"] = jump_init_threshold
            if math.isfinite(jump_bandwidth):
                params["jump_bandwidth"] = jump_bandwidth
            out.setdefault(family, {})[target] = params
    return out


def apply_sparse_hyperparams_for_target(args: argparse.Namespace) -> None:
    hyperparams = getattr(args, "_sparse_hyperparams", None)
    if not hyperparams:
        return
    target = format_k_value(args.vtk_target_l0)
    l1_params = hyperparams.get("l1_sae", {}).get(target)
    if l1_params is not None:
        args.l1_lambda = float(l1_params["sparsity_lambda"])
    jump_params = hyperparams.get("jumprelu_sae", {}).get(target)
    if jump_params is not None:
        args.jump_lambda = float(jump_params["sparsity_lambda"])
        if "jump_init_threshold" in jump_params:
            args.jump_init_threshold = float(jump_params["jump_init_threshold"])
        if "jump_bandwidth" in jump_params:
            args.jump_bandwidth = float(jump_params["jump_bandwidth"])


def ensure_results_csv_fieldnames(results_csv: Path) -> None:
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = reader.fieldnames or []
        rows = list(reader)
    if existing == CSV_FIELDNAMES:
        return
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def append_row(row: dict[str, object], results_csv: Path):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    ensure_results_csv_fieldnames(results_csv)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def run_one(cnn_model, method: str, stage_data: StageData, args, device: torch.device):
    hidden_dim = int(args.hidden_multiplier * STAGE_DIM)
    is_variable_topk = method.startswith("variable_topk")
    is_sparse_penalty = method in SPARSE_PENALTY_METHODS
    decoder_norm = decoder_normalization_enabled(method)
    train_vectors = maps_to_vectors(stage_data.train_maps)
    val_vectors = maps_to_vectors(stage_data.val_maps)
    mean = train_vectors.mean(dim=0, keepdim=True)
    std = train_vectors.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    batch_size = args.vtk_batch_size if is_variable_topk else args.batch_size
    train_loader = make_vector_loader(train_vectors, mean, std, batch_size=batch_size, shuffle=True)
    val_loader = make_vector_loader(val_vectors, mean, std, batch_size=batch_size, shuffle=False)
    model = build_model(method, args, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(f"\n[{STAGE_NAME} / {method}] d={STAGE_DIM} hidden={hidden_dim} train_vectors={len(train_vectors)} batch={batch_size}", flush=True)
    t0 = time.perf_counter()
    model, train_info = train_with_early_stopping(model, method, train_loader, val_loader, args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_memory_mb = 0.0
    runtime = time.perf_counter() - t0

    eval_metrics = evaluate_downstream(
        cnn_model,
        model,
        stage_data.test_maps,
        stage_data.test_labels,
        mean,
        std,
        args,
        device,
    )
    dead_pct = compute_dead_pct(model, train_vectors, mean, std, batch_size=args.dead_eval_batch_size, device=device)

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / checkpoint_name(method, args)
    k_max = int(getattr(unwrap_model(model), "k_max", 0)) if is_variable_topk else ""
    sparsity_lambda = ""
    if method in L1_METHODS:
        sparsity_lambda = args.l1_lambda
    elif method in JUMPRELU_METHODS:
        sparsity_lambda = args.jump_lambda
    torch.save(
        {
            "stage": STAGE_NAME,
            "method": method,
            "config": {
                "d": STAGE_DIM,
                "hidden_dim": hidden_dim,
                "hidden_multiplier": args.hidden_multiplier,
                "k": args.k if method == "topk_nonorm" else None,
                "k_max": k_max if is_variable_topk else None,
                "target_l0": args.vtk_target_l0 if method != "topk_nonorm" else None,
                "sparsity_lambda": sparsity_lambda if is_sparse_penalty else None,
                "jump_init_threshold": args.jump_init_threshold if method in JUMPRELU_METHODS else None,
                "jump_bandwidth": args.jump_bandwidth if method in JUMPRELU_METHODS else None,
                "decoder_normalization": decoder_norm,
            },
            "train_info": train_info,
            "eval_metrics": eval_metrics,
            "model_state_dict": unwrap_model(model).state_dict(),
            "mean": mean,
            "std": std,
        },
        ckpt_path,
    )

    row = {
        "stage": STAGE_NAME,
        "method": method,
        "status": "ok",
        "error": "",
        "d": STAGE_DIM,
        "hidden_dim": hidden_dim,
        "hidden_multiplier": args.hidden_multiplier,
        "k": args.k if method == "topk_nonorm" else "",
        "k_max": k_max,
        "target_l0": args.vtk_target_l0 if method != "topk_nonorm" else "",
        "lambda_prior": args.vtk_lambda_prior if is_variable_topk else "",
        "beta": args.vtk_beta if is_variable_topk else "",
        "best_epoch": train_info["best_epoch"],
        "epochs_trained": train_info["epochs_trained"],
        "stopped_reason": train_info["stopped_reason"],
        "runtime_seconds": round(runtime, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "dead_pct": round(dead_pct, 4),
        "suffix_max_abs_diff": stage_data.suffix_max_abs_diff,
        "checkpoint_path": str(ckpt_path),
        "sparsity_lambda": sparsity_lambda,
        "sparsity_target_l0": args.vtk_target_l0 if is_sparse_penalty else "",
        "jump_init_threshold": args.jump_init_threshold if method in JUMPRELU_METHODS else "",
        "jump_bandwidth": args.jump_bandwidth if method in JUMPRELU_METHODS else "",
        "decoder_normalization": "" if decoder_norm is None else ("enabled" if decoder_norm else "disabled"),
        **{key: round(value, 8) for key, value in eval_metrics.items()},
    }
    append_row(row, args.results_dir / RESULTS_CSV)
    write_report(args)
    write_pareto_plot(args)
    write_relative_improvement_plot(args)
    write_sparse_comparison_plots(args)
    write_decoder_norm_ablation_plots(args)
    print(
        f"[{STAGE_NAME} / {method}] done runtime={runtime:.1f}s "
        f"NMSE={eval_metrics['test_nmse']:.4f} L0={eval_metrics['l0']:.2f} "
        f"agree={eval_metrics['top1_agreement']:.3f} KL={eval_metrics['logit_kl']:.4f}",
        flush=True,
    )


def architecture_diff_markdown() -> str:
    return """| Architecture | K selector input | K selector | Decoder normalization | Notes |
| --- | --- | --- | --- | --- |
| `topk_nonorm` | Fixed `k=64` | None | Disabled | Matched TopK baseline. |
| `variable_topk_original` | Normalized SAE input vector `x` | `Linear(d -> k_max)` | Disabled by architecture | Original Variable TopK baseline in this clean repo. |
| `variable_topk_topvals_mlp_nonorm` | `concat(log1p(topk_vals), cumulative_topk_mass)` | `MLP(2*k_max -> 256 -> k_max)` | Disabled | Variant used in the Stage4 timing pilot; the selector reads sorted activation evidence instead of the raw input vector. |
| `l1_sae` | None | None | Enabled | ReLU SAE with fixed L1 activation penalty. |
| `jumprelu_sae` | None | None | Enabled | JumpReLU SAE with fixed expected-L0 surrogate penalty. |
| `l1_sae_no_decoder_norm` | None | None | Disabled | L1 ablation matching `l1_sae` except the decoder is not normalized at initialization. |
| `jumprelu_sae_no_decoder_norm` | None | None | Disabled | JumpReLU ablation matching `jumprelu_sae` except the decoder is not normalized at initialization. |
"""


def write_report(args: argparse.Namespace):
    results_csv = args.results_dir / RESULTS_CSV
    report_path = args.results_dir / REPORT_MD
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=report_sort_key)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    topk_mse_by_k = {
        format_k_value(float(row["k"])): float(row["test_mse"])
        for row in ok_rows
        if row.get("method") == "topk_nonorm" and row.get("k") not in (None, "")
    }

    lines = [
        "# ConvNeXt Stage4 SAE Architecture Comparison\n\n",
        "This run compares SAE architectures on ConvNeXt-Tiny CUB Stage4 maps before global average pooling.\n\n",
        "## Stage4 Feature Source\n\n",
        "| Field | Value |\n",
        "| --- | --- |\n",
        f"| Hook | `backbone.features[{STAGE_FEATURE_IDX}]` |\n",
        f"| Map shape | `[N, {STAGE_DIM}, {STAGE_HW}, {STAGE_HW}]` |\n",
        f"| SAE vectorization | `[N, {STAGE_DIM}, {STAGE_HW}, {STAGE_HW}] -> [N * {STAGE_HW * STAGE_HW}, {STAGE_DIM}]` |\n",
        f"| Hidden dimension | `{args.hidden_multiplier}d = {args.hidden_multiplier * STAGE_DIM}` |\n",
        "| Position | After the last ConvNeXt stage, before `avgpool` and the classifier. |\n\n",
        "## Architecture Differences\n\n",
        architecture_diff_markdown(),
        "\n## Results\n\n",
        "| Method | Runtime s | Best epoch | Epochs | MSE | NMSE | L0 | Dead % | Top-1 agree | Orig acc | Recon acc | Logit KL | MSE vs TopK |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in ok_rows:
        mse = float(row["test_mse"])
        target = row["k"] if row["method"] == "topk_nonorm" else row["target_l0"]
        topk_mse = topk_mse_by_k.get(format_k_value(float(target)))
        rel = "" if topk_mse is None else f"{100.0 * (mse - topk_mse) / topk_mse:+.2f}%"
        lines.append(
            f"| `{row['method']}` | {float(row['runtime_seconds']):.1f} | {row['best_epoch']} | {row['epochs_trained']} | "
            f"{mse:.6f} | {float(row['test_nmse']):.6f} | {float(row['l0']):.2f} | "
            f"{float(row['dead_pct']):.2f} | {float(row['top1_agreement']):.4f} | "
            f"{float(row['original_top1_accuracy']):.4f} | {float(row['reconstructed_top1_accuracy']):.4f} | "
            f"{float(row['logit_kl']):.6f} | {rel} |\n"
        )
    if args.sweep_logspace:
        sweep_values = sweep_values_for_args(args)
        topk_k_desc = ", ".join(map(str, sweep_values))
        vtk_target_desc = topk_k_desc
        vtk_kmax_desc = ", ".join(str(round(args.vtk_kmax_factor * k_value)) for k_value in sweep_values)
    else:
        topk_k_desc = str(args.k)
        vtk_target_desc = str(args.vtk_target_l0)
        vtk_kmax_desc = str(round(args.vtk_kmax_factor * args.vtk_target_l0))

    lines.extend(
        [
            "\n## Training Configuration\n\n",
            "| Field | Value |\n",
            "| --- | --- |\n",
            f"| Seed | `{SEED}` |\n",
            f"| Optimizer | `Adam(lr={args.lr})` |\n",
            f"| Max epochs | `{args.epochs}` |\n",
            f"| Early stopping | `patience={args.patience}`, EMA alpha `{args.ema_alpha}` |\n",
            f"| TopK batch size | `{args.batch_size}` vectors |\n",
            f"| Variable TopK batch size | `{args.vtk_batch_size}` vectors |\n",
            f"| TopK k | `{topk_k_desc}` |\n",
            f"| Variable target L0 | `{vtk_target_desc}` |\n",
            f"| Variable k_max | `{vtk_kmax_desc}` |\n",
            f"| Sweep enabled | `{args.sweep_logspace}` |\n",
            f"| Sweep k values | `{topk_k_desc if args.sweep_logspace else 'n/a'}` |\n",
            f"| lambda_prior | `{args.vtk_lambda_prior}` |\n",
            f"| beta | `{args.vtk_beta}` |\n",
            f"| budget weight | `{args.vtk_budget_weight}` |\n",
            f"| hard weight | `{args.vtk_hard_weight}` |\n",
            f"| L1 lambda | `{args.l1_lambda}` |\n",
            f"| JumpReLU lambda | `{args.jump_lambda}` |\n",
            f"| JumpReLU init threshold | `{args.jump_init_threshold}` |\n",
            f"| JumpReLU bandwidth | `{args.jump_bandwidth}` |\n",
            f"| Sparse hyperparams CSV | `{getattr(args, 'sparse_hyperparams_csv', None) or 'n/a'}` |\n",
        ]
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")


def report_sort_key(row: dict[str, object]) -> tuple[int, float, str]:
    method = row.get("method", "")
    method_rank = METHODS.index(method) if method in METHODS else 999
    raw_target = row.get("k") if method == "topk_nonorm" else row.get("target_l0")
    try:
        target = float(raw_target)
    except (TypeError, ValueError):
        target = math.inf
    return method_rank, target, str(method)


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def kmax_label_for_row(row: dict[str, object]) -> str:
    method = str(row.get("method", ""))
    if method == "topk_nonorm":
        return format_k_value(float(row.get("k") or 0))
    return format_k_value(float(row.get("k_max") or 0))


def target_label_for_row(row: dict[str, object]) -> str:
    method = str(row.get("method", ""))
    target = row.get("k") if method == "topk_nonorm" else row.get("target_l0")
    return format_k_value(float(target or 0))


def pareto_frontier_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    frontier = []
    best_mse = math.inf
    for l0, mse in sorted(points):
        if mse < best_mse:
            frontier.append((l0, mse))
            best_mse = mse
    return frontier


def plot_label_for_method(method: str, args: argparse.Namespace) -> str:
    if method == "topk_nonorm":
        return "TopK no-norm"
    if method == "variable_topk_original":
        return "Variable TopK"
    if method == "l1_sae":
        return "L1 SAE"
    if method == "jumprelu_sae":
        return "JumpReLU SAE"
    if method == "l1_sae_no_decoder_norm":
        return "L1 SAE no decoder norm"
    if method == "jumprelu_sae_no_decoder_norm":
        return "JumpReLU SAE no decoder norm"
    if method == "variable_topk_topvals_mlp_nonorm":
        return (
            "Variable TopK topvals MLP no-norm, "
            f"kmax_factor={args.vtk_kmax_factor:g}, "
            f"budget={args.vtk_budget_weight:g}, hard={args.vtk_hard_weight:g}"
        )
    return method


def configure_report_plot_style(plt) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 21,
            "axes.titlesize": 27,
            "axes.labelsize": 24,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 21,
            "legend.title_fontsize": 21,
            "figure.titlesize": 27,
            "lines.linewidth": 3.6,
            "lines.markersize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_pareto_plot(args: argparse.Namespace) -> None:
    results_csv = args.results_dir / RESULTS_CSV
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    points_by_method: dict[str, list[tuple[float, float, str]]] = {}
    all_points = []
    for row in rows:
        l0 = finite_float(row.get("l0"))
        mse = finite_float(row.get("test_mse"))
        if not (math.isfinite(l0) and math.isfinite(mse)):
            continue
        method = str(row.get("method", ""))
        label = f"Kmax={kmax_label_for_row(row)}"
        points_by_method.setdefault(method, []).append((l0, mse, label))
        all_points.append((l0, mse))
    if not all_points:
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5))
    markers = {
        "topk_nonorm": "o",
        "variable_topk_original": "s",
        "variable_topk_topvals_mlp_nonorm": "^",
    }
    for method in ("topk_nonorm", "variable_topk_original", "variable_topk_topvals_mlp_nonorm"):
        method_points = sorted(points_by_method.get(method, []))
        if not method_points:
            continue
        xs = [point[0] for point in method_points]
        ys = [point[1] for point in method_points]
        ax.plot(
            xs,
            ys,
            marker=markers.get(method, "o"),
            linewidth=2,
            markersize=6,
            label=plot_label_for_method(method, args),
        )
        if method == "topk_nonorm":
            xytext = (5, 10)
            va = "bottom"
        else:
            xytext = (5, -10)
            va = "top"
        for x, y, label in method_points:
            ax.annotate(
                label,
                (x, y),
                xytext=xytext,
                textcoords="offset points",
                fontsize=7,
                va=va,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            )

    frontier = pareto_frontier_points(all_points)
    if len(frontier) >= 2:
        ax.plot(
            [point[0] for point in frontier],
            [point[1] for point in frontier],
            "k--",
            linewidth=1.5,
            label="Pareto frontier",
        )

    ax.set_title("ConvNeXt Stage4 SAE Pareto Sweep: MSE vs L0")
    ax.set_xlabel("Effective L0 (mean active features)")
    ax.set_ylabel("Test MSE")
    ax.margins(x=0.05, y=0.12)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.results_dir / PARETO_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_sparse_comparison_plots(args: argparse.Namespace) -> None:
    results_csv = args.results_dir / RESULTS_CSV
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    methods = (
        "topk_nonorm",
        "variable_topk_original",
        "l1_sae",
        "jumprelu_sae",
        "l1_sae_no_decoder_norm",
        "jumprelu_sae_no_decoder_norm",
    )
    if not any(row.get("method") in SPARSE_PENALTY_METHODS for row in rows):
        return

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_report_plot_style(plt)

    styles = {
        "topk_nonorm": {"marker": "o", "color": "tab:blue"},
        "variable_topk_original": {"marker": "s", "color": "tab:orange"},
        "l1_sae": {"marker": "^", "color": "tab:green"},
        "jumprelu_sae": {"marker": "D", "color": "tab:red"},
        "l1_sae_no_decoder_norm": {"marker": "v", "color": "tab:olive"},
        "jumprelu_sae_no_decoder_norm": {"marker": "X", "color": "tab:pink"},
    }

    def annotate_kmax_labels(ax) -> None:
        label_positions = {
            8: (5.0, 0.0540, "center", "bottom"),
            11: (22.0, 0.0515, "left", "bottom"),
            16: (40.0, 0.0496, "left", "bottom"),
            23: (57.0, 0.0476, "left", "bottom"),
            32: (74.0, 0.0456, "left", "bottom"),
            45: (91.0, 0.0434, "left", "bottom"),
            64: (109.0, 0.0404, "left", "bottom"),
            91: (126.0, 0.0375, "left", "bottom"),
            128: (144.0, 0.0346, "left", "bottom"),
            181: (195.0, 0.0310, "left", "bottom"),
            256: (222.0, 0.0263, "left", "bottom"),
        }
        label_rows = [
            row
            for row in rows
            if row.get("method") in {"topk_nonorm", "variable_topk_original"}
            and math.isfinite(finite_float(row.get("l0")))
            and math.isfinite(finite_float(row.get("test_mse")))
        ]
        grouped: dict[int, list[dict[str, object]]] = {}
        for row in label_rows:
            kmax = int(round(finite_float(row.get("k") if row.get("method") == "topk_nonorm" else row.get("k_max"))))
            grouped.setdefault(kmax, []).append(row)
        for kmax, group in sorted(grouped.items()):
            default_x = sum(finite_float(row.get("l0")) for row in group) / len(group)
            default_y = max(finite_float(row.get("test_mse")) for row in group)
            text_x, text_y, ha, va = label_positions.get(kmax, (default_x, default_y, "left", "bottom"))
            for row in group:
                ax.plot(
                    [text_x, finite_float(row.get("l0"))],
                    [text_y, finite_float(row.get("test_mse"))],
                    color="0.25",
                    linewidth=1.2,
                    zorder=1,
                )
            ax.text(
                text_x,
                text_y,
                f"Kmax={kmax}",
                fontsize=PLOT_ANNOTATION_SIZE,
                ha=ha,
                va=va,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
                zorder=5,
            )

    def plot_metric(
        png_filename: str,
        pdf_filename: str,
        title: str,
        ylabel: str,
        metric: str,
        scale: float = 1.0,
        annotate_kmax: bool = False,
        original_line: bool = False,
        log_y: bool = False,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for method in methods:
            method_rows = sorted(
                [
                    row
                    for row in rows
                    if row.get("method") == method
                    and math.isfinite(finite_float(row.get("l0")))
                    and math.isfinite(finite_float(row.get(metric)))
                ],
                key=lambda row: finite_float(row.get("l0")),
            )
            if not method_rows:
                continue
            xs = [finite_float(row.get("l0")) for row in method_rows]
            ys = [scale * finite_float(row.get(metric)) for row in method_rows]
            style = styles[method]
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                color=style["color"],
                label=plot_label_for_method(method, args),
            )
        if annotate_kmax:
            annotate_kmax_labels(ax)
        if original_line:
            original_values = [
                finite_float(row.get("original_top1_accuracy"))
                for row in rows
                if math.isfinite(finite_float(row.get("original_top1_accuracy")))
            ]
            if original_values:
                original = scale * original_values[0]
                ax.axhline(
                    original,
                    color="black",
                    linestyle="--",
                    linewidth=2.5,
                    label=f"Original ConvNeXt ({original:.2f}%)",
                )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Effective L0 (mean active features)")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        if y_min is not None or y_max is not None:
            current_min, current_max = ax.get_ylim()
            ax.set_ylim(bottom=y_min if y_min is not None else current_min, top=y_max if y_max is not None else current_max)
        if log_y and y_min == 80.0 and y_max in {92.0, 100.0}:
            ticks = [80, 82, 84, 86, 88, 90, 92] if y_max == 92.0 else [80, 85, 90, 95, 100]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(tick) for tick in ticks])
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True, borderpad=0.55, labelspacing=0.55, handlelength=2.0)
        ax.margins(x=0.05, y=0.12)
        fig.savefig(args.results_dir / png_filename, dpi=PLOT_DPI, bbox_inches="tight")
        fig.savefig(args.results_dir / pdf_filename, bbox_inches="tight")
        plt.close(fig)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(
        SPARSE_PARETO_PNG,
        SPARSE_PARETO_PDF,
        "MSE vs L0",
        "Test MSE",
        "test_mse",
    )
    plot_metric(
        SPARSE_AGREEMENT_PNG,
        SPARSE_AGREEMENT_PDF,
        "Top-1 agreement vs L0",
        "Top-1 agreement (%)",
        "top1_agreement",
        scale=100.0,
        log_y=True,
        y_min=80.0,
        y_max=100.0,
    )
    plot_metric(
        SPARSE_ACCURACY_PNG,
        SPARSE_ACCURACY_PDF,
        "Top-1 accuracy vs L0",
        "Reconstructed Top-1 accuracy (%)",
        "reconstructed_top1_accuracy",
        scale=100.0,
        original_line=True,
        log_y=True,
        y_min=80.0,
        y_max=92.0,
    )


def read_ok_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def write_decoder_norm_ablation_plots(args: argparse.Namespace) -> None:
    baseline_csv = getattr(args, "sparse_hyperparams_csv", None)
    if baseline_csv is None:
        return
    baseline_csv = Path(baseline_csv)
    current_csv = args.results_dir / RESULTS_CSV
    baseline_rows = read_ok_rows(baseline_csv)
    current_rows = read_ok_rows(current_csv)
    if not baseline_rows or not current_rows:
        return

    pairs = (
        ("l1_sae", "l1_sae_no_decoder_norm", "L1 SAE"),
        ("jumprelu_sae", "jumprelu_sae_no_decoder_norm", "JumpReLU SAE"),
    )
    rows_by_method_target = {
        (str(row.get("method", "")), target_label_for_row(row)): row
        for row in [*baseline_rows, *current_rows]
        if str(row.get("method", "")) in SPARSE_PENALTY_METHODS
    }
    if not any(method in {"l1_sae_no_decoder_norm", "jumprelu_sae_no_decoder_norm"} for method, _target in rows_by_method_target):
        return

    summary_rows = []
    targets = sorted(
        {
            target
            for method, target in rows_by_method_target
            if method in SPARSE_PENALTY_METHODS and target != "0"
        },
        key=lambda value: float(value.replace("p", ".")),
    )
    for norm_method, no_norm_method, family_label in pairs:
        for target in targets:
            norm = rows_by_method_target.get((norm_method, target))
            no_norm = rows_by_method_target.get((no_norm_method, target))
            if norm is None or no_norm is None:
                continue
            norm_mse = finite_float(norm.get("test_mse"))
            no_norm_mse = finite_float(no_norm.get("test_mse"))
            norm_agreement = finite_float(norm.get("top1_agreement"))
            no_norm_agreement = finite_float(no_norm.get("top1_agreement"))
            norm_accuracy = finite_float(norm.get("reconstructed_top1_accuracy"))
            no_norm_accuracy = finite_float(no_norm.get("reconstructed_top1_accuracy"))
            if not all(
                math.isfinite(value)
                for value in [norm_mse, no_norm_mse, norm_agreement, no_norm_agreement, norm_accuracy, no_norm_accuracy]
            ):
                continue
            summary_rows.append(
                {
                    "family": family_label,
                    "target_l0": target,
                    "norm_l0": finite_float(norm.get("l0")),
                    "no_norm_l0": finite_float(no_norm.get("l0")),
                    "norm_mse": norm_mse,
                    "no_norm_mse": no_norm_mse,
                    "mse_delta_no_norm_minus_norm": no_norm_mse - norm_mse,
                    "norm_top1_agreement": norm_agreement,
                    "no_norm_top1_agreement": no_norm_agreement,
                    "top1_agreement_delta_no_norm_minus_norm": no_norm_agreement - norm_agreement,
                    "norm_reconstructed_top1_accuracy": norm_accuracy,
                    "no_norm_reconstructed_top1_accuracy": no_norm_accuracy,
                    "reconstructed_top1_accuracy_delta_no_norm_minus_norm": no_norm_accuracy - norm_accuracy,
                }
            )
    if not summary_rows:
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / DECODER_NORM_ABLATION_CSV).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_report_plot_style(plt)
    styles = {
        "l1_sae": {"marker": "^", "color": "tab:green", "linestyle": "-"},
        "l1_sae_no_decoder_norm": {"marker": "v", "color": "tab:olive", "linestyle": "--"},
        "jumprelu_sae": {"marker": "D", "color": "tab:red", "linestyle": "-"},
        "jumprelu_sae_no_decoder_norm": {"marker": "X", "color": "tab:pink", "linestyle": "--"},
    }

    def plot_metric(
        png_filename: str,
        pdf_filename: str,
        title: str,
        ylabel: str,
        metric: str,
        scale: float = 1.0,
        original_line: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, constrained_layout=True)
        for norm_method, no_norm_method, _family_label in pairs:
            for method in (norm_method, no_norm_method):
                method_rows = sorted(
                    [
                        row
                        for row in rows_by_method_target.values()
                        if row.get("method") == method
                        and math.isfinite(finite_float(row.get("l0")))
                        and math.isfinite(finite_float(row.get(metric)))
                    ],
                    key=lambda row: finite_float(row.get("l0")),
                )
                if not method_rows:
                    continue
                style = styles[method]
                ax.plot(
                    [finite_float(row.get("l0")) for row in method_rows],
                    [scale * finite_float(row.get(metric)) for row in method_rows],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    label=plot_label_for_method(method, args),
                )
        if original_line:
            original_values = [
                finite_float(row.get("original_top1_accuracy"))
                for row in [*baseline_rows, *current_rows]
                if math.isfinite(finite_float(row.get("original_top1_accuracy")))
            ]
            if original_values:
                original = scale * original_values[0]
                ax.axhline(
                    original,
                    color="black",
                    linestyle=":",
                    linewidth=2.5,
                    label=f"Original ConvNeXt ({original:.2f}%)",
                )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Effective L0 (mean active features)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True, borderpad=0.55, labelspacing=0.55, handlelength=2.0)
        ax.margins(x=0.05, y=0.12)
        fig.savefig(args.results_dir / png_filename, dpi=PLOT_DPI, bbox_inches="tight")
        fig.savefig(args.results_dir / pdf_filename, bbox_inches="tight")
        plt.close(fig)

    plot_metric(
        DECODER_NORM_ABLATION_MSE_PNG,
        DECODER_NORM_ABLATION_MSE_PDF,
        "Decoder normalization ablation: MSE vs L0",
        "Test MSE",
        "test_mse",
    )
    plot_metric(
        DECODER_NORM_ABLATION_AGREEMENT_PNG,
        DECODER_NORM_ABLATION_AGREEMENT_PDF,
        "Decoder normalization ablation: Top-1 agreement vs L0",
        "Top-1 agreement (%)",
        "top1_agreement",
        scale=100.0,
    )
    plot_metric(
        DECODER_NORM_ABLATION_ACCURACY_PNG,
        DECODER_NORM_ABLATION_ACCURACY_PDF,
        "Decoder normalization ablation: Top-1 accuracy vs L0",
        "Reconstructed Top-1 accuracy (%)",
        "reconstructed_top1_accuracy",
        scale=100.0,
        original_line=True,
    )


def interpolate_linear_with_extrapolation(x_values: list[float], y_values: list[float], x: float) -> tuple[float, bool]:
    if len(x_values) < 2:
        raise ValueError("Need at least two TopK points for interpolation.")
    pairs = sorted(zip(x_values, y_values, strict=True))
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    extrapolated = x < xs[0] or x > xs[-1]
    if x <= xs[0]:
        x0, x1 = xs[0], xs[1]
        y0, y1 = ys[0], ys[1]
    elif x >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]
        y0, y1 = ys[-2], ys[-1]
    else:
        for idx in range(len(xs) - 1):
            if xs[idx] <= x <= xs[idx + 1]:
                x0, x1 = xs[idx], xs[idx + 1]
                y0, y1 = ys[idx], ys[idx + 1]
                break
        else:
            raise RuntimeError(f"Could not locate interpolation interval for x={x}")
    if x1 == x0:
        return y0, extrapolated
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0), extrapolated


def write_relative_improvement_plot(args: argparse.Namespace) -> None:
    results_csv = args.results_dir / RESULTS_CSV
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]

    topk_rows = [row for row in rows if row.get("method") == "topk_nonorm"]
    var_rows = [row for row in rows if str(row.get("method", "")).startswith("variable_topk")]
    topk_points = [
        (finite_float(row.get("l0")), finite_float(row.get("test_mse")))
        for row in topk_rows
        if math.isfinite(finite_float(row.get("l0"))) and math.isfinite(finite_float(row.get("test_mse")))
    ]
    if len(topk_points) < 2 or not var_rows:
        return
    topk_l0 = [point[0] for point in sorted(topk_points)]
    topk_mse = [point[1] for point in sorted(topk_points)]

    comparison_rows = []
    for row in sorted(var_rows, key=report_sort_key):
        var_l0 = finite_float(row.get("l0"))
        var_mse = finite_float(row.get("test_mse"))
        if not (math.isfinite(var_l0) and math.isfinite(var_mse)):
            continue
        topk_interp, extrapolated = interpolate_linear_with_extrapolation(topk_l0, topk_mse, var_l0)
        mse_delta = topk_interp - var_mse
        rel_improvement = 100.0 * (topk_interp - var_mse) / topk_interp
        comparison_rows.append(
            {
                "method": row.get("method", ""),
                "target_k": target_label_for_row(row),
                "k_max": kmax_label_for_row(row),
                "var_l0": var_l0,
                "var_mse": var_mse,
                "topk_interpolated_mse": topk_interp,
                "mse_delta_topk_minus_vartopk": mse_delta,
                "relative_improvement_pct": rel_improvement,
                "topk_value_extrapolated": extrapolated,
            }
        )
    if not comparison_rows:
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / RELATIVE_IMPROVEMENT_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "target_k",
            "k_max",
            "var_l0",
            "var_mse",
            "topk_interpolated_mse",
            "mse_delta_topk_minus_vartopk",
            "relative_improvement_pct",
            "topk_value_extrapolated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(
                {
                    **row,
                    "var_l0": f"{row['var_l0']:.8f}",
                    "var_mse": f"{row['var_mse']:.8f}",
                    "topk_interpolated_mse": f"{row['topk_interpolated_mse']:.8f}",
                    "mse_delta_topk_minus_vartopk": f"{row['mse_delta_topk_minus_vartopk']:.8f}",
                    "relative_improvement_pct": f"{row['relative_improvement_pct']:.4f}",
                }
            )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [row["var_l0"] for row in comparison_rows]
    ys = [row["relative_improvement_pct"] for row in comparison_rows]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.scatter(xs, ys, color="#2ca02c", s=58, zorder=3)

    ax.set_title("Variable TopK Relative MSE Improvement vs Interpolated TopK")
    ax.set_xlabel("Variable TopK effective L0")
    ax.set_ylabel("Relative MSE improvement over TopK (%)")
    ax.margins(x=0.05, y=0.18)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.results_dir / RELATIVE_IMPROVEMENT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sweep_values_for_args(args: argparse.Namespace) -> list[int]:
    step_k = getattr(args, "sweep_step_k", None)
    if step_k is not None:
        return step_k_values(args.sweep_min_k, args.sweep_max_k, int(step_k))
    return logspace_k_values(args.sweep_min_k, args.sweep_max_k, args.sweep_points)


def args_for_sweep_point(args: argparse.Namespace, k_value: int) -> argparse.Namespace:
    point_args = copy.copy(args)
    point_args.k = int(k_value)
    point_args.vtk_target_l0 = float(k_value)
    apply_sparse_hyperparams_for_target(point_args)
    return point_args


def iter_run_args(args: argparse.Namespace):
    if not args.sweep_logspace:
        apply_sparse_hyperparams_for_target(args)
        yield args
        return
    for k_value in sweep_values_for_args(args):
        yield args_for_sweep_point(args, k_value)


def run_compare(args: argparse.Namespace) -> int:
    args.output_dir = Path(args.output_dir)
    args.results_dir = Path(args.results_dir)
    args.sparse_hyperparams_csv = Path(args.sparse_hyperparams_csv) if args.sparse_hyperparams_csv else None
    args._sparse_hyperparams = load_sparse_hyperparams(args.sparse_hyperparams_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    paths = ExternalPaths.from_toml(args.config)
    missing = paths.missing()
    if missing:
        raise FileNotFoundError("Missing external paths: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    print(f"Device: {device}", flush=True)
    print(f"Results: {args.results_dir / RESULTS_CSV}", flush=True)
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, sort_keys=True), flush=True)

    cnn_model = load_convnext_checkpoint(paths.convnext_checkpoint, device)
    done = set() if args.fresh else completed_keys(args.results_dir / RESULTS_CSV)
    stage_data = get_stage_data(cnn_model, paths, args, device)
    for point_args in iter_run_args(args):
        if args.sweep_logspace:
            print(f"\n[sweep] k={point_args.k} vtk_target_l0={point_args.vtk_target_l0:g}", flush=True)
        for method in args.methods:
            key = result_key(method, point_args)
            if key in done:
                print(f"[skip] {STAGE_NAME}/{key} already completed", flush=True)
                continue
            run_one(cnn_model, method, stage_data, point_args, device)
            done.add(key)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_report(args)
    write_pareto_plot(args)
    write_relative_improvement_plot(args)
    write_sparse_comparison_plots(args)
    write_decoder_norm_ablation_plots(args)
    print(f"Done. Wrote {args.results_dir / RESULTS_CSV}", flush=True)
    return 0


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("convnext-stage4-compare", help="Train Stage4 ConvNeXt SAE architecture comparison.")
    parser.add_argument("--config", type=Path, default=Path("config/external_paths.example.toml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["topk_nonorm", "variable_topk_original"])
    parser.add_argument("--fresh", action="store_true", help="Ignore completed rows and rerun selected methods.")
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("-subset-size", "--subset-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--eval-image-batch-size", type=int, default=16)
    parser.add_argument("--eval-vector-batch-size", type=int, default=4096)
    parser.add_argument("--dead-eval-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--vtk-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--lr-scheduler-min", type=float, default=1e-5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--hidden-multiplier", type=int, default=4)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--vtk-target-l0", type=float, default=64.0)
    parser.add_argument("--sweep-logspace", action="store_true")
    parser.add_argument("--sweep-min-k", type=float, default=8.0)
    parser.add_argument("--sweep-max-k", type=float, default=256.0)
    parser.add_argument("--sweep-points", type=int, default=11)
    parser.add_argument("--sweep-step-k", type=int, default=None)
    parser.add_argument("--vtk-kmax-factor", type=float, default=1.0)
    parser.add_argument("--vtk-lambda-prior", type=float, default=0.003)
    parser.add_argument("--vtk-beta", type=float, default=3e-4)
    parser.add_argument("--vtk-hard-weight", type=float, default=0.0)
    parser.add_argument("--vtk-expected-weight", type=float, default=1.0)
    parser.add_argument("--vtk-budget-weight", type=float, default=0.0)
    parser.add_argument("--vtk-selector-hidden", type=int, default=256)
    parser.add_argument("--l1-lambda", type=float, default=1e-4)
    parser.add_argument("--jump-lambda", type=float, default=3e-4)
    parser.add_argument("--jump-init-threshold", type=float, default=0.01)
    parser.add_argument("--jump-bandwidth", type=float, default=0.05)
    parser.add_argument(
        "--sparse-hyperparams-csv",
        type=Path,
        default=None,
        help="Reuse sparse lambda and JumpReLU settings from a previous Stage4 results CSV by target L0.",
    )
    parser.add_argument("--suffix-check-images", type=int, default=32)
    parser.set_defaults(func=run_compare)
