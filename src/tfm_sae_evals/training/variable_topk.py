"""Canonical Variable TopK SAE training loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
import torch.optim as optim

from tfm_sae_evals.models.sae import unwrap_model
from tfm_sae_evals.models.variable_topk_loss import (
    compute_discrete_kl,
    compute_prefix_expected_mse,
    forward_variable_topk_chunked_exact,
)

VariableTopKLossImpl = Literal["full", "chunked_exact"]


@dataclass(frozen=True)
class VariableTopKTrainingConfig:
    num_epochs: int
    beta: float
    lambda_prior: float
    hard_recon_weight: float = 0.35
    learning_rate: float = 1e-3
    loss_impl: VariableTopKLossImpl = "full"
    prefix_chunk_size: int = 64
    selector_temp_start: float = 1.5
    selector_temp_end: float = 0.35
    use_compile: bool = False
    max_train_steps_per_epoch: int | None = None
    max_eval_batches: int | None = None
    log_every_epochs: int = 10


@dataclass(frozen=True)
class VariableTopKTrainingResult:
    model: torch.nn.Module
    val_metrics: dict[str, float]
    peak_memory_mb: float


def selector_temperature_for_epoch(epoch_idx: int, num_epochs: int, start: float, end: float) -> float:
    if num_epochs <= 1:
        return float(end)
    alpha = epoch_idx / float(num_epochs - 1)
    return float(start + alpha * (end - start))


def _features_from_batch(batch) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        for key in ("features", "x", "inputs"):
            value = batch.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    raise TypeError("Expected each batch to be a tensor, tuple/list with a tensor first, or dict with features/x/inputs.")


def forward_variable_topk_for_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    loss_impl: VariableTopKLossImpl,
    prefix_chunk_size: int,
) -> dict[str, torch.Tensor]:
    if loss_impl == "full":
        out = model(features, return_aux=True)
        required = {"k_eval", "encoded", "reconstruction", "prefix_reconstruction", "k_probs"}
        missing = required.difference(out)
        if missing:
            raise ValueError(f"Expected Variable TopK auxiliary outputs; missing {sorted(missing)}")
        expected_mse, _ = compute_prefix_expected_mse(
            out["prefix_reconstruction"],
            features,
            out["k_probs"],
        )
        out["expected_mse"] = expected_mse
        return out
    if loss_impl == "chunked_exact":
        return forward_variable_topk_chunked_exact(
            model,
            features,
            prefix_chunk_size=prefix_chunk_size,
        )
    raise ValueError(f"Unsupported loss_impl: {loss_impl!r}")


def evaluate_variable_topk_sae(
    model: torch.nn.Module,
    data_loader,
    *,
    device: torch.device | str,
    loss_impl: VariableTopKLossImpl = "full",
    prefix_chunk_size: int = 64,
    max_batches: int | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    total_elements = 0
    hard_recon_mse_sum = 0.0
    hard_recon_mae_sum = 0.0
    expected_recon_mse_sum = 0.0
    mean_k_eval_sum = 0.0
    mean_expected_k_sum = 0.0
    mean_entropy_sum = 0.0
    mean_abs_k_gap_sum = 0.0
    n_samples = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            features = _features_from_batch(batch).to(device, non_blocking=True)
            out = forward_variable_topk_for_loss(model, features, loss_impl, prefix_chunk_size)

            hard_recon_mse_sum += F.mse_loss(out["reconstruction"], features, reduction="sum").item()
            hard_recon_mae_sum += F.l1_loss(out["reconstruction"], features, reduction="sum").item()
            expected_recon_mse_sum += out["expected_mse"].sum().item() * features.shape[1]
            mean_k_eval_sum += out["k_eval"].sum().item()
            mean_expected_k_sum += out["expected_k"].sum().item()
            mean_entropy_sum += out["entropy_k"].sum().item()
            mean_abs_k_gap_sum += (out["expected_k"] - out["k_eval"].float()).abs().sum().item()
            total_elements += features.numel()
            n_samples += features.size(0)

    return {
        "hard_recon_mse": hard_recon_mse_sum / max(total_elements, 1),
        "hard_recon_mae": hard_recon_mae_sum / max(total_elements, 1),
        "expected_recon_mse": expected_recon_mse_sum / max(total_elements, 1),
        "mean_k_eval": mean_k_eval_sum / max(n_samples, 1),
        "mean_expected_k": mean_expected_k_sum / max(n_samples, 1),
        "mean_entropy_k": mean_entropy_sum / max(n_samples, 1),
        "mean_abs_k_gap": mean_abs_k_gap_sum / max(n_samples, 1),
    }


def _should_log_epoch(epoch_idx: int, num_epochs: int, log_every_epochs: int) -> bool:
    if log_every_epochs <= 0:
        return False
    return epoch_idx == 0 or (epoch_idx + 1) % log_every_epochs == 0 or epoch_idx == num_epochs - 1


def train_variable_topk_sae(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: VariableTopKTrainingConfig,
    *,
    device: torch.device | str,
    log_fn: Callable[[str], None] | None = print,
) -> VariableTopKTrainingResult:
    device = torch.device(device)
    model = model.to(device)
    if config.use_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("VariableTopKTrainingConfig.use_compile=True requires torch.compile.")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=float(config.learning_rate))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())
    best_val_mse = float("inf")
    peak_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(config.num_epochs)):
        model.train()
        selector_temp = selector_temperature_for_epoch(
            epoch,
            int(config.num_epochs),
            float(config.selector_temp_start),
            float(config.selector_temp_end),
        )

        total_loss_sum = 0.0
        exact_recon_mse_sum = 0.0
        hard_recon_mse_sum = 0.0
        n_samples = 0

        for step_idx, batch in enumerate(train_loader):
            if config.max_train_steps_per_epoch is not None and step_idx >= config.max_train_steps_per_epoch:
                break
            features = _features_from_batch(batch).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = forward_variable_topk_for_loss(model, features, config.loss_impl, config.prefix_chunk_size)
                exact_recon_mse = out["expected_mse"].mean()
                kl_k = compute_discrete_kl(out["k_probs"], out["k_log_probs"], config.lambda_prior).mean()
                hard_recon_mse = F.mse_loss(out["reconstruction"], features)
                total_loss = exact_recon_mse + (config.beta * kl_k) + (config.hard_recon_weight * hard_recon_mse)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = features.size(0)
            total_loss_sum += total_loss.item() * batch_size
            exact_recon_mse_sum += exact_recon_mse.item() * batch_size
            hard_recon_mse_sum += hard_recon_mse.item() * batch_size
            n_samples += batch_size

        val_metrics = evaluate_variable_topk_sae(
            model,
            val_loader,
            device=device,
            loss_impl=config.loss_impl,
            prefix_chunk_size=config.prefix_chunk_size,
            max_batches=config.max_eval_batches,
        )

        if val_metrics["hard_recon_mse"] < best_val_mse:
            best_val_mse = val_metrics["hard_recon_mse"]
            best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())

        if device.type == "cuda":
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / (1024**2))

        if log_fn is not None and _should_log_epoch(epoch, int(config.num_epochs), int(config.log_every_epochs)):
            log_fn(
                f"  Epoch {epoch + 1:3d}/{config.num_epochs} | "
                f"temp={selector_temp:.3f} | "
                f"loss={total_loss_sum / max(n_samples, 1):.6f} | "
                f"train_exact_mse={exact_recon_mse_sum / max(n_samples, 1):.6f} | "
                f"train_hard_mse={hard_recon_mse_sum / max(n_samples, 1):.6f} | "
                f"val_hard_mse={val_metrics['hard_recon_mse']:.6f} | "
                f"k_eval={val_metrics['mean_k_eval']:.1f} | "
                f"E[k]={val_metrics['mean_expected_k']:.1f}"
            )

    unwrap_model(model).load_state_dict(best_state_dict)
    final_val_metrics = evaluate_variable_topk_sae(
        model,
        val_loader,
        device=device,
        loss_impl=config.loss_impl,
        prefix_chunk_size=config.prefix_chunk_size,
        max_batches=config.max_eval_batches,
    )
    return VariableTopKTrainingResult(
        model=model,
        val_metrics=final_val_metrics,
        peak_memory_mb=peak_memory_mb,
    )
