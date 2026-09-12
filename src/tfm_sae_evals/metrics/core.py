"""Core SAE metric helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


REQUIRED_SAE_OUTPUT_KEYS = {"encoded", "reconstruction", "k_eval", "expected_k", "entropy_k"}


def require_sae_output(output: dict) -> None:
    missing = REQUIRED_SAE_OUTPUT_KEYS.difference(output)
    if missing:
        raise ValueError(f"SAE output missing required keys: {sorted(missing)}")


def reconstruction_metrics(features: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    mse_per_sample = (reconstruction - features).pow(2).mean(dim=1)
    denom = features.pow(2).mean(dim=1).clamp_min(1e-12)
    return {
        "hard_recon_mse": float(mse_per_sample.mean().item()),
        "nmse": float((mse_per_sample / denom).mean().item()),
        "cosine_sim": float(F.cosine_similarity(reconstruction, features, dim=1).mean().item()),
    }
