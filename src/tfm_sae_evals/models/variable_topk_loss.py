"""Variable TopK loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .sae import unwrap_model


def make_discrete_log_prior(k_max: int, lambda_prior: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k_values = torch.arange(1, int(k_max) + 1, device=device, dtype=dtype)
    log_prior = -float(lambda_prior) * k_values
    return log_prior - torch.logsumexp(log_prior, dim=0)


def compute_discrete_kl(k_probs: torch.Tensor, k_log_probs: torch.Tensor, lambda_prior: float) -> torch.Tensor:
    log_prior = make_discrete_log_prior(
        k_max=k_probs.shape[-1],
        lambda_prior=lambda_prior,
        device=k_probs.device,
        dtype=k_probs.dtype,
    )
    return (k_probs * (k_log_probs - log_prior.unsqueeze(0))).sum(dim=-1)


def compute_prefix_expected_mse(prefix_reconstruction: torch.Tensor, target: torch.Tensor, k_probs: torch.Tensor):
    per_k_mse = (prefix_reconstruction - target.unsqueeze(1)).pow(2).mean(dim=-1)
    return (k_probs * per_k_mse).sum(dim=1), per_k_mse


def compute_chunked_prefix_expected_mse(
    base_model,
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    target: torch.Tensor,
    k_probs: torch.Tensor,
    prefix_chunk_size: int = 64,
    return_per_k_mse: bool = False,
):
    chunk_size = min(max(int(prefix_chunk_size), 1), topk_vals.shape[1])
    expected_mse = target.new_zeros(target.shape[0])
    per_k_chunks = []
    running_reconstruction = target.new_zeros(target.shape[0], target.shape[1])

    for start in range(0, topk_vals.shape[1], chunk_size):
        end = min(start + chunk_size, topk_vals.shape[1])
        cols = base_model._decoder_columns(topk_idx[:, start:end])
        contributions = cols * topk_vals[:, start:end].unsqueeze(-1)
        prefix_reconstruction = running_reconstruction.unsqueeze(1) + contributions.cumsum(dim=1)
        per_k_mse = (prefix_reconstruction - target.unsqueeze(1)).pow(2).mean(dim=-1)
        expected_mse = expected_mse + (k_probs[:, start:end] * per_k_mse).sum(dim=1)
        if return_per_k_mse:
            per_k_chunks.append(per_k_mse)
        running_reconstruction = prefix_reconstruction[:, -1]

    return expected_mse, torch.cat(per_k_chunks, dim=1) if return_per_k_mse else None


def forward_variable_topk_chunked_exact(model, x: torch.Tensor, prefix_chunk_size: int = 64):
    base_model = unwrap_model(model)
    selector_input, pre_act = base_model.encode_pre_activation(x)
    topk_vals, topk_idx = base_model._topk_candidates(pre_act)
    k_logits = base_model.k_selector(selector_input)
    k_log_probs = F.log_softmax(k_logits, dim=-1)
    k_probs = k_log_probs.exp()
    expected_mse, _ = compute_chunked_prefix_expected_mse(base_model, topk_vals, topk_idx, x, k_probs, prefix_chunk_size)
    k_eval = k_logits.argmax(dim=-1) + 1
    encoded = base_model._scatter_encoded(
        pre_act,
        topk_idx,
        topk_vals * base_model._hard_slot_mask_from_k(k_eval, pre_act.dtype),
    )
    reconstruction = base_model.decoder(encoded)
    return {
        "pre_act": pre_act,
        "topk_vals": topk_vals,
        "topk_idx": topk_idx,
        "k_logits": k_logits,
        "k_log_probs": k_log_probs,
        "k_probs": k_probs,
        "expected_k": base_model._expected_k_from_probs(k_probs),
        "entropy_k": -(k_probs * k_log_probs).sum(dim=-1),
        "k_eval": k_eval,
        "encoded": encoded,
        "reconstruction": reconstruction,
        "expected_mse": expected_mse,
    }
