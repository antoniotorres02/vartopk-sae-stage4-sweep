"""Common SAE architectures used by the CUB feature-space evaluations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k: int, normalize_decoder: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = min(int(k), self.hidden_dim)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.b_e = nn.Parameter(torch.zeros(self.hidden_dim))
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.b_d = nn.Parameter(torch.zeros(self.input_dim))
        self.normalize_decoder_enabled = bool(normalize_decoder)
        if self.normalize_decoder_enabled:
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        weight = self.decoder.weight.data
        self.decoder.weight.data = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        pre_act = F.relu(self.encoder(x - self.b_d) + self.b_e)
        if self.k < self.hidden_dim:
            topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=1)
            encoded = torch.zeros_like(pre_act)
            encoded.scatter_(1, topk_idx, topk_vals)
        else:
            encoded = pre_act
        return encoded, pre_act

    def decode(self, encoded):
        return self.decoder(encoded) + self.b_d

    def forward(self, x, return_aux: bool = False):
        encoded, pre_act = self.encode(x)
        reconstruction = self.decode(encoded)
        k_eval = (encoded > 0).sum(dim=1)
        out = {
            "pre_act": pre_act,
            "encoded": encoded,
            "reconstruction": reconstruction,
            "k_eval": k_eval,
            "expected_k": k_eval.float(),
            "entropy_k": torch.zeros_like(k_eval, dtype=x.dtype),
        }
        return out if return_aux else (encoded, reconstruction)


class L1SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, normalize_decoder: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim)
        self.normalize_decoder_enabled = bool(normalize_decoder)
        if self.normalize_decoder_enabled:
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        weight = self.decoder.weight.data
        self.decoder.weight.data = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-8)

    def forward(self, x, return_aux: bool = False):
        encoded = F.relu(self.encoder(x))
        reconstruction = self.decoder(encoded)
        k_eval = (encoded > 0).sum(dim=1)
        out = {
            "encoded": encoded,
            "reconstruction": reconstruction,
            "k_eval": k_eval,
            "expected_k": k_eval.float(),
            "entropy_k": torch.zeros_like(k_eval, dtype=x.dtype),
            "l1_penalty": encoded.abs().sum(dim=1),
        }
        return out if return_aux else (encoded, reconstruction)


class JumpReLUSparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        init_threshold: float = 0.01,
        bandwidth: float = 0.05,
        normalize_decoder: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bandwidth = float(bandwidth)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))
        self.log_threshold = nn.Parameter(torch.full((self.hidden_dim,), float(np.log(init_threshold))))
        self.normalize_decoder_enabled = bool(normalize_decoder)
        if self.normalize_decoder_enabled:
            self.normalize_decoder()

    @property
    def threshold(self):
        return torch.exp(self.log_threshold)

    @torch.no_grad()
    def normalize_decoder(self):
        weight = self.decoder.weight.data
        self.decoder.weight.data = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        hidden_pre = self.encoder(x - self.b_dec) + self.b_enc
        base_acts = F.relu(hidden_pre)
        soft_gate = torch.sigmoid((hidden_pre - self.threshold) / max(self.bandwidth, 1e-8))
        hard_gate = (hidden_pre > self.threshold).to(base_acts.dtype)
        encoded = base_acts * (hard_gate.detach() - soft_gate.detach() + soft_gate)
        return encoded, hidden_pre, soft_gate

    def decode(self, encoded):
        return self.decoder(encoded) + self.b_dec

    def forward(self, x, return_aux: bool = False):
        encoded, hidden_pre, soft_gate = self.encode(x)
        reconstruction = self.decode(encoded)
        k_eval = (encoded > 0).sum(dim=1)
        expected_k = soft_gate.sum(dim=1)
        out = {
            "hidden_pre": hidden_pre,
            "encoded": encoded,
            "reconstruction": reconstruction,
            "k_eval": k_eval,
            "expected_k": expected_k,
            "entropy_k": torch.zeros_like(k_eval, dtype=x.dtype),
            "l0_surrogate": expected_k,
            "threshold": self.threshold,
        }
        return out if return_aux else (encoded, reconstruction)


class OriginalVariableTopKSparseAutoencoder(nn.Module):
    """Variable TopK model matching the unconstrained original architecture."""

    method_name = "variable_topk_exact_original"

    def __init__(self, input_dim: int, hidden_dim: int, k_max: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_max = min(int(k_max), self.hidden_dim)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.k_selector = nn.Linear(self.input_dim, self.k_max)
        self.register_buffer("slot_positions", torch.arange(1, self.k_max + 1, dtype=torch.float32), persistent=False)

    def encode_pre_activation(self, x):
        return x, F.relu(self.encoder(x))

    def _topk_candidates(self, pre_act):
        return torch.topk(pre_act, self.k_max, dim=1)

    def _decoder_columns(self, topk_idx):
        return self.decoder.weight.transpose(0, 1)[topk_idx]

    def _scatter_encoded(self, pre_act, topk_idx, weighted_topk_vals):
        encoded = torch.zeros_like(pre_act)
        encoded.scatter_(1, topk_idx, weighted_topk_vals)
        return encoded

    def _hard_slot_mask_from_k(self, k_values, dtype):
        return (self.slot_positions.view(1, -1) <= k_values.unsqueeze(1)).to(dtype=dtype)

    def _expected_k_from_probs(self, k_probs):
        return (k_probs * self.slot_positions.to(device=k_probs.device, dtype=k_probs.dtype)).sum(dim=-1)

    def forward(self, x, return_aux: bool = False):
        selector_input, pre_act = self.encode_pre_activation(x)
        topk_vals, topk_idx = self._topk_candidates(pre_act)
        cols = self._decoder_columns(topk_idx)
        prefix_reconstruction = (cols * topk_vals.unsqueeze(-1)).cumsum(dim=1)
        k_logits = self.k_selector(selector_input)
        k_log_probs = F.log_softmax(k_logits, dim=-1)
        k_probs = k_log_probs.exp()
        k_eval = k_logits.argmax(dim=-1) + 1
        batch_idx = torch.arange(x.shape[0], device=x.device)
        reconstruction = prefix_reconstruction[batch_idx, k_eval - 1]
        encoded = self._scatter_encoded(pre_act, topk_idx, topk_vals * self._hard_slot_mask_from_k(k_eval, pre_act.dtype))
        out = {
            "pre_act": pre_act,
            "topk_vals": topk_vals,
            "topk_idx": topk_idx,
            "prefix_reconstruction": prefix_reconstruction,
            "k_logits": k_logits,
            "k_log_probs": k_log_probs,
            "k_probs": k_probs,
            "expected_k": self._expected_k_from_probs(k_probs),
            "entropy_k": -(k_probs * k_log_probs).sum(dim=-1),
            "k_eval": k_eval,
            "encoded": encoded,
            "reconstruction": reconstruction,
        }
        return out if return_aux else (encoded, reconstruction)


class TopValsMLPNoNormVariableTopKSparseAutoencoder(nn.Module):
    """Variable TopK model whose selector reads sorted top-activation evidence.

    This is the architecture used by the ConvNeXt Stage4 timing pilot. It keeps
    the original prefix-reconstruction Variable TopK mechanism, but replaces the
    original linear selector over the input vector with an MLP over top-k
    activation evidence.
    """

    method_name = "variable_topk_topvals_mlp_nonorm"

    def __init__(self, input_dim: int, hidden_dim: int, k_max: int, selector_hidden: int = 256):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k_max = min(int(k_max), self.hidden_dim)
        self.selector_hidden = int(selector_hidden)
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.b_e = nn.Parameter(torch.zeros(self.hidden_dim))
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.b_d = nn.Parameter(torch.zeros(self.input_dim))
        self.k_selector = nn.Sequential(
            nn.Linear(2 * self.k_max, self.selector_hidden),
            nn.ReLU(),
            nn.Linear(self.selector_hidden, self.k_max),
        )
        self.register_buffer("slot_positions", torch.arange(1, self.k_max + 1, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def normalize_decoder(self):
        return None

    def encode_pre_activation(self, x):
        x_centered = x - self.b_d
        return x_centered, F.relu(self.encoder(x_centered) + self.b_e)

    def _topk_candidates(self, pre_act):
        return torch.topk(pre_act, self.k_max, dim=1)

    def _decoder_columns(self, topk_idx):
        return self.decoder.weight.transpose(0, 1)[topk_idx]

    def _scatter_encoded(self, pre_act, topk_idx, weighted_topk_vals):
        encoded = torch.zeros_like(pre_act)
        encoded.scatter_(1, topk_idx, weighted_topk_vals)
        return encoded

    def _prefix_reconstructions(self, topk_vals, topk_idx):
        cols = self._decoder_columns(topk_idx)
        contributions = cols * topk_vals.unsqueeze(-1)
        return contributions.cumsum(dim=1) + self.b_d.view(1, 1, -1)

    def _hard_slot_mask_from_k(self, k_values, dtype):
        return (self.slot_positions.view(1, -1) <= k_values.unsqueeze(1)).to(dtype=dtype)

    def _expected_k_from_probs(self, k_probs):
        return (k_probs * self.slot_positions.to(device=k_probs.device, dtype=k_probs.dtype)).sum(dim=-1)

    def _selector_input_from_topvals(self, topk_vals):
        log_vals = torch.log1p(topk_vals)
        cumulative = topk_vals.cumsum(dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        return torch.cat([log_vals, cumulative], dim=1)

    def forward(self, x, return_aux: bool = False):
        _x_centered, pre_act = self.encode_pre_activation(x)
        topk_vals, topk_idx = self._topk_candidates(pre_act)
        prefix_reconstruction = self._prefix_reconstructions(topk_vals, topk_idx)
        k_logits = self.k_selector(self._selector_input_from_topvals(topk_vals))
        k_log_probs = F.log_softmax(k_logits, dim=-1)
        k_probs = k_log_probs.exp()
        k_eval = k_logits.argmax(dim=-1) + 1
        batch_idx = torch.arange(x.shape[0], device=x.device)
        reconstruction = prefix_reconstruction[batch_idx, k_eval - 1]
        encoded = self._scatter_encoded(pre_act, topk_idx, topk_vals * self._hard_slot_mask_from_k(k_eval, pre_act.dtype))
        out = {
            "pre_act": pre_act,
            "topk_vals": topk_vals,
            "topk_idx": topk_idx,
            "prefix_reconstruction": prefix_reconstruction,
            "expected_reconstruction": (prefix_reconstruction * k_probs.unsqueeze(-1)).sum(dim=1),
            "k_logits": k_logits,
            "k_log_probs": k_log_probs,
            "k_probs": k_probs,
            "expected_k": self._expected_k_from_probs(k_probs),
            "entropy_k": -(k_probs * k_log_probs).sum(dim=-1),
            "k_eval": k_eval,
            "encoded": encoded,
            "reconstruction": reconstruction,
        }
        return out if return_aux else (encoded, reconstruction)
