import torch

from tfm_sae_evals.models import OriginalVariableTopKSparseAutoencoder
from tfm_sae_evals.models.variable_topk_loss import compute_chunked_prefix_expected_mse, compute_prefix_expected_mse


def test_chunked_prefix_expected_mse_matches_full():
    torch.manual_seed(0)
    model = OriginalVariableTopKSparseAutoencoder(input_dim=6, hidden_dim=10, k_max=5)
    x = torch.randn(3, 6)
    out = model(x, return_aux=True)
    full, full_per_k = compute_prefix_expected_mse(out["prefix_reconstruction"], x, out["k_probs"])
    chunked, chunked_per_k = compute_chunked_prefix_expected_mse(
        model,
        out["topk_vals"],
        out["topk_idx"],
        x,
        out["k_probs"],
        prefix_chunk_size=2,
        return_per_k_mse=True,
    )
    assert torch.allclose(chunked, full, atol=1e-6)
    assert torch.allclose(chunked_per_k, full_per_k, atol=1e-6)
