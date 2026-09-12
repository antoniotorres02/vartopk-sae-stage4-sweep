import torch

from tfm_sae_evals.metrics import require_sae_output
from tfm_sae_evals.models import (
    JumpReLUSparseAutoencoder,
    L1SparseAutoencoder,
    OriginalVariableTopKSparseAutoencoder,
    TopValsMLPNoNormVariableTopKSparseAutoencoder,
    TopKSparseAutoencoder,
)


def test_sae_forward_contracts():
    x = torch.randn(4, 8)
    models = [
        TopKSparseAutoencoder(8, 16, 3),
        L1SparseAutoencoder(8, 16),
        JumpReLUSparseAutoencoder(8, 16),
        OriginalVariableTopKSparseAutoencoder(input_dim=8, hidden_dim=16, k_max=5),
        TopValsMLPNoNormVariableTopKSparseAutoencoder(input_dim=8, hidden_dim=16, k_max=5, selector_hidden=4),
    ]
    for model in models:
        out = model(x, return_aux=True)
        require_sae_output(out)
        assert out["reconstruction"].shape == x.shape
        assert out["encoded"].shape[0] == x.shape[0]


def test_topk_no_decoder_norm_forward_contract():
    x = torch.randn(4, 8)
    model = TopKSparseAutoencoder(8, 16, 3, normalize_decoder=False)
    out = model(x, return_aux=True)
    require_sae_output(out)
    assert out["reconstruction"].shape == x.shape
    assert model.normalize_decoder_enabled is False


def test_l1_and_jumprelu_no_decoder_norm_forward_contracts():
    x = torch.randn(4, 8)
    models = [
        L1SparseAutoencoder(8, 16, normalize_decoder=False),
        JumpReLUSparseAutoencoder(8, 16, normalize_decoder=False),
    ]
    for model in models:
        out = model(x, return_aux=True)
        require_sae_output(out)
        assert out["reconstruction"].shape == x.shape
        assert model.normalize_decoder_enabled is False
