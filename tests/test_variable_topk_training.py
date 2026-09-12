import torch
from torch.utils.data import DataLoader, TensorDataset

from tfm_sae_evals.models import OriginalVariableTopKSparseAutoencoder
from tfm_sae_evals.training import (
    VariableTopKTrainingConfig,
    evaluate_variable_topk_sae,
    forward_variable_topk_for_loss,
    train_variable_topk_sae,
)


def test_variable_topk_training_helpers_run_on_feature_batches():
    torch.manual_seed(0)
    features = torch.randn(12, 6)
    labels = torch.zeros(12, dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)
    model = OriginalVariableTopKSparseAutoencoder(input_dim=6, hidden_dim=10, k_max=5)

    out = forward_variable_topk_for_loss(model, features[:4], "full", prefix_chunk_size=2)
    assert out["expected_mse"].shape == (4,)

    metrics = evaluate_variable_topk_sae(
        model,
        loader,
        device="cpu",
        loss_impl="chunked_exact",
        prefix_chunk_size=2,
    )
    assert set(metrics) == {
        "hard_recon_mse",
        "hard_recon_mae",
        "expected_recon_mse",
        "mean_k_eval",
        "mean_expected_k",
        "mean_entropy_k",
        "mean_abs_k_gap",
    }

    result = train_variable_topk_sae(
        model,
        loader,
        loader,
        VariableTopKTrainingConfig(
            num_epochs=1,
            beta=0.01,
            lambda_prior=0.1,
            loss_impl="chunked_exact",
            prefix_chunk_size=2,
        ),
        device="cpu",
        log_fn=None,
    )
    assert result.peak_memory_mb == 0.0
    assert result.val_metrics["hard_recon_mse"] >= 0.0
