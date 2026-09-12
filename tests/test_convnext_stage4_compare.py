from argparse import Namespace

import torch

from tfm_sae_evals.experiments.convnext_stage4_compare import (
    apply_sparse_hyperparams_for_target,
    iter_run_args,
    load_sparse_hyperparams,
    loss_for_batch,
    logspace_k_values,
    maps_to_vectors,
    result_key,
    step_k_values,
    vectors_to_maps,
)
from tfm_sae_evals.models import JumpReLUSparseAutoencoder, L1SparseAutoencoder


def test_stage_map_vector_round_trip():
    maps = torch.randn(2, 768, 7, 7)
    vectors = maps_to_vectors(maps)
    assert vectors.shape == (2 * 7 * 7, 768)
    restored = vectors_to_maps(vectors, tuple(maps.shape))
    assert restored.shape == maps.shape
    assert torch.equal(restored, maps)


def test_logspace_sweep_values_match_requested_range():
    assert logspace_k_values(8, 256, 11) == [8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]


def test_step_sweep_values_match_requested_range():
    assert step_k_values(8, 256, 8) == [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
        248,
        256,
    ]


def test_logspace_sweep_sets_topk_and_variable_target_l0():
    args = Namespace(
        sweep_logspace=True,
        sweep_min_k=8,
        sweep_max_k=256,
        sweep_points=11,
        k=64,
        vtk_target_l0=64.0,
    )

    points = list(iter_run_args(args))

    assert [point.k for point in points] == [8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]
    assert [point.vtk_target_l0 for point in points] == [
        8.0,
        11.0,
        16.0,
        23.0,
        32.0,
        45.0,
        64.0,
        91.0,
        128.0,
        181.0,
        256.0,
    ]


def test_step_sweep_sets_topk_and_variable_target_l0():
    args = Namespace(
        sweep_logspace=True,
        sweep_min_k=8,
        sweep_max_k=256,
        sweep_points=11,
        sweep_step_k=8,
        k=64,
        vtk_target_l0=64.0,
    )

    points = list(iter_run_args(args))

    assert [point.k for point in points] == step_k_values(8, 256, 8)
    assert [point.vtk_target_l0 for point in points] == [float(k) for k in step_k_values(8, 256, 8)]


def test_sparse_methods_use_target_l0_for_resume_key():
    args = Namespace(k=64, vtk_target_l0=23.0)

    assert result_key("l1_sae", args) == "l1_sae:target=23"
    assert result_key("jumprelu_sae", args) == "jumprelu_sae:target=23"
    assert result_key("l1_sae_no_decoder_norm", args) == "l1_sae_no_decoder_norm:target=23"
    assert result_key("jumprelu_sae_no_decoder_norm", args) == "jumprelu_sae_no_decoder_norm:target=23"


def test_sparse_losses_are_finite():
    x = torch.randn(4, 8)
    args = Namespace(l1_lambda=1e-4, jump_lambda=3e-4)

    l1_loss, l1_out = loss_for_batch(L1SparseAutoencoder(8, 16), x, "l1_sae", args)
    jump_loss, jump_out = loss_for_batch(JumpReLUSparseAutoencoder(8, 16), x, "jumprelu_sae", args)
    l1_no_norm_loss, l1_no_norm_out = loss_for_batch(
        L1SparseAutoencoder(8, 16, normalize_decoder=False),
        x,
        "l1_sae_no_decoder_norm",
        args,
    )
    jump_no_norm_loss, jump_no_norm_out = loss_for_batch(
        JumpReLUSparseAutoencoder(8, 16, normalize_decoder=False),
        x,
        "jumprelu_sae_no_decoder_norm",
        args,
    )

    assert torch.isfinite(l1_loss)
    assert torch.isfinite(jump_loss)
    assert torch.isfinite(l1_no_norm_loss)
    assert torch.isfinite(jump_no_norm_loss)
    assert "l1_penalty" in l1_out
    assert "l0_surrogate" in jump_out
    assert "l1_penalty" in l1_no_norm_out
    assert "l0_surrogate" in jump_no_norm_out


def test_sparse_hyperparams_csv_applies_by_target(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "stage,method,status,target_l0,sparsity_lambda,jump_init_threshold,jump_bandwidth",
                "stage4,l1_sae,ok,16,0.025,,",
                "stage4,jumprelu_sae,ok,16,0.012,0.01,0.05",
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        vtk_target_l0=16.0,
        l1_lambda=1e-4,
        jump_lambda=3e-4,
        jump_init_threshold=0.02,
        jump_bandwidth=0.1,
        _sparse_hyperparams=load_sparse_hyperparams(csv_path),
    )

    apply_sparse_hyperparams_for_target(args)

    assert args.l1_lambda == 0.025
    assert args.jump_lambda == 0.012
    assert args.jump_init_threshold == 0.01
    assert args.jump_bandwidth == 0.05
