from tfm_sae_evals.metrics import pareto_frontier, select_l0_bins


def test_pareto_frontier_minimizes_y_over_x():
    rows = [
        {"l0": 1, "mse": 0.5},
        {"l0": 2, "mse": 0.6},
        {"l0": 3, "mse": 0.4},
    ]
    assert pareto_frontier(rows, x_key="l0", y_key="mse") == [rows[0], rows[2]]


def test_select_l0_bins():
    rows = [
        {"architecture": "topk", "val_mean_k_eval": 31, "val_hard_recon_mse": 0.2},
        {"architecture": "topk", "val_mean_k_eval": 33, "val_hard_recon_mse": 0.1},
    ]
    selected = select_l0_bins(rows, targets=[32])
    assert selected[0]["status"] == "selected"
    assert selected[0]["val_hard_recon_mse"] == 0.1
