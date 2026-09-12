"""High-K evaluation summary helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_highk_diagnostics(results_root: str | Path) -> pd.DataFrame:
    path = (
        Path(results_root)
        / "main/research/sae_cub/cnn_feature_sae/convergence_highk/artifacts/diagnostics/"
        / "diag_convergence_summary.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing high-K diagnostic summary: {path}")
    return pd.read_csv(path)


def highk_pair_table(results_root: str | Path) -> pd.DataFrame:
    df = load_highk_diagnostics(results_root)
    preferred = [
        "pair_id",
        "L0_target",
        "L0_v_actual",
        "L0_t_actual",
        "MSE_v_test",
        "MSE_t_test",
        "delta_pct",
        "VTK_wins",
        "VTK_n_dead_ever",
        "TopK_n_dead_ever",
    ]
    columns = [col for col in preferred if col in df]
    return df[columns] if columns else df
