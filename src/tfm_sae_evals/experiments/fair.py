"""Fair SAE evaluation summary helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fair_aggregate(results_root: str | Path) -> pd.DataFrame:
    path = (
        Path(results_root)
        / "fair-sae-methodology-eval/research/sae_cub/cnn_feature_sae/sweep_results/fair_eval_full/"
        / "fair_eval_full_aggregate_selected_summary.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing fair aggregate summary: {path}")
    return pd.read_csv(path)


def fair_winner_table(results_root: str | Path) -> pd.DataFrame:
    df = load_fair_aggregate(results_root)
    columns = [
        col for col in [
            "target_l0",
            "architecture",
            "mean_test_l0",
            "mean_test_mse",
            "std_test_mse",
            "n_seeds",
        ]
        if col in df
    ]
    return df[columns] if columns else df
