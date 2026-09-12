"""Explainability evaluation summary helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_explainability_summary(results_root: str | Path, run_name: str = "variable_vs_topk_explainability") -> pd.DataFrame:
    path = (
        Path(results_root)
        / "vtk-topk-explainability-eval/research/sae_cub/cnn_feature_sae/eval_results"
        / run_name
        / "variable_topk_vs_matched_topk_summary.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing explainability summary: {path}")
    return pd.read_csv(path)


def explainability_delta_table(results_root: str | Path, run_name: str = "variable_vs_topk_explainability") -> pd.DataFrame:
    df = load_explainability_summary(results_root, run_name=run_name)
    preferred = [
        "variable_checkpoint",
        "topk_checkpoint",
        "variable_l0",
        "topk_l0",
        "delta_hard_recon_mse_positive_variable_better",
        "delta_head_agreement_positive_variable_better",
        "delta_mean_class_purity_at_n_positive_variable_better",
        "delta_mean_best_attr_auc_positive_variable_better",
    ]
    columns = [col for col in preferred if col in df]
    return df[columns] if columns else df
