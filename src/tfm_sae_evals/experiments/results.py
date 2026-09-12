"""Lightweight result archive helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results_manifest(results_dir: str | Path) -> pd.DataFrame:
    root = Path(results_dir)
    manifest = root.parent / "docs" / "results_manifest.csv"
    if not manifest.exists():
        manifest = root / "results_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Could not find results manifest near {root}")
    return pd.read_csv(manifest)


def summarize_results(results_dir: str | Path) -> pd.DataFrame:
    manifest = load_results_manifest(results_dir)
    return (
        manifest.groupby(["source_worktree", "category", "extension"], dropna=False)
        .agg(files=("target_path", "count"), bytes=("size_bytes", "sum"))
        .reset_index()
        .sort_values(["source_worktree", "category", "extension"])
    )
