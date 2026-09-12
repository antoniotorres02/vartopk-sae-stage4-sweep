"""Command-line interface for the clean SAE evaluation package."""

from __future__ import annotations

import argparse
from pathlib import Path

from tfm_sae_evals.config import ExternalPaths
from tfm_sae_evals.experiments.convnext_stage4_compare import add_parser as add_convnext_stage4_parser
from tfm_sae_evals.experiments.convnext_stage4_compare_only_topk_versus_topk import (
    add_parser as add_convnext_stage4_only_topk_parser,
)
from tfm_sae_evals.experiments.explainability import explainability_delta_table
from tfm_sae_evals.experiments.fair import fair_winner_table
from tfm_sae_evals.experiments.highk import highk_pair_table
from tfm_sae_evals.experiments.results import summarize_results


def cmd_check_paths(args: argparse.Namespace) -> int:
    paths = ExternalPaths.from_toml(args.config)
    missing = paths.missing()
    if missing:
        for name, path in missing.items():
            print(f"MISSING {name}: {path}")
        return 1
    print("All external paths exist.")
    return 0


def cmd_summarize_results(args: argparse.Namespace) -> int:
    summary = summarize_results(args.results_dir)
    print(summary.to_string(index=False))
    return 0


def cmd_fair(args: argparse.Namespace) -> int:
    table = fair_winner_table(args.results_dir)
    print(table.to_string(index=False))
    return 0


def cmd_highk(args: argparse.Namespace) -> int:
    table = highk_pair_table(args.results_dir)
    print(table.to_string(index=False))
    return 0


def cmd_explainability(args: argparse.Namespace) -> int:
    table = explainability_delta_table(args.results_dir, run_name=args.run_name)
    print(table.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sae-evals")
    sub = parser.add_subparsers(dest="command", required=True)

    check_paths = sub.add_parser("check-paths", help="Validate external data/checkpoint paths.")
    check_paths.add_argument("--config", type=Path, default=Path("config/external_paths.example.toml"))
    check_paths.set_defaults(func=cmd_check_paths)

    summarize = sub.add_parser("summarize-results", help="Summarize copied lightweight result artifacts.")
    summarize.add_argument("--results-dir", type=Path, default=Path("results"))
    summarize.set_defaults(func=cmd_summarize_results)

    fair = sub.add_parser("fair", help="Print the fair SAE aggregate winner table.")
    fair.add_argument("--results-dir", type=Path, default=Path("results"))
    fair.set_defaults(func=cmd_fair)

    highk = sub.add_parser("highk", help="Print the high-K convergence diagnostic table.")
    highk.add_argument("--results-dir", type=Path, default=Path("results"))
    highk.set_defaults(func=cmd_highk)

    explainability = sub.add_parser("explainability", help="Print Variable TopK vs TopK explainability deltas.")
    explainability.add_argument("--results-dir", type=Path, default=Path("results"))
    explainability.add_argument("--run-name", default="variable_vs_topk_explainability")
    explainability.set_defaults(func=cmd_explainability)

    add_convnext_stage4_parser(sub)
    add_convnext_stage4_only_topk_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
