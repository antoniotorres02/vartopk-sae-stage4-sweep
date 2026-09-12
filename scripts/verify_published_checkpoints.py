#!/usr/bin/env python3
"""Verify that published checkpoints match the recorded sweep results.

The sweep writes ``stage4_arch_compare_results.csv``; every checkpoint embeds the
evaluation metrics that produced its row. This script re-reads both and fails if
they disagree, which is the cheapest way for a reader to confirm that the files
published on Hugging Face are the ones behind the reported table.

Expected layout (mirrors the Hugging Face repository)::

    checkpoints/geomspace_sweep/stage4_*.pt   (40 files)
    checkpoints/logspace_reuse/stage4_*.pt    (4 files, targets 8 and 256)
    results/convnext_stage4_geomspace_sweep/stage4_arch_compare_results.csv

Usage::

    python scripts/verify_published_checkpoints.py
    python scripts/verify_published_checkpoints.py --checkpoints-root /path/to/hf/snapshot/checkpoints
    python scripts/verify_published_checkpoints.py --sha256sums path/to/SHA256SUMS

Exits with status 0 when every row matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import torch

METRIC_TOLERANCE = 1e-6
# CSV column -> key inside the checkpoint's ``eval_metrics`` dictionary.
METRIC_FIELDS = {
    "test_mse": "test_mse",
    "test_nmse": "test_nmse",
    "l0": "l0",
    "top1_agreement": "top1_agreement",
    "reconstructed_top1_accuracy": "reconstructed_top1_accuracy",
    "logit_kl": "logit_kl",
}


def sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_checkpoint(root: Path, filename: str) -> Path | None:
    """Locate ``filename`` anywhere under ``root`` (checkpoint folders are flat)."""
    direct = root / filename
    if direct.exists():
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the sweep CSV, keeping only successfully completed rows."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"No completed rows found in {csv_path}")
    return rows


def verify_row(row: dict[str, str], root: Path) -> tuple[bool, str]:
    """Compare one CSV row against the metrics embedded in its checkpoint."""
    filename = Path(str(row.get("checkpoint_path", ""))).name
    if not filename:
        return False, "row has no checkpoint_path"
    path = find_checkpoint(root, filename)
    if path is None:
        return False, f"{filename}: not found under {root}"

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("eval_metrics") or {}
    method = checkpoint.get("method")
    if method != row.get("method"):
        return False, f"{filename}: method {method!r} != row {row.get('method')!r}"

    mismatches = []
    for column, key in METRIC_FIELDS.items():
        recorded = row.get(column)
        if recorded in (None, ""):
            continue
        embedded = metrics.get(key)
        if embedded is None:
            mismatches.append(f"{column}: missing in checkpoint")
            continue
        if abs(float(embedded) - float(recorded)) > METRIC_TOLERANCE:
            mismatches.append(f"{column}: checkpoint {float(embedded):.10g} != csv {float(recorded):.10g}")
    if mismatches:
        return False, f"{filename}: " + "; ".join(mismatches)
    return True, f"{filename}: ok"


def verify_sha256sums(sums_path: Path, root: Path) -> tuple[int, int]:
    """Check every entry of a ``sha256sum``-style manifest against the files."""
    ok = bad = 0
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, raw_name = line.partition("  ")
        name = Path(raw_name.strip().lstrip("*"))
        target = root.parent / name if (root.parent / name).exists() else find_checkpoint(root, name.name)
        if target is None:
            print(f"  MISSING  {name}")
            bad += 1
            continue
        if sha256(target) == digest:
            ok += 1
        else:
            print(f"  MISMATCH {name}")
            bad += 1
    return ok, bad


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/convnext_stage4_geomspace_sweep/stage4_arch_compare_results.csv"),
    )
    parser.add_argument("--sha256sums", type=Path, default=None, help="Optional SHA256SUMS manifest to verify.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.csv)
    failures = 0
    for row in rows:
        ok, message = verify_row(row, args.checkpoints_root)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {row['method']:26s} target={row.get('k') or row.get('target_l0'):>4s}  {message}")
        failures += 0 if ok else 1
    print(f"\n{len(rows) - failures}/{len(rows)} rows match their checkpoint metrics.")

    if args.sha256sums is not None and args.sha256sums.exists():
        ok, bad = verify_sha256sums(args.sha256sums, args.checkpoints_root)
        print(f"{ok} file digests verified, {bad} failed.")
        failures += bad

    if failures:
        print("Verification FAILED.")
        return 1
    print("Verification OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
