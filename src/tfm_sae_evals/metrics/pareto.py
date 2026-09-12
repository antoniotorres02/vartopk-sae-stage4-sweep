"""Pareto and L0-target selection helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable


def finite_float(value: object, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def l0_tolerance(target: float, relative: float = 0.10, absolute: float = 16.0) -> float:
    return max(float(absolute), abs(float(target)) * float(relative))


def pareto_frontier(rows: Iterable[dict[str, object]], *, x_key: str, y_key: str) -> list[dict[str, object]]:
    valid_rows = [
        row for row in rows
        if math.isfinite(finite_float(row.get(x_key))) and math.isfinite(finite_float(row.get(y_key)))
    ]
    sorted_rows = sorted(valid_rows, key=lambda row: (finite_float(row.get(x_key)), finite_float(row.get(y_key))))
    frontier = []
    best_y = math.inf
    for row in sorted_rows:
        y_value = finite_float(row.get(y_key))
        if y_value < best_y:
            frontier.append(dict(row))
            best_y = y_value
    return frontier


def select_l0_bins(
    rows: Iterable[dict[str, object]],
    *,
    targets: Iterable[float],
    architecture_key: str = "architecture",
    l0_key: str = "val_mean_k_eval",
    score_key: str = "val_hard_recon_mse",
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        architecture = str(row.get(architecture_key, ""))
        if architecture:
            grouped[architecture].append(row)

    selections = []
    for architecture, arch_rows in sorted(grouped.items()):
        for target in targets:
            tolerance = l0_tolerance(float(target))
            candidates = []
            for row in arch_rows:
                l0_value = finite_float(row.get(l0_key))
                score_value = finite_float(row.get(score_key))
                if math.isfinite(l0_value) and math.isfinite(score_value) and abs(l0_value - float(target)) <= tolerance:
                    candidates.append((score_value, abs(l0_value - float(target)), row))
            if not candidates:
                selections.append({"architecture": architecture, "target_l0": target, "tolerance": tolerance, "status": "missing"})
                continue
            _score, delta, selected = min(candidates, key=lambda item: (item[0], item[1]))
            out = dict(selected)
            out.update({"target_l0": target, "tolerance": tolerance, "l0_delta": delta, "status": "selected"})
            selections.append(out)
    return selections
