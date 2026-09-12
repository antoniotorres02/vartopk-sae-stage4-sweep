#!/usr/bin/env bash
#
# Reproduce the ConvNeXt Stage4 SAE sweep (44 runs: 4 architecture families x 11
# nominal targets) reported in the TFM memory.
#
#   bash run_sweep.sh                 # full 44-run campaign
#   bash run_sweep.sh --dry-list      # only print the target list the CLI will use
#
# Requires:
#   - config/external_paths.toml (copy of config/external_paths.example.toml)
#     pointing to CUB and to the ConvNeXt-Tiny CUB classifier.
#   - a CUDA device for the timings reported in the memory (~3.3 h in total);
#     add --cpu to the command below to run on CPU.
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-config/external_paths.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/convnext_stage4_geomspace_sweep}"
RESULTS_DIR="${RESULTS_DIR:-results/convnext_stage4_geomspace_sweep}"
SWEEP_POINTS="${SWEEP_POINTS:-11}"

if [[ "${1:-}" == "--dry-list" ]]; then
  "$PYTHON" - <<'PY'
import math

targets = [round(8 * math.pow(256 / 8, i / 10)) for i in range(11)]
print("nominal targets:", targets)
assert targets == [8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]
PY
  exit 0
fi

export PYTHONPATH="${PYTHONPATH:-src}"

"$PYTHON" -m tfm_sae_evals.cli.main convnext-stage4-compare \
  --config "$CONFIG" \
  --fresh \
  --methods topk_nonorm variable_topk_original l1_sae jumprelu_sae \
  --subset-size 512 \
  --epochs 300 \
  --sweep-logspace \
  --sweep-min-k 8 \
  --sweep-max-k 256 \
  --sweep-points "$SWEEP_POINTS" \
  --vtk-kmax-factor 1.0 \
  --vtk-lambda-prior 0.003 \
  --vtk-beta 0.0003 \
  --vtk-hard-weight 0.0 \
  --vtk-expected-weight 1.0 \
  --vtk-budget-weight 0.0 \
  --sparse-hyperparams-csv config/sparse_hyperparameters.csv \
  --output-dir "$OUTPUT_DIR" \
  --results-dir "$RESULTS_DIR"
