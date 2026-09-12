#!/usr/bin/env bash
#
# Download the published artifacts (checkpoints, cached activations, frozen
# backbone) from Hugging Face and place them where the sweep expects them.
#
#   bash scripts/prepare_artifacts.sh              # -> ./artifacts
#   bash scripts/prepare_artifacts.sh /data/sae    # custom destination
#
set -euo pipefail

cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-antoniotorres02/vartopk-sae-stage4-checkpoints}"
DEST="${1:-artifacts}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-outputs/convnext_stage4_geomspace_sweep}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' CLI (huggingface_hub) is required: pip install huggingface_hub" >&2
  exit 1
fi

hf download "$HF_REPO" --local-dir "$DEST"

# The sweep looks for the Stage-4 cache under its own run directory with a name
# derived from the split sizes; mirror the downloaded file there so the
# feature-extraction pass is skipped.
mkdir -p "$RUN_OUTPUT_DIR/features"
CACHE_TARGET="$RUN_OUTPUT_DIR/features/stage4_maps_train512_val512_test512.pt"
if [[ ! -f "$CACHE_TARGET" ]]; then
  cp "$DEST/features/stage4_maps_subset512.pt" "$CACHE_TARGET"
fi

cat <<EOF

Artifacts downloaded to: $DEST
Stage-4 cache linked at: $CACHE_TARGET

Set these keys in config/external_paths.toml:
  convnext_checkpoint = "$DEST/backbone/cub_convnext_tiny_classifier.pt"
  cub_feature_cache   = "$DEST/features/stage4_maps_subset512.pt"

Verify the published checkpoints against the recorded results with:
  python scripts/verify_published_checkpoints.py --checkpoints-root "$DEST/checkpoints" --sha256sums "$DEST/SHA256SUMS"
EOF
