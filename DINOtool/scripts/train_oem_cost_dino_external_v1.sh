#!/usr/bin/env bash
set -euo pipefail

# This run is deliberately external-data-only. LoveDA is not read by this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DATA_ROOT="${DINO_OEM_DATA_ROOT:-${CODE_ROOT}/../datasets/OpenEarthMap_wo_xBD}"
OUTPUT_DIR="${DINO_COST_OUTPUT_DIR:-${CODE_ROOT}/../results/DINOv3Cost_OEM_external_v1}"

exec "${PROJECT_ROOT}/.venv/bin/dino-ovss" train-oem-cost-dino \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
  --crop-size 512 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 2e-4 \
  --visual-lr 1e-5 \
  --warmup-steps 300 \
  --cost-hidden-dim 128 \
  --cost-context-blocks 6 \
  --cost-attention-heads 8 \
  --cost-window-size 7 \
  --cost-dropout 0.1 \
  --train-last-visual-blocks 2 \
  --class-balance-power 0.5 \
  --open-vocabulary-preservation-weight 0.15 \
  --ignored-cost-preservation-weight 0.1 \
  --allow-missing-source-images
