#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${DINO_CODE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
DATA_ROOT="${DINO_OEM_DATA_ROOT:-${CODE_ROOT}/../datasets/OpenEarthMap_wo_xBD}"
OUTPUT_DIR="${DINO_UOT_OUTPUT_DIR:-${CODE_ROOT}/../outputs/uot_full}"
CHECKPOINT_DIR="${DINO_CHECKPOINT_DIR:-${CODE_ROOT}/ckpt/DINO}"
PYTHON_BIN="${DINO_PYTHON:-python}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m dinotool.cli train-oem-uot \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${DINO_UOT_EPOCHS:-30}" \
  --crop-size "${DINO_UOT_CROP_SIZE:-512}" \
  --batch-size "${DINO_UOT_BATCH_SIZE:-8}" \
  --num-workers "${DINO_UOT_NUM_WORKERS:-8}" \
  --train-last-visual-blocks "${DINO_UOT_TRAIN_LAST_BLOCKS:-2}" \
  --uot-num-modes "${DINO_UOT_NUM_MODES:-4}" \
  --uot-sinkhorn-iterations "${DINO_UOT_SINKHORN_ITERATIONS:-15}" \
  --allow-missing-source-images \
  --device "${DINO_DEVICE:-cuda}" \
  --code-root "${CODE_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"
