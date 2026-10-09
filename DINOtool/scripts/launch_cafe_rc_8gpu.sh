#!/usr/bin/env bash
set -euo pipefail
BASE=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
PROJECT="$BASE/code/DINOtool"
PYTHON="$BASE/envs/tessera-cu128/bin/python"
OFFICIAL="$BASE/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
MODE="${1:-train}"
RUN="${2:?Provide a fresh run directory}"
case "$MODE" in
  smoke) EXTRA=(--smoke --workers 0 --log-every 1) ;;
  train) EXTRA=() ;;
  *) echo "mode must be train or smoke" >&2; exit 2 ;;
esac
if [[ -e "$RUN" ]]; then
  echo "Run directory already exists: $RUN" >&2
  exit 2
fi
mkdir -p "$(dirname "$RUN")"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
set +e
"$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  scripts/train_cafe_rc.py --official-root "$OFFICIAL" --checkpoint "$BASE/code/ckpt/CAFe-DINO/weights.pth" \
  --bpe-path "$BASE/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz" \
  --data-root "$BASE/datasets/OpenEarthMap_wo_xBD" --output-dir "$RUN" \
  --allow-missing-source-images --epochs 30 --crop-size 448 --batch-size 2 --accum-steps 2 \
  --workers 4 --amp bf16 --memory-fraction 0.65 --variant regional "${EXTRA[@]}" \
  > "${RUN}.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" > "${RUN}.exit_code"
exit "$STATUS"
