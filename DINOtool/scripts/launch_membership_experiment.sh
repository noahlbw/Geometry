#!/usr/bin/env bash
set -euo pipefail
BASE=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
PROJECT="$BASE/code/DINOtool"
PYTHON="$BASE/envs/tessera-cu128/bin/python"
OFFICIAL="$BASE/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
RUN="${1:?Provide a fresh experiment directory}"
if [[ -e "$RUN" ]]; then
  echo "Refusing existing experiment: $RUN" >&2
  exit 2
fi
mkdir -p "$RUN"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$PROJECT:$PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}"
trap 'printf "%s\n" "$?" > "$RUN/suite.exit_code"' EXIT
DDP=("$PYTHON" -m torch.distributed.run --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0)
BASE_CKPT="$BASE/code/ckpt/CAFe-DINO/weights.pth"
BPE="$BASE/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz"
SOURCE="$BASE/datasets/OpenEarthMap_wo_xBD"
PLAIN="$PROJECT/results/cafe_rc/20260912_cafe_baseline_oem_8gpu_v1/best_inference.pt"
RC="$PROJECT/results/cafe_rc/20260912_cafe_rc_oem_8gpu_v1/best_inference.pt"
TRAIN=(--official-root "$OFFICIAL" --checkpoint "$BASE_CKPT" --bpe-path "$BPE" --data-root "$SOURCE"
       --allow-missing-source-images --epochs 30 --crop-size 448 --batch-size 2 --accum-steps 2
       --workers 4 --amp bf16 --memory-fraction 0.65 --seed 20260912)
EVAL=(--official-root "$OFFICIAL" --base-checkpoint "$BASE_CKPT" --bpe-path "$BPE" --amp bf16)
stage() {
  local label="$1"
  shift
  printf '%s\n' "$label" > "$RUN/current_stage.txt"
  printf 'Starting %s\n' "$label"
  local status=0
  "$@" > "$RUN/$label.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$RUN/$label.exit_code"
  if [[ "$status" != 0 ]]; then
    echo "Failed stage: $label; inspect $RUN/$label.log" >&2
    return "$status"
  fi
}
stage tests "$PYTHON" -m pytest -q tests/test_cafe_rc.py tests/test_cafe_membership.py
stage source_frozen "${DDP[@]}" scripts/audit_cafe_source_errors.py "${EVAL[@]}" --weights "$PLAIN" --frozen --data-root "$SOURCE" --allow-missing-source-images --output-dir "$RUN/source_frozen"
stage source_plain "${DDP[@]}" scripts/audit_cafe_source_errors.py "${EVAL[@]}" --weights "$PLAIN" --data-root "$SOURCE" --allow-missing-source-images --output-dir "$RUN/source_plain"
stage source_rc "${DDP[@]}" scripts/audit_cafe_source_errors.py "${EVAL[@]}" --weights "$RC" --data-root "$SOURCE" --allow-missing-source-images --output-dir "$RUN/source_rc"
stage loveda_trained_plain "${DDP[@]}" scripts/eval_cafe_rc_loveda.py "${EVAL[@]}" --weights "$PLAIN" --data-root "$BASE/datasets/LoveDA/val" --output-dir "$RUN/loveda_trained_plain"
for variant in mask_plain mask_dual mask_fixed; do
  stage "smoke_$variant" "${DDP[@]}" scripts/train_cafe_rc.py "${TRAIN[@]}" --variant "$variant" --output-dir "$RUN/smoke_$variant" --smoke --workers 0 --log-every 1
done
for variant in mask_plain mask_dual mask_fixed; do
  stage "train_$variant" "${DDP[@]}" scripts/train_cafe_rc.py "${TRAIN[@]}" --variant "$variant" --output-dir "$RUN/train_$variant"
  stage "audit_$variant" "${DDP[@]}" scripts/audit_cafe_source_errors.py "${EVAL[@]}" --weights "$RUN/train_$variant/best_inference.pt" --data-root "$SOURCE" --allow-missing-source-images --output-dir "$RUN/audit_$variant"
done
printf 'complete\n' > "$RUN/current_stage.txt"
