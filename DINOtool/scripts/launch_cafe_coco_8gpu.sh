#!/usr/bin/env bash
# Launch one matched CAFe COCO arm. Run prepare_coco_stuff.py first.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <plain|cost_only|concat|vc> <smoke|screen|full> <fresh-output-dir>" >&2
  exit 2
fi

arm=$1
mode=$2
output=$3
base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data=$base/datasets/COCOStuff2017
official=$base/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
python=$base/envs/tessera-cu128/bin/python
checkpoint=$base/code/ckpt/CAFe-DINO/weights.pth
bpe=$base/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz

case "$arm" in plain|cost_only|concat|vc) ;; *) echo "invalid arm: $arm" >&2; exit 2 ;; esac
case "$mode" in smoke) updates=0; smoke=(--smoke) ;; screen) updates=1000; smoke=() ;; full) updates=0; smoke=() ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac

[[ -f "$data/manifests/train2017_cafe41.json" ]] || { echo "Missing prepared COCO train manifest" >&2; exit 1; }
[[ -f "$data/manifests/val2017_source_dev.json" ]] || { echo "Missing prepared COCO source-dev manifest" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "Output must be fresh: $output" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$project:${PYTHONPATH:-}"

args=(
  "$project/scripts/train_cafe_coco.py"
  --official-root "$official"
  --checkpoint "$checkpoint"
  --bpe-path "$bpe"
  --data-root "$data"
  --output-dir "$output"
  --arm "$arm"
  --epochs 3
  --max-updates "$updates"
  --crop-size 224
  --batch-size 2
  --accum-steps 2
  --workers 4
  --lr 2e-4
  --cafe-lr 2e-5
  --weight-decay 0.01
  --warmup-steps 100
  --label-smoothing 0.1
  --content-dim 128
  --heads 4
  --kernel-size 7
  --blocks 2
  --amp bf16
  --seed 20260912
  --memory-fraction 0.70
  "${smoke[@]}"
)

mkdir -p "$(dirname "$output")"
nice -n 10 "$python" -m torch.distributed.run --standalone --nproc_per_node=8 "${args[@]}"
