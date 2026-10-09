#!/usr/bin/env bash
set -euo pipefail
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
run=${REPAIR_RUN_NAME:-cafe_semantic_repair_20260921}
output=$bundle/evaluation/${EVAL_RUN_NAME:-ade150_semantic_repair_full_20260921}
[[ ! -e "$output" ]] || { echo "Existing evaluation must be preserved" >&2; exit 2; }
cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
/data/miniconda3/envs/pfu/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/eval_cafe_region_assembly_ade20k.py \
  --weights "$bundle/results/$run/best_inference.pt" \
  --official-root "$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c" \
  --base-checkpoint "$bundle/ckpt/CAFe-DINO/weights.pth" \
  --bpe-path "$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz" \
  --data-root /data/test/cafe-efa/data/processed/ade20k --output-dir "$output" \
  --amp bf16 --memory-fraction 0.35 > "$output.log" 2>&1
