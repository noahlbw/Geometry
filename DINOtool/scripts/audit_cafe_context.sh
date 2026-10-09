#!/usr/bin/env bash
set -euo pipefail
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
for setting in window224 whole512; do
  window=224
  stride=112
  [[ "$setting" != whole512 ]] || { window=512; stride=512; }
  /data/miniconda3/envs/pfu/bin/torchrun --standalone --nproc_per_node=8 \
    scripts/audit_region_assembly_ade.py --variants adapted --images 64 \
    --window "$window" --stride "$stride" \
    --weights "$bundle/results/cafe_semantic_repair_20260921/best_inference.pt" \
    --official-root "$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c" \
    --base-checkpoint "$bundle/ckpt/CAFe-DINO/weights.pth" \
    --bpe-path "$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz" \
    --data-root /data/test/cafe-efa/data/processed/ade20k \
    --output-dir "$bundle/evaluation/ade_context_${setting}_20260921" \
    > "$bundle/evaluation/ade_context_${setting}_20260921.log" 2>&1
done
