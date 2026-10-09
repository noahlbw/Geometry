#!/usr/bin/env bash
set -euo pipefail
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
output=$bundle/results/${REPAIR_RUN_NAME:-cafe_semantic_repair_20260921}
[[ ! -e "$output" ]] || { echo "Existing repair run must be preserved" >&2; exit 2; }
cd "$project"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
/data/miniconda3/envs/pfu/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/train_cafe_semantic_repair.py \
  --weights "$bundle/results/cafe_region_assembly_20260920/full_4gpu/best_inference.pt" \
  --official-root "$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c" \
  --base-checkpoint "$bundle/ckpt/CAFe-DINO/weights.pth" \
  --bpe-path "$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz" \
  --coco-root /data/test/datasets/COCOStuff2017 --oem-root /data/test/datasets/OpenEarthMap_wo_xBD \
  --output-dir "$output" --updates 400 --validate-every 200 \
  --source-coco-images 512 --source-oem-images 64 --batch-size 1 --accum-steps 2 \
  --proposal-weight "${REPAIR_PROPOSAL_WEIGHT:-0}" \
  > "$output.log" 2>&1
