#!/usr/bin/env bash
set -euo pipefail
protocol=${1:?Pass P or D}
gpus=${2:?Pass four GPU IDs}
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
output=$bundle/evaluation/semantic_repair_loveda_${protocol,,}_full_20260921
[[ ! -e "$output" ]] || { echo "Existing evaluation must be preserved" >&2; exit 2; }
cd "$project"
export CUDA_VISIBLE_DEVICES=$gpus
export OMP_NUM_THREADS=4
/data/miniconda3/envs/pfu/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/eval_cafe_vc_protocols.py \
  --weights "$bundle/results/cafe_semantic_repair_20260921/best_inference.pt" \
  --official-root "$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c" \
  --base-checkpoint "$bundle/ckpt/CAFe-DINO/weights.pth" \
  --bpe-path "$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz" \
  --data-root /data/test/cafe-efa/data/processed/loveda/val \
  --protocol "$protocol" --output-dir "$output" --amp bf16 --memory-fraction 0.4 \
  > "$output.log" 2>&1
