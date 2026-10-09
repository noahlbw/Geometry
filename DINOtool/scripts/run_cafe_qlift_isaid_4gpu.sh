#!/usr/bin/env bash
# Evaluate all four source-selected Q-Lift arms on prepared iSAID validation.
# This is target-only evaluation: it does not train or choose a target checkpoint.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GATE_RUN_ROOT ISAID_EXTRACT_ROOT EVALUATION_ROOT" >&2
  exit 2
fi

gate_root=$1
isaid_root=$2
evaluation_root=$3
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
target_root=$(dirname "$isaid_root")

[[ -d $gate_root && -d $isaid_root && ! -e $evaluation_root ]] || {
  echo "gate root and prepared iSAID extraction must exist; evaluation root must be fresh" >&2
  exit 2
}
[[ -f $target_root/DOWNLOAD_COMPLETE && -f $target_root/EXTRACT_COMPLETE ]] || {
  echo "iSAID root lacks completed download/extraction provenance" >&2
  exit 2
}
[[ -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
  echo "missing locked code, environment, or CAFe artifacts" >&2
  exit 2
}

for arm in skip haar image_lift query_lift; do
  "$python" "$project/scripts/verify_cafe_qlift_gate_contract.py" \
      --run-root "$gate_root/$arm" --arm "$arm"
done

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

for arm in skip haar image_lift query_lift; do
  weights=$gate_root/$arm/best_inference.pt
  [[ -f $weights ]] || { echo "Missing source-selected checkpoint: $weights" >&2; exit 1; }
  "$python" -m torch.distributed.run \
      --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
      "$project/scripts/eval_cafe_qlift_isaid.py" \
      --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
      --data-root "$isaid_root" --weights "$weights" --output-dir "$evaluation_root/$arm" \
      --amp bf16 --memory-fraction 0.75
done
