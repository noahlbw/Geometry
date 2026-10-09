#!/usr/bin/env bash
# Pre-registered external LoveDA P/D evaluation for a completed Q-Lift gate.
#
# This script evaluates every capacity-matched arm with its source-selected
# checkpoint. It never reads a target metric to choose an arm or checkpoint.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 GATE_RUN_ROOT EVALUATION_ROOT" >&2
    exit 2
fi

gate_root=$1
evaluation_root=$2
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
loveda=/data/test/cafe-efa/data/processed/loveda/val

[[ -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe && -d $loveda ]] || {
    echo "missing Q-Lift code, frozen CAFe artifact, or locked LoveDA validation root" >&2
    exit 2
}
[[ -d $gate_root && ! -e $evaluation_root ]] || {
    echo "gate root must exist and evaluation root must be fresh" >&2
    exit 2
}

for arm in skip haar image_lift query_lift; do
    arm_root=$gate_root/$arm
    [[ -f $arm_root/best_inference.pt && -f $arm_root/status.json && ! -e $arm_root/failure.json ]] || {
        echo "missing selected checkpoint or failed source run for $arm" >&2
        exit 2
    }
    "$python" "$project/scripts/verify_cafe_qlift_gate_contract.py" --run-root "$arm_root" --arm "$arm"
done

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$evaluation_root"
for protocol in P D; do
    for arm in skip haar image_lift query_lift; do
        output=$evaluation_root/$protocol/$arm
        "$python" -m torch.distributed.run \
            --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
            "$project/scripts/eval_cafe_vc_protocols.py" \
            --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
            --data-root "$loveda" --weights "$gate_root/$arm/best_inference.pt" \
            --protocol "$protocol" --output-dir "$output" --amp bf16 --memory-fraction 0.75
    done
done
