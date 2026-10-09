#!/usr/bin/env bash
# Post-hoc mechanism audit for the fixed source-selected query_lift checkpoint.
# It never trains, selects checkpoints, or reads a target metric before running
# all predeclared P/U and detail interventions.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 GATE_RUN_ROOT AUDIT_OUTPUT_ROOT" >&2
    exit 2
fi

gate_root=$1
audit_root=$2
evaluation_root=$(dirname "$audit_root")
native_root=$evaluation_root/native
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
loveda=/data/test/cafe-efa/data/processed/loveda/val
query_root=$gate_root/query_lift

[[ -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe && -d $loveda && -d $native_root ]] || {
    echo "missing Q-Lift code, frozen CAFe artifact, locked LoveDA validation root, or completed native references" >&2
    exit 2
}
[[ -f $query_root/best_inference.pt && -f $query_root/status.json && ! -e $query_root/failure.json && ! -e $audit_root ]] || {
    echo "query_lift source-selected checkpoint is missing, failed, or audit output is not fresh" >&2
    exit 2
}
"$python" "$project/scripts/verify_cafe_qlift_gate_contract.py" --run-root "$query_root" --arm query_lift

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$audit_root"
for protocol in P D; do
    for mode in pu-shuffled pu-image-only details-zero; do
        output=$audit_root/$protocol/$mode
        "$python" -m torch.distributed.run \
            --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
            "$project/scripts/eval_cafe_vc_protocols.py" \
            --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
            --data-root "$loveda" --weights "$query_root/best_inference.pt" \
            --native-reference-dir "$native_root/$protocol/query_lift" \
            --protocol "$protocol" --qlift-audit "$mode" --output-dir "$output" \
            --amp bf16 --memory-fraction 0.75
    done
done
