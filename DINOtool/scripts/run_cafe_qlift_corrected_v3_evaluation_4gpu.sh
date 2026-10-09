#!/usr/bin/env bash
# Fixed post-gate sequence for the corrected Q-Lift v3 source contract.
# This launcher never trains, chooses a checkpoint from LoveDA, or evaluates
# a subset of arms.  It is intentionally not called by the source trainer.
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

[[ -d $gate_root && ! -e $evaluation_root ]] || {
    echo "gate root must exist and evaluation root must be fresh" >&2
    exit 2
}
[[ -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe && -d $loveda ]] || {
    echo "missing locked code, environment, CAFe artifacts, or LoveDA root" >&2
    exit 2
}

# This is the sole source-contract gate.  It rejects the historical 2,612-step
# compressed cosine run, before any target files are created.
for arm in skip haar image_lift query_lift; do
    "$python" "$project/scripts/verify_cafe_qlift_gate_contract.py" \
        --run-root "$gate_root/$arm" --arm "$arm"
done

# The native launcher evaluates P and D for every arm in a fixed order and
# creates the provenance/digest bindings required by the spatial replay.
bash "$project/scripts/run_cafe_qlift_loveda_4gpu.sh" "$gate_root" "$evaluation_root/native"

# These are inference-only interventions of the already source-selected
# query_lift checkpoint; they are never target-selected.
bash "$project/scripts/run_cafe_qlift_mechanism_audit_4gpu.sh" "$gate_root" "$evaluation_root/mechanism"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

for protocol in P D; do
    for arm in skip haar image_lift query_lift; do
        "$python" -m torch.distributed.run \
            --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
            "$project/scripts/audit_cafe_qlift_loveda_spatial.py" \
            --reference-evaluation-dir "$evaluation_root/native/$protocol/$arm" \
            --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
            --data-root "$loveda" --audit-output-dir "$evaluation_root/spatial/$protocol/$arm" \
            --memory-fraction 0.75
    done
done

"$python" "$project/scripts/analyze_cafe_qlift_gate.py" \
    --gate-root "$gate_root" --loveda-root "$evaluation_root/native" \
    --mechanism-root "$evaluation_root/mechanism" --spatial-root "$evaluation_root/spatial" \
    --output-dir "$evaluation_root/report"
