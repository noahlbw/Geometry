#!/usr/bin/env bash
# COCO-only training followed by fixed, post-hoc LoveDA P/D evaluation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 FRESH_OUTPUT_ROOT" >&2
    exit 2
fi

output_root=$1
bundle=${OVSS_BUNDLE:-/data/test/code/ovss_cafe_ped_v2_20260919}
project=${OVSS_PROJECT:-$bundle/DINOtool_query_reconstruction_20260926}
python=${PYTHON_BIN:-/data/miniconda3/envs/pfu/bin/python}
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=${COCO_ROOT:-/data/test/datasets/COCOStuff2017}
loveda=${LOVEDA_ROOT:-/data/test/cafe-efa/data/processed/loveda/val}

[[ ! -e $output_root && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "Output must be fresh and pinned CAFe artifacts must exist" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $coco/manifests/val2017_source_dev.json ]] || {
    echo "Locked COCO-Stuff manifests are missing" >&2
    exit 2
}
[[ -d $loveda/images && -d $loveda/masks ]] || {
    echo "Locked LoveDA validation root is missing" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$output_root"
exec > >(tee -a "$output_root/pipeline.log") 2>&1

stage() {
    printf '%s\n' "$1" > "$output_root/stage.txt"
    printf '%s %s\n' "$(date -Is)" "$1"
}

train() {
    local destination=$1
    shift
    "$python" -m torch.distributed.run \
        --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_query_reconstruction.py" \
        --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
        --data-root "$coco" --output-dir "$destination" \
        --epochs 8 --max-updates 20896 --crop-size 224 \
        --batch-size 2 --accum-steps 4 --workers 4 \
        --lr 1e-4 --cafe-lr 2e-5 --weight-decay 0.01 --warmup-steps 500 \
        --kd-weight 0.02 --label-smoothing 0.1 \
        --relation-dim 32 --spatial-weight 0.05 --class-weight 0.05 \
        --solver-steps 4 --solver-step-size 0.7017543859649122 --relation-weight 0.05 \
        --guide-dim 16 --residual-scale 0.10 --consistency-weight 0.05 \
        --amp bf16 --memory-fraction 0.78 --seed 20260926 "$@"
}

evaluate() {
    local protocol=$1
    "$python" -m torch.distributed.run \
        --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/eval_cafe_vc_protocols.py" \
        --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
        --data-root "$loveda" --weights "$output_root/full/best_inference.pt" \
        --protocol "$protocol" --output-dir "$output_root/loveda/$protocol" \
        --amp bf16 --memory-fraction 0.78
}

stage unit_tests
"$python" "$project/tests/test_relational_parallel.py"
"$python" "$project/tests/test_query_reconstruction.py"
"$python" -m py_compile \
    "$project/dinotool/cafe_query_reconstruction.py" \
    "$project/scripts/train_cafe_coco.py" \
    "$project/scripts/train_cafe_query_reconstruction.py" \
    "$project/scripts/eval_cafe_vc_protocols.py"

stage source_smoke
train "$output_root/smoke" --smoke --workers 2

stage coco_full_training
train "$output_root/full"

stage loveda_P
evaluate P

stage loveda_D
evaluate D

stage complete
