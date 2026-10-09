#!/usr/bin/env bash
# Source-only A100 run: CAFe spatial and freshly initialized class aggregators
# train jointly through an anchored per-query parallel fusion gate.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 FRESH_OUTPUT_ROOT" >&2
    exit 2
fi

output_root=$1
bundle=/data/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_task_parallel_20260928
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=/data/datasets/COCOStuff2017

[[ ! -e $output_root && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "A fresh output root and pinned model artifacts are required" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $coco/manifests/val2017_source_dev.json ]] || {
    echo "Locked COCO source manifests are missing" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
        --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_coco.py" \
        --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
        --data-root "$coco" --output-dir "$destination" \
        --arm task_parallel --channel-init fresh --corrected-class-attention \
        --visual-tune-blocks 0 --epochs 16 --max-updates 20896 --crop-size 224 \
        --batch-size 1 --accum-steps 4 --workers 2 \
        --lr 1e-4 --cafe-lr 2e-5 --weight-decay 0.01 --warmup-steps 500 \
        --kd-weight 0 --label-smoothing 0.1 \
        --amp bf16 --memory-fraction 0.90 --seed 20260928 "$@"
}

stage preflight
"$python" -m py_compile \
    "$project/dinotool/cafe_pca.py" \
    "$project/scripts/train_cafe_coco.py" \
    "$project/tests/check_task_parallel.py"
"$python" "$project/tests/check_task_parallel.py"

stage source_smoke
train "$output_root/smoke" --smoke --workers 1

stage coco_full_training
train "$output_root/full"

stage complete
