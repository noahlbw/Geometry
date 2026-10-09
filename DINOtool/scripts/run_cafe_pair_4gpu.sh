#!/usr/bin/env bash
# Four-GPU source-only training for DINO-PairFusion-v1.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_ROOT" >&2
    exit 2
fi

output_root=$1
bundle=${OVSS_BUNDLE:-/data/test/code/ovss_cafe_ped_v2_20260919}
project=${OVSS_PROJECT:-$bundle/DINOtool_pairfusion_20260923}
python=${PYTHON_BIN:-/data/miniconda3/envs/pfu/bin/python}
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=${COCO_ROOT:-/data/test/datasets/COCOStuff2017}
oem=${OEM_ROOT:-/data/test/datasets/OpenEarthMap_wo_xBD}

[[ ! -e $output_root && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "Output must be fresh and PairFusion artifacts must exist" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv ]] || {
    echo "Locked COCO/OEM source data is missing" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

run_train() {
    "$python" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
        --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_pair.py" \
        --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
        --coco-root "$coco" --oem-root "$oem" --output-dir "$1" \
        --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --weight-decay 0.01 \
        --kd-weight 0.02 --label-smoothing 0.1 --amp bf16 --memory-fraction 0.75 \
        --seed 20260923 "$@"
}

mkdir -p "$output_root"
"$python" "$project/tests/test_cafe_pair.py"

# Verify the real DINO/CAFe/DDP data path before committing a full source run.
run_train "$output_root/smoke" --smoke --batch-size 2 --accum-steps 2 --workers 2

# The source-only protocol remains fixed: COCO/OEM only, source-selected weights.
run_train "$output_root/full" --updates 20896 --validate-every 2612 --checkpoint-every 500 \
    --warmup-steps 0 --batch-size 2 --accum-steps 4 --workers 4
