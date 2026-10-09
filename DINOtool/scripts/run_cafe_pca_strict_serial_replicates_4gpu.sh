#!/usr/bin/env bash
# Locked, source-only serial controls for the strict PCA-DINO screen.
#
# This is intentionally not a PairFusion launcher. It provides the missing
# matched serial controls needed to distinguish a topology effect from the
# observed EPL-versus-parallel difference. Each invocation runs the three
# pre-registered seeds sequentially and stops after validation at step 2612.
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
seeds=(20260923 34117 73531)

[[ ! -e $output_root && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "output root must be fresh and all locked PCA-DINO artifacts must exist" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv ]] || {
    echo "locked COCO/OEM source data is missing" >&2
    exit 2
}

# The caller chooses the physical allocation. The approved run uses 4,5,6,7.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$output_root"
for seed in "${seeds[@]}"; do
    output="$output_root/seed${seed}/serial_clean"
    log="$output_root/seed${seed}/serial_clean.log"
    [[ ! -e $output ]] || {
        echo "refusing to overwrite existing output: $output" >&2
        exit 2
    }
    "$python" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
        --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_pca.py" \
        --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
        --coco-root "$coco" --oem-root "$oem" --output-dir "$output" --arm serial \
        --corrected-class-attention --experts 4 --reduction-ratio 4 --fod-weight 0 \
        --evidence-dim 512 --memory-tokens 128 --memory-heads 8 --read-heads 8 \
        --interaction-dim 128 --message-hidden 1024 --visual-blocks 2 \
        --pyramid-grids 8 4 2 --support-points 8 --surround-points 8 --geometry-mode legacy \
        --updates 20896 --validate-every 2612 --stop-after-validation 2612 --checkpoint-every 500 \
        --warmup-steps 0 --batch-size 2 --accum-steps 4 --workers 4 \
        --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --weight-decay 0.01 \
        --kd-weight 0.02 --label-smoothing 0.1 --amp bf16 --memory-fraction 0.75 \
        --seed "$seed" 2>&1 | tee "$log"
done
