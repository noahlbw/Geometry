#!/usr/bin/env bash
# Four-GPU source-only PCA-DINO smoke, first-validation screen, or full run.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "usage: $0 OUTPUT_DIR [smoke|screen|full] [serial|parallel|pca_epl|pca_epl_fod]" >&2
    exit 2
fi

output=$1
mode=${2:-smoke}
arm=${3:-pca_epl_fod}
bundle=${OVSS_BUNDLE:-/data/test/code/ovss_cafe_ped_v2_20260919}
# The dated reproducible snapshot contains the extensible locked trainer used
# by the current source-only studies. OVSS_PROJECT can override it explicitly.
project=${OVSS_PROJECT:-$bundle/DINOtool_region_assembly_20260920}
python=${PYTHON_BIN:-/data/miniconda3/envs/pfu/bin/python}
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=${COCO_ROOT:-/data/test/datasets/COCOStuff2017}
oem=${OEM_ROOT:-/data/test/datasets/OpenEarthMap_wo_xBD}

case $mode in
    smoke) schedule=(--smoke --batch-size 2 --accum-steps 2 --workers 2) ;;
    screen) schedule=(--updates 20896 --validate-every 2612 --stop-after-validation 2612 --checkpoint-every 500
                      --warmup-steps 500 --batch-size 2 --accum-steps 4 --workers 4) ;;
    full) schedule=(--updates 20896 --validate-every 2612 --checkpoint-every 500 --warmup-steps 500
                    --batch-size 2 --accum-steps 4 --workers 4) ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

case $arm in
    serial|parallel|pca_epl) fod=() ;;
    pca_epl_fod) fod=(--fod-weight 0.001) ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

[[ ! -e $output && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "output must be fresh and PCA-DINO artifacts must exist" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv ]] || {
    echo "locked COCO/OEM source data is missing" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$python" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    "$project/scripts/train_cafe_pca.py" \
    --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
    --coco-root "$coco" --oem-root "$oem" --output-dir "$output" --arm "$arm" \
    --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --weight-decay 0.01 \
    --kd-weight 0.02 --label-smoothing 0.1 --amp bf16 --memory-fraction 0.75 \
    --seed 20260923 "${schedule[@]}" "${fod[@]}"
