#!/usr/bin/env bash
# Controlled four-A800-GPU Q-Lift-DINO launch. The caller supplies a fresh run.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 {skip|haar|image_lift|query_lift} OUTPUT_DIR [trainer arguments]" >&2
    exit 2
fi

arm=$1
output=$2
shift 2
case "$arm" in
    skip|haar|image_lift|query_lift) ;;
    *) echo "unknown Q-Lift arm: $arm" >&2; exit 2 ;;
esac

bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=/data/test/datasets/COCOStuff2017
oem=/data/test/datasets/OpenEarthMap_wo_xBD

[[ -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "missing Q-Lift code, environment, or frozen CAFe artifact" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv ]] || {
    echo "locked COCO/OEM source data is incomplete" >&2
    exit 2
}
[[ ! -e $output ]] || { echo "refusing to overwrite run: $output" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

exec "$python" -m torch.distributed.run \
    --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    "$project/scripts/train_cafe_qlift.py" \
    --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" \
    --coco-root "$coco" --oem-root "$oem" --output-dir "$output" --arm "$arm" \
    --updates 20896 --validate-every 2612 --checkpoint-every 500 --warmup-steps 500 \
    --batch-size 2 --accum-steps 4 --workers 4 \
    --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 \
    --weight-decay 0.01 --kd-weight 0.02 --label-smoothing 0.1 \
    --qlift-levels 2 --qlift-guide-dim 128 --qlift-hidden-dim 256 --qlift-kernel 3 \
    --qlift-query-chunk 8 --qlift-tune-visual-blocks 0 \
    --amp bf16 --memory-fraction 0.75 --seed 20260921 "$@"
