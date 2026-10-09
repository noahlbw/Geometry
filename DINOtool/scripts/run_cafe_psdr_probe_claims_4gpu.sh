#!/usr/bin/env bash
# Source-only mechanism screen for the three PSDR evidence-addressing claims.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_ROOT" >&2
    exit 2
fi

root=$1
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=/data/test/datasets/COCOStuff2017
oem=/data/test/datasets/OpenEarthMap_wo_xBD
loveda=/data/test/cafe-efa/data/processed/loveda/val

[[ ! -e $root && -x $python && -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "output must be fresh and CAFe artifacts must exist" >&2; exit 2;
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv && -d $loveda ]] || {
    echo "locked source data or LoveDA validation is missing" >&2; exit 2;
}
mkdir -p "$root/source" "$root/loveda"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

common=(
    --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe"
    --coco-root "$coco" --oem-root "$oem" --arm psdr_probe
    --updates 20896 --validate-every 2612 --stop-after-validation 2612 --checkpoint-every 500 --warmup-steps 500
    --batch-size 2 --accum-steps 4 --workers 4
    --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --weight-decay 0.01 --kd-weight 0.02 --label-smoothing 0.1
    --evidence-dim 256 --evidence-heads 8 --evidence-grids 8 4 --regions-per-read 4 --samples-per-region 4
    --evidence-rounds 2 --evidence-query-chunk 8 --evidence-visual-blocks 2 --amp bf16 --memory-fraction 0.75 --seed 20260922
)

run() {
    local label=$1 address=$2 content=$3 utility=$4
    "$python" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_psdr_probe.py" "${common[@]}" --output-dir "$root/source/$label" \
        --address-mode "$address" --content-mode "$content" --utility-weight "$utility"
}

# M1: who specifies the address? The only changing field is address_mode.
run m1_image_only       image_only  member_geometry 0
run m1_query_only       query_only  member_geometry 0
run m1_pixel_only       pixel_only  member_geometry 0
run m1_pixel_query      pixel_query member_geometry 0

# M2: what is read? M1's pixel-query member result is the third comparison.
run m2_region_mean      pixel_query region_mean     0
run m2_flat_attention   pixel_query flat_attention  0

# M3: identical actual-outcome supervision is applied to both readers.
run m3_member_utility   pixel_query member_geometry 0.10
run m3_flat_utility     pixel_query flat_attention  0.10

for label in m1_image_only m1_query_only m1_pixel_only m1_pixel_query m2_region_mean m2_flat_attention m3_member_utility m3_flat_utility; do
    weights=$root/source/$label/best_inference.pt
    [[ -f $weights ]] || { echo "missing source-selected weight: $weights" >&2; exit 3; }
    for protocol in P D; do
        "$python" -m torch.distributed.run --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
            "$project/scripts/eval_cafe_vc_protocols.py" --official-root "$official" --base-checkpoint "$checkpoint" \
            --bpe-path "$bpe" --data-root "$loveda" --weights "$weights" --protocol "$protocol" \
            --output-dir "$root/loveda/$protocol/$label" --amp bf16 --memory-fraction 0.75
    done
done
