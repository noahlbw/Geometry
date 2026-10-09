#!/usr/bin/env bash
# Four-A800-GPU source-only run for the query-conditioned region assembly model.
set -euo pipefail

bundle=/data/test/code/ovss_cafe_ped_v2_20260919
project=$bundle/DINOtool_region_assembly_20260920
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
coco=/data/test/datasets/COCOStuff2017
oem=/data/test/datasets/OpenEarthMap_wo_xBD
root=$bundle/results/cafe_region_assembly_20260920
smoke=$root/smoke_4gpu
full=$root/full_4gpu

[[ -x $python ]] || { echo "missing Python environment: $python" >&2; exit 2; }
[[ -d $project && -d $official && -f $checkpoint && -f $bpe ]] || {
    echo "missing project or CAFe/DINO artifacts" >&2
    exit 2
}
[[ -f $coco/manifests/train2017_cafe41.json && -f $oem/xbd_files.csv ]] || {
    echo "locked COCO/OEM source data is incomplete" >&2
    exit 2
}
[[ ! -e $root ]] || { echo "refusing to overwrite existing run root: $root" >&2; exit 2; }
mkdir -p $root

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

common=(
    --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe"
    --coco-root "$coco" --oem-root "$oem"
    --arm query_assembly
    --updates 20896 --validate-every 2612 --checkpoint-every 500 --warmup-steps 500
    --batch-size 2 --accum-steps 4 --workers 4
    --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6
    --weight-decay 0.01 --kd-weight 0.02 --label-smoothing 0.1 --membership-weight 0.1
    --assembly-dim 256 --assembly-heads 8 --assembly-regions 128 --assembly-parents 32
    --assembly-proposal-layers 3 --assembly-rounds 2 --assembly-stages 3 --assembly-query-chunk 8
    --amp bf16 --memory-fraction 0.75 --seed 20260920
)

launch() {
    local output=$1
    shift
    "$python" -m torch.distributed.run \
        --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        "$project/scripts/train_cafe_region_assembly.py" \
        "${common[@]}" --output-dir "$output" "$@"
}

launch "$smoke" --smoke
launch "$full"
