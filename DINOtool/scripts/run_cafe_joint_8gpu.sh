#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <fresh-output-root>" >&2; exit 2; }
output=$1
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=$base/envs/tessera-cu128/bin/python
official=$base/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$base/code/ckpt/CAFe-DINO/weights.pth
bpe=$base/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
mkdir -p "$output"
exec >> "$output/queue.log" 2>&1
stage=starting
trap 'rc=$?; if (( rc != 0 )); then echo "failed:$stage:$rc" > "$output/stage.txt"; fi' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CAFE_OFFICIAL_ROOT="$official"
export PYTHONPATH="$project:$project/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
set_stage() {
    stage=$1
    echo "$stage" > "$output/stage.txt"
    date -Is
    echo "$stage"
}
set_stage regression
"$python" -m pytest -q "$project/tests/test_cafe_joint.py" "$project/tests/test_cafe_rs.py" "$project/tests/test_cafe_vc.py" "$project/tests/test_cafe_vc_protocols.py" "$project/tests/test_coco_stuff.py"
common=(--official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe"
        --coco-root "$base/datasets/COCOStuff2017" --oem-root "$base/datasets/OpenEarthMap_wo_xBD"
        --updates 20896 --validate-every 2612 --warmup-steps 500 --batch-size 2 --accum-steps 2 --workers 4
        --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --kd-weight 0.02 --label-smoothing 0.1
        --weight-decay 0.01 --amp bf16 --seed 20260913 --memory-fraction 0.70)
for arm in plain concat rs rs_aux; do
    set_stage "smoke_$arm"
    "$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/train_cafe_joint.py" \
        "${common[@]}" --arm "$arm" --smoke --output-dir "$output/smoke/$arm"
done
# Fixed order and recipes; no target score can change later training jobs.
for arm in rs plain concat rs_aux; do
    set_stage "full_$arm"
    "$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/train_cafe_joint.py" \
        "${common[@]}" --arm "$arm" --output-dir "$output/full/$arm"
done
for arm in rs plain concat rs_aux; do
    for protocol in source-audit P D; do
        set_stage "eval_${arm}_${protocol}"
        data=$base/datasets/COCOStuff2017
        [[ "$protocol" == source-audit ]] || data=$base/datasets/LoveDA/val
        "$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/eval_cafe_vc_protocols.py" \
            --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
            --data-root "$data" --weights "$output/full/$arm/best_inference.pt" \
            --protocol "$protocol" --output-dir "$output/evaluation/$arm/$protocol" --amp bf16
    done
done
set_stage complete
