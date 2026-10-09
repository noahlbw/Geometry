#!/usr/bin/env bash
# Controlled CAFe-PED-v2 study. LoveDA is only evaluated after source selection.
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
launch=("$python" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master-addr 127.0.0.1 --master-port 29583)

set_stage() {
    stage=$1
    echo "$stage" > "$output/stage.txt"
    date -Is
    echo "$stage"
}

common=(--official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe"
        --coco-root "$base/datasets/COCOStuff2017" --oem-root "$base/datasets/OpenEarthMap_wo_xBD"
        --updates 20896 --validate-every 2612 --checkpoint-every 500 --warmup-steps 500
        --batch-size 2 --accum-steps 2 --workers 4
        --evidence-dim 512 --memory-tokens 128 --memory-heads 8 --read-heads 8
        --interaction-dim 128 --message-hidden 1024 --visual-blocks 2
        --pyramid-grids 8 4 2 --support-points 8 --surround-points 8
        --new-lr 1e-4 --head-lr 2e-5 --visual-lr 2e-6 --kd-weight 0.02 --label-smoothing 0.1
        --weight-decay 0.01 --amp bf16 --seed 20260919 --memory-fraction 0.55)

set_stage regression
"$python" -m py_compile "$project/dinotool/parallel_evidence.py" "$project/dinotool/cafe_ped.py" \
    "$project/scripts/train_cafe_ped.py" "$project/scripts/eval_cafe_vc_protocols.py"
"$python" -m pytest -q "$project/tests/test_cafe_ped.py"

for arm in plain naive_parallel anchored_multiscale paired_support; do
    set_stage "smoke_$arm"
    "${launch[@]}" "$project/scripts/train_cafe_ped.py" \
        "${common[@]}" --arm "$arm" --smoke --output-dir "$output/smoke/$arm"
done

for arm in plain naive_parallel anchored_multiscale paired_support; do
    set_stage "full_$arm"
    "${launch[@]}" "$project/scripts/train_cafe_ped.py" \
        "${common[@]}" --arm "$arm" --output-dir "$output/full/$arm"
done

for arm in plain naive_parallel anchored_multiscale paired_support; do
    for protocol in source-audit P D; do
        set_stage "eval_${arm}_${protocol}"
        data=$base/datasets/COCOStuff2017
        [[ "$protocol" == source-audit ]] || data=$base/datasets/LoveDA/val
        "${launch[@]}" "$project/scripts/eval_cafe_vc_protocols.py" \
            --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
            --data-root "$data" --weights "$output/full/$arm/best_inference.pt" \
            --protocol "$protocol" --output-dir "$output/evaluation/$arm/$protocol" --amp bf16
    done
done

set_stage complete
