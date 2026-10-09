#!/usr/bin/env bash
# Independent, fail-fast RS stage-A training and post-training locked evaluation.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <fresh-output-root>" >&2; exit 2; }
output=$1
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=$base/envs/tessera-cu128/bin/python
official=$base/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
data=$base/datasets/COCOStuff2017
checkpoint=$base/code/ckpt/CAFe-DINO/weights.pth
bpe=$base/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
mkdir -p "$output"
exec >> "$output/queue.log" 2>&1
trap 'rc=$?; if (( rc != 0 )); then echo "failed:$rc" > "$output/stage.txt"; fi' EXIT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CAFE_OFFICIAL_ROOT="$official"
export PYTHONPATH="$project:$project/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo regression > "$output/stage.txt"
"$python" -m pytest -q "$project/tests/test_cafe_rs.py" "$project/tests/test_cafe_vc.py" "$project/tests/test_cafe_vc_protocols.py"
echo equivalence > "$output/stage.txt"
"$python" "$project/scripts/verify_cafe_rs.py" --official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" --data-root "$data" --output "$output/equivalence.json"
common=(--official-root "$official" --checkpoint "$checkpoint" --bpe-path "$bpe" --data-root "$data"
        --epochs 8 --crop-size 224 --batch-size 2 --accum-steps 2 --workers 4
        --lr 2e-4 --weight-decay 0.01 --warmup-steps 500 --label-smoothing 0.1
        --content-dim 256 --heads 4 --stages 3 --modes 4 --amp bf16 --seed 20260913
        --region-loss-weight 0.05 --affinity-loss-weight 0.02 --kd-weight 0.02
        --memory-fraction 0.70 --log-every 20)
echo smoke > "$output/stage.txt"
"$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/train_cafe_rs.py" "${common[@]}" --smoke --output-dir "$output/smoke"
echo full > "$output/stage.txt"
"$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/train_cafe_rs.py" "${common[@]}" --output-dir "$output/full"
for protocol in source-audit P D; do
    echo "$protocol" > "$output/stage.txt"
    eval_data=$data
    [[ "$protocol" == source-audit ]] || eval_data=$base/datasets/LoveDA/val
    "$python" -m torch.distributed.run --standalone --nproc_per_node=8 "$project/scripts/eval_cafe_vc_protocols.py" \
        --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
        --data-root "$eval_data" --weights "$output/full/best_inference.pt" --protocol "$protocol" \
        --output-dir "$output/evaluation/$protocol" --amp bf16
done
echo complete > "$output/stage.txt"
