#!/usr/bin/env bash
set -euo pipefail

project=${1:-/data/code/ovss/dino_ovss_training_free_20260924}
run_name=${2:-dger_consensus_full_1669_8shard_20260927}
session_prefix=${3:-dgercons27}
python_bin=${PYTHON_BIN:-/data/miniconda3/envs/pfu/bin/python}
cd "$project"
for shard in {0..7}; do
    if tmux has-session -t "${session_prefix}_s${shard}" 2>/dev/null; then
        echo "Session already exists: ${session_prefix}_s${shard}" >&2
        exit 1
    fi
done
if [[ -e "results/$run_name" ]]; then
    echo "Output already exists; use a new run name to retain the previous result." >&2
    exit 1
fi
mkdir -p "results/$run_name"
for shard in {0..7}; do
    printf -v command '%q ' env "CUDA_VISIBLE_DEVICES=$shard" PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "PYTHONPATH=$project/DINOtool:$project/.parallel_readout_20260924" \
        "$python_bin" "$project/DINOtool/scripts/eval_dger_loveda.py" \
        --dinov3-repo "$project/DINOtool/dinov3_hub" --checkpoint-dir "$project/weights" \
        --data-root "$project/data/loveda/val" --output-dir "$project/results/$run_name/s$shard" \
        --vocabulary-config "$project/DINOtool/configs/tcpr_loveda_vip_v1.json" \
        --consensus --max-images 0 --num-shards 8 --shard-index "$shard" --progress-every 32
    printf -v log '%q' "$project/results/$run_name/s$shard.log"
    tmux new-session -d -s "${session_prefix}_s${shard}" "$command > $log 2>&1"
done
echo "Launched ${session_prefix}_s0..s7 in $project/results/$run_name"
