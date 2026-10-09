#!/usr/bin/env bash
# Launch the fixed LoveDA P/D evaluation only when all eight GPUs are idle.
set -euo pipefail

base=/data/code/ovss/dino_ovss_training_free_20260924
tool="$base/DINOtool_gear_20260929"
python_bin=/data/miniconda3/envs/pfu/bin/python
run="$tool/results/gear_fullv2_loveda_20260929"
third_party="$base/.parallel_readout_20260924"

[[ -f "$tool/scripts/eval_gear_ov.py" && -f "$tool/configs/gar_llm_raw20_loveda_v1.json" ]] || {
    echo 'Evaluator or fixed vocabulary missing.' >&2
    exit 2
}

for gpu in {0..7}; do
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    session="gear29_fullv2_loveda_s${gpu}"
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run/s$gpu" ]]; then
        echo "Session or output already exists: $session / $run/s$gpu" >&2
        exit 4
    fi
done

for gpu in {0..7}; do
    session="gear29_fullv2_loveda_s${gpu}"
    mkdir -p "$run/s$gpu"
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_gear_ov.py --dataset loveda --dinov3-repo '$base/DINOtool/dinov3_hub' --checkpoint-dir '$base/weights' --data-root '$base/data/loveda/val' --vocabulary-config '$tool/configs/gar_llm_raw20_loveda_v1.json' --output-dir '$run/s$gpu' --num-shards 8 --shard-index '$gpu' --progress-every 4 > '$run/s$gpu.log' 2>&1"
done
echo 'Started eight frozen GEAR-OV LoveDA shards.'
