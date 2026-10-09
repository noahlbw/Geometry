#!/usr/bin/env bash
# Launch independent frozen GEAR-OV shards on an explicitly selected idle GPU range.
set -euo pipefail

mode=${1:?usage: launch_gear_ov_a800.sh MODE DATASET DATA_ROOT VOCAB GPU_FIRST SHARDS [LIMIT]}
dataset=${2:?dataset required}
data_root=${3:?data root required}
vocabulary=${4:?vocabulary path required}
first_gpu=${5:?first GPU required}
shards=${6:?shard count required}
limit=${7:-0}

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
third_party=/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run="$tool/results/gear_${mode}_${dataset}_20260929"

[[ "$first_gpu" =~ ^[0-7]$ && "$shards" =~ ^[1-8]$ && "$limit" =~ ^[0-9]+$ ]] || {
    echo 'Invalid GPU range, shard count or sample limit.' >&2
    exit 2
}
(( first_gpu + shards <= 8 )) || { echo 'GPU range exceeds device 7.' >&2; exit 2; }
[[ -d "$data_root" && -f "$vocabulary" && -f "$tool/scripts/eval_gear_ov.py" ]] || {
    echo 'Required data, vocabulary or evaluator is missing.' >&2
    exit 2
}

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    session="gear29_${mode}_${dataset}_s${shard}"
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run/s$shard" ]]; then
        echo "Session or output already exists: $session / $run/s$shard" >&2
        exit 4
    fi
done

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session="gear29_${mode}_${dataset}_s${shard}"
    mkdir -p "$run/s$shard"
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_gear_ov.py --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$weights' --data-root '$data_root' --vocabulary-config '$vocabulary' --output-dir '$run/s$shard' --num-shards '$shards' --shard-index '$shard' --max-images '$limit' --progress-every 2 > '$run/s$shard.log' 2>&1"
done
echo "Started $dataset $mode shards on GPUs $first_gpu-$((first_gpu + shards - 1))."
