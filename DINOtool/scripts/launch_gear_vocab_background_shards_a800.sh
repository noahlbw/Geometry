#!/usr/bin/env bash
# Launch full fixed-readout vocabulary/threshold ablation on idle contiguous GPUs.
set -euo pipefail

dataset=${1:?usage: launch_gear_vocab_background_shards_a800.sh DATASET official|all20 FIRST_GPU SHARDS}
vocabulary=${2:?vocabulary required}
first_gpu=${3:?first GPU required}
shards=${4:?shard count required}

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
upstream=$base/third_party/VIP_official_5bd25ee
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run=$tool/results/vip_gear_vocab_background_20260930/$dataset/gear_${vocabulary}_fast

[[ "$dataset" == vdd || "$dataset" == potsdam ]] || exit 2
[[ "$vocabulary" == official || "$vocabulary" == all20 ]] || exit 2
[[ "$first_gpu" =~ ^[0-7]$ && "$shards" =~ ^[1-8]$ ]] || exit 2
(( first_gpu + shards <= 8 )) || exit 2
if [[ "$dataset" == vdd ]]; then
    data='/data/test/datasets/VDD Release Version/VDD'
    official=$upstream/configs/cls_vdd.txt
    all20=$tool/configs/grounded_vdd_official20.json
else
    data=/data/test/datasets/Potsdam/preprocessed_RGB
    official=$upstream/configs/cls_potsdam.txt
    all20=$tool/configs/grounded_potsdam20.json
fi
if [[ "$vocabulary" == official ]]; then
    vocab=$official
    extra=(--vip-official-classes)
else
    vocab=$all20
    extra=()
fi
[[ -d "$data" && -f "$vocab" && -f "$tool/scripts/eval_gear_vocab_background.py" ]] || exit 2

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session=vgt30_${dataset}_gear_${vocabulary}_fast_s${shard}
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run/s$shard" ]]; then
        echo "Session or output already exists: $session / $run/s$shard" >&2
        exit 4
    fi
done

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session=vgt30_${dataset}_gear_${vocabulary}_fast_s${shard}
    mkdir -p "$run/s$shard"
    printf -v command '%q ' "$python_bin" scripts/eval_gear_vocab_background.py \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
        --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
        --vocabulary-config "$vocab" --output-dir "$run/s$shard" \
        --num-shards "$shards" --shard-index "$shard" --progress-every 5 "${extra[@]}"
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$run/s$shard.log' 2>&1"
done
echo "Started $dataset $vocabulary on GPUs $first_gpu-$((first_gpu + shards - 1))."
