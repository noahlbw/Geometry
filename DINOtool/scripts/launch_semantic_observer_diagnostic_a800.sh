#!/usr/bin/env bash
set -euo pipefail
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
run=$tool/results/semantic_observer_diagnostic_16_20261001
dataset=${1:?vdd|potsdam}
shard=${2:?shard 0-3}
gpu=${3:?GPU 0-7}
[[ "$shard" =~ ^[0-3]$ && "$gpu" =~ ^[0-7]$ ]] || exit 2
case "$dataset" in
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
    *) exit 2 ;;
esac
session=sod01_${dataset}_s$shard
output=$run/$dataset/s$shard
[[ ! -e "$output" ]] || { echo "Existing output: $output" >&2; exit 4; }
! tmux has-session -t "$session" 2>/dev/null || exit 4
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used utilization <<< "$status"
[[ -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] \
    && (( used <= 1024 && utilization <= 10 )) || { echo "GPU $gpu occupied" >&2; exit 3; }
mkdir -p "$run/$dataset"
printf -v command '%q ' /data/miniconda3/envs/pfu/bin/python scripts/diagnose_geometry_semantic_observers.py \
    --dataset "$dataset" --shard-index "$shard" --dinov3-repo "$tool/dinov3_hub" \
    --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" --vocabulary-config "$tool/configs/$vocab" \
    --upstream-root "$base/third_party/VIP_official_5bd25ee" \
    --source-diagnostic-dir "$tool/results/geometry_readout_diagnostic_8_20261001_r2/$dataset" \
    --output-dir "$output"
tmux new-session -d -s "$session" \
    "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$run/$dataset/s$shard.log' 2>&1"
echo "Started $session on GPU $gpu"
