#!/usr/bin/env bash
# Launch pinned VIP inference only on explicitly idle A800 GPUs.
set -euo pipefail

mode=${1:?usage: launch_vip_official_a800.sh MODE DATASET FIRST_GPU SHARDS [LIMIT]}
dataset=${2:?dataset required}
first_gpu=${3:?first GPU required}
shards=${4:?shard count required}
limit=${5:-0}

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
upstream=$base/third_party/VIP_official_5bd25ee
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run=$tool/results/vip_official_${mode}_${dataset}_20260929

case "$dataset" in
    loveda)
        data=/data/test/cafe-efa/data/processed/loveda/val
        vocab=$tool/configs/gar_llm_raw20_loveda_v1.json ;;
    udd5)
        data=/data/test/datasets/UDD5/extracted/UDD/UDD5
        vocab=$tool/configs/hero_udd5_vip20.json ;;
    oem)
        data=/data/test/datasets/OpenEarthMap_wo_xBD
        vocab=$tool/configs/hero_oem_vip20.json ;;
    vdd)
        data='/data/test/datasets/VDD Release Version/VDD'
        vocab='' ;;
    potsdam)
        data=/data/test/datasets/Potsdam/preprocessed_RGB
        vocab='' ;;
    vaihingen)
        data=/data/test/datasets/Vaihingen/preprocessed_complete
        vocab='' ;;
    landcoverai)
        data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29
        vocab=$tool/configs/gear_landcoverai_v1_20.json ;;
    flair1)
        data=/data/test/datasets/FLAIR1_target_20260922
        vocab=$tool/configs/gear_flair1_main12_20.json ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

[[ "$first_gpu" =~ ^[0-7]$ && "$shards" =~ ^[1-8]$ && "$limit" =~ ^[0-9]+$ ]] || exit 2
(( first_gpu + shards <= 8 )) || { echo 'GPU range exceeds device 7.' >&2; exit 2; }
[[ -d "$data" && -d "$upstream" && -f "$tool/scripts/eval_vip_official_eight.py" ]] || {
    echo 'Missing data, pinned upstream, or evaluator.' >&2; exit 2;
}
[[ -z "$vocab" || -f "$vocab" ]] || { echo "Missing vocabulary: $vocab" >&2; exit 2; }

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session="vip29_${mode}_${dataset}_s${shard}"
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2; exit 3
    fi
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run/s$shard" ]]; then
        echo "Session or output already exists: $session / $run/s$shard" >&2; exit 4
    fi
done

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session="vip29_${mode}_${dataset}_s${shard}"
    mkdir -p "$run/s$shard"
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_vip_official_eight.py --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --upstream-root '$upstream' --data-root '$data' --vocabulary-config '$vocab' --output-dir '$run/s$shard' --num-shards '$shards' --shard-index '$shard' --max-images '$limit' --progress-every 5 > '$run/s$shard.log' 2>&1"
done
echo "Started VIP $dataset $mode on GPUs $first_gpu-$((first_gpu + shards - 1))."
