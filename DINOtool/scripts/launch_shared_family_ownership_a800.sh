#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: launch_shared_family_ownership_a800.sh DATASET SHARDS [FIRST_GPU]}
shards=${2:?shard count required}
first_gpu=${3:-0}
[[ "$shards" =~ ^[1-8]$ && "$first_gpu" =~ ^[0-7]$ ]] || exit 2
(( first_gpu + shards <= 8 )) || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
case "$dataset" in
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=gar_llm_raw20_loveda_v1.json ;;
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
    vaihingen) data=/data/test/datasets/Vaihingen/preprocessed_complete; vocab=grounded_vaihingen20.json ;;
    landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; vocab=gear_landcoverai_v1_20.json ;;
    flair1) data=/data/test/datasets/FLAIR1_target_20260922; vocab=gear_flair1_main12_20.json ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
root=$tool/results/shared_family_ownership_${dataset}_20260930
[[ -d "$data" && -f "$tool/configs/$vocab" ]] || {
    echo "Missing dataset or vocabulary for $dataset" >&2; exit 2;
}
for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session=sfo30_${dataset}_s$shard
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu busy: $status" >&2; exit 3
    fi
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$root/s$shard" ]]; then
        echo "Session or output exists: $session / $root/s$shard" >&2; exit 4
    fi
done
mkdir -p "$root"
for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    output=$root/s$shard
    log=$root/s$shard.log
    command="cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 /data/miniconda3/envs/pfu/bin/python scripts/eval_gear_ov.py --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --data-root '$data' --vocabulary-config '$tool/configs/$vocab' --output-dir '$output' --shared-family-ownership --num-shards $shards --shard-index $shard --progress-every 10 > '$log' 2>&1"
    tmux new-session -d -s "sfo30_${dataset}_s$shard" "$command"
done
echo "Started $dataset on GPUs $first_gpu-$((first_gpu + shards - 1))."
