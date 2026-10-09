#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: launch_competitive_ownership_a800.sh DATASET SHARD GPU}
shard=${2:?shard required}
gpu=${3:?GPU required}
[[ "$shard" =~ ^[0-3]$ && "$gpu" =~ ^[0-7]$ ]] || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
case "$dataset" in
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=gar_llm_raw20_loveda_v1.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
    flair1) data=/data/test/datasets/FLAIR1_target_20260922; vocab=gear_flair1_main12_20.json ;;
    landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; vocab=gear_landcoverai_v1_20.json ;;
    vdd) data=/data/test/datasets/VDD_Release_Version; vocab=gear_vdd_main7_20.json ;;
    potsdam) data=/data/test/datasets/ISPRS_Potsdam; vocab=gear_potsdam_main6_20.json ;;
    vaihingen) data=/data/test/datasets/ISPRS_Vaihingen; vocab=gear_vaihingen_main5_20.json ;;
    *) echo "Unsupported dataset $dataset" >&2; exit 2 ;;
esac
root=$tool/results/competitive_ownership_${dataset}_20260930
output=$root/s$shard
session=cor30_${dataset}_s$shard
[[ -f "$tool/configs/$vocab" && -d "$data" ]] || {
    echo "Missing vocabulary or dataset: $vocab / $data" >&2; exit 2;
}
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
IFS=, read -r used utilization <<< "$status"
if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
    echo "GPU $gpu busy: $status" >&2; exit 3
fi
if tmux has-session -t "$session" 2>/dev/null || [[ -e "$output" ]]; then
    echo "Session or output exists: $session / $output" >&2; exit 4
fi
mkdir -p "$root"
log=$root/s$shard.log
command="cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 /data/miniconda3/envs/pfu/bin/python scripts/eval_gear_ov.py --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --data-root '$data' --vocabulary-config '$tool/configs/$vocab' --output-dir '$output' --competitive-ownership --num-shards 4 --shard-index '$shard' --progress-every 2 > '$log' 2>&1"
tmux new-session -d -s "$session" "$command"
echo "Started $dataset shard $shard on GPU $gpu."
