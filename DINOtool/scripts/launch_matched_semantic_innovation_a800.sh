#!/usr/bin/env bash
set -euo pipefail
mode=${1:?full|diagnostic}
dataset=${2:?dataset}
shards=${3:?shards}
first=${4:?first GPU}
[[ "$mode" == full || "$mode" == diagnostic ]] || exit 2
[[ "$shards" =~ ^[1-8]$ && "$first" =~ ^[0-7]$ ]] && (( first+shards <= 8 )) || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
case "$dataset" in
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=gar_llm_raw20_loveda_v1.json ;;
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
    vaihingen) data=/data/test/datasets/Vaihingen/preprocessed_complete; vocab=grounded_vaihingen20.json ;;
    landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; vocab=gear_landcoverai_v1_20.json ;;
    flair1) data=/data/test/datasets/FLAIR1_target_20260922; vocab=gear_flair1_main12_20.json ;;
    *) exit 2 ;;
esac
root=$tool/results/matched_counterfactual_v4_${mode}_${dataset}_20261001
extra=()
if [[ "$mode" == diagnostic ]]; then
    [[ "$dataset" == vdd || "$dataset" == potsdam ]] || exit 2
    extra=(--source-diagnostic "$tool/results/geometry_readout_diagnostic_8_20261001_r2/$dataset/results.json")
fi
for ((shard=0; shard<shards; shard++)); do
    gpu=$((first+shard))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    [[ -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] \
        && (( used <= 1024 && utilization <= 10 )) || { echo "GPU $gpu occupied" >&2; exit 3; }
    [[ ! -e "$root/s$shard" ]] && ! tmux has-session -t "rcf04_${mode}_${dataset}_s$shard" 2>/dev/null || exit 4
done
mkdir -p "$root"
for ((shard=0; shard<shards; shard++)); do
    gpu=$((first+shard))
    printf -v command '%q ' /data/miniconda3/envs/pfu/bin/python scripts/eval_matched_semantic_innovation.py \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" --checkpoint-dir "$base/ckpt/DINO" \
        --data-root "$data" --vocabulary-config "$tool/configs/$vocab" \
        --upstream-root "$base/third_party/VIP_official_5bd25ee" --output-dir "$root/s$shard" \
        --num-shards "$shards" --shard-index "$shard" "${extra[@]}"
    tmux new-session -d -s "rcf04_${mode}_${dataset}_s$shard" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$root/s$shard.log' 2>&1"
done
echo "Started $mode $dataset on GPUs $first-$((first+shards-1))"
