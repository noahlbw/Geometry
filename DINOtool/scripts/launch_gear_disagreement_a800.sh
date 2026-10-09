#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: launch_gear_disagreement_a800.sh select|evaluate DATASET GPU}
dataset=${2:?dataset required}
gpu=${3:?GPU required}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
case "$dataset" in
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=gar_llm_raw20_loveda_v1.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
    flair1) data=/data/test/datasets/FLAIR1_target_20260922; vocab=gear_flair1_main12_20.json ;;
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
    landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; vocab=gear_landcoverai_v1_20.json ;;
    *) exit 2 ;;
esac
root=$tool/results/gear_alias_disagreement_20260929
session=gasdis29_${stage}_${dataset}
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
IFS=, read -r used utilization <<< "$status"
if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
    echo "GPU $gpu busy: $status" >&2
    exit 3
fi
case "$stage" in
    select)
        output=$root/$dataset/selection
        script=$tool/scripts/select_gear_disagreement_aliases.py
        args="--images 64 --tiles-per-image 2 --keep-per-class 15"
        vocabulary=$tool/configs/$vocab ;;
    evaluate)
        output=$root/$dataset/evaluation
        script=$tool/scripts/eval_gear_ov.py
        args="--exact-alias-groups --progress-every 10"
        vocabulary=$root/$dataset/selection/selected_vocab.json ;;
    *) exit 2 ;;
esac
if tmux has-session -t "$session" 2>/dev/null || [[ -e "$output" ]]; then
    echo "Session or output exists: $session / $output" >&2
    exit 4
fi
[[ -f "$script" && -f "$vocabulary" ]] || { echo "Missing script or vocabulary" >&2; exit 2; }
mkdir -p "$root/$dataset"
log=$root/$dataset/$stage.log
command="cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 /data/miniconda3/envs/pfu/bin/python '$script' --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --data-root '$data' --vocabulary-config '$vocabulary' --output-dir '$output' $args > '$log' 2>&1"
tmux new-session -d -s "$session" "$command"
echo "Started $stage $dataset on GPU $gpu."
