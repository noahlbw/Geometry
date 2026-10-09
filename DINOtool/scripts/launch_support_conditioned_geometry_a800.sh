#!/usr/bin/env bash
# New outputs only; no automatic resumption of a stopped experiment queue.
set -euo pipefail
mode=${1:?usage: launch_support_conditioned_geometry_a800.sh smoke GPU | dataset DATASET SHARDS FIRST_GPU}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
root=$tool/results/support_conditioned_geometry_full_20261001

gpu_idle() {
    local status used utilization
    status=$(nvidia-smi --id="$1" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    [[ -z "$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader)" ]] \
        && (( used <= 1024 && utilization <= 10 ))
}

dataset_paths() {
    case "$1" in
        loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=gar_llm_raw20_loveda_v1.json ;;
        udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
        oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
        vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
        potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
        vaihingen) data=/data/test/datasets/Vaihingen/preprocessed_complete; vocab=grounded_vaihingen20.json ;;
        landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; vocab=gear_landcoverai_v1_20.json ;;
        flair1) data=/data/test/datasets/FLAIR1_target_20260922; vocab=gear_flair1_main12_20.json ;;
        *) echo "Unknown dataset" >&2; exit 2 ;;
    esac
}

evaluate() {
    local dataset=$1 shards=$2 shard=$3 gpu=$4 output=$5 limit=${6:-0}
    dataset_paths "$dataset"
    [[ ! -e "$output" ]] || { echo "Refusing existing output $output" >&2; return 4; }
    gpu_idle "$gpu" || { echo "GPU $gpu occupied; no launch." >&2; return 3; }
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$tool:$tool/scripts:$third_party" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
        nice -n 10 "$python_bin" "$tool/scripts/eval_support_conditioned_geometry.py" \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
        --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
        --vocabulary-config "$tool/configs/$vocab" \
        --reference-manifest "$tool/configs/cross_scale_reference_20261001.json" \
        --output-dir "$output" --num-shards "$shards" --shard-index "$shard" \
        --max-images "$limit" --progress-every 2
}

cd "$tool"
case "$mode" in
    smoke)
        gpu=${2:?GPU required}
        [[ "$gpu" =~ ^[0-7]$ ]] || exit 2
        evaluate oem 1 0 "$gpu" "$tool/results/support_conditioned_smoke_vectorized_20261001" 1
        ;;
    dataset)
        dataset=${2:?dataset required}; shards=${3:?shard count required}; first=${4:?first GPU required}
        [[ "$shards" =~ ^[1-8]$ && "$first" =~ ^[0-7]$ ]] || exit 2
        (( first + shards <= 8 )) || exit 2
        dataset_paths "$dataset"
        [[ -d "$data" && -f "$tool/configs/$vocab" && ! -e "$root/$dataset" ]] || exit 4
        for ((shard=0; shard<shards; shard++)); do
            gpu=$((first + shard))
            gpu_idle "$gpu" || { echo "GPU $gpu occupied; no launch." >&2; exit 3; }
            ! tmux has-session -t "scg01_${dataset}_s$shard" 2>/dev/null || exit 4
        done
        mkdir -p "$root/$dataset"
        for ((shard=0; shard<shards; shard++)); do
            gpu=$((first + shard))
            tmux new-session -d -s "scg01_${dataset}_s$shard" \
                "exec bash '$tool/scripts/launch_support_conditioned_geometry_a800.sh' shard '$dataset' '$shards' '$shard' '$gpu' > '$root/$dataset/s$shard.log' 2>&1"
        done
        echo "Launched $dataset: $shards shards, GPUs $first-$((first + shards - 1))"
        ;;
    shard)
        dataset=${2:?}; shards=${3:?}; shard=${4:?}; gpu=${5:?}
        evaluate "$dataset" "$shards" "$shard" "$gpu" "$root/$dataset/s$shard"
        ;;
    *) echo "Unknown launch mode" >&2; exit 2 ;;
esac
