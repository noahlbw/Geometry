#!/usr/bin/env bash
# One complete-model run. Eight workers execute the fixed dataset queue.
set -euo pipefail

mode=${1:?usage: launch_cross_scale_geometry_a800.sh smoke|full|worker [GPU]}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
root=$tool/results/cross_scale_geometry_full_20261001
manifest=$tool/configs/cross_scale_reference_20261001.json

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
        *) echo "Unsupported dataset $1" >&2; exit 2 ;;
    esac
}

gpu_idle() {
    local gpu=$1 status used utilization processes
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    [[ -z "$processes" ]] && (( used <= 1024 && utilization <= 10 ))
}

evaluate() {
    local dataset=$1 shards=$2 shard=$3 gpu=$4 output=$5 limit=${6:-0}
    dataset_paths "$dataset"
    [[ ! -e "$output/results.json" ]] || { echo "Refusing existing result $output" >&2; return 4; }
    env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$tool:$tool/scripts:$third_party" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
        nice -n 10 "$python_bin" "$tool/scripts/eval_cross_scale_geometry.py" \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
        --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
        --vocabulary-config "$tool/configs/$vocab" --reference-manifest "$manifest" \
        --output-dir "$output" --num-shards "$shards" --shard-index "$shard" \
        --max-images "$limit" --progress-every 2
}

merge_ready() {
    local dataset=$1 shards=$2 lockfd
    [[ ! -f "$root/$dataset/merged.json" ]] || return 0
    exec {lockfd}>"$root/$dataset/.merge.lock"
    if flock -w 30 "$lockfd"; then
        if [[ ! -f "$root/$dataset/merged.json" ]] && "$python_bin" -c \
            'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); paths=[root/f"s{i}"/"results.json" for i in range(n)]; sys.exit(0 if all(p.is_file() and json.loads(p.read_text()).get("status")=="complete" for p in paths) else 1)' \
            "$root/$dataset" "$shards"; then
            local inputs=()
            for ((i=0; i<shards; i++)); do inputs+=("$root/$dataset/s$i"); done
            env PYTHONPATH="$tool:$tool/scripts:$third_party" "$python_bin" \
                "$tool/scripts/merge_gear_ov_shards.py" --inputs "${inputs[@]}" \
                --output "$root/$dataset/merged.json"
        fi
        flock -u "$lockfd"
    fi
    exec {lockfd}>&-
}

cd "$tool"
case "$mode" in
    smoke)
        gpu=${2:-0}
        [[ "$gpu" =~ ^[0-7]$ ]] || exit 2
        gpu_idle "$gpu" || { echo "GPU $gpu is occupied." >&2; exit 3; }
        evaluate oem 1 0 "$gpu" "$tool/results/cross_scale_geometry_smoke_20261001" 1
        ;;
    full)
        [[ ! -e "$root" && -f "$manifest" ]] || { echo "Run exists or manifest missing." >&2; exit 4; }
        for dataset in vdd potsdam udd5 oem vaihingen landcoverai loveda flair1; do
            dataset_paths "$dataset"
            [[ -d "$data" && -f "$tool/configs/$vocab" ]] || { echo "Missing input for $dataset" >&2; exit 2; }
        done
        for gpu in {0..7}; do
            gpu_idle "$gpu" || { echo "GPU $gpu is occupied." >&2; exit 3; }
            ! tmux has-session -t "csgg01_g$gpu" 2>/dev/null || exit 4
        done
        for dataset in vdd potsdam udd5 oem vaihingen landcoverai loveda flair1; do mkdir -p "$root/$dataset"; done
        for gpu in {0..7}; do
            tmux new-session -d -s "csgg01_g$gpu" \
                "exec bash '$tool/scripts/launch_cross_scale_geometry_a800.sh' worker $gpu > '$root/gpu$gpu.log' 2>&1"
        done
        echo "Started eight complete-model workers, GPUs 0-7. Outputs: $root"
        ;;
    worker)
        gpu=${2:?GPU required}
        [[ "$gpu" =~ ^[0-7]$ ]] || exit 2
        if (( gpu < 4 )); then
            shard=$gpu
            jobs=(vdd potsdam udd5 flair1)
        else
            shard=$((gpu - 4))
            jobs=(oem vaihingen landcoverai loveda flair1)
        fi
        for dataset in "${jobs[@]}"; do
            shards=4
            dataset_shard=$shard
            if [[ "$dataset" == flair1 ]]; then shards=8; dataset_shard=$gpu; fi
            output=$root/$dataset/s$dataset_shard
            [[ ! -e "$output" ]] || { echo "Existing output $output" >&2; exit 4; }
            while ! gpu_idle "$gpu"; do
                echo "Waiting without interference: GPU $gpu, next $dataset shard $dataset_shard"
                sleep 30
            done
            echo "$(date -Is) START $dataset $dataset_shard/$shards GPU $gpu"
            if evaluate "$dataset" "$shards" "$dataset_shard" "$gpu" "$output" \
                > "$root/$dataset/s$dataset_shard.log" 2>&1; then
                echo "$(date -Is) COMPLETE $dataset shard $dataset_shard"
                merge_ready "$dataset" "$shards"
            else
                echo "$(date -Is) FAILED $dataset shard $dataset_shard; inspect $root/$dataset/s$dataset_shard.log" >&2
                exit 1
            fi
        done
        echo "$(date -Is) COMPLETE worker GPU $gpu"
        ;;
    *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac
