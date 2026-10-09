#!/usr/bin/env bash
# Launch one frozen vocabulary-selection job or independent GEAR evaluation shards.
set -euo pipefail

stage=${1:?usage: launch_gear_alias_study_a800.sh select|evaluate DATASET VARIANT FIRST_GPU SHARDS [LIMIT]}
dataset=${2:?dataset required}
variant=${3:?variant required}
first_gpu=${4:?first GPU required}
shards=${5:?shard count required}
limit=${6:-0}

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
root=$tool/results/gear_alias_study_20260929

case "$dataset" in
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; base_vocab=gar_llm_raw20_loveda_v1.json ;;
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; base_vocab=hero_udd5_vip20.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; base_vocab=hero_oem_vip20.json ;;
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; base_vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; base_vocab=grounded_potsdam20.json ;;
    vaihingen) data=/data/test/datasets/Vaihingen/preprocessed_complete; base_vocab=grounded_vaihingen20.json ;;
    landcoverai) data=/data/test/datasets/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29; base_vocab=gear_landcoverai_v1_20.json ;;
    flair1) data=/data/test/datasets/FLAIR1_target_20260922; base_vocab=gear_flair1_main12_20.json ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
[[ "$first_gpu" =~ ^[0-7]$ && "$shards" =~ ^[1-8]$ && "$limit" =~ ^[0-9]+$ ]] || exit 2
(( first_gpu + shards <= 8 )) || exit 2
[[ -d "$data" ]] || { echo "Missing dataset: $data" >&2; exit 2; }

if [[ "$stage" == select ]]; then
    [[ "$variant" == all20 && "$shards" == 1 && "$dataset" != vdd && "$dataset" != potsdam ]] || exit 2
    vocabulary=$tool/configs/$base_vocab
    run=$root/$dataset/selection
    script=$tool/scripts/select_gear_aliases.py
elif [[ "$stage" == evaluate ]]; then
    script=$tool/scripts/eval_gear_ov.py
    case "$variant" in
        official)
            [[ "$dataset" == vdd || "$dataset" == potsdam ]] || exit 2
            vocabulary=$base/third_party/VIP_official_5bd25ee/configs/cls_$dataset.txt ;;
        all20) vocabulary=$tool/configs/$base_vocab ;;
        selected|random) vocabulary=$root/$dataset/selection/${variant}_vocab.json ;;
        *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
    esac
    run=$root/$dataset/$variant
else
    echo "Unsupported stage: $stage" >&2
    exit 2
fi
[[ -f "$vocabulary" && -f "$script" ]] || { echo "Missing vocabulary or evaluator" >&2; exit 2; }

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    session="gas29_${stage}_${dataset}_${variant}_s${shard}"
    output=$run
    [[ "$stage" == evaluate ]] && output=$run/s$shard
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$output" ]]; then
        echo "Session or output already exists: $session / $output" >&2
        exit 4
    fi
done

for ((shard=0; shard<shards; shard++)); do
    gpu=$((first_gpu + shard))
    session="gas29_${stage}_${dataset}_${variant}_s${shard}"
    if [[ "$stage" == select ]]; then
        mkdir -p "$root/$dataset"
        log=$root/$dataset/selection.log
        arguments="--selection-images 64 --tiles-per-image 4"
        output=$run
    else
        mkdir -p "$run"
        log=$run/s$shard.log
        arguments="--num-shards $shards --shard-index $shard --max-images $limit --progress-every 2"
        [[ "$variant" == official ]] && arguments="$arguments --vip-official-classes"
        output=$run/s$shard
    fi
    command="exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' '$script' --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --data-root '$data' --vocabulary-config '$vocabulary' --output-dir '$output' $arguments > '$log' 2>&1"
    tmux new-session -d -s "$session" "$command"
done
echo "Started $stage $dataset $variant on GPUs $first_gpu-$((first_gpu + shards - 1))."
