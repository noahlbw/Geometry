#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?vdd|potsdam}
shard=${2:?0|1}
gpu=${3:?physical GPU4-7}
mode=${4:-run}
[[ "$gpu" =~ ^[4-7]$ && "$shard" =~ ^[0-1]$ ]] || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
case "$dataset" in
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
    *) exit 2 ;;
esac
root=$tool/results/geometry_language_diagnostic_20261002
source=$base/ckpt/Qwen25VL7B_GeometryDiagnostic_20261002
suffix=${dataset}_s${shard}
extra=()
if [[ "$mode" == smoke || "$mode" == smoke-start ]]; then
    suffix=smoke_$dataset
    extra=(--smoke)
fi
output=$root/$suffix
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 4; }
[[ -f "$source/language_source_manifest.json" ]] || exit 5
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used utilization <<< "$status"
[[ -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] \
    && (( used <= 1024 && utilization <= 10 )) || { echo "GPU $gpu occupied" >&2; exit 3; }
if [[ "$mode" == start || "$mode" == smoke-start ]]; then
    session=gqo02_$suffix
    ! tmux has-session -t "$session" 2>/dev/null || exit 4
    mkdir -p "$root"
    nested=run
    [[ "$mode" != smoke-start ]] || nested=smoke
    exec tmux new-session -d -s "$session" \
        "exec bash '$tool/scripts/launch_geometry_language_diagnostic_a800.sh' '$dataset' '$shard' '$gpu' '$nested' > '$root/$suffix.log' 2>&1"
fi
[[ "$mode" == run || "$mode" == smoke ]] || exit 2
cd "$tool"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$tool:$tool/scripts" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    nice -n 10 /data/miniconda3/envs/pfu/bin/python scripts/diagnose_geometry_language_observation.py \
    --dataset "$dataset" --data-root "$data" --vocabulary-config "$tool/configs/$vocab" \
    --source-diagnostic-dir "$tool/results/geometry_readout_diagnostic_8_20261001_r2/$dataset" \
    --language-source "$source" --output-dir "$output" --num-shards 2 --shard-index "$shard" "${extra[@]}"
