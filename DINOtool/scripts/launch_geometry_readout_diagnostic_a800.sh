#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?usage: launch_geometry_readout_diagnostic_a800.sh vdd|potsdam GPU}
gpu=${2:?GPU required}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
case "$dataset" in
    vdd) data='/data/test/datasets/VDD Release Version/VDD'; vocab=grounded_vdd_official20.json ;;
    potsdam) data=/data/test/datasets/Potsdam/preprocessed_RGB; vocab=grounded_potsdam20.json ;;
    *) exit 2 ;;
esac
root=$tool/results/geometry_readout_diagnostic_8_20261001_r2
output=$root/$dataset
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 4; }
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used utilization <<< "$status"
[[ -z "$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)" ]] \
    && (( used <= 1024 && utilization <= 10 )) || { echo "GPU $gpu occupied" >&2; exit 3; }
if [[ "${3:-run}" == start ]]; then
    session=grdiag01_$dataset
    ! tmux has-session -t "$session" 2>/dev/null || exit 4
    mkdir -p "$root"
    exec tmux new-session -d -s "$session" \
        "exec bash '$tool/scripts/launch_geometry_readout_diagnostic_a800.sh' '$dataset' '$gpu' > '$root/$dataset.log' 2>&1"
fi
cd "$tool"
exec env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$tool:$tool/scripts:$third_party" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
    nice -n 10 /data/miniconda3/envs/pfu/bin/python scripts/diagnose_geometry_readout_streams.py \
    --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
    --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
    --vocabulary-config "$tool/configs/$vocab" \
    --reference-manifest "$tool/configs/cross_scale_reference_20261001.json" \
    --output-dir "$output"
