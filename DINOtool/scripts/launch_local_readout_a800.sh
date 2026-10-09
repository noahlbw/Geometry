#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: launch_local_readout_a800.sh screen|full [vdd|potsdam]}
dataset=${2:-}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
third_party=/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c

case "$mode:$dataset" in
    screen:)
        datasets=(vdd vdd potsdam potsdam)
        shard_indices=(0 1 0 1)
        shards=2
        ;;
    full:vdd|full:potsdam)
        datasets=("$dataset" "$dataset" "$dataset" "$dataset")
        shard_indices=(0 1 2 3)
        shards=4
        ;;
    *) echo "Expected screen, or full with vdd|potsdam" >&2; exit 2 ;;
esac

for offset in 0 1 2 3; do
    gpu=$((offset + 4))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    if (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    ds=${datasets[$offset]}
    shard=${shard_indices[$offset]}
    session="lr29_${mode}_${ds}_s${shard}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session exists: $session" >&2
        exit 4
    fi
    run="$tool/results/local_readout_2x2_${mode}_${ds}_20260929"
    if [[ -e "$run/s$shard" ]]; then
        echo "Shard output exists: $run/s$shard" >&2
        exit 5
    fi
done

for offset in 0 1 2 3; do
    gpu=$((offset + 4))
    ds=${datasets[$offset]}
    shard=${shard_indices[$offset]}
    if [[ "$ds" == vdd ]]; then
        root='/data/test/datasets/VDD Release Version/VDD'
        vocabulary=grounded_vdd_official20.json
        ontology=official
        limit=16
    else
        root=/data/test/datasets/Potsdam/preprocessed_RGB
        vocabulary=grounded_potsdam20.json
        ontology=legacy
        limit=32
    fi
    if [[ "$mode" == full ]]; then
        limit=0
    fi
    run="$tool/results/local_readout_2x2_${mode}_${ds}_20260929"
    mkdir -p "$run/s$shard"
    session="lr29_${mode}_${ds}_s${shard}"
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_local_readout_2x2.py --dataset '$ds' --vdd-ontology '$ontology' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$weights' --data-root '$root' --vocabulary-config '$tool/configs/$vocabulary' --output-dir '$run/s$shard' --memory-fraction 0.20 --num-shards '$shards' --shard-index '$shard' --max-images '$limit' --geometry-depth 2 --prefix-policy preserve --progress-every 2 > '$run/s$shard.log' 2>&1"
done
echo "Started $mode local-readout 2x2 evaluation on A800 GPUs 4-7."
