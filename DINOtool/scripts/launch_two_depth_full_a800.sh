#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: launch_two_depth_full_a800.sh vdd|potsdam preserve|block}
prefix_policy=${2:?missing prefix policy}
case "$prefix_policy" in
    preserve|block) ;;
    *) echo "Expected preserve or block" >&2; exit 2 ;;
esac

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
case "$dataset" in
    vdd)
        root='/data/test/datasets/VDD Release Version/VDD'
        vocabulary=grounded_vdd_official20.json
        ontology=official
        ;;
    potsdam)
        root=/data/test/datasets/Potsdam/preprocessed_RGB
        vocabulary=grounded_potsdam20.json
        ontology=legacy
        ;;
    *) echo "Expected vdd or potsdam" >&2; exit 2 ;;
esac
run=$tool/results/${dataset}_two_depth_${prefix_policy}_full_20260929

if [[ -e "$run" ]]; then
    echo "Run directory already exists: $run" >&2
    exit 2
fi
for shard in 0 1 2 3; do
    gpu=$((shard + 4))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    if (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    if tmux has-session -t "td29_${dataset}_${prefix_policy}_s$shard" 2>/dev/null; then
        echo "Session exists: td29_${dataset}_${prefix_policy}_s$shard" >&2
        exit 4
    fi
done

mkdir -p "$run"
for shard in 0 1 2 3; do
    gpu=$((shard + 4))
    output=$run/s$shard
    tmux new-session -d -s "td29_${dataset}_${prefix_policy}_s$shard" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_grounded_context.py --dataset '$dataset' --vdd-ontology '$ontology' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$weights' --data-root '$root' --vocabulary-config '$tool/configs/$vocabulary' --output-dir '$output' --memory-fraction 0.20 --num-shards 4 --shard-index $shard --geometry-depth 2 --prefix-policy '$prefix_policy' --progress-every 4 > '$run/s$shard.log' 2>&1"
done
echo "Started $dataset two-depth $prefix_policy full evaluation on GPUs 4-7 under $run"
