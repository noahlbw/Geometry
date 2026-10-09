#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
root='/data/test/datasets/VDD Release Version/VDD'
run=$tool/results/vdd_two_depth_official20_full_20260929

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
    if tmux has-session -t "vhead29_full_s$shard" 2>/dev/null; then
        echo "Session exists: vhead29_full_s$shard" >&2
        exit 4
    fi
done

mkdir -p "$run"
for shard in 0 1 2 3; do
    gpu=$((shard + 4))
    output=$run/s$shard
    tmux new-session -d -s "vhead29_full_s$shard" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_grounded_context.py --dataset vdd --vdd-ontology official --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$weights' --data-root '$root' --vocabulary-config '$tool/configs/grounded_vdd_official20.json' --output-dir '$output' --memory-fraction 0.20 --num-shards 4 --shard-index $shard --geometry-depth 2 --prefix-policy preserve --progress-every 1 > '$run/s$shard.log' 2>&1"
done
echo "Started full VDD two-depth readout on GPUs 4-7 under $run"
