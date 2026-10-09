#!/usr/bin/env bash
# Full VDD (GPUs 0-3) and Potsdam (GPUs 4-7), with one frozen selection rule.
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run=$tool/results/geometry_vip_reliability_full_20260930

[[ -d '/data/test/datasets/VDD Release Version/VDD' &&
   -d /data/test/datasets/Potsdam/preprocessed_RGB &&
   -f "$tool/configs/grounded_vdd_official20.json" &&
   -f "$tool/configs/grounded_potsdam20.json" &&
   -f "$tool/scripts/eval_geometry_vip_reliability.py" &&
   -f "$tool/dinotool/branch_reliability.py" ]] || { echo 'Missing required input.' >&2; exit 2; }
[[ ! -e "$run" ]] || { echo "Refusing existing run: $run" >&2; exit 4; }
for gpu in {0..7}; do
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2; exit 3
    fi
    if tmux has-session -t "gvr30_gpu$gpu" 2>/dev/null; then
        echo "Session exists: gvr30_gpu$gpu" >&2; exit 4
    fi
done
for gpu in {0..7}; do
    if (( gpu < 4 )); then
        dataset=vdd; shard=$gpu; data='/data/test/datasets/VDD Release Version/VDD'
        vocab=$tool/configs/grounded_vdd_official20.json
    else
        dataset=potsdam; shard=$((gpu - 4)); data=/data/test/datasets/Potsdam/preprocessed_RGB
        vocab=$tool/configs/grounded_potsdam20.json
    fi
    mkdir -p "$run/$dataset/s$shard"
    printf -v command '%q ' "$python_bin" scripts/eval_geometry_vip_reliability.py \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" --checkpoint-dir "$base/ckpt/DINO" \
        --data-root "$data" --vocabulary-config "$vocab" \
        --upstream-root "$base/third_party/VIP_official_5bd25ee" \
        --output-dir "$run/$dataset/s$shard" --num-shards 4 --shard-index "$shard" --progress-every 2
    tmux new-session -d -s "gvr30_gpu$gpu" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$run/$dataset/s$shard.log' 2>&1"
done
echo 'Started full VDD 80 and Potsdam 504, GPUs 0-7, four shards each.'
