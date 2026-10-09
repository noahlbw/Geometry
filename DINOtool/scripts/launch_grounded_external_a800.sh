#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: launch_grounded_external_a800.sh DATASET DATA_ROOT VOCAB OUTPUT_ROOT SHARDS}
data_root=${2:?missing data root}
vocab=${3:?missing vocabulary config}
output_root=${4:?missing output root}
shards=${5:-8}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
hub=$tool/dinov3_hub
script=$tool/scripts/eval_grounded_context.py

for gpu in $(seq 0 $((shards - 1))); do
    session=gb28_${dataset}_s${gpu}
    tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 1; }
done

for gpu in $(seq 0 $((shards - 1))); do
    output=$output_root/s$gpu
    mkdir -p "$output_root"
    session=gb28_${dataset}_s${gpu}
    command="CUDA_VISIBLE_DEVICES=$gpu $python $script --dataset $dataset --dinov3-repo $hub --checkpoint-dir $weights --data-root '$data_root' --output-dir $output --vocabulary-config '$vocab' --methods G_all20_uniform,BoundedUnion --memory-fraction 0.20 --num-shards $shards --shard-index $gpu --progress-every 1"
    tmux new-session -d -s "$session" "cd $tool && export PYTHONPATH=$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1; mkdir -p '$output'; $command > '$output_root/s$gpu.log' 2>&1"
done
echo "launched $dataset on $shards GPUs"
