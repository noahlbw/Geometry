#!/usr/bin/env bash
# Run one matched-head screen or full evaluation on two idle A800 GPUs.
set -euo pipefail

dataset=${1:?usage: launch_matched_head_fov_a800.sh vdd|potsdam screen|full}
mode=${2:?screen or full required}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
upstream=$base/third_party/VIP_official_5bd25ee
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run=$tool/results/matched_head_fov_20260930/$mode/$dataset

case "$dataset" in
    vdd) first_gpu=4; data='/data/test/datasets/VDD Release Version/VDD'
         vocab=$tool/configs/grounded_vdd_official20.json ;;
    potsdam) first_gpu=6; data=/data/test/datasets/Potsdam/preprocessed_RGB
             vocab=$tool/configs/grounded_potsdam20.json ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
case "$mode" in
    screen) if [[ "$dataset" == vdd ]]; then max_images=16; else max_images=32; fi ;;
    full) max_images=0 ;;
    *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac
[[ -d "$data" && -f "$vocab" && -d "$upstream" &&
   -f "$tool/scripts/eval_matched_head_fov.py" ]] || exit 2

for shard in 0 1; do
    gpu=$((first_gpu + shard))
    session=mhf30_${mode}_${dataset}_s${shard}
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
    IFS=, read -r used utilization <<< "$status"
    if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run/s$shard" ]]; then
        echo "Existing session or output: $session / $run/s$shard" >&2
        exit 4
    fi
done

for shard in 0 1; do
    gpu=$((first_gpu + shard))
    session=mhf30_${mode}_${dataset}_s${shard}
    mkdir -p "$run/s$shard"
    printf -v command '%q ' "$python_bin" scripts/eval_matched_head_fov.py \
        --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
        --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
        --vocabulary-config "$vocab" --upstream-root "$upstream" \
        --output-dir "$run/s$shard" --max-images "$max_images" \
        --num-shards 2 --shard-index "$shard" --progress-every 2
    tmux new-session -d -s "$session" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$run/s$shard.log' 2>&1"
done
echo "Started $mode $dataset on GPUs $first_gpu-$((first_gpu + 1))."
