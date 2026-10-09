#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
root='/data/test/datasets/VDD Release Version/VDD'
run=$tool/results/vdd_visual_head_official20_screen16_20260929
arms=(one_preserve one_block two_preserve two_block)
depths=(1 1 2 2)
policies=(preserve block preserve block)

if [[ -e "$run" ]]; then
    echo "Run directory already exists: $run" >&2
    exit 2
fi
for index in 0 1 2 3; do
    gpu=$((index + 4))
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    if (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    if tmux has-session -t "vhead29_${arms[index]}" 2>/dev/null; then
        echo "Session exists: vhead29_${arms[index]}" >&2
        exit 4
    fi
done

mkdir -p "$run"
for index in 0 1 2 3; do
    gpu=$((index + 4))
    output=$run/${arms[index]}
    tmux new-session -d -s "vhead29_${arms[index]}" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_grounded_context.py --dataset vdd --vdd-ontology official --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$weights' --data-root '$root' --vocabulary-config '$tool/configs/grounded_vdd_official20.json' --output-dir '$output' --memory-fraction 0.20 --max-images 16 --sample-seed 20260923 --geometry-depth '${depths[index]}' --prefix-policy '${policies[index]}' --progress-every 1 > '$run/${arms[index]}.log' 2>&1"
done
echo "Started four official VDD visual-head screen arms on GPUs 4-7 under $run"
