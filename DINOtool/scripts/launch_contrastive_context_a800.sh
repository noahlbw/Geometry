#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
checkpoint_dir=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
version=${1:-v1}
case "$version" in
    v1)
        run=$tool/results/contrastive_context_screen_20260929
        session_prefix=cc29
        evaluator=eval_contrastive_context.py
        ;;
    v2)
        run=$tool/results/contrastive_context_v2_screen_20260929
        session_prefix=cc29v2
        evaluator=eval_contrastive_context_v2.py
        ;;
    *) echo "Expected v1 or v2" >&2; exit 2 ;;
esac

if [[ -e "$run" ]]; then
    echo "Run directory already exists: $run" >&2
    exit 2
fi

datasets=(vdd potsdam udd5 oem)
limits=(16 32 16 64)
roots=(
    '/data/test/datasets/VDD Release Version/VDD'
    '/data/test/datasets/Potsdam/preprocessed_RGB'
    '/data/test/datasets/UDD5/extracted/UDD/UDD5'
    '/data/test/datasets/OpenEarthMap_wo_xBD'
)
vocabularies=(
    grounded_vdd20.json grounded_potsdam20.json
    hero_udd5_vip20.json hero_oem_vip20.json
)

for index in "${!datasets[@]}"; do
    gpu=$((index + 4))
    session=${session_prefix}_${datasets[index]}
    status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r used utilization <<< "$status"
    if (( used > 1024 || utilization > 10 )); then
        echo "GPU $gpu is busy: $status" >&2
        exit 3
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Session already exists: $session" >&2
        exit 4
    fi
done

mkdir -p "$run"
for index in "${!datasets[@]}"; do
    gpu=$((index + 4))
    dataset=${datasets[index]}
    output=$run/$dataset
    tmux new-session -d -s "${session_prefix}_$dataset" \
        "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/$evaluator --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$checkpoint_dir' --data-root '${roots[index]}' --vocabulary-config '$tool/configs/${vocabularies[index]}' --output-dir '$output' --memory-fraction 0.20 --max-images '${limits[index]}' --sample-seed 20260923 --progress-every 1 > '$run/$dataset.log' 2>&1"
done
echo "Started four contrastive-context screens under $run"
