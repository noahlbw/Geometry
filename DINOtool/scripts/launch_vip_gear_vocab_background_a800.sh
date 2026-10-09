#!/usr/bin/env bash
# Full VDD/Potsdam factorial: readout x vocabulary, with both threshold states per run.
set -euo pipefail

dataset=${1:?usage: launch_vip_gear_vocab_background_a800.sh DATASET vip|gear official|all20 GPU}
readout=${2:?readout required}
vocabulary=${3:?vocabulary required}
gpu=${4:?GPU required}

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
upstream=$base/third_party/VIP_official_5bd25ee
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
run=$tool/results/vip_gear_vocab_background_20260930/$dataset/${readout}_${vocabulary}
session=vgt30_${dataset}_${readout}_${vocabulary}

[[ "$dataset" == vdd || "$dataset" == potsdam ]] || exit 2
[[ "$readout" == vip || "$readout" == gear ]] || exit 2
[[ "$vocabulary" == official || "$vocabulary" == all20 ]] || exit 2
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2

if [[ "$dataset" == vdd ]]; then
    data='/data/test/datasets/VDD Release Version/VDD'
    official=$upstream/configs/cls_vdd.txt
    all20=$tool/configs/grounded_vdd_official20.json
else
    data=/data/test/datasets/Potsdam/preprocessed_RGB
    official=$upstream/configs/cls_potsdam.txt
    all20=$tool/configs/grounded_potsdam20.json
fi
if [[ "$vocabulary" == official ]]; then
    vocab=$official
else
    vocab=$all20
fi

[[ -d "$data" && -f "$vocab" && -d "$upstream" ]] || {
    echo 'Missing data, vocabulary, or pinned VIP source.' >&2
    exit 2
}
if tmux has-session -t "$session" 2>/dev/null || [[ -e "$run" ]]; then
    echo "Session or output already exists: $session / $run" >&2
    exit 4
fi
status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
IFS=, read -r used utilization <<< "$status"
if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
    echo "GPU $gpu is busy: $status" >&2
    exit 3
fi

if [[ "$readout" == vip ]]; then
    evaluator=scripts/eval_vip_official_eight.py
    extra=(--upstream-root "$upstream" --vocabulary-source "$vocabulary")
    if [[ "$vocabulary" == all20 ]]; then
        extra=(--upstream-root "$upstream" --vocabulary-source json)
    fi
else
    evaluator=scripts/eval_gear_vocab_background.py
    extra=()
    if [[ "$vocabulary" == official ]]; then
        extra=(--vip-official-classes)
    fi
fi
[[ -f "$tool/$evaluator" ]] || { echo "Missing evaluator: $evaluator" >&2; exit 2; }
mkdir -p "$run"
printf -v command '%q ' "$python_bin" "$evaluator" --dataset "$dataset" \
    --dinov3-repo "$tool/dinov3_hub" --checkpoint-dir "$base/ckpt/DINO" \
    --data-root "$data" --vocabulary-config "$vocab" --output-dir "$run" \
    --progress-every 5 "${extra[@]}"
if [[ "$readout" == vip ]]; then
    command+=" --background-ablation"
fi
tmux new-session -d -s "$session" \
    "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 $command > '$run.log' 2>&1"
echo "Started $session on GPU $gpu."
