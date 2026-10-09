#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: launch_gear_udd5_dynamic_a800.sh select|evaluate ARM GPU}
arm=${2:?arm required}
gpu=${3:?GPU required}
[[ "$gpu" =~ ^[0-7]$ ]] || exit 2
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
third_party=$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
data=/data/test/datasets/UDD5/extracted/UDD/UDD5
root=$tool/results/gear_udd5_dynamic_20260929
session=gud29_${stage}_${arm}

status=$(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader)
IFS=, read -r used utilization <<< "$status"
if [[ -n "$processes" ]] || (( used > 1024 || utilization > 10 )); then
    echo "GPU $gpu busy: $status" >&2
    exit 3
fi

case "$stage" in
    select)
        [[ "$arm" == dynamic ]] || exit 2
        output=$root/selection
        script=$tool/scripts/select_gear_dynamic_udd5.py
        vocabulary=$tool/configs/hero_udd5_vip20.json
        args="--images 40 --tiles-per-image 2 --sample-seed 20260929 --min-aliases 4 --z-threshold 1.0 --support-threshold 0.01" ;;
    evaluate)
        [[ "$arm" =~ ^[a-z0-9_]+$ ]] || exit 2
        output=$root/evaluations/$arm
        script=$tool/scripts/eval_gear_ov.py
        vocabulary=$root/vocabs/$arm.json
        args="--exact-alias-groups --progress-every 10" ;;
    *) exit 2 ;;
esac
if tmux has-session -t "$session" 2>/dev/null || [[ -e "$output" ]]; then
    echo "Session or output exists: $session / $output" >&2
    exit 4
fi
[[ -f "$script" && -f "$vocabulary" ]] || { echo "Missing script or vocabulary" >&2; exit 2; }
mkdir -p "$root/evaluations"
log=$root/${stage}_${arm}.log
command="cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$tool:$tool/scripts:$third_party' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 /data/miniconda3/envs/pfu/bin/python '$script' --dataset udd5 --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/ckpt/DINO' --data-root '$data' --vocabulary-config '$vocabulary' --output-dir '$output' $args > '$log' 2>&1"
tmux new-session -d -s "$session" "$command"
echo "Started $stage $arm on GPU $gpu."
