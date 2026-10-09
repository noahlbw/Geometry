#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: launch_vip_alias_ablation_a800.sh select|evaluate}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
root=$tool/results/vip_alias_ablation_20260928
python=/data/miniconda3/envs/pfu/bin/python
weights=/data/test/code/ovss_cafe_ped_v2_20260919/ckpt/DINO
hub=$tool/dinov3_hub
script=$tool/scripts/eval_vip_alias_ablation_external.py
visual_udd=$tool/configs/hero_udd5_vip20.json
visual_oem=$tool/configs/hero_oem_vip20.json
prompt_udd=$tool/configs/vip_prompt_udd5_v1.json
prompt_oem=$tool/configs/vip_prompt_oem_v1.json

[[ $stage == select || $stage == evaluate ]] || { printf 'Unsupported stage: %s\n' "$stage" >&2; exit 2; }
for gpu in $(seq 0 7); do
    state=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    [[ $state == 0 ]] || { printf 'GPU %s is busy (%s%%)\n' "$gpu" "$state" >&2; exit 1; }
done
for session in vip28_${stage}_u0 vip28_${stage}_u1 vip28_${stage}_u2 vip28_${stage}_u3 vip28_${stage}_o0 vip28_${stage}_o1 vip28_${stage}_o2 vip28_${stage}_o3; do
    tmux has-session -t "$session" 2>/dev/null && { printf 'Session already exists: %s\n' "$session" >&2; exit 1; }
done

run_dataset() {
    local dataset=$1 gpu=$2 shard=$3 visual=$4 prompt=$5 data=$6
    local out=$root/$stage/$dataset/s$shard
    local selection=$root/select/$dataset/merged.json
    local session=vip28_${stage}_${dataset:0:1}$shard
    local command="CUDA_VISIBLE_DEVICES=$gpu $python $script --stage $stage --dataset $dataset --dinov3-repo $hub --checkpoint-dir $weights --data-root $data --output-dir $out --visual-vocabulary-config $visual --vip-prompt-vocabulary-config $prompt --memory-fraction 0.20 --num-shards 4 --shard-index $shard --progress-every 1"
    if [[ $stage == evaluate ]]; then
        [[ -f $selection ]] || { printf 'Missing selection: %s\n' "$selection" >&2; exit 1; }
        command+=" --selection-file $selection"
    fi
    mkdir -p "$root/$stage/$dataset"
    tmux new-session -d -s "$session" "cd $tool && export PYTHONPATH=$tool:$tool/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1; $command > $root/$stage/$dataset/s$shard.log 2>&1"
}

run_dataset udd5 0 0 "$visual_udd" "$prompt_udd" /data/test/datasets/UDD5/extracted/UDD/UDD5
run_dataset udd5 1 1 "$visual_udd" "$prompt_udd" /data/test/datasets/UDD5/extracted/UDD/UDD5
run_dataset udd5 2 2 "$visual_udd" "$prompt_udd" /data/test/datasets/UDD5/extracted/UDD/UDD5
run_dataset udd5 3 3 "$visual_udd" "$prompt_udd" /data/test/datasets/UDD5/extracted/UDD/UDD5
run_dataset oem 4 0 "$visual_oem" "$prompt_oem" /data/test/datasets/OpenEarthMap_wo_xBD
run_dataset oem 5 1 "$visual_oem" "$prompt_oem" /data/test/datasets/OpenEarthMap_wo_xBD
run_dataset oem 6 2 "$visual_oem" "$prompt_oem" /data/test/datasets/OpenEarthMap_wo_xBD
run_dataset oem 7 3 "$visual_oem" "$prompt_oem" /data/test/datasets/OpenEarthMap_wo_xBD
printf 'Launched VIP alias ablation %s stage on UDD5 and OEM.\n' "$stage"
