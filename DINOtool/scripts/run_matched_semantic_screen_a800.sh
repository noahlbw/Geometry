#!/usr/bin/env bash
set -euo pipefail
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
cd "$tool"
export PYTHONPATH="$tool:$tool/scripts:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Observe the already-running batch; an absent handle alone is not success.
while tmux has-session -t gsi03_suite_main 2>/dev/null; do sleep 30; done
for dataset in vdd potsdam udd5 oem vaihingen landcoverai; do
    [[ -f "$tool/results/geometry_semantic_innovation_v3_full_${dataset}_20261001/merged.json" ]] || {
        echo "V3 batch did not finish $dataset; refusing to proceed" >&2; exit 3;
    }
done
bash scripts/launch_matched_semantic_innovation_a800.sh full udd5 2 0
bash scripts/launch_matched_semantic_innovation_a800.sh full oem 2 2
for dataset in udd5 oem; do
    expected=40
    [[ "$dataset" != oem ]] || expected=384
    /data/miniconda3/envs/pfu/bin/python scripts/finish_matched_semantic_innovation.py \
        --dataset "$dataset" --mode full --shards 2 --expected-images "$expected" \
        --root "$tool/results/matched_counterfactual_v4_full_${dataset}_20261001"
done
bash scripts/launch_matched_semantic_innovation_a800.sh diagnostic vdd 2 0
bash scripts/launch_matched_semantic_innovation_a800.sh diagnostic potsdam 2 2
for dataset in vdd potsdam; do
    /data/miniconda3/envs/pfu/bin/python scripts/finish_matched_semantic_innovation.py \
        --dataset "$dataset" --mode diagnostic --shards 2 --expected-images 8 \
        --root "$tool/results/matched_counterfactual_v4_diagnostic_${dataset}_20261001"
done
echo 'Complete: matched semantic screen'
