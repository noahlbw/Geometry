#!/usr/bin/env bash
set -euo pipefail
group=${1:?main|flair}
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
case "$group" in
    main) datasets=(vdd potsdam udd5 oem vaihingen landcoverai loveda); first=0 ;;
    flair) datasets=(flair1); first=4 ;;
    *) exit 2 ;;
esac
cd "$tool"
export PYTHONPATH="$tool:$tool/scripts:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for dataset in "${datasets[@]}"; do
    bash scripts/launch_geometry_semantic_innovation_a800.sh full "$dataset" 4 "$first"
    /data/miniconda3/envs/pfu/bin/python scripts/finish_geometry_semantic_innovation.py \
        --dataset "$dataset" --shards 4 --root "$tool/results/geometry_semantic_innovation_v3_full_${dataset}_20261001"
done
echo "Complete: $group"
