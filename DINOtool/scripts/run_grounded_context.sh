#!/usr/bin/env bash
set -euo pipefail
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
base=/data/test/code/ovss_cafe_ped_v2_20260919
python_bin=/data/miniconda3/envs/pfu/bin/python
export PYTHONPATH="$tool:$tool/scripts:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$tool"
mode=${1:?test or run}
if [[ "$mode" == test ]]; then
    "$python_bin" - <<'PY'
import runpy
count = 0
for path in ['tests/test_grounded_context.py', 'tests/test_grounded_context_merge.py', 'tests/test_contextual_phrase_readout.py',
             'tests/test_contextual_phrase_shard_merge.py']:
    namespace = runpy.run_path(path)
    for name, function in namespace.items():
        if name.startswith('test_') and callable(function):
            function()
            count += 1
print(f'{count} tensor and merge tests passed')
PY
    exit 0
fi
dataset=${2:?dataset}
gpu=${3:?gpu}
run=${4:?run name}
shards=${5:-1}
index=${6:-0}
maximum=${7:-0}
case "$dataset" in
    loveda) data=/data/test/cafe-efa/data/processed/loveda/val; vocab=tcpr_loveda_vip_v1.json ;;
    udd5) data=/data/test/datasets/UDD5/extracted/UDD/UDD5; vocab=hero_udd5_vip20.json ;;
    oem) data=/data/test/datasets/OpenEarthMap_wo_xBD; vocab=hero_oem_vip20.json ;;
    *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$gpu"
exec nice -n 10 "$python_bin" scripts/eval_grounded_context.py \
    --dataset "$dataset" --dinov3-repo "$tool/dinov3_hub" \
    --checkpoint-dir "$base/ckpt/DINO" --data-root "$data" \
    --vocabulary-config "$tool/configs/$vocab" \
    --output-dir "$tool/results/$run/$dataset/s$index" \
    --num-shards "$shards" --shard-index "$index" --max-images "$maximum" \
    --progress-every 4
