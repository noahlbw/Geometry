#!/usr/bin/env bash
# Fixed source-only retraining with independent, pre-registered P/D evaluation.
set -euo pipefail

base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:?usage: run_cafe_vc_v2_experiment.sh FRESH_OUTPUT_ROOT}
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
mkdir -p "$output"
exec >> "$output/queue.log" 2>&1
stage=initializing
trap 'printf "failed: %s (line %s)\n" "$stage" "$LINENO" > "$output/stage.txt"' ERR
python=$base/envs/tessera-cu128/bin/python
official=$base/code/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$base/code/ckpt/CAFe-DINO/weights.pth
bpe=$base/code/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

set_stage() {
  stage=$1
  printf '%s\n' "$stage" > "$output/stage.txt"
  printf '%s %s\n' "$(date -Is)" "$stage"
}

eval_run() {
  local arm=$1 protocol=$2 destination=$3
  local data=$base/datasets/LoveDA/val
  local selector=(--weights "$output/full/$arm/best_inference.pt")
  [[ "$arm" != frozen ]] || selector=(--frozen)
  [[ "$protocol" != source-audit ]] || data=$base/datasets/COCOStuff2017
  "$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$project/scripts/eval_cafe_vc_protocols.py" \
    --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
    --data-root "$data" --output-dir "$destination" --protocol "$protocol" \
    --amp bf16 "${selector[@]}"
}

set_stage unit_tests
cd "$project"
"$python" -m pytest -q tests/test_coco_stuff.py tests/test_cafe_vc.py tests/test_cafe_vc_protocols.py

set_stage waiting_for_previous_v1_evaluation
while pgrep -f "$base/code/DINOtool/scripts/eval_cafe_vc_loveda.py" > /dev/null; do
  sleep 30
done

set_stage real_weight_equivalence
"$python" "$project/scripts/verify_cafe_vc_v2.py" --official-root "$official" \
  --checkpoint "$checkpoint" --bpe-path "$bpe" --data-root "$base/datasets/COCOStuff2017" \
  --output "$output/real_equivalence.json"

for arm in plain cost_only concat vc; do
  set_stage "smoke_$arm"
  bash "$project/scripts/launch_cafe_coco_8gpu.sh" "$arm" smoke "$output/smoke/$arm"
done

for protocol in P D source-audit; do
  set_stage "frozen_$protocol"
  eval_run frozen "$protocol" "$output/frozen/$protocol"
done

set_stage frozen_D_reproduction_check
"$python" -c 'import json,sys; current=json.load(open(sys.argv[1])); previous=json.load(open(sys.argv[2])); assert current["images"] == previous["images"] == 1669; assert current["window_size"] == previous["window_size"] == 448; assert current["stride"] == previous["stride"] == 224; delta=current["mean_iou_percent"]-previous["mean_iou_percent"]; print("frozen D reproduction delta (percentage points):", delta); assert abs(delta) < 0.05, "Frozen D changed: investigate evaluator before training"' \
  "$output/frozen/D/results.json" \
  "$base/code/DINOtool/results/cafe_rc/20260912_cafe_rc_oem_8gpu_v1/loveda_val_cafe_baseline_8gpu_v1/results.json"

for arm in plain cost_only concat vc; do
  set_stage "screen_$arm"
  bash "$project/scripts/launch_cafe_coco_8gpu.sh" "$arm" screen "$output/screen/$arm"
done

for arm in plain cost_only concat vc; do
  set_stage "full_$arm"
  bash "$project/scripts/launch_cafe_coco_8gpu.sh" "$arm" full "$output/full/$arm"
done

# All checkpoints are selected on source-dev before any adapted target scores.
for arm in plain cost_only concat vc; do
  set_stage "source_audit_$arm"
  eval_run "$arm" source-audit "$output/audit/$arm"
done

for protocol in P D; do
  for arm in plain cost_only concat vc; do
    set_stage "external_${protocol}_$arm"
    eval_run "$arm" "$protocol" "$output/loveda/$protocol/$arm"
  done
done
set_stage complete
