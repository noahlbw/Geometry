#!/usr/bin/env bash
# Wait for the resumable COCO download, then execute the M0/M1 gates once.
set -euo pipefail

base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$base/code/DINOtool
data=$base/datasets/COCOStuff2017
python=$base/envs/tessera-cu128/bin/python
output_root=$project/results/cafe_vc_coco/20260912_m0_m1

mkdir -p "$output_root"
while [[ ! -f "$data/logs/download_complete_at.txt" ]]; do
  sleep 60
done

"$python" "$project/scripts/prepare_coco_stuff.py" --root "$data" --split train --seed 20260912 2>&1 | tee "$output_root/prepare_train.log"
(
  cd "$project"
  "$python" -m pytest tests/test_coco_stuff.py tests/test_cafe_vc.py
) 2>&1 | tee "$output_root/unit_tests.log"

"$project/scripts/launch_cafe_coco_8gpu.sh" vc smoke "$output_root/vc_smoke" 2>&1 | tee "$output_root/vc_smoke.log"
