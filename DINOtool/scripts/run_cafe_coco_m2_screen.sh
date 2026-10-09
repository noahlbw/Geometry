#!/usr/bin/env bash
# Execute the pre-registered matched 1,000-update COCO screen on eight GPUs.
set -euo pipefail

base=/root/siton-data-95873922dc054c44bdb7101cec2c70bb
project=$base/code/DINOtool
output_root=$project/results/cafe_vc_coco/20260912_m2_screen

mkdir -p "$output_root"
for arm in plain cost_only concat vc; do
  "$project/scripts/launch_cafe_coco_8gpu.sh" "$arm" screen "$output_root/$arm" 2>&1 | tee "$output_root/$arm.log"
done
