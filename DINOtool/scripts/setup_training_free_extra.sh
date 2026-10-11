#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE="$ROOT/results/training_free_baselines_full_20261011"
PY="$ROOT/.baseline_env_20261011/bin/python"
while [ ! -f "$SUITE/environment_ready" ] || [ ! -f "$SUITE/weights_ready" ]; do
  sleep 20
done
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
SAM2_BUILD_CUDA=0 "$PY" -m pip install --no-deps --no-build-isolation -e "$SUITE/dependencies/sam2"
"$PY" -m pip install ninja
"$PY" -m pip install --no-deps --no-build-isolation -e "$SUITE/dependencies/SimFeatUp"
"$PY" -c 'from sam2.build_sam import build_sam2; from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv; print("EXTRA_BASELINE_DEPENDENCIES_READY")'
touch "$SUITE/extra_dependencies_ready"
