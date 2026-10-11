#!/usr/bin/env bash
# Dedicated environment: never install into the retained Geometry environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE="$ROOT/results/training_free_baselines_full_20261011"
ENV="$ROOT/.baseline_env_20261011"
mkdir -p "$SUITE"
if [ ! -x "$ENV/bin/python" ]; then
  /data/miniconda3/bin/python -m venv "$ENV"
fi
PY="$ENV/bin/python"
"$PY" -m pip install 'pip<26' 'setuptools<81' wheel
"$PY" -m pip install torch==2.4.0 torchvision==0.19.0
"$PY" -m pip install 'numpy==1.26.4' 'mmengine==0.10.7' 'mmsegmentation==1.2.2' 'mmcv==2.2.0' --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
"$PY" -m pip install 'numpy==1.26.4' 'transformers==4.44.2' 'timm==1.0.9' 'opencv-python==4.8.1.78' 'opencv-python-headless==4.8.1.78' ftfy regex tqdm einops scipy scikit-image matplotlib rasterio tifffile 'imagecodecs==2024.9.22' fairscale omegaconf hydra-core iopath pycocotools 'yapf==0.40.1' prettytable segment-anything 'openpyxl==3.1.5' 'scikit-learn==1.5.2' nltk ipdb 'kornia==0.7.3'
# The compiled wheel is available for torch 2.4 as MMCV 2.2. MMSeg 1.2.2's
# conservative version guard otherwise rejects it before importing any model.
# Relax ONLY that upper bound; no model, operator or numerical rule is replaced.
"$PY" "$ROOT/scripts/training_free_baseline_runtime.py" --allow-mmcv22 --output "$SUITE/runtime.json"
"$PY" -c 'from mmseg.models.segmentors import BaseSegmentor; from mmcv.ops import point_sample; import torch; print("BASELINE_RUNTIME_READY", torch.__version__, torch.cuda.is_available())'
touch "$SUITE/environment_ready"
