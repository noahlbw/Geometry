#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python=/data/miniconda3/envs/pfu/bin/python
log=/data/test/datasets/vaihingen_preprocess_complete28.log

$python - "$tool" > "$log" 2>&1 <<'PY'
from pathlib import Path
import sys

tool = Path(sys.argv[1])
source_path = tool / "../third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c/data/vaihingen_preprocess.py"
source = source_path.read_text(encoding="utf-8")
for old, new in {
    'IMG_DIR = "/home/rfaulken/datasets/vaihingen/top"': 'IMG_DIR = "/data/test/datasets/Vaihingen/top"',
    'LAB_DIR = "/home/rfaulken/datasets/vaihingen/labels"': 'LAB_DIR = "/data/test/datasets/Vaihingen/labels_raw"',
    'OUTPUT_DIR = "/home/rfaulken/datasets/vaihingen/preprocessed_nobg"': 'OUTPUT_DIR = "/data/test/datasets/Vaihingen/preprocessed_complete"',
}.items():
    if old not in source:
        raise RuntimeError(f"Missing replacement anchor: {old}")
    source = source.replace(old, new)
namespace = {"__name__": "__main__"}
exec(compile(source, str(source_path), "exec"), namespace, namespace)
print("PREPROCESS_COMPLETE", flush=True)
PY
