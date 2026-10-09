#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python=/data/miniconda3/envs/pfu/bin/python
log=/data/test/datasets/isprs_preprocess28.log

while [[ ! -f /data/test/datasets/vai_inner28.log ]] || ! grep -q '^COMPLETE$' /data/test/datasets/vai_inner28.log; do
    sleep 15
done

$python - "$tool" > "$log" 2>&1 <<'PY'
from pathlib import Path
import sys

tool = Path(sys.argv[1])

def run(path: Path, replacements: dict[str, str]) -> None:
    source = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"Missing replacement anchor {old!r} in {path}")
        source = source.replace(old, new)
    namespace = {"__name__": "__main__"}
    exec(compile(source, str(path), "exec"), namespace, namespace)

run(
    tool / "../third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c/data/potsdam_preprocess.py",
    {
        'IMG_DIR = "/home/rfaulken/datasets/potsdam/2_Ortho_RGB"': 'IMG_DIR = "/data/test/datasets/Potsdam/2_Ortho_RGB"',
        'LAB_DIR = "/home/rfaulken/datasets/potsdam/labels"': 'LAB_DIR = "/data/test/datasets/Potsdam/labels_raw"',
        'OUTPUT_DIR = "/home/rfaulken/datasets/potsdam/preprocessed_RGB"': 'OUTPUT_DIR = "/data/test/datasets/Potsdam/preprocessed_RGB"',
    },
)
run(
    tool / "../third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c/data/vaihingen_preprocess.py",
    {
        'IMG_DIR = "/home/rfaulken/datasets/vaihingen/top"': 'IMG_DIR = "/data/test/datasets/Vaihingen/top"',
        'LAB_DIR = "/home/rfaulken/datasets/vaihingen/labels"': 'LAB_DIR = "/data/test/datasets/Vaihingen/gts_for_participants"',
        'OUTPUT_DIR = "/home/rfaulken/datasets/vaihingen/preprocessed_nobg"': 'OUTPUT_DIR = "/data/test/datasets/Vaihingen/preprocessed"',
    },
)
print("PREPROCESS_COMPLETE", flush=True)
PY
