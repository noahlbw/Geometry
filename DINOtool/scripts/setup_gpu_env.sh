#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_CUDA_PYTHON="${PROJECT_ROOT}/../../envs/tessera-cu128/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${SHARED_CUDA_PYTHON}" ]]; then
  PYTHON_BIN="${SHARED_CUDA_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ is required"'
"${PYTHON_BIN}" -m venv --system-site-packages "${PROJECT_ROOT}/.venv"
"${PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip 'setuptools<82' wheel
if ! "${PROJECT_ROOT}/.venv/bin/python" -c 'import torch, torchvision; assert torch.cuda.is_available()'; then
  "${PROJECT_ROOT}/.venv/bin/python" -m pip install \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi
"${PROJECT_ROOT}/.venv/bin/python" -m pip install -e "${PROJECT_ROOT}[dev]"

"${PROJECT_ROOT}/.venv/bin/dino-ovss" inspect
