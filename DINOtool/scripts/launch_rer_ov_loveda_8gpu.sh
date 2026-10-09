#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/code/ovss/dino_ovss_training_free_20260924}"
OUTPUT="${OUTPUT:-${ROOT}/results/rer_ov_full_1669_8shard_20260925}"
PYTHON="${PYTHON:-/home/star/anaconda3/bin/python3}"
SESSION_PREFIX="${SESSION_PREFIX:-rerfull25_s}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

cd "${ROOT}"
mkdir -p "${OUTPUT}"
for gpu in $(seq 0 7); do
    session="${SESSION_PREFIX}${gpu}"
    shard_output="${OUTPUT}/s${gpu}"
    log="${OUTPUT}/s${gpu}.log"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "session already exists: ${session}" >&2
        exit 1
    fi
    if [[ -e "${shard_output}/signature.json" ]]; then
        echo "output already initialized: ${shard_output}" >&2
        exit 1
    fi
    command="cd '${ROOT}' && env CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=.parallel_readout_20260924:DINOtool '${PYTHON}' DINOtool/scripts/eval_rer_ov_loveda.py --dinov3-repo DINOtool/dinov3_hub --checkpoint-dir weights --data-root data/loveda/val --output-dir '${shard_output}' --max-images 0 --num-shards 8 --shard-index ${gpu} --progress-every '${PROGRESS_EVERY}' > '${log}' 2>&1"
    tmux new-session -d -s "${session}" "bash -lc \"${command}\""
    echo "launched ${session} on GPU ${gpu}"
done
