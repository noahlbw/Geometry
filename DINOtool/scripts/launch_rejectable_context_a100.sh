#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Use smoke or screen}
base=/data/code/ovss/dino_ovss_training_free_20260924
tool=$base/DINOtool
python_bin=/data/miniconda3/envs/pfu/bin/python
export PYTHONPATH="$tool:$tool/scripts:$base/.parallel_readout_20260924:$base/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1

case "$mode" in
    smoke) minimum_gpus=1; shards=1; udd5_limit=1; oem_limit=0;
           run=$tool/results/rejectable_context_smoke_20260928 ;;
    screen) minimum_gpus=7; shards=4; udd5_limit=16; oem_limit=64;
            run=$tool/results/rejectable_context_screen_20260928 ;;
    *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac

if [[ -e "$run" ]]; then
    echo "Refusing to reuse existing run: $run" >&2
    exit 3
fi

readarray -t gpu_status < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
idle_gpus=()
for row in "${gpu_status[@]}"; do
    IFS=, read -r index used utilization <<< "$row"
    index=${index//[[:space:]]/}
    used=${used//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if [[ "$used" -le 1024 && "$utilization" -le 10 ]]; then
        idle_gpus+=("$index")
    else
        echo "GPU $index is occupied ($used MiB, $utilization%); leaving it untouched" >&2
    fi
done
if (( ${#idle_gpus[@]} < minimum_gpus )); then
    echo "Need $minimum_gpus idle GPUs, found ${#idle_gpus[@]}; launch deferred" >&2
    exit 4
fi

mkdir -p "$run/udd5" "$run/oem"
cd "$tool"
if [[ "$mode" == smoke ]]; then
    CUDA_VISIBLE_DEVICES=${idle_gpus[0]} "$python_bin" scripts/eval_rejectable_context.py \
        --dataset udd5 --dinov3-repo "$tool/dinov3_hub" \
        --checkpoint-dir "$base/weights" --data-root /data/datasets/UDD5 \
        --vocabulary-config "$tool/configs/gar_llm_raw20_udd5_v1.json" \
        --output-dir "$run/udd5/s0" --max-images "$udd5_limit" \
        --progress-every 1
    exit
fi

for ((shard=0; shard<shards; shard++)); do
    for dataset in udd5 oem; do
        if [[ "$dataset" == udd5 ]]; then
            gpu_index=$shard
            limit=$udd5_limit
            data=/data/datasets/UDD5
            vocab=gar_llm_raw20_udd5_v1.json
            session=rcv1s_u$shard
        else
            gpu_index=$((shard + shards))
            limit=$oem_limit
            data=/data/datasets/OpenEarthMap_wo_xBD
            vocab=gar_llm_raw20_oem_v1.json
            session=rcv1s_o$shard
        fi
        if (( gpu_index >= ${#idle_gpus[@]} )); then
            continue
        fi
        gpu=${idle_gpus[$gpu_index]}
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "Session already exists: $session" >&2
            exit 5
        fi
        output=$run/$dataset/s$shard
        logfile=$run/$dataset/s$shard.log
        tmux new-session -d -s "$session" \
            "cd '$tool' && exec env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH='$PYTHONPATH' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_rejectable_context.py --dataset '$dataset' --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/weights' --data-root '$data' --vocabulary-config '$tool/configs/$vocab' --output-dir '$output' --max-images '$limit' --sample-seed 20260923 --num-shards '$shards' --shard-index '$shard' --progress-every 2 > '$logfile' 2>&1"
    done
done
if (( ${#idle_gpus[@]} == 7 )); then
    if tmux has-session -t rcv1s_o3 2>/dev/null; then
        echo "Session already exists: rcv1s_o3" >&2
        exit 5
    fi
    queued="for s in 0 1 2 3; do while tmux has-session -t rcv1s_u\$s 2>/dev/null; do sleep 20; done; done; for s in 0 1 2 3; do test -f '$run/udd5/s'\$s'/results.json' || exit 12; done; cd '$tool' && exec env CUDA_VISIBLE_DEVICES=${idle_gpus[0]} PYTHONPATH='$PYTHONPATH' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 nice -n 10 '$python_bin' scripts/eval_rejectable_context.py --dataset oem --dinov3-repo '$tool/dinov3_hub' --checkpoint-dir '$base/weights' --data-root /data/datasets/OpenEarthMap_wo_xBD --vocabulary-config '$tool/configs/gar_llm_raw20_oem_v1.json' --output-dir '$run/oem/s3' --max-images '$oem_limit' --sample-seed 20260923 --num-shards '$shards' --shard-index 3 --progress-every 2"
    tmux new-session -d -s rcv1s_o3 "$queued > '$run/oem/s3.log' 2>&1"
fi
echo "Started rejectable context screen in $run"
