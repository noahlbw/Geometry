#!/usr/bin/env bash
# Run the post-v1 P/U × update-gain factorial without altering the v1 gate.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GATE_RUN_ROOT V1_EVALUATION_ROOT FACTORIAL_OUTPUT_ROOT" >&2
  exit 2
fi

gate_root=$1
v1_root=$2
factorial_root=$3
project=$(cd "$(dirname "$0")/.." && pwd)
bundle=/data/test/code/ovss_cafe_ped_v2_20260919
python=/data/miniconda3/envs/pfu/bin/python
official=$bundle/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
checkpoint=$bundle/ckpt/CAFe-DINO/weights.pth
bpe=$bundle/ckpt/DINO/bpe_simple_vocab_16e6.txt.gz
loveda=/data/test/cafe-efa/data/processed/loveda/val
report=$v1_root/report/report.json
query_root=$gate_root/query_lift

[[ -d $gate_root && -d $v1_root && -f $report && ! -e $factorial_root ]] || {
  echo "gate/v1 report must exist and factorial output must be fresh" >&2
  exit 2
}
[[ -x $python && -d $official && -f $checkpoint && -f $bpe && -d $loveda && -f $query_root/best_inference.pt ]] || {
  echo "missing locked artifact, source-selected query_lift checkpoint, or LoveDA validation root" >&2
  exit 2
}
"$python" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('preliminary_target_decision',{}).get('decision') == 'PROVISIONAL_GO_FOR_LIMITED_ATTRIBUTION_ONLY' else 3)" "$report" || {
  echo "v1 Q-Lift result did not pass the required provisional gate; factorial is forbidden" >&2
  exit 3
}
"$python" "$project/scripts/verify_cafe_qlift_gate_contract.py" --run-root "$query_root" --arm query_lift

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CAFE_OFFICIAL_ROOT=$official
export PYTHONPATH=$project:$project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$factorial_root"
for protocol in P D; do
  # New Q,Q is mandatory: it must reproduce the original v1 native argmax maps
  # before the remaining cells are interpreted by the analyzer.
  "$python" -m torch.distributed.run \
      --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
      "$project/scripts/eval_cafe_vc_protocols.py" \
      --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
      --data-root "$loveda" --weights "$query_root/best_inference.pt" \
      --protocol "$protocol" --qlift-audit native --output-dir "$factorial_root/$protocol/Q,Q" \
      --amp bf16 --memory-fraction 0.75
  for cell_mode in 'Q,I:pu-query-gain-image' 'I,Q:pu-image-gain-query' 'I,I:pu-image-gain-image'; do
      cell=${cell_mode%%:*}
      mode=${cell_mode#*:}
      "$python" -m torch.distributed.run \
          --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
          "$project/scripts/eval_cafe_vc_protocols.py" \
          --official-root "$official" --base-checkpoint "$checkpoint" --bpe-path "$bpe" \
          --data-root "$loveda" --weights "$query_root/best_inference.pt" \
          --native-reference-dir "$factorial_root/$protocol/Q,Q" \
          --protocol "$protocol" --qlift-audit "$mode" --output-dir "$factorial_root/$protocol/$cell" \
          --amp bf16 --memory-fraction 0.75
  done
done
"$python" "$project/scripts/analyze_cafe_qlift_pu_gain_factorial.py" \
    --v1-report "$report" --v1-native-root "$v1_root/native" --factorial-root "$factorial_root" \
    --output-dir "$factorial_root/report"
