#!/usr/bin/env bash
set -euo pipefail
tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
run=grounded_context_full_20260928
[[ ! -e "$tool/results/$run" ]] || { printf 'Run already exists\n' >&2; exit 1; }
for session in gc28_l0 gc28_l1 gc28_l2 gc28_l3 gc28_u0 gc28_u1 gc28_o0 gc28_o1; do
    if tmux has-session -t "$session" 2>/dev/null; then
        printf 'Session already exists: %s\n' "$session" >&2
        exit 1
    fi
done
mkdir -p "$tool/results/$run"
for shard in 0 1 2 3; do
    tmux new-session -d -s "gc28_l$shard" \
        "bash $tool/scripts/run_grounded_context.sh run loveda $shard $run 4 $shard 0 > $tool/results/$run/loveda_s$shard.log 2>&1"
done
for shard in 0 1; do
    gpu=$((shard+4))
    tmux new-session -d -s "gc28_u$shard" \
        "bash $tool/scripts/run_grounded_context.sh run udd5 $gpu $run 2 $shard 0 > $tool/results/$run/udd5_s$shard.log 2>&1"
    gpu=$((shard+6))
    tmux new-session -d -s "gc28_o$shard" \
        "bash $tool/scripts/run_grounded_context.sh run oem $gpu $run 2 $shard 0 > $tool/results/$run/oem_s$shard.log 2>&1"
done
printf 'Launched frozen 6-arm LoveDA 1669 / UDD5 40 / OEM 384 evaluations\n'
