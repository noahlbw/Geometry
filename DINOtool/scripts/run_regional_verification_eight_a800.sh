#!/usr/bin/env bash
set -euo pipefail

tool=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
python_bin=/data/miniconda3/envs/pfu/bin/python
launcher="$tool/scripts/launch_regional_verification_a800.sh"
merge="$tool/scripts/merge_gear_ov_shards.py"
other=/data/test/datasets

datasets=(udd5 oem vdd potsdam vaihingen landcoverai loveda flair1)
totals=(40 384 80 504 113 1602 1669 15700)
roots=(
    "$other/UDD5/extracted/UDD/UDD5"
    "$other/OpenEarthMap_wo_xBD"
    "$other/VDD Release Version/VDD"
    "$other/Potsdam/preprocessed_RGB"
    "$other/Vaihingen/preprocessed_complete"
    "$other/LandCoverAI_v1_target_20260922/preprocessed_official512_gear29"
    /data/code/ovss/dino_ovss_training_free_20260924/data/loveda/val
    "$other/FLAIR1_target_20260922"
)
vocabs=(
    "$tool/configs/hero_udd5_vip20.json"
    "$tool/configs/hero_oem_vip20.json"
    "$tool/configs/grounded_vdd_official20.json"
    "$tool/configs/grounded_potsdam20.json"
    "$tool/configs/grounded_vaihingen20.json"
    "$tool/configs/gear_landcoverai_v1_20.json"
    /data/code/ovss/dino_ovss_training_free_20260924/DINOtool_gear_20260929/configs/gar_llm_raw20_loveda_v1.json
    "$tool/configs/gear_flair1_main12_20.json"
)

for index in "${!datasets[@]}"; do
    dataset=${datasets[$index]}
    run="$tool/results/geometry_regional_verification_full_${dataset}_20260930"
    echo "Starting $dataset; expected ${totals[$index]} images."
    bash "$launcher" "$dataset" "${roots[$index]}" "${vocabs[$index]}" 4 4 0 full
    while :; do
        active=0
        for shard in 0 1 2 3; do
            if tmux has-session -t "grv30_full_${dataset}_s${shard}" 2>/dev/null; then
                active=1
            fi
        done
        (( active == 0 )) && break
        sleep 15
    done

    inputs=()
    for shard in 0 1 2 3; do
        directory="$run/s$shard"
        if ! "$python_bin" -c 'import json,sys; from pathlib import Path; j=json.loads(Path(sys.argv[1]).read_text()); assert j["status"] == "complete" and j["processed_images"] == j["total_images"]' "$directory/results.json"; then
            echo "Incomplete $dataset shard $shard; inspect $run/s$shard.log" >&2
            exit 1
        fi
        inputs+=("$directory")
    done
    PYTHONPATH="$tool:$tool/scripts" "$python_bin" "$merge" \
        --inputs "${inputs[@]}" --output "$run/merged.json"
    "$python_bin" -c 'import json,sys; from pathlib import Path; j=json.loads(Path(sys.argv[1]).read_text()); n=int(sys.argv[2]); assert j["coverage_verified"] and j["processed_images"] == j["total_images"] == n; assert all(all(count == 20 for count in counts) for counts in j["signature"]["vocabulary"]["alias_counts"].values())' "$run/merged.json" "${totals[$index]}"
    echo "Verified $dataset: ${totals[$index]} unique complete images."
done

echo 'All eight regional verification evaluations completed and merged.'
