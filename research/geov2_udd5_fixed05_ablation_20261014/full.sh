cd /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
export PYTHONPATH=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926:/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
pids=()
CUDA_VISIBLE_DEVICES=1 /data/miniconda3/envs/pfu/bin/python -u /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/source/eval_geov2_udd5_fixed05.py --experiment /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014 --mode full --num-shards 4 --shard-index 0 > /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/full_s0.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=2 /data/miniconda3/envs/pfu/bin/python -u /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/source/eval_geov2_udd5_fixed05.py --experiment /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014 --mode full --num-shards 4 --shard-index 1 > /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/full_s1.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=3 /data/miniconda3/envs/pfu/bin/python -u /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/source/eval_geov2_udd5_fixed05.py --experiment /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014 --mode full --num-shards 4 --shard-index 2 > /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/full_s2.log 2>&1 &
pids+=($!)
CUDA_VISIBLE_DEVICES=4 /data/miniconda3/envs/pfu/bin/python -u /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/source/eval_geov2_udd5_fixed05.py --experiment /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014 --mode full --num-shards 4 --shard-index 3 > /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_udd5_fixed05_ablation_20261014/full_s3.log 2>&1 &
pids+=($!)
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
