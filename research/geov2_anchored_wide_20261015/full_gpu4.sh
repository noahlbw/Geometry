set -e
cd /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_anchored_wide_20261015/source:/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926:/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/scripts:/data/test/code/ovss_cafe_ped_v2_20260919/third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c
/data/miniconda3/envs/pfu/bin/python -u /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_anchored_wide_20261015/source/eval_geov2_anchored_wide.py --experiment /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_anchored_wide_20261015 --dataset ade150 --mode full --shard-index 1 --num-shards 3 > /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geov2_anchored_wide_20261015/full_ade150_s1.log 2>&1
