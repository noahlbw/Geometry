# Cityscapes 对比实验审计（2026-10-30）

协议：val 500 张 leftImg8bit 图像、19 个 trainIds、void=255 忽略。外部模型使用 pinned upstream Cityscapes 配置；成本为 7 张固定图、2 次热身后 7 次 CUDA 同步重复，排除磁盘解码、模型初始化和 target 读取。

| Model | mIoU (%) | Median ms/image | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|
| **Geov2 Anchored** | **57.2676** | **133.69** | **1574.68** | **1838.00** |
| VIP official（paper reported） | 55.7 | — | — | — |
| VIP official（released code local） | 58.7766 | 734.02 | 2348.47 | — |
| CorrCLIP（paper reported） | 49.4 | — | — | — |
| CorrCLIP（official City config local） | 46.9745 | 785.81 | 2611.28 | 4632.00 |
| GLA-CLIP（official City config local） | 41.2743 | 263.48 | 7124.57 | 7980.00 |
| PTC（official City config local） | 40.1996 | 544.35 | 980.56 | 1080.00 |
| PEARL（official City config local） | 37.7867 | 218.29 | 736.96 | 886.00 |
| SegEarth-OV（fixed transfer local） | 34.1768 | 675.22 | 1166.57 | 1264.00 |

Geov2 accuracy: `../cityscapes_20261028/artifacts/anchored_strict_tau08/full500_results.json`. Geov2 cost: `geov2_shared_visual_cost_tau08_exact.json`. VIP paper 55.7 and released-code local 58.7766 are separate quantities; the latter is not the paper table value.
