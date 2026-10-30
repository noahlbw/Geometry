# Cityscapes / iSAID 对比实验审计（2026-10-30）

## 协议

- **Cityscapes**：val 500 张 leftImg8bit 图像，19 个 trainIds，void=255 忽略。外部模型使用官方 Cityscapes 配置；成本为 7 张固定图、2 次热身后 7 次 CUDA 同步重复，排除磁盘解码、模型初始化和 target 读取。
- **iSAID（Geov2/VIP）**：官方 validation tile 协议，896×896、overlap=384、11,644 tiles，16 类（含 background），255 ignore。
- **iSAID（SegEarth official cfg）**：458 张原始 validation 图，官方 cfg_iSAID.py；该行只作为官方配置复核，不与 tile runner 的 mIoU 做严格同协议排序。
- 显存列统一为 peak allocated；reserved 另存于成本 JSON。

## Cityscapes

| Model | mIoU (%) | Median ms/image | Peak allocated MiB | Peak reserved MiB | 来源 |
|---|---:|---:|---:|---:|---|
| **Geov2 Anchored** | **57.2676** | **133.69** | **1574.68** | **1838.00** | strict full500 + shared-visual cost |
| VIP official（论文 reported） | 55.7 | — | — | — | VIP paper Cityscapes table |
| VIP official（released code local） | **58.7766** | **734.02** | **2348.47** | — | pinned released code, default 336/112 |
| CorrCLIP（论文 reported） | 49.4 | — | — | — | CorrCLIP paper table |
| CorrCLIP（official City config local） | 46.9745 | 785.81 | 2611.28 | 4632.00 | official City config |
| GLA-CLIP（official City config local） | 41.2743 | 263.48 | 7124.57 | 7980.00 | official City config |
| PTC（official City config local） | 40.1996 | 544.35 | 980.56 | 1080.00 | official City config |
| PEARL（official City config local） | 37.7867 | 218.29 | 736.96 | 886.00 | official City config |
| SegEarth-OV（fixed transfer local） | 34.1768 | 675.22 | 1166.57 | 1264.00 | no dedicated City config |

Geov2 精度：`research/cityscapes_20261028/artifacts/anchored_strict_tau08/full500_results.json`；Geov2 成本：`research/cityscapes_baselines_20261030/geov2_shared_visual_cost_tau08_exact.json`。VIP local 的 58.7766 是官方 released code 复现值，论文值 55.7 单独保留。

## iSAID

| Model | mIoU (%) | Median ms/tile | Peak allocated MiB | 来源 |
|---|---:|---:|---:|---|
| **Geov2 Anchored (best alias bank)** | **28.9181** | 167.50† | 1702.19† | `isaid_alias_g0_screen_20261017/refine_four_full.json` |
| Geov2 Anchored (formal frozen) | 27.0134 | 167.50 | 1702.19 | `isaid_baselines_tiles16_20261017/` |
| VIP official（论文 reported） | 26.1 | — | — | VIP paper iSAID table |
| VIP official（full tile local） | **26.1325** | **127.34** | **1816.76** | `vip_official_isaid_tiles16_20261017/official_tiles_r3.json` |
| SegEarth-OV（official cfg_iSAID） | **18.9291** | **295.53** | **2596.45** | official prob_thd=0.4; 3042 MiB reserved |
| PTC | 12.6610 | 248.60 | 716.22 | fixed transfer local |
| PEARL | 10.8161 | 65.30 | 476.60 | fixed transfer local |
| CorrCLIP | 10.2868 | 308.79 | 2609.10 | fixed transfer local |
| GLA-CLIP | 10.0159 | 56.95 | 1033.90 | fixed transfer local |

Geov2 最佳 alias bank 是 `refine_four_segmentation`，`g0=0.5`，全量 mIoU 为 28.9181%；formal 冻结 bank `kev60_segmentation` 为 27.0134%。†成本沿用同一 Anchored runner；best bank 只改变文本 bank，单独成本未另存。SegEarth 正式值为 18.9291%，来自 458 张原始 validation 图和官方 `cfg_iSAID.py`（`prob_thd=0.4`）；旧值 22.5194% 使用 VDD 配置，已作废。
