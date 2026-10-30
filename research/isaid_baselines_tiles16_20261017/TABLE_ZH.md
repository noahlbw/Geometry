# iSAID 统一结果汇总（标准 tile 协议）

协议：官方 validation RGB，896×896、overlap=384，11,644 tiles；16 类，background 计入，255 ignore。

| Model | mIoU (%) | Median ms | Peak allocated MiB | Accuracy source | Cost source |
|---|---:|---:|---:|---|---|
| VIP official | 26.1325 | 127.34 | 1816.8 | summary.json | vip/official_tiles_r3.json |
| Geov2 Anchored（best alias bank） | **28.9181** | 167.50† | 1702.2† | `isaid_alias_g0_screen_20261017/refine_four_full.json` | best full-label alias upper bound |
| Geov2 Anchored（formal frozen） | 27.0134 | 167.50 | 1702.2 | summary.json | geov2/anchored_tiles_cost_20261017.json |
| SegEarth-OV（official cfg_iSAID） | **18.9291** | **295.53** | **2596.45** | `isaid_baselines_segearth04_20261030/merged.json` | official prob_thd=0.4; 3042 MiB reserved |
| PTC | 12.6610 | 248.60 | 716.2 | summary.json | native/PTC_cost.json |
| PEARL | 10.8161 | 65.30 | 476.6 | summary.json | native/PEARL_cost.json |
| CorrCLIP | 10.2868 | 308.79 | 2607.4 | summary.json | native/CorrCLIP_cost.json |
| GLA-CLIP | 10.0159 | 56.95 | 1033.9 | summary.json | native/GLA-CLIP_cost.json |

成本边界：native baseline 为固定 7 tile×7 repeats，含 BGR/resize/preprocess；Geov2 成本记录为 Anchored runner；VIP 为全 11,644 tile 聚合记录。†best alias bank 精度沿用 Anchored runner 成本，未单独记录 best-bank cost；效率排序只能在同一成本边界内解释。旧 SegEarth 22.5194 使用 cfg_vdd(prob=.3)，已作废。
