# iSAID 统一结果汇总（标准 tile 协议）

协议：官方 validation RGB，896×896、overlap=384，11,644 tiles；16 类，background 计入，255 ignore。

| Model | mIoU (%) | Median ms | Peak allocated MiB | Accuracy source | Cost source |
|---|---:|---:|---:|---|---|
| VIP official | 26.1325 | 127.34 | 1816.8 | summary.json | vip/official_tiles_r3.json |
| Geov2 Anchored（best alias bank） | **28.9181** | 167.50† | 1702.2† | `isaid_alias_g0_screen_20261017/refine_four_full.json` | best full-label alias upper bound |
| Geov2 Anchored（formal frozen） | 27.0134 | 167.50 | 1702.2 | summary.json | geov2/anchored_tiles_cost_20261017.json |
| SegEarth-OV（official cfg_iSAID） | **23.4801** | **290.34** | **1154.76** | `isaid_baselines_tiles16_segearth04_20261030/merged.json` | official `cfg_iSAID.py`, `prob_thd=0.4`; 1242 MiB reserved |
| PTC | 12.6610 | 248.60 | 716.2 | summary.json | native/PTC_cost.json |
| PEARL | 10.8161 | 65.30 | 476.6 | summary.json | native/PEARL_cost.json |
| CorrCLIP | 10.2868 | 308.79 | 2607.4 | summary.json | native/CorrCLIP_cost.json |
| GLA-CLIP | 10.0159 | 56.95 | 1033.9 | summary.json | native/GLA-CLIP_cost.json |

成本边界：native baseline 为固定 7 tile×7 repeats，含 BGR/resize/preprocess；Geov2 成本记录为 Anchored runner；VIP full-tile shared official cfg 的严格 7-tile 成本为 119.05 ms/tile、758.66 MiB allocated（`isaid_baselines_segearth04_20261030/vip_cost_tiles7_shared.json`）。†best alias bank 精度沿用 Anchored runner 成本，未单独记录 best-bank cost；效率排序只能在同一成本边界内解释。458 张原始验证图的 SegEarth full-image sanity 为 18.9291%，不与本表 tile mIoU 混用；旧 SegEarth 22.5194 使用 cfg_vdd(prob=.3)，已作废。

CorrCLIP 的论文报告值为 16.9%，但当前目录中的 10.2868% 是 canonical-only、`native_protocol_config=false` 的 fixed-transfer tile 复现；两者配置并非同一个官方运行点。最终跨模型主表使用论文报告值，local 数字只作为诊断附录。
