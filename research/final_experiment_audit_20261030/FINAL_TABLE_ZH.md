# Geov2 最终实验表（按用户指定的最佳配置口径，2026-10-30）

主比较使用每个协议的 Geov2 最佳 alias bank/g0；`formal` 同时保留开发集冻结后的可复现实验值。带 `*` 的 best 值来自全量标签扫描/选择，只能称为标签选择上界。VIP 主比较列使用论文官方报告值，不用本地优化/迁移结果。

## 主比较（Geov2 best* vs VIP 官方报告）

| 协议 | Geov2 best* | Geov2 formal | 最佳 bank | g0 | VIP 官方 paper | CorrCLIP paper | PEARL paper | GLA-CLIP paper | PTC paper | SegEarth paper |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| VDD | **59.9810** | 59.9810 | `official_imagenet` | 1.0 | 54.3 | 37.7 | — | 45.0 | — | 45.3 |
| Potsdam | **55.4424** | 54.4390 | `official_imagenet` | 0.75 | 49.8 | 42.6 | — | — | — | 47.1 |
| UDD5 | **55.2575** | 55.2575 | `kev60_segmentation` | 0.5 | — | — | — | — | — | 50.6 |
| OEM | **44.7355** | 44.7355 | `original` | 0.25 | — | — | — | — | — | 40.3 |
| Vaihingen | **58.8021** | 58.8021 | `kev40_imagenet` | 0.25 | 47.0 | 24.7 | — | 31.9 | — | 29.1 |
| LandCover.ai | **68.9235** | 68.9235 | `original_imagenet` | 0.5 | — | — | — | — | — | — |
| LoveDA-P | **70.0386** | 70.0386 | `original` | 0.25 | — | — | — | — | — | — |
| LoveDA-D | **45.8370** | 45.8370 | `original` | 0.25 | — | — | — | — | — | — |
| FLAIR1 | **44.1187** | 44.1187 | `original` | 0.25 | — | — | — | — | — | — |
| VOC20 | **93.2805** | 93.2804 | `semantic_imagenet` | 1.0 | 92.5 | 88.8 | 86.9 | 84.7 | 82.7 | — |
| VOC21 | **75.2094** | 74.6882 | `official_segmentation` | 1.0 | 73.2 | 74.8 | 64.1 | 66.7 | 67.2 | — |
| ADE150 | **31.4852** | 31.4852 | `semantic_segmentation` | 0.25 | 29.1 | 26.9 | 19.4 | 20.5 | 20.9 | — |
| Context60 | **44.5929** | 44.5927 | `mix_f6f525631e7f743d226c` | 0.75 | 42.6 | 44.2 | 35.1 | 36.3 | 36.2 | — |
| Context59 | **46.7853** | 46.7853 | `official_segmentation` | 0.75 | — | — | — | — | — | — |
| COCO-Object81 | **49.6209** | 49.6209 | `official_imagenet` | 0.5 | 47.4 | 43.7 | 37.3 | 37.7 | 38.6 | — |
| COCO-Stuff171 | **33.7882** | 33.4209 | `official_segmentation` | 0.75 | 33.3 | 31.6 | 26.3 | 27.2 | 27.5 | — |
| Cityscapes | **57.2676** | 57.2676 | `city_dev_label_selected_alias_bank` | 0.75 | 55.7 | 49.4 | — | 41.2 | 40.0 | — |
| iSAID | **28.9181** | 27.0134 | `refine_four_segmentation (best); kev60_segmentation (formal)` | 0.5 | 26.1 | 16.9 | — | — | — | 21.7 |

## Alias bank 文件位置

| 协议 | bank | 文件 |
|---|---|---|
| VDD | `official_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/vdd.json` |
| Potsdam | `official_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/potsdam.json` |
| UDD5 | `kev60_segmentation` | `research/query_slot_best_alias_20261013/vocabularies/udd5.json` |
| OEM | `original` | `research/query_slot_best_alias_20261013/vocabularies/oem.json` |
| Vaihingen | `kev40_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/vaihingen.json` |
| LandCover.ai | `original_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/landcoverai.json` |
| LoveDA-P/D | `original` | `research/query_slot_best_alias_20261013/vocabularies/loveda.json` |
| FLAIR1 | `original` | `research/query_slot_best_alias_20261013/vocabularies/flair1.json` |
| VOC20 | `semantic_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/voc20.json` |
| VOC21 | `official_segmentation` | `research/query_slot_best_alias_20261013/vocabularies/voc21.json` |
| ADE150 | `semantic_segmentation` | `research/query_slot_best_alias_20261013/vocabularies/ade150.json` |
| Context60 | `mix_f6f525631e7f743d226c` | `research/query_slot_best_alias_20261013/vocabularies/context60.json` |
| Context59 | `official_segmentation` | `research/query_slot_best_alias_20261013/vocabularies/context59.json` |
| COCO-Object81 | `official_imagenet` | `research/query_slot_best_alias_20261013/vocabularies/coco_object81.json` |
| COCO-Stuff171 | `official_segmentation` | `research/query_slot_best_alias_20261013/vocabularies/coco_stuff171.json` |
| Cityscapes | `city_dev_label_selected_alias_bank` | `research/cityscapes_20261028/artifacts/city_dev_label_selected_alias_bank.json` |
| iSAID best | `refine_four_segmentation` | `research/isaid_alias_g0_screen_20261017/hybrid_refine/isaid_vocabulary_refine.json` |
| iSAID formal | `kev60_segmentation` | `research/isaid_alias_g0_screen_20261017/configs/isaid_vocabulary_kev60.json` |

## Geov2 成本（A800）

| 协议 | Median ms/image | Peak allocated MiB | Peak reserved MiB | 成本来源/边界 |
|---|---:|---:|---:|---|
| VDD | 295.89 | 2643.89 | 3254 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| Potsdam | 152.01 | 2143.30 | 2750 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| UDD5 | 291.99 | 2276.79 | 3150 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| OEM | 150.22 | 2170.24 | 2780 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| Vaihingen | 153.20 | 2153.60 | 2750 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| LandCover.ai | 144.91 | 2159.30 | 2752 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| LoveDA-P | 147.32 | 2187.57 | 2778 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| LoveDA-D | 147.83 | 2187.57 | 2778 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| FLAIR1 | 143.36 | 2183.50 | 2868 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| VOC20 | 143.72 | 2253.26 | 2998 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| VOC21 | 139.99 | 2247.24 | 2958 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| ADE150 | 169.33 | 3448.46 | 6876 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| Context60 | 152.06 | 2622.49 | 3422 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| Context59 | 146.22 | 2440.32 | 3280 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| COCO-Object81 | 151.16 | 2722.19 | 4168 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| COCO-Stuff171 | 173.78 | 3324.08 | 6728 | `research/query_slot_best_alias_20261013/RESULTS.json` |
| Cityscapes | 133.69 | 1574.68 | 1838 | `research/cityscapes_baselines_20261030/geov2_shared_visual_cost_tau08_exact.json` |
| iSAID | 167.50 | 1702.19 | — | `research/isaid_baselines_tiles16_20261017/geov2/anchored_tiles_cost_20261017.json` |

Cityscapes 成本为固定 7 图；iSAID 的 Geov2/VIP 26–28% 结果采用同一 11,644-tile runner，VIP shared official cfg 为 119.05 ms/tile、758.66 MiB allocated，Geov2 为 167.50 ms/tile、1702.19 MiB allocated。SegEarth 官方 cfg_iSAID 的 18.9291% 采用 458 张原始验证图，协议不同，不能与 tile runner 做严格 head-to-head。iSAID best alias bank 的精度来自 `research/isaid_alias_g0_screen_20261017/refine_four_full.json`；该 bank 的单独成本未另存，因此成本表列正式 Anchored runner 成本并明确标注。

## 本地复现值（附录，不替代 VIP 官方 paper）

| 协议 | VIP local | CorrCLIP local | PEARL local | GLA-CLIP local | PTC local | SegEarth local |
|---|---:|---:|---:|---:|---:|---:|
| VDD | 52.5765 | 37.1270 | 37.9121 | 37.8462 | 41.1191 | 44.8106 |
| Potsdam | 44.8995 | 32.2386 | 31.5545 | 34.8473 | 33.6962 | 48.1943 |
| UDD5 | 44.9713 | 40.0991 | 43.3762 | 47.7428 | 39.8159 | 50.5568 |
| OEM | 35.0388 | 24.4936 | 27.7070 | 30.7688 | 29.4806 | 35.6405 |
| Vaihingen | 49.1039 | 25.0724 | 26.1222 | 28.5356 | 25.8137 | 29.2379 |
| LandCover.ai | 54.9420 | 30.8354 | 30.9757 | 31.8306 | 33.0437 | 32.7043 |
| LoveDA-P | 56.3110 | 43.7985 | 47.5212 | 50.6544 | 50.6271 | 53.4885 |
| LoveDA-D | 36.1457 | 26.6162 | 29.6031 | 31.0830 | 31.1501 | 36.8928 |
| FLAIR1 | 36.6074 | 20.3994 | 19.3884 | 19.4995 | 20.7257 | 18.2782 |
| VOC20 | 92.5051 | 87.5405 | 86.9143 | 84.2087 | 82.7334 | 73.0056 |
| VOC21 | 73.2591 | 73.9381 | 64.0821 | 66.2608 | 67.1752 | 57.1139 |
| ADE150 | 29.4630 | 26.4301 | 19.3537 | 20.0191 | 20.9090 | 17.8261 |
| Context60 | 42.5987 | 45.1164 | 36.5263 | 37.6621 | 37.7346 | 32.6773 |
| Context59 | 45.9359 | 47.7036 | 38.7213 | 40.0670 | 40.5982 | 35.5333 |
| COCO-Object81 | 48.9955 | 44.6265 | 38.6592 | 39.0724 | 39.6947 | 31.2886 |
| COCO-Stuff171 | 33.4940 | 31.8365 | 26.3142 | 26.9368 | 27.4517 | 23.3325 |
| Cityscapes | 58.7766 | 46.9745 | 37.7867 | 41.2743 | 40.1996 | 34.1768 |
| iSAID（tile local） | 26.1325 | 10.2868 | 10.8161 | 10.0159 | 12.6610 | — |
| iSAID（458 full local） | 20.2096 | — | — | — | — | **18.9291** |

Cityscapes VIP local 是 released code 默认配置（336/112），不用于 VIP 官方 paper 主比较。iSAID tile local 的成本来自 `research/isaid_baselines_segearth04_20261030/vip_cost_tiles7_shared.json`；458 full local 的 20.2096 使用原始图协议且 text tower lifecycle 不同。SegEarth local=18.9291 使用官方 `cfg_iSAID.py` 的 `prob_thd=0.4`；旧的 22.5194 使用 VDD 配置，已作废。外部 local 具体成本见 [TRAINING_FREE_BASELINES_FULL_20261011.md](../TRAINING_FREE_BASELINES_FULL_20261011.md)。

## 解释边界

1. “best” 是你要求的最佳 alias bank/g0 口径；其中 iSAID 28.9181 的 bank 为 `refine_four_segmentation`，12 个类别保留 canonical，4 个类别加入明确 alias。
2. `formal` 是开发集冻结后再跑验证的可复现主结果；best* 允许全量标签选择，适合作为调参上界/内部最佳记录，不应在论文中伪装成无标签测试结果。
3. VIP 的 paper 列是官方论文值：Cityscapes 55.7、iSAID 26.1；本地 released code 值只放附录。
4. 论文和 GitHub 应同时保留 best* 与 formal 两列，避免把不同选择协议混成一个数字。

来源：`research/geov2_best_all_datasets_20261020/BEST_RESULTS.json`、`research/query_slot_best_alias_20261013/RESULTS.json`、`research/isaid_alias_g0_screen_20261017/refine_four_full.json`、`research/TRAINING_FREE_BASELINES_FULL_20261011.md`。
