# Geov3 论文实验表（2026-10-31 重跑）

本文件只记录 **Geov3** 的本轮结果。Geov3 的固定执行端点是：DINOv3 + DonorBefore 局部分支 + QuerySlot（K=24）宽视野分支 + Anchored 读出/写回（β=1），宽视野类内聚合使用 log-sum-exp；推理使用 BF16 冻结权重、fast assignment、FP32 结构求解和 batch=8。每个数据集只读取一个已经冻结的 bank、温度、背景规则和 `g0`。全量评测器不读取标签，标签只在最后累积 confusion matrix 和计算 mIoU 时使用。

主协议、bank、`g0`、样本总数和结果文件见 [`geov3_full_20261031/protocol.json`](../../artifacts/geometry_github/research/geov3_full_20261031/protocol.json) 与 [`DATASET_MANIFEST.json`](../../artifacts/geometry_github/research/geov3_full_20261031/DATASET_MANIFEST.json)。只有 `status=complete` 且 `processed_images == global_sample_count` 的结果才进入下表。

## Table 1. 自然图像主结果

| Dataset | PEARL | GLA-CLIP | PTC | CorrCLIP | SegEarth-OV | VIP | **Geov3** |
|---|---:|---:|---:|---:|---:|---:|---:|
| VOC20 | 86.9143 | 84.2087 | 82.7334 | 87.5405 | 73.0056 | 92.5051 | **93.4071** |
| VOC21 | 64.0821 | 66.2608 | 67.1752 | 73.9381 | 57.1139 | 73.2591 | **75.4482** |
| ADE150 | 19.3537 | 20.0191 | 20.9090 | 26.4301 | 17.8261 | 29.1387 | **31.6741** |
| Context59 | 38.7213 | 40.0670 | 40.5982 | **47.7036** | 35.5333 | — | **47.3866** |
| Context60 / PC60 | 36.5263 | 37.6621 | 37.7346 | **45.1164** | 32.6773 | 42.5987 | **44.8198** |
| COCO-Object81 | 38.6592 | 39.0724 | 39.6947 | 44.6265 | 31.2886 | 48.9955 | **49.7192** |
| COCO-Stuff171 | 26.3142 | 26.9368 | 27.4517 | 31.8365 | 23.3325 | 33.4867 | **33.3064** |
| **Mean (7 protocols)** | 44.3673 | 44.8896 | 45.1853 | 51.0274 | 38.6825 | 53.3306* | **53.6802** |

Geov3 的全量证据文件位于 `geov3_full_20261031/remote_completed/<dataset>_s0/results.json`。每行的完整样本数分别为 1,449、1,449、2,000、5,105、5,105、5,000、5,000。

## Table 2. 遥感与航拍主结果

| Dataset | PEARL | GLA-CLIP | PTC | CorrCLIP | SegEarth-OV | VIP | **Geov3** |
|---|---:|---:|---:|---:|---:|---:|---:|
| VDD | 37.9121 | 37.8462 | 41.1191 | 37.1270 | 44.8106 | 52.0647 | **61.2364** |
| Potsdam | 31.5545 | 34.8473 | 33.6962 | 32.2386 | 48.1943 | 44.8995 | **55.6684** |
| UDD5 | 43.3762 | 47.7428 | 39.8159 | 40.0991 | **50.5568** | 44.9713 | **56.0569** |
| OEM | 27.7070 | 30.7688 | 29.4806 | 24.4936 | 35.6405 | 35.0388 | **44.8435** |
| Vaihingen | 26.1222 | 28.5356 | 25.8137 | 25.0724 | 29.2379 | 47.3620 | **59.2932** |
| LandCover.ai | 30.9757 | 31.8306 | 33.0437 | 30.8354 | 32.7043 | 54.9420 | **68.1581** |
| LoveDA-P | 47.5212 | 50.6544 | 50.6271 | 43.7985 | 53.4885 | 56.3110 | **70.2428** |
| LoveDA-D | 29.6031 | 31.0830 | 31.1501 | 26.6162 | 36.8928 | 36.1457 | **46.1671** |
| FLAIR1 | 19.3884 | 19.4995 | 20.7257 | 20.3994 | 18.2782 | 36.6074 | **44.1914** |
| **Mean（9 个遥感协议）** | — | — | — | — | — | — | **56.2064** |

VDD、Potsdam、UDD5、OEM、Vaihingen、LandCover.ai、FLAIR1 的结果文件位于 `geov3_full_20261031/remote_completed/`；LoveDA 的修正后 P/D finalize runner 已覆盖全部 1,669 张图，结果为 `loveda_fixed/results.json`。

## Table 2b. iSAID 与 Cityscapes 附加协议

| Dataset / protocol | PEARL | GLA-CLIP | PTC | CorrCLIP | SegEarth-OV | VIP | **Geov3** | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| iSAID，标准 16 类、11,644 tiles | — | — | — | 16.9 | 23.4801 | 26.1 | **29.0883** | 最佳 `refine_four_segmentation` bank；全量完成 |
| Cityscapes，500 val images | — | 41.2 | 40.0 | 49.4 | 34.1768 | 55.7 | **56.5384** | Geov3 全量完成；真实 leftImg8bit val |

iSAID 最佳 bank 结果见 `geov3_full_20261031/isaid_bestbank_corrected_full.json`；旧 bank 的 27.0878% 仅保留作对照。Cityscapes 结果见 `geov3_full_20261031/cityscapes_geov3_results.json`。

## Table 2c. Geov3 统一速度与显存（A800-SXM4-80GB）

所有数字来自同一 `geov3_final` 端点、BF16、batch=8、fast assignment、Anchored 写回；每个数据集固定 7 张图、2 次 warmup、7 次同步重复，统计单张图 median latency 和进程峰值 allocated 显存。速度/显存测试不读取标签，也不包含模型初始化、文本编码和 RGB 解码。

| Dataset | Geov3 median ms / peak allocated MiB | peak reserved MiB | 状态 |
|---|---:|---:|---|
| VDD | 214.86 / 1362.19 | 1696 | complete |
| Potsdam | 67.08 / 896.58 | 1284 | complete |
| VOC20 | 60.00 / 952.20 | 1320 | complete |
| VOC21 | 60.29 / 938.81 | 1312 | complete |
| ADE150 | 81.44 / 1726.64 | 2400 | complete |
| Context60 / PC60 | 70.14 / 1112.45 | 1992 | complete |
| Context59 | 66.27 / 1096.56 | 1972 | complete |
| COCO-Object81 | 73.55 / 1270.47 | 1652 | complete |
| COCO-Stuff171 | 90.33 / 1728.61 | 3012 | complete |
| UDD5 | 221.15 / 1002.59 | 1366 | complete |
| OEM | 72.79 / 928.95 | 1316 | complete |
| Vaihingen | 70.63 / 904.15 | 1286 | complete |
| LandCover.ai | 62.72 / 909.73 | 1290 | complete |
| FLAIR1 | 58.99 / 954.15 | 1340 | complete |

原始文件位于 `artifacts/geometry_github/research/geov3_full_20261031/cost/geov3_final/<dataset>_s0/results.json`。`peak allocated` 用于主比较；`peak reserved` 作为运行时缓存上界一并保留。

## Table 3. Geov3 全量结果审计

| Dataset | Frozen bank | `g0` | 全量样本 | Geov3 mIoU (%) | 结果文件 |
|---|---|---:|---:|---:|---|
| VDD | official_imagenet | 1.00 | 80 | 61.2364 | `remote_completed/vdd_s0/results.json` |
| Potsdam | official_imagenet | 0.75 | 504 | 55.6684 | `remote_completed/potsdam_s0/results.json` |
| VOC20 | semantic_imagenet | 1.00 | 1,449 | 93.4071 | `remote_completed/voc20_s0/results.json` |
| VOC21 | official_segmentation | 1.00 | 1,449 | 75.4482 | `remote_completed/voc21_s0/results.json` |
| ADE150 | semantic_segmentation | 0.25 | 2,000 | 31.6741 | `remote_completed/ade150_s0/results.json` |
| Context60 | mix_f6f525631e7f743d226c | 0.75 | 5,105 | 44.8198 | `remote_completed/context60_s0/results.json` |
| Context59 | official_segmentation | 0.75 | 5,105 | 47.3866 | `remote_completed/context59_s0/results.json` |
| COCO-Object81 | official_imagenet | 0.50 | 5,000 | 49.7192 | `remote_completed/coco_object81_s0/results.json` |
| COCO-Stuff171 | official_segmentation | 0.50 | 5,000 | 33.3064 | `remote_completed/coco_stuff171_s0/results.json` |
| UDD5 | kev60_segmentation | 0.50 | 40 | 56.0569 | `remote_completed/udd5_s0/results.json` |
| OEM | original | 0.25 | 384 | 44.8435 | `remote_completed/oem_s0/results.json` |
| Vaihingen | kev40_imagenet | 0.25 | 113 | 59.2932 | `remote_completed/vaihingen_s0/results.json` |
| LandCover.ai | original_imagenet | 0.50 | 1,602 | 68.1581 | `remote_completed/landcoverai_s0/results.json` |
| FLAIR1 | original | 0.25 | 15,700 | 44.1914 | `remote_completed/flair1_s0/results.json` |

## Table 4. 整体路径消融（Geov3）

固定同一 Geov3 bank、文本、窗口和 `g0`，只改变最终路径：LocalOnly、WideOnly、等权 logit 融合和 Anchored 写回。

| Dataset | Local only | Wide only | Equal logit fusion | **Full Geov3** |
|---|---:|---:|---:|---:|
| VDD | 50.9495 | 59.3818 | 60.4198 | **61.1828** |
| Potsdam | 38.3740 | 46.1399 | 53.4771 | **55.6656** |
| ADE150 | 27.8541 | 27.4468 | 29.5030 | **31.6647** |
| VOC21 | 41.4950 | 64.9093 | 75.1675 | **75.4703** |

原始文件：`geov3_full_20261031/path_ablation/full/<dataset>/results.json`。该表用于机制消融；主表使用同一 Geov3 endpoint 的独立全量协议数字。

## Table 4a. Local/Wide/Geov3 的质量分解

同一 Geov3 端点、同一 frozen bank 和 `g0`，只改变最终路径。`Boundary IoU` 使用 GT 边界膨胀容差 2；`Small` 为 GT 连通区域面积 ≤1024 像素，`Large` 为面积 ≥16384 像素。所有指标只在有效标签上计算。

| Dataset / path | mIoU | Pixel Acc. | Boundary IoU | Small-object IoU | Large-region IoU |
|---|---:|---:|---:|---:|---:|
| VDD / Local | 50.9495 | 72.1189 | 2.5521 | 4.7058 | 35.3125 |
| VDD / Wide | 59.3876 | 83.0169 | 3.1897 | 5.0873 | 49.0490 |
| VDD / Equal fusion | 60.4207 | 82.6653 | 3.2740 | 4.2278 | 49.0850 |
| VDD / **Geov3** | **61.1828** | 82.9766 | **3.3932** | 4.2949 | 48.7803 |
| Potsdam / Local | 38.3740 | 54.5881 | 8.7159 | 15.7824 | 29.7380 |
| Potsdam / Wide | 46.1179 | 75.1803 | 9.4861 | 7.1070 | **50.3760** |
| Potsdam / Equal fusion | 53.4689 | 77.8462 | 10.4243 | 8.6489 | 46.5071 |
| Potsdam / **Geov3** | **55.6874** | **78.1591** | **11.0462** | **10.6751** | 44.4758 |
| ADE150 / Local | 27.8541 | 53.3023 | 18.5580 | 8.3798 | 6.7931 |
| ADE150 / Wide | 27.4568 | 63.2161 | 18.1155 | 7.8330 | **22.6755** |
| ADE150 / Equal fusion | 29.4908 | **64.7024** | 18.9298 | 8.7082 | 21.4455 |
| ADE150 / **Geov3** | **31.6649** | 64.1997 | **20.2420** | **9.5890** | 14.6005 |
| VOC21 / Local | 41.4950 | 82.4668 | 14.6729 | 32.4949 | 53.6405 |
| VOC21 / Wide | 64.9069 | 90.7116 | 21.0624 | 27.5901 | 57.5365 |
| VOC21 / Equal fusion | 75.1606 | 93.7010 | 23.1241 | **34.3479** | 60.3013 |
| VOC21 / **Geov3** | **75.4647** | **93.8637** | **23.2798** | 33.6604 | **61.1377** |

原始全量文件：`artifacts/geometry_github/research/geov3_path_diagnostics_20261101_corrected/full/<dataset>/results.json`。

## Table 4b. Local/Wide 正确性转移

四格按像素统计，行是 Local 是否正确，列是 Wide 是否正确。`Local wrong → Wide correct` 是宽视野提供的潜在有益修正，`Local correct → Wide wrong` 是宽视野破坏局部正确结果的潜在干扰。

| Dataset | Both wrong | Local wrong → Wide correct | Local correct → Wide wrong | Both correct |
|---|---:|---:|---:|---:|
| VDD | 11.84% | **16.04%** | 5.14% | 66.98% |
| Potsdam | 16.24% | **29.18%** | 8.58% | 46.00% |
| ADE150 | 34.56% | **16.47%** | 7.36% | 41.62% |
| VOC21 | 10.00% | **12.07%** | 4.27% | 73.66% |

这张表只描述 Local 与 Wide 的互补性；最终 Geov3 的提升还要结合结构写回的筛选效果，不能把所有 Wide 修正都视为可直接写回。

## Table 4c. 结构写回算子消融

固定同一 Local/Wide 观测和 Geov3 的 `g0`，只改变跨视野差异的写回算子。`DirectFusion` 为直接写回，`GramPropagation` 为 $G^\top G$ 写回，`SpectralFilter` 为 $H=(I+G^\top G)^{-1}G^\top G$ 的闭式滤波，也就是 Geov3 的结构锚定写回。

| Dataset | Local | Direct fusion | Gram propagation | **SpectralFilter / Geov3** |
|---|---:|---:|---:|---:|
| VDD | 50.9495 | 59.3827 | 60.4424 | **61.1846** |
| Potsdam | 38.3740 | 49.1425 | 49.1491 | **55.6874** |
| ADE150 | 27.8541 | 31.0417 | 31.1322 | **31.6645** |
| VOC21 | 41.4950 | 64.9100 | 64.0094 | **75.4653** |

结果文件：`artifacts/geometry_github/research/geov3_correction_ablation_20261101/full/<dataset>/results.json`。这组实验直接检验结构传播是否只是等权/直接融合，以及闭式谱滤波是否具有额外作用。

## Table 4d. QuerySlot 槽数量敏感性

固定 Geov3 其余配置、bank 和 `g0`，只改变宽视野 QuerySlot 容量。结果显示 K=24 已经进入稳定区间；K 增大到 48 只带来小幅收益，因此终版固定 K=24 以控制显存和延迟。

| K | VDD mIoU | Potsdam mIoU |
|---:|---:|---:|
| 8 | 60.8991 | 55.0474 |
| 16 | 61.0960 | 55.5103 |
| **24（终版）** | **61.1828** | 55.6445 |
| 32 | 61.1884 | 55.6924 |
| 48 | 61.2538 | **55.7191** |

结果文件：`artifacts/geometry_github/research/geov3_k_sweep_20261101/<dataset>/k<K>/results.json`。K=24 是精度-成本折中，K=48 的增益分别仅 +0.0710 pp 和 +0.0746 pp。

## Table 4e. 结构关系组成消融

固定 Geov3 的 Local/Wide 观测和 `g0`，只改变结构关系 $G$ 的来源。Appearance-only 使用局部 DINO patch 外观相似度，Spatial-only 使用二维坐标高斯先验，Full 使用 Geov3 的外观-空间联合关系和闭式谱滤波。

| Dataset | Local | Appearance-only | Spatial-only | **Full appearance + spatial** |
|---|---:|---:|---:|---:|
| VDD | 50.9495 | 60.1021 | 46.9617 | **61.1852** |
| Potsdam | 38.3740 | 48.8035 | 38.7238 | **55.6446** |

结果文件：`artifacts/geometry_github/research/geov3_relation_ablation_20261101/<dataset>/results.json`。空间先验单独使用几乎不能替代外观关系；联合关系仍高于 Appearance-only。

## Table 4f. Local、Wide、融合和 Geov3

`Weighted fusion` 固定为 $0.25L+0.75W$，不使用标签选择权重；`Equal fusion` 为 $0.5L+0.5W$。这组结果与结构写回算子消融使用同一 Geov3 端点。

| Dataset | Local | Wide | Equal fusion | Weighted fusion (0.25/0.75) | **Geov3** |
|---|---:|---:|---:|---:|---:|
| VDD | 50.9495 | 59.3857 | 60.4209 | 60.3670 | **61.1854** |
| Potsdam | 38.3740 | 46.0975 | 53.4372 | 49.1409 | **55.6443** |
| ADE150 | 27.8541 | 27.4547 | 29.4886 | 28.3115 | **31.6645** |
| VOC21 | 41.4950 | 64.9073 | 75.1613 | 71.8355 | **75.4645** |

结果文件：`artifacts/geometry_github/research/geov3_fusion_ablation_20261101/full/<dataset>/results.json`。

## Table 5. SGDE 核心机制消融（A–E）

所有变体固定 DINOv3、QuerySlot K=24、alias bank、窗口、`G`、log-sum-exp 和最终评分；只改变 SGDE 信息进入模型的位置：A Native Attention，B Post-hoc Propagation，C Direct Structural Propagation，D Donor Reconstruction Only，E Full SGDE。

| Dataset | A Native attention | B Post-hoc propagation | C Direct structural propagation | D Donor reconstruction only | **E Full SGDE** |
|---|---:|---:|---:|---:|---:|
| VDD | 58.9728 | 59.2511 | 58.8438 | 61.0976 | **61.2364** |
| Potsdam | 48.5638 | 49.0547 | 48.6508 | 54.5936 | **55.6692** |
| ADE150 | 28.1788 | 28.9773 | 28.1929 | 31.2159 | **31.6739** |
| VOC21 | 72.6713 | 73.4370 | 73.0030 | 74.9195 | **75.4484** |

上述 A–E 均来自修正版 Geov3 runner，固定 Anchored、β=1、shared_visual=False、BF16、batch=8；结果文件位于 `geov3_sgde_ablation_20261031/corrected_full/`。旧配置结果只保留在 `remote_full/` 作诊断。

## Table 6. 结果与文件状态

| 项目 | 状态 |
|---|---|
| Geov3 主协议（14 个，另计 LoveDA P/D） | 已完成，全部覆盖预期样本 |
| LoveDA P/D | 已完成 1,669/1,669，70.2428 / 46.1671 |
| iSAID 11,644 tiles | 最佳 bank 全量完成，29.0883% |
| 路径消融 VDD/Potsdam/ADE150/VOC21 | 已完成 |
| SGDE A–E VDD/Potsdam/ADE150/VOC21 | 修正版全量完成 |
| 路径质量诊断（Pixel/Boundary/Small/Large/转移） | VDD/Potsdam/ADE150/VOC21 全量完成，见 Table 4a–4b |
| 结构写回算子消融 | VDD/Potsdam/ADE150/VOC21 全量完成，见 Table 4c |
| QuerySlot K 敏感性 | VDD/Potsdam，K∈{8,16,24,32,48} 全量完成，见 Table 4d |
| 结构关系组成消融 | VDD/Potsdam 全量完成，见 Table 4e |
| Local/Wide/Fusion/Geov3 | VDD/Potsdam/ADE150/VOC21 全量完成，见 Table 4f |
| Cityscapes | 真实 leftImg8bit val 全量完成，56.5384% |
| 统一速度/显存（14 个主协议） | 已完成；使用独立 cost runner，见 Table 2c |
| JEV alias 筛词对照 | 已有 Geov2 冻结六 bank 报告；本轮未混入 Geov3 主表，避免模型版本混用 |


审计限制：当前结果是本轮所用 bank/g0 的完整复跑，还没有证明每行达到所有历史 bank 与参数的最佳值。速度与显存已按同一终版配置独立复测并列入 Table 2c；不复用旧 Geov2 效率表。
