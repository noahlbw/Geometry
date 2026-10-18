# Qwen20 candidates → VIP paper-rule screening: 8-domain full evaluation

本实验不读取 VIP 官方词表。每个类别由 Qwen2.5-VL-7B 生成一个固定 20-query bank：canonical 类别名 + 19 个文本候选。生成阶段不读取图像、mask、mIoU 或 VIP 分数；只做 JSON 格式、重复和 exact-rival 的机械检查。原始 prompt/response 保存在 `vocabularies/qwen20_vip_20261018/generation.json`。

随后对每个数据集的全部无标签评测图像计算 VIP paper-rule 的 VG/SC：保留 canonical；候选还必须满足 cosine≥0.7、VG 高于 canonical、SC 低于 canonical。没有数量 quota。冻结后的 bank 再对同一完整图像序列评测，因此这是 transductive 的 VIP 筛词复现，不应当写成独立测试集结果。mask 只在验证阶段加载。

## 结果

`VIP_All20` 是 Qwen20 全候选；`VIP_Distilled` 是完全相同观测经过 VG/SC 筛选后的 bank。

| 数据集/协议 | 图像数 | Qwen All20 | VIP Distilled | 筛选增益 pp | 保留数/类 | 保留率 | 历史 VIP All20* |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vdd/vdd | 80 | 44.3534 | 49.0980 | +4.7446 | [2,15,3,9,3,2,3] | 26.4% | 52.0647 |
| potsdam/potsdam | 504 | 32.9385 | 21.0067 | -11.9318 | [12,1,10,4,4,6] | 30.8% | 44.8995 |
| udd5/udd5 | 40 | 40.3424 | 43.7111 | +3.3687 | [11,1,7,9,9] | 37.0% | 41.4336 |
| oem/oem | 384 | 26.0358 | 29.0695 | +3.0337 | [8,11,15,4,7,6,6,10] | 41.9% | 33.2064 |
| loveda/P | 1669 | 41.1142 | 48.0535 | +6.9393 | [7,2,3,9,5,6] | 26.7% | 56.3110 |
| loveda/D | 1669 | 27.7844 | 30.0183 | +2.2339 | [12,7,2,3,10,4,6] | 31.4% | 36.1457 |
| vaihingen/vaihingen | 113 | 27.5140 | 19.7298 | -7.7842 | [11,4,8,1,4] | 28.0% | 41.9062 |
| landcoverai/landcoverai | 1602 | 54.4455 | 57.0822 | +2.6367 | [6,5,5,1,3] | 20.0% | 54.9420 |
| flair1/flair1 | 15700 | 27.9287 | 31.5313 | +3.6026 | [5,1,11,3,3,5,11,5,5,15,2,12] | 32.5% | 35.2977 |

8 协议均值（LoveDA 只计 D）：All20 **35.1678**，Distilled **35.1559**，平均筛选增益 **-0.0120 pp**；Distilled 胜出 **6/8** 个协议。

*历史 VIP All20 仅在仓库存在对应冻结记录时列出；不同记录的输入/数值修复口径不完全相同，不能把它当作本次 Qwen bank 的同词表对照。

## 各类筛选变化

| 数据集/协议 | 类别 | All20 IoU | Distilled IoU | Δ pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | other | 36.4508 | 28.1045 | -8.3463 |
| vdd/vdd | wall | 1.5377 | 17.2759 | +15.7382 |
| vdd/vdd | road | 36.2233 | 52.9412 | +16.7179 |
| vdd/vdd | vegetation | 66.3300 | 76.3948 | +10.0648 |
| vdd/vdd | vehicle | 1.3503 | 10.2703 | +8.9200 |
| vdd/vdd | roof | 83.3798 | 78.1292 | -5.2506 |
| vdd/vdd | water | 85.2017 | 80.5698 | -4.6319 |
| potsdam/potsdam | impervious surface | 49.0205 | 43.3651 | -5.6554 |
| potsdam/potsdam | building | 30.2842 | 6.4888 | -23.7954 |
| potsdam/potsdam | low vegetation | 37.0778 | 43.3206 | +6.2428 |
| potsdam/potsdam | tree | 45.1618 | 15.6515 | -29.5103 |
| potsdam/potsdam | car | 29.0423 | 9.9958 | -19.0465 |
| potsdam/potsdam | clutter | 7.0441 | 7.2184 | +0.1743 |
| udd5/udd5 | vegetation | 64.8402 | 69.0417 | +4.2015 |
| udd5/udd5 | building | 75.6812 | 80.9467 | +5.2655 |
| udd5/udd5 | road | 36.1085 | 34.6736 | -1.4349 |
| udd5/udd5 | vehicle | 15.6168 | 17.9560 | +2.3392 |
| udd5/udd5 | other | 9.4654 | 15.9377 | +6.4723 |
| oem/oem | bareland | 7.0698 | 6.9372 | -0.1326 |
| oem/oem | rangeland | 10.8432 | 19.3568 | +8.5136 |
| oem/oem | developed space | 24.7808 | 24.8362 | +0.0554 |
| oem/oem | road | 26.1096 | 23.4032 | -2.7064 |
| oem/oem | tree | 25.4675 | 29.3325 | +3.8650 |
| oem/oem | water | 61.3840 | 61.1563 | -0.2277 |
| oem/oem | agriculture land | 49.9384 | 64.9320 | +14.9936 |
| oem/oem | building | 2.6931 | 2.6015 | -0.0916 |
| loveda/P | building | 48.0206 | 60.4044 | +12.3838 |
| loveda/P | road | 47.6826 | 65.8479 | +18.1653 |
| loveda/P | water | 47.2428 | 57.2119 | +9.9691 |
| loveda/P | barren | 29.6363 | 25.0898 | -4.5465 |
| loveda/P | tree | 7.1016 | 9.9740 | +2.8724 |
| loveda/P | farm | 67.0016 | 69.7928 | +2.7912 |
| loveda/D | background | 13.0798 | 16.5647 | +3.4849 |
| loveda/D | building | 25.1944 | 25.5916 | +0.3972 |
| loveda/D | road | 43.4146 | 47.8661 | +4.4515 |
| loveda/D | water | 42.7052 | 51.5549 | +8.8497 |
| loveda/D | barren | 22.6342 | 20.1035 | -2.5307 |
| loveda/D | tree | 4.0346 | 3.4000 | -0.6346 |
| loveda/D | farm | 43.4277 | 45.0477 | +1.6200 |
| vaihingen/vaihingen | impervious surface | 40.2965 | 39.5327 | -0.7638 |
| vaihingen/vaihingen | building | 16.7665 | 22.7691 | +6.0026 |
| vaihingen/vaihingen | low vegetation | 29.7845 | 26.6045 | -3.1800 |
| vaihingen/vaihingen | tree | 37.1406 | 1.6773 | -35.4633 |
| vaihingen/vaihingen | car | 13.5817 | 8.0653 | -5.5164 |
| landcoverai/landcoverai | background | 74.3000 | 75.9300 | +1.6300 |
| landcoverai/landcoverai | building | 40.2376 | 47.8230 | +7.5854 |
| landcoverai/landcoverai | woodland | 64.9831 | 68.4903 | +3.5072 |
| landcoverai/landcoverai | water | 68.8061 | 64.0965 | -4.7096 |
| landcoverai/landcoverai | road | 23.9005 | 29.0712 | +5.1707 |
| flair1/flair1 | building | 49.6259 | 51.6138 | +1.9879 |
| flair1/flair1 | pervious surface | 1.5339 | 0.2277 | -1.3062 |
| flair1/flair1 | impervious surface | 54.2523 | 52.5940 | -1.6583 |
| flair1/flair1 | bare soil | 9.5526 | 10.6727 | +1.1201 |
| flair1/flair1 | water | 57.7391 | 71.0180 | +13.2789 |
| flair1/flair1 | coniferous | 12.8576 | 14.8126 | +1.9550 |
| flair1/flair1 | deciduous | 34.8117 | 50.0020 | +15.1903 |
| flair1/flair1 | brushwood | 1.4877 | 1.8560 | +0.3683 |
| flair1/flair1 | vineyard | 61.0632 | 63.2458 | +2.1826 |
| flair1/flair1 | herbaceous vegetation | 13.6598 | 30.7830 | +17.1232 |
| flair1/flair1 | agricultural land | 21.6720 | 10.5709 | -11.1011 |
| flair1/flair1 | plowed land | 16.8890 | 20.9786 | +4.0896 |

## 筛选与推理成本

selection 是 VG/SC 全图统计；evaluation 使用缓存 logits，同时重建 All20 和 Distilled。峰值显存为单个 worker 的 allocated MiB，初始化和 Qwen 离线生成不计入。

| 数据集 | 筛选秒 | 筛选峰值 MiB | 验证秒 | 验证峰值 MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 36.80 | 1784.6 | 37.83 | 844.5 |
| potsdam | 171.00 | 1770.2 | 42.23 | 69.9 |
| udd5 | 16.93 | 1757.1 | 16.08 | 566.4 |
| oem | 137.11 | 1799.9 | 37.39 | 83.3 |
| loveda | 420.55 | 1804.3 | 117.87 | 74.8 |
| vaihingen | 32.83 | 1757.1 | 8.18 | 63.0 |
| landcoverai | 436.41 | 1757.1 | 87.71 | 19.2 |
| flair1 | 2084.39 | 1853.5 | 393.60 | 42.2 |

## 复现边界

- VIP 的公开仓库没有公开完整 distillation implementation；这里实现并固定了论文文字规则，代码和 config 已保存。
- VDD、Potsdam、Vaihingen 也显式走本次 Qwen JSON，不会回退到 VIP 官方 short-query 文件。
- Qwen generation、VG/SC selection、冻结 bank、full evaluation 的 JSON 和 SHA256 都保留；selection 阶段没有加载 mask。
- 结果包含负增益（例如 Potsdam、Vaihingen），因此不能把 VG/SC 筛选描述成对所有域必然有效。

完整机器可读汇总：`comparison_summary.json`；候选词表：`vocabularies/qwen20_vip_20261018/`。
