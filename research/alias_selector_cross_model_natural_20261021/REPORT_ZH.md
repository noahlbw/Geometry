# Geov2 六类 alias bank 严格对照实验（2026-10-21）

这份实验只回答一个问题：在 Geov2 模型完全固定时，JEV 产生的变长 alias bank 是否比类别名、VIP 官方词表、Qwen20 全量词表和 VIP 筛词词表更有效。所有 bank 在同一批次中读取同一份视觉观测；没有为某个 bank 单独调 g0、温度或背景阈值。

## 固定条件

- 模型：Geov2 Anchored；`QuerySlot-24`；局部分支 `DonorBefore`；Anchored writeback；`compact=True, efficient=True`。
- 数据集级 g0（同一数据集内所有 bank 相同）：VOC21 `1.0`，COCO-Object81 `0.5`，ADE150 `0.25`。
- 评测：full 阶段使用全部验证图像；mIoU 按评测脚本的完整类别矩阵计算，VOC21/COCO-Object81 包含背景类，ADE150 无单独背景类。
- 视觉观测只计算一次，再对六个 bank 做读出，因此表中 wall time 是六-bank 批次时间，不是六次独立编码时间。
- 当前冻结的 Qwen20/JEV bank 只覆盖这三个自然图像数据集；遥感数据集没有同一套冻结 JEV response，不能伪造“所有 16 个协议”的 JEV 结果。

## 六个词表的准确含义和位置

以下路径均相对于仓库根目录 `artifacts/geometry_github`。绝对路径为 `C:\Users\AH\Documents\ChatGPT\OVSS\artifacts\geometry_github\` 加下表相对路径。

| 数据集 | bank | alias 总数 | 文件 |
|---|---|---:|---|
| voc21 | canonical_only（类别名-only） | 21 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/canonical_only.json` |
| voc21 | class_fixed_background（类别名+固定背景） | 76 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/class_fixed_background.json` |
| voc21 | vip_official（VIP官方完整bank） | 119 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/vip_official.json` |
| voc21 | vip20_expansion（Qwen20全量扩展） | 456 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/vip20_expansion.json` |
| voc21 | vip_screened（VIP VG/SC筛词） | 188 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/vip_screened.json` |
| voc21 | jev_screened（JEV文本筛词） | 94 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/jev_screened.json` |
| coco_object81 | canonical_only（类别名-only） | 81 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/canonical_only.json` |
| coco_object81 | class_fixed_background（类别名+固定背景） | 112 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/class_fixed_background.json` |
| coco_object81 | vip_official（VIP官方完整bank） | 156 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/vip_official.json` |
| coco_object81 | vip20_expansion（Qwen20全量扩展） | 1632 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/vip20_expansion.json` |
| coco_object81 | vip_screened（VIP VG/SC筛词） | 347 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/vip_screened.json` |
| coco_object81 | jev_screened（JEV文本筛词） | 170 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/jev_screened.json` |
| ade150 | canonical_only（类别名-only） | 150 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/canonical_only.json` |
| ade150 | class_fixed_background（类别名+固定背景） | 150 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/class_fixed_background.json` |
| ade150 | vip_official（VIP官方完整bank） | 234 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/vip_official.json` |
| ade150 | vip20_expansion（Qwen20全量扩展） | 3000 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/vip20_expansion.json` |
| ade150 | vip_screened（VIP VG/SC筛词） | 345 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/vip_screened.json` |
| ade150 | jev_screened（JEV文本筛词） | 162 | `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/jev_screened.json` |

每个 bank 的生成规则：

- `canonical_only`：每类只保留 Qwen20 pool 的第一个 canonical 表达；背景也只有字面量 `background`。这是严格的“只保留类别名”控制。
- `class_fixed_background`：前景每类只保留类别名，VOC/COCO 的背景替换为固定官方背景 group；这是比较前景 alias 选择时更公平的 baseline。
- `vip_official`：VIP 官方配置中的完整多 alias group。它不是类别名-only。
- `vip20_expansion`：同一 Qwen20 candidate pool 的全部候选；VOC/COCO 背景仍固定为官方背景 group，避免背景词数量成为变量。
- `vip_screened`：VIP 的 VG/SC 视觉筛词输出，候选 pool 与 `vip20_expansion` 相同，背景统一固定。
- `jev_screened`：JEV 文本筛词输出；保留 canonical，每类最多加入 2 个通过 JEV membership/conflict 规则的候选，背景统一固定。JEV 不读取 DINO response、图像或 mask。

每个数据集的完整 bank manifest（包含每类 alias 数量、模板、背景规则和来源文件）位于：

- `research/alias_selector_cross_model_natural_20261021/banks/six_controls/voc21/manifest.json`
- `research/alias_selector_cross_model_natural_20261021/banks/six_controls/coco_object81/manifest.json`
- `research/alias_selector_cross_model_natural_20261021/banks/six_controls/ade150/manifest.json`

## 聚合方式：log-sum-exp 与 log-mean-exp

先在同一 64 张 development split 上比较两个宽视野 alias 聚合。`log-sum-exp` 保留 alias 数量带来的累积偏置；`log-mean-exp` 减去每类 `log(alias_count)`，使复制一组 alias 不会仅因数量增加而抬高类别分数。

| bank | LSE dev 平均 mIoU（3 数据集） | LME dev 平均 mIoU（3 数据集） |
|---|---:|---:|
| canonical_only（类别名-only） | 36.6816 | 36.6817 |
| class_fixed_background（类别名+固定背景） | 41.7952 | 36.5719 |
| vip_official（VIP官方完整bank） | 44.7651 | 36.6702 |
| vip20_expansion（Qwen20全量扩展） | 36.5532 | 36.0685 |
| vip_screened（VIP VG/SC筛词） | 34.4709 | 35.9384 |
| jev_screened（JEV文本筛词） | 40.6225 | 36.8818 |

LME 的目的不是让所有数据集的绝对 mIoU 更高，而是把“bank 里有多少词”和“这些词是否有用”分开。后面的 full 结果同时保存两种聚合，避免把聚合选择混入 JEV 结论。

## Full 结果：log-sum-exp（原 Geov2 默认聚合）

| 数据集 | bank | alias 数 | mIoU (%) | Δ vs class_fixed_background (pp) | Δ vs vip_official (pp) | wall time (s) |
|---|---|---:|---:|---:|---:|---:|
| voc21 | canonical_only（类别名-only） | 21 | 51.6885 | -16.5016 | -23.5209 | 315.37 |
| voc21 | class_fixed_background（类别名+固定背景） | 76 | 68.1901 | +0.0000 | -7.0193 | 315.37 |
| voc21 | vip_official（VIP官方完整bank） | 119 | 75.2094 | +7.0193 | +0.0000 | 315.37 |
| voc21 | vip20_expansion（Qwen20全量扩展） | 456 | 55.5687 | -12.6214 | -19.6407 | 315.37 |
| voc21 | vip_screened（VIP VG/SC筛词） | 188 | 55.3815 | -12.8086 | -19.8279 | 315.37 |
| voc21 | jev_screened（JEV文本筛词） | 94 | 65.4252 | -2.7649 | -9.7842 | 315.37 |
| coco_object81 | canonical_only（类别名-only） | 81 | 48.5229 | -0.0061 | -1.0981 | 1341.19 |
| coco_object81 | class_fixed_background（类别名+固定背景） | 112 | 48.5290 | +0.0000 | -1.0920 | 1341.19 |
| coco_object81 | vip_official（VIP官方完整bank） | 156 | 49.6209 | +1.0920 | +0.0000 | 1341.19 |
| coco_object81 | vip20_expansion（Qwen20全量扩展） | 1632 | 46.5800 | -1.9490 | -3.0409 | 1341.19 |
| coco_object81 | vip_screened（VIP VG/SC筛词） | 347 | 41.5995 | -6.9294 | -8.0214 | 1341.19 |
| coco_object81 | jev_screened（JEV文本筛词） | 170 | 47.9831 | -0.5458 | -1.6378 | 1341.19 |
| ade150 | canonical_only（类别名-only） | 150 | 28.9401 | +0.0000 | -1.1349 | 727.47 |
| ade150 | class_fixed_background（类别名+固定背景） | 150 | 28.9401 | +0.0000 | -1.1349 | 727.47 |
| ade150 | vip_official（VIP官方完整bank） | 234 | 30.0750 | +1.1349 | +0.0000 | 727.47 |
| ade150 | vip20_expansion（Qwen20全量扩展） | 3000 | 27.3678 | -1.5723 | -2.7072 | 727.47 |
| ade150 | vip_screened（VIP VG/SC筛词） | 345 | 25.3313 | -3.6088 | -4.7437 | 727.47 |
| ade150 | jev_screened（JEV文本筛词） | 162 | 28.6825 | -0.2576 | -1.3926 | 727.47 |

平均（3 数据集，简单算术平均）：
| bank | 平均 mIoU (%) | 平均 Δ vs class_fixed_background (pp) |
|---|---:|---:|
| canonical_only（类别名-only） | 43.0505 | -5.5026 |
| class_fixed_background（类别名+固定背景） | 48.5531 | +0.0000 |
| vip_official（VIP官方完整bank） | 51.6351 | +3.0821 |
| vip20_expansion（Qwen20全量扩展） | 43.1721 | -5.3809 |
| vip_screened（VIP VG/SC筛词） | 40.7708 | -7.7823 |
| jev_screened（JEV文本筛词） | 47.3636 | -1.1895 |

这些结果对应文件：`research/alias_selector_cross_model_natural_20261021/results_six/full_logsumexp/`。

## Full 结果：log-mean-exp（alias 数量归一化）

| 数据集 | bank | alias 数 | mIoU (%) | Δ vs class_fixed_background (pp) | Δ vs vip_official (pp) | wall time (s) |
|---|---|---:|---:|---:|---:|---:|
| voc21 | canonical_only（类别名-only） | 21 | 51.6887 | -0.2658 | -1.2824 | 311.05 |
| voc21 | class_fixed_background（类别名+固定背景） | 76 | 51.9544 | +0.0000 | -1.0167 | 311.05 |
| voc21 | vip_official（VIP官方完整bank） | 119 | 52.9711 | +1.0167 | +0.0000 | 311.05 |
| voc21 | vip20_expansion（Qwen20全量扩展） | 456 | 53.3816 | +1.4272 | +0.4105 | 311.05 |
| voc21 | vip_screened（VIP VG/SC筛词） | 188 | 51.5563 | -0.3982 | -1.4148 | 311.05 |
| voc21 | jev_screened（JEV文本筛词） | 94 | 53.8010 | +1.8466 | +0.8299 | 311.05 |
| coco_object81 | canonical_only（类别名-only） | 81 | 48.5235 | -0.0003 | +0.0810 | 1353.17 |
| coco_object81 | class_fixed_background（类别名+固定背景） | 112 | 48.5237 | +0.0000 | +0.0813 | 1353.17 |
| coco_object81 | vip_official（VIP官方完整bank） | 156 | 48.4425 | -0.0813 | +0.0000 | 1353.17 |
| coco_object81 | vip20_expansion（Qwen20全量扩展） | 1632 | 46.5792 | -1.9446 | -1.8633 | 1353.17 |
| coco_object81 | vip_screened（VIP VG/SC筛词） | 347 | 48.0025 | -0.5213 | -0.4400 | 1353.17 |
| coco_object81 | jev_screened（JEV文本筛词） | 170 | 48.3552 | -0.1685 | -0.0873 | 1353.17 |
| ade150 | canonical_only（类别名-only） | 150 | 28.9398 | +0.0000 | -0.2781 | 731.11 |
| ade150 | class_fixed_background（类别名+固定背景） | 150 | 28.9398 | +0.0000 | -0.2781 | 731.11 |
| ade150 | vip_official（VIP官方完整bank） | 234 | 29.2179 | +0.2781 | +0.0000 | 731.11 |
| ade150 | vip20_expansion（Qwen20全量扩展） | 3000 | 27.3678 | -1.5720 | -1.8501 | 731.11 |
| ade150 | vip_screened（VIP VG/SC筛词） | 345 | 28.0284 | -0.9114 | -1.1894 | 731.11 |
| ade150 | jev_screened（JEV文本筛词） | 162 | 28.8896 | -0.0502 | -0.3283 | 731.11 |

平均（3 数据集，简单算术平均）：
| bank | 平均 mIoU (%) | 平均 Δ vs class_fixed_background (pp) |
|---|---:|---:|
| canonical_only（类别名-only） | 43.0506 | -0.0887 |
| class_fixed_background（类别名+固定背景） | 43.1393 | +0.0000 |
| vip_official（VIP官方完整bank） | 43.5438 | +0.4045 |
| vip20_expansion（Qwen20全量扩展） | 42.4429 | -0.6965 |
| vip_screened（VIP VG/SC筛词） | 42.5291 | -0.6103 |
| jev_screened（JEV文本筛词） | 43.6819 | +0.5426 |

这些结果对应文件：`research/alias_selector_cross_model_natural_20261021/results_six/full_logmeanexp/`。

## 结论如何读

1. `vip20_expansion` 在三个数据集都低于对应的 VIP 官方 bank，说明把 20 个候选全部写入并不能保证提升；候选数量本身会带来噪声和类别竞争。
2. 在原始 LSE Geov2 上，VIP 官方 bank 是三套 full 结果中最强的统一词表：VOC21 `75.2094`，COCO-Object81 `49.6209`，ADE150 `30.0750`。JEV bank 分别为 `65.4252`、`47.9831`、`28.6825`，因此这套 JEV admission 不能替代 VIP 官方 bank。
3. LME 去掉 alias-count 偏置后，JEV 在 VOC21 达到 `53.8010`，高于类别名-only `51.6887` 和固定背景类别名 `51.9544`；但 COCO-Object81 `48.3552`、ADE150 `28.8896` 略低于类别名 baseline。也就是说，JEV 文本筛词在当前 frozen bank 上有局部收益，但还没有跨数据集稳定收益。
4. 因此不能把“JEV 通过了语义 membership”写成“alias 对 DINO 一定有视觉收益”。JEV 删除明显不合法短语，却不检查 DINO 的视觉响应、类别竞争和 alias 稳定性；这正是 JEV 与 VIP VG/SC 筛词结果不同的原因。
5. 若目标是复现旧 Geov2 的最高 mIoU，主模型应保留 LSE，并明确 JEV 是一个筛词消融；若论文主张“筛词效果”，建议把 LME 作为 alias-ablation 的公平聚合，同时把 LSE 作为敏感性结果。两种聚合都已保存，不能在不同 bank 间混用。

## 复现实验文件

- bank 构造脚本：`tools/build_six_natural_banks.py`。
- 评测脚本：`DINOtool/scripts/eval_cross_selector_natural.py`。
- development LSE 结果：`research/alias_selector_cross_model_natural_20261021/results_six/dev_logsumexp/`。
- development LME 结果：`research/alias_selector_cross_model_natural_20261021/results_six/dev_logmeanexp/`。
- full LSE 结果：`research/alias_selector_cross_model_natural_20261021/results_six/full_logsumexp/`。
- full LME 结果：`research/alias_selector_cross_model_natural_20261021/results_six/full_logmeanexp/`.

评测使用的远端数据和参数：

```text
VOC21: /data/test/datasets/VIP_natural/VOCdevkit/VOC2012, g0=1.0
COCO-Object81: /data/test/datasets/VIP_natural/COCOObject, g0=0.5
ADE150: /data/test/datasets/VIP_natural/ADEChallengeData2016, g0=0.25
wide_branch=query_slot, query_slot_capacity=24, compact=True, efficient=True
```
