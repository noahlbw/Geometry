# Geov2 component ablation (2026-10-14)

四个数据集各取固定 16 张开发图，所有行共享 backbone、输入、文本 bank、背景规则和 24 槽配置。这里的 `Geov2_VIPWide` 是 Geov2 内部的 VIP 宽视野端点；官方 VIP 完整复现属于外部基线，单列在文末。

## Same-input results

| Dataset | Geov2 VIP-wide | Raw32 | QuerySlot raw | DonorBefore | Geov2 full | No alias | Equal fusion | Shuffled query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vdd | 62.2716 | 60.9840 | 61.3598 | 62.2255 | 61.4683 | 37.8678 | 61.1590 | 55.5686 |
| potsdam | 52.2582 | 51.4823 | 50.9304 | 52.2582 | 50.9316 | 41.9412 | 52.3219 | 48.2695 |
| ade150 | 20.9862 | 20.6140 | 20.7428 | 21.2813 | 20.9200 | 21.5039 | 22.6186 | 20.1220 |
| voc20 | 88.0115 | 96.0922 | 97.6098 | 88.0115 | 97.6094 | 81.5827 | 97.3822 | 88.9303 |

| Equal-domain macro | 55.8819 | 57.2931 | 57.6607 | 55.9441 | 57.7323 | 45.7239 | 58.3704 | 53.2226 |

## Component deltas

| Variant | Δ vs Geov2 full (pp) | 95% paired bootstrap CI |
|---|---:|---:|
| Geov2 VIP-wide | -1.8505 | [-3.5399, +0.7746] |
| Raw32 | -0.4392 | [-1.3265, +0.5705] |
| QuerySlot raw | -0.0716 | [-0.2704, +0.0899] |
| DonorBefore | -1.7882 | [-3.4580, +0.8499] |
| Geov2 full | +0.0000 | [51.6118, 59.3097] |
| No alias | -12.0084 | [-17.4581, -4.9004] |
| Equal fusion | +0.6381 | [+0.1848, +3.9771] |
| Shuffled query | -4.5097 | [-6.8341, +0.4048] |

解释：

- Alias：`No alias` 只保留每个类别的 canonical phrase，因此它测的是 alias bank 的贡献，不是把类别从文本输入中删除。
- QuerySlot：`QuerySlot_Raw` 相对 `Raw32` 只增加局部 query 到共享 slot 的路由；`Geov2 full` 再加 DonorBefore。
- DonorBefore：`DonorBefore` 与 `Geov2_VIPWide` 共享 VIP 宽视野，隔离局部分支变化；数值 strength 的配置仍使用原 Geometry 路径。
- Equal fusion：逐点使用 `0.5(L+W)`，没有结构算子 H；因此它是融合控制，不是 Geov2 writer。
- Shuffle：固定全图 slot bank，只滚动局部 query 的空间顺序；性能下降说明 QuerySlot 的提升依赖正确的局部位置对应。
- VIP：官方 VIP 不是 Geov2 的一个组件。外部全验证参考来自 `research/TRAINING_FREE_BASELINES_FULL_20261011.md`，其数值不能与本 16 图子集直接混为同一列。

## External VIP reference (full validation)

| Dataset | External VIP reference mIoU |
|---|---:|
| vdd | 52.5765 |
| potsdam | 44.8995 |
| ade150 | 29.4630 |
| voc20 | 92.5051 |

本表用于部件归因，属于开发屏幕；不能替代 2026-10-09 冻结版全量主结果。原始逐图混淆矩阵和 bootstrap 数据见同目录。
