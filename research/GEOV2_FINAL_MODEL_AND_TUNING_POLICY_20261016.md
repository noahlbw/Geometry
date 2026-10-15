# Geov2 Anchored 终版模型与调参口径

日期：2026-10-16。

本分支固定 **Anchored spatially anchored residual slot reader** 为 Geov2 终版模型。固定的视觉结构是：原 DonorBefore、全图 24 槽语义记忆、局部条件化残差槽读取、原结构写回。Anchored 实现见 `DINOtool/dinotool/anchored_slot_memory.py`，完整数值见 `research/geov2_anchored_wide_20261015/REPORT_ZH.md`。

## 先给结论

当前 Anchored 全量结果**没有在本轮重新挑词表**，也没有用 Anchored 的 mIoU 反过来选择 alias。它读取的是 2026-10-13 已经冻结的每个协议的词表快照，然后只在声明的 `g0` 候选上运行本轮比较。

但是，整个历史管线不能表述为“只调了 g0、词表完全没有针对性”。2026-10-13 之前的词表、模板、部分分数校准和背景规则曾根据带标签开发过程选择，而且不同数据集使用不同 bank。这些设置在 Anchored 本轮被冻结，因此本轮没有联动调参，但它们仍属于历史的任务适配配置。

因此论文必须区分两种口径：

1. **本轮 Anchored 消融口径**：词表、模板、背景规则、局部分数温度和宽视野策略在开跑前冻结；Anchored 与 Legacy/Coordinate 共享这些设置，只比较读取模块和声明的 `g0`。
2. **严格的无词表联动口径**：需要重新建立一个预先固定的语言协议，在所有结果出来前锁定候选词规则、模板、背景校准和 `g0` 搜索预算。现有 Anchored 数字不能追溯性地变成这个口径。

## 历史词表到底怎么来的

2026-10-13 QuerySlot＋DonorBefore 全量实验的协议明确写着“existing best aliases, previous readout/calibration/view policy retained”，并且每个数据集的 alias 来源都标记为此前的带标签开发选择。2026-10-14 的审计还记录了 `labeled_development=true` 与 `full_validation_viewed=true`。

Anchored 七个数据集继承的 bank 和规模如下：

| 数据集 | 冻结 bank | alias 数 | 历史来源 | 是否在 Anchored 本轮重选 |
| --- | --- | ---: | --- | --- |
| VDD | `official_imagenet` | 9 | `kev_alias_search_20261009` | 否 |
| Potsdam | `official_imagenet` | 11 | `kev_alias_search_20261009` | 否 |
| UDD5 | `kev60_segmentation` | 31 | `kev_alias_search_20261009` | 否 |
| OEM | `original` | 160 | `kev_alias_search_20261009` | 否 |
| Vaihingen | `kev40_imagenet` | 69 | `kev_alias_search_20261009` | 否 |
| ADE150 | `semantic_segmentation` | 433 | `kev_alias_search_20261009` | 否 |
| VOC21 | `official_segmentation` | 119 | `curated_variable_refine_natural_20261010` | 否 |

历史 15 协议还包括 VOC20、COCO-Object81、COCO-Stuff171、LandCover.ai、LoveDA、FLAIR、PC60/context60 和 context59；它们也各自使用已冻结的协议词表，而不是一套全局相同的 alias。PC60/context60 还保留了 401 个残余概念作为背景辅助证据。

这意味着：

- 词表不是 Anchored 的在线路由器；推理时不运行 Kev，也不读取标签。
- alias 仍然改变文本覆盖范围、每类 alias 数量和宽视野的 log-sum-exp 刻度，所以它不是一个可以忽略的实现细节。
- Anchored 本轮可以说“在固定语言输入下比较视觉模块”，不能说“整个 Geov2 从未做过任务词表适配”。

## 历史上调过哪些东西

除词表外，2026-10-13 的旧 QuerySlot 配置还保留了各协议此前的读出和校准设置：

| 设置 | 历史情况 | Anchored 本轮处理 |
| --- | --- | --- |
| Alias bank / 模板 | 按协议冻结，历史上带标签开发选择 | 原样冻结，不重选 |
| `g0` / coupling | 旧配置已有协议值；本轮在 `{0.25, 0.5, 0.75, 1}` 重新做声明内扫描，并报告冻结参考值 | Anchored 用 `protocol.json` 的参考值；统一值分析另报 |
| 局部分数温度 | 不同协议为 0.03、0.05、0.07 或 0.14 | 原样冻结 |
| 宽视野分辨率策略 | `long448` 与 `natural_short336_cap672` 按协议继承 | 原样冻结 |
| 宽视野 alias 聚合 `tau` | 当前这些配置为 1 | 原样冻结 |
| 图像级 alias 温度 `tem` | 个别历史配置为 0.3 或 10，其余多为 1；当前 Anchored scorer 关闭 salience 路径 | 原样冻结；不作为 Anchored 新贡献 |
| 背景 bias / threshold | 按数据集的背景定义和校准保留 | 原样冻结 |
| QuerySlot 槽数、beta、query temperature | `K=24`、`beta=0.5`、`temperature=0.1`、5 次聚类 | 全局固定 |
| DonorBefore strength | 2026-10-13 已统一为 `original` 以确保局部路径一致 | Anchored 不再修改 |

所以“本轮只改 `g0`”是准确的；“历史上所有参数都只改过 `g0`”是不准确的。

## 背景特殊处理是否合理

背景处理是任务定义的一部分，和 VIP 为不同协议设置背景类别、阈值的做法属于同一类工程配置，可以保留。但它必须公开，不能隐藏在“统一模型”表述里：

- 普通协议使用保留的 background 类规则；
- VOC21 使用 `union_max`，局部背景从其背景概念集合取最大响应；
- context60/PC60 使用 `residual_max` 读取残余概念作为背景辅助证据；
- bias 和 threshold 仍是每个协议的冻结校准量。

这些规则不改变输出类别数，也不在推理时访问掩码。公平比较时，同一数据集的 Anchored、Legacy、Coordinate 和 VIP 对照必须使用同一背景定义；论文中应将它们归入“任务协议”，不要归入 Anchored 的视觉创新。

## 终版建议的严格规则

从现在开始，Geov2 Anchored 终版只允许以下自由度：

1. 先锁定每个数据集的类别名称、背景 ontology、alias bank、模板和校准规则；锁定后不因 Anchored 结果改词。
2. `K=24`、`beta=0.5`、query temperature `0.1`、5 次 Lloyd 迭代、DonorBefore 和结构算子 `H` 全局固定。
3. `g0` 可以在预先声明的有限集合中选择；主表同时报告统一 `g0=0.5` 和按开发协议选择的冻结参考值。不得看到结果后再把 alias、背景和 `g0` 一起改动。
4. 任何新 alias、Kev 阈值、模板、局部温度、宽视野尺寸或背景校准实验都单独命名为语言/协议适配实验，不能并入 Anchored 主结果。
5. 论文中把现有结果称为“冻结任务语言协议下的 Anchored 视觉模块比较”；若要声称“只调 `g0` 的统一模型”，必须另做严格冻结语言协议的重新评测。

当前分支已经把 Anchored 固定为终版架构和实现入口，但保留历史词表来源和“曾用带标签开发数据”的审计记录。这样既不会把背景 ontology 错误地当成模型创新，也不会把历史上的词表适配隐瞒成零调参。
