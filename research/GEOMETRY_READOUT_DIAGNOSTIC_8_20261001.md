# 原 Geometry 读取诊断：VDD / Potsdam 各 8 个固定样本

日期：2026-10-01。仅使用用户授权的原 Geometry 观测，不是候选模型评测。

## 1. 结论

**当前不能把第二模块设计成“统一删 MLP / LayerNorm”，也不能继续把小目标问题等同于召回不足。** 本次 Potsdam car 召回为 100%，VDD vehicle 为 99.0994%，但精确率分别只有 5.8594% 和 13.5993%。主要问题是非目标区域也被激活。

读出不是单一路径污染：

- Geometry attention 阶段显著改善植被、树木与小目标 IoU，不能因最终误报而把这条流整体否定。
- Potsdam 的部分 car 误报在最早的读出探针中已有错误方向，后续处理降低了错误 margin，但未消除它。VDD vehicle 对 other 的部分误报也贯穿早晚阶段。
- VDD 缓存里一组 water→other 漏检，在第一层 MLP 前后探针中从正 water margin 翻为负；这是另一种有效证据丢失，不能靠统一增强或抑制小目标解决。
- MLP 与最终 LN 的 learned beta 对多个固定类别对有很大的相反贡献，净值远小于单项。单看 beta 的正贡献并宣布“去 bias 即可”会忽视这种耦合。

**下一模块需要辨识同一位置、同一类别对的竞争证据来源，既处理继承的错误泛化响应，也保护被后续语义变换压低的有效响应。此次观测定位了问题，没有证明如何可靠地决定校正方向。**

## 2. 固定协议与观测不变量

从历史已核验的完整集合按 `seed=20261001` 固定抽取 8 个样本，各图使用全部原始滑窗：512 工作区、128 overlap、Hann 概率拼接、输出温度 0.07。Geometry 为两 block、dense 关系、preserve prefix；原 20 aliases/类、原 RS 模板、normalized LME 温度 0.07 全部不变。VDD 使用修正后的官方 ontology 查询。

观测流程：原 `TCPRSegmenter.prepare_image` 先产生正常 Geometry 输出，再重放同一 head 捕获中间状态；每个窗口检查最终特征和 patch 类别预测。标签仅进入事后审计，不控制窗口、词表、系数或读取路径。

| 核验项 | VDD | Potsdam |
|---|---:|---:|
| 完成样本 / 唯一样本 | 8 / 8 | 8 / 8 |
| 原滑窗 / 不变滑窗 | 704 / 704 | 72 / 72 |
| 最大重放特征误差 | 0 | 0 |
| final 与 Geometry_original 整图混淆矩阵 | 完全相同 | 完全相同 |
| 最大类别分数分解误差 | 1.78814e-7 | 1.78814e-7 |
| 标签无关固定特征快照 | 16 | 16 |
| CPU 核验快照坐标与覆盖 | 通过 | 通过 |
| CPU 缓存最大分解误差 | 1.63913e-7 | 1.63913e-7 |
| 诊断 wall time / 秒 | 162.0389 | 11.1357 |
| peak allocated GPU memory / MiB | 3729.3252 | 3729.1040 |

时间包含重放、分解、缓存保存和各阶段整图拼接，不是原模型独立推理时间；不含前面的模型/文本加载。两诊断会话均正常结束，结束后 A800 0–7 卡均 1 MiB、0% 利用率，无 compute 进程。没有重启八集队列。

全部窗口都有阶段指标与分解汇总，但高维 tensor 仅每图按标签无关规则存两个窗口。远端高维缓存合计约 2.12 GB，未为报告下载这些 tensor；本地保存结果、signature 和 CPU 审计 JSON。

## 3. 各阶段读出探针

中间状态统一经过冻结最终 LN / projection，再接相同 alias / LME / 拼接。**这是诊断探针，不是移除后续层的部署模型。** `backbone` 也不是另一个已校准语义教师。

| 阶段 | VDD mIoU | vehicle IoU | roof IoU | water IoU | Potsdam mIoU | car IoU | lowveg IoU | tree IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Backbone 经最终读出 | 39.5596 | 6.8209 | 69.5484 | 79.2822 | 28.1805 | 2.5832 | 39.2949 | 7.5091 |
| Block0 attention 后 | 44.2562 | 9.2127 | 70.5847 | 80.2349 | 38.2621 | 4.4639 | 60.6870 | 31.1998 |
| Block0 MLP 后 | 43.5202 | 11.2833 | 68.2069 | 78.6707 | 37.4766 | 4.7115 | 55.9135 | 33.2481 |
| Block1 attention 后 | 45.2831 | 13.4866 | 69.4612 | 79.3660 | 41.0611 | 5.5922 | 61.5754 | 44.9666 |
| Block1 MLP 后 / 最终 Geometry | 46.5339 | 13.5825 | 70.5414 | 78.8724 | 42.5695 | 5.8594 | 65.3138 | 51.6910 |
| 最终状态不经过 LN 的探针 | 3.3141 | 1.3635 | 0.1729 | 6.7393 | 13.6644 | 12.9278 | 0.8434 | 41.3555 |

这里不能据 car 单项上升采用无 LN 读出：其 Potsdam 整体和低植被已经严重塌缩，VDD 整体也塌缩。也不能用 Block0 MLP 的总体下降推断所有 MLP 有害，最后一层 MLP 在两集提高总体指标。

本次没有 native attention 对照，所以阶段改善不能全部因果归于 Geometry 相对 native 的改动；它描述的是**当前 Geometry 前向路径**的实际状态变化。

### 最终原模型的逐类 IoU

| VDD 类 | IoU / % | Potsdam 类 | IoU / % |
|---|---:|---|---:|
| other | 14.3970 | impervious surface | 52.5068 |
| wall | 28.0050 | building | 77.5183 |
| road | 44.8643 | low vegetation | 65.3138 |
| vegetation | 75.4744 | tree | 51.6910 |
| vehicle | 13.5825 | car | 5.8594 |
| roof | 70.5414 | clutter | 2.5280 |
| water | 78.8724 | — | — |

**这是各 8 个固定样本的统计，不能替换 VDD80 / Potsdam504 全量结果。** 本次 VDD 的 water IoU 明显高于历史完整集合，也说明小集合不是全部困难场景的代表。

## 4. 小目标过预测来自哪些竞争类

| 项目 | VDD vehicle | Potsdam car |
|---|---:|---:|
| TP | 790,874 | 86,203 |
| FP | 5,024,662 | 1,384,992 |
| FN | 7,187 | 0 |
| precision / % | 13.5993 | 5.8594 |
| recall / % | 99.0994 | 100.0000 |
| 预测面积 / % | 6.0579 | 18.3899 |
| GT 面积 / % | 0.8313 | 1.0775 |

Potsdam car 的 FP 来自路面 712,502、tree 383,454、low vegetation 264,642、clutter 22,718、building 1,676。VDD vehicle 的 FP 来自 other 3,433,153、vegetation 1,157,139、road 363,563、roof 44,858、wall 25,211、water 738。

因此 car/vehicle 与各竞争类必须分别审计。vehicle–road 的响应不能解释主要来自 other/vegetation 的误报；删除 road 词也可能将正常 road 推向 vehicle。

## 5. 分解的含义与边界

最后一个 block 的 patch 状态分为：

```text
最后 block 输入（residual，已含之前的 block）
    + native prefix 读取
    + Geometry patch 读取
    + attention projection bias
    + MLP 增量
    + 数值 rounding
        ↓
最终 LN（gamma / beta）与 projection
        ↓
feature L2 normalize → alias cosine → normalized LME
```

按实际完整状态固定 LN 标准差及最终 L2 分母，将每项中心化后乘 gamma，再加独立 beta 项。实际 head 的最终 projection 为 Identity，`norm_bias` 对应最终 LN 的 learned beta。随后使用最终 alias responsibilities `pi_a`：

`L_c = sum_k sum_(a in c) pi_a * contribution_(k,a) + tau * (H(pi_c) - log(n_c))`。

各项精确重构实际分数；alias entropy 是单独一项。**这属于条件加法归因，不等于真的移除 MLP、beta 或某个 alias 后的因果变化**：实际移除会改变归一化分母、responsibilities，甚至后续 Q/K/V。`residual` 不是单独的原始 backbone residual。

全部窗口汇总中，Potsdam car FP 的平均错误 margin 为 +0.017566：residual +0.011983、Geometry patch +0.001149、MLP -0.015189、norm_bias +0.017482、alias entropy +0.002314，其余项补齐。因此单项 beta 大，但 MLP 抵消相当一部分。

VDD vehicle FP 的 margin 为 +0.017585：residual +0.013689、Geometry patch +0.002557、MLP +0.043209、norm_bias -0.043649。**两域的 MLP / beta 单项方向相反，不能以一个统一删项口号解释。**

这些全部窗口聚合混用了不同竞争类，且窗口重叠；不能拿“正确组对最强竞争类”的贡献直接与“错误组对 GT 类”的贡献比较，宣布已找到可靠性信号。下面改用同一 A/B 对。

## 6. 固定 A/B 对：CPU 缓存审计

以下 margin 始终是 A 减 B。只读每图两个固定窗口，所有阶段使用同一位置与类别对；误报组按最终原预测作事后分组。计数是 patch observation centers，不是唯一图像像素数。

| A−B / 实际 GT 与最终预测 | centers | Backbone | B0 attention | B0 MLP | B1 attention | final |
|---|---:|---:|---:|---:|---:|---:|
| car−路面 / GT路面→car | 2017 | +0.015537 | +0.012545 | +0.014329 | +0.013278 | +0.013745 |
| car−lowveg / GT lowveg→car | 549 | +0.035752 | +0.025909 | +0.025893 | +0.025118 | +0.021301 |
| car−tree / GT tree→car | 948 | +0.043112 | +0.031171 | +0.029490 | +0.025504 | +0.020206 |
| vehicle−other / GT other→vehicle | 318 | +0.016817 | +0.014200 | +0.014184 | +0.015314 | +0.016842 |
| roof−wall / GT roof→wall | 384 | -0.008152 | -0.006049 | -0.011536 | -0.011902 | -0.011456 |
| water−other / GT water→other | 98 | +0.005780 | +0.004052 | -0.002634 | -0.001152 | -0.001798 |

### 6.1 Potsdam：继承的错误方向占重要位置

路面、lowveg、tree 的上述 car 误报在 Backbone 探针就为正。尤其 tree→car 从 +0.043112 降到 +0.020206，说明后续流改善了这个固定竞争，却没有完全纠正。

最后 block 的条件贡献：

| car−B / GT B→car | residual | Geometry patch | MLP | LN beta | MLP＋beta |
|---|---:|---:|---:|---:|---:|
| 路面 | +0.009287 | +0.001327 | +0.010105 | -0.008356 | +0.001749 |
| lowveg | +0.014705 | +0.001783 | -0.021849 | +0.024413 | +0.002564 |
| tree | +0.014484 | -0.000803 | -0.054100 | +0.057322 | +0.003222 |

同 car−路面对，207 个 GT car 中心的 residual 为 +0.037497、Geometry patch +0.006522、MLP＋beta +0.016071。真假 car 的平均幅度不同，但都可能正，**平均差异不等于已经找到逐位置判别规则**。car 真值中心来自 4 个 tiles，三个主要误报组均涉及 8 个 tiles。

不能把 Geometry 最后一次 attention 的误报贡献较小，理解成最早 Value / 文本或空间混合已被排除；前面 block 的处理已经进入 residual。也不能据该表宣布错误只来自 beta。

### 6.2 VDD：误报与有效语义丢失是两件事

318 个 other→vehicle 中心涉及 5 个图像。它们 vehicle−other 的 residual +0.013344、Geometry patch +0.003602，MLP＋beta -0.001417：最后 MLP 与 beta 的净项反而稍微反对误报。单独抑制最终 MLP 或加强几何连续性缺少依据。

98 个 water→other 中心的 water−other 在第一层 MLP 后从 +0.004052 翻至 -0.002634。最后 block 的 residual +0.007463、Geometry patch +0.003716、prefix +0.003440，MLP +0.179422 与 beta -0.195864 合为 -0.016442。这支持“某些有效水体竞争证据在语义变换中丢失”的定位。

该 water 组只来自一张图，缓存中的 22 个 GT vehicle 中心也只来自一张图；不是域级规律，不能据它决定水体专用删层策略。roof→wall 组则跨 4 张图，错误方向在最早探针已存在。

## 7. 对第二模块设计的具体约束

本次可支持的问题定义从“更强的小目标激活”收紧为：**结构化读取后的语义竞争既会继承泛化误响应，也会在后续变换中丢失有效方向；需要位置条件、竞争对条件的证据流校正。**

下一候选如果获得另外授权，应遵守：

1. 保留原 Geometry attention 和正常对齐路径，不先做全局去 MLP、去 LN、Value 去均值或逐类面积阈值。
2. 明确核验的是同位置 A/B 的方向和来源，而不是某类 alias 响应是否大；路面→car、tree→car、water→other 是不同错误流。
3. 继承状态与结构读取分工：Geometry 给出读取支持，语义辨识决定应保留什么竞争内容。Geometry margin 不能同时提出错误和充当真值教师。
4. 若尝试将 donor Values 的局部判别部分与场景/共享部分分开，必须定义可观测的分离规则及对齐还原接口；不能因本次 residual 项较大就断言它等于有害共享 Value。
5. 成分分解只用于设计与审计，不把这里的有标签均值、符号或分组拟合成目标域门控。最终是否可纠错仍需要新的可用信息及验证，不能由分解精确性保证。

这不是建议再堆一个 confidence gate。现有证据也不能保证“去偏置＋Geometry”优于原模型，更不能承诺八域同时提升。此次没有证明筛哪些词、VIP 在相同位置哪条流正确，或已经找到可靠第二模块。

## 8. 复现与文件

成功远端根目录：`/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/geometry_readout_diagnostic_8_20261001_r2`。之前两个 collector 失败目录保留，未覆盖成功结果。采集器的 Identity projection / Confusion 接口已修正，原模型文件不变；8 项远端 CPU 测试通过。

代码：[特征重放与分解](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/dinotool/geometry_readout_trace.py)、[固定样本采集器](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/diagnose_geometry_readout_streams.py)、[CPU 固定竞争对审计](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/audit_geometry_readout_cache.py)。CPU 审计不加载模型或图像；核对了 32 个缓存的样本/窗口坐标、完整性、原分数一致和分解重构。

本地原结果：[VDD](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/vdd_results.json)、[Potsdam](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/potsdam_results.json)。固定对审计：[VDD](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/vdd_cache_audit.json)、[Potsdam](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/potsdam_cache_audit.json)。8 个固定样本的完整文件名见各 signature / results。

**本次授权诊断已完成；可靠语义矫正目标仍未实现，八数据集评测保持停止。**

## 9. 后续只读缓存审计：支持信息能纠正什么、不能纠正什么

在同一32个快照上追加CPU诊断，没有模型前向、参数变化或新分割输出。固定四个描述量，不搜索阈值、权重或排序方向。令m为最终原Geometry的A−B margin，G为缓存原关系；去掉对角后重新行归一化得到非self邻域权重W：

```text
原竞争：              m_i
Geometry邻域竞争：   u_i = sum_j W_ij * m_j
整窗非self对照：      b_i = (sum_j m_j - m_i) / (N - 1)
支持相对整窗：        u_i - b_i
当前位置相对邻域：    m_i - u_i
```

这里的整窗对照不是已知背景，可能完全由目标区域组成。邻域评分也不是统计独立教师。去对角仅用于检验是否真的包含邻居信息，不修改原Geometry读取关系。

### 9.1 Geometry邻域存在可用的方向信息，但不能纠正连续语义错误

固定原预测组内，记录u的自然零点符号；这不是拟合后采用的决策阈值，也不是一个已评测模型。

| A−B与原预测组 | 原本真正A / 真正B中心 | 邻域仍偏向A的真正A | 邻域仍偏向A的真正B |
|---|---:|---:|---:|
| vehicle−road，原预测vehicle | 19 / 23 | 19 / 19 | 1 / 23 |
| vehicle−other，原预测vehicle | 19 / 318 | 19 / 19 | 316 / 318 |
| car−路面，原预测car | 207 / 2017 | 207 / 207 | 1677 / 2017 |
| car−lowveg，原预测car | 207 / 549 | 207 / 207 | 444 / 549 |
| car−tree，原预测car | 207 / 948 | 207 / 207 | 792 / 948 |
| water−other，原预测other | 98 / 88 | 87 / 98 | 0 / 88 |

vehicle−road的22个误报中心以及water−other的87个漏检中心，其邻域符号都指向实际正确类。与此同时，316/318个other→vehicle中心仍得到错误vehicle支持；Potsdam多数非car区域误报也未被否定。

因此这个信号能够解释部分孤立/局部竞争错误，却不能验证“视觉连续即语义正确”。尤其VDD的主要vehicle误报来自other和vegetation，不是只有road；此次不能由vehicle−road的小组成功宣称vehicle整体已被解决。

### 9.2 高排序AUC不等于有可用的自然决策点

在原预测A且GT仅为A/B的固定集合上，正样本是真正A，负样本是被误判为A的B；score方向不因标签翻转。

| 原预测A / 固定竞争 | 原margin AUC | 邻域margin AUC | 支持−整窗 AUC | query−支持 AUC |
|---|---:|---:|---:|---:|
| car−路面 | 0.978243 | 0.986408 | 0.990218 | 0.597829 |
| car−lowveg | 0.998865 | 0.999965 | 0.991552 | 0.500066 |
| car−tree | 0.989864 | 0.989176 | 0.977965 | 0.562267 |
| vehicle−other | 0.978484 | 0.976332 | 0.926349 | 0.609897 |
| roof−wall | 0.545664 | 0.610099 | 0.615904 | 0.300226 |

Potsdam原margin已经有很高的局部排序，而所有原预测car的这些A/B中心原margin均为正。邻域也大多仍为正，所以从高AUC推导出可靠拒绝机制是不成立的。不能拿标签求出的最佳阈值作为training-free模块。

同图内配对也报告在JSON中。例如vehicle−other原margin的总体AUC为0.978484，同图配对为0.889138，邻域对应为0.976332/0.917133。后者涉及的真实vehicle中心仅来自一张图；场景间排序不能当作跨域校准证据。

### 9.3 统一“去场景偏置”会伤害正确的大区域语义

支持−整窗相对量在Potsdam原预测car组的自然正号保留了全部207个真car中心，并使44.0754%的路面误报、65.5738%的lowveg误报、81.5401%的tree误报不再为正。但这**不是可以直接采用的统一模型**：

- 在926个正确water中心中，支持−整窗仅51.5119%为正。
- 在98个原预测other的真正water中心中，该量全部不为正；对应88个真正other中心却有44.3182%为正。该组AUC仅0.063660，同图内0.125850。
- 在全部water/other缓存中心，该量AUC为0.442444，同图内0.376178。大范围真实水体也会成为整窗共有语义，减掉整窗响应会连真证据一起去除。
- query−支持在多个误报组接近随机或反向，并不是一个可替代的统一可信度。

### 9.4 对下一实现的影响

这次明确否定了两个直接实现：把Geometry邻居投票当真值，以及对所有语义统一减整窗偏置。前者无法纠正连续的other→vehicle偏差，后者会伤害水体的有效覆盖。

可以保留的事实是：原Geometry支持中确实存在部分纠错方向，但这与可靠语义识别是两件事。最终模块需要在**同一支持上的具体A/B竞争**中区分局部目标证据与上下文借词；还必须识别“整窗就是该类”的情形，不能以不够局部为由拒绝真实水体。整窗/支持均来自同一错误读出时，这项识别仍没有真值来源。

此处不采用诊断后逐类选择信号、不合成gate、不据这些标签设计阈值。要证明实际纠错与mIoU收益，下一步需要另行授权固定候选或新的语义观测，而不是将这份排名审计称为终版效果。停止条件不变。

脚本：[audit_geometry_support_specificity.py](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/audit_geometry_support_specificity.py)。4项CPU测试通过，覆盖AUC并列/空组、均匀对照、排除self泄漏及无非self支持拒绝。

完整自然符号与同图审计：[VDD](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/vdd_support_specificity_sign_audit.json)、[Potsdam](C:/Users/AH/Documents/ChatGPT/OVSS/research/geometry_readout_diagnostic_8_20261001/potsdam_support_specificity_sign_audit.json)。早期仅排名的审计文件另行保留，未覆盖原结果。
