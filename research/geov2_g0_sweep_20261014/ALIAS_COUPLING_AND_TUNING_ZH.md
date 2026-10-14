# Geov2：Alias 的实际用法、耦合机制与调参口径

2026-10-14。依据当前 10/13 QuerySlot＋DonorBefore 冻结配置、10/14 全量 g0 扫描，以及 A800 上固定提交的 VIP 官方源码。本文是代码和开发过程审计；没有新增词表、修改模型配置或运行新评测。

## 1. 当前模型并非所有数据集都采用 Kev 筛选词表

用户所指的 jev 对应代码中的 Kev（`jaredpalmer/kev-4b`）。Kev 是离线语义判断器，既不是图像编码器，也不是推理时的逐图选词网络。

| 协议 | 当前 bank | 词数 | 说明 |
| --- | --- | ---: | --- |
| VDD | official_imagenet | 9 | 官方文件派生词组＋ImageNet 模板；不是 Kev 筛选结果 |
| Potsdam | official_imagenet | 11 | 官方文件派生词组＋ImageNet 模板；不是 Kev 筛选结果 |
| UDD5 | kev60_segmentation | 31 | Kev 阈值 0.6；五类词数为 7/1/2/1/20 |
| OEM | original | 160 | 原每类 20 词，局部 RS6 模板 |
| Vaihingen | kev40_imagenet | 69 | Kev 阈值 0.4；五类词数为 16/16/17/12/8 |
| PC60 | mix_f6f525631e7f743d226c | 264 | 逐类词组混合；另保留 401 个辅助残余概念 |
| ADE150 | semantic_segmentation | 433 | 词义整理后的语义词表 |
| VOC20 | semantic_imagenet | 107 | 词义整理后的语义词表＋ImageNet 模板 |
| VOC21 | official_segmentation | 119 | 官方文件派生词组＋分割模板；其中背景 56 词 |
| COCO-Object81 | official_imagenet | 156 | 官方文件派生词组＋ImageNet 模板 |

`official_*` 的含义需要精确：历史构建器按所有逗号切分，VIP 原生解析器只按逗号加空格切分。VDD/Potsdam 存在实际分组差异。此外 `official_segmentation` 的模板也不是 VIP 原生默认的 ImageNet 模板。因此不能把所有 `official_*` 都称为字节级相同的 VIP 官方语言输入。

本表直接核对当前配置和词表；可读来源为 [10/13 报告](../query_slot_best_alias_20261013/REPORT.md)，逐项计数、来源 trial、开发标签声明见 [审计 JSON](ALIAS_TUNING_AUDIT.json)。这些配置均保留 `labeled_development=true` 和 `full_validation_viewed=true`。PC60 的逐类混合也经过带标签开发选择，并非纯文本规则决定。

## 2. 为什么扩词，扩出的词实际经过什么处理

扩词解决单个数据集标签与预训练视觉语言模型语言空间不匹配的问题：同义词、明确子类、复合标签的自然表达和消歧后的词义，可以覆盖同一输出类别的不同视觉实例。例如 UDD 的 vegetation 可由 grass、bushes、tree canopy 等提供证据；若另一任务把 tree 独立计分，同样的扩展就需要重新检查类别归属。扩词不增加被评测的类别数。

当前结果来自多个历史输入源：原有 20 词池、按类别含义整理的语义池、VIP 官方词文件，以及自然数据集的逐类混合。不能将整套当前模型改述为一个统一的“Qwen 生成、Kev 筛选”管线；单独的 Qwen20＋Kev 实验也没有成为当前默认词表。

Kev 这条具体路径是：

1. 合并已有 original、semantic、official 候选，并按类别去重；Kev 不生成新词。
2. 将完整竞争类别体系、目标类别、自然/遥感域描述一起提供给冻结判断器。
3. 对每个候选判断四种关系：目标特有；目标与其他类共享或过宽；属于其他类/矛盾；无法解释。
4. 使用两种选项顺序，恢复原次序后平均概率，减少选项位置影响。
5. 计算 `p_specific + 0.5*p_shared`；候选阈值为 0.4/0.6/0.8，保留类别名，不补齐固定词数。背景/残余集合保持原有规则。
6. 词表 bank、阈值和模板等最终选项由带标签开发过程决定，然后冻结用于所有图像。

第 2–5 步不读取图片和掩码；第 6 步使用了开发标签。因而准确表述是“文本判断无图像输入，配置选择使用带标签开发数据”，而不是“整个词表选择过程完全无标签”。Kev 的分数不作为在线 Alias 权重，也不在每张图上重跑 Kev。

使用 Kev 的动机是检查类别体系内的语义归属，避免把共现物、部件、上位概念或其他输出类混进目标类。它不是经校准的视觉正确率，语义上可接受也不保证提升 mIoU。旧 VIP 宽视野版本中，同参数筛选控制在 UDD5 为 +2.0612pp、Vaihingen 为 +1.5667pp；这不是当前 QuerySlot 版本的新消融，也不能把其他数据集的整体开发增益归因于 Kev。

自然数据集的另一次新 Qwen20＋Kev 历史实验得到 VOC21 64.1331%，没有超过更强旧词表；因此没有证据支持为统一 g0 而盲目重新扩词。这一历史结果与当前 QuerySlot 模型配置不同，只能说明该尝试并非必然有益。

源码：[Kev 规则](../../DINOtool/dinotool/kev_alias_selection.py)、[离线判断入口](../../DINOtool/scripts/select_kev_aliases.py)、[历史筛选控制](../KEV_ALIAS_SEARCH_20261009.md)。

## 3. 同一批 Alias 如何进入两路分割

每个词保留所属输出类映射；所有入选词都参加读出。先将词放入文本模板，编码为冻结 dino.txt 文本向量并缓存。局部路径对模板向量求均值后重新归一化；当前宽视野使用未重新归一化的模板向量均值做点积。RS6 bank 的局部采用六个遥感模板，宽视野沿用 ImageNet 模板。因此是共享词组和类映射，不保证两路每个文本向量完全相同。

普通类别的局部读出为：

\[
L_{ic}=\frac{\eta}{T_L}\left[\log\sum_{a\in\mathcal A_c}\exp(s^L_{ia}/\eta)-\log|\mathcal A_c|\right],\qquad\eta=0.07.
\]

其中 `eta` 是 Alias 聚合温度，`T_L` 是配置中的局部分数校准温度，二者不同。宽视野当前为：

\[
W_{ic}=\frac1\tau\log\sum_{a\in\mathcal A_c}\exp(\tau\,40s^W_{ia}).
\]

局部归一化词数，宽视野未归一化，含有 `log(alias_count)/tau` 的显式类偏置；改变词数同时改变覆盖与刻度。当前 `CompactContextScorer(salience=False)` 没有启用 VIP 的图像级词显著性加权，保存的 `tem` 不参与当前 QuerySlot 得分，调它不会改变预测。

背景有明确例外：VOC21 的 `union_max` 使局部背景取 56 个背景概念的最大响应；宽视野仍按其类分组读出。sky/wall/grass 等不是 background 的同义词，而是该任务残余类别的具体概念集合。PC60 的 401 个残余概念是辅助背景证据，`residual_max` 保留它们；不是增加 401 个输出类。不能在背景上直接套用“普通同义词平均”的解释。

源码：[文本编码和部署](../../DINOtool/dinotool/frozen_deployment.py)、[局部聚合](../../DINOtool/dinotool/packed_alias_readout.py)、[当前宽视野聚合](../../DINOtool/dinotool/context_memory_observer.py)、[背景和写回](../../DINOtool/dinotool/natural_static_search.py)。

## 4. 已经实现的耦合是什么

第一处是读取耦合：完整低分辨率图像建立 24 槽宽视野记忆，每个局部 raw token 作为 query 选择槽值，并与同坐标的宽视野特征混合。当前 beta=0.5、query temperature=0.1。因此宽视野观测依赖局部视觉 query，不是与局部完全独立的分支。

第二处是写回耦合：局部视觉关系 A 一方面用于 DonorBefore 的局部特征读取，另一方面构造 H=(I+AᵀA)⁻¹AᵀA，将宽视野与局部的差分写回：

\[
Z=L+g_0H(W-L).
\]

读取耦合回答“局部位置借哪些全局信息”，写回耦合回答“借来的差异如何沿局部结构改变预测”。Alias 通过 L 和 W 参与该修正，但当前 Alias 选取不依赖 A/H，视觉 query 路由也不依赖 Kev。因此可以主张局部条件化跨视野读取与结构写回，不能把 Kev 描述为已经实现的视觉结构自适应选词机制。

源码：[QuerySlot 路由](../../DINOtool/dinotool/query_slot_memory.py)、[DonorBefore](../../DINOtool/dinotool/geometry_semantic_response.py)、[H 构造](../../DINOtool/dinotool/efficient_deployment.py)。

## 5. 可直接用于论文的表述

中文方法表述：

> 我们将每个输出类别表示为一组冻结的文本查询，并在局部和宽视野读出中共享查询的类别映射。查询集合来自已有词表、类别词义整理及官方词文件；对采用 Kev 的配置，冻结语言判断器根据完整竞争类别体系评估候选词的语义归属，保护类别名并按固定阈值保留额外查询。文本向量离线编码和缓存，分割推理不运行语言判断器。局部条件化的槽路由建立跨视野读取依赖，随后沿局部视觉结构写回两路语义差分。

必须配套的实验披露：

> 视觉语言模型权重始终冻结，在线推理不访问标签。保留词表、Kev 阈值、模板及部分读出/背景参数曾由带标签开发数据选择；不同协议的冻结语言配置并不相同。我们分别报告共享写回步长结果与任务适配结果，公开所有词表和配置。全量验证扫描属于已查看数据上的开发分析，不作为独立测试。

English draft:

> We represent each output category by a frozen set of text queries and share their category assignments across the local and wide-view readouts. Query sets are drawn from existing vocabularies, category-sense curation, and released class-query files. For configurations using Kev, a frozen language judge evaluates candidate membership relative to the complete competing taxonomy; the class name is protected and additional queries are admitted by a fixed threshold. Text representations are encoded and cached offline, and the language judge is not executed during segmentation. Local visual queries route the wide-view slot memory, while the resulting cross-view score differences are written back through the local structural operator.

> All vision-language weights remain frozen and inference does not access ground-truth masks. Retained vocabularies, Kev admission thresholds, prompt templates, and some readout/background settings were selected using labeled development data. We disclose these task-specific choices and distinguish shared-step results from task-adapted results. The inspected full-validation sweeps are development analyses rather than independent test evaluations.

“扩词＋筛词”本身不是新的主贡献，VIP 已有扩词和视觉引导蒸馏。当前更有证据的主线是局部 query 条件化读取、DonorBefore 和结构写回；Kev 放语言接口或附录，除非后续同参数控制证明其具有稳定独立贡献。

## 6. VIP 官方实际配置了哪些参数

核对的官方提交：`5bd25ee03ec25c1538622cf7da661e8c0461e769`，项目 [MiSsU-HH/VIP](https://github.com/MiSsU-HH/VIP)。下表是源码配置事实，不证明作者使用哪个标签划分、什么搜索预算挑选了这些值。

| 数据集 | tau | tem | prob_thd | 输入 Resize |
| --- | ---: | ---: | ---: | --- |
| VOC20 | 4 | 1（默认） | 0（默认；无背景拒绝） | (2048,336)，保持比例 |
| VOC21 | 3 | 0.3 | 0.28 | (2048,336)，保持比例 |
| PC60 | 2.2 | 1（默认） | 0.10 | (2048,336)，保持比例 |
| ADE150 | 5 | 5 | 0.07 | (2048,336)，保持比例 |
| COCO-Object | 6 | 0.3 | 0.13 | (2048,336)，保持比例 |
| VDD | 1 | 1 | 0.35 | (448,448)，保持比例 |
| Potsdam | 1 | 2 | 0.25 | (448,448)，保持比例 |
| Vaihingen | 1 | 10 | 0.10 | (448,448)，保持比例 |

`tau` 控制 Alias 的 LSE 聚合锐度；`tem` 控制图像级 Alias 显著性权重的温度；`prob_thd` 在启用背景模式时将低置信度预测分配给背景。输入尺寸也随任务变化，例如 Cityscapes 为 (2048,560)，COCO-Stuff 为 (2048,448)。默认滑窗为 crop336/stride112，logit_scale=40；proxy attention 中 1.5 和 0.5 在所核对源码中写死，不能声称它们逐数据集搜索。

官方没有在此版本提供 UDD5 配置；本仓库的 UDD5 VIP 是外部词表上的迁移适配，不是 VIP 官方 UDD5 基准。官方参数摘录及行号保存于 [审计 JSON](ALIAS_TUNING_AUDIT.json)。

源码：[VOC21](https://github.com/MiSsU-HH/VIP/blob/5bd25ee03ec25c1538622cf7da661e8c0461e769/configs/cfg_voc21.py)、[Potsdam](https://github.com/MiSsU-HH/VIP/blob/5bd25ee03ec25c1538622cf7da661e8c0461e769/configs/cfg_potsdam.py)、[默认与读出](https://github.com/MiSsU-HH/VIP/blob/5bd25ee03ec25c1538622cf7da661e8c0461e769/dinosegmentor.py)。

## 7. Geov2 可以调什么，怎样比较

冻结权重不等于所有参数统一，也不等于开发过程无标签。可以报告有明确开发协议的任务适配，但应公开候选范围、使用的图像与标签、最终冻结值和比较预算。

| 参数 | 当前作用 | 建议 |
| --- | --- | --- |
| g0 | 结构修正步长；几乎不增加计算 | 最先调整；共享结果与任务适配结果同时保留 |
| 局部 T_L | 两路相对分数尺度 | 先沿用冻结值，后续独立校准消融 |
| 宽视野 tau | 词聚合锐度及词数偏置 | 可作读出校准；与换词的作用分开记录 |
| 背景阈值/偏置 | 仅影响有背景或残余类的协议 | 可以校准；偏置是我们额外的自由度，VIP 本表只有阈值，需披露 |
| Alias bank/模板/Kev 阈值 | 改变语义覆盖及词数刻度 | 属于语言输入适配；不能靠换词把统一 g0 写成统一模型 |
| QuerySlot K/beta/query temperature | 记忆容量与路由；可能影响成本 | 优先保持当前 24/0.5/0.1 全局固定 |
| 局部 strength | 会改变局部路径 | 保持 original，当前代码的数值 strength 会退回旧局部投影，不能继续声称 DonorBefore 都启用 |
| Geometry temperature/spatial sigma | 视觉关系 A | 保持全局固定，避免同时改变结构机制 |
| tem | 当前 QuerySlot salience 关闭 | 无作用，不纳入有效调参项 |

比较至少保留 VIP 官方词表/官方配置基线；同词表 VIP 是语言控制，不替代强原生 VIP。若声称“同等调参预算”，还需真的用同一开发划分和事先固定预算优化 VIP；给 VIP 使用专为我们选出的词表并不自动公平，已有 UDD5 同词表 VIP 曾明显下降。

当前全量扫描已经查看，按这些全量指标选出的例外属于后验任务适配。未来在未用于开发的新数据上评估，才提供新的泛化证据；不能把当前表称为独立测试或将词表选择写成纯无标签。

## 8. 现有数字支持的具体选择

UDD5 的当前词表＋g0=0.5 已是本轮扫描最佳，mIoU=55.2567%；相对旧 Efficient 的 55.8266% 仍低约 0.57pp，这不是本轮 g0 选错的证据。VOC21 保留现有词表，g0 从 0.5 改为 1，mIoU 从 72.1656% 到 75.2094%，+3.0438pp；仍比旧 Efficient 75.8586% 低 0.6492pp。

| 方案 | 默认 g0 | 例外 | 十协议平均 mIoU |
| --- | ---: | --- | ---: |
| 共享步长 | 0.5 | 无 | 55.8158% |
| 一个例外 | 0.5 | VOC21=1 | 56.1202% |
| 两个例外 | 0.5 | VDD=1、VOC21=1 | 56.3144% |
| 四个例外 | 0.5 | VDD/VOC21=1；OEM/Vaihingen=0.25 | 56.5417% |
| 逐域最佳参考 | — | 每个协议分别选择 | 56.7433% |

前四行由现有全量扫描直接重新汇总，无需重跑；后验组合不是新的独立验证，也不能保证后续数据保持相同收益。四例外对应当前使用统一 0.5 损失超过 1pp 的协议，但“超过 1pp”是查看本轮结果后的描述，不能追认为预注册规则。

建议保留共享 g0=0.5 的清楚基准；若优先兼顾少量例外与效果，可另外公开 VDD/VOC21=1、其他=0.5 的任务适配版。暂不为 UDD5 或为强行统一 g0 重新选词。后续换词实验应分开改变前景词、背景集合和词数偏置，先固定视觉观测与写回，避免将多项变化混成筛词收益。
