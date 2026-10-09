# 最终模型的论点、证据与投稿缺口

本文件是方法/Introduction草稿与主张清单。冻结评测及交付已完成，全部15协议按预定全局判据选Uniform；正式终版结果见 `CVPR_FINAL_MODEL_20261009.md`。下文的执行步骤保留原计划语境，不需要重启实验或另建重复goal。

## 核心判断

目前问题不是“缺一个权重公式”，而是尚未找到能把正确补充覆盖与错误高响应区分开的可靠证据。视觉响应、文本质心、邻域一致性都是相关信号，却可能共同支持同一个错误类别。Astra xhigh的有界审查没有发现值得继续换公式试验的新估计器，因此停止新增筛词路线。

原高分辨率 RivalFineHard 的同源增益是真实的，但包含新增细视野信息，不能移给有界轻量模型。FamilySUM 的ADE增益被类偏移完整解释；跨支持删词价值相关0.00416落在置换区间；最近四域语义约束仅有近零增益。它们分别否定当前模块的增益归因、当前估计器和当前候选的实用价值，不证明所有可能的筛词都不成立。

保守普通局部软处理在VDD/Potsdam仍有约0.108/0.061pp收益，配对区间方向稳定；这是很小的经验收益，不能据此说“完全没用”，也不能升级为已验证的坏词识别。移除它的意义是简化论点与模型，不是保证速度更快或精度完全相同。

## 三个职责如何保留

| 用户希望的职责 | 实际可支持的实现 | 当前可提出的贡献 |
| --- | --- | --- |
| 在哪里读 | 原视觉描述子的结构关系进入冻结语义head；固定局部视野预算 | 有具体算子与读取通路，需要按实际patch-only/original路线陈述 |
| 凭什么判 | 预训练图文语义响应、局部LME、继承的图像salience；可供给变长词库 | 是必要的语义接口；尚不支持独立的竞争可靠性估计创新 |
| 修正能写到哪里 | 宽视野与局部差分经原Geometry关系构造的H写回 | 同观测相对等权融合的增量与误差可测，不能宣称所有域均比局部好 |

完整论文不要求三个职责都各有一个新模块。真正成立的两个机制加清楚的归因，比三个框中一个无法证明的语义筛选主张更可靠。若最终普通权重保留，称经验启发式；PC60额外残余概念的protected竞争规则须单列，不随普通前景权重开关撤除。

## 方法正文草稿

Let a frozen visual backbone yield patch descriptors \(f_i\) and patch coordinates \(x_i\). We form a row-normalized visual relation

\[
A_{ij}=\operatorname{softmax}_j\left(\frac{\langle\hat f_i,\hat f_j\rangle}{t_g}-\frac{\|x_i-x_j\|^2}{2\sigma^2}\right),
\qquad t_g=0.10,\quad\sigma=0.25.
\]

The relation is constructed from backbone descriptors and reused in two frozen semantic-head blocks. It is not recomputed from the modified semantic states. Prefix queries retain their native path. In the original route, patch queries retain native prefix contributions and native patch-group mass while replacing the conditional patch relation. In the patch-only route, patch queries read scaled \(sAV\), \(s\in\{1,2,3\}\), without prefix Values; residual and MLP computations remain. These are distinct operator settings. The mass-preservation proposition applies to the original route, not automatically to patch-only settings.

Normalized resulting descriptors are matched to a declared language bank. Ordinary local aggregation is

\[
L_{pc}=\frac{T}{t_l}\left[\log\sum_{a\in\mathcal A_c}\exp(s_{pca}/T)-\log K_c\right],
\qquad T=0.07.
\]

Here \(t_l\) is the declared local profile temperature; the implementation applies the alias LME first, then this profile scale. The LME derivative with respect to an alias response is proportional to \(\operatorname{softmax}_a(s_{pca}/T)\). Consequently, uniform input coefficients do not imply equal effective word influence. This response dependence does not establish semantic reliability.

The wide-view semantic field \(W\) retains the attributed VIP observer, image-conditioned salience, scoring and numerical empty-row repair. We do not relabel the observer as a newly invented visual reader. RS uses bounded long-edge448 views; natural images use short-edge336 with long-edge cap672. Each branch uses at most four backbone encodings; there are no extra fine-view encodings. Local heads of different strengths reuse backbone tokens, but their head computations are included in deployment latency.

On the local valid-patch grid, invalid donors and rows are masked and the relation is renormalized before forming

\[
H=(I+A^\top A)^{-1}A^\top A,\qquad Z=L+gH(W-L).
\]

At \(g=1\), this is the classical solution to \(\min_Z\|Z-L\|_F^2+\|A(Z-W)\|_F^2\). The contribution being investigated is the specific visual relation/reading pathway and its useful coupled correction under a matched observation budget, not the invention of a quadratic inverse. At \(g=2\), the implementation extrapolates the correction and does not solve the same objective. A PSD spectral filter can have negative entries; neither positive influence everywhere nor semantic correctness is guaranteed.

RS selects between two predeclared text/readout routes through canonical image margins, retaining \(g=0.5\). Natural tasks use a developed canonical-text-rank menu. The procedure is mask-free at inference, but its menus and words have labeled-development provenance. We describe training-free inference and release these choices rather than claim parameter-free optimization.

For fixed geometry and gain, any language intervention obeys

\[
\Delta Z=(I-gH)\Delta L+gH\Delta W.
\]

This makes the competing-class consequence explicit: modifying non-road evidence can improve or impair road IoU through score competition and structural writeback. Alias-response reduction alone is not the desired endpoint. A word-level estimator must justify its final correction field beyond a simpler class-level calculation; equal correction fields necessarily give equal predictions.

## Introduction草稿

Frozen vision-language models support dense prediction without task-specific weight updates, yet their local spatial support and semantic evidence need not agree. A visually coherent region can receive an incorrect label, while a correct wider-view semantic observation can overwrite local detail when it is transferred directly. The central problem is therefore how to organize spatial reading and how to constrain the correction supplied by another view.

We study a structure-guided readout of a frozen semantic head and a relation-supported cross-view correction. A backbone-derived relation organizes local Value reading; the local prediction provides an anchor; and an explicitly attributed wide-view observer supplies a semantic residual whose writeback uses the same visual structure. Matched observations and endpoint controls separate the benefit of obtaining another view from the benefit of the correction mechanism.

Language expansion introduces competing evidence, but high response is not equivalent to reliability and low response is not equivalent to uselessness. Our experiments distinguish language aggregation, category calibration and additional visual observation. The tested lightweight alias heuristics do not establish a transferable semantic-reliability estimator; this negative result limits the proposed contribution rather than motivating an unsupported additional module.

We evaluate one frozen policy across available remote-sensing and natural-image protocols, report complete class competition and background metrics, and measure synchronized full-image latency and memory. The evaluated domains informed prior development, so these results establish exploratory system behavior rather than untouched generalization. Closest-operator comparisons and future independent validation remain separate requirements for a strong novelty claim.

## 现有主张的证据等级

| 主张 | 已有证据 | 结论/缺口 |
| --- | --- | --- |
| 原Geometry不只是平滑 | 原生匹配SpatialOnly/NativeSpatial全量、读取通路审计 | 在该协议成立；不能搬用到所有patch-only强度 |
| 原Geometry优于所有最近算子 | 原生SCLIP/VIPProxy各仅2/8胜 | 不成立，撤回普遍优越性 |
| 有界patch-only2有最近算子优势 | 20261005与SCLIP coupled全量八域平均+0.3075pp，6/8胜 | 是DINO.text适配结果；同时改变prefix/patch读取，非孤立affinity或当前任务路由证据 |
| 结构写回胜过简单融合 | 原固定patch-only2八域+0.8686pp；当前主要四协议Uniform平均+3.3146pp，配对区间[+3.0130,+3.6391] | 两配置分别引用，不拼表；当前15协议仍需齐全 |
| 宽视野修正始终正向 | 原固定模型和当前模型均有低于局部的域 | 不成立；报告局部强而融合负迁移的类别 |
| 普通软权重是有效的词义筛选 | 当前小幅增益、校准解释、失败估计器 | 不成立为主贡献；允许称小收益启发式 |
| 当前任务模型全面超越VIP | 目前VDD/Potsdam/ADE领先，VOC20/21与PC60落后于本地声明配置VIP | 不成立；本地数值修复比较也不等于论文认证复现 |
| 自然普通权重开关不掉分说明泛化有效 | 两版本自然普通前景本来相同 | 不能作为词处理跨域成功证据 |
| training-free意味着不调参或极快 | 未更新网络权重，存在已公开开发菜单和多视野 | 需要报告开发来源与整图成本，不承诺毫秒级任意尺寸 |

## 接下来只完成有决定作用的工作

1. 已运行队列完成全部15协议及自然VIP修正尺度计时；不新增权重公式、词表搜索或新评测配置。
2. 对现有逐图混淆矩阵进行配对source-group统计和类别覆盖/误激活分析，无推理重跑。TP/union充分统计与完整混淆矩阵bootstrap已做数值等价核验，避免分析阶段的类别平方张量。
3. 按冻结的全局非劣门槛选一个普通权重policy；即使一个历史域略好也不能挑域开关。统一简化版通过才选它，否则全局保留经验权重。
4. `tools/finalize_geometry_model.py`要求全量精度、完整修正成本与全量配对统计齐全才写 `final_model.json` 和最终结果报告；缺失时拒绝提前冻结。
5. 本轮交付包括真实的失败结论、架构/API配置、方法/Introduction、匹配消融和成本。CVPR投稿后续还缺最终任务路由的最近算子匹配、真正新来源/官方测试、边界/目标尺度证据及文献优先权核查。它们不能被“讲好三个模块的故事”代替。

计算和时间的实测细节见 `ALIAS_FINALIZATION_20261009.md`；统计见 `alias_finalization_20261009/statistics_partial10/PAIRED_UNCERTAINTY.md`。部分结果不可冒称最终全量结果。
