# CVPR 终版定案：执行协议与方法草稿

2026-10-09。冻结执行已完成：全部15协议、完整成本与配对统计通过；主终版选择全局Uniform。正式结果与冻结配置见 `CVPR_FINAL_MODEL_20261009.md` 和 `alias_finalization_20261009/final_model.json`。本文保留执行协议和方法草稿，以下“待完成”阶段描述属于原计划，不代表需要重启；结论不是全面优于VIP。

## 科学决策

结束新增条件筛词公式的搜索，执行保留版与统一简化版的有限定案。Astra xhigh 已审阅已有证据，没有找到值得占用下一轮试验的实质新可靠性估计器。更强的 sigmoid、换名后的 cosine/agreement、再次构造词族质心，都不足以改变这个判断。

这不是证明所有筛词都无效。原生高分辨率 RivalFineHard 的八域同源增益存在，但它依赖额外细视野；不能把它的准确率贡献移给当前零细视野模型。轻量候选的问题主要是不能区分错误高响应与正确补充覆盖，不再只是实现速度。

已有四域竞争语义约束全量增量为 VDD +0.0005、Potsdam 0、ADE +0.0269、VOC20 +0.0036pp；开销 +0.85–4.84%。它虽便宜，却不足以成为主贡献。ADE FamilySUM 的约 1.98pp 提升又被类别偏移对照完整解释。跨支持删词价值估计相关 0.00416，落在置换区间内。应撤回当前“能够判断像素级坏词”的主张。

## 统一模型接口

保留 `TaxonomyInference.for_task` 的原配置与所有历史结果。新增 `for_finalization(..., ordinary_alias_policy=...)`，明确比较两个全局选择：

| 项目 | Retained | Uniform |
| --- | --- | --- |
| Geometry、视觉尺度和 H | 相同 | 相同 |
| RS 词表、文本模板与图像选库规则 | 相同 | 相同 |
| 自然图像词表、文本秩选读出规则 | 相同 | 相同 |
| 普通前景局部额外软权重 | RS 开；自然关 | 所有任务关 |
| 局部 LME 与宽视野图像 salience | 保留 | 保留 |
| PC60 的 401 概念残余本体与 protected rule | 保留 | 保留 |

Uniform 不是让每个词的实际影响恒等。局部 LME 的响应导数是 softmax，宽视野也已有图像条件 salience。此处取消的是新增 coverage-soft 启发式，不能宣传成取消一切动态语言处理。PC60 的残余规则属于另一个显式信息预算，不把其收益归到前景 alias 筛选。

当前任务路由来自既有有标签开发：RS 在两个冻结库中用图像 canonical margin 选择 ImageNet/strength1 或 RS6/strength3，g=0.5；自然根据文本有效秩选择既定的 g/strength 菜单。没有 dataset-name 分支不等于这些菜单未经过开发。它是 training-free inference，不是 parameter-free 或 untouched evaluation。

## 全量清单与固定取舍

冻结清单：LoveDA 1669、UDD5 40、OEM 384、VDD 80、Potsdam 504、Vaihingen 113、LandCover.ai 1602、FLAIR-1 15700，共 20092 个遥感输入。

自然协议：VOC20/21 各 1449、PC59/60 各 5105、ADE150 2000、COCO Object81 5000、COCO Stuff171 5000，共 25108 次自然协议评测。因此总数是 45200 次协议图像评测，而不是 45200 个独立图像；两组 VOC 和 PC 分别共享来源。Cityscapes 不在当前可用冻结清单；LandCover.ai 代替不可标注评测的 iSAID 镜像。

LoveDA P 在删除背景文本的六类银行上独立读出，D 使用完整七类银行。不能把 D argmax 减一当成 P。八域均值只计 D，P 单列。VOC20 与 VOC21、PC59 与 PC60 的结果不相互替代。

统一简化版通过条件：VDD、Potsdam、VOC21、ADE 四个主要协议的平均 Uniform−Retained 不低于 −0.1pp、最差不低于 −0.3pp；其余完整迁移协议也不得低于 −0.3pp。通过才全局选择 Uniform，否则全局保留 Retained，并将旧权重如实称为经验启发式。不能按域选开关，不能在看结果后修改词表、阈值或门槛。

这只是是否删除额外软权重的模型决策线，不是统计显著性判据、普适非劣保证或 CVPR 录用标准。两版自然普通权重本来相同，预期自然开关为零；这不能算证明筛词泛化有效。迁移本身仍检验整个模型的新组合是否可用。

全局取舍以清单15个dataset/protocol入口为单位；LoveDA主入口取D，P作为独立读出的附加协议单列。冻结代码中的transfer worst按这15个主入口计算，不能把它写成额外保证LoveDA P也满足同一非劣界。P的完整结果与风险仍必须报告。

同时保存同观测控制：Uniform 局部端点、宽视野端点、等权 logit 融合，以及原20词的同视野参考。PC60 的 protected residual 在端点控制中仍使用耦合前景竞争者，因此它们是最终读出端点，不能冒称孤立的局部/宽视野系统。其余协议的端点使用各自已有分支分数。

## 框架与方法草稿

```mermaid
flowchart LR
    X[RGB 图像] --> LV[局部视野：长边896，最多4窗]
    X --> WV[宽视野：最多4窗]
    LV --> F[冻结 DINO 视觉特征]
    F --> A[视觉关系 A]
    A --> GH[Geometry 局部语义读出]
    F --> GH
    T[冻结任务词表与模板] --> GH
    T --> WR[VIP 来源宽视野语义观测]
    WV --> WR
    GH --> L[局部分数 L]
    WR --> W[宽视野分数 W]
    A --> H[结构写回算子 H]
    L --> D[修正场 W-L]
    W --> D
    D --> H
    H --> Z[Z=L+gH(W-L)]
    L --> Z
    Z --> O[概率拼接与原尺寸预测]
```

自然宽视野 short336/cap672；遥感宽视野 long448。4 个局部和4 个宽视野是每幅完整图像的编码上限，不是每个滑窗再开4个细视野。多种局部 head 强度重用冻结 backbone tokens，不是新增 RGB/backbone 编码；但 head 计算仍需计费。

Geometry 的任务是建立视觉读取/支持关系，不认证文本语义。原 Geometry 在两个 DINO.text head block 中保留原生特殊 token 贡献与 patch 总质量，只替换条件 patch 关系。当前 strength1/2/3 的 patch-only route 阻断 patch 查询读取 prefix Values，并按声明强度读取 patch Values；`original` route 则保留原条件质量机制。这些路线不共享全部守恒性质，方法正文必须区分。

设有效 patch 关系为 A，固定写回为：

    H = (I + A^T A)^(-1) A^T A
    Z = L + g H (W-L)

g=1 时 Z 是以下经典二次目标的闭式解：

    min_Z ||Z-L||_F² + ||A(Z-W)||_F²

这个目标把局部 fidelity 与视觉关系上的宽视野一致性放在同一读出中；线性代数/岭式解本身不是新发明。g=0.5/2 是已开发的修正缩放，其中 g=2 可外推，不能说它同样精确求解上述目标或拥有相同局部保真保证。H 虽为对称正半定谱滤波器，元素仍可为负；语言降权经写回后不保证输出处处下降。

任何额外词处理都会改变 L/W，完整效应为：

    delta Z = (I-gH) delta L + gH delta W

所以必须同时量化本类覆盖和竞争类误激活。不存在“只处理 road 的词，才影响 road IoU”的分离。给 wall 加 roof 的错误归属既改变 wall 的分数，也改变 roof 的竞争边界；删词降低 FP 的同时可能损失另一处正确覆盖。词身份相似度或邻居一致性不是可靠性真值。

## Introduction 的主线

冻结视觉语言模型用于密集预测时，局部空间支持与类别语义竞争会发生两种不同错配：视觉关系可能保留边界却不能选择正确类别，宽视野语义可能识别类别却在写回时覆盖局部细节。提高全局响应或永久删除语言表达，都不能直接解决这两个错配。

本工作的研究问题应是：如何利用同一冻结模型的视觉结构，组织局部读取并限制宽视野语义修正的空间通路。Geometry 提供局部的关系组织；来自明确归属的宽视野观测产生跨视野修正；结构写回保留局部锚点并在关系支持上利用修正。语言聚合作为输入接口接受不同数量的表达，其额外语义筛选主张由消融决定，而不是先验要求第三个框必须存在。

论文贡献候选是一个明确的关系读取算子，以及同信息预算下有用的结构耦合机制。VIP 宽视野观测、LME、图像 salience、经典二次重建和任务调参都应归属/披露。不能把“用了不同的组合”写成每个继承部件都新，也不能把原 Geometry 的守恒性移给 patch-only2。

## CVPR 证据边界

20261005 PatchOnly2 原20全量八遥感平均 47.6399，强测量 VIP 平均44.0039；同信息等权融合46.7713、局部47.0452。原完整耦合相对等权融合八域点估计均更高，但相对局部有多个域退化。它是保留的独立性能/归因参考，不可与本轮任务配置拼表冒充同一模型。

20261007 任务候选 VDD55.2382、Potsdam49.7888、VOC21 70.4508、PC60 41.6004、ADE31.1862；对应本地 finite official-query VIP52.0647、44.8995、73.2591、42.5987、29.1387。它只胜3/5，不能写成自然与遥感全面 SOTA。词表/残余本体不同的完整系统比较不能单独归因 Geometry。

近邻对照必须匹配最终算子、分辨率与读出。已有20261001全量原 Geometry 近邻实验使用原生分辨率和条件质量保持路线，不能直接等同于当前 bounded patch-only strength1/2/3 的终版结果；引用时标明迁移差异。20261005 另有 bounded 固定 patch-only2 与 SCLIP coupled 的全量八域对照：前者平均 +0.3075pp、6/8 胜。它们是 DINO.text 中的算子适配，不是官方 SCLIP 系统；prefix/patch 读取同时变化，未隔离 affinity kernel，且不等同于当前 strength1/3 任务路由。已有此证据，不能写成完全没有最近算子对照。当前未新增文献检索，不做未经核实的优先权承诺。

现有全部主要域长期参与开发，本轮冻结不消除选择偏倚。完整论文仍需真正未参与方法/词表选择的新区域、来源或官方测试条件；旧验证集再次重分组不能获得 untouched validation。

## 成本与复现

独立进程计时比较 Retained、Uniform、同冻结词池的 VIP 和作者声明配置 VIP。每协议7个预先均匀索引完整图，每图两次预热、7次 CUDA 同步；报告中位数/p95、实际编码次数和 peak allocated/reserved。包括缩放、所有视觉前向、词聚合、写回、原尺寸恢复及 CPU 预测输出，解码/初始化/文本编码另报。它不是全量平均延迟；其他 GPU/主机负载仍可能影响绝对时间。

局部 alias 点积/归约为 O(PCKD)/O(PCK)，没有位置×类×词×全部竞争类张量。Geometry A/H 的 P²存储和固定窗二次求解仍存在，不声称线性 attention。额外 alias/语义头目标不超过母模型20%，整图目标2–3×公平VIP，超过6×停止扩展。

遥感不支持官方 VIP 词表的五域，成本中的 VIPOfficial 标明外部原20适配；不能将其称作者官方该域结果，也不是蒸馏后短词表的耗时。同词池计时并非同视觉观测对照：VIP 保留其标准视野，RS 不运行我们的 Geometry 选库。读出归因由本轮同观测端点/融合控制负责。

本轮可执行文件：

- `tools/alias_finalization_experiment.py`：prepare/deploy/launch/status/collect；不恢复其他暂停自动化。
- `DINOtool/scripts/eval_alias_finalization.py`：无标签 API/scalar 检查、完整 P/D/自然协议和逐图混淆矩阵。
- `DINOtool/scripts/run_alias_finalization.py`：仅空闲GPU队列、完整覆盖/目标计数核验，错误时保留原输出。
- `DINOtool/scripts/benchmark_alias_finalization.py`：每个候选独立进程的完整图像成本。
- `research/alias_finalization_20261009/protocol.json`：冻结输入、门槛、路线与声明。

首次 LoveDA 无标签检查错误为 `AttributeError: 'LoveDASample' object has no attribute 'read'`：底层 RGB 加载器接收路径，外层传了样本对象。已修复 `sample.image_path` 接口，扩充加载器测试；失败目录/日志保留，修复检查单独写入 `smoke_loader_repair/loveda`。其余完成检查与 VDD 已完成成本复用，恢复状态与最终结果另存，不改模型/词表/阈值。

第二处是核验器的历史对象命名错误。冻结清单中 `expected_retained_scalar` 实际来自 `NaturalShortEdge` 的 packed 部署输出，不是本轮独立 scalar 归约。VDD 的 Retained 完整混淆矩阵与历史完全一致；scalar 混淆矩阵 L1=36，差异出现在已记录的浮点近并列处（最大 argmax 间隔约7.5e-8、概率差约1.8e-7），两者 mIoU 都55.2382。核验改为部署复现部署，独立 scalar 仍逐图执行严格概率/近并列检查。原错误合并与控制器失败文件保留；验证恢复写 `verified_merged.json`、`verification_status.json` 和 `verification_results.json`，复用所有完整推理。

成本审查还发现自然 VIP 的作者配置不能仅靠 `VIPSettings` 复现：历史完整 comparator 使用 short-edge336/maxlong2048，而 adapter 的通用 `predict` 使用 long448。已有自然 `VIPOfficial.json` 是官方词表/settings＋long448 的尺度控制，不是自然官方协议计时。保留这些原文件，新增 `VIPOfficialProtocol.json` 调用同一个已验证的 `official_prediction`；只在主队列完成后启动7个独立计时进程。最终成本表排除旧自然 long448 的“官方”比率。模型/图像评测不重启，RS 成本不受影响。

下一次 goal 继续时先读取 manager status；不要重启重复实验。只有全部证据、成本、最终架构/方法和贡献边界定案后才完成 goal。
