# Geov2：固定结构写回与步长分析

本轮模型采用 **2026-10-13 最佳 Alias + QuerySlot + DonorBefore + 固定结构写回**。自适应 gate 保留为候选消融，不作为默认最终模型。十协议扫描为 VDD、Potsdam、UDD5、OEM、Vaihingen、PC60、ADE150、VOC20、VOC21、COCO-Object81；PC60 替换 LandCover.ai。

## 完整流程

1. 整图缩放后只编码一次宽视野，用冻结 DINO 特征形成 24 个软槽。初始化、五轮球面聚类和软分配全部来自当前图像，不训练参数。宽视野的两个冻结语义 head block 通过槽执行 Value 聚合；该路径没有执行 VIP 的 self-correction。
2. 局部图像按照冻结的窗口策略编码。原始 patch 的特征相似性与空间距离构成视觉关系 A；每个局部窗口有 32×32 个 patch。
3. DonorBefore 保留 native attention 对特殊 token 的贡献和 patch 总质量，将 patch donor 条件化为自身，再用 A 运输 attention 增量，最后进入冻结 MLP。可写作 x' = x + A u_self，输出 x' + f(x')。特殊 token 不做该空间运输。本轮所有数据集 strength 均为 original，使模块实际执行；数值 strength 的历史配置不会执行该分支。
4. 局部原始 patch 作为 query，在整图槽中心上计算温度为 0.1 的相似度 softmax，读出槽语义。将其与对应坐标的宽视野语义按 beta=0.5 混合并归一化。这使宽视野读出已经依赖局部 query；两路不只是最终 logit 融合时才发生联系。
5. 使用冻结文本缓存得到局部 class scores L 和 QuerySlot 对齐的宽视野 class scores W，随后执行下面的结构写回。窗口内上采样、softmax、Hann 概率拼接，最后恢复原图尺寸并应用已有背景校准。

实现仍保留局部与宽视野两个视觉实例的原有数值角色；不能把另一个 native-wide 单实例候选的显存数字用于本模型。`FrozenGeometry` 中的 `vip` 变量和 `VIPQueries` 数据容器保留了历史命名；QuerySlot 路径实际宽视野计算由 ContextMemoryObserver、read_context_head 和 query_slot_features 完成。官方 VIP 是独立比较基线。

## 固定结构写回

将无效位置屏蔽、重新归一化后的局部关系记为 A。本轮使用的目标为

\[
 Z^\star=\arg\min_Z\frac12\|Z-L\|_F^2+\frac12\|A(Z-W)\|_F^2.
\]

其闭式解为

\[
 H=(I+A^\top A)^{-1}A^\top A,\qquad
 Z^\star=L+H(W-L),\qquad Z=L+g_0(Z^\star-L).
\]

本轮历史实现采用 FP64 Cholesky；`geov2.py` 的 degree-weighted FP32 native-wide 候选是另一条实验路径。二者不可混称为同一个被评测的算子。

g0=0 为局部端点，g0=1 才是该目标的精确最优解。0<g0<1 是阻尼，g0>1 是超松弛。沿这条直线，二次目标在 g0=1 最低；分割 mIoU 的最优 g0 不必等于 1，因为二次一致性目标并非真实标签损失。不能把任何 g0 的输出都叫作该目标的最优解。

若 A^T A 的特征值为 rho，则 H 在该方向的增益为 rho/(1+rho)。弱结构方向的差分受到抑制；这不是所有边界或细节绝对不被改写的保证。W=L 时恒等。大步长也会放大宽视野错误，必须如实报告下降的协议。

## Alias 与步长为何相关

每个类有冻结短语集合。局部用温度 0.07 的 log-mean-exp 聚合，再除以任务已有的 local temperature。宽视野使用 40 倍模板均值点积和按类 log-sum-exp；当前 CompactContextScorer 的 salience 关闭，不能把旧 VIP 的图像 salience 讲成本轮模块。

若类 c 有 k_c 个同分 Alias，宽视野 LSE 相对 LME 会多出 log(k_c)/tau。不同类别词数不等时，这相当于 class bias；结构写回会把这种 bias 一并乘上 g0。因此 g0 不只取决于画面尺度或场景，还受词表、两路分数刻度和背景规则影响。

本轮对同一组视觉观测比较三种语言接口：

- retained：保留已有最佳 Alias。
- canonical：每类仅保留第一条 canonical phrase，模板不变。
- count_normalized：所有 Alias 不变，只从宽视野 class score 扣除 log(k_c)/tau。

canonical 同时改变词义覆盖和词数；count_normalized 只隔离显式词数偏置，不消除增加短语带来的全部统计影响。PC60 在三种接口下都保留 401 个残余概念及 residual_max。已有词表、温度、背景校准经过带标签开发选择；推理不读标签不代表开发过程从未用标签。

## 本轮实验与结论边界

固定扫描 g0 = 0、0.25、0.5、0.75、1、1.5、2、3，另加宽视野端点及字面意义的等权 logit 融合。三种语言接口共 30 个同观测 arms。评测全量验证集，mIoU 从累计混淆矩阵计算，不平均逐图 IoU。

报告包括十协议等权平均、五遥感/五自然平均、逐域最优参考、统一步长损失、留一数据集选步长，以及 VOC20/VOC21 共同留出的版本。逐域最优是带标签调参参考；留一只隔离 g0 选择，其他历史配置仍为任务特定。

统一 g0 能否取代逐域 g0，应由平均增益、最坏域损失和留出表现共同判断。Alias 相关性只能作为十个协议上的描述，不能据此声称已经得到可靠的无标签自动选步长规则。

完整结果与曲线：`research/geov2_g0_sweep_20261014/REPORT.md`；原始逐图充分统计、冻结词表与配置同目录保存。扫描的 30-arm 墙钟耗时不是单次模型延迟，不能用于速度主表。
