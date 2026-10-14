# Geov2 方法

## 3. 方法概述

设输入图像为 \(I\)，输出类别数为 \(C\)。Geov2 以冻结的 DINO/dino.txt 视觉语言模型为基础，在同一幅图像上建立局部视野和宽视野两种观测。局部视野保留高分辨率细节，宽视野提供跨区域语义上下文；两路观测最终都映射到同一个 \(C\) 类 logit 空间。模型不更新视觉或文本权重，也不使用 SAM 或训练好的分割头。

Geov2 的处理链路为：

1. 从局部窗口估计视觉关系矩阵 \(A\)，并用 DonorBefore 顺序得到局部特征 \(F^L\)；
2. 对整图宽视野建立容量受限的 QuerySlot 记忆，并由局部 raw token 路由读取语义上下文，得到宽视野特征 \(F^W\)；
3. 用冻结文本查询把两路特征转换为局部和宽视野 logits \(L,W\)；
4. 使用同一个视觉关系 \(A\) 构造结构写回算子 \(H\)，将经过结构筛选的跨视野差分写回局部 logits；
5. 对各局部窗口进行上采样、Hann 加权拼接，并恢复到输入图像尺寸。

因此，局部和宽视野的耦合发生在两个位置：QuerySlot 在**读取端**使用局部视觉内容选择宽视野上下文，结构写回在**输出端**使用局部关系约束跨视野差分。两路不是最后才进行等权平均的独立分支。

## 3.1 冻结视觉语言读出

给定一个类别 \(c\)，记其冻结 Alias 集合为 \(\mathcal A_c\)，其中每个 Alias 是一个文本查询，且每个查询保留一个父类别映射 \(p(a)=c\)。文本查询在推理前编码并缓存；推理过程只进行视觉特征与缓存文本特征的点积。Alias 的作用是扩大类别的语言覆盖并完成类内聚合，不改变输出类别数，也不参与 QuerySlot 的槽分配。

局部特征与 Alias 的分数经过温度为 \(\eta\) 的 log-mean-exp 聚合：

\[
 L_{ic}=\frac{\eta}{T_L}
 \left[\log\sum_{a\in\mathcal A_c}
 \exp\left(\frac{s^L_{ia}}{\eta}\right)-\log|\mathcal A_c|\right],
\]

其中 \(T_L\) 是任务配置的局部分数校准温度。宽视野沿用冻结的宽分支聚合规则。当前实现先对模板向量求均值，再计算宽视野特征与 Alias 的余弦响应，并使用继承的 log-sum-exp：

\[
 W_{ic}=\frac{1}{\tau}\log\sum_{a\in\mathcal A_c}
 \exp\left(\tau\,40\,\langle f^W_i,t_a\rangle\right).
\]

两路共享 Alias 的父类别映射和背景规则，但由于局部与宽视野采用的冻结模板可以不同，二者的文本向量不必逐项相同。这样可以区分“共享类别语义接口”和“视野观测差异”。

## 3.2 局部结构关系与 DonorBefore

输入图像先按冻结协议缩放。局部视野使用至多四个 \(512\times512\) 窗口，每个窗口被划分为 \(N=32\times32\) 个 patch。对第 \(i,j\) 个 patch，使用 DINO raw 特征的余弦相似度和归一化坐标距离构造结构 logits：

\[
 r_{ij}=\frac{\langle \bar q_i,\bar q_j\rangle}{\tau_A}
 -\frac{\|u_i-u_j\|_2^2}{2\sigma_A^2},
 \qquad
 A_{ij}=\operatorname{softmax}_j(r_{ij}).
\]

当前冻结实现使用 \(\tau_A=0.1\)、\(\sigma_A=0.25\)，并对越界 patch 进行有效性屏蔽。\(A\) 只由当前局部图像的冻结视觉特征和位置构成，不包含文本或标签。

原始 DINO head 在每个冻结 attention block 中产生 native attention 增量。DonorBefore 保留特殊 token 的原始贡献和 patch 的自身质量，将 patch 增量先沿 \(A\) 运输，再进入后续冻结 MLP。对 patch token 可概括写为：

\[
 \widetilde u_i=\sum_j A_{ij}u^{\rm self}_j,
 \qquad
 x_i'=x_i+\widetilde u_i,
 \qquad
 y_i=x_i'+f(x_i'),
\]

其中 \(u^{\rm self}\) 是保留 native attention 自身质量后的 attention increment，\(f\) 是冻结的归一化、MLP 和缩放层。运输发生在 MLP 之前，因此局部结构先决定哪些响应可以作为 donor 传递，再由冻结非线性完成语义响应。Geov2 在当前正式配置中对原始强度使用 DonorBefore；数值 strength 的历史分支仍保留为对照，不能与该分支混称。

## 3.3 QuerySlot 宽视野记忆

宽视野图像只通过冻结 backbone 编码一次。设其 raw patch 特征为

\[
 R=(r_1,\ldots,r_M)\in\mathbb R^{M\times d_r}.
\]

我们用容量为 \(K\) 的软槽表示整图上下文。当前正式配置为 \(K=24\)，以 raw 特征进行五轮确定性的球面 Lloyd 更新，然后得到软分配矩阵 \(P\in\mathbb R^{M\times K}\)：

\[
 P_{mk}=\operatorname{softmax}_k
 \left(\frac{\langle \bar r_m,c_k\rangle}{\tau_q}\right),
 \qquad \tau_q=0.1.
\]

对每个槽 \(k\)，先用同一分配矩阵聚合 raw Key 和冻结语义 Value：

\[
 c_k=\operatorname{norm}\left(\frac{1}{m_k}
 \sum_mP_{mk}\,\operatorname{norm}(r_m)\right),
 \qquad
 v_k=\operatorname{norm}\left(\frac{1}{m_k}
 \sum_mP_{mk}\,s_m\right),
 \quad m_k=\sum_mP_{mk}.
\]

这里 \(s_m\) 不是新训练得到的类别向量，而是把 raw token 送入冻结 dino.txt head、并用同一槽记忆读取得到的视觉语义特征。因而 Key 负责视觉寻址，Value 负责在冻结语义空间中传递上下文。

对于局部窗口中的 patch \(i\)，使用该位置的局部 raw token \(q_i\) 计算槽路由：

\[
 \rho_{ik}=\operatorname{softmax}_k
 \left(\frac{\langle\operatorname{norm}(q_i),c_k\rangle}{\tau_q}\right),
 \qquad
 v_i^{\rm route}=\sum_k\rho_{ik}v_k.
\]

同时，从整图语义网格按坐标采样得到 \(v_i^{\rm coord}\)。最终宽视野特征为：

\[
 f_i^W=\operatorname{norm}\left((1-\beta)v_i^{\rm coord}
 +\beta v_i^{\rm route}\right),
 \qquad \beta=0.5.
\]

这一步是 Geov2 的读取耦合：同一整图槽库会被不同局部 query 读取成不同的宽视野特征。QuerySlot 不直接读取文本，也不把 24 个槽解释成 24 个物体或类别；它只提供容量受限的视觉上下文寻址。它相对于完整 \(M\times M\) 全局读出的额外路由计算为 \(O(MKd_r+NKd_r)\)，并且只保存槽分配和槽值，不显式构造 \(M\times M\) 记忆矩阵。

## 3.4 结构锚定的跨视野写回

局部和宽视野读出得到同一 patch 网格上的 logits \(L,W\in\mathbb R^{N\times C}\)。直接使用 \(W\) 或等权平均会把宽视野的区域级错误直接传播到局部细节，因此 Geov2 只允许跨视野差异沿局部视觉结构传播。具体地，定义结构锚定目标：

\[
 Z^\star=\arg\min_Z
 \frac12\|Z-L\|_F^2
 +\frac12\|A(Z-W)\|_F^2.
\]

第一项使结果贴近局部观测，第二项要求结果在 \(A\) 所表达的视觉关系上与宽视野观测一致。令

\[
 H=(I+A^\top A)^{-1}A^\top A,
\]

则闭式解为：

\[
 Z^\star=L+H(W-L).
\]

实际部署采用冻结步长 \(g_0\)：

\[
 \boxed{Z=L+g_0H(W-L)}.
\]

其中 \(g_0=0\) 为局部端点，\(g_0=1\) 到达该二次目标的闭式解，\(0<g_0<1\) 是阻尼，\(g_0>1\) 是超松弛。默认共享配置使用 \(g_0=0.5\)；若报告任务适配结果，必须同时公开开发集、候选范围和冻结后的例外值。推理时不读取标签。

该写回具有三个直接性质。第一，若 \(W=L\)，则 \(Z=L\)，跨视野一致时不会引入修正。第二，若 \(A^\top A\) 在某方向的特征值为 \(\lambda\ge0\)，则 \(H\) 在该方向的增益为

\[
 \frac{\lambda}{1+\lambda},
\]

因此结构弱的方向被抑制，结构强的方向才更容易传递。第三，\(H\) 是局部关系的函数，所以同一个关系同时服务于局部读取和宽视野写回，形成视觉结构上的双向耦合。需要强调的是，\(H\) 约束的是空间传播方向，并不认证宽视野类别判断一定正确，也不保证所有边界细节绝对不变。

## 3.5 窗口拼接与输出

对每个局部窗口，\(Z\) 在 \(32\times32\) patch 网格上双线性上采样到窗口大小。窗口内 logits 先沿类别维 softmax，再用 Hann 权重进行重叠区域拼接：

\[
 p(x)=\frac{\sum_t h_t(x)\operatorname{softmax}(Z_t(x))}
 {\sum_t h_t(x)}.
\]

随后将拼接结果双线性恢复到原图分辨率，并应用冻结的背景偏置、背景阈值或残余概念规则。所有这些步骤均为后处理，不引入可学习参数。

## 3.6 复杂度与推理预算

设局部 patch 数为 \(N\)，宽视野 patch 数为 \(M\)，槽数为 \(K\)，类别数为 \(C\)，Alias 总数为 \(Q\)。局部结构关系和写回的主要代价为 \(O(N^2d_r)\) 的关系计算以及 \(O(N^3)\) 的有界线性求解；这些计算在每个局部窗口内执行，窗口数由冻结视野协议限制。QuerySlot 的槽构建和局部路由为 \(O(MKd_r+NKd_r)\)，避免了对整图语义 token 做无界的全局读出。文本分数计算为 \(O(NQ)\) 的实现相关形式，Alias 文本向量在图像之间复用缓存。

Geov2 的计算预算由固定视野协议约束：局部最多四次 \(512\times512\) 编码，宽视野使用一次整图缩放编码，细粒度额外 forward 为零。低显存实现还缓存静态空间项、复用关系计算，并以空间 slab 分块完成密集恢复和最终校准；这些是执行优化，不改变模型方程或预测目标。

## 3.7 训练与参数使用边界

Geov2 是 training-free 推理方法：视觉 backbone、文本编码器、head、QuerySlot 聚类和结构求解均不进行梯度更新。文本 Alias、模板、背景规则以及 \(g_0\) 的任务配置可以在开发阶段冻结，但不能把带标签开发选择写成完全无标签的模型选择。正式实验应分别报告：

1. 统一配置下的跨数据集结果；
2. 使用开发数据冻结后的任务适配结果；
3. QuerySlot、DonorBefore、结构写回、Alias 控制和等权融合的同观测消融。

这样可以把性能来源分解为三种因素：宽视野观测本身、局部条件化读取，以及结构约束的差分写回，而不会把 Alias 输入接口或执行优化误写成新的视觉语义学习模块。
