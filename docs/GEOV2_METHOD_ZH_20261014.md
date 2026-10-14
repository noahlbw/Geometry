# Geov2：See Wide, Write Local

> 结构锚定的跨视野开放词表分割
>
> 本页按论文的 Introduction → Related Work → Method → Experiments → References → Appendix 组织，同时给出一条可以逐行实现和复现的推理流程。默认完整模型指：**冻结 Alias 词表 + 局部 DonorBefore + 宽视野 QuerySlot + 固定结构写回**。当前统一主配置使用 `g0=0.5`；逐数据集扫描得到的最佳步长只作为开发分析，不作为无标签推理时的在线选择器。

## 0. 先固定名词和模型边界

Geov2 的输入是一张 RGB 图像和一个类别词表，输出是原图大小的像素级类别 ID。模型不更新 DINOv3、DINO.text、视觉 head 或文本编码器的权重；推理阶段不读取目标标签。词表、温度、背景规则和 `g0` 是离线冻结的部署配置，曾经可以用开发集选择，因此“training-free”描述的是推理过程不训练，而不是声称配置从未看过标签。

下面几条名称必须分开：

| 名称 | 在 Geov2 中的含义 |
| --- | --- |
| **Geov2 Full** | 本页的完整模型：局部 DonorBefore、宽视野 QuerySlot、Alias 读出和结构写回全部开启。 |
| **QuerySlot** | 宽视野的压缩记忆和局部 query 路由。它不是官方 VIP，也不等于最终预测。 |
| **DonorBefore** | 局部冻结视觉 head 内部的执行顺序：把 patch 语义增量沿视觉关系运输后再进入当前位置的 MLP。 |
| **结构写回** | 在类别分数层用 $H$ 过滤 $W-L$，然后按 $g_0$ 写回局部分数。 |
| **官方 VIP** | 独立比较基线。它不是 Geov2 的宽视野分支，也不是 `LegacyDonorVIPWide`。 |
| **Raw32/Raw24** | 宽视野在坐标网格上直接计分再采样的控制，改变了计分顺序和槽容量，不能等同于去掉 QuerySlot 后的 Geov2。 |
| **EqualLogitFusion** | 在同一对 $L,W$ 上做普通等权 logit 融合，用来检验结构写回是否提供了额外作用。 |

`compact` 和 `efficient` 是执行后端选项，不改变上述数学模型。`efficient` 缓存固定空间项、分块处理 Alias，并在文本读出后释放文本塔；它不等于另一个模型。

## 1. 论文叙事流程

### 1.1 Introduction 的五段逻辑

1. **背景。** 冻结视觉语言模型使开放词表稠密预测可以迁移到遥感、无人机和高分辨率场景；这些场景的像素标注昂贵，training-free 推理具有实际价值。
2. **相关工作缺口。** 改注意力的方法主要处理空间偏置，扩展上下文的方法增加编码或分割成本，传播后处理通常只在一个视野内平滑。它们共同默认单一视野的预测可以直接相信，没有明确处理“两个视野冲突时，谁在什么位置起作用”。
3. **Motivation。** 局部视野保留小目标、边界和纹理，但缺少整图语义；宽视野理解场景和长程关系，却会牺牲小结构。两种证据分别承担“在哪里”和“是什么”，不能用无条件平均解决。
4. **方法概览。** Geov2 让局部 raw token 查询压缩的宽视野语义槽；同一局部视觉关系在 DonorBefore 中组织局部读取，并在类别分数层构造谱滤波器 $H$，只把宽视野残差中与局部结构一致的方向写回。
5. **贡献和边界。** 贡献是双视野的结构组织和可解释的闭式写回流程；不把 DINO、官方 VIP 的观测器、已有 Alias 词表或经典二次解冒充为新发明。实验需同时报告提升、失败域、延迟、显存和消融。

### 1.2 Related Work 的三类组织

- **注意力和空间校正：** SCLIP、NACLIP、ProxyCLIP、VIP self-correction 等，重点是单视野的空间偏置。
- **上下文扩展：** Trident、OV-Stitcher、CorrCLIP 等，重点是增加上下文或外部分割先验，通常带来额外编码或模型成本。
- **预测传播和后处理：** LPOSS、TLP 等，在已有预测上进行平滑或传播。

Geov2 的位置是跨视野耦合：宽视野先作为可查询的观测，局部结构再决定哪些差分可以写回。它不是把传播算子本身宣称为全新的线性代数定理。

### 1.3 Method 应按模块和步骤写

Method 先定义输入和冻结资产，再依次介绍局部视野、宽视野、Alias 读出、结构关系、写回和输出恢复。每个模块都回答三个问题：输入是什么、输出是什么、它如何被下一模块使用。完整公式见第 3 节。

### 1.4 Experiments 应回答研究问题

| 研究问题 | 实验 |
| --- | --- |
| RQ1：整体是否有效？ | 与官方 VIP、CorrCLIP、PEARL、GLA-CLIP、PTC、SegEarth-OV、LPOSS 等基线比较。 |
| RQ2：两路是否互补？ | LocalOnly、WideOnly、EqualLogitFusion、结构写回和互补可视化。 |
| RQ3：各机制是否必要？ | NoQueryRouting、ShuffledQuery、NoDonorBefore、Raw24/32、DiagonalH、UniformBlend。 |
| RQ4：语言接口怎样影响结果？ | Retained、Canonical、CountNormalized 和未筛 Alias 的同观测对照。 |
| RQ5：性能代价是什么？ | A800 batch=1 的同步延迟、峰值 allocated/reserved 显存和精度—延迟 Pareto 图。 |
| RQ6：是否可迁移、何时失败？ | 遥感/自然图像分层、物体尺寸与边界分桶、失败类别及步长敏感性。 |

## 2. 总体数据流

```mermaid
flowchart TD
    I[RGB 图像 I] --> IL[局部输入：长边 896]
    I --> IW[宽视野输入：遥感长边 448；自然 short 336 / cap 672]
    IL --> BL[冻结 DINOv3 + 局部视觉 head]
    IW --> BW[冻结 DINOv3 + 宽视野 head]
    BL --> Q[局部 raw patch q_i]
    Q --> A[视觉关系 A / 有效关系 G]
    A --> DB[DonorBefore：先运输增量，再过 MLP]
    DB --> FL[局部特征 F_L]
    BW --> R[宽视野 raw patch r_j 和 value v_j]
    R --> S[24 个语义槽：球面 Lloyd + soft assignment]
    Q --> QR[局部 query 路由到槽]
    S --> QR
    R --> B[宽视野坐标特征采样]
    QR --> FW[QuerySlot 特征]
    B --> FW
    FL --> L[Alias 读出 L]
    FW --> W[Alias 读出 W]
    G --> H[H=(I+GᵀG)^−1GᵀG]
    L --> D[差分 W−L]
    W --> D
    H --> Z[Z=L+g0 H(W−L)]
    D --> Z
    Z --> O[softmax、Hann 拼接、恢复原尺寸、背景校准、argmax]
```

整条链路只在冻结模型上做前向计算。宽视野只编码一次；局部窗口按固定策略编码；每个窗口最终只做一次结构写回求解。

## 3. Geov2 完整推理流程

### Step 1：读取输入和冻结配置

读取 RGB 图像 $I\in\mathbb R^{3\times H_0\times W_0}$、类别名、每类 Alias 集合、prompt 模板和数据集配置。配置还固定局部/宽视野尺寸、温度、背景规则、`g0`、上游 commit 和 checkpoint 身份。

Alias 文本在推理前编码并缓存；缓存身份包含词表、模板、checkpoint 和上游版本。在线阶段只做图像特征和缓存文本特征的相似度，不重新搜索 Alias，不用标签决定词或步长。

### Step 2：构造局部视野

把图像按比例缩放到最长边 896，再以 512 像素窗口和 128 像素重叠取局部窗口，最多四个窗口。每个窗口经过冻结 DINOv3 和视觉 head，得到 $32\times32$ 个 patch token。记第 $i$ 个有效 patch 的 raw 特征和归一化坐标为 $q_i$、$p_i$。

局部视野的职责是保留边界、小目标和纹理。它不负责独立决定全部全局语义；这些信息由宽视野提出，再由结构写回筛选。

### Step 3：从局部 raw token 构造视觉关系 $A$

先对 raw token 做 L2 归一化，定义 patch $i$ 对 donor $j$ 的关系 logit：

$$
\ell_{ij}=\frac{\langle \hat q_i,\hat q_j\rangle}{T_A}
-\frac{\|p_i-p_j\|_2^2}{2\sigma_A^2},
\qquad T_A=0.1,\;\sigma_A=0.25.
$$

对有效 donor 做行 softmax：

$$A_{ij}=\operatorname{softmax}_j(\ell_{ij}).$$

边缘窗口中的 padding 不作为 donor；把有效项重新归一化得到 $G$。$A$ 表示“哪些局部位置在视觉上可以互相借证据”，不是类别概率，也不直接包含文本。

### Step 4：执行局部 DonorBefore

冻结视觉 head 的原生注意力产生 value 和注意力权重。对 patch token，保留原生特殊 token/prefix 的贡献和 patch 总质量，得到当前位置的 self-conditional patch 增量 $u_i$。然后在 MLP 之前沿 $A$ 运输：

$$
\tilde u_i=\sum_jA_{ij}u_j,\qquad
x'_i=x_i+\tilde u_i,\qquad
x_i^{\mathrm{next}}=x'_i+f(x'_i),
$$

其中 $f$ 是冻结 head 的第二个残差分支（LayerNorm、MLP、LayerScale 和残差仍使用原模型）。prefix token 走原生路径，不做空间运输；两层语义 block 都按该顺序执行。

“Before”指运输发生在非线性 MLP 之前。由于一般有 $f(Au)\ne Af(u)$，先运输 evidence 再让当前位置的 MLP 解释它，与在最终类别图上做平滑不同。DonorBefore 没有新参数，也没有训练。

### Step 5：构造宽视野语义记忆

宽视野从原始图像独立生成 $I^W$。遥感配置通常将长边限制为 448；自然图像使用 short 336、长边上限 672 的配置。宽视野只编码一次，用于提供跨窗口上下文。

设宽视野 raw patch 为 $r_j$、冻结 head 的 value 为 $v_j$，归一化后取 $K=\min(24,N_W)$ 个槽。

#### 5.1 五轮球面 Lloyd 压缩

使用确定性的 farthest-point 初始化中心 $c_k$，再执行五轮球面 Lloyd：

$$
a_j=\arg\max_k\langle \hat r_j,c_k\rangle,
$$

$$
c_k\leftarrow \operatorname{norm}\left(\sum_{j:a_j=k}\hat r_j\right).
$$

空槽保留旧中心。最终用温度 $T_s=0.1$ 的 soft assignment：

$$
P_{jk}=\frac{\exp(\langle\hat r_j,c_k\rangle/T_s)}
{\sum_{k'}\exp(\langle\hat r_j,c_{k'}\rangle/T_s)}.
$$

槽的语义 value 原型为：

$$
V_k=\operatorname{norm}\left(
\frac{\sum_jP_{jk}v_j}{\sum_jP_{jk}}
\right).
$$

#### 5.2 为什么叫“隐式关系”

槽分配隐含了 patch-to-patch 关系：

$$
M=P\operatorname{diag}(P^\top\mathbf 1)^{-1}P^\top.
$$

实现不构造 $N_W\times N_W$ 的 $M$，而是先汇聚 $V_k$ 再用 $P$ 分发，计算量从 $O(N_W^2D)$ 降为 $O(N_WKD)$，其中 $K=24\ll N_W$。因此 QuerySlot 保留整图的主要语义模式，同时避免完整宽视野关系矩阵的显存开销。

### Step 6：用局部 query 路由宽视野槽

对每个局部 raw query $q_i$，用同一组槽 raw 中心计算路由：

$$
\pi_{ik}=\operatorname{softmax}_k
\left(\frac{\langle \hat q_i,c_k\rangle}{T_q}\right),
\qquad T_q=0.1,
$$

$$
r_i=\operatorname{norm}\left(\sum_k\pi_{ik}V_k\right).
$$

同时把宽视野语义特征网格双线性采样到局部位置，记为 $b_i$。当前正式配置以固定 $β=0.5$ 融合坐标证据和 query 选出的槽值：

$$
F_i^W=\operatorname{norm}\left((1-\beta)b_i+\beta r_i\right),
\qquad\beta=0.5.
$$

因此宽视野不是所有局部位置共享同一个 donor：不同局部 query 有不同的 $\pi_i$。需要注意，QuerySlot 路由直接使用 raw query 和槽中心，不是把 Alias 文本送进槽路由。

### Step 7：Alias 文本读出局部分数 $L$ 和宽视野分数 $W$

对每个类别 $c$，设 Alias 集合为 $\mathcal A_c$，文本编码器和 prompt 模板产生归一化文本向量 $e_{c\ell}$。Alias 只负责把视觉特征变成类别证据，不参与 $A$、$G$ 或 $H$ 的构造。

#### 7.1 局部端

局部相似度为 $s^L_{ic\ell}=\langle F_i^L,e_{c\ell}\rangle$，使用数量归一化的 log-mean-exp：

$$
L_{ic}=\kappa_L T_L\left[
\log\sum_{\ell=1}^{m_c}\exp(s^L_{ic\ell}/T_L)-\log m_c
\right].
$$

$-\log m_c$ 防止仅仅增加同义词数量就获得固定 logit 偏置；log-mean-exp 仍然允许少数高质量 Alias 提供强证据。$\kappa_L$ 是冻结配置中的局部分数校准。

#### 7.2 宽视野端

先把同一 Alias 在多个模板下的文本向量平均并归一化，再对类别内 Alias 使用 log-sum-exp：

$$
W_{ic}=\frac1\tau\log\sum_{\ell=1}^{m_c}
\exp\left(\tau\langle F_i^W,\bar e_{c\ell}\rangle\right).
$$

宽视野的职责更接近“整图是否存在该类别的强上下文证据”，所以保留高响应 Alias。当前正式 QuerySlot 配置关闭旧 VIP 的图像条件 salience；不能把历史 VIP salience 写成 Geov2 的模块。

#### 7.3 Alias 扩展、筛选和推理边界

Alias 扩展用于覆盖同一类别的外观、材质、视角和数据集用语；历史仓库中称为 Kev 的离线开发筛词流程用于从候选池保留一组冻结词。它不是在线动态选择，也不是本方法的视觉耦合机制。若使用“JEV”称呼该流程，应在论文中统一为仓库实际名称并给出词表版本。

主实验必须说明：词表是带标签开发选择后冻结的；推理时不读取标签；Canonical、Unscreened 和 CountNormalized 是同观测语言控制。由于宽端 LSE 没有减去 $\log m_c$，Alias 数量会改变类别偏置并影响最佳 $g_0$，所以不能把步长差异单独归因于场景尺度。

### Step 8：结构锚定跨视野写回

对每个局部窗口和类别，$L$ 是细节端分数，$W$ 是 QuerySlot 条件化的宽视野分数。用有效关系 $G$ 定义目标：

$$
Z^*=\arg\min_Z
\frac12\lVert Z-L\rVert_F^2+
\frac12\lVert G(Z-W)\rVert_F^2.
$$

其闭式解是：

$$
K=G^\top G,\qquad
H=(I+K)^{-1}K,
$$

$$
Z^*=L+H(W-L).
$$

部署时采用冻结步长：

$$
\boxed{Z=L+g_0H(W-L)}.
$$

实际实现通过 Cholesky 求解，不显式形成矩阵逆；写回求解使用 FP64，分数和视觉特征遵循各自冻结 dtype。

#### 8.1 $Z^*$ 和最终 $Z$ 的关系

- $Z^*$ 是上述二次目标的精确解，只有 $g_0=1$ 时部署输出才等于它。
- $g_0<1$ 是阻尼：向 $Z^*$ 走一部分，保守吸收宽视野信息。
- $g_0>1$ 是超松弛：沿同一方向超过 $Z^*$，可能放大宽视野区域错误。
- $g_0=0$ 退化为局部端点 $L$；$W=L$ 时任意 $g_0$ 都保持不变。

因此论文中不能把所有使用 `g0=0.5` 的输出都称作“目标最优解”；应称为“沿闭式修正方向的固定阻尼写回”。二次目标在 $g_0=1$ 处最小，但真实 mIoU 的最佳步长可以不同，因为目标不是标签损失。

#### 8.2 谱解释和“写回局部”

若 $K$ 的特征值为 $\rho\ge0$，$H$ 在该方向的增益为：

$$
\frac{\rho}{1+\rho}\in[0,1).
$$

结构弱的方向被抑制，结构支持强的方向更容易写回。这里的“写回局部”不是把整张宽视野类别图复制到局部，而是把 $W-L$ 中与局部视觉关系一致的谱分量加入 $L$。$H$ 可能含负元素，所以它不是非负概率传播，也不保证每个像素都变好。

固定 $H$ 下，Alias 或任一语言接口改变引起的分数变化满足：

$$
\Delta Z=(I-g_0H)\Delta L+g_0H\Delta W.
$$

这说明 Alias 虽然不进入路由，却可以通过 $L/W$ 和最终竞争边界影响输出；同样，某一类别的 Alias 变化可能改变其他类别的 argmax。

### Step 9：窗口恢复和最终输出

对每个窗口的 $Z$ 做类别 softmax，按已有 Hann 概率权重把重叠窗口拼回全图，再恢复到 $(H_0,W_0)$。最后应用数据集冻结的背景 union、bias 和 threshold 规则，取每个像素的 argmax，输出 CPU `uint8` 类别图和可选的分数诊断。

该顺序不能任意交换：QuerySlot 是“先采样/混合特征再计分”，Raw24/Raw32 是“先整图计分再采样”；Alias 的非线性聚合、归一化和背景校准使两者不等价。

## 4. 三层耦合究竟发生在哪里

Geov2 不是把两个最终 mask 做平均，而是在三处发生耦合：

1. **特征读取耦合：** 局部 raw query $q_i$ 选择宽视野槽 $\pi_i$，所以宽分支特征依赖局部位置和局部外观。
2. **局部结构耦合：** 同一局部视觉关系 $A$ 在 DonorBefore 中组织 patch evidence，并生成有效写回关系 $G$。
3. **分数写回耦合：** $H$ 只允许结构支持的 $W-L$ 分量写入局部 $L$。

这解释了为什么 `EqualLogitFusion` 不是完整模型：等权融合只在最后一步组合两个分数，绕过了 query 路由、DonorBefore 和结构滤波。

## 5. 计算量、显存和实现注意事项

- QuerySlot 的宽视野关系使用 $O(N_WKD)$，$K=24$；不构造完整 $N_W^2$ 关系。
- 局部视觉关系和 $H$ 仍与每个 512 窗口的 patch 数有关；它不是线性 attention 的理论替代。
- 宽视野整图只编码一次，文本向量提前缓存；推理完成文本读出后释放文本塔。
- 结构求解是每窗口一次线性求解；不要把 30-arm 或 99 配置扫描的总墙钟时间当作单次延迟。
- A800 成本表使用 batch=1、固定图像、CUDA 同步重复；耗时包含缩放、视觉编码、读出、写回、恢复和 CPU 输出，不含初始化、图像解码和离线文本编码。显存报告是 `max_memory_allocated`，与 `nvidia-smi` 驻留值不是同一口径。
- 当前 Full 的 QuerySlot/DonorBefore 是否实际执行，要检查元数据中的 `wide_branch=query_slot`、局部 `strength=original` 和 `donor_before=true`。默认 CLI 可能仍选择历史 VIP 宽分支，复现时必须显式指定 QuerySlot。

## 6. 实验协议和已有证据

### 6.1 主配置

统一步长扫描覆盖 VDD、Potsdam、UDD5、OEM、Vaihingen、PC60、ADE150、VOC20、VOC21、COCO-Object81；PC60 替换 LandCover.ai。十协议平均的统一 `g0=0.5` 为 **55.8158% mIoU**，相对同观测等权融合高 **3.1730pp**，但只有 7/10 个协议领先，VDD、VOC20、VOC21 仍可能低于融合。逐域最佳是带标签开发参考，不能写成无调参统一上界。

### 6.2 关键部件记录

| 协议 | Geov2 Full | 去掉 QuerySlot | 去掉 DonorBefore | 等权融合 | Full 成本 |
| --- | ---: | ---: | ---: | ---: | --- |
| UDD5 | 55.2574 | 55.5846 | 54.7910 | 42.5775 | 288.20 ms / 2276.79 MiB |
| VOC21 | 72.1656 | 71.0325 | 71.9698 | 74.8935 | 142.47 ms / 2247.24 MiB |
| COCO-Object81 | 49.6209 | 49.4760 | 49.6042 | 42.1355 | 153.40 ms / 2722.19 MiB |

这些数字来自同观测、同词表和固定 `g0=0.5` 的已查看验证集。它们支持 QuerySlot 和 DonorBefore 在部分协议的条件收益，但不支持“每个数据集每个部件都必然提升”。例如 UDD5 关闭路由反而略高；VOC21 的等权融合高于固定 0.5 写回。论文应同时披露这些失败和区间。

UDD5 的结构写回相对等权融合高 **12.6799pp**；VOC21 的 Full 相对关闭路由高 **1.1331pp**，配对 95% 区间为 **[+0.8550,+1.4184]**；COCO-Object81 的 Full 相对等权融合高 **7.4853pp**。这些差值是对应协议内的机制证据，不应横向拼成单一普遍结论。

### 6.3 消融顺序

先固定完整模型的 Alias、窗口、温度和 `g0`，再做同观测消融：

1. `LocalOnly` 与 `WideOnly`：观察两种视野各自提供什么。
2. `EqualLogitFusion`：检验朴素融合是否污染细节。
3. `NoQueryRouting` 与 `ShuffledQuery`：区分“加入路由”和“正确空间对应”。
4. `NoDonorBefore`：检验局部 head 内部的执行顺序。
5. `Raw24/Raw32`：检验槽容量和坐标计分顺序，避免把它们误写成纯路由控制。
6. `UniformBlend`、`DiagonalH`、`SpatialH`：检验谱结构和跨位置关系。
7. `Canonical`、`Unscreened`、`WideCountNormalized`：隔离语言词数、扩词和显式类别偏置。

删除差值是条件贡献，不能直接相加。允许每个对照重新选择最佳 `g0` 后，若结论变化，应如实报告，不能只保留对完整模型有利的固定步长表。

### 6.4 可视化和机制分析

- **互补性图：** RGB、真值、LocalOnly、WideOnly、Geov2、`W-L`、`H(W-L)`，按预设规则选图，不挑最漂亮的例子。
- **位置对应图：** QuerySlot 正确 query 与循环移位 query；展示路由是否依赖空间对应。
- **尺度/边界分桶：** 按物体面积、离边界距离和输入分辨率报告局部/宽视野/Geov2 的差异。
- **效率图：** mIoU—延迟 Pareto 图，官方 VIP 和 Geov2 使用相同计时边界。
- **失败图：** 宽视野区域错误沿 $H$ 写回、步长过小导致局部误激活残留，分别对应目标函数可预言的两种失败。

## 7. 论文中应如何准确表述创新

可以写：

> Geov2 uses local raw queries to read a compressed wide-context memory and reuses the local visual relation to constrain the residual written back from the wide observation. The resulting pipeline couples two views at feature readout and score correction while preserving a frozen, training-free deployment path.

中文可写为：

> Geov2 让局部 raw token 查询压缩的宽视野语义记忆，并复用局部视觉关系约束宽视野残差向局部分数的写回。这样两个视野在特征读取和分数修正两处耦合，同时保持冻结模型、无训练和每窗口一次线性求解的部署形式。

不要写成：

- “QuerySlot 就是 VIP”；
- “Alias 在路由中参与了视觉槽选择”；
- “$Z^*$ 在所有 `g0` 下都是输出”；
- “所有数据集都超过等权融合或官方 VIP”；
- “每个部件都在每个数据集上稳定提升”；
- “结构写回保证边界和细节绝不改变”；
- “training-free 等于词表、步长从未用开发标签选择”。

## 8. 复现清单

复现前检查以下项目：

1. 使用仓库中对应日期的 YAML、Alias manifest、prompt 模板和上游 commit。
2. 显式指定 `--wide-branch query_slot --query-slot-capacity 24 --efficient`。
3. 检查 `local_strength=original`，否则 DonorBefore 可能被绕过。
4. 固定 `beta=0.5`、槽温度和 query 温度，除非这是预声明的消融。
5. 使用缓存文本向量或记录重新编码的缓存身份；不要把首次文本编码时间混入在线延迟。
6. 对全量 split 合并混淆矩阵后计算 mIoU，不平均逐图 mIoU。
7. 记录 allocated 和 reserved 两种显存，说明它们不是 `nvidia-smi` 驻留值。
8. 保留逐图混淆矩阵、配置、词表和 cost 原始 JSON，避免只提交一张汇总表。

主要入口：

- [QuerySlot＋DonorBefore 模型与复现](QUERY_SLOT_DONOR_BEFORE_20261013.md)
- [固定结构写回说明](GEOV2_FIXED_WRITER_20261014.md)
- [十协议 g0 扫描](../research/geov2_g0_sweep_20261014/REPORT_ZH.md)
- [UDD5 部件消融](../research/geov2_udd5_fixed05_ablation_20261014/REPORT_ZH.md)
- [VOC21 部件消融](../research/geov2_voc21_fixed05_ablation_20261014/REPORT_ZH.md)
- [COCO-Object81 部件消融](../research/geov2_coco81_bestg_ablation_20261014/REPORT_ZH.md)
- [实验与源码索引](EXPERIMENT_INDEX.md)

## 9. References

仓库内的实现、配置和统计是本文的主要可复现依据；论文相关工作可引用下列公开来源：

1. VIP：<https://arxiv.org/html/2605.12325v2>
2. LPOSS：<https://arxiv.org/abs/2503.19777>
3. Trident：<https://arxiv.org/pdf/2411.09219>
4. OV-Stitcher：<https://arxiv.org/abs/2604.08110>
5. Standalone DINOv3 for RS OVSS：<https://arxiv.org/html/2608.03023v1>
6. DINO Soars：<https://arxiv.org/abs/2605.03175>

## Appendix A：最小伪代码

```text
given image I, classes C, frozen alias bank E, config P

IL = resize_long_side(I, 896)
for local window x in windows(IL, size=512, overlap=128):
    q, native_head = frozen_dino_local(x)
    A = row_softmax(cosine(q, q) / TA - spatial_distance(q) / (2*sigmaA^2))
    FL = donor_before(native_head, A)       # transport before frozen MLP
    save local raw q, FL, valid mask, G

IW = resize_wide(I, P.wide_size)
r, v, FW_grid = frozen_dino_wide(IW)
centers = spherical_lloyd(r, K=24, rounds=5)
Pslot = softmax(cosine(r, centers) / 0.1)
V = normalized_weighted_values(Pslot, v)

for each local position i:
    pi = softmax(cosine(q_i, centers) / 0.1)
    routed = normalize(pi @ V)
    sampled = bilinear_sample(FW_grid, position_i)
    FW_i = normalize(0.5 * sampled + 0.5 * routed)

L = local_alias_readout(FL, E)             # local LME, count-normalized
W = wide_alias_readout(FW, E)              # template mean + wide LSE
H = solve_cholesky(I + G.T @ G, G.T @ G)
Z = L + g0 * H @ (W - L)                   # g0=0.5 in unified main config

prob = softmax(Z + frozen_background_rules(P))
output = hann_stitch_and_resize(prob, original_size=(H0, W0))
return argmax(output)
```

## Appendix B：读者最容易混淆的两条公式

$$
Z^*=L+H(W-L)
$$

是二次目标在 $g_0=1$ 时的闭式解；

$$
Z=L+g_0H(W-L)
$$

是部署输出。Geov2 当前使用的是第二条，默认统一 `g0=0.5`，所以输出是沿结构修正方向的阻尼写回，而不是把上面的 $Z^*$ 直接当作每个数据集的最终预测。

