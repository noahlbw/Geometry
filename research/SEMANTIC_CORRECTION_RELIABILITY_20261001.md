# 怎样实现可靠的语义矫正：核验目标审计与实施边界

日期：2026-10-01。对应目标：找到如何实现可靠的语义矫正。

本轮只分析已保存输出，不加载图像、不运行GPU推理、不生成新词表，也不恢复Potsdam失败后停止的实验。新增离线审计脚本和六项标准库单测；没有把一个未经验证的候选称为最终模型。

## 1. 新证据：旧代理预测的不是实际纠错

核对 `diagnose_gear_alias_marginals.py` 与 `select_gear_disagreement_aliases.py` 后，方向必须澄清：

- 保存的 `mean_logprob_gain` 是保留alias相对删除alias，给真实标签带来的平均log-probability增益。
- 保存的 `flip_balance` 是保留alias保护的正确像素数，减去保留alias造成的错误像素数。不是删除收益；与全量LOO文件的beneficial/harmful定义方向不同。
- `native_counterfactual_gain` 用去除被审alias的native分布给上述概率变化评分。历史low15规则保留这个值**最小**的15个词，实际依赖的是与native的互补/冲突，不是信任native高分。
- 新审计沿用历史排序方向，使用其负值作为保留排名。没有因结果而翻转某些类别的方向，也没有拟合阈值。

| 诊断集 | aliases | 对真实标签概率增益的AUC | 对正确像素净增益的AUC | 后者类内配对AUC |
|---|---:|---:|---:|---:|
| UDD5 | 100 | 0.8185 | 0.5841 | 0.5259 |
| OEM | 160 | 0.8581 | 0.7356 | 0.6885 |
| LoveDA D | 140 | 0.7035 | 0.4636 | 0.5145 |
| FLAIR-1 | 240 | 0.6684 | 0.6258 | 0.6408 |
| LandCover.ai | 100 | 0.8352 | 0.6277 | 0.5805 |

类内AUC只比较同类正负alias，按有效正负对数加权，不把类间score偏置当成排名能力。净增益为零的alias在该目标的AUC中排除；OEM/FLAIR/LandCover.ai分别有4/3/2个这样的alias。

UDD5 road的两种AUC分别为 **0.8788与0.1042**；other分别为 **0.1900与0.0300**。LoveDA tree为 **0.3200与0.1300**，barren为 **0.0101与0.0100**。这说明一个跨alias的较高总体AUC可以掩盖特定竞争类别的反向排名。不能据此选择各类不同的正负号，那会变成用这些诊断标签调选择器。

例如UDD5 `rural road` 的真实标签概率贡献为正，但保留它的正确像素净贡献为-400；`road junction` 为-341。UDD5有15个alias出现“真实标签概率增益为正，但正确像素净增益为负”，OEM21个、LoveDA17个、FLAIR46个、LandCover.ai13个。

这些是旧固定诊断集合的**局部Geometry patch统计**：UDD5覆盖40图/80工作区，其他四集各64图，LoveDA/OEM126工作区，FLAIR/LandCover.ai64工作区。它们不是完整滑窗mIoU，不等于最新SupportConditioned核验器的逐位置准确率，也不能单独解释全部全量退化。它们直接否定的是“既然旧代理的概率AUC较高，就能可靠判断纠错方向”这一推断。

复现脚本：[audit_semantic_proxy_reliability.py](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/audit_semantic_proxy_reliability.py)；完整输出：[离线审计JSON](C:/Users/AH/Documents/ChatGPT/OVSS/research/semantic_proxy_reliability_audit_20261001.json)。六项单测覆盖符号、并列AUC、无正负对、零净变化、类内配对及原计数约定，均通过。未安装依赖。

## 2. 为什么连真实标签概率增益也不能保证纠错

两个像素都属于A。原输出的A概率为0.55与0.10，候选为0.49与0.20：

- 第一个像素从正确A变成错误B。
- 第二个仍错误，但真实A的概率翻倍。
- 平均真实标签log-probability增益为 `(log(0.49/0.55)+log(0.20/0.10))/2 > 0`。
- 正确像素数却从1降至0。

所以存在三层不同目标：响应/概率拟合、离散预测正确性、宏平均IoU。不能用前一个目标的代理成功代替后一个。推理energy下降、教师KL改善、alias可靠性排名都处于这个问题之内。

Potsdam最新SupportConditioned是真实支持裁剪、固定20词分母、保留原Multiscale的完整实现，仍39.9311<40.4433；car预测面积扩大而precision下降。它表明纠错方向错误并非仅由删词分母或context接口造成。

## 3. 必须重新定义核验对象

旧问题：alias a全局是否可信？

正确问题：在位置i，原预测为A，候选拟改为B，这一次改变能否减少错误，而不会只是把错误搬到别的类别？

给定可用观测E，若真正的类别后验为q，则一次A→B对像素准确率的条件期望增益**精确**为：

`E[delta correct | E] = q(B | E) - q(A | E)`。

truth=B是beneficial，truth=A是harmful，其余类别是wrong-to-wrong。三种必须分开统计。这个公式指出需要什么信息，不声称现有native/global softmax就是q。

若要宏平均IoU，还需考虑被影响类的分母。记T_c=TP、D_c=GT_count+pred_count-TP，则小批量A→B的一阶单像素变化为：

`delta IoU_A ≈ [-q_A * D_A + T_A * (1-q_A)] / D_A^2`

`delta IoU_B ≈ [ q_B * D_B - T_B * (1-q_B)] / D_B^2`。

整体mIoU取两者之和除以类别数。它解释为什么相同正确像素净收益也可能产生不同mIoU。**q与GT_count在无标签目标域不可直接取得**；不能把这个公式直接变成带GT的推理loss。单次更新的精确IoU也不是上述一阶式，批量变化及独立场景统计仍需评测。

## 4. 当前信号到底能证明什么

| 信号 | 能提供的信息 | 不能直接证明的内容 |
|---|---|---|
| raw DINO Geometry关系 | 视觉相似、邻近、可定位的读取支持 | A/B真实语义归属 |
| Geometry alias margin | 当前语义读出的竞争假设 | 校正该读出所需的真值 |
| tight/wide native global一致 | 同源编码器对两个裁剪的整体偏好 | 小对象像素的类别；一致错误也会通过 |
| 留词族 | 降低直接用同词提出并验证假设的泄漏 | 消除相邻词和同模型的系统偏差 |
| 旧native反事实代理 | 与native冲突的alias排序，在部分概率目标有信息 | 对实际改对方向的统一跨类可靠性 |
| VIP宽视野dense读出 | 已有匹配实验支持的互补语义观测 | 每次Geometry/VIP分歧谁正确；其置信度也未校准 |

“视觉关系可靠”与“语义核验可靠”必须分工。信息源来自同一冻结编码器不等于完全没有互补，但不能当统计独立的证据投票，也不能把tanh margin、熵、1.96倍离散度直接称为置信区间。

## 5. 可实现的研究路线：完整候选只有两个逻辑部分

保留Geometry/Multiscale作为结构化观测；第二部分只实现**位置条件的竞争核验与有界重读**。不再额外串接全局筛词、TextGraph、重构energy、平滑器。以下为未验证的实现约束，不是已成功的新终版。

### 5.1 Proposal与核验数据流

1. 先得到原Multiscale以及一个明确来源的语义proposal。记录真的会改变预测的位置和原A/拟改B；不为每类强制制造prototype或强制删除词。
2. Geometry将这些分歧位置组织为细粒度软支持，并约束读取路径。一个支持可含多类，不能将support分区当成对象mask。
3. 核验器针对**同一支持、同一个A/B对**取得语义观测；优先保留稠密native或已有宽视野dense信息，不再广播整个裁剪的global父类胜负。
4. 分别测量支持内和周边对照上的A/B语义差异，判断B证据究竟来自目标支持，还是背景/上下文。对照的选取必须不读标签；相关词族和模板作为敏感性检查，不算独立投票。
5. 同一核验结果控制实际支持内proposal的写回。未知保留原输出，不使用“alias没证据”授权删除竞争类。

可待检验的局部对照量，例如：

`E_v(r; A,B) = mean_(i in r)[s_v(i,B)-s_v(i,A)]`

`D_v(r; A,B) = E_v(r; A,B) - E_v(background_control; A,B)`。

E回答绝对竞争，D回答相对背景的定位。**D正不等于B是真类**；不能只用D，否则背景中的真水体/连续植被可能因缺少对照被抑制。也不能直接将E或D经sigmoid当成q。该量与旧COR的“高置信伪类别anchor上词族响应均值比较”不同，但区别本身不保证更好。

此设计新增的假设是“目标支持的竞争证据可从环境偏好中分离”，而不是“再做两次crop就更可靠”。它需要位置级beneficial/harmful审计，不是再比较一个全局alias AUC。

### 5.2 为什么不能直接宣布闭环可靠

正确的写回接口可以保证：零证据精确回退、支持外不改、原slot分母不变、没有donor不注入、明确记录每个A/B变化。它不能保证q正确。原本正确的类别若被核验器错误否定，反对称A/B更新或固定分母仍会伤害预测。

如果后验只知道在某个集合Q内，保守准则可以写成：

`min_(q in Q) [q_B - q_A] > 0`。

但若Q只允许“同源视图给出的分布”，不保证覆盖真实后验，这不是可靠性证书；若Q允许任意类别分布，其最小值为-1，没有非平凡纠错获准。二者之间缺的就是**可检验的语义约束或校准来源**，不能靠缩小Q并命名为confidence解决。

## 6. 两种条件下分别能做到什么

### 严格training-free、只保留现有冻结模型

可继续检验位置对照语义和已有dense互补性；不更新网络权重，不从目标标签拟合router。必须在真实A→B分歧上验证方向、覆盖和跨类稳定性，然后评测完整候选。不能承诺任意域可靠或全域提升；如果这些观测仍无法区分beneficial/harmful，仅换门控或图求解没有依据。

VIP宽视野可以作为已验证的互补来源，但需要明确归属。若耦合机制不胜同输入预算的简单融合，不能把它包装成方法创新。当前停止条件仍有效，本轮没有执行这一路线。

### 允许独立源域校准或新的语义模型

源域带标签的纠错风险估计可以把核验目标直接对准A→B的beneficial/harmful/wrong-to-wrong，而不拟合一般分类置信度；测试需类别留出、场景留出和跨域校准检查，目标标签只评测。即使冻结Geometry，只要训练这个核验器，完整系统就**不再是严格training-free**。它是一个需要用户接受的任务设定变化，本轮不实施。

额外冻结语义模型也不自动可靠：需验证它在小对象、遥感视角与目标ontology上的定位和A/B判别；不能因模型更大就默认是真值。模型选择、资源及公平比较同样需事先明确。

## 7. 本轮结论与下一步门槛

已取得的新证据是：旧筛词代理的高概率AUC在多个域不对应实际正确像素净收益，UDD5 road甚至强烈反向。因此旧反事实proxy不应直接作为终版纠错器或动态删词的依据。

建议保留Geometry，不继续全局alias硬删；将核验单位改为“位置上的真实A→B变化”，优先验证背景对照后的稠密语义判别是否有信息。这个方向尚未验证，不宣称比旧方案一定有效。

任何后续候选获准恢复后，必须记录：

- beneficial/harmful/wrong-to-wrong，按原A、拟改B和独立场景分组，而非只统计平均trust。
- 在同一候选变化集合上的方向判别和覆盖；拒绝全部没有意义，高精度但几乎不覆盖也不能当最终提升。
- 完整候选的mIoU、TP/FP/FN、car/vehicle precision及water/roof recall，不能用log-probability或准确率代替。
- 固定同一规则向已开发域外迁移，与Multiscale、同预算简单融合比较；禁止按有标签审计挑类别阈值和词表。

**目标保持未完成：本轮确定了一个此前被代理指标掩盖的具体瓶颈，以及可执行的纠错核验标准；尚未获得经验证可靠的语义纠错器。**

## 8. 后续离线证据：保守正确性路由也不是IoU校正

继续分析已经完成的Geometry/VIP完整输出，没有重新运行图像推理或选择词表。已生成 [纠错运行点与IoU精确分解JSON](C:/Users/AH/Documents/ChatGPT/OVSS/research/semantic_correction_operating_point_20261001.json)，可由 [审计脚本](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/audit_semantic_correction_operating_point.py)复现。

### 8.1 必须检查的准确率运行点

对固定proposal，B为可改对像素、H为会改错像素，核验器有益保留率为r、错误接受率为f，则正确像素净变化为：

`delta correct = B*r - H*f`。

已有Potsdam Multiscale→VIP20：B=50,253,983、H=34,602,300。GroundedLeaveFamilyOut保留16,306,907个有益变化、接受7,946,357个有害变化，净正确像素 **+8,360,550**；有益保留率32.44898%，有害拒绝率77.03518%。在该保留率下，正确像素不退化所需的最低拒绝率为52.87334%，所以该路由已超过准确率盈亏线。

但是它的Potsdam mIoU只有40.9863，低于MeanProb50的42.4761、MaxConfidence的42.9481与VIP20的42.3519。因此，不能把“总改对多于改错”作为语义校正目标的充分完成条件，更不能直接把77%拒绝率视为最终成功。

该盈亏线是有标签、固定proposal的事后诊断，不是无标签可部署的阈值，也不是mIoU阈值。wrong-to-wrong不改变像素准确率，但可以改变各类FP与mIoU，不能在IoU分析中当作无效变化丢弃。

### 8.2 TP与FP的精确IoU分解

固定同一评测GT，设某类原TP为T、GT像素数为G、原FP为F，新TP变化为dT、新FP变化为dF。原IoU为I=T/(G+F)，则：

`delta IoU = (dT - I*dF) / (G+F+dF)`。

这是有限计数更新的**精确恒等式**，不是前文的单像素一阶近似；原/新union都非零时成立。将右侧分为TP项与FP项，相加精确复现完整混淆矩阵的IoU变化。

| Potsdam相对Multiscale | 类 | dTP | dFP | TP项/百分点 | FP项/百分点 | IoU净变化/百分点 |
|---|---|---:|---:|---:|---:|---:|
| VIP20 | car | -240,869 | -64,740,458 | -0.7350 | +19.9483 | +19.2133 |
| VIP20 | low vegetation | -11,848,884 | -383,563 | -10.8516 | +0.1435 | -10.7081 |
| MeanProb50 | car | -47,706 | -59,276,629 | -0.1248 | +15.6546 | +15.5299 |
| MeanProb50 | low vegetation | -9,417,989 | -427,644 | -8.6288 | +0.1601 | -8.4687 |
| GroundedLeaveFamilyOut | car | -6,989 | -25,101,747 | -0.0097 | +3.5005 | +3.4908 |
| GroundedLeaveFamilyOut | low vegetation | -3,816,213 | -354,794 | -3.4941 | +0.1327 | -3.3614 |

这明确定位了一个问题：Grounded机制保护了几乎全部car TP，但放弃了大量car FP校正；同时仍未充分保住low vegetation的TP。全局统一“更加保守”不能同时解决这两件事。VDD同样存在两种不同方向：VIP20的vehicle IoU提升主要来自FP下降，water/roof主要来自TP恢复。

上述按真实类分组的计数只用于解释，不能在推理时按GT选择分支。特别是car GT上的有害变化比有益多，并不意味着应拒绝所有与car有关的变化：大量有价值的car FP修正在**非car GT**位置发生。

## 9. 已实现的最小诊断接口与还缺的证据

后续局部核验必须记录`(原预测A, 拟改B, 真实类C)`，而不只记录真实类边际。现已加入 [semantic_correction_audit.py](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/dinotool/semantic_correction_audit.py)，输出`[old_prediction, proposal_prediction, ground_truth]`三维计数，按chunk计算以限制全景图临时内存。

已接入本地Geometry/VIP评测的**预测完成后**审计；原预测逻辑不变、标签不进入模型。合并器检查三维计数能否精确复现Multiscale和VIP20的混淆矩阵，拒绝混合有/无新统计的分片。新合并元数据修正旧`Geometry_only_correct`误名为实际的`Multiscale_only_correct`；旧JSON保留不变。

新增运行点8项、三维计数6项单测，加上原代理审计6项，共20项CPU单测通过；相关评测/合并文件通过语法编译。**没有运行或部署新的远端评测**，不能将这些测试当作语义效果证据。历史汇总不足以恢复三维联合计数，不能从两个混淆矩阵捏造A/B/C路径。

因此下一实现需要把“语义观察”和“竞争决策”明确分开：

1. 语义观察器测量同一支持上的目标证据是否来自对象本身，而非背景借词；这仍是需要新观测验证的候选假设。
2. 决策器根据实际A→B变化评估对两类的TP/FP风险，而不是仅比较global置信度或预测是否一致。
3. Geometry约束同一个核验变量能够控制的读取/写回范围，不作为判定语义真值的教师。

若用模型自身softmax去估计GT类面积或TP/FP，会继承同一偏差；精确恒等式没有消除语义后验不可识别的困难。不能据这些审计GT计算最终门控或类阈值。

本轮推进的是纠错目标与必要观测接口，不是证明一个新核验器已经可靠。Potsdam失败后的停实验条件仍然有效；在获得新的位置条件语义证据之前，不再把类似global/一致性门控作为已经找到的解决办法。

## 10. 后续读出机制核验

已进一步核验ClearCLIP与CLIP Surgery的固定官方源码，并分解此前完整head/text因素实验的TP/FP交互。共同alias分数减法在当前LME下不改变分类；实际特征共享模式、类别相关权重及空间归一化是不同操作，不能混为一种消噪。

固定同词、同评分和视野时，RS→ImageNet使Geometry Potsdam car IoU增加27.5500点，却损失8,246,233个lowveg TP；VDD wall IoU下降20.4145点。它们加强了“全局语义重加权无法自动可靠纠错”的证据，不构成新模型结果。新增13项检查使semantic审计CPU测试总数达到33项。

源码证据、精确交互与未验证的下一读取假设见[语义读取机制核验](C:/Users/AH/Documents/ChatGPT/OVSS/research/SEMANTIC_READOUT_MECHANISMS_20261001.md)。本轮未部署或启动任何GPU实验，目标仍未完成。

## 11. 后续各8样本原Geometry观测

用户另行授权有限读取诊断，现已完成VDD/Potsdam各8个固定样本，原模型输出保持完全不变。阶段与固定A/B分解表明错误扩张和有效证据丢失并非同一方向；MLP、LN beta的大单项会相互抵消，不能直接成为正确性门控。缓存标签只用于事后解释，未拟合参数。

详细结果见[原Geometry读取诊断](C:/Users/AH/Documents/ChatGPT/OVSS/research/GEOMETRY_READOUT_DIAGNOSTIC_8_20261001.md)。此次没有候选A→B校正、没有新的beneficial/harmful验证，也没有恢复八集评测；可靠语义矫正仍是未完成目标。
