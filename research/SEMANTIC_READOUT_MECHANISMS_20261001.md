# 可靠语义矫正：读出机制核验与固定实验交互分解

日期：2026-10-01。继续目标“找到如何实现可靠的语义矫正”，但没有恢复Potsdam失败后停止的GPU实验。这里包括官方源码核验、既有完整结果的离线计数分析与CPU数学测试；不是一个新模型的性能报告。

## 1. 外部证据范围

本轮按`research-lookup`核对两项直接相关的机制。此前Parallel Search因缺少授权失败，记录见[失败响应](C:/Users/AH/Documents/ChatGPT/OVSS/research/semantic_evidence_lookup_20261001.json)。本轮只通过正常证书验证的HTTPS读取已知作者仓库的固定commit，不声称完成全面检索或证明研究空白。没有调用登录、安装依赖或运行外部源码。

### ClearCLIP：改变实际视觉读出，不是只减一个logit

- 作者README明确为ECCV 2024；题名为 *ClearCLIP: Decomposing CLIP Representations for Dense Vision-Language Inference*。[论文入口](https://arxiv.org/abs/2407.12442)
- 固定commit：`ad68a404d55d48d27330b93554eb64a234ff717f`。[源码](https://github.com/mc-lan/ClearCLIP/blob/ad68a404d55d48d27330b93554eb64a234ff717f/open_clip/transformer.py#L500)
- `VisionTransformer.forward`的500–567行：`ignore_residual=True`时输出累加custom attention，而非残差＋attention＋MLP。每层原始`blk(x)`仍推进下一层输入；最后保留norm与投影。
- `custom_attn`的589–622行：ClearCLIP分支为QQ self-self attention，再读取冻结V并经过out projection。
- 作者论点是CLIP的全局对比训练与残差噪声削弱局部判别；上述源码与其操作一致。但这不是DINO.text残差必定有害的证据。

与当前Geometry的关键区别：当前`_intervened_head`在被改层重新计算Q/K/V，并沿修改后的残差＋MLP状态推进下一层；不是ClearCLIP的原始状态推进＋attention-only输出。因此“借鉴ClearCLIP”需要定义真正不同的读取流，不能只把prefix blocking或已有深度消融重新命名。

### CLIP Surgery：共同减法、类别权重与空间归一化必须区分

- 作者README给出早期预印本 *CLIP Surgery for Better Explainability with Enhancement in Open-Vocabulary Tasks*，[arXiv](https://arxiv.org/abs/2304.05653)。最终题名 *A closer look at the explainability of Contrastive language-image pre-training*，Pattern Recognition 162，111409（2025），DOI `10.1016/j.patcog.2025.111409`。
- 固定commit：`d4696d47f49cfe70f49140afe5eb94f94c5f59bc`。[源码](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/clip/clip.py#L287)
- `clip_feature_surgery`的287–308行先以CLS与文本的分数形成类别权重w，再计算逐通道乘积`f_i * t_a * w_a`，减去沿label维的平均特征，最后对通道求和。
- `get_similarity_map`的271–284行对每类响应做空间min-max归一化，再插值；这会改变像素内的类间竞争，不能与共同减法等同。

令原余弦为s、固定权重为w，上述feature surgery的输出可精确写为：

`s'_ia = w_a * s_ia - mean_b(w_b * s_ib)`。

最后一项在同一像素对所有alias相同。对当前同温度normalized LME，`LME(s'_c)=LME(w*s_c)-共同项`。所以：

1. **固定w时，共同项不改变类间argmax或softmax。** 它可以改变可视化、绝对分数阈值等后续行为，不能据此否定原论文的整套方法。
2. **类别相关w可以改变竞争。** 它是场景条件的语义重加权，不是共同分数消噪；同源CLS并没有因此成为可靠像素标签。
3. 若使用同一个redundant text向量r，`f_i·(t_a-r)=f_i·t_a-f_i·r`也只是共同分数减法；若r随类别不同，该不变量不适用。
4. 空间min-max、别的模板表示、阈值或另一attention head产生的变化必须分别归因。

据此排除的只是“在当前LME前对所有alias减共同分数，就能纠正分类”这一具体移植。没有排除实际Value处理，也没有证明它们适用于DINO.text。

## 2. Geometry本身保留了什么

对行归一化关系A，写Values为`V=1*mu+V_detail`，则：

`A*V=1*mu+A*V_detail`。

空间关系变化不消除共享Value模式。这里的mu是特征向量，经文本投影可得到**类别相关**偏置；它与上节对所有alias相同的标量不同。当前Geometry还保留prefix、残差、MLP以及最终归一化，不能将上述attention阶段恒等式直接等同于最终logit分解。

尚未测定共享模式是否有害、是否造成car误报。纯净的大水体或农田也可能形成共享模式，直接减特征均值可能损害它们。**这只是需要进一步辨识的语义读取位置，不是已经找到的校正器。**

## 3. 既有完整实验：不是一种模板能解决所有类

离线脚本[固定head/text交互审计](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/audit_semantic_readout_interactions.py)读取此前完整VDD80/Potsdam504的匹配实验。两head使用同20个alias字符串、同输入视野、VIPScore、无背景阈值；文本从六个RS模板平均特征改为ImageNet模板逐模板相似度平均，因此包含模板家族与组合形式，不是纯措辞。

只使用已经保存的混淆矩阵，校验各arm真实类总数及顺序一致，并沿用精确TP/FP IoU分解。没有选择词、拟合类权重、生成新预测或选择“最佳类别head”。[完整离线JSON](C:/Users/AH/Documents/ChatGPT/OVSS/research/semantic_readout_interactions_20261001.json)

### Potsdam：car变好并不代表读取整体可靠

| RS→ImageNet | car IoU变化/点 | car TP变化 | car FP变化 | low vegetation TP变化 | tree IoU变化/点 |
|---|---:|---:|---:|---:|---:|
| Geometry | +27.5500 | -639,475 | -60,666,554 | -8,246,233 | -1.5024 |
| VIP | +19.4988 | -232,203 | -67,489,831 | -6,854,718 | +4.5618 |

在这个特定匹配协议中，Geometry car精确率11.64%→40.54%，召回98.51%→92.05%；lowveg召回17.79%→10.00%，tree召回30.73%→29.44%。这与历史512/128主模型的绝对值不同，不能混作同一个配置。

VIP在同文本改动下增加5,194,803个tree TP，Geometry反而减少1,110,954个。tree的head/text IoU交互为+6.0642点，car为-8.0512点；总体交互仅+0.3387点。一个总体平均会掩盖非常不同的类间响应。

### VDD：全局语义重加权会造成覆盖塌缩

| RS→ImageNet | wall IoU变化/点 | wall TP变化 | wall FP变化 | vegetation TP变化 | vehicle IoU变化/点 |
|---|---:|---:|---:|---:|---:|
| Geometry | -20.4145 | -16,486,306 | -28,397,994 | -43,492,623 | +15.7400 |
| VIP | +5.0122 | -10,917,238 | -40,075,343 | -27,078,258 | +14.4711 |

Geometry wall精确率42.08%→86.13%，但召回68.20%→15.13%；减少FP并不足以抵消覆盖丢失。wall的head/text IoU交互为+25.4267点，总体交互为+4.2300点。

同样的词、同样的文本改动，经过不同视觉读取会形成不同竞争结果。因此不能把“VIP模板有效”“VIP筛词有效”直接移植成Geometry的可靠性准则，也不能用有标签逐类优胜head拼成模型。

这些是开发集事后描述性交互，没有场景级置信区间；不是每个像素的因果路径，也不是统计独立的域验证。

## 4. 对下一模型的实际约束

本轮证据使以下路径失去依据：

- 只移植CLIP Surgery的共同分数减法，再声称它改善当前类别竞争。
- 只换全局模板或CLS类别权重，并以car提升忽略wall/低植被覆盖损失。
- 继续用更强Geometry传播来推断语义必定更准确。
- 直接用外部CLIP论文替当前DINO.text残差处理提供效果保证。

仍值得检验的是**位置条件的语义成分辨识**：在同一个Geometry支持内，区分对象本身的A/B证据与场景借词、泛化词或残差共享响应；然后在读取内部控制它们，而非对所有类别做共同减法。这个假设尚未验证。

如果之后获准恢复实验，首先缺少的观测应是在真实支持上记录attention输出、prefix贡献、残差/MLP前后及对应A/B margins，并与三维`(旧A,候选B,真实C)`审计对齐。标签只能解释这些路径，不能用来选择测试时系数。若错误偏置主要不在这些流中，必须拒绝这个假设；不能因为它有代数解释就继续包装。

当前没有这些实际特征流数据，本轮不会捏造分解结果，不实施盲目的Value去均值终版。跨域可靠性仍需冻结完整规则和真正的独立验证；局部数学身份或源码相似不能证明mIoU提升。

## 5. 验证与状态

新增6项CPU代数检查：共同alias减法、固定权重feature surgery、类别权重可改竞争、共同redundant text、Geometry共享Value保留、空间min-max可改竞争。新增7项交互审计检查：零交互、head无关文本变化、正负交互、GT不一致、类别顺序不一致、未完成结果、零预测精确率。

连同原20项semantic审计测试，**33项CPU单测通过**。它们不执行外部Torch模型，不证明校正效果。复现：

```text
F:/APP/codetool/anaconda3/python.exe -m unittest discover -s DINOtool/tests -p "test_semantic_*.py" -v
F:/APP/codetool/anaconda3/python.exe DINOtool/scripts/audit_semantic_readout_interactions.py research/matched_head_text_vdd_full_20260930_results.json research/matched_head_text_potsdam_full_20260930_results.json
```

严格training-free边界不变，Potsdam失败后的停止条件不变。本轮新增证据否定了两个具体的简单移植方向，并明确了语义读取流的待检验位置；**“可靠语义矫正”目标仍未完成。**

## 6. 现有数据能否支持继续离线定位

2026-10-01后续只读检查了A800工具目录下`results`的实际文件清单，共30,513个文件。没有发现`.npz/.npy/.pt/.pth/.safetensors/.pkl/.h5/.hdf5/.bin`特征或logit缓存，也没有遗留的`.dat/.mmap/.memmap`概率数组。此前部分旧实验保存了PNG预测，但最近的`geometry_vip_reliability_full_20260930`、`matched_head_text_20260930`、`support_conditioned_geometry_full_20261001`与`cross_scale_geometry_full_20261001`目录仅有JSON、日志及部分旧锁文件，没有逐位置特征流。

核对对应评测源码：每个样本的预测只进入混淆矩阵、变化计数和汇总诊断；`ProbabilityAccumulator`的临时`probabilities.dat/normalizer.dat`在context退出时清理。保存的标签预测即便存在，也不足以反推attention、prefix、残差、MLP及norm前后的高维特征。上述清单检查范围是这个授权工具目录的`results`，不声称检查过整台服务器的所有存储。

同次`nvidia-smi`确认0–7卡均1 MiB、0%利用率。旧`.lock`不能当成正在运行的证据；当前没有可等待的本轮推理。未删除文件、部署代码或启动任务。

因此当前推进的真实边界是：汇总结果能够继续解释TP/FP权衡，却不能验证错误具体进入哪条语义读取流。最新Potsdam候选仍退化，可靠校正没有完成；反复增加合成单测或重写报告不能替代缺失的实际观测。下一步需要用户明确允许一轮有限、固定样本的读取诊断；在此之前不恢复GPU推理，不调目标标签参数，也不将“可靠语义矫正”的目标缩小成完成文档。

## 7. 后续授权诊断：实际特征流已采集

用户随后仅授权原Geometry在VDD/Potsdam各8个固定样本上的观测。该诊断已完成：704/72个全部原滑窗，重放误差0，最终整图混淆矩阵与原模型完全相同；每图保留两个标签无关窗口的特征，并在CPU上核验固定A/B分解。没有更改模型/词表/阈值或恢复八集候选评测。第6节的“没有缓存”描述的是授权前的状态，不再代表此次采集后的状态。

本次Potsdam car精确率5.8594%、召回100%；VDD vehicle精确率13.5993%、召回99.0994%。固定类别对显示：部分car/vehicle误报早期探针已有错误方向，后续Geometry流只部分降低；部分water漏检则在第一层MLP前后探针中翻向other。MLP和最终LN beta存在大的相反贡献，不能仅凭某个单项正贡献实施全局删除。归因使用实际归一化与alias responsibilities，是条件分数重构而非因果删层效果。

完整协议、阶段表、固定对数值与局限见[各8样本原Geometry读取诊断](C:/Users/AH/Documents/ChatGPT/OVSS/research/GEOMETRY_READOUT_DIAGNOSTIC_8_20261001.md)。该诊断定位了继承误响应与后续有效证据丢失两类路径，没有得到经验证可靠的纠错器，目标仍未完成。
