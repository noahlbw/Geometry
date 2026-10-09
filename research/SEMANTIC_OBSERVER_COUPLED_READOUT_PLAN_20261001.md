# Geometry 语义纠错：新增授权后的固定研究计划

2026-10-01。用户已授权继续实验、A800八卡可用；此前仅原Geometry诊断的边界不再阻止新增实验。目标仍是可靠语义矫正及统一八域效果，不把完成诊断当作目标完成。

## 当前证据

保留原Geometry/Multiscale。32个缓存快照显示，邻域语义可以纠正部分vehicle–road误报及water漏检，但大部分other→vehicle误报也得到邻域支持。统一整窗去偏置会抑制正确水体。不能据高AUC采用标签拟合阈值，也不能统一删除MLP/LN。

## 第一阶段：固定稠密观察

沿用VDD/Potsdam各8个固定样本，固定现有20词/类、标签映射、checkpoint。VDD四个分片用GPU0–3，Potsdam四个分片用GPU4–7。比较：

- Geometry：原512/128、RS模板、normalized LME。
- Native_Local_RS：同输入、同文本、同聚合的原生head。
- VIP_Matched_Local_RS：同512输入、同精度与文本，pinned proxy扩展到32×32；不是VIP官方完整模型。
- VIP_Broad_RS：官方宽视野视觉读出、相同RS文本与LME。
- VIP_Broad_ImageNet_LME：相同宽视野，改为pinned ImageNet逐模板平均相似度与LME。
- VIP_Broad_Profile：相同宽视野与词表，pinned ImageNet/原评分；无背景阈值。
- MeanProb50：Geometry与VIP_Broad_Profile的简单概率平均，作为明确对照。

宽视野固定长边448、336窗口/112 stride；官方VIP运行时使用独立冻结副本，不改变原Geometry的fp32权重/bf16 AMP。局部head对照不改变模板、视野或聚合；跨局部/宽视野比较明确包含视野及官方精度差异。

保存每个窗口的所有class scores及原Geometry非self支持读出、各阶段整图预测PNG、完整混淆矩阵、[原A,候选B,真实C]变化计数。标签只在预测后审计。合并时验证8/8唯一覆盖和原Geometry混淆矩阵与此前诊断完全相同。

## 后续实现门槛

只用这些诊断判定信息源是否值得继续，不按真实类拟合分支、词表或阈值，不把真实标签上的最优门控称为training-free。若同预算不同head/视野仍不能辨别具体纠错方向，就拒绝把它作为可靠教师的主张。

耦合候选必须具体改变Geometry读取的内容或定位，而不是仅给两个完整分割结果改名融合。冻结一个统一规则后，优先验证完整VDD80/Potsdam504，再向另外六域迁移；保留原Geometry、Multiscale和同预算简单操作作为对照。每阶段报告mIoU和TP/FP/FN，不以总正确像素、trust均值或AUC代替。若失败，保留结果并修改或拒绝假设，不按数据集挑最优配置。

这份计划不是成功模型声明，不承诺超过VIP或CVPR录用。八集均已用于开发，后续评测不能重新称为独立未触碰测试。
