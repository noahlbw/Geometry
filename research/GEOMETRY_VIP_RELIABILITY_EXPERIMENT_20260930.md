# Geometry / VIP 互补与选择可行性实验

2026-09-30。用户授权 A800 8 卡实验。仅 VDD 80 图、Potsdam 504 tiles；不恢复已停止的区域核验 LoveDA / FLAIR-1，不重跑八个数据集。

## 预声明设置

- GPU 0–3：VDD 四分片；4–7：Potsdam 四分片。启动前八卡进程、显存与利用率检查均须通过。
- 固定原来的 20 aliases、类别定义、掩码与完整样本序列。权重冻结，参数不按目标标签选择。
- Geometry / Multiscale：保留 512 / 128、三视图、RS 模板、LME 0.07、Hann 概率融合的完整强控制。
- VIP20：保留 pinned proxy 头、ImageNet 模板、448 长边、336 / 112、原始 scorer，关闭置信度转背景。
- 两个强分支视野／模板／scorer 不同，这是实践组合可行性实验，不冒充只改变视觉头的 matched factorial。使用两个相同 checkpoint 的冻结运行时副本，以防官方 VIP 的 fp16 转换污染原 Geometry 控制。

## 七个输出

Geometry、Multiscale、VIP20、MeanProb50、MaxConfidence、LeaveFamilyOut、GroundedLeaveFamilyOut。

MeanProb50 使用两分支完整概率的等权平均。MaxConfidence 在两者不同预测处按原生最大 softmax 置信度选择；它只是参照，不将不同分支的 softmax 视作已校准正确率。

LeaveFamilyOut：以 Multiscale / VIP 的当前竞争类别为单位，分别找出两分支该类别最高响应 alias；用 RS 文本 cosine>=0.97 定义近同义 family，将两个被审查 family 同时从两类别、两文本分支中留出。使用两边存留的归一化 LME cosine margin 进行比较，VIP 的 v-g margin 必须正，且 VIP 与 Geometry 的同方向 margin 之和为正，才建议切到 VIP。缺少存留参考时未知、不修改。这里的 Geometry 参考来自局部 alias 响应，最终基线仍是 Multiscale。

GroundedLeaveFamilyOut：在上述条件基础上，原始 DINO 特征的共享邻域参考也须支持相同方向。32x32 网格；detail raw 特征池化到该网格；query self 与 padding 不进入邻域；半径 4 patches 内按 raw feature cosine / 0.10 加位置项选 32 邻居。两类别读取完全相同邻域，被审查 family 始终留出。无邻居时未知。没有 all-view probability>=0.75、固定 Top-K 留词、每类强制原型或 GT 面积先验。

局部二值选择图通过 512 Hann 重叠融合，阈值固定 0.5 后在完整图像分支预测之间选择。该版本只检验无标签选择信号，不物理删除词，不声称实现了此前提出的完整语义迁移终版。

## 标签与评价

全部分支预测及选择图先完成，之后才加载目标掩码。全分辨率统计：

1. 各方法 mIoU、逐类 IoU、预测面积、precision、recall。
2. Multiscale / VIP 的四种正确性交集，逐类报告；GT oracle 仅报 pixel accuracy，不报可部署结果或 mIoU 上限。
3. 对 Multiscale 与 VIP 分别统计有益／有害／错到错改动。
4. 各选择器 beneficial_retention、harmful_rejection、只一方正确时的正确路由率。
5. wall time、aggregate GPU time、峰值显存、完整唯一覆盖与规则一致性。

判断以联合选择是否优于强分支和简单融合为主，不能仅因改动变多、oracle 高或路由率高就宣布模型成功。若 GroundedLeaveFamilyOut 不优于简单参照，不能通过改参数或更换命名将本次结果包装为成功。以上两个数据集已用于开发，结果仍是探索性验证。

## 输出

远端 `results/geometry_vip_reliability_full_20260930/{vdd,potsdam}/s0..s3`；会话 `gvr30_gpu0..gvr30_gpu7`。

实现 `dinotool/branch_reliability.py`、`scripts/eval_geometry_vip_reliability.py`；合并 `scripts/merge_geometry_vip_reliability.py`；启动 `scripts/launch_geometry_vip_reliability_a800.sh`。

阶段：已通过五项远端 CPU 张量测试、Python 编译检查和 launcher shell 检查。单图 Potsdam 三个控制的混淆矩阵与原评测器完全一致，样本 key 一致；正确性交集计数与有效混淆矩阵像素数一致；单分片 merger 完成。单图仅用于接口/控制复现，不选择配置。

8 卡全量已完成：`gvr30_gpu0..7`，VDD 每分片 20 张、Potsdam 每分片 126 tiles。历史结果均未覆盖。

## 已验证完整结果

两组合并均 coverage_verified=true，完整唯一序列为 80 / 504；样本 SHA、词表 SHA 与 checkpoint manifest 匹配历史强控制。Geometry、Multiscale 和 VIP20 的完整混淆矩阵分别与历史实现完全一致。两集 selector 参数一致。

| 方法 | VDD | Potsdam |
|---|---:|---:|
| Geometry | 38.8511 | 40.6892 |
| Multiscale | 41.2235 | 40.4433 |
| VIP20，无背景阈值 | 51.0697 | 42.3519 |
| 固定等权概率融合 | 53.8316 | 42.4761 |
| 最大置信度选择 | 51.7483 | 42.9481 |
| 留词组选择 | 52.9364 | 41.1135 |
| 留词组＋Geometry 邻域支持 | 52.9117 | 40.9863 |

结论：确有可利用的分支互补，VDD 固定融合相对同词表 VIP20 +2.7619。但复杂选择规则没有超过两集各自的最佳简单参照；加 Geometry 邻域支持还使留词组选择在两集都略下降。

GroundedLeaveFamilyOut 的有益切换保留率 / 有害切换拒绝率：VDD 71.55% / 72.77%，Potsdam 32.45% / 77.04%。Potsdam 的否决同时丢掉多数有益切换。不能仅凭拒绝率高宣称可靠核验成功。

VDD wall 516.65s、aggregate shard time 2050.86s、峰值 CUDA 5788.23 MiB；Potsdam 275.36s / 1093.01s / 5663.68 MiB。均为七输出联合评测、两个冻结运行时副本的成本，不是单方法独立 latency。

本轮不继续其他数据集、不按这两个结果选择不同数据集的 winner 组成统一模型。固定融合应保留为新的必要对照，后续模块须证明超过简单组合。现有计数属于探索性开发，不代表未接触测试集或统计显著性。

完整逐类 IoU、precision/recall/面积、改动计数及控制复现见 `GEOMETRY_VIP_RELIABILITY_FULL_RESULTS_20260930.md`；原始合并文件 `geometry_vip_reliability_20260930/{vdd,potsdam}_merged.json`。
