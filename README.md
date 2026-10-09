# Geometry: frozen structure-guided open-vocabulary segmentation

2026-10-10更新：新增当前Kev开发选择后的冻结模型、15协议的实际alias与YAML，以及无需历史实验缓存的[独立推理／评测入口](docs/DEPLOYMENT_20261010.md)。原2026-10-09模型和实验记录保留在下面，不能将两版结果混用。最新精度见[Kev完整结果](research/KEV_ALIAS_SEARCH_20261009.md)和[相同词表VIP对照](research/VIP_KEV_SAME_VOCABULARY_20261010.md)，执行优化与验证见[效率报告](research/COMPACT_DEPLOYMENT_20261010.md)。

```bash
python DINOtool/scripts/infer_frozen_geometry.py --help
python DINOtool/scripts/evaluate_frozen_geometry.py --help
```

当前配置是带标签开发选择后的任务特定配置；Kev只在离线词表判断中使用，单图分割不加载Kev。词表不是每类强制20个，VDD/Potsdam采用规范化解析的官方短词；不要将所有收益归为Kev筛词。精确历史二进制文本缓存不纳入Git，独立入口从词表重新编码并记录本机缓存身份，浮点并列处可能与历史缓存有细小差异。

以下为2026-10-09原版研究快照。该版冻结终版为 **Geometry局部读取＋VIP来源的宽视野观测＋Geometry关系支持的耦合写回**。原版主入口为 `TaxonomyInference.for_finalization(..., ordinary_alias_policy="uniform")`；2026-10-10部署配置使用上面的独立入口。

![Final Geometry architecture](docs/architecture.svg)

这份仓库同时保存最终代码和探索历史。旧实验模块仍在源码中，但不代表全部属于终版；`DINOtool/README.md`是早期工具箱说明，其默认DinoSplat/训练适配/SAM等路线不是这里的最终模型。以本页、冻结配置与下列说明为准。

## 阅读入口

| 内容 | 文档 |
| --- | --- |
| 终版框架、公式、职责与创新边界 | [架构说明](docs/ARCHITECTURE.md) |
| 筛词／降权尝试、证据与取舍 | [Alias实验综述](docs/ALIAS_STUDY.md) |
| 全量精度、VIP对比、速度和显存 | [完整实验表](docs/RESULTS.md) |
| 环境、模型入口、数据与复现要求 | [复现说明](docs/REPRODUCIBILITY.md) |
| 历史报告和筛词代码的完整索引 | [实验与源码索引](docs/EXPERIMENT_INDEX.md) |
| 完整终版原始分析 | [最终模型报告](research/CVPR_FINAL_MODEL_20261009.md) |
| 词表、模板、参数与选择标准 | [protocol.json](research/alias_finalization_20261009/protocol.json) |
| 最终策略和比较来源 | [final_model.json](research/alias_finalization_20261009/final_model.json) |

## 原版结果（2026-10-09）

- 八遥感全量20,092图：终版等域平均 **49.1833 mIoU**，较强已测VIP参考 **44.0039**，差 **+5.1794pp**，八域均领先该参考。LoveDA主均值只计D一次，P另报。
- 同观测等权logit融合为 **46.9965**；结构耦合提升 **+2.1869pp**，条件95%配对区间 **[+2.0106,+2.3440]**。
- 自然图像没有全面超过VIP：ADE150为 **31.1862 vs 29.1387**；VOC21为 **70.4508 vs 73.2591**；COCOObject81为 **44.2396 vs 48.9955**。全部结果与缺失比较均保留。
- 整图抽样中位延迟 **185.30–368.71ms**，约 **1.717–3.231×VIP**；峰值allocated **5514.66–6538.80MiB**。每协议7图×7次独立进程同步计时，不是全数据集平均延迟。

## Alias最终决定

关闭额外普通局部coverage-soft重分配，保留原局部LME和宽视野图像salience。关闭后主要四域平均代价 **0.0422pp**，最差迁移入口代价 **0.1497pp**，通过预先冻结的实际简化门槛。软权重并非完全无用，但不足以证明“识别有害alias”。原Retained代码和全部结果保留；没有按域挑选权重开关。

Uniform不等于各词实际影响相同：LME的响应导数与softmax责任成正比，宽视野已有图像条件salience。自然普通前景两臂本来相同，不能用它们相等证明词处理泛化。PC60的额外401残余概念与protected rule另行保留和披露。

## 比较与投稿边界

VIP为本地、明确修复全mask attention行后的实现测量，不是认证的论文数字。VDD/Potsdam/Vaihingen使用官方短词；另外五个遥感域是外部20词及论文规则蒸馏适配。旧NaN污染或旧Vaihingen输入下的低分不作为主比较。

全部15协议共45,200次图像评测；VOC20/21和PC59/60共享来源，不能称45,200个独立图像。LandCover.ai代替未获得可评分标签的iSAID；Cityscapes未纳入。既有词表和配置参与过有标签开发，冻结重跑不等于独立测试。

论文主线是视觉关系组织局部读取和限制跨视野修正写回。明确归属VIP观测/salience与经典二次解；不把现有alias启发式、变长词表支持或所有历史算子都当作新贡献。最终任务路线的最近算子公平对照、独立来源验证及空间机制证据仍是投稿缺口。

## 仓库范围

`DINOtool/dinotool`保存实现，`DINOtool/scripts`保存评测／诊断／统计脚本，`DINOtool/tests`保存测试；`research`保存相关原始报告、冻结配置、汇总混淆矩阵和计时证据。`RELEASE_MANIFEST.json`列出快照内容。

权重、数据集、二进制文本缓存、逐图NPZ和本机SSH管理配置不纳入Git。第三方实现与checkpoint遵循各自上游条件，来源见[复现与归属](docs/REPRODUCIBILITY.md)。
