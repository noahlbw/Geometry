# Geometry + Cross-scale Relation Grounding：完整模型八数据集评测

## 固定模型

实现签名：`geometry-cross-scale-relative-relations-v1-20261001`。

本轮按用户要求直接评测完整模型，不运行逐模块消融或参数搜索。

- 冻结 DINOv3 / DINO.text；两个 Geometry head block；保留每个原始 detail crop 的 prefix 和 native patch attention mass。
- 每个 512 工作区共用 local 一次、detail 四次、context 一次 backbone 编码。
- fine 关系与 full-context 关系使用同一 enclosing 1024 图像坐标，temperature=0.1、sigma=0.25。
- 细关系聚合到已知 4×4 footprint；增量为 log(coarse conditional + 1e-6) − log(fine reference + 1e-6)。
- coarse 可用 donor 质量处理外圈覆盖缺口；只读取实际 fine Values，不把 coarse Values 写入 fine。
- 两个冻结 head block 都根据真实前一层状态重新计算 Q/K/V；原 projection / LayerScale / residual / MLP / normalization 保留。
- 每类固定历史 20 aliases，原遥感模板、normalized LME temperature=0.07 不变；没有筛词、额外 gate 或逐图优化。
- 输出为 (Local + Detail + Grounded)/3；tile512、overlap128、输出 temperature=0.07、Hann 概率拼接沿用原协议。
- 参数统一；本轮预测路径不读取目标标签，运行前配置已冻结，没有按中途指标调参。模型设计参考过此前验证集结果，因此这是探索性开发验证，不是未见测试集。

源码：[完整读出](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/dinotool/cross_scale_geometry.py)、[评测器](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/eval_cross_scale_geometry.py)、[八卡队列](C:/Users/AH/Documents/ChatGPT/OVSS/DINOtool/scripts/launch_cross_scale_geometry_a800.sh)。

## 验证与启动

5 项 CPU 正确性检查通过：四 crop 网格映射、有效成员关系聚合、零增量/零覆盖恒等、padding 排除与格内条件保持、两层 Values 与 origin prefix 的独立 Geometry 等价。

OEM 单图完整推理检查通过：9 个工作区，推理 2.2398 秒，峰值 allocated CUDA memory 3799.88 MiB；此单图指标不用于选择模型，也不作为性能结论。

A800 0–7 卡启动前均空闲。2026-10-01 00:14:56（Asia/Shanghai）启动 tmux `csgg01_g0..g7`。

远端结果：`/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/cross_scale_geometry_full_20261001`。

| 数据集 | 固定完整图像数 | 分片数 | 调度 |
|---|---:|---:|---|
| VDD | 80 | 4 | GPU0–3 首轮 |
| Potsdam | 504 | 4 | GPU0–3 第二轮 |
| UDD5 | 40 | 4 | GPU0–3 第三轮 |
| OEM | 384 | 4 | GPU4–7 首轮 |
| Vaihingen | 113 | 4 | GPU4–7 第二轮 |
| LandCover.ai | 1602 | 4 | GPU4–7 第三轮 |
| LoveDA | 1669 | 4 | GPU4–7 第四轮，统一计算 P/D |
| FLAIR-1 | 15700 | 8 | 各 worker 完成先前任务后运行 |

每次转入下一任务先检查 GPU 空闲，不干扰其他任务。一个 worker 失败则停止该 worker 并保留日志，不覆盖、重启已有输出。最后一个完成分片负责合并完整结果，校验唯一覆盖及统一签名。

`cross_scale_reference_20261001.json` 保存历史样本序列与词表指纹；每个 evaluator 在模型加载前确认数据与现有 full-20 对照匹配。

## 结果状态

**已按用户条件停止，不再运行。** Potsdam 完整结果为 39.5178，低于历史匹配 Multiscale 的 40.4433，触发“变差就停止”的条件。2026-10-01 00:34:01（Asia/Shanghai）停止本轮 `csgg01_g0..g7` 及其所属评测进程，保留日志和输出。最终检查 A800 0–7 卡均空闲；其他 tmux 任务未受影响。

以下五集有完整合并结果，均为 `coverage_verified=true`，数量、唯一覆盖、模型签名及历史样本/词表指纹匹配。差值单位为 mIoU 百分点。

| 数据集 | 图像 | Geometry | Multiscale | CS_Grounded | 新版 − Multiscale |
|---|---:|---:|---:|---:|---:|
| VDD | 80 | 38.8511 | 41.2235 | 35.9126 | -5.3109 |
| Potsdam | 504 | 40.6892 | 40.4433 | 39.5178 | -0.9255 |
| OEM | 384 | 44.6097 | 45.6498 | 45.8245 | +0.1747 |
| Vaihingen | 113 | 4.8494 | 4.5630 | 4.7516 | +0.1886 |
| LandCover.ai | 1602 | 59.2340 | 59.5143 | 59.0699 | -0.4444 |

其余集合中止时状态：UDD5 38/40（3/4 分片完成），LoveDA 1266/1669（0/4 完整分片），FLAIR-1 272/15700（2 个分片已有输出，0 个完整分片）。不合并、不报告这些集合为全量分数。

| 数据集 | 并行 wall 秒 | 各分片评测耗时之和 秒 | 峰值 allocated CUDA MiB |
|---|---:|---:|---:|
| VDD | 552.54 | 2167.68 | 3798.47 |
| Potsdam | 250.54 | 976.39 | 3796.23 |
| OEM | 187.43 | 728.59 | 3799.88 |
| Vaihingen | 56.14 | 220.03 | 3796.25 |
| LandCover.ai | 94.77 | 368.07 | 3782.85 |

耗时之和沿用 JSON 的 `aggregate_gpu_seconds` 字段，它是分片评测 wall time 求和，不是硬件 profiler 测得的纯 CUDA kernel 时间。

完整结果：[VDD](C:/Users/AH/Documents/ChatGPT/OVSS/research/cross_scale_geometry_vdd_full_20261001_results.json)、[Potsdam](C:/Users/AH/Documents/ChatGPT/OVSS/research/cross_scale_geometry_potsdam_full_20261001_results.json)、[OEM](C:/Users/AH/Documents/ChatGPT/OVSS/research/cross_scale_geometry_oem_full_20261001_results.json)、[Vaihingen](C:/Users/AH/Documents/ChatGPT/OVSS/research/cross_scale_geometry_vaihingen_full_20261001_results.json)、[LandCover.ai](C:/Users/AH/Documents/ChatGPT/OVSS/research/cross_scale_geometry_landcoverai_full_20261001_results.json)。

本轮结论：不采用此版本替代当前 Geometry/Multiscale 主模型。VDD roof/water 大幅退化，Potsdam tree 退化，car 过预测基本未解决；不能从 OEM 的小幅提升宣称跨域机制成立。详细架构、失败分析与论文缺口见 [综合汇报](C:/Users/AH/Documents/ChatGPT/OVSS/research/OVSS_COMPREHENSIVE_REVIEW_AND_CVPR_20261001.md)。

对照采用已有完整 Geometry / Multiscale；VIP 对照须区分同20词无背景阈值与官方短词+阈值。重点分析终版在八个域的每类收益/损失、car/vehicle 精确率/召回率与预测面积、关系偏移程度、推理时间和显存。

注意：iSAID 在当前环境没有有标签验证，LandCover.ai 为第八集；OEM 是现有 384 张可用验证图像；Potsdam 504 tiles 来自14个 parent scenes。Vaihingen 沿用现有三波段输入协议，不能把其此前异常低分直接解释为本模型问题。八集均已用于探索性开发，不能称为 untouched validation。
