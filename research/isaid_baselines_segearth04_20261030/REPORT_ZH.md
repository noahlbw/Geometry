# iSAID SegEarth-OV 官方配置复核（2026-10-30）

本目录记录 SegEarth-OV 在 iSAID validation split 的官方配置复核结果。评测使用 458 张原始 RGB 图像、16 类（background + 15 个前景类）和标准包含背景的 mIoU；标签只在预测完成后用于统计。

## 配置

- 上游仓库：`likyoo/SegEarth-OV`
- 固定 commit：`3e22a969b32c6d751bdbba64a88a0b670e630f55`
- 上游配置：`configs/cfg_iSAID.py`
- `prob_thd=0.4`（之前的 `cfg_vdd.py` 迁移值 0.3 不属于 iSAID 官方配置）
- `Resize=(448,448)`，保持原生 iSAID baseline runner、canonical taxonomy 和 SimFeatUp/JBU 权重

## 结果

| 项目 | 数值 |
|---|---:|
| 全量验证图像 | 458 |
| 标准 mIoU（含背景） | **18.9291%** |
| Pixel accuracy | 88.7072% |
| 非背景 mIoU（诊断） | 14.2667% |
| 7 张固定 cost 图 median | **295.5316 ms/image** |
| Peak allocated | **2596.45 MiB** |
| Peak reserved | **3042.00 MiB** |

成本边界包含 BGR 转换、官方 resize/preprocess、原生推理和 postprocess，排除磁盘 decode、模型/文本初始化和 metric/target 读取。成本只用于同一 runner 的相对比较；iSAID 既有其他结果若采用不同图像数或生命周期，不能直接排序。

## 与旧记录的差异

旧记录中的 `22.5194%` 使用 `cfg_vdd.py` 的 `prob_thd=0.3`，因此只能作为迁移配置诊断；本目录的 `18.9291%` 才是按 iSAID 上游配置得到的数值。两者差异为 −3.5903 pp。

## 机器可读证据

- `protocol.json`：配置、词表和 commit
- `merged.json`：458 张图的完整混淆矩阵、mIoU 和 per-class 统计
- `cost.json`：7 张固定 cost 图的 latency/memory
- `suite_results.json`：suite 完成状态和汇总
