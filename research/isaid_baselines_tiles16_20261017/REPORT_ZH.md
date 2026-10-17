# iSAID validation：统一官方 tile、16 类背景计入的复现结果

本轮把 iSAID 的评测协议重新对齐到 VIP 论文表格。正式比较使用同一份官方 validation RGB、semantic mask 和 instance mask，先把 458 张原图按官方规则切成 896×896 tiles，overlap=384，共 11,644 个 tiles。所有模型只读取 RGB；mask 仅在预测完成后用于评分。

## 正确的 mIoU 口径

iSAID 是 16 类：background + 15 个前景类。黑色 `(0,0,0)` 是有效的 background，必须计入 mIoU；只有 padding/无效区域值 255 被忽略。因而混淆矩阵为 16×16，mIoU 是 16 类 IoU 的平均值。此前把 background 当作 ignore、只平均 15 个前景类的数字属于 foreground-only 诊断，不能与 VIP 论文表格比较。

VIP 的官方设置同时使用 `cls_iSAID.txt` alias、`prob_thd=0.3`、`background=True`，并采用 896×896、overlap=384 的滑窗协议。复现得到 **26.1325%**，与论文表格 **26.1%** 对齐，这是协议正确的交叉检查。

## 统一 tile 协议结果

| 方法 | mIoU (%)，16 类含 background | 固定 7 tile 中位延迟 (ms) | 峰值 allocated (MiB) | 备注 |
|---|---:|---:|---:|---|
| VIP（官方复现） | **26.1325** | 127.34 | 1,816.8 | 官方 alias，`prob_thd=0.3`，background enabled |
| Geov2 Anchored | **27.0134** | **167.50** | **1,702.2** | DonorBefore + QuerySlot K=24 + coupled writeback，g0=0.5 |
| SegEarth-OV | 22.5194 | 292.26 | 1,155.0 | 11,644 tiles，16 类 |
| PTC | 12.6610 | 248.60 | 716.1 | 11,644 tiles，16 类 |
| PEARL | 10.8161 | 65.30 | 476.6 | 11,644 tiles，16 类 |
| CorrCLIP | 10.2868 | 308.79 | 2,609.1 | 在线 SAM2，11,644 tiles，16 类 |
| GLA-CLIP | 10.0159 | 56.95 | 1,033.9 | 11,644 tiles，16 类 |

native baseline 的每个 `merged.json` 都验证了 `processed_images=total_images=11,644`，并且预测混淆矩阵的 target 行和与统一 reference target count 一致。CorrCLIP 在旧的 458 张超大原图协议上会因 SAM2 RLE 的 INT_MAX 限制失败；切成官方 tile 后已成功完成全量评测。

Geov2 在同一协议下比官方 VIP 高 **0.8809 pp**（27.0134 vs. 26.1325），但单 tile 中位延迟为 VIP 的约 **1.32×**（167.50 vs. 127.34 ms），峰值 allocated 显存为 **1,702.2 MiB**，略低于 VIP 的 1,816.8 MiB。相对最强 native baseline SegEarth-OV，Geov2 高 **4.4940 pp**。

## 为什么以前的数字偏高

旧 VIP 38.5393%、旧 Geov2 34.1863% 以及旧 native 数字都使用了 foreground-only 统计，且旧 VIP 还只使用每类一个 canonical query、`prob_thd=0.0`。它们忽略了数量极大的 background 误分，因此会显著高估 mIoU。旧 raw-image 结果还和官方 tile 结果混用了不同输入协议。正式主表只能使用本报告的 16 类、官方 tile 结果。

显存数字也必须在同一边界下比较。本表 baseline 的显存是固定 7 tile、热身后的 allocated 峰值；VIP 的 1,816.8 MiB 使用 text-cache 后的单次 tile 测量。此前 8,701 MiB 的 VIP 数字保留了完整 text tower，并跨 458 张原图取峰值，不能拿来直接证明 VIP 算法本身更占显存。

## 可复核文件

- VIP：A800 `results/vip_official_isaid_tiles16_20261017/official_tiles_r3.json`
- native suite：A800 `results/training_free_baselines_isaid_tiles16_20261017_r2/`
- tile manifest：`/data-cache/isaid_official_tiles_20261017/samples.json`
- Geov2 输出：A800 `results/isaid_qwen_kev_20261016_r1/deployment/anchored_tiles16_r2.json`

Geov2 全量输出记录为 11,644/11,644 tiles，wall time 2,370.29 s；统一 tile 成本输出为 `anchored_tiles_cost_20261017.json`。
