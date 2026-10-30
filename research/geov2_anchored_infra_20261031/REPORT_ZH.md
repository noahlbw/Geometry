# Geov2 Anchored-infra：执行层优化（2026-10-31）

本目录记录 Geov2 Anchored 终版的 AI-infra 优化。模型语义和论文公式保持不变：

\[
F^W=Y_0+\beta(\hat\Pi-\Pi)C_{\rm sem},\qquad
Z=L+g_0H(W-L).
\]

局部端点仍由冻结的 `DonorBefore` 路径产生；宽视野仍是 Anchored QuerySlot-24 读出；结构写回仍由同一个 `H` 完成。没有切换到 `shared_canvas`，没有改变 alias bank、g0、温度、背景规则或 QuerySlot 槽数量。

## 代码入口

- `DINOtool/dinotool/anchored_infra.py`：执行配置、共享视觉权重存储、流式 logit 准备和 TF32 开关。
- `DINOtool/dinotool/geov2_wide_optimized.py`：Anchored 读出、Cholesky 形式的 `H` 作用、宽视野中间量复用。
- `DINOtool/dinotool/natural_static_search.py`：接收已经压缩成类别 logit canvas 的中间量。
- `DINOtool/scripts/eval_geov2_wide_optimization.py`：`--variant anchored_infra` 评测入口。
- `DINOtool/scripts/audit_anchored_infra_parity.py`：与 FP64 compatibility 路径的数值审计。

运行配置为 `anchored_infra`：冻结视觉 backbone 只保留一份参数存储，宽视野 head 仍独立；K=24、Anchored residual readout、结构求解 FP32、局部分支保持 DonorBefore。宽视野中间量在单图推理期间按 tile 生命周期管理；兼容 Anchored evaluator 还支持在转成类别 logit canvas 后立即释放高维特征列表，避免把整幅图的 DINO 特征长期驻留。

结构求解使用等价的因子形式。对归一化后的关系矩阵记为 `A`，原式中的
`H=(I+A^T A)^(-1)A^T A` 作用在差分 `D=W-L` 上实现为：

`H D = D - (I+A^T A)^(-1) D`。

因此只保存 Cholesky 因子，不显式保存完整 `H`；这是执行表示的变化，不是模型目标的变化。所有局部/宽视野观测、写回顺序和最终 argmax 保持不变。

## 全量验证集

全量结果使用优化协议中冻结的 alias bank、g0、温度和背景规则；标签只在最后计算 mIoU。A800 全量结果如下：

| 数据集 | Anchored-infra mIoU (%) | 图像数 | g0 | alias bank |
|---|---:|---:|---:|---|
| VDD | 60.5606 | 80 | 1.00 | official_imagenet |
| Potsdam | 54.9837 | 504 | 0.50 | official_imagenet |
| ADE150 | 31.5768 | 2000 | 0.25 | semantic_segmentation |
| VOC20 | 93.2900 | 1449 | 1.00 | semantic_imagenet |
| VOC21 | 74.8120 | 1449 | 0.75 | official_segmentation |
| COCO-Object81 | 49.6131 | 5000 | 0.50 | official_imagenet |

逐图 confusion matrix 和配置保存在 `full/<dataset>/results.json`；COCO-Object81 的完整结果已补入同一位置。

## A800 成本

成本边界与历史表一致：同一 A800、batch=1、每协议 7 张固定图、每张 2 次 warm-up 和 7 次 CUDA 同步重复；包含 RGB resize、视觉 forward、Anchored 读出、结构求解、恢复和 CPU 标签输出；不含解码、初始化、离线文本编码和标签读取。显存是独立进程的 `torch.cuda.max_memory_allocated` 峰值。

| 数据集 | 历史 Anchored (ms / MiB) | Anchored-infra (ms / MiB) | 延迟变化 | 显存变化 |
|---|---:|---:|---:|---:|
| VDD | 308.94 / 2063.89 | 305.10 / 1988.88 | -1.24% | -3.63% |
| Potsdam | 171.97 / 1572.68 | 175.27 / 1569.95 | +1.92% | -0.17% |
| ADE150 | 186.27 / 2872.10 | 186.18 / 2352.69 | -0.05% | -18.08% |
| VOC20 | 166.42 / 2253.26 | 159.87 / 1625.04 | -3.94% | -27.88% |
| VOC21 | 163.98 / 1669.35 | 159.85 / 1613.50 | -2.52% | -3.35% |
| COCO-Object81 | 176.84 / 2722.19 | 167.04 / 1895.07 | -5.54% | -30.38% |

负号表示优化后更低。结果显示主要收益是显存下降和稳定的执行成本；Potsdam 的延迟基本持平，不能表述为所有数据集都大幅加速。

VIP 的 A800 allocated memory 参考为：VDD 2332.6 MiB、Potsdam 1739.3 MiB、ADE150 2172.5 MiB、VOC20 1743.9 MiB、VOC21 1744.8 MiB、COCO-Object81 1922.4 MiB。Anchored-infra 分别低 14.8%、9.7%、高 8.3%、低 6.8%、低 7.5%、低 1.4%；ADE150 仍略高，应在论文中如实披露。

## 数值审计与测试

- 本地：`21 passed`，覆盖 Cholesky writer、slot assignment、Anchored readout、批量采样和 DonorBefore 基础行为。
- A800 smoke：assignment 最大绝对误差约 `5.66e-7`；FP64 因子作用误差约 `8.94e-15`；FP32 因子作用误差约 `4.81e-6`；局部 DonorBefore 特征保持一致。
- 与 FP64 compatibility path 的固定成本图对照：VDD、Potsdam、VOC21 的变化像素比例约 `0.17%`、`0.24%`、`0.22%`，逐图 mIoU 仅有千分之几变化。该路径应描述为“执行层近似，误差有界”，不应声称逐像素 bit-exact。

对照输出保存在 `parity/`。成本和全量 JSON 均保留配置、cache identity、数据覆盖、GPU 和计时边界，便于复核。
