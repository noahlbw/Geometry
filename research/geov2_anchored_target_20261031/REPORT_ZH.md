# Geov2 Anchored：AI-infra 加速与显存优化（2026-10-31）

本实验固定终版 Geov2 Anchored 语义路径，只改变推理执行方式。局部视野仍使用 `DonorBefore`，宽视野仍使用 Anchored QuerySlot（`K=24`），结构写回仍为：

\[
F^W=Y_0+\beta(\hat\Pi-\Pi)C_{\rm sem},\qquad
Z=L+g_0H(W-L).
\]

没有使用 `shared_canvas`，没有改变 alias bank、g0、温度、背景规则、窗口尺寸或文本缓存。

## 执行优化

- **BF16 冻结权重存储**：仅将冻结视觉模块中的 Linear/Conv 权重从 FP32 存储转换为 BF16；归一化、RoPE、关系矩阵和写回求解保持原精度路径。
- **固定形状局部批处理**：4 个独立的 512px 局部窗口组成 micro-batch；`DonorBefore` 对每个窗口逐样本执行，数学路径不变。
- **batch-safe attention shim**：只修复上游 attention 诊断分支的 batch=1 限制，QK、softmax 和 SDPA 运算保持不变。
- **TF32 与 `torch.compile`**：只编译固定形状视觉/投影 kernel，使用 `max-autotune-no-cudagraphs`，避免跨 tile 输出被 CUDA Graph 覆盖。
- **Anchored 写回的因子形式**：保存 Cholesky 因子而不是显式保存 \(H\)，用
  \[
  HD=D-(I+A^\top A)^{-1}D
  \]
  计算同一个结构算子作用，目标函数和写回公式不变。

代码入口：

- `DINOtool/dinotool/anchored_infra.py`
- `DINOtool/dinotool/batched_attention.py`
- `DINOtool/dinotool/geov2_wide_optimized.py`
- `DINOtool/scripts/eval_geov2_wide_optimization.py`

最终候选配置名为 `anchored_infra_fast_bf16`。

## 成本结果（A800）

成本协议固定 7 张图、2 次 warm-up、7 次 CUDA 同步重复；包含 RGB resize、两路视觉 forward、Anchored 读出、结构写回、恢复和 CPU 输出，不含模型初始化、磁盘解码、离线文本编码和标签读取。主显存指标是独立进程的 `torch.cuda.max_memory_allocated`，同时记录 reserved 峰值。

| 数据集 | 同协议 Anchored-infra 基线 ms / MiB | Fast-BF16 ms / MiB | 速度变化 | allocated 变化 | reserved 峰值 |
|---|---:|---:|---:|---:|---:|
| VDD | 308.31 / 1988.88 | **232.60 / 1360.68** | -24.56% | -31.59% | 1636.00 |
| Potsdam | 177.04 / 1569.95 | **95.50 / 896.58** | -46.06% | -42.89% | 1316.00 |
| VOC20 | 166.27 / 1625.04 | **86.99 / 951.16** | -47.68% | -41.47% | 1326.00 |
| VOC21 | 161.35 / 1613.50 | **88.18 / 938.55** | -45.35% | -41.83% | 1318.00 |

四个协议均满足本轮目标：VDD ≤250.24 ms / ≤1651.2 MiB，Potsdam ≤134.56 / ≤1252，VOC20 ≤128.16 / ≤1340，VOC21 ≤130.64 / ≤1335.2。MiB 门槛按 allocated 峰值判断；reserved 作为部署时的 allocator 参考单独列出。

## 精度验证

屏查使用预先固定的屏查集合；标签只用于最后计算 mIoU。全量验证覆盖全部图像。

| 数据集 | 屏查 mIoU (%) | 全量图像数 | Fast-BF16 全量 mIoU (%) | Anchored-infra 全量 mIoU (%) | 差值 pp |
|---|---:|---:|---:|---:|---:|
| VDD | 61.3525 | 80 | **60.5575** | 60.5606 | -0.0031 |
| Potsdam | 51.7519 | 504 | **54.9991** | 54.9837 | +0.0154 |
| VOC20 | 83.8204 | 1449 | **93.2877** | 93.2900 | -0.0023 |
| VOC21 | 75.0961 | 1449 | **74.8200** | 74.8120 | +0.0080 |

全量结果说明 BF16/编译执行只带来千分之几 pp 的数值变化，未改变模型行为结论。

## 数值审计

独立 smoke 结果如下：

| 数据集 | assignment 最大误差 | slot argmax 不一致率 | FP64 因子作用误差 | FP32 因子作用误差 | 局部特征最大误差 | 局部平均 cosine |
|---|---:|---:|---:|---:|---:|---:|
| VDD | 1.28e-6 | 0 | 1.32e-14 | 1.18e-4 | 1.04e-2 | 0.999920 |
| Potsdam | 3.84e-6 | 0 | 1.14e-14 | 5.31e-5 | 1.18e-2 | 0.999932 |
| VOC20 | 3.58e-7 | 0 | 1.14e-14 | 1.22e-4 | 1.21e-2 | 0.999925 |
| VOC21 | 4.26e-6 | 0 | 1.34e-14 | 1.17e-4 | 6.04e-3 | 0.999950 |

局部特征的 `local_features_equal=false` 是 BF16/TF32 执行造成的有限数值差异；四个数据集均通过 `max_abs ≤ 5e-2`、`mean cosine ≥ 0.995` 的 smoke gate，且 slot 的离散分配完全一致。默认 FP32/FP64 compatibility 路径和原有 DonorBefore 单元测试保持不变。

## 复现

```powershell
$env:PYTHONPATH='C:/Users/AH/Documents/ChatGPT/OVSS/tools;C:/Users/AH/Documents/ChatGPT/OVSS/artifacts/geometry_github/tools'
python tools/run_anchored_target.py prepare
python tools/run_anchored_target.py launch --variants anchored_infra_fast_bf16 --mode cost
python tools/run_anchored_target.py launch --variants anchored_infra_fast_bf16 --mode screen
python tools/run_anchored_target.py launch --variants anchored_infra_fast_bf16 --mode full
python tools/run_anchored_target.py launch --variants anchored_infra_fast_bf16 --mode smoke
```

原始 JSON、统计矩阵和协议文件位于本目录的 `cost/`、`screen/`、`full/`、`smoke/` 子目录。相关单元测试：

```powershell
$env:PYTHONPATH='C:/Users/AH/Documents/ChatGPT/OVSS/artifacts/geometry_github/DINOtool'
python -m pytest DINOtool/tests/test_anchored_infra.py DINOtool/tests/test_anchored_slot_memory.py DINOtool/tests/test_fast_dense.py -q
```

本地结果为 **15 passed**，`compileall` 通过。
