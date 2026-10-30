# Geov2 Anchored 20% 目标核对

用户给定的 Geov2 基准是 `ms / MiB`。本轮部署目标按每个数据集分别取基准的 80%：

| 数据集 | Geov2 基准 | 20% 目标（速度 / 显存） | `anchored_infra_fast_bf16` | 速度降幅 / 显存降幅 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| VDD | 312.8 / 2064 | ≤250.24 / ≤1651.20 | 232.60 / 1360.68 | 25.64% / 34.08% | 通过 |
| Potsdam | 168.2 / 1565 | ≤134.56 / ≤1252.00 | 95.50 / 896.58 | 43.22% / 42.71% | 通过 |
| VOC20 | 160.2 / 1675 | ≤128.16 / ≤1340.00 | 86.99 / 951.16 | 45.70% / 43.21% | 通过 |
| VOC21 | 163.3 / 1669 | ≤130.64 / ≤1335.20 | 88.18 / 938.55 | 46.00% / 43.77% | 通过 |

表中 `anchored_infra_fast_bf16` 的显存采用与 VIP/Geov2 成本表一致的独立进程 `torch.cuda.max_memory_allocated`。CUDA allocator 的 `reserved` 峰值另行记录在 [REPORT_ZH.md](REPORT_ZH.md)；它不是本表的主比较指标。

## 设计方案

### 保持的模型语义

- 局部分支继续使用 `DonorBefore`。
- 宽视野继续使用 Anchored QuerySlot，槽数固定为 `K=24`。
- 保持
  \[
  F^W=Y_0+\beta(\hat\Pi-\Pi)C_{\rm sem},
  \qquad Z=L+g_0H(W-L).
  \]
- alias bank、g0、局部/宽视野温度、背景规则、窗口和文本缓存全部冻结。
- 不使用 `shared_canvas`，不使用宽视野浅层截断作为最终路径。

### 只改变执行系统

1. 共享冻结视觉参数的物理存储，避免局部和宽视野各保留一份权重。
2. 将宽视野中间特征按 tile 转成类别 logit 后立即释放，避免保存整幅高维特征。
3. 使用固定形状的 4-window local micro-batch，减少 kernel launch 和 Python dispatch。
4. 为上游 attention 的 batch 诊断路径增加 batch-safe 实现；B=1 仍调用原实现，B>1 使用相同的 QK/softmax/SDPA 运算。
5. 视觉 Linear/Conv 冻结权重采用 BF16 存储，关系矩阵、归一化和结构写回保持稳定精度。
6. 用 Cholesky 因子计算同一个 \(H\) 的作用，不创建 \(N\times N\) 的显式写回矩阵。
7. 对固定形状视觉与投影 kernel 使用 `torch.compile(mode="max-autotune-no-cudagraphs")`，避免 CUDA Graph 跨 tile 覆盖输出。

## 精度与数值证据

全量 mIoU 相对原 Anchored-infra 的变化为：VDD `-0.0031 pp`、Potsdam `+0.0154 pp`、VOC20 `-0.0023 pp`、VOC21 `+0.0080 pp`。smoke 中四个数据集 slot argmax 不一致率均为 `0`；局部特征平均 cosine 为 `0.999920`–`0.999950`。因此该版本的性能收益来自执行层，而不是改变 alias、结构写回或模型目标。

完整成本、屏查、全量 JSON 和复现命令见 [REPORT_ZH.md](REPORT_ZH.md)。
