# Geov2 宽视野与耦合读出优化：最终取舍（2026-10-15）

局部 DonorBefore 路径冻结。本轮只改变宽视野存储/槽路由和读出执行；所有结果来自 A800，全量 mIoU 与固定成本图均已保存。

## 推荐部署

速度优先使用 `shared_canvas`：共享冻结视觉 backbone 参数，保留原 QuerySlot-24 的 canvas 读出和结构写回。它的输出变化来自共享模型舍入/权重存储，局部 DonorBefore 不变。

显存优先、精度优先使用 `optimized48`：共享完整视觉模型、48 个语义槽、局部 patch 直接读出、gather/scatter 槽构造和 Cholesky 因子写回。它会改变宽视野读出，属于模型候选。

没有一个候选同时在每个数据集上最快且最高；VDD 是 direct/48 槽的速度反例，因此论文主部署应选 `shared_canvas`，`optimized48` 作为精度/显存消融。两个候选都通过 `--wide-readout` 显式选择，默认 `incumbent` 仍用于复现旧结果。

## 全量 mIoU

| 数据集 | QuerySlot-24 | shared_canvas | Δ | optimized48 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| VDD | 59.9785 | 59.9025 | -0.0759 | 60.0908 | +0.1123 |
| Potsdam | 54.4383 | 54.4527 | +0.0144 | 54.4812 | +0.0429 |
| ADE150 | 31.4850 | 31.4919 | +0.0069 | 31.5409 | +0.0559 |
| VOC20 | 93.2805 | 93.2990 | +0.0186 | 93.3615 | +0.0811 |
| VOC21 | 74.6883 | 74.6516 | -0.0368 | 74.6481 | -0.0402 |
| COCO-Object81 | 49.6207 | 49.6244 | +0.0037 | 49.6506 | +0.0299 |

shared_canvas 与 optimized48 的逐图配对区间均保存在各自 SUMMARY/REPORT；区间覆盖 0 的变化按持平解释。VOC20/VOC21 共用图像集合。

## 成本对照

| 数据集 | 基准 ms / MiB | shared_canvas ms / MiB | optimized48 ms / MiB |
| --- | ---: | ---: | ---: |
| VDD | 314.16 / 2643 | 305.01 / 2064 | 426.03 / 1936 |
| Potsdam | 179.27 / 2143 | 173.82 / 1565 | 165.15 / 1498 |
| ADE150 | 175.29 / 3448 | 183.90 / 2872 | 171.48 / 2290 |
| VOC20 | 166.42 / 2253 | 161.46 / 1675 | 163.61 / 1555 |
| VOC21 | 162.83 / 2247 | 157.87 / 1669 | 165.26 / 1543 |
| COCO-Object81 | 176.84 / 2722 | 171.13 / 2145 | 167.09 / 1844 |

成本含 RGB tensor 到原尺寸 CPU uint8 标签回传，不含文件解码、初始化、文本编码；固定 7 张图、2 次 warmup、7 次同步重复。现在两个候选的六个协议成本均已完成。

## 实现

1. `FrozenGeometry(shared_visual=False)` 的旧路径保持不变；`OptimizedWideReadout` 只接受 efficient + coupled + DonorBefore/original。
2. `shared_canvas` 只把 VIP observer 的 visual backbone 指向 Geometry 的冻结 backbone，保留独立 wide head 和 QuerySlot canvas；它复用了参数存储，不改变 DonorBefore 局部计算。
3. `optimized48` 共享完整 visual model，wide RGB 只编码一次；全图 raw patch 经过 48 槽 spherical Lloyd memory。每个局部 32×32 raw query 读取槽值，`PatchClassScorer` 直接在 patch 网格上算 class logits。
4. 局部每个 tile 仍由 DonorBefore `ResponseGeometry.project(..., "original")` 产生 L；结构修正仍是 `Z=L+g0 H(W-L)`。只把 H 存成 Cholesky 因子，用 `D - solve(RRᵀ,D)` 得到 HD，不形成 N×N 中间写回矩阵。
5. gather/scatter 槽构造与 incumbent 的 assignment 最大绝对误差小于 1e−6、argmax 相同；因子化 H action 与原实现误差约 1e−14；remote smoke 检查 local_features_equal=true。

## 已排除的方案

原生语义头在 Potsdam/VDD 开发样本明显退化；336 宽视野分辨率损害遥感；低正则 regression 在 VOC20 退化；48 槽提升有限但 VDD direct 读出明显变慢。shared_visual_canvas 进一步共享完整 head，但受执行层竞争影响，VDD 成本约 438 ms，暂不作为默认。

## 历史参考

旧 Efficient：VOC21 75.8586%、COCO-Object81 50.3999%、VDD 60.4352%，约 153–187 ms、2179–2620 MiB（依数据集）。当前候选的优势主要是运行显存；不把 shared_canvas/optimized48 的小幅 mIoU 波动写成全面 SOTA。

## 文件

- [精度优先报告](OPTIMIZED48_REPORT_ZH.md)
- [速度优先报告](SHAREDCANVAS_REPORT_ZH.md)
- [机器可读精度优先汇总](SUMMARY.json)
- [实现源码快照](source)
- [槽构造 microbenchmark](assignment_benchmark.json)
