# QuerySlot＋DonorBefore：已有最佳 alias 的全量结果

2026-10-13。全部 15 数据集、16 协议完成全量 mIoU 与独立成本测量。词表采用之前已完成带标签开发搜索的 retained 最佳版本；本轮没有再按新全量成绩挑选词表。

为使 DonorBefore 在所有数据集实际执行，局部 strength 统一为 original。其中原先使用数值 strength 的协议同时发生局部路径变化；下表相对旧版的差值是整体配置差值，不能仅归因于宽视野替换。QuerySlot 固定 24 槽、beta=0.5、温度=0.1；PC60 保留 401 个残余概念。其余 g、词聚合、背景校准和视野策略沿用各自旧最佳配置。

| 协议 | 图像 | 旧 Efficient mIoU % | QuerySlot＋DonorBefore % | Δpp | 耗时 ms | 峰值 allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| vdd | 80 | 60.4352 | 59.9810 | -0.4542 | 295.89 | 2643.89 |
| potsdam | 504 | 56.1934 | 54.4390 | -1.7544 | 152.01 | 2143.30 |
| voc20 | 1449 | 92.7936 | 93.2804 | +0.4868 | 143.72 | 2253.26 |
| voc21 | 1449 | 75.8586 | 74.6882 | -1.1704 | 139.99 | 2247.24 |
| ade150 | 2000 | 31.3398 | 31.4852 | +0.1454 | 169.33 | 3448.46 |
| context60 | 5105 | 44.7613 | 44.5927 | -0.1686 | 152.06 | 2622.49 |
| udd5 | 40 | 55.8266 | 55.2575 | -0.5691 | 291.99 | 2276.79 |
| oem | 384 | 45.1052 | 44.7355 | -0.3697 | 150.22 | 2170.24 |
| vaihingen | 113 | 58.8562 | 58.8021 | -0.0541 | 153.20 | 2153.60 |
| landcoverai | 1602 | 65.5400 | 68.9235 | +3.3835 | 144.91 | 2159.30 |
| loveda/P | 1669 | 69.6117 | 70.0386 | +0.4269 | 147.32 | 2187.57 |
| loveda/D | 1669 | 45.3171 | 45.8370 | +0.5199 | 147.83 | 2187.57 |
| context59 | 5105 | 46.3075 | 46.7853 | +0.4778 | 146.22 | 2440.32 |
| coco_object81 | 5000 | 50.3999 | 49.6209 | -0.7790 | 151.16 | 2722.19 |
| coco_stuff171 | 5000 | 33.7661 | 33.4209 | -0.3452 | 173.78 | 3324.08 |
| flair1 | 15700 | 43.6579 | 44.1187 | +0.4608 | 143.36 | 2183.50 |

## 成本边界

A800，batch=1。每个数据集使用旧成本清单相同的 7 张固定完整图；每图 2 次预热、7 次同步重复。报告所用成本记录在本轮全量任务不运行时逐项独立测量。耗时为这些重复的中位数；显存为 CUDA max_memory_allocated 的峰值。包含 CPU RGB 缩放、全部编码、读出、结构求解、原尺寸恢复和 CPU uint8 输出；排除图像解码、初始化、离线文本编码和标签读取。文本塔释放，两个视觉模型保留原数值角色并常驻。reserved 原始值另存在 CSV/JSON 中。

与旧 Efficient 的耗时对照属于跨轮测量；运行状态可能影响绝对速度。LoveDA P/D 是独立读出，各自预测和计时。完整覆盖为 45,200 次数据集图像评测；含附加 LoveDA P 后为 46,869 次协议图像评测，来源之间存在共享图像。

FLAIR 的初次成本与首批全量任务有约 30 秒重叠，原记录保留在 cost/flair1.json。本报告仅使用全量结束后 cost_recheck/cost/flair1.json 的独立重测。

## alias 与局部路径

| 数据集 | alias bank | alias 总数 | 每类范围 | 旧 strength | 新 strength | 残余概念 |
|---|---|---:|---:|---|---|---:|
| vdd | official_imagenet | 9 | 1–2 | original | original | 0 |
| potsdam | official_imagenet | 11 | 1–4 | 3.0 | original | 0 |
| voc20 | semantic_imagenet | 107 | 3–18 | 2.0 | original | 0 |
| voc21 | official_segmentation | 119 | 1–56 | 1.0 | original | 0 |
| ade150 | semantic_segmentation | 433 | 1–18 | original | original | 0 |
| context60 | mix_f6f525631e7f743d226c | 264 | 1–21 | 2.0 | original | 401 |
| udd5 | kev60_segmentation | 31 | 1–20 | original | original | 0 |
| oem | original | 160 | 20–20 | 1.0 | original | 0 |
| vaihingen | kev40_imagenet | 69 | 8–17 | original | original | 0 |
| landcoverai | original_imagenet | 100 | 20–20 | 2.0 | original | 0 |
| loveda | original | 140 | 20–20 | original | original | 0 |
| context59 | official_segmentation | 144 | 1–16 | 3.0 | original | 0 |
| coco_object81 | official_imagenet | 156 | 1–32 | 1.0 | original | 0 |
| coco_stuff171 | official_segmentation | 409 | 1–15 | 1.0 | original | 0 |
| flair1 | original | 240 | 20–20 | original | original | 0 |

## 数据与实现

所有合并记录核验唯一完整图像覆盖、各分片配置一致、标注像素计数与既有协议一致及逐图统计之和。推理本身不读取标签；标签仅由外层评测脚本计算混淆矩阵。

- 汇总：RESULTS.csv、RESULTS.json；词表：ALIAS_MANIFEST.csv、vocabularies/。
- 冻结配置：protocol.json、configs/；独立成本：cost/；全量指标：merged/。
- 逐图 TP/target/predicted 充分统计：full/*.npz。
- A800 独立源码与结果：results/query_slot_best_alias_20261013/。
- 执行入口：DINOtool/scripts/eval_query_slot_best_alias.py、run_query_slot_best_alias.py。
- 本实验独立保存；2026-10-09 论文主表继续使用其原冻结结果。
