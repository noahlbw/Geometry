# Geov3 路径诊断（2026-11-01）

固定 Geov3 QuerySlot K=24 + DonorBefore + Anchored；只改变最终读出路径。四个数据集均为全量完成，标签仅用于最后评分。

- `results.json`：每个数据集的 mIoU、Pixel Accuracy、Boundary IoU、Small-object IoU、Large-region IoU 和 Local/Wide 四格转移。
- Small-object：GT 连通区域面积 ≤ 1024 像素；Large-region：≥ 16384 像素。
- Boundary IoU：GT 边界膨胀容差 2，忽略无效标签。
- 转移矩阵行是 Local correct {0,1}，列是 Wide correct {0,1}。

脚本：`artifacts/geometry_github/tools/eval_geov3_path_diagnostics.py`。
