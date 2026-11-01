# Geov3 结构写回算子消融（2026-11-01）

四个数据集均为全量结果，固定 Geov3 QuerySlot K=24、DonorBefore、Anchored、相同 alias bank 和数据集 g0。

- LocalOnly：不写回。
- DirectFusion：直接使用 `L + g0(W-L)`。
- GramPropagation：使用 `L + g0 G^T G(W-L)`。
- SpectralFilter：使用 `L + g0 (I+G^T G)^(-1)G^T G(W-L)`，即 Geov3。

结果见 `full/<dataset>/results.json`，脚本为 `artifacts/geometry_github/tools/eval_geov3_correction_ablation.py`。
