# iSAID 官方 896×896 tile 协议：SegEarth-OV 配置复核（2026-10-30）

本目录记录 SegEarth-OV 按上游 iSAID 配置的完整重跑。协议与已存 VIP/Geov2 tiles16 结果一致：458 张原图转换为 11,644 个 896×896 tiles，重叠 384，16 类（background + 15 类），padding 标签 255 忽略。

## 配置与结果

- 仓库：`https://github.com/likyoo/SegEarth-OV`
- commit：`3e22a969b32c6d751bdbba64a88a0b670e630f55`
- 配置：`configs/cfg_iSAID.py`
- `prob_thd=0.4`；原生 SimFeatUp/JBU、ViT-B/16、canonical iSAID taxonomy
- 全量 tile mIoU（含 background）：**23.4801%**
- cost：7 个固定 tiles、每个 7 次重复，median **290.3425 ms/tile**
- peak allocated **1154.76 MiB**，peak reserved **1242.00 MiB**

## 同协议参考

| 模型 | mIoU (%) | median ms/tile | peak allocated MiB |
|---|---:|---:|---:|
| VIP official tile rerun | 26.1325 | 127.3356 | 1816.76 |
| Geov2 Anchored tile rerun | 27.0134 | 167.5024 | 1702.19 |
| SegEarth-OV official cfg_iSAID | **23.4801** | **290.3425** | **1154.76** |

旧目录中的 SegEarth 22.5194 使用 `cfg_vdd.py` 的 `prob_thd=0.3`，属于 fixed-transfer 诊断，不能冒充 iSAID 官方配置；本次结果高 0.9607 pp。

机器可读证据：`protocol.json`、`merged.json`、`cost.json`、`suite_results.json`。
