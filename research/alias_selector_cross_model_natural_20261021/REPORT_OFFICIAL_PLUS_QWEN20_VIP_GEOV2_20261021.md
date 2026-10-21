# Official + Qwen20 → VIP VG/SC → Geov2（2026-10-21）

## 实验目的

把 VIP 官方 alias 原样放进候选池，再用同一份 Qwen20 候选补足每个前景类约 20 个候选。先用 VIP 的 VG/SC 规则筛 Qwen 新词，再把官方词强制保留，形成最终 `official ∪ VIP-screened-Qwen` bank；最后固定该 bank，在 Geov2 Anchored 的同一配置下比较 log-sum-exp 与 log-mean-exp。

## 候选池与筛词规则

- 前景：官方 alias 全部保留；不足 20 个时按原 Qwen 顺序补齐；官方超过 20 个的类别全部保留。
- VOC21/COCO-Object81 背景：完整官方背景组固定保留（分别 56/32 词），不让泛背景 Qwen 短语进入筛选。
- VIP 筛选：64 张 VOC、64 张 COCO、96 张 ADE 开发图；不读取 mask，不用 mIoU。候选需满足 text cosine ≥ 0.70、VG 高于 canonical、SC 低于 canonical。
- 最终保留规则：官方 alias union VIP 筛出的 Qwen alias。这样 VIP 官方词不会因 VG/SC 噪声被误删。

候选池清单：`candidate_pools_official_plus_qwen20/manifest.json`。
最终 bank：`vip_official_plus_qwen20_unionofficial/{voc21,coco_object81,ade150}/vip_bank.json`。

## VIP 全量验证（最终 bank）

| 数据集 | alias 总数 | VIP mIoU (%) | 用时 (s) |
|---|---:|---:|---:|
| voc21 | 224 | 67.3114 | 116.76 |
| coco_object81 | 387 | 44.6662 | 578.80 |
| ade150 | 444 | 26.9737 | 259.32 |

对应 JSON：`results_vip_unionofficial/{dataset}.json`。

## Geov2 全量验证（最终 bank，批量读出路径）

固定模型：Anchored、QuerySlot-24、DonorBefore 局部路径、compact+efficient；g0 为 VOC21=1.0、COCO-Object81=0.5、ADE150=0.25。两种聚合读取同一 bank、同一图像和同一配置。

| 数据集 | alias 总数 | log-mean-exp mIoU | log-sum-exp mIoU | Δ(sum−mean) | mean 用时(s) | sum 用时(s) |
|---|---:|---:|---:|---:|---:|---:|
| voc21 | 224 | 51.3139 | 53.1948 | +1.8809 | 243.42 | 245.55 |
| coco_object81 | 387 | 47.2134 | 46.9204 | -0.2930 | 911.92 | 900.04 |
| ade150 | 444 | 28.7718 | 28.7089 | -0.0629 | 391.64 | 391.57 |

结果目录：`results_unionofficial_batch/logmeanexp/` 和 `results_unionofficial_batch/logsumexp/`。

## 结论

- sum/mean 没有跨数据集统一赢家：VOC21 的 sum 高 1.8809 pp；COCO-Object81 的 mean 高 0.2930 pp；ADE150 的 mean 高 0.0629 pp。
- 如果只能选一个固定聚合，当前三集平均上 sum 略高，但 COCO/ADE 更稳定地偏向 mean；论文中应报告逐数据集结果，不要写成某一种聚合普遍更好。
- “官方词能否被 VIP 筛出来”的问题通过候选池和 union 规则解决：官方词全部进入候选空间并在最终 bank 中保留；VIP 只决定新增 Qwen alias。单纯让 VG/SC 删除官方词会导致 VOC21/COCO/ADE 的前景官方 alias 丢失，不能作为最终 bank。
- 本轮 Geov2 的 profile（VOC tau=3、COCO tau=6、ADE tau=5/tem=5、background_rule=retained）已写入每个结果 JSON；不要与历史 tau=1/union_max 的旧快照直接混比。

## 旧的单纯筛词输出（诊断）

为便于复核，`vip_official_plus_qwen20_fixedbg/` 是“官方背景固定、前景允许删除官方词”的原始 VIP 输出；它不作为最终 bank。其 Geov2 结果在 `results_official_plus_qwen20_truefixedbg/`，只用于说明强制保留官方 alias 的必要性。
