# Geov2 定性对比：UDD5 与 VOC21（2026-10-14）

本页保留 VOC21 的四个自然图像样例，并用 UDD5 替换之前的 VDD 遥感样例。每个数据集从 16 张候选图中自动选择四张，评分规则为：

```text
Geov2 单图 mIoU − max(VIP wide, PEARL, PTC)
```

所有方法使用同一 RGB、同一 ground truth 和同一类别调色板。每行列顺序为 RGB、Ground truth、VIP wide、PEARL、PTC、Geov2。`VIP wide` 是 Geov2 内部的宽视野端点，未经过 Geov2 局部结构写回；PEARL 和 PTC 是冻结的原生 training-free 基线。

## 遥感：UDD5

UDD5 的四张图分别领先最强对照 **+28.0、+21.5、+14.8、+12.3pp**。

| 图像 | VIP wide | PEARL | PTC | Geov2 | Geov2 margin |
|---|---:|---:|---:|---:|---:|
| `000161.JPG` | 16.3 | 35.6 | 18.4 | **63.7** | **+28.0** |
| `001461.JPG` | 5.6 | 30.9 | 9.5 | **52.4** | **+21.5** |
| `DJI_0303.JPG` | 24.9 | 37.6 | 39.6 | **54.3** | **+14.8** |
| `DJI_0511.JPG` | 15.0 | 38.2 | 35.2 | **50.5** | **+12.3** |

![UDD5 strong comparison](qualitative_20261014/udd5/strong_comparison_contact_sheet.png)

## 自然图像：VOC21

以下四张保持不变，包括用户指定的 `2007_000862`。

| 图像 | VIP wide | PEARL | PTC | Geov2 | Geov2 margin |
|---|---:|---:|---:|---:|---:|
| `2008_008268` | 46.3 | 50.4 | 46.3 | **82.5** | **+32.0** |
| `2010_005606` | 70.2 | 51.3 | 63.2 | **88.4** | **+18.2** |
| `2007_000862` | 37.9 | 33.1 | 33.1 | **53.3** | **+15.3** |
| `2008_007828` | 72.0 | 47.8 | 35.6 | **80.3** | **+8.3** |

![VOC21 strong comparison](qualitative_20261014/voc21/strong_comparison_contact_sheet.png)

## 遥感候选集搜索记录

同时检查了 UDD5、Potsdam 和 Vaihingen，各自使用 16 张候选图。UDD5 被选为展示集；三套数据的前四名及精确数值见 [`remote_dataset_search.json`](qualitative_20261014/remote_dataset_search.json)。

![Selection score summary](qualitative_20261014/strong_selection_summary.png)

逐图 JSON：[`UDD5`](qualitative_20261014/udd5/strong_cases.json)、[`VOC21`](qualitative_20261014/voc21/strong_cases.json)。这些图用于定性展示，完整数据集结论仍以全量 mIoU 表为准。
