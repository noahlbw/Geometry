# 空间锚定宽视野：冒烟检查

状态：**complete**。已完成7/7个数据集。

局部DonorBefore、Alias、类别计分和背景校准沿用冻结配置；本轮只改变宽视野读取。固定24槽、聚类5次、温度0.1、残差系数0.5。Coordinate/Legacy/Anchored使用相同步长搜索预算。

这里按预声明ID核对唯一覆盖，逐图TP/predicted/target与分片混淆矩阵一致；mIoU由累计计数重新计算，未平均分片mIoU。缺失或未完成的数据集不填入精度数字。

**本页不是完整验证集结果，不能替代全量精度结论。**

## 相同参考g0

| 配置 | VDD | Potsdam | UDD5 | OEM | Vaihingen | ADE150 | VOC21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g0 | 1 | 0.5 | 0.5 | 0.25 | 0.25 | 0.25 | 1 |
| Coordinate | 38.0169 | 59.7170 | 64.5999 | 25.9571 | 52.6090 | 23.4127 | 56.2633 |
| Legacy | 36.8328 | 58.8855 | 65.6840 | 26.7271 | 51.4387 | 23.8189 | 53.0504 |
| Anchored | 38.5775 | 60.0463 | 64.7097 | 26.0901 | 52.1653 | 22.0639 | 56.7164 |
| NoPrior | 38.6033 | 59.5756 | 64.6752 | 26.0232 | 52.4637 | 22.1084 | 56.7034 |
| NoReference | 38.7965 | 60.5905 | 64.5270 | 25.8095 | 52.8952 | 21.9202 | 57.0729 |
| AbsoluteMix | 36.6677 | 59.2290 | 65.7009 | 26.7071 | 51.4755 | 23.7878 | 53.5736 |
| Shuffled | 35.5964 | 56.6244 | 64.4030 | 26.4058 | 51.4677 | 23.0388 | 43.4766 |
| 最佳普通加权融合 | 40.5609 | 60.5026 | 64.0044 | 30.9013 | 52.7978 | 53.3102 | 63.8401 |

普通融合若提供多个alpha，以上一行显示其独立最优alpha；不是只比较等权融合。

## 配对净收益

各项为Anchored−对照，单位pp。区间覆盖0时不称为稳定提升。Potsdam/Vaihingen按原航片、OEM按地理区域分组，其余按图像重采样。

| 数据集 | 对照 | Anchored−对照及95%区间 | 重采样单元 |
| --- | --- | ---: | --- |
| VDD | Coordinate | +0.5606 [+0.5606, +0.5606] | image (1) |
| VDD | Legacy | +1.7447 [+1.7447, +1.7447] | image (1) |
| VDD | NoPrior | -0.0258 [-0.0258, -0.0258] | image (1) |
| VDD | NoReference | -0.2190 [-0.2190, -0.2190] | image (1) |
| VDD | AbsoluteMix | +1.9097 [+1.9097, +1.9097] | image (1) |
| VDD | Shuffled | +2.9811 [+2.9811, +2.9811] | image (1) |
| VDD | UniformBlend | -1.9835 [-1.9835, -1.9835] | image (1) |
| Potsdam | Coordinate | +0.3293 [+0.3293, +0.3293] | original aerial scene (1) |
| Potsdam | Legacy | +1.1608 [+1.1608, +1.1608] | original aerial scene (1) |
| Potsdam | NoPrior | +0.4707 [+0.4707, +0.4707] | original aerial scene (1) |
| Potsdam | NoReference | -0.5442 [-0.5442, -0.5442] | original aerial scene (1) |
| Potsdam | AbsoluteMix | +0.8173 [+0.8173, +0.8173] | original aerial scene (1) |
| Potsdam | Shuffled | +3.4219 [+3.4219, +3.4219] | original aerial scene (1) |
| Potsdam | UniformBlend | -0.4563 [-0.4563, -0.4563] | original aerial scene (1) |
| UDD5 | Coordinate | +0.1098 [+0.1098, +0.1098] | image (1) |
| UDD5 | Legacy | -0.9743 [-0.9743, -0.9743] | image (1) |
| UDD5 | NoPrior | +0.0345 [+0.0345, +0.0345] | image (1) |
| UDD5 | NoReference | +0.1827 [+0.1827, +0.1827] | image (1) |
| UDD5 | AbsoluteMix | -0.9912 [-0.9912, -0.9912] | image (1) |
| UDD5 | Shuffled | +0.3067 [+0.3067, +0.3067] | image (1) |
| UDD5 | UniformBlend | +0.7053 [+0.7053, +0.7053] | image (1) |
| OEM | Coordinate | +0.1329 [+0.1329, +0.1329] | geographic region (1) |
| OEM | Legacy | -0.6371 [-0.6371, -0.6371] | geographic region (1) |
| OEM | NoPrior | +0.0668 [+0.0668, +0.0668] | geographic region (1) |
| OEM | NoReference | +0.2805 [+0.2805, +0.2805] | geographic region (1) |
| OEM | AbsoluteMix | -0.6171 [-0.6171, -0.6171] | geographic region (1) |
| OEM | Shuffled | -0.3158 [-0.3158, -0.3158] | geographic region (1) |
| OEM | UniformBlend | -4.8112 [-4.8112, -4.8112] | geographic region (1) |
| Vaihingen | Coordinate | -0.4437 [-0.4437, -0.4437] | original aerial scene (1) |
| Vaihingen | Legacy | +0.7266 [+0.7266, +0.7266] | original aerial scene (1) |
| Vaihingen | NoPrior | -0.2984 [-0.2984, -0.2984] | original aerial scene (1) |
| Vaihingen | NoReference | -0.7299 [-0.7299, -0.7299] | original aerial scene (1) |
| Vaihingen | AbsoluteMix | +0.6899 [+0.6899, +0.6899] | original aerial scene (1) |
| Vaihingen | Shuffled | +0.6976 [+0.6976, +0.6976] | original aerial scene (1) |
| Vaihingen | UniformBlend | -0.6325 [-0.6325, -0.6325] | original aerial scene (1) |
| ADE150 | Coordinate | -1.3488 [-1.3488, -1.3488] | image (1) |
| ADE150 | Legacy | -1.7550 [-1.7550, -1.7550] | image (1) |
| ADE150 | NoPrior | -0.0445 [-0.0445, -0.0445] | image (1) |
| ADE150 | NoReference | +0.1438 [+0.1438, +0.1438] | image (1) |
| ADE150 | AbsoluteMix | -1.7239 [-1.7239, -1.7239] | image (1) |
| ADE150 | Shuffled | -0.9749 [-0.9749, -0.9749] | image (1) |
| ADE150 | UniformBlend | -31.2463 [-31.2463, -31.2463] | image (1) |
| VOC21 | Coordinate | +0.4531 [+0.4531, +0.4531] | image (1) |
| VOC21 | Legacy | +3.6660 [+3.6660, +3.6660] | image (1) |
| VOC21 | NoPrior | +0.0130 [+0.0130, +0.0130] | image (1) |
| VOC21 | NoReference | -0.3564 [-0.3564, -0.3564] | image (1) |
| VOC21 | AbsoluteMix | +3.1428 [+3.1428, +3.1428] | image (1) |
| VOC21 | Shuffled | +13.2398 [+13.2398, +13.2398] | image (1) |
| VOC21 | UniformBlend | -7.1237 [-7.1237, -7.1237] | image (1) |

## 相同预算各自重选g0

| 数据集 | 方法 | 最优g0或alpha | mIoU | Anchored最佳−对照最佳及95%区间 |
| --- | --- | ---: | ---: | ---: |
| VDD | Coordinate | 0.25 | 42.2133 | +0.1294 [+0.1294, +0.1294] |
| VDD | Legacy | 0.25 | 42.2318 | +0.1108 [+0.1108, +0.1108] |
| VDD | Anchored | 0.25 | 42.3426 | +0.0000 [+0.0000, +0.0000] |
| VDD | UniformBlend | 0.125 | 40.5609 | +1.7817 [+1.7817, +1.7817] |
| Potsdam | Coordinate | 0.5 | 59.7170 | +0.3293 [+0.3293, +0.3293] |
| Potsdam | Legacy | 0.5 | 58.8855 | +1.1608 [+1.1608, +1.1608] |
| Potsdam | Anchored | 0.5 | 60.0463 | +0.0000 [+0.0000, +0.0000] |
| Potsdam | UniformBlend | 0.25 | 60.5026 | -0.4563 [-0.4563, -0.4563] |
| UDD5 | Coordinate | 0.5 | 64.5999 | +0.1098 [+0.1098, +0.1098] |
| UDD5 | Legacy | 0.5 | 65.6840 | -0.9743 [-0.9743, -0.9743] |
| UDD5 | Anchored | 0.5 | 64.7097 | +0.0000 [+0.0000, +0.0000] |
| UDD5 | UniformBlend | 0.25 | 64.0044 | +0.7053 [+0.7053, +0.7053] |
| OEM | Coordinate | 1.0 | 30.7491 | -0.0485 [-0.0485, -0.0485] |
| OEM | Legacy | 0.75 | 30.6102 | +0.0904 [+0.0904, +0.0904] |
| OEM | Anchored | 0.75 | 30.7006 | +0.0000 [+0.0000, +0.0000] |
| OEM | UniformBlend | 0.375 | 30.9013 | -0.2007 [-0.2007, -0.2007] |
| Vaihingen | Coordinate | 0.5 | 52.8274 | +0.1074 [+0.1074, +0.1074] |
| Vaihingen | Legacy | 0.75 | 52.2650 | +0.6699 [+0.6699, +0.6699] |
| Vaihingen | Anchored | 1.0 | 52.9348 | +0.0000 [+0.0000, +0.0000] |
| Vaihingen | UniformBlend | 0.25 | 52.7978 | +0.1370 [+0.1370, +0.1370] |
| ADE150 | Coordinate | 1.0 | 54.0296 | -5.2329 [-5.2329, -5.2329] |
| ADE150 | Legacy | 1.0 | 43.2408 | +5.5558 [+5.5558, +5.5558] |
| ADE150 | Anchored | 1.0 | 48.7966 | +0.0000 [+0.0000, +0.0000] |
| ADE150 | UniformBlend | 1.0 | 53.3102 | -4.5136 [-4.5136, -4.5136] |
| VOC21 | Coordinate | 1.0 | 56.2633 | +0.4531 [+0.4531, +0.4531] |
| VOC21 | Legacy | 1.0 | 53.0504 | +3.6660 [+3.6660, +3.6660] |
| VOC21 | Anchored | 1.0 | 56.7164 | +0.0000 [+0.0000, +0.0000] |
| VOC21 | UniformBlend | 1.0 | 63.8401 | -7.1237 [-7.1237, -7.1237] |

## 已完成域等权平均

| 对照 | 平均净收益pp | 95%区间 | 正/负点估计域数 |
| --- | ---: | --- | ---: |
| AbsoluteMix | +0.4611 | [+0.4611, +0.4611] | 4/3 |
| Coordinate | -0.0295 | [-0.0295, -0.0295] | 5/2 |
| Legacy | +0.5617 | [+0.5617, +0.5617] | 4/3 |
| NoPrior | +0.0309 | [+0.0309, +0.0309] | 4/3 |
| NoReference | -0.1775 | [-0.1775, -0.1775] | 3/4 |
| Shuffled | +2.7652 | [+2.7652, +2.7652] | 5/2 |
| UniformBlend | -6.5069 | [-6.5069, -6.5069] | 1/6 |

上述平均只覆盖已完成域；若阶段未完成，不能称全体数据集平均。

## 解释与边界

- Coordinate：保留24槽语义头，使用坐标读出；仍有局部和宽视野两路以及结构写回。
- Legacy：已有局部raw-query槽读取及绝对特征混合。
- Anchored：空间槽归属先验＋相对同位置宽视野raw的证据更新＋槽混合残差。
- NoPrior、NoReference、AbsoluteMix分别检查先验、相对参照和残差减项；Shuffled破坏query与输出位置对应。打乱下降不自动证明相对Coordinate有收益。
- 参考g0与独立最优g0均保留。最优步长用开发标签选取；推理不读取标签。配对bootstrap条件于选定参数，不含完整的参数选择不确定性或多重比较校正。
- 历史Alias、背景校准和验证集查看仍有开发偏差。五折标量选择结果保存在SUMMARY中，仅隔离本轮标量选择。
- 所有负结果保留；本轮对照不能单独证明新方法优于LPOSS等未测方法。

完整配置见[ALL_RESULTS.csv](ALL_RESULTS.csv)，配对区间、覆盖审计与五折分析见[SUMMARY.json](SUMMARY.json)。

协议记录的限制：

Previously inspected validation; historical aliases/calibration used development labels. Full-set scalar selection is development analysis. Group bootstrap is conditional on selected parameters; no untouched-test claim. P spatial mapping matches incumbent sampler, including its padding behavior. Same head/scoring/normalization for all controls. Only local raw query is shuffled by a half-sequence roll, not a multiseed random control. No new adaptive gate.
