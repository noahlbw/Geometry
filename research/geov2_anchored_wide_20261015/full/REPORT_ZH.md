# 空间锚定宽视野：完整验证集实验

状态：**complete**。已完成7/7个数据集。

局部DonorBefore、Alias、类别计分和背景校准沿用冻结配置；本轮只改变宽视野读取。固定24槽、聚类5次、温度0.1、残差系数0.5。Coordinate/Legacy/Anchored使用相同步长搜索预算。

这里按预声明ID核对唯一覆盖，逐图TP/predicted/target与分片混淆矩阵一致；mIoU由累计计数重新计算，未平均分片mIoU。缺失或未完成的数据集不填入精度数字。

## 相同参考g0

| 配置 | VDD | Potsdam | UDD5 | OEM | Vaihingen | ADE150 | VOC21 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g0 | 1 | 0.5 | 0.5 | 0.25 | 0.25 | 0.25 | 1 |
| Coordinate | 60.1696 | 54.8613 | 55.5989 | 44.7603 | 59.1831 | 31.4642 | 74.4098 |
| Legacy | 59.9018 | 54.4523 | 55.2819 | 44.7328 | 58.8200 | 31.4922 | 75.1860 |
| Anchored | 60.5318 | 54.9576 | 55.8063 | 44.7973 | 59.2232 | 31.5649 | 74.9766 |
| NoPrior | 59.9973 | 54.8595 | 55.7212 | 44.7436 | 59.1955 | 31.4769 | 74.6310 |
| NoReference | 60.4049 | 55.2748 | 55.8452 | 44.8022 | 59.4011 | 31.5982 | 74.6618 |
| AbsoluteMix | 60.0474 | 54.4721 | 55.3313 | 44.7540 | 58.8827 | 31.5239 | 75.2359 |
| Shuffled | 56.5999 | 52.0580 | 54.2724 | 44.6455 | 58.2828 | 30.1348 | 64.4297 |
| 最佳普通加权融合 | 59.6304 | 54.9757 | 55.3790 | 44.6972 | 59.1166 | 31.3839 | 74.2921 |

普通融合若提供多个alpha，以上一行显示其独立最优alpha；不是只比较等权融合。

## 配对净收益

各项为Anchored−对照，单位pp。区间覆盖0时不称为稳定提升。Potsdam/Vaihingen按原航片、OEM按地理区域分组，其余按图像重采样。

| 数据集 | 对照 | Anchored−对照及95%区间 | 重采样单元 |
| --- | --- | ---: | --- |
| VDD | Coordinate | +0.3622 [+0.2412, +0.4832] | image (80) |
| VDD | Legacy | +0.6300 [-0.0801, +1.4058] | image (80) |
| VDD | NoPrior | +0.5345 [+0.0697, +1.1890] | image (80) |
| VDD | NoReference | +0.1269 [-0.1272, +0.3492] | image (80) |
| VDD | AbsoluteMix | +0.4843 [-0.1514, +1.2143] | image (80) |
| VDD | Shuffled | +3.9318 [+2.5531, +5.7092] | image (80) |
| VDD | UniformBlend | +0.9014 [+0.6559, +1.1604] | image (80) |
| Potsdam | Coordinate | +0.0962 [+0.0531, +0.1432] | original aerial scene (14) |
| Potsdam | Legacy | +0.5053 [+0.3124, +0.7233] | original aerial scene (14) |
| Potsdam | NoPrior | +0.0981 [+0.0438, +0.1442] | original aerial scene (14) |
| Potsdam | NoReference | -0.3173 [-0.4002, -0.2483] | original aerial scene (14) |
| Potsdam | AbsoluteMix | +0.4854 [+0.3051, +0.6871] | original aerial scene (14) |
| Potsdam | Shuffled | +2.8995 [+2.4726, +3.3778] | original aerial scene (14) |
| Potsdam | UniformBlend | -0.0181 [-0.5156, +0.5586] | original aerial scene (14) |
| UDD5 | Coordinate | +0.2074 [+0.1408, +0.2826] | image (40) |
| UDD5 | Legacy | +0.5243 [+0.1583, +0.9410] | image (40) |
| UDD5 | NoPrior | +0.0850 [+0.0248, +0.1424] | image (40) |
| UDD5 | NoReference | -0.0389 [-0.1727, +0.0781] | image (40) |
| UDD5 | AbsoluteMix | +0.4750 [+0.1398, +0.8595] | image (40) |
| UDD5 | Shuffled | +1.5338 [+0.6262, +2.6415] | image (40) |
| UDD5 | UniformBlend | +0.4273 [+0.1420, +0.6650] | image (40) |
| OEM | Coordinate | +0.0370 [+0.0208, +0.0514] | geographic region (75) |
| OEM | Legacy | +0.0645 [+0.0055, +0.1215] | geographic region (75) |
| OEM | NoPrior | +0.0537 [+0.0235, +0.0959] | geographic region (75) |
| OEM | NoReference | -0.0049 [-0.0258, +0.0164] | geographic region (75) |
| OEM | AbsoluteMix | +0.0432 [-0.0083, +0.0953] | geographic region (75) |
| OEM | Shuffled | +0.1518 [+0.0829, +0.2248] | geographic region (75) |
| OEM | UniformBlend | +0.1001 [+0.0638, +0.1373] | geographic region (75) |
| Vaihingen | Coordinate | +0.0401 [-0.0056, +0.0792] | original aerial scene (17) |
| Vaihingen | Legacy | +0.4032 [+0.2351, +0.5448] | original aerial scene (17) |
| Vaihingen | NoPrior | +0.0277 [-0.0133, +0.0740] | original aerial scene (17) |
| Vaihingen | NoReference | -0.1779 [-0.2414, -0.1100] | original aerial scene (17) |
| Vaihingen | AbsoluteMix | +0.3405 [+0.1831, +0.4678] | original aerial scene (17) |
| Vaihingen | Shuffled | +0.9404 [+0.7286, +1.1850] | original aerial scene (17) |
| Vaihingen | UniformBlend | +0.1066 [+0.0408, +0.1744] | original aerial scene (17) |
| ADE150 | Coordinate | +0.1007 [+0.0839, +0.1241] | image (2000) |
| ADE150 | Legacy | +0.0727 [-0.0252, +0.1560] | image (2000) |
| ADE150 | NoPrior | +0.0881 [+0.0648, +0.1091] | image (2000) |
| ADE150 | NoReference | -0.0333 [-0.0612, +0.0060] | image (2000) |
| ADE150 | AbsoluteMix | +0.0410 [-0.0482, +0.1207] | image (2000) |
| ADE150 | Shuffled | +1.4301 [+1.1896, +1.6279] | image (2000) |
| ADE150 | UniformBlend | +0.1810 [+0.1411, +0.2239] | image (2000) |
| VOC21 | Coordinate | +0.5668 [+0.4938, +0.6385] | image (1449) |
| VOC21 | Legacy | -0.2094 [-0.5991, +0.1462] | image (1449) |
| VOC21 | NoPrior | +0.3456 [+0.2632, +0.4342] | image (1449) |
| VOC21 | NoReference | +0.3148 [+0.1892, +0.4383] | image (1449) |
| VOC21 | AbsoluteMix | -0.2593 [-0.6079, +0.0666] | image (1449) |
| VOC21 | Shuffled | +10.5469 [+9.8207, +11.3471] | image (1449) |
| VOC21 | UniformBlend | +0.6845 [+0.5996, +0.7808] | image (1449) |

## 相同预算各自重选g0

| 数据集 | 方法 | 最优g0或alpha | mIoU | Anchored最佳−对照最佳及95%区间 |
| --- | --- | ---: | ---: | ---: |
| VDD | Coordinate | 1.0 | 60.1696 | +0.3622 [+0.2412, +0.4832] |
| VDD | Legacy | 1.0 | 59.9018 | +0.6300 [-0.0801, +1.4058] |
| VDD | Anchored | 1.0 | 60.5318 | +0.0000 [+0.0000, +0.0000] |
| VDD | UniformBlend | 0.5 | 59.6304 | +0.9014 [+0.6559, +1.1604] |
| Potsdam | Coordinate | 0.75 | 55.2080 | +0.2812 [+0.2019, +0.3549] |
| Potsdam | Legacy | 0.75 | 54.4849 | +1.0043 [+0.7413, +1.3182] |
| Potsdam | Anchored | 0.75 | 55.4893 | +0.0000 [+0.0000, +0.0000] |
| Potsdam | UniformBlend | 0.375 | 54.9757 | +0.5136 [+0.4224, +0.5816] |
| UDD5 | Coordinate | 0.5 | 55.5989 | +0.2074 [+0.1408, +0.2826] |
| UDD5 | Legacy | 0.5 | 55.2819 | +0.5243 [+0.1583, +0.9410] |
| UDD5 | Anchored | 0.5 | 55.8063 | +0.0000 [+0.0000, +0.0000] |
| UDD5 | UniformBlend | 0.25 | 55.3790 | +0.4273 [+0.1420, +0.6650] |
| OEM | Coordinate | 0.25 | 44.7603 | +0.0370 [+0.0208, +0.0514] |
| OEM | Legacy | 0.25 | 44.7328 | +0.0645 [+0.0055, +0.1215] |
| OEM | Anchored | 0.25 | 44.7973 | +0.0000 [+0.0000, +0.0000] |
| OEM | UniformBlend | 0.125 | 44.6972 | +0.1001 [+0.0638, +0.1373] |
| Vaihingen | Coordinate | 0.25 | 59.1831 | +0.0401 [-0.0056, +0.0792] |
| Vaihingen | Legacy | 0.25 | 58.8200 | +0.4032 [+0.2351, +0.5448] |
| Vaihingen | Anchored | 0.25 | 59.2232 | +0.0000 [+0.0000, +0.0000] |
| Vaihingen | UniformBlend | 0.125 | 59.1166 | +0.1066 [+0.0408, +0.1744] |
| ADE150 | Coordinate | 0.25 | 31.4642 | +0.1007 [+0.0839, +0.1241] |
| ADE150 | Legacy | 0.25 | 31.4922 | +0.0727 [-0.0252, +0.1560] |
| ADE150 | Anchored | 0.25 | 31.5649 | +0.0000 [+0.0000, +0.0000] |
| ADE150 | UniformBlend | 0.125 | 31.3839 | +0.1810 [+0.1411, +0.2239] |
| VOC21 | Coordinate | 1.0 | 74.4098 | +0.5668 [+0.4938, +0.6385] |
| VOC21 | Legacy | 1.0 | 75.1860 | -0.2094 [-0.5991, +0.1462] |
| VOC21 | Anchored | 1.0 | 74.9766 | +0.0000 [+0.0000, +0.0000] |
| VOC21 | UniformBlend | 0.5 | 74.2921 | +0.6845 [+0.5996, +0.7808] |

## 已完成域等权平均

| 对照 | 平均净收益pp | 95%区间 | 正/负点估计域数 |
| --- | ---: | --- | ---: |
| AbsoluteMix | +0.2300 | [+0.1063, +0.3643] | 6/1 |
| Coordinate | +0.2015 | [+0.1765, +0.2262] | 7/0 |
| Legacy | +0.2844 | [+0.1515, +0.4275] | 6/1 |
| NoPrior | +0.1761 | [+0.1055, +0.2731] | 7/0 |
| NoReference | -0.0187 | [-0.0662, +0.0253] | 2/5 |
| Shuffled | +3.0621 | [+2.7874, +3.3905] | 7/0 |
| UniformBlend | +0.3404 | [+0.2495, +0.4383] | 6/1 |

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
