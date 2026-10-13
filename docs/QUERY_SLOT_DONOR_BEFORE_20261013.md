# Geometry QuerySlot＋DonorBefore：模型与复现

本页对应2026-10-13完成的15数据集／16协议实验。完整结果见[报告](../research/query_slot_best_alias_20261013/REPORT.md)、[CSV](../research/query_slot_best_alias_20261013/RESULTS.csv)和[JSON](../research/query_slot_best_alias_20261013/RESULTS.json)。本版相对旧Efficient有7项mIoU提升、9项下降，适合作为速度与精度取舍的实验配置；没有达到旧VOC21的75.8586%和VDD的60.4352%。

## 完整计算路径

1. **局部DonorBefore。** 原尺寸RGB按既有Geometry规则缩放、切成512窗口，重用冻结DINO特征与视觉关系A。在两层语义头中，先计算保留原生特殊token贡献与patch注意力质量的self-conditional增量u，再沿A运输patch增量，最后经过当前query处的冻结MLP。patch路径为`x' = x + A u_self; y = x' + f(x')`，prefix仍走原生路径。关系估计、窗口重建和结构写回保持既有定义。
2. **宽视野QuerySlot。** 整图按协议的宽视野尺寸编码一次。原始patch描述子通过确定性farthest-point初始化、5轮球面Lloyd迭代及温度0.1的soft assignment形成最多24个槽；`center=False`、`hard=False`。冻结语义头通过槽进行因子化读取。每个槽保留归一化的原始描述子中心与语义value原型；每个已计算的局部raw token以温度0.1选择槽。输出为坐标采样的整图语义特征与槽读出的等权插值，再归一化：`W_feature = normalize(0.5 B_coordinate + 0.5 softmax(q C^T / 0.1) V)`。没有新增训练参数或RGB编码器。
3. **文本读出与结构修正。** 局部和宽视野分别使用其冻结的alias聚合规则得到L、W，随后继续使用`Z = L + g H(W-L)`。各协议的g、背景bias／阈值、词聚合和视野策略来自此前保留配置。窗口概率通过Hann权重重建，恢复原尺寸并输出CPU uint8类别ID。LoveDA P/D各自独立读出。

入口为[`FrozenGeometry`](../DINOtool/dinotool/frozen_deployment.py)。主要实现包括[QuerySlot](../DINOtool/dinotool/query_slot_memory.py)、[槽语义观测](../DINOtool/dinotool/context_memory_observer.py)、[DonorBefore封装](../DINOtool/dinotool/response_geometry_deployment.py)、[语义头顺序](../DINOtool/dinotool/geometry_semantic_response.py)及[Efficient执行](../DINOtool/dinotool/efficient_deployment.py)。保留局部FP32与宽分支FP16两套冻结视觉权重的原数值角色；在线推理释放文本塔。

## alias、配置与选择范围

每类alias及模板在[`vocabularies/`](../research/query_slot_best_alias_20261013/vocabularies)，完整模型参数在[`configs/`](../research/query_slot_best_alias_20261013/configs)。[alias清单](../research/query_slot_best_alias_20261013/ALIAS_MANIFEST.csv)列出来源、数量和局部strength。PC60保留401个残余概念。

这里的“最佳alias”是此前带标签开发搜索保留的词表；本轮没有针对QuerySlot重新搜索，也不声称词表在该新结构下全局最优。单图推理不读取标签；标签仅由外层评测计算混淆矩阵。

为保证DonorBefore在所有协议实际生效，本轮将局部`strength`统一为`original`。原数值strength路径会绕过DonorBefore；因此相对旧Efficient的差值包含局部路径变化。默认CLI仍选择旧VIP宽分支，运行本版必须显式传入`--wide-branch query_slot --query-slot-capacity 24 --efficient`并使用本轮配置。

## 安装与单图推理

环境、两个固定上游版本、checkpoint文件名及安装命令沿用[部署说明](DEPLOYMENT_20261010.md#环境与上游)。在仓库根目录设置：

```bash
export PYTHONPATH="$PWD/DINOtool:$PWD/DINOtool/scripts:$PWD/third_party/DINO_Soars"
python DINOtool/scripts/infer_frozen_geometry.py \
  --config research/query_slot_best_alias_20261013/configs/voc21.yaml \
  --checkpoint-dir /path/to/checkpoints \
  --dinov3-repo DINOtool/dinov3_hub \
  --upstream-root third_party/VIP \
  --text-cache /path/to/cache/voc21.pt \
  --wide-branch query_slot --query-slot-capacity 24 --efficient \
  --input /path/to/image.jpg --output /path/to/new_prediction.png
```

输出元数据包含`wide_branch=query_slot`和`local_branch=donor_before`。首次运行从公开alias编码文本并写缓存，之后复用；缓存身份包含词表、checkpoint清单和VIP版本。历史实验使用既有精确缓存张量；这些张量不入Git，重新编码在浮点并列处可能有细小差异。首次文本编码的时间和峰值显存不属于表中的缓存后在线成本。

## 独立全量评测

```bash
python DINOtool/scripts/evaluate_frozen_geometry.py \
  --config research/query_slot_best_alias_20261013/configs/voc21.yaml \
  --data-root /path/to/VOC2012 \
  --checkpoint-dir /path/to/checkpoints \
  --dinov3-repo DINOtool/dinov3_hub \
  --upstream-root third_party/VIP \
  --text-cache /path/to/cache/voc21.pt \
  --wide-branch query_slot --query-slot-capacity 24 --efficient \
  --output /path/to/new_results/voc21.json
```

替换配置和数据根目录评测其他数据集；split、标签映射、RGB波段和分片行为沿用[部署说明](DEPLOYMENT_20261010.md#全量评测)。模型核对配置声明的图像数。`evaluate_frozen_geometry.py`只依赖模型配置、词表、上游、权重和数据，不依赖本轮历史结果或A800绝对路径。

本轮批量入口为[`eval_query_slot_best_alias.py`](../DINOtool/scripts/eval_query_slot_best_alias.py)和[`run_query_slot_best_alias.py`](../DINOtool/scripts/run_query_slot_best_alias.py)。它们另外使用`protocol.json`中的机器路径与既有baseline覆盖清单；迁移环境需要复制输入、修改路径并在新目录生成结果。批量评测显式断言两个模块生效及文本塔释放。当前调度器已修复最后一个成本任务结束前提前启动全量的问题；该修复不改变模型推理。

## 对比与成本口径

新旧mIoU均为既有完整split的全量结果。完整覆盖45,200次数据集图像评测，计入LoveDA附加P读出为46,869次协议图像评测；VOC20/21和PC59/60共享来源。

耗时／显存为A800、batch=1，固定7张完整图，每图2次预热、7次CUDA同步重复。耗时是49次重复的中位数；显存是峰值`max_memory_allocated`，`reserved`也保存在CSV/JSON。边界包含CPU RGB缩放、全部编码、读出、求解、原尺寸恢复和CPU uint8输出；排除解码、初始化、离线文本编码和标签读取。旧Efficient成本来自仓库既有记录，属于跨轮对照。

FLAIR首次成本与首批全量任务短暂重叠，仅保留为审计记录；最终表使用[`cost_recheck/cost/flair1.json`](../research/query_slot_best_alias_20261013/cost_recheck/cost/flair1.json)的独立补测。不要把全量评测wall time换算为本表延迟。

## 可审查证据

- `research/query_slot_best_alias_20261013/merged/`：15份全量混淆矩阵、逐类指标和覆盖检查。
- `full/`：39份分片JSON和39份逐图TP／target／predicted统计NPZ；没有图像或mask。
- `cost/`、`cost_recheck/`、`cost_keys.json`：固定输入清单、逐次耗时和显存原始记录。
- `source/`：A800实际执行的源码快照；其中`__init__.py`记录当时依赖路径，仅供审计，正常运行使用`DINOtool/`实现。
- `protocol.json`、`state.json`、`suite_results.json`：输入参数、调度状态和完成记录。
- 旧Efficient对照：[原报告](../research/EFFICIENT_GEOMETRY_20261012.md)、[原成本](../research/efficient_geometry_20261012/cost)。

可以在仓库根目录运行`python tools/report_query_slot_best_alias.py`，从这些证据重新生成表格。2026-10-09论文冻结主表与本轮带标签开发适配的工程结果分别保存。
