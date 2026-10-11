# 五个公开 training-free 方法的全量复现

本轮依次评测 PEARL、GLA-CLIP、PTC、CorrCLIP（会议 v1）和 SegEarth-OV。每个方法覆盖现有 8 个遥感数据集和 7 个自然图像协议，共 45,200 次协议图像评测。VOC20/21、PC59/60 共享照片；LoveDA P/D 分别推理，遥感平均只计 D 一次。

实时汇总见 [全量比较表](../research/TRAINING_FREE_BASELINES_FULL_20261011.md)，冻结设置见 [protocol.json](../research/training_free_baselines_full_20261011/protocol.json)。`pending` 表示尚未完成，不能当作零分或已完成结果。

## 比较原则

- 作者支持的协议保留原词表文件字节、原解析器、阈值、模板、缩放与模型参数。PEARL 显式启用 TLP；GLA-CLIP 显式启用作者 `predict.sh` 中的完整开关；PTC 保留其校准参数。
- 未支持的域固定迁移设置，不用目标标签调词或阈值。遥感采用规范类别名、长边 448；SegEarth-OV 迁移自然图像采用公开任务词表、短边 448/长边上限 2048。具体设置逐项记录，不能称这些是作者报告的官方结果。
- CorrCLIP 使用 MetaCLIP、DINO 和原算法的在线 SAM2 HieraLarge 掩码生成；掩码生成计入推理成本。这个资源设置与论文预先缓存掩码的计时不同。
- SegEarth-OV 保留 SimFeatUp 预训练权重和原生 CUDA JBU。OEM 额外背景、Vaihingen 额外 clutter 查询保留；预测到这些非评分类别的像素计入错误列，不删除后重新 argmax。
- 原 Geometry 模型、参数和历史结果不变。表中 Geometry 是之前经过带标签开发及已查看验证集选择的最佳实测配置，VIP 是已核对的本地数值修复实现，不是论文数字。骨干、词表与资源不同，因此这是完整方法比较，不能单独归因到 Geometry 算子。

## 运行入口

本轮使用独立的 `.baseline_env_20261011`，不修改 Geometry 的原环境。安装脚本在 `DINOtool/scripts/setup_training_free_baselines.sh` 和 `setup_training_free_extra.sh`。Torch 2.4 的编译 MMCV 轮子为 2.2；仅放宽 MMSeg 的版本检查上界，并记录修改前后哈希，不替换任何推理算子。

已有授权 A800 工作区可运行：

```bash
python tools/training_free_baseline_experiment.py prepare
python tools/training_free_baseline_experiment.py fetch
python tools/training_free_baseline_experiment.py deploy
python tools/training_free_baseline_experiment.py start
python tools/training_free_baseline_experiment.py status
python tools/training_free_baseline_experiment.py collect
```

SSH 使用本机已验证的 known-hosts、SSH key/agent 或已有 askpass；凭据、权重和数据均不上传。更换服务器可设置 `GEOMETRY_BASELINE_SSH_HOST`、`GEOMETRY_BASELINE_SSH_USER`、`GEOMETRY_BASELINE_REMOTE_ROOT` 和 `GEOMETRY_BASELINE_CONTROLLER_PYTHON`。公开代码由源码获取脚本下载，仓库中记录其 URL 与固定 commit，不重新分发第三方源码。

其他机器直接使用远端三个核心入口也可以：

```bash
# 按固定 commit 获取五份作者源码，放入 SUITE/sources/；按本机路径修改 protocol.json 的 data_root。
PYTHONPATH=DINOtool:DINOtool/scripts python DINOtool/scripts/export_training_free_samples.py --source SUITE/protocol.json --output SUITE/samples.json
python DINOtool/scripts/prepare_training_free_weights.py --root SUITE
# 安装独立依赖后，用对应环境运行；output 必须是新目录。
PYTHONPATH=DINOtool:DINOtool/scripts python DINOtool/scripts/eval_training_free_baseline.py --root SUITE --output NEW_OUTPUT --method PEARL --dataset voc20 --phase evaluate --shards 1 --shard-index 0
```

单机路径与下载位置需要配置；脚本中的本轮 A800 路径不意味着数据和权重包含在 Git 中。`protocol.json`、`words/`、`cost_keys.json` 和 `references/` 已提供，不依赖旧视觉特征缓存。

## 验证与成本

每个方法／协议先做不读取标注的 smoke；完整输出需要原分辨率、有限数值和合法类别。合并时验证全部唯一样本 key、分片身份、逐图混淆矩阵之和，并与已核对 VIP 的各类别有效标注像素数比较。失败日志和已有输出保留，控制器仅在空闲 GPU 上启动新工作。

耗时使用与先前 VIP 相同的 7 张固定整图，每图 2 次预热、7 次同步计时，单卡单模型进程；包含官方 resize、预处理、模型和后处理，排除外围文件解码、初始化和指标计算。记录峰值 allocated/reserved 显存。LoveDA 成本测 D 单协议，避免把 P+D 两次推理误记为单次。CorrCLIP 原生 SAM2 内部的读图与生成仍计入。

全量 JSON 保存逐类 IoU、precision、recall、预测面积像素数、有效目标像素数、残余类以外 mIoU 和运行成本。下载与 GitHub 更新只追加本轮结果，不更换较差候选或覆盖历史记录。

## PTC 执行修复与失败恢复

原生 PTC 在 UDD5 的非正方形裁剪上将特征网格宽度误写为高度，触发 reshape 错误；LoveDA 上部分 ProxyCLIP 注意力行全为负无穷，softmax 产生 NaN。原源码、失败输出和日志均保留。单独源码副本仅修正宽度下标，并对全屏蔽行回退到自身 Value；参数、词表、校准规则及正常注意力行不变。不读取标签的检查确认三个故障图像恢复有限输出、UDD5 矩形图可运行，原本有限图像的预测逐像素完全一致。

这些结果标记为本地执行修复版，不等于作者原样发布的论文成绩。[修复记录](../research/training_free_baselines_full_20261011/source_repairs.json)包含完整 diff、源码哈希和诊断证据。合并文件的 `execution_variants` 标明各分片版本；已完成分片不重跑。失败分片恢复到新 `sN_attemptN` 目录，缺少逐图检查点的失败前缀会重新计算，原记录不覆盖。验证后可用 `resume --retry-failed-method PTC` 恢复失败任务。
