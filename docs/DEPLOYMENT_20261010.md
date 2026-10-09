# 当前冻结模型的独立复现

本入口运行2026-10-09 Kev实验最终冻结的任务配置，不进行搜索、调参或读取目标标签来生成预测。15个协议的配置在 `configs/frozen_20261010/*.yaml`；实际使用的每类alias、顺序、模板及PC60残余本体在 `vocabularies/frozen_20261010/*.json`，对应TXT便于查阅。YAML中的背景bias/阈值、Geometry读出strength、耦合gain、tau/tem及视野策略都是原冻结值。

框架仍为Geometry局部读取、VIP来源的宽视野观测、Geometry支持的耦合写回。`compact`只是执行优化：释放文本塔、只计算所需Geometry头、缓存宽分支类别索引，以及将最终类别ID无损转成uint8回传CPU。PC60不再回传最终不用的中间预测。保留Geometry FP32权重与宽分支FP16权重、原BF16 autocast、模板逐项乘法和双精度重建；没有减少视野或alias。未在这里将两条分支合为一套不同精度的权重。

## 环境与上游

实验环境是Linux、Python3.12、A800、PyTorch2.6.0/CUDA12.4。观测到的版本见 `research/compact_deployment_20261010/environment.json`。先安装匹配硬件的PyTorch，然后安装工具包和补充依赖：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ./DINOtool
python -m pip install opencv-python-headless einops timm PyYAML
mkdir -p third_party
git clone https://github.com/rfaulk/DINO_Soars.git third_party/DINO_Soars
git -C third_party/DINO_Soars checkout f6704c1b76137543328567c8f8b49a2dc318824c
git clone https://github.com/MiSsU-HH/VIP.git third_party/VIP
git -C third_party/VIP checkout 5bd25ee03ec25c1538622cf7da661e8c0461e769
export PYTHONPATH="$PWD/DINOtool:$PWD/DINOtool/scripts:$PWD/third_party/DINO_Soars"
```

这是已测环境信息，不保证所有其他依赖版本都等价。若已有环境，避免为复现覆盖其原有CUDA环境。测试可用 `python DINOtool/tests/test_compact_deployment.py`；完整测试另外需要pytest。

自行取得有访问许可的以下原始资产，放在同一checkpoint目录；不需要SAT权重：

- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`
- `bpe_simple_vocab_16e6.txt.gz`

权重来自DINOv3/DINO.text官方发布。遵循上游代码、模型和数据各自的许可，仓库不包含权重或数据。VIP模板文件副本保留来源，主运行仍校验VIP的固定commit。

## 单张图像

在仓库根目录运行：

```bash
python DINOtool/scripts/infer_frozen_geometry.py \
  --config configs/frozen_20261010/vdd.yaml \
  --checkpoint-dir /path/to/checkpoints \
  --dinov3-repo DINOtool/dinov3_hub \
  --upstream-root third_party/VIP \
  --text-cache /path/to/cache/vdd.pt \
  --input /path/to/image.jpg --output /path/to/new_prediction.png
```

首次运行按JSON词表编码并缓存文本，之后复用缓存；未读取目标mask。首次编码仍需文本塔的显存和离线时间，不能拿缓存后的在线峰值代替这个成本。已有缓存时，compact构造会在加载CUDA之前丢弃文本塔，避免初始化阶段也加载两套GPU文本权重；重新编码后也释放CPU/GPU文本塔，不将其保留到图像推理阶段。

输出PNG存储类别索引，类别顺序以JSON为准，显示颜色不是类别ID。`--reference`切换回原执行路径，保留相同配置和词表。缓存身份包含词表、checkpoint清单和上游版本；更换资产后使用新缓存路径。PC60会显式编码额外401个残余概念，不能省略。

## 全量评测

```bash
python DINOtool/scripts/evaluate_frozen_geometry.py \
  --config configs/frozen_20261010/potsdam.yaml \
  --data-root /path/to/datasets/potsdam \
  --checkpoint-dir /path/to/checkpoints \
  --dinov3-repo DINOtool/dinov3_hub \
  --upstream-root third_party/VIP \
  --text-cache /path/to/cache/potsdam.pt \
  --output /path/to/new_results/potsdam.json
```

替换配置和data-root即可评测其他协议，路径示例见 `assets.example.yaml`。遥感loader在 `eval_gear_ov.protocol` / `dinotool.rs_external`，自然loader在 `dinotool.natural_evaluation`；按这些loader准备官方split、RGB波段与原始标签映射。程序核对YAML预期图像数，不自动猜测另一split。默认单分片保存全部覆盖、原始混淆矩阵、逐类指标及缓存身份；可用 `--num-shards N --shard-index i`分片，分片输出不声称全量覆盖。

LoveDA分别运行真正六类P与七类D读出，输出两个混淆矩阵；P不是D的argmax减一。为使入口清楚、可独立复现，当前公开评测入口在P/D之间重复视觉观测；独立成本表测量单个primary D，不应拿该双协议评测wall time充当单图延迟。

## 复现与论文边界

原冻结精度来自原文本缓存；重新编码可能有GPU/CPU浮点归约或批次差异。新的复现结果应保存自己的混淆矩阵，不覆盖历史证据。全新文本编码与已缓存执行的验证范围在效率报告列明，不将未测协议说成逐像素等价。

任务参数和部分词表经过带标签开发选择；“training-free”指不更新视觉/文本模型权重，不表示未用标签选择配置。词表筛选是离线外部冻结语言先验，不是单图视觉动态筛词。开发选择、离线成本、初始化、在线成本分开报告。VIP参照为本地有限值修复实现，不是经认证的论文数字。
