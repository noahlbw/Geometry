# 代码入口、复现与来源

## 快照与环境

这份仓库保存2026-10-09冻结模型和相关探索代码，不是新的评测或重新调参。终版由 `research/alias_finalization_20261009/protocol.json` 和 `final_model.json`共同定义，ordinary alias policy固定为Uniform。

实验在Linux/A800、Python/PyTorch CUDA环境完成。推荐Python≥3.10，按实际CUDA安装兼容PyTorch/torchvision，再安装：

```bash
python -m pip install -e ./DINOtool
python -m pip install opencv-python-headless einops timm
export PYTHONPATH="$PWD/DINOtool:$PWD/DINOtool/scripts:$PWD/third_party/DINO_Soars"
```

`DINOtool/pyproject.toml`记录工具箱依赖；上游DINOv3/DINO.text仍需自身依赖。这不是已验证所有版本组合的锁文件。DINOtool默认CLI面向早期工具箱，不默认选择当前终版。

## 上游与权重

- VIP：[MiSsU-HH/VIP](https://github.com/MiSsU-HH/VIP)，固定commit `5bd25ee03ec25c1538622cf7da661e8c0461e769`。
- DINOv3官方模型与权重：[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)。实验本地DINO.text factory来自固定DINO_Soars来源 `f6704c1b76137543328567c8f8b49a2dc318824c`，由 `DINOtool/dinov3_hub/hubconf.py`读取原始checkpoint。
- 两个主checkpoint为 `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` 和 `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`，以及 `bpe_simple_vocab_16e6.txt.gz`。原始清单也记录SAT checkpoint，但当前终版构造没有启用额外SAT视觉分支。
- 原Geometry、SCLIP/VIPProxy等源码近邻边界见[机制与创新审计](../research/geometry_publication_20261001/NOVELTY_AND_MECHANISM_AUDIT.md)。归属第三方的算子与经典求解不改名宣称新发明。第三方代码、checkpoint和数据遵循各自许可／访问条件，本快照不重新授权这些资产。

## 最终推理API

核心入口在 `DINOtool/dinotool/taxonomy_inference.py`：

```python
from dinotool.taxonomy_inference import TaxonomyInference

# geometry: GeometryExecution(TCPRSegmenter(...))
# vip: FiniteVIPObserver(...)
# banks/queries: 用冻结词库与模板编码的TCPRTextBank / VIPQueries字典
# image: RGB float tensor [3,H,W]，数值范围[0,1]；不向predict传mask
model = TaxonomyInference.for_finalization(
    geometry, vip, banks, queries,
    ordinary_alias_policy="uniform",
    family=entry["family"],
    background=entry["background_index"],
    residual_features=residual_features,  # 普通协议None；PC60按冻结残余本体供给
    local_background="retained",
)
prediction, diagnostics = model.predict(image)
```

遥感要求 `original_imagenet` 与 `focused20`，自然要求 `semantic_segmentation`。不要将全词库字典中的旧比较bank当成实际运行bank。详细编码、profile和评测调用可读 `eval_development_readout.load_models` 与 `eval_alias_finalization.build_models`。

## 评测与历史缓存

冻结评测脚本不是不依赖资产的一键命令。它还需要数据集、checkpoint、固定VIP checkout、历史原20文本缓存，以及PC60独立残余缓存。源码、所有词/模板、class ID和协议在仓库；二进制文本缓存与逐图NPZ不在Git。文本可按源码重新编码，但浮点并列处不保证与原缓存bitwise一致。

原始入口及必要参数为：

```bash
python DINOtool/scripts/eval_alias_finalization.py \
  --dataset vdd --mode full \
  --suite-root /path/to/prepared/alias_finalization_20261009 \
  --data-root /path/to/vdd \
  --original-cache /path/to/original20/text_cache/vdd.pt \
  --dinov3-repo "$PWD/DINOtool/dinov3_hub" \
  --checkpoint-dir /path/to/ckpt/DINO \
  --upstream-root "$PWD/third_party/VIP" \
  --num-shards 1 --shard-index 0 \
  --output-dir /path/to/new-output/vdd/s0
```

准备suite时复制冻结 `protocol.json`，按本机资产路径配置数据加载，并在新输出目录执行。历史签名中的checkpoint路径/mtime用于当时身份验证，不应伪造成本机身份。`eval_curated20_patchonly2.models`展示原20缓存生成，`eval_development_readout.load_models`展示补充词库编码。PC60残余构造来自 `dinotool/residual_ontology.py`及对应脚本。

跨机器重新编码是新的复现实验，应保存自己的来源、环境及数值差异，不覆盖本仓库原始证据。本次没有为发布重新运行GPU评测。

## 结果与统计

`research/alias_finalization_20261009/DATASET/merged.json`保存完整混淆矩阵、类顺序、覆盖检查、身份和诊断；`cost_*.json`保存每图每次延迟、实际调用数、allocated/reserved及预处理。

LoveDA P是独立六类读出，不是D argmax减一。历史P独立文本缓存有3个局部向量与D切片不同；只补算历史参考，1669张逐图混淆精确复现。补算来源与原参考保存在 `reference_recovery`，候选结果不变。配对统计只使用候选/端点，不使用被替换的参考项。

`statistics_full/paired_statistics.json`为2000次配对文件名来源组bootstrap，TP/union充分统计重建通过。来源组是采集代理，不是认证独立采集单元；区间不消除长期开发偏倚。逐图NPZ没有打包，因此仓库可重新导出主表，但重做bootstrap需要原逐图数组或新的全量评测。

无GPU即可重新生成本页链接的实验表：

```bash
python tools/export_geometry_tables.py --root .
```

## 测试与计时口径

冻结开发时的2个加载器／bank测试和9个推理API测试已通过。源码中的相关测试为 `test_alias_finalization.py` 与 `test_taxonomy_inference.py`。发布打包另外检查源文件语法、词表/结果清单、表格数值和文件范围；不把这些检查冒充新GPU性能验证。

成本每协议7个确定性完整图×7次预热后同步重复，每个模式独立进程。包含缩放、所有编码/语义头/alias归约/写回/拼接/原尺寸恢复及CPU预测，不含解码、模型加载和文本编码，后者另记。自然VIP声明协议short336/cap2048与本模型short336/cap672不同；五个非官方遥感域的VIP声明计时实际为外部20词适配。完整精度与成本的这种协议边界均须保留。

## 尚未完成的投稿证据

最终任务路由的最近算子匹配、真正新来源/官方测试、目标尺度和边界机制证据、文献优先权复核仍未完成。当前冻结设计与科学定案完成，不代表所有域SOTA或CVPR录用保证。没有证明词表扰动稳健性，也没有自动坏词识别主张。
