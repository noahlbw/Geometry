# DINOtool

Training-free open-vocabulary semantic segmentation for remote-sensing images,
built on the official DINOv3 ViT-L/16 and dino.txt checkpoints.

It also includes an explicitly separate **external-data adaptation** branch:
the DINOv3 LVD dino.txt backbone and DINOv3 SAT-493M backbone stay frozen, while
a small dense bridge learns to map remote-sensing patch features into the dino.txt
text space. This preserves text-conditioned class scoring at inference rather
than training a fixed LoveDA classifier.

The default `dinosplat` mode reimplements DinoSplat-OV (arXiv:2608.03023):

1. remote-sensing prompt and synonym aggregation;
2. Text-aware Laplacian Propagation (TLP) on DINO patch logits;
3. RGB-guided Gaussian Splatting Upsampling (GSUP) with per-image optimization;
4. global-anchor and Hann-weighted sliding-window fusion.

See [RESEARCH.md](RESEARCH.md) for the method comparison and the exact boundary
between published equations and implementation choices.

## Local assets

The defaults expect this layout:

```text
code/
  DINOtool/
  dinov3/
  ckpt/DINO/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
    bpe_simple_vocab_16e6.txt.gz
```

## Install

```bash
cd /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code/DINOtool
bash scripts/setup_gpu_env.sh
```

The setup creates an isolated environment and reuses the existing compatible
CUDA 12.8 runtime when available. It does not modify the system Python
environment.

## Inspect

```bash
.venv/bin/dino-ovss inspect
```

## Inference

Use a class config when synonyms matter:

```bash
.venv/bin/dino-ovss infer \
  --image /data/scene.tif \
  --class-config configs/remote_sensing_classes.example.json \
  --output-dir /data/scene_dino_ovss \
  --mode dinosplat \
  --tile-size 512 --overlap 128 \
  --confidence-threshold 0.35
```

Or pass an ordered class list directly:

```bash
.venv/bin/dino-ovss infer \
  --image /data/scene.png \
  --classes background building road water vegetation bare-soil vehicle \
  --output-dir /data/scene_dino_ovss
```

Every pixel is classified among the supplied classes. Include `background` or
`other land cover` for practical extraction tasks. Pixels below the optional
confidence threshold receive label `255` (`unknown`).

## Modes

- `baseline`: official dino.txt cosine segmentation plus bilinear upsampling.
- `tlp`: baseline plus Text-aware Laplacian Propagation.
- `dinosplat`: TLP plus test-time Gaussian Splatting Upsampling; default.
- `dinosplat-sat`: experimental `dinosplat` variant that multiplies TLP edges by
  frozen DINOv3 SAT-493M neighbor affinity. This is not part of the paper.

## Outputs

The output directory contains:

- `labels.png`: indexed class IDs, with `255` reserved for unknown;
- `labels_color.png`: deterministic color rendering;
- `confidence.png`: maximum class probability;
- `preview_overlay.png`: quick visual inspection;
- `legend.json` and `metadata.json`;
- `labels.tif` and `confidence.tif` for GeoTIFF input, preserving CRS/transform.

Large probability accumulators automatically use a disk-backed temporary array.
GSUP is the expensive stage; reduce `--gsup-steps`, `--gsup-optimization-size`,
or use `--mode tlp` for a faster ablation.

## JSON agent entry point

```bash
echo '{"image":"/data/scene.tif","classes":["background","building","road","water"],"output_dir":"/data/result"}' \
  | .venv/bin/dino-ovss-agent
```

## Fast dense DINO agent path

For latency-sensitive agent calls, `infer-dino-dense` loads only DINOv3 and
does one image forward regardless of the number of requested classes. Text
prototypes are batched and cached, and the dense class map is produced by one
matrix multiplication followed by bilinear upsampling. The default 512-pixel
long side is intended for an 800x800 RGB image; increase it when small objects
matter more than latency. This path is training-free and does not load SAM3.

```bash
.venv/bin/dino-ovss infer-dino-dense \
  --image /data/scene.png \
  --classes building road water vehicle \
  --output-dir /data/scene_fast_dense \
  --dino-input-resolution 512
```

The JSONL worker keeps DINO weights resident and caches the most recent image
features. Reusing an `image_id` for the same immutable image avoids another
ViT forward when an agent asks follow-up questions:

```bash
.venv/bin/dino-ovss-agent --serve-dino-dense \
  --code-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code \
  --dino-input-resolution 512 --image-cache-size 1 --warmup
```

```json
{"image":"/data/scene.png","image_id":"scene-42","classes":["building","road","water"],"output_dir":"/data/scene_fast_dense"}
```

The regular `dino-ovss-agent` JSON entry point accepts `"mode":"fast-dense"`
as a one-shot request. An external-data CAFe cost-aggregation checkpoint can
be supplied with `--cost-aggregation-checkpoint` for a quality experiment,
but its attention decoder is slower than the default frozen DINO path.

## Dynamic SAM3 + DINO agent OVSS

`infer-sam3-dino` is the agent-oriented path for RGB images. It accepts the
caller vocabulary instead of the fixed LoveDA prompt list. By default every
supplied foreground concept is sent to SAM3; background is inserted
automatically when absent. DINOv3 supplies global evidence and verifies only
ambiguous SAM regions, while SAM3 still draws every mask.

```bash
.venv/bin/dino-ovss infer-sam3-dino \
  --image /data/scene.png \
  --classes building road water solar-panel vehicle \
  --output-dir /data/scene_agent_ovss \
  --sam3-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code/SegEarth-OV3 \
  --profile fast
```

Profiles are fixed without benchmark labels:

- `fast`: official SAM3 dual-head aggregation, one canonical prompt per
  routed class, no geometric re-prompting, and frozen DINO region verification;
- `balanced`: one prompt per routed class and at most two geometric prompt
  calls per image;
- `accurate`: up to two prompts per routed class, four geometric calls, TLP,
  and frozen SAT structure features.

`--max-active-classes N` is an explicit latency/recall trade-off: it lets DINO
route only the top `N` foreground concepts to SAM3. The default is `0`, which
retains every caller-supplied class. Thus the tool never runs the fixed
35-prompt LoveDA bank unless a caller explicitly supplies 35 concepts. When a
limit is used, the response records all omitted concepts.

For an agent that asks several questions about the same image, use the
persistent JSONL worker. It loads models once and caches the SAM3 image state
and DINO patch features for the most recent image:

```bash
.venv/bin/dino-ovss-agent --serve-sam3-dino \
  --sam3-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code/SegEarth-OV3 \
  --profile fast --image-cache-size 1 --warmup
```

Send one request per line while the worker is running:

```json
{"image":"/data/scene.png","image_id":"scene-42","classes":["building","road","water"],"output_dir":"/data/scene_buildings"}
```

Use the same `image_id` only for the same immutable image. Each response is a
single JSON line and includes active classes, prompt count, cache-hit state,
stage timings, and paths to `labels.png`, `confidence.png`, and `metadata.json`.
`--warmup` spends a short synthetic pass before the worker reads requests, so
CUDA initialization does not inflate the first real agent response.

## LoveDA benchmark

The labeled LoveDA validation split contains 1,669 images. LoveDA's official
test labels are private, so local mIoU is computed on validation; test results
require a server submission. The benchmark maps ground-truth IDs `1..7` to the
seven ordered prediction IDs and ignores ground-truth `0`/`255`.

Run the three published ablations in one shared-feature pass:

```bash
.venv/bin/dino-ovss benchmark-loveda \
  --data-root /data/LoveDA/Val \
  --output-dir /data/results/DINOtool_LoveDA_val \
  --modes baseline tlp dinosplat \
  --tile-size 512 --overlap 128
```

`results.json` is updated every ten images. Predictions are stored below
`predictions/<mode>/`, so the same command resumes safely after interruption.
The command refuses to reuse an output directory when model, prompts, data,
or inference settings differ. Use `--max-images N` for a deterministic smoke
test and `--class-config configs/loveda.json` to make the default synonym set
explicit.

### TCPR training-free screen

TCPR keeps the frozen Geometry readout as its first route and adds a
class-competition-conditioned reread of the same final-block value tokens.
The fixed LoveDA vocabulary contains 20 VIP-style candidate phrases per class;
canonical names are always kept and at most eight aliases per class are chosen
from pilot-map consistency and activation concentration. No LLM or LoveDA mask
is used during inference.

```bash
PYTHONPATH=. python scripts/eval_tcpr_loveda_e1.py \
  --dinov3-repo /data/code/ovss/dino_ovss_training_free_20260924/DINOtool/dinov3_hub \
  --checkpoint-dir /data/code/ovss/dino_ovss_training_free_20260924/weights \
  --data-root /data/code/ovss/dino_ovss_training_free_20260924/data/loveda/val \
  --output-dir /data/code/ovss/dino_ovss_training_free_20260924/results/tcpr_e1_64_20260924
```

The output compares only `G_geometry` and `TCPR`, and reports reference/evidence
coverage plus beneficial and harmful changes. P and D share one backbone and
two pilot reads per tile; their text-conditioned final rereads remain separate.

## SAM 3 SegEarth-OV3 baseline

`benchmark-loveda-sam3` runs the official SegEarth-OV3 SAM 3 inference logic
without the MMSeg runner: SAM 3's instance and semantic heads are max-fused,
the presence score filters absent categories, and aliases are aggregated by
class. This avoids building an `mmcv` CUDA extension that is not available for
every PyTorch runtime while keeping the model path and post-processing intact.

Obtain the official repository and its `facebook/sam3` `sam3.pt` checkpoint,
then run the labeled LoveDA validation benchmark:

```bash
.venv/bin/dino-ovss benchmark-loveda-sam3 \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/LoveDA/val \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/SegEarthOV3_LoveDA_val \
  --sam3-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code/SegEarth-OV3
```

The command records the complete checkpoint manifest, the official LoveDA
prompt file, two thresholds, all predictions, and a resumable `results.json`.
Its default settings match `configs/cfg_loveda.py` in the SegEarth-OV3
repository: input resolution `1008`, object confidence threshold `0.5`, and
background probability threshold `0.5`.

## Training-free SAM 3 + DINOv3 region verification

`benchmark-loveda-sam3-dino` keeps SAM 3 responsible for every mask. Its
default path deliberately preserves the official SegEarth-OV3 mask and score
path before applying any DINO decision:

1. retain exactly the validated SegEarth-OV3 aliases and official dual-head
   max aggregation;
2. use frozen DINOv3/DINO.text to verify ambiguous SAM regions. DINO may only
   choose from SAM's local top candidates and never paints a new mask.

The default configuration is fixed without LoveDA labels. Experimental PGRF,
aerial prompts, box re-prompting, TLP, and SAT structure features are opt-in
through `--pgrf`, `--constrained-prompts`, `--geometry-reprompt`, `--dino-tlp`,
and `--dino-sat`. `--legacy-pixel-fusion` is kept only to reproduce the retired
posterior-blending ablation. Use a fresh output directory for each fixed
configuration:

```bash
.venv/bin/dino-ovss benchmark-loveda-sam3-dino \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/LoveDA/val \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/SAM3_DINO_REGION_LoveDA_val \
  --sam3-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/code/SegEarth-OV3
```

LoveDA masks are read only by the benchmark metric after predictions are
written. They are never used to choose prompts, thresholds, or corrections.

## External-data DINO adaptation

`train-oem-dino` is intentionally isolated from the LoveDA benchmark. It trains
only on the official OpenEarthMap train split and selects checkpoints by the
official OpenEarthMap validation split. LoveDA validation must not be passed to
this command; it is loaded only later by the independent benchmark command.

OpenEarthMap has 5,000 high-resolution remote-sensing images with eight land
cover classes that cover most LoveDA semantics. Its source imagery and labels
have mixed upstream licenses; the official attribution and the local provenance
file must be retained with any resulting model.

```bash
bash scripts/download_openearthmap.sh \
  /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/OpenEarthMap

.venv/bin/dino-ovss train-oem-dino \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/OpenEarthMap_wo_xBD \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/DINOv3SAT_OEM_adapter_v1 \
  --epochs 20 --crop-size 512 --batch-size 8 \
  --allow-missing-source-images
```

The V2 training controls retain open-vocabulary behavior while fitting external
source labels: class-balanced source loss, label smoothing, frozen dino.txt
auxiliary-concept score preservation, and ignored-pixel feature preservation.
They are recorded in `training_config.json` and default on for new runs. For a
larger external-only run, increase the crop and adapter capacity rather than
using LoveDA for tuning:

```bash
.venv/bin/dino-ovss train-oem-dino \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/OpenEarthMap_wo_xBD \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/DINOv3SAT_OEM_adapter_v2 \
  --epochs 30 --crop-size 768 --batch-size 16 \
  --adapter-hidden-dim 512 --adapter-context-blocks 4 \
  --class-balance-power 0.5 --open-vocabulary-preservation-weight 0.25 \
  --allow-missing-source-images
```

The official archive unpacks as `OpenEarthMap_wo_xBD`: it deliberately omits
the xBD RGB images while retaining their split entries. The explicit flag
skips only absent source images and records declared, available, and selected
sample counts in the checkpoint provenance. It must not be used for an
arbitrarily incomplete dataset.

The adapter checkpoint remains prompt-conditioned. Evaluate it on the untouched
LoveDA validation split with a new result directory:

```bash
.venv/bin/dino-ovss benchmark-loveda \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/LoveDA/val \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/DINOv3SAT_OEM_to_LoveDA_v1 \
  --adapter-checkpoint /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/DINOv3SAT_OEM_adapter_v1/best.pt \
  --modes tlp --tile-size 512 --overlap 128
```

This is cross-dataset semantic transfer with an open-vocabulary text interface.
Because the source data contains semantically overlapping land-cover classes,
it must not be described as an unseen-class LoveDA zero-shot result.

## DINO Cost-Aggregation OVSS

`train-oem-cost-dino` is the stronger DINO-only adaptation path. It keeps the
DINOv3 `dino.txt` text tower and vision head frozen, optionally tunes only the
last two visual backbone blocks, and learns a decoder shared across every
prompted class-cost map. It is therefore not an eight-class OpenEarthMap head:
at inference, it accepts a new ordered text vocabulary.

The reproducible external-only training run is:

```bash
bash scripts/train_oem_cost_dino_external_v1.sh
```

It selects only by OpenEarthMap validation mIoU. It never opens LoveDA during
training, checkpoint selection, hyperparameter selection, or pseudo-label
generation. `best.pt` includes optimizer state for resuming; use the smaller
`best_inference.pt` for inference after a run completes:

```bash
.venv/bin/dino-ovss infer \
  --image /data/scene.tif \
  --class-config configs/remote_sensing_classes.example.json \
  --output-dir /data/scene_dino_cost_ovss \
  --cost-aggregation-checkpoint \
  /root/siton-data-95873922dc054c44bdb7101cec2c70bb/results/DINOv3Cost_OEM_external_v1/best_inference.pt \
  --mode dinosplat
```

The cost decoder is DINO-only, so it supports `baseline`, `tlp`, and
`dinosplat`, but not `dinosplat-sat`.

## Multi-prototype unbalanced-OT DINO OVSS

`train-oem-uot` is the experimental remote-sensing adaptation proposed for
class similarity and intra-class variation. It keeps DINOv3/dino.txt as the
open-vocabulary anchor, adds shared visual modes for each prompted class, and
solves a KL-relaxed (unbalanced) transport problem from patch mass to class
modes plus an explicit unknown sink. Unlike a balanced assignment, absent
classes are not forced to occupy pixels; unlike a fixed classifier, the class
vocabulary is supplied at inference time.

The external-only full run is reproducible with:

```bash
bash scripts/train_oem_uot_external_v1.sh
```

The script trains on OpenEarthMap train, selects by OpenEarthMap validation,
and records the complete data provenance. LoveDA is never opened by the
training process. Use the resulting checkpoint with either inference or the
independent LoveDA benchmark:

```bash
.venv/bin/dino-ovss benchmark-loveda \
  --data-root /root/siton-data-95873922dc054c44bdb7101cec2c70bb/datasets/LoveDA/val \
  --output-dir /root/siton-data-95873922dc054c44bdb7101cec2c70bb/outputs/DINOv3UOT_OEM_to_LoveDA \
  --uot-checkpoint /root/siton-data-95873922dc054c44bdb7101cec2c70bb/outputs/uot_full/best_inference.pt \
  --modes baseline --tile-size 512 --overlap 128
```

`best.pt` retains optimizer state for resuming and `best_inference.pt` omits
it for deployment. The UOT decoder is DINO-only and supports `baseline`,
`tlp`, and `dinosplat`; it cannot be combined with `dinosplat-sat`.

## STRIDE-OV training-free inference core

`dinotool.stride_ov` implements the training-free STRIDE-OV patch-grid model.
It reuses one frozen DINO.text image forward for text-aligned and raw backbone
patches, forms the existing TLP proposal, then limits only supported directed
class-contrast flow across connected structure interfaces. It does not train a
decoder, broadcast region labels, or run a second vision backbone.

```python
from dinotool.config import CheckpointConfig
from dinotool.prompts import ClassSpec
from dinotool.stride_ov import StrideOVDINOTextSegmenter

model = StrideOVDINOTextSegmenter(CheckpointConfig.from_roots())
result = model.segment(
    rgb,  # [B, 3, H, W], H/W divisible by 16
    [ClassSpec("road", ("road",)), ClassSpec("building", ("building",))],
)
patch_logits = result.logits
```

The core returns raw, TLP-proposal, and constrained patch logits plus per-image
structure, budget, and solver diagnostics. Dataset-specific upsampling,
sliding-window stitching, and evaluation remain outside the method core.

## Tests

```bash
.venv/bin/python -m pytest
```
