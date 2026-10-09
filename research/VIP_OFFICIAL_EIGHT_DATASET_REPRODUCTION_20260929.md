# Pinned VIP inference on eight locked remote-sensing datasets

Date: 2026-09-29. This record separates upstream-configured VIP inference from
cross-dataset transfer. It does not claim to reproduce the paper's published
benchmark table, which uses different preparation and/or evaluation units.

## Source and implementation

- VIP source: `third_party/VIP_official`, commit
  `5bd25ee03ec25c1538622cf7da661e8c0461e769`.
- Inference port: `DINOtool/dinotool/vip_official_adapter.py`; evaluator:
  `DINOtool/scripts/eval_vip_official_eight.py`; verified shard merger:
  `DINOtool/scripts/merge_vip_official_shards.py`.
- The pinned upstream `VisionHead.proxy_attn` method is called directly for
  **both** frozen DINO.text vision-head blocks. The port loads the same frozen
  DINOv3 LVD and DINO.text head/text checkpoints already used by GEAR-OV.
- Each alias uses the pinned upstream OpenAI ImageNet text templates. Dense
  patch-text similarities, global pooled-patch alias saliency, within-class
  softmax alias weights, weight-scaled logits and tau log-sum-exp follow
  `dinosegmentor.py`. No model weights or thresholds are trained.
- Images preserve aspect ratio with long edge 448, use bilinear resize,
  336-pixel crops with stride 112, raw-logit averaging over overlap, bilinear
  resize back to the original mask size, then softmax/argmax and the configured
  confidence-to-background rule. The official VIP head fixes a 21x21 patch
  grid; when an adapted image has a short edge below 336, the port pads that
  crop to 336 and records how often this occurs.
- Public `dinosegmentor.py` depends on MMSeg and invokes a hub constructor
  whose published copy has an empty checkpoint path. This independent port
  uses the already validated local DINO loader and calls the pinned VIP
  attention method; it should not be described as an unmodified execution of
  the upstream MMSeg script.

## Reproduction boundary

| Dataset | VIP query source | VIP score/background settings | Locked scored set |
|---|---|---|---|
| Potsdam | upstream `cls_potsdam.txt` | upstream tau=1, tem=2, background threshold=0.25 to class 5 | 504 prepared RGB tiles, six classes |
| VDD | upstream `cls_vdd.txt` | upstream tau=1, tem=1, threshold=0.35 to class 0 | 80 images, official seven-class ontology |
| Vaihingen | upstream `cls_vaihingen.txt` | upstream tau=1, tem=10, threshold=0.1 to clutter class 5 | 113 prepared tiles; five scored classes, clutter GT ignored |
| LoveDA P/D | fixed `gar_llm_raw20_loveda_v1.json` | VIP constructor defaults: tau=4, tem=1, no confidence reassignment | 1,669 images, P/D from GEAR |
| UDD5 | fixed `hero_udd5_vip20.json` | VIP constructor defaults | 40 images |
| OEM | fixed `hero_oem_vip20.json` | VIP constructor defaults | 384 available images |
| LandCover.ai | fixed `gear_landcoverai_v1_20.json` | VIP constructor defaults | 1,602 official validation patches |
| FLAIR-1 | fixed `gear_flair1_main12_20.json` | VIP constructor defaults | 15,700 paired test-archive images |

For the last five datasets, VIP publishes neither dataset configurations nor
official evolved alias files. Using fixed 20-alias vocabularies reproduces its
**inference rule**, but not a published candidate-generation/distillation
protocol. These are explicitly cross-dataset adaptations, not official VIP
benchmark results. The vocabularies are identical to the corresponding
GEAR-OV run, and no target mask or metric was used to choose a VIP threshold.
Vaihingen includes the upstream sixth `clutter` query, whose predictions count
as false negatives for the five scored classes. UDD5 required 24 short-edge
crop pads; the other completed datasets required none.

The upstream VIP paper reports other validation splits, tile preparation and
ignore rules. Matched comparisons below hold the GEAR-OV sample keys and masks
fixed, not the paper's original benchmark protocol. Vaihingen's existing
first-three-band input produces low scores for all methods; this is not an
official native-RGB reproduction. iSAID is not substituted silently: labeled
iSAID validation was unavailable in the preceding eight-set GEAR run, so
LandCover.ai remains the explicit eighth set.

## Verified full results

All merged results have `coverage_verified=true`, unique complete sample keys,
matching inference signatures and the same global sample-key SHA-256 as the
corresponding GEAR-OV v2 result. Values are mIoU percent.

| Dataset | Images | VIP | GEAR-OV v2 | VIP - GEAR |
|---|---:|---:|---:|---:|
| LoveDA P | 1,669 | 42.3170 | 64.4423 | -22.1253 |
| LoveDA D | same images | 33.9828 | 42.4984 | -8.5156 |
| UDD5 | 40 | 41.4336 | 50.0877 | -8.6541 |
| OEM | 384 | 28.9516 | 46.4536 | -17.5020 |
| VDD | 80 | 52.0647 | 39.2423 | +12.8224 |
| Potsdam | 504 | 44.0643 | 40.4554 | +3.6089 |
| Vaihingen | 113 | 5.6908 | 4.7272 | +0.9636 |
| LandCover.ai | 1,602 | 37.3519 | 59.5201 | -22.1682 |
| FLAIR-1 | 15,700 | 19.3010 | 43.2535 | -23.9525 |

LoveDA P is a second metric on the same 1,669 images, not a ninth dataset.
Counting LoveDA D once, the eight-set equal-weight auxiliary average is
32.8551 for this VIP run versus 40.7798 for GEAR-OV v2. Different taxonomies,
ground resolutions and evaluation units make the individual rows primary.
LoveDA D foreground mIoU is 36.3863, UDD5 excluding `other` is 44.1680,
and LandCover.ai foreground is 31.1774.

| Dataset | Parallel wall seconds | Aggregate GPU seconds | Peak allocated MiB |
|---|---:|---:|---:|
| LoveDA | 117.40 | 466.34 | 1,821.66 |
| UDD5 | 22.30 | 22.30 | 2,265.28 |
| OEM | 52.91 | 52.91 | 1,819.40 |
| VDD | 50.71 | 50.71 | 2,527.10 |
| Potsdam | 80.08 | 80.08 | 1,754.38 |
| Vaihingen | 15.53 | 15.53 | 1,754.38 |
| LandCover.ai | 179.65 | 179.65 | 1,773.01 |
| FLAIR-1 | 277.93 | 1,882.37 | 1,878.05 |

These are combined per-run inference measurements, including text encoding.
The results use 4 LoveDA, 7 FLAIR-1, and one shard for each other dataset;
wall and aggregate GPU time are not per-image complexity comparisons.

### Interpretation

The upstream-configured VDD and Potsdam runs score 52.0647 and 44.0643 on
our locked masks, exceeding GEAR-OV v2 by 12.8224 and 3.6089 points. This
supports VIP as a strong relevant comparator. Vaihingen remains very low for
both methods under the existing first-three-band preparation.

The five transfer runs do **not** show that published VIP inherently fails on
LoveDA, OEM, LandCover.ai or FLAIR-1. They keep all 20 externally supplied
aliases per class; VIP does not publish evolved aliases or dataset-specific
settings for those datasets, so its candidate-distillation stage is not
reproduced there. Their poor scores establish only that this fixed-vocabulary,
constructor-default transfer is not robust. For example, OEM `tree` IoU is
24.13 versus GEAR's 61.36; LandCover.ai `woodland` is 8.72 versus 77.97;
FLAIR-1 `building` is 16.79 versus 57.60. Conversely, VDD `water` is 88.74
versus GEAR's 40.95. These class changes warrant a vocabulary/protocol audit
before claiming one architecture dominates the other.

Full per-class IoU, confusion matrices, exact manifests, wall time and peak
CUDA memory are preserved in `research/vip_official_DATASET_full_20260929_results.json`.

Four standard-library regression tests and eight one-image GPU smokes passed.
The initial six parallel smoke attempts did not enter inference because the
launcher-created empty result directories were mistaken for prior output;
the evaluator was corrected to reject nonempty directories and all six were
rerun under a new name. The earlier unnormalized-alias-mean Potsdam smoke was
also discarded before any full run. Neither failed smoke contributes to the
reported full results.
