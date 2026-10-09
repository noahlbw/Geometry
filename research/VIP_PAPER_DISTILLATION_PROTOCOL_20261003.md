# VIP Paper-Rule Vocabulary Distillation: Frozen Protocol

## Scope

Run full unlabeled distillation and cache-paired VIP_All20 versus VIP_Distilled on LoveDA1669, UDD540, OEM384, LandCover.ai1602 and FLAIR-115700 images (19395 total). Preserve the original fixed20 pools, 80 ImageNet templates, checkpoints, class order, masks and pinned repository resize/crop/scoring settings. Do not change RivalFineHard20 or select a new model.

Selection reads images only and averages scores equally over images with high-activation support. This is transductive evaluation-set vocabulary selection, not pure single-image inference.

For each alias, replace only its own canonical class name; leave rivals canonical. Compute multiclass softmax probabilities, VG from two random walks of the squared all-layer mean attention, and multiclass entropy SC on probability>=0.4 patches. Retain canonical unconditionally. Admit an alias only if cosine>=0.7, VG exceeds its canonical and SC falls below its canonical. No fixed retention count and no label-based parameter choice.

## Numerical Issue Found Before Full Evaluation

Pinned VIP can mask every proxy-attention entry in a row when raw feature similarity is globally high. `softmax([-inf, ...])` is undefined and returns NaN. Its earlier evaluator did not reject nonfinite probabilities before argmax. Historical VIP results therefore cannot establish a fair superiority claim on affected images.

Mask-free audit of the first8 images/domain:

| Dataset | Nonfinite upstream crops | Total crops |
| --- | ---: | ---: |
| LoveDA | 2 | 32 |
| UDD5 | 0 | 16 |
| OEM | 1 | 32 |
| LandCover.ai | 29 | 32 |
| FLAIR-1 | 14 | 32 |

The existing fixed identity self-Value fallback on all-masked rows made every audited feature finite. On every originally finite crop, the repaired and upstream projected features matched exactly (maximum error0). This is a numerical completion of an undefined case, not a fitted alias or class correction.

The new run explicitly uses this same repair for both full20 and distilled arms. VDD80, Potsdam504 and corrected-input Vaihingen113 official-short-query VIP evaluations are rerun with the same repair, for a valid updated eight-domain comparison (20092 images). They do not regenerate or select additional aliases.

## Provenance

Paper: <https://arxiv.org/html/2605.12325v2>, Section3.3 and AppendixA.1. Pinned upstream commit: `5bd25ee03ec25c1538622cf7da661e8c0461e769`.

Public pinned VIP does not expose distillation code. This implementation follows its published equations, with explicit conventions for attention prefix slicing, normalized template-mean cosine, padding, and pooling sliding crops within each image. Keep long-edge448/crop336/stride112 from the pinned repository; do not substitute the appendix shorter-edge336/crop224 protocol mid-comparison.

Remote run: `results/vip_paper_distillation_finite_full_20261003`; controller: `vpd03_finite_controller`. The original failed smoke and image-only diagnostic remain at `results/vip_paper_distillation_full_20261003`.

Local manager: `tools/vip_paper_distillation_experiment.py` (`status`, `collect`). Final report target: `research/VIP_PAPER_DISTILLATION_FULL_20261003.md`. Final reporting must distinguish the numerical-repair gain from the vocabulary-distillation gain and compare against the unchanged retained model.

Verification before launch: nine distillation tests, two finite-attention tests, Python compilation. Full runs are gated on five mask-free source/cache smoke checks; no result is claimed until complete coverage, source identities and confusion sums are verified.
