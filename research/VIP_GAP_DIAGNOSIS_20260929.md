# VIP gap diagnosis and next decisions

Date: 2026-09-29. This is a read-only model/protocol diagnosis, not a new evaluation.
Public sources below were fetched directly after the literature-search service
reported that it required authentication. No model or dataset protocol was changed.

## Observed performance

| Dataset | Current full evaluation | Geometry | BoundedUnion | VIP paper Table 2 |
| --- | --- | ---: | ---: | ---: |
| Potsdam | 504 1000x1000 tiles from 14 source images | 31.7524 | 30.9471 | 49.8 |
| VDD | 80 images | 37.2709 | 40.6988 | 54.3 |

These are not yet protocol-matched comparisons with the published numbers.
The separate v2 screen is documented in CONTRASTIVE_CONTEXT_A800_SCREEN_20260929.md.
It improves Potsdam but degrades VDD, UDD5 and OEM against BoundedUnion.
Its graph adds only 0.0039 to 0.0289 mIoU points over the no-graph arm.

Potsdam BoundedUnion car GT fraction is about 1.96%, prediction fraction 29.19%,
recall 99.45%, precision 6.69%. This is a severe false-positive problem, not evidence
that missed small objects are the dominant failure. Building IoU is 77.7414;
low vegetation 17.2785, tree 33.4506, car 6.6911, clutter 6.1765.

## New verified findings

1. The VDD dataset author's class table explicitly defines index 1 as `wall`
   (`building (roof not included)` in its UAVid mapping), index 5 as `roof`,
   and index 0 as `other`. VIP uses `facade` for index 1 and `surface,other`
   for index 0. Our grounded_vdd20.json uses `building` with roof/rooftop
   aliases for index 1 and a background-oriented vocabulary for index 0.
   This is a semantic query specification problem; it does not establish that
   numeric label IDs or image-mask alignment are wrong. Its score impact is unmeasured.

2. VIP's public vision_tower.py replaces attention in both head blocks.
   Patch attention operates on patch values only; prefix tokens are separated.
   It normalizes DINO features, subtracts 1.5 times the mean similarity, masks
   nonpositive edges, then applies softmax. Our tcpr.py/parallel_readout.py
   changes only the final head block, uses cosine/0.1 plus a spatial Gaussian,
   preserves native patch-attention mass, and retains patch-to-prefix reads.
   These are verified structural differences. A causal contribution to the
   car bias has not yet been measured.

3. VIP's public remote-sensing configurations resize to 448x448 with aspect
   ratio preserved. Its segmentor constructor defaults to crop 336/stride 112.
   Its appendix describes shorter-side 336 and crop 224/stride 112. These
   are not interchangeable; a reproduction must pin the actual configuration
   and implementation version. Our evaluation uses 512 tiles, 128 overlap,
   and probability/Hann blending at original resolution.

4. VIP's public configs use confidence-based background assignment:
   Potsdam threshold 0.25, bg_idx=5; VDD threshold 0.35, bg_idx=0. Our current
   evaluator uses ordinary class argmax without this rule. The thresholds
   cannot be transplanted directly because score scales/aggregation differ.

5. VIP points to SegEarth-OV preprocessing, which specifies noBoundary labels
   for Potsdam. We used DINO_Soars preprocessing into 1000x1000 tiles and have
   zero ignored pixels in the full result. Boundary-mask compatibility is
   unresolved. SegEarth-OV's documentation and converter defaults also do not
   give a sufficiently unambiguous tile-count specification to infer the
   exact paper evaluation solely from the README. Inspect the generated
   manifest, masks and configuration before claiming a matched comparison.

6. Our text encoding uses the correct patch-aligned half of DINO.text output.
   There is no evidence from this inspection of a global/patch text-half mixup.
   VIP's public code uses ImageNet templates, global pooled-patch saliency
   weighting and log-sum-exp aggregation; ours uses six aerial templates and
   normalized log-mean-exp over the fixed 20 aliases. The earlier local
   VIP-style filtering experiment is not a full official VIP reproduction.

## Mechanism limitations in the current v2

- Every alias is assigned a nearest competing alias even when the match is weak.
- Every class pair is fitted with equal weight; inactive rivals can change a
  pixel's decision. There is no measured evidence-reliability weight.
- context_pairs is computed after local/context log-mean-exp union, so it is
  not an independent contextual observation.
- Pairwise responses reuse the same feature-text scores. They do not by
  themselves supply an independent visual test of correctness.
- Pair targets from nonlinear nearest-alias matching need not be cycle
  consistent across three classes; reconstruction can require compromises.
- BoundedUnion has a known algebraic asymmetry. For local l and context h,
  u = tau*log((exp(l/tau)+exp(h/tau))/2). As h tends far below l, the reduction
  saturates at tau*log(2), about 0.0485 for tau=0.07. Stronger positive context
  is preserved much more readily. This is a candidate amplification mechanism,
  not a measured attribution of the observed car errors.

## Priority order

1. Fix semantic query definitions against dataset documentation (wall/facade,
   roof, other). Keep the historical vocabulary and results as separate versions.
   Compare canonical-only and 20-alias vocabularies under identical readout.
2. Establish a pinned official VIP reference on the same images and evaluator;
   separately reproduce its published protocol. Do not silently replace our
   current masks, split, resize or ignore policy.
3. Diagnose readout with a 2x2 comparison: one/two modified head blocks crossed
   with preserved/blocked patch-to-prefix attention. Hold vocabulary, patch
   graph, scale and evaluation fixed. Then test the official sparse affinity
   as a separate factor. Track car precision/recall and predicted area, not
   just mIoU. Do not constrain area to GT fractions at inference.
4. Only on a repaired local baseline, compare equal-budget raw bounded context,
   ordinary averaging, log-mean-exp union and signed residual reconstruction.
   Retain pre-union local/context observations separately. Diagnose actual
   active class pairs and test whether proposed evidence distinguishes useful
   from harmful changes before adding graph terms.
5. A candidate research contribution is localization-supported semantic
   transport: class evidence must have localized visual support, then geometry
   carries its signed contextual increment. This remains a hypothesis. Neither
   weighted Huber fitting nor a Laplacian is itself a novel contribution.
   Require improvement over a strong matched baseline and a shuffled-evidence
   control, and retain an untouched dataset or geographic split for final claims.

## Public sources checked

- VIP paper: https://arxiv.org/html/2605.12325v2
- VIP vision head: https://github.com/MiSsU-HH/VIP/blob/main/dinov3/eval/text/vision_tower.py
- VIP segmentor: https://github.com/MiSsU-HH/VIP/blob/main/dinosegmentor.py
- VIP Potsdam config: https://github.com/MiSsU-HH/VIP/blob/main/configs/cfg_potsdam.py
- VIP VDD config: https://github.com/MiSsU-HH/VIP/blob/main/configs/cfg_VDD.py
- VIP VDD query names: https://github.com/MiSsU-HH/VIP/blob/main/configs/cls_vdd.txt
- VDD dataset class definitions: https://github.com/RussRobin/VDD#class-id
- Referenced preprocessing: https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md
- Potsdam converter: https://github.com/likyoo/SegEarth-OV/blob/main/tools/dataset_converters/potsdam.py
