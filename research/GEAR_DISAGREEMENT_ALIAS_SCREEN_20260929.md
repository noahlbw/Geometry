# GEAR-OV alias screening: preregistered image-only screen

Date: 2026-09-29. This is an exploratory study, not a verified performance claim.

## Question

Can a label-free alias selection rule avoid the coverage collapse of the old
VIP-style hard screen while removing aliases that hurt Geometry and GEAR-OV?

The previous selector retained the class-name alias unconditionally and kept
another alias only if its visual-grounding consistency exceeded that name and
its class-posterior entropy was lower. That selector reduced full-set GEAR-OV
performance on all six tested 20-alias datasets. Its fixed canonical anchor is
especially questionable for abstract names such as `rangeland` and
`impervious surface`.

## Diagnostic, not training

`DINOtool/scripts/diagnose_gear_alias_marginals.py` computes each alias's
leave-one-out contribution to local Geometry log-probability and patch-level
correct/wrong flips on fixed-seed image samples. These target labels are
**audit-only** and never enter the selection script or GEAR inference.

| Dataset | Aliases | Positive GT marginal | Old selected positive | Old rejected positive | AUC of inverse native-counterfactual score |
|---|---:|---:|---:|---:|---:|
| OEM | 160 | 79 | 18/36 | 61/124 | 0.8581 |
| FLAIR-1 | 240 | 124 | 15/35 | 109/205 | 0.6684 |
| LoveDA D | 140 | 76 | 34/59 | 42/81 | 0.7035 |
| UDD5 | 100 | 48 | not used for rule design | not used | 0.8185 |
| LandCover.ai | 100 | 50 | not used for rule design | not used | 0.8352 |

The AUCs concern rank agreement on sampled local patches, **not full-image
mIoU or proof that pruning helps**. The audit computes the proxy with the
alias removed from the native teacher. An independent implementation in
`DINOtool/scripts/select_gear_disagreement_aliases.py` does not read masks.

## Frozen candidate rule

For alias *a*, compare full Geometry class probabilities with the
counterfactual probabilities when *a* is removed. Score that change under a
native DINO.text class distribution computed **without a**. The candidate
retains the **lowest** 15 native-counterfactual gains among each class's 20
aliases. This is a test of whether Geometry's disagreement with its native
readout contains useful complementary evidence; it is not assumed true.

The rule uses at most 64 unlabeled images per dataset, at most two fixed-seed
512-pixel tiles per image, the same frozen DINOv3/DINO.text weights, and no
target labels. It is transductive vocabulary selection, not single-image
inference. Class names are not forced into the retained bank. The evaluator's
`--exact-alias-groups` option prevents it from silently adding them back.

Examples of the frozen selections:

- OEM `rangeland`: keep `grassland`, `short grass`, `pasture`, etc.; omit the
  `rangeland` original name.
- FLAIR-1 `impervious surface`: keep `asphalt`, `street`, `parking lot`, etc.;
  omit the abstract original name.
- LoveDA `water`: keep `river`, `lake`, `open water`; omit `dark surface` and
  `reflective surface`.

These examples are qualitative, not ground-truth proof of individual benefit.

## Full evaluation status

Five full evaluations were launched on A800 on GPUs 0–4, one per dataset:
OEM 384, FLAIR-1 15,700, LoveDA 1,669 (P and D), UDD5 40, and LandCover.ai
1,602. The 15-word vocabularies are frozen. Compare Geometry, Multiscale, and
GEAR-OV against the existing matched 20-word results. Do not revise the
retention count or score based on interim target metrics. The active heartbeat
`monitor-gear-disagreement-alias-evaluation` monitors these runs and will
append verified results after completion. The old alias-study heartbeat is
paused.

## Verified full-set outcome

All five evaluations completed. The single-shard results were merged with
`scripts/merge_gear_ov_shards.py`. Each merged result has `status=complete`,
`coverage_verified=true`, unique sample keys, exactly the expected image count,
the same global sample-key SHA as its all-20 and old-screen comparisons, and
`exact_alias_groups=true` with exactly 15 aliases per class. No target labels
were read by the selector; target labels enter only the final metrics and the
separate audit described above. These are full evaluations, not the diagnostic
patch samples.

| Dataset/split | Images | GEAR all20 | Old hard screen | New 15-word screen | New minus all20 |
|---|---:|---:|---:|---:|---:|
| OEM | 384 | 46.4536 | 40.9423 | 46.5527 | +0.0991 |
| FLAIR-1 | 15,700 | 43.2535 | 29.9676 | 43.8325 | +0.5790 |
| LoveDA P | 1,669 | 64.4423 | 58.0165 | 64.1135 | -0.3288 |
| LoveDA D | 1,669 | 42.4984 | 32.6438 | 43.5486 | +1.0502 |
| UDD5 | 40 | 50.0877 | 42.0858 | 48.1441 | -1.9436 |
| LandCover.ai | 1,602 | 59.5201 | 53.6367 | 62.5223 | +3.0022 |

The two other readouts, with the same exact vocabularies, show the same broad
pattern. Each cell is all20 → new15 (difference in parentheses), mIoU %:

| Dataset/split | Geometry | Multiscale |
|---|---:|---:|
| OEM | 44.61 → 44.72 (+0.11) | 45.65 → 45.58 (-0.07) |
| FLAIR-1 | 43.81 → 44.31 (+0.50) | 44.13 → 45.10 (+0.97) |
| LoveDA P | 64.96 → 64.82 (-0.14) | 67.39 → 67.31 (-0.07) |
| LoveDA D | 42.66 → 43.74 (+1.08) | 43.97 → 45.20 (+1.23) |
| UDD5 | 50.56 → 48.62 (-1.94) | 51.04 → 49.70 (-1.34) |
| LandCover.ai | 59.23 → 62.52 (+3.29) | 59.51 → 62.33 (+2.82) |

GEAR-OV per-class IoU changes, all20 → new15 (percentage points):

- OEM: bareland -1.74, rangeland +2.76, developed space -3.53,
  road -0.38, tree +1.94, water +1.54, agriculture land -0.50,
  building +0.71.
- FLAIR-1: building +1.09, pervious +0.69, impervious +1.75,
  bare soil +6.94, water -0.06, coniferous ~0, deciduous +0.13,
  brushwood -1.06, vineyard -0.50, herbaceous +3.28,
  agricultural land -5.27, plowed ~0.
- LoveDA P: building +1.21, road +11.12, water -8.26, barren -0.44,
  tree -4.31, farm -1.30.
- LoveDA D: background +1.25, building +1.69, road +8.01,
  water -3.98, barren +1.62, tree -1.62, farm +0.39.
- UDD5: vegetation -1.46, building -6.04, road -15.21,
  vehicle +9.07, other +3.92.
- LandCover.ai: background +1.41, building +1.98, woodland +0.36,
  water +0.61, road +10.66.

New-screen wall times / peak CUDA memory (single GPU per dataset): OEM
810 s / 3,894 MiB, FLAIR-1 4,560 s / 3,860 MiB, LoveDA 4,732 s /
3,893 MiB, UDD5 1,025 s / 3,885 MiB, LandCover.ai 402 s / 3,861 MiB.
These timings are not speedups over the prior multi-shard all20 runs.

## Decision

The old selector's collapse was avoidable: new15 beats the old screen on
every evaluated split. But the new rule **does not consistently beat all20**:
GEAR-OV improves on four of six reported splits, falls on LoveDA P and UDD5,
and causes large opposing class changes. The patch-level AUC was useful for
finding a less destructive candidate, but did not predict full-set mIoU
reliably. This is not yet a defensible final alias-selection module.

Next necessary control: compare against several *equal-count random 15-word
vocabularies* using the same images and readouts. If the selected bank fails
to beat them, any observed gain could be from reducing vocabulary size rather
than selecting better evidence. Then inspect cross-class confusion (not only
per-alias local gain), especially LoveDA road/water and UDD5 road/vehicle,
before changing the rule. Do not tune the present 15-word result on these
target labels and present it as an independent test.
