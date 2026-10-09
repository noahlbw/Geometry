# Four-arm local readout diagnosis

Date: 2026-09-29. This is a frozen inference diagnosis on the existing VDD and
Potsdam validation protocols, not a proposed segmentation module.

## Locked 2 x 2 intervention

| Arm | Text evidence | Patch connectivity |
| --- | --- | --- |
| C_dense | Canonical class name only | Existing dense Geometry |
| A_dense | All 20 fixed aliases per class | Existing dense Geometry |
| C_sparse | Canonical class name only | Sparse Geometry connectivity |
| A_sparse | All 20 fixed aliases per class | Sparse Geometry connectivity |

All arms share the frozen DINOv3/DINO.text weights, two intervened vision-head
blocks, preserved patch-to-prefix access, feature/position Geometry logits,
512-pixel tiles, 128-pixel overlap, bilinear upscaling, Hann probability blend,
and the existing labels and class order. The sparse arm changes only which
patch-to-patch edges are available: normalized backbone feature cosine must
exceed 1.5 times its query-row mean, and self edges are always retained.
The Geometry logits on surviving edges are unchanged. This is a VIP-inspired
sparsity control, **not** the complete official VIP visual head or evaluation.
The alias aggregation uses normalized log-mean-exp at temperature 0.07 in both
canonical and all-20 arms. Target masks are read only after whole-image
prediction for metric and error-intersection counts.

## Fixed-seed screen

| Dataset | Images | C_dense | A_dense | C_sparse | A_sparse | Four-arm shared error | Oracle any-correct accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VDD | 16 | 30.2246 | 39.6995 | 29.8498 | 39.4864 | 31.47% | 68.53% |
| Potsdam | 32 tiles | 20.7203 | 39.1428 | 20.7649 | 39.4089 | 29.84% | 70.16% |

Scores are mIoU percentages. The oracle is a GT-only diagnostic: it counts a
pixel as recoverable if at least one of the four arms is correct. It cannot be
used as an inference rule. On VDD, the all-20 dense arm improves mIoU but its
vehicle IoU is 10.73 versus 33.81 for canonical dense. On Potsdam, the car IoU
is 9.69 for all-20 dense versus 5.96 for canonical dense. Sparse connectivity
keeps about 12.7% of possible patch edges on VDD and 13.2% on Potsdam; its
mIoU effect is small and changes sign across these screens.

The full error report includes a 16-cell correctness pattern over the four
arms, the same patterns conditioned on each true class, exclusive corrections,
and pairwise useful/harmful changes. The all-four-wrong intersection is 60.43M
of 192M valid VDD screen pixels and 9.55M of 32M Potsdam screen pixels. These
large shared-error sets suggest that switching between these four readouts
alone has limited headroom. A different local observation remains a hypothesis
until the full evaluations confirm this pattern.

## Full evaluation status

Both full evaluations have complete unique sample coverage. `A_dense` exactly
reproduces the earlier two-depth Geometry references: 38.8511 VDD mIoU and
40.6892 Potsdam mIoU. VDD has 80 validation images; Potsdam has 504 validation
tiles from 14 source images. Published VIP scores remain protocol-incomparable
with this evaluator.

| Full VDD, 80 images | C_dense | A_dense | C_sparse | A_sparse |
| --- | ---: | ---: | ---: | ---: |
| mIoU | 31.3999 | **38.8511** | 30.9045 | 38.6295 |
| Pixel accuracy | 44.03 | **61.70** | 43.20 | 61.40 |

| Full Potsdam, 504 tiles | C_dense | A_dense | C_sparse | A_sparse |
| --- | ---: | ---: | ---: | ---: |
| mIoU | 21.2713 | 40.6892 | 21.2980 | **40.9438** |
| Pixel accuracy | 42.25 | 64.41 | 42.26 | **64.69** |

All four arms are wrong on 292,098,006 of 960,000,000 valid VDD pixels
(30.43%). The GT-only any-correct oracle reaches 69.57% pixel accuracy,
7.87 points above the best actual arm. The sparse graph retains about 12.01%
of patch edges, but loses 0.2216 mIoU against all-20 dense. Against A_dense,
A_sparse fixes 2,457,209 pixels relative to A_dense and harms 5,370,794; both are wrong
with different labels on another 3,955,145 changed pixels. This is exploratory
diagnostic evidence, not a claim of a usable oracle selection mechanism.

| VDD class | C_dense IoU | A_dense IoU | Four-arm shared error within class |
| --- | ---: | ---: | ---: |
| other | 18.19 | 15.25 | 37.09% |
| wall | 4.47 | 22.97 | 15.08% |
| road | 52.72 | 48.57 | 5.38% |
| vegetation | 28.03 | 70.03 | 25.61% |
| vehicle | 36.78 | 9.48 | 7.54% |
| roof | 52.76 | 65.85 | 22.34% |
| water | 26.85 | 39.82 | 57.83% |

The shared-error percentage conditions on the true class and is a pixel-error
measure, whereas IoU also includes false positives from other classes. Vehicle
illustrates that distinction: a low shared-error rate does not mean the all-20
arm has reliable vehicle precision. VDD vehicle occupies 0.52% of GT pixels,
but A_dense predicts 5.03% vehicle with 9.56% precision. C_dense predicts
1.08% with 39.83% precision. More aliases do not help every class.

On Potsdam, all four arms are wrong on 142,887,975 of 504,000,000 valid pixels
(28.35%). The GT-only any-correct oracle reaches 71.65% pixel accuracy,
6.96 points above A_sparse. A_sparse retains about 13.30% of patch edges and
gains 0.2546 mIoU over A_dense. Relative to A_dense, it fixes 2,844,759
previously wrong pixels but damages 1,445,820 correct pixels; both are wrong
with different labels on another 1,990,371 changed pixels. Its per-class IoU
is 53.18 impervious surface, 79.53 building, 40.50 low vegetation, 52.62 tree,
11.06 car, and 8.77 clutter.
Potsdam car occupies 1.97% of GT pixels, while A_sparse predicts 17.62% car
with 11.07% precision. Sparse connectivity does not solve the object expansion.

| Full dataset | Best arm pixel accuracy | Four-arm shared error | Fraction of best-arm errors shared | Oracle any-correct accuracy | Parallel wall / summed GPU time | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VDD, 80 images | 61.70% (A_dense) | 30.43% | 79.4% | 69.57% | 213 / 845 s | 3731 MiB |
| Potsdam, 504 tiles | 64.69% (A_sparse) | 28.35% | 80.3% | 71.65% | 117 / 464 s | 3730 MiB |

## Decision for the second module

The 20-alias text basis is essential for overall accuracy, but some classes,
especially VDD vehicle, are harmed. Sparsifying the same Geometry relation is
near neutral and has opposite signs across domains. Approximately four fifths
of the best arm's mistakes are shared by all four arms. A module that only
chooses between these four outputs cannot fix that shared set, even with GT
selection. The next candidate therefore needs evidence unavailable to this
2 x 2 set, such as a new local view or resolution, and must show that it
corrects shared-error pixels without expanding car/vehicle false positives.
This diagnosis does not prove that every possible context or calibration rule
fails, nor does the GT oracle provide a deployable routing rule or an mIoU
upper bound. Use a single predeclared rule across datasets in the next test.

## Artifacts

- `DINOtool/dinotool/tcpr.py`: dense/sparse Geometry connectivity switch.
- `DINOtool/dinotool/local_readout_audit.py`: exact error intersections.
- `DINOtool/scripts/eval_local_readout_2x2.py`: matched four-arm inference.
- `DINOtool/scripts/merge_local_readout_2x2_shards.py`: coverage-checked merge.
- `DINOtool/scripts/launch_local_readout_a800.sh`: A800 4-7 launcher.
- `research/local_readout_2x2_screen_vdd_20260929_results.json`.
- `research/local_readout_2x2_screen_potsdam_20260929_results.json`.
- `research/local_readout_2x2_full_vdd_20260929_results.json`.
- `research/local_readout_2x2_full_potsdam_20260929_results.json`.
