# VIP / Geometry vocabulary and background factorial: VDD and Potsdam

Date: 2026-09-30. Frozen DINOv3 and DINO.text weights. This is a diagnostic
factorial, not a new model-selection result. VDD uses all 80 validation images;
Potsdam uses all 504 prepared RGB tiles. Every arm uses the identical fixed
sample sequence and mask mapping within its dataset. Checkpoint manifests match
the previous VIP or GEAR evaluations, and the four-shard Geometry/Multiscale
merges verify complete unique coverage.

Within each readout, only the vocabulary changes: pinned VIP official query
groups versus the previous dataset-specific 20-alias JSON. The background
factor is evaluated from the same image predictions: ordinary argmax versus
the fixed VIP rule (VDD: max probability < 0.35 -> other; Potsdam: < 0.25 ->
clutter). The numerical threshold is *not calibrated* for Geometry/Multiscale;
its transfer is a diagnostic, not a fair tuned baseline. VIP and GEAR retain
their own text templates, alias aggregation, visual readout, resize/sliding
windows and probability blending. Thus cross-readout rows are not a pure
architecture ablation. VIP official VDD has one query string per class;
Potsdam has counts [2,1,3,2,1,1]. The 20-word arms have 20 per class.

## Full mIoU percent

| Dataset | Readout | Vocabulary | Threshold off | Threshold on |
|---|---|---|---:|---:|
| VDD | VIP | official short | 46.5152 | 52.0647 |
| VDD | VIP | all 20 | 51.0697 | 51.4827 |
| VDD | Multiscale | official short | 26.0008 | 5.5571 |
| VDD | Multiscale | all 20 | 41.2235 | 3.3809 |
| Potsdam | VIP | official short | 43.5166 | 44.0643 |
| Potsdam | VIP | all 20 | 42.3519 | 41.6635 |
| Potsdam | Multiscale | official short | 27.5923 | 23.4476 |
| Potsdam | Multiscale | all 20 | 40.4433 | 24.2140 |

VIP official-short/threshold-on exactly reproduces the prior matched 52.0647
and 44.0643. Multiscale all20/threshold-off exactly reproduces 41.2235 and
40.4433. GEAR also produced Geometry controls in the result JSONs; their
all20/threshold-off mIoUs reproduce 38.8511 and 40.6892.

## Class competition

VDD ground-truth vehicle covers 0.52% and water 15.19% of scored pixels.
Each cell below is predicted area / precision / recall, in percent.

| Readout, vocabulary, threshold | Vehicle | Water |
|---|---:|---:|
| VIP short, off | 2.08 / 17.0 / 67.9 | 16.14 / 89.5 / 95.1 |
| VIP short, on | 0.87 / 32.2 / 54.1 | 15.00 / 94.6 / 93.5 |
| VIP 20, off | 1.32 / 26.1 / 66.4 | 16.69 / 86.8 / 95.3 |
| VIP 20, on | 0.71 / 39.7 / 54.0 | 15.35 / 91.0 / 92.0 |
| Multiscale short, off | 1.90 / 25.0 / 91.2 | 4.20 / 86.0 / 23.8 |
| Multiscale 20, off | 5.17 / 9.3 / 92.7 | 8.13 / 91.3 / 48.9 |

For Multiscale, short queries collapse vegetation recall to 1.2% and water
recall to 23.8%; all20 recovers them to 74.7% and 48.9%, but grows vehicle
false positives, especially from other. The VIP threshold improves the
official-short VDD mIoU by 5.5495 points, but improves VIP all20 by only
0.4130. Under Multiscale the same numeric threshold changes predicted other
area from 19.29% to 99.56% for all20, explaining its catastrophic drop.

Potsdam ground-truth car covers 1.96% and low vegetation 21.01% of pixels.

| Readout, vocabulary, threshold | Car area / P / R | Low vegetation area / P / R |
|---|---:|---:|
| VIP short, off | 0.64 / 58.6 / 19.1 | 16.67 / 76.1 / 60.4 |
| VIP short, on | 0.63 / 59.5 / 19.1 | 16.36 / 76.8 / 59.8 |
| VIP 20, off | 6.44 / 29.6 / 97.0 | 7.19 / 90.9 / 31.1 |
| VIP 20, on | 6.18 / 30.8 / 97.0 | 6.07 / 92.4 / 26.7 |
| Multiscale short, off | 8.19 / 23.5 / 97.9 | 1.79 / 94.7 / 8.1 |
| Multiscale 20, off | 19.34 / 10.1 / 99.4 | 9.61 / 92.4 / 42.3 |

Adding 20 aliases enlarges car in *both* readouts. In VIP it changes car
prediction from 0.64% to 6.44%, with most extra false positives from
impervious surface (0.82 million -> 16.14 million pixels). It also suppresses
low vegetation and reduces full mIoU by 1.1647 without threshold or 2.4008
with threshold. In Multiscale, however, the same expansion rescues building
IoU 30.39 -> 79.62 and low-vegetation IoU 8.04 -> 40.86, while worsening car
IoU 23.37 -> 10.10. The net all20 gain is 12.8510 points. Its fixed 0.25
threshold sends predicted clutter area from 5.34% to 56.95% and is therefore
not portable from VIP.

## Interpretation and next diagnostic

The two models do not exhibit the same vocabulary preference. VDD VIP all20
even beats official short *without* threshold, while Potsdam VIP prefers short
for overall mIoU. Our readout needs broad coverage for vegetation, water and
building, yet extra car/vehicle evidence increases cross-class false positives.
The effect is class-conditional and competitive, not a universal target alias
count. These full-vocabulary swaps cannot identify which individual alias is
harmful. The next controlled diagnostic should hold one readout and all other
class vocabularies fixed while removing one alias or one semantic alias family,
tracking its own-class true positives and rival-class false positives. Do not
choose retained aliases from these validation labels and then report the same
split as independent validation.

Raw outputs: `research/vip_gear_vocab_background_20260930/{vdd,potsdam}_{vip,gear}_{official,all20}.json`.
Code: `DINOtool/scripts/eval_vip_official_eight.py`,
`DINOtool/scripts/eval_gear_vocab_background.py`, and the two dedicated A800
launchers. VIP adapter unit tests (4) and one-image GPU smokes passed; pytest
is unavailable in the A800 environment.
