# Bounded wide-view competitive alias attenuation

Frozen words/profiles/gain/residual handling/long448. Inherited count offsets retained; bounded gate inside alias LSE, no redistribution. Explicit residual classes and PC60 frozen local rival remain unchanged. No new visual observations. Prior labelled development: exploratory.

| Protocol | Frozen | BoundedWideRival | Delta |
| --- | ---: | ---: | ---: |
| vdd | 55.2382 | 55.3916 | +0.1534 |
| potsdam | 49.7888 | 49.7155 | -0.0733 |
| voc21 | 70.3179 | 70.1599 | -0.1580 |
| context60 | 41.5558 | 41.5218 | -0.0340 |
| ade150 | 31.1076 | 31.1001 | -0.0075 |

Default remains retained; suppression bound is not an IoU guarantee.

## Verification and decision

All9138 protocol images completed without failures, with verified unique full coverage, per-image confusion sums, checkpoint/class identity, paired target counts and exact Frozen confusion replay. Only VDD improves; four domains decline. Do not promote this rule as a general improvement or select a VDD exception from these developed full-test scores. Existing default/best results are unchanged. Small differences are descriptive point estimates, not demonstrated statistical significance.

## Class-level evidence

| Protocol / class | Frozen IoU | BoundedWideRival IoU | Delta |
| --- | ---: | ---: | ---: |
| VDD road | 42.0360 | 42.6702 | +0.6342 |
| VDD vehicle | 28.7947 | 29.3075 | +0.5128 |
| VDD vegetation | 66.7112 | 66.2327 | -0.4785 |
| Potsdam low vegetation | 43.1983 | 42.2467 | -0.9516 |
| Potsdam car | 36.7792 | 36.7506 | -0.0286 |
| VOC21 aeroplane | 62.3288 | 63.1147 | +0.7859 |
| VOC21 train | 59.3679 | 60.2289 | +0.8610 |
| VOC21 motorbike | 72.3352 | 68.7340 | -3.6012 |
| PC60 mouse | 43.4396 | 42.8446 | -0.5950 |
| ADE150 armchair | 9.2313 | 8.5943 | -0.6370 |

VDD vehicle precision31.9787→32.6435%, recall74.3063→74.1456%, predicted area1.2097→1.1825%. Potsdam car precision36.9756→36.9471%, recall98.5758→98.5738%, area5.2384→5.2423%: this gate does not solve car overprediction there.

VOC background precision95.5793→95.2721%, recall93.3811→93.7922%, area71.6318→72.1792%. PC60 background precision37.6944→37.1852%, recall32.7488→33.4534%, area7.2329→7.4897%. Explicit residual scores are unchanged at the intervention, but other class-score changes still alter final background competition. Leaving residual queries untouched does not preserve residual predictions.

Improved/declined class counts: VDD4/3; Potsdam3/3; VOC12/9; PC60 26/33 (one unchanged); ADE59/91. Aggregate confusion results show a coverage/precision trade-off, not a universal semantic advantage. They do not isolate which individual aliases caused changes.

## Consequence for the model objective

The bounded intervention avoids the large calibration shifts of count normalization and the stronger modulation of the rejected self-consistency gain rule, but being smaller is not evidence of useful semantic selection. Own/rival text centroid separation is not a certificate that an alias is visually wrong on a patch. Increasing the gate strength or tuning per-dataset exceptions on these scores would not resolve that missing evidence.

Retain the existing task-conditioned deployment. Further work must supply a distinct, testable source of alias/branch reliability or a justified task-level calibration objective, with one frozen rule and matched vocabulary/information-budget controls. The current experiment does not establish that all dynamic weights are ineffective, nor that every alias helps every class. No standalone timing is claimed for this rejected candidate.
