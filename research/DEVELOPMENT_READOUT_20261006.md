# Fixed-Weight Readout And Vocabulary Development

Supervised hyperparameter development. ADE uses96 training images; other protocols use fixed validation development subsets. Their full scores include tuning images; disjoint complements are separate. Prior development means none is an untouched test claim. No backbone training or new fine views.

| Protocol | Dev source/count | Full reference | Full tuned | Heldout count | Heldout reference | Heldout tuned | Target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | fixed_labeled_validation_development_subset/16 | 53.4600 | 56.2807 | 64 | 54.6531 | 56.3804 | None |
| potsdam | fixed_labeled_validation_development_subset/64 | 45.9339 | 51.1828 | 440 | 45.9512 | 51.1164 | None |
| voc21 | fixed_labeled_validation_development_subset/64 | 29.6765 | 70.5894 | 1385 | 29.6305 | 70.4941 | None |
| context60 | fixed_labeled_validation_development_subset/64 | 36.7248 | 40.2949 | 5041 | 36.7208 | 40.2852 | None |
| coco_object81 | fixed_labeled_validation_development_subset/64 | 24.5733 | 46.8965 | 4936 | 24.4789 | 46.9080 | None |
| ade150 | ADE_training/96 | 24.6903 | 31.1076 | 2000 | 24.6903 | 31.1076 | 30.0 |
| context59 | fixed_labeled_validation_development_subset/64 | 40.9221 | 45.1541 | 5041 | 40.9363 | 45.1654 | 49.0 |
| voc20 | fixed_labeled_validation_development_subset/64 | 89.3515 | 92.7596 | 1385 | 89.3972 | 92.7056 | 92.0 |
| coco_stuff171 | fixed_labeled_validation_development_subset/64 | 29.8811 | 33.1979 | 4936 | 29.8257 | 33.1789 | None |

Current status: complete

Full-tuned and heldout-tuned are frozen development selections, not retrospective full-set maxima. Each JSON retains per-class IoU/precision/recall, exact baseline replay and actual encoding budgets. Suite multi-profile wall time is not deployed per-image latency.

## Frozen Selected Settings

| Protocol | Vocabulary/template | Geometry patch read | Coupling gain | Local temperature | Wide tau/tem | Background log-odds / rejection |
| --- | --- | ---: | ---: | ---: | --- | --- |
| VDD | focused20 / RS6 | 1 | 0.5 | 0.07 | 1 / 1 | 0 / 0.25 |
| Potsdam | focused20 / RS6 | 3 | 0.5 | 0.07 | 1 / 1 | 0 / 0 |
| VOC21 | qualified pool / segmentation | 2 | 1 | 0.05 | 1 / 1 | 0 / 0.2 |
| Context60 | qualified pool / segmentation | 2 | 1 | 0.07 | 1 / 0.3 | -2 / 0 |
| COCO-Object81 | original words / segmentation | 2 | 1 | 0.05 | 1 / 1 | -2 / 0.1 |
| ADE150 | qualified pool / segmentation | original Geometry | 0.5 | 0.07 | 1 / 1 | 0 / 0 |
| Context59 | qualified pool / segmentation | 2 | 1 | 0.07 | 1 / 0.3 | 0 / 0 |
| VOC20 | qualified pool / ImageNet | 2 | 1 | 0.1 | 1 / 1 | 0 / 0 |
| COCO-Stuff171 | qualified pool / segmentation | 1 | 0.5 | 0.07 | 1 / 1 | 0 / 0 |

The choices are one joint selection over the declared grid, so vocabulary,
template and readout effects cannot be separated from this table alone. ADE's
setting was chosen on96 ADE training images; its2000 validation images were not
used to choose it. Other protocols chose their setting using labelled validation
development subsets. In particular, the16-image VDD subset is small relative to
51 profiles and remains the highest overfit risk; the separate64-image complement
gain is positive but does not remove dataset-level validation reuse.

## Residual Background Check

| Protocol | Background IoU before/after | Precision before/after | Recall before/after | Predicted area before/after | Ground-truth area |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC21 | 22.96 / 89.24 | 98.63 / 94.39 | 23.03 / 94.24 | 17.12 / 73.20 | 73.32 |
| Context60 | 4.38 / 0.00 | 41.32 / 0.00 | 4.68 / 0.00 | 0.94 / 0.00 | 8.33 |
| COCO-Object81 | 15.68 / 77.47 | 97.78 / 80.55 | 15.73 / 95.29 | 11.14 / 81.90 | 69.23 |

The natural background issue is not yet solved uniformly. VOC21 recovers
background coverage, while Context60's selected pooled gain collapses residual
background and COCO-Object81 over-predicts it by12.67 area points. A single
global residual-class bias is therefore not a transferable correction.

## Goal And Comparator Readout

ADE150 reaches31.1076 against the30 goal. VOC20 reaches92.7596 against the92
goal. Context59 reaches45.1541 and misses the49 goal by3.8459 points; its largest
class losses are keyboard (-36.8726) and ground (-14.7482), despite gains on
mouse (+36.4074) and bird (+28.4998). Context60 reaches40.2949. VOC21 reaches
70.5894: a40.9129-point gain over this reader's original setting, but remains
2.6697 below the previously measured finite VIP value73.2591. COCO-Object81
reaches46.8965 and remains2.0990 below its previous local finite VIP value
48.9955. These VIP values are earlier local comparators, not published numbers
for every protocol.

VDD reaches56.2807: road rises17.8677 points and vehicle12.6423, while vegetation
falls16.1211 and water11.6445. Potsdam reaches51.1828 and all six classes improve;
car rises9.1973 and low vegetation6.1745. These totals are therefore useful
readout trade-offs, not evidence that every selected alias helps every class.

The next module question is specifically class-conditional competitive
calibration with an explicit coverage constraint: retain correct class support
while reducing the observed false activation. Context59/60 and COCO-Object81
show that choosing a background offset by aggregate mIoU can hide a severe
residual-class regression. VDD shows the corresponding risk among foreground
classes. Any further design should be selected by a declared joint objective
that reports per-class IoU and precision/recall alongside mIoU.
