# Fixed-model natural text adaptation

Geometry and the retained fine-rival/coupled equations are unchanged. Actual unequal alias counts are supported without padding. Selection combines canonical taxonomy, curated lexical synonyms and attributed official VIP query pools. Calibration uses at most64 unlabeled evaluation images. Target masks never choose aliases, templates, tau/tem or threshold.

This is transductive self-calibration, not a guarantee of semantic correctness or independent validation. The fixed16 pilot determines whether this single frozen candidate proceeds to full evaluation; it does not select per-domain winners. The VIP comparator uses its own official resize/queries/settings with explicit empty-row numerical repair, not certified reproduction of the published table.

Calibration uses a no-admission coupled-score proxy and cross-view pseudo-labels, not the exact full fine-rival output. Unequal counts can still affect the inherited wide-view log-sum-exp scores; actual-count support is not proof of count-invariant predictions.

## Frozen calibration

| Dataset | Template | tau | tem | Background threshold | Pool aliases | Retained aliases | Per-class count range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| voc20 | seg_template | 5.0 | 0.3 | 0.000000 | 116 | 116 | 3-23 |
| voc21 | seg_template | 5.0 | 0.3 | 0.237790 | 172 | 172 | 3-56 |
| ade150 | seg_template | 5.0 | 0.3 | 0.000000 | 382 | 382 | 1-24 |
| coco_stuff171 | seg_template | 5.0 | 0.3 | 0.000000 | 639 | 638 | 1-17 |
| coco_object81 | seg_template | 5.0 | 0.3 | 0.093264 | 325 | 324 | 3-32 |

## Pilot results

| Dataset | Images | Geometry_Pool | Coupled_Pool | RivalFine_Pool | Adaptive_NoThreshold | Adaptive_RivalFine | VIP_Official_Finite | VIP_Official_NoThreshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voc20 | 16 | 58.5696 | 59.8790 | 59.8778 | 59.8371 | 59.8371 | 61.7996 | 61.7996 |
| voc21 | 16 | 20.6137 | 40.1591 | 40.3215 | 38.2802 | 40.5066 | 47.9158 | 46.2789 |
| ade150 | 16 | 13.2944 | 12.6107 | 12.5814 | 13.0995 | 13.0995 | 10.9411 | 11.7786 |
| coco_stuff171 | 16 | 16.3528 | 16.4359 | 16.4534 | 16.1602 | 16.1602 | 18.1272 | 18.0467 |
| coco_object81 | 16 | 9.9131 | 14.9862 | 15.1241 | 14.3608 | 14.2865 | 19.8103 | 17.6664 |

## Full results

| Dataset | Images | Geometry_Pool | Coupled_Pool | RivalFine_Pool | Adaptive_NoThreshold | Adaptive_RivalFine | VIP_Official_Finite | VIP_Official_NoThreshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voc20 | 1449 | 84.6048 | 91.8149 | 91.8146 | 89.6651 | 89.6651 | 92.5051 | 92.5051 |
| voc21 | 1449 | 28.5474 | 57.2839 | 57.3617 | 48.8813 | 63.6504 | 73.2591 | 67.7496 |
| ade150 | 2000 | 26.3876 | 28.7972 | 28.7845 | 28.8354 | 28.8354 | 29.1387 | 26.8429 |
| coco_object81 | 5000 | 24.6716 | 43.0603 | 43.2917 | 38.2471 | 47.1188 | 48.9955 | 43.7073 |

## Running coverage snapshot

These jobs are incomplete; progress is not a final benchmark result. Full tables above contain only verified merges.

| Dataset | Phase | Shard | Processed | Shard total |
| --- | --- | ---: | ---: | ---: |
| coco_stuff171 | full | 0 | 1920 | 2500 |
| coco_stuff171 | full | 1 | 1580 | 2500 |

## Matched-class pilot diagnosis

Ordinary pilot mIoU can change its denominator when a method stops predicting an absent class. The promotion gate uses one union-of-class-support mask for every arm. These scores are diagnostic and are not published full benchmark scores.

Pilot promotion passed: False. Mean adaptation gain over the unadapted pool: -0.093669pp.

The pilot table above uses these matched-class scores, not arm-dependent class denominators. Higher pseudo-label confidence or a sharper parameter setting is not itself evidence of higher IoU.

Full evaluation is an explicit paired diagnostic expansion. The failed pilot decision remains unchanged; no selection/configuration was replaced after seeing its metrics.

## Protocols Excluded From This Frozen Suite

These are the exclusions recorded at initialization, not the current data inventory.

- context59: [Errno 2] No such file or directory: '/data/test/datasets/VIP_natural/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'
- context60: [Errno 2] No such file or directory: '/data/test/datasets/VIP_natural/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'
- cityscapes19: Incorrect full validation coverage: cityscapes19
