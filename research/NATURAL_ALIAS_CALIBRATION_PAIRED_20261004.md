# Frozen-reader natural alias/count paired experiment

The Geometry/fine-rival/coupled model is unchanged. New launches use physical GPUs 4-7. Source aliases, templates, tau/tem and background threshold are held fixed. Count calibration is selected from at most64 unlabeled images before masks are read. The full-normalization arm is a diagnostic counterfactual, not a label-selected deployment setting.

## Image-only selection

| Protocol | Images | Chosen strength | Half: pseudo-error delta | Half: upper95 delta | Full: pseudo-error delta | Full: upper95 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| voc20 | 64 | 0.0 | -0.5904pp | +0.0774pp | -0.3944pp | +0.3781pp |
| voc21 | 64 | 0.0 | +2.5740pp | +3.7497pp | +6.0637pp | +8.1872pp |
| ade150 | 64 | 1.0 | -1.8123pp | -0.3456pp | -2.5328pp | -0.9782pp |
| coco_stuff171 | 64 | 0.0 | -0.5177pp | +0.0135pp | -0.5783pp | +0.3772pp |
| coco_object81 | 64 | 0.0 | +0.4424pp | +1.2782pp | +1.5835pp | +2.7265pp |

## Pilot paired results

All listed arms use one union-of-class-support mask. When an identical-sample original suite result exists, its arms also contribute to that common support and its finite VIP comparator is included. Pilot scores are diagnostic, not full benchmark numbers.

| Protocol | Images | Source | Count-selected | Full norm diagnostic | Full norm delta | Curated words | VIP | Common classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voc20 | 16 | 59.8371 | 59.8371 | 58.5861 | -1.2510 | pending | 61.7996 | 20 |
| voc21 | 16 | 40.5066 | 40.5066 | 40.0506 | -0.4560 | pending | 47.9158 | 21 |
| ade150 | 16 | 13.0995 | 13.0777 | 13.0777 | -0.0218 | 13.5018 | 10.9411 | 139 |
| coco_stuff171 | 16 | 16.1602 | 16.1602 | 16.5132 | +0.3530 | 16.1573 | 18.1272 | 150 |

voc20: verified unique coverage=True; matched original source confusions=True; wall=14.52s; peak CUDA memory=6716.29MiB.

voc21: verified unique coverage=True; matched original source confusions=True; wall=14.75s; peak CUDA memory=6716.29MiB.

ade150: verified unique coverage=True; matched original source confusions=True; wall=211.94s; peak CUDA memory=7161.78MiB.

coco_stuff171: verified unique coverage=True; matched original source confusions=True; wall=235.64s; peak CUDA memory=8287.72MiB.

## Full paired results

All listed arms use one union-of-class-support mask. When an identical-sample original suite result exists, its arms also contribute to that common support and its finite VIP comparator is included. Pilot scores are diagnostic, not full benchmark numbers.

| Protocol | Images | Source | Count-selected | Full norm diagnostic | Full norm delta | Curated words | VIP | Common classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ade150 | 2000 | 28.8354 | 28.2600 | 28.2600 | -0.5754 | 28.8891 | 29.1387 | 150 |

ade150: verified unique coverage=True; matched original source confusions=True; wall=23744.78s; peak CUDA memory=7251.28MiB.

## Interpretation

A retained strength of zero leaves the source unchanged and cannot be presented as an IoU gain. Count normalization only removes the explicit broad log-sum-exp multiplicity offset. It does not neutralize the count-dependent salience profile or make arbitrary wrong aliases harmless. Text curation is a separate word-only candidate: some ADE base/apparel synonyms and the empty COCO-Stuff query are removed without changing source parameters. Measured full effects appear above; cross-protocol gains are not established.

The investigation was motivated by developed natural-image results. Image-only selection is transductive; these diagnostics are exploratory, not untouched independent validation or proof of CVPR readiness.

Artifacts: research/natural_count_calibration_20261004/{selection,pilot,full}/PROTOCOL/.
