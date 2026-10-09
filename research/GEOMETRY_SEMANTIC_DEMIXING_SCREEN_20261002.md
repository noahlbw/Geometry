# Geometry joint semantic demixing: verified results

Signature `geometry-joint-semantic-residual-demixing-v1-20261002`. Phase `screen`. All eight outputs verified.

## Model and scope

Unchanged frozen DINOv3/DINO.text -> original Geometry descriptors Y and relation G -> joint all-alias nonnegative reconstruction under local/Geometry residual observations -> per-class explained descriptor norm -> native512/128/Hann probability assembly.

Fixed objective `.25*(||Y-XT||^2+||G(Y-XT)||^2)+.5*.01*||X||^2`, X>=0. No target-label fitting, external teacher, crop, hard alias deletion or class quota. All20 aliases per class remain. A zero coefficient is a query-local explanation, not a globally removed word.

This is a96-image development screen (UDD5 full40, seven domains8 each), not full eight-domain SOTA evidence, unless phase is explicitly full. LoveDA D enters the equal-domain mean once; P is separate. Corrected-IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. Matched VIPProxy_Two is not complete official VIP. No NNLS/FISTA originality claim.

## Matched mIoU

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | TextOnlyDemix | MeanLogit_Demix | ShuffledGeometryDemix | Geometry_SemanticDemix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 32.5585 | 32.8817 | 32.3360 | 32.6434 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 44.4430 | 44.0518 | 42.7113 | 44.2334 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 47.6291 | 48.5262 | 47.0963 | 48.1350 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 36.8823 | 37.3084 | 36.2111 | 36.7299 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 58.6148 | 60.2012 | 57.5717 | 58.9453 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 33.7564 | 34.9347 | 33.3986 | 33.7734 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 53.0000 | 53.0491 | 50.0854 | 52.6004 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 53.4830 | 56.9888 | 53.6635 | 53.9376 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 36.7301 | 38.3761 | 34.4530 | 36.7334 |

| Method | Equal-domain mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| TextOnlyDemix | 42.310300 |
| MeanLogit_Demix | 43.264592 |
| ShuffledGeometryDemix | 41.244403 |
| Geometry_SemanticDemix | 42.348325 |

Primary wins vs Geometry: 3/8. Promotion passed: False.

Primary mean delta versus Geometry: -1.482244pp.

All matched Geometry/SCLIP/VIPProxy per-image baseline matrices exactly reproduce the historical reference; complete unique fixed sample coverage, checkpoints, vocabularies and configs verified.

## vdd/vdd class competition

Beneficial/harmful changed pixels: 6577808/6542437. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 25.3949 | 24.7295 | -0.6654 | 38.928/42.213 | 33.567/48.434 | +1607520 | +7655942 |
| wall | 11.8982 | 17.8932 | 5.9950 | 11.984/94.332 | 18.118/93.519 | -15213 | -5055967 |
| road | 24.1970 | 20.3932 | -3.8038 | 24.741/91.668 | 20.863/90.059 | -21304 | +831182 |
| vegetation | 62.0257 | 72.4379 | 10.4122 | 84.684/69.863 | 89.440/79.213 | +1938764 | -680655 |
| vehicle | 5.6965 | 13.7367 | 8.0402 | 5.697/99.956 | 13.737/99.961 | +11 | -2329180 |
| roof | 60.7577 | 58.5064 | -2.2513 | 88.015/66.238 | 89.683/62.728 | -937446 | -481744 |
| water | 33.6534 | 20.8069 | -12.8465 | 93.059/34.520 | 88.766/21.370 | -2536961 | +25051 |

## potsdam/potsdam class competition

Beneficial/harmful changed pixels: 858354/568806. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 63.2560 | 11.5685 | 75.391/62.178 | 84.459/71.589 | +272614 | -206341 |
| building | 78.2026 | 81.3677 | 3.1651 | 80.677/96.226 | 84.029/96.253 | +263 | -46402 |
| low vegetation | 33.4221 | 20.8390 | -12.5831 | 72.470/38.283 | 81.878/21.847 | -252218 | -148964 |
| tree | 62.5813 | 65.5548 | 2.9735 | 90.654/66.897 | 88.428/71.706 | +92130 | +47639 |
| car | 11.6582 | 22.5028 | 10.8446 | 11.672/99.026 | 22.580/98.500 | -987 | -771568 |
| clutter | 4.5652 | 11.8803 | 7.3151 | 7.745/10.007 | 13.774/46.355 | +177746 | +836088 |

## udd5/udd5 class competition

Beneficial/harmful changed pixels: 29203204/53369047. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 77.2000 | -5.2937 | 96.486/85.049 | 96.053/79.729 | -6931924 | +233137 |
| building | 83.9035 | 76.2137 | -7.6898 | 88.360/94.329 | 93.304/80.623 | -23673571 | -11469269 |
| road | 45.7856 | 25.6887 | -20.0969 | 70.678/56.522 | 66.262/29.554 | -15910528 | -4956468 |
| vehicle | 10.0092 | 29.1140 | 19.1048 | 10.032/97.748 | 29.478/95.929 | -64056 | -22789138 |
| other | 30.5846 | 32.4585 | 1.8739 | 52.854/42.059 | 37.099/72.184 | +22414236 | +63147581 |

Geometry non_residual_mean_iou_percent: 55.548.

Geometry_SemanticDemix non_residual_mean_iou_percent: 52.0541.

## oem/oem class competition

Beneficial/harmful changed pixels: 464806/700952. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +272146 |
| rangeland | 47.7780 | 47.0110 | -0.7670 | 65.294/64.041 | 64.854/63.082 | -14096 | +2160 |
| developed space | 28.5294 | 20.8533 | -7.6761 | 61.562/34.713 | 72.276/22.666 | -207147 | -223192 |
| road | 40.8291 | 39.1937 | -1.6354 | 45.753/79.140 | 43.063/81.350 | +7743 | +48086 |
| tree | 56.0657 | 62.2623 | 6.1966 | 87.448/60.973 | 81.083/72.844 | +231747 | +160915 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -321 |
| agriculture land | 79.1353 | 62.2202 | -16.9151 | 90.641/86.177 | 85.744/69.400 | -215851 | +33979 |
| building | 62.4971 | 62.2989 | -0.1982 | 65.584/92.995 | 67.674/88.692 | -38542 | -57627 |

## loveda/P class competition

Beneficial/harmful changed pixels: 108870/198245. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 64.4987 | 64.7393 | 0.2406 | 65.367/97.982 | 67.776/93.528 | -1333 | -2228 |
| road | 66.9348 | 48.7321 | -18.2027 | 67.652/98.440 | 49.446/97.123 | -4295 | +170259 |
| water | 81.1851 | 76.8707 | -4.3144 | 86.764/92.661 | 81.804/92.726 | +617 | +62074 |
| barren | 24.3403 | 25.2839 | 0.9436 | 63.873/28.226 | 87.633/26.219 | -6715 | -41040 |
| tree | 53.7307 | 52.4408 | -1.2899 | 64.518/76.267 | 74.454/63.946 | -62424 | -101347 |
| farm | 86.2626 | 85.6051 | -0.6575 | 95.330/90.069 | 95.230/89.441 | -15225 | +1657 |

## loveda/D class competition

Beneficial/harmful changed pixels: 379170/826160. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 34.1983 | 28.7350 | -5.4633 | 65.750/41.611 | 61.563/35.017 | -243361 | +6930 |
| building | 31.1862 | 34.5068 | 3.3206 | 31.858/93.665 | 35.599/91.834 | -548 | -10238 |
| road | 45.4818 | 29.7218 | -15.7600 | 45.835/98.335 | 29.989/97.095 | -4044 | +360108 |
| water | 61.9193 | 51.8531 | -10.0662 | 65.955/91.007 | 57.215/84.693 | -60386 | +156446 |
| barren | 14.5296 | 15.4451 | 0.9155 | 59.063/16.157 | 63.355/16.960 | +2689 | -4645 |
| tree | 26.8904 | 21.2873 | -5.6031 | 30.053/71.873 | 24.744/60.378 | -58243 | +82854 |
| farm | 54.9851 | 54.8649 | -0.1202 | 69.572/72.395 | 72.850/68.967 | -83097 | -144465 |

Geometry foreground_mean_iou_percent: 39.1654.

Geometry_SemanticDemix foreground_mean_iou_percent: 34.6132.

## vaihingen/vaihingen class competition

Beneficial/harmful changed pixels: 617611/289995. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 57.6752 | 8.4750 | 82.397/54.979 | 74.751/71.629 | +405119 | +302889 |
| building | 74.6610 | 80.6415 | 5.9805 | 75.500/98.533 | 81.801/98.272 | -4470 | -173190 |
| low vegetation | 46.7115 | 39.5271 | -7.1844 | 92.073/48.669 | 97.203/39.982 | -175705 | -61479 |
| tree | 71.8329 | 68.3577 | -3.4752 | 82.551/84.692 | 73.268/91.072 | +105949 | +254540 |
| car | 8.7566 | 16.8006 | 8.0440 | 8.773/97.975 | 16.944/95.192 | -3277 | -650376 |

## landcoverai/landcoverai class competition

Beneficial/harmful changed pixels: 42698/155200. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 75.4554 | -7.1536 | 96.651/85.044 | 92.585/80.309 | -67287 | +49517 |
| building | 34.9562 | 36.3162 | 1.3600 | 35.044/99.292 | 36.527/98.439 | -271 | -4126 |
| woodland | 78.3638 | 64.5499 | -13.8139 | 86.499/89.284 | 70.810/87.953 | -5975 | +100205 |
| water | 93.6635 | 72.2591 | -21.4044 | 93.664/100.000 | 91.180/77.689 | -38701 | +1300 |
| road | 14.9318 | 21.1073 | 6.1755 | 15.629/77.002 | 22.634/75.781 | -268 | -34394 |

Geometry foreground_mean_iou_percent: 55.4788.

Geometry non_residual_mean_iou_percent: 55.4788.

Geometry_SemanticDemix foreground_mean_iou_percent: 48.5581.

Geometry_SemanticDemix non_residual_mean_iou_percent: 48.5581.

## flair1/flair1 class competition

Beneficial/harmful changed pixels: 126614/164396. These totals do not determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 58.5374 | 8.8395 | 50.726/96.082 | 59.556/97.161 | +1623 | -41166 |
| pervious surface | 57.0621 | 49.1712 | -7.8909 | 92.130/59.986 | 85.604/53.604 | -22958 | +13995 |
| impervious surface | 52.6509 | 42.4735 | -10.1774 | 62.624/76.777 | 47.755/79.340 | +8802 | +140721 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -23195 |
| water | 73.2328 | 57.7987 | -15.4341 | 77.143/93.526 | 64.737/84.357 | -8527 | +16960 |
| coniferous | 43.0233 | 42.1986 | -0.8247 | 61.711/58.690 | 70.452/51.273 | -874 | -1757 |
| deciduous | 54.0229 | 57.5281 | 3.5052 | 78.972/63.099 | 74.410/71.717 | +30750 | +28058 |
| brushwood | 18.6136 | 14.8169 | -3.7967 | 23.421/47.556 | 24.777/26.932 | -21264 | -76014 |
| vineyard | NA | NA | NA | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 57.6983 | -2.5346 | 94.317/62.501 | 97.048/58.729 | -25334 | -13296 |
| agricultural land | 18.7339 | 23.8446 | 5.1107 | 21.647/58.198 | 28.773/58.198 | +0 | -4256 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -2268 |

## Numerical diagnostics and costs

| Dataset | Max shard seconds | Sum shard seconds | Peak MiB | Mean iterations | Mean relative KKT | Active alias coefficient fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 700.294 | 700.294 | 3819.585 | 598.29 | 0.0004692 | 0.03701 |
| potsdam | 61.461 | 61.461 | 3817.283 | 525.33 | 0.0004539 | 0.03636 |
| udd5 | 677.795 | 2627.026 | 3818.417 | 490.53 | 0.0004636 | 0.04007 |
| oem | 69.868 | 69.868 | 3818.889 | 693.61 | 0.0004709 | 0.02999 |
| loveda | 115.934 | 115.934 | 3820.627 | 511.28 | 0.0004795 | 0.04256 |
| vaihingen | 55.995 | 55.995 | 3816.034 | 494.11 | 0.0004671 | 0.03913 |
| landcoverai | 6.213 | 6.213 | 3716.371 | 500.88 | 0.0004580 | 0.04754 |
| flair1 | 12.508 | 12.508 | 3718.195 | 893.00 | 0.0004735 | 0.01869 |

Primary-only single-window medians Geometry/primary: 0.024723/0.264699s (10.7067x). Primary peak allocated memory 3716.373MiB.

Evaluation durations include all control heads and output assembly; summed GPU durations are allocated runtime, not kernel-active GPU time. Smoke has no target-mask loads, and both FP32/bf16 immutable-cache and decoder-identity checks have error0.

## Decision and limits

Compared with TextOnlyDemix the primary mean changes by +0.038025pp; compared with fixed same-source logit fusion it changes by -0.916266pp. The matched ablations do not establish a useful overall Geometry-coupling advantage.

VDD and Potsdam reduce vehicle/car false activation, but VDD water and Potsdam low vegetation lose substantial correct coverage. Class competition is altered, not universally corrected. Potsdam and Vaihingen also trail the text-only decoder, so their gains over original Geometry cannot be attributed entirely to the additional residual coupling.

The geometric metric observes deterministic pooled residuals and adds no independent class identity. Explaining an inherited wrong feature does not certify the explanation's semantic correctness. A lower reconstruction objective or sparse coefficients alone is not a segmentation gain. Group mass differs from cosine/LME and is not calibrated by the frozen contrastive training objective.

A failed gate rejects this fixed candidate; no label-based ridge, scoring, word-count or domain adjustments are made. Original Geometry and prior best models remain preserved. Complete official-VIP superiority and CVPR originality remain research goals, not achieved claims.
