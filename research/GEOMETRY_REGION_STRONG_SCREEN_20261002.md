# Geometry-supported strong semantic observer: results

Implementation: `geometry-region-so400m-footprint-joint-v1-20261002`. Phase: screen; status: complete.

Verified datasets: 8/8; pending: none.

## Model and evaluation boundary

Unchanged original Geometry defines connected supports and restricted pooling priors. Frozen SigLIP2 SO400M reads real detail/context RGB with its original trained semantic pool. Actual stride14 footprints are projected without stretching its unseen input border. A single fixed joint KL inference produces the primary output; all20 aliases remain. There is no class-specific or per-domain winner choice or target-label fitting.

This is a two-encoder observer-quality test, not a claim that the stronger borrowed checkpoint or standard KL objective is itself an original contribution. Confidence and view agreement are not correctness guarantees. WebLI provenance is disclosed; its evaluation-image overlap is not independently audited.

The screen uses96 images: full40 UDD5 and eight fixed-seed images per other domain. Unless phase is full, these are not full-set eight-domain benchmark numbers. All domains are development data. LandCover.ai replaces unlabeled iSAID. Matched VIPProxy_Two is not the official full VIP system.

## Verified mIoU

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | CropGlobalFusion | RegionSemantic | RegionFusion | Geometry_RegionStrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 31.2310 | 29.2803 | 32.0273 | 32.9216 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 29.7536 | 30.6748 | 37.1257 | 38.3952 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 44.3511 | 42.8095 | 46.5366 | 48.0467 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 30.9004 | 26.8914 | 30.6766 | 32.3310 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 52.7696 | 38.6959 | 49.5693 | 54.9680 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 34.8269 | 22.8551 | 34.0951 | 35.3908 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 39.1455 | 36.7251 | 44.0316 | 45.8222 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 52.7344 | 48.6051 | 56.1834 | 57.8141 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 26.9219 | 18.4193 | 27.4231 | 31.2528 |

## Cross-domain decision

LoveDA D enters the equal-domain mean once.

| Method | Equal-domain mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| CropGlobalFusion | 36.233102 |
| RegionSemantic | 32.032580 |
| RegionFusion | 38.512417 |
| Geometry_RegionStrong | 40.246801 |

Predeclared promotion gate passed: False.

| Gate check | Passed |
| --- | --- |
| mean_vs_Geometry | False |
| mean_vs_VIPProxy_Two | False |
| mean_vs_RegionFusion | True |
| retain_potsdam_potsdam | False |
| focus_potsdam_vs_Geometry | False |
| focus_potsdam_vs_RegionFusion | True |
| retain_oem_oem | False |
| retain_flair1_flair1 | False |
| retain_landcoverai_landcoverai | False |
| retain_loveda_P | False |
| retain_loveda_D | False |
| retain_vaihingen_vaihingen | False |
| retain_vdd_vdd | True |
| focus_vdd_vs_Geometry | True |
| focus_vdd_vs_RegionFusion | True |
| retain_udd5_udd5 | False |

A failed gate stops full rollout; no post-result parameter changes are permitted.

## Corrections and class competition

| Dataset/protocol | Delta vs Geometry | Delta vs VIPProxy | Delta vs RegionFusion | Beneficial | Harmful |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | +0.975445 | +1.816016 | +0.894373 | 7708458 | 6908070 |
| potsdam/potsdam | -1.957637 | -4.342546 | +1.269536 | 684491 | 757625 |
| udd5/udd5 | -2.508583 | -1.860397 | +1.510108 | 15963283 | 20168058 |
| oem/oem | -7.023354 | -6.658931 | +1.654392 | 226455 | 857735 |
| loveda/P | -7.857393 | -13.601247 | +5.398664 | 68332 | 229060 |
| loveda/D | -3.065007 | -0.276170 | +1.295701 | 362507 | 453473 |
| vaihingen/vaihingen | -4.410311 | -4.717177 | +1.790593 | 396020 | 632542 |
| landcoverai/landcoverai | -3.090767 | -1.493241 | +1.630683 | 47236 | 107259 |
| flair1/flair1 | -7.589929 | -8.090757 | +3.829690 | 63586 | 298387 |

Pixel correction totals measure pixel accuracy, not mIoU. Per-class coverage and false activation follow.

### vdd/vdd

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 25.3949 | 18.8077 | -6.5872 | 38.928/42.213 | 33.408/30.087 | -3133131 | -1615878 |
| wall | 11.8982 | 11.1512 | -0.7470 | 11.984/94.332 | 11.229/94.158 | -3273 | +964459 |
| road | 24.1970 | 24.4929 | 0.2959 | 24.741/91.668 | 25.229/89.358 | -30586 | -185469 |
| vegetation | 62.0257 | 74.0857 | 12.0600 | 84.684/69.863 | 85.283/84.946 | +3127481 | +419522 |
| vehicle | 5.6965 | 10.2431 | 4.5466 | 5.697/99.956 | 10.243/99.998 | +95 | -1765486 |
| roof | 60.7577 | 66.0229 | 5.2652 | 88.015/66.238 | 84.324/75.260 | +2410032 | +1327979 |
| water | 33.6534 | 25.6479 | -8.0055 | 93.059/34.520 | 90.228/26.381 | -1570230 | +54485 |

### potsdam/potsdam

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 53.2133 | 1.5258 | 75.391/62.178 | 76.831/63.385 | +34955 | -34228 |
| building | 78.2026 | 71.5198 | -6.6828 | 80.677/96.226 | 73.043/97.166 | +9175 | +125070 |
| low vegetation | 33.4221 | 34.8485 | 1.4264 | 72.470/38.283 | 61.625/44.507 | +95513 | +202139 |
| tree | 62.5813 | 53.3385 | -9.2428 | 90.654/66.897 | 91.883/55.976 | -209251 | -37403 |
| car | 11.6582 | 13.4411 | 1.7829 | 11.672/99.026 | 13.441/99.995 | +1815 | -197651 |
| clutter | 4.5652 | 4.0101 | -0.5551 | 7.745/10.007 | 6.793/8.915 | -5341 | +15207 |

### udd5/udd5

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 83.6526 | 1.1589 | 96.486/85.049 | 94.417/88.006 | +3853464 | +2745282 |
| building | 83.9035 | 83.1629 | -0.7406 | 88.360/94.329 | 85.511/96.804 | +4273946 | +6868688 |
| road | 45.7856 | 35.3421 | -10.4435 | 70.678/56.522 | 74.027/40.345 | -9544139 | -5482989 |
| vehicle | 10.0092 | 8.8446 | -1.1646 | 10.032/97.748 | 8.853/98.996 | +43958 | +5025534 |
| other | 30.5846 | 29.2314 | -1.3532 | 52.854/42.059 | 55.347/38.253 | -2832004 | -4951740 |

Geometry non_residual_mean_iou_percent: 55.5480.

Geometry_RegionStrong non_residual_mean_iou_percent: 52.7506.

### oem/oem

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +122677 |
| rangeland | 47.7780 | 44.1934 | -3.5846 | 65.294/64.041 | 64.249/58.605 | -79901 | -20997 |
| developed space | 28.5294 | 12.1092 | -16.4202 | 61.562/34.713 | 70.564/12.753 | -377608 | -281222 |
| road | 40.8291 | 35.3757 | -5.4534 | 45.753/79.140 | 41.892/69.459 | -33910 | +8811 |
| tree | 56.0657 | 47.2886 | -8.7771 | 87.448/60.973 | 86.495/51.058 | -193548 | -15235 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +2962 |
| agriculture land | 79.1353 | 78.5317 | -0.6036 | 90.641/86.177 | 88.546/87.412 | +15894 | +31008 |
| building | 62.4971 | 41.1492 | -21.3479 | 65.584/92.995 | 41.640/97.215 | +37793 | +783276 |

### loveda/P

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 64.4987 | 36.7058 | -27.7929 | 65.367/97.982 | 36.781/99.449 | +439 | +35622 |
| road | 66.9348 | 65.2195 | -1.7153 | 67.652/98.440 | 66.578/96.966 | -4805 | +5241 |
| water | 81.1851 | 75.6332 | -5.5519 | 86.764/92.661 | 79.676/93.712 | +10053 | +93428 |
| barren | 24.3403 | 19.3342 | -5.0061 | 63.873/28.226 | 41.953/26.395 | -6126 | +68788 |
| tree | 53.7307 | 51.3960 | -2.3347 | 64.518/76.267 | 66.626/69.216 | -35724 | -36838 |
| farm | 86.2626 | 81.5191 | -4.7435 | 95.330/90.069 | 95.304/84.930 | -124565 | -5513 |

### loveda/D

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 34.1983 | 33.2731 | -0.9252 | 65.750/41.611 | 67.461/39.634 | -72967 | -94426 |
| building | 31.1862 | 17.3486 | -13.8376 | 31.858/93.665 | 17.365/99.449 | +1731 | +81675 |
| road | 45.4818 | 42.2971 | -3.1847 | 45.835/98.335 | 42.868/96.947 | -4526 | +42367 |
| water | 61.9193 | 53.4942 | -8.4251 | 65.955/91.007 | 55.553/93.521 | +24047 | +266357 |
| barren | 14.5296 | 17.5398 | 3.0102 | 59.063/16.157 | 61.494/19.704 | +11870 | +3815 |
| tree | 26.8904 | 29.8541 | 2.9637 | 30.053/71.873 | 35.191/66.315 | -28159 | -228765 |
| farm | 54.9851 | 53.9288 | -1.0563 | 69.572/72.395 | 68.744/71.448 | -22962 | +19943 |

Geometry foreground_mean_iou_percent: 39.1654.

Geometry_RegionStrong foreground_mean_iou_percent: 35.7438.

### vaihingen/vaihingen

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 43.8377 | -5.3625 | 82.397/54.979 | 82.723/48.256 | -163577 | -40551 |
| building | 74.6610 | 63.7819 | -10.8791 | 75.500/98.533 | 64.154/99.098 | +9679 | +400758 |
| low vegetation | 46.7115 | 42.5761 | -4.1354 | 92.073/48.669 | 94.078/43.749 | -99518 | -29052 |
| tree | 71.8329 | 68.5467 | -3.2862 | 82.551/84.692 | 77.342/85.771 | +17912 | +120006 |
| car | 8.7566 | 10.3684 | 1.6118 | 8.773/97.975 | 10.400/97.110 | -1018 | -214639 |

### landcoverai/landcoverai

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 78.3505 | -4.2585 | 96.651/85.044 | 97.794/79.760 | -75082 | -16316 |
| building | 34.9562 | 28.6469 | -6.3093 | 35.044/99.292 | 28.714/99.194 | -31 | +19769 |
| woodland | 78.3638 | 78.4846 | 0.1208 | 86.499/89.284 | 83.724/92.615 | +14953 | +18265 |
| water | 93.6635 | 92.1034 | -1.5601 | 93.664/100.000 | 92.103/100.000 | +0 | +3137 |
| road | 14.9318 | 11.4851 | -3.4467 | 15.629/77.002 | 11.878/77.626 | +137 | +35168 |

Geometry foreground_mean_iou_percent: 55.4788.

Geometry non_residual_mean_iou_percent: 55.4788.

Geometry_RegionStrong foreground_mean_iou_percent: 52.6800.

Geometry_RegionStrong non_residual_mean_iou_percent: 52.6800.

### flair1/flair1

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 38.0843 | -11.6136 | 50.726/96.082 | 38.431/97.686 | +2413 | +95067 |
| pervious surface | 57.0621 | 44.0686 | -12.9935 | 92.130/59.986 | 76.363/51.029 | -32219 | +38386 |
| impervious surface | 52.6509 | 50.0162 | -2.6347 | 62.624/76.777 | 63.862/69.761 | -24094 | -21792 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +23583 |
| water | 73.2328 | 67.6471 | -5.5857 | 77.143/93.526 | 70.695/94.009 | +449 | +10470 |
| coniferous | 43.0233 | 36.0756 | -6.9477 | 61.711/58.690 | 60.880/46.962 | -1382 | -735 |
| deciduous | 54.0229 | 35.1738 | -18.8491 | 78.972/63.099 | 87.131/37.101 | -92768 | -40398 |
| brushwood | 18.6136 | 16.2240 | -2.3896 | 23.421/47.556 | 18.175/60.183 | +13019 | +119043 |
| vineyard | undefined | undefined | undefined | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 47.0049 | -13.2280 | 94.317/62.501 | 97.502/47.578 | -100219 | -17106 |
| agricultural land | 18.7339 | 9.4866 | -9.2473 | 21.647/58.198 | 10.180/58.198 | +0 | +19356 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +8927 |

## Diagnostics and cost

The evaluator executes every control together; its elapsed times are not deployed-primary latency. Aggregate time is elapsed GPU allocation, not measured kernel-active time.

| Dataset | Max shard seconds | Sum shard seconds | Peak allocated MiB | Influence | View agreement | Changed patch fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 1149.625 | 1149.625 | 8755.559 | 0.37572 | 0.55756 | 0.23352 |
| potsdam | 110.356 | 110.356 | 8753.389 | 0.36078 | 0.49494 | 0.25616 |
| udd5 | 5139.994 | 5139.994 | 8752.088 | 0.46268 | 0.64993 | 0.12369 |
| oem | 104.510 | 104.510 | 8754.083 | 0.45415 | 0.62815 | 0.22613 |
| loveda | 116.023 | 116.023 | 8755.413 | 0.23958 | 0.39877 | 0.14754 |
| vaihingen | 109.109 | 109.109 | 8751.146 | 0.43717 | 0.63221 | 0.20203 |
| landcoverai | 12.800 | 12.800 | 8746.908 | 0.40207 | 0.62612 | 0.08960 |
| flair1 | 13.471 | 13.471 | 8750.083 | 0.37665 | 0.48905 | 0.24609 |

### Real-checkpoint smoke and window timing

Original Geometry and native cache errors:0; trained full-support pool error: 0.0.729 actual observer tokens. No masks loaded.

Window medians: Geometry 0.025123s; primary 1.461670s; ratio 58.1795x.

Window-only timing; both encoders remain resident in both measurements. Not independently deployed Geometry peak memory or full-image latency.

Better joint inference than same-information fusion alone is insufficient. The model must also improve original Geometry and strong matched alternatives, with acceptable cross-domain losses and compute. These results cannot establish superiority to full official VIP or CVPR readiness.
