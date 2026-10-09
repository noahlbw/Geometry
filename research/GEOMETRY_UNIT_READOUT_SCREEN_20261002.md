# Geometry-defined semantic units: verified results

Signature `geometry-semantic-unit-detail-lift-v1-196-20261002`. Status complete. Original Geometry and all earlier outputs preserved.

## Architecture and scope

Original frozen DINOv3 -> Geometry relation/fine readout ->196 connected geometry-defined units -> original two-block semantic head re-encoding -> unit-mean replacement with retained fine descriptor detail -> unchanged normalized descriptors/all20 aliases/LME/window assembly.

Lift: `Y = Yg + U(Yunits - S Yg)`, with `S U = I`. Unit means and within-unit differences are defined before final L2 normalization, not promised invariants of final probabilities. No additional teacher, backbone crop, alias selection, trained parameter or fitted threshold. MST/token compression/coarse-detail reconstruction have precedent; this report does not establish mathematical originality or CVPR readiness.

This is the96-image screen: UDD5 full40, eight fixed images per other domain. Not a full eight-dataset SOTA benchmark. Masks are evaluation-only. All domains informed development. Corrected IRRG Vaihingen is used; LandCover.ai replaces unlabeled iSAID. Matched VIPProxy_Two is not the full official VIP system.

## Matched mIoU

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | UnitOnly | MeanLogit_Unit | Spatial_UnitDetail | Geometry_UnitDetail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 29.6232 | 32.0288 | 20.4202 | 17.7412 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 31.4921 | 36.9829 | 26.9251 | 34.7371 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 46.3978 | 49.7625 | 33.2255 | 30.4186 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 37.4391 | 39.7982 | 28.9896 | 27.7567 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 50.0312 | 60.9726 | 30.5287 | 28.9570 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 32.1554 | 39.1435 | 23.1282 | 21.2474 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 40.7273 | 50.0099 | 32.3924 | 36.8466 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 53.8709 | 63.1771 | 38.8116 | 29.3329 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 26.1063 | 36.9827 | 19.2742 | 16.9178 |

LoveDA D enters the equal-domain mean once; P remains separate.

| Method | Mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| UnitOnly | 37.226535 |
| MeanLogit_Unit | 43.485685 |
| Spatial_UnitDetail | 27.895863 |
| Geometry_UnitDetail | 26.874787 |

Primary wins vs Geometry: 0/8. Prospective promotion passed: False.

Primary equal-domain mean delta: -16.955782pp. The fixed candidate is rejected; no full rollout was launched.

| Gate | Passed |
| --- | --- |
| mean_vs_Geometry | False |
| mean_vs_SCLIP_Two | False |
| mean_vs_VIPProxy_Two | False |
| mean_vs_MeanLogit_Unit | False |
| mean_vs_Spatial_UnitDetail | False |
| retain_potsdam_potsdam | False |
| focus_potsdam_vs_Geometry | False |
| focus_potsdam_vs_VIPProxy_Two | False |
| focus_potsdam_vs_MeanLogit_Unit | False |
| focus_potsdam_vs_Spatial_UnitDetail | True |
| retain_oem_oem | False |
| retain_loveda_P | False |
| retain_loveda_D | False |
| retain_landcoverai_landcoverai | False |
| retain_vaihingen_vaihingen | False |
| retain_flair1_flair1 | False |
| retain_vdd_vdd | False |
| focus_vdd_vs_Geometry | False |
| focus_vdd_vs_VIPProxy_Two | False |
| focus_vdd_vs_MeanLogit_Unit | False |
| focus_vdd_vs_Spatial_UnitDetail | False |
| retain_udd5_udd5 | False |

## vdd/vdd class competition

Primary delta vs Geometry: -14.204942pp. Beneficial/harmful pixels: 4889155/22588380. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 25.3949 | 22.7556 | -2.6393 | 38.928/42.213 | 36.022/38.191 | -1039269 | +414377 |
| wall | 11.8982 | 10.3377 | -1.5605 | 11.984/94.332 | 10.748/73.014 | -398942 | -1619540 |
| road | 24.1970 | 12.6782 | -11.5188 | 24.741/91.668 | 14.178/54.509 | -492026 | +676689 |
| vegetation | 62.0257 | 36.3562 | -25.6695 | 84.684/69.863 | 47.813/60.274 | -1988222 | +11021433 |
| vehicle | 5.6965 | 4.2915 | -1.4050 | 5.697/99.956 | 4.317/87.771 | -27636 | +658884 |
| roof | 60.7577 | 26.2192 | -34.5385 | 88.015/66.238 | 79.944/28.065 | -10196405 | -528427 |
| water | 33.6534 | 11.5502 | -22.1032 | 93.059/34.520 | 29.066/16.084 | -3556725 | +7075809 |

## potsdam/potsdam class competition

Primary delta vs Geometry: -5.615724pp. Beneficial/harmful pixels: 712749/1048626. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 43.3944 | -8.2931 | 75.391/62.178 | 64.247/57.210 | -143918 | +334329 |
| building | 78.2026 | 63.0329 | -15.1697 | 80.677/96.226 | 78.807/75.899 | -198440 | -25734 |
| low vegetation | 33.4221 | 32.5489 | -0.8732 | 72.470/38.283 | 63.542/40.023 | +26714 | +129228 |
| tree | 62.5813 | 46.7551 | -15.8262 | 90.654/66.897 | 61.196/66.458 | -8412 | +675273 |
| car | 11.6582 | 17.4568 | 5.7986 | 11.672/99.026 | 17.637/94.460 | -8557 | -577804 |
| clutter | 4.5652 | 5.2345 | 0.6693 | 7.745/10.007 | 10.641/9.340 | -3264 | -199415 |

## udd5/udd5 class competition

Primary delta vs Geometry: -20.136656pp. Beneficial/harmful pixels: 14428183/116448862. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 45.9035 | -36.5902 | 96.486/85.049 | 56.075/71.676 | -17426408 | +69126157 |
| building | 83.9035 | 51.3300 | -32.5735 | 88.360/94.329 | 85.969/56.023 | -66162316 | -5669605 |
| road | 45.7856 | 24.4490 | -21.3366 | 70.678/56.522 | 42.804/36.312 | -11923878 | +14791321 |
| vehicle | 10.0092 | 11.8060 | 1.7968 | 10.032/97.748 | 12.157/80.348 | -612779 | -10424953 |
| other | 30.5846 | 18.6047 | -11.9799 | 52.854/42.059 | 29.023/34.136 | -5895298 | +34197759 |

Geometry non_residual_mean_iou_percent: 55.5480.

Geometry_UnitDetail non_residual_mean_iou_percent: 33.3721.

## oem/oem class competition

Primary delta vs Geometry: -11.597676pp. Beneficial/harmful pixels: 536733/1585735. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -246008 |
| rangeland | 47.7780 | 34.8041 | -12.9739 | 65.294/64.041 | 58.246/46.374 | -259676 | -11700 |
| developed space | 28.5294 | 25.5535 | -2.9759 | 61.562/34.713 | 50.154/34.252 | -7922 | +212658 |
| road | 40.8291 | 30.5386 | -10.2905 | 45.753/79.140 | 35.640/68.088 | -38712 | +102025 |
| tree | 56.0657 | 41.4444 | -14.6213 | 87.448/60.973 | 58.127/59.085 | -36860 | +660065 |
| water | 0.0000 | 0.0253 | 0.0253 | 0.000/0.000 | 0.026/2.931 | +80 | +308073 |
| agriculture land | 79.1353 | 44.3354 | -34.7999 | 90.641/86.177 | 78.304/50.544 | -458442 | +65700 |
| building | 62.4971 | 45.3518 | -17.1453 | 65.584/92.995 | 59.696/65.366 | -247470 | -41811 |

## loveda/P class competition

Primary delta vs Geometry: -33.868383pp. Beneficial/harmful pixels: 52413/1602763. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 64.4987 | 8.8916 | -55.6071 | 65.367/97.982 | 8.979/90.147 | -2345 | +257968 |
| road | 66.9348 | 39.8517 | -27.0831 | 67.652/98.440 | 41.532/90.786 | -24953 | +263192 |
| water | 81.1851 | 41.9406 | -39.2445 | 86.764/92.661 | 74.877/48.809 | -419434 | +21430 |
| barren | 24.3403 | 16.2522 | -8.0881 | 63.873/28.226 | 36.401/22.697 | -18501 | +79278 |
| tree | 53.7307 | 22.5617 | -31.1690 | 64.518/76.267 | 24.421/74.770 | -7582 | +959967 |
| farm | 86.2626 | 44.2441 | -42.0185 | 95.330/90.069 | 93.612/45.622 | -1077535 | -31485 |

## loveda/D class competition

Primary delta vs Geometry: -17.208465pp. Beneficial/harmful pixels: 422280/2058616. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 34.1983 | 26.5819 | -7.6164 | 65.750/41.611 | 55.687/33.713 | -291501 | +190150 |
| building | 31.1862 | 5.7148 | -25.4714 | 31.858/93.665 | 5.755/89.141 | -1354 | +376953 |
| road | 45.4818 | 26.9767 | -18.5051 | 45.835/98.335 | 27.771/90.410 | -25835 | +387694 |
| water | 61.9193 | 33.0023 | -28.9170 | 65.955/91.007 | 52.718/46.878 | -422075 | -47166 |
| barren | 14.5296 | 10.5600 | -3.9696 | 59.063/16.157 | 27.521/14.628 | -5115 | +91439 |
| tree | 26.8904 | 14.9686 | -11.9218 | 30.053/71.873 | 16.032/69.291 | -13078 | +991232 |
| farm | 54.9851 | 30.9273 | -24.0578 | 69.572/72.395 | 67.969/36.204 | -877378 | -353966 |

Geometry foreground_mean_iou_percent: 39.1654.

Geometry_UnitDetail foreground_mean_iou_percent: 20.3583.

## vaihingen/vaihingen class competition

Primary delta vs Geometry: -13.385877pp. Beneficial/harmful pixels: 510166/1497093. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 39.0775 | -10.1227 | 82.397/54.979 | 56.693/55.706 | +17692 | +749579 |
| building | 74.6610 | 57.5234 | -17.1376 | 75.500/98.533 | 70.527/75.727 | -390645 | -5625 |
| low vegetation | 46.7115 | 35.1336 | -11.5779 | 92.073/48.669 | 72.908/40.409 | -167062 | +218966 |
| tree | 71.8329 | 39.3088 | -32.5241 | 82.551/84.692 | 54.970/57.979 | -443664 | +491486 |
| car | 8.7566 | 13.1896 | 4.4330 | 8.773/97.975 | 13.278/95.217 | -3248 | -467479 |

## landcoverai/landcoverai class competition

Primary delta vs Geometry: -31.571973pp. Beneficial/harmful pixels: 32297/615539. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 54.5689 | -28.0401 | 96.651/85.044 | 91.759/57.381 | -393095 | +31348 |
| building | 34.9562 | 13.3649 | -21.5913 | 35.044/99.292 | 13.428/96.610 | -852 | +139426 |
| woodland | 78.3638 | 39.6265 | -38.7373 | 86.499/89.284 | 49.699/66.162 | -103805 | +238058 |
| water | 93.6635 | 24.8144 | -68.8491 | 93.664/100.000 | 32.700/50.715 | -85490 | +169323 |
| road | 14.9318 | 14.2897 | -0.6421 | 15.629/77.002 | 14.927/77.002 | +0 | +5087 |

Geometry foreground_mean_iou_percent: 55.4788.

Geometry non_residual_mean_iou_percent: 55.4788.

Geometry_UnitDetail foreground_mean_iou_percent: 23.0239.

Geometry_UnitDetail non_residual_mean_iou_percent: 23.0239.

## flair1/flair1 class competition

Primary delta vs Geometry: -21.924947pp. Beneficial/harmful pixels: 48124/593357. Pixel totals are not an mIoU objective.

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 37.5345 | -12.1634 | 50.726/96.082 | 45.588/67.996 | -42271 | -18326 |
| pervious surface | 57.0621 | 25.1696 | -31.8925 | 92.130/59.986 | 83.398/26.497 | -120466 | +542 |
| impervious surface | 52.6509 | 38.4695 | -14.1814 | 62.624/76.777 | 49.543/63.250 | -46453 | +63857 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +48338 |
| water | 73.2328 | 27.7420 | -45.4908 | 77.143/93.526 | 34.124/59.731 | -31427 | +81459 |
| coniferous | 43.0233 | 3.8567 | -39.1666 | 61.711/58.690 | 4.364/24.924 | -3979 | +60079 |
| deciduous | 54.0229 | 24.4324 | -29.5905 | 78.972/63.099 | 41.551/37.227 | -92320 | +126905 |
| brushwood | 18.6136 | 8.4161 | -10.1975 | 23.421/47.556 | 10.626/28.810 | -19328 | +89523 |
| vineyard | undefined | 0.0000 | undefined | 0.000/0.000 | 0.000/0.000 | +0 | +12553 |
| herbaceous vegetation | 60.2329 | 32.1622 | -28.0707 | 94.317/62.501 | 83.373/34.367 | -188940 | +20736 |
| agricultural land | 18.7339 | 5.2309 | -13.5030 | 21.647/58.198 | 5.442/57.431 | -49 | +50322 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +9245 |

## Diagnostics and costs

| Dataset | Max shard seconds | Sum shard seconds | Peak MiB | Units | Singleton unit fraction | Largest unit patches | Mean reconstruction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 149.115 | 149.115 | 3818.962 | 196.000 | 0.73583 | 422.253 | 0.000000223 |
| potsdam | 9.912 | 9.912 | 3816.758 | 196.000 | 0.63301 | 282.264 | 0.000000176 |
| udd5 | 156.894 | 614.642 | 3817.987 | 196.000 | 0.66743 | 321.573 | 0.000000216 |
| oem | 10.885 | 10.885 | 3818.165 | 196.000 | 0.63897 | 247.269 | 0.000000147 |
| loveda | 14.744 | 14.744 | 3819.479 | 196.000 | 0.66596 | 309.097 | 0.000000184 |
| vaihingen | 8.293 | 8.293 | 3815.604 | 196.000 | 0.65427 | 199.972 | 0.000000176 |
| landcoverai | 1.638 | 1.638 | 3724.494 | 196.000 | 0.72959 | 422.375 | 0.000000166 |
| flair1 | 2.327 | 2.327 | 3725.588 | 196.000 | 0.63329 | 211.875 | 0.000000140 |

Independent window medians Geometry/primary: 0.024748/0.030845s (1.2464x), including partition. Peak allocated primary: 3715.736MiB.

FP32/bf16 singleton Geometry and native-cache errors0; unit mean error below2e-4; no target masks loaded in smoke. MST membership can change between FP32 and bf16 at nearly tied edges; identical budgets are not identical supports. This does not invalidate exact original baseline replay but limits precision-invariance claims.

Evaluator timing includes all controls and output assembly, not standalone primary latency. Summed durations describe GPU allocation, not kernel-active time.

## Control recovery and interpretation

The first run passed already-normalized native patches to the VIP proxy, which normalizes them again. Its exact reference check failed and stopped promotion. The failed outputs remain preserved. The clean repeat corrects only the proxy cache input; all primary rules, samples, checkpoints and vocabularies are unchanged.

VDD, Potsdam, OEM and LoveDA original Geometry and primary per-image confusion matrices repeat exactly, including both LoveDA P and D. The primary losses were therefore not introduced by this control-interface correction. Details: `geometry_unit_readout_screen_20261002_r2/control_repeat_audit.json`.

The results reject this fixed unit re-encoding/detail-lifting method, not Geometry or every training-free coupling. Connected units are not certified semantic units; some contain hundreds of patches. Pooled tokens and changed prefix/patch attention allocations need not match the frozen head's training distribution. These are plausible limitations, not separately identified causal effects.

Most importantly, the exact pre-normalization reconstruction invariants do not preserve text-relative feature directions or class margins. The dense primary loses against both UnitOnly and equal-information logit fusion on every screened protocol, so retained within-unit differences do not rescue this reconstruction. Higher Potsdam/Vaihingen car IoU coexists with large building/tree losses; reduced false car activation is not sufficient evidence of an overall useful mechanism.

A failed gate rejects this fixed candidate and prevents full rollout. No outcome-based budget, grouping, scale, lift or class tuning. The broader objective remains a useful globally fixed coupling with strong full-system comparisons and independent validation.
