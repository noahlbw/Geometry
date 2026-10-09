# Geometry-balanced SAT feature transport: eight-domain screen

2026-10-02. Suite status: **complete**. Verified datasets: 8/8. Pending: none.

This is a fixed exploratory development screen, not a full eight-domain benchmark. Eight fixed-seed images per domain, except full40 UDD5; planned96 unique images. All eight datasets informed earlier development. No untouched/SOTA claim.

## Complete candidate

Frozen LVD -> native Geometry relation G. Frozen SAT observes identical coordinates. G inverse incoming-mass weights balance an image-only orthogonal Procrustes fit into native LVD feature coordinates. Valid transported SAT patches and unchanged native prefixes enter the original two-block Geometry head with the original G. One primary output; no score gate, alias pruning, class calibration or per-domain method switching.

Weights stay frozen; per-window rotations/means are inferred adaptation states. The extra SAT backbone and established Procrustes operator are explicitly attributed. This implementation alone is not an independent CVPR contribution.

## Verified mIoU

Values in percent, from matched full-image predictions. LoveDA P/D share images. SCLIP_Two/VIPProxy_Two are matched DINO.text operator adaptations, not official systems.

| Dataset/protocol | Images | Geometry | SCLIP_Two | VIPProxy_Two | SAT_Relation | SAT_Unaligned | SAT_UniformTransport | MeanLogit_SATTransport | Geometry_SATTransport |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 31.9462 | 30.6800 | 31.1056 | 31.9602 | 4.8771 | 29.9506 | 31.8699 | 30.1339 |
| potsdam/potsdam | 8 | 40.3528 | 44.0525 | 42.7377 | 39.7292 | 11.4395 | 29.7240 | 36.6561 | 29.6576 |
| udd5/udd5 | 40 | 50.5553 | 50.2837 | 49.9071 | 50.3989 | 14.5970 | 44.8306 | 49.0821 | 44.8297 |
| oem/oem | 8 | 39.3543 | 37.3120 | 38.9899 | 39.3408 | 3.2804 | 39.5465 | 40.2078 | 39.4946 |
| loveda/P | 8 | 62.8254 | 68.2854 | 68.5692 | 62.2669 | 6.6273 | 57.4056 | 62.3893 | 57.9481 |
| loveda/D | 8 | 38.4558 | 36.5043 | 35.6670 | 38.0381 | 7.9913 | 38.8224 | 39.6366 | 38.6844 |
| vaihingen/vaihingen | 8 | 50.2325 | 52.0872 | 50.5393 | 49.6924 | 14.3164 | 42.9689 | 49.3455 | 43.3831 |
| landcoverai/landcoverai | 8 | 60.9049 | 61.7170 | 59.3073 | 60.4616 | 17.6054 | 65.7693 | 63.7119 | 65.6673 |
| flair1/flair1 | 8 | 38.8428 | 37.6225 | 39.3436 | 38.4312 | 2.5327 | 33.7220 | 37.7244 | 33.9304 |

All three baseline per-image confusion arrays match the earlier complete reference exactly on each screened sample. Unique coverage, fixed20 vocabulary and checkpoint identity were verified remotely and after download.

## Eight-domain decision

LoveDA D counts once in the equal-domain mean.

| Method | Equal-domain mean |
| --- | ---: |
| Geometry | 43.830569 |
| SCLIP_Two | 43.782393 |
| VIPProxy_Two | 43.449702 |
| SAT_Relation | 43.506554 |
| SAT_Unaligned | 9.579974 |
| SAT_UniformTransport | 40.666798 |
| MeanLogit_SATTransport | 43.529300 |
| Geometry_SATTransport | 40.722626 |

Predeclared promotion gate passed: **False**.

| Gate check | Passed |
| --- | --- |
| mean_vs_Geometry | False |
| mean_vs_VIPProxy_Two | False |
| mean_vs_SAT_UniformTransport | True |
| mean_vs_MeanLogit_SATTransport | False |
| retain_potsdam_potsdam | False |
| focus_potsdam_vs_Geometry | False |
| focus_potsdam_vs_MeanLogit_SATTransport | False |
| retain_oem_oem | True |
| retain_flair1_flair1 | False |
| retain_loveda_P | False |
| retain_loveda_D | True |
| retain_landcoverai_landcoverai | True |
| retain_vaihingen_vaihingen | False |
| retain_vdd_vdd | False |
| focus_vdd_vs_Geometry | False |
| focus_vdd_vs_MeanLogit_SATTransport | False |
| retain_udd5_udd5 | False |

No full rollout is permitted when this gate fails. No post-result coefficient or per-dataset configuration changes were made.

## Primary changes

| Dataset/protocol | vs Geometry | vs VIPProxy_Two | vs UniformTransport | vs MeanLogit | Beneficial | Harmful |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | -1.812278 | -0.971707 | +0.183273 | -1.736035 | 4724402 | 6990282 |
| potsdam/potsdam | -10.695228 | -13.080137 | -0.066355 | -6.998524 | 345659 | 1304051 |
| udd5/udd5 | -5.725560 | -5.077374 | -0.000851 | -4.252316 | 11783775 | 36431800 |
| oem/oem | +0.140240 | +0.504663 | -0.051975 | -0.713194 | 487580 | 497006 |
| loveda/P | -4.877304 | -10.621159 | +0.542495 | -4.441264 | 87954 | 282235 |
| loveda/D | +0.228575 | +3.017412 | -0.137957 | -0.952230 | 821536 | 644923 |
| vaihingen/vaihingen | -6.849327 | -7.156193 | +0.414222 | -5.962377 | 353908 | 947641 |
| landcoverai/landcoverai | +4.762433 | +6.359960 | -0.102053 | +1.955364 | 108754 | 36533 |
| flair1/flair1 | -4.912401 | -5.413230 | +0.208318 | -3.794078 | 85917 | 251068 |

Pixel transitions diagnose pixel accuracy, not the mIoU objective.

## Foreground metrics

| Dataset/protocol | Metric | Geometry | UniformTransport | MeanLogit | Primary |
| --- | --- | ---: | ---: | ---: | ---: |
| loveda/D | foreground_mean_iou_percent | 39.1654 | 37.8910 | 39.6085 | 37.9481 |
| udd5/udd5 | non_residual_mean_iou_percent | 55.5480 | 48.8030 | 53.6508 | 48.8647 |
| landcoverai/landcoverai | non_residual_mean_iou_percent | 55.4788 | 60.3645 | 58.2016 | 60.2524 |

## Per-class IoU and competition

### vdd/vdd

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 25.3949 | 28.2682 | 2.8733 | 38.928/42.213 | 42.118/46.227 | +1037191 | -696665 |
| wall | 11.8982 | 12.1383 | 0.2401 | 11.984/94.332 | 12.203/95.801 | +27484 | -67136 |
| road | 24.1970 | 23.8886 | -0.3084 | 24.741/91.668 | 26.124/73.630 | -238840 | -935050 |
| vegetation | 62.0257 | 52.0324 | -9.9933 | 84.684/69.863 | 84.562/57.494 | -2564704 | -443464 |
| vehicle | 5.6965 | 2.7121 | -2.9844 | 5.697/99.956 | 2.712/99.563 | -893 | +4346161 |
| roof | 60.7577 | 60.1421 | -0.6156 | 88.015/66.238 | 87.779/65.638 | -160146 | +31787 |
| water | 33.6534 | 31.7554 | -1.8980 | 93.059/34.520 | 92.274/32.623 | -365972 | +30247 |
### potsdam/potsdam

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 51.6875 | 48.5931 | -3.0944 | 75.391/62.178 | 64.972/65.843 | +106158 | +440373 |
| building | 78.2026 | 65.0231 | -13.1795 | 80.677/96.226 | 76.957/80.743 | -151146 | +11022 |
| low vegetation | 33.4221 | 20.8526 | -12.5695 | 72.470/38.283 | 66.245/23.332 | -229430 | -40728 |
| tree | 62.5813 | 31.4680 | -31.1133 | 90.654/66.897 | 94.716/32.030 | -668042 | -97908 |
| car | 11.6582 | 9.6934 | -1.9648 | 11.672/99.026 | 9.698/99.522 | +929 | +332257 |
| clutter | 4.5652 | 2.3154 | -2.2498 | 7.745/10.007 | 3.455/6.559 | -16861 | +313376 |
### udd5/udd5

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 82.4937 | 72.6163 | -9.8774 | 96.486/85.049 | 97.202/74.167 | -14180722 | -1254299 |
| building | 83.9035 | 80.9068 | -2.9967 | 88.360/94.329 | 85.953/93.235 | -1890581 | +4855672 |
| road | 45.7856 | 35.1749 | -10.6107 | 70.678/56.522 | 69.071/41.751 | -8714649 | -2804619 |
| vehicle | 10.0092 | 6.7607 | -3.2485 | 10.032/97.748 | 6.770/97.969 | +7793 | +16640905 |
| other | 30.5846 | 28.6900 | -1.8946 | 52.854/42.059 | 47.219/42.234 | +130134 | +7210366 |
### oem/oem

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -116458 |
| rangeland | 47.7780 | 46.8916 | -0.8864 | 65.294/64.041 | 62.396/65.363 | +19431 | +78681 |
| developed space | 28.5294 | 34.7467 | 6.2173 | 61.562/34.713 | 54.105/49.268 | +250283 | +345940 |
| road | 40.8291 | 44.9639 | 4.1348 | 45.753/79.140 | 55.613/70.132 | -31552 | -132609 |
| tree | 56.0657 | 50.2577 | -5.8080 | 87.448/60.973 | 89.342/53.463 | -146605 | -46357 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -2579 |
| agriculture land | 79.1353 | 76.6268 | -2.5085 | 90.641/86.177 | 86.835/86.699 | +6724 | +54642 |
| building | 62.4971 | 62.4699 | -0.0272 | 65.584/92.995 | 73.220/80.970 | -107707 | -171834 |
### loveda/P

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 64.4987 | 63.5507 | -0.9480 | 65.367/97.982 | 71.789/84.704 | -3974 | -5575 |
| road | 66.9348 | 59.9157 | -7.0191 | 67.652/98.440 | 60.923/97.315 | -3669 | +50039 |
| water | 81.1851 | 76.9025 | -4.2826 | 86.764/92.661 | 86.554/87.337 | -50928 | -5430 |
| barren | 24.3403 | 23.5002 | -0.8401 | 63.873/28.226 | 79.632/25.003 | -10785 | -32021 |
| tree | 53.7307 | 42.3104 | -11.4203 | 64.518/76.267 | 48.813/76.055 | -1071 | +191592 |
| farm | 86.2626 | 81.5089 | -4.7537 | 95.330/90.069 | 95.253/84.960 | -123854 | -4324 |
### loveda/D

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 34.1983 | 43.1020 | 8.9037 | 65.750/41.611 | 62.750/57.922 | +602041 | +469088 |
| building | 31.1862 | 35.1174 | 3.9312 | 31.858/93.665 | 39.867/74.670 | -5685 | -26251 |
| road | 45.4818 | 46.7060 | 1.2242 | 45.835/98.335 | 47.397/96.971 | -4446 | -27992 |
| water | 61.9193 | 62.1522 | 0.2329 | 65.955/91.007 | 71.452/82.685 | -79597 | -133336 |
| barren | 14.5296 | 5.2123 | -9.3173 | 59.063/16.157 | 56.779/5.428 | -35902 | -23647 |
| tree | 26.8904 | 26.9417 | 0.0513 | 30.053/71.873 | 31.535/64.910 | -35277 | -133519 |
| farm | 54.9851 | 51.5593 | -3.4258 | 69.572/72.395 | 76.158/61.484 | -264521 | -300956 |
### vaihingen/vaihingen

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 49.2002 | 46.2469 | -2.9533 | 82.397/54.979 | 68.957/58.407 | +83406 | +353972 |
| building | 74.6610 | 63.7021 | -10.9589 | 75.500/98.533 | 70.948/86.183 | -211542 | +56816 |
| low vegetation | 46.7115 | 38.4730 | -8.2385 | 92.073/48.669 | 92.334/39.742 | -180546 | -18011 |
| tree | 71.8329 | 61.7847 | -10.0482 | 82.551/84.692 | 87.455/67.793 | -280662 | -135802 |
| car | 8.7566 | 6.7089 | -2.0477 | 8.773/97.975 | 6.736/94.248 | -4389 | +336758 |
### landcoverai/landcoverai

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 87.3268 | 4.7178 | 96.651/85.044 | 96.150/90.491 | +77411 | +9612 |
| building | 34.9562 | 39.5614 | 4.6052 | 35.044/99.292 | 39.688/99.201 | -29 | -10579 |
| woodland | 78.3638 | 82.0687 | 3.7049 | 86.499/89.284 | 92.415/87.996 | -5785 | -30140 |
| water | 93.6635 | 97.0868 | 3.4233 | 93.664/100.000 | 97.087/100.000 | +0 | -6530 |
| road | 14.9318 | 22.2927 | 7.3609 | 15.629/77.002 | 23.622/79.844 | +624 | -34584 |
### flair1/flair1

| Class | Geometry IoU | Primary IoU | Delta | Geometry P/R | Primary P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 38.2902 | -11.4077 | 50.726/96.082 | 42.581/79.165 | -25462 | +20192 |
| pervious surface | 57.0621 | 60.5149 | 3.4528 | 92.130/59.986 | 94.587/62.686 | +9711 | -5529 |
| impervious surface | 52.6509 | 44.5831 | -8.0678 | 62.624/76.777 | 50.691/78.723 | +6684 | +105615 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -19405 |
| water | 73.2328 | 62.0645 | -11.1683 | 77.143/93.526 | 82.503/71.472 | -20509 | -11674 |
| coniferous | 43.0233 | 33.6797 | -9.3436 | 61.711/58.690 | 88.578/35.209 | -2767 | -3756 |
| deciduous | 54.0229 | 28.2191 | -25.8038 | 78.972/63.099 | 80.406/30.303 | -117026 | -33602 |
| brushwood | 18.6136 | 15.5397 | -3.0739 | 23.421/47.556 | 17.387/59.394 | +12206 | +130653 |
| vineyard | undefined | undefined | undefined | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 56.6593 | -3.5736 | 94.317/62.501 | 95.178/58.334 | -27988 | -5446 |
| agricultural land | 18.7339 | 33.6834 | 14.9495 | 21.647/58.198 | 44.434/58.198 | +0 | -8813 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -3084 |

## Diagnostics and evaluator cost

Timings below execute all eight arms together, not the deployed primary alone. Sum shard time is elapsed allocation time, not kernel-active GPU time.

| Dataset | Max shard seconds | Sum shard seconds | Peak allocated MiB | Weighted fit before/after | Uniform fit after | Cosine |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| vdd | 494.889 | 494.889 | 4976.113 | 53.0813/15.2190 | 14.9840 | 0.88560 |
| potsdam | 43.056 | 43.056 | 4973.003 | 57.8598/18.3294 | 18.1975 | 0.86405 |
| udd5 | 2075.925 | 2075.925 | 4974.698 | 55.9861/16.7112 | 16.5012 | 0.87624 |
| oem | 41.238 | 41.238 | 4976.447 | 54.8325/14.0607 | 13.8774 | 0.89117 |
| loveda | 48.961 | 48.961 | 4976.103 | 53.6817/14.2780 | 14.0781 | 0.89403 |
| vaihingen | 46.865 | 46.865 | 4973.718 | 59.5240/16.4167 | 16.1911 | 0.87369 |
| landcoverai | 5.275 | 5.275 | 4892.213 | 53.6893/13.5452 | 13.2650 | 0.90235 |
| flair1 | 5.892 | 5.892 | 4893.308 | 57.4448/14.3512 | 14.1671 | 0.89230 |

### Independent deployed cost

One1024x1024 LoveDA image on physical GPU7, no labels/control heads; three image repetitions and ten window repetitions. Geometry measured before SAT allocation. Text/model loading excluded; probability assembly included. Not full-suite throughput.

| Method | Image median seconds | Window median ms | Image peak MiB |
| --- | ---: | ---: | ---: |
| Geometry | 0.322504 | 21.729353 | 3605.347 |
| Geometry_SATTransport | 2.472816 | 259.925626 | 4775.104 |

Primary image/window ratios: 7.6675x/11.9620x. Standalone primary features exactly match the evaluator. This cost is not justified by the current performance evidence.

## Mechanism interpretation and boundaries

1. Feature-coordinate agreement is not language-decision agreement. A lower reconstruction error and roughly0.87 feature cosine can still cross the nonlinear head's class boundaries.
2. Same-coordinate pairing and Geometry weighting do not supply semantic labels. They can preserve or reproduce native errors rather than distinguish car false positives.
3. Uniform-transport and mean-logit controls isolate the claimed coupling. An isolated LandCover.ai gain is insufficient, particularly when uniform transport also improves.
4. Potsdam loses vegetation/tree/building coverage while car prediction area increases. This is not a successful small-object correction and cannot be repaired merely by naming the fit a trust module.
5. This rejects the fixed image-local orthogonal feature-replacement hypothesis. It does not prove every use of SAT, every alignment or every training-free Geometry coupling is impossible.

No coefficient/depth/alias set was tuned separately on these labels. Masks enter only after image predictions. However candidate development and rejection use labeled validation metrics, so these are development data.

The historical official-configured VIP VDD52.0647/Potsdam44.0643 references have different system protocols and full coverage. They must not be compared to the eight-image point estimates as a fair victory test. Corrected Vaihingen IRRG is used; invalid historical black-input VIP results are excluded. LandCover.ai substitutes for iSAID, which has no semantic masks here.

## Artifacts

- `DINOtool/dinotool/sat_geometry_transport.py` and `scripts/eval_sat_geometry_transport.py`.
- `scripts/run_sat_geometry_transport_suite.py`: physical GPU4-7 queue, frozen gate and exact per-image baseline checks.
- `research/GEOMETRY_SAT_TRANSPORT_PROTOCOL_20261002.md`: prospective architecture and gate.
- `research/sat_geometry_transport_screen_20261002/`: downloaded verified results, samples, per-image arrays, smoke and cost.
- 18 local/remote unit tests passed; actual-checkpoint FP32/bf16 identity and padding errors are exactly0.
- Failed loading/numerical smoke logs were preserved in separate initial/r3 remote roots; final smoke is r4.
