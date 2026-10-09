# Geometry-conditioned localized likelihood: verified cached-window screen

Signature geometry-localized-alias-likelihood-v1-20261002. All eight domains completed on physical A800 GPUs4-7. UDD5 full40 images, seven other domains8 each:96 images, observed at one/two original512 windows per image. These are window-only statistics, NOT full-image dataset mIoU. LoveDA D counts once in the equal-domain mean; P is separate. Corrected IRRG Vaihingen; LandCover.ai replaces unlabeled iSAID. Windows can overlap; all domains are development data.

## Complete candidate and boundary

Unchanged Geometry scores/relation + fixed frozen GroundingDINO all20-alias observations -> within-box Geometry support h=b*(G*1[b>0]) -> normalized spatial density D=N*h/sum(h) -> per-alias unit-uniform-pseudocount likelihood e=(1+sum(s*D))/(1+sum(s)) -> all20-alias arithmetic class density -> g'=g+.07*log(e). No confidence, temperature or class tuning.

Whole-window and empty observations are exactly neutral; identity Geometry exactly recovers the same-source no-Geometry control. Missing detections do not veto a class. Nevertheless positive observed locations lower relative likelihood outside their support and can harm unseen objects. A normalized spatial density is NOT a calibrated semantic posterior or correctness certificate.

Ten core tests passed locally/remotely. Real caches passed identity control; original Geometry window confusions and all transition reconstructions match exactly. All candidate scores were saved from image-only arrays before masks were loaded by a separate audit. The design was motivated by a prior labeled development audit, so this is NOT untouched validation.

The detector introduces additional pretrained detection supervision and processor resizing. No target labels fit inference parameters, weights or aliases, but this is NOT a matched single-encoder system. Standard Bayes updates and pseudocounts are existing machinery; originality and complete official-VIP superiority remain unproven.

## Window-only mIoU

| Domain/protocol | Geometry | BoxLocalLikelihood | MeanProb_BoxLikelihood | ShuffledLocalLikelihood | Geometry_LocalLikelihood | Delta vs Geometry | Delta vs BoxLocalLikelihood |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 37.8993 | 37.7433 | 37.9962 | 37.7227 | 37.7829 | -0.1165 | +0.0396 |
| potsdam/potsdam | 40.7676 | 40.3792 | 41.1521 | 40.3936 | 40.7685 | +0.0008 | +0.3893 |
| udd5/udd5 | 45.1736 | 44.6772 | 45.0244 | 44.6828 | 44.7689 | -0.4048 | +0.0917 |
| oem/oem | 38.9652 | 36.4458 | 37.6802 | 36.4657 | 36.5909 | -2.3743 | +0.1451 |
| loveda/P | 65.1016 | 59.4690 | 62.2940 | 59.5118 | 60.7086 | -4.3930 | +1.2396 |
| loveda/D | 35.8503 | 34.5590 | 35.2959 | 34.6193 | 34.6247 | -1.2256 | +0.0657 |
| vaihingen/vaihingen | 47.6994 | 50.6736 | 50.0876 | 50.8631 | 51.3604 | +3.6610 | +0.6868 |
| landcoverai/landcoverai | 60.9049 | 60.1290 | 60.5235 | 60.1443 | 60.2919 | -0.6130 | +0.1629 |
| flair1/flair1 | 38.8428 | 36.9995 | 38.1494 | 37.0786 | 37.1553 | -1.6875 | +0.1558 |

| Method | Equal-domain window mean |
| --- | ---: |
| Geometry | 43.262886 |
| BoxLocalLikelihood | 42.700817 |
| MeanProb_BoxLikelihood | 43.238662 |
| ShuffledLocalLikelihood | 42.746255 |
| Geometry_LocalLikelihood | 42.917934 |

Primary wins vs Geometry: 2/8. Prospective promotion gate passed: False. Full-image evaluation launched:False.

## Class competition and corrections

### vdd/vdd

Corrected/corrupted/wrong-to-wrong: 82128/81013/149007. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 26.0715 | 25.1232 | -0.9483 | 36.234/48.174 | 35.313/46.542 | -15947 | +4676 |
| wall | 38.3866 | 40.1387 | 1.7521 | 38.392/99.965 | 40.141/99.983 | +34 | -21577 |
| road | 38.0339 | 34.0972 | -3.9367 | 38.541/96.658 | 34.277/98.483 | +2228 | +42354 |
| vegetation | 43.0651 | 44.7833 | 1.7182 | 87.726/45.826 | 86.999/47.995 | +23907 | +8384 |
| vehicle | 6.1250 | 8.6910 | 2.5660 | 6.125/100.000 | 8.691/100.000 | +0 | -51562 |
| roof | 79.4937 | 77.5332 | -1.9605 | 85.587/91.780 | 84.114/90.835 | -9126 | +16408 |
| water | 34.1196 | 34.1137 | -0.0059 | 93.828/34.903 | 93.766/34.905 | +19 | +202 |

### potsdam/potsdam

Corrected/corrupted/wrong-to-wrong: 186314/104356/198386. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 59.5266 | 60.4142 | 0.8876 | 76.864/72.521 | 75.081/75.566 | +46520 | +49658 |
| building | 77.4448 | 68.0955 | -9.3493 | 78.142/98.861 | 72.685/91.515 | -30711 | +28171 |
| low vegetation | 27.6598 | 28.7814 | 1.1216 | 74.343/30.579 | 72.923/32.225 | +13302 | +11410 |
| tree | 63.9191 | 67.1803 | 3.2612 | 90.604/68.457 | 89.408/72.990 | +54349 | +18558 |
| car | 11.7333 | 15.6110 | 3.8777 | 11.735/99.847 | 15.657/98.172 | -1201 | -159306 |
| clutter | 4.3221 | 4.5282 | 0.2061 | 6.018/13.299 | 6.467/13.122 | -301 | -30449 |

### udd5/udd5

Corrected/corrupted/wrong-to-wrong: 542807/708115/578380. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 76.0818 | 75.8716 | -0.2102 | 98.330/77.078 | 97.870/77.146 | +4267 | +23261 |
| building | 76.4461 | 71.9803 | -4.4658 | 81.792/92.124 | 80.803/86.828 | -384748 | +8686 |
| road | 38.9432 | 40.6285 | 1.6853 | 69.690/46.884 | 65.640/51.603 | +158475 | +222342 |
| vehicle | 6.0762 | 8.0541 | 1.9779 | 6.082/98.387 | 8.067/97.997 | -498 | -513984 |
| other | 28.3209 | 27.3099 | -1.0110 | 46.627/41.906 | 42.456/43.359 | +57196 | +425003 |

### oem/oem

Corrected/corrupted/wrong-to-wrong: 73145/195751/127190. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -1026 |
| rangeland | 49.8863 | 48.9121 | -0.9742 | 70.829/62.786 | 70.703/61.346 | -12272 | -3737 |
| developed space | 29.5609 | 28.2768 | -1.2841 | 56.548/38.249 | 52.597/37.947 | -3140 | +50068 |
| road | 44.7030 | 41.4366 | -3.2664 | 53.455/73.192 | 49.467/71.850 | -3821 | +27511 |
| tree | 48.3718 | 46.2225 | -2.1493 | 88.422/51.643 | 86.588/49.787 | -17812 | +9122 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +25711 |
| agriculture land | 78.6245 | 76.1524 | -2.4721 | 82.863/93.891 | 84.043/89.024 | -23743 | -12266 |
| building | 60.5747 | 51.7266 | -8.8481 | 64.095/91.688 | 58.986/80.780 | -61818 | +27223 |

### loveda/P

Corrected/corrupted/wrong-to-wrong: 4064/7610/2336. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 51.7113 | 26.4061 | -25.3052 | 53.236/94.751 | 46.751/37.764 | -836 | -590 |
| road | 68.3951 | 68.0215 | -0.3736 | 68.398/99.995 | 68.137/99.752 | -318 | +584 |
| water | 87.3809 | 86.9187 | -0.4622 | 96.082/90.609 | 95.684/90.465 | -979 | +2620 |
| barren | 50.2389 | 50.4075 | 0.1686 | 73.017/61.693 | 73.867/61.348 | -325 | -1032 |
| tree | 48.0837 | 47.8893 | -0.1944 | 56.616/76.138 | 56.548/75.772 | -667 | -219 |
| farm | 84.7995 | 84.6086 | -0.1909 | 94.855/88.888 | 94.658/88.851 | -421 | +2183 |

### loveda/D

Corrected/corrupted/wrong-to-wrong: 13066/29563/25497. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 32.7230 | 32.1255 | -0.5975 | 69.509/38.207 | 68.994/37.546 | -13043 | +2228 |
| building | 12.0412 | 3.8564 | -8.1848 | 12.266/86.776 | 4.148/35.446 | -753 | +2912 |
| road | 37.7236 | 37.7864 | 0.0628 | 37.725/99.989 | 37.823/99.746 | -319 | -1421 |
| water | 63.8501 | 63.1019 | -0.7482 | 70.946/86.456 | 70.145/86.273 | -1248 | +8930 |
| barren | 25.1752 | 26.4250 | 1.2498 | 63.178/29.504 | 63.737/31.101 | +1505 | +470 |
| tree | 22.8266 | 22.3913 | -0.4353 | 25.062/71.902 | 24.564/71.687 | -394 | +9420 |
| farm | 56.6124 | 56.6866 | 0.0742 | 66.338/79.430 | 66.580/79.232 | -2245 | -6042 |

### vaihingen/vaihingen

Corrected/corrupted/wrong-to-wrong: 492063/155592/280691. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 47.6657 | 62.1412 | 14.4755 | 84.267/52.322 | 78.236/75.128 | +336740 | +164344 |
| building | 74.7317 | 68.7881 | -5.9436 | 75.718/98.286 | 71.927/94.034 | -32581 | +39717 |
| low vegetation | 35.9788 | 40.1782 | 4.1994 | 91.316/37.253 | 87.460/42.634 | +51566 | +24633 |
| tree | 73.0101 | 68.9073 | -4.1028 | 84.251/84.549 | 79.798/83.468 | -9686 | +47718 |
| car | 7.1106 | 16.7874 | 9.6768 | 7.112/99.811 | 17.264/85.884 | -9568 | -612883 |

### landcoverai/landcoverai

Corrected/corrupted/wrong-to-wrong: 5527/20676/4046. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 81.6030 | -1.0060 | 96.651/85.044 | 96.594/84.020 | -14542 | +215 |
| building | 34.9562 | 35.3962 | 0.4400 | 35.044/99.292 | 35.571/98.628 | -211 | -1718 |
| woodland | 78.3638 | 76.1273 | -2.2365 | 86.499/89.284 | 83.868/89.187 | -437 | +14453 |
| water | 93.6635 | 93.6458 | -0.0177 | 93.664/100.000 | 93.646/100.000 | +0 | +35 |
| road | 14.9318 | 14.6872 | -0.2446 | 15.629/77.002 | 15.354/77.189 | +41 | +2164 |

### flair1/flair1

Corrected/corrupted/wrong-to-wrong: 23546/56699/39419. Pixel correction counts do not alone determine class-balanced mIoU.

| Class | Geometry IoU | Primary IoU | Delta | Geometry precision/recall | Primary precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 47.1074 | -2.5905 | 50.726/96.082 | 50.338/88.009 | -12151 | -9793 |
| pervious surface | 57.0621 | 57.5351 | 0.4730 | 92.130/59.986 | 91.849/60.631 | +2319 | +922 |
| impervious surface | 52.6509 | 50.6515 | -1.9994 | 62.624/76.777 | 59.863/76.698 | -268 | +19239 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +4902 |
| water | 73.2328 | 64.8965 | -8.3363 | 77.143/93.526 | 68.871/91.834 | -1574 | +12830 |
| coniferous | 43.0233 | 41.1311 | -1.8922 | 61.711/58.690 | 57.711/58.876 | +22 | +793 |
| deciduous | 54.0229 | 52.5517 | -1.4712 | 78.972/63.099 | 79.021/61.073 | -7232 | -2094 |
| brushwood | 18.6136 | 18.0491 | -0.5645 | 23.421/47.556 | 22.650/47.049 | -523 | +5340 |
| vineyard | NA | NA | NA | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 58.2284 | -2.0045 | 94.317/62.501 | 94.052/60.454 | -13746 | +382 |
| agricultural land | 18.7339 | 18.5573 | -0.1766 | 21.647/58.198 | 21.411/58.198 | +0 | +189 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +443 |

## Mechanism verdict

Geometry conditioning improves the same-source BoxLocalLikelihood control in all eight domains and both LoveDA protocols, by+0.217117pp equal-domain mean. It also improves the spatially shuffled support mean by+0.171679pp. This is useful evidence that the actual Geometry supports can improve localization distribution placement under this fixed update, not evidence of universal semantic correctness or originality.

However, the primary mean42.917934 is below original Geometry43.262886 (-0.344952pp) and the same-source fixed probability blend43.238662 (-0.320727pp). Its two domain wins are Vaihingen+3.6610 and Potsdam+0.0008pp; the latter is not a practically meaningful gain. These are overlapping window-only development statistics without independent scene-level uncertainty, not full-domain results.

VDD vehicle improves6.1250->8.6910 IoU, but road38.0339->34.0972 and roof79.4937->77.5332. Potsdam car11.7333->15.6110 improves while building77.4448->68.0955. UDD5 building76.4461->71.9803 and OEM building60.5747->51.7266. LoveDA P's building51.7113->26.4061 contributes most of that protocol's -4.3930pp loss. Thus the lost large/partially observed building coverage is not merely an all-class uniform-box calibration problem.

The model explicitly leaves a completely undetected alias neutral, but an alias with some positive detections receives below-uniform likelihood outside those detected locations. That relative negative update relies on a localization-completeness assumption the detector does not provide. The current results are consistent with harmful partial-coverage redistribution, but do not by themselves establish it as the unique cause of every lost building pixel. Further direction should distinguish positive observed support from unknown/unobserved area; do not fit a new influence coefficient or silently relabel this failed gate as passed.

## Cost and gate

This run reuses frozen cached observations; it is NOT deployed inference timing. It excludes the original DINO and detector forwards. A deployment would still pay both encoders and the detector's alias-query cost. No deployed speed or memory advantage is claimed.

Failed gate checks:

- mean_vs_MeanProb_BoxLikelihood
- retain_vdd_vdd
- retain_oem_oem
- net_corrections_oem_oem
- retain_udd5_udd5
- net_corrections_udd5_udd5
- retain_loveda_P
- net_corrections_loveda_P
- retain_loveda_D
- net_corrections_loveda_D
- retain_landcoverai_landcoverai
- net_corrections_landcoverai_landcoverai
- retain_flair1_flair1
- net_corrections_flair1_flair1

No full-image rollout or outcome-driven threshold/pseudocount/alias changes. The failed raw-box gate remains failed and is not retroactively relabeled. SAM3 is gated; its weights were not downloaded or used. Scholarly Search lacked authorization, and no verified novelty gap is claimed. Original Geometry and historical best results are preserved.
