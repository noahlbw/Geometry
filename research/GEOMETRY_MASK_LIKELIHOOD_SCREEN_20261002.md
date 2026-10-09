# Frozen pixel-support Geometry likelihood: verified eight-domain screen

Complete fixed candidate on physical A800 GPUs4-7. Same96 development image IDs:40 UDD5, eight from each other domain; one/two original512 windows each. NOT full-image dataset mIoU. Corrected IRRG Vaihingen; LandCover.ai replaces unlabeled iSAID; LoveDA D enters the mean once, P separately.

Public SAM2.1 tiny revision de431c4043854a71d8101e17995dfe596bf101a5 supplies image-only pixel supports to the same cached GroundingDINO observations. Safetensors/standard Transformers, exact checkpoint loading (no missing/extra/shape-mismatched weights). No image/prompt/label upload. It adds pretrained detection and segmentation supervision and is not a matched single-encoder system. SAM2/Grounded-SAM and ordinary likelihood updates are borrowed mechanisms, not our invention.

Same20 aliases/class, phrase scores/cutoff .25, original Geometry, temperature .07, pseudocount1. Only the support changes: b times SAM2 sigmoid-mask patch means. No score/alias selection, NMS, query cap, sign clipping or dataset-specific routing. Constant box support remains neutral. All candidate scores/supports were saved before loading labels. Unit Geometry exactly recovers mask-only control; original Geometry and both original box-readout scores/confusions reproduce exactly.

## Window mIoU

| Domain/protocol | Geometry | BoxLocalLikelihood | Geometry_LocalLikelihood | MaskLocalLikelihood | MeanProb_MaskLikelihood | ShuffledMaskLikelihood | Geometry_MaskLikelihood |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 37.8993 | 37.7433 | 37.7829 | 37.9049 | 38.1309 | 37.8765 | 37.9075 |
| potsdam/potsdam | 40.7676 | 40.3792 | 40.7685 | 42.1349 | 41.9626 | 42.1435 | 42.2103 |
| udd5/udd5 | 45.1736 | 44.6772 | 44.7689 | 44.9074 | 45.2272 | 44.9023 | 44.9234 |
| oem/oem | 38.9652 | 36.4458 | 36.5909 | 37.4314 | 38.1241 | 37.4601 | 37.5019 |
| loveda/P | 65.1016 | 59.4690 | 60.7086 | 62.7577 | 62.9508 | 62.9573 | 62.4067 |
| loveda/D | 35.8503 | 34.5590 | 34.6247 | 34.7697 | 35.4189 | 34.8230 | 34.7749 |
| vaihingen/vaihingen | 47.6994 | 50.6736 | 51.3604 | 54.1044 | 51.7721 | 54.1689 | 54.2809 |
| landcoverai/landcoverai | 60.9049 | 60.1290 | 60.2919 | 60.7505 | 60.7987 | 60.7021 | 60.7821 |
| flair1/flair1 | 38.8428 | 36.9995 | 37.1553 | 37.2322 | 38.0901 | 37.3114 | 37.2819 |

| Method | Equal-domain mean |
| --- | ---: |
| Geometry | 43.262886 |
| BoxLocalLikelihood | 42.700817 |
| Geometry_LocalLikelihood | 42.917934 |
| MaskLocalLikelihood | 43.654426 |
| MeanProb_MaskLikelihood | 43.690578 |
| ShuffledMaskLikelihood | 43.673482 |
| Geometry_MaskLikelihood | 43.707858 |

Primary wins vs Geometry:3/8; promotion gate passed:False. No full-image rollout or retrospectively selected control.

Primary delta vs Geometry:+0.444972pp; vs original box-coupled readout:+0.789924pp; vs mask-only:+0.053433pp; vs same-source simple blend:+0.017280pp. An average gain does not establish eight-domain stability or useful novel coupling.

## Per-class effect and transitions

### vdd/vdd

Corrected/corrupted/wrong-to-wrong:79817/75481/156010.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| other | 26.0715 | 25.1232 | 25.1462 | -0.9253 | 35.178/46.859 | -12850 | +15336 |
| wall | 38.3866 | 40.1387 | 40.2065 | 1.8199 | 40.209/99.986 | +41 | -22360 |
| road | 38.0339 | 34.0972 | 34.4430 | -3.5909 | 34.608/98.635 | +2414 | +39354 |
| vegetation | 43.0651 | 44.7833 | 45.5315 | 2.4664 | 88.253/48.469 | +29126 | +438 |
| vehicle | 6.1250 | 8.6910 | 9.1295 | 3.0045 | 9.130/99.907 | -10 | -57584 |
| roof | 79.4937 | 77.5332 | 76.7495 | -2.7442 | 83.657/90.287 | -14412 | +21067 |
| water | 34.1196 | 34.1137 | 34.1465 | 0.0269 | 94.008/34.906 | +27 | -587 |

### potsdam/potsdam

Corrected/corrupted/wrong-to-wrong:196937/66188/171233.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 59.5266 | 60.4142 | 61.7964 | 2.2698 | 74.981/77.848 | +81380 | +63332 |
| building | 77.4448 | 68.0955 | 72.9360 | -4.5088 | 77.336/92.764 | -25488 | -1955 |
| low vegetation | 27.6598 | 28.7814 | 29.4779 | 1.8181 | 73.341/33.016 | +19687 | +11698 |
| tree | 63.9191 | 67.1803 | 67.9423 | 4.0232 | 90.459/73.187 | +56719 | +7436 |
| car | 11.7333 | 15.6110 | 16.4337 | 4.7004 | 16.530/96.573 | -2348 | -188866 |
| clutter | 4.3221 | 4.5282 | 4.6751 | 0.3530 | 6.611/13.770 | +799 | -22394 |

### udd5/udd5

Corrected/corrupted/wrong-to-wrong:578869/721536/581494.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vegetation | 76.0818 | 75.8716 | 76.2297 | 0.1479 | 98.057/77.399 | +20166 | +14103 |
| building | 76.4461 | 71.9803 | 71.9542 | -4.4919 | 80.977/86.591 | -402034 | -12164 |
| road | 38.9432 | 40.6285 | 40.9318 | 1.9886 | 65.526/52.165 | +177368 | +236892 |
| vehicle | 6.0762 | 8.0541 | 8.2800 | 2.2038 | 8.297/97.620 | -979 | -562196 |
| other | 28.3209 | 27.3099 | 27.2210 | -1.0999 | 42.107/43.502 | +62812 | +466032 |

### oem/oem

Corrected/corrupted/wrong-to-wrong:85583/164839/103198.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -2093 |
| rangeland | 49.8863 | 48.9121 | 49.3260 | -0.5603 | 70.576/62.096 | -5882 | +253 |
| developed space | 29.5609 | 28.2768 | 29.6560 | 0.0951 | 51.645/41.056 | +29223 | +94219 |
| road | 44.7030 | 41.4366 | 43.4558 | -1.2472 | 53.053/70.607 | -7358 | -3558 |
| tree | 48.3718 | 46.2225 | 47.3253 | -1.0465 | 87.176/50.866 | -7451 | +6919 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +18845 |
| agriculture land | 78.6245 | 76.1524 | 75.0284 | -3.5961 | 83.144/88.488 | -26362 | -7212 |
| building | 60.5747 | 51.7266 | 55.2240 | -5.3507 | 63.535/80.849 | -61426 | -28117 |

### loveda/P

Corrected/corrupted/wrong-to-wrong:5940/9044/2898.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 51.7113 | 26.4061 | 35.8242 | -15.8871 | 64.876/44.444 | -738 | -868 |
| road | 68.3951 | 68.0215 | 67.7430 | -0.6521 | 67.745/99.995 | +0 | +1843 |
| water | 87.3809 | 86.9187 | 87.3378 | -0.0431 | 96.175/90.481 | -873 | -652 |
| barren | 50.2389 | 50.4075 | 51.2308 | 0.9919 | 74.083/62.417 | +683 | -908 |
| tree | 48.0837 | 47.8893 | 47.8491 | -0.2346 | 56.289/76.140 | +5 | +1428 |
| farm | 84.7995 | 84.6086 | 84.4552 | -0.3443 | 94.642/88.696 | -2181 | +2261 |

### loveda/D

Corrected/corrupted/wrong-to-wrong:20093/33262/29421.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 32.7230 | 32.1255 | 32.2909 | -0.4321 | 69.198/37.711 | -9785 | +514 |
| building | 12.0412 | 3.8564 | 4.0333 | -8.0079 | 4.282/40.968 | -672 | +4329 |
| road | 37.7236 | 37.7864 | 37.5683 | -0.1553 | 37.570/99.989 | +0 | +1435 |
| water | 63.8501 | 63.1019 | 63.4004 | -0.4497 | 70.537/86.238 | -1487 | +4179 |
| barren | 25.1752 | 26.4250 | 26.8897 | 1.7145 | 62.848/31.972 | +2326 | +1606 |
| tree | 22.8266 | 22.3913 | 22.6359 | -0.1907 | 24.775/72.389 | +888 | +8767 |
| farm | 56.6124 | 56.6866 | 56.6060 | -0.0064 | 66.605/79.038 | -4439 | -7661 |

### vaihingen/vaihingen

Corrected/corrupted/wrong-to-wrong:549180/106639/274594.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 47.6657 | 62.1412 | 65.5141 | 17.8484 | 77.449/80.957 | +422806 | +203812 |
| building | 74.7317 | 68.7881 | 73.6213 | -1.1104 | 77.773/93.240 | -38671 | -37326 |
| low vegetation | 35.9788 | 40.1782 | 41.1498 | 5.1710 | 86.106/44.077 | +65387 | +34208 |
| tree | 73.0101 | 68.9073 | 70.1847 | -2.8254 | 80.067/85.044 | +4429 | +48077 |
| car | 7.1106 | 16.7874 | 20.9348 | 13.8242 | 21.859/83.203 | -11410 | -691312 |

### landcoverai/landcoverai

Corrected/corrupted/wrong-to-wrong:8451/17237/3773.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 82.6090 | 81.6030 | 81.9927 | -0.6163 | 96.561/84.459 | -8310 | +861 |
| building | 34.9562 | 35.3962 | 37.0396 | 2.0834 | 37.146/99.235 | -18 | -5125 |
| woodland | 78.3638 | 76.1273 | 76.3604 | -2.0034 | 84.161/89.176 | -487 | +12782 |
| water | 93.6635 | 93.6458 | 93.5715 | -0.0920 | 93.572/100.000 | +0 | +182 |
| road | 14.9318 | 14.6872 | 14.9460 | 0.0142 | 15.639/77.134 | +29 | +86 |

### flair1/flair1

Corrected/corrupted/wrong-to-wrong:23906/57819/39667.

| Class | Geometry IoU | Original box coupling | Mask coupling | Delta vs Geometry | Mask precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 49.6979 | 47.1074 | 49.5369 | -0.1610 | 53.050/88.208 | -11852 | -22980 |
| pervious surface | 57.0621 | 57.5351 | 57.3709 | 0.3088 | 90.964/60.838 | +3064 | +3305 |
| impervious surface | 52.6509 | 50.6515 | 49.8471 | -2.8038 | 58.645/76.866 | +308 | +28786 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +3929 |
| water | 73.2328 | 64.8965 | 66.2209 | -7.0119 | 70.196/92.122 | -1306 | +10603 |
| coniferous | 43.0233 | 41.1311 | 41.0824 | -1.9409 | 57.615/58.876 | +22 | +813 |
| deciduous | 54.0229 | 52.5517 | 52.5022 | -1.5207 | 79.002/61.017 | -7432 | -2084 |
| brushwood | 18.6136 | 18.0491 | 15.8887 | -2.7249 | 20.403/41.798 | -5937 | +7809 |
| vineyard | NA | NA | NA | NA | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 58.2284 | 58.5560 | -1.6769 | 93.842/60.896 | -10780 | +1546 |
| agricultural land | 18.7339 | 18.5573 | 19.0955 | 0.3616 | 22.131/58.198 | +0 | -376 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +2562 |

## Support audit

Separate label-only audit of already saved observations. Mass is score-weighted across all20 aliases and all qualifying queries; overlapping windows/queries count repeatedly. This is patch-averaged supported-label precision, not object AP or dense class IoU. Greater precision can result from lost coverage, so correct mass retention is also reported.

| Domain/protocol | Class | Box support precision | Mask support precision | Mask/box correct mass |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | other | 27.1830 | 23.6472 | 0.6409 |
| vdd/vdd | wall | 28.4236 | 33.5514 | 0.9426 |
| vdd/vdd | road | 21.8836 | 33.7104 | 0.9033 |
| vdd/vdd | vegetation | 73.7181 | 94.1223 | 0.8487 |
| vdd/vdd | vehicle | 35.4175 | 53.7044 | 0.6750 |
| vdd/vdd | roof | 33.1619 | 45.8844 | 0.8960 |
| vdd/vdd | water | 43.2212 | 54.2711 | 0.8626 |
| potsdam/potsdam | impervious surface | 64.3101 | 82.9703 | 0.6271 |
| potsdam/potsdam | building | 41.0705 | 58.5853 | 0.8113 |
| potsdam/potsdam | low vegetation | 23.6263 | 26.3654 | 0.6643 |
| potsdam/potsdam | tree | 58.2396 | 83.2962 | 0.7722 |
| potsdam/potsdam | car | 33.0389 | 60.5795 | 0.7877 |
| potsdam/potsdam | clutter | 4.7085 | 4.1991 | 0.6055 |
| udd5/udd5 | vegetation | 75.6137 | 84.9247 | 0.8096 |
| udd5/udd5 | building | 65.5433 | 70.5582 | 0.7953 |
| udd5/udd5 | road | 27.6142 | 37.5149 | 0.8717 |
| udd5/udd5 | vehicle | 21.4442 | 29.2251 | 0.7545 |
| udd5/udd5 | other | 15.8181 | 15.5508 | 0.6448 |
| oem/oem | bareland | 0.0000 | 0.0000 | NA |
| oem/oem | rangeland | 9.4963 | 1.8316 | 0.1348 |
| oem/oem | developed space | 23.6347 | 21.9032 | 0.5293 |
| oem/oem | road | 11.2843 | 38.4854 | 0.6308 |
| oem/oem | tree | 65.7794 | 78.3075 | 0.8979 |
| oem/oem | water | 0.0342 | 0.0009 | 0.0093 |
| oem/oem | agriculture land | 24.2957 | 27.2749 | 0.7335 |
| oem/oem | building | 41.2065 | 58.1154 | 0.7171 |
| loveda/P | building | 0.0000 | 0.0000 | NA |
| loveda/P | road | 0.0000 | 0.0000 | NA |
| loveda/P | water | 67.3332 | 97.0251 | 0.8346 |
| loveda/P | barren | 0.0139 | 0.0096 | 0.5598 |
| loveda/P | tree | 0.0000 | 0.0000 | NA |
| loveda/P | farm | 40.4346 | 45.3894 | 0.8067 |
| loveda/D | background | 47.6438 | 49.6600 | 0.7947 |
| loveda/D | building | 0.0000 | 0.0000 | NA |
| loveda/D | road | 0.0000 | 0.0000 | NA |
| loveda/D | water | 42.3349 | 69.8752 | 0.8346 |
| loveda/D | barren | 0.0132 | 0.0093 | 0.5598 |
| loveda/D | tree | 0.0000 | 0.0000 | NA |
| loveda/D | farm | 18.9425 | 19.9317 | 0.8067 |
| vaihingen/vaihingen | impervious surface | 42.4946 | 59.1803 | 0.5556 |
| vaihingen/vaihingen | building | 48.7961 | 65.8798 | 0.7727 |
| vaihingen/vaihingen | low vegetation | 11.2468 | 14.5848 | 0.6163 |
| vaihingen/vaihingen | tree | 24.6408 | 35.6435 | 0.6627 |
| vaihingen/vaihingen | car | 24.1327 | 46.2931 | 0.7236 |
| landcoverai/landcoverai | background | 66.4319 | 63.7326 | 0.7076 |
| landcoverai/landcoverai | building | 24.0567 | 28.1185 | 0.8427 |
| landcoverai/landcoverai | woodland | 0.0000 | 0.0000 | NA |
| landcoverai/landcoverai | water | 0.0000 | 0.0000 | NA |
| landcoverai/landcoverai | road | 4.3450 | 6.8445 | 0.5820 |
| flair1/flair1 | building | 47.2039 | 69.1219 | 0.8158 |
| flair1/flair1 | pervious surface | 15.0486 | 20.8035 | 0.7716 |
| flair1/flair1 | impervious surface | 3.7492 | 11.5809 | 0.5735 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | NA |
| flair1/flair1 | water | 0.0000 | 0.0000 | NA |
| flair1/flair1 | coniferous | 0.0000 | 0.0000 | NA |
| flair1/flair1 | deciduous | 0.3655 | 0.0877 | 0.1251 |
| flair1/flair1 | brushwood | 0.3356 | 0.0323 | 0.0492 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | NA |
| flair1/flair1 | herbaceous vegetation | 33.6850 | 35.1950 | 0.7247 |
| flair1/flair1 | agricultural land | 0.3552 | 0.2057 | 0.4120 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | NA |

## Runtime and interpretation

| Domain | SAM2 seconds | Cached readout seconds | Collection wall seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 1.5029 | 0.6971 | 5.7197 | 694.144 |
| potsdam | 1.5008 | 0.5515 | 3.8987 | 958.246 |
| udd5 | 5.5826 | 2.4616 | 23.5046 | 841.025 |
| oem | 1.5815 | 0.7105 | 4.4194 | 833.258 |
| loveda | 1.3415 | 1.0559 | 4.7486 | 514.551 |
| vaihingen | 1.5127 | 0.5452 | 3.5955 | 914.194 |
| landcoverai | 0.8735 | 0.3110 | 1.9972 | 524.695 |
| flair1 | 1.0314 | 0.5533 | 2.9550 | 657.161 |

Collection costs include SAM2 and cached readouts, but exclude original DINO and detector forwards. Not independent deployed-primary latency/memory. Parallel suite wall:61.1088 seconds, including startup/control/audit. Eight source datasets were already used for development; this is not untouched validation or superiority over complete official VIP.

Mask refinement improves the original rectangular-support readout in every protocol, so spatial imprecision matters. Remaining domain losses show that refinement alone does not resolve cross-class semantic likelihood validity. The small matched-coupling advantage does not meet the requested useful-original-module goal. No post-label gain, confidence, mask-source or alias search was performed. Preserve original Geometry and all historical candidates.
