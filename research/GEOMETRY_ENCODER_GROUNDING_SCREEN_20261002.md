# Native encoder-position Geometry coupling: verified screen

Fixed candidate evaluated on physical A800 GPUs0-7. Same96 development image IDs (UDD5 all40 IDs; eight each from seven other domains) and original one/two512 windows per image. WINDOW-ONLY statistics, not full-image dataset mIoU; overlapping windows can duplicate pixels. Corrected three-band IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. LoveDA D enters the equal-domain mean once and P is separate.

Pinned frozen public GroundingDINO tiny revision a2bb814dd30d776dcf7e30523b00659f4f141c71. Additional detection pretraining, not a matched single-encoder comparison or semantic-mask pixel supervision. No checkpoint download, remote custom code, target-label parameter fitting, alias deletion, confidence threshold or dataset routing. Standard Transformers forward also computes unused decoder outputs.

Native enc_outputs_class before object-query selection supplies mean sigmoid token responses over exact alias spans. Levelwise validity-weighted interpolation supplies32x32 observations; equal-level pooling and arithmetic mean of all20 aliases/class. b=.07*log(probability). Original cached Geometry and the unit-weight fidelity solver are unchanged. Unknown numeric observations recover exact original scores. LoveDA P maps the same D observation by class/alias identity. All response/candidate caches were saved before labels. Original Geometry confusions and transition endpoints recover exactly. Detector observation and quadratic solver are borrowed mechanisms.

## Window mIoU

| Domain/protocol | Geometry | EncoderGrounding | MeanLogit_Encoder | MeanProb_Encoder | Shuffled_EncoderCoupling | Geometry_EncoderCoupling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 37.8993 | 14.8814 | 29.1639 | 29.0721 | 36.1684 | 31.4494 |
| potsdam/potsdam | 40.7676 | 20.9370 | 36.2959 | 35.1980 | 48.2715 | 39.4649 |
| udd5/udd5 | 45.1736 | 27.3700 | 40.5017 | 40.1769 | 50.5134 | 43.5894 |
| oem/oem | 38.9652 | 12.6777 | 34.5372 | 35.1743 | 37.2686 | 36.4377 |
| loveda/P | 65.1016 | 16.9528 | 44.8020 | 45.4550 | 55.1203 | 45.4353 |
| loveda/D | 35.8503 | 9.1451 | 26.7590 | 26.9292 | 31.2614 | 26.7572 |
| vaihingen/vaihingen | 47.6994 | 31.7482 | 46.0683 | 45.3796 | 60.5805 | 50.2349 |
| landcoverai/landcoverai | 60.9049 | 12.6352 | 45.1765 | 44.7504 | 53.6916 | 48.2980 |
| flair1/flair1 | 38.8428 | 7.8252 | 28.2020 | 29.7097 | 30.0798 | 28.0736 |

| Method | Equal-domain mean |
| --- | ---: |
| Geometry | 43.26288635 |
| EncoderGrounding | 17.15245727 |
| MeanLogit_Encoder | 35.83803752 |
| MeanProb_Encoder | 35.79879145 |
| Shuffled_EncoderCoupling | 43.47941279 |
| Geometry_EncoderCoupling | 38.03813361 |

Prospective gate passed: False. Primary vs Geometry: -5.22475274pp; vs logit mean: +2.20009609pp; vs probability mean: +2.23934216pp; vs shuffled coupling: -5.44127919pp.

Failed checks: retain_vdd_vdd, retain_potsdam_potsdam, retain_udd5_udd5, retain_oem_oem, retain_loveda_P, retain_loveda_D, retain_landcoverai_landcoverai, retain_flair1_flair1

## Class effects

### vdd/vdd

Corrected/corrupted/wrong-to-wrong: 136363/591874/632669.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| other | 26.0715 | 5.8087 | 22.1766 | 22.9015 | -3.1700 | 36.709/37.845 | -100930 | -190805 |
| wall | 38.3866 | 10.8042 | 14.8747 | 15.4296 | -22.9570 | 15.430/99.972 | +13 | +738439 |
| road | 38.0339 | 7.6228 | 29.4673 | 30.6281 | -7.4058 | 30.900/97.209 | +673 | +77214 |
| vegetation | 43.0651 | 38.8222 | 46.7871 | 48.5866 | 5.5215 | 95.792/49.646 | +42103 | -46630 |
| vehicle | 6.1250 | 9.5206 | 17.7260 | 25.3730 | 19.2480 | 25.383/99.850 | -16 | -132548 |
| roof | 79.4937 | 19.5862 | 40.6591 | 44.1250 | -35.3687 | 74.810/51.825 | -385633 | +19256 |
| water | 34.1196 | 12.0049 | 32.4561 | 33.1019 | -1.0177 | 96.660/33.485 | -11721 | -9415 |

### potsdam/potsdam

Corrected/corrupted/wrong-to-wrong: 442013/510387/464268.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 59.5266 | 37.0399 | 56.6840 | 60.1876 | 0.6610 | 76.766/73.594 | +16397 | +6804 |
| building | 77.4448 | 25.8699 | 45.7497 | 49.1691 | -28.2757 | 50.716/94.159 | -19655 | +266921 |
| low vegetation | 27.6598 | 14.0267 | 18.4852 | 20.8216 | -6.8382 | 48.759/26.654 | -31711 | +141031 |
| tree | 63.9191 | 33.7400 | 55.6621 | 57.7218 | -6.1973 | 86.369/63.507 | -59353 | +35059 |
| car | 11.7333 | 9.8698 | 32.9513 | 39.9601 | 28.2268 | 40.780/95.211 | -3325 | -439467 |
| clutter | 4.3221 | 5.0757 | 8.2427 | 8.9293 | 4.6072 | 11.207/30.525 | +29273 | +58026 |

### udd5/udd5

Corrected/corrupted/wrong-to-wrong: 1567509/2356664/1916739.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vegetation | 76.0818 | 48.9432 | 68.3582 | 72.6374 | -3.4444 | 94.915/75.579 | -94193 | +172233 |
| building | 76.4461 | 37.9193 | 59.0508 | 62.4240 | -14.0221 | 78.573/75.230 | -1227388 | +487 |
| road | 38.9432 | 27.8071 | 40.1064 | 41.5925 | 2.6493 | 49.502/72.247 | +851797 | +1790414 |
| vehicle | 6.0762 | 7.9358 | 14.3629 | 19.3829 | 13.3067 | 19.474/97.652 | -938 | -1424418 |
| other | 28.3209 | 14.2447 | 20.6300 | 21.9102 | -6.4107 | 38.362/33.815 | -318433 | +250439 |

### oem/oem

Corrected/corrupted/wrong-to-wrong: 364474/397857/240532.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -80730 |
| rangeland | 49.8863 | 9.1851 | 47.2902 | 48.6430 | -1.2433 | 72.992/59.320 | -29528 | -33302 |
| developed space | 29.5609 | 5.9140 | 33.4287 | 35.0851 | 5.5242 | 57.937/47.077 | +91907 | +49854 |
| road | 44.7030 | 9.5543 | 33.8268 | 36.6123 | -8.0907 | 43.078/70.923 | -6458 | +85334 |
| tree | 48.3718 | 33.9813 | 49.2792 | 52.1084 | 3.7366 | 87.165/56.439 | +46045 | +14870 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +2346 |
| agriculture land | 78.6245 | 12.9122 | 66.2722 | 65.8936 | -12.7309 | 75.285/84.082 | -47855 | +39935 |
| building | 60.5747 | 29.8744 | 46.2000 | 53.1589 | -7.4158 | 63.708/76.249 | -87494 | -44924 |

### loveda/P

Corrected/corrupted/wrong-to-wrong: 36654/320049/106288.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 51.7113 | 0.0993 | 1.4061 | 1.8362 | -49.8751 | 1.836/100.000 | +77 | +77206 |
| road | 68.3951 | 26.4549 | 68.5339 | 66.5521 | -1.8430 | 66.971/99.069 | -1212 | +3481 |
| water | 87.3809 | 41.7517 | 82.1185 | 82.0753 | -5.3056 | 90.336/89.976 | -4305 | +40294 |
| barren | 50.2389 | 3.6104 | 7.3464 | 7.9361 | -42.3028 | 10.922/22.496 | -36948 | +151454 |
| tree | 48.0837 | 23.5973 | 48.1563 | 51.2275 | 3.1438 | 61.095/76.030 | -197 | -18125 |
| farm | 84.7995 | 6.2032 | 61.2506 | 62.9843 | -21.8152 | 90.152/67.638 | -240810 | +29085 |

### loveda/D

Corrected/corrupted/wrong-to-wrong: 216128/691946/306749.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 32.7230 | 3.1304 | 25.2747 | 25.3618 | -7.3612 | 56.692/31.456 | -133154 | +143382 |
| building | 12.0412 | 0.0716 | 1.1202 | 1.5872 | -10.4540 | 1.587/100.000 | +194 | +81857 |
| road | 37.7236 | 18.7210 | 37.7378 | 37.4420 | -0.2816 | 37.642/98.602 | -1817 | -2242 |
| water | 63.8501 | 25.4880 | 61.6019 | 60.0044 | -3.8457 | 65.777/87.240 | +5325 | +67832 |
| barren | 25.1752 | 1.7868 | 6.9692 | 7.4754 | -17.6998 | 12.916/15.072 | -13605 | +79575 |
| tree | 22.8266 | 11.9550 | 18.2265 | 20.2118 | -2.6148 | 21.714/74.505 | +4752 | +97906 |
| farm | 56.6124 | 2.8627 | 36.3829 | 35.2179 | -21.3945 | 54.789/49.646 | -337513 | +7508 |

### vaihingen/vaihingen

Corrected/corrupted/wrong-to-wrong: 638525/384953/326419.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 47.6657 | 42.2530 | 55.2929 | 59.7924 | 12.1267 | 76.477/73.267 | +309261 | +188512 |
| building | 74.7317 | 37.7374 | 55.7521 | 58.5514 | -16.1803 | 64.357/86.651 | -89163 | +126226 |
| low vegetation | 35.9788 | 19.9719 | 35.0194 | 38.6932 | 2.7144 | 73.160/45.095 | +75143 | +124593 |
| tree | 73.0101 | 38.8005 | 62.4452 | 65.3579 | -7.6522 | 76.916/81.306 | -29057 | +77013 |
| car | 7.1106 | 19.9779 | 21.8317 | 28.7795 | 21.6689 | 30.798/81.453 | -12612 | -769916 |

### landcoverai/landcoverai

Corrected/corrupted/wrong-to-wrong: 35420/338419/24687.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 82.6090 | 15.1797 | 59.1763 | 62.6573 | -19.9517 | 96.963/63.912 | -300291 | -13431 |
| building | 34.9562 | 6.8075 | 18.1479 | 23.9912 | -10.9650 | 24.219/96.220 | -976 | +37180 |
| woodland | 78.3638 | 35.1776 | 58.4887 | 62.7290 | -15.6348 | 67.577/89.737 | +2031 | +130724 |
| water | 93.6635 | 3.5083 | 84.2370 | 85.3267 | -8.3368 | 86.936/97.876 | -3684 | +13777 |
| road | 14.9318 | 2.5028 | 5.8326 | 6.7857 | -8.1461 | 6.929/76.642 | -79 | +134749 |

### flair1/flair1

Corrected/corrupted/wrong-to-wrong: 105572/375203/131536.

| Class | Geometry IoU | Encoder IoU | Logit mean IoU | Coupled IoU | Coupling delta | Coupled precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 49.6979 | 33.0424 | 50.1617 | 53.3921 | 3.6942 | 54.532/96.231 | +224 | -19713 |
| pervious surface | 57.0621 | 2.0647 | 44.6545 | 40.9381 | -16.1240 | 78.096/46.248 | -49417 | +28229 |
| impervious surface | 52.6509 | 23.8895 | 40.8587 | 42.4382 | -10.2127 | 50.039/73.642 | -10765 | +95148 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +19670 |
| water | 73.2328 | 6.2502 | 64.2952 | 60.9746 | -12.2582 | 71.138/81.017 | -11633 | +4797 |
| coniferous | 43.0233 | 2.1250 | 33.4656 | 32.9360 | -10.0873 | 42.912/58.622 | -8 | +4899 |
| deciduous | 54.0229 | 23.2741 | 52.0512 | 53.5947 | -0.4282 | 76.575/64.105 | +3588 | +10025 |
| brushwood | 18.6136 | 0.2311 | 18.0850 | 16.8504 | -1.7632 | 21.103/45.538 | -2081 | +15213 |
| vineyard | NA | 0.0000 | 0.0000 | 0.0000 | NA | 0.000/0.000 | +0 | +2034 |
| herbaceous vegetation | 60.2329 | 2.9910 | 30.8846 | 31.7221 | -28.5108 | 92.237/32.592 | -200855 | -6869 |
| agricultural land | 18.7339 | 0.0350 | 3.9674 | 4.0373 | -14.6966 | 4.082/78.786 | +1316 | +104879 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | +11319 |

## Cost

| Domain | Physical windows | Encoder seconds | Cached readout seconds | Collection wall seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 16 | 6.155 | 0.164 | 9.011 | 1691.213 |
| potsdam | 16 | 4.380 | 0.164 | 5.773 | 1672.037 |
| udd5 | 80 | 19.561 | 0.812 | 32.347 | 1648.727 |
| oem | 16 | 6.236 | 0.174 | 7.658 | 1690.851 |
| loveda | 16 | 4.357 | 0.326 | 6.433 | 1682.284 |
| vaihingen | 16 | 4.296 | 0.165 | 5.365 | 1648.727 |
| landcoverai | 8 | 2.394 | 0.089 | 2.983 | 1648.727 |
| flair1 | 8 | 5.357 | 0.098 | 6.325 | 1692.543 |

Parallel suite wall: 80.950s. Cost includes the full detector forward with unused decoder and all-arm cached readouts; excludes original DINO forwards, model initialization and label audit. Not standalone deployed-primary latency or memory.

No automatic full-image rollout or retrospective threshold/aggregation/source-scale search. A failed fixed observation does not rule out every frozen semantic source. All eight datasets are development data, not untouched validation. Original Geometry and historical Anchored_VIP are preserved. This screen alone cannot establish a useful original final module, CVPR readiness or superiority over complete official VIP.

## Interpretation

Only Vaihingen improves vs Geometry. The fixed native observation is weak even before coupling, and spatially shuffled coupling beats actual Geometry coupling by5.44127919pp mean. Actual geometric support does not validate these responses as reliable dense class observations. Native proposal classification is detection supervision, not semantic-mask supervision; this test does not establish which combination of dense alignment, calibration and interpolation causes the losses.

VDD roof loses385,633 true positives while wall adds738,439 false positives. Potsdam car improves IoU mainly by reducing false positives439,467, with3,325 fewer true positives, while building loses28.2757pp. Improved car IoU alone is not evidence that missing small objects were found. No source, threshold or dataset-specific winner is promoted. Reject this fixed route and preserve the historical best model. These window results must not replace full-image metrics.
