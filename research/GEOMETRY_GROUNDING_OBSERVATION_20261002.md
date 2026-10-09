# Frozen GroundingDINO local-observation diagnostic

All eight fixed development domains completed on physical A800 GPUs4-7. UDD5 full40 images and seven domains8 images each (96 total), but ONLY one or two image-only 512 windows per image. Results below are window-only statistics, NOT full-image dataset mIoU. Windows may overlap. Corrected IRRG Vaihingen; LandCover.ai replaces unlabeled iSAID.

The additional172M-parameter frozen detector supplies pretrained object-localization supervision. Pinned public revision a2bb814dd30d776dcf7e30523b00659f4f141c71; safetensors only, standard Transformers, no remote code or target-image upload. This is NOT a matched single-encoder comparison, an original detector, or evidence of surpassing complete official VIP.

All20 aliases per class are preserved with exact token spans. BoxMean uses mean sigmoid phrase responses and fixed0.25 confidence. BoxNativeMax is an audit control, not a retrospectively chosen replacement. Box rasterization uses fractional patch area, max class evidence and original Geometry fallback when absent. It is not a semantic mask or the proposed final coupled model. Raw boxes, alias scores and Geometry arrays were collected before any masks were loaded.

## Window-only metrics

| Domain/protocol | Images | Geometry | BoxMean | Delta | BoxNativeMax | Corrected | Corrupted | Box coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 8 | 37.8993 | 4.1291 | -33.7702 | 4.4664 | 488617 | 1993312 | 100.00% |
| potsdam/potsdam | 8 | 40.7676 | 8.6138 | -32.1538 | 8.4785 | 147709 | 2605095 | 100.00% |
| udd5/udd5 | 40 | 45.1736 | 24.1372 | -21.0365 | 25.3497 | 1664516 | 7239799 | 91.20% |
| oem/oem | 8 | 38.9652 | 6.9555 | -32.0097 | 2.8782 | 627405 | 2049056 | 100.00% |
| loveda/P | 8 | 65.1016 | 9.9669 | -55.1347 | 9.9669 | 114843 | 1037955 | 93.71% |
| loveda/D | 8 | 35.8503 | 6.3642 | -29.4861 | 6.3555 | 1127042 | 1846162 | 100.00% |
| vaihingen/vaihingen | 8 | 47.6994 | 21.6744 | -26.0250 | 21.0182 | 153207 | 1207206 | 77.65% |
| landcoverai/landcoverai | 8 | 60.9049 | 13.5520 | -47.3529 | 13.5520 | 212534 | 622741 | 100.00% |
| flair1/flair1 | 8 | 38.8428 | 0.6587 | -38.1841 | 0.5790 | 34806 | 1407672 | 100.00% |

Prospective source gate passed: **False**. Every protocol must have positive net corrections. No threshold/score/alias tuning or full rollout occurred.

## Per-class competition

### vdd/vdd

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| other | 26.0715 | 28.2146 | 2.1431 | 36.234/48.174 | 28.375/98.035 | +487191 | +1589595 |
| wall | 38.3866 | 0.0000 | -38.3866 | 38.392/99.965 | 0.000/0.000 | -190495 | +497124 |
| road | 38.0339 | 0.0000 | -38.0339 | 38.541/96.658 | 0.000/0.000 | -117999 | -188168 |
| vegetation | 43.0651 | 0.0000 | -43.0651 | 87.726/45.826 | 0.000/0.000 | -505090 | -70668 |
| vehicle | 6.1250 | 0.0000 | -6.1250 | 6.125/100.000 | 0.000/0.000 | -10697 | -162411 |
| roof | 79.4937 | 0.6894 | -78.8043 | 85.587/91.780 | 47.621/0.695 | -879128 | -141800 |
| water | 34.1196 | 0.0000 | -34.1196 | 93.828/34.903 | 0.000/0.000 | -288477 | -18977 |

### potsdam/potsdam

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 59.5266 | 0.0000 | -59.5266 | 76.864/72.521 | 0.000/0.000 | -1107803 | -333454 |
| building | 77.4448 | 0.7964 | -76.6484 | 78.142/98.861 | 38.741/0.807 | -409924 | -110274 |
| low vegetation | 27.6598 | 0.0000 | -27.6598 | 74.343/30.579 | 0.000/0.000 | -247044 | -85259 |
| tree | 63.9191 | 1.1537 | -62.7654 | 90.604/68.457 | 55.106/1.165 | -806915 | -73754 |
| car | 11.7333 | 45.5966 | 33.8633 | 11.735/99.847 | 74.971/53.784 | -33039 | -525760 |
| clutter | 4.3221 | 4.1360 | -0.1861 | 6.018/13.299 | 4.136/100.000 | +147339 | +3585887 |

### udd5/udd5

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| vegetation | 76.0818 | 10.0618 | -66.0200 | 98.330/77.078 | 66.148/10.608 | -4177279 | +258919 |
| building | 76.4461 | 55.9744 | -20.4717 | 81.792/92.124 | 69.604/74.083 | -1310745 | +860464 |
| road | 38.9432 | 24.3709 | -14.5723 | 69.690/46.884 | 39.780/38.619 | -277598 | +1278625 |
| vehicle | 6.0762 | 12.1318 | 6.0556 | 6.082/98.387 | 12.848/68.522 | -38138 | -1346505 |
| other | 28.3209 | 18.1469 | -10.1740 | 46.627/41.906 | 22.652/47.712 | +228477 | +4523780 |

### oem/oem

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| bareland | 0.0000 | NA | NA | 0.000/0.000 | 0.000/0.000 | +0 | -417518 |
| rangeland | 49.8863 | 0.0000 | -49.8863 | 70.829/62.786 | 0.000/0.000 | -534812 | -220263 |
| developed space | 29.5609 | 29.3725 | -0.1884 | 56.548/38.249 | 29.632/97.105 | +612748 | +2094773 |
| road | 44.7030 | 0.3795 | -44.3235 | 53.455/73.192 | 7.501/0.398 | -207171 | -167404 |
| tree | 48.3718 | 0.0408 | -48.3310 | 88.422/51.643 | 2.591/0.041 | -495370 | -49951 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +3245 |
| agriculture land | 78.6245 | 15.7747 | -62.8498 | 82.863/93.891 | 23.088/33.244 | -295877 | +445547 |
| building | 60.5747 | 3.1208 | -57.4539 | 64.095/91.688 | 43.144/3.255 | -501169 | -266778 |

### loveda/P

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 51.7113 | 0.0829 | -51.6284 | 53.236/94.751 | 0.083/10.157 | -1241 | +177148 |
| road | 68.3951 | 1.4768 | -66.9183 | 68.398/99.995 | 9.560/1.717 | -128704 | -39238 |
| water | 87.3809 | 1.8398 | -85.5411 | 96.082/90.609 | 70.782/1.854 | -602985 | -19902 |
| barren | 50.2389 | 0.0000 | -50.2389 | 73.017/61.693 | 0.000/0.000 | -58154 | -21391 |
| tree | 48.0837 | 9.3603 | -38.7234 | 56.616/76.138 | 92.619/9.431 | -121783 | -105143 |
| farm | 84.7995 | 47.0416 | -37.7579 | 94.855/88.888 | 50.271/87.984 | -10245 | +931638 |

### loveda/D

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 32.7230 | 44.5496 | 11.8266 | 69.509/38.207 | 46.362/91.934 | +1059640 | +1767224 |
| building | 12.0412 | 0.0000 | -12.0412 | 12.266/86.776 | 0.000/0.000 | -1273 | +11887 |
| road | 37.7236 | 0.0000 | -37.7236 | 37.725/99.989 | 0.000/0.000 | -130945 | -216158 |
| water | 63.8501 | 0.0000 | -63.8501 | 70.946/86.456 | 0.000/0.000 | -587366 | -240536 |
| barren | 25.1752 | 0.0000 | -25.1752 | 63.178/29.504 | 0.000/0.000 | -27812 | -16210 |
| tree | 22.8266 | 0.0000 | -22.8266 | 25.062/71.902 | 0.000/0.000 | -131268 | -392502 |
| farm | 56.6124 | 0.0000 | -56.6124 | 66.338/79.430 | 0.000/0.000 | -900096 | -194585 |

### vaihingen/vaihingen

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| impervious surface | 47.6657 | 23.3648 | -24.3009 | 84.267/52.322 | 61.903/27.289 | -369621 | +103738 |
| building | 74.7317 | 31.8076 | -42.9241 | 75.718/98.286 | 32.275/95.649 | -20208 | +1296526 |
| low vegetation | 35.9788 | 20.6487 | -15.3301 | 91.316/37.253 | 67.642/22.912 | -137439 | +71085 |
| tree | 73.0101 | 23.7322 | -49.2779 | 84.251/84.549 | 59.319/28.346 | -503527 | +32557 |
| car | 7.1106 | 8.8187 | 1.7081 | 7.112/99.811 | 9.238/66.036 | -23204 | -449907 |

### landcoverai/landcoverai

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| background | 82.6090 | 67.7600 | -14.8490 | 96.651/85.044 | 67.760/100.000 | +212534 | +634241 |
| building | 34.9562 | 0.0000 | -34.9562 | 35.044/99.292 | 0.000/0.000 | -31548 | -58477 |
| woodland | 78.3638 | 0.0000 | -78.3638 | 86.499/89.284 | 0.000/0.000 | -400826 | -62561 |
| water | 93.6635 | 0.0000 | -93.6635 | 93.664/100.000 | 0.000/0.000 | -173462 | -11735 |
| road | 14.9318 | 0.0000 | -14.9318 | 15.629/77.002 | 0.000/0.000 | -16905 | -91261 |

### flair1/flair1

| Class | Geometry IoU | BoxMean IoU | Delta | Geometry precision/recall | BoxMean precision/recall | TP delta | FP delta |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| building | 49.6979 | 1.6703 | -48.0276 | 50.726/96.082 | 58.456/1.690 | -142064 | -138662 |
| pervious surface | 57.0621 | 4.7712 | -52.2909 | 92.130/59.986 | 9.723/8.566 | -184970 | +267639 |
| impervious surface | 52.6509 | 0.0000 | -52.6509 | 62.624/76.777 | 0.000/0.000 | -263669 | -157363 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | -71111 |
| water | 73.2328 | 0.0000 | -73.2328 | 77.143/93.526 | 0.000/0.000 | -86972 | -23465 |
| coniferous | 43.0233 | 0.0000 | -43.0233 | 61.711/58.690 | 0.000/0.000 | -6916 | -4291 |
| deciduous | 54.0229 | 0.0000 | -54.0229 | 78.972/63.099 | 0.000/0.000 | -225154 | -59951 |
| brushwood | 18.6136 | 0.0000 | -18.6136 | 23.421/47.556 | 0.000/0.000 | -49032 | -160317 |
| vineyard | NA | NA | NA | 0.000/0.000 | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 0.4307 | -59.8022 | 94.317/62.501 | 14.492/0.442 | -416761 | -7779 |
| agricultural land | 18.7339 | 0.3734 | -18.3605 | 21.647/58.198 | 0.373/100.000 | +2672 | +1691971 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | +0 | +36195 |

## Cost

| Dataset | Protocol windows | Wall seconds | Geometry seconds | Observer seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 16 | 18.012 | 0.625 | 5.947 | 5126.091 |
| potsdam | 16 | 14.787 | 0.635 | 4.092 | 5108.835 |
| udd5 | 80 | 78.484 | 2.355 | 19.688 | 5086.234 |
| oem | 16 | 17.293 | 0.639 | 6.013 | 5127.519 |
| loveda | 32 | 25.193 | 0.637 | 4.164 | 5117.571 |
| vaihingen | 16 | 14.495 | 0.611 | 4.021 | 5086.234 |
| landcoverai | 8 | 7.390 | 0.403 | 2.088 | 5086.234 |
| flair1 | 8 | 10.850 | 0.400 | 4.933 | 5127.108 |

Protocol windows count LoveDA P/D separately; both reuse the same detector call. Times include collection and compressed-array writes but exclude model initialization and label audit. Component sums include warmup and are not synchronized steady-state deployed medians. Peak memory keeps both encoders resident.

## Decision boundary

The failure is not uniform lack of all object information. On these windows, Potsdam car improves11.7333->45.5966 IoU (precision11.735->74.971%, recall99.847->53.784%), but low vegetation drops27.6598->0 and tree63.9191->1.1537. VDD other recall rises48.174->98.035% while road/vegetation/vehicle/water recall collapses to0. Broad/background/residual detections dominate several class competitions; six domains have100% box coverage. This rejects treating a detection box as a dense, exclusive class assignment, not the possibility of useful object-specific localization evidence. No inference thresholds or class-specific rules were changed from these labels.

A failed source gate rejects this fixed observer/rasterization route as a justified next solver input. It does not prove every detected box useless or every localization model impossible. Box interiors mix classes, aliases induce different object queries, and unseen detections are not negatives. These window diagnostics cannot establish independent cross-domain generalization or full-dataset SOTA. Original Geometry, historical best candidates and all previous failed screens remain unchanged. The useful original Geometry coupling and complete eight-domain official-VIP superiority are still unachieved.
