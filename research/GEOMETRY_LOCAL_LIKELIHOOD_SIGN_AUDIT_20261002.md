# Fixed sign-removal counterfactual audit

All eight domains' cached windows from96 development images completed on physical A800 GPUs4-7. UDD5 full40 image IDs; seven other domains8 each. These are one/two windows per image, NOT full-image dataset mIoU; windows can overlap. Corrected IRRG Vaihingen; LandCover.ai replaces unlabeled iSAID. LoveDA D counts once in the equal-domain mean, P is separate.

This is a fixed computational diagnostic, NOT a new final model, confidence-threshold search or independent validation. Geometry and the full candidate replay authoritative original matrices exactly. All sign-counterfactual arrays were saved before masks. Weights,20 aliases, samples, temperature and original confidence rule are unchanged; no detector or encoder reruns.

PositiveOnly = g+max(g'-g,0); NegativeOnly = g+min(g'-g,0), for the Geometry-conditioned and matched box-only source. Zero is the neutral score increment. Positive-only evidence can still corrupt a correct class by boosting another class. The same fixed splits apply to every domain.

## Window-only mIoU

| Domain/protocol | Geometry | Geometry_LocalLikelihood | PositiveOnly_Geometry | NegativeOnly_Geometry | PositiveOnly_Box | NegativeOnly_Box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 37.8993 | 37.7829 | 37.8189 | 38.2361 | 37.7855 | 38.2368 |
| potsdam/potsdam | 40.7676 | 40.7685 | 39.3887 | 42.6538 | 39.0573 | 42.6148 |
| udd5/udd5 | 45.1736 | 44.7689 | 44.6307 | 45.4976 | 44.5389 | 45.4902 |
| oem/oem | 38.9652 | 36.5909 | 36.8748 | 38.6900 | 36.7466 | 38.6900 |
| loveda/P | 65.1016 | 60.7086 | 60.6087 | 65.0583 | 59.4052 | 65.0567 |
| loveda/D | 35.8503 | 34.6247 | 34.6151 | 35.8433 | 34.5480 | 35.8434 |
| vaihingen/vaihingen | 47.6994 | 51.3604 | 45.1482 | 55.5026 | 44.6142 | 55.3800 |
| landcoverai/landcoverai | 60.9049 | 60.2919 | 60.3515 | 60.8152 | 60.2105 | 60.8026 |
| flair1/flair1 | 38.8428 | 37.1553 | 37.3139 | 38.7269 | 37.1631 | 38.7252 |

| Method | Equal-domain window mean |
| --- | ---: |
| Geometry | 43.262886 |
| Geometry_LocalLikelihood | 42.917934 |
| PositiveOnly_Geometry | 42.017725 |
| NegativeOnly_Geometry | 44.495695 |
| PositiveOnly_Box | 41.833011 |
| NegativeOnly_Box | 44.472885 |

## Correction balance

| Domain/protocol | Full corrected/corrupted | PositiveOnly corrected/corrupted | NegativeOnly corrected/corrupted |
| --- | --- | --- | --- |
| vdd/vdd | 82128/81013 | 55125/57369 | 47641/27547 |
| potsdam/potsdam | 186314/104356 | 58521/91891 | 152706/15284 |
| udd5/udd5 | 542807/708115 | 204866/300619 | 397704/414119 |
| oem/oem | 73145/195751 | 16821/112751 | 65314/95169 |
| loveda/P | 4064/7610 | 2503/4822 | 1674/2859 |
| loveda/D | 13066/29563 | 5909/11576 | 7650/18392 |
| vaihingen/vaihingen | 492063/155592 | 35795/121031 | 518075/42790 |
| landcoverai/landcoverai | 5527/20676 | 3552/14269 | 2219/8808 |
| flair1/flair1 | 23546/56699 | 1465/38144 | 22829/19497 |

## Per-class effect

### vdd/vdd

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| other | 26.0715 | 25.1232 | 24.5711 | 26.6368 | 35.197/44.870 |
| wall | 38.3866 | 40.1387 | 38.2265 | 41.2141 | 38.231/99.968 |
| road | 38.0339 | 34.0972 | 37.7506 | 34.0280 | 38.319/96.222 |
| vegetation | 43.0651 | 44.7833 | 45.6314 | 43.6051 | 86.834/49.023 |
| vehicle | 6.1250 | 8.6910 | 6.3106 | 9.2758 | 6.311/100.000 |
| roof | 79.4937 | 77.5332 | 78.1831 | 78.7143 | 84.482/91.294 |
| water | 34.1196 | 34.1137 | 34.0587 | 34.1785 | 93.352/34.905 |

### potsdam/potsdam

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| impervious surface | 59.5266 | 60.4142 | 58.3045 | 61.5087 | 77.477/70.204 |
| building | 77.4448 | 68.0955 | 69.7717 | 76.4327 | 73.583/93.090 |
| low vegetation | 27.6598 | 28.7814 | 26.9748 | 29.6122 | 72.548/30.041 |
| tree | 63.9191 | 67.1803 | 65.6080 | 66.3204 | 89.464/71.102 |
| car | 11.7333 | 15.6110 | 11.4234 | 17.0193 | 11.441/98.676 |
| clutter | 4.3221 | 4.5282 | 4.2498 | 5.0297 | 5.924/13.074 |

### udd5/udd5

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| vegetation | 76.0818 | 75.8716 | 75.7609 | 76.4020 | 97.981/76.963 |
| building | 76.4461 | 71.9803 | 75.3145 | 73.2565 | 80.839/91.682 |
| road | 38.9432 | 40.6285 | 38.8768 | 41.1379 | 67.579/47.790 |
| vehicle | 6.0762 | 8.0541 | 6.2911 | 8.0325 | 6.300/97.889 |
| other | 28.3209 | 27.3099 | 26.9104 | 28.6590 | 45.492/39.717 |

### oem/oem

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 |
| rangeland | 49.8863 | 48.9121 | 48.8294 | 49.8934 | 70.704/61.215 |
| developed space | 29.5609 | 28.2768 | 26.2992 | 31.1665 | 57.749/32.565 |
| road | 44.7030 | 41.4366 | 41.9498 | 44.4267 | 49.739/72.817 |
| tree | 48.3718 | 46.2225 | 47.2599 | 47.2696 | 86.801/50.919 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 |
| agriculture land | 78.6245 | 76.1524 | 77.2564 | 77.6963 | 83.768/90.858 |
| building | 60.5747 | 51.7266 | 53.4037 | 59.0673 | 56.158/91.589 |

### loveda/P

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| building | 51.7113 | 26.4061 | 25.6828 | 51.5619 | 46.367/36.537 |
| road | 68.3951 | 68.0215 | 68.6258 | 67.7988 | 68.743/99.752 |
| water | 87.3809 | 86.9187 | 86.9970 | 87.3082 | 95.553/90.668 |
| barren | 50.2389 | 50.4075 | 50.0210 | 50.6202 | 74.175/60.569 |
| tree | 48.0837 | 47.8893 | 47.5878 | 48.3863 | 56.331/75.406 |
| farm | 84.7995 | 84.6086 | 84.7378 | 84.6745 | 94.692/88.963 |

### loveda/D

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| background | 32.7230 | 32.1255 | 32.5543 | 32.2954 | 69.232/38.061 |
| building | 12.0412 | 3.8564 | 3.7587 | 11.9959 | 4.039/35.174 |
| road | 37.7236 | 37.7864 | 38.0959 | 37.4557 | 38.133/99.746 |
| water | 63.8501 | 63.1019 | 63.1485 | 63.8143 | 70.065/86.482 |
| barren | 25.1752 | 26.4250 | 25.3797 | 26.1951 | 65.114/29.374 |
| tree | 22.8266 | 22.3913 | 22.5249 | 22.6924 | 24.787/71.171 |
| farm | 56.6124 | 56.6866 | 56.8438 | 56.4544 | 66.682/79.393 |

### vaihingen/vaihingen

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| impervious surface | 47.6657 | 62.1412 | 44.8789 | 66.5802 | 84.895/48.773 |
| building | 74.7317 | 68.7881 | 70.2076 | 73.6657 | 71.435/97.611 |
| low vegetation | 35.9788 | 40.1782 | 35.1515 | 41.2918 | 88.872/36.770 |
| tree | 73.0101 | 68.9073 | 68.8369 | 72.8858 | 80.632/82.474 |
| car | 7.1106 | 16.7874 | 6.6661 | 23.0897 | 6.698/93.350 |

### landcoverai/landcoverai

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| background | 82.6090 | 81.6030 | 81.9126 | 82.1453 | 96.618/84.330 |
| building | 34.9562 | 35.3962 | 34.3457 | 36.4126 | 34.459/99.053 |
| woodland | 78.3638 | 76.1273 | 77.0126 | 77.1302 | 84.965/89.163 |
| water | 93.6635 | 93.6458 | 93.6458 | 93.6635 | 93.646/100.000 |
| road | 14.9318 | 14.6872 | 14.8408 | 14.7246 | 15.522/77.189 |

### flair1/flair1

| Class | Geometry | Full | PositiveOnly | NegativeOnly | PositiveOnly precision/recall |
| --- | ---: | ---: | ---: | ---: | --- |
| building | 49.6979 | 47.1074 | 46.0658 | 51.2556 | 47.174/95.148 |
| pervious surface | 57.0621 | 57.5351 | 55.8515 | 58.6880 | 92.114/58.656 |
| impervious surface | 52.6509 | 50.6515 | 51.1671 | 52.0813 | 62.301/74.114 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 |
| water | 73.2328 | 64.8965 | 65.6322 | 72.5826 | 69.248/92.631 |
| coniferous | 43.0233 | 41.1311 | 43.0287 | 41.1336 | 61.722/58.690 |
| deciduous | 54.0229 | 52.5517 | 53.4730 | 53.0877 | 79.381/62.098 |
| brushwood | 18.6136 | 18.0491 | 18.4439 | 18.2031 | 23.242/47.187 |
| vineyard | NA | NA | NA | NA | 0.000/0.000 |
| herbaceous vegetation | 60.2329 | 58.2284 | 57.9520 | 60.5361 | 94.350/60.036 |
| agricultural land | 18.7339 | 18.5573 | 18.8383 | 18.4277 | 21.786/58.198 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 |

## Interpretation boundary

PositiveOnly_Geometry wins vs Geometry in0/8 domains. Its equal-domain delta is-1.245161pp. Removing a sign component measures the conditional effect of that fixed computational intervention, not a causal image-semantic truth.

The earlier explanation that harmful updates mainly came from lower likelihood outside incomplete detections was a hypothesis, not a finding. PositiveOnly removes those negative increments yet fails on every domain and both LoveDA protocols. Positive location boosts therefore also cause harmful class competition under this readout. This does not uniquely separate loose box coverage, incorrect phrase response, or likelihood calibration.

NegativeOnly_Geometry wins in4/8 domains and changes the equal-domain mean by+1.232809pp. However, its advantage over NegativeOnly_Box is only+0.022810pp. The favorable negative component is neither a universal eight-domain correction nor strong evidence of a useful new Geometry coupling. It is not promoted after this labeled audit.

Neither split was selected for a full rollout. Do not use a different winning split per dataset, call zero-clipping a CVPR contribution, or claim a window metric is full-domain superiority over complete official VIP. Another coverage-only or positive-only gate is not supported by these results. A successor needs a different, demonstrably useful pixel-specific semantic observation, with its measurement source and extra supervision explicit, before another reconstruction solver. The full useful Geometry-coupling objective remains active and unachieved.
