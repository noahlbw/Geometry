# Native wide-view cross-validation: verified development screen

Fixed96 unique COMPLETE images: UDD5 full40 and eight each from seven other domains. All domains informed prior development; no untouched validation or SOTA claim. Corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. Same frozen DINOv3/DINO.text, original Geometry, all20 aliases, six RS templates, LME/temperature .07. Native wide head has no VIP proxy or salience. Two same-scale448-long-edge crop grids differ by56px; zero RGB padding can affect border observations. Analytic opposite-view coefficient is per tile, not class-specific correctness or independent semantic evidence.

| Dataset/protocol | Geometry | Native wide | Simple mean | Ungated coupling | Validated coupling | Delta vs Geometry |
|---|---:|---:|---:|---:|---:|---:|
| vdd/vdd | 31.9462 | 30.9058 | 37.0770 | 36.4111 | 36.4547 | +4.5085 |
| potsdam/potsdam | 40.3528 | 16.6687 | 28.9390 | 29.0435 | 29.0498 | -11.3030 |
| udd5/udd5 | 50.5553 | 27.2628 | 43.2309 | 42.8899 | 42.8674 | -7.6879 |
| oem/oem | 39.3543 | 32.4714 | 38.9272 | 39.1952 | 39.1952 | -0.1592 |
| loveda/P | 62.8254 | 41.2301 | 54.5419 | 54.5885 | 54.5885 | -8.2369 |
| loveda/D | 38.4558 | 24.5558 | 35.7360 | 35.5080 | 35.5080 | -2.9478 |
| vaihingen/vaihingen | 50.2325 | 30.6825 | 44.7163 | 44.4237 | 44.4237 | -5.8088 |
| landcoverai/landcoverai | 60.9049 | 48.4338 | 59.7572 | 58.8651 | 58.8643 | -2.0406 |
| flair1/flair1 | 38.8428 | 23.8535 | 34.4836 | 33.8774 | 33.8774 | -4.9653 |
| Eight-domain mean, LoveDA D once | 43.830569 | 29.354269 | 40.358380 | 40.026737 | 40.030056 | -3.800513 |

Prospective advancement checks (no automatic full rollout):

```json
{
  "passed": false,
  "mean_miou": {
    "Geometry": 43.83056914141856,
    "NativeWide": 29.354269310258303,
    "MeanLogit_Native": 40.35837978688807,
    "Anchored_Native": 40.026737262650045,
    "Validated_Native": 40.03005571939474
  },
  "values": {
    "potsdam": {
      "potsdam": {
        "Geometry": 40.35283588074661,
        "NativeWide": 16.668659181256203,
        "MeanLogit_Native": 28.93897446379859,
        "Anchored_Native": 29.043485148689268,
        "Validated_Native": 29.049788253641122
      }
    },
    "oem": {
      "oem": {
        "Geometry": 39.35432655571748,
        "NativeWide": 32.47135092165708,
        "MeanLogit_Native": 38.927221746946266,
        "Anchored_Native": 39.19515337565755,
        "Validated_Native": 39.19515337565755
      }
    },
    "flair1": {
      "flair1": {
        "Geometry": 38.84276353131189,
        "NativeWide": 23.853494097691332,
        "MeanLogit_Native": 34.4835756320416,
        "Anchored_Native": 33.87744641494982,
        "Validated_Native": 33.87744641494982
      }
    },
    "loveda": {
      "P": {
        "Geometry": 62.825368077331035,
        "NativeWide": 41.230067916888494,
        "MeanLogit_Native": 54.54193746264203,
        "Anchored_Native": 54.58846031095867,
        "Validated_Native": 54.58846031095867
      },
      "D": {
        "Geometry": 38.45582823361516,
        "NativeWide": 24.555801414547748,
        "MeanLogit_Native": 35.73596110303802,
        "Anchored_Native": 35.508006484926284,
        "Validated_Native": 35.508006484926284
      }
    },
    "vaihingen": {
      "vaihingen": {
        "Geometry": 50.23246409731753,
        "NativeWide": 30.682496030038063,
        "MeanLogit_Native": 44.716257676242364,
        "Anchored_Native": 44.423709830511434,
        "Validated_Native": 44.423709830511434
      }
    },
    "landcoverai": {
      "landcoverai": {
        "Geometry": 60.90485348087695,
        "NativeWide": 48.43380855681101,
        "MeanLogit_Native": 59.75715414970777,
        "Anchored_Native": 58.86514259404857,
        "Validated_Native": 58.86425478783469
      }
    },
    "vdd": {
      "vdd": {
        "Geometry": 31.946179952916985,
        "NativeWide": 30.90578897954791,
        "MeanLogit_Native": 37.07698523829734,
        "Anchored_Native": 36.41106143374008,
        "Validated_Native": 36.45472720479078
      }
    },
    "udd5": {
      "udd5": {
        "Geometry": 50.55530139884586,
        "NativeWide": 27.26275530051708,
        "MeanLogit_Native": 43.23090828503261,
        "Anchored_Native": 42.88989281867739,
        "Validated_Native": 42.867359402846276
      }
    }
  },
  "checks": {
    "mean_vs_Geometry": false,
    "mean_vs_MeanLogit_Native": false,
    "mean_vs_Anchored_Native": true,
    "retain_potsdam_potsdam": false,
    "potsdam_vs_Geometry": false,
    "potsdam_vs_Anchored_Native": true,
    "retain_oem_oem": true,
    "retain_flair1_flair1": false,
    "retain_loveda_P": false,
    "retain_loveda_D": false,
    "retain_vaihingen_vaihingen": false,
    "retain_landcoverai_landcoverai": false,
    "retain_vdd_vdd": true,
    "vdd_vs_Geometry": true,
    "vdd_vs_Anchored_Native": true,
    "retain_udd5_udd5": false
  }
}
```

## Historical VIP-based coupling on the same IDs

Historical Anchored_VIP uses a borrowed VIP observer with different broad text/scoring settings. This is a performance reference, not an equal-information operator ablation or published VIP benchmark.

| Dataset/protocol | Historical Anchored VIP | Native validated | Delta |
|---|---:|---:|---:|
| vdd/vdd | 57.0012 | 36.4547 | -20.5464 |
| potsdam/potsdam | 41.7806 | 29.0498 | -12.7308 |
| udd5/udd5 | 49.2117 | 42.8674 | -6.3443 |
| oem/oem | 31.9533 | 39.1952 | +7.2419 |
| loveda/P | 62.4863 | 54.5885 | -7.8979 |
| loveda/D | 36.5600 | 35.5080 | -1.0520 |
| vaihingen/vaihingen | 51.4491 | 44.4237 | -7.0254 |
| landcoverai/landcoverai | 66.9060 | 58.8643 | -8.0418 |
| flair1/flair1 | 33.6084 | 33.8774 | +0.2690 |
| Eight-domain mean, LoveDA D once | 46.058780 | 40.030056 | -6.028725 |

## Per-class outcomes

### vdd/vdd

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| other | 25.3949 | 31.7710 | 31.7702 | 6.3753 | 38.928/42.213 | 47.332/49.144 | 27.946 |
| wall | 11.8982 | 14.7889 | 14.7868 | 2.8886 | 11.984/94.332 | 14.888/95.606 | 12.518 |
| road | 24.1970 | 28.9951 | 29.1789 | 4.9819 | 24.741/91.668 | 31.800/77.976 | 3.382 |
| vegetation | 62.0257 | 41.4070 | 41.5301 | -20.4956 | 84.684/69.863 | 95.788/42.303 | 9.539 |
| vehicle | 5.6965 | 6.3815 | 6.3797 | 0.6832 | 5.697/99.956 | 6.380/99.926 | 3.700 |
| roof | 60.7577 | 72.4164 | 72.4183 | 11.6606 | 88.015/66.238 | 81.974/86.135 | 29.237 |
| water | 33.6534 | 59.1175 | 59.1190 | 25.4656 | 93.059/34.520 | 91.739/62.443 | 13.679 |

Diagnostics:
```json
{
  "tiles": 704,
  "validation_coefficient_view1": 0.9994853657077659,
  "validation_coefficient_view2": 0.9981848928569392,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 0.9893465909090909,
  "validation_error_before": 2.1946565671899045,
  "validation_error_after": 0.8279985619510193,
  "mean_absolute_ungated_innovation": 0.0044510533109794114,
  "mean_absolute_validated_innovation": 0.004446768614963565,
  "solver_relative_residual": 8.732394789893889e-07,
  "wide_crops_view1_per_tile": 0.022727272727272728,
  "wide_crops_view2_per_tile": 0.06818181818181818
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 21767614 | 17728831 |
| MeanLogit_Native | 14830326 | 7638237 |
| Anchored_Native | 14175766 | 7579247 |
| Validated_Native | 14172784 | 7552335 |

### potsdam/potsdam

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| impervious surface | 51.6875 | 41.8482 | 41.8484 | -9.8391 | 75.391/62.178 | 66.687/52.909 | 28.729 |
| building | 78.2026 | 74.3445 | 74.3445 | -3.8581 | 80.677/96.226 | 76.789/95.894 | 15.239 |
| low vegetation | 33.4221 | 14.8804 | 14.9086 | -18.5135 | 72.470/38.283 | 78.279/15.552 | 3.811 |
| tree | 62.5813 | 31.5906 | 31.5987 | -30.9826 | 90.654/66.897 | 95.791/32.044 | 8.011 |
| car | 11.6582 | 8.4297 | 8.4297 | -3.2285 | 11.672/99.026 | 8.430/99.958 | 27.778 |
| clutter | 4.5652 | 3.1676 | 3.1688 | -1.3964 | 7.745/10.007 | 4.214/11.328 | 16.432 |

Diagnostics:
```json
{
  "tiles": 72,
  "validation_coefficient_view1": 0.9996967919998698,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 0.9930555555555556,
  "validation_error_before": 1.3207608880423718,
  "validation_error_after": 0.47351012010544236,
  "mean_absolute_ungated_innovation": 0.0038058725347380256,
  "mean_absolute_validated_innovation": 0.0038049586841629613,
  "solver_relative_residual": 7.513170968328117e-07,
  "wide_crops_view1_per_tile": 0.4444444444444444,
  "wide_crops_view2_per_tile": 1.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 78094 | 2747461 |
| MeanLogit_Native | 55140 | 1345124 |
| Anchored_Native | 40244 | 1321024 |
| Validated_Native | 40246 | 1320401 |

### udd5/udd5

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| vegetation | 82.4937 | 63.9641 | 63.9786 | -18.5151 | 96.486/85.049 | 97.871/64.882 | 19.635 |
| building | 83.9035 | 81.1705 | 81.1687 | -2.7348 | 88.360/94.329 | 82.278/98.366 | 46.935 |
| road | 45.7856 | 33.1386 | 32.9998 | -12.7858 | 70.678/56.522 | 69.240/38.669 | 7.489 |
| vehicle | 10.0092 | 7.4576 | 7.4591 | -2.5501 | 10.032/97.748 | 7.466/98.823 | 10.596 |
| other | 30.5846 | 28.7187 | 28.7307 | -1.8539 | 52.854/42.059 | 46.917/42.568 | 15.344 |

Diagnostics:
```json
{
  "tiles": 3232,
  "validation_coefficient_view1": 0.9942644954091953,
  "validation_coefficient_view2": 0.9872265548242422,
  "coefficient_zero_fraction": 0.001547029702970297,
  "coefficient_one_fraction": 0.9730816831683168,
  "validation_error_before": 1.196901015035567,
  "validation_error_after": 0.4511215124707285,
  "mean_absolute_ungated_innovation": 0.0038000779490950194,
  "mean_absolute_validated_innovation": 0.0037800431735935365,
  "solver_relative_residual": 7.87386190914674e-07,
  "wide_crops_view1_per_tile": 0.024752475247524754,
  "wide_crops_view2_per_tile": 0.06311881188118812
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 17644685 | 114800238 |
| MeanLogit_Native | 14296773 | 41988631 |
| Anchored_Native | 13112285 | 42486343 |
| Validated_Native | 12996498 | 42419936 |

### oem/oem

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 9.655 |
| rangeland | 47.7780 | 46.9954 | 46.9954 | -0.7826 | 65.294/64.041 | 66.795/61.322 | 17.577 |
| developed space | 28.5294 | 32.1904 | 32.1904 | 3.6610 | 61.562/34.713 | 53.729/44.537 | 18.567 |
| road | 40.8291 | 38.5077 | 38.5077 | -2.3214 | 45.753/79.140 | 43.818/76.061 | 7.920 |
| tree | 56.0657 | 48.8384 | 48.8384 | -7.2273 | 87.448/60.973 | 89.831/51.697 | 14.635 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.046 |
| agriculture land | 79.1353 | 83.7843 | 83.7843 | 4.6490 | 90.641/86.177 | 91.452/90.904 | 16.659 |
| building | 62.4971 | 63.2451 | 63.2451 | 0.7480 | 65.584/92.995 | 68.998/88.353 | 14.940 |

Diagnostics:
```json
{
  "tiles": 67,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 1.0,
  "validation_error_before": 1.6001158710317587,
  "validation_error_after": 0.5533909534609094,
  "mean_absolute_ungated_innovation": 0.003577133345264775,
  "mean_absolute_validated_innovation": 0.003577133369590364,
  "solver_relative_residual": 8.416918594538225e-07,
  "wide_crops_view1_per_tile": 0.47761194029850745,
  "wide_crops_view2_per_tile": 1.0746268656716418
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 675679 | 1281954 |
| MeanLogit_Native | 332387 | 395484 |
| Anchored_Native | 274334 | 318020 |
| Validated_Native | 274334 | 318020 |

### loveda/P

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| building | 64.4987 | 60.8401 | 60.8401 | -3.6586 | 65.367/97.982 | 61.738/97.664 | 1.034 |
| road | 66.9348 | 42.5837 | 42.5837 | -24.3511 | 67.652/98.440 | 42.771/98.984 | 16.479 |
| water | 81.1851 | 77.1695 | 77.1695 | -4.0156 | 86.764/92.661 | 82.203/92.649 | 23.548 |
| barren | 24.3403 | 15.6525 | 15.6525 | -8.6878 | 63.873/28.226 | 77.590/16.394 | 1.544 |
| tree | 53.7307 | 41.4541 | 41.4541 | -12.2766 | 64.518/76.267 | 84.320/44.917 | 5.896 |
| farm | 86.2626 | 89.8308 | 89.8308 | 3.5682 | 95.330/90.069 | 95.982/93.341 | 51.499 |

Diagnostics:
```json
{
  "tiles": 72,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 1.0,
  "validation_error_before": 1.3702942517471175,
  "validation_error_after": 0.43530760597954377,
  "mean_absolute_ungated_innovation": 0.003956279924346341,
  "mean_absolute_validated_innovation": 0.00395627986613868,
  "solver_relative_residual": 9.097012107556818e-07,
  "wide_crops_view1_per_tile": 0.4444444444444444,
  "wide_crops_view2_per_tile": 1.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 154848 | 841827 |
| MeanLogit_Native | 163352 | 275164 |
| Anchored_Native | 143317 | 260860 |
| Validated_Native | 143317 | 260860 |

### loveda/D

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| background | 34.1983 | 40.3515 | 40.3515 | 6.1532 | 65.750/41.611 | 67.249/50.221 | 33.334 |
| building | 31.1862 | 31.3604 | 31.3604 | 0.1742 | 31.858/93.665 | 32.077/93.348 | 1.053 |
| road | 45.4818 | 33.5943 | 33.5943 | -11.8875 | 45.835/98.335 | 33.722/98.884 | 11.560 |
| water | 61.9193 | 58.3089 | 58.3089 | -3.6104 | 65.955/91.007 | 61.305/92.266 | 17.409 |
| barren | 14.5296 | 7.9802 | 7.9802 | -6.5494 | 59.063/16.157 | 63.593/8.362 | 0.532 |
| tree | 26.8904 | 26.1396 | 26.1396 | -0.7508 | 30.053/71.873 | 45.024/38.394 | 5.225 |
| farm | 54.9851 | 50.8212 | 50.8212 | -4.1639 | 69.572/72.395 | 65.681/69.195 | 30.887 |

Diagnostics:
```json
{
  "tiles": 72,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 1.0,
  "validation_error_before": 1.3881317211918438,
  "validation_error_after": 0.4442738623335156,
  "mean_absolute_ungated_innovation": 0.003535493839687357,
  "mean_absolute_validated_innovation": 0.0035354938687911877,
  "solver_relative_residual": 9.199716252005096e-07,
  "wide_crops_view1_per_tile": 0.4444444444444444,
  "wide_crops_view2_per_tile": 1.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 651952 | 1627028 |
| MeanLogit_Native | 686596 | 549215 |
| Anchored_Native | 551228 | 492978 |
| Validated_Native | 551228 | 492978 |

### vaihingen/vaihingen

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| impervious surface | 49.2002 | 41.2948 | 41.2948 | -7.9054 | 82.397/54.979 | 68.116/51.190 | 23.008 |
| building | 74.6610 | 73.0427 | 73.0427 | -1.6183 | 75.500/98.533 | 73.970/98.313 | 28.646 |
| low vegetation | 46.7115 | 29.7972 | 29.7972 | -16.9143 | 92.073/48.669 | 92.002/30.590 | 8.462 |
| tree | 71.8329 | 71.2116 | 71.2116 | -0.6213 | 82.551/84.692 | 88.490/78.481 | 18.534 |
| car | 8.7566 | 6.7722 | 6.7722 | -1.9844 | 8.773/97.975 | 6.783/97.722 | 21.349 |

Diagnostics:
```json
{
  "tiles": 72,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 1.0,
  "validation_error_before": 1.1557639611705743,
  "validation_error_after": 0.39891917484231354,
  "mean_absolute_ungated_innovation": 0.003950837831426825,
  "mean_absolute_validated_innovation": 0.003950837792621719,
  "solver_relative_residual": 5.060940869346572e-07,
  "wide_crops_view1_per_tile": 0.4444444444444444,
  "wide_crops_view2_per_tile": 1.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 276849 | 2092180 |
| MeanLogit_Native | 139941 | 679433 |
| Anchored_Native | 83672 | 648779 |
| Validated_Native | 83672 | 648779 |

### landcoverai/landcoverai

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| background | 82.6090 | 82.4999 | 82.4957 | -0.1133 | 96.651/85.044 | 96.524/85.022 | 59.686 |
| building | 34.9562 | 34.4245 | 34.4245 | -0.5317 | 35.044/99.292 | 34.493/99.424 | 4.367 |
| woodland | 78.3638 | 76.5684 | 76.5692 | -1.7946 | 86.499/89.284 | 92.282/81.808 | 18.977 |
| water | 93.6635 | 90.8264 | 90.8264 | -2.8371 | 93.664/100.000 | 90.826/100.000 | 9.107 |
| road | 14.9318 | 10.0065 | 10.0055 | -4.9263 | 15.629/77.002 | 10.306/77.416 | 7.863 |

Diagnostics:
```json
{
  "tiles": 8,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 0.9660378992557526,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 0.9375,
  "validation_error_before": 0.6264321074144663,
  "validation_error_after": 0.268593794655138,
  "mean_absolute_ungated_innovation": 0.00274225024622865,
  "mean_absolute_validated_innovation": 0.0027083958848379552,
  "solver_relative_residual": 9.241106511126418e-07,
  "wide_crops_view1_per_tile": 4.0,
  "wide_crops_view2_per_tile": 9.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 67411 | 315504 |
| MeanLogit_Native | 47645 | 63197 |
| Anchored_Native | 33863 | 67557 |
| Validated_Native | 33652 | 67392 |

### flair1/flair1

| Class | Geometry IoU | Ungated IoU | Validated IoU | Delta | Geometry P/R | Validated P/R | Validated area % |
|---|---:|---:|---:|---:|---|---|---:|
| building | 49.6979 | 43.0046 | 43.0046 | -6.6933 | 50.726/96.082 | 44.227/93.962 | 15.253 |
| pervious surface | 57.0621 | 66.1813 | 66.1813 | 9.1192 | 92.130/59.986 | 89.195/71.950 | 13.842 |
| impervious surface | 52.6509 | 42.8140 | 42.8140 | -9.8369 | 62.624/76.777 | 49.155/76.847 | 25.612 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 1.364 |
| water | 73.2328 | 57.1302 | 57.1302 | -16.1026 | 77.143/93.526 | 58.749/95.400 | 7.203 |
| coniferous | 43.0233 | 40.2438 | 40.2438 | -2.7795 | 61.711/58.690 | 85.690/43.143 | 0.283 |
| deciduous | 54.0229 | 40.6994 | 40.6994 | -13.3235 | 78.972/63.099 | 82.544/44.532 | 9.183 |
| brushwood | 18.6136 | 8.7775 | 8.7775 | -9.8361 | 23.421/47.556 | 12.976/21.340 | 8.089 |
| vineyard | -- | -- | -- | -- | 0.000/0.000 | 0.000/0.000 | 0.000 |
| herbaceous vegetation | 60.2329 | 52.4425 | 52.4425 | -7.7904 | 94.317/62.501 | 94.415/54.121 | 18.363 |
| agricultural land | 18.7339 | 21.3584 | 21.3584 | 2.6245 | 21.647/58.198 | 25.229/58.198 | 0.703 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | 0.000/0.000 | 0.104 |

Diagnostics:
```json
{
  "tiles": 8,
  "validation_coefficient_view1": 1.0,
  "validation_coefficient_view2": 1.0,
  "coefficient_zero_fraction": 0.0,
  "coefficient_one_fraction": 1.0,
  "validation_error_before": 2.5304257857230548,
  "validation_error_after": 0.8453065072245712,
  "mean_absolute_ungated_innovation": 0.0035999745887238532,
  "mean_absolute_validated_innovation": 0.003599974646931514,
  "solver_relative_residual": 8.906813135922675e-07,
  "wide_crops_view1_per_tile": 4.0,
  "wide_crops_view2_per_tile": 9.0
}
```

| Method | Corrected pixels | Corrupted pixels |
|---|---:|---:|
| NativeWide | 91475 | 469260 |
| MeanLogit_Native | 73579 | 179774 |
| Anchored_Native | 58463 | 168026 |
| Validated_Native | 58463 | 168026 |

## Runtime

Suite wall time: 526.292s. Worker times include all five arms and metric accumulation.

Mask-free standalone primary smoke on aachen/images/aachen_11.tif ([3, 1000, 1000]): median 0.749859s; resident peak 3859.72MiB. This is one-image cost, not full-domain throughput. Native feature replay max error 0.0; three timed predictions identical.

| Dataset | All-arm wall seconds | Peak MiB |
|---|---:|---:|
| vdd | 124.833 | 3827.97 |
| potsdam | 8.982 | 3826.19 |
| udd5 | 476.093 | 3823.76 |
| oem | 10.873 | 3830.55 |
| loveda | 14.490 | 3838.59 |
| vaihingen | 9.097 | 3824.10 |
| landcoverai | 3.525 | 3724.53 |
| flair1 | 4.420 | 3735.52 |

Standalone same-image timings, five synchronized repetitions after warm-up; both complete-image predictions exactly replay evaluator outputs:

| Method | Median seconds | Resident peak MiB |
|---|---:|---:|
| Geometry | 0.394476 | 3819.02 |
| Validated_Native | 0.794037 | 3831.51 |
Candidate/Geometry time ratio: 2.0129. No masks loaded; image aachen/images/aachen_11.tif, shape [3, 1000, 1000].

## Interpretation

A lower cross-view discrepancy is guaranteed by the scalar minimization only for the validating score target; it does not establish correct classes. Views share model/image and can share errors. The scalar cannot reject a bad alias or distinguish a useful correction from a harmful one within the same tile. This screen changes both the wide source and the writer relative to historical VIP coupling; within-screen ungated and mean controls isolate the proposed validation. Original best models are retained.

When both centered observations coincide, d=H(W-G), with H eigenvalues h in[0,1). The unconstrained validation coefficient is sum(s^2*h*t^2)/sum(s^2*h^2*t^2)>=1, so the bounded coefficient is1 wherever a nonzero update exists. Agreement therefore cannot reject a shared erroneous source; an already-shrunk anchored update often saturates the gate. The observed coefficient-one fraction and same-source coupling deltas test this limitation.

Initial smoke failure was a validation-reference mistake: native fused-head output was compared against TCPR's manual FP32-attention path (difference0.0037421109). The smoke was corrected to compare the official encode_image API (actual error0), with no change to candidate inference. The preserved initial smoke preceded labels and all evaluations.
