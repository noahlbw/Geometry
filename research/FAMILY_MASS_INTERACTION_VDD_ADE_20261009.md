# Full class-mass and alias redundancy interaction

Wide only, unchanged inherited20 salience/e: SUM=LSE(tau*e-log(m_family))/tau. MEAN=SUM-log(G/20)/tau; UniformMass=Base+log(G/20)/tau. Original local/Geometry/H/g/templates/scales unchanged.

Calibration interaction with the previously tested relative alias signal, not new semantic reliability. G is declared family count, not validated independent concept count. No historical K, fitted bias coefficient, new views/heads or class-square alias tensor.

Previously label-developed words/profiles; source/model decision informed by prior experiments. Exploratory, not untouched independent validation.

| Dataset | Current Base | Current SUM | Complete Base | MEAN | UniformMass | SUM | SUM Shuffle | VIP20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 55.0754 | 56.6678 | 55.0754 | 55.7070 | 56.0450 | 56.6678 | 55.7663 | 51.4827 |
| ade150 | 27.9111 | 29.5058 | 28.4271 | 28.4098 | 30.4118 | 30.4076 | 30.4116 | 25.4843 |

The joint candidate is rejected. Both saved paper numeric targets and both timing
gates pass, but only VDD establishes an independent alias contribution. VDD
SUM-UniformMass is +0.6228pp (95% paired image interval [0.4817,0.7714]);
SUM-Shuffle is +0.9015pp ([0.6830,1.1231]). Its calibration interaction with the
previous MEAN effect is -0.0088pp ([-0.0598,0.0408]), so the new class mass is
not shown to strengthen that alias signal.

ADE SUM-UniformMass is -0.0042pp ([-0.0202,0.0092]) and SUM-Shuffle is -0.0040pp
([-0.0239,0.0125]). Thus almost all its +1.9805pp over same-input Base is explained
by the disclosed class-mass calibration, not choosing useful aliases. It remains
0.7786pp below the retained 433-query 31.1862 reference. Current20 SUM 29.5058
has no corresponding Current20 uniform-mass control and cannot supply the missing
alias attribution. Neither that control nor UniformMass is promoted.

VDD road IoU increases 41.7969 -> 44.1870 under calibration alone, then to
45.2370 with the family weights. Vehicle changes 28.7693 -> 30.4368 -> 31.4747.
Water falls 83.0581 -> 82.9757 -> 82.9489; improvement is not universal per class.
ADE building/sky/grass/person recover strongly under UniformMass itself; their
large gains must not be presented as word-specific selection success.

| Dataset | Base ms | SUM ms | Matched20 VIP ms | Official VIP ms | SUM/official |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 455.89 | 454.98 | 177.83 | 177.42 | 2.564x |
| ade150 | 353.85 | 337.08 | 112.31 | 111.56 | 3.021x |

Three first/middle/last complete images, five rotated warmed synchronized singleton repetitions. Every image passes both6x gates; not full-set average timing. Includes encoders/alias/writer/stitch/restoration/argmax, excludes decode/setup/text. Memory includes both resident models/banks.

SUM uses cached integer member groups while Base retains the inherited Boolean
indexing path. Its slightly lower measured latency is not evidence that the
semantic rule itself speeds inference. Shared resident peaks are 6224.65MiB VDD
and 7124.86MiB ADE, not isolated deployment peaks. All five workers and the
controller are terminal; a final read-only check found all eight GPUs idle.

54.3/29.1 are saved paper numeric targets, not certified paper-protocol reproduction. Explicit empty-row VIP numerical repair retained. Sources and profiles were previously developed using labels. G counts declared expression families, not demonstrated distinct semantic concepts. SUM and MEAN share the same alias relative weights; all new effects arise through the disclosed class-mass interaction.

vdd paired image-bootstrap effects:
```json
{
  "miou": {
    "Current20_Base": 55.0754,
    "Current20_FamilySum": 56.6678,
    "Complete20_Base": 55.0754,
    "Complete20_FamilyLME": 55.707,
    "Complete20_UniformMass": 56.045,
    "Complete20_FamilySum": 56.6678,
    "Complete20_FamilySumShuffle": 55.7663,
    "VIP_Complete20": 51.4827
  },
  "paired": {
    "Complete20_Base": {
      "delta_pp": 1.5923999999999978,
      "ci95_pp": [
        1.2921902342784009,
        1.9704808962312328
      ]
    },
    "Complete20_UniformMass": {
      "delta_pp": 0.622799999999998,
      "ci95_pp": [
        0.4816972557311727,
        0.77137943446661
      ]
    },
    "Complete20_FamilySumShuffle": {
      "delta_pp": 0.9014999999999986,
      "ci95_pp": [
        0.6830490192031808,
        1.1230715760442103
      ]
    },
    "Complete20_FamilyLME": {
      "delta_pp": 0.960799999999999,
      "ci95_pp": [
        0.7727004507232609,
        1.1958014923663636
      ]
    },
    "Current20_Base": {
      "delta_pp": 1.5923999999999978,
      "ci95_pp": [
        1.2921902342784009,
        1.9704808962312328
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 5.1850999999999985,
      "ci95_pp": [
        3.8992932771136095,
        6.612545211392005
      ]
    }
  },
  "interaction_pp": -0.008800000000000807,
  "interaction_ci95_pp": [
    -0.0597923924203041,
    0.040814239785804826
  ],
  "alias_effect_supported": true,
  "identity_effect_supported": true,
  "total_gain_supported": true,
  "above_numeric_target": true,
  "historical433_delta_pp": null,
  "predecessor_endpoint_replay_verified": true,
  "downloaded_unique_coverage_verified": true
}
```

| Class | Base IoU | UniformMass IoU | SUM IoU | SUM-Uniform pp | SUM-Base pp | TP change vs Uniform | FP change vs Uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 35.2114 | 37.3076 | 38.1923 | 0.8847 | 2.9809 | +2786689 | +1053106 |
| wall | 47.0676 | 47.0781 | 48.0704 | 0.9923 | 1.0028 | +412052 | +137302 |
| road | 41.7969 | 44.1870 | 45.2370 | 1.0500 | 3.4401 | -169874 | -3026162 |
| vegetation | 66.4209 | 66.7818 | 66.9225 | 0.1407 | 0.5016 | +601441 | +159919 |
| vehicle | 28.7693 | 30.4368 | 31.4747 | 1.0379 | 2.7054 | -16612 | -450847 |
| roof | 83.2036 | 83.5479 | 83.8290 | 0.2811 | 0.6254 | -342462 | -1204035 |
| water | 83.0581 | 82.9757 | 82.9489 | -0.0268 | -0.1092 | +3217 | +56266 |

ade150 paired image-bootstrap effects:
```json
{
  "miou": {
    "Current20_Base": 27.9111,
    "Current20_FamilySum": 29.5058,
    "Complete20_Base": 28.4271,
    "Complete20_FamilyLME": 28.4098,
    "Complete20_UniformMass": 30.4118,
    "Complete20_FamilySum": 30.4076,
    "Complete20_FamilySumShuffle": 30.4116,
    "VIP_Complete20": 25.4843
  },
  "paired": {
    "Complete20_Base": {
      "delta_pp": 1.9804999999999993,
      "ci95_pp": [
        1.5873221669353517,
        2.4858639392627095
      ]
    },
    "Complete20_UniformMass": {
      "delta_pp": -0.00420000000000087,
      "ci95_pp": [
        -0.02018107042282411,
        0.00917417419102575
      ]
    },
    "Complete20_FamilySumShuffle": {
      "delta_pp": -0.004000000000001336,
      "ci95_pp": [
        -0.02389884689502697,
        0.01246061943264127
      ]
    },
    "Complete20_FamilyLME": {
      "delta_pp": 1.997799999999998,
      "ci95_pp": [
        1.6077144062360413,
        2.504700035669491
      ]
    },
    "Current20_Base": {
      "delta_pp": 2.4964999999999975,
      "ci95_pp": [
        1.969574122225072,
        3.040080872383311
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 4.923299999999998,
      "ci95_pp": [
        4.428766917972511,
        5.431945605832129
      ]
    }
  },
  "interaction_pp": 0.013099999999997891,
  "interaction_ci95_pp": [
    -0.002971066124514454,
    0.026780444573298996
  ],
  "alias_effect_supported": false,
  "identity_effect_supported": false,
  "total_gain_supported": true,
  "above_numeric_target": true,
  "historical433_delta_pp": -0.7786000000000008,
  "predecessor_endpoint_replay_verified": true,
  "downloaded_unique_coverage_verified": true
}
```

| Class | Base IoU | UniformMass IoU | SUM IoU | SUM-Uniform pp | SUM-Base pp | TP change vs Uniform | FP change vs Uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wall | 29.4245 | 33.4897 | 33.5873 | 0.0976 | 4.1628 | +94078 | +42941 |
| building | 56.7754 | 64.4927 | 64.6900 | 0.1973 | 7.9146 | -79064 | -316282 |
| sky | 72.1083 | 81.4131 | 81.4481 | 0.0350 | 9.3398 | +8022 | -7595 |
| floor | 57.4646 | 58.7287 | 58.7090 | -0.0197 | 1.2444 | +3351 | +17993 |
| tree | 52.9124 | 61.0809 | 61.2256 | 0.1447 | 8.3132 | +40471 | +8749 |
| ceiling | 56.9034 | 53.7564 | 53.8348 | 0.0784 | -3.0686 | +19493 | -17718 |
| road | 70.4468 | 69.4036 | 69.3973 | -0.0063 | -1.0495 | +1110 | +3599 |
| bed | 53.2757 | 60.2901 | 60.2538 | -0.0363 | 6.9781 | -1875 | +7765 |
| windowpane | 39.1184 | 42.2141 | 42.2342 | 0.0201 | 3.1158 | +7290 | +10831 |
| grass | 28.7547 | 44.0294 | 44.0363 | 0.0069 | 15.2816 | +709 | -128 |
| cabinet | 40.8962 | 49.2786 | 49.2759 | -0.0027 | 8.3797 | +1354 | +3329 |
| sidewalk | 50.0388 | 47.7511 | 47.7255 | -0.0256 | -2.3133 | +421 | +8205 |
| person | 32.3176 | 62.1024 | 62.2191 | 0.1167 | 29.9015 | -19202 | -49518 |
| earth | 6.1588 | 10.9932 | 10.9980 | 0.0048 | 4.8392 | +433 | +381 |
| door | 35.2983 | 36.0792 | 36.0846 | 0.0054 | 0.7863 | +3275 | +7549 |
| table | 37.7151 | 40.5313 | 40.5674 | 0.0361 | 2.8523 | -7923 | -25915 |
| mountain | 31.4960 | 36.8118 | 36.8514 | 0.0396 | 5.3554 | +1777 | -1861 |
| plant | 28.8954 | 37.6491 | 37.6323 | -0.0168 | 8.7369 | -395 | +2224 |
| curtain | 66.1882 | 64.7797 | 64.7755 | -0.0042 | -1.4127 | +169 | +685 |
| chair | 35.6584 | 42.8095 | 42.9116 | 0.1021 | 7.2532 | +12599 | +12378 |
| car | 63.4679 | 65.1720 | 65.1009 | -0.0711 | 1.6330 | +100 | +6567 |
| water | 37.9303 | 49.6956 | 49.6960 | 0.0004 | 11.7657 | -36 | -122 |
| painting | 25.3364 | 29.7223 | 29.7733 | 0.0510 | 4.4369 | +14604 | +41913 |
| sofa | 54.9155 | 50.0737 | 49.9595 | -0.1142 | -4.9560 | +906 | +14684 |
| shelf | 29.8764 | 32.5437 | 32.4887 | -0.0550 | 2.6123 | -3864 | -4366 |
| house | 18.8871 | 14.0707 | 14.4394 | 0.3687 | -4.4477 | +17451 | +27074 |
| sea | 27.6019 | 27.8751 | 27.8797 | 0.0046 | 0.2778 | +175 | +266 |
| mirror | 32.6086 | 32.4095 | 32.4291 | 0.0196 | -0.1795 | +1057 | +1560 |
| rug | 29.4893 | 31.1809 | 31.1808 | -0.0001 | 1.6915 | +255 | +853 |
| field | 12.5910 | 16.7704 | 16.7569 | -0.0135 | 4.1659 | -1887 | -5242 |
| armchair | 21.9974 | 12.8923 | 12.2071 | -0.6852 | -9.7903 | -14996 | -12036 |
| seat | 26.4386 | 28.5496 | 28.4973 | -0.0523 | 2.0587 | -34 | +4044 |
| fence | 27.8704 | 26.0452 | 25.9436 | -0.1016 | -1.9268 | -1031 | +10708 |
| desk | 34.7370 | 33.9102 | 33.8528 | -0.0574 | -0.8842 | +638 | +5920 |
| rock | 33.1511 | 34.9354 | 34.8666 | -0.0688 | 1.7155 | -1713 | +530 |
| wardrobe | 42.0983 | 44.2737 | 44.2577 | -0.0160 | 2.1594 | -151 | +524 |
| lamp | 32.5292 | 30.1314 | 30.1097 | -0.0217 | -2.4195 | -89 | +1875 |
| bathtub | 51.9402 | 49.3756 | 49.3237 | -0.0519 | -2.6165 | +141 | +2172 |
| railing | 19.5671 | 19.3173 | 19.1828 | -0.1345 | -0.3843 | -3073 | +12075 |
| cushion | 47.5770 | 29.8800 | 29.8716 | -0.0084 | -17.7054 | -95 | +17 |
| base | 3.7919 | 3.7937 | 3.8210 | 0.0273 | 0.0291 | +422 | +1997 |
| box | 28.5447 | 23.6815 | 23.5843 | -0.0972 | -4.9604 | -1184 | -149 |
| column | 30.8630 | 35.8253 | 35.8987 | 0.0734 | 5.0357 | -604 | -6315 |
| signboard | 26.0044 | 27.5735 | 27.4642 | -0.1093 | 1.4598 | +154 | +8726 |
| chest of drawers | 29.7410 | 31.0237 | 31.0224 | -0.0013 | 1.2814 | +1402 | +4587 |
| counter | 37.5350 | 43.3892 | 43.4392 | 0.0500 | 5.9042 | +1016 | +1116 |
| sand | 44.3926 | 39.7822 | 39.7309 | -0.0513 | -4.6617 | -5 | +2235 |
| sink | 39.3191 | 36.7065 | 36.8135 | 0.1070 | -2.5056 | -779 | -7418 |
| skyscraper | 26.4469 | 15.5600 | 15.9632 | 0.4032 | -10.4837 | +7613 | +15170 |
| fireplace | 48.4232 | 51.5602 | 51.5646 | 0.0044 | 3.1414 | -216 | -555 |
| refrigerator | 67.0728 | 67.3956 | 67.1514 | -0.2442 | 0.0786 | +410 | +5151 |
| grandstand | 23.1068 | 28.1722 | 28.0416 | -0.1306 | 4.9348 | +540 | +6364 |
| path | 4.1205 | 3.8993 | 3.8975 | -0.0018 | -0.2230 | +33 | +1772 |
| stairs | 36.4871 | 38.5237 | 38.5728 | 0.0491 | 2.0857 | +926 | +425 |
| runway | 40.3009 | 34.0403 | 34.0358 | -0.0045 | -6.2651 | +37 | +318 |
| case | 22.6269 | 23.0048 | 22.8950 | -0.1098 | 0.2681 | -59 | +10928 |
| pool table | 79.2281 | 80.9349 | 80.9590 | 0.0241 | 1.7309 | -169 | -442 |
| pillow | 37.0292 | 21.1471 | 21.1468 | -0.0003 | -15.8824 | +0 | +17 |
| screen door | 0.0885 | 0.0460 | 0.0460 | 0.0000 | -0.0425 | +0 | +1164 |
| stairway | 13.7109 | 16.0215 | 16.0217 | 0.0002 | 2.3108 | +683 | +4253 |
| river | 19.5303 | 16.7127 | 16.7399 | 0.0272 | -2.7904 | +14 | -1748 |
| bridge | 16.9548 | 27.7675 | 27.7180 | -0.0495 | 10.7632 | +200 | +2610 |
| bookcase | 16.9215 | 17.2595 | 17.3524 | 0.0929 | 0.4309 | +1806 | +4135 |
| blind | 31.2292 | 32.1748 | 32.1813 | 0.0065 | 0.9521 | +2 | -475 |
| coffee table | 49.4171 | 49.4101 | 49.5709 | 0.1608 | 0.1538 | +3487 | +3524 |
| toilet | 45.8187 | 47.0663 | 47.2160 | 0.1497 | 1.3973 | -149 | -5427 |
| flower | 16.8688 | 9.2629 | 9.3943 | 0.1314 | -7.4745 | +881 | +356 |
| book | 8.2284 | 11.9314 | 11.8658 | -0.0656 | 3.6374 | -430 | -361 |
| hill | 6.5991 | 5.9654 | 5.8770 | -0.0884 | -0.7221 | -2569 | -11504 |
| bench | 31.7802 | 33.7796 | 33.7152 | -0.0644 | 1.9350 | +1072 | +6501 |
| countertop | 11.3911 | 16.6114 | 16.6343 | 0.0229 | 5.2432 | +537 | +1226 |
| stove | 35.1250 | 27.2118 | 27.2095 | -0.0023 | -7.9155 | -1027 | -3601 |
| palm | 24.2952 | 23.2396 | 23.2346 | -0.0050 | -1.0606 | -32 | +189 |
| kitchen island | 10.4141 | 22.1091 | 22.0987 | -0.0104 | 11.6846 | -245 | -727 |
| computer | 42.1122 | 36.9030 | 36.8698 | -0.0332 | -5.2424 | +120 | +1280 |
| swivel chair | 23.9621 | 6.4748 | 6.4786 | 0.0038 | -17.4835 | +17 | +25 |
| boat | 34.5926 | 33.2521 | 33.2227 | -0.0294 | -1.3699 | +152 | +1293 |
| bar | 21.6512 | 24.0548 | 23.7993 | -0.2555 | 2.1481 | +903 | +18675 |
| arcade machine | 20.8112 | 7.9296 | 8.0554 | 0.1258 | -12.7558 | +328 | +39 |
| hovel | 14.4525 | 12.4241 | 12.4020 | -0.0221 | -2.0505 | +4372 | +38590 |
| bus | 67.6411 | 64.7935 | 64.7762 | -0.0173 | -2.8649 | +10 | +178 |
| towel | 39.8170 | 52.2599 | 52.2165 | -0.0434 | 12.3995 | +21 | +574 |
| light | 9.6913 | 5.1088 | 5.0603 | -0.0485 | -4.6310 | -271 | +851 |
| truck | 15.8812 | 17.0378 | 17.0011 | -0.0367 | 1.1199 | +4 | +1635 |
| tower | 4.5706 | 14.9081 | 14.8694 | -0.0387 | 10.2988 | +852 | +9132 |
| chandelier | 31.7228 | 40.5729 | 40.5149 | -0.0580 | 8.7921 | +127 | +1546 |
| awning | 18.2674 | 22.8935 | 22.9509 | 0.0574 | 4.6835 | -552 | -4060 |
| streetlight | 20.6064 | 21.6173 | 21.6030 | -0.0143 | 0.9966 | +151 | +1187 |
| booth | 10.8845 | 6.7110 | 7.0287 | 0.3177 | -3.8558 | +1553 | -128709 |
| television receiver | 31.8064 | 31.7180 | 31.7917 | 0.0737 | -0.0147 | +1260 | +1854 |
| airplane | 17.6805 | 17.7966 | 17.7677 | -0.0289 | 0.0872 | +1 | +2107 |
| dirt track | 1.0410 | 2.9185 | 2.9149 | -0.0036 | 1.8739 | +2 | +383 |
| apparel | 3.5870 | 6.5867 | 6.5233 | -0.0634 | 2.9363 | +77 | +12748 |
| pole | 4.0165 | 13.9269 | 13.8880 | -0.0389 | 9.8715 | +12 | +2801 |
| land | 5.6221 | 3.9174 | 3.8725 | -0.0449 | -1.7496 | +1246 | +48617 |
| bannister | 7.1113 | 9.2243 | 9.2883 | 0.0640 | 2.1770 | +2149 | +17350 |
| escalator | 33.1105 | 33.5887 | 33.6179 | 0.0292 | 0.5074 | +201 | +269 |
| ottoman | 49.1550 | 36.9889 | 36.6583 | -0.3306 | -12.4967 | -688 | +3026 |
| bottle | 28.1991 | 31.0666 | 30.9765 | -0.0901 | 2.7774 | -139 | +455 |
| buffet | 9.6494 | 6.8098 | 7.2297 | 0.4199 | -2.4197 | +2052 | -5345 |
| poster | 6.2571 | 7.1694 | 7.3146 | 0.1452 | 1.0575 | -1010 | -62690 |
| stage | 5.6007 | 13.5496 | 13.4993 | -0.0503 | 7.8986 | +88 | +3395 |
| van | 13.5528 | 35.0273 | 34.9791 | -0.0482 | 21.4263 | +14 | +522 |
| ship | 4.6749 | 6.5103 | 6.5122 | 0.0019 | 1.8373 | +1 | -89 |
| fountain | 28.8249 | 32.2235 | 32.2145 | -0.0090 | 3.3896 | +129 | +543 |
| conveyer belt | 46.1919 | 45.8706 | 46.1993 | 0.3287 | 0.0074 | -263 | -3021 |
| canopy | 10.7751 | 10.6688 | 11.0638 | 0.3950 | 0.2887 | +3508 | -10498 |
| washer | 74.8620 | 66.0090 | 65.9622 | -0.0468 | -8.8998 | -284 | -84 |
| plaything | 10.1451 | 10.2712 | 10.0598 | -0.2114 | -0.0853 | -285 | +10597 |
| swimming pool | 23.5363 | 35.1700 | 35.2137 | 0.0437 | 11.6774 | -8 | -728 |
| stool | 17.0401 | 24.0927 | 24.0614 | -0.0313 | 7.0213 | -177 | -309 |
| barrel | 9.4830 | 21.5823 | 21.0961 | -0.4862 | 11.6131 | +0 | +921 |
| basket | 24.5849 | 31.0965 | 31.0654 | -0.0311 | 6.4805 | +10 | +631 |
| waterfall | 19.3217 | 23.4174 | 23.4253 | 0.0079 | 4.1036 | +15 | -131 |
| tent | 41.3524 | 48.7930 | 48.7015 | -0.0915 | 7.3491 | +0 | +842 |
| bag | 23.3426 | 26.0639 | 26.0206 | -0.0433 | 2.6780 | -500 | -883 |
| minibike | 61.6533 | 62.1620 | 62.1785 | 0.0165 | 0.5252 | -26 | -114 |
| cradle | 48.1042 | 53.9668 | 53.6212 | -0.3456 | 5.5170 | -157 | +1931 |
| oven | 20.6047 | 31.3077 | 30.9644 | -0.3433 | 10.3597 | +67 | +5158 |
| ball | 9.8719 | 12.5853 | 12.4190 | -0.1663 | 2.5471 | -158 | +1344 |
| food | 46.7674 | 35.5385 | 35.5169 | -0.0216 | -11.2505 | -225 | -248 |
| step | 2.1542 | 2.3003 | 2.3008 | 0.0005 | 0.1466 | +0 | -71 |
| tank | 24.1075 | 23.9047 | 24.5886 | 0.6839 | 0.4811 | +4593 | +3412 |
| trade name | 3.7123 | 2.8812 | 2.8799 | -0.0013 | -0.8324 | -25 | -638 |
| microwave | 73.1425 | 74.7167 | 74.5463 | -0.1704 | 1.4038 | +1024 | +2735 |
| pot | 26.1630 | 31.2829 | 31.2521 | -0.0308 | 5.0891 | -121 | +313 |
| animal | 59.8187 | 58.2169 | 58.5560 | 0.3391 | -1.2627 | +2546 | +754 |
| bicycle | 45.5564 | 45.3142 | 45.3068 | -0.0074 | -0.2496 | +14 | +68 |
| lake | 7.2217 | 2.8413 | 2.8462 | 0.0049 | -4.3755 | +0 | -1379 |
| dishwasher | 40.0853 | 43.7138 | 42.9562 | -0.7576 | 2.8709 | -1286 | -79 |
| screen | 31.6663 | 38.6944 | 38.6598 | -0.0346 | 6.9935 | -9 | +548 |
| blanket | 8.7200 | 15.0715 | 15.0477 | -0.0238 | 6.3277 | -188 | +1048 |
| sculpture | 15.4491 | 42.1202 | 42.6191 | 0.4989 | 27.1700 | +1414 | +839 |
| hood | 27.0307 | 29.6956 | 29.7664 | 0.0708 | 2.7357 | +42 | -1117 |
| sconce | 8.2567 | 17.2899 | 17.2838 | -0.0061 | 9.0271 | +64 | +558 |
| vase | 17.5210 | 14.7275 | 14.7630 | 0.0355 | -2.7580 | +267 | +365 |
| traffic light | 15.2664 | 19.0132 | 19.0656 | 0.0524 | 3.7992 | -225 | -2514 |
| tray | 8.3296 | 13.2419 | 13.2288 | -0.0131 | 4.8992 | +14 | +442 |
| ashcan | 31.9794 | 11.6155 | 11.5533 | -0.0622 | -20.4261 | +85 | +7773 |
| fan | 28.3487 | 46.9541 | 47.0962 | 0.1421 | 18.7475 | +480 | +403 |
| pier | 4.6902 | 4.7382 | 4.6966 | -0.0416 | 0.0064 | +9 | +7440 |
| crt screen | 0.4653 | 0.3564 | 0.3453 | -0.0111 | -0.1200 | -25 | +567 |
| plate | 13.2446 | 24.1353 | 24.0193 | -0.1160 | 10.7747 | +56 | +3876 |
| monitor | 18.3620 | 19.0998 | 19.1183 | 0.0185 | 0.7563 | +620 | +2367 |
| bulletin board | 4.5751 | 5.4435 | 5.3965 | -0.0470 | 0.8214 | -56 | +2815 |
| shower | 0.5723 | 1.1400 | 1.1391 | -0.0009 | 0.5668 | +7 | +1812 |
| radiator | 34.4497 | 54.3377 | 54.3338 | -0.0039 | 19.8841 | +55 | +123 |
| glass | 10.2224 | 9.7322 | 9.7764 | 0.0442 | -0.4460 | +107 | +38 |
| clock | 43.3692 | 45.3298 | 45.3185 | -0.0113 | 1.9493 | -12 | +3 |
| flag | 29.5265 | 50.2063 | 50.0586 | -0.1477 | 20.5321 | +11 | +861 |

Joint acceptance gate: False.
If UniformMass explains target crossing without independent alias/identity advantage, the original alias goal remains incomplete. Do not promote controls or tune bias strength after results. No conditional reliability or CVPR novelty claim follows from this calibration interaction.
