# Fixed20 wide-only expression-family exponential means

Wide only: fixed inherited20 salience/profiled e; LSE(tau*e+log[20/(G*m_family)])/tau. Uniform families exact Base. Local remains original per-bank LME; original Geometry/H/g/templates/scales unchanged.

Static expression-mass correction, not image-conditioned semantic reliability or full end-to-end duplication invariance. No word deletion, extra views/heads, historical logK or logG class prior.

Previously label-developed words/profiles; source/model decision informed by prior experiments. Exploratory, not untouched independent validation.

ADE Complete20 preserves the entire language-only union of the actually executed historical433 bank and the existing301 bare roots:455 semantic strings, at most18 per class. Frozen neutral wrappers fill each class to exactly20 distinct strings and retain their generating-root identity. Current20 retains its original strings and generation metadata. VDD Current20 and Complete20 are identical original140-query banks; grouping uses the previously declared conservative lexical equivalences, not fabricated generation provenance. There is no new Qwen generation or score-dependent word pruning.

| Full dataset | Current Base | Current family | Complete Base | Complete family | Identity shuffle | VIP Complete20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 55.0754 | 55.7070 | 55.0754 | 55.7070 | 54.8167 | 51.4827 |
| ade150 | 27.9111 | 27.9007 | 28.4271 | 28.4098 | 28.4306 | 25.4843 |

The VDD treatment gain is +0.6316pp (paired95% CI[0.4894,0.7775]); its advantage over the identity null is +0.8903pp (CI[0.7058,1.1038]). No input change is involved. ADE input construction alone gains +0.5160pp; the actual treatment then loses0.0173pp (CI[-0.0296,-0.0055]) and loses0.0208pp to the identity null (CI[-0.0350,-0.0070]). The primary misses29.1 and remains2.7764pp below the retained433-query31.1862 model. Both the original and complete ADE20 inputs decline under this rule. Thus the joint candidate is rejected, even though execution cost passes.

The kernel depends only on each slot's declared family size. The identity null tests matching redundancy weights to specific alias identities; it does not validate all semantic family relations or image-dependent alias correctness. Higher response or a lexical equivalence remains insufficient evidence of semantic reliability.

| Dataset | Complete Base ms | Family ms | Same20 VIP ms | Official VIP ms | Family/official |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 466.77 | 465.34 | 181.13 | 180.20 | 2.582x |
| ade150 | 349.99 | 329.39 | 110.94 | 109.07 | 3.020x |

First/middle/last complete images; five rotated warmed synchronized singleton repetitions. Each image passes both6x limits. Both backbones and text banks reside in memory; this is not isolated deployment peak. Multi-arm full wall time is not latency.

The treated execution reuses precomputed member indices, whereas the inherited singleton Base retains its original Boolean selection. Lower measured times therefore are not evidence that semantic correction intrinsically accelerates inference. These are three-image timing measurements, not full-dataset mean latency. Full2080 unique keys, source/class/checkpoint identities, exact original-Base per-image replay, paired scored targets and independently downloaded confusion-archive sums were verified. All five inference workers and the controller are terminal; no inference was restarted. The desktop collector's initial OpenMP conflict was resolved by removing its unnecessary PyTorch imports; frozen remote inference sources and outputs were preserved.

54.3/29.1 are saved VIP paper numeric targets, not certified paper-protocol reproduction. VIP uses the explicit self-Value empty-row numerical repair. Inputs/profiles have developed-label provenance; paired bootstrap is conditional on those inputs, not selection-adjusted. Historical433 is a separate performance reference, not a matched treatment comparator.

vdd paired1000 image-bootstrap treatment differences:
```json
{
  "Complete20_Base": {
    "delta_pp": 0.6315999999999988,
    "ci95_pp": [
      0.48943527988354435,
      0.7774911323593023
    ]
  },
  "Complete20_FamilyShuffle": {
    "delta_pp": 0.8903000000000034,
    "ci95_pp": [
      0.705838946582481,
      1.1037537783227986
    ]
  },
  "Current20_Base": {
    "delta_pp": 0.6315999999999988,
    "ci95_pp": [
      0.48943527988354435,
      0.7774911323593023
    ]
  },
  "VIP_Complete20": {
    "delta_pp": 4.2242999999999995,
    "ci95_pp": [
      2.8744906569926005,
      5.639552360423973
    ]
  }
}
```

| Class | Complete Base IoU | Family IoU | Delta | TP change | FP change |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 35.2114 | 36.2054 | 0.9940 | +3026292 | +1023496 |
| wall | 47.0676 | 48.0587 | 0.9911 | +417508 | +143715 |
| road | 41.7969 | 42.8848 | 1.0879 | -141482 | -3409080 |
| vegetation | 66.4209 | 66.5851 | 0.1642 | +690806 | +171646 |
| vehicle | 28.7693 | 29.6743 | 0.9050 | -10867 | -430370 |
| roof | 83.2036 | 83.5144 | 0.3108 | -300300 | -1248407 |
| water | 83.0581 | 83.0262 | -0.0319 | +2165 | +64878 |

ade150 paired1000 image-bootstrap treatment differences:
```json
{
  "Complete20_Base": {
    "delta_pp": -0.01729999999999876,
    "ci95_pp": [
      -0.02956793023130455,
      -0.005453478616756247
    ]
  },
  "Complete20_FamilyShuffle": {
    "delta_pp": -0.02079999999999771,
    "ci95_pp": [
      -0.03502476216867363,
      -0.006991018117370197
    ]
  },
  "Current20_Base": {
    "delta_pp": 0.4986999999999995,
    "ci95_pp": [
      0.13524093661869765,
      0.7918176355087767
    ]
  },
  "VIP_Complete20": {
    "delta_pp": 2.9254999999999995,
    "ci95_pp": [
      2.518599121351261,
      3.288276852640824
    ]
  }
}
```

| Class | Complete Base IoU | Family IoU | Delta | TP change | FP change |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | 29.4245 | 29.4107 | -0.0138 | -3472 | +26430 |
| building | 56.7754 | 56.6864 | -0.0890 | -119312 | -122534 |
| sky | 72.1083 | 72.0785 | -0.0298 | -15360 | -4651 |
| floor | 57.4646 | 57.4640 | -0.0006 | +2075 | +3979 |
| tree | 52.9124 | 53.0840 | 0.1716 | +43680 | +6181 |
| ceiling | 56.9034 | 56.9842 | 0.0808 | +12037 | -25932 |
| road | 70.4468 | 70.4527 | 0.0059 | +1164 | -208 |
| bed | 53.2757 | 53.2528 | -0.0229 | -1571 | +4813 |
| windowpane | 39.1184 | 39.1529 | 0.0345 | +3072 | -3941 |
| grass | 28.7547 | 28.7566 | 0.0019 | +79 | -442 |
| cabinet | 40.8962 | 40.8811 | -0.0151 | -2413 | -1957 |
| sidewalk | 50.0388 | 50.0435 | 0.0047 | -35 | -1244 |
| person | 32.3176 | 32.1526 | -0.1650 | -16730 | -6823 |
| earth | 6.1588 | 6.1584 | -0.0004 | -33 | +54 |
| door | 35.2983 | 35.3105 | 0.0122 | +817 | -1066 |
| table | 37.7151 | 37.7245 | 0.0094 | -10405 | -29404 |
| mountain | 31.4960 | 31.4722 | -0.0238 | -1691 | -835 |
| plant | 28.8954 | 28.8878 | -0.0076 | -411 | +387 |
| curtain | 66.1882 | 66.1976 | 0.0094 | -343 | -1386 |
| chair | 35.6584 | 35.7811 | 0.1227 | +11511 | +10735 |
| car | 63.4679 | 63.4356 | -0.0323 | +91 | +2977 |
| water | 37.9303 | 37.9308 | 0.0005 | +2 | -62 |
| painting | 25.3364 | 25.0761 | -0.2603 | +14075 | +106332 |
| sofa | 54.9155 | 54.7325 | -0.1830 | +753 | +17946 |
| shelf | 29.8764 | 29.8444 | -0.0320 | -3207 | -5952 |
| house | 18.8871 | 18.8255 | -0.0616 | +8350 | +75472 |
| sea | 27.6019 | 27.6037 | 0.0018 | -9 | -179 |
| mirror | 32.6086 | 32.5822 | -0.0264 | +953 | +5999 |
| rug | 29.4893 | 29.4897 | 0.0004 | +251 | +748 |
| field | 12.5910 | 12.5817 | -0.0093 | -923 | +1033 |
| armchair | 21.9974 | 21.2183 | -0.7791 | -19758 | -14875 |
| seat | 26.4386 | 26.3895 | -0.0491 | +122 | +7648 |
| fence | 27.8704 | 27.8169 | -0.0535 | -997 | +2830 |
| desk | 34.7370 | 34.6979 | -0.0391 | +341 | +3416 |
| rock | 33.1511 | 33.0963 | -0.0548 | -1339 | -125 |
| wardrobe | 42.0983 | 42.0733 | -0.0250 | -390 | +420 |
| lamp | 32.5292 | 32.5226 | -0.0066 | -119 | +171 |
| bathtub | 51.9402 | 51.8665 | -0.0737 | +140 | +2548 |
| railing | 19.5671 | 19.4715 | -0.0956 | -1073 | +15827 |
| cushion | 47.5770 | 47.5519 | -0.0251 | -59 | +638 |
| base | 3.7919 | 3.8059 | 0.0140 | +223 | +1128 |
| box | 28.5447 | 28.4580 | -0.0867 | -1053 | +800 |
| column | 30.8630 | 31.0518 | 0.1888 | -1138 | -20690 |
| signboard | 26.0044 | 25.9023 | -0.1021 | +370 | +10041 |
| chest of drawers | 29.7410 | 29.7372 | -0.0038 | +1623 | +5665 |
| counter | 37.5350 | 37.3917 | -0.1433 | -887 | +3179 |
| sand | 44.3926 | 44.3904 | -0.0022 | -117 | -190 |
| sink | 39.3191 | 39.4793 | 0.1602 | -543 | -7819 |
| skyscraper | 26.4469 | 26.7497 | 0.3028 | +10907 | +16262 |
| fireplace | 48.4232 | 48.4722 | 0.0490 | -96 | -1946 |
| refrigerator | 67.0728 | 66.8391 | -0.2337 | +145 | +4714 |
| grandstand | 23.1068 | 22.9870 | -0.1198 | -741 | +4154 |
| path | 4.1205 | 4.1276 | 0.0071 | -10 | -13942 |
| stairs | 36.4871 | 36.5119 | 0.0248 | +44 | -1040 |
| runway | 40.3009 | 40.2789 | -0.0220 | +4 | +1176 |
| case | 22.6269 | 22.5801 | -0.0468 | +109 | +4922 |
| pool table | 79.2281 | 79.2764 | 0.0483 | -181 | -716 |
| pillow | 37.0292 | 37.0293 | 0.0001 | +0 | -3 |
| screen door | 0.0885 | 0.0885 | 0.0000 | +0 | +639 |
| stairway | 13.7109 | 13.6852 | -0.0257 | +123 | +4008 |
| river | 19.5303 | 19.5380 | 0.0077 | -7 | -383 |
| bridge | 16.9548 | 16.9182 | -0.0366 | +135 | +5707 |
| bookcase | 16.9215 | 16.9344 | 0.0129 | +1009 | +4535 |
| blind | 31.2292 | 31.2440 | 0.0148 | +19 | -1627 |
| coffee table | 49.4171 | 49.4371 | 0.0200 | +1981 | +3502 |
| toilet | 45.8187 | 45.9959 | 0.1772 | -145 | -6693 |
| flower | 16.8688 | 16.9536 | 0.0848 | +804 | +184 |
| book | 8.2284 | 8.1252 | -0.1032 | -636 | -371 |
| hill | 6.5991 | 6.6221 | 0.0230 | -672 | -20753 |
| bench | 31.7802 | 31.7761 | -0.0041 | +1604 | +5276 |
| countertop | 11.3911 | 11.3970 | 0.0059 | +423 | +2306 |
| stove | 35.1250 | 35.1040 | -0.0210 | -2128 | -5279 |
| palm | 24.2952 | 24.2985 | 0.0033 | -25 | -294 |
| kitchen island | 10.4141 | 10.4265 | 0.0124 | -50 | -2852 |
| computer | 42.1122 | 42.1747 | 0.0625 | +839 | +430 |
| swivel chair | 23.9621 | 23.9036 | -0.0585 | +29 | +1873 |
| boat | 34.5926 | 34.5806 | -0.0120 | +175 | +763 |
| bar | 21.6512 | 21.4602 | -0.1910 | +662 | +15255 |
| arcade machine | 20.8112 | 21.1579 | 0.3467 | +959 | +69 |
| hovel | 14.4525 | 14.3760 | -0.0765 | +745 | +13868 |
| bus | 67.6411 | 67.6252 | -0.0159 | -6 | +128 |
| towel | 39.8170 | 39.6789 | -0.1381 | +29 | +3146 |
| light | 9.6913 | 9.5878 | -0.1035 | -1082 | +779 |
| truck | 15.8812 | 15.8843 | 0.0031 | +2 | -117 |
| tower | 4.5706 | 4.5502 | -0.0204 | +373 | +35187 |
| chandelier | 31.7228 | 31.6824 | -0.0404 | +71 | +1680 |
| awning | 18.2674 | 18.4100 | 0.1426 | -387 | -8911 |
| streetlight | 20.6064 | 20.5662 | -0.0402 | +37 | +1868 |
| booth | 10.8845 | 11.3960 | 0.5115 | +1667 | -68900 |
| television receiver | 31.8064 | 31.9692 | 0.1628 | +1382 | -8 |
| airplane | 17.6805 | 17.6750 | -0.0055 | +2 | +407 |
| dirt track | 1.0410 | 1.0412 | 0.0002 | +1 | -136 |
| apparel | 3.5870 | 3.5686 | -0.0184 | -2 | +11141 |
| pole | 4.0165 | 4.0001 | -0.0164 | -18 | +18597 |
| land | 5.6221 | 5.4397 | -0.1824 | +634 | +46322 |
| bannister | 7.1113 | 6.9694 | -0.1419 | -373 | +3130 |
| escalator | 33.1105 | 33.1089 | -0.0016 | +51 | +173 |
| ottoman | 49.1550 | 48.8611 | -0.2939 | -2176 | -2109 |
| bottle | 28.1991 | 28.1237 | -0.0754 | -171 | +195 |
| buffet | 9.6494 | 10.2462 | 0.5968 | +3294 | -4957 |
| poster | 6.2571 | 6.4311 | 0.1740 | -1084 | -94745 |
| stage | 5.6007 | 5.5889 | -0.0118 | +3 | +5023 |
| van | 13.5528 | 13.5093 | -0.0435 | +6 | +4223 |
| ship | 4.6749 | 4.6717 | -0.0032 | -2 | +496 |
| fountain | 28.8249 | 28.7648 | -0.0601 | -6 | +1937 |
| conveyer belt | 46.1919 | 46.5582 | 0.3663 | -224 | -3373 |
| canopy | 10.7751 | 11.1372 | 0.3621 | +2851 | -22375 |
| washer | 74.8620 | 74.8555 | -0.0065 | -129 | -129 |
| plaything | 10.1451 | 9.9561 | -0.1890 | +79 | +13471 |
| swimming pool | 23.5363 | 23.5518 | 0.0155 | +0 | -582 |
| stool | 17.0401 | 16.9882 | -0.0519 | -38 | +1725 |
| barrel | 9.4830 | 9.4024 | -0.0806 | +0 | +786 |
| basket | 24.5849 | 24.5389 | -0.0460 | -19 | +1348 |
| waterfall | 19.3217 | 19.3292 | 0.0075 | -2 | -360 |
| tent | 41.3524 | 41.3251 | -0.0273 | +0 | +349 |
| bag | 23.3426 | 23.3054 | -0.0372 | -458 | -857 |
| minibike | 61.6533 | 61.6801 | 0.0268 | -12 | -138 |
| cradle | 48.1042 | 47.7351 | -0.3691 | -153 | +2674 |
| oven | 20.6047 | 20.3358 | -0.2689 | +39 | +9392 |
| ball | 9.8719 | 9.8420 | -0.0299 | -168 | +162 |
| food | 46.7674 | 46.6898 | -0.0776 | -694 | -22 |
| step | 2.1542 | 2.1572 | 0.0030 | +4 | -342 |
| tank | 24.1075 | 24.0755 | -0.0320 | +22 | +1174 |
| trade name | 3.7123 | 3.7036 | -0.0087 | -65 | -410 |
| microwave | 73.1425 | 72.9193 | -0.2232 | +856 | +3054 |
| pot | 26.1630 | 26.1628 | -0.0002 | -14 | -49 |
| animal | 59.8187 | 59.8396 | 0.0209 | +210 | +137 |
| bicycle | 45.5564 | 45.5666 | 0.0102 | +15 | -18 |
| lake | 7.2217 | 7.2322 | 0.0105 | -2 | -3478 |
| dishwasher | 40.0853 | 39.4618 | -0.6235 | -1033 | -1 |
| screen | 31.6663 | 31.6724 | 0.0061 | -13 | -190 |
| blanket | 8.7200 | 8.7086 | -0.0114 | -216 | +1136 |
| sculpture | 15.4491 | 15.3657 | -0.0834 | +45 | +4191 |
| hood | 27.0307 | 27.1890 | 0.1583 | -111 | -4124 |
| sconce | 8.2567 | 8.2467 | -0.0100 | -10 | +1654 |
| vase | 17.5210 | 17.5345 | 0.0135 | +211 | +823 |
| traffic light | 15.2664 | 15.3230 | 0.0566 | -165 | -3040 |
| tray | 8.3296 | 8.3373 | 0.0077 | +45 | -112 |
| ashcan | 31.9794 | 31.9544 | -0.0250 | +35 | +434 |
| fan | 28.3487 | 28.1023 | -0.2464 | +151 | +4778 |
| pier | 4.6902 | 4.6702 | -0.0200 | +5 | +3550 |
| crt screen | 0.4653 | 0.4644 | -0.0009 | -7 | -820 |
| plate | 13.2446 | 13.1936 | -0.0510 | +49 | +5641 |
| monitor | 18.3620 | 18.4148 | 0.0528 | +429 | +441 |
| bulletin board | 4.5751 | 4.5208 | -0.0543 | -78 | +6304 |
| shower | 0.5723 | 0.5718 | -0.0005 | -3 | +2158 |
| radiator | 34.4497 | 34.4177 | -0.0320 | +6 | +494 |
| glass | 10.2224 | 10.2605 | 0.0381 | +96 | +59 |
| clock | 43.3692 | 43.3503 | -0.0189 | -15 | +18 |
| flag | 29.5265 | 29.4744 | -0.0521 | +16 | +946 |

Joint numeric/cost/treatment/identity gate passed: False.
This tests static expression mass only. No conditional semantic-reliability contribution or CVPR novelty is established. No control promotion or post-score rule adjustment.
