# Existing bare-root selection: VDD and ADE150

The existing-root role improves ADE Base by +0.4993pp and its fixed within-family representative control by +1.3185pp, at 2.11% singleton overhead. This supports a bare-root versus wrapper role signal. However, Root minus canonical-only is -0.5186pp, and its difference from the retained433-query model is -2.7758pp. The frozen candidate is rejected: it misses the ADE numeric target and cannot establish an advantage from retaining multiple roots over the protected canonical control. Do not promote that control or tune a mixture after scoring. VDD only verifies exact compatibility. This trial separates low execution cost from unresolved semantic evidence quality.

Prior-developed fixed20 pools and task profiles; exploratory, not untouched validation. ADE static selection uses exact generation metadata, no images or masks to choose words. VDD has no verified family provenance and uses exact original fallback.

Original20 profiled e; Root=log20+LSE_g(e_existing_bare_root(g))-logG. Original20 salience unchanged; no survivor renormalization, historical K, new queries, logG prior or fitted coefficient.

Original Base, independently replayed previous CQ, fixed seed within-family representative selection (including canonical family), and original profiled canonical+log20. No control promotion or seed selection.

Unchanged scalar local and L+.5H(W-L), original Geometry/profile/templates/physical views; no extra RGB/head/fine forwards or all-rival tensor.

This tests bare-root versus wrapper role, not image-conditioned semantic correctness. VDD fallback is compatibility, never positive alias contribution. Historical433 ADE31.1862 remains the deployment performance reference; it does not enter the candidate.

54.3/29.1 are saved VIP paper numeric targets, not certified reproduction of paper protocols.


vdd full 80 images:

| Method | mIoU |
| --- | ---: |
| Fixed20_TaskCoupled | 55.0754 |
| GeneratedFamily_Quotient | 55.0754 |
| GeneratedRoot | 55.0754 |
| GeneratedRoot_RepresentativeShuffle | 55.0754 |
| GeneratedRoot_CanonicalOnly | 51.0687 |

Retained matched20 finite VIP: 51.4827; Root delta: +3.5927pp.
The matched VIP result is reused from the verified identical-input template trial, not rerun during the current full evaluation.

Paired1000 image bootstrap (conditional on developed inputs; not method-selection adjusted):
```json
{
  "Fixed20_TaskCoupled": {
    "delta_pp": 0.0,
    "ci95_pp": [
      0.0,
      0.0
    ]
  },
  "GeneratedFamily_Quotient": {
    "delta_pp": 0.0,
    "ci95_pp": [
      0.0,
      0.0
    ]
  },
  "GeneratedRoot_RepresentativeShuffle": {
    "delta_pp": 0.0,
    "ci95_pp": [
      0.0,
      0.0
    ]
  },
  "GeneratedRoot_CanonicalOnly": {
    "delta_pp": 4.006700000000002,
    "ci95_pp": [
      2.6666942241905636,
      5.134172753690096
    ]
  }
}
```

| Class | Base | CQ | Root | RepresentativeShuffle | CanonicalOnly |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 35.2114 | 35.2114 | 35.2114 | 35.2114 | 36.5018 |
| wall | 47.0676 | 47.0676 | 47.0676 | 47.0676 | 22.4683 |
| road | 41.7969 | 41.7969 | 41.7969 | 41.7969 | 40.5072 |
| vegetation | 66.4209 | 66.4209 | 66.4209 | 66.4209 | 66.1424 |
| vehicle | 28.7693 | 28.7693 | 28.7693 | 28.7693 | 27.5934 |
| roof | 83.2036 | 83.2036 | 83.2036 | 83.2036 | 81.7350 |
| water | 83.0581 | 83.0581 | 83.0581 | 83.0581 | 82.5329 |

Largest class changes versus same-input Base:

| Class | Delta IoU pp | Delta correct coverage pixels | Delta false positives |
| --- | ---: | ---: | ---: |
| other | +0.0000 | +0 | +0 |
| wall | +0.0000 | +0 | +0 |
| road | +0.0000 | +0 | +0 |
| vegetation | +0.0000 | +0 | +0 |
| vehicle | +0.0000 | +0 | +0 |
| roof | +0.0000 | +0 | +0 |
| water | +0.0000 | +0 | +0 |

ade150 full 2000 images:

| Method | mIoU |
| --- | ---: |
| Fixed20_TaskCoupled | 27.9111 |
| GeneratedFamily_Quotient | 27.9545 |
| GeneratedRoot | 28.4104 |
| GeneratedRoot_RepresentativeShuffle | 27.0919 |
| GeneratedRoot_CanonicalOnly | 28.9290 |

Retained matched20 finite VIP: 25.0377; Root delta: +3.3727pp.
The matched VIP result is reused from the verified identical-input template trial, not rerun during the current full evaluation.

Paired1000 image bootstrap (conditional on developed inputs; not method-selection adjusted):
```json
{
  "Fixed20_TaskCoupled": {
    "delta_pp": 0.4992999999999981,
    "ci95_pp": [
      0.38177073253446825,
      0.612688169792617
    ]
  },
  "GeneratedFamily_Quotient": {
    "delta_pp": 0.45589999999999975,
    "ci95_pp": [
      0.33803771549132444,
      0.5760017540828233
    ]
  },
  "GeneratedRoot_RepresentativeShuffle": {
    "delta_pp": 1.3185000000000002,
    "ci95_pp": [
      1.0511930484235146,
      1.5109363167708911
    ]
  },
  "GeneratedRoot_CanonicalOnly": {
    "delta_pp": -0.5185999999999993,
    "ci95_pp": [
      -0.6786948219665676,
      -0.3047158516398162
    ]
  }
}
```

| Class | Base | CQ | Root | RepresentativeShuffle | CanonicalOnly |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | 37.2644 | 37.4706 | 38.9998 | 36.7987 | 40.3009 |
| building | 53.9374 | 54.2808 | 53.4251 | 53.2500 | 54.9875 |
| sky | 67.0706 | 67.3996 | 66.8524 | 65.0284 | 69.2670 |
| floor | 54.4595 | 54.3868 | 56.4480 | 54.4700 | 62.0163 |
| tree | 46.5507 | 46.9034 | 48.2978 | 47.1166 | 52.3816 |
| ceiling | 62.2003 | 62.2766 | 62.8270 | 63.7883 | 65.0194 |
| road | 71.7554 | 71.7131 | 71.5350 | 71.1480 | 70.7766 |
| bed | 61.3060 | 61.2601 | 62.3332 | 58.3475 | 64.6436 |
| windowpane | 38.9361 | 38.9865 | 39.5238 | 38.5975 | 41.0908 |
| grass | 28.6987 | 29.0278 | 26.0932 | 28.4925 | 27.2015 |
| cabinet | 40.1862 | 40.7482 | 39.4851 | 40.6996 | 40.1917 |
| sidewalk | 51.5996 | 51.5311 | 51.7263 | 51.9866 | 48.9922 |
| person | 52.9238 | 52.8107 | 54.4828 | 53.0944 | 43.4762 |
| earth | 6.8683 | 6.9085 | 6.3999 | 5.6363 | 8.5286 |
| door | 35.6291 | 35.6300 | 36.2420 | 35.5718 | 36.0042 |
| table | 37.2110 | 37.3359 | 37.5916 | 36.3146 | 37.1091 |
| mountain | 29.3978 | 29.4319 | 30.7536 | 32.1616 | 34.3258 |
| plant | 24.9594 | 24.9364 | 25.2411 | 23.6474 | 27.0589 |
| curtain | 65.9894 | 65.9511 | 65.2582 | 67.0666 | 67.1156 |
| chair | 39.4071 | 39.5152 | 39.1856 | 27.8578 | 39.9952 |
| car | 63.8844 | 63.9359 | 63.8714 | 64.3020 | 64.9079 |
| water | 42.0743 | 42.0740 | 37.5130 | 43.1715 | 40.7127 |
| painting | 37.4475 | 37.5804 | 33.7954 | 39.5058 | 34.0353 |
| sofa | 55.8983 | 55.5538 | 55.9897 | 54.3265 | 57.1072 |
| shelf | 29.7881 | 29.7496 | 30.1332 | 29.7984 | 30.2065 |
| house | 18.5089 | 18.8770 | 17.3950 | 17.8057 | 18.7355 |
| sea | 25.3428 | 25.0949 | 25.2397 | 22.1776 | 27.6952 |
| mirror | 33.9334 | 34.0181 | 35.0231 | 32.5453 | 35.2352 |
| rug | 26.1042 | 25.9671 | 27.9792 | 26.5483 | 35.3915 |
| field | 12.0354 | 12.0524 | 11.8737 | 12.0662 | 12.5685 |
| armchair | 17.3946 | 16.0748 | 15.3181 | 21.5348 | 25.8412 |
| seat | 28.6772 | 28.3591 | 27.7615 | 27.0698 | 31.2801 |
| fence | 29.4465 | 29.2670 | 29.2603 | 29.8508 | 29.3401 |
| desk | 35.2448 | 35.4477 | 34.7222 | 34.4350 | 35.0157 |
| rock | 33.7962 | 33.9051 | 32.2904 | 33.4066 | 32.7566 |
| wardrobe | 44.3601 | 44.6471 | 45.0878 | 43.1511 | 43.5494 |
| lamp | 31.1605 | 31.2332 | 31.8721 | 31.4226 | 31.7140 |
| bathtub | 55.0249 | 54.8388 | 55.0480 | 55.1446 | 55.7607 |
| railing | 20.0725 | 20.0854 | 20.2244 | 19.1177 | 20.8430 |
| cushion | 36.2775 | 37.3162 | 41.5270 | 18.1579 | 40.5863 |
| base | 2.4363 | 2.3851 | 2.3949 | 2.5969 | 2.6873 |
| box | 29.4203 | 29.5192 | 29.4375 | 28.3649 | 29.3565 |
| column | 27.8378 | 28.0281 | 33.6270 | 23.7229 | 34.9761 |
| signboard | 25.3258 | 25.2908 | 25.2636 | 24.4299 | 25.7727 |
| chest of drawers | 29.5348 | 29.5295 | 29.7536 | 29.2472 | 29.4547 |
| counter | 17.2187 | 17.4576 | 15.8461 | 14.0903 | 15.4813 |
| sand | 45.5215 | 45.5798 | 44.8239 | 46.1925 | 50.2177 |
| sink | 41.1202 | 41.3474 | 43.3874 | 40.7752 | 43.3425 |
| skyscraper | 26.3116 | 26.7076 | 26.1381 | 27.0142 | 25.2660 |
| fireplace | 43.3445 | 43.4494 | 43.2718 | 44.2995 | 44.1081 |
| refrigerator | 67.0769 | 66.8031 | 65.3776 | 65.5881 | 67.9908 |
| grandstand | 31.0854 | 31.1291 | 30.7146 | 32.5724 | 30.7099 |
| path | 4.0839 | 3.9609 | 4.8438 | 3.6762 | 4.5343 |
| stairs | 26.9875 | 27.2467 | 25.1651 | 26.4280 | 24.7134 |
| runway | 39.8775 | 39.8721 | 39.5691 | 36.1289 | 39.1960 |
| case | 23.1177 | 23.1269 | 23.2095 | 23.5895 | 24.4191 |
| pool table | 77.9326 | 77.9738 | 77.8052 | 77.2430 | 78.2962 |
| pillow | 17.1551 | 16.5481 | 13.3825 | 13.0050 | 12.4572 |
| screen door | 0.2022 | 0.1944 | 0.2163 | 0.2189 | 0.1298 |
| stairway | 6.5581 | 6.5481 | 6.5973 | 12.8466 | 6.7879 |
| river | 19.4599 | 19.4210 | 18.5520 | 18.8687 | 16.8071 |
| bridge | 17.1714 | 17.2649 | 17.1219 | 19.0932 | 17.1202 |
| bookcase | 17.2769 | 17.3055 | 17.6225 | 17.0961 | 17.9658 |
| blind | 31.0291 | 31.1072 | 31.7626 | 29.6609 | 32.4920 |
| coffee table | 45.7021 | 45.8129 | 46.6859 | 49.2345 | 44.8229 |
| toilet | 46.7573 | 47.0907 | 50.5195 | 45.0843 | 43.6861 |
| flower | 16.8811 | 16.2127 | 17.6643 | 13.4973 | 21.6592 |
| book | 7.7727 | 7.7467 | 8.9411 | 9.2115 | 11.2471 |
| hill | 6.3364 | 6.3384 | 6.1057 | 6.6509 | 5.5805 |
| bench | 32.9964 | 32.9372 | 36.6214 | 33.9193 | 35.4230 |
| countertop | 11.9826 | 12.0673 | 13.0376 | 13.0335 | 13.7120 |
| stove | 35.7841 | 35.6065 | 37.0806 | 32.2368 | 39.7728 |
| palm | 21.9829 | 22.0353 | 22.2667 | 21.2556 | 21.4515 |
| kitchen island | 10.2192 | 10.4165 | 10.5006 | 9.7510 | 11.2131 |
| computer | 47.7402 | 47.1056 | 44.2961 | 53.8996 | 42.8721 |
| swivel chair | 24.7402 | 24.5043 | 24.2360 | 22.5916 | 25.4764 |
| boat | 37.1507 | 37.4351 | 36.1203 | 30.5348 | 36.8862 |
| bar | 23.9647 | 24.8793 | 22.8082 | 14.2876 | 25.6933 |
| arcade machine | 18.8127 | 19.0811 | 19.8044 | 19.4739 | 19.4337 |
| hovel | 8.4273 | 8.3979 | 9.1423 | 10.6513 | 9.6213 |
| bus | 68.6518 | 68.3005 | 67.9641 | 68.8936 | 63.9151 |
| towel | 42.9216 | 43.1818 | 46.7632 | 40.2862 | 46.5787 |
| light | 12.1365 | 12.2269 | 11.8493 | 10.7123 | 11.4309 |
| truck | 16.7389 | 16.7849 | 17.6111 | 18.8350 | 16.0316 |
| tower | 4.0319 | 4.0849 | 4.7595 | 3.1120 | 5.3369 |
| chandelier | 36.0461 | 35.9248 | 36.5990 | 38.6306 | 36.4876 |
| awning | 11.2917 | 11.3250 | 11.4664 | 10.2538 | 12.0446 |
| streetlight | 15.8958 | 15.9452 | 18.7169 | 14.9005 | 19.3043 |
| booth | 4.8938 | 4.8372 | 4.9231 | 4.7433 | 4.2786 |
| television receiver | 26.3777 | 25.8529 | 24.4604 | 26.5190 | 24.3386 |
| airplane | 15.8662 | 15.8129 | 15.2780 | 17.0773 | 16.9761 |
| dirt track | 1.1217 | 1.1176 | 1.2319 | 1.0082 | 1.0744 |
| apparel | 5.8622 | 5.9073 | 6.5669 | 6.5576 | 6.9337 |
| pole | 4.1056 | 4.1761 | 4.1765 | 5.2989 | 4.4200 |
| land | 5.6973 | 5.8947 | 3.7748 | 4.6176 | 7.4077 |
| bannister | 6.8542 | 6.7909 | 5.7881 | 5.9590 | 6.3111 |
| escalator | 9.9283 | 10.1706 | 15.1236 | 7.1499 | 15.1126 |
| ottoman | 42.7400 | 43.5547 | 43.0028 | 35.1098 | 42.8294 |
| bottle | 27.0822 | 26.9942 | 27.7051 | 23.5561 | 36.2037 |
| buffet | 5.0641 | 4.9921 | 4.7922 | 5.2613 | 4.0386 |
| poster | 6.8226 | 6.8577 | 5.8867 | 6.6041 | 5.7369 |
| stage | 7.0271 | 7.0426 | 5.7193 | 6.0801 | 5.8739 |
| van | 14.0697 | 14.1754 | 16.3787 | 9.8829 | 17.6094 |
| ship | 4.7030 | 4.7398 | 5.3565 | 4.8732 | 5.4911 |
| fountain | 27.5551 | 27.6978 | 31.9535 | 25.8723 | 28.6320 |
| conveyer belt | 43.2556 | 43.4805 | 42.3072 | 39.5612 | 45.6879 |
| canopy | 4.3400 | 4.3512 | 4.9288 | 4.6767 | 6.5082 |
| washer | 68.1974 | 68.6790 | 70.0924 | 71.1434 | 70.1208 |
| plaything | 19.3225 | 19.6884 | 19.6419 | 15.7102 | 19.3338 |
| swimming pool | 23.8212 | 23.8172 | 23.4546 | 27.7872 | 23.5800 |
| stool | 19.4463 | 19.5850 | 20.8796 | 16.5834 | 20.1037 |
| barrel | 6.3902 | 6.5362 | 7.4731 | 4.2476 | 9.1832 |
| basket | 25.6323 | 26.1212 | 29.2255 | 28.7418 | 29.1459 |
| waterfall | 19.9715 | 20.0029 | 18.6742 | 16.4893 | 18.6884 |
| tent | 26.5077 | 26.5752 | 28.1467 | 23.7752 | 30.0339 |
| bag | 25.5179 | 25.4540 | 25.6789 | 24.4849 | 25.7991 |
| minibike | 62.6657 | 62.8904 | 62.6567 | 59.9007 | 45.6209 |
| cradle | 36.7544 | 36.4252 | 40.4760 | 25.6817 | 40.3141 |
| oven | 15.7612 | 15.3644 | 16.8535 | 13.9776 | 19.5978 |
| ball | 13.9422 | 14.6773 | 15.0912 | 10.1167 | 13.7807 |
| food | 55.6801 | 55.6562 | 54.7305 | 55.4837 | 54.1935 |
| step | 3.0960 | 3.0378 | 2.9532 | 1.9073 | 4.1178 |
| tank | 23.2868 | 23.4175 | 21.7183 | 17.9571 | 24.1512 |
| trade name | 3.3983 | 3.4057 | 3.0416 | 3.4335 | 2.9570 |
| microwave | 74.4844 | 74.2789 | 73.0068 | 73.2506 | 76.1312 |
| pot | 27.1000 | 27.3691 | 30.6568 | 28.7132 | 29.6371 |
| animal | 45.3065 | 45.7288 | 43.3209 | 49.2690 | 45.0844 |
| bicycle | 45.2669 | 45.1239 | 45.5695 | 45.5810 | 45.4682 |
| lake | 5.6604 | 5.6750 | 7.9319 | 5.6019 | 6.0383 |
| dishwasher | 41.5590 | 39.7971 | 44.3595 | 36.8961 | 50.7272 |
| screen | 33.9537 | 33.8033 | 32.8979 | 34.9755 | 31.4916 |
| blanket | 17.4460 | 17.6741 | 19.7492 | 16.8577 | 21.0713 |
| sculpture | 18.5921 | 18.5240 | 20.5418 | 17.7889 | 22.2084 |
| hood | 26.3627 | 26.3562 | 28.1075 | 26.2026 | 27.6543 |
| sconce | 9.6496 | 9.9558 | 13.1898 | 10.5841 | 13.8320 |
| vase | 18.5593 | 18.0595 | 19.5939 | 16.7186 | 24.3073 |
| traffic light | 15.3468 | 15.5193 | 16.7104 | 17.4899 | 16.3721 |
| tray | 9.4485 | 9.4907 | 10.7101 | 8.3293 | 10.7267 |
| ashcan | 34.1460 | 34.2440 | 35.8365 | 32.7463 | 34.7235 |
| fan | 35.8356 | 36.3931 | 42.7153 | 38.0974 | 42.7186 |
| pier | 6.8117 | 6.7433 | 6.2380 | 4.9719 | 6.6797 |
| crt screen | 0.4191 | 0.4201 | 0.4553 | 0.6833 | 0.4795 |
| plate | 15.4118 | 16.0970 | 25.2737 | 13.5870 | 26.0975 |
| monitor | 11.1859 | 11.3137 | 9.3711 | 13.1506 | 6.5484 |
| bulletin board | 2.4041 | 2.4361 | 2.8867 | 2.7234 | 3.0438 |
| shower | 0.5785 | 0.5838 | 0.5823 | 0.5848 | 0.6071 |
| radiator | 35.3187 | 35.4338 | 37.7916 | 32.0646 | 39.1699 |
| glass | 9.9264 | 9.9522 | 9.0364 | 8.8783 | 9.4426 |
| clock | 43.7738 | 43.8280 | 43.8944 | 42.1200 | 44.9091 |
| flag | 34.4843 | 34.7776 | 38.9265 | 33.7890 | 40.2615 |

Largest class changes versus same-input Base:

| Class | Delta IoU pp | Delta correct coverage pixels | Delta false positives |
| --- | ---: | ---: | ---: |
| water | -4.5613 | -287159 | -147067 |
| pillow | -3.7726 | -70622 | -110536 |
| painting | -3.6521 | -187736 | -147228 |
| computer | -3.4441 | +860 | +87263 |
| grass | -2.6055 | -295745 | -92413 |
| armchair | -2.0765 | -44803 | -29076 |
| flag | +4.4422 | -1607 | -53408 |
| escalator | +5.1953 | +20401 | +11660 |
| cushion | +5.2495 | +40181 | -81750 |
| column | +5.7892 | -74455 | -748222 |
| fan | +6.8797 | -6665 | -76701 |
| plate | +9.8619 | +228 | -441610 |

| Dataset | Base ms | Root ms | VIP20 ms | Official VIP ms | Root/official |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 442.98 | 442.83 | 182.57 | 182.05 | 2.432x |
| ade150 | 350.89 | 358.31 | 111.29 | 109.96 | 3.259x |

First/middle/last full images/domain;5 rotated warmed synchronized singleton repetitions of Base/Root/matched20 VIP/official-query VIP. Each image must satisfy <=6x both VIP references before any full masks.

Peak memory includes both resident backbones and caches. Multi-arm full-evaluation wall time is not singleton deployment latency.

Full ADE2000 Root must exceed29.1 and Base/CQ/representative-shuffle/canonical-only with positive paired95% intervals; VDD80 must exceed54.3, exact fallback and pass each-image speed gates. VDD fallback cannot establish useful alias handling in both domains or complete that scientific objective.

Frozen candidate gates passed: False. VDD has no positive alias contribution; the broader scientific goal remains incomplete.
