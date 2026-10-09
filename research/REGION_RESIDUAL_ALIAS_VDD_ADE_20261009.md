# Regional residual alias weights: full VDD/ADE

Complete20 unchanged. Per Geometry32x32 tile:16 anchors at rows/cols4,12,20,28, top32 positive valid non-self original-affinity donors, original normalized masses; drop empty regions, max64/image. Align unscaled raw wide21x21 template-mean fields directly by bilinear physical-centre sampling and overlap averaging. Pool local/wide raw20-alias responses, remove within-class common response and each alias global regional offset. q=regional Pearson correlation, zero for <2 regions/variance<=1e-12, clamp[-1,1]. w=G*softmax(q); write only inherited wide LSE(tau*e+log(w))/tau. Original local/Geometry/H/g/templates/views and inherited e/salience unchanged.

Same-mass BaseUniform exact predecessor. ClassPooled/zero q exactly BaseUniform, algebra/smoke identity. All20 within-class seeded derangement directly permutes final log weights. Full sorted image inventory seeded bijection derangement supplies donor q with aligned class/alias identity; mask-free complete q bank and recipient/donor mapping frozen before masks. FamilySUM is separate strong reference, not stacked.

Cross-observer regional residual consistency, not semantic truth, competitive pseudo-labels or independent semantic reliability. Shared textures and shared errors may agree. Direct raw-field sampling is a declared proxy, not exact final upsample/stitch/sample composition. Image shuffle preserves complete-bank distribution, not every recipient weight multiset.

Previously label-developed words/profiles; source/model decision informed by prior experiments. Exploratory, not untouched independent validation.

| Dataset | Complete Base | BaseUniform | Regional | AliasShuffle | ImageShuffle | FamilySUM | VIP20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 55.0754 | 56.0450 | 55.9487 | 56.0199 | 56.0171 | 56.6678 | 51.4827 |
| ade150 | 28.4271 | 30.4118 | 30.4095 | 30.4040 | 30.4069 | 30.4076 | 25.4843 |

The candidate is rejected. VDD regional-minus-BaseUniform is -0.0963pp,
paired95% interval [-0.1589,-0.0330]; ADE is -0.0023pp,
[-0.0223,0.0189]. Neither domain establishes a reliable advantage over both
alias-identity and image-identity shuffles. Clearing the saved paper numeric
targets does not demonstrate an alias contribution: BaseUniform already clears
them. The original joint alias objective remains unmet; no control is promoted.

The signal is present, but does not rank useful aliases successfully under this
frozen rule. Mean absolute regional correlation is 0.7647 VDD / 0.6327 ADE,
and mean within-class correlation standard deviation is 0.1305 / 0.1649.
Fresh per-image correlations exactly replay the mask-free prepass on all2080
images. Thus zero signal or wrong donor-image mapping does not explain the
failure. Shared response structure still does not establish semantic truth;
the results do not prove all possible alias weighting methods are ineffective.

Cost is acceptable under the declared6x gate: primary overhead against the
same-reader BaseUniform is 1.76% VDD / 13.33% ADE. The ADE primary remains
0.7767pp below the retained historical433-query31.1862 incumbent. That bank and
the retained model are unchanged. Historical238.62ms timing is not a paired
measurement against this trial and is not used to claim an exact speed ratio.

Original controller stopped when a second GPU-idle check refused an unstarted
ADE prepass launch. Separate recovery preserved the failed status/log, completed
VDD prepass and all existing outputs; it scheduled only missing work. All five
prepass and five full worker outcomes were collected without relaunching any
existing worker. Model rules and their frozen source identities remained fixed.

| Dataset | BaseUniform ms | Regional ms | Matched20 VIP ms | Official VIP ms | Regional/official |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 339.48 | 345.46 | 127.83 | 128.31 | 2.692x |
| ade150 | 341.00 | 386.45 | 114.58 | 113.19 | 3.414x |

Three fixed complete images/domain, five rotated warmed synchronized singleton repetitions. Fresh current-image q included; no q-bank acceleration of primary. Every image must pass both6x gates. Not full-set latency. Includes resize/encoders/alias/writer/restoration/argmax, excludes decode/setup. Peaks include both resident backbones/banks.

54.3/29.1 are saved paper numeric targets, not certified paper-protocol reproduction. Explicit empty-row VIP repair retained. Inputs/profiles and method choice have developed-label provenance; full results exploratory. Historical433 ADE31.1862 remains a separate strong incumbent, not matched20 attribution.

Joint acceptance: False. No null/reference promotion or after-score adjustment.

vdd paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 55.0754,
    "Complete20_Base": 55.0754,
    "Region_BaseUniform": 56.045,
    "RegionResidual_Soft": 55.9487,
    "RegionResidual_AliasShuffle": 56.0199,
    "RegionResidual_ImageShuffle": 56.0171,
    "Complete20_FamilySum": 56.6678,
    "VIP_Complete20": 51.4827
  },
  "paired": {
    "Region_BaseUniform": {
      "delta_pp": -0.09629999999999939,
      "ci95_pp": [
        -0.1589259446366583,
        -0.03295088768185438
      ]
    },
    "RegionResidual_AliasShuffle": {
      "delta_pp": -0.07119999999999749,
      "ci95_pp": [
        -0.17813133625380786,
        0.03977307332452129
      ]
    },
    "RegionResidual_ImageShuffle": {
      "delta_pp": -0.06839999999999691,
      "ci95_pp": [
        -0.1606357246498197,
        0.03539898386988776
      ]
    },
    "Complete20_Base": {
      "delta_pp": 0.8733000000000004,
      "ci95_pp": [
        0.6682190015389635,
        1.12603408469485
      ]
    },
    "Complete20_FamilySum": {
      "delta_pp": -0.7190999999999974,
      "ci95_pp": [
        -0.8667204885672994,
        -0.5733242822672651
      ]
    },
    "Current20_Base": {
      "delta_pp": 0.8733000000000004,
      "ci95_pp": [
        0.6682190015389635,
        1.12603408469485
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 4.466000000000001,
      "ci95_pp": [
        3.1385009232323227,
        5.858252710362527
      ]
    }
  },
  "alias_and_image_effect_supported": false,
  "above_numeric_target": true,
  "historical433_delta_pp": null,
  "mean_within_class_q_std": 0.1304822862148285,
  "mean_abs_q": 0.7646851539611816,
  "diagnostics": {
    "vdd": {
      "geometry_encodings": 4.0,
      "wide_encodings": 2.0,
      "fine_forwards": 0.0,
      "additional_visual_forwards": 0.0,
      "additional_semantic_heads": 0.0,
      "alias_slots_per_class": 20.0,
      "region_count": 64.0,
      "mean_abs_q": 0.7646851874887943,
      "nonzero_q_fraction": 1.0,
      "maximum_region_elements": 8960.0,
      "prepass_q_max_abs": 0.0
    }
  },
  "independent_downloaded_coverage_replay_verified": true
}
```

| Class | BaseUniform IoU | Regional IoU | Delta pp | AliasShuffle IoU | ImageShuffle IoU | FamilySUM IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 37.3076 | 37.1069 | -0.2007 | 37.2702 | 37.2447 | 38.1923 |
| wall | 47.0781 | 46.9338 | -0.1443 | 47.0954 | 47.0612 | 48.0704 |
| road | 44.1870 | 44.1431 | -0.0439 | 44.0267 | 44.2569 | 45.2370 |
| vegetation | 66.7818 | 66.6726 | -0.1092 | 66.9921 | 66.6731 | 66.9225 |
| vehicle | 30.4368 | 30.3272 | -0.1096 | 30.3191 | 30.4134 | 31.4747 |
| roof | 83.5479 | 83.5276 | -0.0203 | 83.5898 | 83.5770 | 83.8290 |
| water | 82.9757 | 82.9298 | -0.0459 | 82.8463 | 82.8935 | 82.9489 |

ade150 paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 27.9111,
    "Complete20_Base": 28.4271,
    "Region_BaseUniform": 30.4118,
    "RegionResidual_Soft": 30.4095,
    "RegionResidual_AliasShuffle": 30.404,
    "RegionResidual_ImageShuffle": 30.4069,
    "Complete20_FamilySum": 30.4076,
    "VIP_Complete20": 25.4843
  },
  "paired": {
    "Region_BaseUniform": {
      "delta_pp": -0.0022999999999981924,
      "ci95_pp": [
        -0.02228415149765457,
        0.018852642964885823
      ]
    },
    "RegionResidual_AliasShuffle": {
      "delta_pp": 0.005500000000001393,
      "ci95_pp": [
        -0.01719112969254457,
        0.029976593634121942
      ]
    },
    "RegionResidual_ImageShuffle": {
      "delta_pp": 0.002600000000001046,
      "ci95_pp": [
        -0.027068021521399997,
        0.03325057799903863
      ]
    },
    "Complete20_Base": {
      "delta_pp": 1.982400000000002,
      "ci95_pp": [
        1.561522295900754,
        2.489851559462617
      ]
    },
    "Complete20_FamilySum": {
      "delta_pp": 0.0019000000000026773,
      "ci95_pp": [
        -0.01782133892084916,
        0.023791378695931526
      ]
    },
    "Current20_Base": {
      "delta_pp": 2.4984,
      "ci95_pp": [
        1.9713088516006063,
        3.066993256757042
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 4.9252,
      "ci95_pp": [
        4.435506664510623,
        5.4231961603489145
      ]
    }
  },
  "alias_and_image_effect_supported": false,
  "above_numeric_target": true,
  "historical433_delta_pp": -0.7766999999999982,
  "mean_within_class_q_std": 0.16494575142860413,
  "mean_abs_q": 0.6327096819877625,
  "diagnostics": {
    "ade150": {
      "geometry_encodings": 3.965,
      "wide_encodings": 2.1435,
      "fine_forwards": 0.0,
      "additional_visual_forwards": 0.0,
      "additional_semantic_heads": 0.0,
      "alias_slots_per_class": 20.0,
      "region_count": 63.42,
      "mean_abs_q": 0.6327099071741105,
      "nonzero_q_fraction": 1.0,
      "maximum_region_elements": 190260.0,
      "prepass_q_max_abs": 0.0
    }
  },
  "independent_downloaded_coverage_replay_verified": true
}
```

| Class | BaseUniform IoU | Regional IoU | Delta pp | AliasShuffle IoU | ImageShuffle IoU | FamilySUM IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wall | 33.4897 | 33.4023 | -0.0874 | 33.3717 | 33.4156 | 33.5873 |
| building | 64.4927 | 64.5075 | 0.0148 | 64.4699 | 64.5152 | 64.6900 |
| sky | 81.4131 | 81.5140 | 0.1009 | 81.3638 | 81.5012 | 81.4481 |
| floor | 58.7287 | 58.5099 | -0.2188 | 58.9559 | 58.5786 | 58.7090 |
| tree | 61.0809 | 61.0016 | -0.0793 | 60.9456 | 61.1065 | 61.2256 |
| ceiling | 53.7564 | 53.7378 | -0.0186 | 53.7377 | 53.7569 | 53.8348 |
| road | 69.4036 | 69.1933 | -0.2103 | 69.4370 | 69.3975 | 69.3973 |
| bed | 60.2901 | 60.1453 | -0.1448 | 60.2080 | 60.1687 | 60.2538 |
| windowpane | 42.2141 | 42.1757 | -0.0384 | 42.2418 | 42.1898 | 42.2342 |
| grass | 44.0294 | 44.2505 | 0.2211 | 43.9719 | 44.0456 | 44.0363 |
| cabinet | 49.2786 | 49.3251 | 0.0465 | 49.2509 | 49.2626 | 49.2759 |
| sidewalk | 47.7511 | 47.6942 | -0.0569 | 47.6877 | 47.7478 | 47.7255 |
| person | 62.1024 | 62.4751 | 0.3727 | 61.8971 | 61.9880 | 62.2191 |
| earth | 10.9932 | 10.9318 | -0.0614 | 11.0637 | 11.0411 | 10.9980 |
| door | 36.0792 | 36.0694 | -0.0098 | 36.0319 | 36.0806 | 36.0846 |
| table | 40.5313 | 40.4890 | -0.0423 | 40.4932 | 40.5519 | 40.5674 |
| mountain | 36.8118 | 36.6419 | -0.1699 | 36.7659 | 36.8501 | 36.8514 |
| plant | 37.6491 | 37.6677 | 0.0186 | 37.5130 | 37.6554 | 37.6323 |
| curtain | 64.7797 | 64.7530 | -0.0267 | 64.7774 | 64.7839 | 64.7755 |
| chair | 42.8095 | 42.7681 | -0.0414 | 42.7999 | 42.8077 | 42.9116 |
| car | 65.1720 | 65.1730 | 0.0010 | 64.9990 | 65.1039 | 65.1009 |
| water | 49.6956 | 49.6090 | -0.0866 | 49.7366 | 49.4739 | 49.6960 |
| painting | 29.7223 | 29.5345 | -0.1878 | 29.6918 | 29.4986 | 29.7733 |
| sofa | 50.0737 | 50.0173 | -0.0564 | 50.0400 | 50.0555 | 49.9595 |
| shelf | 32.5437 | 32.5852 | 0.0415 | 32.5515 | 32.5391 | 32.4887 |
| house | 14.0707 | 14.0723 | 0.0016 | 14.0816 | 14.0926 | 14.4394 |
| sea | 27.8751 | 27.6358 | -0.2393 | 27.7764 | 27.7852 | 27.8797 |
| mirror | 32.4095 | 32.4216 | 0.0121 | 32.4038 | 32.3923 | 32.4291 |
| rug | 31.1809 | 30.8757 | -0.3052 | 31.4745 | 31.0009 | 31.1808 |
| field | 16.7704 | 16.7894 | 0.0190 | 16.6986 | 16.7360 | 16.7569 |
| armchair | 12.8923 | 12.6464 | -0.2459 | 12.7823 | 12.6515 | 12.2071 |
| seat | 28.5496 | 28.6530 | 0.1034 | 28.7618 | 28.5493 | 28.4973 |
| fence | 26.0452 | 26.0636 | 0.0184 | 26.0590 | 26.0006 | 25.9436 |
| desk | 33.9102 | 33.9598 | 0.0496 | 33.8168 | 33.9038 | 33.8528 |
| rock | 34.9354 | 34.9608 | 0.0254 | 35.0275 | 35.1138 | 34.8666 |
| wardrobe | 44.2737 | 44.3248 | 0.0511 | 44.3340 | 44.1626 | 44.2577 |
| lamp | 30.1314 | 30.1624 | 0.0310 | 30.0966 | 30.1368 | 30.1097 |
| bathtub | 49.3756 | 49.3712 | -0.0044 | 49.2450 | 49.2880 | 49.3237 |
| railing | 19.3173 | 19.2412 | -0.0761 | 19.2988 | 19.3375 | 19.1828 |
| cushion | 29.8800 | 29.6589 | -0.2211 | 29.7201 | 30.0126 | 29.8716 |
| base | 3.7937 | 3.8243 | 0.0306 | 3.8017 | 3.7885 | 3.8210 |
| box | 23.6815 | 23.5745 | -0.1070 | 23.6396 | 23.7861 | 23.5843 |
| column | 35.8253 | 35.9149 | 0.0896 | 35.7128 | 35.7539 | 35.8987 |
| signboard | 27.5735 | 27.5806 | 0.0071 | 27.5793 | 27.5729 | 27.4642 |
| chest of drawers | 31.0237 | 31.0948 | 0.0711 | 31.0431 | 31.0233 | 31.0224 |
| counter | 43.3892 | 43.2960 | -0.0932 | 43.5330 | 43.3225 | 43.4392 |
| sand | 39.7822 | 39.7168 | -0.0654 | 39.9076 | 39.8520 | 39.7309 |
| sink | 36.7065 | 36.6739 | -0.0326 | 36.6976 | 36.6459 | 36.8135 |
| skyscraper | 15.5600 | 15.6217 | 0.0617 | 15.5458 | 15.6395 | 15.9632 |
| fireplace | 51.5602 | 51.4887 | -0.0715 | 51.5114 | 51.5405 | 51.5646 |
| refrigerator | 67.3956 | 67.4132 | 0.0176 | 67.4718 | 67.4250 | 67.1514 |
| grandstand | 28.1722 | 28.3939 | 0.2217 | 28.1740 | 28.3045 | 28.0416 |
| path | 3.8993 | 3.8927 | -0.0066 | 3.8603 | 3.9198 | 3.8975 |
| stairs | 38.5237 | 38.4223 | -0.1014 | 38.4093 | 38.4952 | 38.5728 |
| runway | 34.0403 | 34.1670 | 0.1267 | 34.0278 | 34.1859 | 34.0358 |
| case | 23.0048 | 23.0474 | 0.0426 | 22.9820 | 23.1208 | 22.8950 |
| pool table | 80.9349 | 80.9513 | 0.0164 | 80.9277 | 80.9093 | 80.9590 |
| pillow | 21.1471 | 21.1329 | -0.0142 | 20.9942 | 21.0254 | 21.1468 |
| screen door | 0.0460 | 0.0461 | 0.0001 | 0.0472 | 0.0460 | 0.0460 |
| stairway | 16.0215 | 15.9892 | -0.0323 | 15.9634 | 15.8585 | 16.0217 |
| river | 16.7127 | 16.8090 | 0.0963 | 16.7767 | 16.7014 | 16.7399 |
| bridge | 27.7675 | 27.7618 | -0.0057 | 27.8167 | 27.7938 | 27.7180 |
| bookcase | 17.2595 | 17.3234 | 0.0639 | 17.2773 | 17.2853 | 17.3524 |
| blind | 32.1748 | 32.2260 | 0.0512 | 32.1155 | 32.0762 | 32.1813 |
| coffee table | 49.4101 | 49.2700 | -0.1401 | 49.2615 | 49.6018 | 49.5709 |
| toilet | 47.0663 | 46.7814 | -0.2849 | 47.0903 | 47.3086 | 47.2160 |
| flower | 9.2629 | 9.1839 | -0.0790 | 9.2466 | 9.3158 | 9.3943 |
| book | 11.9314 | 12.0098 | 0.0784 | 11.9907 | 11.9023 | 11.8658 |
| hill | 5.9654 | 6.0080 | 0.0426 | 5.9451 | 6.0796 | 5.8770 |
| bench | 33.7796 | 33.8347 | 0.0551 | 33.7063 | 33.7651 | 33.7152 |
| countertop | 16.6114 | 16.6426 | 0.0312 | 16.5664 | 16.6325 | 16.6343 |
| stove | 27.2118 | 27.2029 | -0.0089 | 27.3261 | 26.9734 | 27.2095 |
| palm | 23.2396 | 23.2466 | 0.0070 | 23.2322 | 23.2826 | 23.2346 |
| kitchen island | 22.1091 | 22.1520 | 0.0429 | 22.0952 | 22.2071 | 22.0987 |
| computer | 36.9030 | 37.0570 | 0.1540 | 37.1770 | 37.1970 | 36.8698 |
| swivel chair | 6.4748 | 6.4084 | -0.0664 | 6.3115 | 6.5403 | 6.4786 |
| boat | 33.2521 | 33.0518 | -0.2003 | 33.3609 | 33.3255 | 33.2227 |
| bar | 24.0548 | 24.0181 | -0.0367 | 24.1029 | 24.0531 | 23.7993 |
| arcade machine | 7.9296 | 7.9222 | -0.0074 | 7.9566 | 7.9601 | 8.0554 |
| hovel | 12.4241 | 12.4012 | -0.0229 | 12.5560 | 12.3809 | 12.4020 |
| bus | 64.7935 | 64.8175 | 0.0240 | 64.7242 | 64.8307 | 64.7762 |
| towel | 52.2599 | 52.2610 | 0.0011 | 52.2708 | 52.2354 | 52.2165 |
| light | 5.1088 | 5.0831 | -0.0257 | 5.1001 | 5.0930 | 5.0603 |
| truck | 17.0378 | 17.0934 | 0.0556 | 17.0625 | 17.0607 | 17.0011 |
| tower | 14.9081 | 14.9682 | 0.0601 | 14.7986 | 14.8782 | 14.8694 |
| chandelier | 40.5729 | 40.4876 | -0.0853 | 40.5776 | 40.4890 | 40.5149 |
| awning | 22.8935 | 22.9098 | 0.0163 | 22.9374 | 22.9023 | 22.9509 |
| streetlight | 21.6173 | 21.6198 | 0.0025 | 21.6117 | 21.5755 | 21.6030 |
| booth | 6.7110 | 6.6986 | -0.0124 | 6.7147 | 6.6549 | 7.0287 |
| television receiver | 31.7180 | 31.7517 | 0.0337 | 31.6787 | 31.6411 | 31.7917 |
| airplane | 17.7966 | 18.1199 | 0.3233 | 17.9138 | 17.8884 | 17.7677 |
| dirt track | 2.9185 | 2.9104 | -0.0081 | 2.9204 | 2.9713 | 2.9149 |
| apparel | 6.5867 | 6.5966 | 0.0099 | 6.5032 | 6.4620 | 6.5233 |
| pole | 13.9269 | 13.9591 | 0.0322 | 13.8746 | 13.9759 | 13.8880 |
| land | 3.9174 | 3.9388 | 0.0214 | 3.8537 | 4.0341 | 3.8725 |
| bannister | 9.2243 | 8.8926 | -0.3317 | 9.2803 | 9.2651 | 9.2883 |
| escalator | 33.5887 | 33.9292 | 0.3405 | 33.2924 | 33.3237 | 33.6179 |
| ottoman | 36.9889 | 37.0393 | 0.0504 | 37.0158 | 37.4130 | 36.6583 |
| bottle | 31.0666 | 31.1580 | 0.0914 | 31.2513 | 31.2518 | 30.9765 |
| buffet | 6.8098 | 6.9124 | 0.1026 | 6.8686 | 6.2061 | 7.2297 |
| poster | 7.1694 | 7.0981 | -0.0713 | 7.1713 | 7.1376 | 7.3146 |
| stage | 13.5496 | 13.5269 | -0.0227 | 13.5776 | 13.4827 | 13.4993 |
| van | 35.0273 | 35.0498 | 0.0225 | 35.1640 | 35.0766 | 34.9791 |
| ship | 6.5103 | 6.7271 | 0.2168 | 6.4884 | 6.4573 | 6.5122 |
| fountain | 32.2235 | 32.0909 | -0.1326 | 32.2346 | 32.0623 | 32.2145 |
| conveyer belt | 45.8706 | 46.0703 | 0.1997 | 45.6297 | 45.6427 | 46.1993 |
| canopy | 10.6688 | 10.4774 | -0.1914 | 10.7481 | 10.6242 | 11.0638 |
| washer | 66.0090 | 66.5382 | 0.5292 | 66.2297 | 66.4190 | 65.9622 |
| plaything | 10.2712 | 10.2538 | -0.0174 | 10.2655 | 10.2750 | 10.0598 |
| swimming pool | 35.1700 | 35.2313 | 0.0613 | 35.1344 | 35.2249 | 35.2137 |
| stool | 24.0927 | 24.1624 | 0.0697 | 24.0957 | 24.0923 | 24.0614 |
| barrel | 21.5823 | 21.5493 | -0.0330 | 21.6689 | 21.8096 | 21.0961 |
| basket | 31.0965 | 30.9599 | -0.1366 | 31.0972 | 31.0455 | 31.0654 |
| waterfall | 23.4174 | 23.3107 | -0.1067 | 23.4473 | 23.2165 | 23.4253 |
| tent | 48.7930 | 48.6214 | -0.1716 | 48.7326 | 48.6374 | 48.7015 |
| bag | 26.0639 | 26.2308 | 0.1669 | 26.0637 | 25.9088 | 26.0206 |
| minibike | 62.1620 | 62.0788 | -0.0832 | 62.0975 | 62.0809 | 62.1785 |
| cradle | 53.9668 | 53.7295 | -0.2373 | 53.8925 | 54.0581 | 53.6212 |
| oven | 31.3077 | 31.1063 | -0.2014 | 31.2217 | 31.4915 | 30.9644 |
| ball | 12.5853 | 12.6328 | 0.0475 | 12.8381 | 12.6526 | 12.4190 |
| food | 35.5385 | 35.3651 | -0.1734 | 35.1159 | 35.9304 | 35.5169 |
| step | 2.3003 | 2.3160 | 0.0157 | 2.3337 | 2.2868 | 2.3008 |
| tank | 23.9047 | 24.6581 | 0.7534 | 24.1223 | 23.6608 | 24.5886 |
| trade name | 2.8812 | 2.8705 | -0.0107 | 2.8725 | 2.8891 | 2.8799 |
| microwave | 74.7167 | 74.6091 | -0.1076 | 74.8581 | 74.7089 | 74.5463 |
| pot | 31.2829 | 31.1406 | -0.1423 | 31.2055 | 31.3196 | 31.2521 |
| animal | 58.2169 | 58.5312 | 0.3143 | 58.0612 | 58.3661 | 58.5560 |
| bicycle | 45.3142 | 45.3011 | -0.0131 | 45.3400 | 45.2565 | 45.3068 |
| lake | 2.8413 | 2.9856 | 0.1443 | 2.7267 | 2.8247 | 2.8462 |
| dishwasher | 43.7138 | 43.4760 | -0.2378 | 43.9300 | 43.3495 | 42.9562 |
| screen | 38.6944 | 38.6999 | 0.0055 | 38.6565 | 38.6889 | 38.6598 |
| blanket | 15.0715 | 15.0173 | -0.0542 | 15.1709 | 15.0800 | 15.0477 |
| sculpture | 42.1202 | 42.0973 | -0.0229 | 41.9840 | 42.2471 | 42.6191 |
| hood | 29.6956 | 29.7884 | 0.0928 | 29.7507 | 29.5520 | 29.7664 |
| sconce | 17.2899 | 17.2857 | -0.0042 | 17.2825 | 17.2560 | 17.2838 |
| vase | 14.7275 | 14.7360 | 0.0085 | 14.7360 | 14.7685 | 14.7630 |
| traffic light | 19.0132 | 19.0210 | 0.0078 | 19.0341 | 18.9678 | 19.0656 |
| tray | 13.2419 | 13.2686 | 0.0267 | 13.1774 | 13.2918 | 13.2288 |
| ashcan | 11.6155 | 11.5099 | -0.1056 | 11.5542 | 11.5382 | 11.5533 |
| fan | 46.9541 | 46.9579 | 0.0038 | 46.8436 | 46.9779 | 47.0962 |
| pier | 4.7382 | 4.7186 | -0.0196 | 4.7294 | 4.7530 | 4.6966 |
| crt screen | 0.3564 | 0.3698 | 0.0134 | 0.3589 | 0.3603 | 0.3453 |
| plate | 24.1353 | 24.0724 | -0.0629 | 24.0719 | 24.3117 | 24.0193 |
| monitor | 19.0998 | 19.0385 | -0.0613 | 19.1965 | 19.0957 | 19.1183 |
| bulletin board | 5.4435 | 5.4567 | 0.0132 | 5.4275 | 5.4538 | 5.3965 |
| shower | 1.1400 | 1.1386 | -0.0014 | 1.1378 | 1.1395 | 1.1391 |
| radiator | 54.3377 | 54.3792 | 0.0415 | 54.3688 | 54.3061 | 54.3338 |
| glass | 9.7322 | 9.7315 | -0.0007 | 9.6835 | 9.7097 | 9.7764 |
| clock | 45.3298 | 45.4801 | 0.1503 | 45.3536 | 45.3459 | 45.3185 |
| flag | 50.2063 | 50.2477 | 0.0414 | 50.1424 | 50.1662 | 50.0586 |
