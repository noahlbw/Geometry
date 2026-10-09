# Family pre-salience: full VDD/ADE

h0=mean-one slot softmax(s/tem), h1=mean-one family softmax(mean_g(s)/tem) broadcast to slots. M0=G/20, M1=1/m_family. Wuv=LSE(tau*original_BF16_scaled_raw*h_u+logM_v)/tau. Four00/01/10/11, plus same-size alias-to-family identity shuffle(seed20261011). Exact neutral uniform-salience fallback. Fixed G/local/Geometry/H/g/words/templates/views.

Predesignated11 must exceed same-mass00, old01 FamilySUM and same-size alias membership shuffle with positive paired95% lower bounds on BOTH VDD80/ADE2000, exceed54.3/29.1, and pass each fixed complete timing image double6x. No coefficient/input tuning or control promotion.

Previously label-developed words/profiles; source/model decision informed by prior experiments. Exploratory, not untouched independent validation.

| Dataset |00 uniform|01 old family|10 pre-only|11 primary|MembershipShuffle|VIP20|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 56.0450 | 56.6678 | 56.0232 | 56.6345 | 56.2285 | 51.4827 |
| ade150 | 30.4118 | 30.4076 | 30.4198 | 30.4158 | 30.3973 | 25.4843 |

The predesignated11 candidate is rejected. On VDD it improves over00 and the
membership shuffle, but is worse than the already existing01 by0.0333pp,
paired95% interval [-0.0456,-0.0219]. On ADE it gains only0.0082pp over01,
[0.0019,0.0146], and does not establish a reliable gain over00 or the membership
shuffle. Its interaction between pre-amplification and family pooling is
negative on VDD (-0.0115pp) and unresolved on ADE (+0.0002pp).

The mask-free panel demonstrated a real action, not a semantic improvement.
The full factorial shows no joint reason to add this pre-amplification to the
existing family rule. Crossing54.3/29.1 cannot pass the alias attribution gate,
because00 already crosses both numeric targets. Neither10 nor a null is
promoted after scoring. The strong historical433-query ADE31.1862 remains
0.7704pp above11 and stays unchanged.

Use the corrected timing supplement below for additional-head cost. The
original00 timing implementation unnecessarily computed family amplification
too, and therefore understated added cost. Matched supplement overhead is
2.94% VDD / 24.70% ADE. Whole-image double6x gates still pass, but ADE does not
meet the earlier aspirational10-20% additional-processing budget. No frozen
model rule or full prediction was modified by this timing correction.

| Dataset |00 ms|11 ms|Matched20 VIP ms|Official VIP ms|11/official|
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 353.85 | 354.44 | 128.87 | 128.33 | 2.762x |
| ade150 | 426.98 | 427.31 | 115.05 | 113.49 | 3.765x |

Three fixed complete images, five rotated warmed synchronized singleton repetitions, each must pass double6x. All model/alias/writeback/restoration/argmax included, decode/setup excluded; not full-set latency. Memory includes both resident backbones/banks.

Numerical VIP empty-row repair retained.54.3/29.1 are saved paper numeric targets, not certified paper reproduction. Historical433ADE31.1862 remains a separate strong incumbent. Full developed-domain results are exploratory.

Joint acceptance: False. Four-factor cells are causal controls; no post-result winner promotion.

vdd paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 55.0754,
    "Slot_Uniform": 56.045,
    "Slot_Family": 56.6678,
    "Family_Uniform": 56.0232,
    "Family_Family": 56.6345,
    "Family_Shuffled": 56.2285,
    "VIP_Complete20": 51.4827
  },
  "paired": {
    "Slot_Uniform": {
      "delta_pp": 0.589500000000001,
      "ci95_pp": [
        0.4559847116382304,
        0.745201512241103
      ]
    },
    "Slot_Family": {
      "delta_pp": -0.033299999999997,
      "ci95_pp": [
        -0.04561168933532898,
        -0.021910469946989152
      ]
    },
    "Family_Shuffled": {
      "delta_pp": 0.4060000000000059,
      "ci95_pp": [
        0.25964127679101756,
        0.5849130712505494
      ]
    },
    "Family_Uniform": {
      "delta_pp": 0.6113,
      "ci95_pp": [
        0.4741968715441674,
        0.7651970199400224
      ]
    },
    "Current20_Base": {
      "delta_pp": 1.5591000000000008,
      "ci95_pp": [
        1.2574889127808675,
        1.9264436525686608
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 5.1518000000000015,
      "ci95_pp": [
        3.889968950158241,
        6.475918202113075
      ]
    }
  },
  "independent_preprocessing_and_identity_effect": false,
  "above_numeric_target": true,
  "interaction_pp": -0.011499999999998067,
  "interaction_ci95_pp": [
    -0.01642731650317657,
    -0.006419590596840053
  ],
  "historical433_delta_pp": null,
  "independent_downloaded_coverage_replay_verified": true
}
```

| Class |00 IoU|01 IoU|11 IoU|11-01 pp|Shuffle IoU|
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 37.3076 | 38.1923 | 38.1078 | -0.0845 | 37.9758 |
| wall | 47.0781 | 48.0704 | 48.0662 | -0.0042 | 46.6847 |
| road | 44.1870 | 45.2370 | 45.1107 | -0.1263 | 44.9945 |
| vegetation | 66.7818 | 66.9225 | 66.9093 | -0.0132 | 66.7713 |
| vehicle | 30.4368 | 31.4747 | 31.4647 | -0.0100 | 30.8069 |
| roof | 83.5479 | 83.8290 | 83.8329 | 0.0039 | 83.3663 |
| water | 82.9757 | 82.9489 | 82.9500 | 0.0011 | 82.9997 |

ade150 paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 27.9111,
    "Slot_Uniform": 30.4118,
    "Slot_Family": 30.4076,
    "Family_Uniform": 30.4198,
    "Family_Family": 30.4158,
    "Family_Shuffled": 30.3973,
    "VIP_Complete20": 25.4843
  },
  "paired": {
    "Slot_Uniform": {
      "delta_pp": 0.004000000000001336,
      "ci95_pp": [
        -0.011530423716762873,
        0.0173031628958185
      ]
    },
    "Slot_Family": {
      "delta_pp": 0.008200000000002206,
      "ci95_pp": [
        0.001942677623617062,
        0.014560703343010671
      ]
    },
    "Family_Shuffled": {
      "delta_pp": 0.018499999999999517,
      "ci95_pp": [
        -0.007605135833710985,
        0.04186553491722442
      ]
    },
    "Family_Uniform": {
      "delta_pp": -0.003999999999997783,
      "ci95_pp": [
        -0.019786892850490023,
        0.010094945719971092
      ]
    },
    "Current20_Base": {
      "delta_pp": 2.5046999999999997,
      "ci95_pp": [
        2.021212596439251,
        3.0893909792938667
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 4.9315,
      "ci95_pp": [
        4.424369783571028,
        5.474019743744184
      ]
    }
  },
  "independent_preprocessing_and_identity_effect": false,
  "above_numeric_target": true,
  "interaction_pp": 0.0002000000000030866,
  "interaction_ci95_pp": [
    -0.0008562874838599299,
    0.0011009296068955619
  ],
  "historical433_delta_pp": -0.7703999999999986,
  "independent_downloaded_coverage_replay_verified": true
}
```

| Class |00 IoU|01 IoU|11 IoU|11-01 pp|Shuffle IoU|
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | 33.4897 | 33.5873 | 33.5849 | -0.0024 | 33.1211 |
| building | 64.4927 | 64.6900 | 64.6781 | -0.0119 | 64.5205 |
| sky | 81.4131 | 81.4481 | 81.4631 | 0.0150 | 81.5464 |
| floor | 58.7287 | 58.7090 | 58.7105 | 0.0015 | 58.7762 |
| tree | 61.0809 | 61.2256 | 61.2412 | 0.0156 | 61.1466 |
| ceiling | 53.7564 | 53.8348 | 53.8202 | -0.0146 | 53.5605 |
| road | 69.4036 | 69.3973 | 69.4032 | 0.0059 | 69.5424 |
| bed | 60.2901 | 60.2538 | 60.2592 | 0.0054 | 60.2299 |
| windowpane | 42.2141 | 42.2342 | 42.2390 | 0.0048 | 42.1550 |
| grass | 44.0294 | 44.0363 | 44.0566 | 0.0203 | 43.5860 |
| cabinet | 49.2786 | 49.2759 | 49.3055 | 0.0296 | 49.2521 |
| sidewalk | 47.7511 | 47.7255 | 47.7301 | 0.0046 | 47.8439 |
| person | 62.1024 | 62.2191 | 62.2982 | 0.0791 | 62.4422 |
| earth | 10.9932 | 10.9980 | 11.0136 | 0.0156 | 10.8797 |
| door | 36.0792 | 36.0846 | 36.0853 | 0.0007 | 36.0059 |
| table | 40.5313 | 40.5674 | 40.5687 | 0.0013 | 40.5632 |
| mountain | 36.8118 | 36.8514 | 36.8590 | 0.0076 | 36.8410 |
| plant | 37.6491 | 37.6323 | 37.6590 | 0.0267 | 37.5608 |
| curtain | 64.7797 | 64.7755 | 64.7173 | -0.0582 | 64.6276 |
| chair | 42.8095 | 42.9116 | 42.8871 | -0.0245 | 42.9420 |
| car | 65.1720 | 65.1009 | 65.1537 | 0.0528 | 65.2212 |
| water | 49.6956 | 49.6960 | 49.6899 | -0.0061 | 49.4238 |
| painting | 29.7223 | 29.7733 | 29.7766 | 0.0033 | 29.8347 |
| sofa | 50.0737 | 49.9595 | 49.9600 | 0.0005 | 49.9311 |
| shelf | 32.5437 | 32.4887 | 32.4794 | -0.0093 | 32.4463 |
| house | 14.0707 | 14.4394 | 14.3073 | -0.1321 | 14.2598 |
| sea | 27.8751 | 27.8797 | 27.8640 | -0.0157 | 28.0996 |
| mirror | 32.4095 | 32.4291 | 32.4271 | -0.0020 | 32.4498 |
| rug | 31.1809 | 31.1808 | 31.1787 | -0.0021 | 31.3245 |
| field | 16.7704 | 16.7569 | 16.7553 | -0.0016 | 16.6332 |
| armchair | 12.8923 | 12.2071 | 12.3055 | 0.0984 | 11.8124 |
| seat | 28.5496 | 28.4973 | 28.6182 | 0.1209 | 28.5648 |
| fence | 26.0452 | 25.9436 | 25.9374 | -0.0062 | 25.9570 |
| desk | 33.9102 | 33.8528 | 33.8921 | 0.0393 | 33.8610 |
| rock | 34.9354 | 34.8666 | 34.8865 | 0.0199 | 34.9853 |
| wardrobe | 44.2737 | 44.2577 | 44.2738 | 0.0161 | 44.1795 |
| lamp | 30.1314 | 30.1097 | 30.1371 | 0.0274 | 30.0722 |
| bathtub | 49.3756 | 49.3237 | 49.2670 | -0.0567 | 49.1083 |
| railing | 19.3173 | 19.1828 | 19.1831 | 0.0003 | 19.1799 |
| cushion | 29.8800 | 29.8716 | 30.2619 | 0.3903 | 30.9704 |
| base | 3.7937 | 3.8210 | 3.8126 | -0.0084 | 3.8110 |
| box | 23.6815 | 23.5843 | 23.5383 | -0.0460 | 23.8034 |
| column | 35.8253 | 35.8987 | 35.9202 | 0.0215 | 35.7815 |
| signboard | 27.5735 | 27.4642 | 27.4941 | 0.0299 | 27.5664 |
| chest of drawers | 31.0237 | 31.0224 | 31.0435 | 0.0211 | 30.9759 |
| counter | 43.3892 | 43.4392 | 43.3466 | -0.0926 | 42.8930 |
| sand | 39.7822 | 39.7309 | 39.8451 | 0.1142 | 39.7808 |
| sink | 36.7065 | 36.8135 | 36.8442 | 0.0307 | 36.6432 |
| skyscraper | 15.5600 | 15.9632 | 15.9514 | -0.0118 | 15.8795 |
| fireplace | 51.5602 | 51.5646 | 51.5701 | 0.0055 | 51.5083 |
| refrigerator | 67.3956 | 67.1514 | 67.2010 | 0.0496 | 67.5685 |
| grandstand | 28.1722 | 28.0416 | 28.0547 | 0.0131 | 28.2473 |
| path | 3.8993 | 3.8975 | 3.8781 | -0.0194 | 3.8962 |
| stairs | 38.5237 | 38.5728 | 38.6195 | 0.0467 | 38.7960 |
| runway | 34.0403 | 34.0358 | 33.9451 | -0.0907 | 33.9550 |
| case | 23.0048 | 22.8950 | 22.9056 | 0.0106 | 22.9517 |
| pool table | 80.9349 | 80.9590 | 80.9567 | -0.0023 | 80.8745 |
| pillow | 21.1471 | 21.1468 | 21.0217 | -0.1251 | 21.2629 |
| screen door | 0.0460 | 0.0460 | 0.0459 | -0.0001 | 0.0463 |
| stairway | 16.0215 | 16.0217 | 16.0203 | -0.0014 | 16.1143 |
| river | 16.7127 | 16.7399 | 16.7272 | -0.0127 | 16.6648 |
| bridge | 27.7675 | 27.7180 | 27.7175 | -0.0005 | 27.7296 |
| bookcase | 17.2595 | 17.3524 | 17.3038 | -0.0486 | 17.2544 |
| blind | 32.1748 | 32.1813 | 32.0112 | -0.1701 | 31.9620 |
| coffee table | 49.4101 | 49.5709 | 49.5317 | -0.0392 | 49.4330 |
| toilet | 47.0663 | 47.2160 | 47.2654 | 0.0494 | 47.4620 |
| flower | 9.2629 | 9.3943 | 9.3352 | -0.0591 | 9.1679 |
| book | 11.9314 | 11.8658 | 11.8885 | 0.0227 | 11.8692 |
| hill | 5.9654 | 5.8770 | 5.9211 | 0.0441 | 6.0296 |
| bench | 33.7796 | 33.7152 | 33.7541 | 0.0389 | 33.7099 |
| countertop | 16.6114 | 16.6343 | 16.6505 | 0.0162 | 16.4834 |
| stove | 27.2118 | 27.2095 | 27.2325 | 0.0230 | 27.3288 |
| palm | 23.2396 | 23.2346 | 23.2251 | -0.0095 | 23.2393 |
| kitchen island | 22.1091 | 22.0987 | 22.1169 | 0.0182 | 21.9718 |
| computer | 36.9030 | 36.8698 | 36.8252 | -0.0446 | 36.4096 |
| swivel chair | 6.4748 | 6.4786 | 6.4212 | -0.0574 | 6.4627 |
| boat | 33.2521 | 33.2227 | 33.2330 | 0.0103 | 33.2793 |
| bar | 24.0548 | 23.7993 | 23.8965 | 0.0972 | 24.1481 |
| arcade machine | 7.9296 | 8.0554 | 8.0254 | -0.0300 | 8.0935 |
| hovel | 12.4241 | 12.4020 | 12.4079 | 0.0059 | 12.4751 |
| bus | 64.7935 | 64.7762 | 64.7686 | -0.0076 | 64.6802 |
| towel | 52.2599 | 52.2165 | 52.2606 | 0.0441 | 52.2803 |
| light | 5.1088 | 5.0603 | 5.0669 | 0.0066 | 5.1695 |
| truck | 17.0378 | 17.0011 | 17.0431 | 0.0420 | 17.0449 |
| tower | 14.9081 | 14.8694 | 14.8767 | 0.0073 | 14.8528 |
| chandelier | 40.5729 | 40.5149 | 40.5233 | 0.0084 | 40.4335 |
| awning | 22.8935 | 22.9509 | 22.9325 | -0.0184 | 22.7417 |
| streetlight | 21.6173 | 21.6030 | 21.6153 | 0.0123 | 21.6242 |
| booth | 6.7110 | 7.0287 | 6.9797 | -0.0490 | 6.8959 |
| television receiver | 31.7180 | 31.7917 | 31.8016 | 0.0099 | 31.3599 |
| airplane | 17.7966 | 17.7677 | 17.7799 | 0.0122 | 17.8252 |
| dirt track | 2.9185 | 2.9149 | 2.9170 | 0.0021 | 2.8842 |
| apparel | 6.5867 | 6.5233 | 6.5622 | 0.0389 | 6.6715 |
| pole | 13.9269 | 13.8880 | 13.9054 | 0.0174 | 13.8747 |
| land | 3.9174 | 3.8725 | 3.9145 | 0.0420 | 4.0522 |
| bannister | 9.2243 | 9.2883 | 9.2858 | -0.0025 | 9.1975 |
| escalator | 33.5887 | 33.6179 | 33.6632 | 0.0453 | 33.6735 |
| ottoman | 36.9889 | 36.6583 | 36.6456 | -0.0127 | 37.0101 |
| bottle | 31.0666 | 30.9765 | 30.9681 | -0.0084 | 30.8832 |
| buffet | 6.8098 | 7.2297 | 7.0946 | -0.1351 | 6.3382 |
| poster | 7.1694 | 7.3146 | 7.3072 | -0.0074 | 7.1643 |
| stage | 13.5496 | 13.4993 | 13.5550 | 0.0557 | 13.4790 |
| van | 35.0273 | 34.9791 | 34.9655 | -0.0136 | 34.8702 |
| ship | 6.5103 | 6.5122 | 6.5094 | -0.0028 | 6.5528 |
| fountain | 32.2235 | 32.2145 | 32.2241 | 0.0096 | 32.3467 |
| conveyer belt | 45.8706 | 46.1993 | 46.0697 | -0.1296 | 45.9206 |
| canopy | 10.6688 | 11.0638 | 11.0138 | -0.0500 | 10.4724 |
| washer | 66.0090 | 65.9622 | 65.9926 | 0.0304 | 66.6775 |
| plaything | 10.2712 | 10.0598 | 10.0719 | 0.0121 | 10.3777 |
| swimming pool | 35.1700 | 35.2137 | 35.1983 | -0.0154 | 35.1081 |
| stool | 24.0927 | 24.0614 | 24.0563 | -0.0051 | 24.1380 |
| barrel | 21.5823 | 21.0961 | 21.2229 | 0.1268 | 21.2470 |
| basket | 31.0965 | 31.0654 | 31.1227 | 0.0573 | 30.9971 |
| waterfall | 23.4174 | 23.4253 | 23.4508 | 0.0255 | 23.4062 |
| tent | 48.7930 | 48.7015 | 48.7786 | 0.0771 | 49.2169 |
| bag | 26.0639 | 26.0206 | 26.0361 | 0.0155 | 26.0113 |
| minibike | 62.1620 | 62.1785 | 62.1815 | 0.0030 | 62.1153 |
| cradle | 53.9668 | 53.6212 | 53.6831 | 0.0619 | 54.0429 |
| oven | 31.3077 | 30.9644 | 30.9953 | 0.0309 | 30.5745 |
| ball | 12.5853 | 12.4190 | 12.5358 | 0.1168 | 12.5625 |
| food | 35.5385 | 35.5169 | 35.5622 | 0.0453 | 35.9363 |
| step | 2.3003 | 2.3008 | 2.2958 | -0.0050 | 2.2013 |
| tank | 23.9047 | 24.5886 | 24.4214 | -0.1672 | 24.0979 |
| trade name | 2.8812 | 2.8799 | 2.8792 | -0.0007 | 2.8823 |
| microwave | 74.7167 | 74.5463 | 74.5572 | 0.0109 | 74.6229 |
| pot | 31.2829 | 31.2521 | 31.2071 | -0.0450 | 31.1941 |
| animal | 58.2169 | 58.5560 | 58.5119 | -0.0441 | 58.1057 |
| bicycle | 45.3142 | 45.3068 | 45.3117 | 0.0049 | 45.3372 |
| lake | 2.8413 | 2.8462 | 2.8495 | 0.0033 | 2.9237 |
| dishwasher | 43.7138 | 42.9562 | 43.2220 | 0.2658 | 44.5648 |
| screen | 38.6944 | 38.6598 | 38.6642 | 0.0044 | 38.5997 |
| blanket | 15.0715 | 15.0477 | 15.0759 | 0.0282 | 15.3772 |
| sculpture | 42.1202 | 42.6191 | 42.5111 | -0.1080 | 42.0392 |
| hood | 29.6956 | 29.7664 | 29.7714 | 0.0050 | 29.6359 |
| sconce | 17.2899 | 17.2838 | 17.3223 | 0.0385 | 17.2372 |
| vase | 14.7275 | 14.7630 | 14.7548 | -0.0082 | 14.6444 |
| traffic light | 19.0132 | 19.0656 | 19.1047 | 0.0391 | 18.7233 |
| tray | 13.2419 | 13.2288 | 13.2397 | 0.0109 | 13.2292 |
| ashcan | 11.6155 | 11.5533 | 11.5235 | -0.0298 | 11.4640 |
| fan | 46.9541 | 47.0962 | 46.9808 | -0.1154 | 46.9888 |
| pier | 4.7382 | 4.6966 | 4.6985 | 0.0019 | 4.7309 |
| crt screen | 0.3564 | 0.3453 | 0.3381 | -0.0072 | 0.3763 |
| plate | 24.1353 | 24.0193 | 24.2325 | 0.2132 | 24.2298 |
| monitor | 19.0998 | 19.1183 | 19.1024 | -0.0159 | 18.8776 |
| bulletin board | 5.4435 | 5.3965 | 5.4150 | 0.0185 | 5.3363 |
| shower | 1.1400 | 1.1391 | 1.1428 | 0.0037 | 1.1357 |
| radiator | 54.3377 | 54.3338 | 54.3451 | 0.0113 | 54.2221 |
| glass | 9.7322 | 9.7764 | 9.7470 | -0.0294 | 9.7482 |
| clock | 45.3298 | 45.3185 | 45.3561 | 0.0376 | 45.3571 |
| flag | 50.2063 | 50.0586 | 50.2184 | 0.1598 | 50.0217 |

## Corrected overhead comparator

The original00 timing implementation also computed unused family amplification. Its near-zero reported11 overhead is not the true additional cost. No prediction rule/result changed; the supplement compares against original frozen UniformMass code on the same fixed whole images and verifies identical baseline predictions.

| Dataset | Original Uniform ms |11 ms|Matched20 VIP ms|Official VIP ms|True additional overhead|11/official|
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 481.80 | 495.97 | 188.10 | 188.31 | 2.94% | 2.634x |
| ade150 | 342.73 | 427.39 | 114.59 | 113.46 | 24.70% | 3.767x |

Five rotated warmed singleton repetitions on three fixed complete images/domain, exclusive GPU checks, no labels. Not full-set latency or isolated deployment memory. Original timing artifacts retained.
