# Local lexical-role priors: full VDD/ADE

The frozen primary reaches the requested numerical accuracy and cost targets on
both full datasets: VDD57.2142 and ADE30.8940, with whole-image timing337.81ms
and231.59ms (2.644x and2.065x official-query VIP). All2080 unique images,
fixed20 input slots, paired scored targets, frozen source/word identities and
five predecessor endpoints were independently verified from downloaded arrays.

The new local prior improves the already mass-calibrated FamilySUM by0.5464pp
on VDD and0.4864pp on ADE, with positive paired95% intervals on both. This is a
verified performance candidate, not the final paper model or a conditional
semantic-reliability result. VDD also beats its protected identity null by0.3125pp;
ADE instead loses0.2877pp to that null (31.1817). Thus the documented-root
identity explanation fails on ADE. Do not promote the shuffle, claim that roots
are uniformly better expressions, or erase the earlier rejected mechanisms.

ADE remains0.2922pp below the separately retained433-query31.1862 incumbent.
That incumbent and all retained models remain unchanged. The user requested
VDD/ADE numerical and speed goals are distinct from the stronger research
condition requiring a word-identity advantage in each domain. This report keeps
the failed attribution flag visible; it does not redefine that flag as success.

Per-class competition is substantial. ADE `ball` has one recorded family, so
primary and canonical-protected shuffle have identical own local priors and
identical own wide evidence. Its IoU nevertheless differs by10.6354pp because
other class scores change. An own-class IoU change alone cannot identify an
effective or harmful alias for that class. This is an audit observation, not an
input to a tuned selector.

Complete20 frozen. Local prior p=1/(G*|E_g|): E_g is its exact existing bare root if recorded generation provenance exists, else every original lexical member. Recorded wrappers have local mass0, but remain in inherited FamilySUM wide. Local Lprime=(.07*LSE(x/.07+logp))/.07; normalized mass1/class. Zprime=Lprime+.5H(WFamilySUM-Lprime). Original Geometry/H/local scale/templates/views/inherited wide salience remain fixed.

ClassPooled local p=1/20 exactly replays previous FamilySUM. Canonical-protected noncanonical local-prior shuffle seed20261012 preserves class mass, zero count and complete weight multiset. Original raw/UniformMass/current20/VIP endpoints are exact replay references. No image shuffle for this static prior.

Static documented lexical-role prior, not conditional alias semantic correctness. Zero local weight is explicitly allowed; all20 inputs still participate in wide. Family labels are declared lexical/provenance groups, not proven independent semantic concepts. No additional image/head/fine calls or class-square alias tensors.

Previously label-developed words/profiles; source/model decision informed by prior experiments. Exploratory, not untouched independent validation.

| Dataset |Complete raw|UniformMass|FamilySUM pooled|Local role primary|AliasShuffle|VIP20|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 55.0754 | 56.0450 | 56.6678 | 57.2142 | 56.9017 | 51.4827 |
| ade150 | 28.4271 | 30.4118 | 30.4076 | 30.8940 | 31.1817 | 25.4843 |

| Dataset |FamilySUM ms|Primary ms|VIP20 ms|Official VIP ms|Added cost|Primary/official|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 340.78 | 337.81 | 128.03 | 127.75 | -0.87% | 2.644x |
| ade150 | 340.60 | 231.59 | 113.80 | 112.15 | -32.00% | 2.065x |

Three fixed complete images/domain, five rotated warmed synchronized singleton repetitions, exclusive GPU before/after each. All inference/alias/writeback/restoration/argmax included; decode/setup excluded. Shared resident memory; not full-dataset latency.

The new local reduction uses precomputed grouped member indices and vectorized [patch,class,20] arithmetic. The exact inherited pooled comparator still uses class-wise Boolean selection/counting. A lower primary latency includes this execution difference; it is not evidence that the lexical prior intrinsically accelerates the semantic calculation. Singleton primary performs no comparison-arm work.

Paper54.3/29.1 are numeric targets, not certified reproductions. VIP empty-row self-Value repair remains. Developed-domain results are exploratory. No word/head/view/threshold tuning. Historical433 ADE31.1862 is separate, not a matched20 attribution control.

Numerical accuracy/cost gate: True. Joint supported local-role candidate: False. No control promotion.

vdd paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 55.0754,
    "Complete20_Base": 55.0754,
    "Complete20_UniformMass": 56.045,
    "LocalRole_ClassPooled": 56.6678,
    "LocalRole_Prior": 57.2142,
    "LocalRole_AliasShuffle": 56.9017,
    "VIP_Complete20": 51.4827
  },
  "paired": {
    "LocalRole_ClassPooled": {
      "delta_pp": 0.5463999999999984,
      "ci95_pp": [
        0.41901845797285964,
        0.6814587806526153
      ]
    },
    "LocalRole_AliasShuffle": {
      "delta_pp": 0.3125,
      "ci95_pp": [
        0.16506645948333212,
        0.45145374169712416
      ]
    },
    "Complete20_UniformMass": {
      "delta_pp": 1.1691999999999965,
      "ci95_pp": [
        0.9256018054669612,
        1.4455630301463647
      ]
    },
    "Complete20_Base": {
      "delta_pp": 2.1387999999999963,
      "ci95_pp": [
        1.7402700948904513,
        2.60620310017564
      ]
    },
    "Current20_Base": {
      "delta_pp": 2.1387999999999963,
      "ci95_pp": [
        1.7402700948904513,
        2.60620310017564
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 5.731499999999997,
      "ci95_pp": [
        4.471668306050174,
        6.980885209475231
      ]
    }
  },
  "independent_local_prior_effect": true,
  "above_numeric_target": true,
  "historical433_delta_pp": null,
  "independently_downloaded_coverage_replay_verified": true
}
```

| Class |Pooled IoU|Primary IoU|Delta pp|Shuffle IoU|Delta TP|Delta FP|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 38.1923 | 38.9485 | 0.7562 | 38.6574 | +2436506 | +1002263 |
| wall | 48.0704 | 48.8942 | 0.8238 | 48.9208 | +388122 | +203839 |
| road | 45.2370 | 46.2933 | 1.0563 | 46.1900 | -192330 | -2951980 |
| vegetation | 66.9225 | 67.0955 | 0.1730 | 66.9985 | +747868 | +208229 |
| vehicle | 31.4747 | 32.2642 | 0.7895 | 30.6038 | -78684 | -528266 |
| roof | 83.8290 | 84.0626 | 0.2336 | 83.9922 | -276433 | -985049 |
| water | 82.9489 | 82.9410 | -0.0079 | 82.9489 | +4701 | +21214 |

ade150 paired outcomes:
```json
{
  "miou": {
    "Current20_Base": 27.9111,
    "Complete20_Base": 28.4271,
    "Complete20_UniformMass": 30.4118,
    "LocalRole_ClassPooled": 30.4076,
    "LocalRole_Prior": 30.894,
    "LocalRole_AliasShuffle": 31.1817,
    "VIP_Complete20": 25.4843
  },
  "paired": {
    "LocalRole_ClassPooled": {
      "delta_pp": 0.4863999999999997,
      "ci95_pp": [
        0.32882167335957907,
        0.6057165412214643
      ]
    },
    "LocalRole_AliasShuffle": {
      "delta_pp": -0.28770000000000095,
      "ci95_pp": [
        -0.5102712036541156,
        -0.0276876796885312
      ]
    },
    "Complete20_UniformMass": {
      "delta_pp": 0.48219999999999885,
      "ci95_pp": [
        0.3235619670192647,
        0.605577812762438
      ]
    },
    "Complete20_Base": {
      "delta_pp": 2.466899999999999,
      "ci95_pp": [
        2.0650361154080685,
        2.9345923020813194
      ]
    },
    "Current20_Base": {
      "delta_pp": 2.982899999999997,
      "ci95_pp": [
        2.4788331877037475,
        3.498856414812016
      ]
    },
    "VIP_Complete20": {
      "delta_pp": 5.409699999999997,
      "ci95_pp": [
        4.92054820219663,
        5.90962552509197
      ]
    }
  },
  "independent_local_prior_effect": false,
  "above_numeric_target": true,
  "historical433_delta_pp": -0.2922000000000011,
  "independently_downloaded_coverage_replay_verified": true
}
```

| Class |Pooled IoU|Primary IoU|Delta pp|Shuffle IoU|Delta TP|Delta FP|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wall | 33.5873 | 34.3921 | 0.8048 | 38.4021 | +638469 | -53372 |
| building | 64.6900 | 63.8015 | -0.8885 | 63.7667 | -1092474 | -830854 |
| sky | 81.4481 | 80.1986 | -1.2495 | 79.2619 | -564625 | -71668 |
| floor | 58.7090 | 59.5983 | 0.8893 | 60.5718 | +285744 | -65527 |
| tree | 61.2256 | 61.6828 | 0.4572 | 58.2067 | +263981 | +248059 |
| ceiling | 53.8348 | 55.1106 | 1.2758 | 57.1327 | -217664 | -1251712 |
| road | 69.3973 | 69.8585 | 0.4612 | 65.4100 | +422777 | +458143 |
| bed | 60.2538 | 60.4856 | 0.2318 | 63.7153 | +263007 | +365560 |
| windowpane | 42.2342 | 41.7601 | -0.4741 | 43.5892 | +87877 | +363935 |
| grass | 44.0363 | 41.6819 | -2.3544 | 38.4045 | -358682 | -232532 |
| cabinet | 49.2759 | 48.4979 | -0.7780 | 48.7921 | -131584 | -97257 |
| sidewalk | 47.7255 | 49.5425 | 1.8170 | 45.9128 | -38589 | -578675 |
| person | 62.2191 | 61.5057 | -0.7134 | 63.6540 | -117635 | -76381 |
| earth | 10.9980 | 10.8647 | -0.1333 | 11.3971 | -11758 | -9293 |
| door | 36.0846 | 36.7013 | 0.6167 | 35.9117 | -89371 | -414552 |
| table | 40.5674 | 40.6819 | 0.1145 | 40.7360 | +44986 | +90442 |
| mountain | 36.8514 | 37.6013 | 0.7499 | 37.5276 | +61178 | +38724 |
| plant | 37.6323 | 37.7936 | 0.1613 | 37.4630 | -9242 | -55742 |
| curtain | 64.7755 | 64.7735 | -0.0020 | 65.9714 | -42265 | -65043 |
| chair | 42.9116 | 43.8658 | 0.9542 | 44.3089 | +32322 | -81756 |
| car | 65.1009 | 64.2752 | -0.8257 | 63.8815 | -24591 | +37214 |
| water | 49.6960 | 48.1146 | -1.5814 | 47.5700 | -118640 | -60506 |
| painting | 29.7733 | 29.8713 | 0.0980 | 36.4452 | -11070 | -50867 |
| sofa | 49.9595 | 50.5138 | 0.5543 | 53.2194 | +1897 | -58161 |
| shelf | 32.4887 | 32.3848 | -0.1039 | 32.4159 | +5064 | +29892 |
| house | 14.4394 | 18.4966 | 4.0572 | 18.1943 | +243081 | +502743 |
| sea | 27.8797 | 27.7334 | -0.1463 | 30.1469 | +15404 | +67228 |
| mirror | 32.4291 | 32.2552 | -0.1739 | 31.1640 | -9850 | -15368 |
| rug | 31.1808 | 32.5529 | 1.3721 | 34.0852 | -16390 | -376951 |
| field | 16.7569 | 16.1814 | -0.5755 | 16.0915 | +35725 | +487221 |
| armchair | 12.2071 | 10.5204 | -1.6867 | 24.4436 | -38939 | -55549 |
| seat | 28.4973 | 28.3443 | -0.1530 | 30.3303 | +17866 | +75286 |
| fence | 25.9436 | 25.4929 | -0.4507 | 27.1653 | +7270 | +94963 |
| desk | 33.8528 | 33.6105 | -0.2423 | 34.4905 | -21312 | -46208 |
| rock | 34.8666 | 34.2313 | -0.6353 | 34.1853 | -30591 | -38157 |
| wardrobe | 44.2577 | 44.9725 | 0.7148 | 44.7351 | +9166 | -17659 |
| lamp | 30.1097 | 31.2749 | 1.1652 | 31.4413 | -7127 | -135129 |
| bathtub | 49.3237 | 48.8947 | -0.4290 | 51.3560 | -7885 | -380 |
| railing | 19.1828 | 19.5113 | 0.3285 | 19.7694 | +25785 | +64483 |
| cushion | 29.8716 | 32.4903 | 2.6187 | 32.7560 | +28185 | -9047 |
| base | 3.8210 | 3.7669 | -0.0541 | 2.9745 | -1710 | -27160 |
| box | 23.5843 | 23.1974 | -0.3869 | 22.0365 | -7034 | -10604 |
| column | 35.8987 | 36.4462 | 0.5475 | 37.5336 | -21352 | -92529 |
| signboard | 27.4642 | 26.9204 | -0.5438 | 27.5721 | -15871 | -17336 |
| chest of drawers | 31.0224 | 31.3706 | 0.3482 | 31.2472 | -6841 | -38586 |
| counter | 43.4392 | 41.5391 | -1.9001 | 34.5366 | +18710 | +93728 |
| sand | 39.7309 | 39.1947 | -0.5362 | 37.7351 | -3514 | +14877 |
| sink | 36.8135 | 38.5641 | 1.7506 | 34.2421 | -14510 | -120118 |
| skyscraper | 15.9632 | 20.2739 | 4.3107 | 18.5861 | +94405 | +188608 |
| fireplace | 51.5646 | 49.2055 | -2.3591 | 47.4575 | +3974 | +85036 |
| refrigerator | 67.1514 | 65.7448 | -1.4066 | 67.3943 | +5266 | +34832 |
| grandstand | 28.0416 | 26.6357 | -1.4059 | 30.9081 | -4636 | +33261 |
| path | 3.8975 | 5.7762 | 1.8787 | 5.7430 | +31776 | -137162 |
| stairs | 38.5728 | 38.1164 | -0.4564 | 35.9748 | -34561 | -72094 |
| runway | 34.0358 | 38.6300 | 4.5942 | 39.2453 | +114758 | +107581 |
| case | 22.8950 | 22.0604 | -0.8346 | 24.6745 | -7325 | +55470 |
| pool table | 80.9590 | 81.0495 | 0.0905 | 81.6290 | +2763 | +2536 |
| pillow | 21.1468 | 18.7825 | -2.3643 | 19.8953 | -45848 | -65189 |
| screen door | 0.0460 | 0.0444 | -0.0016 | 0.0443 | -30 | -25034 |
| stairway | 16.0217 | 17.5081 | 1.4864 | 18.7020 | +17104 | -4998 |
| river | 16.7399 | 16.1767 | -0.5632 | 11.9325 | +3678 | +61958 |
| bridge | 27.7180 | 27.1619 | -0.5561 | 28.7942 | +9947 | +58315 |
| bookcase | 17.3524 | 17.6446 | 0.2922 | 17.1669 | -4328 | -44001 |
| blind | 32.1813 | 32.9780 | 0.7967 | 34.4417 | +12670 | -18812 |
| coffee table | 49.5709 | 49.6013 | 0.0304 | 51.4324 | +8569 | +16611 |
| toilet | 47.2160 | 50.3644 | 3.1484 | 46.0330 | -2753 | -105909 |
| flower | 9.3943 | 10.8677 | 1.4734 | 13.2012 | +14212 | +43331 |
| book | 11.8658 | 12.6989 | 0.8331 | 11.6336 | +5180 | +2064 |
| hill | 5.8770 | 5.2849 | -0.5921 | 5.4402 | -9298 | +62556 |
| bench | 33.7152 | 37.9434 | 4.2282 | 36.4054 | -3782 | -204295 |
| countertop | 16.6343 | 17.9705 | 1.3362 | 18.0439 | +24958 | +30579 |
| stove | 27.2095 | 27.0292 | -0.1803 | 27.5045 | -12013 | -30657 |
| palm | 23.2346 | 23.4318 | 0.1972 | 22.3221 | -1091 | -17411 |
| kitchen island | 22.0987 | 21.0217 | -1.0770 | 21.9330 | -5693 | +14419 |
| computer | 36.8698 | 37.1164 | 0.2466 | 36.2733 | +12344 | +26212 |
| swivel chair | 6.4786 | 8.7853 | 2.3067 | 5.9336 | +10369 | +12065 |
| boat | 33.2227 | 34.4599 | 1.2372 | 33.5967 | -2880 | -42275 |
| bar | 23.7993 | 22.7612 | -1.0381 | 25.1106 | -5207 | +41204 |
| arcade machine | 8.0554 | 9.6399 | 1.5845 | 8.7495 | +4170 | +782 |
| hovel | 12.4020 | 11.4960 | -0.9060 | 12.1802 | +28765 | +400978 |
| bus | 64.7762 | 63.0787 | -1.6975 | 61.1283 | +1462 | +18689 |
| towel | 52.2165 | 54.2165 | 2.0000 | 54.5877 | -3602 | -30318 |
| light | 5.0603 | 7.4986 | 2.4383 | 7.7667 | +20117 | +57326 |
| truck | 17.0011 | 16.9187 | -0.0824 | 16.2910 | -1994 | -8141 |
| tower | 14.8694 | 15.1389 | 0.2695 | 15.5817 | +15419 | +78411 |
| chandelier | 40.5149 | 41.4892 | 0.9743 | 44.3151 | +1912 | -15651 |
| awning | 22.9509 | 22.9672 | 0.0163 | 21.1981 | +531 | +1845 |
| streetlight | 21.6030 | 21.9122 | 0.3092 | 20.8010 | -7725 | -45690 |
| booth | 7.0287 | 7.4597 | 0.4310 | 5.6345 | +741 | -175415 |
| television receiver | 31.7917 | 29.7209 | -2.0708 | 36.6026 | +3753 | +76162 |
| airplane | 17.7677 | 15.8941 | -1.8736 | 23.6425 | +421 | +155038 |
| dirt track | 2.9149 | 2.5022 | -0.4127 | 2.8833 | -1033 | +601 |
| apparel | 6.5233 | 6.3456 | -0.1777 | 7.0880 | +4357 | +102331 |
| pole | 13.8880 | 13.1476 | -0.7404 | 14.0256 | +9052 | +123597 |
| land | 3.8725 | 2.8317 | -1.0408 | 3.2633 | +13924 | +1031166 |
| bannister | 9.2883 | 11.6541 | 2.3658 | 9.6172 | +7931 | -105934 |
| escalator | 33.6179 | 38.7332 | 5.1153 | 23.6352 | +24550 | +13354 |
| ottoman | 36.6583 | 38.5309 | 1.8726 | 38.3853 | -117 | -26873 |
| bottle | 30.9765 | 31.4968 | 0.5203 | 30.0899 | +3948 | +7398 |
| buffet | 7.2297 | 7.3271 | 0.0974 | 3.1215 | -2921 | -47506 |
| poster | 7.3146 | 7.3976 | 0.0830 | 8.7070 | -5804 | -105402 |
| stage | 13.4993 | 10.3158 | -3.1835 | 11.6631 | +12725 | +351580 |
| van | 34.9791 | 36.4830 | 1.5039 | 38.7242 | +2270 | -8237 |
| ship | 6.5122 | 7.5622 | 1.0500 | 6.6332 | +5916 | +29014 |
| fountain | 32.2145 | 33.7305 | 1.5160 | 32.5993 | +2266 | -16233 |
| conveyer belt | 46.1993 | 45.6785 | -0.5208 | 45.1943 | +5034 | +14914 |
| canopy | 11.0638 | 10.9149 | -0.1489 | 8.2768 | +12927 | +134418 |
| washer | 65.9622 | 70.9971 | 5.0349 | 59.3868 | +28279 | +5223 |
| plaything | 10.0598 | 9.3665 | -0.6933 | 15.1412 | +2659 | +76477 |
| swimming pool | 35.2137 | 34.5642 | -0.6495 | 35.7375 | +1850 | +16033 |
| stool | 24.0614 | 23.0407 | -1.0207 | 22.6339 | -7059 | -16134 |
| barrel | 21.0961 | 27.1277 | 6.0316 | 14.4742 | -74 | -9364 |
| basket | 31.0654 | 33.7878 | 2.7224 | 32.0232 | -5004 | -63057 |
| waterfall | 23.4253 | 21.9498 | -1.4755 | 24.3646 | +4073 | +57264 |
| tent | 48.7015 | 46.3817 | -2.3198 | 50.1699 | +30 | +22532 |
| bag | 26.0206 | 25.4499 | -0.5707 | 25.2798 | -6724 | -12447 |
| minibike | 62.1785 | 63.4298 | 1.2513 | 64.4968 | -1663 | -7996 |
| cradle | 53.6212 | 54.8809 | 1.2597 | 56.0247 | -4737 | -16596 |
| oven | 30.9644 | 27.2760 | -3.6884 | 35.5702 | -204 | +60231 |
| ball | 12.4190 | 15.8697 | 3.4507 | 26.5051 | +7545 | +4770 |
| food | 35.5169 | 40.8588 | 5.3419 | 46.0849 | +46662 | +31305 |
| step | 2.3008 | 2.7730 | 0.4722 | 3.8545 | +1611 | +3023 |
| tank | 24.5886 | 24.5537 | -0.0349 | 24.5254 | +10162 | +42173 |
| trade name | 2.8799 | 2.6966 | -0.1833 | 2.7379 | -1685 | -28862 |
| microwave | 74.5463 | 72.6922 | -1.8541 | 74.2484 | +5523 | +22859 |
| pot | 31.2521 | 33.5690 | 2.3169 | 30.3135 | -5347 | -65003 |
| animal | 58.5560 | 61.1342 | 2.5782 | 59.7888 | +30936 | +24404 |
| bicycle | 45.3068 | 44.9185 | -0.3883 | 47.2111 | -648 | +531 |
| lake | 2.8462 | 3.2139 | 0.3677 | 3.0995 | +8836 | +181960 |
| dishwasher | 42.9562 | 48.1809 | 5.2247 | 48.1511 | +9304 | +1397 |
| screen | 38.6598 | 36.5111 | -2.1487 | 38.3372 | +2708 | +44948 |
| blanket | 15.0477 | 18.5792 | 3.5315 | 18.2528 | -7839 | -318685 |
| sculpture | 42.6191 | 43.9799 | 1.3608 | 44.7557 | +2821 | -164 |
| hood | 29.7664 | 31.7457 | 1.9793 | 27.8784 | +137 | -32475 |
| sconce | 17.2838 | 19.1110 | 1.8272 | 19.4992 | -1824 | -59978 |
| vase | 14.7630 | 15.5329 | 0.7699 | 18.7862 | -1011 | -36310 |
| traffic light | 19.0656 | 19.6267 | 0.5611 | 22.0064 | +791 | -9782 |
| tray | 13.2288 | 12.8968 | -0.3320 | 13.3903 | -2275 | -8900 |
| ashcan | 11.5533 | 13.3969 | 1.8436 | 14.1373 | +2442 | -162942 |
| fan | 47.0962 | 48.4390 | 1.3428 | 49.2465 | +1546 | -2482 |
| pier | 4.6966 | 4.1842 | -0.5124 | 4.6607 | +1284 | +131815 |
| crt screen | 0.3453 | 0.4290 | 0.0837 | 0.2447 | +416 | +49173 |
| plate | 24.0193 | 30.5139 | 6.4946 | 36.5265 | +1362 | -156874 |
| monitor | 19.1183 | 18.1358 | -0.9825 | 20.6810 | -10883 | -10629 |
| bulletin board | 5.3965 | 6.0856 | 0.6891 | 4.2073 | +3993 | +15208 |
| shower | 1.1391 | 1.0444 | -0.0947 | 1.0724 | +403 | +177906 |
| radiator | 54.3338 | 55.5911 | 1.2573 | 56.8585 | +10 | -6876 |
| glass | 9.7764 | 8.8901 | -0.8863 | 12.5225 | -2177 | -1192 |
| clock | 45.3185 | 44.7729 | -0.5456 | 45.2203 | -791 | -317 |
| flag | 50.0586 | 53.5977 | 3.5391 | 55.4150 | -2267 | -23063 |
