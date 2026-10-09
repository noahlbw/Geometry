# Rival-Preserving Alias Reader: Verified64-Window Screen

Same64 development image IDs, eight/domain, ONE top-left512 window/image with original whole-image wide observations. Frozen weights,20 aliases/class, unchanged semantic source/Geometry/wide readout/reconstruction. One writer change. All fields saved before labels; exact18 historical numerical/per-image controls and full source risk replay checked remotely. Unique coverage/matrix/transition/config identities checked independently after download. LoveDA P/D share images, mean counts D once; corrected IRRG; LandCover.ai substitutes for unlabeled iSAID. Not full-dataset, full-image or untouched validation.

## Prospective Decision

```json
{
  "passed": false,
  "checks": {
    "gain_at_least_point1": true,
    "at_least5_domain_wins": true,
    "no_protocol_loss_over1": true,
    "above_ContrastReversal_Exact": false,
    "above_RivalMeanLogit": true,
    "above_RivalShuffledSupport_Exact": false,
    "above_RivalTextOnly_Exact": true,
    "above_RivalAliasShuffle0_Exact": true,
    "above_RivalAliasShuffle1_Exact": true,
    "above_RivalAliasShuffle2_Exact": true,
    "above_DirectionalMean_Exact": true,
    "above_DirectionalShuffle0_Exact": true,
    "above_DirectionalShuffle1_Exact": true,
    "above_DirectionalShuffle2_Exact": true
  },
  "wins": 5,
  "mean_delta_pp": 0.5808805236204009,
  "mean_delta_vs_collapsed_pp": -0.09951700399731322,
  "worst_protocol_delta_pp": -0.0874516640197811,
  "no_automatic_full_rollout": true
}
```

Gate FAILED: preserve original coupled model and do not promote this source/writer or launch all20,092 images.

## Fixed-Class Paired Metrics

Baseline union-positive classes remain scored, including IoU0 after zero union. This avoids denominator changes.

| Dataset/protocol | Geometry | Anchored_Exact | ContrastReversal_Exact | RivalPreserving_Exact | RivalMeanLogit | RivalShuffledSupport_Exact | RivalTextOnly_Exact | RivalAliasShuffle0_Exact | RivalAliasShuffle1_Exact | RivalAliasShuffle2_Exact | DirectionalMean_Exact | DirectionalShuffle0_Exact | DirectionalShuffle1_Exact | DirectionalShuffle2_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 55.619109 | 54.836328 | 55.293208 | 55.395613 | 53.798923 | 54.168700 | 54.361658 | 54.290570 | 54.726310 | 54.752826 | 54.712213 | 54.654820 |
| potsdam/potsdam | 35.703546 | 38.920258 | 38.592874 | 38.832806 | 38.878269 | 38.733210 | 39.817692 | 38.818082 | 38.885358 | 38.930198 | 38.871178 | 38.863785 | 38.874339 | 38.864110 |
| udd5/udd5 | 30.500979 | 28.175811 | 30.178447 | 29.942522 | 29.973981 | 29.965271 | 28.033410 | 28.611087 | 29.408245 | 28.613619 | 29.016408 | 29.025676 | 28.995073 | 28.995758 |
| oem/oem | 39.820583 | 39.023152 | 39.232807 | 39.122037 | 38.737652 | 38.717890 | 39.663912 | 39.096738 | 39.066693 | 39.112421 | 39.057569 | 39.055697 | 39.061738 | 39.063633 |
| loveda/P | 49.539928 | 50.706552 | 53.152193 | 51.919532 | 43.643607 | 52.323740 | 50.749798 | 51.859535 | 50.934323 | 52.326175 | 50.393144 | 49.845505 | 49.843500 | 50.394383 |
| loveda/D | 33.877920 | 30.736030 | 32.743830 | 32.571870 | 28.840013 | 35.517785 | 29.835409 | 32.202358 | 32.240351 | 32.036987 | 31.691250 | 31.808303 | 31.633017 | 31.687490 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.608847 | 51.777031 | 50.717923 | 51.608204 | 51.282612 | 51.923843 | 51.914651 | 51.949011 | 51.771760 | 51.765309 | 51.764841 | 51.764465 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.865626 | 66.892068 | 66.417529 | 66.503452 | 66.505182 | 66.903120 | 66.913246 | 66.891877 | 66.891267 | 66.891601 | 66.889532 | 66.901248 |
| flair1/flair1 | 35.605867 | 33.608404 | 33.545232 | 33.615973 | 32.368367 | 33.459339 | 32.989548 | 33.637296 | 33.610473 | 33.647469 | 33.603552 | 33.604885 | 33.611239 | 33.602250 |
| Equal-domain mean | 40.531033 | 42.867949 | 43.548346 | 43.448829 | 42.653368 | 43.737596 | 42.740836 | 43.170153 | 43.300085 | 43.184019 | 43.203662 | 43.221010 | 43.192749 | 43.191722 |

## Historical Standard Metrics

| Dataset/protocol | Geometry | Anchored_Exact | ContrastReversal_Exact | RivalPreserving_Exact | RivalMeanLogit | RivalShuffledSupport_Exact | RivalTextOnly_Exact | RivalAliasShuffle0_Exact | RivalAliasShuffle1_Exact | RivalAliasShuffle2_Exact | DirectionalMean_Exact | DirectionalShuffle0_Exact | DirectionalShuffle1_Exact | DirectionalShuffle2_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 55.619109 | 54.836328 | 55.293208 | 55.395613 | 53.798923 | 54.168700 | 54.361658 | 54.290570 | 54.726310 | 54.752826 | 54.712213 | 54.654820 |
| potsdam/potsdam | 35.703546 | 38.920258 | 38.592874 | 38.832806 | 38.878269 | 38.733210 | 39.817692 | 38.818082 | 38.885358 | 38.930198 | 38.871178 | 38.863785 | 38.874339 | 38.864110 |
| udd5/udd5 | 30.500979 | 28.175811 | 30.178447 | 29.942522 | 29.973981 | 29.965271 | 28.033410 | 28.611087 | 29.408245 | 28.613619 | 29.016408 | 29.025676 | 28.995073 | 28.995758 |
| oem/oem | 39.820583 | 39.023152 | 39.232807 | 39.122037 | 38.737652 | 38.717890 | 39.663912 | 39.096738 | 39.066693 | 39.112421 | 39.057569 | 39.055697 | 39.061738 | 39.063633 |
| loveda/P | 49.539928 | 50.706552 | 53.152193 | 51.919532 | 43.643607 | 52.323740 | 50.749798 | 51.859535 | 50.934323 | 52.326175 | 50.393144 | 49.845505 | 49.843500 | 50.394383 |
| loveda/D | 33.877920 | 30.736030 | 32.743830 | 32.571870 | 28.840013 | 35.517785 | 29.835409 | 32.202358 | 32.240351 | 32.036987 | 31.691250 | 31.808303 | 31.633017 | 31.687490 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.608847 | 51.777031 | 50.717923 | 51.608204 | 51.282612 | 51.923843 | 51.914651 | 51.949011 | 51.771760 | 51.765309 | 51.764841 | 51.764465 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.865626 | 66.892068 | 66.417529 | 66.503452 | 66.505182 | 66.903120 | 66.913246 | 66.891877 | 66.891267 | 66.891601 | 66.889532 | 66.901248 |
| flair1/flair1 | 38.842764 | 33.608404 | 33.545232 | 33.615973 | 32.368367 | 33.459339 | 32.989548 | 33.637296 | 33.610473 | 33.647469 | 33.603552 | 33.604885 | 33.611239 | 33.602250 |
| Equal-domain mean | 40.935645 | 42.867949 | 43.548346 | 43.448829 | 42.653368 | 43.737596 | 42.740836 | 43.170153 | 43.300085 | 43.184019 | 43.203662 | 43.221010 | 43.192749 | 43.191722 |

## Directed Admission And Consistency

v_c=sum_d e(c,d)/C is classical graph least squares, not a new solver or semantic certificate. It discards cycles and may dilute sparse margins. Positive potential is relative competition, not an alias boost. Final A/C can change even if its directed observed correction is0. Broad-only admission does not screen local20 words.

| Dataset/protocol | Directed suppression | Signed potential magnitude | Positive fraction | Mean rival risk | Discarded cycle fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 0.016469 | 0.021219 | 0.533901 | 0.039186 | 0.354360 |
| potsdam/potsdam | 0.006002 | 0.009863 | 0.410095 | 0.016983 | 0.367854 |
| udd5/udd5 | 0.031023 | 0.037103 | 0.450757 | 0.050669 | 0.358965 |
| oem/oem | 0.005192 | 0.008268 | 0.434769 | 0.014964 | 0.488971 |
| loveda/P | 0.011163 | 0.015687 | 0.410217 | 0.025332 | 0.265504 |
| loveda/D | 0.013889 | 0.017552 | 0.418614 | 0.027962 | 0.293820 |
| vaihingen/vaihingen | 0.007400 | 0.011536 | 0.390381 | 0.023672 | 0.320962 |
| landcoverai/landcoverai | 0.004023 | 0.006985 | 0.331812 | 0.011305 | 0.318707 |
| flair1/flair1 | 0.002685 | 0.004217 | 0.400258 | 0.008871 | 0.623927 |

Directional controls preserve every directed-pair total and capacities; shuffles also preserve directed-pair spectra. They preserve final class-potential aggregate budgets, NOT final potential spectra or pair covariance/cycle structure. They are not retroactively substituted for the previous independent-class spatial controls.

## Correction Transitions

| Dataset/protocol | Method | Beneficial | Harmful | Wrong-to-wrong |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | Geometry | 200243 | 664465 | 229565 |
| vdd/vdd | ContrastReversal_Exact | 32597 | 11202 | 16644 |
| vdd/vdd | RivalPreserving_Exact | 14863 | 5245 | 7760 |
| vdd/vdd | RivalMeanLogit | 74558 | 38495 | 21803 |
| vdd/vdd | RivalShuffledSupport_Exact | 24702 | 20092 | 11405 |
| vdd/vdd | RivalTextOnly_Exact | 5949 | 2357 | 2282 |
| vdd/vdd | RivalAliasShuffle0_Exact | 7975 | 3417 | 4581 |
| vdd/vdd | RivalAliasShuffle1_Exact | 9551 | 4590 | 6696 |
| vdd/vdd | RivalAliasShuffle2_Exact | 6973 | 3805 | 4269 |
| vdd/vdd | DirectionalMean_Exact | 11271 | 5436 | 7507 |
| vdd/vdd | DirectionalShuffle0_Exact | 11482 | 5384 | 7510 |
| vdd/vdd | DirectionalShuffle1_Exact | 11364 | 5580 | 7496 |
| vdd/vdd | DirectionalShuffle2_Exact | 11051 | 5597 | 7565 |
| potsdam/potsdam | Geometry | 98825 | 248015 | 200337 |
| potsdam/potsdam | ContrastReversal_Exact | 6658 | 17492 | 4969 |
| potsdam/potsdam | RivalPreserving_Exact | 4830 | 8340 | 2525 |
| potsdam/potsdam | RivalMeanLogit | 64252 | 62651 | 38557 |
| potsdam/potsdam | RivalShuffledSupport_Exact | 17139 | 19688 | 9209 |
| potsdam/potsdam | RivalTextOnly_Exact | 48244 | 24661 | 25980 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | 4445 | 8931 | 1928 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | 3828 | 5631 | 1973 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | 4692 | 5512 | 1925 |
| potsdam/potsdam | DirectionalMean_Exact | 3395 | 6058 | 1585 |
| potsdam/potsdam | DirectionalShuffle0_Exact | 3378 | 6274 | 1852 |
| potsdam/potsdam | DirectionalShuffle1_Exact | 3511 | 6153 | 1609 |
| potsdam/potsdam | DirectionalShuffle2_Exact | 3517 | 6549 | 1713 |
| udd5/udd5 | Geometry | 29933 | 87090 | 153751 |
| udd5/udd5 | ContrastReversal_Exact | 27446 | 3929 | 28124 |
| udd5/udd5 | RivalPreserving_Exact | 23007 | 2773 | 25162 |
| udd5/udd5 | RivalMeanLogit | 24593 | 6055 | 38107 |
| udd5/udd5 | RivalShuffledSupport_Exact | 38685 | 3920 | 33345 |
| udd5/udd5 | RivalTextOnly_Exact | 2096 | 7133 | 14672 |
| udd5/udd5 | RivalAliasShuffle0_Exact | 3992 | 1465 | 13412 |
| udd5/udd5 | RivalAliasShuffle1_Exact | 12945 | 1186 | 24562 |
| udd5/udd5 | RivalAliasShuffle2_Exact | 3057 | 1760 | 8266 |
| udd5/udd5 | DirectionalMean_Exact | 18874 | 2679 | 22205 |
| udd5/udd5 | DirectionalShuffle0_Exact | 19474 | 2776 | 22472 |
| udd5/udd5 | DirectionalShuffle1_Exact | 18923 | 2725 | 21778 |
| udd5/udd5 | DirectionalShuffle2_Exact | 18942 | 2790 | 22504 |
| oem/oem | Geometry | 215866 | 181426 | 84092 |
| oem/oem | ContrastReversal_Exact | 13098 | 2295 | 2480 |
| oem/oem | RivalPreserving_Exact | 4808 | 862 | 1051 |
| oem/oem | RivalMeanLogit | 58349 | 56017 | 37223 |
| oem/oem | RivalShuffledSupport_Exact | 4951 | 15448 | 6991 |
| oem/oem | RivalTextOnly_Exact | 39910 | 2363 | 5223 |
| oem/oem | RivalAliasShuffle0_Exact | 3853 | 1426 | 1394 |
| oem/oem | RivalAliasShuffle1_Exact | 2843 | 1644 | 1534 |
| oem/oem | RivalAliasShuffle2_Exact | 4985 | 1344 | 1411 |
| oem/oem | DirectionalMean_Exact | 2676 | 1010 | 1089 |
| oem/oem | DirectionalShuffle0_Exact | 2658 | 1077 | 1081 |
| oem/oem | DirectionalShuffle1_Exact | 2715 | 1024 | 1169 |
| oem/oem | DirectionalShuffle2_Exact | 2877 | 1026 | 1032 |
| loveda/P | Geometry | 78545 | 16195 | 37158 |
| loveda/P | ContrastReversal_Exact | 3001 | 1286 | 1760 |
| loveda/P | RivalPreserving_Exact | 2242 | 711 | 1260 |
| loveda/P | RivalMeanLogit | 18276 | 8176 | 17789 |
| loveda/P | RivalShuffledSupport_Exact | 5267 | 2378 | 5134 |
| loveda/P | RivalTextOnly_Exact | 36 | 783 | 298 |
| loveda/P | RivalAliasShuffle0_Exact | 2078 | 476 | 873 |
| loveda/P | RivalAliasShuffle1_Exact | 1224 | 786 | 1128 |
| loveda/P | RivalAliasShuffle2_Exact | 1779 | 412 | 1080 |
| loveda/P | DirectionalMean_Exact | 586 | 265 | 381 |
| loveda/P | DirectionalShuffle0_Exact | 564 | 257 | 461 |
| loveda/P | DirectionalShuffle1_Exact | 590 | 302 | 358 |
| loveda/P | DirectionalShuffle2_Exact | 559 | 218 | 344 |
| loveda/D | Geometry | 453985 | 129960 | 264121 |
| loveda/D | ContrastReversal_Exact | 51853 | 1697 | 7464 |
| loveda/D | RivalPreserving_Exact | 56926 | 1003 | 6919 |
| loveda/D | RivalMeanLogit | 107268 | 25005 | 53245 |
| loveda/D | RivalShuffledSupport_Exact | 157530 | 16357 | 29919 |
| loveda/D | RivalTextOnly_Exact | 6077 | 48796 | 9837 |
| loveda/D | RivalAliasShuffle0_Exact | 33785 | 847 | 4458 |
| loveda/D | RivalAliasShuffle1_Exact | 37328 | 680 | 4636 |
| loveda/D | RivalAliasShuffle2_Exact | 33401 | 573 | 4452 |
| loveda/D | DirectionalMean_Exact | 41815 | 500 | 4481 |
| loveda/D | DirectionalShuffle0_Exact | 42522 | 497 | 4576 |
| loveda/D | DirectionalShuffle1_Exact | 40983 | 693 | 4518 |
| loveda/D | DirectionalShuffle2_Exact | 41157 | 448 | 4419 |
| vaihingen/vaihingen | Geometry | 97388 | 179006 | 102887 |
| vaihingen/vaihingen | ContrastReversal_Exact | 5316 | 11486 | 6283 |
| vaihingen/vaihingen | RivalPreserving_Exact | 2684 | 4235 | 4087 |
| vaihingen/vaihingen | RivalMeanLogit | 41222 | 60755 | 45576 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | 3508 | 9778 | 7890 |
| vaihingen/vaihingen | RivalTextOnly_Exact | 7659 | 15044 | 12237 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | 3605 | 1860 | 2211 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | 2846 | 1596 | 2577 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | 4004 | 1895 | 2557 |
| vaihingen/vaihingen | DirectionalMean_Exact | 1481 | 2839 | 2096 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | 1537 | 2996 | 2176 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | 1603 | 3049 | 2024 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | 1596 | 3064 | 2184 |
| landcoverai/landcoverai | Geometry | 37157 | 113587 | 11811 |
| landcoverai/landcoverai | ContrastReversal_Exact | 1644 | 1432 | 29 |
| landcoverai/landcoverai | RivalPreserving_Exact | 791 | 629 | 28 |
| landcoverai/landcoverai | RivalMeanLogit | 24012 | 29642 | 1451 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | 3152 | 8735 | 165 |
| landcoverai/landcoverai | RivalTextOnly_Exact | 13696 | 15263 | 284 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | 1208 | 830 | 30 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | 1023 | 577 | 18 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | 610 | 538 | 19 |
| landcoverai/landcoverai | DirectionalMean_Exact | 760 | 687 | 41 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | 717 | 644 | 35 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | 737 | 680 | 30 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | 764 | 591 | 38 |
| flair1/flair1 | Geometry | 134726 | 98118 | 101041 |
| flair1/flair1 | ContrastReversal_Exact | 4222 | 14203 | 5305 |
| flair1/flair1 | RivalPreserving_Exact | 1827 | 3042 | 1487 |
| flair1/flair1 | RivalMeanLogit | 39141 | 70429 | 50241 |
| flair1/flair1 | RivalShuffledSupport_Exact | 2556 | 6787 | 3554 |
| flair1/flair1 | RivalTextOnly_Exact | 12218 | 24898 | 14442 |
| flair1/flair1 | RivalAliasShuffle0_Exact | 1809 | 2523 | 1561 |
| flair1/flair1 | RivalAliasShuffle1_Exact | 1423 | 2755 | 1600 |
| flair1/flair1 | RivalAliasShuffle2_Exact | 2173 | 2243 | 1688 |
| flair1/flair1 | DirectionalMean_Exact | 1052 | 1643 | 875 |
| flair1/flair1 | DirectionalShuffle0_Exact | 1151 | 1737 | 967 |
| flair1/flair1 | DirectionalShuffle1_Exact | 1181 | 1550 | 927 |
| flair1/flair1 | DirectionalShuffle2_Exact | 1048 | 1685 | 922 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU % | Delta pp | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -47.4915 | 70.4934 | 12.9511 | 9.3800 | -604535 | -114546 |
| vdd/vdd | Geometry | wall | 19.8084 | -39.8732 | 20.3031 | 89.0471 | 34.8907 | 4281 | 508235 |
| vdd/vdd | Geometry | road | 24.9744 | 1.7494 | 29.4846 | 62.0157 | 6.9810 | -24557 | -118754 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 25.6066 | 89.0479 | 76.2942 | 19.7359 | 150148 | 33618 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -46.4204 | 3.8880 | 100.0000 | 9.2105 | 6 | 178243 |
| vdd/vdd | Geometry | roof | 63.9822 | -22.2785 | 64.6648 | 98.3769 | 9.1667 | -1294 | 48679 |
| vdd/vdd | Geometry | water | 74.5988 | 21.7649 | 75.8754 | 97.7943 | 10.6353 | 11729 | -71253 |
| vdd/vdd | Anchored_Exact | other | 59.7766 | 0.0000 | 81.1541 | 69.4120 | 43.6684 | 0 | 0 |
| vdd/vdd | Anchored_Exact | wall | 59.6816 | 0.0000 | 65.8225 | 86.4811 | 10.4520 | 0 | 0 |
| vdd/vdd | Anchored_Exact | road | 23.2250 | 0.0000 | 23.3759 | 97.2962 | 13.8146 | 0 | 0 |
| vdd/vdd | Anchored_Exact | vegetation | 44.1426 | 0.0000 | 94.9106 | 45.2128 | 10.9733 | 0 | 0 |
| vdd/vdd | Anchored_Exact | vehicle | 50.3084 | 0.0000 | 50.3286 | 99.9201 | 0.7110 | 0 | 0 |
| vdd/vdd | Anchored_Exact | roof | 86.2607 | 0.0000 | 86.7114 | 99.4009 | 6.9072 | 0 | 0 |
| vdd/vdd | Anchored_Exact | water | 52.8339 | 0.0000 | 55.7407 | 91.0164 | 13.4736 | 0 | 0 |
| vdd/vdd | ContrastReversal_Exact | other | 60.8784 | 1.1018 | 80.2921 | 71.5734 | 45.5116 | 23142 | 15512 |
| vdd/vdd | ContrastReversal_Exact | wall | 60.4642 | 0.7826 | 67.2467 | 85.7037 | 10.1387 | -1297 | -5274 |
| vdd/vdd | ContrastReversal_Exact | road | 23.5324 | 0.3074 | 23.6539 | 97.8637 | 13.7319 | 395 | -2130 |
| vdd/vdd | ContrastReversal_Exact | vegetation | 44.3584 | 0.2158 | 95.8907 | 45.2180 | 10.8624 | 25 | -2351 |
| vdd/vdd | ContrastReversal_Exact | vehicle | 55.1886 | 4.8802 | 56.7518 | 95.2463 | 0.6010 | -351 | -1955 |
| vdd/vdd | ContrastReversal_Exact | roof | 87.9750 | 1.7143 | 88.4784 | 99.3574 | 6.7663 | -55 | -2900 |
| vdd/vdd | ContrastReversal_Exact | water | 56.9368 | 4.1029 | 60.4455 | 90.7482 | 12.3883 | -464 | -22297 |
| vdd/vdd | RivalPreserving_Exact | other | 60.0642 | 0.2876 | 80.8039 | 70.0613 | 44.2679 | 6952 | 5621 |
| vdd/vdd | RivalPreserving_Exact | wall | 59.4998 | -0.1818 | 65.6454 | 86.4050 | 10.4710 | -127 | 525 |
| vdd/vdd | RivalPreserving_Exact | road | 23.4567 | 0.2317 | 23.5789 | 97.8378 | 13.7719 | 377 | -1273 |
| vdd/vdd | RivalPreserving_Exact | vegetation | 44.6084 | 0.4658 | 95.2319 | 45.6274 | 11.0365 | 2003 | -676 |
| vdd/vdd | RivalPreserving_Exact | vehicle | 53.4746 | 3.1662 | 53.6204 | 99.4940 | 0.6645 | -32 | -943 |
| vdd/vdd | RivalPreserving_Exact | roof | 87.8829 | 1.6222 | 88.4015 | 99.3368 | 6.7708 | -81 | -2780 |
| vdd/vdd | RivalPreserving_Exact | water | 54.8678 | 2.0339 | 57.8866 | 91.3203 | 13.0175 | 526 | -10092 |
| vdd/vdd | RivalMeanLogit | other | 61.2334 | 1.4568 | 80.4962 | 71.9010 | 45.6040 | 26650 | 13942 |
| vdd/vdd | RivalMeanLogit | wall | 62.0343 | 2.3527 | 69.4654 | 85.2919 | 9.7677 | -1984 | -12367 |
| vdd/vdd | RivalMeanLogit | road | 22.4594 | -0.7656 | 22.8542 | 92.8583 | 13.4854 | -3089 | -3814 |
| vdd/vdd | RivalMeanLogit | vegetation | 47.0487 | 2.9061 | 97.0564 | 47.7298 | 11.3280 | 12159 | -4719 |
| vdd/vdd | RivalMeanLogit | vehicle | 48.5257 | -1.7827 | 48.7000 | 99.2676 | 0.7299 | -49 | 447 |
| vdd/vdd | RivalMeanLogit | roof | 86.0433 | -0.2174 | 86.9886 | 98.7528 | 6.8403 | -819 | -584 |
| vdd/vdd | RivalMeanLogit | water | 59.7076 | 6.8737 | 62.5794 | 92.8627 | 12.2447 | 3195 | -28968 |
| vdd/vdd | RivalShuffledSupport_Exact | other | 60.3890 | 0.6124 | 79.5123 | 71.5172 | 45.9219 | 22541 | 24718 |
| vdd/vdd | RivalShuffledSupport_Exact | wall | 60.3861 | 0.7045 | 66.5704 | 86.6669 | 10.3568 | 310 | -2307 |
| vdd/vdd | RivalShuffledSupport_Exact | road | 23.6221 | 0.3971 | 23.7486 | 97.7961 | 13.6677 | 348 | -3429 |
| vdd/vdd | RivalShuffledSupport_Exact | vegetation | 41.1812 | -2.9614 | 96.0810 | 41.8848 | 10.0417 | -16077 | -3459 |
| vdd/vdd | RivalShuffledSupport_Exact | vehicle | 58.1178 | 7.8094 | 58.5445 | 98.7617 | 0.6041 | -87 | -2154 |
| vdd/vdd | RivalShuffledSupport_Exact | roof | 89.8217 | 3.5610 | 90.3577 | 99.3439 | 6.6247 | -72 | -5853 |
| vdd/vdd | RivalShuffledSupport_Exact | water | 54.2513 | 1.4174 | 57.8735 | 89.6566 | 12.7832 | -2353 | -12126 |
| vdd/vdd | RivalTextOnly_Exact | other | 60.0618 | 0.2852 | 81.6482 | 69.4355 | 43.4189 | 252 | -5485 |
| vdd/vdd | RivalTextOnly_Exact | wall | 59.6897 | 0.0081 | 66.1791 | 85.8901 | 10.3246 | -986 | -1685 |
| vdd/vdd | RivalTextOnly_Exact | road | 23.1935 | -0.0315 | 23.3435 | 97.3048 | 13.8350 | 6 | 422 |
| vdd/vdd | RivalTextOnly_Exact | vegetation | 44.9564 | 0.8138 | 94.7205 | 46.1118 | 11.2139 | 4343 | 704 |
| vdd/vdd | RivalTextOnly_Exact | vehicle | 50.5831 | 0.2747 | 50.6035 | 99.9201 | 0.7071 | 0 | -81 |
| vdd/vdd | RivalTextOnly_Exact | roof | 85.5003 | -0.7604 | 85.9431 | 99.4009 | 6.9689 | 0 | 1295 |
| vdd/vdd | RivalTextOnly_Exact | water | 52.6077 | -0.2262 | 55.4939 | 91.0031 | 13.5315 | -23 | 1238 |
| vdd/vdd | RivalAliasShuffle0_Exact | other | 59.7170 | -0.0596 | 80.9550 | 69.4778 | 43.8173 | 704 | 2419 |
| vdd/vdd | RivalAliasShuffle0_Exact | wall | 59.6153 | -0.0663 | 65.5362 | 86.8395 | 10.5412 | 598 | 1272 |
| vdd/vdd | RivalAliasShuffle0_Exact | road | 23.4087 | 0.1837 | 23.5264 | 97.9068 | 13.8124 | 425 | -472 |
| vdd/vdd | RivalAliasShuffle0_Exact | vegetation | 44.5428 | 0.4002 | 94.8950 | 45.6363 | 11.0779 | 2046 | 148 |
| vdd/vdd | RivalAliasShuffle0_Exact | vehicle | 49.6561 | -0.6523 | 49.6627 | 99.9734 | 0.7209 | 4 | 204 |
| vdd/vdd | RivalAliasShuffle0_Exact | roof | 88.2542 | 1.9935 | 88.7968 | 99.3123 | 6.7389 | -112 | -3416 |
| vdd/vdd | RivalAliasShuffle0_Exact | water | 53.9869 | 1.1530 | 56.8249 | 91.5324 | 13.2915 | 893 | -4713 |
| vdd/vdd | RivalAliasShuffle1_Exact | other | 59.5330 | -0.2436 | 80.6154 | 69.4791 | 44.0027 | 718 | 6293 |
| vdd/vdd | RivalAliasShuffle1_Exact | wall | 59.6179 | -0.0637 | 65.6037 | 86.7268 | 10.5166 | 410 | 946 |
| vdd/vdd | RivalAliasShuffle1_Exact | road | 23.4317 | 0.2067 | 23.5528 | 97.8522 | 13.7892 | 387 | -920 |
| vdd/vdd | RivalAliasShuffle1_Exact | vegetation | 44.7412 | 0.5986 | 95.1456 | 45.7864 | 11.0850 | 2771 | -427 |
| vdd/vdd | RivalAliasShuffle1_Exact | vehicle | 50.8870 | 0.5786 | 50.9666 | 99.6937 | 0.7005 | -17 | -203 |
| vdd/vdd | RivalAliasShuffle1_Exact | roof | 87.7456 | 1.4849 | 88.2645 | 99.3345 | 6.7811 | -84 | -2560 |
| vdd/vdd | RivalAliasShuffle1_Exact | water | 54.5752 | 1.7413 | 57.5038 | 91.4648 | 13.1248 | 776 | -8090 |
| vdd/vdd | RivalAliasShuffle2_Exact | other | 59.6806 | -0.0960 | 80.9675 | 69.4192 | 43.7736 | 77 | 2129 |
| vdd/vdd | RivalAliasShuffle2_Exact | wall | 59.4588 | -0.2228 | 65.2977 | 86.9270 | 10.5903 | 744 | 2157 |
| vdd/vdd | RivalAliasShuffle2_Exact | road | 23.4133 | 0.1883 | 23.5313 | 97.9024 | 13.8089 | 422 | -542 |
| vdd/vdd | RivalAliasShuffle2_Exact | vegetation | 44.5196 | 0.3770 | 95.0536 | 45.5755 | 11.0446 | 1752 | -255 |
| vdd/vdd | RivalAliasShuffle2_Exact | vehicle | 51.4646 | 1.1562 | 51.4929 | 99.8935 | 0.6947 | -2 | -339 |
| vdd/vdd | RivalAliasShuffle2_Exact | roof | 87.9602 | 1.6995 | 88.5005 | 99.3107 | 6.7614 | -114 | -2943 |
| vdd/vdd | RivalAliasShuffle2_Exact | water | 53.5369 | 0.7030 | 56.4596 | 91.1834 | 13.3265 | 289 | -3375 |
| vdd/vdd | DirectionalMean_Exact | other | 59.9397 | 0.1631 | 80.7974 | 69.8969 | 44.1677 | 5192 | 5278 |
| vdd/vdd | DirectionalMean_Exact | wall | 59.9052 | 0.2236 | 65.6171 | 87.3125 | 10.5855 | 1387 | 1413 |
| vdd/vdd | DirectionalMean_Exact | road | 23.1128 | -0.1122 | 23.2304 | 97.8565 | 13.9812 | 390 | 3103 |
| vdd/vdd | DirectionalMean_Exact | vegetation | 44.3917 | 0.2491 | 95.2647 | 45.3933 | 10.9761 | 872 | -812 |
| vdd/vdd | DirectionalMean_Exact | vehicle | 52.8421 | 2.5337 | 52.8719 | 99.8935 | 0.6766 | -2 | -719 |
| vdd/vdd | DirectionalMean_Exact | roof | 88.9706 | 2.7099 | 89.5113 | 99.3257 | 6.6861 | -95 | -4542 |
| vdd/vdd | DirectionalMean_Exact | water | 53.9220 | 1.0881 | 57.3938 | 89.9132 | 12.9269 | -1909 | -9556 |
| vdd/vdd | DirectionalShuffle0_Exact | other | 59.9634 | 0.1868 | 80.8074 | 69.9217 | 44.1778 | 5457 | 5226 |
| vdd/vdd | DirectionalShuffle0_Exact | wall | 59.8672 | 0.1856 | 65.6125 | 87.2399 | 10.5774 | 1266 | 1365 |
| vdd/vdd | DirectionalShuffle0_Exact | road | 23.1220 | -0.1030 | 23.2400 | 97.8507 | 13.9746 | 386 | 2969 |
| vdd/vdd | DirectionalShuffle0_Exact | vegetation | 44.4365 | 0.2939 | 95.2790 | 45.4370 | 10.9850 | 1083 | -836 |
| vdd/vdd | DirectionalShuffle0_Exact | vehicle | 53.0514 | 2.7430 | 53.0814 | 99.8935 | 0.6739 | -2 | -775 |
| vdd/vdd | DirectionalShuffle0_Exact | roof | 88.9372 | 2.6765 | 89.4780 | 99.3250 | 6.6885 | -96 | -4490 |
| vdd/vdd | DirectionalShuffle0_Exact | water | 53.8920 | 1.0581 | 57.3804 | 89.8629 | 12.9227 | -1996 | -9557 |
| vdd/vdd | DirectionalShuffle1_Exact | other | 59.9404 | 0.1638 | 80.7907 | 69.9029 | 44.1751 | 5256 | 5370 |
| vdd/vdd | DirectionalShuffle1_Exact | wall | 59.9277 | 0.2461 | 65.6231 | 87.3496 | 10.5890 | 1449 | 1425 |
| vdd/vdd | DirectionalShuffle1_Exact | road | 23.1075 | -0.1175 | 23.2252 | 97.8550 | 13.9841 | 389 | 3166 |
| vdd/vdd | DirectionalShuffle1_Exact | vegetation | 44.3658 | 0.2232 | 95.2868 | 45.3612 | 10.9658 | 717 | -873 |
| vdd/vdd | DirectionalShuffle1_Exact | vehicle | 52.6641 | 2.3557 | 52.6937 | 99.8935 | 0.6789 | -2 | -671 |
| vdd/vdd | DirectionalShuffle1_Exact | roof | 89.0782 | 2.8175 | 89.6182 | 99.3281 | 6.6782 | -92 | -4709 |
| vdd/vdd | DirectionalShuffle1_Exact | water | 53.9017 | 1.0678 | 57.3765 | 89.8993 | 12.9288 | -1933 | -9492 |
| vdd/vdd | DirectionalShuffle2_Exact | other | 59.9161 | 0.1395 | 80.7944 | 69.8670 | 44.1504 | 4872 | 5236 |
| vdd/vdd | DirectionalShuffle2_Exact | wall | 59.8346 | 0.1530 | 65.5625 | 87.2591 | 10.5878 | 1298 | 1551 |
| vdd/vdd | DirectionalShuffle2_Exact | road | 23.1180 | -0.1070 | 23.2364 | 97.8435 | 13.9757 | 381 | 2998 |
| vdd/vdd | DirectionalShuffle2_Exact | vegetation | 44.3855 | 0.2429 | 95.2135 | 45.3985 | 10.9833 | 897 | -687 |
| vdd/vdd | DirectionalShuffle2_Exact | vehicle | 52.5711 | 2.2627 | 52.5932 | 99.9201 | 0.6804 | 0 | -642 |
| vdd/vdd | DirectionalShuffle2_Exact | roof | 88.8244 | 2.5637 | 89.3645 | 99.3242 | 6.6969 | -97 | -4312 |
| vdd/vdd | DirectionalShuffle2_Exact | water | 53.9340 | 1.1001 | 57.4046 | 89.9201 | 12.9255 | -1897 | -9598 |
| potsdam/potsdam | Geometry | impervious surface | 48.6703 | -17.2274 | 84.3816 | 53.4888 | 25.9112 | -176352 | -21321 |
| potsdam/potsdam | Geometry | building | 72.8222 | 1.7693 | 79.5088 | 89.6472 | 18.1787 | -11696 | -26826 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 12.1298 | 77.4031 | 28.7716 | 6.4458 | 50250 | 16882 |
| potsdam/potsdam | Geometry | tree | 49.3365 | -6.1699 | 84.7437 | 54.1457 | 11.2731 | -15594 | 17045 |
| potsdam/potsdam | Geometry | car | 11.6958 | -12.8420 | 11.6965 | 99.9455 | 30.6388 | 648 | 338941 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 3.0400 | 7.7773 | 13.2570 | 7.5523 | 3554 | -175531 |
| potsdam/potsdam | Anchored_Exact | impervious surface | 65.8977 | 0.0000 | 85.6706 | 74.0609 | 35.3370 | 0 | 0 |
| potsdam/potsdam | Anchored_Exact | building | 71.0529 | 0.0000 | 74.9985 | 93.1063 | 20.0156 | 0 | 0 |
| potsdam/potsdam | Anchored_Exact | low vegetation | 14.4124 | 0.0000 | 79.9195 | 14.9539 | 3.2447 | 0 | 0 |
| potsdam/potsdam | Anchored_Exact | tree | 55.5064 | 0.0000 | 91.9038 | 58.3601 | 11.2039 | 0 | 0 |
| potsdam/potsdam | Anchored_Exact | car | 24.5378 | 0.0000 | 24.5936 | 99.0837 | 14.4459 | 0 | 0 |
| potsdam/potsdam | Anchored_Exact | clutter | 2.1143 | 0.0000 | 2.6529 | 9.4321 | 15.7528 | 0 | 0 |
| potsdam/potsdam | ContrastReversal_Exact | impervious surface | 64.6878 | -1.2099 | 85.5242 | 72.6413 | 34.7190 | -12169 | -791 |
| potsdam/potsdam | ContrastReversal_Exact | building | 70.9088 | -0.1441 | 74.7567 | 93.2323 | 20.1075 | 426 | 1501 |
| potsdam/potsdam | ContrastReversal_Exact | low vegetation | 14.8765 | 0.4641 | 80.3578 | 15.4379 | 3.3314 | 1760 | 59 |
| potsdam/potsdam | ContrastReversal_Exact | tree | 55.3447 | -0.1617 | 92.0009 | 58.1426 | 11.1504 | -805 | -318 |
| potsdam/potsdam | ContrastReversal_Exact | car | 23.6344 | -0.9034 | 23.6846 | 99.1103 | 15.0043 | 20 | 11691 |
| potsdam/potsdam | ContrastReversal_Exact | clutter | 2.1050 | -0.0093 | 2.6439 | 9.3610 | 15.6873 | -66 | -1308 |
| potsdam/potsdam | RivalPreserving_Exact | impervious surface | 65.3609 | -0.5368 | 85.5730 | 73.4552 | 35.0880 | -5192 | -30 |
| potsdam/potsdam | RivalPreserving_Exact | building | 70.9269 | -0.1260 | 74.8455 | 93.1258 | 20.0607 | 66 | 880 |
| potsdam/potsdam | RivalPreserving_Exact | low vegetation | 15.0481 | 0.6357 | 80.4766 | 15.6183 | 3.3654 | 2416 | 115 |
| potsdam/potsdam | RivalPreserving_Exact | tree | 55.3930 | -0.1134 | 92.1003 | 58.1561 | 11.1409 | -755 | -566 |
| potsdam/potsdam | RivalPreserving_Exact | car | 24.1545 | -0.3833 | 24.2091 | 99.0757 | 14.6742 | -6 | 4793 |
| potsdam/potsdam | RivalPreserving_Exact | clutter | 2.1135 | -0.0008 | 2.6549 | 9.3901 | 15.6708 | -39 | -1682 |
| potsdam/potsdam | RivalMeanLogit | impervious surface | 64.0228 | -1.8749 | 84.6983 | 72.3966 | 34.9395 | -14267 | 5930 |
| potsdam/potsdam | RivalMeanLogit | building | 69.3205 | -1.7324 | 73.5792 | 92.2939 | 20.2237 | -2747 | 7110 |
| potsdam/potsdam | RivalMeanLogit | low vegetation | 16.9396 | 2.5272 | 81.7750 | 17.6042 | 3.7331 | 9638 | 604 |
| potsdam/potsdam | RivalMeanLogit | tree | 57.2170 | 1.7106 | 90.4938 | 60.8760 | 11.8690 | 9309 | 4639 |
| potsdam/potsdam | RivalMeanLogit | car | 23.4752 | -1.0626 | 23.5519 | 98.6302 | 15.0158 | -341 | 12292 |
| potsdam/potsdam | RivalMeanLogit | clutter | 2.2946 | 0.1803 | 2.9420 | 9.4418 | 14.2190 | 9 | -32176 |
| potsdam/potsdam | RivalShuffledSupport_Exact | impervious surface | 67.4022 | 1.5045 | 85.8049 | 75.8612 | 36.1393 | 15433 | 1393 |
| potsdam/potsdam | RivalShuffledSupport_Exact | building | 71.8352 | 0.7823 | 75.8135 | 93.1924 | 19.8187 | 291 | -4420 |
| potsdam/potsdam | RivalShuffledSupport_Exact | low vegetation | 11.5367 | -2.8757 | 78.7235 | 11.9080 | 2.6230 | -11077 | -1960 |
| potsdam/potsdam | RivalShuffledSupport_Exact | tree | 53.7724 | -1.7340 | 92.1170 | 56.3661 | 10.7961 | -7378 | -1175 |
| potsdam/potsdam | RivalShuffledSupport_Exact | car | 25.7907 | 1.2529 | 25.8608 | 98.9601 | 13.7209 | -93 | -15112 |
| potsdam/potsdam | RivalShuffledSupport_Exact | clutter | 2.0621 | -0.0522 | 2.5501 | 9.7280 | 16.9019 | 275 | 23823 |
| potsdam/potsdam | RivalTextOnly_Exact | impervious surface | 63.9693 | -1.9284 | 86.1131 | 71.3273 | 33.8579 | -23433 | -7587 |
| potsdam/potsdam | RivalTextOnly_Exact | building | 69.0491 | -2.0038 | 72.2133 | 94.0329 | 20.9945 | 3133 | 17395 |
| potsdam/potsdam | RivalTextOnly_Exact | low vegetation | 21.1640 | 6.7516 | 81.2526 | 22.2505 | 4.7487 | 26535 | 5006 |
| potsdam/potsdam | RivalTextOnly_Exact | tree | 59.4831 | 3.9767 | 91.0789 | 63.1632 | 12.2358 | 17772 | 3869 |
| potsdam/potsdam | RivalTextOnly_Exact | car | 22.9192 | -1.6186 | 22.9523 | 99.3736 | 15.5242 | 218 | 22395 |
| potsdam/potsdam | RivalTextOnly_Exact | clutter | 2.3215 | 0.2072 | 3.0642 | 8.7411 | 12.6390 | -642 | -64661 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | impervious surface | 65.1788 | -0.7189 | 85.5556 | 73.2380 | 34.9914 | -7054 | -195 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | building | 70.9539 | -0.0990 | 74.8870 | 93.1081 | 20.0458 | 6 | 627 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | low vegetation | 15.1799 | 0.7675 | 80.2297 | 15.7698 | 3.4085 | 2967 | 468 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | tree | 55.4675 | -0.0389 | 92.0591 | 58.2547 | 11.1648 | -390 | -430 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | car | 24.0090 | -0.5288 | 24.0607 | 99.1130 | 14.7702 | 22 | 6779 |
| potsdam/potsdam | RivalAliasShuffle0_Exact | clutter | 2.1195 | 0.0052 | 2.6642 | 9.3923 | 15.6193 | -37 | -2763 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | impervious surface | 65.5131 | -0.3846 | 85.6037 | 73.6247 | 35.1563 | -3739 | -50 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | building | 70.9215 | -0.1314 | 74.8380 | 93.1282 | 20.0633 | 74 | 925 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | low vegetation | 14.9976 | 0.5852 | 80.2360 | 15.5729 | 3.3657 | 2251 | 286 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | tree | 55.4818 | -0.0246 | 92.0722 | 58.2653 | 11.1652 | -351 | -460 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | car | 24.2859 | -0.2519 | 24.3394 | 99.1037 | 14.5998 | 15 | 3211 |
| potsdam/potsdam | RivalAliasShuffle1_Exact | clutter | 2.1122 | -0.0021 | 2.6542 | 9.3750 | 15.6497 | -53 | -2109 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | impervious surface | 65.6228 | -0.2749 | 85.6551 | 73.7252 | 35.1832 | -2877 | -348 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | building | 70.9930 | -0.0599 | 74.9101 | 93.1397 | 20.0464 | 113 | 533 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | low vegetation | 15.0454 | 0.6330 | 80.0166 | 15.6328 | 3.3879 | 2469 | 534 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | tree | 55.4430 | -0.0634 | 92.1065 | 58.2088 | 11.1503 | -560 | -565 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | car | 24.3552 | -0.1826 | 24.4068 | 99.1396 | 14.5647 | 42 | 2449 |
| potsdam/potsdam | RivalAliasShuffle2_Exact | clutter | 2.1217 | 0.0074 | 2.6652 | 9.4245 | 15.6675 | -7 | -1783 |
| potsdam/potsdam | DirectionalMean_Exact | impervious surface | 65.4475 | -0.4502 | 85.6680 | 73.4945 | 35.0678 | -4855 | -790 |
| potsdam/potsdam | DirectionalMean_Exact | building | 71.0369 | -0.0160 | 74.9388 | 93.1708 | 20.0454 | 218 | 407 |
| potsdam/potsdam | DirectionalMean_Exact | low vegetation | 15.0056 | 0.5932 | 79.8831 | 15.5949 | 3.3853 | 2331 | 618 |
| potsdam/potsdam | DirectionalMean_Exact | tree | 55.4988 | -0.0076 | 92.0771 | 58.2820 | 11.1679 | -289 | -467 |
| potsdam/potsdam | DirectionalMean_Exact | car | 24.1219 | -0.4159 | 24.1783 | 99.0438 | 14.6882 | -30 | 5110 |
| potsdam/potsdam | DirectionalMean_Exact | clutter | 2.1164 | 0.0021 | 2.6595 | 9.3912 | 15.6454 | -38 | -2215 |
| potsdam/potsdam | DirectionalShuffle0_Exact | impervious surface | 65.4179 | -0.4798 | 85.6691 | 73.4564 | 35.0492 | -5182 | -854 |
| potsdam/potsdam | DirectionalShuffle0_Exact | building | 71.0283 | -0.0246 | 74.9291 | 93.1711 | 20.0481 | 219 | 462 |
| potsdam/potsdam | DirectionalShuffle0_Exact | low vegetation | 15.0161 | 0.6037 | 79.8197 | 15.6086 | 3.3910 | 2381 | 687 |
| potsdam/potsdam | DirectionalShuffle0_Exact | tree | 55.5011 | -0.0053 | 92.0606 | 58.2912 | 11.1716 | -255 | -422 |
| potsdam/potsdam | DirectionalShuffle0_Exact | car | 24.1000 | -0.4378 | 24.1562 | 99.0438 | 14.7016 | -30 | 5391 |
| potsdam/potsdam | DirectionalShuffle0_Exact | clutter | 2.1194 | 0.0051 | 2.6634 | 9.4009 | 15.6385 | -29 | -2368 |
| potsdam/potsdam | DirectionalShuffle1_Exact | impervious surface | 65.4302 | -0.4675 | 85.6550 | 73.4823 | 35.0673 | -4960 | -696 |
| potsdam/potsdam | DirectionalShuffle1_Exact | building | 71.0353 | -0.0176 | 74.9379 | 93.1696 | 20.0454 | 214 | 411 |
| potsdam/potsdam | DirectionalShuffle1_Exact | low vegetation | 15.0261 | 0.6137 | 79.9310 | 15.6152 | 3.3877 | 2405 | 594 |
| potsdam/potsdam | DirectionalShuffle1_Exact | tree | 55.5159 | 0.0095 | 92.0898 | 58.2958 | 11.1690 | -238 | -495 |
| potsdam/potsdam | DirectionalShuffle1_Exact | car | 24.1219 | -0.4159 | 24.1777 | 99.0518 | 14.6897 | -24 | 5136 |
| potsdam/potsdam | DirectionalShuffle1_Exact | clutter | 2.1167 | 0.0024 | 2.6599 | 9.3901 | 15.6409 | -39 | -2308 |
| potsdam/potsdam | DirectionalShuffle2_Exact | impervious surface | 65.4077 | -0.4900 | 85.6825 | 73.4337 | 35.0329 | -5376 | -1001 |
| potsdam/potsdam | DirectionalShuffle2_Exact | building | 71.0451 | -0.0078 | 74.9388 | 93.1850 | 20.0485 | 266 | 423 |
| potsdam/potsdam | DirectionalShuffle2_Exact | low vegetation | 15.0398 | 0.6274 | 79.8991 | 15.6312 | 3.3925 | 2463 | 637 |
| potsdam/potsdam | DirectionalShuffle2_Exact | tree | 55.4965 | -0.0099 | 92.0944 | 58.2725 | 11.1639 | -324 | -514 |
| potsdam/potsdam | DirectionalShuffle2_Exact | car | 24.0786 | -0.4592 | 24.1343 | 99.0492 | 14.7157 | -26 | 5683 |
| potsdam/potsdam | DirectionalShuffle2_Exact | clutter | 2.1171 | 0.0028 | 2.6602 | 9.3944 | 15.6465 | -35 | -2196 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | 13.9675 | 89.6338 | 64.7597 | 2.2305 | 11032 | 2834 |
| udd5/udd5 | Geometry | building | 83.4755 | -0.0116 | 86.9865 | 95.3877 | 78.8720 | -65859 | -78647 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 | 0 | 39991 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -1.6538 | 3.7107 | 99.9773 | 11.3160 | 0 | 73162 |
| udd5/udd5 | Geometry | other | 5.0713 | -0.6763 | 23.0553 | 6.1044 | 5.3797 | -2330 | 19817 |
| udd5/udd5 | Anchored_Exact | vegetation | 46.2799 | 0.0000 | 93.8774 | 47.7203 | 1.5693 | 0 | 0 |
| udd5/udd5 | Anchored_Exact | building | 83.4871 | 0.0000 | 83.6594 | 99.7539 | 85.7625 | 0 | 0 |
| udd5/udd5 | Anchored_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 | 0 | 0 |
| udd5/udd5 | Anchored_Exact | vehicle | 5.3645 | 0.0000 | 5.3645 | 99.9773 | 7.8274 | 0 | 0 |
| udd5/udd5 | Anchored_Exact | other | 5.7476 | 0.0000 | 29.7284 | 6.6513 | 4.5458 | 0 | 0 |
| udd5/udd5 | ContrastReversal_Exact | vegetation | 50.0302 | 3.7503 | 91.6242 | 52.4280 | 1.7665 | 3048 | 1088 |
| udd5/udd5 | ContrastReversal_Exact | building | 83.6643 | 0.1772 | 84.0005 | 99.5239 | 85.2174 | -3469 | -7964 |
| udd5/udd5 | ContrastReversal_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3406 | 0 | 958 |
| udd5/udd5 | ContrastReversal_Exact | vehicle | 7.0366 | 1.6721 | 7.0367 | 99.9773 | 5.9673 | 0 | -39009 |
| udd5/udd5 | ContrastReversal_Exact | other | 10.1611 | 4.4135 | 37.1614 | 12.2692 | 6.7082 | 23938 | 21410 |
| udd5/udd5 | RivalPreserving_Exact | vegetation | 49.9130 | 3.6331 | 91.6227 | 52.2998 | 1.7622 | 2965 | 1081 |
| udd5/udd5 | RivalPreserving_Exact | building | 83.6748 | 0.1877 | 83.9622 | 99.5925 | 85.3149 | -2435 | -6953 |
| udd5/udd5 | RivalPreserving_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2646 | 0 | -635 |
| udd5/udd5 | RivalPreserving_Exact | vehicle | 6.7660 | 1.4015 | 6.7661 | 99.9773 | 6.2059 | 0 | -34004 |
| udd5/udd5 | RivalPreserving_Exact | other | 9.3588 | 3.6112 | 35.5063 | 11.2755 | 6.4523 | 19704 | 20277 |
| udd5/udd5 | RivalMeanLogit | vegetation | 50.3379 | 4.0580 | 89.8448 | 53.3748 | 1.8341 | 3661 | 1891 |
| udd5/udd5 | RivalMeanLogit | building | 84.3217 | 0.8346 | 84.6340 | 99.5643 | 84.6138 | -2860 | -21232 |
| udd5/udd5 | RivalMeanLogit | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4199 | 0 | 2621 |
| udd5/udd5 | RivalMeanLogit | vehicle | 6.2580 | 0.8935 | 6.2581 | 99.9773 | 6.7097 | 0 | -23439 |
| udd5/udd5 | RivalMeanLogit | other | 8.9522 | 3.2046 | 34.2102 | 10.8139 | 6.4226 | 17737 | 21621 |
| udd5/udd5 | RivalShuffledSupport_Exact | vegetation | 45.2940 | -0.9859 | 95.0473 | 46.3889 | 1.5068 | -862 | -450 |
| udd5/udd5 | RivalShuffledSupport_Exact | building | 84.0440 | 0.5569 | 84.3444 | 99.5780 | 84.9160 | -2653 | -15100 |
| udd5/udd5 | RivalShuffledSupport_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2964 | 0 | 31 |
| udd5/udd5 | RivalShuffledSupport_Exact | vehicle | 7.8017 | 2.4372 | 7.8018 | 99.9773 | 5.3821 | 0 | -51281 |
| udd5/udd5 | RivalShuffledSupport_Exact | other | 12.6867 | 6.9391 | 40.2184 | 15.6351 | 7.8987 | 38280 | 32035 |
| udd5/udd5 | RivalTextOnly_Exact | vegetation | 47.4659 | 1.1860 | 93.3046 | 49.1397 | 1.6259 | 919 | 268 |
| udd5/udd5 | RivalTextOnly_Exact | building | 83.2201 | -0.2670 | 83.3391 | 99.8288 | 86.1567 | 1129 | 7138 |
| udd5/udd5 | RivalTextOnly_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3479 | 0 | 1112 |
| udd5/udd5 | RivalTextOnly_Exact | vehicle | 5.0456 | -0.3189 | 5.0457 | 99.9773 | 8.3220 | 0 | 10374 |
| udd5/udd5 | RivalTextOnly_Exact | other | 4.4354 | -1.3122 | 28.5726 | 4.9885 | 3.5473 | -7085 | -13855 |
| udd5/udd5 | RivalAliasShuffle0_Exact | vegetation | 48.1447 | 1.8648 | 92.7700 | 50.0216 | 1.6646 | 1490 | 509 |
| udd5/udd5 | RivalAliasShuffle0_Exact | building | 83.7292 | 0.2421 | 83.9326 | 99.7114 | 85.4470 | -641 | -5977 |
| udd5/udd5 | RivalAliasShuffle0_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2242 | 0 | -1484 |
| udd5/udd5 | RivalAliasShuffle0_Exact | vehicle | 4.9983 | -0.3662 | 4.9984 | 99.9773 | 8.4008 | 0 | 12026 |
| udd5/udd5 | RivalAliasShuffle0_Exact | other | 6.1832 | 0.4356 | 33.5745 | 7.0451 | 4.2634 | 1678 | -7601 |
| udd5/udd5 | RivalAliasShuffle1_Exact | vegetation | 49.4692 | 3.1893 | 91.8197 | 51.7500 | 1.7400 | 2609 | 970 |
| udd5/udd5 | RivalAliasShuffle1_Exact | building | 83.6527 | 0.1656 | 83.8652 | 99.6980 | 85.5042 | -843 | -4576 |
| udd5/udd5 | RivalAliasShuffle1_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3238 | 0 | 606 |
| udd5/udd5 | RivalAliasShuffle1_Exact | vehicle | 6.4233 | 1.0588 | 6.4234 | 99.9773 | 6.5371 | 0 | -27060 |
| udd5/udd5 | RivalAliasShuffle1_Exact | other | 7.4960 | 1.7484 | 31.0078 | 8.9965 | 5.8950 | 9993 | 18301 |
| udd5/udd5 | RivalAliasShuffle2_Exact | vegetation | 48.3663 | 2.0864 | 92.7905 | 50.2548 | 1.6720 | 1641 | 513 |
| udd5/udd5 | RivalAliasShuffle2_Exact | building | 83.7173 | 0.2302 | 83.9194 | 99.7131 | 85.4619 | -615 | -5691 |
| udd5/udd5 | RivalAliasShuffle2_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2707 | 0 | -507 |
| udd5/udd5 | RivalAliasShuffle2_Exact | vehicle | 5.1549 | -0.2096 | 5.1549 | 99.9773 | 8.1457 | 0 | 6675 |
| udd5/udd5 | RivalAliasShuffle2_Exact | other | 5.8296 | 0.0820 | 30.6611 | 6.7149 | 4.4497 | 271 | -2287 |
| udd5/udd5 | DirectionalMean_Exact | vegetation | 45.6545 | -0.6254 | 94.2528 | 46.9619 | 1.5382 | -491 | -161 |
| udd5/udd5 | DirectionalMean_Exact | building | 83.8551 | 0.3680 | 84.1224 | 99.6225 | 85.1781 | -1982 | -10274 |
| udd5/udd5 | DirectionalMean_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2845 | 0 | -219 |
| udd5/udd5 | DirectionalMean_Exact | vehicle | 6.4461 | 1.0816 | 6.4462 | 99.9773 | 6.5139 | 0 | -27545 |
| udd5/udd5 | DirectionalMean_Exact | other | 9.1263 | 3.3787 | 34.5642 | 11.0324 | 6.4852 | 18668 | 22004 |
| udd5/udd5 | DirectionalShuffle0_Exact | vegetation | 45.5577 | -0.7222 | 94.3696 | 46.8306 | 1.5320 | -576 | -206 |
| udd5/udd5 | DirectionalShuffle0_Exact | building | 83.8479 | 0.3608 | 84.1160 | 99.6213 | 85.1836 | -2000 | -10141 |
| udd5/udd5 | DirectionalShuffle0_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2896 | 0 | -111 |
| udd5/udd5 | DirectionalShuffle0_Exact | vehicle | 6.4787 | 1.1142 | 6.4788 | 99.9773 | 6.4812 | 0 | -28231 |
| udd5/udd5 | DirectionalShuffle0_Exact | other | 9.2442 | 3.4966 | 34.8578 | 11.1746 | 6.5135 | 19274 | 21991 |
| udd5/udd5 | DirectionalShuffle1_Exact | vegetation | 45.5487 | -0.7312 | 94.1748 | 46.8692 | 1.5365 | -551 | -138 |
| udd5/udd5 | DirectionalShuffle1_Exact | building | 83.8285 | 0.3414 | 84.0948 | 99.6236 | 85.2071 | -1965 | -9684 |
| udd5/udd5 | DirectionalShuffle1_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2892 | 0 | -119 |
| udd5/udd5 | DirectionalShuffle1_Exact | vehicle | 6.4528 | 1.0883 | 6.4529 | 99.9773 | 6.5072 | 0 | -27686 |
| udd5/udd5 | DirectionalShuffle1_Exact | other | 9.1454 | 3.3978 | 34.7331 | 11.0432 | 6.4600 | 18714 | 21429 |
| udd5/udd5 | DirectionalShuffle2_Exact | vegetation | 45.5244 | -0.7555 | 94.4086 | 46.7858 | 1.5299 | -605 | -221 |
| udd5/udd5 | DirectionalShuffle2_Exact | building | 83.8608 | 0.3737 | 84.1275 | 99.6234 | 85.1737 | -1969 | -10380 |
| udd5/udd5 | DirectionalShuffle2_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2986 | 0 | 77 |
| udd5/udd5 | DirectionalShuffle2_Exact | vehicle | 6.4596 | 1.0951 | 6.4597 | 99.9773 | 6.5003 | 0 | -27830 |
| udd5/udd5 | DirectionalShuffle2_Exact | other | 9.1340 | 3.3864 | 34.5418 | 11.0460 | 6.4974 | 18726 | 22202 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 | 0 | 14184 |
| oem/oem | Geometry | rangeland | 41.5417 | -7.9844 | 54.4153 | 63.7145 | 17.0320 | 3770 | 81071 |
| oem/oem | Geometry | developed space | 25.6980 | 1.9311 | 65.5490 | 29.7108 | 8.8828 | -32737 | -175342 |
| oem/oem | Geometry | road | 52.3963 | -2.1566 | 60.6188 | 79.4359 | 8.2389 | 7878 | 22137 |
| oem/oem | Geometry | tree | 65.2556 | 9.5951 | 84.8180 | 73.8857 | 23.0134 | 82580 | 43050 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 | 0 | 1153 |
| oem/oem | Geometry | agriculture land | 71.2323 | -10.4177 | 95.7971 | 73.5302 | 16.2828 | -112388 | -80583 |
| oem/oem | Geometry | building | 62.4408 | 15.4121 | 64.4677 | 95.2061 | 17.5919 | 85337 | 59890 |
| oem/oem | Anchored_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 | 0 | 0 |
| oem/oem | Anchored_Exact | rangeland | 49.5261 | 0.0000 | 70.5348 | 62.4456 | 12.8780 | 0 | 0 |
| oem/oem | Anchored_Exact | developed space | 23.7669 | 0.0000 | 38.9362 | 37.8898 | 19.0710 | 0 | 0 |
| oem/oem | Anchored_Exact | road | 54.5529 | 0.0000 | 68.0810 | 73.3007 | 6.7693 | 0 | 0 |
| oem/oem | Anchored_Exact | tree | 55.6605 | 0.0000 | 91.7802 | 58.5807 | 16.8622 | 0 | 0 |
| oem/oem | Anchored_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | Anchored_Exact | agriculture land | 81.6500 | 0.0000 | 82.0066 | 99.4703 | 25.7312 | 0 | 0 |
| oem/oem | Anchored_Exact | building | 47.0287 | 0.0000 | 68.3392 | 60.1297 | 10.4811 | 0 | 0 |
| oem/oem | ContrastReversal_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.6803 | 0 | -10762 |
| oem/oem | ContrastReversal_Exact | rangeland | 49.2046 | -0.3215 | 70.4958 | 61.9652 | 12.7860 | -1427 | -452 |
| oem/oem | ContrastReversal_Exact | developed space | 25.4634 | 1.6965 | 40.5017 | 40.6806 | 19.6842 | 11170 | 1354 |
| oem/oem | ContrastReversal_Exact | road | 54.5099 | -0.0430 | 68.0543 | 73.2540 | 6.7676 | -60 | 26 |
| oem/oem | ContrastReversal_Exact | tree | 55.8490 | 0.1885 | 91.6079 | 58.8604 | 16.9746 | 1509 | 786 |
| oem/oem | ContrastReversal_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | ContrastReversal_Exact | agriculture land | 81.9118 | 0.2618 | 82.2821 | 99.4537 | 25.6408 | -72 | -1775 |
| oem/oem | ContrastReversal_Exact | building | 46.9238 | -0.1049 | 68.2859 | 59.9994 | 10.4666 | -317 | 20 |
| oem/oem | RivalPreserving_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.0749 | 0 | -2703 |
| oem/oem | RivalPreserving_Exact | rangeland | 49.4588 | -0.0673 | 70.5939 | 62.2924 | 12.8356 | -455 | -410 |
| oem/oem | RivalPreserving_Exact | developed space | 24.2414 | 0.4745 | 39.4337 | 38.6209 | 19.1937 | 2926 | -420 |
| oem/oem | RivalPreserving_Exact | road | 54.5205 | -0.0324 | 68.0640 | 73.2618 | 6.7674 | -50 | 11 |
| oem/oem | RivalPreserving_Exact | tree | 55.8248 | 0.1643 | 91.6968 | 58.7970 | 16.9399 | 1167 | 419 |
| oem/oem | RivalPreserving_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalPreserving_Exact | agriculture land | 81.7723 | 0.1223 | 82.1375 | 99.4592 | 25.6873 | -48 | -848 |
| oem/oem | RivalPreserving_Exact | building | 47.1585 | 0.1298 | 68.3976 | 60.2966 | 10.5013 | 406 | 5 |
| oem/oem | RivalMeanLogit | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1861 | 0 | -432 |
| oem/oem | RivalMeanLogit | rangeland | 46.3584 | -3.1677 | 69.3826 | 58.2811 | 12.2187 | -12372 | -1092 |
| oem/oem | RivalMeanLogit | developed space | 25.4830 | 1.7161 | 41.2871 | 39.9660 | 18.9706 | 8310 | -10361 |
| oem/oem | RivalMeanLogit | road | 51.0560 | -3.4969 | 61.7404 | 74.6854 | 7.6055 | 1778 | 15300 |
| oem/oem | RivalMeanLogit | tree | 55.2582 | -0.4023 | 90.4379 | 58.6869 | 17.1435 | 573 | 5172 |
| oem/oem | RivalMeanLogit | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalMeanLogit | agriculture land | 82.9098 | 1.2598 | 83.3417 | 99.3789 | 25.2957 | -396 | -8498 |
| oem/oem | RivalMeanLogit | building | 48.8359 | 1.8072 | 69.7553 | 61.9543 | 10.5799 | 4439 | -2421 |
| oem/oem | RivalShuffledSupport_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3653 | 0 | 3228 |
| oem/oem | RivalShuffledSupport_Exact | rangeland | 49.0738 | -0.4523 | 70.5394 | 61.7246 | 12.7284 | -2142 | -912 |
| oem/oem | RivalShuffledSupport_Exact | developed space | 23.3835 | -0.3834 | 37.6771 | 38.1332 | 19.8349 | 974 | 14627 |
| oem/oem | RivalShuffledSupport_Exact | road | 54.3697 | -0.1832 | 68.1582 | 72.8818 | 6.7229 | -538 | -408 |
| oem/oem | RivalShuffledSupport_Exact | tree | 55.1754 | -0.4851 | 91.8507 | 58.0155 | 16.6867 | -3050 | -535 |
| oem/oem | RivalShuffledSupport_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalShuffledSupport_Exact | agriculture land | 82.3835 | 0.7335 | 82.8096 | 99.3794 | 25.4584 | -394 | -5178 |
| oem/oem | RivalShuffledSupport_Exact | building | 45.3572 | -1.6715 | 67.6334 | 57.9319 | 10.2034 | -5347 | -325 |
| oem/oem | RivalTextOnly_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.0859 | 0 | -43324 |
| oem/oem | RivalTextOnly_Exact | rangeland | 49.4510 | -0.0751 | 70.5038 | 62.3503 | 12.8640 | -283 | -3 |
| oem/oem | RivalTextOnly_Exact | developed space | 29.6961 | 5.9292 | 44.1600 | 47.5522 | 21.1030 | 38674 | 2828 |
| oem/oem | RivalTextOnly_Exact | road | 54.5042 | -0.0487 | 67.9957 | 73.3116 | 6.7788 | 14 | 180 |
| oem/oem | RivalTextOnly_Exact | tree | 55.6117 | -0.0488 | 91.7589 | 58.5353 | 16.8531 | -245 | 58 |
| oem/oem | RivalTextOnly_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalTextOnly_Exact | agriculture land | 81.1538 | -0.4962 | 81.4895 | 99.4950 | 25.9009 | 107 | 3359 |
| oem/oem | RivalTextOnly_Exact | building | 46.8945 | -0.1342 | 68.4393 | 59.8338 | 10.4143 | -720 | -645 |
| oem/oem | RivalAliasShuffle0_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1443 | 0 | -1284 |
| oem/oem | RivalAliasShuffle0_Exact | rangeland | 49.4099 | -0.1162 | 70.5552 | 62.2449 | 12.8329 | -596 | -325 |
| oem/oem | RivalAliasShuffle0_Exact | developed space | 24.0007 | 0.2338 | 39.1894 | 38.2434 | 19.1246 | 1415 | -321 |
| oem/oem | RivalAliasShuffle0_Exact | road | 54.5459 | -0.0070 | 68.0686 | 73.3023 | 6.7706 | 2 | 26 |
| oem/oem | RivalAliasShuffle0_Exact | tree | 55.7999 | 0.1394 | 91.6979 | 58.7689 | 16.9316 | 1015 | 401 |
| oem/oem | RivalAliasShuffle0_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalAliasShuffle0_Exact | agriculture land | 81.8311 | 0.1811 | 82.1948 | 99.4622 | 25.6702 | -35 | -1211 |
| oem/oem | RivalAliasShuffle0_Exact | building | 47.1864 | 0.1577 | 68.3402 | 60.3870 | 10.5258 | 626 | 287 |
| oem/oem | RivalAliasShuffle1_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1965 | 0 | -218 |
| oem/oem | RivalAliasShuffle1_Exact | rangeland | 49.3786 | -0.1475 | 70.5954 | 62.1642 | 12.8089 | -836 | -574 |
| oem/oem | RivalAliasShuffle1_Exact | developed space | 23.8504 | 0.0835 | 39.0064 | 38.0355 | 19.1098 | 583 | 210 |
| oem/oem | RivalAliasShuffle1_Exact | road | 54.5301 | -0.0228 | 68.0690 | 73.2735 | 6.7679 | -35 | 8 |
| oem/oem | RivalAliasShuffle1_Exact | tree | 55.8230 | 0.1625 | 91.6746 | 58.8041 | 16.9460 | 1205 | 506 |
| oem/oem | RivalAliasShuffle1_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalAliasShuffle1_Exact | agriculture land | 81.8621 | 0.2121 | 82.2272 | 99.4606 | 25.6597 | -42 | -1419 |
| oem/oem | RivalAliasShuffle1_Exact | building | 47.0893 | 0.0606 | 68.2953 | 60.2629 | 10.5111 | 324 | 288 |
| oem/oem | RivalAliasShuffle2_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1055 | 0 | -2077 |
| oem/oem | RivalAliasShuffle2_Exact | rangeland | 49.3930 | -0.1331 | 70.5939 | 62.1881 | 12.8141 | -765 | -539 |
| oem/oem | RivalAliasShuffle2_Exact | developed space | 24.1844 | 0.4175 | 39.4191 | 38.4902 | 19.1358 | 2403 | -1079 |
| oem/oem | RivalAliasShuffle2_Exact | road | 54.5172 | -0.0357 | 68.0294 | 73.2961 | 6.7740 | -6 | 102 |
| oem/oem | RivalAliasShuffle2_Exact | tree | 55.8803 | 0.2198 | 91.6626 | 58.8726 | 16.9680 | 1575 | 585 |
| oem/oem | RivalAliasShuffle2_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | RivalAliasShuffle2_Exact | agriculture land | 81.7837 | 0.1337 | 82.1500 | 99.4578 | 25.6831 | -54 | -929 |
| oem/oem | RivalAliasShuffle2_Exact | building | 47.1408 | 0.1121 | 68.3170 | 60.3303 | 10.5195 | 488 | 296 |
| oem/oem | DirectionalMean_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1359 | 0 | -1456 |
| oem/oem | DirectionalMean_Exact | rangeland | 49.3926 | -0.1335 | 70.5407 | 62.2288 | 12.8322 | -644 | -291 |
| oem/oem | DirectionalMean_Exact | developed space | 24.0041 | 0.2372 | 39.1559 | 38.2838 | 19.1612 | 1577 | 265 |
| oem/oem | DirectionalMean_Exact | road | 54.5174 | -0.0355 | 68.0921 | 73.2236 | 6.7610 | -99 | -69 |
| oem/oem | DirectionalMean_Exact | tree | 55.7482 | 0.0877 | 91.6873 | 58.7158 | 16.9182 | 729 | 415 |
| oem/oem | DirectionalMean_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | DirectionalMean_Exact | agriculture land | 81.7311 | 0.0811 | 82.0935 | 99.4627 | 25.7020 | -33 | -564 |
| oem/oem | DirectionalMean_Exact | building | 47.0673 | 0.0386 | 68.3485 | 60.1856 | 10.4895 | 136 | 34 |
| oem/oem | DirectionalShuffle0_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1426 | 0 | -1320 |
| oem/oem | DirectionalShuffle0_Exact | rangeland | 49.3900 | -0.1361 | 70.5402 | 62.2251 | 12.8315 | -655 | -294 |
| oem/oem | DirectionalShuffle0_Exact | developed space | 23.9908 | 0.2239 | 39.1467 | 38.2589 | 19.1531 | 1477 | 201 |
| oem/oem | DirectionalShuffle0_Exact | road | 54.5148 | -0.0381 | 68.0915 | 73.2197 | 6.7607 | -104 | -70 |
| oem/oem | DirectionalShuffle0_Exact | tree | 55.7534 | 0.0929 | 91.6898 | 58.7207 | 16.9192 | 755 | 408 |
| oem/oem | DirectionalShuffle0_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | DirectionalShuffle0_Exact | agriculture land | 81.7267 | 0.0767 | 82.0894 | 99.4622 | 25.7032 | -35 | -538 |
| oem/oem | DirectionalShuffle0_Exact | building | 47.0699 | 0.0412 | 68.3502 | 60.1885 | 10.4897 | 143 | 32 |
| oem/oem | DirectionalShuffle1_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1385 | 0 | -1403 |
| oem/oem | DirectionalShuffle1_Exact | rangeland | 49.4079 | -0.1182 | 70.5473 | 62.2480 | 12.8349 | -587 | -292 |
| oem/oem | DirectionalShuffle1_Exact | developed space | 24.0053 | 0.2384 | 39.1513 | 38.2913 | 19.1672 | 1607 | 357 |
| oem/oem | DirectionalShuffle1_Exact | road | 54.5254 | -0.0275 | 68.0986 | 73.2306 | 6.7610 | -90 | -78 |
| oem/oem | DirectionalShuffle1_Exact | tree | 55.7397 | 0.0792 | 91.6829 | 58.7083 | 16.9169 | 688 | 428 |
| oem/oem | DirectionalShuffle1_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | DirectionalShuffle1_Exact | agriculture land | 81.7464 | 0.0964 | 82.1093 | 99.4622 | 25.6969 | -35 | -665 |
| oem/oem | DirectionalShuffle1_Exact | building | 47.0692 | 0.0405 | 68.3673 | 60.1741 | 10.4846 | 108 | -38 |
| oem/oem | DirectionalShuffle2_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1318 | 0 | -1540 |
| oem/oem | DirectionalShuffle2_Exact | rangeland | 49.3989 | -0.1272 | 70.5444 | 62.2359 | 12.8330 | -623 | -296 |
| oem/oem | DirectionalShuffle2_Exact | developed space | 24.0178 | 0.2509 | 39.1757 | 38.2998 | 19.1595 | 1641 | 166 |
| oem/oem | DirectionalShuffle2_Exact | road | 54.5252 | -0.0277 | 68.0908 | 73.2392 | 6.7626 | -79 | -57 |
| oem/oem | DirectionalShuffle2_Exact | tree | 55.7557 | 0.0952 | 91.6919 | 58.7223 | 16.9193 | 764 | 401 |
| oem/oem | DirectionalShuffle2_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | DirectionalShuffle2_Exact | agriculture land | 81.7248 | 0.0748 | 82.0868 | 99.4634 | 25.7043 | -30 | -520 |
| oem/oem | DirectionalShuffle2_Exact | building | 47.0867 | 0.0580 | 68.3671 | 60.2029 | 10.4896 | 178 | -5 |
| loveda/P | Geometry | building | 7.6311 | -53.2168 | 7.6311 | 100.0000 | 0.3472 | 63 | 3622 |
| loveda/P | Geometry | road | 63.0522 | 9.2969 | 64.4734 | 96.6222 | 12.4720 | -171 | -25873 |
| loveda/P | Geometry | water | 63.2131 | 0.8040 | 68.9718 | 88.3328 | 18.8451 | 314 | -2566 |
| loveda/P | Geometry | barren | 1.9144 | -2.4071 | 34.2731 | 1.9874 | 0.6429 | -3026 | 4258 |
| loveda/P | Geometry | tree | 72.0056 | 32.0503 | 88.8915 | 79.1256 | 14.6560 | 74529 | 18377 |
| loveda/P | Geometry | farm | 89.4231 | 6.4729 | 91.1653 | 97.9076 | 53.0368 | -9359 | -60168 |
| loveda/P | Anchored_Exact | building | 60.8479 | 0.0000 | 72.1893 | 79.4788 | 0.0292 | 0 | 0 |
| loveda/P | Anchored_Exact | road | 53.7553 | 0.0000 | 54.7280 | 96.7995 | 14.7198 | 0 | 0 |
| loveda/P | Anchored_Exact | water | 62.4091 | 0.0000 | 68.1253 | 88.1486 | 19.0394 | 0 | 0 |
| loveda/P | Anchored_Exact | barren | 4.3215 | 0.0000 | 89.7378 | 4.3430 | 0.5366 | 0 | 0 |
| loveda/P | Anchored_Exact | tree | 39.9553 | 0.0000 | 99.3680 | 40.0571 | 6.6373 | 0 | 0 |
| loveda/P | Anchored_Exact | farm | 82.9502 | 0.0000 | 83.2671 | 99.5433 | 59.0377 | 0 | 0 |
| loveda/P | ContrastReversal_Exact | building | 74.7899 | 13.9420 | 84.2271 | 86.9707 | 0.0274 | 23 | -44 |
| loveda/P | ContrastReversal_Exact | road | 53.4658 | -0.2895 | 54.4277 | 96.8006 | 14.8012 | 1 | 942 |
| loveda/P | ContrastReversal_Exact | water | 62.2578 | -0.1513 | 68.0276 | 88.0102 | 19.0369 | -236 | 206 |
| loveda/P | ContrastReversal_Exact | barren | 3.9198 | -0.4017 | 89.7764 | 3.9374 | 0.4863 | -521 | -62 |
| loveda/P | ContrastReversal_Exact | tree | 41.1725 | 1.2172 | 99.2829 | 41.2953 | 6.8483 | 2362 | 83 |
| loveda/P | ContrastReversal_Exact | farm | 83.3073 | 0.3571 | 83.6163 | 99.5584 | 58.8000 | 86 | -2840 |
| loveda/P | RivalPreserving_Exact | building | 67.2727 | 6.4248 | 76.8546 | 84.3648 | 0.0291 | 15 | -16 |
| loveda/P | RivalPreserving_Exact | road | 53.7961 | 0.0408 | 54.7739 | 96.7881 | 14.7058 | -11 | -152 |
| loveda/P | RivalPreserving_Exact | water | 62.3134 | -0.0957 | 68.0498 | 88.0841 | 19.0466 | -110 | 193 |
| loveda/P | RivalPreserving_Exact | barren | 3.9777 | -0.3438 | 89.3802 | 3.9965 | 0.4958 | -445 | -28 |
| loveda/P | RivalPreserving_Exact | tree | 41.0091 | 1.0538 | 99.3176 | 41.1249 | 6.8177 | 2037 | 53 |
| loveda/P | RivalPreserving_Exact | farm | 83.1482 | 0.1980 | 83.4611 | 99.5512 | 58.9051 | 45 | -1581 |
| loveda/P | RivalMeanLogit | building | 10.4235 | -50.4244 | 100.0000 | 10.4235 | 0.0028 | -212 | -94 |
| loveda/P | RivalMeanLogit | road | 56.0839 | 2.3286 | 57.1107 | 96.8939 | 14.1195 | 91 | -7047 |
| loveda/P | RivalMeanLogit | water | 61.8438 | -0.5653 | 67.0753 | 88.8008 | 19.4806 | 1112 | 3999 |
| loveda/P | RivalMeanLogit | barren | 5.4093 | 1.0878 | 64.3981 | 5.5760 | 0.9600 | 1584 | 3322 |
| loveda/P | RivalMeanLogit | tree | 43.8795 | 3.9242 | 99.2718 | 44.0212 | 7.3012 | 7562 | 130 |
| loveda/P | RivalMeanLogit | farm | 84.2217 | 1.2715 | 84.5530 | 99.5369 | 58.1360 | -37 | -10410 |
| loveda/P | RivalShuffledSupport_Exact | building | 69.3023 | 8.4544 | 70.7838 | 97.0684 | 0.0363 | 54 | 29 |
| loveda/P | RivalShuffledSupport_Exact | road | 52.7314 | -1.0239 | 53.6648 | 96.8068 | 15.0126 | 7 | 3385 |
| loveda/P | RivalShuffledSupport_Exact | water | 61.7043 | -0.7048 | 67.3351 | 88.0653 | 19.2447 | -142 | 2520 |
| loveda/P | RivalShuffledSupport_Exact | barren | 5.1963 | 0.8748 | 85.0095 | 5.2444 | 0.6840 | 1158 | 550 |
| loveda/P | RivalShuffledSupport_Exact | tree | 40.8501 | 0.8948 | 98.5076 | 41.1045 | 6.8703 | 1998 | 702 |
| loveda/P | RivalShuffledSupport_Exact | farm | 84.1580 | 1.2078 | 84.5076 | 99.5108 | 58.1521 | -186 | -10075 |
| loveda/P | RivalTextOnly_Exact | building | 61.6368 | 0.7889 | 74.1538 | 78.5016 | 0.0281 | -3 | -10 |
| loveda/P | RivalTextOnly_Exact | road | 53.6451 | -0.1102 | 54.6118 | 96.8057 | 14.7521 | 6 | 368 |
| loveda/P | RivalTextOnly_Exact | water | 62.4081 | -0.0010 | 68.1782 | 88.0583 | 19.0052 | -154 | -243 |
| loveda/P | RivalTextOnly_Exact | barren | 4.2418 | -0.0797 | 89.6236 | 4.2628 | 0.5274 | -103 | -4 |
| loveda/P | RivalTextOnly_Exact | tree | 39.6943 | -0.2610 | 99.3989 | 39.7898 | 6.5909 | -510 | -27 |
| loveda/P | RivalTextOnly_Exact | farm | 82.8727 | -0.0775 | 83.1869 | 99.5463 | 59.0964 | 17 | 663 |
| loveda/P | RivalAliasShuffle0_Exact | building | 66.7494 | 5.9015 | 73.6986 | 87.6221 | 0.0315 | 25 | 2 |
| loveda/P | RivalAliasShuffle0_Exact | road | 53.9454 | 0.1901 | 54.9247 | 96.8006 | 14.6673 | 1 | -610 |
| loveda/P | RivalAliasShuffle0_Exact | water | 62.3150 | -0.0941 | 68.0360 | 88.1105 | 19.0562 | -65 | 259 |
| loveda/P | RivalAliasShuffle0_Exact | barren | 4.0909 | -0.2306 | 89.3419 | 4.1110 | 0.5102 | -298 | -8 |
| loveda/P | RivalAliasShuffle0_Exact | tree | 40.9482 | 0.9929 | 99.3518 | 41.0578 | 6.8042 | 1909 | 25 |
| loveda/P | RivalAliasShuffle0_Exact | farm | 83.1083 | 0.1581 | 83.4227 | 99.5486 | 58.9307 | 30 | -1270 |
| loveda/P | RivalAliasShuffle1_Exact | building | 62.1429 | 1.2950 | 69.7861 | 85.0163 | 0.0323 | 17 | 19 |
| loveda/P | RivalAliasShuffle1_Exact | road | 53.7161 | -0.0392 | 54.6880 | 96.7974 | 14.7303 | -2 | 123 |
| loveda/P | RivalAliasShuffle1_Exact | water | 62.2476 | -0.1615 | 67.9909 | 88.0512 | 19.0560 | -166 | 358 |
| loveda/P | RivalAliasShuffle1_Exact | barren | 3.9377 | -0.3838 | 89.8656 | 3.9553 | 0.4880 | -498 | -65 |
| loveda/P | RivalAliasShuffle1_Exact | tree | 40.4956 | 0.5403 | 99.3268 | 40.6070 | 6.7312 | 1049 | 39 |
| loveda/P | RivalAliasShuffle1_Exact | farm | 83.0661 | 0.1159 | 83.3792 | 99.5500 | 58.9622 | 38 | -912 |
| loveda/P | RivalAliasShuffle2_Exact | building | 69.7970 | 8.9491 | 75.9669 | 89.5765 | 0.0312 | 31 | -7 |
| loveda/P | RivalAliasShuffle2_Exact | road | 53.8145 | 0.0592 | 54.7894 | 96.7995 | 14.7033 | 0 | -191 |
| loveda/P | RivalAliasShuffle2_Exact | water | 62.2430 | -0.1661 | 67.9536 | 88.1046 | 19.0780 | -75 | 522 |
| loveda/P | RivalAliasShuffle2_Exact | barren | 4.1539 | -0.1676 | 89.2346 | 4.1748 | 0.5187 | -216 | 9 |
| loveda/P | RivalAliasShuffle2_Exact | tree | 40.7833 | 0.8280 | 99.3202 | 40.8974 | 6.7798 | 1603 | 48 |
| loveda/P | RivalAliasShuffle2_Exact | farm | 83.1654 | 0.2152 | 83.4810 | 99.5475 | 58.8889 | 24 | -1748 |
| loveda/P | DirectionalMean_Exact | building | 58.7805 | -2.0674 | 70.0581 | 78.5016 | 0.0297 | -3 | 9 |
| loveda/P | DirectionalMean_Exact | road | 53.7630 | 0.0077 | 54.7376 | 96.7943 | 14.7165 | -5 | -34 |
| loveda/P | DirectionalMean_Exact | water | 62.4570 | 0.0479 | 68.1807 | 88.1515 | 19.0246 | 5 | -177 |
| loveda/P | DirectionalMean_Exact | barren | 4.1957 | -0.1258 | 90.0399 | 4.2153 | 0.5191 | -164 | -39 |
| loveda/P | DirectionalMean_Exact | tree | 40.1985 | 0.2432 | 99.3526 | 40.3040 | 6.6792 | 471 | 15 |
| loveda/P | DirectionalMean_Exact | farm | 82.9642 | 0.0140 | 83.2791 | 99.5463 | 59.0310 | 17 | -95 |
| loveda/P | DirectionalShuffle0_Exact | building | 55.4745 | -5.3734 | 68.6747 | 74.2671 | 0.0287 | -16 | 10 |
| loveda/P | DirectionalShuffle0_Exact | road | 53.7519 | -0.0034 | 54.7268 | 96.7923 | 14.7190 | -7 | -2 |
| loveda/P | DirectionalShuffle0_Exact | water | 62.4491 | 0.0400 | 68.1856 | 88.1275 | 19.0180 | -36 | -212 |
| loveda/P | DirectionalShuffle0_Exact | barren | 4.2482 | -0.0733 | 90.0624 | 4.2682 | 0.5255 | -96 | -33 |
| loveda/P | DirectionalShuffle0_Exact | tree | 40.1866 | 0.2313 | 99.3473 | 40.2930 | 6.6778 | 450 | 19 |
| loveda/P | DirectionalShuffle0_Exact | farm | 82.9627 | 0.0125 | 83.2782 | 99.5454 | 59.0310 | 12 | -89 |
| loveda/P | DirectionalShuffle1_Exact | building | 55.5012 | -5.3467 | 68.9970 | 73.9414 | 0.0284 | -17 | 8 |
| loveda/P | DirectionalShuffle1_Exact | road | 53.7671 | 0.0118 | 54.7429 | 96.7912 | 14.7146 | -8 | -53 |
| loveda/P | DirectionalShuffle1_Exact | water | 62.4663 | 0.0572 | 68.1861 | 88.1609 | 19.0251 | 21 | -187 |
| loveda/P | DirectionalShuffle1_Exact | barren | 4.1654 | -0.1561 | 89.9448 | 4.1849 | 0.5159 | -203 | -37 |
| loveda/P | DirectionalShuffle1_Exact | tree | 40.2046 | 0.2493 | 99.3489 | 40.3109 | 6.6806 | 484 | 18 |
| loveda/P | DirectionalShuffle1_Exact | farm | 82.9563 | 0.0061 | 83.2719 | 99.5452 | 59.0354 | 11 | -37 |
| loveda/P | DirectionalShuffle2_Exact | building | 58.7805 | -2.0674 | 70.0581 | 78.5016 | 0.0297 | -3 | 9 |
| loveda/P | DirectionalShuffle2_Exact | road | 53.7506 | -0.0047 | 54.7251 | 96.7933 | 14.7197 | -6 | 4 |
| loveda/P | DirectionalShuffle2_Exact | water | 62.4552 | 0.0461 | 68.1733 | 88.1603 | 19.0286 | 20 | -146 |
| loveda/P | DirectionalShuffle2_Exact | barren | 4.2214 | -0.1001 | 90.1539 | 4.2410 | 0.5216 | -131 | -43 |
| loveda/P | DirectionalShuffle2_Exact | tree | 40.1837 | 0.2284 | 99.3485 | 40.2899 | 6.6772 | 444 | 18 |
| loveda/P | DirectionalShuffle2_Exact | farm | 82.9748 | 0.0246 | 83.2898 | 99.5463 | 59.0234 | 17 | -183 |
| loveda/D | Geometry | background | 39.0474 | 31.7250 | 75.8250 | 44.5998 | 26.3232 | 348157 | 110196 |
| loveda/D | Geometry | building | 4.8538 | -26.4283 | 4.8538 | 100.0000 | 0.3016 | 63 | 5545 |
| loveda/D | Geometry | road | 45.0691 | 0.9421 | 45.8485 | 96.3650 | 9.6637 | -404 | -5317 |
| loveda/D | Geometry | water | 53.8049 | -0.6330 | 58.0149 | 88.1157 | 12.3473 | -36 | 3180 |
| loveda/D | Geometry | barren | 0.0293 | -3.2121 | 0.8665 | 0.0304 | 0.2146 | -4154 | 3564 |
| loveda/D | Geometry | tree | 37.0607 | 1.1933 | 42.3596 | 74.7642 | 16.0550 | 73472 | 192040 |
| loveda/D | Geometry | farm | 57.2802 | 18.4061 | 64.7327 | 83.2647 | 35.0945 | -93073 | -633233 |
| loveda/D | Anchored_Exact | background | 7.3224 | 0.0000 | 75.1732 | 7.5039 | 4.4672 | 0 | 0 |
| loveda/D | Anchored_Exact | building | 31.2821 | 0.0000 | 34.0307 | 79.4788 | 0.0342 | 0 | 0 |
| loveda/D | Anchored_Exact | road | 44.1270 | 0.0000 | 44.7837 | 96.7840 | 9.9365 | 0 | 0 |
| loveda/D | Anchored_Exact | water | 54.4379 | 0.0000 | 58.7421 | 88.1369 | 12.1974 | 0 | 0 |
| loveda/D | Anchored_Exact | barren | 3.2414 | 0.0000 | 82.3610 | 3.2640 | 0.2428 | 0 | 0 |
| loveda/D | Anchored_Exact | tree | 35.8674 | 0.0000 | 97.1427 | 36.2498 | 3.3944 | 0 | 0 |
| loveda/D | Anchored_Exact | farm | 38.8741 | 0.0000 | 38.9455 | 99.5311 | 69.7275 | 0 | 0 |
| loveda/D | ContrastReversal_Exact | background | 12.3885 | 5.0661 | 81.9842 | 12.7352 | 6.9518 | 49098 | 3006 |
| loveda/D | ContrastReversal_Exact | building | 38.9222 | 7.6401 | 41.8680 | 84.6906 | 0.0296 | 16 | -112 |
| loveda/D | ContrastReversal_Exact | road | 43.9562 | -0.1708 | 44.6077 | 96.7840 | 9.9757 | 0 | 822 |
| loveda/D | ContrastReversal_Exact | water | 54.0087 | -0.4292 | 58.3067 | 87.9908 | 12.2681 | -249 | 1732 |
| loveda/D | ContrastReversal_Exact | barren | 2.7892 | -0.4522 | 80.1244 | 2.8086 | 0.2147 | -585 | -3 |
| loveda/D | ContrastReversal_Exact | tree | 36.7162 | 0.8488 | 96.6838 | 37.1845 | 3.4985 | 1783 | 399 |
| loveda/D | ContrastReversal_Exact | farm | 40.4258 | 1.5517 | 40.5002 | 99.5473 | 67.0616 | 93 | -56000 |
| loveda/D | RivalPreserving_Exact | background | 12.8180 | 5.4956 | 83.0094 | 13.1633 | 7.0967 | 53116 | 2028 |
| loveda/D | RivalPreserving_Exact | building | 35.7735 | 4.4914 | 38.3136 | 84.3648 | 0.0322 | 15 | -56 |
| loveda/D | RivalPreserving_Exact | road | 44.2432 | 0.1162 | 44.9056 | 96.7736 | 9.9085 | -10 | -578 |
| loveda/D | RivalPreserving_Exact | water | 54.2433 | -0.1946 | 58.5539 | 88.0500 | 12.2245 | -148 | 717 |
| loveda/D | RivalPreserving_Exact | barren | 2.8827 | -0.3587 | 80.0257 | 2.9036 | 0.2223 | -463 | 33 |
| loveda/D | RivalPreserving_Exact | tree | 37.5513 | 1.6839 | 96.8688 | 38.0127 | 3.5696 | 3363 | 310 |
| loveda/D | RivalPreserving_Exact | farm | 40.4911 | 1.6170 | 40.5670 | 99.5398 | 66.9462 | 50 | -58377 |
| loveda/D | RivalMeanLogit | background | 14.5600 | 7.2376 | 84.0054 | 14.9751 | 7.9778 | 70120 | 3501 |
| loveda/D | RivalMeanLogit | building | 2.0455 | -29.2366 | 3.0457 | 5.8632 | 0.0282 | -226 | 100 |
| loveda/D | RivalMeanLogit | road | 44.6189 | 0.4919 | 45.2722 | 96.8669 | 9.8377 | 80 | -2152 |
| loveda/D | RivalMeanLogit | water | 54.2479 | -0.1900 | 58.2400 | 88.7821 | 12.3926 | 1100 | 2994 |
| loveda/D | RivalMeanLogit | barren | 4.8236 | 1.5822 | 58.1800 | 4.9968 | 0.5261 | 2226 | 3716 |
| loveda/D | RivalMeanLogit | tree | 40.0905 | 4.2231 | 94.3771 | 41.0715 | 3.9586 | 9198 | 2634 |
| loveda/D | RivalMeanLogit | farm | 41.4938 | 2.6197 | 41.5822 | 99.4900 | 65.2790 | -235 | -93056 |
| loveda/D | RivalShuffledSupport_Exact | background | 22.4059 | 15.0835 | 78.1448 | 23.9038 | 13.6895 | 153919 | 39485 |
| loveda/D | RivalShuffledSupport_Exact | building | 49.4176 | 18.1355 | 50.2538 | 96.7427 | 0.0282 | 53 | -179 |
| loveda/D | RivalShuffledSupport_Exact | road | 44.7855 | 0.6585 | 45.4724 | 96.7373 | 9.7813 | -45 | -3210 |
| loveda/D | RivalShuffledSupport_Exact | water | 54.4037 | -0.0342 | 58.8404 | 87.8272 | 12.1342 | -528 | -797 |
| loveda/D | RivalShuffledSupport_Exact | barren | 4.0292 | 0.7878 | 84.6841 | 4.0588 | 0.2936 | 1021 | 45 |
| loveda/D | RivalShuffledSupport_Exact | tree | 29.4436 | -6.4238 | 95.7009 | 29.8383 | 2.8361 | -12231 | 523 |
| loveda/D | RivalShuffledSupport_Exact | farm | 44.1389 | 5.2648 | 44.2660 | 99.3535 | 61.2371 | -1016 | -177040 |
| loveda/D | RivalTextOnly_Exact | background | 2.4036 | -4.9188 | 71.8867 | 2.4265 | 1.5106 | -47653 | -14353 |
| loveda/D | RivalTextOnly_Exact | building | 29.4621 | -1.8200 | 32.0479 | 78.5016 | 0.0359 | -3 | 38 |
| loveda/D | RivalTextOnly_Exact | road | 43.3414 | -0.7856 | 43.9708 | 96.8026 | 10.1222 | 18 | 3875 |
| loveda/D | RivalTextOnly_Exact | water | 54.3793 | -0.0586 | 58.7467 | 87.9732 | 12.1738 | -279 | -216 |
| loveda/D | RivalTextOnly_Exact | barren | 3.3957 | 0.1543 | 82.4236 | 3.4205 | 0.2542 | 201 | 39 |
| loveda/D | RivalTextOnly_Exact | tree | 38.3426 | 2.4752 | 96.7713 | 38.8394 | 3.6509 | 4940 | 438 |
| loveda/D | RivalTextOnly_Exact | farm | 37.5231 | -1.3510 | 37.5882 | 99.5411 | 72.2526 | 57 | 52898 |
| loveda/D | RivalAliasShuffle0_Exact | background | 10.5113 | 3.1889 | 81.3852 | 10.7702 | 5.9224 | 30656 | -139 |
| loveda/D | RivalAliasShuffle0_Exact | building | 36.5952 | 5.3131 | 38.3427 | 88.9251 | 0.0340 | 29 | -34 |
| loveda/D | RivalAliasShuffle0_Exact | road | 44.1665 | 0.0395 | 44.8277 | 96.7684 | 9.9252 | -15 | -223 |
| loveda/D | RivalAliasShuffle0_Exact | water | 54.2022 | -0.2357 | 58.4898 | 88.0870 | 12.2431 | -85 | 1043 |
| loveda/D | RivalAliasShuffle0_Exact | barren | 2.9217 | -0.3197 | 80.4940 | 2.9425 | 0.2239 | -413 | 18 |
| loveda/D | RivalAliasShuffle0_Exact | tree | 37.2215 | 1.3541 | 96.8522 | 37.6772 | 3.5387 | 2723 | 302 |
| loveda/D | RivalAliasShuffle0_Exact | farm | 39.7982 | 0.9241 | 39.8717 | 99.5386 | 68.1128 | 43 | -33905 |
| loveda/D | RivalAliasShuffle1_Exact | background | 10.8638 | 3.5414 | 81.6462 | 11.1357 | 6.1038 | 34086 | 235 |
| loveda/D | RivalAliasShuffle1_Exact | building | 36.1074 | 4.8253 | 38.0481 | 87.6221 | 0.0337 | 25 | -35 |
| loveda/D | RivalAliasShuffle1_Exact | road | 44.2578 | 0.1308 | 44.9202 | 96.7757 | 9.9055 | -8 | -643 |
| loveda/D | RivalAliasShuffle1_Exact | water | 54.1920 | -0.2459 | 58.4807 | 88.0805 | 12.2441 | -96 | 1075 |
| loveda/D | RivalAliasShuffle1_Exact | barren | 3.0687 | -0.1727 | 80.2586 | 3.0920 | 0.2360 | -221 | 79 |
| loveda/D | RivalAliasShuffle1_Exact | tree | 37.2931 | 1.4257 | 96.9315 | 37.7386 | 3.5415 | 2840 | 245 |
| loveda/D | RivalAliasShuffle1_Exact | farm | 39.8998 | 1.0257 | 39.9743 | 99.5349 | 67.9354 | 22 | -37604 |
| loveda/D | RivalAliasShuffle2_Exact | background | 10.4751 | 3.1527 | 81.2669 | 10.7343 | 5.9113 | 30319 | -36 |
| loveda/D | RivalAliasShuffle2_Exact | building | 35.3093 | 4.0272 | 36.8775 | 89.2508 | 0.0354 | 30 | -4 |
| loveda/D | RivalAliasShuffle2_Exact | road | 44.1390 | 0.0120 | 44.7972 | 96.7788 | 9.9330 | -5 | -69 |
| loveda/D | RivalAliasShuffle2_Exact | water | 54.1887 | -0.2492 | 58.4836 | 88.0653 | 12.2414 | -122 | 1044 |
| loveda/D | RivalAliasShuffle2_Exact | barren | 3.0725 | -0.1689 | 80.2624 | 3.0959 | 0.2363 | -216 | 80 |
| loveda/D | RivalAliasShuffle2_Exact | tree | 37.2725 | 1.4051 | 96.9066 | 37.7213 | 3.5408 | 2807 | 263 |
| loveda/D | RivalAliasShuffle2_Exact | farm | 39.8018 | 0.9277 | 39.8761 | 99.5337 | 68.1019 | 15 | -34106 |
| loveda/D | DirectionalMean_Exact | background | 11.5896 | 4.2672 | 80.7970 | 11.9179 | 6.6012 | 41427 | 3325 |
| loveda/D | DirectionalMean_Exact | building | 32.5676 | 1.2855 | 35.7567 | 78.5016 | 0.0321 | -3 | -40 |
| loveda/D | DirectionalMean_Exact | road | 44.1737 | 0.0467 | 44.8355 | 96.7663 | 9.9232 | -17 | -262 |
| loveda/D | DirectionalMean_Exact | water | 54.3856 | -0.0523 | 58.6853 | 88.1275 | 12.2079 | -16 | 236 |
| loveda/D | DirectionalMean_Exact | barren | 3.1367 | -0.1047 | 82.2087 | 3.1582 | 0.2353 | -136 | -20 |
| loveda/D | DirectionalMean_Exact | tree | 35.8911 | 0.0237 | 97.1472 | 36.2734 | 3.3965 | 45 | -2 |
| loveda/D | DirectionalMean_Exact | farm | 40.0945 | 1.2204 | 40.1699 | 99.5337 | 67.6038 | 15 | -44552 |
| loveda/D | DirectionalShuffle0_Exact | background | 11.6549 | 4.3325 | 80.7892 | 11.9871 | 6.6402 | 42077 | 3493 |
| loveda/D | DirectionalShuffle0_Exact | building | 33.2414 | 1.9593 | 36.5706 | 78.5016 | 0.0314 | -3 | -55 |
| loveda/D | DirectionalShuffle0_Exact | road | 44.2038 | 0.0768 | 44.8675 | 96.7622 | 9.9157 | -21 | -415 |
| loveda/D | DirectionalShuffle0_Exact | water | 54.3982 | -0.0397 | 58.6919 | 88.1457 | 12.2090 | 15 | 229 |
| loveda/D | DirectionalShuffle0_Exact | barren | 3.1521 | -0.0893 | 82.7446 | 3.1729 | 0.2349 | -117 | -48 |
| loveda/D | DirectionalShuffle0_Exact | tree | 35.8935 | 0.0261 | 97.1231 | 36.2792 | 3.3978 | 56 | 16 |
| loveda/D | DirectionalShuffle0_Exact | farm | 40.1143 | 1.2402 | 40.1897 | 99.5342 | 67.5709 | 18 | -45245 |
| loveda/D | DirectionalShuffle1_Exact | background | 11.5057 | 4.1833 | 80.5452 | 11.8346 | 6.5755 | 40645 | 3569 |
| loveda/D | DirectionalShuffle1_Exact | building | 32.3925 | 1.1104 | 35.5457 | 78.5016 | 0.0323 | -3 | -36 |
| loveda/D | DirectionalShuffle1_Exact | road | 44.1886 | 0.0616 | 44.8507 | 96.7674 | 9.9200 | -16 | -331 |
| loveda/D | DirectionalShuffle1_Exact | water | 54.3837 | -0.0542 | 58.6777 | 88.1398 | 12.2112 | 5 | 284 |
| loveda/D | DirectionalShuffle1_Exact | barren | 3.1311 | -0.1103 | 82.0502 | 3.1527 | 0.2354 | -143 | -12 |
| loveda/D | DirectionalShuffle1_Exact | tree | 35.7574 | -0.1100 | 97.1450 | 36.1371 | 3.3838 | -215 | -8 |
| loveda/D | DirectionalShuffle1_Exact | farm | 40.0722 | 1.1981 | 40.1475 | 99.5341 | 67.6418 | 17 | -43756 |
| loveda/D | DirectionalShuffle2_Exact | background | 11.5234 | 4.2010 | 80.7839 | 11.8482 | 6.5637 | 40773 | 3192 |
| loveda/D | DirectionalShuffle2_Exact | building | 32.5676 | 1.2855 | 35.7567 | 78.5016 | 0.0321 | -3 | -40 |
| loveda/D | DirectionalShuffle2_Exact | road | 44.1941 | 0.0671 | 44.8577 | 96.7611 | 9.9178 | -22 | -371 |
| loveda/D | DirectionalShuffle2_Exact | water | 54.3837 | -0.0542 | 58.6876 | 88.1175 | 12.2060 | -33 | 214 |
| loveda/D | DirectionalShuffle2_Exact | barren | 3.1619 | -0.0795 | 82.1120 | 3.1838 | 0.2375 | -103 | -7 |
| loveda/D | DirectionalShuffle2_Exact | tree | 35.9115 | 0.0441 | 97.1461 | 36.2944 | 3.3985 | 85 | 0 |
| loveda/D | DirectionalShuffle2_Exact | farm | 40.0702 | 1.1961 | 40.1456 | 99.5332 | 67.6444 | 12 | -43697 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -15.2111 | 77.7719 | 47.8510 | 16.8072 | -137882 | -67750 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 7.0164 | 73.5287 | 98.5836 | 27.6605 | -4810 | -69879 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 9.2961 | 92.4475 | 55.4111 | 17.7225 | 67731 | 17105 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.3793 | 82.7688 | 80.0845 | 19.9693 | -7441 | -21418 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -14.7902 | 10.3220 | 99.7520 | 17.8406 | 784 | 223560 |
| vaihingen/vaihingen | Anchored_Exact | impervious surface | 57.3052 | 0.0000 | 73.8225 | 71.9196 | 26.6125 | 0 | 0 |
| vaihingen/vaihingen | Anchored_Exact | building | 65.7437 | 0.0000 | 65.8760 | 99.6954 | 31.2219 | 0 | 0 |
| vaihingen/vaihingen | Anchored_Exact | low vegetation | 43.7153 | 0.0000 | 96.1772 | 44.4883 | 13.6772 | 0 | 0 |
| vaihingen/vaihingen | Anchored_Exact | tree | 67.2612 | 0.0000 | 79.0950 | 81.8036 | 21.3454 | 0 | 0 |
| vaihingen/vaihingen | Anchored_Exact | car | 25.1095 | 0.0000 | 25.2570 | 97.7270 | 7.1430 | 0 | 0 |
| vaihingen/vaihingen | ContrastReversal_Exact | impervious surface | 56.7782 | -0.5270 | 72.7917 | 72.0742 | 27.0474 | 886 | 8234 |
| vaihingen/vaihingen | ContrastReversal_Exact | building | 65.6430 | -0.1007 | 65.7733 | 99.6991 | 31.2718 | 16 | 1031 |
| vaihingen/vaihingen | ContrastReversal_Exact | low vegetation | 42.2933 | -1.4220 | 96.4220 | 42.9677 | 13.1762 | -9429 | -1078 |
| vaihingen/vaihingen | ContrastReversal_Exact | tree | 67.2546 | -0.0066 | 78.5841 | 82.3475 | 21.6270 | 2354 | 3552 |
| vaihingen/vaihingen | ContrastReversal_Exact | car | 26.0752 | 0.9657 | 26.2338 | 97.7347 | 6.8776 | 3 | -5569 |
| vaihingen/vaihingen | RivalPreserving_Exact | impervious surface | 57.1891 | -0.1161 | 73.7543 | 71.8014 | 26.5934 | -677 | 275 |
| vaihingen/vaihingen | RivalPreserving_Exact | building | 65.5449 | -0.1988 | 65.6756 | 99.6972 | 31.3178 | 8 | 2002 |
| vaihingen/vaihingen | RivalPreserving_Exact | low vegetation | 43.3888 | -0.3265 | 96.3128 | 44.1217 | 13.5454 | -2273 | -491 |
| vaihingen/vaihingen | RivalPreserving_Exact | tree | 67.2583 | -0.0029 | 78.7933 | 82.1246 | 21.5112 | 1389 | 2088 |
| vaihingen/vaihingen | RivalPreserving_Exact | car | 25.5040 | 0.3945 | 25.6559 | 97.7321 | 7.0323 | 2 | -2323 |
| vaihingen/vaihingen | RivalMeanLogit | impervious surface | 55.5759 | -1.7293 | 72.6459 | 70.2839 | 26.4285 | -9370 | 5511 |
| vaihingen/vaihingen | RivalMeanLogit | building | 64.8756 | -0.8681 | 65.1607 | 99.3302 | 31.4490 | -1580 | 6343 |
| vaihingen/vaihingen | RivalMeanLogit | low vegetation | 43.7782 | 0.0629 | 96.2111 | 44.5462 | 13.6901 | 359 | -87 |
| vaihingen/vaihingen | RivalMeanLogit | tree | 65.0448 | -2.2164 | 77.8096 | 79.8585 | 21.1821 | -8419 | 4994 |
| vaihingen/vaihingen | RivalMeanLogit | car | 24.3151 | -0.7944 | 24.5395 | 96.3761 | 7.2503 | -523 | 2772 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | impervious surface | 56.7703 | -0.5349 | 72.5816 | 72.2687 | 27.1989 | 2000 | 10297 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | building | 65.4783 | -0.2654 | 65.6094 | 99.6956 | 31.3488 | 1 | 2661 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | low vegetation | 42.5194 | -1.1959 | 96.3608 | 43.2133 | 13.2599 | -7906 | -845 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | tree | 67.2813 | 0.0201 | 79.1993 | 81.7221 | 21.2960 | -353 | -682 |
| vaihingen/vaihingen | RivalShuffledSupport_Exact | car | 25.9918 | 0.8823 | 26.1521 | 97.6960 | 6.8964 | -12 | -5161 |
| vaihingen/vaihingen | RivalTextOnly_Exact | impervious surface | 56.9331 | -0.3721 | 75.8222 | 69.5618 | 25.0612 | -13507 | -19026 |
| vaihingen/vaihingen | RivalTextOnly_Exact | building | 64.5621 | -1.1816 | 64.6793 | 99.7201 | 31.8075 | 107 | 12173 |
| vaihingen/vaihingen | RivalTextOnly_Exact | low vegetation | 44.5847 | 0.8694 | 96.2684 | 45.3688 | 13.9347 | 5460 | -60 |
| vaihingen/vaihingen | RivalTextOnly_Exact | tree | 67.1781 | -0.0831 | 78.8687 | 81.9236 | 21.4380 | 519 | 1424 |
| vaihingen/vaihingen | RivalTextOnly_Exact | car | 23.1550 | -1.9545 | 23.2752 | 97.8200 | 7.7586 | 36 | 12874 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | impervious surface | 57.3927 | 0.0875 | 74.0682 | 71.8249 | 26.4894 | -542 | -2041 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | building | 65.7061 | -0.0376 | 65.8372 | 99.6979 | 31.2411 | 11 | 392 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | low vegetation | 43.9582 | 0.2429 | 96.2096 | 44.7329 | 13.7477 | 1517 | -37 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | tree | 67.2973 | 0.0361 | 78.9835 | 81.9769 | 21.4208 | 750 | 832 |
| vaihingen/vaihingen | RivalAliasShuffle0_Exact | car | 25.2649 | 0.1554 | 25.4126 | 97.7502 | 7.1010 | 9 | -891 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | impervious surface | 57.3440 | 0.0388 | 73.9540 | 71.8560 | 26.5417 | -364 | -1121 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | building | 65.6488 | -0.0949 | 65.7802 | 99.6968 | 31.2678 | 6 | 957 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | low vegetation | 43.8698 | 0.1545 | 96.1754 | 44.6487 | 13.7268 | 995 | 45 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | tree | 67.2785 | 0.0173 | 78.9873 | 81.9448 | 21.4114 | 611 | 773 |
| vaihingen/vaihingen | RivalAliasShuffle1_Exact | car | 25.4322 | 0.3227 | 25.5832 | 97.7321 | 7.0523 | 2 | -1904 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | impervious surface | 57.3955 | 0.0903 | 74.1047 | 71.7949 | 26.4652 | -714 | -2375 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | building | 65.6791 | -0.0646 | 65.8104 | 99.6972 | 31.2536 | 8 | 657 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | low vegetation | 44.0217 | 0.3064 | 96.2093 | 44.7987 | 13.7680 | 1925 | -20 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | tree | 67.2998 | 0.0386 | 78.9597 | 82.0063 | 21.4349 | 877 | 1001 |
| vaihingen/vaihingen | RivalAliasShuffle2_Exact | car | 25.3489 | 0.2394 | 25.4970 | 97.7606 | 7.0782 | 13 | -1372 |
| vaihingen/vaihingen | DirectionalMean_Exact | impervious surface | 57.2339 | -0.0713 | 73.7613 | 71.8653 | 26.6145 | -311 | 352 |
| vaihingen/vaihingen | DirectionalMean_Exact | building | 65.6119 | -0.1318 | 65.7433 | 99.6963 | 31.2852 | 4 | 1324 |
| vaihingen/vaihingen | DirectionalMean_Exact | low vegetation | 43.4338 | -0.2815 | 96.2896 | 44.1732 | 13.5644 | -1954 | -410 |
| vaihingen/vaihingen | DirectionalMean_Exact | tree | 67.2751 | 0.0139 | 78.9216 | 82.0107 | 21.4464 | 896 | 1223 |
| vaihingen/vaihingen | DirectionalMean_Exact | car | 25.3041 | 0.1946 | 25.4527 | 97.7451 | 7.0894 | 7 | -1131 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | impervious surface | 57.2369 | -0.0683 | 73.7681 | 71.8637 | 26.6115 | -320 | 298 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | building | 65.6080 | -0.1357 | 65.7390 | 99.6972 | 31.2876 | 8 | 1369 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | low vegetation | 43.4225 | -0.2928 | 96.2883 | 44.1617 | 13.5611 | -2025 | -409 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | tree | 67.2609 | -0.0003 | 78.9075 | 82.0049 | 21.4488 | 871 | 1297 |
| vaihingen/vaihingen | DirectionalShuffle0_Exact | car | 25.2982 | 0.1887 | 25.4467 | 97.7451 | 7.0911 | 7 | -1096 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | impervious surface | 57.2435 | -0.0617 | 73.7687 | 71.8735 | 26.6149 | -264 | 313 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | building | 65.6170 | -0.1267 | 65.7488 | 99.6954 | 31.2823 | 0 | 1267 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | low vegetation | 43.4072 | -0.3081 | 96.2860 | 44.1464 | 13.5567 | -2120 | -406 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | tree | 67.2759 | 0.0147 | 78.9147 | 82.0192 | 21.4505 | 933 | 1272 |
| vaihingen/vaihingen | DirectionalShuffle1_Exact | car | 25.2806 | 0.1711 | 25.4293 | 97.7399 | 7.0956 | 5 | -1000 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | impervious surface | 57.2359 | -0.0693 | 73.7562 | 71.8733 | 26.6193 | -265 | 407 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | building | 65.6198 | -0.1239 | 65.7506 | 99.6979 | 31.2823 | 11 | 1255 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | low vegetation | 43.4055 | -0.3098 | 96.2846 | 44.1449 | 13.5565 | -2129 | -402 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | tree | 67.2786 | 0.0174 | 78.9236 | 82.0137 | 21.4467 | 909 | 1215 |
| vaihingen/vaihingen | DirectionalShuffle2_Exact | car | 25.2824 | 0.1729 | 25.4310 | 97.7425 | 7.0953 | 6 | -1007 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -4.8387 | 96.6505 | 85.0437 | 59.6226 | -109633 | -44423 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.5110 | 35.0436 | 99.2919 | 4.2927 | 316 | 20014 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -0.2388 | 86.4992 | 89.2841 | 22.0960 | 32579 | 43002 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9543 | 93.6635 | 100.0000 | 8.8309 | 0 | 7502 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -11.4629 | 15.6288 | 77.0019 | 5.1578 | 308 | 50335 |
| landcoverai/landcoverai | Anchored_Exact | background | 87.4477 | 0.0000 | 93.8549 | 92.7587 | 66.9686 | 0 | 0 |
| landcoverai/landcoverai | Anchored_Exact | building | 44.4672 | 0.0000 | 44.8124 | 98.2973 | 3.3233 | 0 | 0 |
| landcoverai/landcoverai | Anchored_Exact | woodland | 78.6026 | 0.0000 | 94.9565 | 82.0272 | 18.4920 | 0 | 0 |
| landcoverai/landcoverai | Anchored_Exact | water | 97.6178 | 0.0000 | 97.6178 | 100.0000 | 8.4732 | 0 | 0 |
| landcoverai/landcoverai | Anchored_Exact | road | 26.3947 | 0.0000 | 28.8528 | 75.5990 | 2.7429 | 0 | 0 |
| landcoverai/landcoverai | ContrastReversal_Exact | background | 87.4487 | 0.0010 | 93.9520 | 92.6652 | 66.8319 | -1329 | -1537 |
| landcoverai/landcoverai | ContrastReversal_Exact | building | 44.3740 | -0.0932 | 44.7165 | 98.3036 | 3.3307 | 2 | 152 |
| landcoverai/landcoverai | ContrastReversal_Exact | woodland | 78.9032 | 0.3006 | 94.9367 | 82.3695 | 18.5731 | 1537 | 163 |
| landcoverai/landcoverai | ContrastReversal_Exact | water | 97.6321 | 0.0143 | 97.6321 | 100.0000 | 8.4719 | 0 | -26 |
| landcoverai/landcoverai | ContrastReversal_Exact | road | 25.9700 | -0.4247 | 28.3448 | 75.6081 | 2.7924 | 2 | 1036 |
| landcoverai/landcoverai | RivalPreserving_Exact | background | 87.4526 | 0.0049 | 93.9019 | 92.7182 | 66.9058 | -575 | -741 |
| landcoverai/landcoverai | RivalPreserving_Exact | building | 44.4119 | -0.0553 | 44.7549 | 98.3036 | 3.3278 | 2 | 92 |
| landcoverai/landcoverai | RivalPreserving_Exact | woodland | 78.7457 | 0.1431 | 94.9462 | 82.1907 | 18.5309 | 734 | 81 |
| landcoverai/landcoverai | RivalPreserving_Exact | water | 97.6398 | 0.0220 | 97.6398 | 100.0000 | 8.4713 | 0 | -40 |
| landcoverai/landcoverai | RivalPreserving_Exact | road | 26.2104 | -0.1843 | 28.6321 | 75.6035 | 2.7642 | 1 | 446 |
| landcoverai/landcoverai | RivalMeanLogit | background | 87.1203 | -0.3274 | 93.6205 | 92.6187 | 67.0349 | -1990 | 3381 |
| landcoverai/landcoverai | RivalMeanLogit | building | 44.1702 | -0.2970 | 44.5952 | 97.8881 | 3.3256 | -130 | 178 |
| landcoverai/landcoverai | RivalMeanLogit | woodland | 77.8919 | -0.7107 | 94.9372 | 81.2676 | 18.3245 | -3410 | -103 |
| landcoverai/landcoverai | RivalMeanLogit | water | 97.3658 | -0.2520 | 97.3658 | 100.0000 | 8.4951 | 0 | 460 |
| landcoverai/landcoverai | RivalMeanLogit | road | 25.5395 | -0.8552 | 27.8962 | 75.1435 | 2.8199 | -100 | 1714 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | background | 87.1377 | -0.3100 | 93.3993 | 92.8560 | 67.3658 | 1382 | 6949 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | building | 43.8916 | -0.5756 | 44.2234 | 98.3193 | 3.3683 | 7 | 937 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | woodland | 77.5410 | -1.0616 | 95.5102 | 80.4744 | 18.0368 | -6971 | -2576 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | water | 97.8193 | 0.2015 | 97.8193 | 100.0000 | 8.4557 | 0 | -366 |
| landcoverai/landcoverai | RivalShuffledSupport_Exact | road | 26.1276 | -0.2671 | 28.5346 | 75.5944 | 2.7733 | -1 | 639 |
| landcoverai/landcoverai | RivalTextOnly_Exact | background | 87.2359 | -0.2118 | 94.7309 | 91.6846 | 65.5810 | -15263 | -13837 |
| landcoverai/landcoverai | RivalTextOnly_Exact | building | 43.1752 | -1.2920 | 43.4846 | 98.3791 | 3.4276 | 26 | 2162 |
| landcoverai/landcoverai | RivalTextOnly_Exact | woodland | 80.1932 | 1.5906 | 93.3509 | 85.0512 | 19.5035 | 13576 | 7637 |
| landcoverai/landcoverai | RivalTextOnly_Exact | water | 97.4221 | -0.1957 | 97.4221 | 100.0000 | 8.4902 | 0 | 357 |
| landcoverai/landcoverai | RivalTextOnly_Exact | road | 24.4995 | -1.8952 | 26.5505 | 76.0271 | 2.9976 | 94 | 5248 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | background | 87.4637 | 0.0160 | 93.9300 | 92.7035 | 66.8752 | -785 | -1174 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | building | 44.3860 | -0.0812 | 44.7286 | 98.3036 | 3.3298 | 2 | 133 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | woodland | 78.8022 | 0.1996 | 94.9027 | 82.2849 | 18.5606 | 1157 | 282 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | water | 97.6332 | 0.0154 | 97.6332 | 100.0000 | 8.4718 | 0 | -28 |
| landcoverai/landcoverai | RivalAliasShuffle0_Exact | road | 26.2305 | -0.1642 | 28.6540 | 75.6172 | 2.7626 | 4 | 409 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | background | 87.4696 | 0.0219 | 93.9180 | 92.7218 | 66.8969 | -525 | -978 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | building | 44.4125 | -0.0547 | 44.7555 | 98.3036 | 3.3278 | 2 | 91 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | woodland | 78.7808 | 0.1782 | 94.9278 | 82.2428 | 18.5462 | 968 | 169 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | water | 97.6294 | 0.0116 | 97.6294 | 100.0000 | 8.4722 | 0 | -21 |
| landcoverai/landcoverai | RivalAliasShuffle1_Exact | road | 26.2739 | -0.1208 | 28.7078 | 75.6035 | 2.7569 | 1 | 293 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | background | 87.4483 | 0.0006 | 93.8910 | 92.7242 | 66.9179 | -491 | -572 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | building | 44.4087 | -0.0585 | 44.7517 | 98.3036 | 3.3280 | 2 | 97 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | woodland | 78.7020 | 0.0994 | 94.9347 | 82.1517 | 18.5244 | 559 | 119 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | water | 97.6316 | 0.0138 | 97.6316 | 100.0000 | 8.4720 | 0 | -25 |
| landcoverai/landcoverai | RivalAliasShuffle2_Exact | road | 26.2688 | -0.1259 | 28.7011 | 75.6081 | 2.7577 | 2 | 309 |
| landcoverai/landcoverai | DirectionalMean_Exact | background | 87.4487 | 0.0010 | 93.8869 | 92.7286 | 66.9240 | -428 | -507 |
| landcoverai/landcoverai | DirectionalMean_Exact | building | 44.3805 | -0.0867 | 44.7224 | 98.3067 | 3.3303 | 3 | 144 |
| landcoverai/landcoverai | DirectionalMean_Exact | woodland | 78.6993 | 0.0967 | 94.9495 | 82.1376 | 18.5183 | 496 | 55 |
| landcoverai/landcoverai | DirectionalMean_Exact | water | 97.6607 | 0.0429 | 97.6607 | 100.0000 | 8.4694 | 0 | -78 |
| landcoverai/landcoverai | DirectionalMean_Exact | road | 26.2671 | -0.1276 | 28.6991 | 75.6081 | 2.7579 | 2 | 313 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | background | 87.4488 | 0.0011 | 93.8877 | 92.7279 | 66.9230 | -438 | -519 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | building | 44.3838 | -0.0834 | 44.7251 | 98.3099 | 3.3302 | 4 | 141 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | woodland | 78.6987 | 0.0961 | 94.9459 | 82.1397 | 18.5194 | 505 | 70 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | water | 97.6563 | 0.0385 | 97.6563 | 100.0000 | 8.4698 | 0 | -70 |
| landcoverai/landcoverai | DirectionalShuffle0_Exact | road | 26.2705 | -0.1242 | 28.7031 | 75.6081 | 2.7575 | 2 | 305 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | background | 87.4476 | -0.0001 | 93.8865 | 92.7277 | 66.9236 | -441 | -502 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | building | 44.3542 | -0.1130 | 44.6950 | 98.3099 | 3.3325 | 4 | 188 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | woodland | 78.6952 | 0.0926 | 94.9448 | 82.1368 | 18.5190 | 492 | 74 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | water | 97.6656 | 0.0478 | 97.6656 | 100.0000 | 8.4690 | 0 | -87 |
| landcoverai/landcoverai | DirectionalShuffle1_Exact | road | 26.2850 | -0.1097 | 28.7205 | 75.6081 | 2.7559 | 2 | 270 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | background | 87.4551 | 0.0074 | 93.8892 | 92.7335 | 66.9259 | -358 | -537 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | building | 44.4084 | -0.0588 | 44.7501 | 98.3099 | 3.3284 | 4 | 102 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | woodland | 78.7058 | 0.1032 | 94.9504 | 82.1441 | 18.5196 | 525 | 53 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | water | 97.6535 | 0.0357 | 97.6535 | 100.0000 | 8.4701 | 0 | -65 |
| landcoverai/landcoverai | DirectionalShuffle2_Exact | road | 26.2834 | -0.1113 | 28.7185 | 75.6081 | 2.7561 | 2 | 274 |
| flair1/flair1 | Geometry | building | 49.6979 | -7.0776 | 50.7258 | 96.0825 | 13.5991 | -336 | 35681 |
| flair1/flair1 | Geometry | pervious surface | 57.0621 | 9.7868 | 92.1299 | 59.9861 | 11.1728 | 38121 | 2352 |
| flair1/flair1 | Geometry | impervious surface | 52.6509 | 0.6651 | 62.6245 | 76.7765 | 20.0846 | -10906 | -27386 |
| flair1/flair1 | Geometry | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4411 | 0 | -77697 |
| flair1/flair1 | Geometry | water | 73.2328 | 11.7763 | 77.1432 | 93.5263 | 5.3781 | -1918 | -25878 |
| flair1/flair1 | Geometry | coniferous | 43.0233 | -0.5049 | 61.7114 | 58.6897 | 0.5346 | 281 | 832 |
| flair1/flair1 | Geometry | deciduous | 54.0229 | -5.5008 | 78.9723 | 63.0995 | 13.6004 | -27735 | -8079 |
| flair1/flair1 | Geometry | brushwood | 18.6136 | 6.2849 | 23.4212 | 47.5559 | 9.9866 | 29569 | 105554 |
| flair1/flair1 | Geometry | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 | 0 | -34 |
| flair1/flair1 | Geometry | herbaceous vegetation | 60.2329 | 2.7604 | 94.3169 | 62.5013 | 21.2289 | 9983 | -16100 |
| flair1/flair1 | Geometry | agricultural land | 18.7339 | 5.7793 | 21.6468 | 58.1977 | 0.8198 | -451 | -12340 |
| flair1/flair1 | Geometry | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1540 | 0 | -13513 |
| flair1/flair1 | Anchored_Exact | building | 56.7755 | 0.0000 | 58.0396 | 96.3057 | 11.9131 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | pervious surface | 47.2753 | 0.0000 | 91.6998 | 49.3887 | 9.2421 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | impervious surface | 51.9858 | 0.0000 | 59.7781 | 79.9522 | 21.9112 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | water | 61.4565 | 0.0000 | 63.2502 | 95.5889 | 6.7041 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | coniferous | 43.5282 | 0.0000 | 65.7321 | 56.3052 | 0.4815 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | deciduous | 59.5237 | 0.0000 | 78.8015 | 70.8722 | 15.3089 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | brushwood | 12.3287 | 0.0000 | 26.2213 | 18.8771 | 3.5408 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | herbaceous vegetation | 57.4725 | 0.0000 | 90.8252 | 61.0148 | 21.5207 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | agricultural land | 12.9546 | 0.0000 | 13.9145 | 65.2534 | 1.4300 | 0 | 0 |
| flair1/flair1 | Anchored_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 | 0 | 0 |
| flair1/flair1 | ContrastReversal_Exact | building | 56.5714 | -0.2041 | 57.7672 | 96.4699 | 11.9896 | 247 | 1358 |
| flair1/flair1 | ContrastReversal_Exact | pervious surface | 46.5616 | -0.7137 | 91.8524 | 48.5675 | 9.0733 | -2954 | -584 |
| flair1/flair1 | ContrastReversal_Exact | impervious surface | 52.0562 | 0.0704 | 59.8453 | 79.9985 | 21.8993 | 159 | -409 |
| flair1/flair1 | ContrastReversal_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3366 | 0 | 3965 |
| flair1/flair1 | ContrastReversal_Exact | water | 62.3123 | 0.8558 | 64.1679 | 95.5652 | 6.6066 | -22 | -2022 |
| flair1/flair1 | ContrastReversal_Exact | coniferous | 43.3290 | -0.1992 | 65.2447 | 56.3306 | 0.4853 | 3 | 77 |
| flair1/flair1 | ContrastReversal_Exact | deciduous | 59.7799 | 0.2562 | 78.6259 | 71.3797 | 15.4529 | 1811 | 1209 |
| flair1/flair1 | ContrastReversal_Exact | brushwood | 12.9010 | 0.5723 | 26.4495 | 20.1185 | 3.7411 | 1280 | 2919 |
| flair1/flair1 | ContrastReversal_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0034 | 0 | 37 |
| flair1/flair1 | ContrastReversal_Exact | herbaceous vegetation | 56.2345 | -1.2380 | 91.2319 | 59.4474 | 20.8743 | -10526 | -3023 |
| flair1/flair1 | ContrastReversal_Exact | agricultural land | 12.7969 | -0.1577 | 13.7182 | 65.5820 | 1.4577 | 21 | 561 |
| flair1/flair1 | ContrastReversal_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0798 | 0 | 5893 |
| flair1/flair1 | RivalPreserving_Exact | building | 56.7142 | -0.0613 | 57.9673 | 96.3283 | 11.9307 | 34 | 336 |
| flair1/flair1 | RivalPreserving_Exact | pervious surface | 47.1308 | -0.1445 | 91.7711 | 49.2105 | 9.2016 | -641 | -208 |
| flair1/flair1 | RivalPreserving_Exact | impervious surface | 52.0164 | 0.0306 | 59.8112 | 79.9653 | 21.9027 | 45 | -224 |
| flair1/flair1 | RivalPreserving_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1827 | 0 | 739 |
| flair1/flair1 | RivalPreserving_Exact | water | 61.5951 | 0.1386 | 63.3929 | 95.5985 | 6.6897 | 9 | -311 |
| flair1/flair1 | RivalPreserving_Exact | coniferous | 43.6160 | 0.0878 | 65.9443 | 56.2967 | 0.4799 | -1 | -33 |
| flair1/flair1 | RivalPreserving_Exact | deciduous | 59.7275 | 0.2038 | 78.7227 | 71.2256 | 15.4006 | 1261 | 662 |
| flair1/flair1 | RivalPreserving_Exact | brushwood | 12.4665 | 0.1378 | 26.6527 | 18.9770 | 3.5019 | 103 | -918 |
| flair1/flair1 | RivalPreserving_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0 | 3 |
| flair1/flair1 | RivalPreserving_Exact | herbaceous vegetation | 57.2206 | -0.2519 | 90.8663 | 60.7126 | 21.4044 | -2029 | -408 |
| flair1/flair1 | RivalPreserving_Exact | agricultural land | 12.9045 | -0.0501 | 13.8539 | 65.3160 | 1.4376 | 4 | 156 |
| flair1/flair1 | RivalPreserving_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8664 | 0 | 1421 |
| flair1/flair1 | RivalMeanLogit | building | 54.6446 | -2.1309 | 56.0129 | 95.7210 | 12.2692 | -880 | 8345 |
| flair1/flair1 | RivalMeanLogit | pervious surface | 46.0791 | -1.1962 | 88.8694 | 48.9014 | 9.4424 | -1753 | 5951 |
| flair1/flair1 | RivalMeanLogit | impervious surface | 51.0324 | -0.9534 | 59.2514 | 78.6276 | 21.7397 | -4549 | 954 |
| flair1/flair1 | RivalMeanLogit | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.4775 | 0 | 6919 |
| flair1/flair1 | RivalMeanLogit | water | 59.6274 | -1.8291 | 61.9304 | 94.1296 | 6.7424 | -1357 | 2161 |
| flair1/flair1 | RivalMeanLogit | coniferous | 38.6965 | -4.8317 | 57.1479 | 54.5146 | 0.5362 | -211 | 1358 |
| flair1/flair1 | RivalMeanLogit | deciduous | 57.7916 | -1.7321 | 78.7423 | 68.4749 | 14.8022 | -8554 | -2068 |
| flair1/flair1 | RivalMeanLogit | brushwood | 13.5463 | 1.2176 | 28.6071 | 20.4648 | 3.5185 | 1637 | -2105 |
| flair1/flair1 | RivalMeanLogit | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0218 | 0 | 422 |
| flair1/flair1 | RivalMeanLogit | herbaceous vegetation | 55.2680 | -2.2045 | 90.5030 | 58.6707 | 20.7675 | -15742 | -46 |
| flair1/flair1 | RivalMeanLogit | agricultural land | 11.7345 | -1.2201 | 12.4492 | 67.1464 | 1.6446 | 121 | 4379 |
| flair1/flair1 | RivalMeanLogit | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0380 | 0 | 5018 |
| flair1/flair1 | RivalShuffledSupport_Exact | building | 56.7062 | -0.0693 | 57.9779 | 96.2758 | 11.9220 | -45 | 233 |
| flair1/flair1 | RivalShuffledSupport_Exact | pervious surface | 46.8881 | -0.3872 | 92.1203 | 48.8472 | 9.0990 | -1948 | -1051 |
| flair1/flair1 | RivalShuffledSupport_Exact | impervious surface | 51.7888 | -0.1970 | 59.3284 | 80.2964 | 22.1723 | 1182 | 4292 |
| flair1/flair1 | RivalShuffledSupport_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2489 | 0 | 2126 |
| flair1/flair1 | RivalShuffledSupport_Exact | water | 61.2280 | -0.2285 | 63.0828 | 95.4179 | 6.7098 | -159 | 280 |
| flair1/flair1 | RivalShuffledSupport_Exact | coniferous | 43.4777 | -0.0505 | 66.0723 | 55.9742 | 0.4762 | -39 | -72 |
| flair1/flair1 | RivalShuffledSupport_Exact | deciduous | 59.1999 | -0.3238 | 78.6656 | 70.5224 | 15.2596 | -1248 | 216 |
| flair1/flair1 | RivalShuffledSupport_Exact | brushwood | 11.8372 | -0.4915 | 25.8787 | 17.9091 | 3.4037 | -998 | -1876 |
| flair1/flair1 | RivalShuffledSupport_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0052 | 0 | 76 |
| flair1/flair1 | RivalShuffledSupport_Exact | herbaceous vegetation | 57.3205 | -0.1520 | 90.7674 | 60.8694 | 21.4831 | -976 | 188 |
| flair1/flair1 | RivalShuffledSupport_Exact | agricultural land | 13.0658 | 0.1112 | 14.0428 | 65.2534 | 1.4169 | 0 | -274 |
| flair1/flair1 | RivalShuffledSupport_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8031 | 0 | 93 |
| flair1/flair1 | RivalTextOnly_Exact | building | 55.6739 | -1.1016 | 56.4885 | 97.4752 | 12.3888 | 1760 | 8213 |
| flair1/flair1 | RivalTextOnly_Exact | pervious surface | 44.1794 | -3.0959 | 93.5559 | 45.5660 | 8.3576 | -13751 | -4791 |
| flair1/flair1 | RivalTextOnly_Exact | impervious surface | 52.0676 | 0.0818 | 60.2553 | 79.3037 | 21.5614 | -2227 | -5107 |
| flair1/flair1 | RivalTextOnly_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.9124 | 0 | 16035 |
| flair1/flair1 | RivalTextOnly_Exact | water | 60.4753 | -0.9812 | 62.1556 | 95.7211 | 6.8316 | 123 | 2550 |
| flair1/flair1 | RivalTextOnly_Exact | coniferous | 42.8810 | -0.6472 | 63.9065 | 56.5852 | 0.4977 | 33 | 307 |
| flair1/flair1 | RivalTextOnly_Exact | deciduous | 60.4115 | 0.8878 | 78.0834 | 72.7468 | 15.8583 | 6689 | 4829 |
| flair1/flair1 | RivalTextOnly_Exact | brushwood | 10.6927 | -1.6360 | 27.0627 | 15.0217 | 2.7301 | -3975 | -13021 |
| flair1/flair1 | RivalTextOnly_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 0 | -12 |
| flair1/flair1 | RivalTextOnly_Exact | herbaceous vegetation | 57.2219 | -0.2506 | 90.6685 | 60.8027 | 21.4829 | -1424 | 633 |
| flair1/flair1 | RivalTextOnly_Exact | agricultural land | 12.2712 | -0.6834 | 13.0723 | 66.6927 | 1.5556 | 92 | 2543 |
| flair1/flair1 | RivalTextOnly_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8225 | 0 | 501 |
| flair1/flair1 | RivalAliasShuffle0_Exact | building | 56.7368 | -0.0387 | 57.9945 | 96.3184 | 11.9239 | 19 | 208 |
| flair1/flair1 | RivalAliasShuffle0_Exact | pervious surface | 47.1845 | -0.0908 | 91.7506 | 49.2750 | 9.2157 | -409 | -144 |
| flair1/flair1 | RivalAliasShuffle0_Exact | impervious surface | 52.0547 | 0.0689 | 59.8722 | 79.9469 | 21.8753 | -18 | -734 |
| flair1/flair1 | RivalAliasShuffle0_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1465 | 0 | -20 |
| flair1/flair1 | RivalAliasShuffle0_Exact | water | 61.6032 | 0.1467 | 63.4000 | 95.6018 | 6.6891 | 12 | -325 |
| flair1/flair1 | RivalAliasShuffle0_Exact | coniferous | 43.6246 | 0.0964 | 65.9640 | 56.2967 | 0.4798 | -1 | -36 |
| flair1/flair1 | RivalAliasShuffle0_Exact | deciduous | 59.5834 | 0.0597 | 78.8312 | 70.9327 | 15.3162 | 216 | -63 |
| flair1/flair1 | RivalAliasShuffle0_Exact | brushwood | 12.5083 | 0.1796 | 26.3702 | 19.2214 | 3.5850 | 355 | 572 |
| flair1/flair1 | RivalAliasShuffle0_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0025 | 0 | 18 |
| flair1/flair1 | RivalAliasShuffle0_Exact | herbaceous vegetation | 57.3430 | -0.1295 | 90.7940 | 60.8829 | 21.4815 | -886 | 65 |
| flair1/flair1 | RivalAliasShuffle0_Exact | agricultural land | 13.0090 | 0.0544 | 13.9787 | 65.2222 | 1.4227 | -2 | -150 |
| flair1/flair1 | RivalAliasShuffle0_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8618 | 0 | 1323 |
| flair1/flair1 | RivalAliasShuffle1_Exact | building | 56.7339 | -0.0416 | 57.9924 | 96.3157 | 11.9240 | 15 | 214 |
| flair1/flair1 | RivalAliasShuffle1_Exact | pervious surface | 47.1066 | -0.1687 | 91.7326 | 49.1952 | 9.2026 | -696 | -132 |
| flair1/flair1 | RivalAliasShuffle1_Exact | impervious surface | 52.0562 | 0.0704 | 59.8756 | 79.9443 | 21.8734 | -27 | -766 |
| flair1/flair1 | RivalAliasShuffle1_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1749 | 0 | 576 |
| flair1/flair1 | RivalAliasShuffle1_Exact | water | 61.5225 | 0.0660 | 63.3117 | 95.6082 | 6.6989 | 18 | -126 |
| flair1/flair1 | RivalAliasShuffle1_Exact | coniferous | 43.5645 | 0.0363 | 65.8266 | 56.2967 | 0.4808 | -1 | -15 |
| flair1/flair1 | RivalAliasShuffle1_Exact | deciduous | 59.5929 | 0.0692 | 78.8618 | 70.9215 | 15.3078 | 176 | -198 |
| flair1/flair1 | RivalAliasShuffle1_Exact | brushwood | 12.4195 | 0.0908 | 26.1761 | 19.1147 | 3.5916 | 245 | 819 |
| flair1/flair1 | RivalAliasShuffle1_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | 0 | 10 |
| flair1/flair1 | RivalAliasShuffle1_Exact | herbaceous vegetation | 57.3333 | -0.1392 | 90.8335 | 60.8543 | 21.4621 | -1078 | -150 |
| flair1/flair1 | RivalAliasShuffle1_Exact | agricultural land | 12.9962 | 0.0416 | 13.9511 | 65.5038 | 1.4317 | 16 | 20 |
| flair1/flair1 | RivalAliasShuffle1_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8502 | 0 | 1080 |
| flair1/flair1 | RivalAliasShuffle2_Exact | building | 56.7634 | -0.0121 | 58.0170 | 96.3330 | 11.9211 | 41 | 127 |
| flair1/flair1 | RivalAliasShuffle2_Exact | pervious surface | 47.2013 | -0.0740 | 91.7396 | 49.2964 | 9.2208 | -332 | -114 |
| flair1/flair1 | RivalAliasShuffle2_Exact | impervious surface | 52.0608 | 0.0750 | 59.8794 | 79.9484 | 21.8731 | -13 | -786 |
| flair1/flair1 | RivalAliasShuffle2_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1483 | 0 | 17 |
| flair1/flair1 | RivalAliasShuffle2_Exact | water | 61.6037 | 0.1472 | 63.3996 | 95.6039 | 6.6893 | 14 | -323 |
| flair1/flair1 | RivalAliasShuffle2_Exact | coniferous | 43.5613 | 0.0331 | 65.7845 | 56.3221 | 0.4813 | 2 | -7 |
| flair1/flair1 | RivalAliasShuffle2_Exact | deciduous | 59.6094 | 0.0857 | 78.8235 | 70.9759 | 15.3270 | 370 | 10 |
| flair1/flair1 | RivalAliasShuffle2_Exact | brushwood | 12.5698 | 0.2411 | 26.5727 | 19.2592 | 3.5647 | 394 | 107 |
| flair1/flair1 | RivalAliasShuffle2_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0024 | 0 | 17 |
| flair1/flair1 | RivalAliasShuffle2_Exact | herbaceous vegetation | 57.3824 | -0.0901 | 90.7797 | 60.9338 | 21.5029 | -544 | 171 |
| flair1/flair1 | RivalAliasShuffle2_Exact | agricultural land | 13.0175 | 0.0629 | 13.9885 | 65.2222 | 1.4217 | -2 | -171 |
| flair1/flair1 | RivalAliasShuffle2_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8474 | 0 | 1022 |
| flair1/flair1 | DirectionalMean_Exact | building | 56.7541 | -0.0214 | 58.0116 | 96.3210 | 11.9207 | 23 | 137 |
| flair1/flair1 | DirectionalMean_Exact | pervious surface | 47.1085 | -0.1668 | 91.7397 | 49.1952 | 9.2019 | -696 | -147 |
| flair1/flair1 | DirectionalMean_Exact | impervious surface | 51.9799 | -0.0059 | 59.7535 | 79.9822 | 21.9284 | 103 | 258 |
| flair1/flair1 | DirectionalMean_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1810 | 0 | 703 |
| flair1/flair1 | DirectionalMean_Exact | water | 61.5275 | 0.0710 | 63.3345 | 95.5684 | 6.6937 | -19 | -198 |
| flair1/flair1 | DirectionalMean_Exact | coniferous | 43.5282 | 0.0000 | 65.7321 | 56.3052 | 0.4815 | 0 | 0 |
| flair1/flair1 | DirectionalMean_Exact | deciduous | 59.6268 | 0.1031 | 78.7221 | 71.0829 | 15.3699 | 752 | 527 |
| flair1/flair1 | DirectionalMean_Exact | brushwood | 12.3900 | 0.0613 | 26.5247 | 18.8644 | 3.4980 | -13 | -885 |
| flair1/flair1 | DirectionalMean_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0 | 3 |
| flair1/flair1 | DirectionalMean_Exact | herbaceous vegetation | 57.3863 | -0.0862 | 90.8596 | 60.9022 | 21.4728 | -756 | -247 |
| flair1/flair1 | DirectionalMean_Exact | agricultural land | 12.9413 | -0.0133 | 13.8885 | 65.4881 | 1.4378 | 15 | 149 |
| flair1/flair1 | DirectionalMean_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8125 | 0 | 291 |
| flair1/flair1 | DirectionalShuffle0_Exact | building | 56.7634 | -0.0121 | 58.0204 | 96.3237 | 11.9192 | 27 | 102 |
| flair1/flair1 | DirectionalShuffle0_Exact | pervious surface | 47.0909 | -0.1844 | 91.7328 | 49.1780 | 9.1994 | -758 | -138 |
| flair1/flair1 | DirectionalShuffle0_Exact | impervious surface | 51.9764 | -0.0094 | 59.7462 | 79.9868 | 21.9324 | 119 | 325 |
| flair1/flair1 | DirectionalShuffle0_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1840 | 0 | 765 |
| flair1/flair1 | DirectionalShuffle0_Exact | water | 61.5349 | 0.0784 | 63.3414 | 95.5706 | 6.6931 | -17 | -212 |
| flair1/flair1 | DirectionalShuffle0_Exact | coniferous | 43.5159 | -0.0123 | 65.7157 | 56.2967 | 0.4816 | -1 | 2 |
| flair1/flair1 | DirectionalShuffle0_Exact | deciduous | 59.6336 | 0.1099 | 78.7240 | 71.0911 | 15.3713 | 781 | 527 |
| flair1/flair1 | DirectionalShuffle0_Exact | brushwood | 12.4102 | 0.0815 | 26.5810 | 18.8829 | 3.4940 | 6 | -988 |
| flair1/flair1 | DirectionalShuffle0_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0 | 4 |
| flair1/flair1 | DirectionalShuffle0_Exact | herbaceous vegetation | 57.3854 | -0.0871 | 90.8587 | 60.9016 | 21.4728 | -760 | -243 |
| flair1/flair1 | DirectionalShuffle0_Exact | agricultural land | 12.9479 | -0.0067 | 13.8947 | 65.5194 | 1.4378 | 17 | 148 |
| flair1/flair1 | DirectionalShuffle0_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8127 | 0 | 294 |
| flair1/flair1 | DirectionalShuffle1_Exact | building | 56.7627 | -0.0128 | 58.0204 | 96.3217 | 11.9190 | 24 | 100 |
| flair1/flair1 | DirectionalShuffle1_Exact | pervious surface | 47.1207 | -0.1546 | 91.7658 | 49.2010 | 9.2004 | -675 | -200 |
| flair1/flair1 | DirectionalShuffle1_Exact | impervious surface | 51.9944 | 0.0086 | 59.7592 | 80.0063 | 21.9330 | 186 | 270 |
| flair1/flair1 | DirectionalShuffle1_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1805 | 0 | 692 |
| flair1/flair1 | DirectionalShuffle1_Exact | water | 61.5288 | 0.0723 | 63.3359 | 95.5684 | 6.6936 | -19 | -201 |
| flair1/flair1 | DirectionalShuffle1_Exact | coniferous | 43.5273 | -0.0009 | 65.7418 | 56.2967 | 0.4814 | -1 | -2 |
| flair1/flair1 | DirectionalShuffle1_Exact | deciduous | 59.6363 | 0.1126 | 78.7307 | 71.0894 | 15.3696 | 775 | 498 |
| flair1/flair1 | DirectionalShuffle1_Exact | brushwood | 12.4207 | 0.0920 | 26.5890 | 18.9032 | 3.4967 | 27 | -952 |
| flair1/flair1 | DirectionalShuffle1_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0 | 3 |
| flair1/flair1 | DirectionalShuffle1_Exact | herbaceous vegetation | 57.3931 | -0.0794 | 90.8591 | 60.9101 | 21.4757 | -703 | -239 |
| flair1/flair1 | DirectionalShuffle1_Exact | agricultural land | 12.9507 | -0.0039 | 13.8979 | 65.5194 | 1.4375 | 17 | 141 |
| flair1/flair1 | DirectionalShuffle1_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8110 | 0 | 259 |
| flair1/flair1 | DirectionalShuffle2_Exact | building | 56.7436 | -0.0319 | 58.0036 | 96.3131 | 11.9214 | 11 | 163 |
| flair1/flair1 | DirectionalShuffle2_Exact | pervious surface | 47.0960 | -0.1793 | 91.7368 | 49.1824 | 9.1998 | -742 | -145 |
| flair1/flair1 | DirectionalShuffle2_Exact | impervious surface | 51.9736 | -0.0122 | 59.7491 | 79.9752 | 21.9282 | 79 | 276 |
| flair1/flair1 | DirectionalShuffle2_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1809 | 0 | 700 |
| flair1/flair1 | DirectionalShuffle2_Exact | water | 61.5261 | 0.0696 | 63.3325 | 95.5695 | 6.6940 | -18 | -193 |
| flair1/flair1 | DirectionalShuffle2_Exact | coniferous | 43.5359 | 0.0077 | 65.7613 | 56.2967 | 0.4812 | -1 | -5 |
| flair1/flair1 | DirectionalShuffle2_Exact | deciduous | 59.6324 | 0.1087 | 78.7288 | 71.0855 | 15.3691 | 761 | 502 |
| flair1/flair1 | DirectionalShuffle2_Exact | brushwood | 12.3845 | 0.0558 | 26.5073 | 18.8606 | 3.4996 | -17 | -848 |
| flair1/flair1 | DirectionalShuffle2_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0 | 3 |
| flair1/flair1 | DirectionalShuffle2_Exact | herbaceous vegetation | 57.3922 | -0.0803 | 90.8646 | 60.9065 | 21.4732 | -727 | -269 |
| flair1/flair1 | DirectionalShuffle2_Exact | agricultural land | 12.9427 | -0.0119 | 13.8887 | 65.5194 | 1.4384 | 17 | 161 |
| flair1/flair1 | DirectionalShuffle2_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8126 | 0 | 292 |

## Shared Cached-Source Cost

Suite wall 143.7659s. All40 numerical/source/reader/control tests and mask-free real-checkpoint smoke passed. Timings include all29 arms and exclude prior64-150.5 masked forwards/window; NOT standalone deployment latency.

| Dataset | Worker seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 26.2777 | 5569.1924 |
| potsdam | 23.9338 | 5565.6602 |
| udd5 | 19.5719 | 5562.2344 |
| oem | 28.2356 | 5573.5127 |
| loveda | 51.3201 | 5589.7495 |
| vaihingen | 21.8030 | 5562.2344 |
| landcoverai | 20.5931 | 5562.2344 |
| flair1 | 35.6574 | 5590.8799 |

All raw risk/directed/score fields remain remote. Merged results, per-image confusions/transitions and logs are local. No post-result gain/fill/seed/threshold tuning or domain routing. Prior failed gates remain failed.
