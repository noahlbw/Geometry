# Matched-Contribution Contextual Alias Source: Verified Screen

Same64 development image IDs, eight/domain, ONE top-left512 window/image and original whole-image wide observations. Frozen weights,20 aliases/class, unchanged Geometry/shared128px supports/writer/reconstruction. All actual mean-template logits/salience, source margins/risks and prediction fields saved before masks. All29 historical numerical/per-image endpoints replay exactly; actual unmasked local/wide prediction path checked. Corrected IRRG; LandCover.ai substitutes for unlabeled iSAID; LoveDA D counts once in domain mean. Not full-dataset, complete-image or independent validation.

## Prospective Decision

```json
{
  "passed": false,
  "checks": {
    "mean_gain": true,
    "domain_wins": true,
    "worst_protocol_loss": true,
    "above_ContrastReversal_Exact": false,
    "above_RivalPreserving_Exact": false,
    "above_RivalShuffledSupport_Exact": false,
    "above_RivalAliasShuffle0_Exact": true,
    "above_RivalAliasShuffle1_Exact": true,
    "above_RivalAliasShuffle2_Exact": true,
    "above_DirectionalMean_Exact": true,
    "above_DirectionalShuffle0_Exact": true,
    "above_DirectionalShuffle1_Exact": true,
    "above_DirectionalShuffle2_Exact": true,
    "above_MatchedMeanLogit": true,
    "above_MatchedShuffledSupport_Exact": false,
    "above_MatchedTextOnly_Exact": true,
    "above_MatchedAliasShuffle0_Exact": true,
    "above_MatchedAliasShuffle1_Exact": true,
    "above_MatchedAliasShuffle2_Exact": true,
    "above_MatchedDirectionalMean_Exact": true,
    "above_MatchedDirectionalShuffle0_Exact": true,
    "above_MatchedDirectionalShuffle1_Exact": true,
    "above_MatchedDirectionalShuffle2_Exact": true
  },
  "wins": 5,
  "mean_delta_pp": 0.4330253868062073,
  "worst_protocol_delta_pp": -0.0683121925649246,
  "no_automatic_full_rollout": true
}
```

FAILED promotion: retain original model. Do not relax the gate, choose a fill/seed or launch full datasets.

## Fixed-Class Matched Metrics

| Dataset/protocol | Geometry | Anchored_Exact | ContrastReversal_Exact | RivalPreserving_Exact | MatchedContribution_Exact | MatchedMeanLogit | MatchedShuffledSupport_Exact | MatchedTextOnly_Exact | MatchedAliasShuffle0_Exact | MatchedAliasShuffle1_Exact | MatchedAliasShuffle2_Exact | MatchedDirectionalMean_Exact | MatchedDirectionalShuffle0_Exact | MatchedDirectionalShuffle1_Exact | MatchedDirectionalShuffle2_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 55.619109 | 54.836328 | 54.192579 | 54.510755 | 55.558063 | 53.838820 | 53.901163 | 53.983981 | 54.024389 | 54.291933 | 54.267746 | 54.216695 | 54.235800 |
| potsdam/potsdam | 35.703546 | 38.920258 | 38.592874 | 38.832806 | 38.888837 | 38.948977 | 38.567637 | 39.607123 | 38.783466 | 38.878042 | 38.924406 | 38.880572 | 38.904587 | 38.891441 | 38.884233 |
| udd5/udd5 | 30.500979 | 28.175811 | 30.178447 | 29.942522 | 29.514001 | 29.626866 | 29.349219 | 27.892101 | 28.503573 | 29.181831 | 28.447684 | 28.713675 | 28.728075 | 28.683358 | 28.709316 |
| oem/oem | 39.820583 | 39.023152 | 39.232807 | 39.122037 | 39.085050 | 38.720748 | 38.820046 | 39.351841 | 39.080528 | 39.048332 | 39.082455 | 39.057212 | 39.056081 | 39.056386 | 39.056437 |
| loveda/P | 49.539928 | 50.706552 | 53.152193 | 51.919532 | 52.063326 | 42.904692 | 51.422121 | 50.776042 | 51.539392 | 51.145015 | 52.442798 | 50.307679 | 50.368499 | 49.796689 | 50.637254 |
| loveda/D | 33.877920 | 30.736030 | 32.743830 | 32.571870 | 32.467504 | 29.070146 | 33.424368 | 29.189692 | 31.995500 | 31.956498 | 31.705090 | 31.517042 | 31.360912 | 31.557418 | 31.536274 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.608847 | 51.777031 | 51.758650 | 50.685271 | 51.928278 | 51.807291 | 51.829442 | 51.839446 | 51.872479 | 51.838327 | 51.826691 | 51.815796 | 51.828226 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.865626 | 66.892068 | 66.889881 | 66.419588 | 66.345847 | 66.458741 | 66.888100 | 66.904010 | 66.877268 | 66.886971 | 66.890978 | 66.886034 | 66.895058 |
| flair1/flair1 | 35.605867 | 33.608404 | 33.545232 | 33.615973 | 33.611294 | 32.346716 | 33.515416 | 33.087267 | 33.612488 | 33.582721 | 33.636904 | 33.616255 | 33.614026 | 33.610046 | 33.613446 |
| Equal-domain mean | 40.531033 | 42.867949 | 43.548346 | 43.448829 | 43.300974 | 42.541133 | 43.438609 | 42.654109 | 43.074283 | 43.171858 | 43.071334 | 43.100248 | 43.081137 | 43.089647 | 43.094849 |

Class set is fixed to original union-positive classes. A disappearing scored class stays IoU0. Historical standard means are in summary.json; they differ only when scored unions differ.

## Mechanism Activity

| Dataset/protocol | Source | Mean risk | Active fraction | Unknown fraction | Directed magnitude | Potential magnitude |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | MatchedContribution_Exact | 0.028045 | 0.081207 | 0.918793 | 0.011482 | 0.016655 |
| vdd/vdd | MatchedShuffledSupport_Exact | 0.050746 | 0.133078 | 0.866922 | 0.019582 | 0.024549 |
| vdd/vdd | MatchedTextOnly_Exact | 0.000653 | 0.003061 | 0.996939 | 0.000448 | 0.000897 |
| potsdam/potsdam | MatchedContribution_Exact | 0.013620 | 0.035543 | 0.964457 | 0.005846 | 0.009381 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | 0.038155 | 0.101352 | 0.898648 | 0.018248 | 0.025769 |
| potsdam/potsdam | MatchedTextOnly_Exact | 0.020667 | 0.054167 | 0.945833 | 0.008299 | 0.014902 |
| udd5/udd5 | MatchedContribution_Exact | 0.023186 | 0.071101 | 0.928899 | 0.018628 | 0.025588 |
| udd5/udd5 | MatchedShuffledSupport_Exact | 0.044880 | 0.140398 | 0.859602 | 0.032212 | 0.035445 |
| udd5/udd5 | MatchedTextOnly_Exact | 0.031118 | 0.064000 | 0.936000 | 0.010498 | 0.020996 |
| oem/oem | MatchedContribution_Exact | 0.013068 | 0.037590 | 0.962410 | 0.005028 | 0.008195 |
| oem/oem | MatchedShuffledSupport_Exact | 0.053558 | 0.151274 | 0.848726 | 0.025521 | 0.026663 |
| oem/oem | MatchedTextOnly_Exact | 0.029940 | 0.094531 | 0.905469 | 0.007664 | 0.013457 |
| loveda/P | MatchedContribution_Exact | 0.020379 | 0.063851 | 0.936149 | 0.011256 | 0.016697 |
| loveda/P | MatchedShuffledSupport_Exact | 0.061886 | 0.180603 | 0.819397 | 0.034016 | 0.032352 |
| loveda/P | MatchedTextOnly_Exact | 0.008616 | 0.041667 | 0.958333 | 0.005726 | 0.010877 |
| loveda/D | MatchedContribution_Exact | 0.020781 | 0.065277 | 0.934723 | 0.012201 | 0.016429 |
| loveda/D | MatchedShuffledSupport_Exact | 0.059489 | 0.170894 | 0.829106 | 0.037319 | 0.036374 |
| loveda/D | MatchedTextOnly_Exact | 0.022427 | 0.071429 | 0.928571 | 0.019346 | 0.034609 |
| vaihingen/vaihingen | MatchedContribution_Exact | 0.018077 | 0.043217 | 0.956783 | 0.006661 | 0.010934 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | 0.043045 | 0.108584 | 0.891416 | 0.016742 | 0.023473 |
| vaihingen/vaihingen | MatchedTextOnly_Exact | 0.004561 | 0.016000 | 0.984000 | 0.001914 | 0.003829 |
| landcoverai/landcoverai | MatchedContribution_Exact | 0.010373 | 0.031306 | 0.968694 | 0.003683 | 0.006617 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | 0.034928 | 0.109829 | 0.890171 | 0.012321 | 0.017746 |
| landcoverai/landcoverai | MatchedTextOnly_Exact | 0.039133 | 0.088000 | 0.912000 | 0.020442 | 0.040885 |
| flair1/flair1 | MatchedContribution_Exact | 0.009103 | 0.025863 | 0.974137 | 0.002595 | 0.004283 |
| flair1/flair1 | MatchedShuffledSupport_Exact | 0.021413 | 0.056661 | 0.943339 | 0.006264 | 0.009165 |
| flair1/flair1 | MatchedTextOnly_Exact | 0.026228 | 0.091667 | 0.908333 | 0.011399 | 0.019247 |

## Beneficial And Harmful Dense Changes

| Dataset/protocol | Method | Beneficial | Harmful | Wrong-to-wrong |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | MatchedContribution_Exact | 11263 | 5054 | 4828 |
| vdd/vdd | ContrastReversal_Exact | 32597 | 11202 | 16644 |
| vdd/vdd | RivalPreserving_Exact | 14863 | 5245 | 7760 |
| vdd/vdd | MatchedShuffledSupport_Exact | 27693 | 15243 | 11530 |
| potsdam/potsdam | MatchedContribution_Exact | 4224 | 6476 | 2928 |
| potsdam/potsdam | ContrastReversal_Exact | 6658 | 17492 | 4969 |
| potsdam/potsdam | RivalPreserving_Exact | 4830 | 8340 | 2525 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | 16087 | 24681 | 12433 |
| udd5/udd5 | MatchedContribution_Exact | 12773 | 1290 | 20496 |
| udd5/udd5 | ContrastReversal_Exact | 27446 | 3929 | 28124 |
| udd5/udd5 | RivalPreserving_Exact | 23007 | 2773 | 25162 |
| udd5/udd5 | MatchedShuffledSupport_Exact | 23908 | 2780 | 28518 |
| oem/oem | MatchedContribution_Exact | 3368 | 1418 | 1311 |
| oem/oem | ContrastReversal_Exact | 13098 | 2295 | 2480 |
| oem/oem | RivalPreserving_Exact | 4808 | 862 | 1051 |
| oem/oem | MatchedShuffledSupport_Exact | 7127 | 12589 | 6087 |
| loveda/P | MatchedContribution_Exact | 2352 | 455 | 1037 |
| loveda/P | ContrastReversal_Exact | 3001 | 1286 | 1760 |
| loveda/P | RivalPreserving_Exact | 2242 | 711 | 1260 |
| loveda/P | MatchedShuffledSupport_Exact | 4403 | 5881 | 4996 |
| loveda/D | MatchedContribution_Exact | 50651 | 799 | 5560 |
| loveda/D | ContrastReversal_Exact | 51853 | 1697 | 7464 |
| loveda/D | RivalPreserving_Exact | 56926 | 1003 | 6919 |
| loveda/D | MatchedShuffledSupport_Exact | 116817 | 16076 | 22472 |
| vaihingen/vaihingen | MatchedContribution_Exact | 2222 | 4165 | 3367 |
| vaihingen/vaihingen | ContrastReversal_Exact | 5316 | 11486 | 6283 |
| vaihingen/vaihingen | RivalPreserving_Exact | 2684 | 4235 | 4087 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | 9867 | 12430 | 13967 |
| landcoverai/landcoverai | MatchedContribution_Exact | 974 | 765 | 19 |
| landcoverai/landcoverai | ContrastReversal_Exact | 1644 | 1432 | 29 |
| landcoverai/landcoverai | RivalPreserving_Exact | 791 | 629 | 28 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | 1165 | 7478 | 223 |
| flair1/flair1 | MatchedContribution_Exact | 1838 | 4089 | 2129 |
| flair1/flair1 | ContrastReversal_Exact | 4222 | 14203 | 5305 |
| flair1/flair1 | RivalPreserving_Exact | 1827 | 3042 | 1487 |
| flair1/flair1 | MatchedShuffledSupport_Exact | 2765 | 5966 | 2600 |

## Per-Class Outcome

| Dataset/protocol | Method | Class | IoU % | Delta pp | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -47.4915 | 70.4934 | 12.9511 | 9.3800 | -604535 | -114546 |
| vdd/vdd | Geometry | wall | 19.8084 | -39.8732 | 20.3031 | 89.0471 | 34.8907 | 4281 | 508235 |
| vdd/vdd | Geometry | road | 24.9744 | 1.7494 | 29.4846 | 62.0157 | 6.9810 | -24557 | -118754 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 25.6066 | 89.0479 | 76.2942 | 19.7359 | 150148 | 33618 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -46.4204 | 3.8880 | 100.0000 | 9.2105 | 6 | 178243 |
| vdd/vdd | Geometry | roof | 63.9822 | -22.2785 | 64.6648 | 98.3769 | 9.1667 | -1294 | 48679 |
| vdd/vdd | Geometry | water | 74.5988 | 21.7649 | 75.8754 | 97.7943 | 10.6353 | 11729 | -71253 |
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
| vdd/vdd | MatchedContribution_Exact | other | 60.0057 | 0.2291 | 80.9340 | 69.8844 | 44.0852 | 5058 | 3682 |
| vdd/vdd | MatchedContribution_Exact | wall | 59.2043 | -0.4773 | 65.4584 | 86.1047 | 10.4644 | -628 | 888 |
| vdd/vdd | MatchedContribution_Exact | road | 23.5191 | 0.2941 | 23.6439 | 97.8062 | 13.7296 | 355 | -2137 |
| vdd/vdd | MatchedContribution_Exact | vegetation | 44.4190 | 0.2764 | 95.1468 | 45.4488 | 11.0032 | 1140 | -513 |
| vdd/vdd | MatchedContribution_Exact | vehicle | 50.9475 | 0.6391 | 50.9787 | 99.8802 | 0.7016 | -3 | -193 |
| vdd/vdd | MatchedContribution_Exact | roof | 87.1398 | 0.8791 | 87.6411 | 99.3479 | 6.8303 | -67 | -1546 |
| vdd/vdd | MatchedContribution_Exact | water | 54.1125 | 1.2786 | 57.0854 | 91.2209 | 13.1858 | 354 | -6390 |
| vdd/vdd | MatchedShuffledSupport_Exact | other | 60.7356 | 0.9590 | 79.7895 | 71.7781 | 45.9293 | 25334 | 22080 |
| vdd/vdd | MatchedShuffledSupport_Exact | wall | 61.4645 | 1.7829 | 68.0231 | 86.4403 | 10.1091 | -68 | -7123 |
| vdd/vdd | MatchedShuffledSupport_Exact | road | 23.5923 | 0.3673 | 23.7253 | 97.6798 | 13.6648 | 267 | -3408 |
| vdd/vdd | MatchedShuffledSupport_Exact | vegetation | 42.0813 | -2.0613 | 95.8313 | 42.8660 | 10.3037 | -11337 | -2704 |
| vdd/vdd | MatchedShuffledSupport_Exact | vehicle | 56.7541 | 6.4457 | 56.9446 | 99.4141 | 0.6252 | -38 | -1761 |
| vdd/vdd | MatchedShuffledSupport_Exact | roof | 89.3313 | 3.0706 | 89.8575 | 99.3487 | 6.6618 | -66 | -5079 |
| vdd/vdd | MatchedShuffledSupport_Exact | water | 54.9474 | 2.1135 | 58.4917 | 90.0675 | 12.7060 | -1642 | -14455 |
| potsdam/potsdam | Geometry | impervious surface | 48.6703 | -17.2274 | 84.3816 | 53.4888 | 25.9112 | -176352 | -21321 |
| potsdam/potsdam | Geometry | building | 72.8222 | 1.7693 | 79.5088 | 89.6472 | 18.1787 | -11696 | -26826 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 12.1298 | 77.4031 | 28.7716 | 6.4458 | 50250 | 16882 |
| potsdam/potsdam | Geometry | tree | 49.3365 | -6.1699 | 84.7437 | 54.1457 | 11.2731 | -15594 | 17045 |
| potsdam/potsdam | Geometry | car | 11.6958 | -12.8420 | 11.6965 | 99.9455 | 30.6388 | 648 | 338941 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 3.0400 | 7.7773 | 13.2570 | 7.5523 | 3554 | -175531 |
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
| potsdam/potsdam | MatchedContribution_Exact | impervious surface | 65.4674 | -0.4303 | 85.5864 | 73.5799 | 35.1421 | -4123 | 35 |
| potsdam/potsdam | MatchedContribution_Exact | building | 71.0321 | -0.0208 | 74.9657 | 93.1211 | 20.0275 | 50 | 200 |
| potsdam/potsdam | MatchedContribution_Exact | low vegetation | 15.0059 | 0.5935 | 80.3797 | 15.5765 | 3.3604 | 2264 | 163 |
| potsdam/potsdam | MatchedContribution_Exact | tree | 55.4416 | -0.0648 | 92.0318 | 58.2371 | 11.1648 | -455 | -366 |
| potsdam/potsdam | MatchedContribution_Exact | car | 24.2702 | -0.2676 | 24.3222 | 99.1276 | 14.6136 | 33 | 3483 |
| potsdam/potsdam | MatchedContribution_Exact | clutter | 2.1157 | 0.0014 | 2.6568 | 9.4095 | 15.6916 | -21 | -1263 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | impervious surface | 67.2330 | 1.3353 | 85.7187 | 75.7141 | 36.1055 | 14172 | 1945 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | building | 72.0047 | 0.9518 | 75.9935 | 93.2057 | 19.7746 | 336 | -5390 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | low vegetation | 11.0769 | -3.3355 | 78.1110 | 11.4317 | 2.5379 | -12809 | -2014 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | tree | 53.1712 | -2.3352 | 92.8621 | 55.4370 | 10.5329 | -10816 | -3256 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | car | 25.8312 | 1.2934 | 25.8981 | 99.0106 | 13.7082 | -55 | -15417 |
| potsdam/potsdam | MatchedShuffledSupport_Exact | clutter | 2.0888 | -0.0255 | 2.5688 | 10.0541 | 17.3409 | 578 | 32726 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | 13.9675 | 89.6338 | 64.7597 | 2.2305 | 11032 | 2834 |
| udd5/udd5 | Geometry | building | 83.4755 | -0.0116 | 86.9865 | 95.3877 | 78.8720 | -65859 | -78647 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 | 0 | 39991 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -1.6538 | 3.7107 | 99.9773 | 11.3160 | 0 | 73162 |
| udd5/udd5 | Geometry | other | 5.0713 | -0.6763 | 23.0553 | 6.1044 | 5.3797 | -2330 | 19817 |
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
| udd5/udd5 | MatchedContribution_Exact | vegetation | 50.3423 | 4.0624 | 91.4799 | 52.8188 | 1.7825 | 3301 | 1170 |
| udd5/udd5 | MatchedContribution_Exact | building | 83.5751 | 0.0880 | 83.7877 | 99.6973 | 85.5826 | -854 | -2920 |
| udd5/udd5 | MatchedContribution_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3001 | 0 | 109 |
| udd5/udd5 | MatchedContribution_Exact | vehicle | 6.2846 | 0.9201 | 6.2847 | 99.9773 | 6.6814 | 0 | -24033 |
| udd5/udd5 | MatchedContribution_Exact | other | 7.3681 | 1.6205 | 31.5258 | 8.7719 | 5.6534 | 9036 | 14191 |
| udd5/udd5 | MatchedShuffledSupport_Exact | vegetation | 45.9205 | -0.3594 | 94.2729 | 47.2384 | 1.5470 | -312 | -157 |
| udd5/udd5 | MatchedShuffledSupport_Exact | building | 83.6916 | 0.2045 | 83.9492 | 99.6347 | 85.3643 | -1798 | -6554 |
| udd5/udd5 | MatchedShuffledSupport_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4168 | 0 | 2555 |
| udd5/udd5 | MatchedShuffledSupport_Exact | vehicle | 7.1642 | 1.7997 | 7.1643 | 99.9773 | 5.8610 | 0 | -41237 |
| udd5/udd5 | MatchedShuffledSupport_Exact | other | 9.9697 | 4.2221 | 36.1106 | 12.1049 | 6.8110 | 23238 | 24265 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 | 0 | 14184 |
| oem/oem | Geometry | rangeland | 41.5417 | -7.9844 | 54.4153 | 63.7145 | 17.0320 | 3770 | 81071 |
| oem/oem | Geometry | developed space | 25.6980 | 1.9311 | 65.5490 | 29.7108 | 8.8828 | -32737 | -175342 |
| oem/oem | Geometry | road | 52.3963 | -2.1566 | 60.6188 | 79.4359 | 8.2389 | 7878 | 22137 |
| oem/oem | Geometry | tree | 65.2556 | 9.5951 | 84.8180 | 73.8857 | 23.0134 | 82580 | 43050 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 | 0 | 1153 |
| oem/oem | Geometry | agriculture land | 71.2323 | -10.4177 | 95.7971 | 73.5302 | 16.2828 | -112388 | -80583 |
| oem/oem | Geometry | building | 62.4408 | 15.4121 | 64.4677 | 95.2061 | 17.5919 | 85337 | 59890 |
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
| oem/oem | MatchedContribution_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1790 | 0 | -577 |
| oem/oem | MatchedContribution_Exact | rangeland | 49.4089 | -0.1172 | 70.5987 | 62.2096 | 12.8177 | -701 | -530 |
| oem/oem | MatchedContribution_Exact | developed space | 23.8582 | 0.0913 | 39.0926 | 37.9735 | 19.0366 | 335 | -1037 |
| oem/oem | MatchedContribution_Exact | road | 54.5039 | -0.0490 | 68.0160 | 73.2875 | 6.7745 | -17 | 124 |
| oem/oem | MatchedContribution_Exact | tree | 55.8559 | 0.1954 | 91.6490 | 58.8511 | 16.9643 | 1459 | 626 |
| oem/oem | MatchedContribution_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | MatchedContribution_Exact | agriculture land | 81.7895 | 0.1395 | 82.1550 | 99.4590 | 25.6818 | -49 | -960 |
| oem/oem | MatchedContribution_Exact | building | 47.2641 | 0.2354 | 68.3467 | 60.5091 | 10.5461 | 923 | 404 |
| oem/oem | MatchedShuffledSupport_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1324 | 0 | -1528 |
| oem/oem | MatchedShuffledSupport_Exact | rangeland | 49.1360 | -0.3901 | 70.5169 | 61.8403 | 12.7564 | -1798 | -685 |
| oem/oem | MatchedShuffledSupport_Exact | developed space | 24.1166 | 0.3497 | 38.4546 | 39.2765 | 20.0165 | 5550 | 13760 |
| oem/oem | MatchedShuffledSupport_Exact | road | 54.4315 | -0.1214 | 68.0241 | 73.1473 | 6.7607 | -197 | 23 |
| oem/oem | MatchedShuffledSupport_Exact | tree | 55.1298 | -0.5307 | 91.9700 | 57.9176 | 16.6369 | -3578 | -1023 |
| oem/oem | MatchedShuffledSupport_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| oem/oem | MatchedShuffledSupport_Exact | agriculture land | 82.2982 | 0.6482 | 82.6965 | 99.4181 | 25.5031 | -226 | -4432 |
| oem/oem | MatchedShuffledSupport_Exact | building | 45.4483 | -1.5804 | 67.7608 | 57.9870 | 10.1939 | -5213 | -653 |
| loveda/P | Geometry | building | 7.6311 | -53.2168 | 7.6311 | 100.0000 | 0.3472 | 63 | 3622 |
| loveda/P | Geometry | road | 63.0522 | 9.2969 | 64.4734 | 96.6222 | 12.4720 | -171 | -25873 |
| loveda/P | Geometry | water | 63.2131 | 0.8040 | 68.9718 | 88.3328 | 18.8451 | 314 | -2566 |
| loveda/P | Geometry | barren | 1.9144 | -2.4071 | 34.2731 | 1.9874 | 0.6429 | -3026 | 4258 |
| loveda/P | Geometry | tree | 72.0056 | 32.0503 | 88.8915 | 79.1256 | 14.6560 | 74529 | 18377 |
| loveda/P | Geometry | farm | 89.4231 | 6.4729 | 91.1653 | 97.9076 | 53.0368 | -9359 | -60168 |
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
| loveda/P | MatchedContribution_Exact | building | 67.8010 | 6.9531 | 77.5449 | 84.3648 | 0.0288 | 15 | -19 |
| loveda/P | MatchedContribution_Exact | road | 53.7648 | 0.0095 | 54.7418 | 96.7871 | 14.7142 | -12 | -53 |
| loveda/P | MatchedContribution_Exact | water | 62.3666 | -0.0425 | 68.1164 | 88.0788 | 19.0268 | -119 | -27 |
| loveda/P | MatchedContribution_Exact | barren | 4.2157 | -0.1058 | 89.6393 | 4.2363 | 0.5240 | -137 | -9 |
| loveda/P | MatchedContribution_Exact | tree | 41.0541 | 1.0988 | 99.3096 | 41.1716 | 6.8260 | 2126 | 60 |
| loveda/P | MatchedContribution_Exact | farm | 83.1777 | 0.2275 | 83.4934 | 99.5475 | 58.8802 | 24 | -1849 |
| loveda/P | MatchedShuffledSupport_Exact | building | 67.3469 | 6.4990 | 68.9095 | 96.7427 | 0.0372 | 53 | 40 |
| loveda/P | MatchedShuffledSupport_Exact | road | 51.5079 | -2.2474 | 52.3887 | 96.8389 | 15.3834 | 38 | 7650 |
| loveda/P | MatchedShuffledSupport_Exact | water | 61.8057 | -0.6034 | 67.5698 | 87.8717 | 19.1357 | -472 | 1587 |
| loveda/P | MatchedShuffledSupport_Exact | barren | 4.9029 | 0.5814 | 85.7645 | 4.9431 | 0.6390 | 771 | 416 |
| loveda/P | MatchedShuffledSupport_Exact | tree | 38.9955 | -0.9598 | 98.8791 | 39.1686 | 6.5222 | -1695 | 361 |
| loveda/P | MatchedShuffledSupport_Exact | farm | 83.9738 | 1.0236 | 84.3203 | 99.5131 | 58.2826 | -173 | -8576 |
| loveda/D | Geometry | background | 39.0474 | 31.7250 | 75.8250 | 44.5998 | 26.3232 | 348157 | 110196 |
| loveda/D | Geometry | building | 4.8538 | -26.4283 | 4.8538 | 100.0000 | 0.3016 | 63 | 5545 |
| loveda/D | Geometry | road | 45.0691 | 0.9421 | 45.8485 | 96.3650 | 9.6637 | -404 | -5317 |
| loveda/D | Geometry | water | 53.8049 | -0.6330 | 58.0149 | 88.1157 | 12.3473 | -36 | 3180 |
| loveda/D | Geometry | barren | 0.0293 | -3.2121 | 0.8665 | 0.0304 | 0.2146 | -4154 | 3564 |
| loveda/D | Geometry | tree | 37.0607 | 1.1933 | 42.3596 | 74.7642 | 16.0550 | 73472 | 192040 |
| loveda/D | Geometry | farm | 57.2802 | 18.4061 | 64.7327 | 83.2647 | 35.0945 | -93073 | -633233 |
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
| loveda/D | MatchedContribution_Exact | background | 12.1342 | 4.8118 | 83.7163 | 12.4275 | 6.6434 | 46210 | -572 |
| loveda/D | MatchedContribution_Exact | building | 35.7735 | 4.4914 | 38.3136 | 84.3648 | 0.0322 | 15 | -56 |
| loveda/D | MatchedContribution_Exact | road | 43.9558 | -0.1712 | 44.6111 | 96.7663 | 9.9731 | -17 | 785 |
| loveda/D | MatchedContribution_Exact | water | 54.1747 | -0.2632 | 58.4726 | 88.0530 | 12.2419 | -143 | 1077 |
| loveda/D | MatchedContribution_Exact | barren | 3.0956 | -0.1458 | 80.8803 | 3.1185 | 0.2362 | -187 | 49 |
| loveda/D | MatchedContribution_Exact | tree | 37.8469 | 1.9795 | 96.8887 | 38.3126 | 3.5970 | 3935 | 313 |
| loveda/D | MatchedContribution_Exact | farm | 40.2918 | 1.4177 | 40.3673 | 99.5379 | 67.2761 | 39 | -51448 |
| loveda/D | MatchedShuffledSupport_Exact | background | 18.5762 | 11.2538 | 78.8511 | 19.5502 | 11.0959 | 113059 | 25954 |
| loveda/D | MatchedShuffledSupport_Exact | building | 42.2475 | 10.9654 | 42.8571 | 96.7427 | 0.0330 | 53 | -77 |
| loveda/D | MatchedShuffledSupport_Exact | road | 42.7639 | -1.3631 | 43.3761 | 96.8047 | 10.2612 | 20 | 6788 |
| loveda/D | MatchedShuffledSupport_Exact | water | 54.1370 | -0.3009 | 58.5602 | 87.7562 | 12.1824 | -649 | 335 |
| loveda/D | MatchedShuffledSupport_Exact | barren | 3.9526 | 0.7112 | 81.9782 | 3.9872 | 0.2979 | 929 | 228 |
| loveda/D | MatchedShuffledSupport_Exact | tree | 29.5376 | -6.3298 | 95.9061 | 29.9148 | 2.8373 | -12085 | 402 |
| loveda/D | MatchedShuffledSupport_Exact | farm | 42.7558 | 3.8817 | 42.8611 | 99.4287 | 63.2922 | -586 | -134371 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -15.2111 | 77.7719 | 47.8510 | 16.8072 | -137882 | -67750 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 7.0164 | 73.5287 | 98.5836 | 27.6605 | -4810 | -69879 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 9.2961 | 92.4475 | 55.4111 | 17.7225 | 67731 | 17105 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.3793 | 82.7688 | 80.0845 | 19.9693 | -7441 | -21418 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -14.7902 | 10.3220 | 99.7520 | 17.8406 | 784 | 223560 |
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
| vaihingen/vaihingen | MatchedContribution_Exact | impervious surface | 57.1977 | -0.1075 | 73.6677 | 71.8972 | 26.6602 | -128 | 1127 |
| vaihingen/vaihingen | MatchedContribution_Exact | building | 65.5576 | -0.1861 | 65.6883 | 99.6974 | 31.3118 | 9 | 1876 |
| vaihingen/vaihingen | MatchedContribution_Exact | low vegetation | 43.3013 | -0.4140 | 96.3057 | 44.0327 | 13.5190 | -2825 | -491 |
| vaihingen/vaihingen | MatchedContribution_Exact | tree | 67.2085 | -0.0527 | 78.8076 | 82.0349 | 21.4838 | 1001 | 1902 |
| vaihingen/vaihingen | MatchedContribution_Exact | car | 25.5281 | 0.4186 | 25.6806 | 97.7270 | 7.0252 | 0 | -2471 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | impervious surface | 56.6604 | -0.6448 | 71.2030 | 73.5042 | 28.1995 | 9078 | 24204 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | building | 66.4375 | 0.6938 | 66.5792 | 99.6806 | 30.8876 | -64 | -6948 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | low vegetation | 42.1993 | -1.5160 | 96.3971 | 42.8756 | 13.1513 | -10000 | -1028 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | tree | 67.3691 | 0.1079 | 79.5833 | 81.4455 | 21.1215 | -1550 | -3144 |
| vaihingen/vaihingen | MatchedShuffledSupport_Exact | car | 26.9751 | 1.8656 | 27.1508 | 97.6572 | 6.6401 | -27 | -10521 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -4.8387 | 96.6505 | 85.0437 | 59.6226 | -109633 | -44423 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.5110 | 35.0436 | 99.2919 | 4.2927 | 316 | 20014 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -0.2388 | 86.4992 | 89.2841 | 22.0960 | 32579 | 43002 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9543 | 93.6635 | 100.0000 | 8.8309 | 0 | 7502 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -11.4629 | 15.6288 | 77.0019 | 5.1578 | 308 | 50335 |
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
| landcoverai/landcoverai | MatchedContribution_Exact | background | 87.4537 | 0.0060 | 93.9147 | 92.7071 | 66.8887 | -734 | -942 |
| landcoverai/landcoverai | MatchedContribution_Exact | building | 44.3803 | -0.0869 | 44.7229 | 98.3036 | 3.3302 | 2 | 142 |
| landcoverai/landcoverai | MatchedContribution_Exact | woodland | 78.7741 | 0.1715 | 94.9267 | 82.2363 | 18.5450 | 939 | 172 |
| landcoverai/landcoverai | MatchedContribution_Exact | water | 97.6189 | 0.0011 | 97.6189 | 100.0000 | 8.4731 | 0 | -2 |
| landcoverai/landcoverai | MatchedContribution_Exact | road | 26.2223 | -0.1724 | 28.6456 | 75.6081 | 2.7631 | 2 | 421 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | background | 87.0620 | -0.3857 | 93.6120 | 92.5610 | 66.9992 | -2809 | 3452 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | building | 43.3693 | -1.0979 | 43.6876 | 98.3477 | 3.4106 | 16 | 1815 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | woodland | 77.9591 | -0.6435 | 95.0749 | 81.2400 | 18.2918 | -3534 | -666 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | water | 97.6332 | 0.0154 | 97.6332 | 100.0000 | 8.4718 | 0 | -28 |
| landcoverai/landcoverai | MatchedShuffledSupport_Exact | road | 25.7057 | -0.6890 | 28.0227 | 75.6627 | 2.8265 | 14 | 1740 |
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
| flair1/flair1 | MatchedContribution_Exact | building | 56.7009 | -0.0746 | 57.9522 | 96.3317 | 11.9342 | 39 | 405 |
| flair1/flair1 | MatchedContribution_Exact | pervious surface | 47.1936 | -0.0817 | 91.7779 | 49.2769 | 9.2133 | -402 | -201 |
| flair1/flair1 | MatchedContribution_Exact | impervious surface | 52.0631 | 0.0773 | 59.8884 | 79.9376 | 21.8669 | -50 | -880 |
| flair1/flair1 | MatchedContribution_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1770 | 0 | 619 |
| flair1/flair1 | MatchedContribution_Exact | water | 61.7063 | 0.2498 | 63.5106 | 95.5985 | 6.6773 | 9 | -571 |
| flair1/flair1 | MatchedContribution_Exact | coniferous | 43.6468 | 0.1186 | 66.0263 | 56.2882 | 0.4792 | -2 | -46 |
| flair1/flair1 | MatchedContribution_Exact | deciduous | 59.6840 | 0.1603 | 78.7445 | 71.1460 | 15.3791 | 977 | 496 |
| flair1/flair1 | MatchedContribution_Exact | brushwood | 12.2882 | -0.0405 | 25.8888 | 18.9566 | 3.6014 | 82 | 1188 |
| flair1/flair1 | MatchedContribution_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0021 | 0 | 10 |
| flair1/flair1 | MatchedContribution_Exact | herbaceous vegetation | 57.1282 | -0.3443 | 90.9254 | 60.5824 | 21.3446 | -2904 | -787 |
| flair1/flair1 | MatchedContribution_Exact | agricultural land | 12.9245 | -0.0301 | 13.8797 | 65.2534 | 1.4335 | 0 | 75 |
| flair1/flair1 | MatchedContribution_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8913 | 0 | 1943 |
| flair1/flair1 | MatchedShuffledSupport_Exact | building | 56.7032 | -0.0723 | 57.9664 | 96.2991 | 11.9273 | -10 | 308 |
| flair1/flair1 | MatchedShuffledSupport_Exact | pervious surface | 46.6053 | -0.6700 | 91.9660 | 48.5833 | 9.0651 | -2897 | -814 |
| flair1/flair1 | MatchedShuffledSupport_Exact | impervious surface | 51.9288 | -0.0570 | 59.6193 | 80.1024 | 22.0109 | 516 | 1573 |
| flair1/flair1 | MatchedShuffledSupport_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2903 | 0 | 2994 |
| flair1/flair1 | MatchedShuffledSupport_Exact | water | 61.4536 | -0.0029 | 63.3147 | 95.4351 | 6.6865 | -143 | -226 |
| flair1/flair1 | MatchedShuffledSupport_Exact | coniferous | 43.6656 | 0.1374 | 66.4953 | 55.9827 | 0.4733 | -38 | -135 |
| flair1/flair1 | MatchedShuffledSupport_Exact | deciduous | 59.2201 | -0.3036 | 78.7688 | 70.4684 | 15.2280 | -1441 | -255 |
| flair1/flair1 | MatchedShuffledSupport_Exact | brushwood | 11.9931 | -0.3356 | 25.8292 | 18.2932 | 3.4834 | -602 | -602 |
| flair1/flair1 | MatchedShuffledSupport_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0022 | 0 | 13 |
| flair1/flair1 | MatchedShuffledSupport_Exact | herbaceous vegetation | 57.5880 | 0.1155 | 90.6451 | 61.2268 | 21.6384 | 1424 | 1043 |
| flair1/flair1 | MatchedShuffledSupport_Exact | agricultural land | 13.0271 | 0.0725 | 14.0054 | 65.0970 | 1.4173 | -10 | -256 |
| flair1/flair1 | MatchedShuffledSupport_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7776 | 0 | -442 |

## Actual Observation Cost

Suite wall398.4252s. All50 tests and mask-free real-checkpoint smoke passed. Worker times include new intervention observations,40-arm decoding and cache persistence, NOT standalone primary deployment latency.

| Dataset | Worker seconds | True masked forwards | Support-control masked forwards | True observation GPU seconds | Control GPU seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 95.2214 | 512 | 512 | 11.8772 | 11.8682 | 5569.7939 |
| potsdam | 151.7937 | 1152 | 1152 | 27.2713 | 27.2736 | 5566.1758 |
| udd5 | 71.5290 | 512 | 512 | 11.8004 | 11.7919 | 5562.6641 |
| oem | 198.8594 | 1204 | 1204 | 27.3232 | 27.3019 | 5574.2002 |
| loveda | 273.1568 | 1152 | 1152 | 27.7964 | 27.7782 | 5590.8667 |
| vaihingen | 132.7888 | 1152 | 1152 | 25.5112 | 25.4999 | 5562.6641 |
| landcoverai | 137.6352 | 1152 | 1152 | 27.7948 | 27.7757 | 5562.6641 |
| flair1 | 283.3176 | 1152 | 1152 | 26.9241 | 26.9127 | 5844.0029 |

Actual source logits/salience/margins remain remote. Local merged files, confusions, transitions and logs are verified. The source uses original templates/profiling and rival class aggregation; it is not a semantic correctness certificate. Its class-margin identity holds at a wide token, not after nonlinear alias aggregation of interpolated margins. The unchanged writer still admits query risks across crop contributions. Canonical protection does not repair false canonical responses. The test rejects or supports this complete source, not the general existence of useful alias admission.
