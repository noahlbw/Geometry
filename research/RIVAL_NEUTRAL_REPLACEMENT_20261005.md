# Rival-neutral alias replacement

Same64 developed top-left512 windows with full-image wide context. NOT full datasets or independent validation. Geometry/finite VIP wide/fine/all20/risk support/reconstruction frozen; five earlier scores and per-image endpoints exact. Scores persist before masks; LoveDA D once, P separate; common scored-class support.

On original eligible alias/rival slots, primary caps the wide alias at the original rival log-mean-exp. Fine-cap alternative adds the original sampled fine alias/rival margin. All K slots remain; no survivor renormalization. Source class deltas cannot increase, but pair antisymmetrization/solver/H can increase final class scores. Hard fixed-mass/unprojected/strength-matched endpoints are controls, not promotable new candidates.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalProjected_Exact | FineBudgetOnly_Exact | RivalNeutral_Projected | RivalFineCap_Projected | HardFixedMass_Projected | NeutralUnprojected | ProjectedHardStrengthMatched | NeutralAliasShuffle0 | NeutralAliasShuffle1 | NeutralAliasShuffle2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.3441 | 54.1183 | 54.1333 | 53.7933 | 50.7883 | 49.4830 | 53.9544 | 54.5849 | 53.2510 | 53.4981 | 52.9984 |
| potsdam/potsdam | 35.7035 | 38.9203 | 40.6173 | 41.1231 | 40.9475 | 41.0258 | 42.1928 | 43.9302 | 41.0302 | 41.1951 | 39.8409 | 40.1079 | 40.2463 |
| udd5/udd5 | 30.5010 | 28.1758 | 34.0394 | 34.6174 | 34.6906 | 33.5597 | 35.5119 | 36.4177 | 33.3218 | 34.2813 | 29.3710 | 30.5794 | 29.3562 |
| oem/oem | 39.8205 | 39.0232 | 39.7906 | 40.4484 | 40.1611 | 40.4522 | 41.4402 | 42.1255 | 40.4410 | 40.1721 | 40.0416 | 39.9928 | 39.9898 |
| loveda/P | 49.5399 | 50.7066 | 52.5879 | 52.6436 | 52.0588 | 51.6984 | 46.2235 | 45.7072 | 51.7371 | 51.6470 | 50.1998 | 50.3967 | 50.9854 |
| loveda/D | 33.8779 | 30.7360 | 37.2156 | 37.6023 | 37.2513 | 36.7269 | 37.9298 | 38.9757 | 36.7815 | 37.1760 | 35.4501 | 35.5559 | 35.3061 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 52.9446 | 53.3059 | 53.3440 | 53.5057 | 54.5369 | 55.6349 | 53.5122 | 53.5418 | 53.0884 | 53.0452 | 53.2211 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.7836 | 67.6095 | 67.6475 | 68.2617 | 68.2753 | 67.6407 | 67.5346 | 67.5117 | 67.5288 | 67.3180 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 34.3507 | 34.4546 | 34.2658 | 34.5566 | 35.1708 | 34.2613 | 34.2250 | 34.1518 | 34.1027 | 34.2834 |
| Eight-domain mean | 40.5310 | 42.8679 | 45.0622 | 45.4187 | 45.3240 | 45.1221 | 45.6523 | 46.2516 | 45.1179 | 45.3389 | 44.0883 | 44.3014 | 44.0899 |

## Predeclared Gates

```json
{
  "RivalNeutral_Projected": {
    "mean_gain_vs_projected_pp": -0.29660383121536427,
    "worst_protocol_gain_pp": -1.0576874141326087,
    "gain_vs_strongest_control_pp": -1.1295070486999208,
    "gain_vs_primary_alias_null_pp": 0.9622617034267549,
    "own_null_matched": true,
    "accuracy_gate": false,
    "mechanism_gate": false
  },
  "RivalFineCap_Projected": {
    "mean_gain_vs_projected_pp": 0.23355199874944788,
    "worst_protocol_gain_pp": -6.420100479704956,
    "gain_vs_strongest_control_pp": -0.5993512187351087,
    "gain_vs_primary_alias_null_pp": 1.492417533391567,
    "own_null_matched": false,
    "accuracy_gate": false,
    "mechanism_gate": false
  }
}
```

Matched alias-null applies to primary. Fine-cap needs its own matched null before a mechanism claim. No automatic full20092 rollout, label fitting, diagnostic promotion or retained-model replacement.

## Source Normalization Diagnostics

| Dataset/protocol | Active hard actions increasing own class | Mean normalization term | Neutral source absolute action |
| --- | ---: | ---: | ---: |
| vdd/vdd | 0.378068 | 0.284932 | 0.057635 |
| potsdam/potsdam | 0.402262 | 0.189639 | 0.028391 |
| udd5/udd5 | 0.295421 | 0.184069 | 0.067241 |
| oem/oem | 0.456297 | 0.169522 | 0.023973 |
| loveda/P | 0.363418 | 0.258631 | 0.077677 |
| loveda/D | 0.361603 | 0.245705 | 0.076223 |
| vaihingen/vaihingen | 0.367243 | 0.218857 | 0.030565 |
| landcoverai/landcoverai | 0.482358 | 0.199507 | 0.020552 |
| flair1/flair1 | 0.465683 | 0.158789 | 0.020746 |

The fraction averages per-window active-pair fractions. Positive directed source actions are not automatically harmful. Normalizer decomposition errors<=1e-10, primary/fixed source-positive maxima<=1e-12, canonical risk0 and all shuffled per-query/class/rival rejected counts exactly preserved.

## Independent Window-Context Cost

The fixed-mass diagnostic additionally executes the original hard writer to verify the normalization decomposition. Its recorded latency includes that audit and is not an optimized fixed-mass deployment latency; no speed ranking is inferred from it.

Warmed, synchronized, alternated three repetitions, loading/text encoding/GT excluded; resident backbones included. CPU singleton score checks are included for candidates/fixed-mass but absent for hard; do not claim tiny differences. Every singleton score is bitwise equal to all-arm scores in warmup and three timed runs. No unused primary writer runs on singleton alternatives. Fine visual forwards remain.

| Dataset | Cached hard s | Neutral s | Fine cap s | Fixed-mass control s |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.325801 | 0.327201 | 0.328517 | 0.315691 |
| potsdam | 0.274070 | 0.275941 | 0.277515 | 0.286162 |
| udd5 | 0.336703 | 0.337105 | 0.339778 | 0.347463 |
| oem | 0.280418 | 0.282338 | 0.289230 | 0.301591 |
| loveda | 0.324404 | 0.330632 | 0.334444 | 0.349973 |
| vaihingen | 0.273973 | 0.276623 | 0.277397 | 0.287569 |
| landcoverai | 0.274818 | 0.278536 | 0.281263 | 0.292344 |
| flair1 | 0.284380 | 0.288534 | 0.292109 | 0.304960 |

| Dataset | Cached hard MiB | Neutral MiB | Fine cap MiB | Fixed-mass MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 5566.000 | 5566.000 | 5566.000 | 5566.000 |
| potsdam | 5564.028 | 5564.028 | 5564.028 | 5564.028 |
| udd5 | 5556.948 | 5556.948 | 5556.948 | 5556.948 |
| oem | 5573.711 | 5573.711 | 5573.711 | 5573.711 |
| loveda | 5594.520 | 5594.520 | 5594.520 | 5594.520 |
| vaihingen | 5559.755 | 5559.755 | 5559.755 | 5559.755 |
| landcoverai | 5559.755 | 5559.755 | 5559.755 | 5559.755 |
| flair1 | 5797.519 | 5797.519 | 5797.519 | 5797.519 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs projected hard | Precision | Recall | Area |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -42.7749 | 70.4934 | 12.9511 | 9.3800 |
| vdd/vdd | Geometry | wall | 19.8085 | -33.3364 | 20.3032 | 89.0471 | 34.8906 |
| vdd/vdd | Geometry | road | 24.9744 | 3.3126 | 29.4846 | 62.0157 | 6.9810 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 15.4815 | 89.0479 | 76.2942 | 19.7359 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -34.3717 | 3.8880 | 100.0000 | 9.2105 |
| vdd/vdd | Geometry | roof | 63.9827 | -25.8988 | 64.6650 | 98.3777 | 9.1667 |
| vdd/vdd | Geometry | water | 74.5988 | 8.0463 | 75.8754 | 97.7943 | 10.6353 |
| vdd/vdd | NoAdmission_Exact | other | 59.7766 | 4.7166 | 81.1541 | 69.4120 | 43.6684 |
| vdd/vdd | NoAdmission_Exact | wall | 59.6816 | 6.5367 | 65.8225 | 86.4811 | 10.4520 |
| vdd/vdd | NoAdmission_Exact | road | 23.2250 | 1.5632 | 23.3759 | 97.2962 | 13.8146 |
| vdd/vdd | NoAdmission_Exact | vegetation | 44.1426 | -10.1251 | 94.9106 | 45.2128 | 10.9733 |
| vdd/vdd | NoAdmission_Exact | vehicle | 50.3084 | 12.0487 | 50.3286 | 99.9201 | 0.7110 |
| vdd/vdd | NoAdmission_Exact | roof | 86.2607 | -3.6208 | 86.7114 | 99.4009 | 6.9072 |
| vdd/vdd | NoAdmission_Exact | water | 52.8339 | -13.7186 | 55.7407 | 91.0164 | 13.4736 |
| vdd/vdd | RivalFineHard_Exact | other | 57.2425 | 2.1825 | 81.0538 | 66.0849 | 41.6267 |
| vdd/vdd | RivalFineHard_Exact | wall | 58.4390 | 5.2941 | 63.3706 | 88.2481 | 11.0782 |
| vdd/vdd | RivalFineHard_Exact | road | 21.3171 | -0.3447 | 21.3725 | 98.7975 | 15.3427 |
| vdd/vdd | RivalFineHard_Exact | vegetation | 50.4296 | -3.8381 | 95.0722 | 51.7831 | 12.5465 |
| vdd/vdd | RivalFineHard_Exact | vehicle | 42.1815 | 3.9218 | 42.1815 | 100.0000 | 0.8490 |
| vdd/vdd | RivalFineHard_Exact | roof | 89.7289 | -0.1526 | 90.1757 | 99.4508 | 6.6452 |
| vdd/vdd | RivalFineHard_Exact | water | 61.0704 | -5.4821 | 64.1803 | 92.6489 | 11.9117 |
| vdd/vdd | FineRivalProjected_Exact | other | 55.0600 | 0.0000 | 80.8961 | 63.2893 | 39.9435 |
| vdd/vdd | FineRivalProjected_Exact | wall | 53.1449 | 0.0000 | 57.0418 | 88.6096 | 12.3578 |
| vdd/vdd | FineRivalProjected_Exact | road | 21.6618 | 0.0000 | 21.7082 | 99.0245 | 15.1402 |
| vdd/vdd | FineRivalProjected_Exact | vegetation | 54.2677 | 0.0000 | 94.7736 | 55.9419 | 13.5969 |
| vdd/vdd | FineRivalProjected_Exact | vehicle | 38.2597 | 0.0000 | 38.2597 | 100.0000 | 0.9360 |
| vdd/vdd | FineRivalProjected_Exact | roof | 89.8815 | 0.0000 | 90.3136 | 99.4706 | 6.6363 |
| vdd/vdd | FineRivalProjected_Exact | water | 66.5525 | 0.0000 | 68.9089 | 95.1129 | 11.3894 |
| vdd/vdd | FineBudgetOnly_Exact | other | 56.2592 | 1.1992 | 80.7432 | 64.9776 | 41.0867 |
| vdd/vdd | FineBudgetOnly_Exact | wall | 54.3088 | 1.1639 | 58.4429 | 88.4759 | 12.0433 |
| vdd/vdd | FineBudgetOnly_Exact | road | 22.3716 | 0.7098 | 22.4247 | 98.9527 | 14.6457 |
| vdd/vdd | FineBudgetOnly_Exact | vegetation | 52.7891 | -1.4786 | 95.4959 | 54.1370 | 13.0587 |
| vdd/vdd | FineBudgetOnly_Exact | vehicle | 38.4045 | 0.1448 | 38.4045 | 100.0000 | 0.9325 |
| vdd/vdd | FineBudgetOnly_Exact | roof | 90.4037 | 0.5222 | 90.8606 | 99.4468 | 6.5948 |
| vdd/vdd | FineBudgetOnly_Exact | water | 64.3964 | -2.1561 | 66.9438 | 94.4206 | 11.6384 |
| vdd/vdd | RivalNeutral_Projected | other | 55.2626 | 0.2026 | 80.5862 | 63.7496 | 40.3887 |
| vdd/vdd | RivalNeutral_Projected | wall | 52.1863 | -0.9586 | 55.9576 | 88.5628 | 12.5906 |
| vdd/vdd | RivalNeutral_Projected | road | 22.9005 | 1.2387 | 22.9440 | 99.1797 | 14.3471 |
| vdd/vdd | RivalNeutral_Projected | vegetation | 53.7982 | -0.4695 | 95.1163 | 55.3264 | 13.3988 |
| vdd/vdd | RivalNeutral_Projected | vehicle | 36.6074 | -1.6523 | 36.6074 | 100.0000 | 0.9782 |
| vdd/vdd | RivalNeutral_Projected | roof | 90.1866 | 0.3051 | 90.6393 | 99.4492 | 6.6111 |
| vdd/vdd | RivalNeutral_Projected | water | 65.6117 | -0.9408 | 67.5935 | 95.7226 | 11.6855 |
| vdd/vdd | RivalFineCap_Projected | other | 46.1576 | -8.9024 | 79.7304 | 52.2941 | 33.4867 |
| vdd/vdd | RivalFineCap_Projected | wall | 37.7453 | -15.3996 | 39.6473 | 88.7235 | 17.8023 |
| vdd/vdd | RivalFineCap_Projected | road | 22.8748 | 1.2130 | 22.8924 | 99.6667 | 14.4501 |
| vdd/vdd | RivalFineCap_Projected | vegetation | 60.0262 | 5.7585 | 94.9070 | 62.0241 | 15.0540 |
| vdd/vdd | RivalFineCap_Projected | vehicle | 32.9227 | -5.3370 | 32.9227 | 100.0000 | 1.0877 |
| vdd/vdd | RivalFineCap_Projected | roof | 86.1041 | -3.7774 | 86.5412 | 99.4168 | 6.9219 |
| vdd/vdd | RivalFineCap_Projected | water | 69.6876 | 3.1351 | 71.3323 | 96.7974 | 11.1973 |
| vdd/vdd | HardFixedMass_Projected | other | 37.9511 | -17.1089 | 79.4379 | 42.0854 | 27.0487 |
| vdd/vdd | HardFixedMass_Projected | wall | 29.4305 | -23.7144 | 30.5393 | 89.0184 | 23.1885 |
| vdd/vdd | HardFixedMass_Projected | road | 23.5259 | 1.8641 | 23.5433 | 99.6868 | 14.0534 |
| vdd/vdd | HardFixedMass_Projected | vegetation | 66.4379 | 12.1702 | 94.9990 | 68.8457 | 16.6935 |
| vdd/vdd | HardFixedMass_Projected | vehicle | 31.8315 | -6.4282 | 31.8315 | 100.0000 | 1.1250 |
| vdd/vdd | HardFixedMass_Projected | roof | 84.7864 | -5.0951 | 85.2346 | 99.3835 | 7.0256 |
| vdd/vdd | HardFixedMass_Projected | water | 72.4177 | 5.8652 | 73.8990 | 97.3065 | 10.8653 |
| vdd/vdd | NeutralUnprojected | other | 55.3978 | 0.3378 | 80.5984 | 63.9220 | 40.4919 |
| vdd/vdd | NeutralUnprojected | wall | 52.5268 | -0.6181 | 56.3446 | 88.5742 | 12.5057 |
| vdd/vdd | NeutralUnprojected | road | 22.8469 | 1.1851 | 22.8898 | 99.1854 | 14.3819 |
| vdd/vdd | NeutralUnprojected | vegetation | 53.7501 | -0.5176 | 95.1982 | 55.2480 | 13.3683 |
| vdd/vdd | NeutralUnprojected | vehicle | 37.3019 | -0.9578 | 37.3019 | 100.0000 | 0.9600 |
| vdd/vdd | NeutralUnprojected | roof | 90.2452 | 0.3637 | 90.7018 | 99.4452 | 6.6062 |
| vdd/vdd | NeutralUnprojected | water | 65.6122 | -0.9403 | 67.5925 | 95.7255 | 11.6860 |
| vdd/vdd | ProjectedHardStrengthMatched | other | 55.4692 | 0.4092 | 80.7327 | 63.9327 | 40.4312 |
| vdd/vdd | ProjectedHardStrengthMatched | wall | 53.4523 | 0.3074 | 57.1677 | 89.1592 | 12.4070 |
| vdd/vdd | ProjectedHardStrengthMatched | road | 22.1846 | 0.5228 | 22.1981 | 99.7270 | 14.9110 |
| vdd/vdd | ProjectedHardStrengthMatched | vegetation | 52.9980 | -1.2697 | 94.1982 | 54.7864 | 13.3974 |
| vdd/vdd | ProjectedHardStrengthMatched | vehicle | 39.3565 | 1.0968 | 39.3565 | 100.0000 | 0.9099 |
| vdd/vdd | ProjectedHardStrengthMatched | roof | 92.7851 | 2.9036 | 93.2171 | 99.5030 | 6.4317 |
| vdd/vdd | ProjectedHardStrengthMatched | water | 65.8485 | -0.7040 | 68.1636 | 95.0950 | 11.5118 |
| vdd/vdd | NeutralAliasShuffle0 | other | 56.5781 | 1.5181 | 80.9147 | 65.2912 | 41.1975 |
| vdd/vdd | NeutralAliasShuffle0 | wall | 54.0215 | 0.8766 | 58.1950 | 88.2805 | 12.0679 |
| vdd/vdd | NeutralAliasShuffle0 | road | 24.5129 | 2.8511 | 24.5845 | 98.8248 | 13.3418 |
| vdd/vdd | NeutralAliasShuffle0 | vegetation | 51.8295 | -2.4382 | 94.5710 | 53.4189 | 13.0115 |
| vdd/vdd | NeutralAliasShuffle0 | vehicle | 35.3046 | -2.9551 | 35.3046 | 100.0000 | 1.0143 |
| vdd/vdd | NeutralAliasShuffle0 | roof | 90.6783 | 0.7968 | 91.2166 | 99.3534 | 6.5629 |
| vdd/vdd | NeutralAliasShuffle0 | water | 59.8322 | -6.7203 | 61.5588 | 95.5221 | 12.8041 |
| vdd/vdd | NeutralAliasShuffle1 | other | 56.4906 | 1.4306 | 80.2438 | 65.6166 | 41.7490 |
| vdd/vdd | NeutralAliasShuffle1 | wall | 54.0790 | 0.9341 | 58.1833 | 88.4609 | 12.0950 |
| vdd/vdd | NeutralAliasShuffle1 | road | 24.6110 | 2.9492 | 24.6803 | 98.8722 | 13.2964 |
| vdd/vdd | NeutralAliasShuffle1 | vegetation | 51.1042 | -3.1635 | 95.5873 | 52.3390 | 12.6129 |
| vdd/vdd | NeutralAliasShuffle1 | vehicle | 38.2072 | -0.0525 | 38.2072 | 100.0000 | 0.9373 |
| vdd/vdd | NeutralAliasShuffle1 | roof | 90.0941 | 0.2126 | 90.6004 | 99.3835 | 6.6095 |
| vdd/vdd | NeutralAliasShuffle1 | water | 59.9006 | -6.6519 | 61.8008 | 95.1175 | 12.7000 |
| vdd/vdd | NeutralAliasShuffle2 | other | 56.4386 | 1.3786 | 80.7054 | 65.2417 | 41.2730 |
| vdd/vdd | NeutralAliasShuffle2 | wall | 53.8500 | 0.7051 | 58.0445 | 88.1684 | 12.0838 |
| vdd/vdd | NeutralAliasShuffle2 | road | 24.1886 | 2.5268 | 24.2579 | 98.8320 | 13.5224 |
| vdd/vdd | NeutralAliasShuffle2 | vegetation | 51.7520 | -2.5157 | 95.0026 | 53.2003 | 12.8994 |
| vdd/vdd | NeutralAliasShuffle2 | vehicle | 34.4464 | -3.8133 | 34.4464 | 100.0000 | 1.0396 |
| vdd/vdd | NeutralAliasShuffle2 | roof | 89.5338 | -0.3477 | 90.0072 | 99.4160 | 6.6553 |
| vdd/vdd | NeutralAliasShuffle2 | water | 60.7796 | -5.7729 | 62.7050 | 95.1909 | 12.5265 |
| potsdam/potsdam | Geometry | impervious surface | 48.6704 | -17.6961 | 84.3816 | 53.4889 | 25.9113 |
| potsdam/potsdam | Geometry | building | 72.8222 | -0.0441 | 79.5088 | 89.6472 | 18.1787 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 4.8098 | 77.4031 | 28.7716 | 6.4458 |
| potsdam/potsdam | Geometry | tree | 49.3362 | -8.7199 | 84.7436 | 54.1454 | 11.2731 |
| potsdam/potsdam | Geometry | car | 11.6958 | -13.4835 | 11.6965 | 99.9455 | 30.6388 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 2.6162 | 7.7773 | 13.2570 | 7.5523 |
| potsdam/potsdam | NoAdmission_Exact | impervious surface | 65.8977 | -0.4688 | 85.6706 | 74.0609 | 35.3370 |
| potsdam/potsdam | NoAdmission_Exact | building | 71.0529 | -1.8134 | 74.9985 | 93.1063 | 20.0156 |
| potsdam/potsdam | NoAdmission_Exact | low vegetation | 14.4124 | -7.3200 | 79.9195 | 14.9539 | 3.2447 |
| potsdam/potsdam | NoAdmission_Exact | tree | 55.5064 | -2.5497 | 91.9038 | 58.3601 | 11.2039 |
| potsdam/potsdam | NoAdmission_Exact | car | 24.5378 | -0.6415 | 24.5936 | 99.0837 | 14.4459 |
| potsdam/potsdam | NoAdmission_Exact | clutter | 2.1143 | -0.4238 | 2.6529 | 9.4321 | 15.7528 |
| potsdam/potsdam | RivalFineHard_Exact | impervious surface | 66.2446 | -0.1219 | 85.2705 | 74.8044 | 35.8593 |
| potsdam/potsdam | RivalFineHard_Exact | building | 72.5275 | -0.3388 | 76.5613 | 93.2276 | 19.6326 |
| potsdam/potsdam | RivalFineHard_Exact | low vegetation | 19.6733 | -2.0591 | 81.8153 | 20.5728 | 4.3604 |
| potsdam/potsdam | RivalFineHard_Exact | tree | 57.9390 | -0.1171 | 92.6969 | 60.7103 | 11.5554 |
| potsdam/potsdam | RivalFineHard_Exact | car | 24.9082 | -0.2711 | 24.9554 | 99.2460 | 14.2598 |
| potsdam/potsdam | RivalFineHard_Exact | clutter | 2.4111 | -0.1270 | 3.0821 | 9.9702 | 14.3326 |
| potsdam/potsdam | FineRivalProjected_Exact | impervious surface | 66.3665 | 0.0000 | 85.2000 | 75.0146 | 35.9898 |
| potsdam/potsdam | FineRivalProjected_Exact | building | 72.8663 | 0.0000 | 76.9013 | 93.2829 | 19.5574 |
| potsdam/potsdam | FineRivalProjected_Exact | low vegetation | 21.7324 | 0.0000 | 81.8728 | 22.8310 | 4.8357 |
| potsdam/potsdam | FineRivalProjected_Exact | tree | 58.0561 | 0.0000 | 92.6395 | 60.8635 | 11.5917 |
| potsdam/potsdam | FineRivalProjected_Exact | car | 25.1793 | 0.0000 | 25.2161 | 99.4242 | 14.1377 |
| potsdam/potsdam | FineRivalProjected_Exact | clutter | 2.5381 | 0.0000 | 3.2649 | 10.2339 | 13.8877 |
| potsdam/potsdam | FineBudgetOnly_Exact | impervious surface | 66.1042 | -0.2623 | 85.2298 | 74.6567 | 35.8056 |
| potsdam/potsdam | FineBudgetOnly_Exact | building | 72.9459 | 0.0796 | 77.0489 | 93.1965 | 19.5018 |
| potsdam/potsdam | FineBudgetOnly_Exact | low vegetation | 21.5223 | -0.2101 | 81.8118 | 22.6038 | 4.7911 |
| potsdam/potsdam | FineBudgetOnly_Exact | tree | 57.6287 | -0.4274 | 92.5777 | 60.4203 | 11.5150 |
| potsdam/potsdam | FineBudgetOnly_Exact | car | 24.9889 | -0.1904 | 25.0268 | 99.3976 | 14.2408 |
| potsdam/potsdam | FineBudgetOnly_Exact | clutter | 2.4949 | -0.0432 | 3.1966 | 10.2059 | 14.1457 |
| potsdam/potsdam | RivalNeutral_Projected | impervious surface | 66.0422 | -0.3243 | 85.1249 | 74.6581 | 35.8503 |
| potsdam/potsdam | RivalNeutral_Projected | building | 73.2765 | 0.4102 | 77.5331 | 93.0300 | 19.3454 |
| potsdam/potsdam | RivalNeutral_Projected | low vegetation | 20.9895 | -0.7429 | 80.9402 | 22.0808 | 4.7307 |
| potsdam/potsdam | RivalNeutral_Projected | tree | 58.4154 | 0.3593 | 92.6158 | 61.2689 | 11.6719 |
| potsdam/potsdam | RivalNeutral_Projected | car | 24.8942 | -0.2851 | 24.9303 | 99.4215 | 14.2994 |
| potsdam/potsdam | RivalNeutral_Projected | clutter | 2.5372 | -0.0009 | 3.2518 | 10.3501 | 14.1023 |
| potsdam/potsdam | RivalFineCap_Projected | impervious surface | 65.3560 | -1.0105 | 84.6666 | 74.1303 | 35.7896 |
| potsdam/potsdam | RivalFineCap_Projected | building | 73.0965 | 0.2302 | 77.2761 | 93.1105 | 19.4265 |
| potsdam/potsdam | RivalFineCap_Projected | low vegetation | 26.8952 | 5.1628 | 79.4209 | 28.9099 | 6.3122 |
| potsdam/potsdam | RivalFineCap_Projected | tree | 59.6377 | 1.5816 | 92.7994 | 62.5313 | 11.8888 |
| potsdam/potsdam | RivalFineCap_Projected | car | 24.9851 | -0.1942 | 25.0132 | 99.5518 | 14.2707 |
| potsdam/potsdam | RivalFineCap_Projected | clutter | 3.1865 | 0.6484 | 4.1994 | 11.6696 | 12.3121 |
| potsdam/potsdam | HardFixedMass_Projected | impervious surface | 65.0409 | -1.3256 | 83.9342 | 74.2895 | 36.1794 |
| potsdam/potsdam | HardFixedMass_Projected | building | 73.6407 | 0.7744 | 78.5192 | 92.2194 | 18.9360 |
| potsdam/potsdam | HardFixedMass_Projected | low vegetation | 34.5009 | 12.7685 | 79.5505 | 37.8586 | 8.2526 |
| potsdam/potsdam | HardFixedMass_Projected | tree | 61.1790 | 3.1229 | 91.8354 | 64.6980 | 12.4299 |
| potsdam/potsdam | HardFixedMass_Projected | car | 25.0333 | -0.1460 | 25.0546 | 99.6622 | 14.2629 |
| potsdam/potsdam | HardFixedMass_Projected | clutter | 4.1864 | 1.6483 | 5.8094 | 13.0321 | 9.9391 |
| potsdam/potsdam | NeutralUnprojected | impervious surface | 66.0252 | -0.3413 | 85.1231 | 74.6378 | 35.8414 |
| potsdam/potsdam | NeutralUnprojected | building | 73.3160 | 0.4497 | 77.5857 | 93.0179 | 19.3298 |
| potsdam/potsdam | NeutralUnprojected | low vegetation | 20.9616 | -0.7708 | 80.8399 | 22.0574 | 4.7315 |
| potsdam/potsdam | NeutralUnprojected | tree | 58.4702 | 0.4141 | 92.6222 | 61.3265 | 11.6821 |
| potsdam/potsdam | NeutralUnprojected | car | 24.8781 | -0.3012 | 24.9146 | 99.4149 | 14.3075 |
| potsdam/potsdam | NeutralUnprojected | clutter | 2.5302 | -0.0079 | 3.2427 | 10.3253 | 14.1078 |
| potsdam/potsdam | ProjectedHardStrengthMatched | impervious surface | 66.1355 | -0.2310 | 84.9815 | 74.8884 | 36.0216 |
| potsdam/potsdam | ProjectedHardStrengthMatched | building | 73.2477 | 0.3814 | 77.3168 | 93.2965 | 19.4551 |
| potsdam/potsdam | ProjectedHardStrengthMatched | low vegetation | 21.7654 | 0.0330 | 82.1713 | 22.8442 | 4.8209 |
| potsdam/potsdam | ProjectedHardStrengthMatched | tree | 58.3299 | 0.2738 | 92.5978 | 61.1827 | 11.6578 |
| potsdam/potsdam | ProjectedHardStrengthMatched | car | 25.0753 | -0.1040 | 25.1131 | 99.4029 | 14.1926 |
| potsdam/potsdam | ProjectedHardStrengthMatched | clutter | 2.6170 | 0.0789 | 3.3659 | 10.5234 | 13.8520 |
| potsdam/potsdam | NeutralAliasShuffle0 | impervious surface | 65.1686 | -1.1979 | 85.2180 | 73.4743 | 35.2434 |
| potsdam/potsdam | NeutralAliasShuffle0 | building | 72.8135 | -0.0528 | 77.0811 | 92.9336 | 19.4387 |
| potsdam/potsdam | NeutralAliasShuffle0 | low vegetation | 17.2808 | -4.4516 | 80.6681 | 18.0274 | 3.8753 |
| potsdam/potsdam | NeutralAliasShuffle0 | tree | 57.3106 | -0.7455 | 91.9316 | 60.3460 | 11.5817 |
| potsdam/potsdam | NeutralAliasShuffle0 | car | 24.1580 | -1.0213 | 24.1934 | 99.3976 | 14.7314 |
| potsdam/potsdam | NeutralAliasShuffle0 | clutter | 2.3136 | -0.2245 | 2.9235 | 9.9831 | 15.1297 |
| potsdam/potsdam | NeutralAliasShuffle1 | impervious surface | 65.7212 | -0.6453 | 85.2758 | 74.1338 | 35.5356 |
| potsdam/potsdam | NeutralAliasShuffle1 | building | 72.7519 | -0.1144 | 77.0062 | 92.9422 | 19.4594 |
| potsdam/potsdam | NeutralAliasShuffle1 | low vegetation | 18.0313 | -3.7011 | 80.0233 | 18.8812 | 4.0915 |
| potsdam/potsdam | NeutralAliasShuffle1 | tree | 57.0935 | -0.9626 | 92.1253 | 60.0228 | 11.4954 |
| potsdam/potsdam | NeutralAliasShuffle1 | car | 24.6743 | -0.5050 | 24.7176 | 99.2952 | 14.4041 |
| potsdam/potsdam | NeutralAliasShuffle1 | clutter | 2.3755 | -0.1626 | 3.0051 | 10.1833 | 15.0140 |
| potsdam/potsdam | NeutralAliasShuffle2 | impervious surface | 66.4722 | 0.1057 | 85.7748 | 74.7080 | 35.6025 |
| potsdam/potsdam | NeutralAliasShuffle2 | building | 72.5346 | -0.3317 | 76.7112 | 93.0179 | 19.5501 |
| potsdam/potsdam | NeutralAliasShuffle2 | low vegetation | 17.6033 | -4.1291 | 79.2039 | 18.4563 | 4.0408 |
| potsdam/potsdam | NeutralAliasShuffle2 | tree | 57.5410 | -0.5151 | 92.2407 | 60.4679 | 11.5662 |
| potsdam/potsdam | NeutralAliasShuffle2 | car | 24.9817 | -0.1976 | 25.0195 | 99.3989 | 14.2452 |
| potsdam/potsdam | NeutralAliasShuffle2 | clutter | 2.3452 | -0.1929 | 2.9685 | 10.0466 | 14.9952 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | -8.3139 | 89.6338 | 64.7597 | 2.2305 |
| udd5/udd5 | Geometry | building | 83.4755 | -2.4045 | 86.9865 | 95.3877 | 78.8720 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -2.9205 | 3.7107 | 99.9773 | 11.3160 |
| udd5/udd5 | Geometry | other | 5.0713 | -6.9433 | 23.0553 | 6.1044 | 5.3797 |
| udd5/udd5 | NoAdmission_Exact | vegetation | 46.2799 | -22.2814 | 93.8774 | 47.7203 | 1.5693 |
| udd5/udd5 | NoAdmission_Exact | building | 83.4871 | -2.3929 | 83.6594 | 99.7539 | 85.7625 |
| udd5/udd5 | NoAdmission_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 |
| udd5/udd5 | NoAdmission_Exact | vehicle | 5.3645 | -1.2667 | 5.3645 | 99.9773 | 7.8274 |
| udd5/udd5 | NoAdmission_Exact | other | 5.7476 | -6.2670 | 29.7284 | 6.6513 | 4.5458 |
| udd5/udd5 | RivalFineHard_Exact | vegetation | 67.6962 | -0.8651 | 91.0639 | 72.5133 | 2.4583 |
| udd5/udd5 | RivalFineHard_Exact | building | 85.5987 | -0.2813 | 85.6693 | 99.9037 | 83.8762 |
| udd5/udd5 | RivalFineHard_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6799 |
| udd5/udd5 | RivalFineHard_Exact | vehicle | 7.0104 | 0.3792 | 7.0105 | 99.9773 | 5.9896 |
| udd5/udd5 | RivalFineHard_Exact | other | 9.8917 | -2.1229 | 39.5036 | 11.6576 | 5.9959 |
| udd5/udd5 | FineRivalProjected_Exact | vegetation | 68.5613 | 0.0000 | 90.9603 | 73.5744 | 2.4971 |
| udd5/udd5 | FineRivalProjected_Exact | building | 85.8800 | 0.0000 | 86.0701 | 99.7434 | 83.3517 |
| udd5/udd5 | FineRivalProjected_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3867 |
| udd5/udd5 | FineRivalProjected_Exact | vehicle | 6.6312 | 0.0000 | 6.6313 | 99.9773 | 6.3321 |
| udd5/udd5 | FineRivalProjected_Exact | other | 12.0146 | 0.0000 | 44.6062 | 14.1216 | 6.4323 |
| udd5/udd5 | FineBudgetOnly_Exact | vegetation | 68.7476 | 0.1863 | 90.9648 | 73.7860 | 2.5042 |
| udd5/udd5 | FineBudgetOnly_Exact | building | 85.4999 | -0.3801 | 85.5929 | 99.8730 | 83.9252 |
| udd5/udd5 | FineBudgetOnly_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9439 |
| udd5/udd5 | FineBudgetOnly_Exact | vehicle | 6.8445 | 0.2133 | 6.8446 | 99.9773 | 6.1348 |
| udd5/udd5 | FineBudgetOnly_Exact | other | 12.3612 | 0.3466 | 45.4328 | 14.5163 | 6.4919 |
| udd5/udd5 | RivalNeutral_Projected | vegetation | 66.0774 | -2.4839 | 91.4502 | 70.4281 | 2.3776 |
| udd5/udd5 | RivalNeutral_Projected | building | 85.7067 | -0.1733 | 85.9886 | 99.6189 | 83.3265 |
| udd5/udd5 | RivalNeutral_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6083 |
| udd5/udd5 | RivalNeutral_Projected | vehicle | 6.2004 | -0.4308 | 6.2005 | 99.9773 | 6.7721 |
| udd5/udd5 | RivalNeutral_Projected | other | 9.8142 | -2.2004 | 39.6328 | 11.5391 | 5.9156 |
| udd5/udd5 | RivalFineCap_Projected | vegetation | 71.5191 | 2.9578 | 89.6923 | 77.9238 | 2.6822 |
| udd5/udd5 | RivalFineCap_Projected | building | 86.2077 | 0.3277 | 86.8489 | 99.1508 | 82.1134 |
| udd5/udd5 | RivalFineCap_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6685 |
| udd5/udd5 | RivalFineCap_Projected | vehicle | 6.8407 | 0.2095 | 6.8408 | 99.9773 | 6.1382 |
| udd5/udd5 | RivalFineCap_Projected | other | 12.9921 | 0.9775 | 48.0141 | 15.1188 | 6.3978 |
| udd5/udd5 | HardFixedMass_Projected | vegetation | 73.3235 | 4.7622 | 89.2792 | 80.4028 | 2.7803 |
| udd5/udd5 | HardFixedMass_Projected | building | 86.4035 | 0.5235 | 87.1783 | 98.9818 | 81.6637 |
| udd5/udd5 | HardFixedMass_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.9347 |
| udd5/udd5 | HardFixedMass_Projected | vehicle | 7.0260 | 0.3948 | 7.0261 | 99.9773 | 5.9763 |
| udd5/udd5 | HardFixedMass_Projected | other | 15.3357 | 3.3211 | 53.9529 | 17.6452 | 6.6450 |
| udd5/udd5 | NeutralUnprojected | vegetation | 65.8890 | -2.6723 | 91.2817 | 70.3139 | 2.3781 |
| udd5/udd5 | NeutralUnprojected | building | 85.6863 | -0.1937 | 85.9592 | 99.6308 | 83.3649 |
| udd5/udd5 | NeutralUnprojected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0067 |
| udd5/udd5 | NeutralUnprojected | vehicle | 6.3894 | -0.2418 | 6.3895 | 99.9773 | 6.5718 |
| udd5/udd5 | NeutralUnprojected | other | 8.6443 | -3.3703 | 36.4255 | 10.1803 | 5.6785 |
| udd5/udd5 | ProjectedHardStrengthMatched | vegetation | 68.3997 | -0.1616 | 91.0091 | 73.3566 | 2.4884 |
| udd5/udd5 | ProjectedHardStrengthMatched | building | 85.9277 | 0.0477 | 86.2986 | 99.5022 | 82.9299 |
| udd5/udd5 | ProjectedHardStrengthMatched | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.8133 |
| udd5/udd5 | ProjectedHardStrengthMatched | vehicle | 6.3191 | -0.3121 | 6.3192 | 99.9773 | 6.6449 |
| udd5/udd5 | ProjectedHardStrengthMatched | other | 10.7599 | -1.2547 | 41.9482 | 12.6424 | 6.1234 |
| udd5/udd5 | NeutralAliasShuffle0 | vegetation | 52.6139 | -15.9474 | 91.5714 | 55.2916 | 1.8641 |
| udd5/udd5 | NeutralAliasShuffle0 | building | 83.9114 | -1.9686 | 84.0628 | 99.7858 | 85.3782 |
| udd5/udd5 | NeutralAliasShuffle0 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3559 |
| udd5/udd5 | NeutralAliasShuffle0 | vehicle | 4.7864 | -1.8448 | 4.7864 | 99.9773 | 8.7728 |
| udd5/udd5 | NeutralAliasShuffle0 | other | 5.5432 | -6.4714 | 34.6569 | 6.1901 | 3.6290 |
| udd5/udd5 | NeutralAliasShuffle1 | vegetation | 54.0938 | -14.4675 | 91.6180 | 56.9103 | 1.9177 |
| udd5/udd5 | NeutralAliasShuffle1 | building | 83.8474 | -2.0326 | 84.0165 | 99.7606 | 85.4037 |
| udd5/udd5 | NeutralAliasShuffle1 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4593 |
| udd5/udd5 | NeutralAliasShuffle1 | vehicle | 6.7278 | 0.0966 | 6.7279 | 99.9773 | 6.2412 |
| udd5/udd5 | NeutralAliasShuffle1 | other | 8.2277 | -3.7869 | 33.4402 | 9.8390 | 5.9781 |
| udd5/udd5 | NeutralAliasShuffle2 | vegetation | 51.4705 | -17.0908 | 92.1010 | 53.8475 | 1.8050 |
| udd5/udd5 | NeutralAliasShuffle2 | building | 84.1703 | -1.7097 | 84.3091 | 99.8049 | 85.1451 |
| udd5/udd5 | NeutralAliasShuffle2 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3712 |
| udd5/udd5 | NeutralAliasShuffle2 | vehicle | 4.8816 | -1.7496 | 4.8817 | 99.9773 | 8.6016 |
| udd5/udd5 | NeutralAliasShuffle2 | other | 6.2586 | -5.7560 | 35.2420 | 7.0718 | 4.0771 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 |
| oem/oem | Geometry | rangeland | 41.5417 | -8.3175 | 54.4153 | 63.7145 | 17.0320 |
| oem/oem | Geometry | developed space | 25.6977 | 2.2107 | 65.5485 | 29.7106 | 8.8828 |
| oem/oem | Geometry | road | 52.3958 | -3.0184 | 60.6185 | 79.4351 | 8.2388 |
| oem/oem | Geometry | tree | 65.2556 | 7.5232 | 84.8180 | 73.8857 | 23.0134 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 |
| oem/oem | Geometry | agriculture land | 71.2323 | -11.3134 | 95.7971 | 73.5302 | 16.2828 |
| oem/oem | Geometry | building | 62.4408 | 7.8917 | 64.4677 | 95.2061 | 17.5919 |
| oem/oem | NoAdmission_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 |
| oem/oem | NoAdmission_Exact | rangeland | 49.5261 | -0.3331 | 70.5348 | 62.4456 | 12.8780 |
| oem/oem | NoAdmission_Exact | developed space | 23.7669 | 0.2799 | 38.9362 | 37.8898 | 19.0710 |
| oem/oem | NoAdmission_Exact | road | 54.5529 | -0.8613 | 68.0810 | 73.3007 | 6.7693 |
| oem/oem | NoAdmission_Exact | tree | 55.6605 | -2.0719 | 91.7802 | 58.5807 | 16.8622 |
| oem/oem | NoAdmission_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | NoAdmission_Exact | agriculture land | 81.6500 | -0.8957 | 82.0066 | 99.4703 | 25.7312 |
| oem/oem | NoAdmission_Exact | building | 47.0287 | -7.5204 | 68.3392 | 60.1297 | 10.4811 |
| oem/oem | RivalFineHard_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4062 |
| oem/oem | RivalFineHard_Exact | rangeland | 49.6933 | -0.1659 | 70.6164 | 62.6472 | 12.9046 |
| oem/oem | RivalFineHard_Exact | developed space | 23.8549 | 0.3679 | 40.6619 | 36.5937 | 17.6369 |
| oem/oem | RivalFineHard_Exact | road | 54.9225 | -0.4917 | 68.1954 | 73.8350 | 6.8072 |
| oem/oem | RivalFineHard_Exact | tree | 56.9837 | -0.7487 | 91.1586 | 60.3173 | 17.4805 |
| oem/oem | RivalFineHard_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| oem/oem | RivalFineHard_Exact | agriculture land | 82.0069 | -0.5388 | 82.3962 | 99.4271 | 25.5984 |
| oem/oem | RivalFineHard_Exact | building | 50.8636 | -3.6855 | 69.6827 | 65.3182 | 11.1660 |
| oem/oem | FineRivalProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6617 |
| oem/oem | FineRivalProjected_Exact | rangeland | 49.8592 | 0.0000 | 70.6753 | 62.8643 | 12.9385 |
| oem/oem | FineRivalProjected_Exact | developed space | 23.4870 | 0.0000 | 41.7734 | 34.9185 | 16.3817 |
| oem/oem | FineRivalProjected_Exact | road | 55.4142 | 0.0000 | 68.4843 | 74.3824 | 6.8287 |
| oem/oem | FineRivalProjected_Exact | tree | 57.7324 | 0.0000 | 90.8900 | 61.2783 | 17.8115 |
| oem/oem | FineRivalProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineRivalProjected_Exact | agriculture land | 82.5457 | 0.0000 | 82.9578 | 99.4017 | 25.4186 |
| oem/oem | FineRivalProjected_Exact | building | 54.5491 | 0.0000 | 70.4544 | 70.7286 | 11.9585 |
| oem/oem | FineBudgetOnly_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.5567 |
| oem/oem | FineBudgetOnly_Exact | rangeland | 49.8649 | 0.0057 | 70.6703 | 62.8774 | 12.9422 |
| oem/oem | FineBudgetOnly_Exact | developed space | 23.7934 | 0.3064 | 41.0958 | 36.1075 | 17.2188 |
| oem/oem | FineBudgetOnly_Exact | road | 55.3529 | -0.0613 | 68.7562 | 73.9549 | 6.7626 |
| oem/oem | FineBudgetOnly_Exact | tree | 57.7722 | 0.0398 | 90.9006 | 61.3183 | 17.8210 |
| oem/oem | FineBudgetOnly_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| oem/oem | FineBudgetOnly_Exact | agriculture land | 82.8190 | 0.2733 | 83.2403 | 99.3925 | 25.3300 |
| oem/oem | FineBudgetOnly_Exact | building | 51.6864 | -2.8627 | 69.7801 | 66.5924 | 11.3680 |
| oem/oem | RivalNeutral_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6332 |
| oem/oem | RivalNeutral_Projected | rangeland | 49.9901 | 0.1309 | 70.7125 | 63.0430 | 12.9685 |
| oem/oem | RivalNeutral_Projected | developed space | 23.7275 | 0.2405 | 41.4271 | 35.7062 | 16.8913 |
| oem/oem | RivalNeutral_Projected | road | 55.5506 | 0.1364 | 68.6768 | 74.4011 | 6.8113 |
| oem/oem | RivalNeutral_Projected | tree | 57.8444 | 0.1120 | 90.9926 | 61.3578 | 17.8145 |
| oem/oem | RivalNeutral_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0023 |
| oem/oem | RivalNeutral_Projected | agriculture land | 83.0534 | 0.5077 | 83.4799 | 99.3886 | 25.2563 |
| oem/oem | RivalNeutral_Projected | building | 53.4517 | -1.0974 | 70.5334 | 68.8194 | 11.6227 |
| oem/oem | RivalFineCap_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.8617 |
| oem/oem | RivalFineCap_Projected | rangeland | 49.9382 | 0.0790 | 70.6895 | 62.9787 | 12.9595 |
| oem/oem | RivalFineCap_Projected | developed space | 23.7208 | 0.2338 | 43.8294 | 34.0815 | 15.2390 |
| oem/oem | RivalFineCap_Projected | road | 55.9920 | 0.5778 | 68.8877 | 74.9439 | 6.8400 |
| oem/oem | RivalFineCap_Projected | tree | 59.1970 | 1.4646 | 90.3815 | 63.1771 | 18.4667 |
| oem/oem | RivalFineCap_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0114 |
| oem/oem | RivalFineCap_Projected | agriculture land | 83.9327 | 1.3870 | 84.4197 | 99.3175 | 24.9573 |
| oem/oem | RivalFineCap_Projected | building | 58.7407 | 4.1916 | 71.8102 | 76.3454 | 12.6644 |
| oem/oem | HardFixedMass_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.3174 |
| oem/oem | HardFixedMass_Projected | rangeland | 49.9459 | 0.0867 | 70.5596 | 63.0945 | 13.0072 |
| oem/oem | HardFixedMass_Projected | developed space | 22.8729 | -0.6141 | 45.4059 | 31.5494 | 13.6170 |
| oem/oem | HardFixedMass_Projected | road | 56.7392 | 1.3250 | 69.4442 | 75.6176 | 6.8461 |
| oem/oem | HardFixedMass_Projected | tree | 60.3220 | 2.5896 | 89.6413 | 64.8419 | 19.1098 |
| oem/oem | HardFixedMass_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0155 |
| oem/oem | HardFixedMass_Projected | agriculture land | 85.0999 | 2.5542 | 85.7354 | 99.1365 | 24.5295 |
| oem/oem | HardFixedMass_Projected | building | 62.0243 | 7.4752 | 71.9163 | 81.8487 | 13.5573 |
| oem/oem | NeutralUnprojected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6304 |
| oem/oem | NeutralUnprojected | rangeland | 49.9839 | 0.1247 | 70.7190 | 63.0279 | 12.9642 |
| oem/oem | NeutralUnprojected | developed space | 23.7231 | 0.2361 | 41.3778 | 35.7327 | 16.9240 |
| oem/oem | NeutralUnprojected | road | 55.5566 | 0.1424 | 68.6913 | 74.3949 | 6.8093 |
| oem/oem | NeutralUnprojected | tree | 57.8266 | 0.0942 | 90.9965 | 61.3359 | 17.8074 |
| oem/oem | NeutralUnprojected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0023 |
| oem/oem | NeutralUnprojected | agriculture land | 83.0471 | 0.5014 | 83.4728 | 99.3897 | 25.2588 |
| oem/oem | NeutralUnprojected | building | 53.3912 | -1.1579 | 70.5394 | 68.7133 | 11.6038 |
| oem/oem | ProjectedHardStrengthMatched | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.7164 |
| oem/oem | ProjectedHardStrengthMatched | rangeland | 49.8213 | -0.0379 | 70.6380 | 62.8337 | 12.9391 |
| oem/oem | ProjectedHardStrengthMatched | developed space | 23.1790 | -0.3080 | 40.6333 | 35.0482 | 16.9039 |
| oem/oem | ProjectedHardStrengthMatched | road | 55.3870 | -0.0272 | 68.5732 | 74.2290 | 6.8058 |
| oem/oem | ProjectedHardStrengthMatched | tree | 57.6324 | -0.1000 | 91.0437 | 61.0963 | 17.7286 |
| oem/oem | ProjectedHardStrengthMatched | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | ProjectedHardStrengthMatched | agriculture land | 82.7965 | 0.2508 | 83.2059 | 99.4094 | 25.3448 |
| oem/oem | ProjectedHardStrengthMatched | building | 52.5606 | -1.9885 | 69.9517 | 67.8884 | 11.5608 |
| oem/oem | NeutralAliasShuffle0 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3342 |
| oem/oem | NeutralAliasShuffle0 | rangeland | 49.9477 | 0.0885 | 70.6894 | 62.9939 | 12.9626 |
| oem/oem | NeutralAliasShuffle0 | developed space | 24.0200 | 0.5330 | 40.5280 | 37.0951 | 17.9376 |
| oem/oem | NeutralAliasShuffle0 | road | 55.2173 | -0.1969 | 68.6599 | 73.8241 | 6.7601 |
| oem/oem | NeutralAliasShuffle0 | tree | 56.9672 | -0.7652 | 91.2724 | 60.2491 | 17.4390 |
| oem/oem | NeutralAliasShuffle0 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0009 |
| oem/oem | NeutralAliasShuffle0 | agriculture land | 82.7792 | 0.2335 | 83.1829 | 99.4172 | 25.3538 |
| oem/oem | NeutralAliasShuffle0 | building | 51.4014 | -3.1477 | 70.0216 | 65.9047 | 11.2118 |
| oem/oem | NeutralAliasShuffle1 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2985 |
| oem/oem | NeutralAliasShuffle1 | rangeland | 49.8739 | 0.0147 | 70.6793 | 62.8845 | 12.9420 |
| oem/oem | NeutralAliasShuffle1 | developed space | 24.1865 | 0.6995 | 40.5530 | 37.4724 | 18.1089 |
| oem/oem | NeutralAliasShuffle1 | road | 55.2502 | -0.1640 | 68.6011 | 73.9510 | 6.7775 |
| oem/oem | NeutralAliasShuffle1 | tree | 57.2493 | -0.4831 | 91.1720 | 60.6090 | 17.5624 |
| oem/oem | NeutralAliasShuffle1 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0011 |
| oem/oem | NeutralAliasShuffle1 | agriculture land | 83.0680 | 0.5223 | 83.4968 | 99.3856 | 25.2504 |
| oem/oem | NeutralAliasShuffle1 | building | 50.3149 | -4.2342 | 69.5278 | 64.5492 | 11.0591 |
| oem/oem | NeutralAliasShuffle2 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6688 |
| oem/oem | NeutralAliasShuffle2 | rangeland | 49.9648 | 0.1056 | 70.6952 | 63.0164 | 12.9662 |
| oem/oem | NeutralAliasShuffle2 | developed space | 23.6854 | 0.1984 | 40.7258 | 36.1460 | 17.3938 |
| oem/oem | NeutralAliasShuffle2 | road | 55.1549 | -0.2593 | 68.3055 | 74.1254 | 6.8229 |
| oem/oem | NeutralAliasShuffle2 | tree | 57.6508 | -0.0816 | 91.0563 | 61.1113 | 17.7305 |
| oem/oem | NeutralAliasShuffle2 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0017 |
| oem/oem | NeutralAliasShuffle2 | agriculture land | 82.7717 | 0.2260 | 83.1716 | 99.4225 | 25.3586 |
| oem/oem | NeutralAliasShuffle2 | building | 50.6905 | -3.8586 | 69.8773 | 64.8644 | 11.0576 |
| loveda/P | Geometry | building | 7.6311 | -41.2981 | 7.6311 | 100.0000 | 0.3472 |
| loveda/P | Geometry | road | 63.0522 | 2.8911 | 64.4734 | 96.6222 | 12.4720 |
| loveda/P | Geometry | water | 63.2131 | 0.2889 | 68.9718 | 88.3328 | 18.8451 |
| loveda/P | Geometry | barren | 1.9144 | 0.2420 | 34.2731 | 1.9874 | 0.6429 |
| loveda/P | Geometry | tree | 72.0056 | 15.4145 | 88.8915 | 79.1256 | 14.6560 |
| loveda/P | Geometry | farm | 89.4231 | 3.8393 | 91.1653 | 97.9076 | 53.0368 |
| loveda/P | NoAdmission_Exact | building | 60.8479 | 11.9187 | 72.1893 | 79.4788 | 0.0292 |
| loveda/P | NoAdmission_Exact | road | 53.7553 | -6.4058 | 54.7280 | 96.7995 | 14.7198 |
| loveda/P | NoAdmission_Exact | water | 62.4091 | -0.5151 | 68.1253 | 88.1486 | 19.0394 |
| loveda/P | NoAdmission_Exact | barren | 4.3215 | 2.6491 | 89.7378 | 4.3430 | 0.5366 |
| loveda/P | NoAdmission_Exact | tree | 39.9553 | -16.6358 | 99.3680 | 40.0571 | 6.6373 |
| loveda/P | NoAdmission_Exact | farm | 82.9502 | -2.6336 | 83.2671 | 99.5433 | 59.0377 |
| loveda/P | RivalFineHard_Exact | building | 53.9715 | 5.0423 | 59.0200 | 86.3192 | 0.0388 |
| loveda/P | RivalFineHard_Exact | road | 58.5446 | -1.6165 | 59.7219 | 96.7425 | 13.4810 |
| loveda/P | RivalFineHard_Exact | water | 62.5489 | -0.3753 | 67.9247 | 88.7680 | 19.2298 |
| loveda/P | RivalFineHard_Exact | barren | 2.3466 | 0.6742 | 65.0543 | 2.3766 | 0.4051 |
| loveda/P | RivalFineHard_Exact | tree | 53.2126 | -3.3785 | 97.3091 | 54.0073 | 9.1381 |
| loveda/P | RivalFineHard_Exact | farm | 84.9031 | -0.6807 | 85.2129 | 99.5737 | 57.7072 |
| loveda/P | FineRivalProjected_Exact | building | 48.9292 | 0.0000 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | FineRivalProjected_Exact | road | 60.1611 | 0.0000 | 61.4250 | 96.6927 | 13.1005 |
| loveda/P | FineRivalProjected_Exact | water | 62.9242 | 0.0000 | 68.0543 | 89.3018 | 19.3086 |
| loveda/P | FineRivalProjected_Exact | barren | 1.6724 | 0.0000 | 41.1787 | 1.7134 | 0.4613 |
| loveda/P | FineRivalProjected_Exact | tree | 56.5911 | 0.0000 | 96.7147 | 57.7003 | 9.8230 |
| loveda/P | FineRivalProjected_Exact | farm | 85.5838 | 0.0000 | 85.8926 | 99.5816 | 57.2550 |
| loveda/P | FineBudgetOnly_Exact | building | 44.3478 | -4.5814 | 44.4122 | 99.6743 | 0.0595 |
| loveda/P | FineBudgetOnly_Exact | road | 60.5514 | 0.3903 | 61.8333 | 96.6896 | 13.0136 |
| loveda/P | FineBudgetOnly_Exact | water | 63.0869 | 0.1627 | 68.1903 | 89.3950 | 19.2903 |
| loveda/P | FineBudgetOnly_Exact | barren | 1.8502 | 0.1778 | 62.0227 | 1.8714 | 0.3345 |
| loveda/P | FineBudgetOnly_Exact | tree | 57.0813 | 0.4902 | 96.4614 | 58.3021 | 9.9515 |
| loveda/P | FineBudgetOnly_Exact | farm | 85.4354 | -0.1484 | 85.7461 | 99.5778 | 57.3507 |
| loveda/P | RivalNeutral_Projected | building | 43.7322 | -5.1970 | 43.7322 | 100.0000 | 0.0606 |
| loveda/P | RivalNeutral_Projected | road | 59.3050 | -0.8561 | 60.5146 | 96.7394 | 13.3040 |
| loveda/P | RivalNeutral_Projected | water | 62.8178 | -0.1064 | 68.0016 | 89.1780 | 19.2968 |
| loveda/P | RivalNeutral_Projected | barren | 1.9550 | 0.2826 | 33.0427 | 2.0356 | 0.6831 |
| loveda/P | RivalNeutral_Projected | tree | 56.2826 | -0.3085 | 96.7149 | 57.3795 | 9.7683 |
| loveda/P | RivalNeutral_Projected | farm | 86.0978 | 0.5140 | 86.4278 | 99.5584 | 56.8872 |
| loveda/P | RivalFineCap_Projected | building | 7.3727 | -41.5565 | 7.3727 | 100.0000 | 0.3594 |
| loveda/P | RivalFineCap_Projected | road | 62.8087 | 2.6476 | 64.1917 | 96.6834 | 12.5346 |
| loveda/P | RivalFineCap_Projected | water | 63.6245 | 0.7003 | 68.7411 | 89.5264 | 19.1638 |
| loveda/P | RivalFineCap_Projected | barren | 1.1820 | -0.4904 | 3.9541 | 1.6581 | 4.6493 |
| loveda/P | RivalFineCap_Projected | tree | 62.1871 | 5.5960 | 95.6079 | 64.0159 | 11.0243 |
| loveda/P | RivalFineCap_Projected | farm | 80.1661 | -5.4177 | 86.5362 | 91.5899 | 52.2685 |
| loveda/P | HardFixedMass_Projected | building | 1.1404 | -47.7888 | 1.1404 | 100.0000 | 2.3235 |
| loveda/P | HardFixedMass_Projected | road | 66.1987 | 6.0376 | 67.7849 | 96.5859 | 11.8582 |
| loveda/P | HardFixedMass_Projected | water | 64.5042 | 1.5800 | 69.5540 | 89.8830 | 19.0153 |
| loveda/P | HardFixedMass_Projected | barren | 1.1262 | -0.5462 | 3.2283 | 1.7001 | 5.8389 |
| loveda/P | HardFixedMass_Projected | tree | 65.9827 | 9.3916 | 94.2365 | 68.7574 | 12.0132 |
| loveda/P | HardFixedMass_Projected | farm | 75.2908 | -10.2930 | 86.2843 | 85.5267 | 48.9509 |
| loveda/P | NeutralUnprojected | building | 44.0459 | -4.8833 | 44.0459 | 100.0000 | 0.0602 |
| loveda/P | NeutralUnprojected | road | 59.2657 | -0.8954 | 60.4721 | 96.7435 | 13.3139 |
| loveda/P | NeutralUnprojected | water | 62.8243 | -0.0999 | 68.0127 | 89.1721 | 19.2924 |
| loveda/P | NeutralUnprojected | barren | 1.9523 | 0.2799 | 33.5218 | 2.0310 | 0.6717 |
| loveda/P | NeutralUnprojected | tree | 56.2558 | -0.3353 | 96.7251 | 57.3480 | 9.7619 |
| loveda/P | NeutralUnprojected | farm | 86.0785 | 0.4947 | 86.4086 | 99.5582 | 56.8998 |
| loveda/P | ProjectedHardStrengthMatched | building | 50.0843 | 1.1551 | 50.9434 | 96.7427 | 0.0503 |
| loveda/P | ProjectedHardStrengthMatched | road | 59.1822 | -0.9789 | 60.3904 | 96.7300 | 13.3301 |
| loveda/P | ProjectedHardStrengthMatched | water | 61.3062 | -1.6180 | 66.2474 | 89.1534 | 19.8023 |
| loveda/P | ProjectedHardStrengthMatched | barren | 1.7474 | 0.0750 | 10.4501 | 2.0551 | 2.1804 |
| loveda/P | ProjectedHardStrengthMatched | tree | 54.4836 | -2.1075 | 96.6456 | 55.5338 | 9.4609 |
| loveda/P | ProjectedHardStrengthMatched | farm | 83.0781 | -2.5057 | 85.9939 | 96.0787 | 55.1759 |
| loveda/P | NeutralAliasShuffle0 | building | 38.1366 | -10.7926 | 38.1366 | 100.0000 | 0.0695 |
| loveda/P | NeutralAliasShuffle0 | road | 58.0182 | -2.1429 | 59.1563 | 96.7902 | 13.6166 |
| loveda/P | NeutralAliasShuffle0 | water | 63.4299 | 0.5057 | 68.7470 | 89.1317 | 19.0777 |
| loveda/P | NeutralAliasShuffle0 | barren | 2.2114 | 0.5390 | 65.6764 | 2.2373 | 0.3777 |
| loveda/P | NeutralAliasShuffle0 | tree | 54.0454 | -2.5457 | 95.9776 | 55.2979 | 9.4863 |
| loveda/P | NeutralAliasShuffle0 | farm | 85.3576 | -0.2262 | 85.6890 | 99.5489 | 57.3723 |
| loveda/P | NeutralAliasShuffle1 | building | 45.4815 | -3.4477 | 45.4815 | 100.0000 | 0.0583 |
| loveda/P | NeutralAliasShuffle1 | road | 57.1109 | -3.0502 | 58.2201 | 96.7715 | 13.8329 |
| loveda/P | NeutralAliasShuffle1 | water | 62.5633 | -0.3609 | 67.7581 | 89.0836 | 19.3457 |
| loveda/P | NeutralAliasShuffle1 | barren | 2.3617 | 0.6893 | 74.2115 | 2.3813 | 0.3558 |
| loveda/P | NeutralAliasShuffle1 | tree | 50.1157 | -6.4754 | 97.1744 | 50.8568 | 8.6170 |
| loveda/P | NeutralAliasShuffle1 | farm | 84.7470 | -0.8368 | 85.0715 | 99.5519 | 57.7904 |
| loveda/P | NeutralAliasShuffle2 | building | 45.2802 | -3.6490 | 45.2802 | 100.0000 | 0.0585 |
| loveda/P | NeutralAliasShuffle2 | road | 57.9121 | -2.2490 | 59.0608 | 96.7508 | 13.6331 |
| loveda/P | NeutralAliasShuffle2 | water | 62.3939 | -0.5303 | 67.5388 | 89.1193 | 19.4163 |
| loveda/P | NeutralAliasShuffle2 | barren | 2.7553 | 1.0829 | 47.8620 | 2.8406 | 0.6580 |
| loveda/P | NeutralAliasShuffle2 | tree | 52.1007 | -4.4904 | 97.3551 | 52.8488 | 8.9379 |
| loveda/P | NeutralAliasShuffle2 | farm | 85.4704 | -0.1134 | 85.8027 | 99.5489 | 57.2962 |
| loveda/D | Geometry | background | 39.0474 | 8.3540 | 75.8250 | 44.5998 | 26.3232 |
| loveda/D | Geometry | building | 4.8538 | -26.7087 | 4.8538 | 100.0000 | 0.3016 |
| loveda/D | Geometry | road | 45.0691 | -4.2060 | 45.8485 | 96.3650 | 9.6637 |
| loveda/D | Geometry | water | 53.8049 | -1.1962 | 58.0149 | 88.1157 | 12.3473 |
| loveda/D | Geometry | barren | 0.0293 | -1.5884 | 0.8665 | 0.0304 | 0.2146 |
| loveda/D | Geometry | tree | 37.0607 | -11.4112 | 42.3596 | 74.7642 | 16.0550 |
| loveda/D | Geometry | farm | 57.2802 | 10.6858 | 64.7327 | 83.2647 | 35.0945 |
| loveda/D | NoAdmission_Exact | background | 7.3224 | -23.3710 | 75.1732 | 7.5039 | 4.4672 |
| loveda/D | NoAdmission_Exact | building | 31.2821 | -0.2804 | 34.0307 | 79.4788 | 0.0342 |
| loveda/D | NoAdmission_Exact | road | 44.1270 | -5.1481 | 44.7837 | 96.7840 | 9.9365 |
| loveda/D | NoAdmission_Exact | water | 54.4379 | -0.5632 | 58.7421 | 88.1369 | 12.1974 |
| loveda/D | NoAdmission_Exact | barren | 3.2414 | 1.6237 | 82.3610 | 3.2640 | 0.2428 |
| loveda/D | NoAdmission_Exact | tree | 35.8674 | -12.6045 | 97.1427 | 36.2498 | 3.3944 |
| loveda/D | NoAdmission_Exact | farm | 38.8741 | -7.7203 | 38.9455 | 99.5311 | 69.7275 |
| loveda/D | RivalFineHard_Exact | background | 25.9735 | -4.7199 | 85.4376 | 27.1766 | 14.2353 |
| loveda/D | RivalFineHard_Exact | building | 38.0886 | 6.5261 | 39.8551 | 89.5765 | 0.0329 |
| loveda/D | RivalFineHard_Exact | road | 48.5879 | -0.6872 | 49.4050 | 96.7083 | 9.0000 |
| loveda/D | RivalFineHard_Exact | water | 54.7088 | -0.2923 | 58.8346 | 88.6384 | 12.2475 |
| loveda/D | RivalFineHard_Exact | barren | 1.7765 | 0.1588 | 68.4211 | 1.7912 | 0.1604 |
| loveda/D | RivalFineHard_Exact | tree | 45.9895 | -2.4824 | 93.8092 | 47.4290 | 4.5990 |
| loveda/D | RivalFineHard_Exact | farm | 45.3841 | -1.2103 | 45.4771 | 99.5512 | 59.7249 |
| loveda/D | FineRivalProjected_Exact | background | 30.6934 | 0.0000 | 84.6654 | 32.5002 | 17.1790 |
| loveda/D | FineRivalProjected_Exact | building | 31.5625 | 0.0000 | 31.6946 | 98.6971 | 0.0456 |
| loveda/D | FineRivalProjected_Exact | road | 49.2751 | 0.0000 | 50.1318 | 96.6481 | 8.8640 |
| loveda/D | FineRivalProjected_Exact | water | 55.0011 | 0.0000 | 58.9493 | 89.1446 | 12.2935 |
| loveda/D | FineRivalProjected_Exact | barren | 1.6177 | 0.0000 | 66.8262 | 1.6308 | 0.1495 |
| loveda/D | FineRivalProjected_Exact | tree | 48.4719 | 0.0000 | 92.5745 | 50.4327 | 4.9555 |
| loveda/D | FineRivalProjected_Exact | farm | 46.5944 | 0.0000 | 47.1297 | 97.6203 | 56.5129 |
| loveda/D | FineBudgetOnly_Exact | background | 28.8149 | -1.8785 | 86.6920 | 30.1485 | 15.5634 |
| loveda/D | FineBudgetOnly_Exact | building | 30.0098 | -1.5527 | 30.0098 | 100.0000 | 0.0488 |
| loveda/D | FineBudgetOnly_Exact | road | 49.3559 | 0.0808 | 50.2207 | 96.6284 | 8.8465 |
| loveda/D | FineBudgetOnly_Exact | water | 55.1009 | 0.0998 | 59.0131 | 89.2607 | 12.2962 |
| loveda/D | FineBudgetOnly_Exact | barren | 1.7088 | 0.0911 | 70.4364 | 1.7211 | 0.1497 |
| loveda/D | FineBudgetOnly_Exact | tree | 49.0748 | 0.6029 | 92.1383 | 51.2196 | 5.0567 |
| loveda/D | FineBudgetOnly_Exact | farm | 46.6943 | 0.0999 | 46.7946 | 99.5431 | 58.0387 |
| loveda/D | RivalNeutral_Projected | background | 32.9426 | 2.2492 | 78.9275 | 36.1194 | 20.4801 |
| loveda/D | RivalNeutral_Projected | building | 26.5801 | -4.9824 | 26.5801 | 100.0000 | 0.0551 |
| loveda/D | RivalNeutral_Projected | road | 48.5963 | -0.6788 | 49.4236 | 96.6699 | 8.9931 |
| loveda/D | RivalNeutral_Projected | water | 54.7872 | -0.2139 | 58.7506 | 89.0366 | 12.3201 |
| loveda/D | RivalNeutral_Projected | barren | 1.6899 | 0.0722 | 60.5684 | 1.7087 | 0.1728 |
| loveda/D | RivalNeutral_Projected | tree | 47.5330 | -0.9389 | 92.5828 | 49.4147 | 4.8551 |
| loveda/D | RivalNeutral_Projected | farm | 44.9590 | -1.6354 | 46.9438 | 91.4041 | 53.1238 |
| loveda/D | RivalFineCap_Projected | background | 38.5436 | 7.8502 | 74.3137 | 44.4678 | 26.7791 |
| loveda/D | RivalFineCap_Projected | building | 23.1001 | -8.4624 | 23.1001 | 100.0000 | 0.0634 |
| loveda/D | RivalFineCap_Projected | road | 49.8171 | 0.5420 | 50.7080 | 96.5931 | 8.7583 |
| loveda/D | RivalFineCap_Projected | water | 55.6925 | 0.6914 | 59.6553 | 89.3434 | 12.1751 |
| loveda/D | RivalFineCap_Projected | barren | 1.4615 | -0.1562 | 57.4108 | 1.4775 | 0.1576 |
| loveda/D | RivalFineCap_Projected | tree | 53.3049 | 4.8330 | 89.2406 | 56.9659 | 5.8066 |
| loveda/D | RivalFineCap_Projected | farm | 43.5890 | -3.0054 | 48.2609 | 81.8274 | 46.2599 |
| loveda/D | HardFixedMass_Projected | background | 43.0654 | 12.3720 | 73.0855 | 51.1825 | 31.3407 |
| loveda/D | HardFixedMass_Projected | building | 21.4535 | -10.1090 | 21.4535 | 100.0000 | 0.0682 |
| loveda/D | HardFixedMass_Projected | road | 51.4749 | 2.1998 | 52.4653 | 96.4625 | 8.4535 |
| loveda/D | HardFixedMass_Projected | water | 56.8435 | 1.8424 | 60.8174 | 89.6901 | 11.9888 |
| loveda/D | HardFixedMass_Projected | barren | 1.0187 | -0.5990 | 38.8208 | 1.0353 | 0.1634 |
| loveda/D | HardFixedMass_Projected | tree | 55.2633 | 6.7914 | 82.8156 | 62.4213 | 6.8563 |
| loveda/D | HardFixedMass_Projected | farm | 43.7104 | -2.8840 | 50.5923 | 76.2660 | 41.1291 |
| loveda/D | NeutralUnprojected | background | 33.0628 | 2.3694 | 79.1229 | 36.2228 | 20.4879 |
| loveda/D | NeutralUnprojected | building | 26.8357 | -4.7268 | 26.8357 | 100.0000 | 0.0546 |
| loveda/D | NeutralUnprojected | road | 48.5785 | -0.6966 | 49.4058 | 96.6678 | 8.9961 |
| loveda/D | NeutralUnprojected | water | 54.8027 | -0.1984 | 58.7748 | 89.0220 | 12.3130 |
| loveda/D | NeutralUnprojected | barren | 1.6834 | 0.0657 | 59.1032 | 1.7032 | 0.1765 |
| loveda/D | NeutralUnprojected | tree | 47.4336 | -1.0383 | 92.6220 | 49.2963 | 4.8414 |
| loveda/D | NeutralUnprojected | farm | 45.0739 | -1.5205 | 47.0245 | 91.5727 | 53.1305 |
| loveda/D | ProjectedHardStrengthMatched | background | 36.6106 | 5.9172 | 74.1143 | 41.9784 | 25.3479 |
| loveda/D | ProjectedHardStrengthMatched | building | 28.2913 | -3.2712 | 28.3974 | 98.6971 | 0.0509 |
| loveda/D | ProjectedHardStrengthMatched | road | 48.5064 | -0.7687 | 49.3318 | 96.6657 | 9.0094 |
| loveda/D | ProjectedHardStrengthMatched | water | 55.0089 | 0.0078 | 59.0476 | 88.9410 | 12.2450 |
| loveda/D | ProjectedHardStrengthMatched | barren | 1.7257 | 0.1080 | 64.5515 | 1.7422 | 0.1653 |
| loveda/D | ProjectedHardStrengthMatched | tree | 47.2406 | -1.2313 | 93.0833 | 48.9592 | 4.7844 |
| loveda/D | ProjectedHardStrengthMatched | farm | 42.8487 | -3.7457 | 46.9059 | 83.2040 | 48.3971 |
| loveda/D | NeutralAliasShuffle0 | background | 28.6761 | -2.0173 | 83.2033 | 30.4382 | 16.3718 |
| loveda/D | NeutralAliasShuffle0 | building | 26.3068 | -5.2557 | 26.3068 | 100.0000 | 0.0556 |
| loveda/D | NeutralAliasShuffle0 | road | 46.9048 | -2.3703 | 47.6595 | 96.7342 | 9.3321 |
| loveda/D | NeutralAliasShuffle0 | water | 54.0399 | -0.9612 | 57.9727 | 88.8466 | 12.4588 |
| loveda/D | NeutralAliasShuffle0 | barren | 1.7543 | 0.1366 | 65.1489 | 1.7710 | 0.1665 |
| loveda/D | NeutralAliasShuffle0 | tree | 45.4602 | -3.0117 | 92.2535 | 47.2644 | 4.6604 |
| loveda/D | NeutralAliasShuffle0 | farm | 45.0086 | -1.5858 | 45.9073 | 95.8317 | 56.9547 |
| loveda/D | NeutralAliasShuffle1 | background | 28.9803 | -1.7131 | 86.8896 | 30.3054 | 15.6089 |
| loveda/D | NeutralAliasShuffle1 | building | 25.7550 | -5.8075 | 25.7550 | 100.0000 | 0.0568 |
| loveda/D | NeutralAliasShuffle1 | road | 46.7997 | -2.4754 | 47.5558 | 96.7145 | 9.3506 |
| loveda/D | NeutralAliasShuffle1 | water | 54.8224 | -0.1787 | 58.8861 | 88.8196 | 12.2618 |
| loveda/D | NeutralAliasShuffle1 | barren | 2.2200 | 0.6023 | 62.6626 | 2.2497 | 0.2199 |
| loveda/D | NeutralAliasShuffle1 | tree | 43.9975 | -4.4744 | 92.2762 | 45.6798 | 4.5030 |
| loveda/D | NeutralAliasShuffle1 | farm | 46.3163 | -0.2781 | 46.5459 | 98.9463 | 57.9989 |
| loveda/D | NeutralAliasShuffle2 | background | 25.5471 | -5.1463 | 81.8052 | 27.0862 | 14.8179 |
| loveda/D | NeutralAliasShuffle2 | building | 27.5090 | -4.0535 | 27.5090 | 100.0000 | 0.0532 |
| loveda/D | NeutralAliasShuffle2 | road | 47.5099 | -1.7652 | 48.2924 | 96.7020 | 9.2068 |
| loveda/D | NeutralAliasShuffle2 | water | 55.0298 | 0.0287 | 59.1627 | 88.7357 | 12.1930 |
| loveda/D | NeutralAliasShuffle2 | barren | 1.8897 | 0.2720 | 63.6741 | 1.9103 | 0.1838 |
| loveda/D | NeutralAliasShuffle2 | tree | 45.3580 | -3.1139 | 92.3687 | 47.1239 | 4.6407 |
| loveda/D | NeutralAliasShuffle2 | farm | 44.2990 | -2.2954 | 44.9189 | 96.9789 | 58.9047 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -17.2825 | 77.7719 | 47.8510 | 16.8072 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 5.2927 | 73.5287 | 98.5836 | 27.6605 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 7.2372 | 92.4475 | 55.4111 | 17.7225 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.0271 | 82.7688 | 80.0845 | 19.9693 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -15.9787 | 10.3220 | 99.7520 | 17.8406 |
| vaihingen/vaihingen | NoAdmission_Exact | impervious surface | 57.3052 | -2.0714 | 73.8225 | 71.9196 | 26.6125 |
| vaihingen/vaihingen | NoAdmission_Exact | building | 65.7437 | -1.7237 | 65.8760 | 99.6954 | 31.2219 |
| vaihingen/vaihingen | NoAdmission_Exact | low vegetation | 43.7153 | -2.0589 | 96.1772 | 44.4883 | 13.6772 |
| vaihingen/vaihingen | NoAdmission_Exact | tree | 67.2612 | -0.3522 | 79.0950 | 81.8036 | 21.3454 |
| vaihingen/vaihingen | NoAdmission_Exact | car | 25.1095 | -1.1885 | 25.2570 | 97.7270 | 7.1430 |
| vaihingen/vaihingen | RivalFineHard_Exact | impervious surface | 58.8103 | -0.5663 | 74.7374 | 73.4017 | 26.8285 |
| vaihingen/vaihingen | RivalFineHard_Exact | building | 67.0764 | -0.3910 | 67.2196 | 99.6834 | 30.5942 |
| vaihingen/vaihingen | RivalFineHard_Exact | low vegetation | 44.9502 | -0.8240 | 95.9989 | 45.8084 | 14.1092 |
| vaihingen/vaihingen | RivalFineHard_Exact | tree | 67.5208 | -0.0926 | 78.7110 | 82.6067 | 21.6601 |
| vaihingen/vaihingen | RivalFineHard_Exact | car | 26.3654 | 0.0674 | 26.5220 | 97.8096 | 6.8081 |
| vaihingen/vaihingen | FineRivalProjected_Exact | impervious surface | 59.3766 | 0.0000 | 75.4775 | 73.5690 | 26.6259 |
| vaihingen/vaihingen | FineRivalProjected_Exact | building | 67.4674 | 0.0000 | 67.6072 | 99.6944 | 30.4222 |
| vaihingen/vaihingen | FineRivalProjected_Exact | low vegetation | 45.7742 | 0.0000 | 95.8882 | 46.6907 | 14.3975 |
| vaihingen/vaihingen | FineRivalProjected_Exact | tree | 67.6134 | 0.0000 | 78.6701 | 82.7907 | 21.7196 |
| vaihingen/vaihingen | FineRivalProjected_Exact | car | 26.2980 | 0.0000 | 26.4463 | 97.9130 | 6.8348 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | impervious surface | 59.3423 | -0.0343 | 76.2324 | 72.8142 | 26.0918 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | building | 67.4593 | -0.0081 | 67.5948 | 99.7037 | 30.4306 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | low vegetation | 46.6048 | 0.8306 | 95.6768 | 47.6074 | 14.7126 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | tree | 67.6624 | 0.0490 | 78.6651 | 82.8697 | 21.7417 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | car | 25.6512 | -0.6468 | 25.7806 | 98.0808 | 7.0233 |
| vaihingen/vaihingen | RivalNeutral_Projected | impervious surface | 59.5573 | 0.1807 | 76.3976 | 72.9866 | 26.0971 |
| vaihingen/vaihingen | RivalNeutral_Projected | building | 67.9358 | 0.4684 | 68.0654 | 99.7206 | 30.2253 |
| vaihingen/vaihingen | RivalNeutral_Projected | low vegetation | 46.8601 | 1.0859 | 95.6098 | 47.8905 | 14.8105 |
| vaihingen/vaihingen | RivalNeutral_Projected | tree | 67.7410 | 0.1276 | 78.6673 | 82.9852 | 21.7714 |
| vaihingen/vaihingen | RivalNeutral_Projected | car | 25.4343 | -0.8637 | 25.5524 | 98.2152 | 7.0957 |
| vaihingen/vaihingen | RivalFineCap_Projected | impervious surface | 60.8263 | 1.4497 | 77.7155 | 73.6767 | 25.8970 |
| vaihingen/vaihingen | RivalFineCap_Projected | building | 69.2850 | 1.8176 | 69.4205 | 99.7192 | 29.6349 |
| vaihingen/vaihingen | RivalFineCap_Projected | low vegetation | 48.6762 | 2.9020 | 94.8202 | 50.0059 | 15.5935 |
| vaihingen/vaihingen | RivalFineCap_Projected | tree | 67.8181 | 0.2047 | 78.4361 | 83.3604 | 21.9343 |
| vaihingen/vaihingen | RivalFineCap_Projected | car | 26.0787 | -0.2193 | 26.1864 | 98.4476 | 6.9403 |
| vaihingen/vaihingen | HardFixedMass_Projected | impervious surface | 61.9472 | 2.5706 | 79.0381 | 74.1255 | 25.6188 |
| vaihingen/vaihingen | HardFixedMass_Projected | building | 70.9264 | 3.4590 | 71.1124 | 99.6325 | 28.9046 |
| vaihingen/vaihingen | HardFixedMass_Projected | low vegetation | 50.7946 | 5.0204 | 93.5530 | 52.6371 | 16.6363 |
| vaihingen/vaihingen | HardFixedMass_Projected | tree | 67.9433 | 0.3299 | 78.3806 | 83.6127 | 22.0163 |
| vaihingen/vaihingen | HardFixedMass_Projected | car | 26.5629 | 0.2649 | 26.6657 | 98.5690 | 6.8240 |
| vaihingen/vaihingen | NeutralUnprojected | impervious surface | 59.5684 | 0.1918 | 76.3803 | 73.0191 | 26.1146 |
| vaihingen/vaihingen | NeutralUnprojected | building | 67.9401 | 0.4727 | 68.0714 | 99.7169 | 30.2215 |
| vaihingen/vaihingen | NeutralUnprojected | low vegetation | 46.8318 | 1.0576 | 95.6208 | 47.8583 | 14.7988 |
| vaihingen/vaihingen | NeutralUnprojected | tree | 67.7673 | 0.1539 | 78.6793 | 83.0113 | 21.7750 |
| vaihingen/vaihingen | NeutralUnprojected | car | 25.4532 | -0.8448 | 25.5717 | 98.2126 | 7.0902 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | impervious surface | 59.4675 | 0.0909 | 75.5996 | 73.5925 | 26.5914 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | building | 67.6764 | 0.2090 | 67.8182 | 99.6919 | 30.3267 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | low vegetation | 46.2522 | 0.4780 | 95.7255 | 47.2276 | 14.5878 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | tree | 67.6907 | 0.0773 | 78.6817 | 82.8937 | 21.7434 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | car | 26.6224 | 0.3244 | 26.7747 | 97.9078 | 6.7506 |
| vaihingen/vaihingen | NeutralAliasShuffle0 | impervious surface | 58.8793 | -0.4973 | 76.1036 | 72.2339 | 25.9277 |
| vaihingen/vaihingen | NeutralAliasShuffle0 | building | 67.7701 | 0.3027 | 67.8965 | 99.7261 | 30.3021 |
| vaihingen/vaihingen | NeutralAliasShuffle0 | low vegetation | 46.5386 | 0.7644 | 95.5423 | 47.5716 | 14.7223 |
| vaihingen/vaihingen | NeutralAliasShuffle0 | tree | 67.6925 | 0.0791 | 78.7663 | 82.8027 | 21.6962 |
| vaihingen/vaihingen | NeutralAliasShuffle0 | car | 24.5615 | -1.7365 | 24.6699 | 98.2436 | 7.3517 |
| vaihingen/vaihingen | NeutralAliasShuffle1 | impervious surface | 58.5604 | -0.8162 | 75.1631 | 72.6112 | 26.3892 |
| vaihingen/vaihingen | NeutralAliasShuffle1 | building | 67.6364 | 0.1690 | 67.7680 | 99.7136 | 30.3558 |
| vaihingen/vaihingen | NeutralAliasShuffle1 | low vegetation | 46.1105 | 0.3363 | 95.6810 | 47.0907 | 14.5523 |
| vaihingen/vaihingen | NeutralAliasShuffle1 | tree | 67.6655 | 0.0521 | 79.0005 | 82.5053 | 21.5542 |
| vaihingen/vaihingen | NeutralAliasShuffle1 | car | 25.2533 | -1.0447 | 25.3685 | 98.2332 | 7.1485 |
| vaihingen/vaihingen | NeutralAliasShuffle2 | impervious surface | 59.1444 | -0.2322 | 76.1662 | 72.5764 | 26.0292 |
| vaihingen/vaihingen | NeutralAliasShuffle2 | building | 67.5659 | 0.0985 | 67.6965 | 99.7152 | 30.3884 |
| vaihingen/vaihingen | NeutralAliasShuffle2 | low vegetation | 46.6383 | 0.8641 | 95.6768 | 47.6423 | 14.7234 |
| vaihingen/vaihingen | NeutralAliasShuffle2 | tree | 67.6669 | 0.0535 | 78.8022 | 82.7248 | 21.6660 |
| vaihingen/vaihingen | NeutralAliasShuffle2 | car | 25.0901 | -1.2079 | 25.2053 | 98.2100 | 7.1930 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -5.6280 | 96.6505 | 85.0437 | 59.6226 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.6375 | 35.0436 | 99.2919 | 4.2927 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -3.1775 | 86.4992 | 89.2841 | 22.0960 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9444 | 93.6635 | 100.0000 | 8.8309 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -12.0061 | 15.6288 | 77.0019 | 5.1578 |
| landcoverai/landcoverai | NoAdmission_Exact | background | 87.4477 | -0.7893 | 93.8549 | 92.7587 | 66.9686 |
| landcoverai/landcoverai | NoAdmission_Exact | building | 44.4672 | -0.1265 | 44.8124 | 98.2973 | 3.3233 |
| landcoverai/landcoverai | NoAdmission_Exact | woodland | 78.6026 | -2.9387 | 94.9565 | 82.0272 | 18.4920 |
| landcoverai/landcoverai | NoAdmission_Exact | water | 97.6178 | 0.0099 | 97.6178 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | NoAdmission_Exact | road | 26.3947 | -0.5432 | 28.8528 | 75.5990 | 2.7429 |
| landcoverai/landcoverai | RivalFineHard_Exact | background | 87.9828 | -0.2542 | 94.6196 | 92.6163 | 66.3254 |
| landcoverai/landcoverai | RivalFineHard_Exact | building | 44.5590 | -0.0347 | 44.9037 | 98.3067 | 3.3169 |
| landcoverai/landcoverai | RivalFineHard_Exact | woodland | 80.5571 | -0.9842 | 94.4300 | 84.5759 | 19.1729 |
| landcoverai/landcoverai | RivalFineHard_Exact | water | 97.6134 | 0.0055 | 97.6134 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | RivalFineHard_Exact | road | 26.7045 | -0.2334 | 29.2139 | 75.6627 | 2.7113 |
| landcoverai/landcoverai | FineRivalProjected_Exact | background | 88.2370 | 0.0000 | 95.1004 | 92.4393 | 65.8639 |
| landcoverai/landcoverai | FineRivalProjected_Exact | building | 44.5937 | 0.0000 | 44.9185 | 98.4043 | 3.3191 |
| landcoverai/landcoverai | FineRivalProjected_Exact | woodland | 81.5413 | 0.0000 | 93.8335 | 86.1583 | 19.6558 |
| landcoverai/landcoverai | FineRivalProjected_Exact | water | 97.6079 | 0.0000 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineRivalProjected_Exact | road | 26.9379 | 0.0000 | 29.4886 | 75.6946 | 2.6872 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | background | 88.0617 | -0.1753 | 94.8162 | 92.5159 | 66.1161 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | building | 44.5675 | -0.0262 | 44.8926 | 98.4012 | 3.3209 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | woodland | 80.8303 | -0.7110 | 94.0020 | 85.2259 | 19.4082 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | water | 97.6173 | 0.0094 | 97.6173 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | road | 26.9705 | 0.0326 | 29.5339 | 75.6536 | 2.6816 |
| landcoverai/landcoverai | RivalNeutral_Projected | background | 88.0983 | -0.1387 | 94.8076 | 92.5645 | 66.1568 |
| landcoverai/landcoverai | RivalNeutral_Projected | building | 44.5606 | -0.0331 | 44.8934 | 98.3634 | 3.3195 |
| landcoverai/landcoverai | RivalNeutral_Projected | woodland | 80.8714 | -0.6699 | 94.0972 | 85.1933 | 19.3812 |
| landcoverai/landcoverai | RivalNeutral_Projected | water | 97.6250 | 0.0171 | 97.6250 | 100.0000 | 8.4725 |
| landcoverai/landcoverai | RivalNeutral_Projected | road | 27.0824 | 0.1445 | 29.6667 | 75.6627 | 2.6699 |
| landcoverai/landcoverai | RivalFineCap_Projected | background | 88.6492 | 0.4122 | 95.6688 | 92.3559 | 65.4136 |
| landcoverai/landcoverai | RivalFineCap_Projected | building | 44.6819 | 0.0882 | 45.0041 | 98.4232 | 3.3134 |
| landcoverai/landcoverai | RivalFineCap_Projected | woodland | 82.8117 | 1.2704 | 93.3567 | 87.9973 | 20.1779 |
| landcoverai/landcoverai | RivalFineCap_Projected | water | 97.6272 | 0.0193 | 97.6272 | 100.0000 | 8.4723 |
| landcoverai/landcoverai | RivalFineCap_Projected | road | 27.5385 | 0.6006 | 30.2105 | 75.6901 | 2.6228 |
| landcoverai/landcoverai | HardFixedMass_Projected | background | 88.8974 | 0.6604 | 96.1081 | 92.2171 | 65.0167 |
| landcoverai/landcoverai | HardFixedMass_Projected | building | 44.8945 | 0.3008 | 45.2158 | 98.4421 | 3.2985 |
| landcoverai/landcoverai | HardFixedMass_Projected | woodland | 84.5119 | 2.9706 | 92.4991 | 90.7298 | 20.9973 |
| landcoverai/landcoverai | HardFixedMass_Projected | water | 94.2712 | -3.3367 | 97.5331 | 96.5739 | 8.1900 |
| landcoverai/landcoverai | HardFixedMass_Projected | road | 28.8014 | 1.8635 | 31.7340 | 75.7083 | 2.4975 |
| landcoverai/landcoverai | NeutralUnprojected | background | 88.0946 | -0.1424 | 94.8063 | 92.5617 | 66.1557 |
| landcoverai/landcoverai | NeutralUnprojected | building | 44.5396 | -0.0541 | 44.8721 | 98.3634 | 3.3211 |
| landcoverai/landcoverai | NeutralUnprojected | woodland | 80.8692 | -0.6721 | 94.0988 | 85.1895 | 19.3800 |
| landcoverai/landcoverai | NeutralUnprojected | water | 97.6228 | 0.0149 | 97.6228 | 100.0000 | 8.4727 |
| landcoverai/landcoverai | NeutralUnprojected | road | 27.0776 | 0.1397 | 29.6609 | 75.6627 | 2.6704 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | background | 88.0036 | -0.2334 | 94.7724 | 92.4935 | 66.1306 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | building | 44.6492 | 0.0555 | 44.9827 | 98.3665 | 3.3131 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | woodland | 80.7478 | -0.7935 | 94.0346 | 85.1074 | 19.3745 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | water | 97.5425 | -0.0654 | 97.5914 | 99.9487 | 8.4711 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | road | 26.7300 | -0.2079 | 29.2376 | 75.7083 | 2.7107 |
| landcoverai/landcoverai | NeutralAliasShuffle0 | background | 88.0133 | -0.2237 | 94.7187 | 92.5554 | 66.2124 |
| landcoverai/landcoverai | NeutralAliasShuffle0 | building | 44.3525 | -0.2412 | 44.6771 | 98.3886 | 3.3365 |
| landcoverai/landcoverai | NeutralAliasShuffle0 | woodland | 80.7120 | -0.8293 | 94.2371 | 84.9026 | 19.2864 |
| landcoverai/landcoverai | NeutralAliasShuffle0 | water | 97.6283 | 0.0204 | 97.6283 | 100.0000 | 8.4723 |
| landcoverai/landcoverai | NeutralAliasShuffle0 | road | 26.8526 | -0.0853 | 29.3987 | 75.6126 | 2.6925 |
| landcoverai/landcoverai | NeutralAliasShuffle1 | background | 88.0092 | -0.2278 | 94.6458 | 92.6206 | 66.3100 |
| landcoverai/landcoverai | NeutralAliasShuffle1 | building | 44.4467 | -0.1470 | 44.7771 | 98.3665 | 3.3283 |
| landcoverai/landcoverai | NeutralAliasShuffle1 | woodland | 80.5417 | -0.9996 | 94.3094 | 84.6558 | 19.2156 |
| landcoverai/landcoverai | NeutralAliasShuffle1 | water | 97.6239 | 0.0160 | 97.6239 | 100.0000 | 8.4726 |
| landcoverai/landcoverai | NeutralAliasShuffle1 | road | 27.0225 | 0.0846 | 29.6039 | 75.6035 | 2.6735 |
| landcoverai/landcoverai | NeutralAliasShuffle2 | background | 87.8093 | -0.4277 | 94.4576 | 92.5793 | 66.4125 |
| landcoverai/landcoverai | NeutralAliasShuffle2 | building | 44.4568 | -0.1369 | 44.7926 | 98.3414 | 3.3263 |
| landcoverai/landcoverai | NeutralAliasShuffle2 | woodland | 79.9786 | -1.5627 | 94.2918 | 84.0480 | 19.0812 |
| landcoverai/landcoverai | NeutralAliasShuffle2 | water | 97.6332 | 0.0253 | 97.6332 | 100.0000 | 8.4718 |
| landcoverai/landcoverai | NeutralAliasShuffle2 | road | 26.7120 | -0.2259 | 29.2297 | 75.6172 | 2.7082 |
| flair1/flair1 | Geometry | building | 49.6979 | -7.7062 | 50.7258 | 96.0825 | 13.5991 |
| flair1/flair1 | Geometry | pervious surface | 57.0621 | 10.2549 | 92.1299 | 59.9861 | 11.1728 |
| flair1/flair1 | Geometry | impervious surface | 52.6509 | -0.2497 | 62.6245 | 76.7765 | 20.0846 |
| flair1/flair1 | Geometry | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4411 |
| flair1/flair1 | Geometry | water | 73.2328 | 8.4491 | 77.1432 | 93.5263 | 5.3781 |
| flair1/flair1 | Geometry | coniferous | 43.0233 | 0.2968 | 61.7114 | 58.6897 | 0.5346 |
| flair1/flair1 | Geometry | deciduous | 54.0229 | -6.8819 | 78.9723 | 63.0995 | 13.6004 |
| flair1/flair1 | Geometry | brushwood | 18.6136 | 2.9830 | 23.4212 | 47.5559 | 9.9866 |
| flair1/flair1 | Geometry | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | Geometry | herbaceous vegetation | 60.2329 | 2.6126 | 94.3169 | 62.5013 | 21.2289 |
| flair1/flair1 | Geometry | agricultural land | 18.7339 | 5.3034 | 21.6468 | 58.1977 | 0.8198 |
| flair1/flair1 | Geometry | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1540 |
| flair1/flair1 | NoAdmission_Exact | building | 56.7755 | -0.6286 | 58.0396 | 96.3057 | 11.9131 |
| flair1/flair1 | NoAdmission_Exact | pervious surface | 47.2753 | 0.4681 | 91.6998 | 49.3887 | 9.2421 |
| flair1/flair1 | NoAdmission_Exact | impervious surface | 51.9858 | -0.9148 | 59.7781 | 79.9522 | 21.9112 |
| flair1/flair1 | NoAdmission_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 |
| flair1/flair1 | NoAdmission_Exact | water | 61.4565 | -3.3272 | 63.2502 | 95.5889 | 6.7041 |
| flair1/flair1 | NoAdmission_Exact | coniferous | 43.5282 | 0.8017 | 65.7321 | 56.3052 | 0.4815 |
| flair1/flair1 | NoAdmission_Exact | deciduous | 59.5237 | -1.3811 | 78.8015 | 70.8722 | 15.3089 |
| flair1/flair1 | NoAdmission_Exact | brushwood | 12.3287 | -3.3019 | 26.2213 | 18.8771 | 3.5408 |
| flair1/flair1 | NoAdmission_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | NoAdmission_Exact | herbaceous vegetation | 57.4725 | -0.1478 | 90.8252 | 61.0148 | 21.5207 |
| flair1/flair1 | NoAdmission_Exact | agricultural land | 12.9546 | -0.4759 | 13.9145 | 65.2534 | 1.4300 |
| flair1/flair1 | NoAdmission_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 |
| flair1/flair1 | RivalFineHard_Exact | building | 57.1752 | -0.2289 | 58.4382 | 96.3576 | 11.8382 |
| flair1/flair1 | RivalFineHard_Exact | pervious surface | 46.6274 | -0.1798 | 91.7085 | 48.6795 | 9.1085 |
| flair1/flair1 | RivalFineHard_Exact | impervious surface | 52.5746 | -0.3260 | 60.4793 | 80.0896 | 21.6944 |
| flair1/flair1 | RivalFineHard_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1744 |
| flair1/flair1 | RivalFineHard_Exact | water | 63.6878 | -1.0959 | 65.7363 | 95.3351 | 6.4334 |
| flair1/flair1 | RivalFineHard_Exact | coniferous | 42.7771 | 0.0506 | 63.1055 | 57.0434 | 0.5081 |
| flair1/flair1 | RivalFineHard_Exact | deciduous | 60.3808 | -0.5240 | 78.8158 | 72.0787 | 15.5666 |
| flair1/flair1 | RivalFineHard_Exact | brushwood | 14.8835 | -0.7471 | 28.7712 | 23.5675 | 4.0288 |
| flair1/flair1 | RivalFineHard_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0207 |
| flair1/flair1 | RivalFineHard_Exact | herbaceous vegetation | 57.5245 | -0.0958 | 91.4034 | 60.8148 | 21.3145 |
| flair1/flair1 | RivalFineHard_Exact | agricultural land | 13.1175 | -0.3130 | 14.1254 | 64.7685 | 1.3981 |
| flair1/flair1 | RivalFineHard_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9143 |
| flair1/flair1 | FineRivalProjected_Exact | building | 57.4041 | 0.0000 | 58.6643 | 96.3928 | 11.7969 |
| flair1/flair1 | FineRivalProjected_Exact | pervious surface | 46.8072 | 0.0000 | 91.9102 | 48.8185 | 9.1145 |
| flair1/flair1 | FineRivalProjected_Exact | impervious surface | 52.9006 | 0.0000 | 60.8089 | 80.2670 | 21.6246 |
| flair1/flair1 | FineRivalProjected_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0982 |
| flair1/flair1 | FineRivalProjected_Exact | water | 64.7837 | 0.0000 | 66.9300 | 95.2835 | 6.3152 |
| flair1/flair1 | FineRivalProjected_Exact | coniferous | 42.7265 | 0.0000 | 62.4459 | 57.5017 | 0.5176 |
| flair1/flair1 | FineRivalProjected_Exact | deciduous | 60.9048 | 0.0000 | 78.7253 | 72.9040 | 15.7630 |
| flair1/flair1 | FineRivalProjected_Exact | brushwood | 15.6306 | 0.0000 | 29.7469 | 24.7769 | 4.0967 |
| flair1/flair1 | FineRivalProjected_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineRivalProjected_Exact | herbaceous vegetation | 57.6203 | 0.0000 | 91.5155 | 60.8721 | 21.3084 |
| flair1/flair1 | FineRivalProjected_Exact | agricultural land | 13.4305 | 0.0000 | 14.5039 | 64.4712 | 1.3554 |
| flair1/flair1 | FineRivalProjected_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9893 |
| flair1/flair1 | FineBudgetOnly_Exact | building | 57.3732 | -0.0309 | 58.6294 | 96.4001 | 11.8048 |
| flair1/flair1 | FineBudgetOnly_Exact | pervious surface | 47.2072 | 0.4000 | 92.1302 | 49.1908 | 9.1621 |
| flair1/flair1 | FineBudgetOnly_Exact | impervious surface | 52.9893 | 0.0887 | 60.9714 | 80.1886 | 21.5459 |
| flair1/flair1 | FineBudgetOnly_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0326 |
| flair1/flair1 | FineBudgetOnly_Exact | water | 65.2431 | 0.4594 | 67.4205 | 95.2835 | 6.2693 |
| flair1/flair1 | FineBudgetOnly_Exact | coniferous | 42.6496 | -0.0769 | 62.2418 | 57.5356 | 0.5196 |
| flair1/flair1 | FineBudgetOnly_Exact | deciduous | 60.7870 | -0.1178 | 78.7072 | 72.7507 | 15.7335 |
| flair1/flair1 | FineBudgetOnly_Exact | brushwood | 15.8816 | 0.2510 | 29.5663 | 25.5470 | 4.2498 |
| flair1/flair1 | FineBudgetOnly_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0168 |
| flair1/flair1 | FineBudgetOnly_Exact | herbaceous vegetation | 57.7631 | 0.1428 | 91.5243 | 61.0276 | 21.3608 |
| flair1/flair1 | FineBudgetOnly_Exact | agricultural land | 13.5617 | 0.1312 | 14.6595 | 64.4243 | 1.3400 |
| flair1/flair1 | FineBudgetOnly_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9648 |
| flair1/flair1 | RivalNeutral_Projected | building | 57.4001 | -0.0040 | 58.6530 | 96.4121 | 11.8015 |
| flair1/flair1 | RivalNeutral_Projected | pervious surface | 46.7187 | -0.0885 | 91.7785 | 48.7593 | 9.1165 |
| flair1/flair1 | RivalNeutral_Projected | impervious surface | 52.9532 | 0.0526 | 60.9046 | 80.2215 | 21.5784 |
| flair1/flair1 | RivalNeutral_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1218 |
| flair1/flair1 | RivalNeutral_Projected | water | 64.3157 | -0.4680 | 66.3717 | 95.4050 | 6.3765 |
| flair1/flair1 | RivalNeutral_Projected | coniferous | 42.7350 | 0.0085 | 62.7265 | 57.2811 | 0.5133 |
| flair1/flair1 | RivalNeutral_Projected | deciduous | 60.5861 | -0.3187 | 78.6754 | 72.4901 | 15.6834 |
| flair1/flair1 | RivalNeutral_Projected | brushwood | 15.1196 | -0.5110 | 29.0868 | 23.9467 | 4.0492 |
| flair1/flair1 | RivalNeutral_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0177 |
| flair1/flair1 | RivalNeutral_Projected | herbaceous vegetation | 57.8157 | 0.1954 | 91.3358 | 61.1705 | 21.4550 |
| flair1/flair1 | RivalNeutral_Projected | agricultural land | 13.5453 | 0.1148 | 14.6228 | 64.7685 | 1.3506 |
| flair1/flair1 | RivalNeutral_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9360 |
| flair1/flair1 | RivalFineCap_Projected | building | 57.6853 | 0.2812 | 58.9282 | 96.4725 | 11.7537 |
| flair1/flair1 | RivalFineCap_Projected | pervious surface | 46.4709 | -0.3363 | 91.8527 | 48.4688 | 9.0549 |
| flair1/flair1 | RivalFineCap_Projected | impervious surface | 53.4264 | 0.5258 | 61.4624 | 80.3392 | 21.4139 |
| flair1/flair1 | RivalFineCap_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1137 |
| flair1/flair1 | RivalFineCap_Projected | water | 65.6848 | 0.9011 | 67.9087 | 95.2512 | 6.2221 |
| flair1/flair1 | RivalFineCap_Projected | coniferous | 41.8459 | -0.8806 | 60.1003 | 57.9430 | 0.5420 |
| flair1/flair1 | RivalFineCap_Projected | deciduous | 61.1682 | 0.2634 | 78.6564 | 73.3415 | 15.8715 |
| flair1/flair1 | RivalFineCap_Projected | brushwood | 16.5864 | 0.9558 | 30.5841 | 26.6003 | 4.2777 |
| flair1/flair1 | RivalFineCap_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0247 |
| flair1/flair1 | RivalFineCap_Projected | herbaceous vegetation | 57.9256 | 0.3053 | 91.5681 | 61.1895 | 21.4072 |
| flair1/flair1 | RivalFineCap_Projected | agricultural land | 13.8851 | 0.4546 | 15.0365 | 64.4556 | 1.3071 |
| flair1/flair1 | RivalFineCap_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0116 |
| flair1/flair1 | HardFixedMass_Projected | building | 58.3790 | 0.9749 | 59.5933 | 96.6273 | 11.6412 |
| flair1/flair1 | HardFixedMass_Projected | pervious surface | 46.0062 | -0.8010 | 92.0247 | 47.9167 | 8.9350 |
| flair1/flair1 | HardFixedMass_Projected | impervious surface | 54.3475 | 1.4469 | 62.4904 | 80.6603 | 21.1458 |
| flair1/flair1 | HardFixedMass_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1172 |
| flair1/flair1 | HardFixedMass_Projected | water | 68.8020 | 4.0183 | 71.3958 | 94.9845 | 5.9016 |
| flair1/flair1 | HardFixedMass_Projected | coniferous | 41.0048 | -1.7217 | 58.0062 | 58.3164 | 0.5651 |
| flair1/flair1 | HardFixedMass_Projected | deciduous | 61.6430 | 0.7382 | 78.5921 | 74.0822 | 16.0449 |
| flair1/flair1 | HardFixedMass_Projected | brushwood | 19.6041 | 3.9735 | 32.6903 | 32.8736 | 4.9460 |
| flair1/flair1 | HardFixedMass_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0405 |
| flair1/flair1 | HardFixedMass_Projected | herbaceous vegetation | 57.7945 | 0.1742 | 91.9527 | 60.8735 | 21.2075 |
| flair1/flair1 | HardFixedMass_Projected | agricultural land | 14.4678 | 1.0373 | 15.7858 | 63.4074 | 1.2248 |
| flair1/flair1 | HardFixedMass_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.2304 |
| flair1/flair1 | NeutralUnprojected | building | 57.4003 | -0.0038 | 58.6559 | 96.4047 | 11.8000 |
| flair1/flair1 | NeutralUnprojected | pervious surface | 46.6863 | -0.1209 | 91.7852 | 48.7221 | 9.1089 |
| flair1/flair1 | NeutralUnprojected | impervious surface | 52.9605 | 0.0599 | 60.9144 | 80.2212 | 21.5748 |
| flair1/flair1 | NeutralUnprojected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1315 |
| flair1/flair1 | NeutralUnprojected | water | 64.2780 | -0.5057 | 66.3315 | 95.4050 | 6.3803 |
| flair1/flair1 | NeutralUnprojected | coniferous | 42.7314 | 0.0049 | 62.7289 | 57.2726 | 0.5132 |
| flair1/flair1 | NeutralUnprojected | deciduous | 60.6081 | -0.2967 | 78.6775 | 72.5198 | 15.6894 |
| flair1/flair1 | NeutralUnprojected | brushwood | 15.0893 | -0.5413 | 29.0907 | 23.8681 | 4.0354 |
| flair1/flair1 | NeutralUnprojected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0180 |
| flair1/flair1 | NeutralUnprojected | herbaceous vegetation | 57.8145 | 0.1942 | 91.3291 | 61.1722 | 21.4571 |
| flair1/flair1 | NeutralUnprojected | agricultural land | 13.5674 | 0.1369 | 14.6469 | 64.7997 | 1.3490 |
| flair1/flair1 | NeutralUnprojected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9422 |
| flair1/flair1 | ProjectedHardStrengthMatched | building | 57.3299 | -0.0742 | 58.5924 | 96.3775 | 11.8094 |
| flair1/flair1 | ProjectedHardStrengthMatched | pervious surface | 46.8556 | 0.0484 | 91.8603 | 48.8852 | 9.1319 |
| flair1/flair1 | ProjectedHardStrengthMatched | impervious surface | 52.7516 | -0.1490 | 60.6288 | 80.2378 | 21.6810 |
| flair1/flair1 | ProjectedHardStrengthMatched | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1054 |
| flair1/flair1 | ProjectedHardStrengthMatched | water | 64.1716 | -0.6121 | 66.2421 | 95.3555 | 6.3856 |
| flair1/flair1 | ProjectedHardStrengthMatched | coniferous | 42.8899 | 0.1634 | 63.1535 | 57.2047 | 0.5092 |
| flair1/flair1 | ProjectedHardStrengthMatched | deciduous | 60.6643 | -0.2405 | 78.7159 | 72.5677 | 15.6922 |
| flair1/flair1 | ProjectedHardStrengthMatched | brushwood | 14.9925 | -0.6381 | 29.0559 | 23.6499 | 4.0033 |
| flair1/flair1 | ProjectedHardStrengthMatched | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0195 |
| flair1/flair1 | ProjectedHardStrengthMatched | herbaceous vegetation | 57.6722 | 0.0519 | 91.4320 | 60.9671 | 21.3612 |
| flair1/flair1 | ProjectedHardStrengthMatched | agricultural land | 13.3720 | -0.0585 | 14.4327 | 64.5338 | 1.3634 |
| flair1/flair1 | ProjectedHardStrengthMatched | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9380 |
| flair1/flair1 | NeutralAliasShuffle0 | building | 57.4697 | 0.0656 | 58.5984 | 96.7569 | 11.8547 |
| flair1/flair1 | NeutralAliasShuffle0 | pervious surface | 47.0705 | 0.2633 | 91.8715 | 49.1160 | 9.1739 |
| flair1/flair1 | NeutralAliasShuffle0 | impervious surface | 52.7592 | -0.1414 | 60.6572 | 80.2058 | 21.6622 |
| flair1/flair1 | NeutralAliasShuffle0 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0876 |
| flair1/flair1 | NeutralAliasShuffle0 | water | 63.9875 | -0.7962 | 65.9944 | 95.4631 | 6.4168 |
| flair1/flair1 | NeutralAliasShuffle0 | coniferous | 42.8836 | 0.1571 | 63.4952 | 56.9162 | 0.5039 |
| flair1/flair1 | NeutralAliasShuffle0 | deciduous | 60.1107 | -0.7941 | 78.5578 | 71.9088 | 15.5810 |
| flair1/flair1 | NeutralAliasShuffle0 | brushwood | 14.2110 | -1.4196 | 27.9347 | 22.4366 | 3.9503 |
| flair1/flair1 | NeutralAliasShuffle0 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0104 |
| flair1/flair1 | NeutralAliasShuffle0 | herbaceous vegetation | 57.9428 | 0.3225 | 91.2629 | 61.3458 | 21.5336 |
| flair1/flair1 | NeutralAliasShuffle0 | agricultural land | 13.3863 | -0.0442 | 14.4329 | 64.8623 | 1.3703 |
| flair1/flair1 | NeutralAliasShuffle0 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8552 |
| flair1/flair1 | NeutralAliasShuffle1 | building | 57.3369 | -0.0672 | 58.5071 | 96.6293 | 11.8576 |
| flair1/flair1 | NeutralAliasShuffle1 | pervious surface | 47.2692 | 0.4620 | 92.1044 | 49.2655 | 9.1786 |
| flair1/flair1 | NeutralAliasShuffle1 | impervious surface | 52.7646 | -0.1360 | 60.6635 | 80.2070 | 21.6602 |
| flair1/flair1 | NeutralAliasShuffle1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0383 |
| flair1/flair1 | NeutralAliasShuffle1 | water | 63.2108 | -1.5729 | 65.1510 | 95.5007 | 6.5025 |
| flair1/flair1 | NeutralAliasShuffle1 | coniferous | 42.8854 | 0.1589 | 63.2578 | 57.1113 | 0.5075 |
| flair1/flair1 | NeutralAliasShuffle1 | deciduous | 60.4444 | -0.4604 | 78.6740 | 72.2886 | 15.6401 |
| flair1/flair1 | NeutralAliasShuffle1 | brushwood | 13.9978 | -1.6328 | 28.0726 | 21.8255 | 3.8239 |
| flair1/flair1 | NeutralAliasShuffle1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0094 |
| flair1/flair1 | NeutralAliasShuffle1 | herbaceous vegetation | 58.0082 | 0.3879 | 91.2031 | 61.4462 | 21.5830 |
| flair1/flair1 | NeutralAliasShuffle1 | agricultural land | 13.3152 | -0.1153 | 14.3464 | 64.9406 | 1.3802 |
| flair1/flair1 | NeutralAliasShuffle1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8186 |
| flair1/flair1 | NeutralAliasShuffle2 | building | 57.3001 | -0.1040 | 58.5398 | 96.4360 | 11.8272 |
| flair1/flair1 | NeutralAliasShuffle2 | pervious surface | 47.2731 | 0.4659 | 91.8143 | 49.3531 | 9.2239 |
| flair1/flair1 | NeutralAliasShuffle2 | impervious surface | 52.6166 | -0.2840 | 60.5010 | 80.1490 | 21.7027 |
| flair1/flair1 | NeutralAliasShuffle2 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0165 |
| flair1/flair1 | NeutralAliasShuffle2 | water | 64.5069 | -0.2768 | 66.6021 | 95.3501 | 6.3508 |
| flair1/flair1 | NeutralAliasShuffle2 | coniferous | 42.8781 | 0.1516 | 63.3255 | 57.0434 | 0.5064 |
| flair1/flair1 | NeutralAliasShuffle2 | deciduous | 60.3469 | -0.5579 | 78.5489 | 72.2547 | 15.6577 |
| flair1/flair1 | NeutralAliasShuffle2 | brushwood | 15.0479 | -0.5827 | 29.7810 | 23.3231 | 3.8518 |
| flair1/flair1 | NeutralAliasShuffle2 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0118 |
| flair1/flair1 | NeutralAliasShuffle2 | herbaceous vegetation | 57.9728 | 0.3525 | 91.0494 | 61.4764 | 21.6301 |
| flair1/flair1 | NeutralAliasShuffle2 | agricultural land | 13.4578 | 0.0273 | 14.5224 | 64.7372 | 1.3593 |
| flair1/flair1 | NeutralAliasShuffle2 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8618 |

All source/output options preserved. No claim of SOTA or every-alias utility from these developed windows.
