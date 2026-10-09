# Rival-margin soft attenuation

Same64 developed top-left512 windows, eight/domain, with full-image wide context. NOT full datasets or independent validation. Original Geometry/finite VIP wide/fine observations/all20/risk support/reconstruction unchanged. Five earlier endpoint scores and per-image predictions replay bitwise; scores persist before masks. LoveDA D once, P separate; common scored-class support.

Primary log weight=beta*(fine margin-wide margin) on original eligible alias/rival slots; other slots weight1. The sampled source margin reaches the fine margin before class normalization. This is not an identity of final class or dense scores, nor a calibrated error probability. Same normalized writer, fine-target segment projection and posterior solver.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalProjected_Exact | FineBudgetOnly_Exact | MarginTransfer_Projected | MarginNeutral_Projected | MarginDeficit_Projected | RatioSoft_Projected | TransferClassMean_Projected | ProjectedHardStrengthMatched | TransferAliasShuffle0 | TransferAliasShuffle1 | TransferAliasShuffle2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.3441 | 54.1183 | 54.1333 | 54.4115 | 54.3761 | 54.4047 | 54.4872 | 53.8226 | 54.4625 | 53.4716 | 54.1493 | 53.3909 |
| potsdam/potsdam | 35.7035 | 38.9203 | 40.6173 | 41.1231 | 40.9475 | 40.2309 | 39.7057 | 39.5657 | 39.6135 | 38.9764 | 40.1928 | 39.0419 | 39.0873 | 39.1315 |
| udd5/udd5 | 30.5010 | 28.1758 | 34.0394 | 34.6174 | 34.6906 | 33.1589 | 30.8106 | 30.4739 | 29.8426 | 28.2922 | 33.3400 | 28.7316 | 29.5035 | 28.3858 |
| oem/oem | 39.8205 | 39.0232 | 39.7906 | 40.4484 | 40.1611 | 39.9486 | 39.6089 | 39.4287 | 39.3465 | 39.0596 | 39.9306 | 39.1911 | 39.1938 | 39.1672 |
| loveda/P | 49.5399 | 50.7066 | 52.5879 | 52.6436 | 52.0588 | 51.2916 | 50.7699 | 51.8603 | 52.0094 | 50.4435 | 51.1618 | 49.4982 | 50.5447 | 51.5858 |
| loveda/D | 33.8779 | 30.7360 | 37.2156 | 37.6023 | 37.2513 | 35.9488 | 34.6104 | 33.7563 | 33.6620 | 30.9138 | 36.1029 | 31.9737 | 32.2849 | 31.2619 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 52.9446 | 53.3059 | 53.3440 | 52.7414 | 52.3794 | 52.3168 | 52.3623 | 51.8455 | 52.7511 | 52.0816 | 51.9864 | 52.1333 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.7836 | 67.6095 | 67.4199 | 67.2335 | 67.2065 | 67.2959 | 66.9482 | 67.4863 | 67.0908 | 67.0974 | 66.9492 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 34.3507 | 34.4546 | 34.0701 | 33.9003 | 33.8602 | 33.9121 | 33.6204 | 34.0868 | 33.6899 | 33.7276 | 33.7955 |
| Eight-domain mean | 40.5310 | 42.8679 | 45.0622 | 45.4187 | 45.3240 | 44.7412 | 44.0781 | 43.8766 | 43.8153 | 42.9348 | 44.7941 | 43.1590 | 43.3788 | 43.0269 |

## Predeclared Gates

```json
{
  "MarginTransfer_Projected": {
    "mean_gain_vs_projected_pp": -0.6774796434878496,
    "worst_protocol_gain_pp": -1.6534865433173422,
    "mean_gain_vs_strongest_control_pp": -0.05288152990045347,
    "mean_gain_vs_transfer_alias_null_pp": 1.5530161323276843,
    "own_alias_null_matched": true,
    "accuracy_gate": false,
    "mechanism_gate": false
  },
  "MarginNeutral_Projected": {
    "mean_gain_vs_projected_pp": -1.3406177774065355,
    "worst_protocol_gain_pp": -3.806800452972446,
    "mean_gain_vs_strongest_control_pp": -0.7160196638191394,
    "mean_gain_vs_transfer_alias_null_pp": 0.8898779984089984,
    "own_alias_null_matched": false,
    "accuracy_gate": false,
    "mechanism_gate": false
  },
  "MarginDeficit_Projected": {
    "mean_gain_vs_projected_pp": -1.5421197213694313,
    "worst_protocol_gain_pp": -4.143474692685217,
    "mean_gain_vs_strongest_control_pp": -0.9175216077820352,
    "mean_gain_vs_transfer_alias_null_pp": 0.6883760544461026,
    "own_alias_null_matched": false,
    "accuracy_gate": false,
    "mechanism_gate": false
  }
}
```

Null spectra are matched to primary transfer, not neutral/deficit. Controls are not promotable model candidates. A nonprimary accuracy pass needs its own matched controls before an alias-mechanism claim. No automatic full20092 rollout.

## Label-Free Source Diagnostics

| Dataset/protocol | Eligible comparisons/window |1-risk residual-positive fraction | Transfer active weight |1-risk active weight |
| --- | ---: | ---: | ---: | ---: |
| vdd/vdd | 85748.000000 | 0.438741 | 0.369487 | 0.500642 |
| potsdam/potsdam | 40419.250000 | 0.358152 | 0.473217 | 0.489797 |
| udd5/udd5 | 33422.875000 | 0.499037 | 0.351286 | 0.539044 |
| oem/oem | 62800.000000 | 0.343666 | 0.484984 | 0.483198 |
| loveda/P | 66532.250000 | 0.530335 | 0.392549 | 0.542607 |
| loveda/D | 88674.250000 | 0.525386 | 0.401620 | 0.541493 |
| vaihingen/vaihingen | 32010.250000 | 0.319503 | 0.474771 | 0.466607 |
| landcoverai/landcoverai | 23834.000000 | 0.357037 | 0.526610 | 0.501505 |
| flair1/flair1 | 147350.250000 | 0.305713 | 0.535055 | 0.480836 |

All source-transfer margin identity errors<=1e-10; canonical risk0; shuffled weight spectra exactly preserved. Class-mean control preserves log-weight sums, not arithmetic weight mass or class-score changes. Equal-strength hard control inherits one alias-derived window scalar.

## Independent Window-Context Cost

Neutral/deficit singleton execution currently computes the unused primary transfer correction before its own correction. Reported times reflect this implementation, not a minimum attainable latency for those alternatives. No optimized speed claim is made for failed candidates.

Warmed/synchronized/alternated three executions, resident backbones included; loading/text encoding/GT excluded. Candidate timing includes small CPU score-equality validation absent for hard. Do not claim sub-percent differences. All three singleton candidate scores match all-arm scores bitwise in warmup and three timed executions. Fine forwards remain; no new visual observation.

| Dataset | Cached hard s | Transfer s | Neutral s | Deficit s |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.310728 | 0.312981 | 0.320156 | 0.319764 |
| potsdam | 0.276652 | 0.275834 | 0.287744 | 0.288548 |
| udd5 | 0.283352 | 0.284482 | 0.293253 | 0.292087 |
| oem | 0.287525 | 0.286402 | 0.300865 | 0.296976 |
| loveda | 0.323612 | 0.325839 | 0.347248 | 0.346657 |
| vaihingen | 0.269110 | 0.271298 | 0.281681 | 0.282070 |
| landcoverai | 0.267938 | 0.268867 | 0.279432 | 0.279276 |
| flair1 | 0.287932 | 0.287431 | 0.303061 | 0.301808 |

| Dataset | Cached hard MiB | Transfer MiB | Neutral MiB | Deficit MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 5566.055 | 5566.055 | 5566.055 | 5566.055 |
| potsdam | 5564.075 | 5564.075 | 5564.075 | 5564.075 |
| udd5 | 5556.987 | 5556.987 | 5556.987 | 5556.987 |
| oem | 5573.773 | 5573.773 | 5573.773 | 5573.773 |
| loveda | 5594.622 | 5594.622 | 5594.622 | 5594.622 |
| vaihingen | 5559.794 | 5559.794 | 5559.794 | 5559.794 |
| landcoverai | 5559.794 | 5559.794 | 5559.794 | 5559.794 |
| flair1 | 5797.613 | 5797.613 | 5797.613 | 5797.613 |

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
| vdd/vdd | MarginTransfer_Projected | other | 57.4791 | 2.4191 | 80.8384 | 66.5457 | 42.0287 |
| vdd/vdd | MarginTransfer_Projected | wall | 56.3580 | 3.2131 | 60.9078 | 88.2967 | 11.5325 |
| vdd/vdd | MarginTransfer_Projected | road | 22.4352 | 0.7734 | 22.5058 | 98.6222 | 14.5442 |
| vdd/vdd | MarginTransfer_Projected | vegetation | 50.4369 | -3.8308 | 95.2366 | 51.7422 | 12.5150 |
| vdd/vdd | MarginTransfer_Projected | vehicle | 42.3695 | 4.1098 | 42.3695 | 100.0000 | 0.8452 |
| vdd/vdd | MarginTransfer_Projected | roof | 90.0775 | 0.1960 | 90.5455 | 99.4294 | 6.6166 |
| vdd/vdd | MarginTransfer_Projected | water | 61.7241 | -4.8284 | 64.5916 | 93.2903 | 11.9178 |
| vdd/vdd | MarginNeutral_Projected | other | 58.8573 | 3.7973 | 81.0112 | 68.2768 | 43.0300 |
| vdd/vdd | MarginNeutral_Projected | wall | 57.9323 | 4.7874 | 63.0856 | 87.6421 | 11.0518 |
| vdd/vdd | MarginNeutral_Projected | road | 22.7835 | 1.1217 | 22.8782 | 98.2156 | 14.2485 |
| vdd/vdd | MarginNeutral_Projected | vegetation | 47.7673 | -6.5004 | 95.3932 | 48.8952 | 11.8070 |
| vdd/vdd | MarginNeutral_Projected | vehicle | 45.8066 | 7.5469 | 45.8066 | 100.0000 | 0.7818 |
| vdd/vdd | MarginNeutral_Projected | roof | 88.9413 | -0.9402 | 89.4174 | 99.4049 | 6.6984 |
| vdd/vdd | MarginNeutral_Projected | water | 58.5441 | -8.0084 | 61.5333 | 92.3380 | 12.3825 |
| vdd/vdd | MarginDeficit_Projected | other | 59.0279 | 3.9679 | 80.9423 | 68.5558 | 43.2427 |
| vdd/vdd | MarginDeficit_Projected | wall | 58.9104 | 5.7655 | 64.3062 | 87.5324 | 10.8285 |
| vdd/vdd | MarginDeficit_Projected | road | 23.2779 | 1.6161 | 23.3763 | 98.2228 | 13.9459 |
| vdd/vdd | MarginDeficit_Projected | vegetation | 47.4573 | -6.8104 | 95.3421 | 48.5837 | 11.7380 |
| vdd/vdd | MarginDeficit_Projected | vehicle | 46.5794 | 8.3197 | 46.5794 | 100.0000 | 0.7688 |
| vdd/vdd | MarginDeficit_Projected | roof | 89.1102 | -0.7713 | 89.5791 | 99.4160 | 6.6871 |
| vdd/vdd | MarginDeficit_Projected | water | 56.4699 | -10.0826 | 59.3755 | 92.0253 | 12.7890 |
| vdd/vdd | RatioSoft_Projected | other | 59.4050 | 4.3450 | 81.0873 | 68.9597 | 43.4196 |
| vdd/vdd | RatioSoft_Projected | wall | 59.4498 | 6.3049 | 65.0280 | 87.3904 | 10.6909 |
| vdd/vdd | RatioSoft_Projected | road | 23.3332 | 1.6714 | 23.4400 | 98.0849 | 13.8885 |
| vdd/vdd | RatioSoft_Projected | vegetation | 47.1191 | -7.1486 | 95.3560 | 48.2258 | 11.6498 |
| vdd/vdd | RatioSoft_Projected | vehicle | 47.4326 | 9.1729 | 47.4326 | 100.0000 | 0.7550 |
| vdd/vdd | RatioSoft_Projected | roof | 88.6761 | -1.2054 | 89.1360 | 99.4215 | 6.7207 |
| vdd/vdd | RatioSoft_Projected | water | 55.9943 | -10.5582 | 58.8994 | 91.9046 | 12.8755 |
| vdd/vdd | TransferClassMean_Projected | other | 59.7982 | 4.7382 | 81.1115 | 69.4724 | 43.7294 |
| vdd/vdd | TransferClassMean_Projected | wall | 59.7702 | 6.6253 | 65.9053 | 86.5242 | 10.4441 |
| vdd/vdd | TransferClassMean_Projected | road | 23.2027 | 1.5409 | 23.3433 | 97.4700 | 13.8586 |
| vdd/vdd | TransferClassMean_Projected | vegetation | 44.4759 | -9.7918 | 94.9424 | 45.5552 | 11.0527 |
| vdd/vdd | TransferClassMean_Projected | vehicle | 49.5905 | 11.3308 | 49.5970 | 99.9734 | 0.7218 |
| vdd/vdd | TransferClassMean_Projected | roof | 86.5027 | -3.3788 | 86.9621 | 99.3930 | 6.8867 |
| vdd/vdd | TransferClassMean_Projected | water | 53.4183 | -13.1342 | 56.4100 | 90.9684 | 13.3067 |
| vdd/vdd | ProjectedHardStrengthMatched | other | 57.5122 | 2.4522 | 80.8832 | 66.5596 | 42.0142 |
| vdd/vdd | ProjectedHardStrengthMatched | wall | 56.5758 | 3.4309 | 61.1465 | 88.3296 | 11.4918 |
| vdd/vdd | ProjectedHardStrengthMatched | road | 22.5753 | 0.9135 | 22.6465 | 98.6280 | 14.4547 |
| vdd/vdd | ProjectedHardStrengthMatched | vegetation | 50.6264 | -3.6413 | 95.1814 | 51.9581 | 12.5745 |
| vdd/vdd | ProjectedHardStrengthMatched | vehicle | 41.3023 | 3.0426 | 41.3023 | 100.0000 | 0.8670 |
| vdd/vdd | ProjectedHardStrengthMatched | roof | 89.9689 | 0.0874 | 90.4266 | 99.4405 | 6.6260 |
| vdd/vdd | ProjectedHardStrengthMatched | water | 62.6765 | -3.8760 | 65.0841 | 94.4270 | 11.9718 |
| vdd/vdd | TransferAliasShuffle0 | other | 59.3833 | 4.3233 | 81.3792 | 68.7210 | 43.1141 |
| vdd/vdd | TransferAliasShuffle0 | wall | 58.4019 | 5.2570 | 64.0715 | 86.8419 | 10.7824 |
| vdd/vdd | TransferAliasShuffle0 | road | 23.3991 | 1.7373 | 23.5173 | 97.8981 | 13.8165 |
| vdd/vdd | TransferAliasShuffle0 | vegetation | 45.2139 | -9.0538 | 94.6628 | 46.3967 | 11.2901 |
| vdd/vdd | TransferAliasShuffle0 | vehicle | 46.6837 | 8.4240 | 46.6837 | 100.0000 | 0.7671 |
| vdd/vdd | TransferAliasShuffle0 | roof | 87.7945 | -2.0870 | 88.2889 | 99.3661 | 6.7814 |
| vdd/vdd | TransferAliasShuffle0 | water | 53.4246 | -13.1279 | 56.1868 | 91.5734 | 13.4484 |
| vdd/vdd | TransferAliasShuffle1 | other | 59.3207 | 4.2607 | 80.9989 | 68.9101 | 43.4358 |
| vdd/vdd | TransferAliasShuffle1 | wall | 59.2327 | 6.0878 | 64.8840 | 87.1806 | 10.6889 |
| vdd/vdd | TransferAliasShuffle1 | road | 23.1302 | 1.4684 | 23.2498 | 97.8249 | 13.9650 |
| vdd/vdd | TransferAliasShuffle1 | vegetation | 45.7690 | -8.4987 | 95.2357 | 46.8415 | 11.3297 |
| vdd/vdd | TransferAliasShuffle1 | vehicle | 49.6595 | 11.3998 | 49.6595 | 100.0000 | 0.7211 |
| vdd/vdd | TransferAliasShuffle1 | roof | 87.5049 | -2.3766 | 87.9800 | 99.3867 | 6.8066 |
| vdd/vdd | TransferAliasShuffle1 | water | 54.4284 | -12.1241 | 57.5258 | 90.9979 | 13.0528 |
| vdd/vdd | TransferAliasShuffle2 | other | 59.5962 | 4.5362 | 81.4422 | 68.9610 | 43.2312 |
| vdd/vdd | TransferAliasShuffle2 | wall | 59.3592 | 6.2143 | 65.4241 | 86.4925 | 10.5170 |
| vdd/vdd | TransferAliasShuffle2 | road | 23.2297 | 1.5679 | 23.3610 | 97.6367 | 13.8718 |
| vdd/vdd | TransferAliasShuffle2 | vegetation | 45.6915 | -8.5762 | 94.8262 | 46.8597 | 11.3831 |
| vdd/vdd | TransferAliasShuffle2 | vehicle | 45.8626 | 7.6029 | 45.8626 | 100.0000 | 0.7808 |
| vdd/vdd | TransferAliasShuffle2 | roof | 86.4928 | -3.3887 | 86.9466 | 99.4001 | 6.8884 |
| vdd/vdd | TransferAliasShuffle2 | water | 53.5044 | -13.0481 | 56.4354 | 91.1522 | 13.3276 |
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
| potsdam/potsdam | MarginTransfer_Projected | impervious surface | 66.1508 | -0.2157 | 85.3980 | 74.5873 | 35.7018 |
| potsdam/potsdam | MarginTransfer_Projected | building | 72.2420 | -0.6243 | 76.2637 | 93.1968 | 19.7027 |
| potsdam/potsdam | MarginTransfer_Projected | low vegetation | 18.6933 | -3.0391 | 81.7708 | 19.5062 | 4.1366 |
| potsdam/potsdam | MarginTransfer_Projected | tree | 57.0965 | -0.9596 | 92.4129 | 59.9047 | 11.4371 |
| potsdam/potsdam | MarginTransfer_Projected | car | 24.8931 | -0.2862 | 24.9340 | 99.3457 | 14.2864 |
| potsdam/potsdam | MarginTransfer_Projected | clutter | 2.3094 | -0.2287 | 2.9360 | 9.7646 | 14.7355 |
| potsdam/potsdam | MarginNeutral_Projected | impervious surface | 65.9567 | -0.4098 | 85.3885 | 74.3478 | 35.5911 |
| potsdam/potsdam | MarginNeutral_Projected | building | 71.9322 | -0.9341 | 75.9894 | 93.0903 | 19.7512 |
| potsdam/potsdam | MarginNeutral_Projected | low vegetation | 16.9289 | -4.8035 | 81.4922 | 17.6058 | 3.7464 |
| potsdam/potsdam | MarginNeutral_Projected | tree | 56.4574 | -1.5987 | 92.1410 | 59.3136 | 11.3576 |
| potsdam/potsdam | MarginNeutral_Projected | car | 24.7322 | -0.4471 | 24.7780 | 99.2579 | 14.3636 |
| potsdam/potsdam | MarginNeutral_Projected | clutter | 2.2270 | -0.3111 | 2.8139 | 9.6473 | 15.1900 |
| potsdam/potsdam | MarginDeficit_Projected | impervious surface | 66.1848 | -0.1817 | 85.6773 | 74.4186 | 35.5050 |
| potsdam/potsdam | MarginDeficit_Projected | building | 71.5549 | -1.3114 | 75.4806 | 93.2240 | 19.9129 |
| potsdam/potsdam | MarginDeficit_Projected | low vegetation | 16.4668 | -5.2656 | 80.6556 | 17.1438 | 3.6859 |
| potsdam/potsdam | MarginDeficit_Projected | tree | 56.2125 | -1.8436 | 92.0235 | 59.0917 | 11.3296 |
| potsdam/potsdam | MarginDeficit_Projected | car | 24.7822 | -0.3971 | 24.8294 | 99.2393 | 14.3312 |
| potsdam/potsdam | MarginDeficit_Projected | clutter | 2.1931 | -0.3450 | 2.7702 | 9.5257 | 15.2354 |
| potsdam/potsdam | RatioSoft_Projected | impervious surface | 66.3408 | -0.0257 | 85.6932 | 74.6039 | 35.5867 |
| potsdam/potsdam | RatioSoft_Projected | building | 71.5406 | -1.3257 | 75.4672 | 93.2202 | 19.9156 |
| potsdam/potsdam | RatioSoft_Projected | low vegetation | 16.4040 | -5.3284 | 80.8258 | 17.0682 | 3.6619 |
| potsdam/potsdam | RatioSoft_Projected | tree | 56.2867 | -1.7694 | 92.0564 | 59.1601 | 11.3387 |
| potsdam/potsdam | RatioSoft_Projected | car | 24.9126 | -0.2667 | 24.9614 | 99.2207 | 14.2527 |
| potsdam/potsdam | RatioSoft_Projected | clutter | 2.1962 | -0.3419 | 2.7736 | 9.5429 | 15.2443 |
| potsdam/potsdam | TransferClassMean_Projected | impervious surface | 65.8810 | -0.4855 | 85.6818 | 74.0313 | 35.3183 |
| potsdam/potsdam | TransferClassMean_Projected | building | 71.0990 | -1.7673 | 75.0062 | 93.1734 | 20.0280 |
| potsdam/potsdam | TransferClassMean_Projected | low vegetation | 14.6016 | -7.1308 | 80.0192 | 15.1541 | 3.2840 |
| potsdam/potsdam | TransferClassMean_Projected | tree | 55.6313 | -2.4248 | 91.8756 | 58.5096 | 11.2360 |
| potsdam/potsdam | TransferClassMean_Projected | car | 24.5221 | -0.6572 | 24.5766 | 99.1037 | 14.4588 |
| potsdam/potsdam | TransferClassMean_Projected | clutter | 2.1237 | -0.4144 | 2.6673 | 9.4364 | 15.6748 |
| potsdam/potsdam | ProjectedHardStrengthMatched | impervious surface | 66.2013 | -0.1652 | 85.3928 | 74.6556 | 35.7367 |
| potsdam/potsdam | ProjectedHardStrengthMatched | building | 72.2119 | -0.6544 | 76.2259 | 93.2033 | 19.7138 |
| potsdam/potsdam | ProjectedHardStrengthMatched | low vegetation | 18.4169 | -3.3155 | 81.9311 | 19.1966 | 4.0630 |
| potsdam/potsdam | ProjectedHardStrengthMatched | tree | 57.0650 | -0.9911 | 92.5416 | 59.8160 | 11.4043 |
| potsdam/potsdam | ProjectedHardStrengthMatched | car | 24.9564 | -0.2229 | 24.9995 | 99.3138 | 14.2444 |
| potsdam/potsdam | ProjectedHardStrengthMatched | clutter | 2.3052 | -0.2329 | 2.9260 | 9.7991 | 14.8379 |
| potsdam/potsdam | TransferAliasShuffle0 | impervious surface | 65.6511 | -0.7154 | 85.5466 | 73.8417 | 35.2835 |
| potsdam/potsdam | TransferAliasShuffle0 | building | 71.5232 | -1.3431 | 75.5621 | 93.0463 | 19.8535 |
| potsdam/potsdam | TransferAliasShuffle0 | low vegetation | 14.6993 | -7.0331 | 80.1100 | 15.2561 | 3.3024 |
| potsdam/potsdam | TransferAliasShuffle0 | tree | 55.8459 | -2.2102 | 91.8541 | 58.7558 | 11.2860 |
| potsdam/potsdam | TransferAliasShuffle0 | car | 24.3928 | -0.7865 | 24.4404 | 99.2087 | 14.5548 |
| potsdam/potsdam | TransferAliasShuffle0 | clutter | 2.1388 | -0.3993 | 2.6842 | 9.5236 | 15.7198 |
| potsdam/potsdam | TransferAliasShuffle1 | impervious surface | 65.8393 | -0.5272 | 85.6043 | 74.0366 | 35.3528 |
| potsdam/potsdam | TransferAliasShuffle1 | building | 71.2021 | -1.6642 | 75.1579 | 93.1167 | 19.9754 |
| potsdam/potsdam | TransferAliasShuffle1 | low vegetation | 14.9605 | -6.7719 | 80.0927 | 15.5382 | 3.3642 |
| potsdam/potsdam | TransferAliasShuffle1 | tree | 55.8388 | -2.2173 | 91.8727 | 58.7404 | 11.2807 |
| potsdam/potsdam | TransferAliasShuffle1 | car | 24.5402 | -0.6391 | 24.5933 | 99.1276 | 14.4525 |
| potsdam/potsdam | TransferAliasShuffle1 | clutter | 2.1430 | -0.3951 | 2.6949 | 9.4730 | 15.5744 |
| potsdam/potsdam | TransferAliasShuffle2 | impervious surface | 66.1799 | -0.1866 | 85.8432 | 74.2877 | 35.3740 |
| potsdam/potsdam | TransferAliasShuffle2 | building | 71.1792 | -1.6871 | 75.1224 | 93.1320 | 19.9881 |
| potsdam/potsdam | TransferAliasShuffle2 | low vegetation | 14.7009 | -7.0315 | 79.6062 | 15.2762 | 3.3277 |
| potsdam/potsdam | TransferAliasShuffle2 | tree | 55.8570 | -2.1991 | 91.9221 | 58.7404 | 11.2747 |
| potsdam/potsdam | TransferAliasShuffle2 | car | 24.7265 | -0.4528 | 24.7730 | 99.2473 | 14.3650 |
| potsdam/potsdam | TransferAliasShuffle2 | clutter | 2.1456 | -0.3925 | 2.6945 | 9.5300 | 15.6705 |
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
| udd5/udd5 | MarginTransfer_Projected | vegetation | 64.4186 | -4.1427 | 91.4859 | 68.5268 | 2.3125 |
| udd5/udd5 | MarginTransfer_Projected | building | 85.1728 | -0.7072 | 85.2526 | 99.8902 | 84.2748 |
| udd5/udd5 | MarginTransfer_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9466 |
| udd5/udd5 | MarginTransfer_Projected | vehicle | 6.5211 | -0.1101 | 6.5212 | 99.9773 | 6.4391 |
| udd5/udd5 | MarginTransfer_Projected | other | 9.6818 | -2.3328 | 38.5845 | 11.4457 | 6.0271 |
| udd5/udd5 | MarginNeutral_Projected | vegetation | 55.2583 | -13.3030 | 91.8696 | 58.0996 | 1.9524 |
| udd5/udd5 | MarginNeutral_Projected | building | 84.4944 | -1.3856 | 84.5842 | 99.8745 | 84.9273 |
| udd5/udd5 | MarginNeutral_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6782 |
| udd5/udd5 | MarginNeutral_Projected | vehicle | 6.3364 | -0.2948 | 6.3365 | 99.9773 | 6.6267 |
| udd5/udd5 | MarginNeutral_Projected | other | 7.9639 | -4.0507 | 33.1486 | 9.4877 | 5.8154 |
| udd5/udd5 | MarginDeficit_Projected | vegetation | 54.5278 | -14.0335 | 91.8517 | 57.2995 | 1.9259 |
| udd5/udd5 | MarginDeficit_Projected | building | 84.4000 | -1.4800 | 84.5250 | 99.8250 | 84.9447 |
| udd5/udd5 | MarginDeficit_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6346 |
| udd5/udd5 | MarginDeficit_Projected | vehicle | 6.1407 | -0.4905 | 6.1408 | 99.9773 | 6.8379 |
| udd5/udd5 | MarginDeficit_Projected | other | 7.3013 | -4.7133 | 31.2443 | 8.6989 | 5.6569 |
| udd5/udd5 | RatioSoft_Projected | vegetation | 52.1970 | -16.3643 | 92.3707 | 54.5487 | 1.8231 |
| udd5/udd5 | RatioSoft_Projected | building | 84.2683 | -1.6117 | 84.4030 | 99.8110 | 85.0556 |
| udd5/udd5 | RatioSoft_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5595 |
| udd5/udd5 | RatioSoft_Projected | vehicle | 5.8257 | -0.8055 | 5.8258 | 99.9773 | 7.2076 |
| udd5/udd5 | RatioSoft_Projected | other | 6.9222 | -5.0924 | 31.0418 | 8.1800 | 5.3541 |
| udd5/udd5 | TransferClassMean_Projected | vegetation | 46.6450 | -21.9163 | 93.7799 | 48.1342 | 1.5846 |
| udd5/udd5 | TransferClassMean_Projected | building | 83.5146 | -2.3654 | 83.6868 | 99.7543 | 85.7348 |
| udd5/udd5 | TransferClassMean_Projected | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3086 |
| udd5/udd5 | TransferClassMean_Projected | vehicle | 5.4231 | -1.2081 | 5.4231 | 99.9773 | 7.7428 |
| udd5/udd5 | TransferClassMean_Projected | other | 5.8782 | -6.1364 | 29.9190 | 6.8167 | 4.6292 |
| udd5/udd5 | ProjectedHardStrengthMatched | vegetation | 64.8899 | -3.6714 | 91.3843 | 69.1184 | 2.3350 |
| udd5/udd5 | ProjectedHardStrengthMatched | building | 85.3077 | -0.5723 | 85.4113 | 99.8581 | 84.0911 |
| udd5/udd5 | ProjectedHardStrengthMatched | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9357 |
| udd5/udd5 | ProjectedHardStrengthMatched | vehicle | 6.3834 | -0.2478 | 6.3835 | 99.9773 | 6.5779 |
| udd5/udd5 | ProjectedHardStrengthMatched | other | 10.1191 | -1.8955 | 39.9981 | 11.9301 | 6.0602 |
| udd5/udd5 | TransferAliasShuffle0 | vegetation | 49.8134 | -18.7479 | 92.3757 | 51.9492 | 1.7362 |
| udd5/udd5 | TransferAliasShuffle0 | building | 83.3856 | -2.4944 | 83.5584 | 99.7526 | 85.8650 |
| udd5/udd5 | TransferAliasShuffle0 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2782 |
| udd5/udd5 | TransferAliasShuffle0 | vehicle | 5.1419 | -1.4893 | 5.1420 | 99.9773 | 8.1662 |
| udd5/udd5 | TransferAliasShuffle0 | other | 5.3173 | -6.6973 | 30.9896 | 6.0315 | 3.9545 |
| udd5/udd5 | TransferAliasShuffle1 | vegetation | 49.9511 | -18.6102 | 92.6095 | 52.0249 | 1.7343 |
| udd5/udd5 | TransferAliasShuffle1 | building | 83.3391 | -2.5409 | 83.5039 | 99.7637 | 85.9306 |
| udd5/udd5 | TransferAliasShuffle1 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3899 |
| udd5/udd5 | TransferAliasShuffle1 | vehicle | 6.8968 | 0.2656 | 6.8969 | 99.9773 | 6.0883 |
| udd5/udd5 | TransferAliasShuffle1 | other | 7.3303 | -4.6843 | 30.5221 | 8.7984 | 5.8569 |
| udd5/udd5 | TransferAliasShuffle2 | vegetation | 47.4225 | -21.1388 | 93.5672 | 49.0208 | 1.6174 |
| udd5/udd5 | TransferAliasShuffle2 | building | 83.4953 | -2.3847 | 83.6653 | 99.7572 | 85.7593 |
| udd5/udd5 | TransferAliasShuffle2 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2694 |
| udd5/udd5 | TransferAliasShuffle2 | vehicle | 5.2543 | -1.3769 | 5.2544 | 99.9773 | 7.9915 |
| udd5/udd5 | TransferAliasShuffle2 | other | 5.7567 | -6.2579 | 30.7963 | 6.6121 | 4.3623 |
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
| oem/oem | MarginTransfer_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4624 |
| oem/oem | MarginTransfer_Projected | rangeland | 49.7672 | -0.0920 | 70.6538 | 62.7350 | 12.9159 |
| oem/oem | MarginTransfer_Projected | developed space | 23.7461 | 0.2591 | 40.7805 | 36.2441 | 17.4176 |
| oem/oem | MarginTransfer_Projected | road | 55.0530 | -0.3612 | 68.3533 | 73.8856 | 6.7961 |
| oem/oem | MarginTransfer_Projected | tree | 57.1809 | -0.5515 | 91.1494 | 60.5423 | 17.5475 |
| oem/oem | MarginTransfer_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 |
| oem/oem | MarginTransfer_Projected | agriculture land | 82.2303 | -0.3154 | 82.6246 | 99.4230 | 25.5266 |
| oem/oem | MarginTransfer_Projected | building | 51.6114 | -2.9377 | 69.8215 | 66.4305 | 11.3336 |
| oem/oem | MarginNeutral_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3682 |
| oem/oem | MarginNeutral_Projected | rangeland | 49.7449 | -0.1143 | 70.6193 | 62.7270 | 12.9205 |
| oem/oem | MarginNeutral_Projected | developed space | 23.8113 | 0.3243 | 40.0258 | 37.0192 | 18.1255 |
| oem/oem | MarginNeutral_Projected | road | 54.8799 | -0.5343 | 68.3097 | 73.6247 | 6.7764 |
| oem/oem | MarginNeutral_Projected | tree | 56.6103 | -1.1221 | 91.4192 | 59.7871 | 17.2774 |
| oem/oem | MarginNeutral_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | MarginNeutral_Projected | agriculture land | 82.0960 | -0.4497 | 82.4749 | 99.4435 | 25.5782 |
| oem/oem | MarginNeutral_Projected | building | 49.7288 | -4.8203 | 69.3311 | 63.7530 | 10.9537 |
| oem/oem | MarginDeficit_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3128 |
| oem/oem | MarginDeficit_Projected | rangeland | 49.6724 | -0.1868 | 70.6126 | 62.6169 | 12.8991 |
| oem/oem | MarginDeficit_Projected | developed space | 23.7092 | 0.2222 | 39.7112 | 37.0426 | 18.2807 |
| oem/oem | MarginDeficit_Projected | road | 54.7708 | -0.6434 | 68.1708 | 73.5897 | 6.7870 |
| oem/oem | MarginDeficit_Projected | tree | 56.4495 | -1.2829 | 91.4382 | 59.5997 | 17.2197 |
| oem/oem | MarginDeficit_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | MarginDeficit_Projected | agriculture land | 81.8982 | -0.6475 | 82.2786 | 99.4387 | 25.6380 |
| oem/oem | MarginDeficit_Projected | building | 48.9293 | -5.6198 | 68.8821 | 62.8138 | 10.8627 |
| oem/oem | RatioSoft_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3546 |
| oem/oem | RatioSoft_Projected | rangeland | 49.6960 | -0.1632 | 70.5999 | 62.6643 | 12.9112 |
| oem/oem | RatioSoft_Projected | developed space | 23.4945 | 0.0075 | 39.3924 | 36.7950 | 18.3055 |
| oem/oem | RatioSoft_Projected | road | 54.7647 | -0.6495 | 68.1794 | 73.5686 | 6.7842 |
| oem/oem | RatioSoft_Projected | tree | 56.2655 | -1.4669 | 91.4878 | 59.3738 | 17.1451 |
| oem/oem | RatioSoft_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | RatioSoft_Projected | agriculture land | 81.8862 | -0.6595 | 82.2643 | 99.4419 | 25.6433 |
| oem/oem | RatioSoft_Projected | building | 48.6654 | -5.8837 | 68.6537 | 62.5680 | 10.8562 |
| oem/oem | TransferClassMean_Projected | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1994 |
| oem/oem | TransferClassMean_Projected | rangeland | 49.4919 | -0.3673 | 70.5247 | 62.3991 | 12.8702 |
| oem/oem | TransferClassMean_Projected | developed space | 23.8270 | 0.3400 | 39.0874 | 37.8996 | 19.0021 |
| oem/oem | TransferClassMean_Projected | road | 54.5568 | -0.8574 | 68.0562 | 73.3366 | 6.7750 |
| oem/oem | TransferClassMean_Projected | tree | 55.7629 | -1.9695 | 91.7317 | 58.7140 | 16.9095 |
| oem/oem | TransferClassMean_Projected | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | TransferClassMean_Projected | agriculture land | 81.6368 | -0.9089 | 81.9943 | 99.4687 | 25.7346 |
| oem/oem | TransferClassMean_Projected | building | 47.2016 | -7.3475 | 68.4131 | 60.3550 | 10.5090 |
| oem/oem | ProjectedHardStrengthMatched | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.5032 |
| oem/oem | ProjectedHardStrengthMatched | rangeland | 49.8003 | -0.0589 | 70.6383 | 62.8000 | 12.9321 |
| oem/oem | ProjectedHardStrengthMatched | developed space | 23.5901 | 0.1031 | 40.6573 | 35.9778 | 17.3420 |
| oem/oem | ProjectedHardStrengthMatched | road | 55.0506 | -0.3636 | 68.3442 | 73.8918 | 6.7976 |
| oem/oem | ProjectedHardStrengthMatched | tree | 57.0750 | -0.6574 | 91.1624 | 60.4180 | 17.5089 |
| oem/oem | ProjectedHardStrengthMatched | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| oem/oem | ProjectedHardStrengthMatched | agriculture land | 82.2460 | -0.2997 | 82.6402 | 99.4234 | 25.5219 |
| oem/oem | ProjectedHardStrengthMatched | building | 51.6825 | -2.8666 | 69.6948 | 66.6639 | 11.3941 |
| oem/oem | TransferAliasShuffle0 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2050 |
| oem/oem | TransferAliasShuffle0 | rangeland | 49.5714 | -0.2878 | 70.5278 | 62.5230 | 12.8952 |
| oem/oem | TransferAliasShuffle0 | developed space | 23.8341 | 0.3471 | 39.2301 | 37.7842 | 18.8753 |
| oem/oem | TransferAliasShuffle0 | road | 54.6551 | -0.7591 | 68.2064 | 73.3397 | 6.7604 |
| oem/oem | TransferAliasShuffle0 | tree | 55.8019 | -1.9305 | 91.6841 | 58.7768 | 16.9364 |
| oem/oem | TransferAliasShuffle0 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | TransferAliasShuffle0 | agriculture land | 81.7186 | -0.8271 | 82.0786 | 99.4661 | 25.7076 |
| oem/oem | TransferAliasShuffle0 | building | 47.9476 | -6.6015 | 68.7596 | 61.3020 | 10.6201 |
| oem/oem | TransferAliasShuffle1 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2015 |
| oem/oem | TransferAliasShuffle1 | rangeland | 49.5715 | -0.2877 | 70.5757 | 62.4856 | 12.8788 |
| oem/oem | TransferAliasShuffle1 | developed space | 23.9002 | 0.4132 | 39.3169 | 37.8699 | 18.8764 |
| oem/oem | TransferAliasShuffle1 | road | 54.6477 | -0.7665 | 68.0909 | 73.4604 | 6.7830 |
| oem/oem | TransferAliasShuffle1 | tree | 56.0293 | -1.7031 | 91.5954 | 59.0659 | 17.0362 |
| oem/oem | TransferAliasShuffle1 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | TransferAliasShuffle1 | agriculture land | 81.8199 | -0.7258 | 82.1841 | 99.4613 | 25.6733 |
| oem/oem | TransferAliasShuffle1 | building | 47.5817 | -6.9674 | 68.6414 | 60.7977 | 10.5509 |
| oem/oem | TransferAliasShuffle2 | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3378 |
| oem/oem | TransferAliasShuffle2 | rangeland | 49.5041 | -0.3551 | 70.5283 | 62.4156 | 12.8730 |
| oem/oem | TransferAliasShuffle2 | developed space | 23.8076 | 0.3206 | 39.4989 | 37.4726 | 18.5923 |
| oem/oem | TransferAliasShuffle2 | road | 54.5550 | -0.8592 | 67.9537 | 73.4526 | 6.7960 |
| oem/oem | TransferAliasShuffle2 | tree | 56.1961 | -1.5363 | 91.5481 | 59.2711 | 17.1042 |
| oem/oem | TransferAliasShuffle2 | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | TransferAliasShuffle2 | agriculture land | 81.5914 | -0.9543 | 81.9487 | 99.4684 | 25.7489 |
| oem/oem | TransferAliasShuffle2 | building | 47.6831 | -6.8660 | 68.7511 | 60.8770 | 10.5478 |
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
| loveda/P | MarginTransfer_Projected | building | 51.0830 | 2.1538 | 53.3962 | 92.1824 | 0.0457 |
| loveda/P | MarginTransfer_Projected | road | 57.5103 | -2.6508 | 58.6494 | 96.7331 | 13.7262 |
| loveda/P | MarginTransfer_Projected | water | 62.6622 | -0.2620 | 67.9342 | 88.9803 | 19.2732 |
| loveda/P | MarginTransfer_Projected | barren | 2.2727 | 0.6003 | 84.3013 | 2.2824 | 0.3002 |
| loveda/P | MarginTransfer_Projected | tree | 49.9922 | -6.5989 | 97.9639 | 50.5171 | 8.4904 |
| loveda/P | MarginTransfer_Projected | farm | 84.2289 | -1.3549 | 84.5381 | 99.5676 | 58.1642 |
| loveda/P | MarginNeutral_Projected | building | 53.4221 | 4.4929 | 56.2000 | 91.5309 | 0.0432 |
| loveda/P | MarginNeutral_Projected | road | 56.0628 | -4.0983 | 57.1338 | 96.7643 | 14.0949 |
| loveda/P | MarginNeutral_Projected | water | 62.5480 | -0.3762 | 67.9498 | 88.7234 | 19.2131 |
| loveda/P | MarginNeutral_Projected | barren | 2.6737 | 1.0013 | 87.3796 | 2.6841 | 0.3406 |
| loveda/P | MarginNeutral_Projected | tree | 46.2267 | -10.3644 | 98.5928 | 46.5337 | 7.7710 |
| loveda/P | MarginNeutral_Projected | farm | 83.6860 | -1.8978 | 83.9950 | 99.5624 | 58.5373 |
| loveda/P | MarginDeficit_Projected | building | 61.6667 | 12.7375 | 69.6237 | 84.3648 | 0.0321 |
| loveda/P | MarginDeficit_Projected | road | 55.7560 | -4.4051 | 56.8178 | 96.7570 | 14.1722 |
| loveda/P | MarginDeficit_Projected | water | 62.5767 | -0.3475 | 68.0611 | 88.5920 | 19.1533 |
| loveda/P | MarginDeficit_Projected | barren | 3.3635 | 1.6911 | 89.9979 | 3.3761 | 0.4159 |
| loveda/P | MarginDeficit_Projected | tree | 44.5113 | -12.0798 | 99.1428 | 44.6832 | 7.4206 |
| loveda/P | MarginDeficit_Projected | farm | 83.2875 | -2.2963 | 83.6017 | 99.5508 | 58.8059 |
| loveda/P | RatioSoft_Projected | building | 62.2871 | 13.3579 | 71.1111 | 83.3876 | 0.0311 |
| loveda/P | RatioSoft_Projected | road | 55.9553 | -4.2058 | 57.0262 | 96.7528 | 14.1198 |
| loveda/P | RatioSoft_Projected | water | 62.5576 | -0.3666 | 68.0423 | 88.5856 | 19.1572 |
| loveda/P | RatioSoft_Projected | barren | 3.5962 | 1.9238 | 90.6195 | 3.6097 | 0.4416 |
| loveda/P | RatioSoft_Projected | tree | 44.4283 | -12.1628 | 99.1389 | 44.6004 | 7.4072 |
| loveda/P | RatioSoft_Projected | farm | 83.2320 | -2.3518 | 83.5471 | 99.5489 | 58.8431 |
| loveda/P | TransferClassMean_Projected | building | 59.0799 | 10.1507 | 69.7143 | 79.4788 | 0.0302 |
| loveda/P | TransferClassMean_Projected | road | 53.7972 | -6.3639 | 54.7714 | 96.7995 | 14.7082 |
| loveda/P | TransferClassMean_Projected | water | 62.4280 | -0.4962 | 68.1385 | 88.1644 | 19.0392 |
| loveda/P | TransferClassMean_Projected | barren | 4.2015 | 2.5291 | 89.5197 | 4.2223 | 0.5229 |
| loveda/P | TransferClassMean_Projected | tree | 40.1810 | -16.4101 | 99.3382 | 40.2888 | 6.6777 |
| loveda/P | TransferClassMean_Projected | farm | 82.9735 | -2.6103 | 83.2901 | 99.5440 | 59.0218 |
| loveda/P | ProjectedHardStrengthMatched | building | 50.4472 | 1.5180 | 52.8090 | 91.8567 | 0.0461 |
| loveda/P | ProjectedHardStrengthMatched | road | 57.5558 | -2.6053 | 58.6971 | 96.7321 | 13.7149 |
| loveda/P | ProjectedHardStrengthMatched | water | 62.6252 | -0.2990 | 67.8684 | 89.0185 | 19.3001 |
| loveda/P | ProjectedHardStrengthMatched | barren | 2.2477 | 0.5753 | 81.7005 | 2.2591 | 0.3066 |
| loveda/P | ProjectedHardStrengthMatched | tree | 49.8375 | -6.7536 | 97.8087 | 50.4002 | 8.4842 |
| loveda/P | ProjectedHardStrengthMatched | farm | 84.2577 | -1.3261 | 84.5646 | 99.5711 | 58.1481 |
| loveda/P | TransferAliasShuffle0 | building | 50.3817 | 1.4525 | 54.8857 | 85.9935 | 0.0415 |
| loveda/P | TransferAliasShuffle0 | road | 54.7341 | -5.4270 | 55.7466 | 96.7881 | 14.4492 |
| loveda/P | TransferAliasShuffle0 | water | 62.7724 | -0.1518 | 68.3247 | 88.5381 | 19.0678 |
| loveda/P | TransferAliasShuffle0 | barren | 3.2564 | 1.5840 | 90.2020 | 3.2679 | 0.4017 |
| loveda/P | TransferAliasShuffle0 | tree | 42.6935 | -13.8976 | 98.9192 | 42.8936 | 7.1395 |
| loveda/P | TransferAliasShuffle0 | farm | 83.1511 | -2.4327 | 83.4658 | 99.5487 | 58.9004 |
| loveda/P | TransferAliasShuffle1 | building | 59.0909 | 10.1617 | 66.1578 | 84.6906 | 0.0339 |
| loveda/P | TransferAliasShuffle1 | road | 54.2165 | -5.9446 | 55.2121 | 96.7809 | 14.5879 |
| loveda/P | TransferAliasShuffle1 | water | 62.4618 | -0.4624 | 68.0057 | 88.4554 | 19.1393 |
| loveda/P | TransferAliasShuffle1 | barren | 3.5863 | 1.9139 | 89.7207 | 3.6011 | 0.4450 |
| loveda/P | TransferAliasShuffle1 | tree | 40.8899 | -15.7012 | 99.2328 | 41.0196 | 6.8060 |
| loveda/P | TransferAliasShuffle1 | farm | 83.0230 | -2.5608 | 83.3391 | 99.5452 | 58.9878 |
| loveda/P | TransferAliasShuffle2 | building | 62.8019 | 13.8727 | 70.8447 | 84.6906 | 0.0317 |
| loveda/P | TransferAliasShuffle2 | road | 54.5853 | -5.5758 | 55.5951 | 96.7798 | 14.4873 |
| loveda/P | TransferAliasShuffle2 | water | 62.3894 | -0.5348 | 67.9556 | 88.3949 | 19.1403 |
| loveda/P | TransferAliasShuffle2 | barren | 4.2001 | 2.5277 | 89.2381 | 4.2215 | 0.5245 |
| loveda/P | TransferAliasShuffle2 | tree | 42.2334 | -14.3577 | 99.2801 | 42.3631 | 7.0256 |
| loveda/P | TransferAliasShuffle2 | farm | 83.3048 | -2.2790 | 83.6210 | 99.5480 | 58.7906 |
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
| loveda/D | MarginTransfer_Projected | background | 24.3875 | -6.3059 | 85.8115 | 25.4122 | 13.2530 |
| loveda/D | MarginTransfer_Projected | building | 34.2548 | 2.6923 | 35.1852 | 92.8339 | 0.0386 |
| loveda/D | MarginTransfer_Projected | road | 47.6934 | -1.5817 | 48.4791 | 96.7134 | 9.1724 |
| loveda/D | MarginTransfer_Projected | water | 54.5697 | -0.4314 | 58.5732 | 88.8689 | 12.3342 |
| loveda/D | MarginTransfer_Projected | barren | 1.7972 | 0.1795 | 69.5873 | 1.8114 | 0.1595 |
| loveda/D | MarginTransfer_Projected | tree | 44.2662 | -4.2057 | 94.6033 | 45.4129 | 4.3666 |
| loveda/D | MarginTransfer_Projected | farm | 44.6729 | -1.9215 | 44.7635 | 99.5491 | 60.6758 |
| loveda/D | MarginNeutral_Projected | background | 20.3746 | -10.3188 | 84.0842 | 21.1918 | 11.2791 |
| loveda/D | MarginNeutral_Projected | building | 35.1250 | 3.5625 | 36.3049 | 91.5309 | 0.0369 |
| loveda/D | MarginNeutral_Projected | road | 46.5288 | -2.7463 | 47.2701 | 96.7394 | 9.4095 |
| loveda/D | MarginNeutral_Projected | water | 54.4078 | -0.5933 | 58.4882 | 88.6348 | 12.3196 |
| loveda/D | MarginNeutral_Projected | barren | 1.9821 | 0.3644 | 70.0109 | 1.9991 | 0.1749 |
| loveda/D | MarginNeutral_Projected | tree | 40.7130 | -7.7589 | 95.6010 | 41.4903 | 3.9478 |
| loveda/D | MarginNeutral_Projected | farm | 43.1417 | -3.4527 | 43.2265 | 99.5475 | 62.8323 |
| loveda/D | MarginDeficit_Projected | background | 14.9507 | -15.7427 | 81.7746 | 15.4661 | 8.4641 |
| loveda/D | MarginDeficit_Projected | building | 36.5253 | 4.9628 | 38.6397 | 86.9707 | 0.0329 |
| loveda/D | MarginDeficit_Projected | road | 46.0202 | -3.2549 | 46.7458 | 96.7373 | 9.5149 |
| loveda/D | MarginDeficit_Projected | water | 54.5624 | -0.4387 | 58.7042 | 88.5498 | 12.2624 |
| loveda/D | MarginDeficit_Projected | barren | 2.5750 | 0.9573 | 80.2169 | 2.5914 | 0.1979 |
| loveda/D | MarginDeficit_Projected | tree | 40.3784 | -8.0935 | 96.3832 | 40.9997 | 3.8694 |
| loveda/D | MarginDeficit_Projected | farm | 41.2821 | -5.3123 | 41.3616 | 99.5370 | 65.6583 |
| loveda/D | RatioSoft_Projected | background | 13.1430 | -17.5504 | 80.1578 | 13.5850 | 7.5846 |
| loveda/D | RatioSoft_Projected | building | 38.0342 | 6.4717 | 40.3323 | 86.9707 | 0.0316 |
| loveda/D | RatioSoft_Projected | road | 46.2854 | -2.9897 | 47.0219 | 96.7269 | 9.4580 |
| loveda/D | RatioSoft_Projected | water | 54.6191 | -0.3820 | 58.7621 | 88.5674 | 12.2528 |
| loveda/D | RatioSoft_Projected | barren | 2.7300 | 1.1123 | 81.3884 | 2.7471 | 0.2068 |
| loveda/D | RatioSoft_Projected | tree | 40.1411 | -8.3308 | 96.4957 | 40.7349 | 3.8400 |
| loveda/D | RatioSoft_Projected | farm | 40.6813 | -5.9131 | 40.7590 | 99.5332 | 66.6264 |
| loveda/D | TransferClassMean_Projected | background | 8.1069 | -22.5865 | 77.0862 | 8.3071 | 4.8227 |
| loveda/D | TransferClassMean_Projected | building | 31.2821 | -0.2804 | 34.0307 | 79.4788 | 0.0342 |
| loveda/D | TransferClassMean_Projected | road | 44.1408 | -5.1343 | 44.7979 | 96.7840 | 9.9334 |
| loveda/D | TransferClassMean_Projected | water | 54.4205 | -0.5806 | 58.7125 | 88.1580 | 12.2065 |
| loveda/D | TransferClassMean_Projected | barren | 3.1328 | 1.5151 | 81.6314 | 3.1550 | 0.2367 |
| loveda/D | TransferClassMean_Projected | tree | 36.2188 | -12.2531 | 97.0574 | 36.6210 | 3.4322 |
| loveda/D | TransferClassMean_Projected | farm | 39.0945 | -7.4999 | 39.1666 | 99.5318 | 69.3343 |
| loveda/D | ProjectedHardStrengthMatched | background | 24.7684 | -5.9250 | 85.4846 | 25.8558 | 13.5360 |
| loveda/D | ProjectedHardStrengthMatched | building | 35.3160 | 3.7535 | 36.3057 | 92.8339 | 0.0374 |
| loveda/D | ProjectedHardStrengthMatched | road | 47.8179 | -1.4572 | 48.6104 | 96.7031 | 9.1466 |
| loveda/D | ProjectedHardStrengthMatched | water | 54.5689 | -0.4322 | 58.5564 | 88.9052 | 12.3427 |
| loveda/D | ProjectedHardStrengthMatched | barren | 1.7945 | 0.1768 | 68.9205 | 1.8091 | 0.1608 |
| loveda/D | ProjectedHardStrengthMatched | tree | 43.6288 | -4.8431 | 94.5277 | 44.7593 | 4.3072 |
| loveda/D | ProjectedHardStrengthMatched | farm | 44.8259 | -1.7685 | 44.9169 | 99.5501 | 60.4692 |
| loveda/D | TransferAliasShuffle0 | background | 10.4575 | -20.2359 | 80.4703 | 10.7299 | 5.9673 |
| loveda/D | TransferAliasShuffle0 | building | 34.6720 | 3.1095 | 37.0529 | 84.3648 | 0.0333 |
| loveda/D | TransferAliasShuffle0 | road | 44.4701 | -4.8050 | 45.1401 | 96.7705 | 9.8567 |
| loveda/D | TransferAliasShuffle0 | water | 54.1641 | -0.8370 | 58.3885 | 88.2166 | 12.2824 |
| loveda/D | TransferAliasShuffle0 | barren | 2.7100 | 1.0923 | 82.0525 | 2.7261 | 0.2035 |
| loveda/D | TransferAliasShuffle0 | tree | 37.5254 | -10.9465 | 96.8015 | 37.9965 | 3.5705 |
| loveda/D | TransferAliasShuffle0 | farm | 39.8167 | -6.7777 | 39.8895 | 99.5442 | 68.0863 |
| loveda/D | TransferAliasShuffle1 | background | 10.4605 | -20.2329 | 79.0984 | 10.7579 | 6.0866 |
| loveda/D | TransferAliasShuffle1 | building | 36.9251 | 5.3626 | 38.2313 | 91.5309 | 0.0350 |
| loveda/D | TransferAliasShuffle1 | road | 44.3987 | -4.8764 | 45.0671 | 96.7674 | 9.8723 |
| loveda/D | TransferAliasShuffle1 | water | 54.6813 | -0.3198 | 58.9826 | 88.2331 | 12.1609 |
| loveda/D | TransferAliasShuffle1 | barren | 3.1683 | 1.5506 | 80.7760 | 3.1924 | 0.2421 |
| loveda/D | TransferAliasShuffle1 | tree | 36.5720 | -11.8999 | 96.7902 | 37.0209 | 3.4792 |
| loveda/D | TransferAliasShuffle1 | farm | 39.7883 | -6.8061 | 39.8628 | 99.5323 | 68.1237 |
| loveda/D | TransferAliasShuffle2 | background | 7.1769 | -23.5165 | 74.1203 | 7.3614 | 4.4447 |
| loveda/D | TransferAliasShuffle2 | building | 33.8521 | 2.2896 | 36.0000 | 85.0163 | 0.0346 |
| loveda/D | TransferAliasShuffle2 | road | 44.4484 | -4.8267 | 45.1201 | 96.7591 | 9.8599 |
| loveda/D | TransferAliasShuffle2 | water | 54.7417 | -0.2594 | 59.0947 | 88.1398 | 12.1250 |
| loveda/D | TransferAliasShuffle2 | barren | 2.9294 | 1.3117 | 81.1349 | 2.9495 | 0.2227 |
| loveda/D | TransferAliasShuffle2 | tree | 36.8476 | -11.6243 | 96.5439 | 37.3402 | 3.5182 |
| loveda/D | TransferAliasShuffle2 | farm | 38.8369 | -7.7575 | 38.9080 | 99.5316 | 69.7949 |
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
| vaihingen/vaihingen | MarginTransfer_Projected | impervious surface | 58.6769 | -0.6997 | 74.9514 | 72.9900 | 26.6018 |
| vaihingen/vaihingen | MarginTransfer_Projected | building | 66.8296 | -0.6378 | 66.9668 | 99.6942 | 30.7130 |
| vaihingen/vaihingen | MarginTransfer_Projected | low vegetation | 44.9512 | -0.8230 | 96.0515 | 45.7974 | 14.0981 |
| vaihingen/vaihingen | MarginTransfer_Projected | tree | 67.4940 | -0.1194 | 78.7849 | 82.4855 | 21.6080 |
| vaihingen/vaihingen | MarginTransfer_Projected | car | 25.7554 | -0.5426 | 25.8979 | 97.9078 | 6.9791 |
| vaihingen/vaihingen | MarginNeutral_Projected | impervious surface | 58.1410 | -1.2356 | 74.5589 | 72.5302 | 26.5734 |
| vaihingen/vaihingen | MarginNeutral_Projected | building | 66.4827 | -0.9847 | 66.6180 | 99.6954 | 30.8742 |
| vaihingen/vaihingen | MarginNeutral_Projected | low vegetation | 44.5333 | -1.2409 | 96.0671 | 45.3602 | 13.9612 |
| vaihingen/vaihingen | MarginNeutral_Projected | tree | 67.4271 | -0.1863 | 78.9407 | 82.2158 | 21.4949 |
| vaihingen/vaihingen | MarginNeutral_Projected | car | 25.3129 | -0.9851 | 25.4546 | 97.8484 | 7.0964 |
| vaihingen/vaihingen | MarginDeficit_Projected | impervious surface | 58.0340 | -1.3426 | 74.3033 | 72.6063 | 26.6928 |
| vaihingen/vaihingen | MarginDeficit_Projected | building | 66.2963 | -1.1711 | 66.4307 | 99.6958 | 30.9614 |
| vaihingen/vaihingen | MarginDeficit_Projected | low vegetation | 44.2235 | -1.5507 | 96.1608 | 45.0184 | 13.8425 |
| vaihingen/vaihingen | MarginDeficit_Projected | tree | 67.3845 | -0.2289 | 78.8997 | 82.1971 | 21.5012 |
| vaihingen/vaihingen | MarginDeficit_Projected | car | 25.6455 | -0.6525 | 25.7922 | 97.8303 | 7.0022 |
| vaihingen/vaihingen | RatioSoft_Projected | impervious surface | 58.0658 | -1.3108 | 74.3186 | 72.6414 | 26.7002 |
| vaihingen/vaihingen | RatioSoft_Projected | building | 66.2831 | -1.1843 | 66.4155 | 99.7000 | 30.9697 |
| vaihingen/vaihingen | RatioSoft_Projected | low vegetation | 44.2974 | -1.4768 | 96.1358 | 45.1004 | 13.8713 |
| vaihingen/vaihingen | RatioSoft_Projected | tree | 67.3978 | -0.2156 | 78.9209 | 82.1939 | 21.4945 |
| vaihingen/vaihingen | RatioSoft_Projected | car | 25.7673 | -0.5307 | 25.9190 | 97.7786 | 6.9643 |
| vaihingen/vaihingen | TransferClassMean_Projected | impervious surface | 57.3237 | -2.0529 | 73.8018 | 71.9684 | 26.6381 |
| vaihingen/vaihingen | TransferClassMean_Projected | building | 65.7981 | -1.6693 | 65.9309 | 99.6947 | 31.1957 |
| vaihingen/vaihingen | TransferClassMean_Projected | low vegetation | 43.7066 | -2.0676 | 96.1837 | 44.4780 | 13.6731 |
| vaihingen/vaihingen | TransferClassMean_Projected | tree | 67.2708 | -0.3426 | 79.0833 | 81.8304 | 21.3555 |
| vaihingen/vaihingen | TransferClassMean_Projected | car | 25.1283 | -1.1697 | 25.2761 | 97.7270 | 7.1376 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | impervious surface | 58.6552 | -0.7214 | 74.8977 | 73.0074 | 26.6273 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | building | 66.8036 | -0.6638 | 66.9395 | 99.6970 | 30.7264 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | low vegetation | 44.9582 | -0.8160 | 96.0155 | 45.8129 | 14.1081 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | tree | 67.4915 | -0.1219 | 78.8231 | 82.4399 | 21.5856 |
| vaihingen/vaihingen | ProjectedHardStrengthMatched | car | 25.8471 | -0.4509 | 25.9919 | 97.8897 | 6.9526 |
| vaihingen/vaihingen | TransferAliasShuffle0 | impervious surface | 57.6108 | -1.7658 | 73.9645 | 72.2655 | 26.6892 |
| vaihingen/vaihingen | TransferAliasShuffle0 | building | 66.3639 | -1.1035 | 66.4988 | 99.6951 | 30.9294 |
| vaihingen/vaihingen | TransferAliasShuffle0 | low vegetation | 43.9725 | -1.8017 | 96.1038 | 44.7707 | 13.7745 |
| vaihingen/vaihingen | TransferAliasShuffle0 | tree | 67.3524 | -0.2610 | 78.9553 | 82.0892 | 21.4578 |
| vaihingen/vaihingen | TransferAliasShuffle0 | car | 25.1083 | -1.1897 | 25.2516 | 97.7890 | 7.1491 |
| vaihingen/vaihingen | TransferAliasShuffle1 | impervious surface | 57.4726 | -1.9040 | 73.7836 | 72.2207 | 26.7380 |
| vaihingen/vaihingen | TransferAliasShuffle1 | building | 66.0486 | -1.4188 | 66.1820 | 99.6956 | 31.0776 |
| vaihingen/vaihingen | TransferAliasShuffle1 | low vegetation | 43.8590 | -1.9152 | 96.1207 | 44.6494 | 13.7348 |
| vaihingen/vaihingen | TransferAliasShuffle1 | tree | 67.3141 | -0.2993 | 79.1511 | 81.8219 | 21.3350 |
| vaihingen/vaihingen | TransferAliasShuffle1 | car | 25.2378 | -1.0602 | 25.3809 | 97.8148 | 7.1146 |
| vaihingen/vaihingen | TransferAliasShuffle2 | impervious surface | 57.7579 | -1.6187 | 74.0895 | 72.3774 | 26.6854 |
| vaihingen/vaihingen | TransferAliasShuffle2 | building | 66.2935 | -1.1739 | 66.4300 | 99.6910 | 30.9602 |
| vaihingen/vaihingen | TransferAliasShuffle2 | low vegetation | 44.0549 | -1.7193 | 96.1555 | 44.8448 | 13.7899 |
| vaihingen/vaihingen | TransferAliasShuffle2 | tree | 67.3536 | -0.2598 | 78.9806 | 82.0636 | 21.4442 |
| vaihingen/vaihingen | TransferAliasShuffle2 | car | 25.2067 | -1.0913 | 25.3518 | 97.7812 | 7.1203 |
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
| landcoverai/landcoverai | MarginTransfer_Projected | background | 87.9273 | -0.3097 | 94.6358 | 92.5394 | 66.2589 |
| landcoverai/landcoverai | MarginTransfer_Projected | building | 44.4984 | -0.0953 | 44.8303 | 98.3634 | 3.3242 |
| landcoverai/landcoverai | MarginTransfer_Projected | woodland | 80.5076 | -1.0337 | 94.2884 | 84.6351 | 19.2152 |
| landcoverai/landcoverai | MarginTransfer_Projected | water | 97.6101 | 0.0022 | 97.6101 | 100.0000 | 8.4738 |
| landcoverai/landcoverai | MarginTransfer_Projected | road | 26.5559 | -0.3820 | 29.0361 | 75.6627 | 2.7279 |
| landcoverai/landcoverai | MarginNeutral_Projected | background | 87.7694 | -0.4676 | 94.3451 | 92.6431 | 66.5376 |
| landcoverai/landcoverai | MarginNeutral_Projected | building | 44.4601 | -0.1336 | 44.7980 | 98.3319 | 3.3256 |
| landcoverai/landcoverai | MarginNeutral_Projected | woodland | 79.8859 | -1.6554 | 94.6466 | 83.6664 | 18.9233 |
| landcoverai/landcoverai | MarginNeutral_Projected | water | 97.6140 | 0.0061 | 97.6140 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | MarginNeutral_Projected | road | 26.4380 | -0.4999 | 28.8986 | 75.6400 | 2.7400 |
| landcoverai/landcoverai | MarginDeficit_Projected | background | 87.7263 | -0.5107 | 94.2901 | 92.6481 | 66.5801 |
| landcoverai/landcoverai | MarginDeficit_Projected | building | 44.4767 | -0.1170 | 44.8194 | 98.3099 | 3.3232 |
| landcoverai/landcoverai | MarginDeficit_Projected | woodland | 79.6803 | -1.8610 | 94.5884 | 83.4862 | 18.8942 |
| landcoverai/landcoverai | MarginDeficit_Projected | water | 97.6123 | 0.0044 | 97.6123 | 100.0000 | 8.4736 |
| landcoverai/landcoverai | MarginDeficit_Projected | road | 26.5369 | -0.4010 | 29.0168 | 75.6400 | 2.7289 |
| landcoverai/landcoverai | RatioSoft_Projected | background | 87.8007 | -0.4363 | 94.3470 | 92.6762 | 66.5600 |
| landcoverai/landcoverai | RatioSoft_Projected | building | 44.5182 | -0.0755 | 44.8576 | 98.3288 | 3.3210 |
| landcoverai/landcoverai | RatioSoft_Projected | woodland | 79.8496 | -1.6917 | 94.5922 | 83.6691 | 18.9348 |
| landcoverai/landcoverai | RatioSoft_Projected | water | 97.6112 | 0.0033 | 97.6112 | 100.0000 | 8.4737 |
| landcoverai/landcoverai | RatioSoft_Projected | road | 26.7000 | -0.2379 | 29.2126 | 75.6354 | 2.7104 |
| landcoverai/landcoverai | TransferClassMean_Projected | background | 87.4948 | -0.7422 | 93.9468 | 92.7220 | 66.8766 |
| landcoverai/landcoverai | TransferClassMean_Projected | building | 44.4403 | -0.1534 | 44.7838 | 98.3036 | 3.3257 |
| landcoverai/landcoverai | TransferClassMean_Projected | woodland | 78.8374 | -2.7039 | 94.8836 | 82.3377 | 18.5763 |
| landcoverai/landcoverai | TransferClassMean_Projected | water | 97.6162 | 0.0083 | 97.6162 | 100.0000 | 8.4733 |
| landcoverai/landcoverai | TransferClassMean_Projected | road | 26.3522 | -0.5857 | 28.8007 | 75.6081 | 2.7482 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | background | 87.9820 | -0.2550 | 94.6627 | 92.5743 | 66.2652 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | building | 44.5249 | -0.0688 | 44.8552 | 98.3728 | 3.3227 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | woodland | 80.6039 | -0.9374 | 94.3184 | 84.7173 | 19.2277 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | water | 97.6112 | 0.0033 | 97.6112 | 100.0000 | 8.4737 |
| landcoverai/landcoverai | ProjectedHardStrengthMatched | road | 26.7097 | -0.2282 | 29.2200 | 75.6627 | 2.7107 |
| landcoverai/landcoverai | TransferAliasShuffle0 | background | 87.6346 | -0.6024 | 94.1622 | 92.6694 | 66.6858 |
| landcoverai/landcoverai | TransferAliasShuffle0 | building | 44.4304 | -0.1633 | 44.7730 | 98.3067 | 3.3266 |
| landcoverai/landcoverai | TransferAliasShuffle0 | woodland | 79.3921 | -2.1492 | 94.7298 | 83.0607 | 18.7698 |
| landcoverai/landcoverai | TransferAliasShuffle0 | water | 97.6167 | 0.0088 | 97.6167 | 100.0000 | 8.4733 |
| landcoverai/landcoverai | TransferAliasShuffle0 | road | 26.3800 | -0.5579 | 28.8353 | 75.5990 | 2.7446 |
| landcoverai/landcoverai | TransferAliasShuffle1 | background | 87.6385 | -0.5985 | 94.1462 | 92.6893 | 66.7114 |
| landcoverai/landcoverai | TransferAliasShuffle1 | building | 44.4138 | -0.1799 | 44.7542 | 98.3162 | 3.3283 |
| landcoverai/landcoverai | TransferAliasShuffle1 | woodland | 79.3598 | -2.1815 | 94.7587 | 83.0033 | 18.7511 |
| landcoverai/landcoverai | TransferAliasShuffle1 | water | 97.6151 | 0.0072 | 97.6151 | 100.0000 | 8.4734 |
| landcoverai/landcoverai | TransferAliasShuffle1 | road | 26.4599 | -0.4780 | 28.9300 | 75.6035 | 2.7358 |
| landcoverai/landcoverai | TransferAliasShuffle2 | background | 87.4901 | -0.7469 | 93.9810 | 92.6834 | 66.8244 |
| landcoverai/landcoverai | TransferAliasShuffle2 | building | 44.4163 | -0.1774 | 44.7568 | 98.3162 | 3.3281 |
| landcoverai/landcoverai | TransferAliasShuffle2 | woodland | 78.8642 | -2.6771 | 94.7655 | 82.4562 | 18.6262 |
| landcoverai/landcoverai | TransferAliasShuffle2 | water | 97.6189 | 0.0110 | 97.6189 | 100.0000 | 8.4731 |
| landcoverai/landcoverai | TransferAliasShuffle2 | road | 26.3562 | -0.5817 | 28.8042 | 75.6172 | 2.7482 |
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
| flair1/flair1 | MarginTransfer_Projected | building | 57.0822 | -0.3219 | 58.3450 | 96.3469 | 11.8558 |
| flair1/flair1 | MarginTransfer_Projected | pervious surface | 47.0501 | 0.2429 | 91.8230 | 49.1076 | 9.1772 |
| flair1/flair1 | MarginTransfer_Projected | impervious surface | 52.5494 | -0.3512 | 60.4391 | 80.1016 | 21.7121 |
| flair1/flair1 | MarginTransfer_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0931 |
| flair1/flair1 | MarginTransfer_Projected | water | 63.4553 | -1.3284 | 65.4447 | 95.4286 | 6.4684 |
| flair1/flair1 | MarginTransfer_Projected | coniferous | 42.9180 | 0.1915 | 63.1421 | 57.2641 | 0.5098 |
| flair1/flair1 | MarginTransfer_Projected | deciduous | 60.5423 | -0.3625 | 78.7039 | 72.4032 | 15.6590 |
| flair1/flair1 | MarginTransfer_Projected | brushwood | 14.3222 | -1.3084 | 28.5972 | 22.2950 | 3.8345 |
| flair1/flair1 | MarginTransfer_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0110 |
| flair1/flair1 | MarginTransfer_Projected | herbaceous vegetation | 57.5959 | -0.0244 | 91.2773 | 60.9506 | 21.3915 |
| flair1/flair1 | MarginTransfer_Projected | agricultural land | 13.3254 | -0.1051 | 14.3584 | 64.9406 | 1.3791 |
| flair1/flair1 | MarginTransfer_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9086 |
| flair1/flair1 | MarginNeutral_Projected | building | 56.9519 | -0.4522 | 58.2183 | 96.3210 | 11.8784 |
| flair1/flair1 | MarginNeutral_Projected | pervious surface | 47.1240 | 0.3168 | 91.7705 | 49.2033 | 9.2003 |
| flair1/flair1 | MarginNeutral_Projected | impervious surface | 52.3373 | -0.5633 | 60.1951 | 80.0372 | 21.7826 |
| flair1/flair1 | MarginNeutral_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1049 |
| flair1/flair1 | MarginNeutral_Projected | water | 62.7894 | -1.9943 | 64.7059 | 95.4953 | 6.5468 |
| flair1/flair1 | MarginNeutral_Projected | coniferous | 43.1343 | 0.4078 | 64.2192 | 56.7804 | 0.4970 |
| flair1/flair1 | MarginNeutral_Projected | deciduous | 60.1186 | -0.7862 | 78.7404 | 71.7679 | 15.5144 |
| flair1/flair1 | MarginNeutral_Projected | brushwood | 13.6290 | -2.0016 | 27.7231 | 21.1408 | 3.7506 |
| flair1/flair1 | MarginNeutral_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0083 |
| flair1/flair1 | MarginNeutral_Projected | herbaceous vegetation | 57.5600 | -0.0603 | 91.1091 | 60.9856 | 21.4433 |
| flair1/flair1 | MarginNeutral_Projected | agricultural land | 13.1586 | -0.2719 | 14.1574 | 65.0970 | 1.4020 |
| flair1/flair1 | MarginNeutral_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8714 |
| flair1/flair1 | MarginDeficit_Projected | building | 56.9394 | -0.4647 | 58.2016 | 96.3310 | 11.8830 |
| flair1/flair1 | MarginDeficit_Projected | pervious surface | 47.1954 | 0.3882 | 91.7955 | 49.2739 | 9.2110 |
| flair1/flair1 | MarginDeficit_Projected | impervious surface | 52.3148 | -0.5858 | 60.1675 | 80.0334 | 21.7915 |
| flair1/flair1 | MarginDeficit_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1065 |
| flair1/flair1 | MarginDeficit_Projected | water | 62.4456 | -2.3381 | 64.3365 | 95.5050 | 6.5851 |
| flair1/flair1 | MarginDeficit_Projected | coniferous | 43.1128 | 0.3863 | 64.0957 | 56.8398 | 0.4985 |
| flair1/flair1 | MarginDeficit_Projected | deciduous | 60.1787 | -0.7261 | 78.7551 | 71.8413 | 15.5273 |
| flair1/flair1 | MarginDeficit_Projected | brushwood | 13.3693 | -2.2613 | 27.6392 | 20.5686 | 3.6602 |
| flair1/flair1 | MarginDeficit_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0054 |
| flair1/flair1 | MarginDeficit_Projected | herbaceous vegetation | 57.6055 | -0.0148 | 91.0445 | 61.0657 | 21.4868 |
| flair1/flair1 | MarginDeficit_Projected | agricultural land | 13.1605 | -0.2700 | 14.1589 | 65.1126 | 1.4022 |
| flair1/flair1 | MarginDeficit_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8425 |
| flair1/flair1 | RatioSoft_Projected | building | 57.0415 | -0.3626 | 58.3003 | 96.3529 | 11.8656 |
| flair1/flair1 | RatioSoft_Projected | pervious surface | 47.1067 | 0.2995 | 91.8325 | 49.1666 | 9.1873 |
| flair1/flair1 | RatioSoft_Projected | impervious surface | 52.3974 | -0.5032 | 60.2435 | 80.0922 | 21.7800 |
| flair1/flair1 | RatioSoft_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1181 |
| flair1/flair1 | RatioSoft_Projected | water | 62.6394 | -2.1443 | 64.5471 | 95.4942 | 6.5629 |
| flair1/flair1 | RatioSoft_Projected | coniferous | 43.1187 | 0.3922 | 64.2062 | 56.7634 | 0.4970 |
| flair1/flair1 | RatioSoft_Projected | deciduous | 60.1916 | -0.7132 | 78.7899 | 71.8307 | 15.5182 |
| flair1/flair1 | RatioSoft_Projected | brushwood | 13.5872 | -2.0434 | 27.7710 | 21.0128 | 3.7215 |
| flair1/flair1 | RatioSoft_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0075 |
| flair1/flair1 | RatioSoft_Projected | herbaceous vegetation | 57.6257 | 0.0054 | 91.0492 | 61.0863 | 21.4929 |
| flair1/flair1 | RatioSoft_Projected | agricultural land | 13.2370 | -0.1935 | 14.2558 | 64.9406 | 1.3890 |
| flair1/flair1 | RatioSoft_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8601 |
| flair1/flair1 | TransferClassMean_Projected | building | 56.7704 | -0.6337 | 58.0333 | 96.3084 | 11.9147 |
| flair1/flair1 | TransferClassMean_Projected | pervious surface | 47.2437 | 0.4365 | 91.7083 | 49.3517 | 9.2343 |
| flair1/flair1 | TransferClassMean_Projected | impervious surface | 51.9805 | -0.9201 | 59.7698 | 79.9545 | 21.9149 |
| flair1/flair1 | TransferClassMean_Projected | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1528 |
| flair1/flair1 | TransferClassMean_Projected | water | 61.5654 | -3.2183 | 63.3680 | 95.5835 | 6.6912 |
| flair1/flair1 | TransferClassMean_Projected | coniferous | 43.5116 | 0.7851 | 65.6482 | 56.3391 | 0.4824 |
| flair1/flair1 | TransferClassMean_Projected | deciduous | 59.5706 | -1.3342 | 78.7870 | 70.9504 | 15.3286 |
| flair1/flair1 | TransferClassMean_Projected | brushwood | 12.4143 | -3.2163 | 26.3929 | 18.9886 | 3.5386 |
| flair1/flair1 | TransferClassMean_Projected | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0017 |
| flair1/flair1 | TransferClassMean_Projected | herbaceous vegetation | 57.4581 | -0.1622 | 90.8440 | 60.9901 | 21.5075 |
| flair1/flair1 | TransferClassMean_Projected | agricultural land | 12.9297 | -0.5008 | 13.8857 | 65.2534 | 1.4329 |
| flair1/flair1 | TransferClassMean_Projected | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8004 |
| flair1/flair1 | ProjectedHardStrengthMatched | building | 57.1616 | -0.2425 | 58.4264 | 96.3509 | 11.8397 |
| flair1/flair1 | ProjectedHardStrengthMatched | pervious surface | 47.0415 | 0.2343 | 91.8070 | 49.1029 | 9.1779 |
| flair1/flair1 | ProjectedHardStrengthMatched | impervious surface | 52.5690 | -0.3316 | 60.4407 | 80.1444 | 21.7231 |
| flair1/flair1 | ProjectedHardStrengthMatched | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0933 |
| flair1/flair1 | ProjectedHardStrengthMatched | water | 63.5508 | -1.2329 | 65.5452 | 95.4308 | 6.4586 |
| flair1/flair1 | ProjectedHardStrengthMatched | coniferous | 42.9191 | 0.1926 | 63.3836 | 57.0689 | 0.5061 |
| flair1/flair1 | ProjectedHardStrengthMatched | deciduous | 60.4121 | -0.4927 | 78.7509 | 72.1776 | 15.6009 |
| flair1/flair1 | ProjectedHardStrengthMatched | brushwood | 14.4453 | -1.1853 | 28.5942 | 22.5966 | 3.8868 |
| flair1/flair1 | ProjectedHardStrengthMatched | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0126 |
| flair1/flair1 | ProjectedHardStrengthMatched | herbaceous vegetation | 57.5878 | -0.0325 | 91.2362 | 60.9598 | 21.4044 |
| flair1/flair1 | ProjectedHardStrengthMatched | agricultural land | 13.3544 | -0.0761 | 14.3974 | 64.8310 | 1.3730 |
| flair1/flair1 | ProjectedHardStrengthMatched | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9236 |
| flair1/flair1 | TransferAliasShuffle0 | building | 56.8765 | -0.5276 | 58.0813 | 96.4812 | 11.9262 |
| flair1/flair1 | TransferAliasShuffle0 | pervious surface | 47.2966 | 0.4894 | 91.8298 | 49.3742 | 9.2263 |
| flair1/flair1 | TransferAliasShuffle0 | impervious surface | 52.1086 | -0.7920 | 59.9039 | 80.0174 | 21.8830 |
| flair1/flair1 | TransferAliasShuffle0 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1295 |
| flair1/flair1 | TransferAliasShuffle0 | water | 61.9445 | -2.8392 | 63.7731 | 95.5760 | 6.6482 |
| flair1/flair1 | TransferAliasShuffle0 | coniferous | 43.4231 | 0.6966 | 65.4126 | 56.3646 | 0.4844 |
| flair1/flair1 | TransferAliasShuffle0 | deciduous | 59.6546 | -1.2502 | 78.7613 | 71.0905 | 15.3639 |
| flair1/flair1 | TransferAliasShuffle0 | brushwood | 12.5830 | -3.0476 | 26.2324 | 19.4735 | 3.6512 |
| flair1/flair1 | TransferAliasShuffle0 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0021 |
| flair1/flair1 | TransferAliasShuffle0 | herbaceous vegetation | 57.4930 | -0.1273 | 91.0392 | 60.9417 | 21.4443 |
| flair1/flair1 | TransferAliasShuffle0 | agricultural land | 12.8986 | -0.5319 | 13.8506 | 65.2378 | 1.4362 |
| flair1/flair1 | TransferAliasShuffle0 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8048 |
| flair1/flair1 | TransferAliasShuffle1 | building | 56.8932 | -0.5109 | 58.1064 | 96.4599 | 11.9184 |
| flair1/flair1 | TransferAliasShuffle1 | pervious surface | 47.2691 | 0.4619 | 91.8756 | 49.3311 | 9.2137 |
| flair1/flair1 | TransferAliasShuffle1 | impervious surface | 52.1183 | -0.7823 | 59.9125 | 80.0250 | 21.8820 |
| flair1/flair1 | TransferAliasShuffle1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0972 |
| flair1/flair1 | TransferAliasShuffle1 | water | 61.7782 | -3.0055 | 63.6016 | 95.5652 | 6.6654 |
| flair1/flair1 | TransferAliasShuffle1 | coniferous | 43.3340 | 0.6075 | 65.0073 | 56.5173 | 0.4887 |
| flair1/flair1 | TransferAliasShuffle1 | deciduous | 60.0550 | -0.8498 | 78.8138 | 71.6165 | 15.4672 |
| flair1/flair1 | TransferAliasShuffle1 | brushwood | 12.7355 | -2.8951 | 27.1735 | 19.3348 | 3.4996 |
| flair1/flair1 | TransferAliasShuffle1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0021 |
| flair1/flair1 | TransferAliasShuffle1 | herbaceous vegetation | 57.5976 | -0.0227 | 90.9015 | 61.1213 | 21.5401 |
| flair1/flair1 | TransferAliasShuffle1 | agricultural land | 12.9499 | -0.4806 | 13.9097 | 65.2378 | 1.4301 |
| flair1/flair1 | TransferAliasShuffle1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7954 |
| flair1/flair1 | TransferAliasShuffle2 | building | 56.8251 | -0.5790 | 58.0858 | 96.3210 | 11.9055 |
| flair1/flair1 | TransferAliasShuffle2 | pervious surface | 47.3348 | 0.5276 | 91.7760 | 49.4315 | 9.2424 |
| flair1/flair1 | TransferAliasShuffle2 | impervious surface | 52.0149 | -0.8857 | 59.7944 | 79.9918 | 21.9161 |
| flair1/flair1 | TransferAliasShuffle2 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0947 |
| flair1/flair1 | TransferAliasShuffle2 | water | 62.4410 | -2.3427 | 64.3336 | 95.5007 | 6.5851 |
| flair1/flair1 | TransferAliasShuffle2 | coniferous | 43.2649 | 0.5384 | 64.9078 | 56.4749 | 0.4891 |
| flair1/flair1 | TransferAliasShuffle2 | deciduous | 59.9255 | -0.9793 | 78.7514 | 71.4837 | 15.4508 |
| flair1/flair1 | TransferAliasShuffle2 | brushwood | 13.2412 | -2.3894 | 27.9017 | 20.1282 | 3.5481 |
| flair1/flair1 | TransferAliasShuffle2 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0032 |
| flair1/flair1 | TransferAliasShuffle2 | herbaceous vegetation | 57.5345 | -0.0858 | 90.8808 | 61.0596 | 21.5233 |
| flair1/flair1 | TransferAliasShuffle2 | agricultural land | 12.9637 | -0.4668 | 13.9264 | 65.2222 | 1.4280 |
| flair1/flair1 | TransferAliasShuffle2 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8136 |

All implementations and raw outcomes retained. Developed pilots do not prove SOTA or every-alias utility.
