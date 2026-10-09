# Frozen Geometry: cached rival screening and soft admission

Verified64 developed top-left512 windows (eight/domain), not full datasets. Same20 aliases, original Geometry, finite VIP wide observer and reconstruction. Full-image wide context retained. LoveDA D counted once; P separately. Ground truth only after score persistence. No rule/threshold/strength fitted to these masks.

Original versus cached scores, risks and diagnostics are bitwise identical on all64 windows. Historical no-admission and hard-admission predictions replay exactly per image. VDD first full image also has zero prediction mismatches and identical diagnostics.

The legacy window Geometry writer interpolated cosine/.07 in the archived score dtype; the current full evaluator interpolates cosine in float32 before dividing by.07. Their confusion differences total10 L1 counts (VDD2, Potsdam4, OEM4). A no-forward audit of stored scores reproduces every legacy per-image Geometry confusion exactly using its original interpolation. This is not a cached-backend discrepancy; current Geometry follows the retained full evaluator.

## Fixed Common Scored-Class mIoU

The same union of scored classes is used across every arm within each dataset/protocol; absent classes predicted by any arm remain in this denominator. Native denominators are retained in summary.json.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineSoft_Weighted | FineSoft_Excess | ReuseSoft_Weighted | ReuseSoft_Excess | ReuseClassMean_Weighted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.3441 | 54.2567 | 54.3308 | 53.4326 | 53.2956 | 53.7325 |
| potsdam/potsdam | 35.7035 | 38.9203 | 40.6173 | 39.3772 | 39.4361 | 39.0708 | 39.0663 | 38.9357 |
| udd5/udd5 | 30.5010 | 28.1758 | 34.0394 | 29.7268 | 29.5160 | 28.4351 | 28.4045 | 28.1877 |
| oem/oem | 39.8205 | 39.0232 | 39.7906 | 39.2104 | 39.2784 | 39.0872 | 39.1932 | 39.0106 |
| loveda/P | 49.5399 | 50.7066 | 52.5879 | 51.9329 | 51.8737 | 51.3924 | 51.8809 | 50.3858 |
| loveda/D | 33.8779 | 30.7360 | 37.2156 | 32.7899 | 33.3658 | 31.4870 | 32.0440 | 30.6992 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 52.9446 | 52.1839 | 52.2902 | 51.8531 | 51.9227 | 51.8165 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.1432 | 67.1392 | 66.8383 | 66.8489 | 66.9111 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 33.7562 | 33.8539 | 33.7378 | 33.8091 | 33.5992 |
| Eight-domain mean | 40.5310 | 42.8679 | 45.0622 | 43.5555 | 43.6513 | 42.9927 | 43.0730 | 42.8616 |

## Independent Window-Context Speed

Warmed synchronized median of three independent executions, alternating order. Full-image wide context plus one512 local window; not full-image throughput. Both resident backbones count toward peak allocated memory. Fine/wide stages include original crop/text preparation, not pure backbone latency.

| Dataset | Original hard s | Cached hard s | Speedup | Fine weighted s | Fine excess s | Reuse weighted s | Reuse excess s | No admission s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 0.417578 | 0.374575 | 1.114806 | 0.375186 | 0.373907 | 0.263997 | 0.261507 | 0.250670 |
| potsdam | 0.357453 | 0.271662 | 1.315799 | 0.271923 | 0.268642 | 0.160022 | 0.155981 | 0.137165 |
| udd5 | 0.328948 | 0.286669 | 1.147485 | 0.285435 | 0.282708 | 0.171291 | 0.169383 | 0.158360 |
| oem | 0.361715 | 0.275057 | 1.315054 | 0.274303 | 0.270991 | 0.162707 | 0.157946 | 0.137010 |
| loveda | 0.489384 | 0.323124 | 1.514537 | 0.322172 | 0.315685 | 0.196513 | 0.187056 | 0.147564 |
| vaihingen | 0.360560 | 0.273257 | 1.319492 | 0.276996 | 0.269521 | 0.159430 | 0.155954 | 0.135389 |
| landcoverai | 0.343286 | 0.260368 | 1.318464 | 0.259260 | 0.258037 | 0.150690 | 0.144119 | 0.129094 |
| flair1 | 0.368420 | 0.284023 | 1.297149 | 0.282835 | 0.289594 | 0.183232 | 0.167426 | 0.149395 |

## Cached Hard Stage Costs

| Dataset | Wide s | Geometry s | Fine s | Operator s | Reader s | Dense s | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 0.181293 | 0.023897 | 0.099885 | 0.002493 | 0.025445 | 0.000205 | 5565.727 |
| potsdam | 0.100915 | 0.023378 | 0.100319 | 0.002487 | 0.036178 | 0.000252 | 5563.794 |
| udd5 | 0.105923 | 0.024118 | 0.102388 | 0.002508 | 0.025512 | 0.000213 | 5556.753 |
| oem | 0.102937 | 0.023519 | 0.100544 | 0.002514 | 0.036898 | 0.000213 | 5573.398 |
| loveda | 0.112254 | 0.023838 | 0.102460 | 0.002501 | 0.071439 | 0.000442 | 5594.012 |
| vaihingen | 0.101948 | 0.023805 | 0.102505 | 0.002480 | 0.035944 | 0.000238 | 5559.560 |
| landcoverai | 0.095711 | 0.023360 | 0.099090 | 0.002464 | 0.034928 | 0.000206 | 5559.560 |
| flair1 | 0.103406 | 0.023622 | 0.101467 | 0.002498 | 0.045088 | 0.000219 | 5798.074 |

## Interpretation

On this panel, cached hard admission remains strongest:45.0622 mean mIoU versus
42.8679 without admission (+2.1942pp), improving all eight main domains. Fine excess
soft reaches43.6513 (+0.7833pp); fine normalized weights43.5555 (+0.6876pp). Both
improve all eight domains versus no admission, but neither beats hard admission here.
Reusing local Geometry features reaches43.0730 at best (+0.2051pp), with losses on
VDD and LandCover.ai. Class-mean weights42.8616 are nearly identity; alias-specific
allocation matters, but this does not establish a universal optimal weighting rule.

Mean independently timed window-context latency falls from0.3784s to0.2936s for
the equivalent cached hard path (1.289x ratio of mean times). Per-domain speedup
ranges1.115-1.515x. The best of these soft variants still uses four fine forwards;
removing them reduces mean latency to0.1749s, but loses most of the screening gain.
The fine stage itself is about0.10s; the current evidence favors retaining its
observation when accuracy is prioritized, not discarding it because caching alone
has already improved speed.

VDD illustrates the tradeoff: hard admission gains vegetation6.2870pp and
water8.2365pp over no admission, while losing vehicle8.1269pp and road1.9079pp.
Fine excess soft limits the vehicle loss to1.7440pp and slightly improves road,
but also recovers much less vegetation/water. It is a more conservative correction,
not a free accuracy improvement. UDD5 gains primarily vegetation/background on
these particular windows; its road IoU is0 in every arm, so this panel cannot
answer whole-UDD5 road performance.

FineSoft_Weighted retains continuous1-risk alias mass in normalized log-mean-exp; FineSoft_Excess suppresses only above-uniform responsibilities. Both retain fine RGB evidence. Reuse variants are genuinely different rules using existing Geometry alias scores instead, with no extra fine forwards. ReuseClassMean keeps each pair risk sum but removes alias specificity; comparison to it separates alias handling from class-only calibration.

A speed gain with unchanged outputs establishes execution optimization, not a new screening contribution. Changes in mIoU test the new soft rules. These results cannot establish that every alias is useful or that VIP screening is universally inferior. All options are preserved; no candidate automatically replaces the retained model.

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs hard | Precision | Recall | Predicted area |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 12.2851 | -44.9574 | 70.4934 | 12.9511 | 9.3800 |
| vdd/vdd | Geometry | wall | 19.8085 | -38.6305 | 20.3032 | 89.0471 | 34.8906 |
| vdd/vdd | Geometry | road | 24.9744 | 3.6573 | 29.4846 | 62.0157 | 6.9810 |
| vdd/vdd | Geometry | vegetation | 69.7492 | 19.3196 | 89.0479 | 76.2942 | 19.7359 |
| vdd/vdd | Geometry | vehicle | 3.8880 | -38.2935 | 3.8880 | 100.0000 | 9.2105 |
| vdd/vdd | Geometry | roof | 63.9827 | -25.7462 | 64.6650 | 98.3777 | 9.1667 |
| vdd/vdd | Geometry | water | 74.5988 | 13.5284 | 75.8754 | 97.7943 | 10.6353 |
| vdd/vdd | NoAdmission_Exact | other | 59.7766 | 2.5341 | 81.1541 | 69.4120 | 43.6684 |
| vdd/vdd | NoAdmission_Exact | wall | 59.6816 | 1.2426 | 65.8225 | 86.4811 | 10.4520 |
| vdd/vdd | NoAdmission_Exact | road | 23.2250 | 1.9079 | 23.3759 | 97.2962 | 13.8146 |
| vdd/vdd | NoAdmission_Exact | vegetation | 44.1426 | -6.2870 | 94.9106 | 45.2128 | 10.9733 |
| vdd/vdd | NoAdmission_Exact | vehicle | 50.3084 | 8.1269 | 50.3286 | 99.9201 | 0.7110 |
| vdd/vdd | NoAdmission_Exact | roof | 86.2607 | -3.4682 | 86.7114 | 99.4009 | 6.9072 |
| vdd/vdd | NoAdmission_Exact | water | 52.8339 | -8.2365 | 55.7407 | 91.0164 | 13.4736 |
| vdd/vdd | RivalFineHard_Exact | other | 57.2425 | 0.0000 | 81.0538 | 66.0849 | 41.6267 |
| vdd/vdd | RivalFineHard_Exact | wall | 58.4390 | 0.0000 | 63.3706 | 88.2481 | 11.0782 |
| vdd/vdd | RivalFineHard_Exact | road | 21.3171 | 0.0000 | 21.3725 | 98.7975 | 15.3427 |
| vdd/vdd | RivalFineHard_Exact | vegetation | 50.4296 | 0.0000 | 95.0722 | 51.7831 | 12.5465 |
| vdd/vdd | RivalFineHard_Exact | vehicle | 42.1815 | 0.0000 | 42.1815 | 100.0000 | 0.8490 |
| vdd/vdd | RivalFineHard_Exact | roof | 89.7289 | 0.0000 | 90.1757 | 99.4508 | 6.6452 |
| vdd/vdd | RivalFineHard_Exact | water | 61.0704 | 0.0000 | 64.1803 | 92.6489 | 11.9117 |
| vdd/vdd | FineSoft_Weighted | other | 59.6177 | 2.3752 | 81.2923 | 69.0978 | 43.3969 |
| vdd/vdd | FineSoft_Weighted | wall | 60.4109 | 1.9719 | 66.3668 | 87.0661 | 10.4364 |
| vdd/vdd | FineSoft_Weighted | road | 23.1678 | 1.8507 | 23.2875 | 97.8292 | 13.9430 |
| vdd/vdd | FineSoft_Weighted | vegetation | 46.0396 | -4.3900 | 94.9195 | 47.2027 | 11.4552 |
| vdd/vdd | FineSoft_Weighted | vehicle | 48.9697 | 6.7882 | 48.9697 | 100.0000 | 0.7313 |
| vdd/vdd | FineSoft_Weighted | roof | 87.8443 | -1.8846 | 88.3025 | 99.4128 | 6.7835 |
| vdd/vdd | FineSoft_Weighted | water | 53.7467 | -7.3237 | 56.7222 | 91.1077 | 13.2537 |
| vdd/vdd | FineSoft_Excess | other | 59.6895 | 2.4470 | 81.2003 | 69.2610 | 43.5486 |
| vdd/vdd | FineSoft_Excess | wall | 59.7520 | 1.3130 | 65.5222 | 87.1548 | 10.5817 |
| vdd/vdd | FineSoft_Excess | road | 23.3602 | 2.0431 | 23.4700 | 98.0375 | 13.8640 |
| vdd/vdd | FineSoft_Excess | vegetation | 46.0408 | -4.3888 | 95.1482 | 47.1477 | 11.4143 |
| vdd/vdd | FineSoft_Excess | vehicle | 48.5644 | 6.3829 | 48.5644 | 100.0000 | 0.7374 |
| vdd/vdd | FineSoft_Excess | roof | 88.2913 | -1.4376 | 88.7592 | 99.4065 | 6.7482 |
| vdd/vdd | FineSoft_Excess | water | 54.6175 | -6.4529 | 57.5649 | 91.4290 | 13.1058 |
| vdd/vdd | ReuseSoft_Weighted | other | 59.3015 | 2.0590 | 81.3715 | 68.6169 | 43.0529 |
| vdd/vdd | ReuseSoft_Weighted | wall | 58.7569 | 0.3179 | 64.3664 | 87.0835 | 10.7629 |
| vdd/vdd | ReuseSoft_Weighted | road | 23.2917 | 1.9746 | 23.4120 | 97.8407 | 13.8705 |
| vdd/vdd | ReuseSoft_Weighted | vegetation | 45.1745 | -5.2551 | 94.6104 | 46.3677 | 11.2893 |
| vdd/vdd | ReuseSoft_Weighted | vehicle | 47.3339 | 5.1524 | 47.3339 | 100.0000 | 0.7565 |
| vdd/vdd | ReuseSoft_Weighted | roof | 87.2305 | -2.4984 | 87.7002 | 99.3898 | 6.8285 |
| vdd/vdd | ReuseSoft_Weighted | water | 52.9393 | -8.1311 | 55.8674 | 90.9915 | 13.4394 |
| vdd/vdd | ReuseSoft_Excess | other | 59.1030 | 1.8605 | 81.2605 | 68.4298 | 42.9941 |
| vdd/vdd | ReuseSoft_Excess | wall | 58.2782 | -0.1608 | 63.7202 | 87.2184 | 10.8889 |
| vdd/vdd | ReuseSoft_Excess | road | 23.3950 | 2.0779 | 23.5129 | 97.9010 | 13.8195 |
| vdd/vdd | ReuseSoft_Excess | vegetation | 45.1825 | -5.2471 | 94.7432 | 46.3443 | 11.2678 |
| vdd/vdd | ReuseSoft_Excess | vehicle | 46.3723 | 4.1908 | 46.3723 | 100.0000 | 0.7722 |
| vdd/vdd | ReuseSoft_Excess | roof | 87.4313 | -2.2976 | 87.9001 | 99.3938 | 6.8133 |
| vdd/vdd | ReuseSoft_Excess | water | 53.3069 | -7.7635 | 56.1127 | 91.4243 | 13.4443 |
| vdd/vdd | ReuseClassMean_Weighted | other | 59.7959 | 2.5534 | 81.1565 | 69.4364 | 43.6825 |
| vdd/vdd | ReuseClassMean_Weighted | wall | 59.7236 | 1.2846 | 65.8643 | 86.4973 | 10.4473 |
| vdd/vdd | ReuseClassMean_Weighted | road | 23.2292 | 1.9121 | 23.3784 | 97.3249 | 13.8172 |
| vdd/vdd | ReuseClassMean_Weighted | vegetation | 44.1726 | -6.2570 | 94.9050 | 45.2455 | 10.9818 |
| vdd/vdd | ReuseClassMean_Weighted | vehicle | 50.0500 | 7.8685 | 50.0667 | 99.9334 | 0.7148 |
| vdd/vdd | ReuseClassMean_Weighted | roof | 86.2902 | -3.4387 | 86.7419 | 99.4001 | 6.9047 |
| vdd/vdd | ReuseClassMean_Weighted | water | 52.8660 | -8.2044 | 55.7974 | 90.9609 | 13.4517 |
| potsdam/potsdam | Geometry | impervious surface | 48.6704 | -17.5742 | 84.3816 | 53.4889 | 25.9113 |
| potsdam/potsdam | Geometry | building | 72.8222 | 0.2947 | 79.5088 | 89.6472 | 18.1787 |
| potsdam/potsdam | Geometry | low vegetation | 26.5422 | 6.8689 | 77.4031 | 28.7716 | 6.4458 |
| potsdam/potsdam | Geometry | tree | 49.3362 | -8.6028 | 84.7436 | 54.1454 | 11.2731 |
| potsdam/potsdam | Geometry | car | 11.6958 | -13.2124 | 11.6965 | 99.9455 | 30.6388 |
| potsdam/potsdam | Geometry | clutter | 5.1543 | 2.7432 | 7.7773 | 13.2570 | 7.5523 |
| potsdam/potsdam | NoAdmission_Exact | impervious surface | 65.8977 | -0.3469 | 85.6706 | 74.0609 | 35.3370 |
| potsdam/potsdam | NoAdmission_Exact | building | 71.0529 | -1.4746 | 74.9985 | 93.1063 | 20.0156 |
| potsdam/potsdam | NoAdmission_Exact | low vegetation | 14.4124 | -5.2609 | 79.9195 | 14.9539 | 3.2447 |
| potsdam/potsdam | NoAdmission_Exact | tree | 55.5064 | -2.4326 | 91.9038 | 58.3601 | 11.2039 |
| potsdam/potsdam | NoAdmission_Exact | car | 24.5378 | -0.3704 | 24.5936 | 99.0837 | 14.4459 |
| potsdam/potsdam | NoAdmission_Exact | clutter | 2.1143 | -0.2968 | 2.6529 | 9.4321 | 15.7528 |
| potsdam/potsdam | RivalFineHard_Exact | impervious surface | 66.2446 | 0.0000 | 85.2705 | 74.8044 | 35.8593 |
| potsdam/potsdam | RivalFineHard_Exact | building | 72.5275 | 0.0000 | 76.5613 | 93.2276 | 19.6326 |
| potsdam/potsdam | RivalFineHard_Exact | low vegetation | 19.6733 | 0.0000 | 81.8153 | 20.5728 | 4.3604 |
| potsdam/potsdam | RivalFineHard_Exact | tree | 57.9390 | 0.0000 | 92.6969 | 60.7103 | 11.5554 |
| potsdam/potsdam | RivalFineHard_Exact | car | 24.9082 | 0.0000 | 24.9554 | 99.2460 | 14.2598 |
| potsdam/potsdam | RivalFineHard_Exact | clutter | 2.4111 | 0.0000 | 3.0821 | 9.9702 | 14.3326 |
| potsdam/potsdam | FineSoft_Weighted | impervious surface | 66.2343 | -0.0103 | 85.6950 | 74.4677 | 35.5210 |
| potsdam/potsdam | FineSoft_Weighted | building | 71.3028 | -1.2247 | 75.2286 | 93.1803 | 19.9702 |
| potsdam/potsdam | FineSoft_Weighted | low vegetation | 15.6521 | -4.0212 | 80.4218 | 16.2722 | 3.5087 |
| potsdam/potsdam | FineSoft_Weighted | tree | 56.1278 | -1.8112 | 91.9926 | 59.0109 | 11.3179 |
| potsdam/potsdam | FineSoft_Weighted | car | 24.7739 | -0.1343 | 24.8272 | 99.1396 | 14.3180 |
| potsdam/potsdam | FineSoft_Weighted | clutter | 2.1725 | -0.2386 | 2.7395 | 9.4999 | 15.3641 |
| potsdam/potsdam | FineSoft_Excess | impervious surface | 66.1262 | -0.1184 | 85.6380 | 74.3742 | 35.5000 |
| potsdam/potsdam | FineSoft_Excess | building | 71.4924 | -1.0351 | 75.4797 | 93.1193 | 19.8908 |
| potsdam/potsdam | FineSoft_Excess | low vegetation | 15.9378 | -3.7355 | 80.5309 | 16.5766 | 3.5695 |
| potsdam/potsdam | FineSoft_Excess | tree | 56.1246 | -1.8144 | 92.0220 | 58.9952 | 11.3113 |
| potsdam/potsdam | FineSoft_Excess | car | 24.7549 | -0.1533 | 24.8056 | 99.1808 | 14.3365 |
| potsdam/potsdam | FineSoft_Excess | clutter | 2.1805 | -0.2306 | 2.7482 | 9.5472 | 15.3919 |
| potsdam/potsdam | ReuseSoft_Weighted | impervious surface | 65.6324 | -0.6122 | 85.5185 | 73.8389 | 35.2938 |
| potsdam/potsdam | ReuseSoft_Weighted | building | 71.0545 | -1.4730 | 74.9818 | 93.1347 | 20.0262 |
| potsdam/potsdam | ReuseSoft_Weighted | low vegetation | 15.2109 | -4.4624 | 80.1576 | 15.8061 | 3.4194 |
| potsdam/potsdam | ReuseSoft_Weighted | tree | 56.0655 | -1.8735 | 91.8742 | 58.9906 | 11.3286 |
| potsdam/potsdam | ReuseSoft_Weighted | car | 24.3232 | -0.5850 | 24.3748 | 99.1369 | 14.5834 |
| potsdam/potsdam | ReuseSoft_Weighted | clutter | 2.1380 | -0.2731 | 2.6975 | 9.3449 | 15.3487 |
| potsdam/potsdam | ReuseSoft_Excess | impervious surface | 65.5875 | -0.6571 | 85.5036 | 73.7931 | 35.2780 |
| potsdam/potsdam | ReuseSoft_Excess | building | 71.1414 | -1.3861 | 75.1259 | 93.0620 | 19.9721 |
| potsdam/potsdam | ReuseSoft_Excess | low vegetation | 15.3862 | -4.2871 | 80.0435 | 15.9999 | 3.4663 |
| potsdam/potsdam | ReuseSoft_Excess | tree | 55.8903 | -2.0487 | 91.8332 | 58.8136 | 11.2997 |
| potsdam/potsdam | ReuseSoft_Excess | car | 24.2508 | -0.6574 | 24.3005 | 99.1635 | 14.6319 |
| potsdam/potsdam | ReuseSoft_Excess | clutter | 2.1415 | -0.2696 | 2.7016 | 9.3610 | 15.3520 |
| potsdam/potsdam | ReuseClassMean_Weighted | impervious surface | 65.8731 | -0.3715 | 85.6556 | 74.0409 | 35.3337 |
| potsdam/potsdam | ReuseClassMean_Weighted | building | 71.0522 | -1.4753 | 74.9932 | 93.1131 | 20.0185 |
| potsdam/potsdam | ReuseClassMean_Weighted | low vegetation | 14.4708 | -5.2025 | 80.0035 | 15.0139 | 3.2543 |
| potsdam/potsdam | ReuseClassMean_Weighted | tree | 55.5793 | -2.3597 | 91.8798 | 58.4504 | 11.2242 |
| potsdam/potsdam | ReuseClassMean_Weighted | car | 24.5203 | -0.3879 | 24.5758 | 99.0877 | 14.4570 |
| potsdam/potsdam | ReuseClassMean_Weighted | clutter | 2.1184 | -0.2927 | 2.6594 | 9.4310 | 15.7124 |
| udd5/udd5 | Geometry | vegetation | 60.2474 | -7.4488 | 89.6338 | 64.7597 | 2.2305 |
| udd5/udd5 | Geometry | building | 83.4755 | -2.1232 | 86.9865 | 95.3877 | 78.8720 |
| udd5/udd5 | Geometry | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.2018 |
| udd5/udd5 | Geometry | vehicle | 3.7107 | -3.2997 | 3.7107 | 99.9773 | 11.3160 |
| udd5/udd5 | Geometry | other | 5.0713 | -4.8204 | 23.0553 | 6.1044 | 5.3797 |
| udd5/udd5 | NoAdmission_Exact | vegetation | 46.2799 | -21.4163 | 93.8774 | 47.7203 | 1.5693 |
| udd5/udd5 | NoAdmission_Exact | building | 83.4871 | -2.1116 | 83.6594 | 99.7539 | 85.7625 |
| udd5/udd5 | NoAdmission_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 |
| udd5/udd5 | NoAdmission_Exact | vehicle | 5.3645 | -1.6459 | 5.3645 | 99.9773 | 7.8274 |
| udd5/udd5 | NoAdmission_Exact | other | 5.7476 | -4.1441 | 29.7284 | 6.6513 | 4.5458 |
| udd5/udd5 | RivalFineHard_Exact | vegetation | 67.6962 | 0.0000 | 91.0639 | 72.5133 | 2.4583 |
| udd5/udd5 | RivalFineHard_Exact | building | 85.5987 | 0.0000 | 85.6693 | 99.9037 | 83.8762 |
| udd5/udd5 | RivalFineHard_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.6799 |
| udd5/udd5 | RivalFineHard_Exact | vehicle | 7.0104 | 0.0000 | 7.0105 | 99.9773 | 5.9896 |
| udd5/udd5 | RivalFineHard_Exact | other | 9.8917 | 0.0000 | 39.5036 | 11.6576 | 5.9959 |
| udd5/udd5 | FineSoft_Weighted | vegetation | 51.8938 | -15.8024 | 92.5581 | 54.1533 | 1.8063 |
| udd5/udd5 | FineSoft_Weighted | building | 84.2172 | -1.3815 | 84.3594 | 99.8002 | 85.0904 |
| udd5/udd5 | FineSoft_Weighted | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6174 |
| udd5/udd5 | FineSoft_Weighted | vehicle | 5.8517 | -1.1587 | 5.8518 | 99.9773 | 7.1756 |
| udd5/udd5 | FineSoft_Weighted | other | 6.6712 | -3.2205 | 30.1822 | 7.8885 | 5.3104 |
| udd5/udd5 | FineSoft_Excess | vegetation | 51.1281 | -16.5681 | 92.9115 | 53.2034 | 1.7678 |
| udd5/udd5 | FineSoft_Excess | building | 84.1388 | -1.4599 | 84.2778 | 99.8044 | 85.1763 |
| udd5/udd5 | FineSoft_Excess | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5717 |
| udd5/udd5 | FineSoft_Excess | vehicle | 5.7892 | -1.2212 | 5.7893 | 99.9773 | 7.2531 |
| udd5/udd5 | FineSoft_Excess | other | 6.5238 | -3.3679 | 29.9117 | 7.7010 | 5.2310 |
| udd5/udd5 | ReuseSoft_Weighted | vegetation | 47.0246 | -20.6716 | 93.9059 | 48.5049 | 1.5946 |
| udd5/udd5 | ReuseSoft_Weighted | building | 83.7033 | -1.8954 | 83.8691 | 99.7644 | 85.5571 |
| udd5/udd5 | ReuseSoft_Weighted | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3255 |
| udd5/udd5 | ReuseSoft_Weighted | vehicle | 5.4431 | -1.5673 | 5.4431 | 99.9773 | 7.7144 |
| udd5/udd5 | ReuseSoft_Weighted | other | 6.0043 | -3.8874 | 29.5987 | 7.0047 | 4.8084 |
| udd5/udd5 | ReuseSoft_Excess | vegetation | 47.0375 | -20.6587 | 93.8530 | 48.5327 | 1.5965 |
| udd5/udd5 | ReuseSoft_Excess | building | 83.6778 | -1.9209 | 83.8391 | 99.7706 | 85.5930 |
| udd5/udd5 | ReuseSoft_Excess | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3201 |
| udd5/udd5 | ReuseSoft_Excess | vehicle | 5.4306 | -1.5798 | 5.4307 | 99.9773 | 7.7321 |
| udd5/udd5 | ReuseSoft_Excess | other | 5.8764 | -4.0153 | 29.2494 | 6.8501 | 4.7584 |
| udd5/udd5 | ReuseClassMean_Weighted | vegetation | 46.2919 | -21.4043 | 93.8789 | 47.7326 | 1.5697 |
| udd5/udd5 | ReuseClassMean_Weighted | building | 83.4859 | -2.1128 | 83.6582 | 99.7540 | 85.7638 |
| udd5/udd5 | ReuseClassMean_Weighted | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2939 |
| udd5/udd5 | ReuseClassMean_Weighted | vehicle | 5.3811 | -1.6293 | 5.3811 | 99.9773 | 7.8032 |
| udd5/udd5 | ReuseClassMean_Weighted | other | 5.7798 | -4.1119 | 29.7602 | 6.6928 | 4.5693 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9017 |
| oem/oem | Geometry | rangeland | 41.5417 | -8.1516 | 54.4153 | 63.7145 | 17.0320 |
| oem/oem | Geometry | developed space | 25.6977 | 1.8428 | 65.5485 | 29.7106 | 8.8828 |
| oem/oem | Geometry | road | 52.3958 | -2.5267 | 60.6185 | 79.4351 | 8.2388 |
| oem/oem | Geometry | tree | 65.2556 | 8.2719 | 84.8180 | 73.8857 | 23.0134 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0565 |
| oem/oem | Geometry | agriculture land | 71.2323 | -10.7746 | 95.7971 | 73.5302 | 16.2828 |
| oem/oem | Geometry | building | 62.4408 | 11.5772 | 64.4677 | 95.2061 | 17.5919 |
| oem/oem | NoAdmission_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 |
| oem/oem | NoAdmission_Exact | rangeland | 49.5261 | -0.1672 | 70.5348 | 62.4456 | 12.8780 |
| oem/oem | NoAdmission_Exact | developed space | 23.7669 | -0.0880 | 38.9362 | 37.8898 | 19.0710 |
| oem/oem | NoAdmission_Exact | road | 54.5529 | -0.3696 | 68.0810 | 73.3007 | 6.7693 |
| oem/oem | NoAdmission_Exact | tree | 55.6605 | -1.3232 | 91.7802 | 58.5807 | 16.8622 |
| oem/oem | NoAdmission_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | NoAdmission_Exact | agriculture land | 81.6500 | -0.3569 | 82.0066 | 99.4703 | 25.7312 |
| oem/oem | NoAdmission_Exact | building | 47.0287 | -3.8349 | 68.3392 | 60.1297 | 10.4811 |
| oem/oem | RivalFineHard_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4062 |
| oem/oem | RivalFineHard_Exact | rangeland | 49.6933 | 0.0000 | 70.6164 | 62.6472 | 12.9046 |
| oem/oem | RivalFineHard_Exact | developed space | 23.8549 | 0.0000 | 40.6619 | 36.5937 | 17.6369 |
| oem/oem | RivalFineHard_Exact | road | 54.9225 | 0.0000 | 68.1954 | 73.8350 | 6.8072 |
| oem/oem | RivalFineHard_Exact | tree | 56.9837 | 0.0000 | 91.1586 | 60.3173 | 17.4805 |
| oem/oem | RivalFineHard_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| oem/oem | RivalFineHard_Exact | agriculture land | 82.0069 | 0.0000 | 82.3962 | 99.4271 | 25.5984 |
| oem/oem | RivalFineHard_Exact | building | 50.8636 | 0.0000 | 69.6827 | 65.3182 | 11.1660 |
| oem/oem | FineSoft_Weighted | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2565 |
| oem/oem | FineSoft_Weighted | rangeland | 49.5714 | -0.1219 | 70.5678 | 62.4917 | 12.8814 |
| oem/oem | FineSoft_Weighted | developed space | 23.7098 | -0.1451 | 39.3111 | 37.3992 | 18.6445 |
| oem/oem | FineSoft_Weighted | road | 54.6462 | -0.2763 | 68.1141 | 73.4308 | 6.7780 |
| oem/oem | FineSoft_Weighted | tree | 56.0244 | -0.9593 | 91.5985 | 59.0593 | 17.0337 |
| oem/oem | FineSoft_Weighted | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FineSoft_Weighted | agriculture land | 81.6756 | -0.3313 | 82.0448 | 99.4521 | 25.7145 |
| oem/oem | FineSoft_Weighted | building | 48.0558 | -2.8078 | 68.6217 | 61.5897 | 10.6914 |
| oem/oem | FineSoft_Excess | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2881 |
| oem/oem | FineSoft_Excess | rangeland | 49.6666 | -0.0267 | 70.5851 | 62.6293 | 12.9067 |
| oem/oem | FineSoft_Excess | developed space | 23.7078 | -0.1471 | 39.3537 | 37.3557 | 18.6027 |
| oem/oem | FineSoft_Excess | road | 54.7341 | -0.1884 | 68.2198 | 73.4666 | 6.7708 |
| oem/oem | FineSoft_Excess | tree | 56.1658 | -0.8179 | 91.5509 | 59.2363 | 17.0936 |
| oem/oem | FineSoft_Excess | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FineSoft_Excess | agriculture land | 81.8807 | -0.1262 | 82.2548 | 99.4477 | 25.6477 |
| oem/oem | FineSoft_Excess | building | 48.0722 | -2.7914 | 68.6408 | 61.6012 | 10.6905 |
| oem/oem | ReuseSoft_Weighted | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2885 |
| oem/oem | ReuseSoft_Weighted | rangeland | 49.6857 | -0.0076 | 70.5903 | 62.6556 | 12.9111 |
| oem/oem | ReuseSoft_Weighted | developed space | 23.5141 | -0.3408 | 38.8975 | 37.2870 | 18.7862 |
| oem/oem | ReuseSoft_Weighted | road | 54.6003 | -0.3222 | 68.0863 | 73.3802 | 6.7761 |
| oem/oem | ReuseSoft_Weighted | tree | 55.8245 | -1.1592 | 91.6570 | 58.8130 | 16.9518 |
| oem/oem | ReuseSoft_Weighted | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | ReuseSoft_Weighted | agriculture land | 81.6757 | -0.3312 | 82.0429 | 99.4551 | 25.7159 |
| oem/oem | ReuseSoft_Weighted | building | 47.3970 | -3.4666 | 68.3938 | 60.6900 | 10.5703 |
| oem/oem | ReuseSoft_Excess | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2048 |
| oem/oem | ReuseSoft_Excess | rangeland | 49.7629 | 0.0696 | 70.6140 | 62.7596 | 12.9282 |
| oem/oem | ReuseSoft_Excess | developed space | 23.8166 | -0.0383 | 39.2346 | 37.7362 | 18.8492 |
| oem/oem | ReuseSoft_Excess | road | 54.5810 | -0.3415 | 68.0442 | 73.3942 | 6.7816 |
| oem/oem | ReuseSoft_Excess | tree | 55.9684 | -1.0153 | 91.6088 | 58.9927 | 17.0126 |
| oem/oem | ReuseSoft_Excess | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | ReuseSoft_Excess | agriculture land | 81.9409 | -0.0660 | 82.3173 | 99.4451 | 25.6276 |
| oem/oem | ReuseSoft_Excess | building | 47.4756 | -3.3880 | 68.3827 | 60.8277 | 10.5961 |
| oem/oem | ReuseClassMean_Weighted | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.1996 |
| oem/oem | ReuseClassMean_Weighted | rangeland | 49.4933 | -0.2000 | 70.5273 | 62.3991 | 12.8697 |
| oem/oem | ReuseClassMean_Weighted | developed space | 23.7818 | -0.0731 | 38.9661 | 37.8993 | 19.0611 |
| oem/oem | ReuseClassMean_Weighted | road | 54.5505 | -0.3720 | 68.0745 | 73.3038 | 6.7702 |
| oem/oem | ReuseClassMean_Weighted | tree | 55.6506 | -1.3331 | 91.7740 | 58.5722 | 16.8609 |
| oem/oem | ReuseClassMean_Weighted | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | ReuseClassMean_Weighted | agriculture land | 81.5725 | -0.4344 | 81.9279 | 99.4710 | 25.7561 |
| oem/oem | ReuseClassMean_Weighted | building | 47.0362 | -3.8274 | 68.3421 | 60.1396 | 10.4824 |
| loveda/P | Geometry | building | 7.6311 | -46.3404 | 7.6311 | 100.0000 | 0.3472 |
| loveda/P | Geometry | road | 63.0522 | 4.5076 | 64.4734 | 96.6222 | 12.4720 |
| loveda/P | Geometry | water | 63.2131 | 0.6642 | 68.9718 | 88.3328 | 18.8451 |
| loveda/P | Geometry | barren | 1.9144 | -0.4322 | 34.2731 | 1.9874 | 0.6429 |
| loveda/P | Geometry | tree | 72.0056 | 18.7930 | 88.8915 | 79.1256 | 14.6560 |
| loveda/P | Geometry | farm | 89.4231 | 4.5200 | 91.1653 | 97.9076 | 53.0368 |
| loveda/P | NoAdmission_Exact | building | 60.8479 | 6.8764 | 72.1893 | 79.4788 | 0.0292 |
| loveda/P | NoAdmission_Exact | road | 53.7553 | -4.7893 | 54.7280 | 96.7995 | 14.7198 |
| loveda/P | NoAdmission_Exact | water | 62.4091 | -0.1398 | 68.1253 | 88.1486 | 19.0394 |
| loveda/P | NoAdmission_Exact | barren | 4.3215 | 1.9749 | 89.7378 | 4.3430 | 0.5366 |
| loveda/P | NoAdmission_Exact | tree | 39.9553 | -13.2573 | 99.3680 | 40.0571 | 6.6373 |
| loveda/P | NoAdmission_Exact | farm | 82.9502 | -1.9529 | 83.2671 | 99.5433 | 59.0377 |
| loveda/P | RivalFineHard_Exact | building | 53.9715 | 0.0000 | 59.0200 | 86.3192 | 0.0388 |
| loveda/P | RivalFineHard_Exact | road | 58.5446 | 0.0000 | 59.7219 | 96.7425 | 13.4810 |
| loveda/P | RivalFineHard_Exact | water | 62.5489 | 0.0000 | 67.9247 | 88.7680 | 19.2298 |
| loveda/P | RivalFineHard_Exact | barren | 2.3466 | 0.0000 | 65.0543 | 2.3766 | 0.4051 |
| loveda/P | RivalFineHard_Exact | tree | 53.2126 | 0.0000 | 97.3091 | 54.0073 | 9.1381 |
| loveda/P | RivalFineHard_Exact | farm | 84.9031 | 0.0000 | 85.2129 | 99.5737 | 57.7072 |
| loveda/P | FineSoft_Weighted | building | 63.5870 | 9.6155 | 79.3220 | 76.2215 | 0.0255 |
| loveda/P | FineSoft_Weighted | road | 55.2770 | -3.2676 | 56.3166 | 96.7684 | 14.3000 |
| loveda/P | FineSoft_Weighted | water | 62.4474 | -0.1015 | 68.0734 | 88.3122 | 19.0893 |
| loveda/P | FineSoft_Weighted | barren | 4.0199 | 1.6733 | 90.4745 | 4.0370 | 0.4947 |
| loveda/P | FineSoft_Weighted | tree | 43.1347 | -10.0779 | 99.2748 | 43.2710 | 7.1765 |
| loveda/P | FineSoft_Weighted | farm | 83.1316 | -1.7715 | 83.4463 | 99.5484 | 58.9139 |
| loveda/P | FineSoft_Excess | building | 62.7737 | 8.8022 | 71.2707 | 84.0391 | 0.0312 |
| loveda/P | FineSoft_Excess | road | 55.4593 | -3.0853 | 56.5090 | 96.7591 | 14.2500 |
| loveda/P | FineSoft_Excess | water | 62.4654 | -0.0835 | 68.0072 | 88.4600 | 19.1399 |
| loveda/P | FineSoft_Excess | barren | 3.7339 | 1.3873 | 89.8024 | 3.7498 | 0.4630 |
| loveda/P | FineSoft_Excess | tree | 43.5853 | -9.6273 | 99.1107 | 43.7565 | 7.2691 |
| loveda/P | FineSoft_Excess | farm | 83.2249 | -1.6782 | 83.5408 | 99.5477 | 58.8469 |
| loveda/P | ReuseSoft_Weighted | building | 62.8272 | 8.8557 | 76.1905 | 78.1759 | 0.0272 |
| loveda/P | ReuseSoft_Weighted | road | 54.3452 | -4.1994 | 55.3474 | 96.7757 | 14.5515 |
| loveda/P | ReuseSoft_Weighted | water | 62.2408 | -0.3081 | 67.9019 | 88.1873 | 19.1105 |
| loveda/P | ReuseSoft_Weighted | barren | 4.1772 | 1.8306 | 90.3756 | 4.1958 | 0.5147 |
| loveda/P | ReuseSoft_Weighted | tree | 41.5808 | -11.6318 | 99.2220 | 41.7168 | 6.9225 |
| loveda/P | ReuseSoft_Weighted | farm | 83.1832 | -1.7199 | 83.5006 | 99.5451 | 58.8736 |
| loveda/P | ReuseSoft_Excess | building | 65.3012 | 11.3297 | 71.5040 | 88.2736 | 0.0327 |
| loveda/P | ReuseSoft_Excess | road | 54.2587 | -4.2859 | 55.2580 | 96.7746 | 14.5749 |
| loveda/P | ReuseSoft_Excess | water | 62.1681 | -0.3808 | 67.7821 | 88.2436 | 19.1565 |
| loveda/P | ReuseSoft_Excess | barren | 4.1127 | 1.7661 | 90.1784 | 4.1312 | 0.5079 |
| loveda/P | ReuseSoft_Excess | tree | 42.0354 | -11.1772 | 99.0369 | 42.2080 | 7.0170 |
| loveda/P | ReuseSoft_Excess | farm | 83.4093 | -1.4938 | 83.7301 | 99.5428 | 58.7109 |
| loveda/P | ReuseClassMean_Weighted | building | 59.0244 | 5.0529 | 70.1449 | 78.8274 | 0.0298 |
| loveda/P | ReuseClassMean_Weighted | road | 53.7009 | -4.8437 | 54.6716 | 96.7995 | 14.7350 |
| loveda/P | ReuseClassMean_Weighted | water | 62.4068 | -0.1421 | 68.1254 | 88.1439 | 19.0384 |
| loveda/P | ReuseClassMean_Weighted | barren | 4.3013 | 1.9547 | 89.6802 | 4.3227 | 0.5344 |
| loveda/P | ReuseClassMean_Weighted | tree | 39.9206 | -13.2920 | 99.3597 | 40.0236 | 6.6323 |
| loveda/P | ReuseClassMean_Weighted | farm | 82.9608 | -1.9423 | 83.2778 | 99.5433 | 59.0301 |
| loveda/D | Geometry | background | 39.0474 | 13.0739 | 75.8250 | 44.5998 | 26.3232 |
| loveda/D | Geometry | building | 4.8538 | -33.2348 | 4.8538 | 100.0000 | 0.3016 |
| loveda/D | Geometry | road | 45.0691 | -3.5188 | 45.8485 | 96.3650 | 9.6637 |
| loveda/D | Geometry | water | 53.8049 | -0.9039 | 58.0149 | 88.1157 | 12.3473 |
| loveda/D | Geometry | barren | 0.0293 | -1.7472 | 0.8665 | 0.0304 | 0.2146 |
| loveda/D | Geometry | tree | 37.0607 | -8.9288 | 42.3596 | 74.7642 | 16.0550 |
| loveda/D | Geometry | farm | 57.2802 | 11.8961 | 64.7327 | 83.2647 | 35.0945 |
| loveda/D | NoAdmission_Exact | background | 7.3224 | -18.6511 | 75.1732 | 7.5039 | 4.4672 |
| loveda/D | NoAdmission_Exact | building | 31.2821 | -6.8065 | 34.0307 | 79.4788 | 0.0342 |
| loveda/D | NoAdmission_Exact | road | 44.1270 | -4.4609 | 44.7837 | 96.7840 | 9.9365 |
| loveda/D | NoAdmission_Exact | water | 54.4379 | -0.2709 | 58.7421 | 88.1369 | 12.1974 |
| loveda/D | NoAdmission_Exact | barren | 3.2414 | 1.4649 | 82.3610 | 3.2640 | 0.2428 |
| loveda/D | NoAdmission_Exact | tree | 35.8674 | -10.1221 | 97.1427 | 36.2498 | 3.3944 |
| loveda/D | NoAdmission_Exact | farm | 38.8741 | -6.5100 | 38.9455 | 99.5311 | 69.7275 |
| loveda/D | RivalFineHard_Exact | background | 25.9735 | 0.0000 | 85.4376 | 27.1766 | 14.2353 |
| loveda/D | RivalFineHard_Exact | building | 38.0886 | 0.0000 | 39.8551 | 89.5765 | 0.0329 |
| loveda/D | RivalFineHard_Exact | road | 48.5879 | 0.0000 | 49.4050 | 96.7083 | 9.0000 |
| loveda/D | RivalFineHard_Exact | water | 54.7088 | 0.0000 | 58.8346 | 88.6384 | 12.2475 |
| loveda/D | RivalFineHard_Exact | barren | 1.7765 | 0.0000 | 68.4211 | 1.7912 | 0.1604 |
| loveda/D | RivalFineHard_Exact | tree | 45.9895 | 0.0000 | 93.8092 | 47.4290 | 4.5990 |
| loveda/D | RivalFineHard_Exact | farm | 45.3841 | 0.0000 | 45.4771 | 99.5512 | 59.7249 |
| loveda/D | FineSoft_Weighted | background | 10.9815 | -14.9920 | 78.3687 | 11.3247 | 6.4670 |
| loveda/D | FineSoft_Weighted | building | 36.0061 | -2.0825 | 40.2027 | 77.5244 | 0.0282 |
| loveda/D | FineSoft_Weighted | road | 45.8807 | -2.7072 | 46.5995 | 96.7477 | 9.5458 |
| loveda/D | FineSoft_Weighted | water | 54.6114 | -0.0974 | 58.8759 | 88.2899 | 12.1908 |
| loveda/D | FineSoft_Weighted | barren | 2.9354 | 1.1589 | 82.1784 | 2.9542 | 0.2202 |
| loveda/D | FineSoft_Weighted | tree | 39.1504 | -6.8391 | 96.9020 | 39.6467 | 3.7217 |
| loveda/D | FineSoft_Weighted | farm | 39.9639 | -5.4202 | 40.0387 | 99.5349 | 67.8263 |
| loveda/D | FineSoft_Excess | background | 13.0457 | -12.9278 | 79.7614 | 13.4924 | 7.5703 |
| loveda/D | FineSoft_Excess | building | 37.1866 | -0.9020 | 39.3805 | 86.9707 | 0.0323 |
| loveda/D | FineSoft_Excess | road | 45.9942 | -2.5937 | 46.7199 | 96.7331 | 9.5197 |
| loveda/D | FineSoft_Excess | water | 54.5388 | -0.1700 | 58.7280 | 88.4337 | 12.2414 |
| loveda/D | FineSoft_Excess | barren | 2.7944 | 1.0179 | 80.6157 | 2.8133 | 0.2138 |
| loveda/D | FineSoft_Excess | tree | 39.3430 | -6.6465 | 96.5743 | 39.8999 | 3.7582 |
| loveda/D | FineSoft_Excess | farm | 40.6577 | -4.7264 | 40.7355 | 99.5323 | 66.6643 |
| loveda/D | ReuseSoft_Weighted | background | 7.9502 | -18.0233 | 77.9268 | 8.1333 | 4.6709 |
| loveda/D | ReuseSoft_Weighted | building | 32.9252 | -5.1634 | 36.1194 | 78.8274 | 0.0319 |
| loveda/D | ReuseSoft_Weighted | road | 44.4131 | -4.1748 | 45.0835 | 96.7601 | 9.8680 |
| loveda/D | ReuseSoft_Weighted | water | 54.1933 | -0.5155 | 58.4527 | 88.1474 | 12.2592 |
| loveda/D | ReuseSoft_Weighted | barren | 3.0951 | 1.3186 | 82.6347 | 3.1153 | 0.2309 |
| loveda/D | ReuseSoft_Weighted | tree | 38.6936 | -7.2959 | 96.8133 | 39.1927 | 3.6825 |
| loveda/D | ReuseSoft_Weighted | farm | 39.1386 | -6.2455 | 39.2107 | 99.5321 | 69.2565 |
| loveda/D | ReuseSoft_Excess | background | 9.4403 | -16.5332 | 80.1555 | 9.6662 | 5.3968 |
| loveda/D | ReuseSoft_Excess | building | 34.7771 | -3.3115 | 36.3515 | 88.9251 | 0.0358 |
| loveda/D | ReuseSoft_Excess | road | 44.3072 | -4.2807 | 44.9753 | 96.7560 | 9.8913 |
| loveda/D | ReuseSoft_Excess | water | 54.0462 | -0.6626 | 58.2655 | 88.1844 | 12.3038 |
| loveda/D | ReuseSoft_Excess | barren | 3.0602 | 1.2837 | 81.8953 | 3.0811 | 0.2305 |
| loveda/D | ReuseSoft_Excess | tree | 39.0527 | -6.9368 | 96.4076 | 39.6294 | 3.7392 |
| loveda/D | ReuseSoft_Excess | farm | 39.6240 | -5.7601 | 39.6986 | 99.5281 | 68.4026 |
| loveda/D | ReuseClassMean_Weighted | background | 7.3476 | -18.6259 | 75.5718 | 7.5263 | 4.4570 |
| loveda/D | ReuseClassMean_Weighted | building | 31.0655 | -7.0231 | 33.8936 | 78.8274 | 0.0340 |
| loveda/D | ReuseClassMean_Weighted | road | 44.0381 | -4.5498 | 44.6921 | 96.7840 | 9.9569 |
| loveda/D | ReuseClassMean_Weighted | water | 54.4172 | -0.2916 | 58.7201 | 88.1322 | 12.2013 |
| loveda/D | ReuseClassMean_Weighted | barren | 3.2221 | 1.4456 | 82.2902 | 3.2446 | 0.2415 |
| loveda/D | ReuseClassMean_Weighted | tree | 35.9193 | -10.0702 | 97.1130 | 36.3070 | 3.4008 |
| loveda/D | ReuseClassMean_Weighted | farm | 38.8847 | -6.4994 | 38.9561 | 99.5311 | 69.7084 |
| vaihingen/vaihingen | Geometry | impervious surface | 42.0941 | -16.7162 | 77.7719 | 47.8510 | 16.8072 |
| vaihingen/vaihingen | Geometry | building | 72.7601 | 5.6837 | 73.5287 | 98.5836 | 27.6605 |
| vaihingen/vaihingen | Geometry | low vegetation | 53.0114 | 8.0612 | 92.4475 | 55.4111 | 17.7225 |
| vaihingen/vaihingen | Geometry | tree | 68.6405 | 1.1197 | 82.7688 | 80.0845 | 19.9693 |
| vaihingen/vaihingen | Geometry | car | 10.3193 | -16.0461 | 10.3220 | 99.7520 | 17.8406 |
| vaihingen/vaihingen | NoAdmission_Exact | impervious surface | 57.3052 | -1.5051 | 73.8225 | 71.9196 | 26.6125 |
| vaihingen/vaihingen | NoAdmission_Exact | building | 65.7437 | -1.3327 | 65.8760 | 99.6954 | 31.2219 |
| vaihingen/vaihingen | NoAdmission_Exact | low vegetation | 43.7153 | -1.2349 | 96.1772 | 44.4883 | 13.6772 |
| vaihingen/vaihingen | NoAdmission_Exact | tree | 67.2612 | -0.2596 | 79.0950 | 81.8036 | 21.3454 |
| vaihingen/vaihingen | NoAdmission_Exact | car | 25.1095 | -1.2559 | 25.2570 | 97.7270 | 7.1430 |
| vaihingen/vaihingen | RivalFineHard_Exact | impervious surface | 58.8103 | 0.0000 | 74.7374 | 73.4017 | 26.8285 |
| vaihingen/vaihingen | RivalFineHard_Exact | building | 67.0764 | 0.0000 | 67.2196 | 99.6834 | 30.5942 |
| vaihingen/vaihingen | RivalFineHard_Exact | low vegetation | 44.9502 | 0.0000 | 95.9989 | 45.8084 | 14.1092 |
| vaihingen/vaihingen | RivalFineHard_Exact | tree | 67.5208 | 0.0000 | 78.7110 | 82.6067 | 21.6601 |
| vaihingen/vaihingen | RivalFineHard_Exact | car | 26.3654 | 0.0000 | 26.5220 | 97.8096 | 6.8081 |
| vaihingen/vaihingen | FineSoft_Weighted | impervious surface | 57.7843 | -1.0260 | 73.9273 | 72.5745 | 26.8168 |
| vaihingen/vaihingen | FineSoft_Weighted | building | 66.1069 | -0.9695 | 66.2421 | 99.6924 | 31.0484 |
| vaihingen/vaihingen | FineSoft_Weighted | low vegetation | 43.9004 | -1.0498 | 96.2025 | 44.6745 | 13.7308 |
| vaihingen/vaihingen | FineSoft_Weighted | tree | 67.3103 | -0.2105 | 78.9244 | 82.0599 | 21.4585 |
| vaihingen/vaihingen | FineSoft_Weighted | car | 25.8179 | -0.5475 | 25.9742 | 97.7218 | 6.9454 |
| vaihingen/vaihingen | FineSoft_Excess | impervious surface | 57.9936 | -0.8167 | 74.4453 | 72.4081 | 26.5692 |
| vaihingen/vaihingen | FineSoft_Excess | building | 66.2402 | -0.8362 | 66.3711 | 99.7032 | 30.9915 |
| vaihingen/vaihingen | FineSoft_Excess | low vegetation | 44.3694 | -0.5808 | 96.0897 | 45.1853 | 13.9041 |
| vaihingen/vaihingen | FineSoft_Excess | tree | 67.3816 | -0.1392 | 78.9282 | 82.1618 | 21.4841 |
| vaihingen/vaihingen | FineSoft_Excess | car | 25.4660 | -0.8994 | 25.6112 | 97.8225 | 7.0511 |
| vaihingen/vaihingen | ReuseSoft_Weighted | impervious surface | 57.2449 | -1.5654 | 73.6262 | 72.0115 | 26.7176 |
| vaihingen/vaihingen | ReuseSoft_Weighted | building | 66.0249 | -1.0515 | 66.1608 | 99.6898 | 31.0858 |
| vaihingen/vaihingen | ReuseSoft_Weighted | low vegetation | 43.7496 | -1.2006 | 96.1271 | 44.5346 | 13.6985 |
| vaihingen/vaihingen | ReuseSoft_Weighted | tree | 67.3161 | -0.2047 | 79.2155 | 81.7561 | 21.3005 |
| vaihingen/vaihingen | ReuseSoft_Weighted | car | 24.9302 | -1.4352 | 25.0735 | 97.7580 | 7.1976 |
| vaihingen/vaihingen | ReuseSoft_Excess | impervious surface | 57.3673 | -1.4430 | 73.9918 | 71.8571 | 26.5285 |
| vaihingen/vaihingen | ReuseSoft_Excess | building | 66.1194 | -0.9570 | 66.2546 | 99.6921 | 31.0425 |
| vaihingen/vaihingen | ReuseSoft_Excess | low vegetation | 44.1248 | -0.8254 | 96.0601 | 44.9380 | 13.8323 |
| vaihingen/vaihingen | ReuseSoft_Excess | tree | 67.3977 | -0.1231 | 79.2806 | 81.8071 | 21.2963 |
| vaihingen/vaihingen | ReuseSoft_Excess | car | 24.6046 | -1.7608 | 24.7394 | 97.8329 | 7.3004 |
| vaihingen/vaihingen | ReuseClassMean_Weighted | impervious surface | 57.2327 | -1.5776 | 73.6037 | 72.0136 | 26.7265 |
| vaihingen/vaihingen | ReuseClassMean_Weighted | building | 65.8129 | -1.2635 | 65.9463 | 99.6935 | 31.1881 |
| vaihingen/vaihingen | ReuseClassMean_Weighted | low vegetation | 43.5757 | -1.3745 | 96.2058 | 44.3376 | 13.6268 |
| vaihingen/vaihingen | ReuseClassMean_Weighted | tree | 67.2394 | -0.2814 | 79.0736 | 81.7944 | 21.3488 |
| vaihingen/vaihingen | ReuseClassMean_Weighted | car | 25.2218 | -1.1436 | 25.3716 | 97.7141 | 7.1098 |
| landcoverai/landcoverai | Geometry | background | 82.6090 | -5.3738 | 96.6505 | 85.0437 | 59.6226 |
| landcoverai/landcoverai | Geometry | building | 34.9562 | -9.6028 | 35.0436 | 99.2919 | 4.2927 |
| landcoverai/landcoverai | Geometry | woodland | 78.3638 | -2.1933 | 86.4992 | 89.2841 | 22.0960 |
| landcoverai/landcoverai | Geometry | water | 93.6635 | -3.9499 | 93.6635 | 100.0000 | 8.8309 |
| landcoverai/landcoverai | Geometry | road | 14.9318 | -11.7727 | 15.6288 | 77.0019 | 5.1578 |
| landcoverai/landcoverai | NoAdmission_Exact | background | 87.4477 | -0.5351 | 93.8549 | 92.7587 | 66.9686 |
| landcoverai/landcoverai | NoAdmission_Exact | building | 44.4672 | -0.0918 | 44.8124 | 98.2973 | 3.3233 |
| landcoverai/landcoverai | NoAdmission_Exact | woodland | 78.6026 | -1.9545 | 94.9565 | 82.0272 | 18.4920 |
| landcoverai/landcoverai | NoAdmission_Exact | water | 97.6178 | 0.0044 | 97.6178 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | NoAdmission_Exact | road | 26.3947 | -0.3098 | 28.8528 | 75.5990 | 2.7429 |
| landcoverai/landcoverai | RivalFineHard_Exact | background | 87.9828 | 0.0000 | 94.6196 | 92.6163 | 66.3254 |
| landcoverai/landcoverai | RivalFineHard_Exact | building | 44.5590 | 0.0000 | 44.9037 | 98.3067 | 3.3169 |
| landcoverai/landcoverai | RivalFineHard_Exact | woodland | 80.5571 | 0.0000 | 94.4300 | 84.5759 | 19.1729 |
| landcoverai/landcoverai | RivalFineHard_Exact | water | 97.6134 | 0.0000 | 97.6134 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | RivalFineHard_Exact | road | 26.7045 | 0.0000 | 29.2139 | 75.6627 | 2.7113 |
| landcoverai/landcoverai | FineSoft_Weighted | background | 87.6603 | -0.3225 | 94.1286 | 92.7308 | 66.7538 |
| landcoverai/landcoverai | FineSoft_Weighted | building | 44.5271 | -0.0319 | 44.8719 | 98.3036 | 3.3191 |
| landcoverai/landcoverai | FineSoft_Weighted | woodland | 79.3220 | -1.2351 | 94.7856 | 82.9413 | 18.7318 |
| landcoverai/landcoverai | FineSoft_Weighted | water | 97.6134 | 0.0000 | 97.6134 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | FineSoft_Weighted | road | 26.5930 | -0.1115 | 29.0865 | 75.6218 | 2.7217 |
| landcoverai/landcoverai | FineSoft_Excess | background | 87.6590 | -0.3238 | 94.1490 | 92.7094 | 66.7239 |
| landcoverai/landcoverai | FineSoft_Excess | building | 44.4910 | -0.0680 | 44.8352 | 98.3036 | 3.3218 |
| landcoverai/landcoverai | FineSoft_Excess | woodland | 79.3522 | -1.2049 | 94.7352 | 83.0131 | 18.7580 |
| landcoverai/landcoverai | FineSoft_Excess | water | 97.6145 | 0.0011 | 97.6145 | 100.0000 | 8.4734 |
| landcoverai/landcoverai | FineSoft_Excess | road | 26.5791 | -0.1254 | 29.0713 | 75.6126 | 2.7228 |
| landcoverai/landcoverai | ReuseSoft_Weighted | background | 87.4533 | -0.5295 | 93.9887 | 92.6347 | 66.7839 |
| landcoverai/landcoverai | ReuseSoft_Weighted | building | 44.0531 | -0.5059 | 44.3900 | 98.3067 | 3.3553 |
| landcoverai/landcoverai | ReuseSoft_Weighted | woodland | 78.9839 | -1.5732 | 94.8919 | 82.4912 | 18.6093 |
| landcoverai/landcoverai | ReuseSoft_Weighted | water | 97.5690 | -0.0444 | 97.5690 | 100.0000 | 8.4774 |
| landcoverai/landcoverai | ReuseSoft_Weighted | road | 26.1321 | -0.5724 | 28.5361 | 75.6218 | 2.7742 |
| landcoverai/landcoverai | ReuseSoft_Excess | background | 87.4855 | -0.4973 | 94.0828 | 92.5795 | 66.6772 |
| landcoverai/landcoverai | ReuseSoft_Excess | building | 43.9594 | -0.5996 | 44.2884 | 98.3382 | 3.3640 |
| landcoverai/landcoverai | ReuseSoft_Excess | woodland | 79.2443 | -1.3128 | 94.8474 | 82.8092 | 18.6898 |
| landcoverai/landcoverai | ReuseSoft_Excess | water | 97.5563 | -0.0571 | 97.5563 | 100.0000 | 8.4785 |
| landcoverai/landcoverai | ReuseSoft_Excess | road | 25.9989 | -0.7056 | 28.3754 | 75.6354 | 2.7904 |
| landcoverai/landcoverai | ReuseClassMean_Weighted | background | 87.4587 | -0.5241 | 93.8825 | 92.7441 | 66.9384 |
| landcoverai/landcoverai | ReuseClassMean_Weighted | building | 44.4492 | -0.1098 | 44.7928 | 98.3036 | 3.3250 |
| landcoverai/landcoverai | ReuseClassMean_Weighted | woodland | 78.6792 | -1.8779 | 94.9438 | 82.1200 | 18.5154 |
| landcoverai/landcoverai | ReuseClassMean_Weighted | water | 97.6140 | 0.0006 | 97.6140 | 100.0000 | 8.4735 |
| landcoverai/landcoverai | ReuseClassMean_Weighted | road | 26.3544 | -0.3501 | 28.8040 | 75.6035 | 2.7477 |
| flair1/flair1 | Geometry | building | 49.6979 | -7.4773 | 50.7258 | 96.0825 | 13.5991 |
| flair1/flair1 | Geometry | pervious surface | 57.0621 | 10.4347 | 92.1299 | 59.9861 | 11.1728 |
| flair1/flair1 | Geometry | impervious surface | 52.6509 | 0.0763 | 62.6245 | 76.7765 | 20.0846 |
| flair1/flair1 | Geometry | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4411 |
| flair1/flair1 | Geometry | water | 73.2328 | 9.5450 | 77.1432 | 93.5263 | 5.3781 |
| flair1/flair1 | Geometry | coniferous | 43.0233 | 0.2462 | 61.7114 | 58.6897 | 0.5346 |
| flair1/flair1 | Geometry | deciduous | 54.0229 | -6.3579 | 78.9723 | 63.0995 | 13.6004 |
| flair1/flair1 | Geometry | brushwood | 18.6136 | 3.7301 | 23.4212 | 47.5559 | 9.9866 |
| flair1/flair1 | Geometry | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | Geometry | herbaceous vegetation | 60.2329 | 2.7084 | 94.3169 | 62.5013 | 21.2289 |
| flair1/flair1 | Geometry | agricultural land | 18.7339 | 5.6164 | 21.6468 | 58.1977 | 0.8198 |
| flair1/flair1 | Geometry | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1540 |
| flair1/flair1 | NoAdmission_Exact | building | 56.7755 | -0.3997 | 58.0396 | 96.3057 | 11.9131 |
| flair1/flair1 | NoAdmission_Exact | pervious surface | 47.2753 | 0.6479 | 91.6998 | 49.3887 | 9.2421 |
| flair1/flair1 | NoAdmission_Exact | impervious surface | 51.9858 | -0.5888 | 59.7781 | 79.9522 | 21.9112 |
| flair1/flair1 | NoAdmission_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 |
| flair1/flair1 | NoAdmission_Exact | water | 61.4565 | -2.2313 | 63.2502 | 95.5889 | 6.7041 |
| flair1/flair1 | NoAdmission_Exact | coniferous | 43.5282 | 0.7511 | 65.7321 | 56.3052 | 0.4815 |
| flair1/flair1 | NoAdmission_Exact | deciduous | 59.5237 | -0.8571 | 78.8015 | 70.8722 | 15.3089 |
| flair1/flair1 | NoAdmission_Exact | brushwood | 12.3287 | -2.5548 | 26.2213 | 18.8771 | 3.5408 |
| flair1/flair1 | NoAdmission_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 |
| flair1/flair1 | NoAdmission_Exact | herbaceous vegetation | 57.4725 | -0.0520 | 90.8252 | 61.0148 | 21.5207 |
| flair1/flair1 | NoAdmission_Exact | agricultural land | 12.9546 | -0.1629 | 13.9145 | 65.2534 | 1.4300 |
| flair1/flair1 | NoAdmission_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 |
| flair1/flair1 | RivalFineHard_Exact | building | 57.1752 | 0.0000 | 58.4382 | 96.3576 | 11.8382 |
| flair1/flair1 | RivalFineHard_Exact | pervious surface | 46.6274 | 0.0000 | 91.7085 | 48.6795 | 9.1085 |
| flair1/flair1 | RivalFineHard_Exact | impervious surface | 52.5746 | 0.0000 | 60.4793 | 80.0896 | 21.6944 |
| flair1/flair1 | RivalFineHard_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1744 |
| flair1/flair1 | RivalFineHard_Exact | water | 63.6878 | 0.0000 | 65.7363 | 95.3351 | 6.4334 |
| flair1/flair1 | RivalFineHard_Exact | coniferous | 42.7771 | 0.0000 | 63.1055 | 57.0434 | 0.5081 |
| flair1/flair1 | RivalFineHard_Exact | deciduous | 60.3808 | 0.0000 | 78.8158 | 72.0787 | 15.5666 |
| flair1/flair1 | RivalFineHard_Exact | brushwood | 14.8835 | 0.0000 | 28.7712 | 23.5675 | 4.0288 |
| flair1/flair1 | RivalFineHard_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0207 |
| flair1/flair1 | RivalFineHard_Exact | herbaceous vegetation | 57.5245 | 0.0000 | 91.4034 | 60.8148 | 21.3145 |
| flair1/flair1 | RivalFineHard_Exact | agricultural land | 13.1175 | 0.0000 | 14.1254 | 64.7685 | 1.3981 |
| flair1/flair1 | RivalFineHard_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9143 |
| flair1/flair1 | FineSoft_Weighted | building | 56.9729 | -0.2023 | 58.2374 | 96.3290 | 11.8755 |
| flair1/flair1 | FineSoft_Weighted | pervious surface | 47.0355 | 0.4081 | 91.7307 | 49.1182 | 9.1884 |
| flair1/flair1 | FineSoft_Weighted | impervious surface | 52.2195 | -0.3551 | 60.0419 | 80.0326 | 21.8368 |
| flair1/flair1 | FineSoft_Weighted | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1578 |
| flair1/flair1 | FineSoft_Weighted | water | 62.0885 | -1.5993 | 63.9527 | 95.5157 | 6.6254 |
| flair1/flair1 | FineSoft_Weighted | coniferous | 43.1628 | 0.3857 | 64.6562 | 56.4919 | 0.4912 |
| flair1/flair1 | FineSoft_Weighted | deciduous | 59.8918 | -0.4890 | 78.8370 | 71.3654 | 15.4085 |
| flair1/flair1 | FineSoft_Weighted | brushwood | 13.1638 | -1.7197 | 27.1613 | 20.3464 | 3.6844 |
| flair1/flair1 | FineSoft_Weighted | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0034 |
| flair1/flair1 | FineSoft_Weighted | herbaceous vegetation | 57.5387 | 0.0142 | 90.9814 | 61.0190 | 21.4852 |
| flair1/flair1 | FineSoft_Weighted | agricultural land | 13.0011 | -0.1164 | 13.9774 | 65.0501 | 1.4191 |
| flair1/flair1 | FineSoft_Weighted | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8245 |
| flair1/flair1 | FineSoft_Excess | building | 56.9773 | -0.1979 | 58.2417 | 96.3297 | 11.8747 |
| flair1/flair1 | FineSoft_Excess | pervious surface | 47.1631 | 0.5357 | 91.8762 | 49.2155 | 9.1920 |
| flair1/flair1 | FineSoft_Excess | impervious surface | 52.3670 | -0.2076 | 60.2217 | 80.0596 | 21.7790 |
| flair1/flair1 | FineSoft_Excess | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1192 |
| flair1/flair1 | FineSoft_Excess | water | 62.4524 | -1.2354 | 64.3441 | 95.5039 | 6.5842 |
| flair1/flair1 | FineSoft_Excess | coniferous | 43.0924 | 0.3153 | 64.3332 | 56.6191 | 0.4947 |
| flair1/flair1 | FineSoft_Excess | deciduous | 59.9828 | -0.3980 | 78.7821 | 71.5400 | 15.4569 |
| flair1/flair1 | FineSoft_Excess | brushwood | 13.4179 | -1.4656 | 27.4155 | 20.8110 | 3.7335 |
| flair1/flair1 | FineSoft_Excess | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0068 |
| flair1/flair1 | FineSoft_Excess | herbaceous vegetation | 57.6262 | 0.1017 | 90.9933 | 61.1120 | 21.5151 |
| flair1/flair1 | FineSoft_Excess | agricultural land | 13.1672 | 0.0497 | 14.1726 | 64.9875 | 1.3982 |
| flair1/flair1 | FineSoft_Excess | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8455 |
| flair1/flair1 | ReuseSoft_Weighted | building | 56.7938 | -0.3814 | 58.0428 | 96.3496 | 11.9178 |
| flair1/flair1 | ReuseSoft_Weighted | pervious surface | 47.2257 | 0.5983 | 91.9830 | 49.2530 | 9.1883 |
| flair1/flair1 | ReuseSoft_Weighted | impervious surface | 52.0485 | -0.5261 | 59.8622 | 79.9501 | 21.8799 |
| flair1/flair1 | ReuseSoft_Weighted | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1646 |
| flair1/flair1 | ReuseSoft_Weighted | water | 61.7191 | -1.9687 | 63.5404 | 95.5620 | 6.6716 |
| flair1/flair1 | ReuseSoft_Weighted | coniferous | 43.5176 | 0.7405 | 65.6965 | 56.3136 | 0.4818 |
| flair1/flair1 | ReuseSoft_Weighted | deciduous | 59.5248 | -0.8560 | 79.0522 | 70.6721 | 15.2172 |
| flair1/flair1 | ReuseSoft_Weighted | brushwood | 13.2869 | -1.5966 | 27.1509 | 20.6481 | 3.7404 |
| flair1/flair1 | ReuseSoft_Weighted | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0021 |
| flair1/flair1 | ReuseSoft_Weighted | herbaceous vegetation | 57.5967 | 0.0722 | 90.8839 | 61.1283 | 21.5468 |
| flair1/flair1 | ReuseSoft_Weighted | agricultural land | 13.1401 | 0.0226 | 14.1294 | 65.2378 | 1.4079 |
| flair1/flair1 | ReuseSoft_Weighted | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7815 |
| flair1/flair1 | ReuseSoft_Excess | building | 56.7026 | -0.4726 | 57.9597 | 96.3157 | 11.9307 |
| flair1/flair1 | ReuseSoft_Excess | pervious surface | 47.4103 | 0.7829 | 92.0190 | 49.4435 | 9.2203 |
| flair1/flair1 | ReuseSoft_Excess | impervious surface | 52.0356 | -0.5390 | 59.8408 | 79.9577 | 21.8898 |
| flair1/flair1 | ReuseSoft_Excess | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0992 |
| flair1/flair1 | ReuseSoft_Excess | water | 62.0349 | -1.6529 | 63.8838 | 95.5426 | 6.6344 |
| flair1/flair1 | ReuseSoft_Excess | coniferous | 43.5025 | 0.7254 | 65.5584 | 56.3900 | 0.4835 |
| flair1/flair1 | ReuseSoft_Excess | deciduous | 59.6030 | -0.7778 | 78.9944 | 70.8288 | 15.2621 |
| flair1/flair1 | ReuseSoft_Excess | brushwood | 13.5283 | -1.3552 | 27.5227 | 21.0147 | 3.7554 |
| flair1/flair1 | ReuseSoft_Excess | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| flair1/flair1 | ReuseSoft_Excess | herbaceous vegetation | 57.6827 | 0.1582 | 90.9142 | 61.2113 | 21.5689 |
| flair1/flair1 | ReuseSoft_Excess | agricultural land | 13.2089 | 0.0914 | 14.2269 | 64.8623 | 1.3902 |
| flair1/flair1 | ReuseSoft_Excess | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7649 |
| flair1/flair1 | ReuseClassMean_Weighted | building | 56.7852 | -0.3900 | 58.0489 | 96.3077 | 11.9114 |
| flair1/flair1 | ReuseClassMean_Weighted | pervious surface | 47.2743 | 0.6469 | 91.7046 | 49.3862 | 9.2412 |
| flair1/flair1 | ReuseClassMean_Weighted | impervious surface | 51.9655 | -0.6091 | 59.7474 | 79.9589 | 21.9243 |
| flair1/flair1 | ReuseClassMean_Weighted | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1502 |
| flair1/flair1 | ReuseClassMean_Weighted | water | 61.4790 | -2.2088 | 63.2741 | 95.5889 | 6.7015 |
| flair1/flair1 | ReuseClassMean_Weighted | coniferous | 43.5225 | 0.7454 | 65.7191 | 56.3052 | 0.4816 |
| flair1/flair1 | ReuseClassMean_Weighted | deciduous | 59.5088 | -0.8720 | 78.7843 | 70.8649 | 15.3106 |
| flair1/flair1 | ReuseClassMean_Weighted | brushwood | 12.2831 | -2.6004 | 26.1833 | 18.7898 | 3.5296 |
| flair1/flair1 | ReuseClassMean_Weighted | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0019 |
| flair1/flair1 | ReuseClassMean_Weighted | herbaceous vegetation | 57.4552 | -0.0693 | 90.8398 | 60.9887 | 21.5080 |
| flair1/flair1 | ReuseClassMean_Weighted | agricultural land | 12.9172 | -0.2003 | 13.8593 | 65.5194 | 1.4415 |
| flair1/flair1 | ReuseClassMean_Weighted | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7982 |

Raw results retain beneficial/harmful/wrong-to-wrong transitions, per-image confusions, all diagnostics, forward counts and stage samples. Timing excludes model loading, text encoding, decoding, score persistence and GT.
