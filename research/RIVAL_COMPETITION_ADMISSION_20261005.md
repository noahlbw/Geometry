# Frozen alias sources: competition-conditioned action

Same64 developed top-left512 windows, eight/domain; full-image wide context. NOT full eight-dataset evaluation. Geometry, finite VIP wide observer,20 aliases/class, original fine witnesses/risk and reconstruction remain fixed. Scores persist before masks. No threshold/strength/domain tuning. Source scores and all three original per-image controls replay exactly. LoveDA D counts once; P is separate. Common scored-class denominator across all nine arms.

Primary posterior solver uses actual unchanged coupled probabilities to weight rival actions. Projection alternatives reject actions opposed to the same fine class evidence and clip aligned magnitude. The candidate still uses the original fine forwards; zero *additional* visual forwards does not mean no fine observation.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalPosterior_Exact | FineActionProjected_Exact | FineRivalProjected_Exact | FineTopTwo_Exact | FinePosteriorStrengthMatched_Exact | FineBudgetOnly_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.4695 | 53.7470 | 54.3441 | 54.6828 | 54.1523 | 54.1183 | 51.8446 | 54.3642 | 54.1333 |
| potsdam/potsdam | 35.7035 | 38.9203 | 40.6173 | 41.0160 | 40.7623 | 41.1231 | 41.9379 | 40.8274 | 40.9475 |
| udd5/udd5 | 30.5010 | 28.1758 | 34.0394 | 34.1825 | 34.4520 | 34.6174 | 30.8653 | 34.4171 | 34.6906 |
| oem/oem | 39.8205 | 39.0232 | 39.7906 | 40.2494 | 39.9717 | 40.4484 | 40.4892 | 40.0239 | 40.1611 |
| loveda/P | 49.5399 | 50.7066 | 52.5879 | 52.8314 | 52.5632 | 52.6436 | 50.1185 | 52.5781 | 52.0588 |
| loveda/D | 33.8779 | 30.7360 | 37.2156 | 37.4683 | 36.9750 | 37.6023 | 36.6930 | 37.0468 | 37.2513 |
| vaihingen/vaihingen | 49.3651 | 51.8270 | 52.9446 | 53.1436 | 53.1363 | 53.3059 | 53.2107 | 52.9310 | 53.3440 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.7464 | 67.5608 | 67.7836 | 68.1473 | 67.6297 | 67.6095 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 34.1776 | 34.2746 | 34.3507 | 34.0159 | 34.0569 | 34.4546 |
| Eight-domain mean | 40.5310 | 42.8679 | 45.0622 | 45.3333 | 45.1606 | 45.4187 | 44.6505 | 45.1621 | 45.3240 |

## Frozen Advancement Gate

| Candidate | Mean delta vs hard, pp | Worst protocol delta, pp | Advance to complete-image pilot |
| --- | ---: | ---: | --- |
| FineRivalPosterior_Exact | +0.2712 | +0.1152 | True |
| FineActionProjected_Exact | +0.0985 | -0.2406 | True |
| FineRivalProjected_Exact | +0.3566 | -0.2258 | True |

Top-two, strength-matched and fine-budget-only arms are diagnostics, not promotable alias candidates. Candidate choice on these windows is development selection, not untouched independent validation. Every implementation/output is preserved. No full20k rollout or retained-model replacement is automatic.

## Independent Window-Context Cost

Same-process warmed synchronized median of three alternated executions; resident backbones included. One local512 window plus full-image wide context, not complete-image throughput. Loading/text encoding/GT excluded.

| Dataset | Cached hard s | Posterior s | Projection s | Posterior+projection s |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.315413 | 0.315798 | 0.317384 | 0.318737 |
| potsdam | 0.284438 | 0.285422 | 0.287228 | 0.288498 |
| udd5 | 0.291938 | 0.293232 | 0.295022 | 0.295136 |
| oem | 0.283511 | 0.284708 | 0.284313 | 0.284763 |
| loveda | 0.325480 | 0.326997 | 0.328425 | 0.330943 |
| vaihingen | 0.274364 | 0.274334 | 0.275253 | 0.278819 |
| landcoverai | 0.269581 | 0.271666 | 0.271894 | 0.273227 |
| flair1 | 0.280339 | 0.281520 | 0.282140 | 0.283853 |

## Peak Allocated Memory

| Dataset | Cached hard MiB | Posterior MiB | Projection MiB | Posterior+projection MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 5565.781 | 5565.781 | 5565.781 | 5565.781 |
| potsdam | 5563.841 | 5563.841 | 5563.841 | 5563.841 |
| udd5 | 5556.792 | 5556.792 | 5556.792 | 5556.792 |
| oem | 5573.461 | 5573.461 | 5573.461 | 5573.461 |
| loveda | 5594.114 | 5594.114 | 5594.114 | 5594.114 |
| vaihingen | 5559.599 | 5559.599 | 5559.599 | 5559.599 |
| landcoverai | 5559.599 | 5559.599 | 5559.599 | 5559.599 |
| flair1 | 5797.144 | 5797.144 | 5797.144 | 5797.144 |

## Pixel Changes Against No Admission

| Dataset/protocol | Method | Corrected | Broken | Wrong-to-wrong |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | RivalFineHard_Exact | 49565 | 46562 | 20548 |
| vdd/vdd | FineRivalPosterior_Exact | 74699 | 68939 | 24280 |
| vdd/vdd | FineActionProjected_Exact | 57016 | 51487 | 22908 |
| vdd/vdd | FineRivalProjected_Exact | 77937 | 79727 | 25381 |
| vdd/vdd | FineTopTwo_Exact | 66399 | 133587 | 30822 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | 62975 | 60361 | 22218 |
| vdd/vdd | FineBudgetOnly_Exact | 69103 | 63036 | 25286 |
| potsdam/potsdam | RivalFineHard_Exact | 43913 | 7377 | 20577 |
| potsdam/potsdam | FineRivalPosterior_Exact | 54926 | 8725 | 24863 |
| potsdam/potsdam | FineActionProjected_Exact | 47891 | 9247 | 22815 |
| potsdam/potsdam | FineRivalProjected_Exact | 58218 | 10535 | 27184 |
| potsdam/potsdam | FineTopTwo_Exact | 76815 | 8078 | 26529 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | 49749 | 8099 | 22772 |
| potsdam/potsdam | FineBudgetOnly_Exact | 53248 | 11437 | 27392 |
| udd5/udd5 | RivalFineHard_Exact | 45375 | 5731 | 73932 |
| udd5/udd5 | FineRivalPosterior_Exact | 48408 | 8458 | 83873 |
| udd5/udd5 | FineActionProjected_Exact | 51809 | 3892 | 62371 |
| udd5/udd5 | FineRivalProjected_Exact | 55139 | 6727 | 71503 |
| udd5/udd5 | FineTopTwo_Exact | 31215 | 6603 | 57769 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | 48484 | 6549 | 80742 |
| udd5/udd5 | FineBudgetOnly_Exact | 56165 | 3980 | 60785 |
| oem/oem | RivalFineHard_Exact | 25549 | 7646 | 5648 |
| oem/oem | FineRivalPosterior_Exact | 40041 | 13834 | 7451 |
| oem/oem | FineActionProjected_Exact | 30545 | 9045 | 7179 |
| oem/oem | FineRivalProjected_Exact | 45692 | 14908 | 9073 |
| oem/oem | FineTopTwo_Exact | 57642 | 24821 | 12938 |
| oem/oem | FinePosteriorStrengthMatched_Exact | 32091 | 8475 | 6194 |
| oem/oem | FineBudgetOnly_Exact | 36028 | 10882 | 9634 |
| loveda/P | RivalFineHard_Exact | 27988 | 2706 | 6329 |
| loveda/P | FineRivalPosterior_Exact | 32128 | 3348 | 9098 |
| loveda/P | FineActionProjected_Exact | 32880 | 3095 | 7648 |
| loveda/P | FineRivalProjected_Exact | 36129 | 3715 | 10067 |
| loveda/P | FineTopTwo_Exact | 25873 | 3854 | 9339 |
| loveda/P | FinePosteriorStrengthMatched_Exact | 27323 | 2806 | 7475 |
| loveda/P | FineBudgetOnly_Exact | 37422 | 3514 | 9174 |
| loveda/D | RivalFineHard_Exact | 207690 | 2693 | 35269 |
| loveda/D | FineRivalPosterior_Exact | 254893 | 13779 | 38454 |
| loveda/D | FineActionProjected_Exact | 223720 | 2990 | 38414 |
| loveda/D | FineRivalProjected_Exact | 264602 | 14333 | 40749 |
| loveda/D | FineTopTwo_Exact | 328270 | 52790 | 38927 |
| loveda/D | FinePosteriorStrengthMatched_Exact | 227614 | 2823 | 35612 |
| loveda/D | FineBudgetOnly_Exact | 244417 | 3417 | 41654 |
| vaihingen/vaihingen | RivalFineHard_Exact | 24562 | 4429 | 13931 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | 28755 | 5151 | 14452 |
| vaihingen/vaihingen | FineActionProjected_Exact | 28894 | 4251 | 13573 |
| vaihingen/vaihingen | FineRivalProjected_Exact | 32344 | 4898 | 14250 |
| vaihingen/vaihingen | FineTopTwo_Exact | 31646 | 6784 | 11879 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | 24215 | 4416 | 13451 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | 34971 | 5718 | 15398 |
| landcoverai/landcoverai | RivalFineHard_Exact | 12707 | 3271 | 86 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | 19125 | 5617 | 141 |
| landcoverai/landcoverai | FineActionProjected_Exact | 14679 | 4290 | 107 |
| landcoverai/landcoverai | FineRivalProjected_Exact | 20626 | 6564 | 140 |
| landcoverai/landcoverai | FineTopTwo_Exact | 29654 | 9353 | 238 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | 15515 | 3691 | 86 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | 16533 | 5578 | 145 |
| flair1/flair1 | RivalFineHard_Exact | 14251 | 8634 | 11338 |
| flair1/flair1 | FineRivalPosterior_Exact | 18395 | 10462 | 12820 |
| flair1/flair1 | FineActionProjected_Exact | 18103 | 8187 | 13960 |
| flair1/flair1 | FineRivalProjected_Exact | 21691 | 10348 | 15217 |
| flair1/flair1 | FineTopTwo_Exact | 24632 | 12112 | 9862 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | 14374 | 8860 | 11477 |
| flair1/flair1 | FineBudgetOnly_Exact | 22819 | 9103 | 17162 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs hard | Precision | Recall | Area |
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
| vdd/vdd | FineRivalPosterior_Exact | other | 56.0157 | -1.2268 | 81.1309 | 64.4066 | 40.5310 |
| vdd/vdd | FineRivalPosterior_Exact | wall | 55.5750 | -2.8640 | 59.8997 | 88.5023 | 11.7539 |
| vdd/vdd | FineRivalPosterior_Exact | road | 21.3628 | 0.0457 | 21.4093 | 98.9929 | 15.3466 |
| vdd/vdd | FineRivalPosterior_Exact | vegetation | 53.5435 | 3.1139 | 94.8794 | 55.1368 | 13.3862 |
| vdd/vdd | FineRivalPosterior_Exact | vehicle | 40.5354 | -1.6461 | 40.5354 | 100.0000 | 0.8834 |
| vdd/vdd | FineRivalPosterior_Exact | roof | 89.7776 | 0.0487 | 90.2080 | 99.4714 | 6.6442 |
| vdd/vdd | FineRivalPosterior_Exact | water | 65.9699 | 4.8995 | 68.3813 | 94.9257 | 11.4547 |
| vdd/vdd | FineActionProjected_Exact | other | 56.8559 | -0.3866 | 80.9404 | 65.6446 | 41.4073 |
| vdd/vdd | FineActionProjected_Exact | wall | 56.1560 | -2.2830 | 60.6340 | 88.3770 | 11.5951 |
| vdd/vdd | FineActionProjected_Exact | road | 21.9292 | 0.6121 | 21.9840 | 98.8765 | 14.9279 |
| vdd/vdd | FineActionProjected_Exact | vegetation | 51.6139 | 1.1843 | 95.3873 | 52.9351 | 12.7833 |
| vdd/vdd | FineActionProjected_Exact | vehicle | 39.7818 | -2.3997 | 39.7818 | 100.0000 | 0.9002 |
| vdd/vdd | FineActionProjected_Exact | roof | 90.0549 | 0.3260 | 90.5050 | 99.4508 | 6.6210 |
| vdd/vdd | FineActionProjected_Exact | water | 62.6744 | 1.6040 | 65.5489 | 93.4608 | 11.7652 |
| vdd/vdd | FineRivalProjected_Exact | other | 55.0600 | -2.1825 | 80.8961 | 63.2893 | 39.9435 |
| vdd/vdd | FineRivalProjected_Exact | wall | 53.1449 | -5.2941 | 57.0418 | 88.6096 | 12.3578 |
| vdd/vdd | FineRivalProjected_Exact | road | 21.6618 | 0.3447 | 21.7082 | 99.0245 | 15.1402 |
| vdd/vdd | FineRivalProjected_Exact | vegetation | 54.2677 | 3.8381 | 94.7736 | 55.9419 | 13.5969 |
| vdd/vdd | FineRivalProjected_Exact | vehicle | 38.2597 | -3.9218 | 38.2597 | 100.0000 | 0.9360 |
| vdd/vdd | FineRivalProjected_Exact | roof | 89.8815 | 0.1526 | 90.3136 | 99.4706 | 6.6363 |
| vdd/vdd | FineRivalProjected_Exact | water | 66.5525 | 5.4821 | 68.9089 | 95.1129 | 11.3894 |
| vdd/vdd | FineTopTwo_Exact | other | 50.6344 | -6.6081 | 77.7021 | 59.2426 | 38.9265 |
| vdd/vdd | FineTopTwo_Exact | wall | 46.9412 | -11.4978 | 49.7856 | 89.1496 | 14.2452 |
| vdd/vdd | FineTopTwo_Exact | road | 21.5050 | 0.1879 | 21.5710 | 98.5978 | 15.1708 |
| vdd/vdd | FineTopTwo_Exact | vegetation | 49.0904 | -1.3392 | 93.5842 | 50.8001 | 12.5041 |
| vdd/vdd | FineTopTwo_Exact | vehicle | 36.4953 | -5.6862 | 36.4953 | 100.0000 | 0.9812 |
| vdd/vdd | FineTopTwo_Exact | roof | 91.8711 | 2.1422 | 92.3417 | 99.4484 | 6.4891 |
| vdd/vdd | FineTopTwo_Exact | water | 66.3743 | 5.3039 | 68.0712 | 96.3802 | 11.6832 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | other | 56.4014 | -0.8411 | 81.2632 | 64.8325 | 40.7327 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | wall | 56.8454 | -1.5936 | 61.5319 | 88.1846 | 11.4010 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | road | 20.8674 | -0.4497 | 20.9216 | 98.7731 | 15.6694 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | vegetation | 53.0077 | 2.5781 | 95.4517 | 54.3813 | 13.1237 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | vehicle | 40.9398 | -1.2417 | 40.9398 | 100.0000 | 0.8747 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | roof | 88.9192 | -0.8097 | 89.3599 | 99.4484 | 6.7057 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | water | 63.5683 | 2.4979 | 66.7665 | 92.9927 | 11.4928 |
| vdd/vdd | FineBudgetOnly_Exact | other | 56.2592 | -0.9833 | 80.7432 | 64.9776 | 41.0867 |
| vdd/vdd | FineBudgetOnly_Exact | wall | 54.3088 | -4.1302 | 58.4429 | 88.4759 | 12.0433 |
| vdd/vdd | FineBudgetOnly_Exact | road | 22.3716 | 1.0545 | 22.4247 | 98.9527 | 14.6457 |
| vdd/vdd | FineBudgetOnly_Exact | vegetation | 52.7891 | 2.3595 | 95.4959 | 54.1370 | 13.0587 |
| vdd/vdd | FineBudgetOnly_Exact | vehicle | 38.4045 | -3.7770 | 38.4045 | 100.0000 | 0.9325 |
| vdd/vdd | FineBudgetOnly_Exact | roof | 90.4037 | 0.6748 | 90.8606 | 99.4468 | 6.5948 |
| vdd/vdd | FineBudgetOnly_Exact | water | 64.3964 | 3.3260 | 66.9438 | 94.4206 | 11.6384 |
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
| potsdam/potsdam | FineRivalPosterior_Exact | impervious surface | 66.4396 | 0.1950 | 85.1714 | 75.1301 | 36.0573 |
| potsdam/potsdam | FineRivalPosterior_Exact | building | 72.7078 | 0.1803 | 76.7219 | 93.2870 | 19.6040 |
| potsdam/potsdam | FineRivalPosterior_Exact | low vegetation | 20.9725 | 1.2992 | 82.2198 | 21.9689 | 4.6334 |
| potsdam/potsdam | FineRivalPosterior_Exact | tree | 58.3025 | 0.3635 | 92.7249 | 61.0973 | 11.6255 |
| potsdam/potsdam | FineRivalPosterior_Exact | car | 25.1882 | 0.2800 | 25.2303 | 99.3417 | 14.1180 |
| potsdam/potsdam | FineRivalPosterior_Exact | clutter | 2.4856 | 0.0745 | 3.1950 | 10.0681 | 13.9618 |
| potsdam/potsdam | FineActionProjected_Exact | impervious surface | 66.1796 | -0.0650 | 85.2939 | 74.7036 | 35.8011 |
| potsdam/potsdam | FineActionProjected_Exact | building | 72.7222 | 0.1947 | 76.7865 | 93.2154 | 19.5724 |
| potsdam/potsdam | FineActionProjected_Exact | low vegetation | 20.5742 | 0.9009 | 81.7502 | 21.5647 | 4.5743 |
| potsdam/potsdam | FineActionProjected_Exact | tree | 57.7307 | -0.2083 | 92.6289 | 60.5106 | 11.5258 |
| potsdam/potsdam | FineActionProjected_Exact | car | 24.9252 | 0.0170 | 24.9670 | 99.3324 | 14.2656 |
| potsdam/potsdam | FineActionProjected_Exact | clutter | 2.4420 | 0.0309 | 3.1244 | 10.0563 | 14.2607 |
| potsdam/potsdam | FineRivalProjected_Exact | impervious surface | 66.3665 | 0.1219 | 85.2000 | 75.0146 | 35.9898 |
| potsdam/potsdam | FineRivalProjected_Exact | building | 72.8663 | 0.3388 | 76.9013 | 93.2829 | 19.5574 |
| potsdam/potsdam | FineRivalProjected_Exact | low vegetation | 21.7324 | 2.0591 | 81.8728 | 22.8310 | 4.8357 |
| potsdam/potsdam | FineRivalProjected_Exact | tree | 58.0561 | 0.1171 | 92.6395 | 60.8635 | 11.5917 |
| potsdam/potsdam | FineRivalProjected_Exact | car | 25.1793 | 0.2711 | 25.2161 | 99.4242 | 14.1377 |
| potsdam/potsdam | FineRivalProjected_Exact | clutter | 2.5381 | 0.1270 | 3.2649 | 10.2339 | 13.8877 |
| potsdam/potsdam | FineTopTwo_Exact | impervious surface | 67.1029 | 0.8583 | 85.1415 | 76.0032 | 36.4892 |
| potsdam/potsdam | FineTopTwo_Exact | building | 72.9615 | 0.4340 | 76.8510 | 93.5133 | 19.6185 |
| potsdam/potsdam | FineTopTwo_Exact | low vegetation | 23.0818 | 3.4085 | 84.6196 | 24.0926 | 4.9372 |
| potsdam/potsdam | FineTopTwo_Exact | tree | 59.8813 | 1.9423 | 92.5773 | 62.9013 | 11.9879 |
| potsdam/potsdam | FineTopTwo_Exact | car | 26.0695 | 1.1613 | 26.1091 | 99.4215 | 13.6538 |
| potsdam/potsdam | FineTopTwo_Exact | clutter | 2.5302 | 0.1191 | 3.2890 | 9.8830 | 13.3134 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | impervious surface | 66.3575 | 0.1129 | 85.2116 | 74.9940 | 35.9750 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | building | 72.6201 | 0.0926 | 76.6797 | 93.2051 | 19.5975 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | low vegetation | 20.3807 | 0.7074 | 82.0137 | 21.3343 | 4.5109 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | tree | 58.1124 | 0.1734 | 92.7014 | 60.8987 | 11.5907 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | car | 25.0347 | 0.1265 | 25.0808 | 99.2712 | 14.1921 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | clutter | 2.4591 | 0.0480 | 3.1524 | 10.0563 | 14.1338 |
| potsdam/potsdam | FineBudgetOnly_Exact | impervious surface | 66.1042 | -0.1404 | 85.2298 | 74.6567 | 35.8056 |
| potsdam/potsdam | FineBudgetOnly_Exact | building | 72.9459 | 0.4184 | 77.0489 | 93.1965 | 19.5018 |
| potsdam/potsdam | FineBudgetOnly_Exact | low vegetation | 21.5223 | 1.8490 | 81.8118 | 22.6038 | 4.7911 |
| potsdam/potsdam | FineBudgetOnly_Exact | tree | 57.6287 | -0.3103 | 92.5777 | 60.4203 | 11.5150 |
| potsdam/potsdam | FineBudgetOnly_Exact | car | 24.9889 | 0.0807 | 25.0268 | 99.3976 | 14.2408 |
| potsdam/potsdam | FineBudgetOnly_Exact | clutter | 2.4949 | 0.0838 | 3.1966 | 10.2059 | 14.1457 |
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
| udd5/udd5 | FineRivalPosterior_Exact | vegetation | 67.7313 | 0.0351 | 90.9645 | 72.6168 | 2.4645 |
| udd5/udd5 | FineRivalPosterior_Exact | building | 86.0483 | 0.4496 | 86.2039 | 99.7907 | 83.2617 |
| udd5/udd5 | FineRivalPosterior_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0578 |
| udd5/udd5 | FineRivalPosterior_Exact | vehicle | 6.8552 | -0.1552 | 6.8553 | 99.9773 | 6.1253 |
| udd5/udd5 | FineRivalPosterior_Exact | other | 10.2779 | 0.3862 | 40.4107 | 12.1139 | 6.0907 |
| udd5/udd5 | FineActionProjected_Exact | vegetation | 68.4985 | 0.8023 | 91.0176 | 73.4647 | 2.4919 |
| udd5/udd5 | FineActionProjected_Exact | building | 85.4756 | -0.1231 | 85.5542 | 99.8926 | 83.9797 |
| udd5/udd5 | FineActionProjected_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0577 |
| udd5/udd5 | FineActionProjected_Exact | vehicle | 6.8014 | -0.2090 | 6.8015 | 99.9773 | 6.1737 |
| udd5/udd5 | FineActionProjected_Exact | other | 11.4844 | 1.5927 | 43.5396 | 13.4940 | 6.2971 |
| udd5/udd5 | FineRivalProjected_Exact | vegetation | 68.5613 | 0.8651 | 90.9603 | 73.5744 | 2.4971 |
| udd5/udd5 | FineRivalProjected_Exact | building | 85.8800 | 0.2813 | 86.0701 | 99.7434 | 83.3517 |
| udd5/udd5 | FineRivalProjected_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3867 |
| udd5/udd5 | FineRivalProjected_Exact | vehicle | 6.6312 | -0.3792 | 6.6313 | 99.9773 | 6.3321 |
| udd5/udd5 | FineRivalProjected_Exact | other | 12.0146 | 2.1229 | 44.6062 | 14.1216 | 6.4323 |
| udd5/udd5 | FineTopTwo_Exact | vegetation | 54.1582 | -13.5380 | 90.0275 | 57.6146 | 1.9757 |
| udd5/udd5 | FineTopTwo_Exact | building | 85.5450 | -0.0537 | 85.6374 | 99.8740 | 83.8824 |
| udd5/udd5 | FineTopTwo_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9048 |
| udd5/udd5 | FineTopTwo_Exact | vehicle | 5.8040 | -1.2064 | 5.8041 | 99.9773 | 7.2346 |
| udd5/udd5 | FineTopTwo_Exact | other | 8.8195 | -1.0722 | 35.5389 | 10.4990 | 6.0024 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | vegetation | 68.9952 | 1.2990 | 90.8595 | 74.1412 | 2.5192 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | building | 85.7756 | 0.1769 | 85.8578 | 99.8886 | 83.6794 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.8969 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | vehicle | 7.0990 | 0.0886 | 7.0992 | 99.9773 | 5.9148 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | other | 10.2159 | 0.3242 | 40.7111 | 12.0014 | 5.9896 |
| udd5/udd5 | FineBudgetOnly_Exact | vegetation | 68.7476 | 1.0514 | 90.9648 | 73.7860 | 2.5042 |
| udd5/udd5 | FineBudgetOnly_Exact | building | 85.4999 | -0.0988 | 85.5929 | 99.8730 | 83.9252 |
| udd5/udd5 | FineBudgetOnly_Exact | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9439 |
| udd5/udd5 | FineBudgetOnly_Exact | vehicle | 6.8445 | -0.1659 | 6.8446 | 99.9773 | 6.1348 |
| udd5/udd5 | FineBudgetOnly_Exact | other | 12.3612 | 2.4695 | 45.4328 | 14.5163 | 6.4919 |
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
| oem/oem | FineRivalPosterior_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6395 |
| oem/oem | FineRivalPosterior_Exact | rangeland | 49.7762 | 0.0829 | 70.6521 | 62.7509 | 12.9194 |
| oem/oem | FineRivalPosterior_Exact | developed space | 23.4347 | -0.4202 | 41.3714 | 35.0871 | 16.6208 |
| oem/oem | FineRivalPosterior_Exact | road | 55.1826 | 0.2601 | 68.3262 | 74.1511 | 6.8232 |
| oem/oem | FineRivalPosterior_Exact | tree | 57.3337 | 0.3500 | 91.0134 | 60.7742 | 17.6410 |
| oem/oem | FineRivalPosterior_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 |
| oem/oem | FineRivalPosterior_Exact | agriculture land | 82.1791 | 0.1722 | 82.5774 | 99.4165 | 25.5395 |
| oem/oem | FineRivalPosterior_Exact | building | 54.0884 | 3.2248 | 70.4889 | 69.9222 | 11.8164 |
| oem/oem | FineActionProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4822 |
| oem/oem | FineActionProjected_Exact | rangeland | 49.8079 | 0.1146 | 70.6549 | 62.7990 | 12.9288 |
| oem/oem | FineActionProjected_Exact | developed space | 23.7905 | -0.0644 | 40.8878 | 36.2629 | 17.3809 |
| oem/oem | FineActionProjected_Exact | road | 55.1296 | 0.2071 | 68.4333 | 73.9300 | 6.7922 |
| oem/oem | FineActionProjected_Exact | tree | 57.3670 | 0.3833 | 91.0213 | 60.8081 | 17.6493 |
| oem/oem | FineActionProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 |
| oem/oem | FineActionProjected_Exact | agriculture land | 82.3758 | 0.3689 | 82.7788 | 99.4124 | 25.4763 |
| oem/oem | FineActionProjected_Exact | building | 51.3032 | 0.4396 | 69.6846 | 66.0433 | 11.2897 |
| oem/oem | FineRivalProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.6617 |
| oem/oem | FineRivalProjected_Exact | rangeland | 49.8592 | 0.1659 | 70.6753 | 62.8643 | 12.9385 |
| oem/oem | FineRivalProjected_Exact | developed space | 23.4870 | -0.3679 | 41.7734 | 34.9185 | 16.3817 |
| oem/oem | FineRivalProjected_Exact | road | 55.4142 | 0.4917 | 68.4843 | 74.3824 | 6.8287 |
| oem/oem | FineRivalProjected_Exact | tree | 57.7324 | 0.7487 | 90.8900 | 61.2783 | 17.8115 |
| oem/oem | FineRivalProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 |
| oem/oem | FineRivalProjected_Exact | agriculture land | 82.5457 | 0.5388 | 82.9578 | 99.4017 | 25.4186 |
| oem/oem | FineRivalProjected_Exact | building | 54.5491 | 3.6855 | 70.4544 | 70.7286 | 11.9585 |
| oem/oem | FineTopTwo_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.8619 |
| oem/oem | FineTopTwo_Exact | rangeland | 49.9101 | 0.2168 | 70.7262 | 62.9050 | 12.9376 |
| oem/oem | FineTopTwo_Exact | developed space | 22.3930 | -1.4619 | 42.2713 | 32.2579 | 14.9553 |
| oem/oem | FineTopTwo_Exact | road | 54.7650 | -0.1575 | 66.9403 | 75.0685 | 7.0507 |
| oem/oem | FineTopTwo_Exact | tree | 57.8721 | 0.8884 | 90.6949 | 61.5252 | 17.9217 |
| oem/oem | FineTopTwo_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FineTopTwo_Exact | agriculture land | 82.3679 | 0.3610 | 82.7744 | 99.4073 | 25.4764 |
| oem/oem | FineTopTwo_Exact | building | 56.6052 | 5.7416 | 69.7923 | 74.9738 | 12.7965 |
| oem/oem | FinePosteriorStrengthMatched_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.4190 |
| oem/oem | FinePosteriorStrengthMatched_Exact | rangeland | 49.6925 | -0.0008 | 70.6348 | 62.6314 | 12.8980 |
| oem/oem | FinePosteriorStrengthMatched_Exact | developed space | 24.0131 | 0.1582 | 41.3356 | 36.4275 | 17.2707 |
| oem/oem | FinePosteriorStrengthMatched_Exact | road | 54.9366 | 0.0141 | 68.2177 | 73.8342 | 6.8049 |
| oem/oem | FinePosteriorStrengthMatched_Exact | tree | 57.2148 | 0.2311 | 91.0528 | 60.6231 | 17.5895 |
| oem/oem | FinePosteriorStrengthMatched_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 |
| oem/oem | FinePosteriorStrengthMatched_Exact | agriculture land | 82.0258 | 0.0189 | 82.4189 | 99.4218 | 25.5900 |
| oem/oem | FinePosteriorStrengthMatched_Exact | building | 52.3083 | 1.4447 | 70.1433 | 67.2908 | 11.4277 |
| oem/oem | FineBudgetOnly_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.5567 |
| oem/oem | FineBudgetOnly_Exact | rangeland | 49.8649 | 0.1716 | 70.6703 | 62.8774 | 12.9422 |
| oem/oem | FineBudgetOnly_Exact | developed space | 23.7934 | -0.0615 | 41.0958 | 36.1075 | 17.2188 |
| oem/oem | FineBudgetOnly_Exact | road | 55.3529 | 0.4304 | 68.7562 | 73.9549 | 6.7626 |
| oem/oem | FineBudgetOnly_Exact | tree | 57.7722 | 0.7885 | 90.9006 | 61.3183 | 17.8210 |
| oem/oem | FineBudgetOnly_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 |
| oem/oem | FineBudgetOnly_Exact | agriculture land | 82.8190 | 0.8121 | 83.2403 | 99.3925 | 25.3300 |
| oem/oem | FineBudgetOnly_Exact | building | 51.6864 | 0.8228 | 69.7801 | 66.5924 | 11.3680 |
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
| loveda/P | FineRivalPosterior_Exact | building | 52.7619 | -1.2096 | 55.9596 | 90.2280 | 0.0427 |
| loveda/P | FineRivalPosterior_Exact | road | 59.3245 | 0.7799 | 60.5410 | 96.7238 | 13.2961 |
| loveda/P | FineRivalPosterior_Exact | water | 62.6874 | 0.1385 | 67.9337 | 89.0319 | 19.2845 |
| loveda/P | FineRivalPosterior_Exact | barren | 1.9020 | -0.4446 | 43.1846 | 1.9508 | 0.5009 |
| loveda/P | FineRivalPosterior_Exact | tree | 54.9404 | 1.7278 | 97.0534 | 55.8724 | 9.4786 |
| loveda/P | FineRivalPosterior_Exact | farm | 85.3719 | 0.4688 | 85.6795 | 99.5812 | 57.3973 |
| loveda/P | FineActionProjected_Exact | building | 50.5155 | -3.4560 | 51.6696 | 95.7655 | 0.0491 |
| loveda/P | FineActionProjected_Exact | road | 59.4934 | 0.9488 | 60.7251 | 96.7031 | 13.2529 |
| loveda/P | FineActionProjected_Exact | water | 62.8038 | 0.2549 | 68.0440 | 89.0771 | 19.2630 |
| loveda/P | FineActionProjected_Exact | barren | 2.1166 | -0.2300 | 63.8515 | 2.1423 | 0.3720 |
| loveda/P | FineActionProjected_Exact | tree | 55.2400 | 2.0274 | 96.8398 | 56.2540 | 9.5644 |
| loveda/P | FineActionProjected_Exact | farm | 85.2101 | 0.3070 | 85.5220 | 99.5737 | 57.4986 |
| loveda/P | FineRivalProjected_Exact | building | 48.9292 | -5.0423 | 49.7487 | 96.7427 | 0.0515 |
| loveda/P | FineRivalProjected_Exact | road | 60.1611 | 1.6165 | 61.4250 | 96.6927 | 13.1005 |
| loveda/P | FineRivalProjected_Exact | water | 62.9242 | 0.3753 | 68.0543 | 89.3018 | 19.3086 |
| loveda/P | FineRivalProjected_Exact | barren | 1.6724 | -0.6742 | 41.1787 | 1.7134 | 0.4613 |
| loveda/P | FineRivalProjected_Exact | tree | 56.5911 | 3.3785 | 96.7147 | 57.7003 | 9.8230 |
| loveda/P | FineRivalProjected_Exact | farm | 85.5838 | 0.6807 | 85.8926 | 99.5816 | 57.2550 |
| loveda/P | FineTopTwo_Exact | building | 41.6098 | -12.3617 | 41.7237 | 99.3485 | 0.0631 |
| loveda/P | FineTopTwo_Exact | road | 58.6835 | 0.1389 | 59.8867 | 96.6896 | 13.4366 |
| loveda/P | FineTopTwo_Exact | water | 62.3757 | -0.1732 | 67.2658 | 89.5616 | 19.5918 |
| loveda/P | FineTopTwo_Exact | barren | 1.5789 | -0.7677 | 40.2285 | 1.6168 | 0.4456 |
| loveda/P | FineTopTwo_Exact | tree | 51.5684 | -1.6442 | 98.0497 | 52.1028 | 8.7493 |
| loveda/P | FineTopTwo_Exact | farm | 84.8949 | -0.0082 | 85.2040 | 99.5744 | 57.7136 |
| loveda/P | FinePosteriorStrengthMatched_Exact | building | 54.0984 | 0.1269 | 59.3258 | 85.9935 | 0.0384 |
| loveda/P | FinePosteriorStrengthMatched_Exact | road | 58.8679 | 0.3233 | 60.0600 | 96.7383 | 13.4046 |
| loveda/P | FinePosteriorStrengthMatched_Exact | water | 62.4217 | -0.1272 | 67.7464 | 88.8167 | 19.2910 |
| loveda/P | FinePosteriorStrengthMatched_Exact | barren | 2.2138 | -0.1328 | 48.2684 | 2.2676 | 0.5209 |
| loveda/P | FinePosteriorStrengthMatched_Exact | tree | 52.9596 | -0.2530 | 97.7206 | 53.6220 | 9.0347 |
| loveda/P | FinePosteriorStrengthMatched_Exact | farm | 84.9073 | 0.0042 | 85.2129 | 99.5793 | 57.7104 |
| loveda/P | FineBudgetOnly_Exact | building | 44.3478 | -9.6237 | 44.4122 | 99.6743 | 0.0595 |
| loveda/P | FineBudgetOnly_Exact | road | 60.5514 | 2.0068 | 61.8333 | 96.6896 | 13.0136 |
| loveda/P | FineBudgetOnly_Exact | water | 63.0869 | 0.5380 | 68.1903 | 89.3950 | 19.2903 |
| loveda/P | FineBudgetOnly_Exact | barren | 1.8502 | -0.4964 | 62.0227 | 1.8714 | 0.3345 |
| loveda/P | FineBudgetOnly_Exact | tree | 57.0813 | 3.8687 | 96.4614 | 58.3021 | 9.9515 |
| loveda/P | FineBudgetOnly_Exact | farm | 85.4354 | 0.5323 | 85.7461 | 99.5778 | 57.3507 |
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
| loveda/D | FineRivalPosterior_Exact | background | 30.1227 | 4.1492 | 84.3761 | 31.9022 | 16.9208 |
| loveda/D | FineRivalPosterior_Exact | building | 33.3728 | -4.7158 | 34.3902 | 91.8567 | 0.0391 |
| loveda/D | FineRivalPosterior_Exact | road | 49.0124 | 0.4245 | 49.8492 | 96.6885 | 8.9180 |
| loveda/D | FineRivalPosterior_Exact | water | 54.8120 | 0.1032 | 58.8605 | 88.8507 | 12.2715 |
| loveda/D | FineRivalPosterior_Exact | barren | 1.6429 | -0.1336 | 65.3870 | 1.6573 | 0.1553 |
| loveda/D | FineRivalPosterior_Exact | tree | 47.0449 | 1.0554 | 93.3889 | 48.6656 | 4.7402 |
| loveda/D | FineRivalPosterior_Exact | farm | 46.2701 | 0.8860 | 46.7869 | 97.6687 | 56.9552 |
| loveda/D | FineActionProjected_Exact | background | 27.1535 | 1.1800 | 86.0315 | 28.4059 | 14.7764 |
| loveda/D | FineActionProjected_Exact | building | 32.5708 | -5.5178 | 32.8571 | 97.3941 | 0.0434 |
| loveda/D | FineActionProjected_Exact | road | 48.9453 | 0.3574 | 49.7863 | 96.6637 | 8.9270 |
| loveda/D | FineActionProjected_Exact | water | 54.8785 | 0.1697 | 58.8905 | 88.9569 | 12.2798 |
| loveda/D | FineActionProjected_Exact | barren | 1.7458 | -0.0307 | 69.5599 | 1.7593 | 0.1549 |
| loveda/D | FineActionProjected_Exact | tree | 47.5775 | 1.5880 | 92.8905 | 49.3754 | 4.8351 |
| loveda/D | FineActionProjected_Exact | farm | 45.9535 | 0.5694 | 46.0489 | 99.5512 | 58.9833 |
| loveda/D | FineRivalProjected_Exact | background | 30.6934 | 4.7199 | 84.6654 | 32.5002 | 17.1790 |
| loveda/D | FineRivalProjected_Exact | building | 31.5625 | -6.5261 | 31.6946 | 98.6971 | 0.0456 |
| loveda/D | FineRivalProjected_Exact | road | 49.2751 | 0.6872 | 50.1318 | 96.6481 | 8.8640 |
| loveda/D | FineRivalProjected_Exact | water | 55.0011 | 0.2923 | 58.9493 | 89.1446 | 12.2935 |
| loveda/D | FineRivalProjected_Exact | barren | 1.6177 | -0.1588 | 66.8262 | 1.6308 | 0.1495 |
| loveda/D | FineRivalProjected_Exact | tree | 48.4719 | 2.4824 | 92.5745 | 50.4327 | 4.9555 |
| loveda/D | FineRivalProjected_Exact | farm | 46.5944 | 1.2103 | 47.1297 | 97.6203 | 56.5129 |
| loveda/D | FineTopTwo_Exact | background | 36.9762 | 11.0027 | 79.6937 | 40.8223 | 22.9241 |
| loveda/D | FineTopTwo_Exact | building | 27.4816 | -10.6070 | 27.6852 | 97.3941 | 0.0515 |
| loveda/D | FineTopTwo_Exact | road | 48.4488 | -0.1391 | 49.2782 | 96.6429 | 9.0171 |
| loveda/D | FineTopTwo_Exact | water | 54.3596 | -0.3492 | 58.0691 | 89.4842 | 12.5274 |
| loveda/D | FineTopTwo_Exact | barren | 1.6181 | -0.1584 | 71.6193 | 1.6285 | 0.1393 |
| loveda/D | FineTopTwo_Exact | tree | 41.8644 | -4.1251 | 95.0595 | 42.7956 | 4.0952 |
| loveda/D | FineTopTwo_Exact | farm | 46.1025 | 0.7184 | 48.3551 | 90.8230 | 51.2455 |
| loveda/D | FinePosteriorStrengthMatched_Exact | background | 28.0410 | 2.0675 | 86.3620 | 29.3402 | 15.2040 |
| loveda/D | FinePosteriorStrengthMatched_Exact | building | 34.2929 | -3.7957 | 35.7702 | 89.2508 | 0.0365 |
| loveda/D | FinePosteriorStrengthMatched_Exact | road | 48.7506 | 0.1627 | 49.5765 | 96.6958 | 8.9677 |
| loveda/D | FinePosteriorStrengthMatched_Exact | water | 54.6327 | -0.0761 | 58.7381 | 88.6577 | 12.2703 |
| loveda/D | FinePosteriorStrengthMatched_Exact | barren | 1.7443 | -0.0322 | 67.1221 | 1.7593 | 0.1606 |
| loveda/D | FinePosteriorStrengthMatched_Exact | tree | 45.7620 | -0.2275 | 93.8951 | 47.1654 | 4.5693 |
| loveda/D | FinePosteriorStrengthMatched_Exact | farm | 46.1045 | 0.7204 | 46.2001 | 99.5533 | 58.7915 |
| loveda/D | FineBudgetOnly_Exact | background | 28.8149 | 2.8414 | 86.6920 | 30.1485 | 15.5634 |
| loveda/D | FineBudgetOnly_Exact | building | 30.0098 | -8.0788 | 30.0098 | 100.0000 | 0.0488 |
| loveda/D | FineBudgetOnly_Exact | road | 49.3559 | 0.7680 | 50.2207 | 96.6284 | 8.8465 |
| loveda/D | FineBudgetOnly_Exact | water | 55.1009 | 0.3921 | 59.0131 | 89.2607 | 12.2962 |
| loveda/D | FineBudgetOnly_Exact | barren | 1.7088 | -0.0677 | 70.4364 | 1.7211 | 0.1497 |
| loveda/D | FineBudgetOnly_Exact | tree | 49.0748 | 3.0853 | 92.1383 | 51.2196 | 5.0567 |
| loveda/D | FineBudgetOnly_Exact | farm | 46.6943 | 1.3102 | 46.7946 | 99.5431 | 58.0387 |
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
| vaihingen/vaihingen | FineRivalPosterior_Exact | impervious surface | 59.1819 | 0.3716 | 74.8509 | 73.8706 | 26.9589 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | building | 67.3035 | 0.2271 | 67.4480 | 99.6827 | 30.4904 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | low vegetation | 45.0244 | 0.0742 | 95.9861 | 45.8884 | 14.1357 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | tree | 67.5573 | 0.0365 | 78.6982 | 82.6756 | 21.6817 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | car | 26.6509 | 0.2855 | 26.8120 | 97.7941 | 6.7334 |
| vaihingen/vaihingen | FineActionProjected_Exact | impervious surface | 59.0689 | 0.2586 | 75.4951 | 73.0807 | 26.4431 |
| vaihingen/vaihingen | FineActionProjected_Exact | building | 67.2334 | 0.1570 | 67.3732 | 99.6924 | 30.5272 |
| vaihingen/vaihingen | FineActionProjected_Exact | low vegetation | 45.8197 | 0.8695 | 95.9021 | 46.7347 | 14.4090 |
| vaihingen/vaihingen | FineActionProjected_Exact | tree | 67.5795 | 0.0587 | 78.6833 | 82.7253 | 21.6988 |
| vaihingen/vaihingen | FineActionProjected_Exact | car | 25.9799 | -0.3855 | 26.1222 | 97.9465 | 6.9220 |
| vaihingen/vaihingen | FineRivalProjected_Exact | impervious surface | 59.3766 | 0.5663 | 75.4775 | 73.5690 | 26.6259 |
| vaihingen/vaihingen | FineRivalProjected_Exact | building | 67.4674 | 0.3910 | 67.6072 | 99.6944 | 30.4222 |
| vaihingen/vaihingen | FineRivalProjected_Exact | low vegetation | 45.7742 | 0.8240 | 95.8882 | 46.6907 | 14.3975 |
| vaihingen/vaihingen | FineRivalProjected_Exact | tree | 67.6134 | 0.0926 | 78.6701 | 82.7907 | 21.7196 |
| vaihingen/vaihingen | FineRivalProjected_Exact | car | 26.2980 | -0.0674 | 26.4463 | 97.9130 | 6.8348 |
| vaihingen/vaihingen | FineTopTwo_Exact | impervious surface | 59.6324 | 0.8221 | 74.9999 | 74.4266 | 27.1079 |
| vaihingen/vaihingen | FineTopTwo_Exact | building | 67.2418 | 0.1654 | 67.3836 | 99.6880 | 30.5211 |
| vaihingen/vaihingen | FineTopTwo_Exact | low vegetation | 44.7149 | -0.2353 | 96.1229 | 45.5362 | 14.0072 |
| vaihingen/vaihingen | FineTopTwo_Exact | tree | 67.6046 | 0.0838 | 78.6966 | 82.7481 | 21.7011 |
| vaihingen/vaihingen | FineTopTwo_Exact | car | 26.8598 | 0.4944 | 27.0393 | 97.5875 | 6.6627 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | impervious surface | 58.8062 | -0.0041 | 74.7158 | 73.4162 | 26.8415 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | building | 66.9969 | -0.0795 | 67.1395 | 99.6840 | 30.6309 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | low vegetation | 44.9115 | -0.0387 | 96.0380 | 45.7594 | 14.0883 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | tree | 67.5249 | 0.0041 | 78.7394 | 82.5816 | 21.6457 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | car | 26.4153 | 0.0499 | 26.5738 | 97.7916 | 6.7935 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | impervious surface | 59.3423 | 0.5320 | 76.2324 | 72.8142 | 26.0918 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | building | 67.4593 | 0.3829 | 67.5948 | 99.7037 | 30.4306 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | low vegetation | 46.6048 | 1.6546 | 95.6768 | 47.6074 | 14.7126 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | tree | 67.6624 | 0.1416 | 78.6651 | 82.8697 | 21.7417 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | car | 25.6512 | -0.7142 | 25.7806 | 98.0808 | 7.0233 |
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
| landcoverai/landcoverai | FineRivalPosterior_Exact | background | 88.2094 | 0.2266 | 95.0155 | 92.4893 | 65.9584 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | building | 44.5981 | 0.0391 | 44.9315 | 98.3634 | 3.3167 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | woodland | 81.4048 | 0.8477 | 93.9842 | 85.8796 | 19.5608 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | water | 97.6085 | -0.0049 | 97.6085 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | road | 26.9113 | 0.2068 | 29.4567 | 75.6946 | 2.6901 |
| landcoverai/landcoverai | FineActionProjected_Exact | background | 88.0350 | 0.0522 | 94.7195 | 92.5786 | 66.2284 |
| landcoverai/landcoverai | FineActionProjected_Exact | building | 44.5808 | 0.0218 | 44.9146 | 98.3602 | 3.3179 |
| landcoverai/landcoverai | FineActionProjected_Exact | woodland | 80.7162 | 0.1591 | 94.2411 | 84.9040 | 19.2859 |
| landcoverai/landcoverai | FineActionProjected_Exact | water | 97.6167 | 0.0033 | 97.6167 | 100.0000 | 8.4733 |
| landcoverai/landcoverai | FineActionProjected_Exact | road | 26.8556 | 0.1511 | 29.3948 | 75.6627 | 2.6946 |
| landcoverai/landcoverai | FineRivalProjected_Exact | background | 88.2370 | 0.2542 | 95.1004 | 92.4393 | 65.8639 |
| landcoverai/landcoverai | FineRivalProjected_Exact | building | 44.5937 | 0.0347 | 44.9185 | 98.4043 | 3.3191 |
| landcoverai/landcoverai | FineRivalProjected_Exact | woodland | 81.5413 | 0.9842 | 93.8335 | 86.1583 | 19.6558 |
| landcoverai/landcoverai | FineRivalProjected_Exact | water | 97.6079 | -0.0055 | 97.6079 | 100.0000 | 8.4740 |
| landcoverai/landcoverai | FineRivalProjected_Exact | road | 26.9379 | 0.2334 | 29.4886 | 75.6946 | 2.6872 |
| landcoverai/landcoverai | FineTopTwo_Exact | background | 88.5913 | 0.6085 | 95.7004 | 92.2636 | 65.3266 |
| landcoverai/landcoverai | FineTopTwo_Exact | building | 44.6408 | 0.0818 | 44.9473 | 98.4956 | 3.3200 |
| landcoverai/landcoverai | FineTopTwo_Exact | woodland | 82.9298 | 2.3727 | 93.3989 | 88.0931 | 20.1907 |
| landcoverai/landcoverai | FineTopTwo_Exact | water | 97.6118 | -0.0016 | 97.6118 | 100.0000 | 8.4737 |
| landcoverai/landcoverai | FineTopTwo_Exact | road | 26.9629 | 0.2584 | 29.5047 | 75.7857 | 2.6889 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | background | 88.1222 | 0.1394 | 94.7922 | 92.6056 | 66.1970 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | building | 44.5462 | -0.0128 | 44.8886 | 98.3162 | 3.3183 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | woodland | 81.0156 | 0.4585 | 94.3568 | 85.1410 | 19.3160 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | water | 97.6145 | 0.0011 | 97.6145 | 100.0000 | 8.4734 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | road | 26.8500 | 0.1455 | 29.3880 | 75.6627 | 2.6952 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | background | 88.0617 | 0.0789 | 94.8162 | 92.5159 | 66.1161 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | building | 44.5675 | 0.0085 | 44.8926 | 98.4012 | 3.3209 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | woodland | 80.8303 | 0.2732 | 94.0020 | 85.2259 | 19.4082 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | water | 97.6173 | 0.0039 | 97.6173 | 100.0000 | 8.4732 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | road | 26.9705 | 0.2660 | 29.5339 | 75.6536 | 2.6816 |
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
| flair1/flair1 | FineRivalPosterior_Exact | building | 57.3282 | 0.1530 | 58.6076 | 96.3317 | 11.8008 |
| flair1/flair1 | FineRivalPosterior_Exact | pervious surface | 46.5588 | -0.0686 | 91.7762 | 48.5858 | 9.0843 |
| flair1/flair1 | FineRivalPosterior_Exact | impervious surface | 52.7327 | 0.1581 | 60.5890 | 80.2638 | 21.7022 |
| flair1/flair1 | FineRivalPosterior_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1623 |
| flair1/flair1 | FineRivalPosterior_Exact | water | 64.1075 | 0.4197 | 66.1925 | 95.3168 | 6.3878 |
| flair1/flair1 | FineRivalPosterior_Exact | coniferous | 42.8136 | 0.0365 | 62.9267 | 57.2556 | 0.5115 |
| flair1/flair1 | FineRivalPosterior_Exact | deciduous | 60.7853 | 0.4045 | 78.7185 | 72.7387 | 15.7286 |
| flair1/flair1 | FineRivalPosterior_Exact | brushwood | 15.1360 | 0.2525 | 29.5360 | 23.6906 | 3.9450 |
| flair1/flair1 | FineRivalPosterior_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0256 |
| flair1/flair1 | FineRivalPosterior_Exact | herbaceous vegetation | 57.4891 | -0.0354 | 91.4452 | 60.7567 | 21.2844 |
| flair1/flair1 | FineRivalPosterior_Exact | agricultural land | 13.1795 | 0.0620 | 14.2018 | 64.6746 | 1.3886 |
| flair1/flair1 | FineRivalPosterior_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9790 |
| flair1/flair1 | FineActionProjected_Exact | building | 57.2798 | 0.1046 | 58.5392 | 96.3802 | 11.8205 |
| flair1/flair1 | FineActionProjected_Exact | pervious surface | 46.9960 | 0.3686 | 91.8941 | 49.0284 | 9.1553 |
| flair1/flair1 | FineActionProjected_Exact | impervious surface | 52.7721 | 0.1975 | 60.7221 | 80.1222 | 21.6165 |
| flair1/flair1 | FineActionProjected_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0839 |
| flair1/flair1 | FineActionProjected_Exact | water | 64.4608 | 0.7730 | 66.5723 | 95.3103 | 6.3510 |
| flair1/flair1 | FineActionProjected_Exact | coniferous | 42.7776 | 0.0005 | 62.7064 | 57.3744 | 0.5143 |
| flair1/flair1 | FineActionProjected_Exact | deciduous | 60.5673 | 0.1865 | 78.7705 | 72.3827 | 15.6413 |
| flair1/flair1 | FineActionProjected_Exact | brushwood | 15.3727 | 0.4892 | 29.1005 | 24.5781 | 4.1540 |
| flair1/flair1 | FineActionProjected_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0182 |
| flair1/flair1 | FineActionProjected_Exact | herbaceous vegetation | 57.6542 | 0.1297 | 91.4728 | 60.9289 | 21.3382 |
| flair1/flair1 | FineActionProjected_Exact | agricultural land | 13.4146 | 0.2971 | 14.4791 | 64.5964 | 1.3604 |
| flair1/flair1 | FineActionProjected_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9464 |
| flair1/flair1 | FineRivalProjected_Exact | building | 57.4041 | 0.2289 | 58.6643 | 96.3928 | 11.7969 |
| flair1/flair1 | FineRivalProjected_Exact | pervious surface | 46.8072 | 0.1798 | 91.9102 | 48.8185 | 9.1145 |
| flair1/flair1 | FineRivalProjected_Exact | impervious surface | 52.9006 | 0.3260 | 60.8089 | 80.2670 | 21.6246 |
| flair1/flair1 | FineRivalProjected_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0982 |
| flair1/flair1 | FineRivalProjected_Exact | water | 64.7837 | 1.0959 | 66.9300 | 95.2835 | 6.3152 |
| flair1/flair1 | FineRivalProjected_Exact | coniferous | 42.7265 | -0.0506 | 62.4459 | 57.5017 | 0.5176 |
| flair1/flair1 | FineRivalProjected_Exact | deciduous | 60.9048 | 0.5240 | 78.7253 | 72.9040 | 15.7630 |
| flair1/flair1 | FineRivalProjected_Exact | brushwood | 15.6306 | 0.7471 | 29.7469 | 24.7769 | 4.0967 |
| flair1/flair1 | FineRivalProjected_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0203 |
| flair1/flair1 | FineRivalProjected_Exact | herbaceous vegetation | 57.6203 | 0.0958 | 91.5155 | 60.8721 | 21.3084 |
| flair1/flair1 | FineRivalProjected_Exact | agricultural land | 13.4305 | 0.3130 | 14.5039 | 64.4712 | 1.3554 |
| flair1/flair1 | FineRivalProjected_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9893 |
| flair1/flair1 | FineTopTwo_Exact | building | 57.0980 | -0.0772 | 58.4027 | 96.2347 | 11.8302 |
| flair1/flair1 | FineTopTwo_Exact | pervious surface | 45.9758 | -0.6516 | 91.3358 | 48.0724 | 9.0316 |
| flair1/flair1 | FineTopTwo_Exact | impervious surface | 52.6461 | 0.0715 | 60.4087 | 80.3802 | 21.7985 |
| flair1/flair1 | FineTopTwo_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.2205 |
| flair1/flair1 | FineTopTwo_Exact | water | 62.8359 | -0.8519 | 64.7113 | 95.5910 | 6.5528 |
| flair1/flair1 | FineTopTwo_Exact | coniferous | 42.5713 | -0.2058 | 63.3165 | 56.5088 | 0.5017 |
| flair1/flair1 | FineTopTwo_Exact | deciduous | 62.1706 | 1.7898 | 78.7522 | 74.7010 | 16.1460 |
| flair1/flair1 | FineTopTwo_Exact | brushwood | 14.2504 | -0.6331 | 33.3865 | 19.9119 | 2.9334 |
| flair1/flair1 | FineTopTwo_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0074 |
| flair1/flair1 | FineTopTwo_Exact | herbaceous vegetation | 57.6004 | 0.0759 | 90.7603 | 61.1884 | 21.5973 |
| flair1/flair1 | FineTopTwo_Exact | agricultural land | 13.0428 | -0.0747 | 14.0373 | 64.7997 | 1.4076 |
| flair1/flair1 | FineTopTwo_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9728 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | building | 57.1858 | 0.0106 | 58.4486 | 96.3596 | 11.8363 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | pervious surface | 46.6095 | -0.0179 | 91.7065 | 48.6606 | 9.1052 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | impervious surface | 52.5825 | 0.0079 | 60.4866 | 80.0952 | 21.6933 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1761 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | water | 63.6476 | -0.0402 | 65.6889 | 95.3448 | 6.4387 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | coniferous | 42.7654 | -0.0117 | 63.0592 | 57.0604 | 0.5087 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | deciduous | 60.3947 | 0.0139 | 78.8145 | 72.0994 | 15.5714 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | brushwood | 14.8288 | -0.0547 | 28.7012 | 23.4773 | 4.0232 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0211 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | herbaceous vegetation | 57.5219 | -0.0026 | 91.4133 | 60.8075 | 21.3096 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | agricultural land | 13.1466 | 0.0291 | 14.1592 | 64.7685 | 1.3948 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9217 |
| flair1/flair1 | FineBudgetOnly_Exact | building | 57.3732 | 0.1980 | 58.6294 | 96.4001 | 11.8048 |
| flair1/flair1 | FineBudgetOnly_Exact | pervious surface | 47.2072 | 0.5798 | 92.1302 | 49.1908 | 9.1621 |
| flair1/flair1 | FineBudgetOnly_Exact | impervious surface | 52.9893 | 0.4147 | 60.9714 | 80.1886 | 21.5459 |
| flair1/flair1 | FineBudgetOnly_Exact | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0326 |
| flair1/flair1 | FineBudgetOnly_Exact | water | 65.2431 | 1.5553 | 67.4205 | 95.2835 | 6.2693 |
| flair1/flair1 | FineBudgetOnly_Exact | coniferous | 42.6496 | -0.1275 | 62.2418 | 57.5356 | 0.5196 |
| flair1/flair1 | FineBudgetOnly_Exact | deciduous | 60.7870 | 0.4062 | 78.7072 | 72.7507 | 15.7335 |
| flair1/flair1 | FineBudgetOnly_Exact | brushwood | 15.8816 | 0.9981 | 29.5663 | 25.5470 | 4.2498 |
| flair1/flair1 | FineBudgetOnly_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0168 |
| flair1/flair1 | FineBudgetOnly_Exact | herbaceous vegetation | 57.7631 | 0.2386 | 91.5243 | 61.0276 | 21.3608 |
| flair1/flair1 | FineBudgetOnly_Exact | agricultural land | 13.5617 | 0.4442 | 14.6595 | 64.4243 | 1.3400 |
| flair1/flair1 | FineBudgetOnly_Exact | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9648 |
