# Competition-conditioned alias action: complete-image development panel

96 unique complete images: full UDD5(40), seven other domains eight each. Not full eight-dataset results and not independent validation. Original Geometry/finite VIP wide observer/fine source/all20/reconstruction are frozen. All three original controls match historical full evaluation per image exactly. Masks only after prediction. LoveDA D once; P separate. Common scored-class denominators.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalPosterior_Exact | FineActionProjected_Exact | FineRivalProjected_Exact | FineTopTwo_Exact | FinePosteriorStrengthMatched_Exact | FineBudgetOnly_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.9462 | 57.0012 | 57.6659 | 57.5076 | 57.8696 | 57.4022 | 56.9915 | 57.2610 | 57.8535 |
| potsdam/potsdam | 40.3528 | 41.7806 | 43.0218 | 43.2584 | 43.0832 | 43.3137 | 43.6218 | 43.1143 | 43.1632 |
| udd5/udd5 | 50.5553 | 49.2117 | 52.9563 | 53.2342 | 52.9202 | 53.1947 | 52.1007 | 53.1920 | 52.9621 |
| oem/oem | 39.3543 | 31.9533 | 32.9765 | 33.4535 | 33.2007 | 33.6778 | 33.6436 | 33.1666 | 33.4083 |
| loveda/P | 62.8254 | 62.4863 | 64.5291 | 64.9639 | 65.0206 | 65.4187 | 65.1316 | 64.5067 | 65.5628 |
| loveda/D | 38.4558 | 36.5599 | 40.6605 | 41.7095 | 41.0880 | 42.0883 | 42.2054 | 41.0195 | 41.6701 |
| vaihingen/vaihingen | 50.2325 | 51.4491 | 52.6486 | 52.9568 | 52.8408 | 53.1461 | 53.1202 | 52.6385 | 53.0670 |
| landcoverai/landcoverai | 60.9049 | 66.9060 | 67.4834 | 67.7464 | 67.5608 | 67.7836 | 68.1473 | 67.6297 | 67.6095 |
| flair1/flair1 | 35.6059 | 33.6084 | 34.0624 | 34.1776 | 34.2746 | 34.3507 | 34.0159 | 34.0569 | 34.4546 |
| Eight-domain mean | 43.4260 | 46.0588 | 47.6844 | 48.0055 | 47.8548 | 48.1196 | 47.9808 | 47.7598 | 48.0235 |

## Frozen Gate

| Candidate | Mean gain vs hard, pp | Worst protocol delta, pp | Pass |
| --- | ---: | ---: | --- |
| FineRivalPosterior_Exact | +0.3211 | -0.1583 | True |
| FineActionProjected_Exact | +0.1703 | -0.0360 | True |
| FineRivalProjected_Exact | +0.4352 | -0.2637 | True |

No domain-specific winner switching. Diagnostic top-two, strength-matched and fine-only controls are not alias candidates. These results do not automatically promote a new model or launch the full20092 suite.

Softmax rival weights are model predictions, not calibrated correctness probabilities. Projection bounds the source pair action before Geometry reconstruction; it does not guarantee a bound on final dense margins. FineBudgetOnly retains alias-derived action magnitudes and changes their direction to fine class evidence, so it is not an alias-free model. These distinctions matter when assigning gains to the screening mechanism.

## Shared Multi-Arm Cost

| Dataset | Images | Wall seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| vdd | 8 | 236.290 | 5587.845 |
| potsdam | 8 | 20.084 | 5583.896 |
| udd5 | 40 | 988.036 | 5575.992 |
| oem | 8 | 21.208 | 5597.532 |
| loveda | 8 | 28.696 | 5616.224 |
| vaihingen | 8 | 18.771 | 5578.271 |
| landcoverai | 8 | 3.501 | 5559.247 |
| flair1 | 8 | 4.322 | 5796.300 |

All nine dense prediction/transition endpoints share forwards in these costs; they are not isolated primary latency. Independent warmed latency/memory is in RIVAL_COMPETITION_ADMISSION_20261005.md.

## Conditional Admission Diagnostics

Tile averages across valid query/class/rival comparisons, not global vocabulary sizes. The input pool stays20 per class; accepted words depend on position and rival, with no fixed retained-count quota.

| Dataset/protocol | Mean retained aliases | Comparisons with deletion | Effective softmax class count | Projected nonzero-action retention | Zero-original-action fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 17.665901 | 0.483894 | 5.208682 | 0.540331 | 0.000000 |
| potsdam/potsdam | 18.583272 | 0.439867 | 4.821688 | 0.594769 | 0.000000 |
| udd5/udd5 | 18.237525 | 0.499539 | 3.707814 | 0.522456 | 0.000000 |
| oem/oem | 18.917550 | 0.371750 | 5.715299 | 0.496031 | 0.000000 |
| loveda/P | 17.840657 | 0.542029 | 4.592901 | 0.522812 | 0.000000 |
| loveda/D | 17.933164 | 0.540643 | 5.539738 | 0.524184 | 0.000000 |
| vaihingen/vaihingen | 18.483083 | 0.422953 | 3.937744 | 0.605930 | 0.000000 |
| landcoverai/landcoverai | 18.836230 | 0.356024 | 3.535141 | 0.480987 | 0.000000 |
| flair1/flair1 | 18.909873 | 0.397940 | 8.884863 | 0.536207 | 0.000000 |

## Residual-Excluded Metrics

Common scored-class support across arms, excluding other/background for these two residual protocols.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalPosterior_Exact | FineActionProjected_Exact | FineRivalProjected_Exact | FineTopTwo_Exact | FinePosteriorStrengthMatched_Exact | FineBudgetOnly_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5 | 55.5480 | 52.7852 | 56.9706 | 57.3236 | 56.8937 | 57.2321 | 56.1853 | 57.2449 | 56.9171 |
| landcoverai | 55.4788 | 61.7706 | 62.3585 | 62.6307 | 62.4423 | 62.6702 | 63.0363 | 62.5066 | 62.4964 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs hard | Precision | Recall | Area |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 25.3949 | -28.9800 | 38.9278 | 42.2128 | 29.1864 |
| vdd/vdd | Geometry | wall | 11.8982 | -45.6975 | 11.9838 | 94.3324 | 15.3444 |
| vdd/vdd | Geometry | road | 24.1970 | 0.4285 | 24.7412 | 91.6676 | 5.1104 |
| vdd/vdd | Geometry | vegetation | 62.0257 | -0.8352 | 84.6841 | 69.8628 | 17.8191 |
| vdd/vdd | Geometry | vehicle | 5.6965 | -19.3144 | 5.6966 | 99.9563 | 4.1453 |
| vdd/vdd | Geometry | roof | 60.7577 | -25.8998 | 88.0153 | 66.2377 | 20.9399 |
| vdd/vdd | Geometry | water | 33.6534 | -59.7394 | 93.0590 | 34.5198 | 7.4546 |
| vdd/vdd | NoAdmission_Exact | other | 53.4471 | -0.9278 | 72.8291 | 66.7587 | 24.6717 |
| vdd/vdd | NoAdmission_Exact | wall | 57.7209 | 0.1252 | 68.4274 | 78.6737 | 2.2412 |
| vdd/vdd | NoAdmission_Exact | road | 25.5594 | 1.7909 | 25.7549 | 97.1163 | 5.2010 |
| vdd/vdd | NoAdmission_Exact | vegetation | 60.0299 | -2.8310 | 96.8776 | 61.2143 | 13.6481 |
| vdd/vdd | NoAdmission_Exact | vehicle | 25.3443 | 0.3334 | 25.5376 | 97.1009 | 0.8983 |
| vdd/vdd | NoAdmission_Exact | roof | 84.4472 | -2.2103 | 84.8283 | 99.4708 | 32.6274 |
| vdd/vdd | NoAdmission_Exact | water | 92.4594 | -0.9334 | 94.6527 | 97.5551 | 20.7123 |
| vdd/vdd | RivalFineHard_Exact | other | 54.3749 | 0.0000 | 73.9753 | 67.2369 | 24.4635 |
| vdd/vdd | RivalFineHard_Exact | wall | 57.5957 | 0.0000 | 65.9286 | 82.0044 | 2.4246 |
| vdd/vdd | RivalFineHard_Exact | road | 23.7685 | 0.0000 | 23.8294 | 98.9358 | 5.7266 |
| vdd/vdd | RivalFineHard_Exact | vegetation | 62.8609 | 0.0000 | 97.0366 | 64.0913 | 14.2661 |
| vdd/vdd | RivalFineHard_Exact | vehicle | 25.0109 | 0.0000 | 25.0386 | 99.5582 | 0.9393 |
| vdd/vdd | RivalFineHard_Exact | roof | 86.6575 | 0.0000 | 87.1372 | 99.3687 | 31.7302 |
| vdd/vdd | RivalFineHard_Exact | water | 93.3928 | 0.0000 | 95.7487 | 97.4330 | 20.4496 |
| vdd/vdd | FineRivalPosterior_Exact | other | 54.5060 | 0.1311 | 73.8629 | 67.5311 | 24.6079 |
| vdd/vdd | FineRivalPosterior_Exact | wall | 54.2033 | -3.3924 | 61.0807 | 82.8001 | 2.6425 |
| vdd/vdd | FineRivalPosterior_Exact | road | 24.4964 | 0.7279 | 24.5554 | 99.0291 | 5.5626 |
| vdd/vdd | FineRivalPosterior_Exact | vegetation | 64.1822 | 1.3213 | 97.0120 | 65.4766 | 14.5781 |
| vdd/vdd | FineRivalPosterior_Exact | vehicle | 25.1222 | 0.1113 | 25.1337 | 99.8170 | 0.9382 |
| vdd/vdd | FineRivalPosterior_Exact | roof | 86.3636 | -0.2939 | 87.4567 | 98.5734 | 31.3613 |
| vdd/vdd | FineRivalPosterior_Exact | water | 93.6796 | 0.2868 | 96.2286 | 97.2502 | 20.3094 |
| vdd/vdd | FineActionProjected_Exact | other | 55.0875 | 0.7126 | 73.9703 | 68.3339 | 24.8643 |
| vdd/vdd | FineActionProjected_Exact | wall | 55.5253 | -2.0704 | 62.9892 | 82.4126 | 2.5504 |
| vdd/vdd | FineActionProjected_Exact | road | 25.7635 | 1.9950 | 25.8365 | 98.9154 | 5.2807 |
| vdd/vdd | FineActionProjected_Exact | vegetation | 63.4757 | 0.6148 | 97.0806 | 64.7110 | 14.3975 |
| vdd/vdd | FineActionProjected_Exact | vehicle | 25.0815 | 0.0706 | 25.1029 | 99.6609 | 0.9379 |
| vdd/vdd | FineActionProjected_Exact | roof | 86.7689 | 0.1114 | 87.3692 | 99.2144 | 31.5968 |
| vdd/vdd | FineActionProjected_Exact | water | 93.3847 | -0.0081 | 95.9243 | 97.2432 | 20.3724 |
| vdd/vdd | FineRivalProjected_Exact | other | 54.7595 | 0.3846 | 73.6767 | 68.0788 | 24.8702 |
| vdd/vdd | FineRivalProjected_Exact | wall | 51.7573 | -5.8384 | 57.8563 | 83.0791 | 2.7991 |
| vdd/vdd | FineRivalProjected_Exact | road | 25.7526 | 1.9841 | 25.8194 | 99.0046 | 5.2889 |
| vdd/vdd | FineRivalProjected_Exact | vegetation | 64.5561 | 1.6952 | 97.0142 | 65.8648 | 14.6642 |
| vdd/vdd | FineRivalProjected_Exact | vehicle | 25.1452 | 0.1343 | 25.1551 | 99.8452 | 0.9377 |
| vdd/vdd | FineRivalProjected_Exact | roof | 86.3331 | -0.3244 | 87.6576 | 98.2800 | 31.1963 |
| vdd/vdd | FineRivalProjected_Exact | water | 93.5112 | 0.1184 | 96.2948 | 97.0014 | 20.2436 |
| vdd/vdd | FineTopTwo_Exact | other | 52.7405 | -1.6344 | 69.7045 | 68.4253 | 26.4212 |
| vdd/vdd | FineTopTwo_Exact | wall | 56.1290 | -1.4667 | 63.5009 | 82.8616 | 2.5436 |
| vdd/vdd | FineTopTwo_Exact | road | 26.0160 | 2.2475 | 26.1029 | 98.7366 | 5.2173 |
| vdd/vdd | FineTopTwo_Exact | vegetation | 64.6161 | 1.7552 | 96.8169 | 66.0186 | 14.7284 |
| vdd/vdd | FineTopTwo_Exact | vehicle | 25.2143 | 0.2034 | 25.2224 | 99.8735 | 0.9355 |
| vdd/vdd | FineTopTwo_Exact | roof | 84.3067 | -2.3508 | 87.1430 | 96.2829 | 30.7428 |
| vdd/vdd | FineTopTwo_Exact | water | 89.9183 | -3.4745 | 96.3624 | 93.0777 | 19.4111 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | other | 54.0239 | -0.3510 | 73.6180 | 66.9941 | 24.4934 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | wall | 55.4550 | -2.1407 | 62.8565 | 82.4852 | 2.5581 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | road | 23.5287 | -0.2398 | 23.5885 | 98.9332 | 5.7849 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | vegetation | 62.9769 | 0.1160 | 97.0624 | 64.2006 | 14.2866 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | vehicle | 24.9431 | -0.0678 | 24.9554 | 99.8020 | 0.9448 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | roof | 86.3816 | -0.2759 | 87.2473 | 98.8643 | 31.5293 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | water | 93.5181 | 0.1253 | 95.9239 | 97.3882 | 20.4029 |
| vdd/vdd | FineBudgetOnly_Exact | other | 55.3012 | 0.9263 | 73.6465 | 68.9446 | 25.1968 |
| vdd/vdd | FineBudgetOnly_Exact | wall | 53.6964 | -3.8993 | 60.5007 | 82.6822 | 2.6640 |
| vdd/vdd | FineBudgetOnly_Exact | road | 26.9416 | 3.1731 | 27.0238 | 98.8834 | 5.0470 |
| vdd/vdd | FineBudgetOnly_Exact | vegetation | 63.8367 | 0.9758 | 97.0967 | 65.0790 | 14.4770 |
| vdd/vdd | FineBudgetOnly_Exact | vehicle | 25.2451 | 0.2342 | 25.2617 | 99.7403 | 0.9328 |
| vdd/vdd | FineBudgetOnly_Exact | roof | 86.8564 | 0.1989 | 87.6316 | 98.9917 | 31.4315 |
| vdd/vdd | FineBudgetOnly_Exact | water | 93.0973 | -0.2955 | 96.0566 | 96.7968 | 20.2510 |
| potsdam/potsdam | Geometry | impervious surface | 51.6875 | -11.3225 | 75.3909 | 62.1781 | 29.8644 |
| potsdam/potsdam | Geometry | building | 78.2026 | 1.1178 | 80.6769 | 96.2262 | 14.5546 |
| potsdam/potsdam | Geometry | low vegetation | 33.4221 | 9.4461 | 72.4703 | 38.2826 | 10.1329 |
| potsdam/potsdam | Geometry | tree | 62.5813 | -3.8701 | 90.6538 | 66.8975 | 17.6734 |
| potsdam/potsdam | Geometry | car | 11.6582 | -13.1154 | 11.6716 | 99.0262 | 19.8762 |
| potsdam/potsdam | Geometry | clutter | 4.5652 | 1.7302 | 7.7447 | 10.0073 | 7.8986 |
| potsdam/potsdam | NoAdmission_Exact | impervious surface | 62.1577 | -0.8523 | 73.6731 | 79.9064 | 39.2742 |
| potsdam/potsdam | NoAdmission_Exact | building | 75.5788 | -1.5060 | 77.0240 | 97.5776 | 15.4589 |
| potsdam/potsdam | NoAdmission_Exact | low vegetation | 21.2307 | -2.7453 | 82.1270 | 22.2591 | 5.1989 |
| potsdam/potsdam | NoAdmission_Exact | tree | 64.8712 | -1.5802 | 92.4963 | 68.4748 | 17.7297 |
| potsdam/potsdam | NoAdmission_Exact | car | 24.2128 | -0.5608 | 24.3214 | 98.1896 | 9.4578 |
| potsdam/potsdam | NoAdmission_Exact | clutter | 2.6325 | -0.2025 | 3.7823 | 7.9698 | 12.8804 |
| potsdam/potsdam | RivalFineHard_Exact | impervious surface | 63.0100 | 0.0000 | 74.0136 | 80.9097 | 39.5844 |
| potsdam/potsdam | RivalFineHard_Exact | building | 77.0848 | 0.0000 | 78.5686 | 97.6086 | 15.1598 |
| potsdam/potsdam | RivalFineHard_Exact | low vegetation | 23.9760 | 0.0000 | 82.4162 | 25.2686 | 5.8811 |
| potsdam/potsdam | RivalFineHard_Exact | tree | 66.4514 | 0.0000 | 92.4527 | 70.2630 | 18.2014 |
| potsdam/potsdam | RivalFineHard_Exact | car | 24.7736 | 0.0000 | 24.8820 | 98.2723 | 9.2525 |
| potsdam/potsdam | RivalFineHard_Exact | clutter | 2.8350 | 0.0000 | 4.1705 | 8.1332 | 11.9208 |
| potsdam/potsdam | FineRivalPosterior_Exact | impervious surface | 63.2264 | 0.2164 | 74.1185 | 81.1407 | 39.6412 |
| potsdam/potsdam | FineRivalPosterior_Exact | building | 77.2586 | 0.1738 | 78.7420 | 97.6197 | 15.1281 |
| potsdam/potsdam | FineRivalPosterior_Exact | low vegetation | 24.5344 | 0.5584 | 82.2026 | 25.9108 | 6.0462 |
| potsdam/potsdam | FineRivalPosterior_Exact | tree | 66.6491 | 0.1977 | 92.4182 | 70.5040 | 18.2706 |
| potsdam/potsdam | FineRivalPosterior_Exact | car | 25.0038 | 0.2302 | 25.1113 | 98.3166 | 9.1722 |
| potsdam/potsdam | FineRivalPosterior_Exact | clutter | 2.8780 | 0.0430 | 4.2539 | 8.1710 | 11.7417 |
| potsdam/potsdam | FineActionProjected_Exact | impervious surface | 62.8480 | -0.1620 | 74.0200 | 80.6352 | 39.4466 |
| potsdam/potsdam | FineActionProjected_Exact | building | 77.1563 | 0.0715 | 78.6486 | 97.5998 | 15.1430 |
| potsdam/potsdam | FineActionProjected_Exact | low vegetation | 24.4691 | 0.4931 | 82.1384 | 25.8443 | 6.0354 |
| potsdam/potsdam | FineActionProjected_Exact | tree | 66.5127 | 0.0613 | 92.4134 | 70.3542 | 18.2327 |
| potsdam/potsdam | FineActionProjected_Exact | car | 24.6199 | -0.1537 | 24.7243 | 98.3144 | 9.3155 |
| potsdam/potsdam | FineActionProjected_Exact | clutter | 2.8934 | 0.0584 | 4.2655 | 8.2526 | 11.8266 |
| potsdam/potsdam | FineRivalProjected_Exact | impervious surface | 63.0462 | 0.0362 | 74.0633 | 80.9100 | 39.5580 |
| potsdam/potsdam | FineRivalProjected_Exact | building | 77.3306 | 0.2458 | 78.8193 | 97.6159 | 15.1127 |
| potsdam/potsdam | FineRivalProjected_Exact | low vegetation | 24.9663 | 0.9903 | 81.8924 | 26.4250 | 6.1896 |
| potsdam/potsdam | FineRivalProjected_Exact | tree | 66.7002 | 0.2488 | 92.3835 | 70.5815 | 18.2976 |
| potsdam/potsdam | FineRivalProjected_Exact | car | 24.8861 | 0.1125 | 24.9902 | 98.3545 | 9.2202 |
| potsdam/potsdam | FineRivalProjected_Exact | clutter | 2.9529 | 0.1179 | 4.3768 | 8.3215 | 11.6220 |
| potsdam/potsdam | FineTopTwo_Exact | impervious surface | 63.6025 | 0.5925 | 74.1432 | 81.7311 | 39.9164 |
| potsdam/potsdam | FineTopTwo_Exact | building | 76.8171 | -0.2677 | 78.2375 | 97.6911 | 15.2368 |
| potsdam/potsdam | FineTopTwo_Exact | low vegetation | 25.1563 | 1.1803 | 82.7476 | 26.5487 | 6.1543 |
| potsdam/potsdam | FineTopTwo_Exact | tree | 67.3306 | 0.8792 | 92.2585 | 71.3625 | 18.5251 |
| potsdam/potsdam | FineTopTwo_Exact | car | 25.9860 | 1.2124 | 26.0999 | 98.3491 | 8.8277 |
| potsdam/potsdam | FineTopTwo_Exact | clutter | 2.8382 | 0.0032 | 4.2476 | 7.8796 | 11.3398 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | impervious surface | 63.1566 | 0.1466 | 74.0782 | 81.0740 | 39.6302 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | building | 77.0747 | -0.0101 | 78.5611 | 97.6040 | 15.1605 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | low vegetation | 24.1450 | 0.1690 | 82.4369 | 25.4544 | 5.9229 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | tree | 66.5187 | 0.0673 | 92.4569 | 70.3358 | 18.2194 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | car | 24.9404 | 0.1668 | 25.0495 | 98.2824 | 9.1916 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | clutter | 2.8505 | 0.0155 | 4.1981 | 8.1557 | 11.8754 |
| potsdam/potsdam | FineBudgetOnly_Exact | impervious surface | 62.7128 | -0.2972 | 74.0412 | 80.3877 | 39.3144 |
| potsdam/potsdam | FineBudgetOnly_Exact | building | 77.2987 | 0.2139 | 78.8027 | 97.5905 | 15.1120 |
| potsdam/potsdam | FineBudgetOnly_Exact | low vegetation | 24.9493 | 0.9733 | 81.8239 | 26.4131 | 6.1920 |
| potsdam/potsdam | FineBudgetOnly_Exact | tree | 66.5797 | 0.1283 | 92.4021 | 70.4358 | 18.2561 |
| potsdam/potsdam | FineBudgetOnly_Exact | car | 24.4859 | -0.2877 | 24.5870 | 98.3481 | 9.3707 |
| potsdam/potsdam | FineBudgetOnly_Exact | clutter | 2.9527 | 0.1177 | 4.3595 | 8.3833 | 11.7548 |
| udd5/udd5 | Geometry | vegetation | 82.4937 | 9.0324 | 96.4862 | 85.0488 | 26.1079 |
| udd5/udd5 | Geometry | building | 83.9035 | -0.7893 | 88.3604 | 94.3293 | 41.9106 |
| udd5/udd5 | Geometry | road | 45.7856 | 2.1332 | 70.6780 | 56.5218 | 10.7242 |
| udd5/udd5 | Geometry | vehicle | 10.0092 | -16.0667 | 10.0323 | 97.7477 | 7.7994 |
| udd5/udd5 | Geometry | other | 30.5846 | -6.3144 | 52.8537 | 42.0591 | 13.4578 |
| udd5/udd5 | NoAdmission_Exact | vegetation | 68.3847 | -5.0766 | 97.4282 | 69.6418 | 21.1716 |
| udd5/udd5 | NoAdmission_Exact | building | 82.5264 | -2.1664 | 83.6805 | 98.3563 | 46.1437 |
| udd5/udd5 | NoAdmission_Exact | road | 39.8083 | -3.8441 | 66.8744 | 49.5860 | 9.9434 |
| udd5/udd5 | NoAdmission_Exact | vehicle | 20.4213 | -5.6546 | 20.9505 | 88.9918 | 3.4003 |
| udd5/udd5 | NoAdmission_Exact | other | 34.9177 | -1.9813 | 48.5109 | 55.4791 | 19.3411 |
| udd5/udd5 | RivalFineHard_Exact | vegetation | 73.4613 | 0.0000 | 97.6728 | 74.7700 | 22.6737 |
| udd5/udd5 | RivalFineHard_Exact | building | 84.6928 | 0.0000 | 85.7573 | 98.5555 | 45.1174 |
| udd5/udd5 | RivalFineHard_Exact | road | 43.6524 | 0.0000 | 68.8074 | 54.4220 | 10.6065 |
| udd5/udd5 | RivalFineHard_Exact | vehicle | 26.0759 | 0.0000 | 26.5277 | 93.8692 | 2.8326 |
| udd5/udd5 | RivalFineHard_Exact | other | 36.8990 | 0.0000 | 51.2389 | 56.8679 | 18.7698 |
| udd5/udd5 | FineRivalPosterior_Exact | vegetation | 73.8847 | 0.4234 | 97.6880 | 75.1997 | 22.8005 |
| udd5/udd5 | FineRivalPosterior_Exact | building | 84.8969 | 0.2041 | 86.0437 | 98.4544 | 44.9211 |
| udd5/udd5 | FineRivalPosterior_Exact | road | 43.7995 | 0.1471 | 68.9854 | 54.5391 | 10.6019 |
| udd5/udd5 | FineRivalPosterior_Exact | vehicle | 26.7132 | 0.6373 | 27.1434 | 94.3993 | 2.7839 |
| udd5/udd5 | FineRivalPosterior_Exact | other | 36.8768 | -0.0222 | 51.0587 | 57.0386 | 18.8925 |
| udd5/udd5 | FineActionProjected_Exact | vegetation | 74.2073 | 0.7460 | 97.6744 | 75.5421 | 22.9075 |
| udd5/udd5 | FineActionProjected_Exact | building | 84.7731 | 0.0803 | 85.8320 | 98.5656 | 45.0828 |
| udd5/udd5 | FineActionProjected_Exact | road | 43.2621 | -0.3903 | 70.1029 | 53.0498 | 10.1480 |
| udd5/udd5 | FineActionProjected_Exact | vehicle | 25.3323 | -0.7436 | 25.7248 | 94.3195 | 2.9350 |
| udd5/udd5 | FineActionProjected_Exact | other | 37.0262 | 0.1272 | 51.1659 | 57.2619 | 18.9267 |
| udd5/udd5 | FineRivalProjected_Exact | vegetation | 74.5093 | 1.0480 | 97.6926 | 75.8441 | 22.9948 |
| udd5/udd5 | FineRivalProjected_Exact | building | 84.9284 | 0.2356 | 86.0908 | 98.4351 | 44.8878 |
| udd5/udd5 | FineRivalProjected_Exact | road | 43.6159 | -0.0365 | 70.1931 | 53.5304 | 10.2268 |
| udd5/udd5 | FineRivalProjected_Exact | vehicle | 25.8748 | -0.2011 | 26.2495 | 94.7708 | 2.8901 |
| udd5/udd5 | FineRivalProjected_Exact | other | 37.0450 | 0.1460 | 51.0910 | 57.4011 | 19.0006 |
| udd5/udd5 | FineTopTwo_Exact | vegetation | 72.2129 | -1.2484 | 97.6486 | 73.4908 | 22.2914 |
| udd5/udd5 | FineTopTwo_Exact | building | 84.5268 | -0.1660 | 86.0137 | 97.9959 | 44.7275 |
| udd5/udd5 | FineTopTwo_Exact | road | 42.4358 | -1.2166 | 67.9294 | 53.0678 | 10.4763 |
| udd5/udd5 | FineTopTwo_Exact | vehicle | 25.5658 | -0.5101 | 25.9619 | 94.3687 | 2.9097 |
| udd5/udd5 | FineTopTwo_Exact | other | 35.7624 | -1.1366 | 49.0766 | 56.8633 | 19.5951 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | vegetation | 73.7894 | 0.3281 | 97.6763 | 75.1079 | 22.7754 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | building | 84.7728 | 0.0800 | 85.8851 | 98.4952 | 45.0227 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | road | 43.9556 | 0.3032 | 68.8078 | 54.8939 | 10.6984 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | vehicle | 26.4618 | 0.3859 | 26.9050 | 94.1396 | 2.8009 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | other | 36.9801 | 0.0811 | 51.4085 | 56.8520 | 18.7026 |
| udd5/udd5 | FineBudgetOnly_Exact | vegetation | 74.7410 | 1.2797 | 97.6843 | 76.0892 | 23.0711 |
| udd5/udd5 | FineBudgetOnly_Exact | building | 84.9177 | 0.2249 | 85.9884 | 98.5548 | 44.9958 |
| udd5/udd5 | FineBudgetOnly_Exact | road | 42.9557 | -0.6967 | 71.0536 | 52.0673 | 9.8268 |
| udd5/udd5 | FineBudgetOnly_Exact | vehicle | 25.0541 | -1.0218 | 25.4126 | 94.6698 | 2.9821 |
| udd5/udd5 | FineBudgetOnly_Exact | other | 37.1420 | 0.2430 | 51.0327 | 57.7087 | 19.1242 |
| oem/oem | Geometry | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.4255 |
| oem/oem | Geometry | rangeland | 47.7780 | 10.0146 | 65.2945 | 64.0414 | 18.7786 |
| oem/oem | Geometry | developed space | 28.5294 | 4.5256 | 61.5617 | 34.7130 | 12.6303 |
| oem/oem | Geometry | road | 40.8291 | -1.9438 | 45.7530 | 79.1399 | 7.8925 |
| oem/oem | Geometry | tree | 56.0657 | 15.1883 | 87.4475 | 60.9727 | 17.7308 |
| oem/oem | Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0647 |
| oem/oem | Geometry | agriculture land | 79.1353 | 9.9570 | 90.6413 | 86.1766 | 15.9338 |
| oem/oem | Geometry | building | 62.4971 | 13.2809 | 65.5844 | 92.9954 | 16.5437 |
| oem/oem | NoAdmission_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.6586 |
| oem/oem | NoAdmission_Exact | rangeland | 37.5959 | -0.1675 | 60.8653 | 49.5812 | 15.5965 |
| oem/oem | NoAdmission_Exact | developed space | 24.0530 | 0.0492 | 36.5063 | 41.3526 | 25.3727 |
| oem/oem | NoAdmission_Exact | road | 42.1205 | -0.6524 | 53.7385 | 66.0817 | 5.6109 |
| oem/oem | NoAdmission_Exact | tree | 38.7562 | -2.1212 | 92.7506 | 39.9668 | 10.9578 |
| oem/oem | NoAdmission_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | NoAdmission_Exact | agriculture land | 68.8611 | -0.3172 | 70.3827 | 96.9561 | 23.0869 |
| oem/oem | NoAdmission_Exact | building | 44.2393 | -4.9769 | 77.0443 | 50.9559 | 7.7166 |
| oem/oem | RivalFineHard_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.9888 |
| oem/oem | RivalFineHard_Exact | rangeland | 37.7634 | 0.0000 | 61.3289 | 49.5660 | 15.4738 |
| oem/oem | RivalFineHard_Exact | developed space | 24.0038 | 0.0000 | 37.5709 | 39.9301 | 23.8057 |
| oem/oem | RivalFineHard_Exact | road | 42.7729 | 0.0000 | 54.3267 | 66.7909 | 5.6097 |
| oem/oem | RivalFineHard_Exact | tree | 40.8774 | 0.0000 | 92.4159 | 42.2963 | 11.6385 |
| oem/oem | RivalFineHard_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | RivalFineHard_Exact | agriculture land | 69.1783 | 0.0000 | 70.7186 | 96.9477 | 22.9753 |
| oem/oem | RivalFineHard_Exact | building | 49.2162 | 0.0000 | 78.2129 | 57.0356 | 8.5082 |
| oem/oem | FineRivalPosterior_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.1869 |
| oem/oem | FineRivalPosterior_Exact | rangeland | 37.8513 | 0.0879 | 61.4412 | 49.6438 | 15.4698 |
| oem/oem | FineRivalPosterior_Exact | developed space | 23.7513 | -0.2525 | 37.9495 | 38.8317 | 22.9199 |
| oem/oem | FineRivalPosterior_Exact | road | 43.0111 | 0.2382 | 54.4636 | 67.1640 | 5.6269 |
| oem/oem | FineRivalPosterior_Exact | tree | 41.4449 | 0.5675 | 92.3021 | 42.9287 | 11.8271 |
| oem/oem | FineRivalPosterior_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FineRivalPosterior_Exact | agriculture land | 69.4020 | 0.2237 | 70.9313 | 96.9871 | 22.9157 |
| oem/oem | FineRivalPosterior_Exact | building | 52.1674 | 2.9512 | 78.4624 | 60.8862 | 9.0538 |
| oem/oem | FineActionProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.9618 |
| oem/oem | FineActionProjected_Exact | rangeland | 37.8906 | 0.1272 | 61.4108 | 49.7314 | 15.5047 |
| oem/oem | FineActionProjected_Exact | developed space | 24.1518 | 0.1480 | 37.9337 | 39.9314 | 23.5788 |
| oem/oem | FineActionProjected_Exact | road | 42.9071 | 0.1342 | 54.5267 | 66.8157 | 5.5912 |
| oem/oem | FineActionProjected_Exact | tree | 41.4348 | 0.5574 | 92.3283 | 42.9122 | 11.8192 |
| oem/oem | FineActionProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| oem/oem | FineActionProjected_Exact | agriculture land | 69.3879 | 0.2096 | 70.9354 | 96.9518 | 22.9060 |
| oem/oem | FineActionProjected_Exact | building | 49.8339 | 0.6177 | 78.1825 | 57.8835 | 8.6381 |
| oem/oem | FineRivalProjected_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.1312 |
| oem/oem | FineRivalProjected_Exact | rangeland | 38.0036 | 0.2402 | 61.5401 | 49.8411 | 15.5063 |
| oem/oem | FineRivalProjected_Exact | developed space | 23.9263 | -0.0775 | 38.3556 | 38.8754 | 22.7027 |
| oem/oem | FineRivalProjected_Exact | road | 43.1018 | 0.3289 | 54.5579 | 67.2416 | 5.6237 |
| oem/oem | FineRivalProjected_Exact | tree | 41.9865 | 1.1091 | 92.1997 | 43.5329 | 12.0068 |
| oem/oem | FineRivalProjected_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| oem/oem | FineRivalProjected_Exact | agriculture land | 69.6417 | 0.4634 | 71.1862 | 96.9786 | 22.8316 |
| oem/oem | FineRivalProjected_Exact | building | 52.7625 | 3.5463 | 78.3523 | 61.7665 | 9.1976 |
| oem/oem | FineTopTwo_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12.3686 |
| oem/oem | FineTopTwo_Exact | rangeland | 37.7357 | -0.0277 | 61.2766 | 49.5523 | 15.4827 |
| oem/oem | FineTopTwo_Exact | developed space | 23.1759 | -0.8279 | 38.2846 | 36.9985 | 21.6467 |
| oem/oem | FineTopTwo_Exact | road | 42.4866 | -0.2863 | 53.0503 | 68.0884 | 5.8563 |
| oem/oem | FineTopTwo_Exact | tree | 41.5856 | 0.7082 | 92.0233 | 43.1407 | 11.9215 |
| oem/oem | FineTopTwo_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FineTopTwo_Exact | agriculture land | 69.4147 | 0.2364 | 70.9323 | 97.0099 | 22.9208 |
| oem/oem | FineTopTwo_Exact | building | 54.7504 | 5.5342 | 77.4864 | 65.1075 | 9.8034 |
| oem/oem | FinePosteriorStrengthMatched_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.9923 |
| oem/oem | FinePosteriorStrengthMatched_Exact | rangeland | 37.8319 | 0.0685 | 61.4265 | 49.6201 | 15.4661 |
| oem/oem | FinePosteriorStrengthMatched_Exact | developed space | 24.1023 | 0.0985 | 37.8841 | 39.8508 | 23.5620 |
| oem/oem | FinePosteriorStrengthMatched_Exact | road | 42.7713 | -0.0016 | 54.3263 | 66.7874 | 5.6095 |
| oem/oem | FinePosteriorStrengthMatched_Exact | tree | 41.1821 | 0.3047 | 92.3521 | 42.6361 | 11.7401 |
| oem/oem | FinePosteriorStrengthMatched_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | FinePosteriorStrengthMatched_Exact | agriculture land | 69.2353 | 0.0570 | 70.7811 | 96.9422 | 22.9537 |
| oem/oem | FinePosteriorStrengthMatched_Exact | building | 50.2098 | 0.9936 | 78.3761 | 58.2837 | 8.6763 |
| oem/oem | FineBudgetOnly_Exact | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.9399 |
| oem/oem | FineBudgetOnly_Exact | rangeland | 38.0056 | 0.2422 | 61.4839 | 49.8815 | 15.5331 |
| oem/oem | FineBudgetOnly_Exact | developed space | 24.3072 | 0.3034 | 38.2590 | 39.9961 | 23.4162 |
| oem/oem | FineBudgetOnly_Exact | road | 43.0763 | 0.3034 | 54.8022 | 66.8128 | 5.5629 |
| oem/oem | FineBudgetOnly_Exact | tree | 41.9922 | 1.1148 | 92.2473 | 43.5284 | 11.9994 |
| oem/oem | FineBudgetOnly_Exact | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 |
| oem/oem | FineBudgetOnly_Exact | agriculture land | 69.6179 | 0.4396 | 71.1757 | 96.9520 | 22.8287 |
| oem/oem | FineBudgetOnly_Exact | building | 50.2671 | 1.0509 | 78.2132 | 58.4516 | 8.7194 |
| loveda/P | Geometry | building | 64.4987 | -15.0478 | 65.3671 | 97.9819 | 0.9799 |
| loveda/P | Geometry | road | 66.9348 | -4.0653 | 67.6522 | 98.4404 | 10.3611 |
| loveda/P | Geometry | water | 81.1851 | 1.6464 | 86.7638 | 92.6613 | 22.3129 |
| loveda/P | Geometry | barren | 24.3403 | 0.1850 | 63.8735 | 28.2261 | 3.2301 |
| loveda/P | Geometry | tree | 53.7307 | 9.1362 | 64.5184 | 76.2666 | 13.0831 |
| loveda/P | Geometry | farm | 86.2626 | -2.0767 | 95.3302 | 90.0685 | 50.0330 |
| loveda/P | NoAdmission_Exact | building | 79.5953 | 0.0488 | 96.0602 | 82.2814 | 0.5600 |
| loveda/P | NoAdmission_Exact | road | 68.0988 | -2.9013 | 69.1302 | 97.8560 | 10.0793 |
| loveda/P | NoAdmission_Exact | water | 79.3816 | -0.1571 | 85.6142 | 91.5997 | 22.3534 |
| loveda/P | NoAdmission_Exact | barren | 23.8204 | -0.3349 | 74.5721 | 25.9262 | 2.5412 |
| loveda/P | NoAdmission_Exact | tree | 36.7931 | -7.8014 | 99.0340 | 36.9257 | 4.1267 |
| loveda/P | NoAdmission_Exact | farm | 87.2288 | -1.1105 | 87.4779 | 99.6747 | 60.3393 |
| loveda/P | RivalFineHard_Exact | building | 79.5465 | 0.0000 | 96.3094 | 82.0475 | 0.5569 |
| loveda/P | RivalFineHard_Exact | road | 71.0001 | 0.0000 | 71.9939 | 98.0928 | 9.7018 |
| loveda/P | RivalFineHard_Exact | water | 79.5387 | 0.0000 | 85.3266 | 92.1420 | 22.5615 |
| loveda/P | RivalFineHard_Exact | barren | 24.1553 | 0.0000 | 75.2651 | 26.2382 | 2.5481 |
| loveda/P | RivalFineHard_Exact | tree | 44.5945 | 0.0000 | 98.5986 | 44.8789 | 5.0377 |
| loveda/P | RivalFineHard_Exact | farm | 88.3393 | 0.0000 | 88.5841 | 99.6882 | 59.5938 |
| loveda/P | FineRivalPosterior_Exact | building | 79.6566 | 0.1101 | 96.3326 | 82.1477 | 0.5575 |
| loveda/P | FineRivalPosterior_Exact | road | 71.4259 | 0.4258 | 72.4093 | 98.1339 | 9.6502 |
| loveda/P | FineRivalPosterior_Exact | water | 79.6291 | 0.0904 | 85.2908 | 92.3051 | 22.6110 |
| loveda/P | FineRivalPosterior_Exact | barren | 24.3892 | 0.2339 | 74.2796 | 26.6389 | 2.6214 |
| loveda/P | FineRivalPosterior_Exact | tree | 46.0027 | 1.4082 | 98.5245 | 46.3219 | 5.2036 |
| loveda/P | FineRivalPosterior_Exact | farm | 88.6798 | 0.3405 | 88.9322 | 99.6811 | 59.3564 |
| loveda/P | FineActionProjected_Exact | building | 79.9031 | 0.3566 | 96.0531 | 82.6155 | 0.5623 |
| loveda/P | FineActionProjected_Exact | road | 71.5324 | 0.5323 | 72.5143 | 98.1422 | 9.6371 |
| loveda/P | FineActionProjected_Exact | water | 79.7198 | 0.1811 | 85.3839 | 92.3180 | 22.5895 |
| loveda/P | FineActionProjected_Exact | barren | 24.2267 | 0.0714 | 75.7980 | 26.2579 | 2.5321 |
| loveda/P | FineActionProjected_Exact | tree | 46.1823 | 1.5878 | 98.3579 | 46.5411 | 5.2371 |
| loveda/P | FineActionProjected_Exact | farm | 88.5592 | 0.2199 | 88.8077 | 99.6851 | 59.4420 |
| loveda/P | FineRivalProjected_Exact | building | 80.0000 | 0.4535 | 96.0217 | 82.7425 | 0.5633 |
| loveda/P | FineRivalProjected_Exact | road | 71.8874 | 0.8873 | 72.8631 | 98.1713 | 9.5938 |
| loveda/P | FineRivalProjected_Exact | water | 79.7760 | 0.2373 | 85.3372 | 92.4481 | 22.6336 |
| loveda/P | FineRivalProjected_Exact | barren | 24.4483 | 0.2930 | 74.3275 | 26.7032 | 2.6260 |
| loveda/P | FineRivalProjected_Exact | tree | 47.4840 | 2.8895 | 98.3052 | 47.8759 | 5.3902 |
| loveda/P | FineRivalProjected_Exact | farm | 88.9166 | 0.5773 | 89.1737 | 99.6768 | 59.1931 |
| loveda/P | FineTopTwo_Exact | building | 80.4062 | 0.8597 | 95.9909 | 83.2002 | 0.5666 |
| loveda/P | FineTopTwo_Exact | road | 70.9642 | -0.0359 | 71.8961 | 98.2063 | 9.7263 |
| loveda/P | FineTopTwo_Exact | water | 79.6078 | 0.0691 | 84.9939 | 92.6266 | 22.7690 |
| loveda/P | FineTopTwo_Exact | barren | 24.6125 | 0.4572 | 70.4234 | 27.4500 | 2.8491 |
| loveda/P | FineTopTwo_Exact | tree | 46.2672 | 1.6727 | 98.9988 | 46.4847 | 5.1969 |
| loveda/P | FineTopTwo_Exact | farm | 88.9315 | 0.5922 | 89.3968 | 99.4182 | 58.8922 |
| loveda/P | FinePosteriorStrengthMatched_Exact | building | 79.4992 | -0.0473 | 96.2999 | 82.0041 | 0.5567 |
| loveda/P | FinePosteriorStrengthMatched_Exact | road | 71.1279 | 0.1278 | 72.1334 | 98.0778 | 9.6816 |
| loveda/P | FinePosteriorStrengthMatched_Exact | water | 79.5147 | -0.0240 | 85.3043 | 92.1358 | 22.5659 |
| loveda/P | FinePosteriorStrengthMatched_Exact | barren | 23.8719 | -0.2834 | 74.1482 | 26.0392 | 2.5669 |
| loveda/P | FinePosteriorStrengthMatched_Exact | tree | 44.6810 | 0.0865 | 98.7279 | 44.9397 | 5.0379 |
| loveda/P | FinePosteriorStrengthMatched_Exact | farm | 88.3457 | 0.0064 | 88.5895 | 99.6895 | 59.5910 |
| loveda/P | FineBudgetOnly_Exact | building | 80.3566 | 0.8101 | 95.8138 | 83.2804 | 0.5682 |
| loveda/P | FineBudgetOnly_Exact | road | 72.0799 | 1.0798 | 73.0561 | 98.1799 | 9.5693 |
| loveda/P | FineBudgetOnly_Exact | water | 79.9082 | 0.3695 | 85.4533 | 92.4894 | 22.6130 |
| loveda/P | FineBudgetOnly_Exact | barren | 24.3846 | 0.2293 | 76.4762 | 26.3619 | 2.5196 |
| loveda/P | FineBudgetOnly_Exact | tree | 47.8516 | 3.2571 | 98.0773 | 48.3048 | 5.4511 |
| loveda/P | FineBudgetOnly_Exact | farm | 88.7960 | 0.4567 | 89.0488 | 99.6814 | 59.2788 |
| loveda/D | Geometry | background | 34.1983 | 11.0783 | 65.7504 | 41.6109 | 28.2483 |
| loveda/D | Geometry | building | 31.1862 | -5.3197 | 31.8582 | 93.6650 | 1.0641 |
| loveda/D | Geometry | road | 45.4818 | -13.8894 | 45.8348 | 98.3352 | 8.4577 |
| loveda/D | Geometry | water | 61.9193 | -0.0187 | 65.9551 | 91.0065 | 15.9605 |
| loveda/D | Geometry | barren | 14.5296 | -5.3978 | 59.0631 | 16.1567 | 1.1070 |
| loveda/D | Geometry | tree | 26.8904 | -8.5924 | 30.0530 | 71.8726 | 14.6543 |
| loveda/D | Geometry | farm | 54.9851 | 6.7071 | 69.5721 | 72.3948 | 30.5080 |
| loveda/D | NoAdmission_Exact | background | 6.4379 | -16.6821 | 78.1616 | 6.5558 | 3.7438 |
| loveda/D | NoAdmission_Exact | building | 36.3815 | -0.1244 | 39.8770 | 80.5840 | 0.7314 |
| loveda/D | NoAdmission_Exact | road | 56.4727 | -2.8985 | 57.1824 | 97.8495 | 6.7459 |
| loveda/D | NoAdmission_Exact | water | 62.7475 | 0.8095 | 66.6173 | 91.5269 | 15.8922 |
| loveda/D | NoAdmission_Exact | barren | 20.0167 | 0.0893 | 48.3384 | 25.4642 | 2.1318 |
| loveda/D | NoAdmission_Exact | tree | 31.2897 | -4.1931 | 87.2023 | 32.7957 | 2.3045 |
| loveda/D | NoAdmission_Exact | farm | 42.5737 | -5.7043 | 42.6508 | 99.5774 | 68.4503 |
| loveda/D | RivalFineHard_Exact | background | 23.1200 | 0.0000 | 86.6691 | 23.9725 | 12.3462 |
| loveda/D | RivalFineHard_Exact | building | 36.5059 | 0.0000 | 40.3043 | 79.4814 | 0.7138 |
| loveda/D | RivalFineHard_Exact | road | 59.3712 | 0.0000 | 60.0572 | 98.1124 | 6.4402 |
| loveda/D | RivalFineHard_Exact | water | 61.9380 | 0.0000 | 65.8706 | 91.2084 | 16.0165 |
| loveda/D | RivalFineHard_Exact | barren | 19.9274 | 0.0000 | 48.0720 | 25.3937 | 2.1377 |
| loveda/D | RivalFineHard_Exact | tree | 35.4828 | 0.0000 | 84.0461 | 38.0453 | 2.7738 |
| loveda/D | RivalFineHard_Exact | farm | 48.2780 | 0.0000 | 48.5832 | 98.7155 | 59.5719 |
| loveda/D | FineRivalPosterior_Exact | background | 28.1696 | 5.0496 | 85.4740 | 29.5860 | 15.4503 |
| loveda/D | FineRivalPosterior_Exact | building | 36.4818 | -0.0241 | 40.2611 | 79.5349 | 0.7150 |
| loveda/D | FineRivalPosterior_Exact | road | 59.5435 | 0.1723 | 60.2171 | 98.1560 | 6.4260 |
| loveda/D | FineRivalPosterior_Exact | water | 61.9958 | 0.0578 | 65.9528 | 91.1762 | 15.9909 |
| loveda/D | FineRivalPosterior_Exact | barren | 20.0426 | 0.1152 | 48.2062 | 25.5431 | 2.1443 |
| loveda/D | FineRivalPosterior_Exact | tree | 36.2137 | 0.7309 | 84.1359 | 38.8677 | 2.8307 |
| loveda/D | FineRivalPosterior_Exact | farm | 49.5198 | 1.2418 | 50.3226 | 96.8791 | 56.4429 |
| loveda/D | FineActionProjected_Exact | background | 24.0032 | 0.8832 | 86.8309 | 24.9101 | 12.8052 |
| loveda/D | FineActionProjected_Exact | building | 36.6463 | 0.1404 | 40.2756 | 80.2633 | 0.7213 |
| loveda/D | FineActionProjected_Exact | road | 59.3061 | -0.0651 | 59.9731 | 98.1591 | 6.4523 |
| loveda/D | FineActionProjected_Exact | water | 62.5857 | 0.6477 | 66.2718 | 91.8382 | 16.0294 |
| loveda/D | FineActionProjected_Exact | barren | 19.9923 | 0.0649 | 48.7629 | 25.3088 | 2.1003 |
| loveda/D | FineActionProjected_Exact | tree | 36.5290 | 1.0462 | 83.1287 | 39.4541 | 2.9082 |
| loveda/D | FineActionProjected_Exact | farm | 48.5532 | 0.2752 | 48.9301 | 98.4381 | 58.9832 |
| loveda/D | FineRivalProjected_Exact | background | 28.7066 | 5.5866 | 85.8123 | 30.1370 | 15.6760 |
| loveda/D | FineRivalProjected_Exact | building | 36.6843 | 0.1784 | 40.2846 | 80.4103 | 0.7225 |
| loveda/D | FineRivalProjected_Exact | road | 59.4775 | 0.1063 | 60.1359 | 98.1925 | 6.4370 |
| loveda/D | FineRivalProjected_Exact | water | 62.5320 | 0.5940 | 66.2677 | 91.7304 | 16.0116 |
| loveda/D | FineRivalProjected_Exact | barren | 20.1811 | 0.2537 | 48.9504 | 25.5607 | 2.1131 |
| loveda/D | FineRivalProjected_Exact | tree | 37.2648 | 1.7820 | 83.1591 | 40.3067 | 2.9700 |
| loveda/D | FineRivalProjected_Exact | farm | 49.7720 | 1.4940 | 50.6086 | 96.7857 | 56.0698 |
| loveda/D | FineTopTwo_Exact | background | 35.0494 | 11.9294 | 77.8949 | 38.9206 | 22.3026 |
| loveda/D | FineTopTwo_Exact | building | 36.8567 | 0.3508 | 40.2089 | 81.5530 | 0.7341 |
| loveda/D | FineTopTwo_Exact | road | 58.6369 | -0.7343 | 59.2850 | 98.1698 | 6.5279 |
| loveda/D | FineTopTwo_Exact | water | 61.8601 | -0.0779 | 65.7290 | 91.3114 | 16.0691 |
| loveda/D | FineTopTwo_Exact | barren | 20.1591 | 0.2317 | 48.1711 | 25.7427 | 2.1626 |
| loveda/D | FineTopTwo_Exact | tree | 34.3649 | -1.1179 | 87.7722 | 36.0928 | 2.5197 |
| loveda/D | FineTopTwo_Exact | farm | 48.5105 | 0.2325 | 51.9401 | 88.0193 | 49.6840 |
| loveda/D | FinePosteriorStrengthMatched_Exact | background | 25.0041 | 1.8841 | 86.6903 | 26.0023 | 13.3883 |
| loveda/D | FinePosteriorStrengthMatched_Exact | building | 36.4919 | -0.0140 | 40.3035 | 79.4180 | 0.7132 |
| loveda/D | FinePosteriorStrengthMatched_Exact | road | 59.4542 | 0.0830 | 60.1411 | 98.1152 | 6.4314 |
| loveda/D | FinePosteriorStrengthMatched_Exact | water | 62.0144 | 0.0764 | 65.9767 | 91.1709 | 15.9841 |
| loveda/D | FinePosteriorStrengthMatched_Exact | barren | 19.8773 | -0.0501 | 48.1007 | 25.3043 | 2.1289 |
| loveda/D | FinePosteriorStrengthMatched_Exact | tree | 35.5121 | 0.0293 | 83.9440 | 38.1000 | 2.7811 |
| loveda/D | FinePosteriorStrengthMatched_Exact | farm | 48.7828 | 0.5048 | 49.1998 | 98.2921 | 58.5729 |
| loveda/D | FineBudgetOnly_Exact | background | 25.5816 | 2.4616 | 86.9462 | 26.6034 | 13.6575 |
| loveda/D | FineBudgetOnly_Exact | building | 37.0352 | 0.5293 | 40.4576 | 81.4060 | 0.7283 |
| loveda/D | FineBudgetOnly_Exact | road | 59.1774 | -0.1938 | 59.8285 | 98.1944 | 6.4702 |
| loveda/D | FineBudgetOnly_Exact | water | 63.0696 | 1.1316 | 66.6443 | 92.1619 | 15.9960 |
| loveda/D | FineBudgetOnly_Exact | barren | 20.1520 | 0.2246 | 49.8201 | 25.2840 | 2.0538 |
| loveda/D | FineBudgetOnly_Exact | tree | 37.5678 | 2.0850 | 82.3446 | 40.8589 | 3.0405 |
| loveda/D | FineBudgetOnly_Exact | farm | 49.1068 | 0.8288 | 49.5664 | 98.1468 | 58.0538 |
| vaihingen/vaihingen | Geometry | impervious surface | 49.2002 | -12.4205 | 82.3971 | 54.9789 | 20.4283 |
| vaihingen/vaihingen | Geometry | building | 74.6610 | 4.8403 | 75.5001 | 98.5334 | 28.1288 |
| vaihingen/vaihingen | Geometry | low vegetation | 46.7115 | 9.8273 | 92.0732 | 48.6687 | 13.4530 |
| vaihingen/vaihingen | Geometry | tree | 71.8329 | -0.2664 | 82.5509 | 84.6923 | 21.4401 |
| vaihingen/vaihingen | Geometry | car | 8.7566 | -14.0617 | 8.7725 | 97.9748 | 16.5499 |
| vaihingen/vaihingen | NoAdmission_Exact | impervious surface | 60.0516 | -1.5691 | 74.5909 | 75.4952 | 30.9871 |
| vaihingen/vaihingen | NoAdmission_Exact | building | 68.2387 | -1.5820 | 68.4663 | 99.5152 | 31.3277 |
| vaihingen/vaihingen | NoAdmission_Exact | low vegetation | 35.3912 | -1.4930 | 96.4478 | 35.8586 | 9.4625 |
| vaihingen/vaihingen | NoAdmission_Exact | tree | 71.8529 | -0.2464 | 81.8858 | 85.4321 | 21.8030 |
| vaihingen/vaihingen | NoAdmission_Exact | car | 21.7110 | -1.1073 | 21.9557 | 95.1174 | 6.4197 |
| vaihingen/vaihingen | RivalFineHard_Exact | impervious surface | 61.6207 | 0.0000 | 75.6239 | 76.8936 | 31.1300 |
| vaihingen/vaihingen | RivalFineHard_Exact | building | 69.8207 | 0.0000 | 70.0761 | 99.4806 | 30.5974 |
| vaihingen/vaihingen | RivalFineHard_Exact | low vegetation | 36.8842 | 0.0000 | 95.8237 | 37.4868 | 9.9565 |
| vaihingen/vaihingen | RivalFineHard_Exact | tree | 72.0993 | 0.0000 | 81.3783 | 86.3448 | 22.1734 |
| vaihingen/vaihingen | RivalFineHard_Exact | car | 22.8183 | 0.0000 | 23.0608 | 95.5947 | 6.1428 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | impervious surface | 62.0890 | 0.4683 | 75.8771 | 77.3593 | 31.2139 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | building | 70.0337 | 0.2130 | 70.2895 | 99.4831 | 30.5052 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | low vegetation | 37.2366 | 0.3524 | 95.8648 | 37.8445 | 10.0472 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | tree | 72.1088 | 0.0095 | 81.3270 | 86.4164 | 22.2057 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | car | 23.3159 | 0.4976 | 23.5556 | 95.8188 | 6.0278 |
| vaihingen/vaihingen | FineActionProjected_Exact | impervious surface | 61.9060 | 0.2853 | 76.2102 | 76.7347 | 30.8266 |
| vaihingen/vaihingen | FineActionProjected_Exact | building | 69.9029 | 0.0822 | 70.1532 | 99.4921 | 30.5673 |
| vaihingen/vaihingen | FineActionProjected_Exact | low vegetation | 37.5590 | 0.6748 | 95.7092 | 38.2022 | 10.1587 |
| vaihingen/vaihingen | FineActionProjected_Exact | tree | 72.1533 | 0.0540 | 81.2767 | 86.5371 | 22.2505 |
| vaihingen/vaihingen | FineActionProjected_Exact | car | 22.6831 | -0.1352 | 22.9105 | 95.8086 | 6.1969 |
| vaihingen/vaihingen | FineRivalProjected_Exact | impervious surface | 62.3747 | 0.7540 | 76.4270 | 77.2335 | 30.9390 |
| vaihingen/vaihingen | FineRivalProjected_Exact | building | 70.1210 | 0.3003 | 70.3721 | 99.4938 | 30.4727 |
| vaihingen/vaihingen | FineRivalProjected_Exact | low vegetation | 37.8730 | 0.9888 | 95.7426 | 38.5217 | 10.2401 |
| vaihingen/vaihingen | FineRivalProjected_Exact | tree | 72.1589 | 0.0596 | 81.2353 | 86.5922 | 22.2760 |
| vaihingen/vaihingen | FineRivalProjected_Exact | car | 23.2028 | 0.3845 | 23.4290 | 96.0048 | 6.0722 |
| vaihingen/vaihingen | FineTopTwo_Exact | impervious surface | 62.5005 | 0.8798 | 76.1317 | 77.7318 | 31.2594 |
| vaihingen/vaihingen | FineTopTwo_Exact | building | 69.8017 | -0.0190 | 70.0458 | 99.5032 | 30.6175 |
| vaihingen/vaihingen | FineTopTwo_Exact | low vegetation | 37.2963 | 0.4121 | 96.4769 | 37.8112 | 9.9747 |
| vaihingen/vaihingen | FineTopTwo_Exact | tree | 72.1392 | 0.0399 | 81.2660 | 86.5290 | 22.2514 |
| vaihingen/vaihingen | FineTopTwo_Exact | car | 23.8634 | 1.0451 | 24.1072 | 95.9343 | 5.8970 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | impervious surface | 61.6255 | 0.0048 | 75.6318 | 76.8929 | 31.1264 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | building | 69.7305 | -0.0902 | 69.9843 | 99.4827 | 30.6381 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | low vegetation | 36.8639 | -0.0203 | 95.8510 | 37.4617 | 9.9470 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | tree | 72.0836 | -0.0157 | 81.3829 | 86.3171 | 22.1650 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | car | 22.8888 | 0.0705 | 23.1329 | 95.5921 | 6.1234 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | impervious surface | 62.2288 | 0.6081 | 76.7959 | 76.6389 | 30.5533 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | building | 70.0572 | 0.2365 | 70.3050 | 99.4994 | 30.5035 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | low vegetation | 38.2721 | 1.3879 | 95.5729 | 38.9628 | 10.3757 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | tree | 72.2064 | 0.1071 | 81.1784 | 86.7254 | 22.3260 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | car | 22.5708 | -0.2475 | 22.7864 | 95.9751 | 6.2415 |
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
| flair1/flair1 | FineRivalPosterior_Exact | deciduous | 60.7855 | 0.4047 | 78.7185 | 72.7389 | 15.7286 |
| flair1/flair1 | FineRivalPosterior_Exact | brushwood | 15.1361 | 0.2526 | 29.5364 | 23.6906 | 3.9450 |
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

## Pixel Changes Against Geometry

| Dataset/protocol | Method | Corrected | Broken | Wrong-to-wrong |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | NoAdmission_Exact | 30699881 | 5340265 | 4854091 |
| vdd/vdd | RivalFineHard_Exact | 30212364 | 4091451 | 4679335 |
| vdd/vdd | FineRivalPosterior_Exact | 29903508 | 3650354 | 4563433 |
| vdd/vdd | FineActionProjected_Exact | 30272259 | 3809634 | 4432276 |
| vdd/vdd | FineRivalProjected_Exact | 29807913 | 3454135 | 4434078 |
| vdd/vdd | FineTopTwo_Exact | 28624155 | 3446944 | 4742346 |
| vdd/vdd | FinePosteriorStrengthMatched_Exact | 29996128 | 4049170 | 4726105 |
| vdd/vdd | FineBudgetOnly_Exact | 30177751 | 3621834 | 4315851 |
| potsdam/potsdam | NoAdmission_Exact | 720293 | 420741 | 571181 |
| potsdam/potsdam | RivalFineHard_Exact | 761742 | 351424 | 535438 |
| potsdam/potsdam | FineRivalPosterior_Exact | 771360 | 339503 | 530771 |
| potsdam/potsdam | FineActionProjected_Exact | 752148 | 338624 | 530258 |
| potsdam/potsdam | FineRivalProjected_Exact | 763112 | 327792 | 525777 |
| potsdam/potsdam | FineTopTwo_Exact | 800909 | 326378 | 536523 |
| potsdam/potsdam | FinePosteriorStrengthMatched_Exact | 767057 | 347648 | 535188 |
| potsdam/potsdam | FineBudgetOnly_Exact | 743374 | 326114 | 525356 |
| udd5/udd5 | NoAdmission_Exact | 29576999 | 37113893 | 19566424 |
| udd5/udd5 | RivalFineHard_Exact | 30326885 | 26778763 | 17879565 |
| udd5/udd5 | FineRivalPosterior_Exact | 29901463 | 25753290 | 17700126 |
| udd5/udd5 | FineActionProjected_Exact | 29639078 | 25567990 | 17506857 |
| udd5/udd5 | FineRivalProjected_Exact | 29313181 | 24670920 | 17322043 |
| udd5/udd5 | FineTopTwo_Exact | 28745442 | 28615644 | 17690321 |
| udd5/udd5 | FinePosteriorStrengthMatched_Exact | 30216594 | 26056177 | 17775617 |
| udd5/udd5 | FineBudgetOnly_Exact | 29158101 | 24627644 | 17195930 |
| oem/oem | NoAdmission_Exact | 459328 | 1251366 | 361802 |
| oem/oem | RivalFineHard_Exact | 442066 | 1156483 | 357793 |
| oem/oem | FineRivalPosterior_Exact | 432001 | 1115511 | 354757 |
| oem/oem | FineActionProjected_Exact | 440309 | 1132514 | 352655 |
| oem/oem | FineRivalProjected_Exact | 429198 | 1089216 | 348751 |
| oem/oem | FineTopTwo_Exact | 412328 | 1083226 | 341459 |
| oem/oem | FinePosteriorStrengthMatched_Exact | 439805 | 1137059 | 356996 |
| oem/oem | FineBudgetOnly_Exact | 439510 | 1111286 | 348674 |
| loveda/P | NoAdmission_Exact | 268850 | 259754 | 110246 |
| loveda/P | RivalFineHard_Exact | 268520 | 211867 | 104320 |
| loveda/P | FineRivalPosterior_Exact | 267828 | 200971 | 103235 |
| loveda/P | FineActionProjected_Exact | 268187 | 201107 | 102559 |
| loveda/P | FineRivalProjected_Exact | 267434 | 190924 | 101808 |
| loveda/P | FineTopTwo_Exact | 260969 | 193320 | 104439 |
| loveda/P | FinePosteriorStrengthMatched_Exact | 268240 | 212035 | 104827 |
| loveda/P | FineBudgetOnly_Exact | 267914 | 189679 | 100798 |
| loveda/D | NoAdmission_Exact | 833269 | 1635495 | 1123346 |
| loveda/D | RivalFineHard_Exact | 1031710 | 1188153 | 868796 |
| loveda/D | FineRivalPosterior_Exact | 1067889 | 1057147 | 784195 |
| loveda/D | FineActionProjected_Exact | 1036814 | 1152114 | 845450 |
| loveda/D | FineRivalProjected_Exact | 1072684 | 1030838 | 766380 |
| loveda/D | FineTopTwo_Exact | 987428 | 858397 | 671524 |
| loveda/D | FinePosteriorStrengthMatched_Exact | 1047490 | 1139672 | 844376 |
| loveda/D | FineBudgetOnly_Exact | 1053009 | 1102286 | 811231 |
| vaihingen/vaihingen | NoAdmission_Exact | 647943 | 382120 | 395711 |
| vaihingen/vaihingen | RivalFineHard_Exact | 673187 | 325279 | 368951 |
| vaihingen/vaihingen | FineRivalPosterior_Exact | 682056 | 314089 | 366193 |
| vaihingen/vaihingen | FineActionProjected_Exact | 669929 | 307776 | 361662 |
| vaihingen/vaihingen | FineRivalProjected_Exact | 678811 | 296884 | 359433 |
| vaihingen/vaihingen | FineTopTwo_Exact | 689918 | 311210 | 373503 |
| vaihingen/vaihingen | FinePosteriorStrengthMatched_Exact | 672780 | 325826 | 369686 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | 668104 | 289447 | 353610 |
| landcoverai/landcoverai | NoAdmission_Exact | 113587 | 37157 | 11811 |
| landcoverai/landcoverai | RivalFineHard_Exact | 114942 | 29076 | 11607 |
| landcoverai/landcoverai | FineRivalPosterior_Exact | 116430 | 26492 | 11615 |
| landcoverai/landcoverai | FineActionProjected_Exact | 115241 | 28422 | 11562 |
| landcoverai/landcoverai | FineRivalProjected_Exact | 116402 | 25910 | 11555 |
| landcoverai/landcoverai | FineTopTwo_Exact | 119826 | 23095 | 11611 |
| landcoverai/landcoverai | FinePosteriorStrengthMatched_Exact | 115120 | 26866 | 11613 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | 115364 | 27979 | 11560 |
| flair1/flair1 | NoAdmission_Exact | 98118 | 134726 | 101041 |
| flair1/flair1 | RivalFineHard_Exact | 100142 | 131133 | 97230 |
| flair1/flair1 | FineRivalPosterior_Exact | 102489 | 131163 | 96723 |
| flair1/flair1 | FineActionProjected_Exact | 100103 | 126795 | 95035 |
| flair1/flair1 | FineRivalProjected_Exact | 101905 | 127170 | 94560 |
| flair1/flair1 | FineTopTwo_Exact | 112246 | 136334 | 99966 |
| flair1/flair1 | FinePosteriorStrengthMatched_Exact | 100259 | 131353 | 97195 |
| flair1/flair1 | FineBudgetOnly_Exact | 100452 | 123344 | 93037 |
