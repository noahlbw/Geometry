# Frozen conditional alias action: key-disjoint transfer panel

112 NEW complete inputs:16/domain in seven domains, disjoint by input key from the current developed64/complete96 panels. UDD5 uses its already-examined40 complete results unchanged, for152 descriptive images overall. This is not scene-disjoint/untouched independent validation: prior domain research and full baselines exist. Fixed seed selection sees neither labels nor scores. All original full per-image Geometry/no-admission/hard controls match exactly. Same20 aliases, checkpoints, Geometry, finite VIP wide/fine observers, risk and reconstruction.

Only FineRivalProjected is the frozen transfer candidate. FineBudgetOnly is a diagnostic retaining alias-derived action magnitudes with fine-class direction, not an alias-free model or a post-result alternative selector. Equivalent fine CUDA graph used throughout. Masks load only after predictions. LoveDA D once; P separately. Common scored-class denominators within each protocol.

| Dataset/protocol | Geometry | NoAdmission_Exact | RivalFineHard_Exact | FineRivalProjected_Exact | FineBudgetOnly_Exact | Candidate-hard pp | Paired bootstrap95% pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vdd/vdd | 35.2105 | 53.6741 | 55.2145 | 56.1993 | 56.1459 | +0.9848 | [+0.3343, +2.0794] |
| potsdam/potsdam | 37.8893 | 40.4691 | 41.8861 | 42.1555 | 42.0015 | +0.2693 | [+0.1359, +0.4136] |
| udd5/udd5 (reused) | 50.5553 | 49.2117 | 52.9563 | 53.1947 | 52.9621 | +0.2384 | [-0.0712, +0.5310] |
| oem/oem | 44.9899 | 37.4936 | 38.7700 | 39.6540 | 39.4742 | +0.8840 | [+0.4301, +1.4599] |
| loveda/P | 67.4823 | 63.0467 | 65.4682 | 66.6116 | 66.5745 | +1.1433 | [+0.5003, +1.9290] |
| loveda/D | 43.1909 | 41.0710 | 43.1298 | 44.2120 | 44.1783 | +1.0822 | [+0.5928, +1.5420] |
| vaihingen/vaihingen | 46.0762 | 49.3326 | 50.4574 | 50.6879 | 50.6056 | +0.2305 | [+0.1151, +0.3233] |
| landcoverai/landcoverai | 60.5117 | 67.4858 | 67.6718 | 67.8106 | 67.7412 | +0.1388 | [-0.0746, +1.0216] |
| flair1/flair1 | 41.5529 | 40.3617 | 40.6560 | 40.7028 | 40.7047 | +0.0468 | [-0.1470, +0.3163] |
| Seven NEW domain mean | 44.2031 | 47.1268 | 48.2551 | 48.7746 | 48.6931 | +0.5195 | [+0.3734, +0.7332] |
| Eight-domain descriptive mean | 44.9971 | 47.3874 | 48.8427 | 49.3271 | 49.2267 | +0.4844 | |

## Frozen Transfer Decision

```json
{
  "seven_new_domain_mean_gain_pp": 0.5194994610814163,
  "new_domain_wins": 7,
  "worst_new_protocol_gain_pp": 0.04678591315848024,
  "empirical_paired_bootstrap95_mean_interval_pp": [
    0.3733800387348869,
    0.7331510975882399
  ],
  "gain_vs_fine_direction_control_pp": 0.08151713079420375,
  "accuracy_transfer_gate": true,
  "stronger_empirical_transfer_gate": true
}
```

Paired image bootstrap:2000 replicates, sum confusion matrices within each sampled domain, fixed common scored-class support, then average seven domains equally. These intervals describe empirical panel variability, not correction for prior method selection or correlated scenes. No dataset-specific model selection or automatic full20092 rollout.

## Independent Window-Context Latency

Seven warmed synchronized alternating repetitions, same first new input per domain. One512 local window plus full-image wide context, not complete-image throughput. Loading/text encoding/decoding/GT excluded. Resident graph storage included in every arm. UDD5 is reused and not retimed here.

| Domain | Eager hard s | Graph hard s | Graph projected s | Projected overhead vs graph hard |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.315686 | 0.288997 | 0.291339 | +0.811% |
| potsdam | 0.273212 | 0.247455 | 0.250060 | +1.053% |
| oem | 0.294788 | 0.263534 | 0.266838 | +1.254% |
| loveda | 0.331934 | 0.305410 | 0.309058 | +1.194% |
| vaihingen | 0.282148 | 0.253691 | 0.254200 | +0.201% |
| landcoverai | 0.276849 | 0.248059 | 0.251797 | +1.507% |
| flair1 | 0.299487 | 0.269878 | 0.272694 | +1.043% |
| Mean | 0.296301 | 0.268146 | 0.270855 | +1.010% |

## Shared Five-Arm Cost

| Domain | Images | Parallel prediction wall s | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| vdd | 16 | 188.039 | 5611.732 |
| potsdam | 16 | 30.899 | 5607.921 |
| oem | 16 | 33.251 | 5621.014 |
| loveda | 16 | 43.850 | 5640.030 |
| vaihingen | 16 | 30.091 | 5601.972 |
| landcoverai | 16 | 5.970 | 5583.104 |
| flair1 | 16 | 6.640 | 5820.032 |

Shared costs include five dense endpoints and transitions; they are not standalone candidate latency.

## Residual-Excluded Performance

| Domain | Original hard | Candidate | Delta pp |
| --- | ---: | ---: | ---: |
| udd5 | 56.9706 | 57.2321 | +0.2615 |
| landcoverai | 63.1659 | 63.2508 | +0.0849 |

## Per-Class Outcomes

| Dataset/protocol | Method | Class | IoU | Delta vs hard | Precision | Recall | Area |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | Geometry | other | 9.8803 | -19.2109 | 18.5568 | 17.4450 | 15.7358 |
| vdd/vdd | Geometry | wall | 21.7998 | -21.9834 | 23.4663 | 75.4279 | 13.0420 |
| vdd/vdd | Geometry | road | 46.4850 | 4.8954 | 48.4096 | 92.1213 | 11.8303 |
| vdd/vdd | Geometry | vegetation | 72.0599 | -0.3803 | 92.0456 | 76.8453 | 27.0300 |
| vdd/vdd | Geometry | vehicle | 8.1906 | -15.3690 | 8.1971 | 99.0476 | 5.4764 |
| vdd/vdd | Geometry | roof | 65.7663 | -20.6497 | 88.7839 | 71.7254 | 23.3739 |
| vdd/vdd | Geometry | water | 22.2918 | -67.3299 | 76.4927 | 23.9312 | 3.5116 |
| vdd/vdd | NoAdmission_Exact | other | 30.4047 | 1.3135 | 53.8956 | 41.0926 | 12.7624 |
| vdd/vdd | NoAdmission_Exact | wall | 38.1835 | -5.5997 | 76.4860 | 43.2619 | 2.2950 |
| vdd/vdd | NoAdmission_Exact | road | 42.0588 | 0.4692 | 43.1369 | 94.3911 | 13.6034 |
| vdd/vdd | NoAdmission_Exact | vegetation | 70.5821 | -1.8582 | 97.4929 | 71.8869 | 23.8731 |
| vdd/vdd | NoAdmission_Exact | vehicle | 20.8375 | -2.7221 | 22.1267 | 78.1486 | 1.6007 |
| vdd/vdd | NoAdmission_Exact | roof | 85.8516 | -0.5644 | 86.5904 | 99.0159 | 33.0848 |
| vdd/vdd | NoAdmission_Exact | water | 87.8002 | -1.8214 | 87.8105 | 99.9868 | 12.7806 |
| vdd/vdd | RivalFineHard_Exact | other | 29.0912 | 0.0000 | 53.3109 | 39.0369 | 12.2569 |
| vdd/vdd | RivalFineHard_Exact | wall | 43.7832 | 0.0000 | 79.9469 | 49.1847 | 2.4962 |
| vdd/vdd | RivalFineHard_Exact | road | 41.5896 | 0.0000 | 42.4788 | 95.2083 | 13.9338 |
| vdd/vdd | RivalFineHard_Exact | vegetation | 72.4402 | 0.0000 | 97.6795 | 73.7087 | 24.4313 |
| vdd/vdd | RivalFineHard_Exact | vehicle | 23.5596 | 0.0000 | 24.7097 | 83.5029 | 1.5316 |
| vdd/vdd | RivalFineHard_Exact | roof | 86.4160 | 0.0000 | 87.2059 | 98.9627 | 32.8336 |
| vdd/vdd | RivalFineHard_Exact | water | 89.6216 | 0.0000 | 89.6469 | 99.9686 | 12.5165 |
| vdd/vdd | FineRivalProjected_Exact | other | 28.9012 | -0.1900 | 51.4050 | 39.7657 | 12.9487 |
| vdd/vdd | FineRivalProjected_Exact | wall | 45.6617 | 1.8785 | 78.6890 | 52.1052 | 2.6867 |
| vdd/vdd | FineRivalProjected_Exact | road | 42.6891 | 1.0995 | 43.5934 | 95.3660 | 13.6000 |
| vdd/vdd | FineRivalProjected_Exact | vegetation | 73.2540 | 0.8138 | 97.6491 | 74.5691 | 24.7242 |
| vdd/vdd | FineRivalProjected_Exact | vehicle | 26.7712 | 3.2116 | 27.5036 | 90.9534 | 1.4988 |
| vdd/vdd | FineRivalProjected_Exact | roof | 85.4038 | -1.0122 | 87.4697 | 97.3090 | 32.1876 |
| vdd/vdd | FineRivalProjected_Exact | water | 90.7143 | 1.0927 | 90.7810 | 99.9190 | 12.3540 |
| vdd/vdd | FineBudgetOnly_Exact | other | 30.3429 | 1.2517 | 52.9915 | 41.5184 | 13.1147 |
| vdd/vdd | FineBudgetOnly_Exact | wall | 44.9817 | 1.1986 | 78.3710 | 51.3573 | 2.6589 |
| vdd/vdd | FineBudgetOnly_Exact | road | 43.6116 | 2.0220 | 44.5975 | 95.1757 | 13.2673 |
| vdd/vdd | FineBudgetOnly_Exact | vegetation | 73.0490 | 0.6088 | 97.6638 | 74.3482 | 24.6473 |
| vdd/vdd | FineBudgetOnly_Exact | vehicle | 24.8224 | 1.2628 | 25.8866 | 85.7914 | 1.5020 |
| vdd/vdd | FineBudgetOnly_Exact | roof | 85.9881 | -0.4279 | 87.5482 | 97.9698 | 32.3771 |
| vdd/vdd | FineBudgetOnly_Exact | water | 90.2256 | 0.6039 | 90.2510 | 99.9687 | 12.4327 |
| potsdam/potsdam | Geometry | impervious surface | 50.0004 | -7.7429 | 72.1575 | 61.9530 | 31.3551 |
| potsdam/potsdam | Geometry | building | 83.1829 | 1.5695 | 84.1323 | 98.6615 | 21.8375 |
| potsdam/potsdam | Geometry | low vegetation | 28.1172 | 1.1165 | 92.0687 | 28.8152 | 5.5016 |
| potsdam/potsdam | Geometry | tree | 46.3573 | -7.6716 | 83.5241 | 51.0230 | 9.5257 |
| potsdam/potsdam | Geometry | car | 11.9431 | -12.0910 | 11.9754 | 97.7903 | 24.8080 |
| potsdam/potsdam | Geometry | clutter | 7.7346 | 0.8383 | 16.0853 | 12.9668 | 6.9721 |
| potsdam/potsdam | NoAdmission_Exact | impervious surface | 56.4265 | -1.3168 | 69.5557 | 74.9333 | 39.3431 |
| potsdam/potsdam | NoAdmission_Exact | building | 79.8701 | -1.7433 | 80.1130 | 99.6218 | 23.1563 |
| potsdam/potsdam | NoAdmission_Exact | low vegetation | 24.2118 | -2.7889 | 91.3797 | 24.7778 | 4.7664 |
| potsdam/potsdam | NoAdmission_Exact | tree | 51.7220 | -2.3069 | 89.1722 | 55.1880 | 9.6507 |
| potsdam/potsdam | NoAdmission_Exact | car | 23.6262 | -0.4078 | 23.7160 | 98.4237 | 12.6080 |
| potsdam/potsdam | NoAdmission_Exact | clutter | 6.9578 | 0.0615 | 11.8761 | 14.3842 | 10.4755 |
| potsdam/potsdam | RivalFineHard_Exact | impervious surface | 57.7434 | 0.0000 | 70.4032 | 76.2537 | 39.5544 |
| potsdam/potsdam | RivalFineHard_Exact | building | 81.6133 | 0.0000 | 81.8582 | 99.6348 | 22.6656 |
| potsdam/potsdam | RivalFineHard_Exact | low vegetation | 27.0007 | 0.0000 | 92.1585 | 27.6356 | 5.2712 |
| potsdam/potsdam | RivalFineHard_Exact | tree | 54.0289 | 0.0000 | 88.4855 | 58.1148 | 10.2413 |
| potsdam/potsdam | RivalFineHard_Exact | car | 24.0341 | 0.0000 | 24.1290 | 98.3894 | 12.3878 |
| potsdam/potsdam | RivalFineHard_Exact | clutter | 6.8963 | 0.0000 | 12.0992 | 13.8209 | 9.8796 |
| potsdam/potsdam | FineRivalProjected_Exact | impervious surface | 58.2429 | 0.4996 | 70.7349 | 76.7332 | 39.6165 |
| potsdam/potsdam | FineRivalProjected_Exact | building | 81.7713 | 0.1579 | 82.0028 | 99.6558 | 22.6304 |
| potsdam/potsdam | FineRivalProjected_Exact | low vegetation | 27.1329 | 0.1322 | 92.3595 | 27.7559 | 5.2827 |
| potsdam/potsdam | FineRivalProjected_Exact | tree | 54.3004 | 0.2715 | 88.3439 | 58.4909 | 10.3241 |
| potsdam/potsdam | FineRivalProjected_Exact | car | 24.4134 | 0.3794 | 24.5094 | 98.4212 | 12.1995 |
| potsdam/potsdam | FineRivalProjected_Exact | clutter | 7.0718 | 0.1755 | 12.3477 | 14.2007 | 9.9469 |
| potsdam/potsdam | FineBudgetOnly_Exact | impervious surface | 57.9878 | 0.2444 | 70.7836 | 76.2343 | 39.3318 |
| potsdam/potsdam | FineBudgetOnly_Exact | building | 81.7570 | 0.1437 | 81.9911 | 99.6521 | 22.6328 |
| potsdam/potsdam | FineBudgetOnly_Exact | low vegetation | 26.8948 | -0.1059 | 92.3366 | 27.5088 | 5.2369 |
| potsdam/potsdam | FineBudgetOnly_Exact | tree | 53.9868 | -0.0421 | 88.3439 | 58.1272 | 10.2599 |
| potsdam/potsdam | FineBudgetOnly_Exact | car | 24.1278 | 0.0937 | 24.2198 | 98.4498 | 12.3490 |
| potsdam/potsdam | FineBudgetOnly_Exact | clutter | 7.2546 | 0.3583 | 12.5052 | 14.7328 | 10.1896 |
| udd5/udd5 | Geometry | vegetation | 82.4937 | 9.0324 | 96.4862 | 85.0488 | 26.1079 |
| udd5/udd5 | Geometry | building | 83.9035 | -0.7893 | 88.3604 | 94.3293 | 41.9106 |
| udd5/udd5 | Geometry | road | 45.7856 | 2.1331 | 70.6780 | 56.5218 | 10.7242 |
| udd5/udd5 | Geometry | vehicle | 10.0092 | -16.0667 | 10.0323 | 97.7477 | 7.7994 |
| udd5/udd5 | Geometry | other | 30.5846 | -6.3144 | 52.8537 | 42.0591 | 13.4578 |
| udd5/udd5 | NoAdmission_Exact | vegetation | 68.3847 | -5.0766 | 97.4282 | 69.6418 | 21.1716 |
| udd5/udd5 | NoAdmission_Exact | building | 82.5264 | -2.1664 | 83.6805 | 98.3563 | 46.1437 |
| udd5/udd5 | NoAdmission_Exact | road | 39.8083 | -3.8441 | 66.8744 | 49.5860 | 9.9434 |
| udd5/udd5 | NoAdmission_Exact | vehicle | 20.4213 | -5.6546 | 20.9505 | 88.9918 | 3.4003 |
| udd5/udd5 | NoAdmission_Exact | other | 34.9177 | -1.9812 | 48.5109 | 55.4791 | 19.3411 |
| udd5/udd5 | RivalFineHard_Exact | vegetation | 73.4613 | 0.0000 | 97.6728 | 74.7700 | 22.6737 |
| udd5/udd5 | RivalFineHard_Exact | building | 84.6928 | 0.0000 | 85.7573 | 98.5555 | 45.1174 |
| udd5/udd5 | RivalFineHard_Exact | road | 43.6524 | 0.0000 | 68.8074 | 54.4220 | 10.6065 |
| udd5/udd5 | RivalFineHard_Exact | vehicle | 26.0759 | 0.0000 | 26.5277 | 93.8692 | 2.8326 |
| udd5/udd5 | RivalFineHard_Exact | other | 36.8990 | 0.0000 | 51.2389 | 56.8679 | 18.7698 |
| udd5/udd5 | FineRivalProjected_Exact | vegetation | 74.5093 | 1.0480 | 97.6926 | 75.8441 | 22.9948 |
| udd5/udd5 | FineRivalProjected_Exact | building | 84.9284 | 0.2357 | 86.0908 | 98.4351 | 44.8878 |
| udd5/udd5 | FineRivalProjected_Exact | road | 43.6159 | -0.0365 | 70.1931 | 53.5304 | 10.2268 |
| udd5/udd5 | FineRivalProjected_Exact | vehicle | 25.8748 | -0.2011 | 26.2495 | 94.7708 | 2.8901 |
| udd5/udd5 | FineRivalProjected_Exact | other | 37.0450 | 0.1460 | 51.0910 | 57.4011 | 19.0006 |
| udd5/udd5 | FineBudgetOnly_Exact | vegetation | 74.7410 | 1.2798 | 97.6843 | 76.0892 | 23.0711 |
| udd5/udd5 | FineBudgetOnly_Exact | building | 84.9177 | 0.2249 | 85.9884 | 98.5548 | 44.9958 |
| udd5/udd5 | FineBudgetOnly_Exact | road | 42.9557 | -0.6967 | 71.0536 | 52.0673 | 9.8268 |
| udd5/udd5 | FineBudgetOnly_Exact | vehicle | 25.0541 | -1.0218 | 25.4126 | 94.6698 | 2.9821 |
| udd5/udd5 | FineBudgetOnly_Exact | other | 37.1420 | 0.2430 | 51.0327 | 57.7087 | 19.1242 |
| oem/oem | Geometry | bareland | 5.4418 | 2.9226 | 6.5871 | 23.8368 | 5.6900 |
| oem/oem | Geometry | rangeland | 35.4024 | 12.1849 | 73.8081 | 40.4891 | 12.4986 |
| oem/oem | Geometry | developed space | 26.2728 | 2.7273 | 33.9328 | 53.7861 | 19.7253 |
| oem/oem | Geometry | road | 36.1208 | 2.6615 | 44.2955 | 66.1848 | 10.7577 |
| oem/oem | Geometry | tree | 65.1220 | 10.4369 | 89.9464 | 70.2343 | 25.0154 |
| oem/oem | Geometry | water | 51.7207 | -1.5880 | 65.5748 | 70.9982 | 1.0715 |
| oem/oem | Geometry | agriculture land | 79.2481 | 10.1646 | 94.2720 | 83.2570 | 4.9183 |
| oem/oem | Geometry | building | 60.5908 | 10.2501 | 70.0417 | 81.7866 | 20.3232 |
| oem/oem | NoAdmission_Exact | bareland | 2.4713 | -0.0479 | 2.9016 | 14.2848 | 7.7411 |
| oem/oem | NoAdmission_Exact | rangeland | 23.4052 | 0.1877 | 64.6097 | 26.8471 | 9.4673 |
| oem/oem | NoAdmission_Exact | developed space | 23.0035 | -0.5420 | 24.8618 | 75.4757 | 37.7788 |
| oem/oem | NoAdmission_Exact | road | 31.9998 | -1.4595 | 48.3527 | 48.6173 | 7.2392 |
| oem/oem | NoAdmission_Exact | tree | 51.4109 | -3.2743 | 95.6108 | 52.6535 | 17.6426 |
| oem/oem | NoAdmission_Exact | water | 53.8162 | 0.5075 | 75.0042 | 65.5774 | 0.8653 |
| oem/oem | NoAdmission_Exact | agriculture land | 65.5304 | -3.5531 | 66.3167 | 98.2229 | 8.2484 |
| oem/oem | NoAdmission_Exact | building | 48.3118 | -2.0290 | 84.0340 | 53.1945 | 11.0174 |
| oem/oem | RivalFineHard_Exact | bareland | 2.5192 | 0.0000 | 2.9340 | 15.1245 | 8.1056 |
| oem/oem | RivalFineHard_Exact | rangeland | 23.2175 | 0.0000 | 65.8362 | 26.3980 | 9.1355 |
| oem/oem | RivalFineHard_Exact | developed space | 23.5455 | 0.0000 | 25.5781 | 74.7665 | 36.3759 |
| oem/oem | RivalFineHard_Exact | road | 33.4593 | 0.0000 | 50.0891 | 50.1943 | 7.2149 |
| oem/oem | RivalFineHard_Exact | tree | 54.6851 | 0.0000 | 94.7363 | 56.3988 | 19.0719 |
| oem/oem | RivalFineHard_Exact | water | 53.3088 | 0.0000 | 74.4393 | 65.2534 | 0.8675 |
| oem/oem | RivalFineHard_Exact | agriculture land | 69.0835 | 0.0000 | 69.9936 | 98.1525 | 7.8095 |
| oem/oem | RivalFineHard_Exact | building | 50.3407 | 0.0000 | 84.5201 | 55.4535 | 11.4192 |
| oem/oem | FineRivalProjected_Exact | bareland | 2.5939 | 0.0747 | 3.0119 | 15.7462 | 8.2203 |
| oem/oem | FineRivalProjected_Exact | rangeland | 23.1212 | -0.0963 | 66.0751 | 26.2357 | 9.0465 |
| oem/oem | FineRivalProjected_Exact | developed space | 23.7917 | 0.2462 | 25.9504 | 74.0935 | 35.5312 |
| oem/oem | FineRivalProjected_Exact | road | 34.2529 | 0.7935 | 50.8419 | 51.2142 | 7.2525 |
| oem/oem | FineRivalProjected_Exact | tree | 56.4316 | 1.7465 | 94.3725 | 58.3967 | 19.8237 |
| oem/oem | FineRivalProjected_Exact | water | 53.5295 | 0.2208 | 74.8054 | 65.3029 | 0.8639 |
| oem/oem | FineRivalProjected_Exact | agriculture land | 72.0526 | 2.9691 | 73.1822 | 97.9028 | 7.4502 |
| oem/oem | FineRivalProjected_Exact | building | 51.4585 | 1.1177 | 84.0387 | 57.0325 | 11.8116 |
| oem/oem | FineBudgetOnly_Exact | bareland | 2.6091 | 0.0900 | 3.0500 | 15.2902 | 7.8826 |
| oem/oem | FineBudgetOnly_Exact | rangeland | 24.0872 | 0.8697 | 66.6009 | 27.3966 | 9.3722 |
| oem/oem | FineBudgetOnly_Exact | developed space | 23.8045 | 0.2590 | 25.8805 | 74.7956 | 35.9648 |
| oem/oem | FineBudgetOnly_Exact | road | 33.9671 | 0.5078 | 51.1145 | 50.3111 | 7.0866 |
| oem/oem | FineBudgetOnly_Exact | tree | 56.0243 | 1.3392 | 94.5009 | 57.9123 | 19.6325 |
| oem/oem | FineBudgetOnly_Exact | water | 53.4707 | 0.1619 | 75.0160 | 65.0561 | 0.8583 |
| oem/oem | FineBudgetOnly_Exact | agriculture land | 70.8332 | 1.7497 | 71.8986 | 97.9508 | 7.5869 |
| oem/oem | FineBudgetOnly_Exact | building | 50.9976 | 0.6569 | 84.3784 | 56.3146 | 11.6160 |
| loveda/P | Geometry | building | 83.5628 | 1.7782 | 89.4197 | 92.7314 | 8.4517 |
| loveda/P | Geometry | road | 56.8741 | -5.1695 | 60.7500 | 89.9138 | 12.0337 |
| loveda/P | Geometry | water | 72.7661 | 1.2889 | 91.9118 | 77.7444 | 18.6468 |
| loveda/P | Geometry | barren | 50.1054 | 0.1703 | 68.7594 | 64.8741 | 8.3414 |
| loveda/P | Geometry | tree | 63.4010 | 10.8497 | 69.9238 | 87.1738 | 17.3635 |
| loveda/P | Geometry | farm | 78.1846 | 3.1672 | 92.4280 | 83.5351 | 35.1628 |
| loveda/P | NoAdmission_Exact | building | 81.6450 | -0.1395 | 92.2048 | 87.6983 | 7.7516 |
| loveda/P | NoAdmission_Exact | road | 61.0624 | -0.9813 | 69.9101 | 82.8322 | 9.6334 |
| loveda/P | NoAdmission_Exact | water | 68.7147 | -2.7625 | 95.4629 | 71.0346 | 16.4037 |
| loveda/P | NoAdmission_Exact | barren | 48.6123 | -1.3228 | 68.5987 | 62.5259 | 8.0583 |
| loveda/P | NoAdmission_Exact | tree | 46.0770 | -6.4743 | 92.2237 | 47.9396 | 7.2398 |
| loveda/P | NoAdmission_Exact | farm | 72.1685 | -2.8489 | 73.9492 | 96.7711 | 50.9132 |
| loveda/P | RivalFineHard_Exact | building | 81.7846 | 0.0000 | 92.7824 | 87.3413 | 7.6720 |
| loveda/P | RivalFineHard_Exact | road | 62.0437 | 0.0000 | 71.0741 | 83.0023 | 9.4951 |
| loveda/P | RivalFineHard_Exact | water | 71.4772 | 0.0000 | 95.7377 | 73.8265 | 16.9995 |
| loveda/P | RivalFineHard_Exact | barren | 49.9351 | 0.0000 | 67.4400 | 65.7981 | 8.6257 |
| loveda/P | RivalFineHard_Exact | tree | 52.5513 | 0.0000 | 91.9358 | 55.0908 | 8.3459 |
| loveda/P | RivalFineHard_Exact | farm | 75.0174 | 0.0000 | 76.9922 | 96.6939 | 48.8619 |
| loveda/P | FineRivalProjected_Exact | building | 82.0442 | 0.2597 | 92.8937 | 87.5384 | 7.6800 |
| loveda/P | FineRivalProjected_Exact | road | 62.7004 | 0.6567 | 71.9295 | 83.0127 | 9.3833 |
| loveda/P | FineRivalProjected_Exact | water | 72.3826 | 0.9054 | 95.5792 | 74.8898 | 17.2729 |
| loveda/P | FineRivalProjected_Exact | barren | 51.4343 | 1.4992 | 67.4389 | 68.4274 | 8.9706 |
| loveda/P | FineRivalProjected_Exact | tree | 54.7744 | 2.2231 | 91.6696 | 57.6437 | 8.7580 |
| loveda/P | FineRivalProjected_Exact | farm | 76.3333 | 1.3159 | 78.4245 | 96.6246 | 47.9352 |
| loveda/P | FineBudgetOnly_Exact | building | 82.2205 | 0.4360 | 92.8865 | 87.7456 | 7.6988 |
| loveda/P | FineBudgetOnly_Exact | road | 62.9366 | 0.8929 | 72.1946 | 83.0735 | 9.3557 |
| loveda/P | FineBudgetOnly_Exact | water | 72.2632 | 0.7860 | 95.5226 | 74.7967 | 17.2616 |
| loveda/P | FineBudgetOnly_Exact | barren | 50.8326 | 0.8975 | 67.6595 | 67.1478 | 8.7741 |
| loveda/P | FineBudgetOnly_Exact | tree | 55.1472 | 2.5958 | 91.6317 | 58.0719 | 8.8267 |
| loveda/P | FineBudgetOnly_Exact | farm | 76.0467 | 1.0293 | 78.1494 | 96.5827 | 48.0830 |
| loveda/D | Geometry | background | 30.7308 | 13.8753 | 57.3442 | 39.8373 | 24.2172 |
| loveda/D | Geometry | building | 44.2716 | 0.0704 | 46.3362 | 90.8556 | 10.4095 |
| loveda/D | Geometry | road | 42.0787 | -8.7249 | 44.4237 | 88.8537 | 10.5933 |
| loveda/D | Geometry | water | 55.8298 | -7.9164 | 78.8508 | 65.6624 | 11.9582 |
| loveda/D | Geometry | barren | 19.0320 | -12.4265 | 41.2152 | 26.1232 | 3.6502 |
| loveda/D | Geometry | tree | 48.3821 | 3.8417 | 54.2637 | 81.6975 | 13.6592 |
| loveda/D | Geometry | farm | 62.0115 | 11.7083 | 76.2988 | 76.8068 | 25.5123 |
| loveda/D | NoAdmission_Exact | background | 13.6250 | -3.2305 | 58.6654 | 15.0719 | 8.9559 |
| loveda/D | NoAdmission_Exact | building | 43.8865 | -0.3147 | 47.2938 | 85.8989 | 9.6424 |
| loveda/D | NoAdmission_Exact | road | 49.3741 | -1.4295 | 55.4935 | 81.7434 | 7.8015 |
| loveda/D | NoAdmission_Exact | water | 62.0801 | -1.6660 | 83.8833 | 70.4876 | 12.0668 |
| loveda/D | NoAdmission_Exact | barren | 31.5789 | 0.1203 | 49.9940 | 46.1588 | 5.3172 |
| loveda/D | NoAdmission_Exact | tree | 39.8233 | -4.7171 | 77.2839 | 45.1027 | 5.2947 |
| loveda/D | NoAdmission_Exact | farm | 47.1290 | -3.1742 | 47.9750 | 96.3935 | 50.9214 |
| loveda/D | RivalFineHard_Exact | background | 16.8554 | 0.0000 | 57.9239 | 19.2071 | 11.5592 |
| loveda/D | RivalFineHard_Exact | building | 44.2012 | 0.0000 | 47.8133 | 85.4033 | 9.4826 |
| loveda/D | RivalFineHard_Exact | road | 50.8036 | 0.0000 | 57.1768 | 82.0073 | 7.5963 |
| loveda/D | RivalFineHard_Exact | water | 63.7462 | 0.0000 | 84.2980 | 72.3352 | 12.3222 |
| loveda/D | RivalFineHard_Exact | barren | 31.4585 | 0.0000 | 48.5794 | 47.1632 | 5.5911 |
| loveda/D | RivalFineHard_Exact | tree | 44.5404 | 0.0000 | 77.6007 | 51.1115 | 5.9756 |
| loveda/D | RivalFineHard_Exact | farm | 50.3032 | 0.0000 | 51.3347 | 96.1590 | 47.4730 |
| loveda/D | FineRivalProjected_Exact | background | 19.8044 | 2.9489 | 60.5488 | 22.7385 | 13.0912 |
| loveda/D | FineRivalProjected_Exact | building | 44.2062 | 0.0050 | 47.7628 | 85.5836 | 9.5126 |
| loveda/D | FineRivalProjected_Exact | road | 51.4333 | 0.6297 | 57.9599 | 82.0390 | 7.4966 |
| loveda/D | FineRivalProjected_Exact | water | 64.3283 | 0.5821 | 83.9568 | 73.3440 | 12.5448 |
| loveda/D | FineRivalProjected_Exact | barren | 31.4871 | 0.0285 | 48.1387 | 47.6513 | 5.7007 |
| loveda/D | FineRivalProjected_Exact | tree | 45.8214 | 1.2810 | 77.1766 | 53.0037 | 6.2309 |
| loveda/D | FineRivalProjected_Exact | farm | 52.4031 | 2.0999 | 53.5692 | 96.0118 | 45.4232 |
| loveda/D | FineBudgetOnly_Exact | background | 19.4410 | 2.5856 | 60.1748 | 22.3117 | 12.9254 |
| loveda/D | FineBudgetOnly_Exact | building | 44.2538 | 0.0526 | 47.7699 | 85.7394 | 9.5286 |
| loveda/D | FineBudgetOnly_Exact | road | 51.6110 | 0.8074 | 58.1723 | 82.0654 | 7.4716 |
| loveda/D | FineBudgetOnly_Exact | water | 64.4326 | 0.6865 | 83.8676 | 73.5482 | 12.5931 |
| loveda/D | FineBudgetOnly_Exact | barren | 31.2306 | -0.2279 | 48.4703 | 46.7537 | 5.5551 |
| loveda/D | FineBudgetOnly_Exact | tree | 46.1473 | 1.6069 | 77.3141 | 53.3747 | 6.2633 |
| loveda/D | FineBudgetOnly_Exact | farm | 52.1316 | 1.8284 | 53.2863 | 96.0091 | 45.6630 |
| vaihingen/vaihingen | Geometry | impervious surface | 44.8452 | -17.9728 | 82.5028 | 49.5586 | 17.6870 |
| vaihingen/vaihingen | Geometry | building | 72.9874 | 3.4658 | 75.7685 | 95.2119 | 27.9192 |
| vaihingen/vaihingen | Geometry | low vegetation | 29.1567 | 5.5281 | 78.8426 | 31.6317 | 6.8723 |
| vaihingen/vaihingen | Geometry | tree | 76.2634 | 0.1903 | 84.7156 | 88.4310 | 31.3389 |
| vaihingen/vaihingen | Geometry | car | 7.1283 | -13.1174 | 7.1416 | 97.4478 | 16.1825 |
| vaihingen/vaihingen | NoAdmission_Exact | impervious surface | 61.2786 | -1.5394 | 80.5726 | 71.9024 | 26.2761 |
| vaihingen/vaihingen | NoAdmission_Exact | building | 67.9244 | -1.5972 | 68.4978 | 98.7827 | 32.0409 |
| vaihingen/vaihingen | NoAdmission_Exact | low vegetation | 22.3182 | -1.3104 | 85.1108 | 23.2250 | 4.6743 |
| vaihingen/vaihingen | NoAdmission_Exact | tree | 75.5594 | -0.5136 | 84.3399 | 87.8902 | 31.2860 |
| vaihingen/vaihingen | NoAdmission_Exact | car | 19.5822 | -0.6635 | 19.7692 | 95.3928 | 5.7226 |
| vaihingen/vaihingen | RivalFineHard_Exact | impervious surface | 62.8180 | 0.0000 | 81.3550 | 73.3827 | 26.5592 |
| vaihingen/vaihingen | RivalFineHard_Exact | building | 69.5217 | 0.0000 | 70.2166 | 98.5963 | 31.1976 |
| vaihingen/vaihingen | RivalFineHard_Exact | low vegetation | 23.6286 | 0.0000 | 87.0123 | 24.4925 | 4.8216 |
| vaihingen/vaihingen | RivalFineHard_Exact | tree | 76.0730 | 0.0000 | 83.9022 | 89.0739 | 31.8728 |
| vaihingen/vaihingen | RivalFineHard_Exact | car | 20.2457 | 0.0000 | 20.4356 | 95.6127 | 5.5488 |
| vaihingen/vaihingen | FineRivalProjected_Exact | impervious surface | 62.9512 | 0.1332 | 81.4348 | 73.4994 | 26.5753 |
| vaihingen/vaihingen | FineRivalProjected_Exact | building | 69.8015 | 0.2799 | 70.5478 | 98.5071 | 31.0231 |
| vaihingen/vaihingen | FineRivalProjected_Exact | low vegetation | 23.9810 | 0.3524 | 87.5284 | 24.8295 | 4.8592 |
| vaihingen/vaihingen | FineRivalProjected_Exact | tree | 76.2075 | 0.1345 | 83.7653 | 89.4139 | 32.0468 |
| vaihingen/vaihingen | FineRivalProjected_Exact | car | 20.4984 | 0.2526 | 20.6823 | 95.8411 | 5.4957 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | impervious surface | 62.5755 | -0.2425 | 81.5286 | 72.9126 | 26.3328 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | building | 69.7911 | 0.2694 | 70.5347 | 98.5119 | 31.0303 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | low vegetation | 24.2653 | 0.6367 | 87.5292 | 25.1342 | 4.9188 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | tree | 76.2896 | 0.2166 | 83.7348 | 89.5617 | 32.1114 |
| vaihingen/vaihingen | FineBudgetOnly_Exact | car | 20.1065 | -0.1392 | 20.2816 | 95.8825 | 5.6067 |
| landcoverai/landcoverai | Geometry | background | 82.9195 | -2.7761 | 90.2873 | 91.0405 | 52.2092 |
| landcoverai/landcoverai | Geometry | building | 30.7681 | -17.2864 | 30.7765 | 99.9108 | 0.9546 |
| landcoverai/landcoverai | Geometry | woodland | 85.0196 | 0.9936 | 95.4044 | 88.6502 | 36.6740 |
| landcoverai/landcoverai | Geometry | water | 82.7009 | -2.2887 | 98.8783 | 83.4841 | 6.2067 |
| landcoverai/landcoverai | Geometry | road | 21.1504 | -14.4429 | 22.3539 | 79.7103 | 3.9555 |
| landcoverai/landcoverai | NoAdmission_Exact | background | 85.4810 | -0.2146 | 88.2565 | 96.4516 | 56.5851 |
| landcoverai/landcoverai | NoAdmission_Exact | building | 47.1462 | -0.9082 | 47.3582 | 99.0594 | 0.6150 |
| landcoverai/landcoverai | NoAdmission_Exact | woodland | 83.5299 | -0.4962 | 97.0745 | 85.6869 | 34.8382 |
| landcoverai/landcoverai | NoAdmission_Exact | water | 84.9183 | -0.0713 | 99.4807 | 85.2964 | 6.3031 |
| landcoverai/landcoverai | NoAdmission_Exact | road | 36.3535 | 0.7602 | 44.4928 | 66.5241 | 1.6585 |
| landcoverai/landcoverai | RivalFineHard_Exact | background | 85.6956 | 0.0000 | 88.6296 | 96.2807 | 56.2470 |
| landcoverai/landcoverai | RivalFineHard_Exact | building | 48.0545 | 0.0000 | 48.2805 | 99.0351 | 0.6032 |
| landcoverai/landcoverai | RivalFineHard_Exact | woodland | 84.0261 | 0.0000 | 96.9310 | 86.3226 | 35.1487 |
| landcoverai/landcoverai | RivalFineHard_Exact | water | 84.9896 | 0.0000 | 99.6087 | 85.2744 | 6.2933 |
| landcoverai/landcoverai | RivalFineHard_Exact | road | 35.5933 | 0.0000 | 43.3003 | 66.6638 | 1.7078 |
| landcoverai/landcoverai | FineRivalProjected_Exact | background | 86.0498 | 0.3543 | 89.1485 | 96.1175 | 55.8248 |
| landcoverai/landcoverai | FineRivalProjected_Exact | building | 48.0607 | 0.0063 | 48.2791 | 99.0675 | 0.6034 |
| landcoverai/landcoverai | FineRivalProjected_Exact | woodland | 84.6750 | 0.6489 | 96.7622 | 87.1441 | 35.5451 |
| landcoverai/landcoverai | FineRivalProjected_Exact | water | 85.0606 | 0.0710 | 99.6274 | 85.3321 | 6.2964 |
| landcoverai/landcoverai | FineRivalProjected_Exact | road | 35.2068 | -0.3865 | 42.7323 | 66.6574 | 1.7303 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | background | 85.9958 | 0.3002 | 89.0215 | 96.1980 | 55.9513 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | building | 47.4260 | -0.6285 | 47.6255 | 99.1243 | 0.6120 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | woodland | 84.4974 | 0.4713 | 96.8031 | 86.9230 | 35.4399 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | water | 85.0778 | 0.0882 | 99.6042 | 85.3665 | 6.3004 |
| landcoverai/landcoverai | FineBudgetOnly_Exact | road | 35.7090 | 0.1157 | 43.5193 | 66.5520 | 1.6963 |
| flair1/flair1 | Geometry | building | 67.0966 | -6.4769 | 68.1989 | 97.6478 | 15.9894 |
| flair1/flair1 | Geometry | pervious surface | 9.7604 | -21.1276 | 24.8509 | 13.8476 | 0.9564 |
| flair1/flair1 | Geometry | impervious surface | 57.7466 | -1.0939 | 72.6067 | 73.8322 | 10.6937 |
| flair1/flair1 | Geometry | bare soil | 69.7864 | 11.1318 | 71.8093 | 96.1199 | 6.0964 |
| flair1/flair1 | Geometry | water | 92.0166 | 9.1580 | 97.2234 | 94.5000 | 24.4604 |
| flair1/flair1 | Geometry | coniferous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2338 |
| flair1/flair1 | Geometry | deciduous | 54.1871 | 8.6727 | 80.8738 | 62.1519 | 5.3313 |
| flair1/flair1 | Geometry | brushwood | 49.0021 | 16.3048 | 64.7139 | 66.8687 | 8.4523 |
| flair1/flair1 | Geometry | vineyard | 0.0000 | 0.0000 | 0.0000 | -- | 0.0034 |
| flair1/flair1 | Geometry | herbaceous vegetation | 30.3753 | -10.7777 | 46.7859 | 46.4089 | 14.8807 |
| flair1/flair1 | Geometry | agricultural land | 30.2466 | -12.5594 | 68.4276 | 35.1523 | 7.5013 |
| flair1/flair1 | Geometry | plowed land | 38.4176 | 17.5320 | 38.7759 | 97.6515 | 5.4010 |
| flair1/flair1 | NoAdmission_Exact | building | 72.9954 | -0.5782 | 74.1971 | 97.8293 | 14.7241 |
| flair1/flair1 | NoAdmission_Exact | pervious surface | 29.2004 | -1.6876 | 62.1447 | 35.5182 | 0.9810 |
| flair1/flair1 | NoAdmission_Exact | impervious surface | 58.7255 | -0.1150 | 73.5753 | 74.4222 | 10.6373 |
| flair1/flair1 | NoAdmission_Exact | bare soil | 58.5702 | -0.0844 | 59.0228 | 98.7075 | 7.6167 |
| flair1/flair1 | NoAdmission_Exact | water | 82.0023 | -0.8563 | 96.0118 | 84.8940 | 22.2513 |
| flair1/flair1 | NoAdmission_Exact | coniferous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2193 |
| flair1/flair1 | NoAdmission_Exact | deciduous | 45.4177 | -0.0968 | 61.6971 | 63.2526 | 7.1121 |
| flair1/flair1 | NoAdmission_Exact | brushwood | 32.7075 | 0.0101 | 71.4215 | 37.6326 | 4.3101 |
| flair1/flair1 | NoAdmission_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | -- | 0.0319 |
| flair1/flair1 | NoAdmission_Exact | herbaceous vegetation | 41.2901 | 0.1372 | 91.0547 | 43.0358 | 7.0903 |
| flair1/flair1 | NoAdmission_Exact | agricultural land | 42.3628 | -0.4432 | 58.6768 | 60.3751 | 15.0247 |
| flair1/flair1 | NoAdmission_Exact | plowed land | 21.0685 | 0.1828 | 21.1338 | 98.5542 | 10.0012 |
| flair1/flair1 | RivalFineHard_Exact | building | 73.5735 | 0.0000 | 74.7967 | 97.8257 | 14.6055 |
| flair1/flair1 | RivalFineHard_Exact | pervious surface | 30.8881 | 0.0000 | 63.4572 | 37.5709 | 1.0162 |
| flair1/flair1 | RivalFineHard_Exact | impervious surface | 58.8405 | 0.0000 | 73.7282 | 74.4503 | 10.6192 |
| flair1/flair1 | RivalFineHard_Exact | bare soil | 58.6545 | 0.0000 | 59.0695 | 98.8165 | 7.6191 |
| flair1/flair1 | RivalFineHard_Exact | water | 82.8585 | 0.0000 | 96.0938 | 85.7466 | 22.4556 |
| flair1/flair1 | RivalFineHard_Exact | coniferous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2280 |
| flair1/flair1 | RivalFineHard_Exact | deciduous | 45.5145 | 0.0000 | 61.7819 | 63.3510 | 7.1134 |
| flair1/flair1 | RivalFineHard_Exact | brushwood | 32.6973 | 0.0000 | 69.6300 | 38.1360 | 4.4801 |
| flair1/flair1 | RivalFineHard_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | -- | 0.0287 |
| flair1/flair1 | RivalFineHard_Exact | herbaceous vegetation | 41.1530 | 0.0000 | 91.8633 | 42.7098 | 6.9746 |
| flair1/flair1 | RivalFineHard_Exact | agricultural land | 42.8060 | 0.0000 | 59.5944 | 60.3096 | 14.7773 |
| flair1/flair1 | RivalFineHard_Exact | plowed land | 20.8856 | 0.0000 | 20.9524 | 98.4985 | 10.0822 |
| flair1/flair1 | FineRivalProjected_Exact | building | 73.5567 | -0.0169 | 74.7129 | 97.9394 | 14.6389 |
| flair1/flair1 | FineRivalProjected_Exact | pervious surface | 29.0929 | -1.7951 | 62.7763 | 35.1580 | 0.9613 |
| flair1/flair1 | FineRivalProjected_Exact | impervious surface | 58.8120 | -0.0284 | 73.8172 | 74.3144 | 10.5870 |
| flair1/flair1 | FineRivalProjected_Exact | bare soil | 58.9607 | 0.3061 | 59.3931 | 98.7804 | 7.5748 |
| flair1/flair1 | FineRivalProjected_Exact | water | 83.6552 | 0.7967 | 96.1203 | 86.5786 | 22.6672 |
| flair1/flair1 | FineRivalProjected_Exact | coniferous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2313 |
| flair1/flair1 | FineRivalProjected_Exact | deciduous | 45.8268 | 0.3123 | 61.7671 | 63.9735 | 7.1850 |
| flair1/flair1 | FineRivalProjected_Exact | brushwood | 32.7368 | 0.0395 | 69.2634 | 38.3009 | 4.5233 |
| flair1/flair1 | FineRivalProjected_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | -- | 0.0231 |
| flair1/flair1 | FineRivalProjected_Exact | herbaceous vegetation | 41.1794 | 0.0264 | 92.1445 | 42.6777 | 6.9481 |
| flair1/flair1 | FineRivalProjected_Exact | agricultural land | 43.5739 | 0.7679 | 60.5960 | 60.8021 | 14.6518 |
| flair1/flair1 | FineRivalProjected_Exact | plowed land | 21.0387 | 0.1530 | 21.1065 | 98.4952 | 10.0082 |
| flair1/flair1 | FineBudgetOnly_Exact | building | 73.4052 | -0.1683 | 74.5645 | 97.9259 | 14.6660 |
| flair1/flair1 | FineBudgetOnly_Exact | pervious surface | 27.5156 | -3.3725 | 61.6288 | 33.2040 | 0.9248 |
| flair1/flair1 | FineBudgetOnly_Exact | impervious surface | 58.9107 | 0.0703 | 73.9227 | 74.3650 | 10.5791 |
| flair1/flair1 | FineBudgetOnly_Exact | bare soil | 59.1702 | 0.5156 | 59.6326 | 98.7065 | 7.5387 |
| flair1/flair1 | FineBudgetOnly_Exact | water | 83.5155 | 0.6570 | 96.1064 | 86.4402 | 22.6342 |
| flair1/flair1 | FineBudgetOnly_Exact | coniferous | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2333 |
| flair1/flair1 | FineBudgetOnly_Exact | deciduous | 46.1299 | 0.6154 | 62.1686 | 64.1328 | 7.1564 |
| flair1/flair1 | FineBudgetOnly_Exact | brushwood | 33.1081 | 0.4108 | 69.0148 | 38.8886 | 4.6093 |
| flair1/flair1 | FineBudgetOnly_Exact | vineyard | 0.0000 | 0.0000 | 0.0000 | -- | 0.0195 |
| flair1/flair1 | FineBudgetOnly_Exact | herbaceous vegetation | 41.4305 | 0.2775 | 92.1659 | 42.9428 | 6.9896 |
| flair1/flair1 | FineBudgetOnly_Exact | agricultural land | 43.9248 | 1.1187 | 60.6600 | 61.4217 | 14.7854 |
| flair1/flair1 | FineBudgetOnly_Exact | plowed land | 21.3463 | 0.4607 | 21.4161 | 98.4963 | 9.8636 |

All schemes preserved; this transfer result does not prove every alias useful or VIP deletion universally ineffective.

Opt-in frozen complete/sharded candidate evaluator: DINOtool/scripts/eval_rival_projected_graph.py.
Same retained evaluator arguments and refusal of existing output. Keeps
Geometry/no-admission/original-hard controls alongside the candidate; no full suite
was launched. A nonzero-risk regression checks bitwise endpoint equality when
removing diagnostic arms or running the candidate alone.
