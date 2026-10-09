# Target/Context Alias Source: Verified64-Window Pilot

64 fixed unique image IDs, eight/domain, ONE top-left512 window/image and original whole-image wide views. Frozen weights, exact20 aliases/class, no label fitting. LoveDA P/D share images; means count D once. Corrected IRRG; LandCover.ai replaces unavailable labeled iSAID. All are development domains. Not complete-image/full-dataset mIoU, not new solver or verified novelty.

## Decision

{
  "passed": false,
  "checks": {
    "gain_at_least_point1": true,
    "at_least5_domain_wins": true,
    "no_protocol_loss_over1": true,
    "above_all_alias_shuffles": true,
    "above_shuffled_support": false,
    "above_context_only": true,
    "above_same_source_mean": true
  },
  "wins": 6,
  "worst_protocol_delta_pp": -0.1417477058914116,
  "mean_delta_pp": 0.25169766602085986,
  "mean_fusion_increment_pp": 0.2023565364445119
}

Gate FAILED. Preserve original coupling. No full rollout, strength tuning, audit-word bans or per-domain routing.

## Fixed-Class Paired Metrics

Baseline union-positive classes are fixed before actions; scored zero-union classes remain IoU0. Raw standard metrics are separately preserved below.

| Dataset/protocol | Geometry | Anchored_Exact | ClassRelativeReject_CG | TargetContext_Exact | ContextOnly_Exact | ShuffledSupport_Exact | ShuffledAlias0_Exact | ShuffledAlias1_Exact | ShuffledAlias2_Exact | MeanLogit_Original | MeanLogit_TargetContext |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 53.739421 | 54.455821 | 51.862834 | 54.712948 | 54.425523 | 54.442462 | 54.422568 | 53.979454 | 54.754018 |
| potsdam/potsdam | 35.703546 | 38.920258 | 39.141453 | 38.949938 | 38.180637 | 38.705541 | 38.913950 | 38.921392 | 38.926613 | 38.967769 | 38.991858 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.170638 | 28.877049 | 28.682729 | 28.514209 | 28.770688 | 28.815465 | 28.785449 | 28.427401 | 29.071982 |
| oem/oem | 39.820583 | 39.023152 | 39.157933 | 39.076361 | 39.389256 | 38.919006 | 39.052029 | 39.011042 | 39.015797 | 38.639963 | 38.677759 |
| loveda/P | 49.539928 | 50.706552 | 50.935333 | 50.681958 | 48.045271 | 51.836092 | 50.191069 | 50.164385 | 50.730409 | 41.845129 | 41.667288 |
| loveda/D | 33.877920 | 30.736030 | 30.729095 | 31.446522 | 30.821280 | 32.594840 | 31.369209 | 31.369831 | 31.349376 | 27.868953 | 28.176375 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.588246 | 51.755697 | 51.263929 | 51.630259 | 51.781464 | 51.769381 | 51.820278 | 50.772066 | 50.718882 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.922266 | 66.929129 | 66.709903 | 66.670479 | 66.956245 | 66.958451 | 66.932060 | 66.426320 | 66.456105 |
| flair1/flair1 | 35.605867 | 33.608404 | 33.476562 | 33.466656 | 33.918403 | 33.417430 | 33.444877 | 33.426929 | 33.473475 | 32.364820 | 32.218620 |
| Equal-domain mean | 40.531033 | 42.867949 | 42.865702 | 43.119647 | 42.603621 | 43.145589 | 43.089248 | 43.089369 | 43.090702 | 42.180843 | 42.383200 |

## Standard Metrics

| Dataset/protocol | Geometry | Anchored_Exact | ClassRelativeReject_CG | TargetContext_Exact | ContextOnly_Exact | ShuffledSupport_Exact | ShuffledAlias0_Exact | ShuffledAlias1_Exact | ShuffledAlias2_Exact | MeanLogit_Original | MeanLogit_TargetContext |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 38.469448 | 53.746954 | 53.739421 | 54.455821 | 51.862834 | 54.712948 | 54.425523 | 54.442462 | 54.422568 | 53.979454 | 54.754018 |
| potsdam/potsdam | 35.703546 | 38.920258 | 39.141453 | 38.949938 | 38.180637 | 38.705541 | 38.913950 | 38.921392 | 38.926613 | 38.967769 | 38.991858 |
| udd5/udd5 | 30.500979 | 28.175811 | 28.170638 | 28.877049 | 28.682729 | 28.514209 | 28.770688 | 28.815465 | 28.785449 | 28.427401 | 29.071982 |
| oem/oem | 39.820583 | 39.023152 | 39.157933 | 39.076361 | 39.389256 | 38.919006 | 39.052029 | 39.011042 | 39.015797 | 38.639963 | 38.677759 |
| loveda/P | 49.539928 | 50.706552 | 50.935333 | 50.681958 | 48.045271 | 51.836092 | 50.191069 | 50.164385 | 50.730409 | 41.845129 | 41.667288 |
| loveda/D | 33.877920 | 30.736030 | 30.729095 | 31.446522 | 30.821280 | 32.594840 | 31.369209 | 31.369831 | 31.349376 | 27.868953 | 28.176375 |
| vaihingen/vaihingen | 49.365065 | 51.826962 | 51.588246 | 51.755697 | 51.263929 | 51.630259 | 51.781464 | 51.769381 | 51.820278 | 50.772066 | 50.718882 |
| landcoverai/landcoverai | 60.904853 | 66.906020 | 66.922266 | 66.929129 | 66.709903 | 66.670479 | 66.956245 | 66.958451 | 66.932060 | 66.426320 | 66.456105 |
| flair1/flair1 | 38.842764 | 33.608404 | 33.476562 | 33.466656 | 33.918403 | 33.417430 | 33.444877 | 33.426929 | 33.473475 | 32.364820 | 32.218620 |
| Equal-domain mean | 40.935645 | 42.867949 | 42.865702 | 43.119647 | 42.603621 | 43.145589 | 43.089248 | 43.089369 | 43.090702 | 42.180843 | 42.383200 |

## Activity And Source Ranking

AUC relates frozen capacity-weighted per-image mean risk to the prior labeled GLOBAL single-alias attenuation utility. It is not querywise correctness, significance or evidence of useful deployment by itself.

| Dataset/protocol | Mean rejection | Positive risk fraction | Supported query fraction | Old action AUC | New action AUC | Context-only AUC | Shuffled-support AUC | Beneficial pixels | Harmful pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 0.079051 | 0.653505 | 1.000000 | 0.4732 | 0.5854 | 0.3367 | 0.5502 | 10515 | 3627 |
| potsdam/potsdam | 0.064494 | 0.493555 | 1.000000 | 0.4905 | 0.4694 | 0.5925 | 0.4675 | 7137 | 6569 |
| udd5/udd5 | 0.085932 | 0.564795 | 1.000000 | 0.4500 | 0.4157 | 0.5586 | 0.7957 | 9608 | 928 |
| oem/oem | 0.096574 | 0.584106 | 1.000000 | 0.4815 | 0.3148 | 0.5315 | 0.4973 | 12763 | 5169 |
| loveda/P | 0.058827 | 0.546459 | 1.000000 | 0.4647 | 0.4723 | 0.5282 | 0.6306 | 655 | 1800 |
| loveda/D | 0.067774 | 0.573093 | 1.000000 | 0.3924 | 0.6003 | 0.4273 | 0.5981 | 34796 | 2679 |
| vaihingen/vaihingen | 0.079693 | 0.520870 | 1.000000 | 0.4069 | 0.4756 | 0.5222 | 0.4395 | 4201 | 6990 |
| landcoverai/landcoverai | 0.058644 | 0.472080 | 1.000000 | 0.4021 | 0.7721 | 0.4535 | 0.4442 | 4522 | 2635 |
| flair1/flair1 | 0.068021 | 0.515946 | 1.000000 | 0.2900 | 0.4325 | 0.4622 | 0.4121 | 6233 | 14117 |

## Prior Action Examples: New Frozen Risk

The action utility is the previous labeled global-window attenuation audit, not an alias chosen for deployment. Scores here are the frozen capacity-weighted per-image means; unknown/fallback and competition can alter the actual joint result.

| Dataset / class / alias | Prior action delta pp | Old score | New score | Context-only score | Shuffled-support score |
| --- | ---: | ---: | ---: | ---: | ---: |
| udd5 / vehicle / cars seen from above | +2.261230 | 0.000000 | 0.144588 | 0.847159 | 0.187065 |
| potsdam / car / traffic vehicle | +0.640937 | 0.000000 | 0.046410 | 0.474668 | 0.077331 |
| vdd / vehicle / vehicles | +0.570833 | 0.000000 | 0.116516 | 0.775869 | 0.133693 |
| potsdam / impervious surface / road pavement | -0.772698 | 0.103774 | 0.032902 | 0.398882 | 0.013291 |
| vaihingen / impervious surface / parking pavement | -1.209770 | 0.076618 | 0.035135 | 0.535908 | 0.003685 |
| loveda / farm / farmland | +1.411040 | 0.000000 | 0.092491 | 0.679320 | 0.268746 |

## Interpretation

Primary improves six of eight clean domains and exceeds each exact-spectrum alias shuffle in the domain mean. This is better evidence than the prior almost-neutral source, but the advantage over alias shuffles is small (about0.029pp above the strongest shuffle) and not a significance claim. Context-only suppression loses in the mean; adding competing-class evidence is important for this fixed operator.

The predeclared spatial-support control scores higher in the mean, so Geometry-specific target support is NOT established. This control preserves MASK area/value spectra, not resulting RETENTION spectra: different masking responses and class calibration remain possible explanations. It neither proves all Geometry support unnecessary nor licenses promoting a favorable dataset. Potsdam car still loses while impervious coverage also declines.

Do not run the full benchmark or tune support/strength from these labels. The next source decision must separate spatial specificity from aggregate class calibration with already frozen observations, before another model sweep. The added source requires64-150.5 forward passes per audited window on average; the cost is material and no amortized deployment claim is established.


## Class Coverage And False Activation

| Dataset/protocol/class | Original IoU | Primary IoU | Delta pp | Original precision | Primary precision | Original recall | Primary recall | Original area | Primary area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd/other | 59.7766 | 60.2012 | 0.4246 | 81.1541 | 80.9440 | 69.4120 | 70.1422 | 43.6684 | 44.2423 |
| vdd/vdd/wall | 59.6816 | 59.9532 | 0.2716 | 65.8225 | 66.3858 | 86.4811 | 86.0867 | 10.4520 | 10.3160 |
| vdd/vdd/road | 23.2250 | 23.1865 | -0.0385 | 23.3759 | 23.3132 | 97.2962 | 97.7099 | 13.8146 | 13.9107 |
| vdd/vdd/vegetation | 44.1426 | 44.2118 | 0.0692 | 94.9106 | 95.3073 | 45.2128 | 45.1956 | 10.9733 | 10.9234 |
| vdd/vdd/vehicle | 50.3084 | 52.6445 | 2.3361 | 50.3286 | 52.7000 | 99.9201 | 99.8003 | 0.7110 | 0.6782 |
| vdd/vdd/roof | 86.2607 | 86.8556 | 0.5949 | 86.7114 | 87.3512 | 99.4009 | 99.3511 | 6.9072 | 6.8532 |
| vdd/vdd/water | 52.8339 | 54.1380 | 1.3041 | 55.7407 | 57.2870 | 91.0164 | 90.7823 | 13.4736 | 13.0762 |
| potsdam/potsdam/impervious surface | 65.8977 | 65.2643 | -0.6334 | 85.6706 | 85.3578 | 74.0609 | 73.4919 | 35.3370 | 35.1941 |
| potsdam/potsdam/building | 71.0529 | 70.8956 | -0.1573 | 74.9985 | 74.7569 | 93.1063 | 93.2092 | 20.0156 | 20.1025 |
| potsdam/potsdam/low vegetation | 14.4124 | 15.4050 | 0.9926 | 79.9195 | 81.3387 | 14.9539 | 15.9694 | 3.2447 | 3.4046 |
| potsdam/potsdam/tree | 55.5064 | 55.8624 | 0.3560 | 91.9038 | 91.9003 | 58.3601 | 58.7552 | 11.2039 | 11.2802 |
| potsdam/potsdam/car | 24.5378 | 24.1387 | -0.3991 | 24.5936 | 24.1886 | 99.0837 | 99.1542 | 14.4459 | 14.6983 |
| potsdam/potsdam/clutter | 2.1143 | 2.1336 | 0.0193 | 2.6529 | 2.6932 | 9.4321 | 9.3126 | 15.7528 | 15.3204 |
| udd5/udd5/vegetation | 46.2799 | 47.6083 | 1.3284 | 93.8774 | 93.1707 | 47.7203 | 49.3297 | 1.5693 | 1.6346 |
| udd5/udd5/building | 83.4871 | 83.5377 | 0.0506 | 83.6594 | 83.7470 | 99.7539 | 99.7017 | 85.7625 | 85.6280 |
| udd5/udd5/road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2949 | 0.2585 |
| udd5/udd5/vehicle | 5.3645 | 5.9193 | 0.5548 | 5.3645 | 5.9194 | 99.9773 | 99.9773 | 7.8274 | 7.0937 |
| udd5/udd5/other | 5.7476 | 7.3200 | 1.5724 | 29.7284 | 32.5541 | 6.6513 | 8.6285 | 4.5458 | 5.3853 |
| oem/oem/bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.2072 | 7.7257 |
| oem/oem/rangeland | 49.5261 | 48.8609 | -0.6652 | 70.5348 | 70.5025 | 62.4456 | 61.4162 | 12.8780 | 12.6715 |
| oem/oem/developed space | 23.7669 | 25.4214 | 1.6545 | 38.9362 | 40.4478 | 37.8898 | 40.6278 | 19.0710 | 19.6849 |
| oem/oem/road | 54.5529 | 54.4536 | -0.0993 | 68.0810 | 67.9967 | 73.3007 | 73.2190 | 6.7693 | 6.7701 |
| oem/oem/tree | 55.6605 | 55.7997 | 0.1392 | 91.7802 | 91.5938 | 58.5807 | 58.8115 | 16.8622 | 16.9631 |
| oem/oem/water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oem/oem/agriculture land | 81.6500 | 81.4126 | -0.2374 | 82.0066 | 81.7680 | 99.4703 | 99.4689 | 25.7312 | 25.8059 |
| oem/oem/building | 47.0287 | 46.6627 | -0.3660 | 68.3392 | 68.3329 | 60.1297 | 59.5374 | 10.4811 | 10.3789 |
| loveda/P/building | 60.8479 | 61.7571 | 0.9092 | 72.1893 | 74.9216 | 79.4788 | 77.8502 | 0.0292 | 0.0275 |
| loveda/P/road | 53.7553 | 53.1125 | -0.6428 | 54.7280 | 54.0606 | 96.7995 | 96.8037 | 14.7198 | 14.9022 |
| loveda/P/water | 62.4091 | 62.1478 | -0.2613 | 68.1253 | 68.2053 | 88.1486 | 87.4963 | 19.0394 | 18.8764 |
| loveda/P/barren | 4.3215 | 3.9695 | -0.3520 | 89.7378 | 90.3493 | 4.3430 | 3.9864 | 0.5366 | 0.4892 |
| loveda/P/tree | 39.9553 | 40.1305 | 0.1752 | 99.3680 | 99.3515 | 40.0571 | 40.2359 | 6.6373 | 6.6680 |
| loveda/P/farm | 82.9502 | 82.9743 | 0.0241 | 83.2671 | 83.2810 | 99.5433 | 99.5582 | 59.0377 | 59.0366 |
| loveda/D/background | 7.3224 | 10.6467 | 3.3243 | 75.1732 | 83.3341 | 7.5039 | 10.8783 | 4.4672 | 5.8419 |
| loveda/D/building | 31.2821 | 32.3848 | 1.1027 | 34.0307 | 35.6716 | 79.4788 | 77.8502 | 0.0342 | 0.0319 |
| loveda/D/road | 44.1270 | 43.4899 | -0.6371 | 44.7837 | 44.1278 | 96.7840 | 96.7829 | 9.9365 | 10.0841 |
| loveda/D/water | 54.4379 | 54.1036 | -0.3343 | 58.7421 | 58.6444 | 88.1369 | 87.4805 | 12.1974 | 12.1267 |
| loveda/D/barren | 3.2414 | 2.9271 | -0.3143 | 82.3610 | 81.6965 | 3.2640 | 2.9464 | 0.2428 | 0.2209 |
| loveda/D/tree | 35.8674 | 36.8241 | 0.9567 | 97.1427 | 97.1307 | 36.2498 | 37.2291 | 3.3944 | 3.4865 |
| loveda/D/farm | 38.8741 | 39.7495 | 0.8754 | 38.9455 | 39.8210 | 99.5311 | 99.5507 | 69.7275 | 68.2078 |
| vaihingen/vaihingen/impervious surface | 57.3052 | 57.1078 | -0.1974 | 73.8225 | 72.9878 | 71.9196 | 72.4122 | 26.6125 | 27.1012 |
| vaihingen/vaihingen/building | 65.7437 | 65.8598 | 0.1161 | 65.8760 | 65.9960 | 99.6954 | 99.6877 | 31.2219 | 31.1628 |
| vaihingen/vaihingen/low vegetation | 43.7153 | 42.7248 | -0.9905 | 96.1772 | 96.3505 | 44.4883 | 43.4276 | 13.6772 | 13.3271 |
| vaihingen/vaihingen/tree | 67.2612 | 67.2409 | -0.0203 | 79.0950 | 78.8506 | 81.8036 | 82.0365 | 21.3454 | 21.4725 |
| vaihingen/vaihingen/car | 25.1095 | 25.8452 | 0.7357 | 25.2570 | 26.0031 | 97.7270 | 97.7037 | 7.1430 | 6.9364 |
| landcoverai/landcoverai/background | 87.4477 | 87.5369 | 0.0892 | 93.8549 | 94.1455 | 92.7587 | 92.5764 | 66.9686 | 66.6306 |
| landcoverai/landcoverai/building | 44.4672 | 44.1597 | -0.3075 | 44.8124 | 44.4956 | 98.2973 | 98.3193 | 3.3233 | 3.3477 |
| landcoverai/landcoverai/woodland | 78.6026 | 79.4007 | 0.7981 | 94.9565 | 94.7969 | 82.0272 | 83.0186 | 18.4920 | 18.7470 |
| landcoverai/landcoverai/water | 97.6178 | 97.6244 | 0.0066 | 97.6178 | 97.6244 | 100.0000 | 100.0000 | 8.4732 | 8.4726 |
| landcoverai/landcoverai/road | 26.3947 | 25.9240 | -0.4707 | 28.8528 | 28.2785 | 75.5990 | 75.6901 | 2.7429 | 2.8020 |
| flair1/flair1/building | 56.7755 | 56.4909 | -0.2846 | 58.0396 | 57.6383 | 96.3057 | 96.5961 | 11.9131 | 12.0322 |
| flair1/flair1/pervious surface | 47.2753 | 47.0237 | -0.2516 | 91.6998 | 92.1297 | 49.3887 | 48.9917 | 9.2421 | 9.1250 |
| flair1/flair1/impervious surface | 51.9858 | 51.6353 | -0.3505 | 59.7781 | 59.2782 | 79.9522 | 80.0192 | 21.9112 | 22.1145 |
| flair1/flair1/bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1475 | 7.3173 |
| flair1/flair1/water | 61.4565 | 62.0656 | 0.6091 | 63.2502 | 63.8889 | 95.5889 | 95.6039 | 6.7041 | 6.6381 |
| flair1/flair1/coniferous | 43.5282 | 43.4192 | -0.1090 | 65.7321 | 65.2673 | 56.3052 | 56.4664 | 0.4815 | 0.4863 |
| flair1/flair1/deciduous | 59.5237 | 60.0711 | 0.5474 | 78.8015 | 78.3090 | 70.8722 | 72.0616 | 15.3089 | 15.6637 |
| flair1/flair1/brushwood | 12.3287 | 12.1635 | -0.1652 | 26.2213 | 27.7179 | 18.8771 | 17.8141 | 3.5408 | 3.1610 |
| flair1/flair1/vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0016 | 0.0016 |
| flair1/flair1/herbaceous vegetation | 57.4725 | 56.1852 | -1.2873 | 90.8252 | 91.0616 | 61.0148 | 59.4646 | 21.5207 | 20.9195 |
| flair1/flair1/agricultural land | 12.9546 | 12.5455 | -0.4091 | 13.9145 | 13.3752 | 65.2534 | 66.9118 | 1.4300 | 1.5254 |
| flair1/flair1/plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7986 | 1.0154 |

## Cost

Suite wall seconds: 166.1478. Shared control experiment, not standalone inference cost.

| Dataset | Worker seconds | Peak allocated MiB | True source forwards/image | True source seconds/image | Unmasked source seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 33.7940 | 5560.3057 | 64.00 | 1.4154 | 0.0227 |
| potsdam | 62.3706 | 5556.9180 | 144.00 | 3.1621 | 0.0876 |
| udd5 | 31.4288 | 5553.6367 | 64.00 | 1.4079 | 0.0227 |
| oem | 66.7863 | 5564.4814 | 150.50 | 3.3610 | 0.0893 |
| loveda | 71.3523 | 5579.9956 | 144.00 | 3.1548 | 0.0865 |
| vaihingen | 61.9617 | 5553.6367 | 144.00 | 3.1791 | 0.0884 |
| landcoverai | 60.3095 | 5553.6367 | 144.00 | 3.0931 | 0.0856 |
| flair1 | 64.3425 | 5576.4824 | 144.00 | 3.1245 | 0.0864 |

Nine mathematical tests and mask-free actual-checkpoint smoke passed. Exact original per-image Geometry/Anchored_Exact/ClassRelativeReject_CG controls, signatures,20-word groups, confusion sums and transition endpoints independently verified. Identity writer/canonical protection/alias spectrum/spatial support spectrum checks pass. Raw pre-mask numerical caches stay remote; merged/per-image/source-score results are local. Target-kept content and Geometry support are NOT trusted labels; masks may be mixed and occluded inputs OOD. Neither risk magnitude nor mean rank AUC can certify semantic correctness.
