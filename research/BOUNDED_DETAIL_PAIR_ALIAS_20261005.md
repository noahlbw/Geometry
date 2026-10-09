# Global Detail2: Frozen Sparse Competitive Alias Pilot

UDD5 full40 and eight complete images/other domain;96 developed images, not full eight datasets or independent validation. Fixed20 aliases, bounded896 Geometry/448 wide, original patch-only2 baseline and H. At most two detail RGB encodings per COMPLETE image, selected globally before masks. Detail uses frozen native DINO.text, not VIP proxy. Sparse baseline-top2 alias weights reuse the contradiction rule; this is not a new semantic-correctness probability.

| Dataset/protocol | Geometry | NoAdmission_Exact | Geometry_PatchOnly2Coupled | Detail2_RivalSoft | Detail2_ClassMean | Detail2_AliasShuffle | Detail2_ObservationMean | Detail2_RivalHard | Primary-baseline pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 42.7217 | 46.3491 | 46.8468 | 46.8323 | 46.8521 | 46.9021 | 46.4052 | 46.5833 | -0.0145 |
| potsdam/potsdam | 45.6466 | 49.7562 | 51.5168 | 51.3694 | 51.5154 | 51.6201 | 50.8465 | 51.1022 | -0.1474 |
| udd5/udd5 | 47.1219 | 47.0096 | 47.0786 | 47.1069 | 47.0805 | 47.0071 | 47.4508 | 47.1604 | +0.0283 |
| oem/oem | 31.4297 | 25.7311 | 25.2530 | 25.2389 | 25.2509 | 25.2555 | 25.2365 | 25.2386 | -0.0141 |
| loveda/P | 70.8274 | 67.0933 | 67.2104 | 67.3866 | 67.2170 | 67.2682 | 68.6392 | 68.2719 | +0.1762 |
| loveda/D | 46.6369 | 41.0700 | 39.4705 | 39.6433 | 39.4751 | 39.4240 | 41.0024 | 40.5395 | +0.1728 |
| vaihingen/vaihingen | 46.0831 | 48.3789 | 49.1468 | 49.5124 | 49.1423 | 49.2970 | 50.1929 | 49.7835 | +0.3656 |
| landcoverai/landcoverai | 57.9352 | 75.1447 | 76.2441 | 76.2025 | 76.2388 | 76.2463 | 75.9127 | 76.1297 | -0.0416 |
| flair1/flair1 | 39.8935 | 37.2481 | 37.6036 | 37.5169 | 37.5980 | 37.6034 | 37.5692 | 37.2729 | -0.0867 |

| Arm | Equal-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| Detail2_RivalSoft | 46.6778 |
| Detail2_ClassMean | 46.6441 |
| Detail2_AliasShuffle | 46.6694 |
| Detail2_ObservationMean | 46.8270 |
| Detail2_RivalHard | 46.7263 |

## Matched Complete-Image Timing

| Domain | Baseline ms | Primary ms | VIP20 ms | Primary/baseline |
| --- | ---: | ---: | ---: | ---: |
| vdd | 325.37 | 441.89 | 124.73 | 1.358x |
| potsdam | 249.40 | 393.65 | 104.76 | 1.578x |
| udd5 | 390.52 | 494.96 | 166.84 | 1.267x |
| oem | 239.11 | 382.08 | 103.22 | 1.598x |
| loveda | 252.25 | 484.13 | 204.08 | 1.919x |
| vaihingen | 231.08 | 366.63 | 97.87 | 1.587x |
| landcoverai | 227.29 | 364.81 | 95.54 | 1.605x |
| flair1 | 235.88 | 385.58 | 102.85 | 1.635x |

Three fixed complete inputs/domain, three synchronized singleton repeats, serial idle-GPU timing. Includes all views, alias action, H writing, restoration and argmax; excludes loading/text/decoding/masks. Shared-resident peaks are not standalone memory.

## Frozen Advancement Gate

```json
{
  "passed": false,
  "checks": {
    "mean_gain": false,
    "domain_wins": false,
    "worst_protocol_loss": true,
    "above_class_mean": false,
    "above_alias_shuffle": false,
    "above_same_source_mean": false,
    "all_eight_timings": true,
    "mean_cost_ratio": false,
    "domain_mean_below1000ms": true
  },
  "domain_wins": 3,
  "mean_gain_pp": 0.03279999999999461,
  "worst_protocol_delta_pp": -0.14740000000000464
}
```

No post-result primary, threshold or domain-route changes and no automatic full20092 promotion. A passed developed pilot still requires vocabulary stress and unchanged full-domain validation. A failed gate rejects this implementation, not every possible conditional alias mechanism.

## Coverage And Competition Outcomes

Confusion rows are targets and columns are predictions. Delta TP counts changes in correct coverage; delta FP counts changes in competitor activation. Negative delta FP is beneficial, but does not establish useful suppression if correct coverage also falls.

| Dataset/protocol | Class | Baseline IoU | Soft IoU | Delta pp | Delta TP pixels | Delta FP pixels | Baseline precision | Soft precision | Baseline recall | Soft recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.6232 | 20.4833 | -0.1399 | -4089 | +176832 | 32.3110 | 31.9884 | 36.3108 | 36.2859 |
| vdd/vdd | wall | 46.3090 | 46.2958 | -0.0132 | -2111 | -3255 | 84.1267 | 84.2224 | 50.7428 | 50.6921 |
| vdd/vdd | road | 15.2940 | 15.2474 | -0.0466 | +4515 | +57294 | 15.7348 | 15.6760 | 84.5199 | 84.7951 |
| vdd/vdd | vegetation | 45.8052 | 45.0352 | -0.7700 | -219849 | -5453 | 96.1118 | 96.0868 | 46.6700 | 45.8766 |
| vdd/vdd | vehicle | 23.8613 | 24.8682 | +1.0069 | -291 | -43642 | 24.5176 | 25.5903 | 89.9132 | 89.8087 |
| vdd/vdd | roof | 85.4306 | 85.4262 | -0.0044 | +171 | +1831 | 85.8504 | 85.8455 | 99.4309 | 99.4315 |
| vdd/vdd | water | 90.6042 | 90.4699 | -0.1343 | +3438 | +34609 | 91.1218 | 90.9708 | 99.3769 | 99.3951 |
| potsdam/potsdam | impervious surface | 75.3698 | 75.1888 | -0.1810 | -2311 | +5833 | 84.9896 | 84.8281 | 86.9432 | 86.8711 |
| potsdam/potsdam | building | 80.4415 | 80.4747 | +0.0332 | +1 | -559 | 80.4757 | 80.5090 | 99.9471 | 99.9472 |
| potsdam/potsdam | low vegetation | 60.1752 | 60.0306 | -0.1446 | -3641 | -560 | 93.9502 | 93.9711 | 62.6009 | 62.4352 |
| potsdam/potsdam | tree | 57.3938 | 57.2948 | -0.0990 | -1545 | -168 | 77.3966 | 77.3764 | 68.9511 | 68.8243 |
| potsdam/potsdam | car | 31.4528 | 30.9613 | -0.4915 | +8 | +8393 | 31.5636 | 31.0682 | 98.8959 | 98.9006 |
| potsdam/potsdam | clutter | 4.2680 | 4.2665 | -0.0015 | -229 | -5222 | 5.7679 | 5.7980 | 14.0993 | 13.9057 |
| udd5/udd5 | vegetation | 66.2953 | 66.2760 | -0.0193 | -31289 | -8173 | 96.5410 | 96.5484 | 67.9084 | 67.8844 |
| udd5/udd5 | building | 81.2771 | 81.2831 | +0.0060 | +35002 | +27605 | 82.2868 | 82.2788 | 98.5127 | 98.5330 |
| udd5/udd5 | road | 37.2416 | 37.2393 | -0.0023 | +19037 | +55533 | 65.9709 | 65.8979 | 46.0967 | 46.1289 |
| udd5/udd5 | vehicle | 16.9950 | 17.1533 | +0.1583 | -2069 | -161979 | 17.8319 | 18.0095 | 78.3585 | 78.2998 |
| udd5/udd5 | other | 33.5843 | 33.5827 | -0.0016 | +15237 | +51096 | 47.1961 | 47.1772 | 53.7992 | 53.8197 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0.0000 | +0 | -1146 | 0.0000 | 0.0000 | N/A | N/A |
| oem/oem | rangeland | 4.3949 | 4.3512 | -0.0437 | -608 | -468 | 66.4403 | 66.5619 | 4.4947 | 4.4484 |
| oem/oem | developed space | 21.7107 | 21.7127 | +0.0020 | +1271 | +5466 | 24.6999 | 24.6894 | 64.2081 | 64.2970 |
| oem/oem | road | 26.4384 | 26.4517 | +0.0133 | +142 | +195 | 45.0353 | 45.0328 | 39.0337 | 39.0645 |
| oem/oem | tree | 10.8604 | 10.8333 | -0.0271 | -305 | -110 | 86.7166 | 86.7573 | 11.0442 | 11.0154 |
| oem/oem | water | 18.0795 | 18.0628 | -0.0167 | +18 | +205 | 22.5533 | 22.5182 | 47.6827 | 47.7242 |
| oem/oem | agriculture land | 69.4583 | 69.4272 | -0.0311 | +14 | +486 | 69.6777 | 69.6454 | 99.5487 | 99.5506 |
| oem/oem | building | 51.0820 | 51.0722 | -0.0098 | -1957 | -3203 | 78.2527 | 78.3501 | 59.5335 | 59.4640 |
| loveda/P | building | 88.2412 | 88.2283 | -0.0129 | +19 | +75 | 92.8501 | 92.8305 | 94.6743 | 94.6799 |
| loveda/P | road | 79.8873 | 79.8315 | -0.0558 | +70 | +767 | 83.5543 | 83.4867 | 94.7923 | 94.8008 |
| loveda/P | water | 62.0850 | 62.9650 | +0.8800 | +12504 | +146 | 98.5503 | 98.5545 | 62.6572 | 63.5519 |
| loveda/P | barren | 70.3607 | 70.3866 | +0.0259 | +0 | -78 | 71.3382 | 71.3648 | 98.0897 | 98.0897 |
| loveda/P | tree | 39.2222 | 39.1600 | -0.0622 | -703 | -33 | 94.6144 | 94.6129 | 40.1179 | 40.0530 |
| loveda/P | farm | 63.4659 | 63.7481 | +0.2822 | -74 | -12693 | 64.3610 | 64.6531 | 97.8555 | 97.8515 |
| loveda/D | background | 4.0203 | 4.2688 | +0.2485 | +6917 | +79 | 49.2011 | 50.6838 | 4.1944 | 4.4538 |
| loveda/D | building | 45.7347 | 45.7393 | +0.0046 | -2 | -76 | 47.1134 | 47.1185 | 93.9861 | 93.9855 |
| loveda/D | road | 63.0492 | 62.9978 | -0.0514 | +61 | +1100 | 65.4549 | 65.3959 | 94.4918 | 94.4993 |
| loveda/D | water | 58.9731 | 59.7985 | +0.8254 | +12436 | +303 | 90.9392 | 91.0265 | 62.6546 | 63.5445 |
| loveda/D | barren | 27.9609 | 27.9989 | +0.0380 | -1 | -723 | 28.1841 | 28.2228 | 97.2456 | 97.2450 |
| loveda/D | tree | 34.5990 | 34.5530 | -0.0460 | -565 | -70 | 81.3407 | 81.3310 | 37.5818 | 37.5296 |
| loveda/D | farm | 41.9559 | 42.1468 | +0.1909 | -33 | -19426 | 42.4483 | 42.6440 | 97.3098 | 97.3080 |
| vaihingen/vaihingen | impervious surface | 60.8940 | 61.4581 | +0.5641 | +18070 | +1423 | 84.1446 | 84.2196 | 68.7868 | 69.4564 |
| vaihingen/vaihingen | building | 65.1989 | 65.2608 | +0.0619 | -33 | -3069 | 65.4089 | 65.4719 | 99.5099 | 99.5083 |
| vaihingen/vaihingen | low vegetation | 27.2861 | 27.3361 | +0.0500 | +780 | +23 | 84.6746 | 84.6946 | 28.7036 | 28.7567 |
| vaihingen/vaihingen | tree | 65.8712 | 65.8702 | -0.0010 | +653 | +1022 | 77.8059 | 77.7673 | 81.1120 | 81.1524 |
| vaihingen/vaihingen | car | 26.4838 | 27.6368 | +1.1530 | +1 | -18870 | 27.2384 | 28.4595 | 90.5299 | 90.5307 |
| landcoverai/landcoverai | background | 82.5166 | 82.4885 | -0.0281 | -512 | -358 | 93.9248 | 93.9697 | 87.1691 | 87.0991 |
| landcoverai/landcoverai | building | N/A | N/A | N/A | +0 | +0 | N/A | N/A | N/A | N/A |
| landcoverai/landcoverai | woodland | 91.5012 | 91.5003 | -0.0009 | +150 | +175 | 95.2495 | 95.2337 | 95.8766 | 95.8916 |
| landcoverai/landcoverai | water | 91.7035 | 91.6035 | -0.1000 | +0 | +397 | 91.7116 | 91.6116 | 99.9904 | 99.9904 |
| landcoverai/landcoverai | road | 39.2550 | 39.2177 | -0.0373 | +28 | +120 | 46.0574 | 45.9655 | 72.6614 | 72.7628 |
| flair1/flair1 | building | 49.1283 | 48.9890 | -0.1393 | -214 | +117 | 51.3696 | 51.2814 | 91.8434 | 91.6380 |
| flair1/flair1 | pervious surface | 17.8467 | 17.9787 | +0.1320 | +87 | +71 | 64.2446 | 64.1526 | 19.8148 | 19.9866 |
| flair1/flair1 | impervious surface | 66.4519 | 66.3896 | -0.0623 | +4 | +410 | 74.8732 | 74.7932 | 85.5244 | 85.5256 |
| flair1/flair1 | bare soil | 21.3471 | 20.8897 | -0.4574 | +1 | +1728 | 21.3761 | 20.9173 | 99.3672 | 99.3731 |
| flair1/flair1 | water | 94.9931 | 94.9931 | +0.0000 | -1 | -1 | 99.7998 | 99.8017 | 95.1745 | 95.1727 |
| flair1/flair1 | coniferous | 5.9298 | 5.9298 | +0.0000 | +0 | +0 | 7.4090 | 7.4090 | 22.8997 | 22.8997 |
| flair1/flair1 | deciduous | 64.3305 | 64.3305 | +0.0000 | -83 | -129 | 81.0241 | 81.0634 | 75.7421 | 75.7077 |
| flair1/flair1 | brushwood | 3.3685 | 3.3322 | -0.0363 | -57 | +136 | 6.8848 | 6.8096 | 6.1874 | 6.1257 |
| flair1/flair1 | vineyard | 51.9276 | 51.9267 | -0.0009 | +0 | +4 | 52.0611 | 52.0602 | 99.5087 | 99.5087 |
| flair1/flair1 | herbaceous vegetation | 37.8267 | 37.5430 | -0.2837 | -1792 | +68 | 86.6034 | 86.4960 | 40.1776 | 39.8805 |
| flair1/flair1 | agricultural land | 38.0928 | 37.9004 | -0.1924 | -1028 | -84 | 83.6119 | 83.5701 | 41.1664 | 40.9519 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0.0000 | +0 | +763 | 0.0000 | 0.0000 | N/A | N/A |
