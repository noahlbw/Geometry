# Frozen Patch-Only Strength2: Full Eight-Domain Comparison

Signature `geometry-bounded896-patch-only-strength2-coupled-v1-20261005`. Completed 8/8 full domains. Retained every original20 alias/class, bounded896 Geometry and independent448 wide views, original Geometry reconstruction relation and wide calibration. No fine/native encodings or alias admission. Patch-only strength2 is one frozen configuration for all domains, not a new innovation or per-domain winner selection. These developed domains are exploratory validation, not untouched independent evidence.

| Dataset/protocol | Images | VIP20 | Official/distilled VIP | Stronger measured VIP | Geometry | Original coupled | Patch-only1 | Frozen patch-only2 | SCLIP coupled | Delta vs stronger VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 1669 | 56.3110 | 55.0261 | 56.3110 | 66.5715 | 62.5150 | 62.4770 | 61.4752 | 60.4750 | +5.1642 |
| loveda/D | 1669 | 36.1457 | 35.3436 | 36.1457 | 42.8650 | 40.0210 | 39.4554 | 38.8353 | 38.1615 | +2.6896 |
| udd5/udd5 | 40 | 41.4336 | 44.9713 | 44.9713 | 47.1219 | 47.0096 | 47.0654 | 47.0786 | 46.7075 | +2.1073 |
| oem/oem | 384 | 33.2064 | 35.0388 | 35.0388 | 43.9751 | 37.2104 | 37.8737 | 36.8400 | 36.1612 | +1.8012 |
| vdd/vdd | 80 | 51.4827 | 52.0647 | 52.0647 | 46.6010 | 53.8090 | 53.5838 | 53.4600 | 53.5257 | +1.3953 |
| potsdam/potsdam | 504 | 42.3028 | 44.8995 | 44.8995 | 40.3779 | 43.4301 | 44.5000 | 45.9339 | 45.6185 | +1.0344 |
| vaihingen/vaihingen | 113 | 47.3620 | 41.9062 | 47.3620 | 48.2368 | 50.7949 | 50.6222 | 51.4282 | 51.1076 | +4.0662 |
| landcoverai/landcoverai | 1602 | 54.9420 | 50.2980 | 54.9420 | 57.0988 | 63.6099 | 64.5704 | 65.1069 | 65.2173 | +10.1649 |
| flair1/flair1 | 15700 | 35.2977 | 36.6074 | 36.6074 | 40.5208 | 42.5038 | 42.9658 | 42.4363 | 42.1597 | +5.8289 |

Mean over 8 completed domains, counting LoveDA D once: primary 47.6399; stronger measured VIP 44.0039.

Matched coupled control: primary minus SCLIP mean +0.3075pp; wins 6/8 completed domains. This is a frozen DINO.text operator adaptation, not the official SCLIP system. SCLIP changes both prefix and patch reads while patch-only2 blocks prefix Values for patch queries; the comparison does not isolate only the affinity kernel.

VIP20 is the finite-row-repaired matched20 local adaptation. The separate official/distilled comparator uses official short words on VDD/Potsdam/corrected-input Vaihingen and frozen distilled words on five other domains. These are not published VIP benchmark numbers. Stronger VIP is the maximum of these two completed protocols, not a claim of the best possible VIP. LandCover.ai replaces unlabeled iSAID.

## Complete-Image Cost

| Domain | VIP20 ms | Coupled ms | Patch-only2 ms | Ratio | Maximum primary ms | Peak shared-resident MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda | 204.02 | 244.14 | 256.03 | 1.255x | 256.56 | 5749.39 |
| udd5 | 170.81 | 391.88 | 401.76 | 2.352x | 448.81 | 5678.96 |
| oem | 102.96 | 231.74 | 242.78 | 2.358x | 243.13 | 5713.26 |
| vdd | 127.47 | 323.23 | 333.62 | 2.617x | 334.06 | 5791.32 |
| potsdam | 101.75 | 229.47 | 240.85 | 2.367x | 241.05 | 5693.86 |
| vaihingen | 100.15 | 228.11 | 239.28 | 2.389x | 239.67 | 5685.42 |
| landcoverai | 94.39 | 218.01 | 228.09 | 2.416x | 229.19 | 5685.42 |
| flair1 | 102.98 | 228.66 | 239.54 | 2.326x | 240.23 | 5748.54 |

Three fixed complete images/domain and three warmed synchronized rotated singleton repeats/image; timing workers run serially after all GPUs are idle. Includes all views, reconstruction, output restoration and argmax; excludes initialization/text/image loading/masks. This is not full-domain throughput. Both backbones are resident; peak allocation is not standalone deployment memory.

## Foreground And Non-Residual Scores

| Domain/protocol | Metric | VIP20 | Official/distilled VIP | Geometry | Coupled | Patch-only2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| loveda/D | foreground_mean_iou_percent | 40.1802 | 39.2398 | 45.5343 | 44.5342 | 44.2269 |
| udd5/udd5 | non_residual_mean_iou_percent | 44.1680 | 48.6691 | 51.2339 | 50.3444 | 50.4523 |
| landcoverai/landcoverai | foreground_mean_iou_percent | 50.1322 | 44.7117 | 51.4953 | 58.9920 | 60.5460 |
| landcoverai/landcoverai | non_residual_mean_iou_percent | not reported | not reported | 51.4953 | 58.9920 | 60.5460 |

## Class Outcomes

| Domain/protocol | Class | Arm | IoU | Precision | Recall | Predicted area % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| loveda/P | building | VIP20 | 81.4273 | 88.1551 | 91.4307 | 11.6955 |
| loveda/P | road | VIP20 | 63.5157 | 80.5282 | 75.0405 | 6.8102 |
| loveda/P | water | VIP20 | 65.2910 | 92.9336 | 68.7018 | 13.5469 |
| loveda/P | barren | VIP20 | 34.6101 | 45.7950 | 58.6274 | 8.7440 |
| loveda/P | tree | VIP20 | 21.2621 | 82.9562 | 22.2333 | 3.0914 |
| loveda/P | farm | VIP20 | 71.7595 | 75.0802 | 94.1943 | 56.1120 |
| loveda/P | building | VIP_OfficialOrDistilled | 79.3618 | 85.2657 | 91.9754 | 12.1638 |
| loveda/P | road | VIP_OfficialOrDistilled | 61.2826 | 85.9698 | 68.0927 | 5.7886 |
| loveda/P | water | VIP_OfficialOrDistilled | 62.3187 | 93.0426 | 65.3646 | 12.8738 |
| loveda/P | barren | VIP_OfficialOrDistilled | 33.9002 | 43.9968 | 59.6324 | 9.2574 |
| loveda/P | tree | VIP_OfficialOrDistilled | 21.6331 | 75.8160 | 23.2365 | 3.5351 |
| loveda/P | farm | VIP_OfficialOrDistilled | 71.6600 | 74.8607 | 94.3696 | 56.3813 |
| loveda/P | building | OriginalCoupled | 88.4549 | 92.2013 | 95.6081 | 11.6931 |
| loveda/P | road | OriginalCoupled | 67.8859 | 79.7586 | 82.0157 | 7.5151 |
| loveda/P | water | OriginalCoupled | 70.6875 | 94.4226 | 73.7676 | 14.3164 |
| loveda/P | barren | OriginalCoupled | 40.9590 | 60.9859 | 55.5018 | 6.2159 |
| loveda/P | tree | OriginalCoupled | 33.1641 | 90.8680 | 34.3076 | 4.3549 |
| loveda/P | farm | OriginalCoupled | 73.9385 | 76.5167 | 95.6416 | 55.9046 |
| loveda/P | building | PatchOnly2 | 88.6825 | 92.9903 | 95.0356 | 11.5245 |
| loveda/P | road | PatchOnly2 | 68.3298 | 82.7305 | 79.6974 | 7.0403 |
| loveda/P | water | PatchOnly2 | 69.1266 | 95.3075 | 71.5622 | 13.7594 |
| loveda/P | barren | PatchOnly2 | 39.8354 | 62.1183 | 52.6177 | 5.7855 |
| loveda/P | tree | PatchOnly2 | 30.4370 | 91.5398 | 31.3180 | 3.9462 |
| loveda/P | farm | PatchOnly2 | 72.4400 | 74.4344 | 96.4330 | 57.9440 |
| loveda/D | background | VIP20 | 11.9386 | 56.5493 | 13.1443 | 8.3911 |
| loveda/D | building | VIP20 | 47.6742 | 50.9468 | 88.1258 | 12.4641 |
| loveda/D | road | VIP20 | 50.1601 | 61.2171 | 73.5248 | 5.6089 |
| loveda/D | water | VIP20 | 55.8824 | 77.5494 | 66.6679 | 10.0666 |
| loveda/D | barren | VIP20 | 22.9091 | 30.0813 | 49.0015 | 7.1096 |
| loveda/D | tree | VIP20 | 16.4150 | 48.9324 | 19.8084 | 2.9837 |
| loveda/D | farm | VIP20 | 48.0404 | 49.8264 | 93.0568 | 53.3761 |
| loveda/D | background | VIP_OfficialOrDistilled | 11.9665 | 51.1257 | 13.5122 | 9.5410 |
| loveda/D | building | VIP_OfficialOrDistilled | 46.5717 | 49.5685 | 88.5098 | 12.8665 |
| loveda/D | road | VIP_OfficialOrDistilled | 51.7150 | 69.3253 | 67.0601 | 4.5174 |
| loveda/D | water | VIP_OfficialOrDistilled | 49.7909 | 76.3158 | 58.8909 | 9.0361 |
| loveda/D | barren | VIP_OfficialOrDistilled | 21.9642 | 29.0677 | 47.3347 | 7.1072 |
| loveda/D | tree | VIP_OfficialOrDistilled | 17.7414 | 50.3753 | 21.4988 | 3.1455 |
| loveda/D | farm | VIP_OfficialOrDistilled | 47.6557 | 49.4243 | 93.0154 | 53.7863 |
| loveda/D | background | OriginalCoupled | 12.9418 | 61.7101 | 14.0718 | 8.2319 |
| loveda/D | building | OriginalCoupled | 51.6895 | 53.2756 | 94.5540 | 12.7887 |
| loveda/D | road | OriginalCoupled | 51.2726 | 58.2574 | 81.0479 | 6.4969 |
| loveda/D | water | OriginalCoupled | 59.2759 | 78.4141 | 70.8344 | 10.5778 |
| loveda/D | barren | OriginalCoupled | 29.7836 | 46.2080 | 45.5908 | 4.3062 |
| loveda/D | tree | OriginalCoupled | 26.4831 | 67.0672 | 30.4419 | 3.3455 |
| loveda/D | farm | OriginalCoupled | 48.7003 | 50.0032 | 94.9214 | 54.2530 |
| loveda/D | background | PatchOnly2 | 6.4859 | 64.8102 | 6.7226 | 3.7446 |
| loveda/D | building | PatchOnly2 | 51.4892 | 53.0442 | 94.6133 | 12.8525 |
| loveda/D | road | PatchOnly2 | 52.4245 | 60.7470 | 79.2812 | 6.0948 |
| loveda/D | water | PatchOnly2 | 60.5242 | 81.3538 | 70.2724 | 10.1147 |
| loveda/D | barren | PatchOnly2 | 28.8864 | 42.2875 | 47.6855 | 4.9216 |
| loveda/D | tree | PatchOnly2 | 26.2322 | 66.5281 | 30.2207 | 3.3481 |
| loveda/D | farm | PatchOnly2 | 45.8048 | 46.6524 | 96.1848 | 58.9237 |
| udd5/udd5 | vegetation | VIP20 | 64.3148 | 95.3964 | 66.3749 | 20.6082 |
| udd5/udd5 | building | VIP20 | 81.1240 | 85.2184 | 94.4087 | 43.4924 |
| udd5/udd5 | road | VIP20 | 19.3962 | 56.3084 | 22.8326 | 5.4377 |
| udd5/udd5 | vehicle | VIP20 | 11.8369 | 12.8662 | 59.6718 | 3.7126 |
| udd5/udd5 | other | VIP20 | 30.4958 | 38.1441 | 60.3317 | 26.7491 |
| udd5/udd5 | vegetation | VIP_OfficialOrDistilled | 67.0805 | 93.5741 | 70.3198 | 22.2583 |
| udd5/udd5 | building | VIP_OfficialOrDistilled | 79.9297 | 82.0560 | 96.8599 | 46.3413 |
| udd5/udd5 | road | VIP_OfficialOrDistilled | 32.1817 | 50.8014 | 46.7529 | 12.3415 |
| udd5/udd5 | vehicle | VIP_OfficialOrDistilled | 15.4845 | 17.3658 | 58.8362 | 2.7121 |
| udd5/udd5 | other | VIP_OfficialOrDistilled | 30.1803 | 47.1682 | 45.5924 | 16.3469 |
| udd5/udd5 | vegetation | OriginalCoupled | 66.3933 | 96.7293 | 67.9181 | 20.7968 |
| udd5/udd5 | building | OriginalCoupled | 80.9728 | 81.9480 | 98.5516 | 47.2128 |
| udd5/udd5 | road | OriginalCoupled | 36.8085 | 65.8275 | 45.5034 | 9.2698 |
| udd5/udd5 | vehicle | OriginalCoupled | 17.2031 | 17.8933 | 81.6859 | 3.6544 |
| udd5/udd5 | other | OriginalCoupled | 33.6700 | 47.5315 | 53.5866 | 19.0662 |
| udd5/udd5 | vegetation | PatchOnly2 | 66.2953 | 96.5410 | 67.9084 | 20.8344 |
| udd5/udd5 | building | PatchOnly2 | 81.2771 | 82.2868 | 98.5127 | 46.9999 |
| udd5/udd5 | road | PatchOnly2 | 37.2416 | 65.9709 | 46.0967 | 9.3702 |
| udd5/udd5 | vehicle | PatchOnly2 | 16.9950 | 17.8319 | 78.3585 | 3.5176 |
| udd5/udd5 | other | PatchOnly2 | 33.5843 | 47.1961 | 53.7992 | 19.2779 |
| oem/oem | bareland | VIP20 | 6.9087 | 7.2452 | 59.7934 | 10.6509 |
| oem/oem | rangeland | VIP20 | 9.0957 | 58.0302 | 9.7361 | 3.5324 |
| oem/oem | developed space | VIP20 | 22.7255 | 28.1419 | 54.1440 | 38.1122 |
| oem/oem | road | VIP20 | 30.3281 | 44.6514 | 48.5979 | 7.6874 |
| oem/oem | tree | VIP20 | 31.4655 | 84.4423 | 33.4019 | 7.4322 |
| oem/oem | water | VIP20 | 61.2792 | 75.3063 | 76.6892 | 2.4038 |
| oem/oem | agriculture land | VIP20 | 61.9014 | 68.1685 | 87.0687 | 15.1049 |
| oem/oem | building | VIP20 | 41.9472 | 64.4559 | 54.5703 | 15.0762 |
| oem/oem | bareland | VIP_OfficialOrDistilled | 6.6615 | 6.9775 | 59.5262 | 11.0101 |
| oem/oem | rangeland | VIP_OfficialOrDistilled | 12.3559 | 56.4549 | 13.6575 | 5.0934 |
| oem/oem | developed space | VIP_OfficialOrDistilled | 22.1125 | 27.3809 | 53.4716 | 38.6849 |
| oem/oem | road | VIP_OfficialOrDistilled | 29.2458 | 46.0591 | 44.4806 | 6.8211 |
| oem/oem | tree | VIP_OfficialOrDistilled | 33.1620 | 83.2263 | 35.5372 | 8.0228 |
| oem/oem | water | VIP_OfficialOrDistilled | 64.2911 | 80.0928 | 76.5185 | 2.2552 |
| oem/oem | agriculture land | VIP_OfficialOrDistilled | 68.4754 | 83.1128 | 79.5422 | 11.3180 |
| oem/oem | building | VIP_OfficialOrDistilled | 44.0064 | 62.9601 | 59.3792 | 16.7945 |
| oem/oem | bareland | OriginalCoupled | 10.0142 | 10.2866 | 79.0896 | 9.9228 |
| oem/oem | rangeland | OriginalCoupled | 17.6999 | 66.9109 | 19.3978 | 6.1037 |
| oem/oem | developed space | OriginalCoupled | 28.0239 | 32.7422 | 66.0406 | 39.9548 |
| oem/oem | road | OriginalCoupled | 33.4645 | 50.0977 | 50.1971 | 7.0772 |
| oem/oem | tree | OriginalCoupled | 36.0697 | 89.6185 | 37.6425 | 7.8920 |
| oem/oem | water | OriginalCoupled | 67.8635 | 77.7888 | 84.1741 | 2.5543 |
| oem/oem | agriculture land | OriginalCoupled | 61.5442 | 65.5575 | 90.9529 | 16.4072 |
| oem/oem | building | OriginalCoupled | 43.0029 | 83.1526 | 47.1072 | 10.0881 |
| oem/oem | bareland | PatchOnly2 | 9.9823 | 10.2547 | 78.9879 | 9.9408 |
| oem/oem | rangeland | PatchOnly2 | 17.7049 | 68.4987 | 19.2742 | 5.9242 |
| oem/oem | developed space | PatchOnly2 | 27.9752 | 33.3700 | 63.3759 | 37.6213 |
| oem/oem | road | PatchOnly2 | 33.0407 | 50.3852 | 48.9750 | 6.8655 |
| oem/oem | tree | PatchOnly2 | 36.4308 | 88.9660 | 38.1548 | 8.0581 |
| oem/oem | water | PatchOnly2 | 67.9548 | 78.9922 | 82.9449 | 2.4786 |
| oem/oem | agriculture land | PatchOnly2 | 57.4689 | 60.0952 | 92.9327 | 18.2881 |
| oem/oem | building | PatchOnly2 | 44.1626 | 81.0347 | 49.2533 | 10.8234 |
| vdd/vdd | other | VIP20 | 36.5655 | 42.4665 | 72.4628 | 32.0986 |
| vdd/vdd | wall | VIP20 | 29.0582 | 87.4659 | 30.3208 | 1.1216 |
| vdd/vdd | road | VIP20 | 45.0432 | 49.4124 | 83.5904 | 9.2860 |
| vdd/vdd | vegetation | VIP20 | 50.2515 | 97.5524 | 50.8932 | 18.6325 |
| vdd/vdd | vehicle | VIP20 | 29.6526 | 39.6519 | 54.0414 | 0.7096 |
| vdd/vdd | roof | VIP20 | 85.5153 | 88.6284 | 96.0545 | 22.8049 |
| vdd/vdd | water | VIP20 | 84.2928 | 90.9998 | 91.9593 | 15.3470 |
| vdd/vdd | other | VIP_OfficialOrDistilled | 35.5829 | 43.7502 | 65.5895 | 28.2015 |
| vdd/vdd | wall | VIP_OfficialOrDistilled | 40.1862 | 56.2839 | 58.4212 | 3.3584 |
| vdd/vdd | road | VIP_OfficialOrDistilled | 40.5200 | 42.4047 | 90.1152 | 11.6652 |
| vdd/vdd | vegetation | VIP_OfficialOrDistilled | 52.5014 | 96.4468 | 53.5369 | 19.8250 |
| vdd/vdd | vehicle | VIP_OfficialOrDistilled | 25.2671 | 32.1751 | 54.0622 | 0.8748 |
| vdd/vdd | roof | VIP_OfficialOrDistilled | 81.6539 | 89.8302 | 89.9710 | 21.0748 |
| vdd/vdd | water | VIP_OfficialOrDistilled | 88.7412 | 94.6192 | 93.4576 | 15.0004 |
| vdd/vdd | other | OriginalCoupled | 34.6165 | 52.1847 | 50.6965 | 18.2748 |
| vdd/vdd | wall | OriginalCoupled | 44.4124 | 75.9742 | 51.6693 | 2.2004 |
| vdd/vdd | road | OriginalCoupled | 41.3459 | 42.5370 | 93.6568 | 12.0859 |
| vdd/vdd | vegetation | OriginalCoupled | 65.0501 | 96.7735 | 66.4922 | 24.5393 |
| vdd/vdd | vehicle | OriginalCoupled | 24.1562 | 26.3075 | 74.7086 | 1.4785 |
| vdd/vdd | roof | OriginalCoupled | 82.5973 | 83.2883 | 99.0055 | 25.0126 |
| vdd/vdd | water | OriginalCoupled | 84.4846 | 88.1803 | 95.2737 | 16.4085 |
| vdd/vdd | other | PatchOnly2 | 32.7096 | 53.4974 | 45.7046 | 16.0711 |
| vdd/vdd | wall | PatchOnly2 | 43.4422 | 71.6955 | 52.4351 | 2.3663 |
| vdd/vdd | road | PatchOnly2 | 40.6842 | 41.6059 | 94.8360 | 12.5120 |
| vdd/vdd | vegetation | PatchOnly2 | 68.1015 | 96.4819 | 69.8357 | 25.8512 |
| vdd/vdd | vehicle | PatchOnly2 | 22.7793 | 24.9941 | 71.9936 | 1.4996 |
| vdd/vdd | roof | PatchOnly2 | 81.4826 | 82.1567 | 99.0030 | 25.3564 |
| vdd/vdd | water | PatchOnly2 | 85.0207 | 88.6520 | 95.4036 | 16.3434 |
| potsdam/potsdam | impervious surface | VIP20 | 61.2488 | 74.7767 | 77.1981 | 32.4902 |
| potsdam/potsdam | building | VIP20 | 77.2218 | 79.8084 | 95.9722 | 28.7639 |
| potsdam/potsdam | low vegetation | VIP20 | 28.5736 | 92.9464 | 29.2069 | 6.6017 |
| potsdam/potsdam | tree | VIP20 | 50.9752 | 84.8504 | 56.0791 | 11.2629 |
| potsdam/potsdam | car | VIP20 | 30.4756 | 30.7704 | 96.9522 | 6.1911 |
| potsdam/potsdam | clutter | VIP20 | 5.3220 | 6.6334 | 21.2098 | 14.6902 |
| potsdam/potsdam | impervious surface | VIP_OfficialOrDistilled | 60.7547 | 65.4790 | 89.3850 | 42.9611 |
| potsdam/potsdam | building | VIP_OfficialOrDistilled | 71.0553 | 84.3751 | 81.8216 | 23.1955 |
| potsdam/potsdam | low vegetation | VIP_OfficialOrDistilled | 52.7964 | 77.0456 | 62.6513 | 17.0838 |
| potsdam/potsdam | tree | VIP_OfficialOrDistilled | 56.8492 | 79.8574 | 66.3655 | 14.1622 |
| potsdam/potsdam | car | VIP_OfficialOrDistilled | 16.8657 | 59.1739 | 19.0866 | 0.6338 |
| potsdam/potsdam | clutter | VIP_OfficialOrDistilled | 11.0760 | 33.3030 | 14.2333 | 1.9636 |
| potsdam/potsdam | impervious surface | OriginalCoupled | 61.0891 | 72.6216 | 79.3681 | 34.3948 |
| potsdam/potsdam | building | OriginalCoupled | 78.6001 | 81.0594 | 96.2834 | 28.4118 |
| potsdam/potsdam | low vegetation | OriginalCoupled | 33.6290 | 93.0373 | 34.4972 | 7.7898 |
| potsdam/potsdam | tree | OriginalCoupled | 54.4128 | 83.4512 | 60.9944 | 12.4555 |
| potsdam/potsdam | car | OriginalCoupled | 26.6264 | 26.6925 | 99.0780 | 7.2934 |
| potsdam/potsdam | clutter | OriginalCoupled | 6.2234 | 8.6467 | 18.1704 | 9.6547 |
| potsdam/potsdam | impervious surface | PatchOnly2 | 62.7987 | 73.9621 | 80.6228 | 34.3053 |
| potsdam/potsdam | building | PatchOnly2 | 79.0175 | 81.0699 | 96.8956 | 28.5888 |
| potsdam/potsdam | low vegetation | PatchOnly2 | 38.5018 | 92.0555 | 39.8250 | 9.0888 |
| potsdam/potsdam | tree | PatchOnly2 | 58.5794 | 81.9854 | 67.2335 | 13.9750 |
| potsdam/potsdam | car | PatchOnly2 | 30.1233 | 30.2159 | 98.9927 | 6.4374 |
| potsdam/potsdam | clutter | PatchOnly2 | 6.5825 | 9.9072 | 16.3986 | 7.6047 |
| vaihingen/vaihingen | impervious surface | VIP20 | 56.9979 | 74.4361 | 70.8709 | 27.0911 |
| vaihingen/vaihingen | building | VIP20 | 64.9474 | 65.8335 | 97.9697 | 39.2030 |
| vaihingen/vaihingen | low vegetation | VIP20 | 30.2954 | 82.1686 | 32.4273 | 8.4567 |
| vaihingen/vaihingen | tree | VIP20 | 62.0184 | 79.1843 | 74.0989 | 21.0254 |
| vaihingen/vaihingen | car | VIP20 | 22.5506 | 24.0881 | 77.9398 | 4.2239 |
| vaihingen/vaihingen | impervious surface | VIP_OfficialOrDistilled | 45.4945 | 49.6039 | 84.5956 | 48.5259 |
| vaihingen/vaihingen | building | VIP_OfficialOrDistilled | 60.4878 | 76.7511 | 74.0568 | 25.4188 |
| vaihingen/vaihingen | low vegetation | VIP_OfficialOrDistilled | 22.1971 | 94.6077 | 22.4816 | 5.0921 |
| vaihingen/vaihingen | tree | VIP_OfficialOrDistilled | 56.9623 | 77.4656 | 68.2756 | 19.8029 |
| vaihingen/vaihingen | car | VIP_OfficialOrDistilled | 24.3892 | 46.8607 | 33.7133 | 0.9392 |
| vaihingen/vaihingen | impervious surface | OriginalCoupled | 60.8181 | 78.8003 | 72.7158 | 26.2569 |
| vaihingen/vaihingen | building | OriginalCoupled | 67.4261 | 68.0474 | 98.6640 | 38.1963 |
| vaihingen/vaihingen | low vegetation | OriginalCoupled | 32.3657 | 88.6454 | 33.7655 | 8.1623 |
| vaihingen/vaihingen | tree | OriginalCoupled | 68.5013 | 80.7096 | 81.9124 | 22.8032 |
| vaihingen/vaihingen | car | OriginalCoupled | 24.8634 | 25.5865 | 89.7936 | 4.5813 |
| vaihingen/vaihingen | impervious surface | PatchOnly2 | 61.5534 | 78.4214 | 74.1046 | 26.8876 |
| vaihingen/vaihingen | building | PatchOnly2 | 68.7479 | 69.3447 | 98.7637 | 37.5196 |
| vaihingen/vaihingen | low vegetation | PatchOnly2 | 32.1553 | 87.2197 | 33.7453 | 8.2908 |
| vaihingen/vaihingen | tree | PatchOnly2 | 67.7382 | 79.5645 | 82.0055 | 23.1577 |
| vaihingen/vaihingen | car | PatchOnly2 | 26.9462 | 27.9126 | 88.6136 | 4.1443 |
| landcoverai/landcoverai | background | VIP20 | 74.1810 | 77.5344 | 94.4907 | 73.1495 |
| landcoverai/landcoverai | building | VIP20 | 44.5106 | 45.8719 | 93.7495 | 1.8703 |
| landcoverai/landcoverai | woodland | VIP20 | 52.0468 | 94.5550 | 53.6549 | 17.9386 |
| landcoverai/landcoverai | water | VIP20 | 64.8284 | 88.0514 | 71.0816 | 4.7813 |
| landcoverai/landcoverai | road | VIP20 | 39.1431 | 47.1286 | 69.7898 | 2.2604 |
| landcoverai/landcoverai | background | VIP_OfficialOrDistilled | 72.6430 | 74.6995 | 96.3486 | 77.4185 |
| landcoverai/landcoverai | building | VIP_OfficialOrDistilled | 50.8202 | 53.8116 | 90.1399 | 1.5329 |
| landcoverai/landcoverai | woodland | VIP_OfficialOrDistilled | 49.3624 | 94.9030 | 50.7067 | 16.8907 |
| landcoverai/landcoverai | water | VIP_OfficialOrDistilled | 38.9371 | 98.9391 | 39.1003 | 2.3406 |
| landcoverai/landcoverai | road | VIP_OfficialOrDistilled | 39.7273 | 52.3144 | 62.2803 | 1.8172 |
| landcoverai/landcoverai | background | OriginalCoupled | 82.0813 | 86.8922 | 93.6809 | 64.7124 |
| landcoverai/landcoverai | building | OriginalCoupled | 47.7667 | 48.5865 | 96.5881 | 1.8192 |
| landcoverai/landcoverai | woodland | OriginalCoupled | 73.0349 | 95.0145 | 75.9453 | 25.2682 |
| landcoverai/landcoverai | water | OriginalCoupled | 74.8843 | 89.1169 | 82.4217 | 5.4777 |
| landcoverai/landcoverai | road | OriginalCoupled | 40.2822 | 44.8154 | 79.9290 | 2.7224 |
| landcoverai/landcoverai | background | PatchOnly2 | 83.3502 | 88.4253 | 93.5577 | 63.5068 |
| landcoverai/landcoverai | building | PatchOnly2 | 49.1710 | 50.1203 | 96.2912 | 1.7581 |
| landcoverai/landcoverai | woodland | PatchOnly2 | 75.8544 | 94.0842 | 79.6535 | 26.7640 |
| landcoverai/landcoverai | water | PatchOnly2 | 75.9163 | 90.4639 | 82.5201 | 5.4026 |
| landcoverai/landcoverai | road | PatchOnly2 | 41.2424 | 46.5536 | 78.3316 | 2.5684 |
| flair1/flair1 | building | VIP20 | 60.2824 | 64.1549 | 90.8982 | 12.2232 |
| flair1/flair1 | pervious surface | VIP20 | 32.1094 | 57.8373 | 41.9224 | 5.3386 |
| flair1/flair1 | impervious surface | VIP20 | 56.6589 | 70.6219 | 74.1315 | 15.7706 |
| flair1/flair1 | bare soil | VIP20 | 23.1537 | 24.1644 | 84.6983 | 15.3219 |
| flair1/flair1 | water | VIP20 | 74.1419 | 88.1892 | 82.3154 | 5.6013 |
| flair1/flair1 | coniferous | VIP20 | 14.4692 | 44.0361 | 17.7294 | 0.9655 |
| flair1/flair1 | deciduous | VIP20 | 48.7353 | 78.2220 | 56.3860 | 10.0552 |
| flair1/flair1 | brushwood | VIP20 | 11.1440 | 32.7620 | 14.4485 | 3.0547 |
| flair1/flair1 | vineyard | VIP20 | 57.6837 | 75.3965 | 71.0595 | 3.6571 |
| flair1/flair1 | herbaceous vegetation | VIP20 | 18.8786 | 76.8164 | 20.0193 | 5.7937 |
| flair1/flair1 | agricultural land | VIP20 | 16.1240 | 35.2572 | 22.9062 | 4.5260 |
| flair1/flair1 | plowed land | VIP20 | 10.1912 | 10.4300 | 81.6572 | 17.6922 |
| flair1/flair1 | building | VIP_OfficialOrDistilled | 58.6727 | 62.1492 | 91.2960 | 12.6729 |
| flair1/flair1 | pervious surface | VIP_OfficialOrDistilled | 7.0286 | 47.1442 | 7.6298 | 1.1920 |
| flair1/flair1 | impervious surface | VIP_OfficialOrDistilled | 45.8843 | 51.0920 | 81.8237 | 24.0608 |
| flair1/flair1 | bare soil | VIP_OfficialOrDistilled | 24.4057 | 41.6864 | 37.0570 | 3.8859 |
| flair1/flair1 | water | VIP_OfficialOrDistilled | 67.1946 | 89.8708 | 72.7004 | 4.8545 |
| flair1/flair1 | coniferous | VIP_OfficialOrDistilled | 14.9779 | 43.0053 | 18.6874 | 1.0421 |
| flair1/flair1 | deciduous | VIP_OfficialOrDistilled | 53.6978 | 73.5507 | 66.5483 | 12.6211 |
| flair1/flair1 | brushwood | VIP_OfficialOrDistilled | 22.1228 | 43.2520 | 31.1703 | 4.9917 |
| flair1/flair1 | vineyard | VIP_OfficialOrDistilled | 59.3478 | 85.3796 | 66.0614 | 3.0023 |
| flair1/flair1 | herbaceous vegetation | VIP_OfficialOrDistilled | 41.9934 | 67.7829 | 52.4651 | 17.2073 |
| flair1/flair1 | agricultural land | VIP_OfficialOrDistilled | 22.6933 | 35.3367 | 38.8097 | 7.6511 |
| flair1/flair1 | plowed land | VIP_OfficialOrDistilled | 21.2695 | 23.3520 | 70.4580 | 6.8183 |
| flair1/flair1 | building | OriginalCoupled | 62.7485 | 64.7300 | 95.3483 | 12.7077 |
| flair1/flair1 | pervious surface | OriginalCoupled | 33.6803 | 69.8821 | 39.3994 | 4.1525 |
| flair1/flair1 | impervious surface | OriginalCoupled | 60.4068 | 71.1734 | 79.9730 | 16.8815 |
| flair1/flair1 | bare soil | OriginalCoupled | 35.3724 | 37.0601 | 88.5941 | 10.4499 |
| flair1/flair1 | water | OriginalCoupled | 80.2617 | 88.2167 | 89.8996 | 6.1155 |
| flair1/flair1 | coniferous | OriginalCoupled | 15.8935 | 43.9264 | 19.9388 | 1.0885 |
| flair1/flair1 | deciduous | OriginalCoupled | 56.3093 | 77.3074 | 67.4597 | 12.1722 |
| flair1/flair1 | brushwood | OriginalCoupled | 26.5738 | 42.2522 | 41.7300 | 6.8409 |
| flair1/flair1 | vineyard | OriginalCoupled | 62.9131 | 80.2304 | 74.4555 | 3.6010 |
| flair1/flair1 | herbaceous vegetation | OriginalCoupled | 31.8945 | 79.4587 | 34.7606 | 9.7254 |
| flair1/flair1 | agricultural land | OriginalCoupled | 23.8671 | 37.6806 | 39.4326 | 7.2903 |
| flair1/flair1 | plowed land | OriginalCoupled | 20.1241 | 20.9712 | 83.2838 | 8.9745 |
| flair1/flair1 | building | PatchOnly2 | 64.8588 | 67.0234 | 95.2569 | 12.2611 |
| flair1/flair1 | pervious surface | PatchOnly2 | 33.1251 | 69.9570 | 38.6189 | 4.0659 |
| flair1/flair1 | impervious surface | PatchOnly2 | 61.9267 | 73.2792 | 79.9892 | 16.3997 |
| flair1/flair1 | bare soil | PatchOnly2 | 36.4787 | 38.4092 | 87.8903 | 10.0028 |
| flair1/flair1 | water | PatchOnly2 | 79.1873 | 88.7142 | 88.0582 | 5.9566 |
| flair1/flair1 | coniferous | PatchOnly2 | 15.5239 | 39.6908 | 20.3160 | 1.2275 |
| flair1/flair1 | deciduous | PatchOnly2 | 57.2671 | 74.0452 | 71.6498 | 13.4978 |
| flair1/flair1 | brushwood | PatchOnly2 | 25.9535 | 47.6032 | 36.3326 | 5.2866 |
| flair1/flair1 | vineyard | PatchOnly2 | 61.4830 | 74.8920 | 77.4467 | 4.0127 |
| flair1/flair1 | herbaceous vegetation | PatchOnly2 | 32.6420 | 78.3921 | 35.8693 | 10.1722 |
| flair1/flair1 | agricultural land | PatchOnly2 | 22.2483 | 36.0843 | 36.7183 | 7.0888 |
| flair1/flair1 | plowed land | PatchOnly2 | 18.5413 | 19.1658 | 85.0517 | 10.0283 |

## Selection Status

Full unique coverage: 20092 images; all-eight accuracy available=True; all completed protocols exceed the stronger measured VIP=True; eight timing panels complete=True. No final-model selection is established by an average or a partial suite. Alias-module efficacy and a defensible CVPR innovation claim are not established by this route/strength configuration.
