# Frozen Semantic-Role Admission: Verified Results

Same64 development top-left512 windows, eight/domain, not full or untouched validation. Frozen Geometry/local20/VIP wide observer/operator and20 context aliases/class. LoveDA D counts once in domain mean; P separately. LandCover.ai substitutes for unlabeled iSAID. Text-only pinned Qwen adds pretrained semantic supervision at vocabulary time. No target-label numerical fitting, routing or post-result seed/coefficient choice. All61 historical numerical/per-image endpoints replay exactly.

## Prospective Decision

```json
{
  "passed": false,
  "checks": {
    "clean_gain": true,
    "wrong_parent_gain": true,
    "wrong_parent_domain_wins": true,
    "clean_paraphrase_safety": true,
    "actual_wrong_parent_damage": true,
    "above_CoherentNativeJoint_Exact": false,
    "above_CoherentNoHoldout_Exact": false,
    "above_ClassAttachmentJoint_Exact": false,
    "above_ClassAttachmentTextOnly_Exact": false,
    "above_SemanticRoleOnly_Exact": true,
    "above_SemanticRoleVisualSupported_Exact": false,
    "above_SemanticRoleHard_Exact": true,
    "above_SemanticRoleMeanLogit": true,
    "above_SemanticRoleAliasShuffle0_Exact": true,
    "above_SemanticRoleAliasShuffle1_Exact": false,
    "above_SemanticRoleAliasShuffle2_Exact": false
  },
  "wrong_parent_domain_wins": 6,
  "clean_gain_pp": 1.2204536666181696,
  "wrong_parent_gain_pp": 0.6007924160416138,
  "worst_clean_paraphrase_delta_pp": -0.01821214871169019,
  "earlier_failed_gates_unchanged": true
}
```

FAIL: no selector promotion, full rollout or winning-control promotion.

## clean

| Dataset/protocol | NoAdmission_Exact | CoherentNativeJoint_Exact | CoherentNoHoldout_Exact | ClassAttachmentJoint_Exact | SemanticRoleJoint_Exact | SemanticRoleOnly_Exact | SemanticRoleVisualSupported_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 53.746954 | 55.927447 | 55.833087 | 55.835045 | 55.833087 | 53.746954 | 55.833087 |
| potsdam/potsdam | 38.920258 | 39.525433 | 39.537067 | 39.599082 | 39.537067 | 38.920258 | 39.537067 |
| udd5/udd5 | 28.175811 | 30.794864 | 30.674206 | 30.645131 | 30.674206 | 28.175811 | 30.674206 |
| oem/oem | 39.023152 | 39.353115 | 39.298606 | 39.310936 | 39.298606 | 39.023152 | 39.298606 |
| loveda/P | 50.706552 | 53.281856 | 53.513897 | 53.496902 | 53.513897 | 50.706552 | 53.513897 |
| loveda/D | 30.736030 | 34.135199 | 33.944372 | 33.829905 | 33.944372 | 30.736030 | 33.944372 |
| vaihingen/vaihingen | 51.826962 | 52.865911 | 52.851821 | 52.846145 | 52.851821 | 51.826962 | 52.851821 |
| landcoverai/landcoverai | 66.906020 | 66.949114 | 66.907578 | 66.965036 | 66.907578 | 66.906020 | 66.907578 |
| flair1/flair1 | 33.608404 | 33.737800 | 33.687914 | 33.628603 | 33.660484 | 33.555369 | 33.683102 |
| Domain mean | 42.867949 | 44.161110 | 44.091831 | 44.082486 | 44.088403 | 42.861320 | 44.091230 |

| All prospective controls | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 44.161110 |
| CoherentNoHoldout_Exact | 44.091831 |
| ClassAttachmentJoint_Exact | 44.082486 |
| ClassAttachmentTextOnly_Exact | 43.521285 |
| SemanticRoleOnly_Exact | 42.861320 |
| SemanticRoleVisualSupported_Exact | 44.091230 |
| SemanticRoleHard_Exact | 44.087081 |
| SemanticRoleMeanLogit | 43.816199 |
| SemanticRoleAliasShuffle0_Exact | 44.065516 |
| SemanticRoleAliasShuffle1_Exact | 44.090729 |
| SemanticRoleAliasShuffle2_Exact | 44.083677 |
| SemanticRoleJoint_Exact | 44.088403 |

| Domain/protocol | Class | Original IoU | Primary IoU | Delta pp | Precision % | Recall % | Area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 59.7766 | 61.3087 | 1.5321 | 81.3606 | 71.3270 | 44.7593 |
| vdd/vdd | wall | 59.6816 | 64.5037 | 4.8221 | 71.1222 | 87.3922 | 9.7751 |
| vdd/vdd | road | 23.2250 | 24.2656 | 1.0406 | 24.3888 | 97.9599 | 13.3312 |
| vdd/vdd | vegetation | 44.1426 | 45.8169 | 1.6743 | 95.9666 | 46.7165 | 11.2134 |
| vdd/vdd | vehicle | 50.3084 | 48.9761 | -1.3323 | 48.9761 | 100.0000 | 0.7312 |
| vdd/vdd | roof | 86.2607 | 89.6779 | 3.4172 | 90.0769 | 99.5086 | 6.6563 |
| vdd/vdd | water | 52.8339 | 56.2827 | 3.4488 | 57.9713 | 95.0794 | 13.5335 |
| potsdam/potsdam | impervious surface | 65.8977 | 66.2596 | 0.3619 | 85.7327 | 74.4714 | 35.5072 |
| potsdam/potsdam | building | 71.0529 | 71.9184 | 0.8655 | 76.0656 | 92.9531 | 19.7023 |
| potsdam/potsdam | low vegetation | 14.4124 | 15.8962 | 1.4838 | 78.0148 | 16.6417 | 3.6991 |
| potsdam/potsdam | tree | 55.5064 | 56.0681 | 0.5617 | 93.1635 | 58.4739 | 11.0740 |
| potsdam/potsdam | car | 24.5378 | 24.9387 | 0.4009 | 24.9741 | 99.4348 | 14.2762 |
| potsdam/potsdam | clutter | 2.1143 | 2.1414 | 0.0271 | 2.6866 | 9.5451 | 15.7413 |
| udd5/udd5 | vegetation | 46.2799 | 51.7198 | 5.4399 | 92.8015 | 53.8814 | 1.7925 |
| udd5/udd5 | building | 83.4871 | 83.9210 | 0.4339 | 84.0247 | 99.8532 | 85.4746 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4610 |
| udd5/udd5 | vehicle | 5.3645 | 7.2681 | 1.9036 | 7.2682 | 99.9773 | 5.7773 |
| udd5/udd5 | other | 5.7476 | 10.4621 | 4.7145 | 39.1012 | 12.4987 | 6.4947 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9330 |
| oem/oem | rangeland | 49.5261 | 49.7962 | 0.2701 | 70.4787 | 62.9202 | 12.9862 |
| oem/oem | developed space | 23.7669 | 22.1134 | -1.6535 | 38.2379 | 34.4003 | 17.6308 |
| oem/oem | road | 54.5529 | 55.1110 | 0.5581 | 69.2338 | 72.9853 | 6.6279 |
| oem/oem | tree | 55.6605 | 56.0455 | 0.3850 | 91.7205 | 59.0320 | 17.0032 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| oem/oem | agriculture land | 81.6500 | 81.3092 | -0.3408 | 81.6604 | 99.4738 | 25.8412 |
| oem/oem | building | 47.0287 | 50.0136 | 2.9849 | 69.5169 | 64.0633 | 10.9776 |
| loveda/P | building | 60.8479 | 76.1773 | 15.3294 | 83.5866 | 89.5765 | 0.0284 |
| loveda/P | road | 53.7553 | 55.8436 | 2.0883 | 56.9027 | 96.7746 | 14.1536 |
| loveda/P | water | 62.4091 | 62.7439 | 0.3348 | 67.9932 | 89.0437 | 19.2701 |
| loveda/P | barren | 4.3215 | 2.3564 | -1.9651 | 91.1659 | 2.3618 | 0.2872 |
| loveda/P | tree | 39.9553 | 41.3957 | 1.4404 | 99.2570 | 41.5244 | 6.8881 |
| loveda/P | farm | 82.9502 | 82.5665 | -0.3837 | 82.8428 | 99.5977 | 59.3725 |
| loveda/D | background | 7.3224 | 14.4463 | 7.1239 | 76.3082 | 15.1247 | 8.8702 |
| loveda/D | building | 31.2821 | 46.3744 | 15.0923 | 49.0196 | 89.5765 | 0.0268 |
| loveda/D | road | 44.1270 | 46.8793 | 2.7523 | 47.6321 | 96.7383 | 9.3379 |
| loveda/D | water | 54.4379 | 54.5619 | 0.1240 | 58.5144 | 88.9838 | 12.3625 |
| loveda/D | barren | 3.2414 | 1.6702 | -1.5712 | 81.0689 | 1.6768 | 0.1267 |
| loveda/D | tree | 35.8674 | 32.7068 | -3.1606 | 97.7244 | 32.9578 | 3.0678 |
| loveda/D | farm | 38.8741 | 40.9717 | 2.0976 | 41.0406 | 99.5917 | 66.2081 |
| vaihingen/vaihingen | impervious surface | 57.3052 | 58.8479 | 1.5427 | 75.0983 | 73.1151 | 26.5953 |
| vaihingen/vaihingen | building | 65.7437 | 66.4389 | 0.6952 | 66.5588 | 99.7296 | 30.9122 |
| vaihingen/vaihingen | low vegetation | 43.7153 | 45.2176 | 1.5023 | 95.9748 | 46.0918 | 14.2000 |
| vaihingen/vaihingen | tree | 67.2612 | 67.4096 | 0.1484 | 78.9471 | 82.1830 | 21.4846 |
| vaihingen/vaihingen | car | 25.1095 | 26.3450 | 1.2355 | 26.5059 | 97.7476 | 6.8079 |
| landcoverai/landcoverai | background | 87.4477 | 87.3541 | -0.0936 | 93.6745 | 92.8298 | 67.1490 |
| landcoverai/landcoverai | building | 44.4672 | 44.7822 | 0.3150 | 45.1296 | 98.3099 | 3.3004 |
| landcoverai/landcoverai | woodland | 78.6026 | 77.9455 | -0.6571 | 94.8215 | 81.4110 | 18.3793 |
| landcoverai/landcoverai | water | 97.6178 | 97.7174 | 0.0996 | 97.7174 | 100.0000 | 8.4645 |
| landcoverai/landcoverai | road | 26.3947 | 26.7388 | 0.3441 | 29.2569 | 75.6491 | 2.7068 |
| flair1/flair1 | building | 56.7755 | 57.5599 | 0.7844 | 58.4651 | 97.3808 | 11.9584 |
| flair1/flair1 | pervious surface | 47.2753 | 46.1263 | -1.1490 | 92.6289 | 47.8839 | 8.8706 |
| flair1/flair1 | impervious surface | 51.9858 | 52.7898 | 0.8040 | 60.5320 | 80.4967 | 21.7857 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.4287 |
| flair1/flair1 | water | 61.4565 | 61.4318 | -0.0247 | 63.2519 | 95.5254 | 6.6994 |
| flair1/flair1 | coniferous | 43.5282 | 43.4286 | -0.0996 | 63.9781 | 57.4847 | 0.5051 |
| flair1/flair1 | deciduous | 59.5237 | 61.3929 | 1.8692 | 78.2019 | 74.0679 | 16.1218 |
| flair1/flair1 | brushwood | 12.3287 | 12.1677 | -0.1610 | 29.1679 | 17.2709 | 2.9123 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0119 |
| flair1/flair1 | herbaceous vegetation | 57.4725 | 56.5675 | -0.9050 | 90.9546 | 59.9395 | 21.1113 |
| flair1/flair1 | agricultural land | 12.9546 | 12.4612 | -0.4934 | 13.3410 | 65.3942 | 1.4946 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1002 |

## wrong_parent

| Dataset/protocol | NoAdmission_Exact | CoherentNativeJoint_Exact | CoherentNoHoldout_Exact | ClassAttachmentJoint_Exact | SemanticRoleJoint_Exact | SemanticRoleOnly_Exact | SemanticRoleVisualSupported_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 47.408779 | 49.661937 | 49.718132 | 49.790934 | 49.628514 | 47.111604 | 49.715715 |
| potsdam/potsdam | 36.070589 | 36.403840 | 36.118895 | 36.242560 | 37.283986 | 36.846160 | 36.253287 |
| udd5/udd5 | 27.616556 | 28.350669 | 28.317171 | 28.318628 | 27.301082 | 26.032349 | 28.290806 |
| oem/oem | 40.053288 | 40.881100 | 40.711109 | 40.756976 | 40.619348 | 39.916217 | 40.702483 |
| loveda/P | 49.989695 | 50.240053 | 50.287711 | 50.065941 | 49.742447 | 49.400226 | 50.265919 |
| loveda/D | 30.333514 | 31.529715 | 31.569316 | 31.472916 | 31.289952 | 29.877249 | 31.556340 |
| vaihingen/vaihingen | 53.060153 | 54.325480 | 54.201623 | 54.356306 | 52.691449 | 50.950979 | 54.205599 |
| landcoverai/landcoverai | 68.123522 | 68.791227 | 68.758172 | 68.803943 | 68.185930 | 67.787935 | 68.724942 |
| flair1/flair1 | 33.246876 | 33.531745 | 33.448926 | 33.418777 | 33.719356 | 33.549687 | 33.460285 |
| Domain mean | 41.989160 | 42.934464 | 42.855418 | 42.895130 | 42.589952 | 41.509023 | 42.863682 |

| All prospective controls | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 42.934464 |
| CoherentNoHoldout_Exact | 42.855418 |
| ClassAttachmentJoint_Exact | 42.895130 |
| ClassAttachmentTextOnly_Exact | 43.528885 |
| SemanticRoleOnly_Exact | 41.509023 |
| SemanticRoleVisualSupported_Exact | 42.863682 |
| SemanticRoleHard_Exact | 42.243275 |
| SemanticRoleMeanLogit | 41.893298 |
| SemanticRoleAliasShuffle0_Exact | 42.537595 |
| SemanticRoleAliasShuffle1_Exact | 42.661983 |
| SemanticRoleAliasShuffle2_Exact | 42.610170 |
| SemanticRoleJoint_Exact | 42.589952 |

| Domain/protocol | Class | Original IoU | Primary IoU | Delta pp | Precision % | Recall % | Area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 56.9300 | 59.5196 | 2.5896 | 80.3117 | 69.6878 | 44.3018 |
| vdd/vdd | wall | 55.1110 | 59.6094 | 4.4984 | 65.7407 | 86.4709 | 10.4638 |
| vdd/vdd | road | 28.2952 | 34.1059 | 5.8107 | 34.6162 | 95.8566 | 9.1908 |
| vdd/vdd | vegetation | 36.9486 | 35.9310 | -1.0176 | 96.7074 | 36.3760 | 8.6645 |
| vdd/vdd | vehicle | 25.1912 | 26.1399 | 0.9487 | 26.1399 | 100.0000 | 1.3700 |
| vdd/vdd | roof | 86.7367 | 89.9004 | 3.1637 | 90.5287 | 99.2339 | 6.6048 |
| vdd/vdd | water | 42.6487 | 42.1934 | -0.4553 | 42.2916 | 99.4528 | 19.4044 |
| potsdam/potsdam | impervious surface | 68.7852 | 69.5641 | 0.7789 | 84.6093 | 79.6420 | 38.4767 |
| potsdam/potsdam | building | 73.8547 | 73.7290 | -0.1257 | 80.7168 | 89.4919 | 17.8757 |
| potsdam/potsdam | low vegetation | 11.7787 | 9.1891 | -2.5896 | 60.9702 | 9.7634 | 2.7769 |
| potsdam/potsdam | tree | 32.2407 | 39.8294 | 7.5887 | 97.5663 | 40.2291 | 7.2749 |
| potsdam/potsdam | car | 26.8975 | 28.3270 | 1.4295 | 28.4133 | 98.9401 | 12.4858 |
| potsdam/potsdam | clutter | 2.8668 | 3.0652 | 0.1984 | 3.5983 | 17.1443 | 21.1101 |
| udd5/udd5 | vegetation | 49.3198 | 47.6951 | -1.6247 | 93.6096 | 49.3003 | 1.6259 |
| udd5/udd5 | building | 86.0240 | 86.1010 | 0.0770 | 86.4936 | 99.4757 | 82.7209 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0772 |
| udd5/udd5 | vehicle | 2.7332 | 2.7067 | -0.0265 | 2.7067 | 99.9773 | 15.5135 |
| udd5/udd5 | other | 0.0058 | 0.0026 | -0.0032 | 0.8391 | 0.0026 | 0.0625 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.7606 |
| oem/oem | rangeland | 49.2840 | 49.3259 | 0.0419 | 70.2806 | 62.3261 | 12.8998 |
| oem/oem | developed space | 31.5090 | 31.1728 | -0.3362 | 51.7058 | 43.9772 | 16.6683 |
| oem/oem | road | 55.9739 | 56.3067 | 0.3328 | 72.2637 | 71.8304 | 6.2495 |
| oem/oem | tree | 54.0503 | 54.2705 | 0.2202 | 93.1321 | 56.5330 | 16.0366 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1011 |
| oem/oem | agriculture land | 81.3052 | 81.1333 | -0.1719 | 81.4159 | 99.5739 | 25.9449 |
| oem/oem | building | 48.3038 | 52.7456 | 4.4418 | 61.3484 | 78.9978 | 15.3391 |
| loveda/P | building | 42.0043 | 41.7073 | -0.2970 | 62.4088 | 55.7003 | 0.0236 |
| loveda/P | road | 54.1953 | 56.3977 | 2.2024 | 57.4485 | 96.8586 | 14.0313 |
| loveda/P | water | 60.8381 | 60.0455 | -0.7926 | 64.3916 | 89.8954 | 20.5426 |
| loveda/P | barren | 3.3653 | 1.9413 | -1.4240 | 99.6006 | 1.9414 | 0.2161 |
| loveda/P | tree | 52.3660 | 51.5851 | -0.7809 | 97.9710 | 52.1422 | 8.7629 |
| loveda/P | farm | 87.1691 | 86.7777 | -0.3914 | 87.1249 | 99.5430 | 56.4234 |
| loveda/D | background | 2.1813 | 4.6510 | 2.4697 | 65.3648 | 4.7685 | 3.2648 |
| loveda/D | building | 32.4926 | 39.0963 | 6.6037 | 49.6259 | 64.8208 | 0.0191 |
| loveda/D | road | 44.1361 | 46.5576 | 2.4215 | 47.2778 | 96.8317 | 9.4170 |
| loveda/D | water | 52.5423 | 51.1507 | -1.3916 | 54.2866 | 89.8525 | 13.4554 |
| loveda/D | barren | 1.9651 | 1.1581 | -0.8070 | 98.4778 | 1.1583 | 0.0721 |
| loveda/D | tree | 39.4804 | 36.9848 | -2.4956 | 75.9691 | 41.8850 | 5.0152 |
| loveda/D | farm | 39.5368 | 39.4312 | -0.1056 | 39.5020 | 99.5475 | 68.7564 |
| vaihingen/vaihingen | impervious surface | 59.1306 | 59.7217 | 0.5911 | 74.8210 | 74.7434 | 27.2883 |
| vaihingen/vaihingen | building | 72.1795 | 71.2157 | -0.9638 | 71.5283 | 99.3900 | 28.6666 |
| vaihingen/vaihingen | low vegetation | 49.0698 | 46.6934 | -2.3764 | 93.9111 | 48.1511 | 15.1605 |
| vaihingen/vaihingen | tree | 66.9292 | 67.4608 | 0.5316 | 84.1646 | 77.2681 | 18.9475 |
| vaihingen/vaihingen | car | 17.9916 | 18.3658 | 0.3742 | 18.3986 | 99.0365 | 9.9371 |
| landcoverai/landcoverai | background | 88.0128 | 87.9905 | -0.0223 | 94.2228 | 93.0083 | 66.8866 |
| landcoverai/landcoverai | building | 50.7657 | 50.7708 | 0.0051 | 51.3452 | 97.8441 | 2.8871 |
| landcoverai/landcoverai | woodland | 79.5889 | 79.3797 | -0.2092 | 94.5037 | 83.2218 | 18.8512 |
| landcoverai/landcoverai | water | 97.5136 | 97.7020 | 0.1884 | 97.7020 | 100.0000 | 8.4659 |
| landcoverai/landcoverai | road | 24.7367 | 25.0867 | 0.3500 | 27.2721 | 75.7903 | 2.9092 |
| flair1/flair1 | building | 52.0630 | 52.2757 | 0.2127 | 52.5389 | 99.0505 | 13.5354 |
| flair1/flair1 | pervious surface | 47.5920 | 45.8481 | -1.7439 | 90.6285 | 48.1299 | 9.1130 |
| flair1/flair1 | impervious surface | 52.0153 | 52.4111 | 0.3958 | 62.2382 | 76.8484 | 20.2282 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.4218 |
| flair1/flair1 | water | 65.6557 | 64.9497 | -0.7060 | 67.6273 | 94.2543 | 6.1826 |
| flair1/flair1 | coniferous | 34.5846 | 39.2422 | 4.6576 | 54.8942 | 57.9175 | 0.5931 |
| flair1/flair1 | deciduous | 53.4475 | 56.3588 | 2.9113 | 80.0724 | 65.5533 | 13.9352 |
| flair1/flair1 | brushwood | 17.9649 | 18.5371 | 0.5722 | 31.1563 | 31.3974 | 4.9565 |
| flair1/flair1 | vineyard | -- | -- | -- | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | herbaceous vegetation | 61.2727 | 61.4402 | 0.1675 | 89.2341 | 66.3591 | 23.8230 |
| flair1/flair1 | agricultural land | 14.3668 | 13.5695 | -0.7973 | 14.9916 | 58.8548 | 1.1971 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0142 |

## paraphrase

| Dataset/protocol | NoAdmission_Exact | CoherentNativeJoint_Exact | CoherentNoHoldout_Exact | ClassAttachmentJoint_Exact | SemanticRoleJoint_Exact | SemanticRoleOnly_Exact | SemanticRoleVisualSupported_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 54.530664 | 55.328998 | 55.415298 | 55.401869 | 55.415298 | 54.530664 | 55.415298 |
| potsdam/potsdam | 41.663326 | 42.224396 | 42.320156 | 42.372603 | 42.320156 | 41.663326 | 42.320156 |
| udd5/udd5 | 31.169271 | 33.291316 | 33.568498 | 33.297857 | 33.568498 | 31.169271 | 33.568498 |
| oem/oem | 39.619434 | 40.095996 | 39.987607 | 39.989834 | 39.987607 | 39.619434 | 39.987607 |
| loveda/P | 48.735654 | 50.319881 | 50.283629 | 50.263914 | 50.283629 | 48.735654 | 50.283629 |
| loveda/D | 29.513994 | 32.114437 | 32.061199 | 31.885494 | 32.061199 | 29.513994 | 32.061199 |
| vaihingen/vaihingen | 52.518481 | 53.516799 | 53.527369 | 53.521288 | 53.527369 | 52.518481 | 53.527369 |
| landcoverai/landcoverai | 65.629553 | 65.944705 | 65.844126 | 65.868127 | 65.844126 | 65.629553 | 65.844126 |
| flair1/flair1 | 34.530135 | 34.534005 | 34.534747 | 34.468500 | 34.511923 | 34.483004 | 34.529446 |
| Domain mean | 43.646857 | 44.631331 | 44.657375 | 44.600696 | 44.654522 | 43.640966 | 44.656712 |

| All prospective controls | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 44.631331 |
| CoherentNoHoldout_Exact | 44.657375 |
| ClassAttachmentJoint_Exact | 44.600696 |
| ClassAttachmentTextOnly_Exact | 43.415082 |
| SemanticRoleOnly_Exact | 43.640966 |
| SemanticRoleVisualSupported_Exact | 44.656712 |
| SemanticRoleHard_Exact | 44.653373 |
| SemanticRoleMeanLogit | 44.087705 |
| SemanticRoleAliasShuffle0_Exact | 44.623962 |
| SemanticRoleAliasShuffle1_Exact | 44.656408 |
| SemanticRoleAliasShuffle2_Exact | 44.655164 |
| SemanticRoleJoint_Exact | 44.654522 |

| Domain/protocol | Class | Original IoU | Primary IoU | Delta pp | Precision % | Recall % | Area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 57.9155 | 60.0975 | 2.1820 | 83.1936 | 68.4019 | 41.9780 |
| vdd/vdd | wall | 56.3455 | 61.6849 | 5.3394 | 66.0438 | 90.3346 | 10.8811 |
| vdd/vdd | road | 21.6181 | 22.9958 | 1.3777 | 23.1075 | 97.9412 | 14.0677 |
| vdd/vdd | vegetation | 48.8482 | 52.5498 | 3.7016 | 92.0290 | 55.0557 | 13.7805 |
| vdd/vdd | vehicle | 52.7210 | 39.6327 | -13.0883 | 39.6327 | 100.0000 | 0.9036 |
| vdd/vdd | roof | 92.2355 | 93.6614 | 1.4259 | 94.1030 | 99.5014 | 6.3711 |
| vdd/vdd | water | 52.0308 | 57.2849 | 5.2541 | 61.4280 | 89.4665 | 12.0180 |
| potsdam/potsdam | impervious surface | 61.6785 | 61.1804 | -0.4981 | 85.5299 | 68.2441 | 32.6152 |
| potsdam/potsdam | building | 70.7181 | 72.1481 | 1.4300 | 76.4006 | 92.8378 | 19.5916 |
| potsdam/potsdam | low vegetation | 33.5899 | 37.1875 | 3.5976 | 75.0770 | 42.4249 | 9.7991 |
| potsdam/potsdam | tree | 59.9611 | 59.4383 | -0.5228 | 93.6477 | 61.9354 | 11.6689 |
| potsdam/potsdam | car | 21.0407 | 20.7759 | -0.2648 | 20.7831 | 99.8338 | 17.2239 |
| potsdam/potsdam | clutter | 2.9917 | 3.1908 | 0.1991 | 4.5974 | 9.4439 | 9.1013 |
| udd5/udd5 | vegetation | 52.6873 | 59.3116 | 6.6243 | 88.1697 | 64.4399 | 2.2563 |
| udd5/udd5 | building | 84.1700 | 84.2639 | 0.0939 | 84.5468 | 99.6045 | 84.7352 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1014 |
| udd5/udd5 | vehicle | 6.6957 | 8.5409 | 1.8452 | 8.5411 | 99.9773 | 4.9163 |
| udd5/udd5 | other | 12.2934 | 15.7260 | 3.4326 | 48.1415 | 18.9334 | 7.9908 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.0229 |
| oem/oem | rangeland | 51.1534 | 51.1980 | 0.0446 | 70.7027 | 64.9845 | 13.3698 |
| oem/oem | developed space | 19.2067 | 17.6025 | -1.6042 | 35.5078 | 25.8750 | 14.2810 |
| oem/oem | road | 55.1295 | 55.7466 | 0.6171 | 70.9532 | 72.2307 | 6.4004 |
| oem/oem | tree | 57.1085 | 57.3904 | 0.2819 | 91.3326 | 60.6962 | 17.5568 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| oem/oem | agriculture land | 82.6661 | 82.3568 | -0.3093 | 82.7697 | 99.3978 | 25.4754 |
| oem/oem | building | 51.6912 | 55.6066 | 3.9154 | 71.5265 | 71.4151 | 11.8936 |
| loveda/P | building | 55.2347 | 62.0202 | 6.7855 | 62.0202 | 100.0000 | 0.0427 |
| loveda/P | road | 50.6803 | 52.8228 | 2.1425 | 53.7429 | 96.8607 | 14.9991 |
| loveda/P | water | 62.5637 | 62.9600 | 0.3963 | 68.8650 | 88.0131 | 18.8060 |
| loveda/P | barren | 3.7827 | 2.1254 | -1.6573 | 91.1392 | 2.1298 | 0.2591 |
| loveda/P | tree | 37.1877 | 39.1728 | 1.9851 | 99.1092 | 39.3112 | 6.5307 |
| loveda/P | farm | 82.9648 | 82.6006 | -0.3642 | 82.8679 | 99.6110 | 59.3624 |
| loveda/D | background | 4.9139 | 11.4117 | 6.4978 | 80.7259 | 11.7313 | 6.5036 |
| loveda/D | building | 29.3384 | 37.8079 | 8.4695 | 37.8079 | 100.0000 | 0.0387 |
| loveda/D | road | 40.8782 | 42.7235 | 1.8453 | 43.3269 | 96.8431 | 10.2769 |
| loveda/D | water | 54.8009 | 54.8153 | 0.0144 | 59.2595 | 87.9650 | 12.0673 |
| loveda/D | barren | 2.9040 | 1.6008 | -1.3032 | 77.5526 | 1.6083 | 0.1270 |
| loveda/D | tree | 35.3763 | 35.9225 | 0.5462 | 97.0016 | 36.3258 | 3.4065 |
| loveda/D | farm | 38.3862 | 40.1468 | 1.7606 | 40.2114 | 99.6013 | 67.5800 |
| vaihingen/vaihingen | impervious surface | 57.2431 | 58.9108 | 1.6677 | 78.5487 | 70.2057 | 24.4153 |
| vaihingen/vaihingen | building | 66.4345 | 67.1297 | 0.6952 | 67.2610 | 99.7102 | 30.5836 |
| vaihingen/vaihingen | low vegetation | 48.4163 | 49.4995 | 1.0832 | 93.9695 | 51.1235 | 16.0863 |
| vaihingen/vaihingen | tree | 67.5330 | 67.4567 | -0.0763 | 78.7242 | 82.4963 | 21.6275 |
| vaihingen/vaihingen | car | 22.9655 | 24.6402 | 1.6747 | 24.7770 | 97.8071 | 7.2874 |
| landcoverai/landcoverai | background | 86.7775 | 86.8935 | 0.1160 | 94.4167 | 91.6003 | 65.7388 |
| landcoverai/landcoverai | building | 39.1418 | 39.8556 | 0.7138 | 40.0416 | 98.8481 | 3.7401 |
| landcoverai/landcoverai | woodland | 79.8202 | 79.7288 | -0.0914 | 93.9801 | 84.0197 | 19.1380 |
| landcoverai/landcoverai | water | 97.4402 | 97.5646 | 0.1244 | 97.5646 | 100.0000 | 8.4778 |
| landcoverai/landcoverai | road | 24.9682 | 25.1782 | 0.2100 | 27.3614 | 75.9360 | 2.9053 |
| flair1/flair1 | building | 55.1554 | 55.9445 | 0.7891 | 57.3240 | 95.8759 | 12.0079 |
| flair1/flair1 | pervious surface | 49.3192 | 48.4459 | -0.8733 | 95.2479 | 49.6458 | 8.9442 |
| flair1/flair1 | impervious surface | 52.3579 | 53.1167 | 0.7588 | 60.9949 | 80.4399 | 21.6051 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.1056 |
| flair1/flair1 | water | 65.3852 | 65.2134 | -0.1718 | 67.5081 | 95.0458 | 6.2455 |
| flair1/flair1 | coniferous | 43.9217 | 43.7060 | -0.2157 | 64.6570 | 57.4253 | 0.4993 |
| flair1/flair1 | deciduous | 61.2313 | 62.4803 | 1.2490 | 77.4752 | 76.3494 | 16.7743 |
| flair1/flair1 | brushwood | 14.7571 | 13.9205 | -0.8366 | 33.5889 | 19.2068 | 2.8124 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0114 |
| flair1/flair1 | herbaceous vegetation | 58.9987 | 58.5074 | -0.4913 | 90.6620 | 62.2592 | 21.9991 |
| flair1/flair1 | agricultural land | 13.2352 | 12.8084 | -0.4268 | 13.7884 | 64.3148 | 1.4223 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5728 |

## llm_style

| Dataset/protocol | NoAdmission_Exact | CoherentNativeJoint_Exact | CoherentNoHoldout_Exact | ClassAttachmentJoint_Exact | SemanticRoleJoint_Exact | SemanticRoleOnly_Exact | SemanticRoleVisualSupported_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | 31.837398 | 33.236409 | 33.251536 | 33.280775 | 33.251536 | 31.837398 | 33.251536 |
| oem/oem | 40.439688 | 40.337843 | 40.387403 | 40.366842 | 40.387403 | 40.439688 | 40.387403 |
| loveda/P | 50.706552 | 53.281856 | 53.513897 | 53.496902 | 53.513897 | 50.706552 | 53.513897 |
| loveda/D | 30.736030 | 34.135199 | 33.944372 | 33.829905 | 33.944372 | 30.736030 | 33.944372 |
| Domain mean | 34.337705 | 35.903150 | 35.861104 | 35.825841 | 35.861104 | 34.337705 | 35.861104 |

| All prospective controls | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 35.903150 |
| CoherentNoHoldout_Exact | 35.861104 |
| ClassAttachmentJoint_Exact | 35.825841 |
| ClassAttachmentTextOnly_Exact | 34.724253 |
| SemanticRoleOnly_Exact | 34.337705 |
| SemanticRoleVisualSupported_Exact | 35.861104 |
| SemanticRoleHard_Exact | 35.861104 |
| SemanticRoleMeanLogit | 35.571622 |
| SemanticRoleAliasShuffle0_Exact | 35.861104 |
| SemanticRoleAliasShuffle1_Exact | 35.861104 |
| SemanticRoleAliasShuffle2_Exact | 35.861104 |
| SemanticRoleJoint_Exact | 35.861104 |

| Domain/protocol | Class | Original IoU | Primary IoU | Delta pp | Precision % | Recall % | Area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | vegetation | 57.1566 | 61.1230 | 3.9664 | 85.7588 | 68.0279 | 2.4489 |
| udd5/udd5 | building | 75.1152 | 75.3211 | 0.2059 | 75.3211 | 100.0000 | 95.4916 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3322 |
| udd5/udd5 | vehicle | 26.7195 | 28.5114 | 1.7919 | 28.5644 | 99.3529 | 1.4608 |
| udd5/udd5 | other | 0.1958 | 1.3022 | 1.1064 | 99.3376 | 1.3023 | 0.2664 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3631 |
| oem/oem | rangeland | 51.8244 | 51.9283 | 0.1039 | 71.0315 | 65.8803 | 13.4913 |
| oem/oem | developed space | 28.5411 | 27.4352 | -1.1059 | 46.3235 | 40.2216 | 17.0162 |
| oem/oem | road | 55.0628 | 55.2935 | 0.2307 | 69.3030 | 73.2283 | 6.6433 |
| oem/oem | tree | 56.6810 | 56.9907 | 0.3097 | 91.1160 | 60.3438 | 17.4963 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 |
| oem/oem | agriculture land | 80.8260 | 80.4793 | -0.3467 | 80.8425 | 99.4449 | 26.0951 |
| oem/oem | building | 50.5822 | 50.9722 | 0.3900 | 67.5761 | 67.4745 | 11.8942 |
| loveda/P | building | 60.8479 | 76.1773 | 15.3294 | 83.5866 | 89.5765 | 0.0284 |
| loveda/P | road | 53.7553 | 55.8436 | 2.0883 | 56.9027 | 96.7746 | 14.1536 |
| loveda/P | water | 62.4091 | 62.7439 | 0.3348 | 67.9932 | 89.0437 | 19.2701 |
| loveda/P | barren | 4.3215 | 2.3564 | -1.9651 | 91.1659 | 2.3618 | 0.2872 |
| loveda/P | tree | 39.9553 | 41.3957 | 1.4404 | 99.2570 | 41.5244 | 6.8881 |
| loveda/P | farm | 82.9502 | 82.5665 | -0.3837 | 82.8428 | 99.5977 | 59.3725 |
| loveda/D | background | 7.3224 | 14.4463 | 7.1239 | 76.3082 | 15.1247 | 8.8702 |
| loveda/D | building | 31.2821 | 46.3744 | 15.0923 | 49.0196 | 89.5765 | 0.0268 |
| loveda/D | road | 44.1270 | 46.8793 | 2.7523 | 47.6321 | 96.7383 | 9.3379 |
| loveda/D | water | 54.4379 | 54.5619 | 0.1240 | 58.5144 | 88.9838 | 12.3625 |
| loveda/D | barren | 3.2414 | 1.6702 | -1.5712 | 81.0689 | 1.6768 | 0.1267 |
| loveda/D | tree | 35.8674 | 32.7068 | -3.1606 | 97.7244 | 32.9578 | 3.0678 |
| loveda/D | farm | 38.8741 | 40.9717 | 2.0976 | 41.0406 | 99.5917 | 66.2081 |

## Incremental Word Effect Over The Visual Class/View Base

Original-model gains are not all word-selection gains. The incremental means are:

```json
{
  "clean": -0.0034288219985256774,
  "wrong_parent": -0.2654657512097103,
  "paraphrase": -0.0028530565649802497,
  "llm_style": 0.0
}
```

| Domain/protocol | Wrong-parent extra delta pp | Class | TP change | FP change | FN change |
| --- | ---: | --- | ---: | ---: | ---: |
| vdd/vdd | -0.089618 | other | -196 | 265 | 196 |
| vdd/vdd | -0.089618 | wall | -474 | -4522 | 474 |
| vdd/vdd | -0.089618 | road | 39 | 4369 | -39 |
| vdd/vdd | -0.089618 | vegetation | -668 | -290 | 668 |
| vdd/vdd | -0.089618 | vehicle | 0 | 265 | 0 |
| vdd/vdd | -0.089618 | roof | 16 | 498 | -16 |
| vdd/vdd | -0.089618 | water | 0 | 698 | 0 |
| potsdam/potsdam | +1.165091 | impervious surface | -2339 | -5286 | 2339 |
| potsdam/potsdam | +1.165091 | building | 3243 | 3543 | -3243 |
| potsdam/potsdam | +1.165091 | low vegetation | -16479 | -57292 | 16479 |
| potsdam/potsdam | +1.165091 | tree | 35482 | 1510 | -35482 |
| potsdam/potsdam | +1.165091 | car | 17 | 674 | -17 |
| potsdam/potsdam | +1.165091 | clutter | 736 | 36191 | -736 |
| udd5/udd5 | -1.016089 | vegetation | -3978 | -1099 | 3978 |
| udd5/udd5 | -1.016089 | building | -79 | -1178 | 79 |
| udd5/udd5 | -1.016089 | road | 0 | -332 | 0 |
| udd5/udd5 | -1.016089 | vehicle | 0 | 6659 | 0 |
| udd5/udd5 | -1.016089 | other | 0 | 7 | 0 |
| oem/oem | -0.091761 | bareland | 0 | 1319 | 0 |
| oem/oem | -0.091761 | rangeland | -1193 | -312 | 1193 |
| oem/oem | -0.091761 | developed space | -1771 | -1815 | 1771 |
| oem/oem | -0.091761 | road | 332 | 822 | -332 |
| oem/oem | -0.091761 | tree | 351 | 203 | -351 |
| oem/oem | -0.091761 | water | 0 | 48 | 0 |
| oem/oem | -0.091761 | agriculture land | 9 | 781 | -9 |
| oem/oem | -0.091761 | building | 210 | 1016 | -210 |
| loveda/P | -0.545264 | building | -14 | 5 | 14 |
| loveda/P | -0.545264 | road | 0 | -1049 | 0 |
| loveda/P | -0.545264 | water | 146 | -986 | -146 |
| loveda/P | -0.545264 | barren | -108 | -143 | 108 |
| loveda/P | -0.545264 | tree | 664 | 104 | -664 |
| loveda/P | -0.545264 | farm | 18 | 1363 | -18 |
| loveda/D | -0.279364 | background | -78 | 273 | 78 |
| loveda/D | -0.279364 | building | -20 | -20 | 20 |
| loveda/D | -0.279364 | road | -6 | -1424 | 6 |
| loveda/D | -0.279364 | water | 235 | 681 | -235 |
| loveda/D | -0.279364 | barren | -1 | -5 | 1 |
| loveda/D | -0.279364 | tree | 164 | 4 | -164 |
| loveda/D | -0.279364 | farm | 0 | 197 | 0 |
| vaihingen/vaihingen | -1.510174 | impervious surface | -6257 | 3964 | 6257 |
| vaihingen/vaihingen | -1.510174 | building | 1308 | 14875 | -1308 |
| vaihingen/vaihingen | -1.510174 | low vegetation | -40585 | -29801 | 40585 |
| vaihingen/vaihingen | -1.510174 | tree | 18354 | 21853 | -18354 |
| vaihingen/vaihingen | -1.510174 | car | 20 | 16269 | -20 |
| landcoverai/landcoverai | -0.572242 | background | -3301 | -22 | 3301 |
| landcoverai/landcoverai | -0.572242 | building | 12 | 2931 | -12 |
| landcoverai/landcoverai | -0.572242 | woodland | 22 | -21 | -22 |
| landcoverai/landcoverai | -0.572242 | water | 0 | -59 | 0 |
| landcoverai/landcoverai | -0.572242 | road | 8 | 430 | -8 |
| flair1/flair1 | +0.270430 | building | 29 | 6339 | -29 |
| flair1/flair1 | +0.270430 | pervious surface | -5411 | -8207 | 5411 |
| flair1/flair1 | +0.270430 | impervious surface | -430 | -994 | 430 |
| flair1/flair1 | +0.270430 | bare soil | 0 | 4446 | 0 |
| flair1/flair1 | +0.270430 | water | 0 | 355 | 0 |
| flair1/flair1 | +0.270430 | coniferous | -40 | -2584 | 40 |
| flair1/flair1 | +0.270430 | deciduous | 1130 | 1107 | -1130 |
| flair1/flair1 | +0.270430 | brushwood | 63 | 1705 | -63 |
| flair1/flair1 | +0.270430 | vineyard | 0 | 0 | 0 |
| flair1/flair1 | +0.270430 | herbaceous vegetation | 1816 | 574 | -1816 |
| flair1/flair1 | +0.270430 | agricultural land | 0 | 102 | 0 |
| flair1/flair1 | +0.270430 | plowed land | 0 | 0 | 0 |

## Semantic Decisions

| Dataset/regime/protocol | Parent | Alias | Rival | Strength |
| --- | --- | --- | --- | ---: |
| vdd/wrong_parent__vdd | other | wall | wall | 0.450519 |
| vdd/wrong_parent__vdd | other | a view of wall | wall | 0.518627 |
| vdd/wrong_parent__vdd | other | wall seen from above | wall | 0.337384 |
| vdd/wrong_parent__vdd | other | an area of wall | wall | 0.689506 |
| vdd/wrong_parent__vdd | wall | road | road | 0.664767 |
| vdd/wrong_parent__vdd | wall | a view of road | road | 0.562867 |
| vdd/wrong_parent__vdd | wall | road seen from above | road | 0.333229 |
| vdd/wrong_parent__vdd | wall | an area of road | road | 0.772977 |
| vdd/wrong_parent__vdd | road | vegetation | vegetation | 0.536060 |
| vdd/wrong_parent__vdd | road | a view of vegetation | vegetation | 0.412971 |
| vdd/wrong_parent__vdd | road | an area of vegetation | vegetation | 0.634148 |
| vdd/wrong_parent__vdd | vegetation | vehicle | vehicle | 0.984467 |
| vdd/wrong_parent__vdd | vegetation | a view of vehicle | vehicle | 0.920329 |
| vdd/wrong_parent__vdd | vegetation | vehicle seen from above | vehicle | 0.741722 |
| vdd/wrong_parent__vdd | vegetation | an area of vehicle | vehicle | 0.861104 |
| vdd/wrong_parent__vdd | vehicle | an area of roof | roof | 0.412579 |
| vdd/wrong_parent__vdd | roof | water | water | 0.755683 |
| vdd/wrong_parent__vdd | roof | a view of water | water | 0.425216 |
| vdd/wrong_parent__vdd | roof | water seen from above | water | 0.358692 |
| vdd/wrong_parent__vdd | roof | an area of water | water | 0.678357 |
| potsdam/wrong_parent__potsdam | impervious surface | building | building | 0.933368 |
| potsdam/wrong_parent__potsdam | impervious surface | a view of building | building | 0.663658 |
| potsdam/wrong_parent__potsdam | impervious surface | building seen from above | building | 0.341917 |
| potsdam/wrong_parent__potsdam | impervious surface | an area of building | building | 0.662290 |
| potsdam/wrong_parent__potsdam | low vegetation | tree | tree | 0.737315 |
| potsdam/wrong_parent__potsdam | low vegetation | a view of tree | tree | 0.840332 |
| potsdam/wrong_parent__potsdam | low vegetation | tree seen from above | tree | 0.784797 |
| potsdam/wrong_parent__potsdam | low vegetation | an area of tree | tree | 0.811181 |
| potsdam/wrong_parent__potsdam | tree | car | car | 0.972338 |
| potsdam/wrong_parent__potsdam | tree | a view of car | car | 0.925171 |
| potsdam/wrong_parent__potsdam | tree | car seen from above | car | 0.711441 |
| potsdam/wrong_parent__potsdam | tree | an area of car | car | 0.963417 |
| potsdam/wrong_parent__potsdam | car | a view of clutter | clutter | 0.474241 |
| potsdam/wrong_parent__potsdam | car | an area of clutter | clutter | 0.553170 |
| udd5/wrong_parent__udd5 | vegetation | building | building | 0.970347 |
| udd5/wrong_parent__udd5 | vegetation | a view of building | building | 0.965734 |
| udd5/wrong_parent__udd5 | vegetation | building seen from above | building | 0.693623 |
| udd5/wrong_parent__udd5 | vegetation | an area of building | building | 0.972530 |
| udd5/wrong_parent__udd5 | building | road | road | 0.603220 |
| udd5/wrong_parent__udd5 | building | a view of road | road | 0.621883 |
| udd5/wrong_parent__udd5 | building | road seen from above | road | 0.486586 |
| udd5/wrong_parent__udd5 | building | an area of road | road | 0.732600 |
| udd5/wrong_parent__udd5 | road | vehicle | vehicle | 0.810506 |
| udd5/wrong_parent__udd5 | road | a view of vehicle | vehicle | 0.858067 |
| udd5/wrong_parent__udd5 | road | vehicle seen from above | vehicle | 0.546105 |
| udd5/wrong_parent__udd5 | road | an area of vehicle | vehicle | 0.478909 |
| oem/wrong_parent__oem | bareland | a view of rangeland | rangeland | 0.495659 |
| oem/wrong_parent__oem | bareland | an area of rangeland | rangeland | 0.511879 |
| oem/wrong_parent__oem | rangeland | developed space | developed space | 0.601541 |
| oem/wrong_parent__oem | rangeland | a view of developed space | developed space | 0.661965 |
| oem/wrong_parent__oem | rangeland | developed space seen from above | developed space | 0.683541 |
| oem/wrong_parent__oem | rangeland | an area of developed space | developed space | 0.722554 |
| oem/wrong_parent__oem | developed space | road seen from above | road | 0.350492 |
| oem/wrong_parent__oem | developed space | an area of road | road | 0.628052 |
| oem/wrong_parent__oem | road | a view of tree | tree | 0.500333 |
| oem/wrong_parent__oem | road | an area of tree | tree | 0.536222 |
| oem/wrong_parent__oem | tree | water | water | 0.707216 |
| oem/wrong_parent__oem | tree | an area of water | water | 0.608224 |
| oem/wrong_parent__oem | agriculture land | a view of building | building | 0.557374 |
| loveda/wrong_parent__P | building | road | road | 0.865542 |
| loveda/wrong_parent__P | building | an area of road | road | 0.518174 |
| loveda/wrong_parent__P | road | water | water | 0.800975 |
| loveda/wrong_parent__P | road | a view of water | water | 0.596037 |
| loveda/wrong_parent__P | road | water seen from above | water | 0.464688 |
| loveda/wrong_parent__P | road | an area of water | water | 0.734044 |
| loveda/wrong_parent__P | water | barren seen from above | barren | 0.493370 |
| loveda/wrong_parent__P | barren | tree | tree | 0.829306 |
| loveda/wrong_parent__P | barren | a view of tree | tree | 0.678105 |
| loveda/wrong_parent__P | barren | tree seen from above | tree | 0.449464 |
| loveda/wrong_parent__P | barren | an area of tree | tree | 0.565089 |
| loveda/wrong_parent__P | farm | building | building | 0.858975 |
| loveda/wrong_parent__P | farm | a view of building | building | 0.661835 |
| loveda/wrong_parent__P | farm | an area of building | building | 0.640655 |
| loveda/wrong_parent__D | background | building | building | 0.897140 |
| loveda/wrong_parent__D | background | a view of building | building | 0.593714 |
| loveda/wrong_parent__D | background | building seen from above | building | 0.422585 |
| loveda/wrong_parent__D | background | an area of building | building | 0.603020 |
| loveda/wrong_parent__D | building | road seen from above | road | 0.339414 |
| loveda/wrong_parent__D | building | an area of road | road | 0.565947 |
| loveda/wrong_parent__D | road | water | water | 0.861571 |
| loveda/wrong_parent__D | road | a view of water | water | 0.586636 |
| loveda/wrong_parent__D | road | water seen from above | water | 0.552487 |
| loveda/wrong_parent__D | road | an area of water | water | 0.765848 |
| loveda/wrong_parent__D | barren | tree | tree | 0.661728 |
| loveda/wrong_parent__D | barren | a view of tree | tree | 0.515069 |
| loveda/wrong_parent__D | barren | tree seen from above | tree | 0.348133 |
| loveda/wrong_parent__D | barren | an area of tree | tree | 0.375528 |
| loveda/wrong_parent__D | farm | background | background | 0.373564 |
| loveda/wrong_parent__D | farm | a view of background | background | 0.419296 |
| loveda/wrong_parent__D | farm | an area of background | background | 0.455740 |
| vaihingen/wrong_parent__vaihingen | impervious surface | building | building | 0.885422 |
| vaihingen/wrong_parent__vaihingen | impervious surface | a view of building | building | 0.807125 |
| vaihingen/wrong_parent__vaihingen | impervious surface | building seen from above | building | 0.489780 |
| vaihingen/wrong_parent__vaihingen | impervious surface | an area of building | building | 0.750443 |
| vaihingen/wrong_parent__vaihingen | building | an area of low vegetation | low vegetation | 0.427408 |
| vaihingen/wrong_parent__vaihingen | low vegetation | tree | tree | 0.814771 |
| vaihingen/wrong_parent__vaihingen | low vegetation | a view of tree | tree | 0.866690 |
| vaihingen/wrong_parent__vaihingen | low vegetation | tree seen from above | tree | 0.824053 |
| vaihingen/wrong_parent__vaihingen | low vegetation | an area of tree | tree | 0.837057 |
| vaihingen/wrong_parent__vaihingen | tree | car | car | 0.891461 |
| vaihingen/wrong_parent__vaihingen | tree | a view of car | car | 0.824066 |
| vaihingen/wrong_parent__vaihingen | tree | an area of car | car | 0.619549 |
| vaihingen/wrong_parent__vaihingen | car | a view of impervious surface | impervious surface | 0.601627 |
| landcoverai/wrong_parent__landcoverai | background | building | building | 0.880426 |
| landcoverai/wrong_parent__landcoverai | background | a view of building | building | 0.677664 |
| landcoverai/wrong_parent__landcoverai | background | building seen from above | building | 0.442929 |
| landcoverai/wrong_parent__landcoverai | background | an area of building | building | 0.731797 |
| landcoverai/wrong_parent__landcoverai | building | woodland | woodland | 0.486515 |
| landcoverai/wrong_parent__landcoverai | building | a view of woodland | woodland | 0.711943 |
| landcoverai/wrong_parent__landcoverai | building | woodland seen from above | woodland | 0.564494 |
| landcoverai/wrong_parent__landcoverai | building | an area of woodland | woodland | 0.825203 |
| landcoverai/wrong_parent__landcoverai | woodland | water | water | 0.882965 |
| landcoverai/wrong_parent__landcoverai | woodland | a view of water | water | 0.531089 |
| landcoverai/wrong_parent__landcoverai | woodland | water seen from above | water | 0.491657 |
| landcoverai/wrong_parent__landcoverai | woodland | an area of water | water | 0.785781 |
| landcoverai/wrong_parent__landcoverai | water | a view of road | road | 0.774728 |
| landcoverai/wrong_parent__landcoverai | water | an area of road | road | 0.921557 |
| landcoverai/wrong_parent__landcoverai | road | background | background | 0.721001 |
| landcoverai/wrong_parent__landcoverai | road | an area of background | background | 0.535386 |
| flair1/clean__flair1 | building | agricultural silo | agricultural land | 0.330040 |
| flair1/clean__flair1 | pervious surface | artificial bare soil | bare soil | 0.732086 |
| flair1/wrong_parent__flair1 | pervious surface | artificial bare soil | bare soil | 0.732086 |
| flair1/wrong_parent__flair1 | pervious surface | impervious surface | impervious surface | 0.829821 |
| flair1/wrong_parent__flair1 | pervious surface | a view of impervious surface | impervious surface | 0.873596 |
| flair1/wrong_parent__flair1 | pervious surface | impervious surface seen from above | impervious surface | 0.477159 |
| flair1/wrong_parent__flair1 | pervious surface | an area of impervious surface | impervious surface | 0.864043 |
| flair1/wrong_parent__flair1 | impervious surface | a view of bare soil | bare soil | 0.457614 |
| flair1/wrong_parent__flair1 | impervious surface | bare soil seen from above | bare soil | 0.447683 |
| flair1/wrong_parent__flair1 | impervious surface | an area of bare soil | bare soil | 0.562106 |
| flair1/wrong_parent__flair1 | bare soil | water | water | 0.606970 |
| flair1/wrong_parent__flair1 | coniferous | deciduous | deciduous | 0.539541 |
| flair1/wrong_parent__flair1 | coniferous | a view of deciduous | deciduous | 0.549644 |
| flair1/wrong_parent__flair1 | coniferous | deciduous seen from above | deciduous | 0.419044 |
| flair1/wrong_parent__flair1 | coniferous | an area of deciduous | deciduous | 0.524816 |
| flair1/wrong_parent__flair1 | plowed land | building | building | 0.682637 |
| flair1/wrong_parent__flair1 | plowed land | a view of building | building | 0.421171 |
| flair1/wrong_parent__flair1 | plowed land | an area of building | building | 0.668892 |
| flair1/paraphrase__flair1 | pervious surface | artificial bare soil | bare soil | 0.732086 |

## Cost

Prediction suite wall 704.3076s.

| Domain | Vocabulary source seconds | Source peak MiB | Multi-arm worker seconds | Combined prediction peak MiB |
| --- | ---: | ---: | ---: | ---: |
| vdd | 18.4572 | 16647.3765 | 149.8770 | 5624.3086 |
| potsdam | 17.7829 | 16624.2446 | 184.7469 | 5612.3994 |
| udd5 | 18.0351 | 16560.1650 | 155.7554 | 5619.8784 |
| oem | 27.8658 | 16727.0283 | 302.6000 | 5679.8452 |
| loveda | 27.6529 | 16650.2632 | 525.4669 | 5698.5620 |
| vaihingen | 16.4059 | 16575.0205 | 175.3322 | 5601.1846 |
| landcoverai | 16.3116 | 16558.5791 | 168.9194 | 5601.1846 |
| flair1 | 32.9749 | 16963.6230 | 306.2972 | 5957.2285 |

Source uses no images/masks and no per-image language forwards. Prediction cost shares69 arms and all regimes; not standalone primary latency. Semantic scores/order consensus are not truth certificates. Historical style banks are not verified raw-provider outputs. Context-only screening does not clean the fixed local20 bank. Good aliases retain existing responsibility; no explicit positive amplification was tested. SemanticRoleHard uses binary risk on excess responsibility, not complete vocabulary deletion.
