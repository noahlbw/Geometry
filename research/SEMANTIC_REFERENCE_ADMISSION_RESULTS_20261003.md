# Semantic-Admissible Native Reference: Verified Prediction Results

Same64 developed top-left512 windows, eight/domain, not full or untouched validation. Frozen weights,20 aliases/class, original Geometry/local20/wide observer/reconstruction. Only context banks are perturbed; all69 preceding numerical/per-image endpoints replay exactly. LoveDA D counts once in means, P separately. LandCover.ai substitutes for unlabeled iSAID. Existing frozen Qwen text-role observations are reused without new language inference or prompt changes.

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
    "above_CoherentNoHoldout_Exact": true,
    "above_ClassAttachmentJoint_Exact": false,
    "above_ClassAttachmentTextOnly_Exact": false,
    "above_SemanticRoleJoint_Exact": true,
    "above_SemanticRoleVisualSupported_Exact": true,
    "above_SemanticReferenceOnly_Exact": false,
    "above_SemanticReferenceNoView_Exact": true,
    "above_SemanticReferenceAttachmentOnly_Exact": true,
    "above_SemanticReferenceCosine_Exact": false,
    "above_SemanticReferenceBinary_Exact": true,
    "above_SemanticReferenceMeanLogit": true,
    "above_SemanticReferenceShuffle0_Exact": true,
    "above_SemanticReferenceShuffle1_Exact": true,
    "above_SemanticReferenceShuffle2_Exact": false
  },
  "wrong_parent_domain_wins": 8,
  "clean_gain_pp": 1.222796431402486,
  "wrong_parent_gain_pp": 0.8826731686481111,
  "worst_clean_paraphrase_delta_pp": -0.0011444236386140005,
  "earlier_failed_gates_unchanged": true
}
```

FAIL: no controller promotion, winning-control selection or full rollout.

## Isolated Increments

```json
{
  "clean": {
    "reference_only_over_old_base": -0.0005150948195478122,
    "complete_over_old_base": -0.0010860572142092906,
    "attachment_over_reference_only": -0.0005709623946614784
  },
  "wrong_parent": {
    "reference_only_over_old_base": 0.032030079427791236,
    "complete_over_old_base": 0.016415001396786977,
    "attachment_over_reference_only": -0.01561507803100426
  },
  "paraphrase": {
    "reference_only_over_old_base": -0.00015752724703332888,
    "complete_over_old_base": -0.0007195909308421733,
    "attachment_over_reference_only": -0.0005620636838088444
  },
  "llm_style": {
    "reference_only_over_old_base": 0.0,
    "complete_over_old_base": 0.0,
    "attachment_over_reference_only": 0.0
  }
}
```

## clean

| Dataset/protocol | NoAdmission_Exact | CoherentNoHoldout_Exact | SemanticRoleJoint_Exact | SemanticRoleVisualSupported_Exact | SemanticReferenceOnly_Exact | SemanticReferenceJoint_Exact | SemanticReferenceCosine_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 53.746954 | 55.833087 | 55.833087 | 55.833087 | 55.833087 | 55.833087 | 55.835045 |
| potsdam/potsdam | 38.920258 | 39.537067 | 39.537067 | 39.537067 | 39.537067 | 39.537067 | 39.599082 |
| udd5/udd5 | 28.175811 | 30.674206 | 30.674206 | 30.674206 | 30.674206 | 30.674206 | 30.645131 |
| oem/oem | 39.023152 | 39.298606 | 39.298606 | 39.298606 | 39.298606 | 39.298606 | 39.310936 |
| loveda/P | 50.706552 | 53.513897 | 53.513897 | 53.513897 | 53.513897 | 53.513897 | 53.496902 |
| loveda/D | 30.736030 | 33.944372 | 33.944372 | 33.944372 | 33.944372 | 33.944372 | 33.829905 |
| vaihingen/vaihingen | 51.826962 | 52.851821 | 52.851821 | 52.851821 | 52.851821 | 52.851821 | 52.846145 |
| landcoverai/landcoverai | 66.906020 | 66.907578 | 66.907578 | 66.907578 | 66.907578 | 66.907578 | 66.965036 |
| flair1/flair1 | 33.608404 | 33.687914 | 33.660484 | 33.683102 | 33.683794 | 33.679226 | 33.621910 |
| Domain mean | 42.867949 | 44.091831 | 44.088403 | 44.091230 | 44.091316 | 44.090745 | 44.081649 |

| Prospective control | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 44.161110 |
| CoherentNoHoldout_Exact | 44.091831 |
| ClassAttachmentJoint_Exact | 44.082486 |
| ClassAttachmentTextOnly_Exact | 43.521285 |
| SemanticRoleJoint_Exact | 44.088403 |
| SemanticRoleVisualSupported_Exact | 44.091230 |
| SemanticReferenceOnly_Exact | 44.091316 |
| SemanticReferenceNoView_Exact | 43.964869 |
| SemanticReferenceAttachmentOnly_Exact | 42.865766 |
| SemanticReferenceCosine_Exact | 44.081649 |
| SemanticReferenceBinary_Exact | 44.089936 |
| SemanticReferenceMeanLogit | 43.818550 |
| SemanticReferenceShuffle0_Exact | 44.083106 |
| SemanticReferenceShuffle1_Exact | 44.092907 |
| SemanticReferenceShuffle2_Exact | 44.088460 |
| SemanticReferenceJoint_Exact | 44.090745 |

| Dataset/protocol | Class | Old class/view IoU | Primary IoU | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 61.3087 | 61.3087 | 81.3606 | 71.3270 | 44.7593 | 0 | 0 |
| vdd/vdd | wall | 64.5037 | 64.5037 | 71.1222 | 87.3922 | 9.7751 | 0 | 0 |
| vdd/vdd | road | 24.2656 | 24.2656 | 24.3888 | 97.9599 | 13.3312 | 0 | 0 |
| vdd/vdd | vegetation | 45.8169 | 45.8169 | 95.9666 | 46.7165 | 11.2134 | 0 | 0 |
| vdd/vdd | vehicle | 48.9761 | 48.9761 | 48.9761 | 100.0000 | 0.7312 | 0 | 0 |
| vdd/vdd | roof | 89.6779 | 89.6779 | 90.0769 | 99.5086 | 6.6563 | 0 | 0 |
| vdd/vdd | water | 56.2827 | 56.2827 | 57.9713 | 95.0794 | 13.5335 | 0 | 0 |
| potsdam/potsdam | impervious surface | 66.2596 | 66.2596 | 85.7327 | 74.4714 | 35.5072 | 0 | 0 |
| potsdam/potsdam | building | 71.9184 | 71.9184 | 76.0656 | 92.9531 | 19.7023 | 0 | 0 |
| potsdam/potsdam | low vegetation | 15.8962 | 15.8962 | 78.0148 | 16.6417 | 3.6991 | 0 | 0 |
| potsdam/potsdam | tree | 56.0681 | 56.0681 | 93.1635 | 58.4739 | 11.0740 | 0 | 0 |
| potsdam/potsdam | car | 24.9387 | 24.9387 | 24.9741 | 99.4348 | 14.2762 | 0 | 0 |
| potsdam/potsdam | clutter | 2.1414 | 2.1414 | 2.6866 | 9.5451 | 15.7413 | 0 | 0 |
| udd5/udd5 | vegetation | 51.7198 | 51.7198 | 92.8015 | 53.8814 | 1.7925 | 0 | 0 |
| udd5/udd5 | building | 83.9210 | 83.9210 | 84.0247 | 99.8532 | 85.4746 | 0 | 0 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4610 | 0 | 0 |
| udd5/udd5 | vehicle | 7.2681 | 7.2681 | 7.2682 | 99.9773 | 5.7773 | 0 | 0 |
| udd5/udd5 | other | 10.4621 | 10.4621 | 39.1012 | 12.4987 | 6.4947 | 0 | 0 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.9330 | 0 | 0 |
| oem/oem | rangeland | 49.7962 | 49.7962 | 70.4787 | 62.9202 | 12.9862 | 0 | 0 |
| oem/oem | developed space | 22.1134 | 22.1134 | 38.2379 | 34.4003 | 17.6308 | 0 | 0 |
| oem/oem | road | 55.1110 | 55.1110 | 69.2338 | 72.9853 | 6.6279 | 0 | 0 |
| oem/oem | tree | 56.0455 | 56.0455 | 91.7205 | 59.0320 | 17.0032 | 0 | 0 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0 | 0 |
| oem/oem | agriculture land | 81.3092 | 81.3092 | 81.6604 | 99.4738 | 25.8412 | 0 | 0 |
| oem/oem | building | 50.0136 | 50.0136 | 69.5169 | 64.0633 | 10.9776 | 0 | 0 |
| loveda/P | building | 76.1773 | 76.1773 | 83.5866 | 89.5765 | 0.0284 | 0 | 0 |
| loveda/P | road | 55.8436 | 55.8436 | 56.9027 | 96.7746 | 14.1536 | 0 | 0 |
| loveda/P | water | 62.7439 | 62.7439 | 67.9932 | 89.0437 | 19.2701 | 0 | 0 |
| loveda/P | barren | 2.3564 | 2.3564 | 91.1659 | 2.3618 | 0.2872 | 0 | 0 |
| loveda/P | tree | 41.3957 | 41.3957 | 99.2570 | 41.5244 | 6.8881 | 0 | 0 |
| loveda/P | farm | 82.5665 | 82.5665 | 82.8428 | 99.5977 | 59.3725 | 0 | 0 |
| loveda/D | background | 14.4463 | 14.4463 | 76.3082 | 15.1247 | 8.8702 | 0 | 0 |
| loveda/D | building | 46.3744 | 46.3744 | 49.0196 | 89.5765 | 0.0268 | 0 | 0 |
| loveda/D | road | 46.8793 | 46.8793 | 47.6321 | 96.7383 | 9.3379 | 0 | 0 |
| loveda/D | water | 54.5619 | 54.5619 | 58.5144 | 88.9838 | 12.3625 | 0 | 0 |
| loveda/D | barren | 1.6702 | 1.6702 | 81.0689 | 1.6768 | 0.1267 | 0 | 0 |
| loveda/D | tree | 32.7068 | 32.7068 | 97.7244 | 32.9578 | 3.0678 | 0 | 0 |
| loveda/D | farm | 40.9717 | 40.9717 | 41.0406 | 99.5917 | 66.2081 | 0 | 0 |
| vaihingen/vaihingen | impervious surface | 58.8479 | 58.8479 | 75.0983 | 73.1151 | 26.5953 | 0 | 0 |
| vaihingen/vaihingen | building | 66.4389 | 66.4389 | 66.5588 | 99.7296 | 30.9122 | 0 | 0 |
| vaihingen/vaihingen | low vegetation | 45.2176 | 45.2176 | 95.9748 | 46.0918 | 14.2000 | 0 | 0 |
| vaihingen/vaihingen | tree | 67.4096 | 67.4096 | 78.9471 | 82.1830 | 21.4846 | 0 | 0 |
| vaihingen/vaihingen | car | 26.3450 | 26.3450 | 26.5059 | 97.7476 | 6.8079 | 0 | 0 |
| landcoverai/landcoverai | background | 87.3541 | 87.3541 | 93.6745 | 92.8298 | 67.1490 | 0 | 0 |
| landcoverai/landcoverai | building | 44.7822 | 44.7822 | 45.1296 | 98.3099 | 3.3004 | 0 | 0 |
| landcoverai/landcoverai | woodland | 77.9455 | 77.9455 | 94.8215 | 81.4110 | 18.3793 | 0 | 0 |
| landcoverai/landcoverai | water | 97.7174 | 97.7174 | 97.7174 | 100.0000 | 8.4645 | 0 | 0 |
| landcoverai/landcoverai | road | 26.7388 | 26.7388 | 29.2569 | 75.6491 | 2.7068 | 0 | 0 |
| flair1/flair1 | building | 57.5880 | 57.5640 | 58.4556 | 97.4187 | 11.9650 | 58 | 207 |
| flair1/flair1 | pervious surface | 46.4695 | 46.3886 | 92.4188 | 48.2236 | 8.9539 | -429 | -272 |
| flair1/flair1 | impervious surface | 52.7526 | 52.7550 | 60.5360 | 80.4088 | 21.7604 | -47 | -113 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3659 | 0 | 604 |
| flair1/flair1 | water | 61.4327 | 61.4322 | 63.2524 | 95.5254 | 6.6994 | 0 | 1 |
| flair1/flair1 | coniferous | 43.4286 | 43.4286 | 63.9781 | 57.4847 | 0.5051 | 0 | 0 |
| flair1/flair1 | deciduous | 61.3912 | 61.3909 | 78.1996 | 74.0670 | 16.1221 | 4 | 9 |
| flair1/flair1 | brushwood | 12.1622 | 12.1644 | 29.1630 | 17.2661 | 2.9119 | 3 | -2 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0119 | 0 | 0 |
| flair1/flair1 | herbaceous vegetation | 56.5666 | 56.5642 | 90.9553 | 59.9355 | 21.1098 | -20 | -5 |
| flair1/flair1 | agricultural land | 12.4635 | 12.4627 | 13.3427 | 65.3942 | 1.4944 | 0 | 2 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1002 | 0 | 0 |

## wrong_parent

| Dataset/protocol | NoAdmission_Exact | CoherentNoHoldout_Exact | SemanticRoleJoint_Exact | SemanticRoleVisualSupported_Exact | SemanticReferenceOnly_Exact | SemanticReferenceJoint_Exact | SemanticReferenceCosine_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 47.408779 | 49.718132 | 49.628514 | 49.715715 | 49.791646 | 49.790707 | 49.870643 |
| potsdam/potsdam | 36.070589 | 36.118895 | 37.283986 | 36.253287 | 37.027222 | 37.177614 | 37.169533 |
| udd5/udd5 | 27.616556 | 28.317171 | 27.301082 | 28.290806 | 28.130332 | 28.088308 | 28.110502 |
| oem/oem | 40.053288 | 40.711109 | 40.619348 | 40.702483 | 40.665139 | 40.658392 | 40.708241 |
| loveda/P | 49.989695 | 50.287711 | 49.742447 | 50.265919 | 50.198686 | 50.192992 | 49.978861 |
| loveda/D | 30.333514 | 31.569316 | 31.289952 | 31.556340 | 31.465379 | 31.414332 | 31.301685 |
| vaihingen/vaihingen | 53.060153 | 54.201623 | 52.691449 | 54.205599 | 53.967926 | 53.822736 | 54.003836 |
| landcoverai/landcoverai | 68.123522 | 68.758172 | 68.185930 | 68.724942 | 68.452335 | 68.398622 | 68.460606 |
| flair1/flair1 | 33.246876 | 33.448926 | 33.719356 | 33.460285 | 33.599604 | 33.623952 | 33.563087 |
| Domain mean | 41.989160 | 42.855418 | 42.589952 | 42.863682 | 42.887448 | 42.871833 | 42.898517 |

| Prospective control | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 42.934464 |
| CoherentNoHoldout_Exact | 42.855418 |
| ClassAttachmentJoint_Exact | 42.895130 |
| ClassAttachmentTextOnly_Exact | 43.528885 |
| SemanticRoleJoint_Exact | 42.589952 |
| SemanticRoleVisualSupported_Exact | 42.863682 |
| SemanticReferenceOnly_Exact | 42.887448 |
| SemanticReferenceNoView_Exact | 42.831514 |
| SemanticReferenceAttachmentOnly_Exact | 41.991892 |
| SemanticReferenceCosine_Exact | 42.898517 |
| SemanticReferenceBinary_Exact | 42.854365 |
| SemanticReferenceMeanLogit | 42.165082 |
| SemanticReferenceShuffle0_Exact | 42.833683 |
| SemanticReferenceShuffle1_Exact | 42.861029 |
| SemanticReferenceShuffle2_Exact | 42.970401 |
| SemanticReferenceJoint_Exact | 42.871833 |

| Dataset/protocol | Class | Old class/view IoU | Primary IoU | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 59.5478 | 59.3229 | 80.2998 | 69.4272 | 44.1427 | -2986 | -281 |
| vdd/vdd | wall | 58.7083 | 58.2947 | 64.0063 | 86.7244 | 10.7788 | -51 | 1662 |
| vdd/vdd | road | 34.8646 | 34.3401 | 34.8273 | 96.0851 | 9.1568 | 198 | 3498 |
| vdd/vdd | vegetation | 36.0462 | 36.5292 | 96.6474 | 36.9980 | 8.8181 | 2337 | -73 |
| vdd/vdd | vehicle | 26.3833 | 27.2694 | 27.2694 | 100.0000 | 1.3132 | 0 | -925 |
| vdd/vdd | roof | 90.2110 | 90.1390 | 90.7389 | 99.2719 | 6.5920 | 64 | 182 |
| vdd/vdd | water | 42.2657 | 42.6397 | 42.7417 | 99.4435 | 19.1982 | -16 | -3609 |
| potsdam/potsdam | impervious surface | 69.4285 | 69.7208 | 84.2595 | 80.1615 | 38.8884 | 2114 | -1105 |
| potsdam/potsdam | building | 73.5739 | 73.6551 | 81.1510 | 88.8567 | 17.6538 | 1095 | 1038 |
| potsdam/potsdam | low vegetation | 11.7166 | 11.6686 | 48.6465 | 13.3079 | 4.7438 | -3589 | -28932 |
| potsdam/potsdam | tree | 30.4584 | 36.2414 | 97.8930 | 36.5263 | 6.5833 | 21781 | 706 |
| potsdam/potsdam | car | 28.3934 | 28.6836 | 28.7797 | 98.8497 | 12.3156 | -51 | -2828 |
| potsdam/potsdam | clutter | 3.1425 | 3.0961 | 3.6746 | 16.4340 | 19.8152 | 76 | 9695 |
| udd5/udd5 | vegetation | 52.7726 | 51.7035 | 92.5892 | 53.9355 | 1.7984 | -977 | -483 |
| udd5/udd5 | building | 86.0474 | 85.9687 | 86.3465 | 99.4936 | 82.8768 | 192 | 1820 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0827 | 0 | -216 |
| udd5/udd5 | vehicle | 2.7632 | 2.7668 | 2.7668 | 99.9773 | 15.1763 | 0 | -411 |
| udd5/udd5 | other | 0.0026 | 0.0026 | 0.7977 | 0.0026 | 0.0658 | 0 | 75 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.7907 | 0 | 1932 |
| oem/oem | rangeland | 49.6025 | 49.5329 | 70.3042 | 62.6381 | 12.9600 | -266 | -9 |
| oem/oem | developed space | 31.3856 | 31.0376 | 51.2865 | 44.0127 | 16.8182 | -1629 | 1103 |
| oem/oem | road | 56.3870 | 56.4152 | 72.5690 | 71.7066 | 6.2125 | 173 | 225 |
| oem/oem | tree | 54.2276 | 54.3175 | 93.1172 | 56.5895 | 16.0552 | 656 | 278 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0980 | 0 | -15 |
| oem/oem | agriculture land | 81.2509 | 81.2098 | 81.4929 | 99.5742 | 25.9205 | 10 | 281 |
| oem/oem | building | 52.8353 | 52.7542 | 61.6987 | 78.4433 | 15.1450 | -1139 | -1600 |
| loveda/P | building | 45.6790 | 45.5665 | 65.1408 | 60.2606 | 0.0245 | 0 | 1 |
| loveda/P | road | 56.0427 | 56.2325 | 57.2750 | 96.8649 | 14.0747 | 6 | -552 |
| loveda/P | water | 59.7575 | 60.0212 | 64.3692 | 89.8842 | 20.5472 | 127 | -914 |
| loveda/P | barren | 2.0231 | 1.9668 | 99.0980 | 1.9671 | 0.2201 | -75 | -130 |
| loveda/P | tree | 51.2684 | 50.7433 | 98.0371 | 51.2641 | 8.6095 | -1011 | 2 |
| loveda/P | farm | 86.9556 | 86.6277 | 86.9719 | 99.5452 | 56.5239 | 31 | 2515 |
| loveda/D | background | 4.6604 | 6.0897 | 66.6297 | 6.2813 | 4.2189 | 14120 | 6084 |
| loveda/D | building | 41.3989 | 39.1969 | 48.6936 | 66.7752 | 0.0201 | -14 | -6 |
| loveda/D | road | 46.2323 | 46.6828 | 47.4052 | 96.8389 | 9.3924 | 1 | -1947 |
| loveda/D | water | 51.1886 | 51.3327 | 54.5001 | 89.8297 | 13.3993 | 196 | -457 |
| loveda/D | barren | 1.1589 | 1.1550 | 98.4085 | 1.1552 | 0.0719 | -5 | -4 |
| loveda/D | tree | 36.9096 | 35.6604 | 76.6444 | 40.0079 | 4.7482 | -3417 | -2014 |
| loveda/D | farm | 39.4366 | 39.7828 | 39.8546 | 99.5491 | 68.1492 | 9 | -12546 |
| vaihingen/vaihingen | impervious surface | 60.9312 | 60.6511 | 75.2916 | 75.7228 | 27.4731 | -646 | 2228 |
| vaihingen/vaihingen | building | 72.7923 | 72.2063 | 72.5972 | 99.2599 | 28.2076 | 745 | 5811 |
| vaihingen/vaihingen | low vegetation | 50.6784 | 49.0688 | 91.4445 | 51.4299 | 16.6296 | -20254 | -19323 |
| vaihingen/vaihingen | tree | 66.6986 | 67.5510 | 85.5327 | 76.2649 | 18.4023 | 14012 | 14763 |
| vaihingen/vaihingen | car | 19.9075 | 19.6364 | 19.6759 | 98.9875 | 9.2874 | 1 | 2663 |
| landcoverai/landcoverai | background | 88.2090 | 88.0876 | 94.2266 | 93.1131 | 66.9592 | -1812 | 13 |
| landcoverai/landcoverai | building | 53.3027 | 51.4964 | 52.0991 | 97.8032 | 2.8441 | -1 | 2043 |
| landcoverai/landcoverai | woodland | 79.3715 | 79.3609 | 94.4785 | 83.2207 | 18.8560 | 17 | 84 |
| landcoverai/landcoverai | water | 97.6695 | 97.7157 | 97.7157 | 100.0000 | 8.4647 | 0 | -84 |
| landcoverai/landcoverai | road | 25.2383 | 25.3325 | 27.5695 | 75.7402 | 2.8759 | -3 | -257 |
| flair1/flair1 | building | 53.4537 | 52.5719 | 52.8514 | 99.0040 | 13.4491 | -41 | 4599 |
| flair1/flair1 | pervious surface | 46.2753 | 47.2419 | 89.7169 | 49.9463 | 9.5530 | 1123 | -5517 |
| flair1/flair1 | impervious surface | 52.3931 | 52.2918 | 62.0629 | 76.8595 | 20.2882 | -392 | 227 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.1333 | 0 | -1600 |
| flair1/flair1 | water | 65.1210 | 65.0925 | 67.7976 | 94.2242 | 6.1651 | -28 | 16 |
| flair1/flair1 | coniferous | 34.3662 | 36.4409 | 49.3245 | 58.2485 | 0.6638 | -1 | -1140 |
| flair1/flair1 | deciduous | 56.2366 | 56.3026 | 80.2077 | 65.3871 | 13.8764 | 537 | 468 |
| flair1/flair1 | brushwood | 18.6834 | 18.6195 | 31.4103 | 31.3771 | 4.9132 | 42 | 819 |
| flair1/flair1 | vineyard | -- | -- | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| flair1/flair1 | herbaceous vegetation | 61.2383 | 61.3028 | 89.2727 | 66.1777 | 23.7476 | 598 | 212 |
| flair1/flair1 | agricultural land | 13.6196 | 13.6235 | 15.0576 | 58.8548 | 1.1918 | 0 | -8 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0183 | 0 | 86 |

## paraphrase

| Dataset/protocol | NoAdmission_Exact | CoherentNoHoldout_Exact | SemanticRoleJoint_Exact | SemanticRoleVisualSupported_Exact | SemanticReferenceOnly_Exact | SemanticReferenceJoint_Exact | SemanticReferenceCosine_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 54.530664 | 55.415298 | 55.415298 | 55.415298 | 55.415298 | 55.415298 | 55.401869 |
| potsdam/potsdam | 41.663326 | 42.320156 | 42.320156 | 42.320156 | 42.320156 | 42.320156 | 42.372603 |
| udd5/udd5 | 31.169271 | 33.568498 | 33.568498 | 33.568498 | 33.568498 | 33.568498 | 33.297857 |
| oem/oem | 39.619434 | 39.987607 | 39.987607 | 39.987607 | 39.987607 | 39.987607 | 39.989834 |
| loveda/P | 48.735654 | 50.283629 | 50.283629 | 50.283629 | 50.283629 | 50.283629 | 50.263914 |
| loveda/D | 29.513994 | 32.061199 | 32.061199 | 32.061199 | 32.061199 | 32.061199 | 31.885494 |
| vaihingen/vaihingen | 52.518481 | 53.527369 | 53.527369 | 53.527369 | 53.527369 | 53.527369 | 53.521288 |
| landcoverai/landcoverai | 65.629553 | 65.844126 | 65.844126 | 65.844126 | 65.844126 | 65.844126 | 65.868127 |
| flair1/flair1 | 34.530135 | 34.534747 | 34.511923 | 34.529446 | 34.533487 | 34.528990 | 34.467147 |
| Domain mean | 43.646857 | 44.657375 | 44.654522 | 44.656712 | 44.657218 | 44.656655 | 44.600527 |

| Prospective control | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 44.631331 |
| CoherentNoHoldout_Exact | 44.657375 |
| ClassAttachmentJoint_Exact | 44.600696 |
| ClassAttachmentTextOnly_Exact | 43.415082 |
| SemanticRoleJoint_Exact | 44.654522 |
| SemanticRoleVisualSupported_Exact | 44.656712 |
| SemanticReferenceOnly_Exact | 44.657218 |
| SemanticReferenceNoView_Exact | 44.495696 |
| SemanticReferenceAttachmentOnly_Exact | 43.645265 |
| SemanticReferenceCosine_Exact | 44.600527 |
| SemanticReferenceBinary_Exact | 44.656647 |
| SemanticReferenceMeanLogit | 44.090214 |
| SemanticReferenceShuffle0_Exact | 44.646936 |
| SemanticReferenceShuffle1_Exact | 44.658450 |
| SemanticReferenceShuffle2_Exact | 44.656305 |
| SemanticReferenceJoint_Exact | 44.656655 |

| Dataset/protocol | Class | Old class/view IoU | Primary IoU | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 60.0975 | 60.0975 | 83.1936 | 68.4019 | 41.9780 | 0 | 0 |
| vdd/vdd | wall | 61.6849 | 61.6849 | 66.0438 | 90.3346 | 10.8811 | 0 | 0 |
| vdd/vdd | road | 22.9958 | 22.9958 | 23.1075 | 97.9412 | 14.0677 | 0 | 0 |
| vdd/vdd | vegetation | 52.5498 | 52.5498 | 92.0290 | 55.0557 | 13.7805 | 0 | 0 |
| vdd/vdd | vehicle | 39.6327 | 39.6327 | 39.6327 | 100.0000 | 0.9036 | 0 | 0 |
| vdd/vdd | roof | 93.6614 | 93.6614 | 94.1030 | 99.5014 | 6.3711 | 0 | 0 |
| vdd/vdd | water | 57.2849 | 57.2849 | 61.4280 | 89.4665 | 12.0180 | 0 | 0 |
| potsdam/potsdam | impervious surface | 61.1804 | 61.1804 | 85.5299 | 68.2441 | 32.6152 | 0 | 0 |
| potsdam/potsdam | building | 72.1481 | 72.1481 | 76.4006 | 92.8378 | 19.5916 | 0 | 0 |
| potsdam/potsdam | low vegetation | 37.1875 | 37.1875 | 75.0770 | 42.4249 | 9.7991 | 0 | 0 |
| potsdam/potsdam | tree | 59.4383 | 59.4383 | 93.6477 | 61.9354 | 11.6689 | 0 | 0 |
| potsdam/potsdam | car | 20.7759 | 20.7759 | 20.7831 | 99.8338 | 17.2239 | 0 | 0 |
| potsdam/potsdam | clutter | 3.1908 | 3.1908 | 4.5974 | 9.4439 | 9.1013 | 0 | 0 |
| udd5/udd5 | vegetation | 59.3116 | 59.3116 | 88.1697 | 64.4399 | 2.2563 | 0 | 0 |
| udd5/udd5 | building | 84.2639 | 84.2639 | 84.5468 | 99.6045 | 84.7352 | 0 | 0 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1014 | 0 | 0 |
| udd5/udd5 | vehicle | 8.5409 | 8.5409 | 8.5411 | 99.9773 | 4.9163 | 0 | 0 |
| udd5/udd5 | other | 15.7260 | 15.7260 | 48.1415 | 18.9334 | 7.9908 | 0 | 0 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.0229 | 0 | 0 |
| oem/oem | rangeland | 51.1980 | 51.1980 | 70.7027 | 64.9845 | 13.3698 | 0 | 0 |
| oem/oem | developed space | 17.6025 | 17.6025 | 35.5078 | 25.8750 | 14.2810 | 0 | 0 |
| oem/oem | road | 55.7466 | 55.7466 | 70.9532 | 72.2307 | 6.4004 | 0 | 0 |
| oem/oem | tree | 57.3904 | 57.3904 | 91.3326 | 60.6962 | 17.5568 | 0 | 0 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0 | 0 |
| oem/oem | agriculture land | 82.3568 | 82.3568 | 82.7697 | 99.3978 | 25.4754 | 0 | 0 |
| oem/oem | building | 55.6066 | 55.6066 | 71.5265 | 71.4151 | 11.8936 | 0 | 0 |
| loveda/P | building | 62.0202 | 62.0202 | 62.0202 | 100.0000 | 0.0427 | 0 | 0 |
| loveda/P | road | 52.8228 | 52.8228 | 53.7429 | 96.8607 | 14.9991 | 0 | 0 |
| loveda/P | water | 62.9600 | 62.9600 | 68.8650 | 88.0131 | 18.8060 | 0 | 0 |
| loveda/P | barren | 2.1254 | 2.1254 | 91.1392 | 2.1298 | 0.2591 | 0 | 0 |
| loveda/P | tree | 39.1728 | 39.1728 | 99.1092 | 39.3112 | 6.5307 | 0 | 0 |
| loveda/P | farm | 82.6006 | 82.6006 | 82.8679 | 99.6110 | 59.3624 | 0 | 0 |
| loveda/D | background | 11.4117 | 11.4117 | 80.7259 | 11.7313 | 6.5036 | 0 | 0 |
| loveda/D | building | 37.8079 | 37.8079 | 37.8079 | 100.0000 | 0.0387 | 0 | 0 |
| loveda/D | road | 42.7235 | 42.7235 | 43.3269 | 96.8431 | 10.2769 | 0 | 0 |
| loveda/D | water | 54.8153 | 54.8153 | 59.2595 | 87.9650 | 12.0673 | 0 | 0 |
| loveda/D | barren | 1.6008 | 1.6008 | 77.5526 | 1.6083 | 0.1270 | 0 | 0 |
| loveda/D | tree | 35.9225 | 35.9225 | 97.0016 | 36.3258 | 3.4065 | 0 | 0 |
| loveda/D | farm | 40.1468 | 40.1468 | 40.2114 | 99.6013 | 67.5800 | 0 | 0 |
| vaihingen/vaihingen | impervious surface | 58.9108 | 58.9108 | 78.5487 | 70.2057 | 24.4153 | 0 | 0 |
| vaihingen/vaihingen | building | 67.1297 | 67.1297 | 67.2610 | 99.7102 | 30.5836 | 0 | 0 |
| vaihingen/vaihingen | low vegetation | 49.4995 | 49.4995 | 93.9695 | 51.1235 | 16.0863 | 0 | 0 |
| vaihingen/vaihingen | tree | 67.4567 | 67.4567 | 78.7242 | 82.4963 | 21.6275 | 0 | 0 |
| vaihingen/vaihingen | car | 24.6402 | 24.6402 | 24.7770 | 97.8071 | 7.2874 | 0 | 0 |
| landcoverai/landcoverai | background | 86.8935 | 86.8935 | 94.4167 | 91.6003 | 65.7388 | 0 | 0 |
| landcoverai/landcoverai | building | 39.8556 | 39.8556 | 40.0416 | 98.8481 | 3.7401 | 0 | 0 |
| landcoverai/landcoverai | woodland | 79.7288 | 79.7288 | 93.9801 | 84.0197 | 19.1380 | 0 | 0 |
| landcoverai/landcoverai | water | 97.5646 | 97.5646 | 97.5646 | 100.0000 | 8.4778 | 0 | 0 |
| landcoverai/landcoverai | road | 25.1782 | 25.1782 | 27.3614 | 75.9360 | 2.9053 | 0 | 0 |
| flair1/flair1 | building | 55.9603 | 55.9599 | 57.3401 | 95.8759 | 12.0046 | 0 | 2 |
| flair1/flair1 | pervious surface | 48.7195 | 48.6500 | 95.1216 | 49.8949 | 9.0010 | -288 | -65 |
| flair1/flair1 | impervious surface | 53.1004 | 53.0998 | 60.9935 | 80.4035 | 21.5958 | -17 | -27 |
| flair1/flair1 | bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0614 | 0 | 388 |
| flair1/flair1 | water | 65.2139 | 65.2139 | 67.5086 | 95.0458 | 6.2455 | 0 | 0 |
| flair1/flair1 | coniferous | 43.7060 | 43.7060 | 64.6570 | 57.4253 | 0.4993 | 0 | 0 |
| flair1/flair1 | deciduous | 62.4799 | 62.4802 | 77.4744 | 76.3500 | 16.7746 | 4 | 4 |
| flair1/flair1 | brushwood | 13.9205 | 13.9220 | 33.5917 | 19.2088 | 2.8125 | 2 | -1 |
| flair1/flair1 | vineyard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0114 | 0 | 0 |
| flair1/flair1 | herbaceous vegetation | 58.5074 | 58.5069 | 90.6617 | 62.2588 | 21.9990 | -3 | 1 |
| flair1/flair1 | agricultural land | 12.8092 | 12.8092 | 13.7893 | 64.3148 | 1.4222 | 0 | 0 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5728 | 0 | 0 |

## llm_style

| Dataset/protocol | NoAdmission_Exact | CoherentNoHoldout_Exact | SemanticRoleJoint_Exact | SemanticRoleVisualSupported_Exact | SemanticReferenceOnly_Exact | SemanticReferenceJoint_Exact | SemanticReferenceCosine_Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | 31.837398 | 33.251536 | 33.251536 | 33.251536 | 33.251536 | 33.251536 | 33.280775 |
| oem/oem | 40.439688 | 40.387403 | 40.387403 | 40.387403 | 40.387403 | 40.387403 | 40.366842 |
| loveda/P | 50.706552 | 53.513897 | 53.513897 | 53.513897 | 53.513897 | 53.513897 | 53.496902 |
| loveda/D | 30.736030 | 33.944372 | 33.944372 | 33.944372 | 33.944372 | 33.944372 | 33.829905 |
| Domain mean | 34.337705 | 35.861104 | 35.861104 | 35.861104 | 35.861104 | 35.861104 | 35.825841 |

| Prospective control | Mean mIoU |
| --- | ---: |
| CoherentNativeJoint_Exact | 35.903150 |
| CoherentNoHoldout_Exact | 35.861104 |
| ClassAttachmentJoint_Exact | 35.825841 |
| ClassAttachmentTextOnly_Exact | 34.724253 |
| SemanticRoleJoint_Exact | 35.861104 |
| SemanticRoleVisualSupported_Exact | 35.861104 |
| SemanticReferenceOnly_Exact | 35.861104 |
| SemanticReferenceNoView_Exact | 35.653085 |
| SemanticReferenceAttachmentOnly_Exact | 34.337705 |
| SemanticReferenceCosine_Exact | 35.825841 |
| SemanticReferenceBinary_Exact | 35.861104 |
| SemanticReferenceMeanLogit | 35.571622 |
| SemanticReferenceShuffle0_Exact | 35.861104 |
| SemanticReferenceShuffle1_Exact | 35.861104 |
| SemanticReferenceShuffle2_Exact | 35.861104 |
| SemanticReferenceJoint_Exact | 35.861104 |

| Dataset/protocol | Class | Old class/view IoU | Primary IoU | Precision % | Recall % | Area % | TP change | FP change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | vegetation | 61.1230 | 61.1230 | 85.7588 | 68.0279 | 2.4489 | 0 | 0 |
| udd5/udd5 | building | 75.3211 | 75.3211 | 75.3211 | 100.0000 | 95.4916 | 0 | 0 |
| udd5/udd5 | road | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3322 | 0 | 0 |
| udd5/udd5 | vehicle | 28.5114 | 28.5114 | 28.5644 | 99.3529 | 1.4608 | 0 | 0 |
| udd5/udd5 | other | 1.3022 | 1.3022 | 99.3376 | 1.3023 | 0.2664 | 0 | 0 |
| oem/oem | bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.3631 | 0 | 0 |
| oem/oem | rangeland | 51.9283 | 51.9283 | 71.0315 | 65.8803 | 13.4913 | 0 | 0 |
| oem/oem | developed space | 27.4352 | 27.4352 | 46.3235 | 40.2216 | 17.0162 | 0 | 0 |
| oem/oem | road | 55.2935 | 55.2935 | 69.3030 | 73.2283 | 6.6433 | 0 | 0 |
| oem/oem | tree | 56.9907 | 56.9907 | 91.1160 | 60.3438 | 17.4963 | 0 | 0 |
| oem/oem | water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 0 | 0 |
| oem/oem | agriculture land | 80.4793 | 80.4793 | 80.8425 | 99.4449 | 26.0951 | 0 | 0 |
| oem/oem | building | 50.9722 | 50.9722 | 67.5761 | 67.4745 | 11.8942 | 0 | 0 |
| loveda/P | building | 76.1773 | 76.1773 | 83.5866 | 89.5765 | 0.0284 | 0 | 0 |
| loveda/P | road | 55.8436 | 55.8436 | 56.9027 | 96.7746 | 14.1536 | 0 | 0 |
| loveda/P | water | 62.7439 | 62.7439 | 67.9932 | 89.0437 | 19.2701 | 0 | 0 |
| loveda/P | barren | 2.3564 | 2.3564 | 91.1659 | 2.3618 | 0.2872 | 0 | 0 |
| loveda/P | tree | 41.3957 | 41.3957 | 99.2570 | 41.5244 | 6.8881 | 0 | 0 |
| loveda/P | farm | 82.5665 | 82.5665 | 82.8428 | 99.5977 | 59.3725 | 0 | 0 |
| loveda/D | background | 14.4463 | 14.4463 | 76.3082 | 15.1247 | 8.8702 | 0 | 0 |
| loveda/D | building | 46.3744 | 46.3744 | 49.0196 | 89.5765 | 0.0268 | 0 | 0 |
| loveda/D | road | 46.8793 | 46.8793 | 47.6321 | 96.7383 | 9.3379 | 0 | 0 |
| loveda/D | water | 54.5619 | 54.5619 | 58.5144 | 88.9838 | 12.3625 | 0 | 0 |
| loveda/D | barren | 1.6702 | 1.6702 | 81.0689 | 1.6768 | 0.1267 | 0 | 0 |
| loveda/D | tree | 32.7068 | 32.7068 | 97.7244 | 32.9578 | 3.0678 | 0 | 0 |
| loveda/D | farm | 40.9717 | 40.9717 | 41.0406 | 99.5917 | 66.2081 | 0 | 0 |

## Cost And Interpretation

Suite wall 789.6501s.

| Dataset | Multi-arm worker seconds | Combined peak allocated MiB |
| --- | ---: | ---: |
| vdd | 173.2133 | 5625.9492 |
| potsdam | 219.6379 | 5613.8057 |
| udd5 | 182.1141 | 5621.4409 |
| oem | 351.0691 | 5684.2202 |
| loveda | 613.8184 | 5705.0132 |
| vaihingen | 205.1282 | 5602.3564 |
| landcoverai | 196.7644 | 5602.3564 |
| flair1 | 344.5128 | 5961.9160 |

These share79 arms/all regimes and cached reference fields; not standalone deployed latency. Empty class mass is unknown; no-semantic sources recover the old base; held families cannot self-confirm. Class/view and extra attachment remain distinct in the increments above. The semantic source is not truth; its previous errors are preserved. No numerical label fitting, dataset routing or post-result seed choice. Historical style is not verified raw LLM output. Geometry/coupling and frozen auxiliary language source retain their attribution; ordinary weighted aggregation/least squares alone are not claimed as novel.
