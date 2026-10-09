# Pool-Isolated Alias Witness: Frozen Eight-Domain Pilot

Same96 developed complete inputs, unchanged20 prediction words, Geometry reader/original H and positive wide/fine information. Only the negative witness uses unprofiled raw alias-minus-rival canonical margins. Canonical is a reference, not semantic ground truth. At most4 Geometry/4 wide/16 bounded fine encodings; no native sliding windows or extra head.

| Domain/protocol | PoolIsolated_Canonical | PoolIsolated_ObservationMean | PoolIsolated_ProfilePool | PoolIsolated_ProfileCanonical | PoolIsolated_RawPool | PoolIsolated_FineOnly | PoolIsolated_WideHard | PoolIsolated_ClassMean | PoolIsolated_AliasShuffle0 | PoolIsolated_AliasShuffle1 | PoolIsolated_AliasShuffle2 | PoolIsolated_ShuffledWrite | Primary-ProfilePool pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 49.8310 | 49.5849 | 49.8732 | 49.8086 | 49.8807 | 49.7966 | 49.7970 | 49.8315 | 49.8167 | 49.8262 | 49.7915 | 49.6838 | -0.0422 |
| potsdam/potsdam | 55.0010 | 54.8216 | 55.1778 | 54.9918 | 55.1868 | 55.0371 | 54.9119 | 55.0824 | 55.1004 | 55.0723 | 55.1298 | 54.9237 | -0.1768 |
| udd5/udd5 | 51.1917 | 50.7777 | 51.2305 | 51.1812 | 51.2391 | 51.1396 | 51.2136 | 51.1588 | 51.0972 | 51.1019 | 51.1931 | 50.7708 | -0.0388 |
| oem/oem | 30.0191 | 29.5309 | 29.9714 | 29.9952 | 29.9857 | 29.7729 | 29.8628 | 30.0933 | 30.1189 | 30.0878 | 30.0741 | 29.7452 | +0.0477 |
| loveda/P | 75.3408 | 75.2590 | 75.5356 | 75.3093 | 75.5503 | 75.3727 | 75.2137 | 75.4178 | 75.4383 | 75.4774 | 75.3181 | 75.3625 | -0.1948 |
| loveda/D | 47.1917 | 45.5946 | 46.6718 | 47.1440 | 46.7310 | 46.5198 | 47.0139 | 47.2175 | 47.2661 | 47.2397 | 47.2936 | 46.9628 | +0.5199 |
| vaihingen/vaihingen | 51.5557 | 51.5130 | 51.8593 | 51.5457 | 51.8576 | 51.2941 | 51.9034 | 51.6279 | 51.6116 | 51.6562 | 51.6598 | 51.2926 | -0.3036 |
| landcoverai/landcoverai | 77.9936 | 77.9179 | 78.0381 | 77.9966 | 78.0417 | 77.8788 | 77.9643 | 78.0271 | 78.0087 | 78.0385 | 78.0250 | 78.1407 | -0.0445 |
| flair1/flair1 | 38.5809 | 38.7829 | 38.9230 | 38.5865 | 38.9278 | 38.6483 | 38.6442 | 38.8158 | 38.8043 | 38.7936 | 38.8449 | 38.5333 | -0.3421 |

Eight-domain means, LoveDA D once:

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| PoolIsolated_Canonical | 50.1706 |
| PoolIsolated_ObservationMean | 49.8154 |
| PoolIsolated_ProfilePool | 50.2181 |
| PoolIsolated_ProfileCanonical | 50.1562 |
| PoolIsolated_RawPool | 50.2313 |
| PoolIsolated_FineOnly | 50.0109 |
| PoolIsolated_WideHard | 50.1639 |
| PoolIsolated_ClassMean | 50.2318 |
| PoolIsolated_AliasShuffle0 | 50.2280 |
| PoolIsolated_AliasShuffle1 | 50.2270 |
| PoolIsolated_AliasShuffle2 | 50.2515 |
| PoolIsolated_ShuffledWrite | 50.0066 |

## Actual Complete-Image Warm Cost

| Domain | Bounded baseline ms | Primary ms | ProfilePool ms | VIP20 ms |
| --- | ---: | ---: | ---: | ---: |
| vdd | 325.76 | 761.85 | 730.44 | 125.25 |
| potsdam | 239.13 | 715.07 | 682.71 | 102.67 |
| udd5 | 289.92 | 640.82 | 619.62 | 116.43 |
| oem | 239.34 | 722.96 | 683.31 | 104.08 |
| loveda | 247.40 | 895.76 | 830.34 | 199.92 |
| vaihingen | 231.41 | 694.04 | 666.19 | 98.51 |
| landcoverai | 225.59 | 687.04 | 659.10 | 95.71 |
| flair1 | 230.97 | 748.72 | 688.46 | 100.88 |

Three fixed full inputs/domain, three warmed synchronized singleton repeats; serial idle GPU. Graph setup is recorded separately. Shared peaks are not standalone memory; warm timings are not full throughput.

13 focused CPU tests and26 regression tests passed remotely with real dependencies. Both mask-free smokes and all96 per-image original/positive/hard/soft endpoint replays verify, as do ordered unique coverage, scored targets, vocabulary/checkpoints and independent mIoU.

## Frozen Advancement Checks

```json
{
  "means": {
    "Geometry": 44.683575,
    "NoAdmission_Exact": 46.3359625,
    "Geometry_PatchOnly2Coupled": 46.645025000000004,
    "PoolIsolated_Canonical": 50.170587499999996,
    "PoolIsolated_ObservationMean": 49.8154375,
    "PoolIsolated_ProfilePool": 50.2181375,
    "PoolIsolated_ProfileCanonical": 50.1562,
    "PoolIsolated_RawPool": 50.231300000000005,
    "PoolIsolated_FineOnly": 50.01089999999999,
    "PoolIsolated_WideHard": 50.1638875,
    "PoolIsolated_ClassMean": 50.2317875,
    "PoolIsolated_AliasShuffle0": 50.2279875,
    "PoolIsolated_AliasShuffle1": 50.227025,
    "PoolIsolated_AliasShuffle2": 50.251475,
    "PoolIsolated_ShuffledWrite": 50.006612499999996
  },
  "speed_ms": {
    "vdd": {
      "Geometry_PatchOnly2Coupled": 325.7616050541401,
      "PoolIsolated_Canonical": 761.8465255945921,
      "PoolIsolated_ProfilePool": 730.4370279889554,
      "VIP_All20": 125.24503407378992
    },
    "potsdam": {
      "Geometry_PatchOnly2Coupled": 239.13076861451069,
      "PoolIsolated_Canonical": 715.0680630778273,
      "PoolIsolated_ProfilePool": 682.7071615650008,
      "VIP_All20": 102.67195656585197
    },
    "udd5": {
      "Geometry_PatchOnly2Coupled": 289.9218046416839,
      "PoolIsolated_Canonical": 640.8239079173654,
      "PoolIsolated_ProfilePool": 619.6221842740973,
      "VIP_All20": 116.43478406282763
    },
    "oem": {
      "Geometry_PatchOnly2Coupled": 239.34047296643257,
      "PoolIsolated_Canonical": 722.9604842141271,
      "PoolIsolated_ProfilePool": 683.3084600511938,
      "VIP_All20": 104.07878062687814
    },
    "loveda": {
      "Geometry_PatchOnly2Coupled": 247.39773998347422,
      "PoolIsolated_Canonical": 895.7564023633798,
      "PoolIsolated_ProfilePool": 830.3445840720087,
      "VIP_All20": 199.91596803689995
    },
    "vaihingen": {
      "Geometry_PatchOnly2Coupled": 231.41180607490242,
      "PoolIsolated_Canonical": 694.0441387705505,
      "PoolIsolated_ProfilePool": 666.1881243344396,
      "VIP_All20": 98.50843506865203
    },
    "landcoverai": {
      "Geometry_PatchOnly2Coupled": 225.58882165079316,
      "PoolIsolated_Canonical": 687.0412437710911,
      "PoolIsolated_ProfilePool": 659.101797034964,
      "VIP_All20": 95.71478158856432
    },
    "flair1": {
      "Geometry_PatchOnly2Coupled": 230.97103422818086,
      "PoolIsolated_Canonical": 748.7206923154494,
      "PoolIsolated_ProfilePool": 688.4630294516683,
      "VIP_All20": 100.88379502606888
    }
  },
  "graph_setup": {
    "vdd": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.46038546692579985
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "potsdam": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.4721154992002994
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "udd5": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.46193797001615167
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "oem": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.469854157185182
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "loveda": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.4628667258657515
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "vaihingen": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.4588897309731692
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "landcoverai": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.46249216399155557
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    },
    "flair1": {
      "implementation": "bounded-fine-coverage-exact-execution-v1-20261006",
      "cached": true,
      "burst": true,
      "graph_setup_seconds": {
        "4": 0.500508019933477
      },
      "graph_setup_visual_forwards": 16,
      "inference_visual_budget_unchanged": 16
    }
  },
  "shared_peak_mib": {
    "vdd": 5865.83544921875,
    "potsdam": 5768.1572265625,
    "udd5": 5743.49853515625,
    "oem": 5807.4892578125,
    "loveda": 5833.74365234375,
    "vaihingen": 5753.95458984375,
    "landcoverai": 5753.95458984375,
    "flair1": 5926.8251953125
  },
  "unique_images": 96,
  "domain_wins": 2,
  "exact_original_positive_hard_soft_replay": true,
  "component_deltas_pp": {
    "reference": -0.06071250000000816,
    "salience": 0.014387499999997999,
    "write_correspondence": 0.16397500000000065
  },
  "accuracy_mechanism_passed": false,
  "gate": {
    "passed": false,
    "checks": {
      "mean_gain": false,
      "domain_wins": false,
      "worst_protocol_loss": true,
      "word_gain": true,
      "above_class_mean": false,
      "above_identity_null": false,
      "reference_factor": false,
      "salience_factor": false,
      "domain_mean_below1000ms": true,
      "mean_ratio_to_profile_pool": true
    }
  }
}
```

No post-result primary/threshold/domain route or favorable-control promotion. Developed inputs, not independent validation. Failure rejects this implementation, not every isolated witness.

## Correct Coverage And Competitor Activation

| Domain/protocol | Class | ProfilePool IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 21.0681 | 21.0246 | +7742 | +95292 |
| vdd/vdd | wall | 55.6308 | 55.6969 | +6529 | +6294 |
| vdd/vdd | road | 16.5334 | 16.5448 | -1730 | -16722 |
| vdd/vdd | vegetation | 48.9341 | 48.5999 | -97027 | -5553 |
| vdd/vdd | vehicle | 26.8570 | 26.9270 | -354 | -3880 |
| vdd/vdd | roof | 87.5307 | 87.5141 | -5581 | -568 |
| vdd/vdd | water | 92.5582 | 92.5096 | +2330 | +13228 |
| potsdam/potsdam | impervious surface | 77.3746 | 77.1010 | -266 | +12521 |
| potsdam/potsdam | building | 83.9116 | 84.0436 | +2115 | +487 |
| potsdam/potsdam | low vegetation | 65.3738 | 65.0587 | -6854 | +691 |
| potsdam/potsdam | tree | 62.6535 | 62.3415 | -3277 | +1922 |
| potsdam/potsdam | car | 33.1180 | 33.0847 | -97 | +215 |
| potsdam/potsdam | clutter | 8.6353 | 8.3767 | -1351 | -6106 |
| udd5/udd5 | vegetation | 73.1690 | 73.0027 | -256861 | -46058 |
| udd5/udd5 | building | 83.7786 | 83.8113 | -79237 | -173906 |
| udd5/udd5 | road | 44.7628 | 44.7353 | -138320 | -264051 |
| udd5/udd5 | vehicle | 19.4049 | 19.1745 | -786 | +180425 |
| udd5/udd5 | other | 35.0373 | 35.2345 | +364221 | +414573 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +9719 |
| oem/oem | rangeland | 6.8975 | 6.6217 | -3849 | -993 |
| oem/oem | developed space | 22.7224 | 22.6229 | -10880 | -33379 |
| oem/oem | road | 29.6258 | 29.6718 | +425 | +346 |
| oem/oem | tree | 27.0871 | 27.2026 | +1941 | +2172 |
| oem/oem | water | 17.0185 | 17.0527 | +37 | -54 |
| oem/oem | agriculture land | 72.9856 | 73.1941 | -802 | -3848 |
| oem/oem | building | 63.4347 | 63.7870 | +22636 | +16529 |
| loveda/P | building | 89.6315 | 89.5126 | +133 | +626 |
| loveda/P | road | 85.1967 | 84.9928 | -1080 | +931 |
| loveda/P | water | 72.6192 | 72.5975 | -260 | +66 |
| loveda/P | barren | 68.0139 | 67.5490 | -190 | +1219 |
| loveda/P | tree | 64.6523 | 64.3435 | -2665 | +1225 |
| loveda/P | farm | 73.1002 | 73.0495 | -718 | +713 |
| loveda/D | background | 14.6919 | 17.5911 | +87169 | +31366 |
| loveda/D | building | 48.4265 | 48.2009 | +244 | +3608 |
| loveda/D | road | 65.9654 | 65.8814 | -1271 | -424 |
| loveda/D | water | 67.7771 | 67.1909 | -7729 | +1639 |
| loveda/D | barren | 26.2285 | 27.8070 | +558 | -29549 |
| loveda/D | tree | 52.2329 | 51.7816 | -10766 | -9255 |
| loveda/D | farm | 51.3805 | 51.8893 | -10924 | -54666 |
| vaihingen/vaihingen | impervious surface | 63.9288 | 63.8306 | -8869 | -9287 |
| vaihingen/vaihingen | building | 70.2291 | 69.6887 | +760 | +23965 |
| vaihingen/vaihingen | low vegetation | 29.7808 | 28.9700 | -12657 | -1245 |
| vaihingen/vaihingen | tree | 68.1148 | 67.9283 | +1044 | +7272 |
| vaihingen/vaihingen | car | 27.2431 | 27.3610 | +213 | -1196 |
| landcoverai/landcoverai | background | 85.1704 | 85.0443 | +394 | +1576 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7243 | 92.6409 | -1440 | -599 |
| landcoverai/landcoverai | water | 93.8996 | 93.8377 | +38 | +275 |
| landcoverai/landcoverai | road | 40.3582 | 40.4513 | -37 | -207 |
| flair1/flair1 | building | 59.6299 | 59.1722 | -126 | +1089 |
| flair1/flair1 | pervious surface | 28.8614 | 28.0451 | -524 | -136 |
| flair1/flair1 | impervious surface | 69.6020 | 69.5002 | -115 | +446 |
| flair1/flair1 | bare soil | 11.4686 | 11.3424 | +2 | +1654 |
| flair1/flair1 | water | 95.6854 | 95.7066 | +10 | -2 |
| flair1/flair1 | coniferous | 2.8284 | 2.0939 | -56 | -132 |
| flair1/flair1 | deciduous | 66.8983 | 66.9224 | -351 | -634 |
| flair1/flair1 | brushwood | 8.3984 | 8.0475 | -708 | -2067 |
| flair1/flair1 | vineyard | 56.9048 | 56.3909 | -2 | +1840 |
| flair1/flair1 | herbaceous vegetation | 35.3889 | 35.3068 | -399 | +390 |
| flair1/flair1 | agricultural land | 31.4099 | 30.4424 | -6095 | -3425 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | +9341 |
