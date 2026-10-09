# Bounded Legacy Fine-Witness Coverage Diagnosis

Same96 developed complete inputs; bounded896 Geometry and independent448 wide, unchanged20 words/checkpoints/G/H. Up to16 quadrant fine encodings of resized896 RGB, NOT native-image physical8. This is source/coverage transfer evidence, not a new final model. Canonical-protected old hard admission and original writer stay unchanged.

| Domain/protocol | Baseline | Hard | Class-only | Identity-null mean | Observation mean | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.8468 | 47.6460 | 46.8812 | 46.7148 | 49.5849 | 0.7992 |
| potsdam/potsdam | 51.5168 | 52.6137 | 51.4909 | 51.6989 | 54.8216 | 1.0969 |
| udd5/udd5 | 47.0786 | 48.4026 | 47.0682 | 46.9423 | 50.7777 | 1.3240 |
| oem/oem | 25.2530 | 26.4794 | 25.2489 | 25.2689 | 29.5309 | 1.2264 |
| loveda/P | 67.2104 | 70.8156 | 67.2995 | 67.4888 | 75.2590 | 3.6052 |
| loveda/D | 39.4705 | 42.3382 | 39.5194 | 39.6204 | 45.5946 | 2.8677 |
| vaihingen/vaihingen | 49.1468 | 50.4579 | 49.0669 | 49.3392 | 51.5130 | 1.3111 |
| landcoverai/landcoverai | 76.2441 | 76.6367 | 76.2652 | 76.3336 | 77.9179 | 0.3926 |
| flair1/flair1 | 37.6036 | 37.3145 | 37.8778 | 37.8467 | 38.7829 | -0.2891 |

Eight-domain means (LoveDA D once):

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| BoundedFineCoverage_Hard | 47.7361 |
| BoundedFineCoverage_ClassMean | 46.6773 |
| BoundedFineCoverage_AliasShuffle0 | 46.7145 |
| BoundedFineCoverage_AliasShuffle1 | 46.7686 |
| BoundedFineCoverage_AliasShuffle2 | 46.6787 |
| BoundedFineCoverage_ObservationMean | 49.8154 |

## Complete-Image Timing

| Domain | Baseline ms | Hard ms | VIP20 ms | Hard/baseline |
| --- | ---: | ---: | ---: | ---: |
| vdd | 324.05 | 992.85 | 125.30 | 3.064x |
| potsdam | 237.79 | 1120.11 | 102.49 | 4.711x |
| udd5 | 291.49 | 843.12 | 116.83 | 2.892x |
| oem | 241.33 | 1135.70 | 104.23 | 4.706x |
| loveda | 249.70 | 1622.53 | 202.12 | 6.498x |
| vaihingen | 235.06 | 1110.05 | 100.34 | 4.722x |
| landcoverai | 228.31 | 1104.17 | 96.34 | 4.836x |
| flair1 | 238.14 | 1156.82 | 103.60 | 4.858x |

Three fixed complete images/domain; three warmed synchronized rotated repeats, serial idle-GPU timing. Includes all16-bounded fine views, alias action, reconstruction, restoration and argmax; excludes model/text initialization and data/mask loading. Both backbones resident; not standalone memory or full-suite throughput.

## Predeclared Decision

```json
{
  "means": {
    "Geometry": 44.683575,
    "NoAdmission_Exact": 46.3359625,
    "Geometry_PatchOnly2Coupled": 46.645025000000004,
    "BoundedFineCoverage_Hard": 47.736125,
    "BoundedFineCoverage_ClassMean": 46.6773125,
    "BoundedFineCoverage_AliasShuffle0": 46.7145,
    "BoundedFineCoverage_AliasShuffle1": 46.7686375,
    "BoundedFineCoverage_AliasShuffle2": 46.678675,
    "BoundedFineCoverage_ObservationMean": 49.8154375
  },
  "speed_ms": {
    "vdd": {
      "Geometry_PatchOnly2Coupled": 324.04689909890294,
      "BoundedFineCoverage_Hard": 992.8522249683738,
      "VIP_All20": 125.3014060202986
    },
    "potsdam": {
      "Geometry_PatchOnly2Coupled": 237.78687797797224,
      "BoundedFineCoverage_Hard": 1120.1115190051496,
      "VIP_All20": 102.48831030912697
    },
    "udd5": {
      "Geometry_PatchOnly2Coupled": 291.4930076804012,
      "BoundedFineCoverage_Hard": 843.1217740289867,
      "VIP_All20": 116.83026600318651
    },
    "oem": {
      "Geometry_PatchOnly2Coupled": 241.3331716476629,
      "BoundedFineCoverage_Hard": 1135.7042810413986,
      "VIP_All20": 104.22597134796281
    },
    "loveda": {
      "Geometry_PatchOnly2Coupled": 249.69777736502388,
      "BoundedFineCoverage_Hard": 1622.5268259489287,
      "VIP_All20": 202.12298996436098
    },
    "vaihingen": {
      "Geometry_PatchOnly2Coupled": 235.0631410566469,
      "BoundedFineCoverage_Hard": 1110.0549530237913,
      "VIP_All20": 100.34229668478172
    },
    "landcoverai": {
      "Geometry_PatchOnly2Coupled": 228.31106232479215,
      "BoundedFineCoverage_Hard": 1104.1680166187386,
      "VIP_All20": 96.3389112924536
    },
    "flair1": {
      "Geometry_PatchOnly2Coupled": 238.14069689251482,
      "BoundedFineCoverage_Hard": 1156.8160412522654,
      "VIP_All20": 103.59885799698532
    }
  },
  "unique_images": 96,
  "domain_wins": 7,
  "original_per_image_endpoints_verified": true,
  "gate": {
    "passed": false,
    "checks": {
      "mean_gain": true,
      "domain_wins": true,
      "worst_protocol_loss": true,
      "above_class_mean": true,
      "above_identity_null": true,
      "above_observation_mean": false,
      "all_timings": true,
      "mean_cost_ratio": false,
      "domain_mean_below1000ms": false
    }
  },
  "mean_gain_pp": 1.0910999999999973,
  "identity_null_mean": 46.72060416666667,
  "mean_cost_ratio": 4.535893208281982
}
```

The cost allowance2 was fixed before results for this diagnosis; old failed1.3 gates remain failed. The method uses no labels, but domains and rules are developed. Neither passing nor a favorable control automatically replaces the retained model, authorizes a full20092 rollout or establishes new novelty. Failure does not isolate coverage, physical scale and reader differences individually.

## Interpretation And Next Decision

The unchanged legacy rule transfers nontrivial alias-specific utility to bounded
complete-image fields:47.7361 versus46.6450 (+1.0911pp),7/8 domain wins,
class-only advantage1.0588pp and mean alias-identity-null advantage1.0155pp.
FLAIR-1 loses0.2891pp. All original three per-image confusions,96 unique keys,
vocabulary/checkpoint identities and every arm's target counts verify. Ten CPU
tests pass on A800 with real dependencies; both mask-free smokes verify exact
legacy fine evidence, original endpoints, singleton primary and frozen heads.
The local pure-function test run used an import-only cv2 stand-in because the
local interpreter lacks cv2; real image/source behavior is checked on A800.

This is not another failed confidence curve and does not show global good/bad
word classification. It also does not isolate coverage versus reader versus
physical-scale differences from earlier sparse/native witness failures.
The fine source reads resized896 quarters; it does not restore original-image
detail that was lost during resizing. The fine VIP proxy is inherited.

Same-fine-information observation mean49.8154 exceeds the hard primary by2.0793pp,
and is higher on every domain and LoveDA P. Thus word-identity utility exists,
but the tested controller is not the best use of the available observations.
The legacy controller reads fine evidence to modify wide alias use; it does not
write the fine observation's positive class innovation directly. The comparison
motivates investigating this distinction, not declaring its causal contribution
or promoting the observation-mean control into a new final candidate.

Matched warmed whole-image primary times843.12-1622.53ms versus228.31-324.05ms
for its paired bounded reference. The mean paired ratio4.5359 fails even the
new predeclared2 allowance; six domain means exceed1000ms. These are three fixed
inputs/domain, not full-suite throughput. Total timing does not apportion the
overhead between additional visual forwards, all-rival alias/stencil work and
writing; do not assign an unsupported percentage to any component.

VDD vehicle gains1.6458pp with3318 additional TP and54679 fewer FP. Potsdam car
gains0.2959pp with381 additional TP and3713 fewer FP; low vegetation gains1.3663pp
with34199 additional TP but4819 additional FP. UDD5 road gains1.4679pp with both
more TP and FP, while vehicle gains1.7501pp chiefly through reduced FP. This is
not a universal precision-only mechanism or an all-class improvement.

Reject final-model promotion/full20092 rollout. Preserve Geometry_PatchOnly2Coupled
and all diagnostic controls. The next necessary work is to reduce the cost of
this actually effective evidence and establish conditional positive/negative
evidence writing with same-information attribution. Do not resume replacing it
with unverified confidence/cosine factors, choose words from these labeled
outcomes, or erase earlier failed gates. The full three-part goal remains open.

All experiment and timing workers exited; the final authoritative GPU check has
no compute processes. Unrelated tmux sessions and paused automations are unchanged.

## Class Coverage And False Activation

| Domain/protocol | Class | Baseline IoU | Hard IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.6232 | 20.3374 | -81154 | +5857 |
| vdd/vdd | wall | 46.3090 | 49.2313 | +116254 | -35041 |
| vdd/vdd | road | 15.2940 | 15.3548 | +40169 | +225731 |
| vdd/vdd | vegetation | 45.8052 | 45.9924 | +39181 | -29701 |
| vdd/vdd | vehicle | 23.8613 | 25.5071 | +3318 | -54679 |
| vdd/vdd | roof | 85.4306 | 85.9370 | +27869 | -152045 |
| vdd/vdd | water | 90.6042 | 91.1619 | +10159 | -115918 |
| potsdam/potsdam | impervious surface | 75.3698 | 76.2524 | +12961 | -25820 |
| potsdam/potsdam | building | 80.4415 | 81.7850 | -595 | -23001 |
| potsdam/potsdam | low vegetation | 60.1752 | 61.5415 | +34199 | +4819 |
| potsdam/potsdam | tree | 57.3938 | 59.6173 | +28843 | -6220 |
| potsdam/potsdam | car | 31.4528 | 31.7487 | +381 | -3713 |
| potsdam/potsdam | clutter | 4.2680 | 4.7371 | +761 | -22615 |
| udd5/udd5 | vegetation | 66.2953 | 68.2330 | +2702505 | +170199 |
| udd5/udd5 | building | 81.2771 | 82.2730 | +83120 | -2433075 |
| udd5/udd5 | road | 37.2416 | 38.7095 | +1226865 | +399976 |
| udd5/udd5 | vehicle | 16.9950 | 18.7451 | +25030 | -1382546 |
| udd5/udd5 | other | 33.5843 | 34.0526 | +215213 | -1007287 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +19575 |
| oem/oem | rangeland | 4.3949 | 4.4009 | +166 | +1935 |
| oem/oem | developed space | 21.7107 | 22.0530 | -37153 | -234125 |
| oem/oem | road | 26.4384 | 27.1126 | +5649 | +3892 |
| oem/oem | tree | 10.8604 | 16.0214 | +59571 | +24327 |
| oem/oem | water | 18.0795 | 18.0049 | +60 | +807 |
| oem/oem | agriculture land | 69.4583 | 69.7606 | -2883 | -8633 |
| oem/oem | building | 51.0820 | 54.4818 | +131029 | +35783 |
| loveda/P | building | 88.2412 | 88.7702 | -148 | -2346 |
| loveda/P | road | 79.8873 | 82.0827 | +2219 | -23312 |
| loveda/P | water | 62.0850 | 67.5811 | +78403 | +1308 |
| loveda/P | barren | 70.3607 | 69.8373 | +698 | +2590 |
| loveda/P | tree | 39.2222 | 49.2252 | +111710 | +1822 |
| loveda/P | farm | 63.4659 | 67.3968 | -2923 | -170021 |
| loveda/D | background | 4.0203 | 7.2570 | +94080 | +55420 |
| loveda/D | building | 45.7347 | 46.4671 | +395 | -10192 |
| loveda/D | road | 63.0492 | 64.8042 | +2523 | -29376 |
| loveda/D | water | 58.9731 | 64.2146 | +80522 | +4201 |
| loveda/D | barren | 27.9609 | 28.1645 | +1534 | +1618 |
| loveda/D | tree | 34.5990 | 40.3646 | +79719 | +29455 |
| loveda/D | farm | 41.9559 | 45.0950 | -3869 | -306030 |
| vaihingen/vaihingen | impervious surface | 60.8940 | 62.6299 | +51935 | -1571 |
| vaihingen/vaihingen | building | 65.1989 | 66.9667 | -334 | -84573 |
| vaihingen/vaihingen | low vegetation | 27.2861 | 27.8583 | +5631 | -11540 |
| vaihingen/vaihingen | tree | 65.8712 | 67.1128 | +38688 | +20908 |
| vaihingen/vaihingen | car | 26.4838 | 27.7218 | +230 | -19374 |
| landcoverai/landcoverai | background | 82.5166 | 83.1464 | +323 | -5472 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 91.5012 | 91.7165 | +4712 | +2669 |
| landcoverai/landcoverai | water | 91.7035 | 92.3645 | -35 | -2641 |
| landcoverai/landcoverai | road | 39.2550 | 39.3195 | +149 | +295 |
| flair1/flair1 | building | 49.1283 | 50.9954 | +150 | -6838 |
| flair1/flair1 | pervious surface | 17.8467 | 19.6286 | +1010 | +41 |
| flair1/flair1 | impervious surface | 66.4519 | 67.0218 | +819 | -2440 |
| flair1/flair1 | bare soil | 21.3471 | 19.2347 | +28 | +8789 |
| flair1/flair1 | water | 94.9931 | 94.9942 | -26 | -28 |
| flair1/flair1 | coniferous | 5.9298 | 5.0282 | -44 | +147 |
| flair1/flair1 | deciduous | 64.3305 | 65.0490 | +4293 | +3459 |
| flair1/flair1 | brushwood | 3.3685 | 3.2967 | -62 | +1814 |
| flair1/flair1 | vineyard | 51.9276 | 51.6435 | +18 | +1258 |
| flair1/flair1 | herbaceous vegetation | 37.8267 | 36.5866 | -5158 | +7619 |
| flair1/flair1 | agricultural land | 38.0928 | 34.2954 | -20238 | -1676 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | +7065 |
