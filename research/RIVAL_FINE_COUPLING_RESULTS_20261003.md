# Physical8 Rival Admission Coupling: Verified Screen

Same64 developed top-left512 windows, eight/domain; not full-image/full-dataset or untouched validation. Frozen weights;20 candidates/class, query/rival-dependent variable retention; local Geometry20 unchanged. Same wide observer and original reconstruction. No fresh observations or mask-based source tuning. LoveDA D counts once, P separately; LandCover.ai substitutes for unlabeled iSAID.

## Frozen Decision

```json
{
  "passed": true,
  "checks": {
    "clean_gain": true,
    "clean_domain_wins": true,
    "clean_protocol_safety": true,
    "wrong_parent_gain": true,
    "above_RivalFineRandom0_Exact": true,
    "above_RivalFineRandom1_Exact": true,
    "above_RivalFineRandom2_Exact": true
  },
  "clean_domain_wins": 8,
  "worst_clean_protocol_delta_pp": 0.45396200630368355,
  "no_automatic_full_rollout": true
}
```

PASS: freeze full-image observation/cost handling before independent validation.

## Alias And Coupling Effects

```json
{
  "clean": {
    "coupled_hard_gain_pp": 2.194216064949032,
    "independent_hard_gain_pp": 4.256642186674156,
    "count_matched_random_gaps_pp": [
      2.0869620079452886,
      1.87914441404385,
      2.3268081158178546
    ],
    "soft_gain_pp": 0.7833311973706287,
    "gap_vs_simple_fusion_pp": 0.5910740163874308
  },
  "wrong_parent": {
    "coupled_hard_gain_pp": 2.2956177041059647,
    "independent_hard_gain_pp": 4.215026154470344,
    "count_matched_random_gaps_pp": [
      1.602376809216679,
      1.3494882684019203,
      2.60992527131031
    ],
    "soft_gain_pp": 0.6762856462187301,
    "gap_vs_simple_fusion_pp": 0.8209983417270763
  },
  "paraphrase": {
    "coupled_hard_gain_pp": 2.0466053504274058,
    "independent_hard_gain_pp": 4.300809141693648,
    "count_matched_random_gaps_pp": [
      2.008580920769468,
      1.7630028160756055,
      2.2924035683118618
    ],
    "soft_gain_pp": 0.7780469920185595,
    "gap_vs_simple_fusion_pp": 0.6287558988796889
  },
  "llm_style": {
    "coupled_hard_gain_pp": 3.1084012269715586,
    "independent_hard_gain_pp": 5.452494222089488,
    "count_matched_random_gaps_pp": [
      2.80697266027979,
      2.3540430387100457,
      3.676508238965255
    ],
    "soft_gain_pp": 0.9816615003575819,
    "gap_vs_simple_fusion_pp": 1.1907766771621482
  }
}
```

## clean

| Dataset/protocol | NoAdmission_Exact | FineAliasViewJoint_Exact | RivalFineHard_Exact | RivalFineSoft_Exact | RivalFineHard_MeanLogit |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 53.746954 | 55.739108 | 54.344136 | 54.330821 | 54.167084 |
| potsdam/potsdam | 38.920258 | 41.136621 | 40.617274 | 39.436072 | 41.080674 |
| udd5/udd5 | 28.175811 | 30.703970 | 34.039388 | 29.515973 | 34.424736 |
| oem/oem | 39.023152 | 39.907197 | 39.790613 | 39.278402 | 39.820078 |
| loveda/P | 50.706552 | 52.206600 | 52.587883 | 51.873736 | 51.449884 |
| loveda/D | 30.736030 | 37.111699 | 37.215569 | 33.365778 | 34.070213 |
| vaihingen/vaihingen | 51.826962 | 53.026018 | 52.944605 | 52.290158 | 52.397266 |
| landcoverai/landcoverai | 66.906020 | 67.977906 | 67.483370 | 67.139166 | 67.072024 |
| flair1/flair1 | 33.608404 | 34.396715 | 34.062366 | 33.853871 | 32.736652 |
| Equal-domain mean | 42.867949 | 44.999904 | 45.062165 | 43.651280 | 44.471091 |

| Endpoint | Mean mIoU |
| --- | ---: |
| Geometry | 40.531033 |
| NoAdmission_Exact | 42.867949 |
| FineAliasViewJoint_Exact | 44.999904 |
| ObserverAll20 | 39.128143 |
| ObserverFinePairHard | 43.384785 |
| RivalFineHard_Exact | 45.062165 |
| RivalFineSoft_Exact | 43.651280 |
| RivalNative16Hard_Exact | 43.863058 |
| RivalFineRandom0_Exact | 42.975203 |
| RivalFineRandom1_Exact | 43.183021 |
| RivalFineRandom2_Exact | 42.735357 |
| RivalFineHard_MeanLogit | 44.471091 |

## wrong_parent

| Dataset/protocol | NoAdmission_Exact | FineAliasViewJoint_Exact | RivalFineHard_Exact | RivalFineSoft_Exact | RivalFineHard_MeanLogit |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 47.408779 | 50.145074 | 49.874370 | 48.803002 | 49.793754 |
| potsdam/potsdam | 36.070589 | 38.493724 | 37.489780 | 36.506298 | 38.321121 |
| udd5/udd5 | 27.616556 | 30.588661 | 31.676345 | 28.215103 | 31.697233 |
| oem/oem | 40.053288 | 41.322503 | 41.396160 | 40.525878 | 41.100023 |
| loveda/P | 49.989695 | 51.254103 | 53.112923 | 51.641387 | 45.621829 |
| loveda/D | 30.333514 | 33.930006 | 36.060892 | 31.634693 | 30.658769 |
| vaihingen/vaihingen | 53.060153 | 54.643893 | 54.564974 | 53.636040 | 54.363815 |
| landcoverai/landcoverai | 68.123522 | 69.477891 | 69.366759 | 68.509352 | 68.935091 |
| flair1/flair1 | 33.246876 | 34.319417 | 33.848938 | 33.493196 | 32.840426 |
| Equal-domain mean | 41.989160 | 44.115146 | 44.284777 | 42.665445 | 43.463779 |

| Endpoint | Mean mIoU |
| --- | ---: |
| Geometry | 40.531033 |
| NoAdmission_Exact | 41.989160 |
| FineAliasViewJoint_Exact | 44.115146 |
| ObserverAll20 | 36.729247 |
| ObserverFinePairHard | 40.944273 |
| RivalFineHard_Exact | 44.284777 |
| RivalFineSoft_Exact | 42.665445 |
| RivalNative16Hard_Exact | 42.715460 |
| RivalFineRandom0_Exact | 42.682401 |
| RivalFineRandom1_Exact | 42.935289 |
| RivalFineRandom2_Exact | 41.674852 |
| RivalFineHard_MeanLogit | 43.463779 |

## paraphrase

| Dataset/protocol | NoAdmission_Exact | FineAliasViewJoint_Exact | RivalFineHard_Exact | RivalFineSoft_Exact | RivalFineHard_MeanLogit |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 54.530664 | 55.099635 | 53.103147 | 54.241801 | 52.645345 |
| potsdam/potsdam | 41.663326 | 43.994626 | 43.551123 | 42.377647 | 43.398771 |
| udd5/udd5 | 31.169271 | 34.496462 | 36.221454 | 33.124857 | 37.210481 |
| oem/oem | 39.619434 | 40.641842 | 40.464528 | 39.928722 | 40.383282 |
| loveda/P | 48.735654 | 50.766964 | 50.593173 | 49.713602 | 53.286945 |
| loveda/D | 29.513994 | 35.886984 | 37.026603 | 31.908856 | 34.478141 |
| vaihingen/vaihingen | 52.518481 | 53.753848 | 54.138431 | 53.194590 | 53.534751 |
| landcoverai/landcoverai | 65.629553 | 66.691516 | 65.977474 | 65.848909 | 65.247612 |
| flair1/flair1 | 34.530135 | 35.219380 | 35.064940 | 34.773851 | 33.619270 |
| Equal-domain mean | 43.646857 | 45.723037 | 45.693463 | 44.424904 | 45.064707 |

| Endpoint | Mean mIoU |
| --- | ---: |
| Geometry | 40.531033 |
| NoAdmission_Exact | 43.646857 |
| FineAliasViewJoint_Exact | 45.723037 |
| ObserverAll20 | 40.612375 |
| ObserverFinePairHard | 44.913184 |
| RivalFineHard_Exact | 45.693463 |
| RivalFineSoft_Exact | 44.424904 |
| RivalNative16Hard_Exact | 44.425854 |
| RivalFineRandom0_Exact | 43.684882 |
| RivalFineRandom1_Exact | 43.930460 |
| RivalFineRandom2_Exact | 43.401059 |
| RivalFineHard_MeanLogit | 45.064707 |

## llm_style

| Dataset/protocol | NoAdmission_Exact | FineAliasViewJoint_Exact | RivalFineHard_Exact | RivalFineSoft_Exact | RivalFineHard_MeanLogit |
| --- | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | 31.837398 | 32.514620 | 34.310310 | 32.027142 | 34.184476 |
| oem/oem | 40.439688 | 40.805695 | 40.812441 | 40.565181 | 40.511300 |
| loveda/P | 50.706552 | 52.206600 | 52.587883 | 51.873736 | 51.449884 |
| loveda/D | 30.736030 | 37.111699 | 37.215569 | 33.365778 | 34.070213 |
| Equal-domain mean | 34.337705 | 36.810671 | 37.446107 | 35.319367 | 36.255330 |

| Endpoint | Mean mIoU |
| --- | ---: |
| Geometry | 34.733161 |
| NoAdmission_Exact | 34.337705 |
| FineAliasViewJoint_Exact | 36.810671 |
| ObserverAll20 | 28.052436 |
| ObserverFinePairHard | 33.504931 |
| RivalFineHard_Exact | 37.446107 |
| RivalFineSoft_Exact | 35.319367 |
| RivalNative16Hard_Exact | 35.969582 |
| RivalFineRandom0_Exact | 34.639134 |
| RivalFineRandom1_Exact | 35.092064 |
| RivalFineRandom2_Exact | 33.769598 |
| RivalFineHard_MeanLogit | 36.255330 |

## Cost And Interpretation

Cached12-endpoint suite wall 147.4824s. No fresh image feature observations; model loading and cached prediction evaluation remain. These are not standalone deployment costs. Raw LLM provenance and whole-model noise robustness are not established. Historical style exists on three domains only; LoveDA style equals clean. Adapted VIP observer and classical consistency/reconstruction remain attributed. Per-class IoU/precision/recall/area, transitions and dynamic retained counts are archived in merged.json.
