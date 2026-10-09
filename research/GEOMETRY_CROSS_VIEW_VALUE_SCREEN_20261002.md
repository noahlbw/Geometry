# Cross-view matched Value innovation: verified complete-image screen

One fixed internally coupled Geometry model on physical A800 GPUs0-7. Same96 image IDs: UDD5 full40; eight each of seven other domains. COMPLETE image predictions, not cached-window statistics or full eight-dataset metrics. Corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. LoveDA D enters the mean once, P separately. All domains are development data.

Primary uses the same frozen DINOv3/DINO.text and no VIP/external semantic source. One extra 1024->512 context view per fine tile. Same current fine Query reads geometry-conditioned local/context learned Keys/Values. Native likelihood partition functions determine the bounded view share; only their matched Value difference is added to original Geometry before the projection/residual/MLP. Both frozen head blocks, original prefix pathway and native patch mass remain active. Single final fine descriptor; all20 aliases and RS/LME/assembly unchanged.

11 CPU tests pass locally/remotely. Actual-checkpoint smoke has zero duplicate-view/no-context errors, zero context Geometry replay error, finite outputs and an unchanged complete head weight fingerprint. Original inference-tensor version-counter smoke failure was corrected in validation only; its log is preserved. Exact original Geometry/SCLIP/VIPProxy per-image confusions and transition endpoints independently verified. Target masks enter only after complete image predictions; no threshold, alias, coefficient or per-domain route fitted.

Cross-attention, geometric priors, differential reads and native projections have precedents. No priority claim or CVPR readiness follows from algebra. Native QK/view evidence is not calibrated semantic reliability; the Value displacement is signed, not a nonnegative attention flow.

## mIoU

| Dataset/protocol | Geometry | SCLIP_Two | VIPProxy_Two | ContextGeometry | MeanLogit_ContextGeometry | Shuffled_ValueInnovation | Geometry_ValueInnovation | Historical Anchored_VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.9462 | 30.6800 | 31.1056 | 38.6567 | 36.4209 | 30.0140 | 33.7193 | 57.0012 |
| potsdam/potsdam | 40.3528 | 44.0525 | 42.7377 | 37.3743 | 39.7125 | 29.7899 | 38.2670 | 41.7806 |
| udd5/udd5 | 50.5553 | 50.2837 | 49.9071 | 50.8727 | 51.5644 | 42.5563 | 50.8125 | 49.2117 |
| oem/oem | 39.3543 | 37.3120 | 38.9899 | 37.0406 | 39.0778 | 37.4363 | 39.8999 | 31.9533 |
| loveda/P | 62.8254 | 68.2854 | 68.5692 | 63.4297 | 66.0230 | 42.8443 | 64.9102 | 62.4863 |
| loveda/D | 38.4558 | 36.5043 | 35.6670 | 34.1337 | 37.1782 | 28.2324 | 39.0910 | 36.5600 |
| vaihingen/vaihingen | 50.2325 | 52.0872 | 50.5393 | 41.4339 | 46.5562 | 41.1048 | 48.5209 | 51.4491 |
| landcoverai/landcoverai | 60.9049 | 61.7170 | 59.3073 | 59.3830 | 61.2873 | 46.5265 | 61.4722 | 66.9060 |
| flair1/flair1 | 38.8428 | 37.6225 | 39.3436 | 29.2016 | 39.4205 | 29.1321 | 38.2545 | 33.6084 |
| Equal-domain mean | 43.830569 | 43.782393 | 43.449702 | 41.012051 | 43.902207 | 35.599071 | 43.754654 | 46.058780 |

Prospective gate passed: False. Primary delta vs Geometry: -0.075915pp; vs same-source simple fusion: -0.147553pp; vs shuffled: +8.155583pp; vs historical Anchored_VIP: -2.304126pp.

Historical Anchored_VIP has matched IDs/checkpoints/vocabulary and exact Geometry control, but different view/text/readout settings and cost; it is not an equal-information operator ablation. VIPProxy_Two is a matched adaptation, not complete official VIP.

Failed checks: mean_vs_Geometry, mean_vs_SCLIP_Two, mean_vs_MeanLogit_ContextGeometry, vdd_vs_MeanLogit_ContextGeometry, retain_potsdam_potsdam, potsdam_vs_Geometry, potsdam_vs_MeanLogit_ContextGeometry, retain_vaihingen_vaihingen, retain_flair1_flair1

## Mechanism Verdict

Primary improves Geometry on 5/8 domain scores: vdd, udd5, oem, loveda, landcoverai. LoveDA P is a separate protocol, not an extra domain. A screen gain does not establish full-dataset transfer or official-VIP superiority.

| Focus domain | Context alone | Fixed logit mean | Internal readout | Internal minus mean |
| --- | ---: | ---: | ---: | ---: |
| vdd | 38.6567 | 36.4209 | 33.7193 | -2.7016 |
| potsdam | 37.3743 | 39.7125 | 38.2670 | -1.4455 |

The source and the writeback must be distinguished. ContextGeometry changes the physical view while retaining original Geometry; its performance is measured above. An internal difference that loses to the same-source logit mean has not shown that it uses that information more effectively. Better results than spatial shuffling establish a dependence on correspondence, not that the unshuffled corrections are semantically reliable.

At one block, the original conditional patch read is G_f V_f, whereas the subtraction is P_f V_f with P_f=softmax(log G_f+QK_f). Therefore the effective fine coefficient is G_f-alpha P_f, not (1-alpha)G_f; it can be negative. Original Geometry deliberately rewrites the native patch relation, but the innovation reintroduces native QK in both matched reads. Duplicate-view cancellation does not make arbitrary cross-view changes harmless. This is an operator limitation, not a separately established explanation of every error.

Normalized priors remove donor-count bias, not scale-dependent learned-Key calibration. The partition-function share measures matching under this frozen head; it does not measure whether a competing class gains true coverage or false activations. Average shares below do not reveal the complete per-query gate distribution.

| Domain/protocol | Mean context share | Corrected | Corrupted | Net corrections |
| --- | ---: | ---: | ---: | ---: |
| vdd/vdd | 0.452210 | 3748801 | 1262694 | +2486107 |
| potsdam/potsdam | 0.459995 | 115183 | 354691 | -239508 |
| udd5/udd5 | 0.461731 | 6345903 | 4773360 | +1572543 |
| oem/oem | 0.479678 | 175864 | 103334 | +72530 |
| loveda/P | 0.462674 | 122721 | 29313 | +93408 |
| loveda/D | 0.462674 | 371437 | 218479 | +152958 |
| vaihingen/vaihingen | 0.428762 | 84633 | 238118 | -153485 |
| landcoverai/landcoverai | 0.546866 | 34044 | 18202 | +15842 |
| flair1/flair1 | 0.533208 | 36212 | 56027 | -19815 |

Potsdam car IoU 11.6582->10.1338, predicted area 19.8762->22.9335%, recall 99.0262->99.2775%. This does not demonstrate recovered small objects: increased car area with already high recall must be read alongside precision, false positives and lost competing-class coverage.

Decision: reject promotion of this fixed candidate. Do not launch a full eight-domain run, tune its share from these labels, or route different winners by dataset. The failure does not prove that Geometry has reached its performance ceiling.

## Per-class competition

### vdd/vdd

Corrected/corrupted/wrong-to-wrong: 3748801/1262694/2417235.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| other | 25.3949 | 26.8259 | 26.6612 | 1.2663 | 41.278/42.953 | +191150 | -1323079 |
| wall | 11.8982 | 14.9332 | 12.7828 | 0.8846 | 12.877/94.576 | +4557 | -991086 |
| road | 24.1970 | 22.2208 | 23.7161 | -0.4809 | 24.267/91.258 | -5426 | +78841 |
| vegetation | 62.0257 | 64.8461 | 62.8761 | 0.8504 | 86.297/69.850 | -2647 | -320108 |
| vehicle | 5.6965 | 6.5322 | 5.3805 | -0.3160 | 5.381/99.988 | +72 | +235041 |
| roof | 60.7577 | 73.4602 | 64.9192 | 4.1615 | 89.374/70.349 | +1098278 | -174928 |
| water | 33.6534 | 46.1280 | 39.6995 | 6.0461 | 93.952/40.741 | +1200123 | +9212 |

Internal diagnostics:
```json
{
  "tiles": 704,
  "mean_context_fraction": 0.45221048845136963,
  "mean_absolute_value_displacement": 0.18877599720144644,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### potsdam/potsdam

Corrected/corrupted/wrong-to-wrong: 115183/354691/235530.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 51.6875 | 47.3191 | 47.2638 | -4.4237 | 72.793/57.405 | -138282 | +33588 |
| building | 78.2026 | 73.9230 | 76.9646 | -1.2380 | 79.514/96.000 | -2206 | +16456 |
| low vegetation | 33.4221 | 35.0559 | 27.7111 | -5.7110 | 73.346/30.814 | -114608 | -51331 |
| tree | 62.5813 | 67.3083 | 63.3490 | 0.7677 | 90.391/67.923 | +19650 | +6198 |
| car | 11.6582 | 10.9777 | 10.1338 | -1.5244 | 10.141/99.278 | +471 | +244113 |
| clutter | 4.5652 | 3.6908 | 4.1795 | -0.3857 | 7.187/9.080 | -4533 | -9516 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "mean_context_fraction": 0.4599949507974088,
  "mean_absolute_value_displacement": 0.22116312156948778,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### udd5/udd5

Corrected/corrupted/wrong-to-wrong: 6345903/4773360/3830431.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| vegetation | 82.4937 | 82.5838 | 82.1892 | -0.3045 | 96.589/84.646 | -524315 | -140251 |
| building | 83.9035 | 85.9325 | 84.9332 | 1.0297 | 88.407/95.578 | +2156754 | +185192 |
| road | 45.7856 | 47.9939 | 46.2048 | 0.4192 | 70.784/57.093 | +337145 | +68818 |
| vehicle | 10.0092 | 9.9994 | 9.7968 | -0.2124 | 9.818/97.839 | +3199 | +778147 |
| other | 30.5846 | 31.3124 | 30.9386 | 0.3540 | 54.831/41.521 | -400240 | -2464449 |

Internal diagnostics:
```json
{
  "tiles": 3232,
  "mean_context_fraction": 0.46173050393417275,
  "mean_absolute_value_displacement": 0.1977834606849782,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### oem/oem

Corrected/corrupted/wrong-to-wrong: 175864/103334/75215.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -60018 |
| rangeland | 47.7780 | 48.5516 | 48.0164 | 0.2384 | 65.925/63.868 | -2553 | -15097 |
| developed space | 28.5294 | 31.6033 | 31.6773 | 3.1479 | 63.418/38.760 | +69587 | +11763 |
| road | 40.8291 | 37.6494 | 40.3495 | -0.4796 | 45.294/78.705 | -1522 | +4298 |
| tree | 56.0657 | 52.6756 | 55.5450 | -0.5207 | 87.287/60.434 | -10511 | +979 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -717 |
| agriculture land | 79.1353 | 85.2397 | 81.3192 | 2.1839 | 91.498/87.966 | +23023 | -9315 |
| building | 62.4971 | 56.9025 | 62.2916 | -0.2055 | 65.665/92.382 | -5494 | -4423 |

Internal diagnostics:
```json
{
  "tiles": 67,
  "mean_context_fraction": 0.479678492854113,
  "mean_absolute_value_displacement": 0.20131749537453722,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### loveda/P

Corrected/corrupted/wrong-to-wrong: 122721/29313/35212.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 64.4987 | 67.5236 | 65.3745 | 0.8758 | 66.212/98.102 | +36 | -554 |
| road | 66.9348 | 67.0740 | 65.3461 | -1.5887 | 66.003/98.500 | +194 | +11952 |
| water | 81.1851 | 80.1724 | 81.6030 | 0.4179 | 86.949/92.993 | +3171 | -1704 |
| barren | 24.3403 | 25.9840 | 25.9659 | 1.6256 | 65.332/30.115 | +6322 | +54 |
| tree | 53.7307 | 62.7500 | 61.2697 | 7.5390 | 77.786/74.264 | -10147 | -105056 |
| farm | 86.2626 | 92.6343 | 89.9020 | 3.6394 | 95.438/93.939 | +93832 | +1900 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "mean_context_fraction": 0.4626743961125612,
  "mean_absolute_value_displacement": 0.2093615439823932,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### loveda/D

Corrected/corrupted/wrong-to-wrong: 371437/218479/141117.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 34.1983 | 24.1758 | 35.9673 | 1.7690 | 70.263/42.425 | +30060 | -137306 |
| building | 31.1862 | 31.0952 | 29.1917 | -1.9945 | 29.847/93.007 | -197 | +5467 |
| road | 45.4818 | 44.1695 | 43.8970 | -1.5848 | 44.212/98.403 | +222 | +25951 |
| water | 61.9193 | 60.1976 | 62.2909 | 0.3716 | 65.506/92.695 | +16154 | +17546 |
| barren | 14.5296 | 16.2787 | 15.5314 | 1.0018 | 62.577/17.122 | +3229 | -3209 |
| tree | 26.8904 | 26.0966 | 30.8312 | 3.9408 | 35.730/69.217 | -13453 | -216736 |
| farm | 54.9851 | 58.2340 | 55.9273 | 0.9422 | 66.979/77.219 | +116943 | +155329 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "mean_context_fraction": 0.4626743961125612,
  "mean_absolute_value_displacement": 0.2093615439823932,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### vaihingen/vaihingen

Corrected/corrupted/wrong-to-wrong: 84633/238118/114581.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| impervious surface | 49.2002 | 41.2253 | 46.5792 | -2.6210 | 80.032/52.704 | -55343 | +34176 |
| building | 74.6610 | 69.7259 | 74.0425 | -0.6185 | 74.725/98.782 | +4259 | +24637 |
| low vegetation | 46.7115 | 42.2493 | 41.6798 | -5.0317 | 93.906/42.838 | -117927 | -28524 |
| tree | 71.8329 | 71.8876 | 72.2207 | 0.3878 | 82.195/85.614 | +15316 | +10693 |
| car | 8.7566 | 7.6927 | 8.0825 | -0.6741 | 8.095/98.153 | +210 | +112503 |

Internal diagnostics:
```json
{
  "tiles": 72,
  "mean_context_fraction": 0.42876190570597017,
  "mean_absolute_value_displacement": 0.20051409200661713,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### landcoverai/landcoverai

Corrected/corrupted/wrong-to-wrong: 34044/18202/8327.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| background | 82.6090 | 82.7314 | 83.8033 | 1.1943 | 96.964/86.061 | +14462 | -3591 |
| building | 34.9562 | 33.3267 | 32.7635 | -2.1927 | 32.828/99.399 | +34 | +6144 |
| woodland | 78.3638 | 83.2134 | 82.6228 | 4.2590 | 91.396/89.592 | +1381 | -24695 |
| water | 93.6635 | 93.7273 | 94.1735 | 0.5100 | 94.174/100.000 | +0 | -1003 |
| road | 14.9318 | 13.4375 | 13.9979 | -0.9339 | 14.614/76.842 | -35 | +7303 |

Internal diagnostics:
```json
{
  "tiles": 8,
  "mean_context_fraction": 0.5468662017956376,
  "mean_absolute_value_displacement": 0.28243220318108797,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

### flair1/flair1

Corrected/corrupted/wrong-to-wrong: 36212/56027/40041.

| Class | Geometry IoU | Context/fine mean IoU | Coupled IoU | Delta vs Geometry | Coupled P/R | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| building | 49.6979 | 44.0633 | 46.5634 | -3.1345 | 47.359/96.516 | +653 | +20990 |
| pervious surface | 57.0621 | 67.7143 | 60.8306 | 3.7685 | 91.918/64.268 | +15403 | +1894 |
| impervious surface | 52.6509 | 51.3397 | 51.8382 | -0.8127 | 61.710/76.418 | -1232 | +5475 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -16238 |
| water | 73.2328 | 73.6234 | 73.4842 | 0.2514 | 79.678/90.434 | -2876 | -4320 |
| coniferous | 43.0233 | 43.6365 | 39.3216 | -3.7017 | 61.532/52.138 | -772 | -450 |
| deciduous | 54.0229 | 59.1076 | 52.5652 | -1.4577 | 80.142/60.437 | -9501 | -6517 |
| brushwood | 18.6136 | 19.7005 | 18.9998 | 0.3862 | 22.751/53.538 | +6168 | +27108 |
| vineyard | NA | NA | NA | NA | 0.000/0.000 | +0 | +0 |
| herbaceous vegetation | 60.2329 | 56.1070 | 56.7305 | -3.5024 | 95.248/58.383 | -27658 | -5732 |
| agricultural land | 18.7339 | 18.3333 | 20.4654 | 1.7315 | 23.992/58.198 | +0 | -1680 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 | +0 | -715 |

Internal diagnostics:
```json
{
  "tiles": 8,
  "mean_context_fraction": 0.5332081224769354,
  "mean_absolute_value_displacement": 0.2660853872075677,
  "prefix_displacement_max_error": 0.0,
  "invalid_query_displacement_max_error": 0.0,
  "context_geometry_replay_max_error": 0.0
}
```

## Cost And Decision

Parallel suite elapsed: 931.3596s. Each duration below includes all seven arms; not standalone-primary cost.

| Dataset | All-arm seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 220.0743 | 3956.5723 |
| potsdam | 14.1459 | 3952.5273 |
| udd5 | 898.4085 | 3955.9702 |
| oem | 15.6433 | 3957.2573 |
| loveda | 20.8081 | 3956.0596 |
| vaihingen | 13.8527 | 3953.9590 |
| landcoverai | 2.1588 | 3833.6060 |
| flair1 | 2.8479 | 3834.7007 |

### Single-arm descriptor benchmark

As-implemented single512-window descriptor extraction, one resident frozen model; candidate includes local preparation, context preparation and coupled head only, no comparator arms. No text encoding/scoring, dense probability assembly, image decoding or initialization timed. Preparation computes native/Geometry auxiliary paths; this is not an optimized or full-image latency.

One resident model; three warmups and20 synchronized trials. Same fixed OEM image; no masks.

| Arm | Median512-window ms | Peak allocated MiB |
| --- | ---: | ---: |
| Geometry | 24.9397 | 3715.7246 |
| Geometry_ValueInnovation | 84.5616 | 3827.2959 |

As-implemented descriptor median ratio: 3.3906x. This is not a whole-image latency ratio or an optimized implementation comparison.

No automatic full rollout or post-result coefficient/support/layer search. Preserve original Geometry, historical Anchored_VIP and all rejected candidates. A failed frozen gate rejects this particular internal readout, not every cross-view method. A pass would still require full-domain results, independent cost and validation before a final-model/CVPR claim.
