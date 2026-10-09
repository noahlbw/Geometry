# Geometry semantic budget: verified fixed96 screen

UDD5 full40; seven other domains8 each. Unchanged20 aliases, six RS templates, same native512/128/Hann.
Developed validation only. Original Geometry/SCLIP_Two/VIPProxy_Two per-image baselines exactly replayed.
Geometry-constrained per-head CSA is a hypothesis; CSA and KL projection are attributed standard machinery.

| Dataset/protocol | Geometry | SCLIP_Two | VIPProxy_Two | CSA_SameShell | Proxy_SameShell | Geometry_SemanticBudget |
|---|---:|---:|---:|---:|---:|---:|
| vdd/vdd | 31.9462 | 30.6800 | 31.1056 | 32.3676 | 31.9050 | 31.9171 |
| potsdam/potsdam | 40.3528 | 44.0525 | 42.7377 | 38.9705 | 40.3696 | 40.2606 |
| udd5/udd5 | 50.5553 | 50.2837 | 49.9071 | 49.7287 | 50.7353 | 50.3421 |
| oem/oem | 39.3543 | 37.3120 | 38.9899 | 39.5034 | 39.3507 | 39.1923 |
| loveda/P | 62.8254 | 68.2854 | 68.5692 | 60.7125 | 63.3524 | 61.0692 |
| loveda/D | 38.4558 | 36.5043 | 35.6670 | 38.1218 | 38.3488 | 38.0923 |
| vaihingen/vaihingen | 50.2325 | 52.0872 | 50.5393 | 49.1961 | 50.3944 | 50.4492 |
| landcoverai/landcoverai | 60.9049 | 61.7170 | 59.3073 | 61.5185 | 60.8127 | 60.7971 |
| flair1/flair1 | 38.8428 | 37.6225 | 39.3436 | 37.6414 | 38.5829 | 38.4751 |
| Eight-domain mean, LoveDA D once | 43.8306 | 43.7824 | 43.4497 | 43.3810 | 43.8124 | 43.6907 |

Predeclared gate passed: `False`. No automatic full rollout.

## Decision And Mechanism

Retain original Geometry. The new candidate loses 0.139842 pp in the equal-domain mean,
improves only Vaihingen (1/8 domains), and loses 1.756157 pp on LoveDA P.
It improves on unconstrained CSA_SameShell by 0.309727 pp, but does not beat Geometry or Proxy_SameShell.
All96 sample keys are unique and covered; the three historical controls exactly replay their per-image confusions.
Seven synthetic tests and real-checkpoint fp32/bf16 identity/constraint checks passed; no masks were loaded in smoke.

The second block's constraint activity is 95.52%-98.05% across domains, with mean row L1 projection
0.863-1.000. This budget frequently changes semantic attention, rather than occasionally correcting an unsafe edge.
Constraints are numerically satisfied, but that does not imply correct semantic discrimination.

VDD vehicle IoU falls 5.6965 -> 5.4209, prediction area grows 4.1453% -> 4.3546%, and recall stays about99.9%.
Potsdam car IoU falls 11.6582 -> 11.5228, area grows 19.8762% -> 20.1165%, and recall stays about99.0%.
Thus this candidate does not solve false vehicle activation. Potsdam low-vegetation recall rises
38.2826% -> 39.2864%, but the gain does not compensate the other classes. UDD5 has2,521,778 beneficial
versus3,894,199 harmful changed pixels; LoveDA P has15,209 versus82,828.

The same-shell controls also matter: Potsdam SCLIP_Two scores44.0525 but CSA_SameShell38.9705.
Moving the relation into Geometry's shell changes patch/prefix allocation and attention strength together;
this is not evidence that QQ/KK relation quality alone explains SCLIP's advantage, nor a pure amplitude ablation.
Even equal row mass does not fix feature-increment magnitude: candidate block1 projection norms exceed
Geometry's (VDD49.5051 vs47.0039; Potsdam51.3842 vs48.2836), without improving their mIoU.

Independent512-window latency rises22.06 ->60.37ms (2.737x), allocated peak3594.01 ->4395.65MiB.
There is no favorable accuracy/cost tradeoff for this candidate. These results reject this frozen
cost/budget construction, not every possible use of head-specific geometry constraints.
Do not relax the budget against these labels or replace the retained model with this failed candidate.
The unresolved target is class-discriminative evidence and correct readout allocation, not simply more geometric agreement.

## Execution

All old OVSS GPU workers/controllers were stopped; partial and completed outputs were preserved.
The default four-shard evaluator initially ran s0 only. Never-launched s1-s3 completed the same frozen96
manifest without replaying s0 or changing the model. Startup and invariant-diagnostic failure logs remain
remote alongside the new outputs; the Proxy_SameShell bf16 relation-normalization issue was repaired before scoring.
All new workers finished and GPUs0-7 are now idle. No paused automation was resumed.

## Independent Single-Window Cost

```json
{
  "Geometry": {
    "window_median_seconds": 0.022055633133277297,
    "runs_seconds": [
      0.022152871126309037,
      0.022044020937755704,
      0.022044189041480422,
      0.0220658159814775,
      0.022055633133277297
    ],
    "peak_allocated_mb": 3594.01123046875
  },
  "Geometry_SemanticBudget": {
    "window_median_seconds": 0.06036648899316788,
    "runs_seconds": [
      0.06042752298526466,
      0.060324923135340214,
      0.06039061094634235,
      0.06029718299396336,
      0.06036648899316788
    ],
    "peak_allocated_mb": 4395.6484375
  }
}
```

## Per-Class Results And Diagnostics

### vdd/vdd

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | other | 25.3949 | 38.9278 | 42.2128 | 29.1864 |
| Geometry | wall | 11.8982 | 11.9838 | 94.3324 | 15.3444 |
| Geometry | road | 24.1970 | 24.7412 | 91.6676 | 5.1104 |
| Geometry | vegetation | 62.0257 | 84.6841 | 69.8628 | 17.8191 |
| Geometry | vehicle | 5.6965 | 5.6966 | 99.9563 | 4.1453 |
| Geometry | roof | 60.7577 | 88.0153 | 66.2377 | 20.9399 |
| Geometry | water | 33.6534 | 93.0590 | 34.5198 | 7.4546 |
| SCLIP_Two | other | 9.0853 | 27.6554 | 11.9177 | 11.5987 |
| SCLIP_Two | wall | 7.2858 | 7.3157 | 94.6941 | 25.2320 |
| SCLIP_Two | road | 23.4986 | 24.0750 | 90.7527 | 5.1994 |
| SCLIP_Two | vegetation | 62.8305 | 77.0438 | 77.3024 | 21.6719 |
| SCLIP_Two | vehicle | 13.7022 | 13.7029 | 99.9647 | 1.7234 |
| SCLIP_Two | roof | 57.8464 | 76.3206 | 70.4993 | 25.7022 |
| SCLIP_Two | water | 40.5110 | 94.1342 | 41.5601 | 8.8724 |
| VIPProxy_Two | other | 12.5663 | 34.5441 | 16.4937 | 12.8511 |
| VIPProxy_Two | wall | 8.1229 | 8.1618 | 94.4626 | 22.5609 |
| VIPProxy_Two | road | 22.8861 | 24.0793 | 82.2023 | 4.7087 |
| VIPProxy_Two | vegetation | 66.5882 | 83.9582 | 76.2952 | 19.6280 |
| VIPProxy_Two | vehicle | 4.8657 | 4.8658 | 99.9246 | 4.8515 |
| VIPProxy_Two | roof | 62.0679 | 78.2770 | 74.9836 | 26.6538 |
| VIPProxy_Two | water | 40.6421 | 95.2966 | 41.4740 | 8.7460 |
| CSA_SameShell | other | 26.3389 | 40.8478 | 42.5795 | 28.0562 |
| CSA_SameShell | wall | 12.0863 | 12.1744 | 94.3528 | 15.1075 |
| CSA_SameShell | road | 24.9911 | 25.6816 | 90.2857 | 4.8490 |
| CSA_SameShell | vegetation | 60.9887 | 83.4099 | 69.4083 | 17.9736 |
| CSA_SameShell | vehicle | 4.8486 | 4.8488 | 99.9211 | 4.8684 |
| CSA_SameShell | roof | 61.9674 | 88.3527 | 67.4798 | 21.2511 |
| CSA_SameShell | water | 35.3523 | 92.6083 | 36.3788 | 7.8942 |
| Proxy_SameShell | other | 24.7389 | 38.5493 | 40.8474 | 28.5197 |
| Proxy_SameShell | wall | 11.8824 | 11.9656 | 94.4699 | 15.3901 |
| Proxy_SameShell | road | 23.5100 | 23.9783 | 92.3304 | 5.3111 |
| Proxy_SameShell | vegetation | 62.6289 | 84.6076 | 70.6823 | 18.0444 |
| Proxy_SameShell | vehicle | 5.6566 | 5.6566 | 99.9881 | 4.1759 |
| Proxy_SameShell | roof | 61.1286 | 88.0576 | 66.6545 | 21.0615 |
| Proxy_SameShell | water | 33.7896 | 92.9528 | 34.6780 | 7.4973 |
| Geometry_SemanticBudget | other | 25.6768 | 39.2134 | 42.6546 | 29.2771 |
| Geometry_SemanticBudget | wall | 12.0752 | 12.1648 | 94.2475 | 15.1025 |
| Geometry_SemanticBudget | road | 23.4315 | 23.9995 | 90.8268 | 5.2200 |
| Geometry_SemanticBudget | vegetation | 62.6586 | 85.2163 | 70.3005 | 17.8188 |
| Geometry_SemanticBudget | vehicle | 5.4209 | 5.4211 | 99.9242 | 4.3546 |
| Geometry_SemanticBudget | roof | 60.2213 | 88.0722 | 65.5691 | 20.7151 |
| Geometry_SemanticBudget | water | 33.9354 | 93.1191 | 34.8082 | 7.5120 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 704,
    "Geometry_SemanticBudget__block0_value_read_norm": 24.24289791692387,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 81.85150135105306,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 13.289004797285253,
    "Geometry_SemanticBudget__block0_block_output_norm": 30.559058877554808,
    "Geometry_SemanticBudget__block0_patch_mass": 0.48810479595241224,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8720998764038086,
    "Geometry_SemanticBudget__block0_dual_mean": 0.25337732930413703,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.1913230788301337,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.7853882686997002,
    "Geometry_SemanticBudget__block0_geometry_budget": 1.8631256830624559,
    "Geometry_SemanticBudget__block0_constraint_violation": 6.515871394764293e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.4093890623612836e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.4318462659774179,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.522841085087169e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 15.727863758802414,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 49.50507002527063,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 43.438777617432855,
    "Geometry_SemanticBudget__block1_block_output_norm": 560.3773475126786,
    "Geometry_SemanticBudget__block1_patch_mass": 0.3509779841300439,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9771352247758345,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5934983785687522,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.727227414873513,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 1.851714938371019,
    "Geometry_SemanticBudget__block1_geometry_budget": 1.8631256830624559,
    "Geometry_SemanticBudget__block1_constraint_violation": 8.331103758378462e-08,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.5931136066263373e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.9997348800640214,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.609200087460605e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 23.2520244798877,
    "Geometry__block0_projected_increment_norm": 78.98083308610049,
    "Geometry__block0_attention_residual_norm": 13.158086092634635,
    "Geometry__block0_block_output_norm": 30.47985182296146,
    "Geometry__block0_patch_mass": 0.48810479595241224,
    "Geometry__block1_value_read_norm": 14.657196611166,
    "Geometry__block1_projected_increment_norm": 47.00388157909567,
    "Geometry__block1_attention_residual_norm": 43.28018223697489,
    "Geometry__block1_block_output_norm": 555.3938054604964,
    "Geometry__block1_patch_mass": 0.3587125282184305,
    "CSA_SameShell__block0_value_read_norm": 23.345482663674787,
    "CSA_SameShell__block0_projected_increment_norm": 79.74681678685275,
    "CSA_SameShell__block0_attention_residual_norm": 13.237166582183404,
    "CSA_SameShell__block0_block_output_norm": 30.535326757214285,
    "CSA_SameShell__block0_patch_mass": 0.48810479595241224,
    "CSA_SameShell__block0_row_mass_error": 4.2612241073088214e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.554997843774883,
    "CSA_SameShell__block1_projected_increment_norm": 48.78712531653318,
    "CSA_SameShell__block1_attention_residual_norm": 43.38784700632095,
    "CSA_SameShell__block1_block_output_norm": 562.0347475572066,
    "CSA_SameShell__block1_patch_mass": 0.352728432002054,
    "CSA_SameShell__block1_row_mass_error": 4.1113658384843305e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 22.93342596834356,
    "Proxy_SameShell__block0_projected_increment_norm": 78.21787243539637,
    "Proxy_SameShell__block0_attention_residual_norm": 13.141225599429823,
    "Proxy_SameShell__block0_block_output_norm": 30.468308692628685,
    "Proxy_SameShell__block0_patch_mass": 0.48810479595241224,
    "Proxy_SameShell__block0_row_mass_error": 2.1615150299939243e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.678329772569917,
    "Proxy_SameShell__block1_projected_increment_norm": 47.03975713253021,
    "Proxy_SameShell__block1_attention_residual_norm": 43.30492848157883,
    "Proxy_SameShell__block1_block_output_norm": 556.5789140354503,
    "Proxy_SameShell__block1_patch_mass": 0.3596650356723165,
    "Proxy_SameShell__block1_row_mass_error": 2.1615150299939243e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "vdd__Geometry_SemanticBudget__top2_margin": 0.012116739828673995,
    "vdd__Geometry__top2_margin": 0.012144676029658347,
    "vdd__CSA_SameShell__top2_margin": 0.011882234877372288,
    "vdd__Proxy_SameShell__top2_margin": 0.01223431131901717,
    "vdd__SCLIP_Two__top2_margin": 0.013861539124429162,
    "vdd__VIPProxy_Two__top2_margin": 0.016549807362025604
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 96000000,
      "changed": 28074917,
      "beneficial": 5927381,
      "harmful": 9721337,
      "wrong_to_wrong": 12426199,
      "base_confusion": [
        [
          10907187,
          6551598,
          2362979,
          1464214,
          2320375,
          2030925,
          201284
        ],
        [
          45,
          1765289,
          38,
          122,
          1025,
          104784,
          46
        ],
        [
          689,
          9205,
          1213794,
          19036,
          81234,
          167,
          0
        ],
        [
          3709316,
          899749,
          455269,
          14486350,
          800569,
          88769,
          295397
        ],
        [
          0,
          88,
          2,
          9,
          226695,
          0,
          0
        ],
        [
          3657627,
          5293801,
          7224,
          15373,
          44405,
          17693075,
          0
        ],
        [
          9744124,
          210860,
          866660,
          1121244,
          505150,
          184555,
          6659653
        ]
      ],
      "proposal_confusion": [
        [
          3079362,
          10713496,
          3066939,
          3402370,
          1181440,
          4192439,
          202516
        ],
        [
          2,
          1772057,
          123,
          55,
          498,
          98614,
          0
        ],
        [
          3,
          20633,
          1201679,
          25930,
          74852,
          1028,
          0
        ],
        [
          1906522,
          1881758,
          378530,
          16028984,
          145644,
          96878,
          297103
        ],
        [
          0,
          2,
          6,
          72,
          226714,
          0,
          0
        ],
        [
          173908,
          7566254,
          12064,
          112493,
          15369,
          18831417,
          0
        ],
        [
          5974969,
          2268539,
          332049,
          1235116,
          9983,
          1453716,
          8017874
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 96000000,
      "changed": 25509950,
      "beneficial": 6147052,
      "harmful": 7903903,
      "wrong_to_wrong": 11458995,
      "base_confusion": [
        [
          10907187,
          6551598,
          2362979,
          1464214,
          2320375,
          2030925,
          201284
        ],
        [
          45,
          1765289,
          38,
          122,
          1025,
          104784,
          46
        ],
        [
          689,
          9205,
          1213794,
          19036,
          81234,
          167,
          0
        ],
        [
          3709316,
          899749,
          455269,
          14486350,
          800569,
          88769,
          295397
        ],
        [
          0,
          88,
          2,
          9,
          226695,
          0,
          0
        ],
        [
          3657627,
          5293801,
          7224,
          15373,
          44405,
          17693075,
          0
        ],
        [
          9744124,
          210860,
          866660,
          1121244,
          505150,
          184555,
          6659653
        ]
      ],
      "proposal_confusion": [
        [
          4261731,
          9085683,
          2743488,
          2556652,
          3482610,
          3536867,
          171531
        ],
        [
          0,
          1767725,
          0,
          130,
          634,
          102860,
          0
        ],
        [
          5,
          16450,
          1088461,
          24734,
          193434,
          1041,
          0
        ],
        [
          2691839,
          1062081,
          333073,
          15820123,
          518497,
          86427,
          223379
        ],
        [
          0,
          13,
          0,
          158,
          226623,
          0,
          0
        ],
        [
          325098,
          6072472,
          17670,
          50575,
          216432,
          20029258,
          0
        ],
        [
          5058387,
          3654070,
          337628,
          390483,
          19203,
          1831204,
          8001271
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 96000000,
      "changed": 5822283,
      "beneficial": 2175775,
      "harmful": 1502861,
      "wrong_to_wrong": 2143647,
      "base_confusion": [
        [
          10907187,
          6551598,
          2362979,
          1464214,
          2320375,
          2030925,
          201284
        ],
        [
          45,
          1765289,
          38,
          122,
          1025,
          104784,
          46
        ],
        [
          689,
          9205,
          1213794,
          19036,
          81234,
          167,
          0
        ],
        [
          3709316,
          899749,
          455269,
          14486350,
          800569,
          88769,
          295397
        ],
        [
          0,
          88,
          2,
          9,
          226695,
          0,
          0
        ],
        [
          3657627,
          5293801,
          7224,
          15373,
          44405,
          17693075,
          0
        ],
        [
          9744124,
          210860,
          866660,
          1121244,
          505150,
          184555,
          6659653
        ]
      ],
      "proposal_confusion": [
        [
          11001922,
          6499735,
          1916857,
          1660365,
          2572035,
          1966477,
          221171
        ],
        [
          42,
          1765671,
          47,
          31,
          1198,
          104325,
          35
        ],
        [
          941,
          7094,
          1195496,
          13476,
          106891,
          227,
          0
        ],
        [
          3353856,
          858891,
          557633,
          14392096,
          1135896,
          98076,
          338971
        ],
        [
          0,
          178,
          1,
          0,
          226615,
          0,
          0
        ],
        [
          3381614,
          5235835,
          3529,
          14164,
          51485,
          18024878,
          0
        ],
        [
          9195562,
          135776,
          981501,
          1174533,
          579519,
          207076,
          7018279
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 96000000,
      "changed": 2646559,
      "beneficial": 837114,
      "harmful": 866731,
      "wrong_to_wrong": 942714,
      "base_confusion": [
        [
          10907187,
          6551598,
          2362979,
          1464214,
          2320375,
          2030925,
          201284
        ],
        [
          45,
          1765289,
          38,
          122,
          1025,
          104784,
          46
        ],
        [
          689,
          9205,
          1213794,
          19036,
          81234,
          167,
          0
        ],
        [
          3709316,
          899749,
          455269,
          14486350,
          800569,
          88769,
          295397
        ],
        [
          0,
          88,
          2,
          9,
          226695,
          0,
          0
        ],
        [
          3657627,
          5293801,
          7224,
          15373,
          44405,
          17693075,
          0
        ],
        [
          9744124,
          210860,
          866660,
          1121244,
          505150,
          184555,
          6659653
        ]
      ],
      "proposal_confusion": [
        [
          10554374,
          6613788,
          2554179,
          1525614,
          2338883,
          2044428,
          207296
        ],
        [
          33,
          1767861,
          50,
          123,
          669,
          102593,
          20
        ],
        [
          411,
          8753,
          1222570,
          19414,
          72684,
          293,
          0
        ],
        [
          3569058,
          846994,
          452840,
          14656270,
          825016,
          85343,
          299898
        ],
        [
          0,
          3,
          2,
          22,
          226767,
          0,
          0
        ],
        [
          3527587,
          5308117,
          5863,
          15322,
          50193,
          17804423,
          0
        ],
        [
          9727438,
          228973,
          863158,
          1105872,
          494648,
          181996,
          6690161
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 96000000,
      "changed": 3230688,
      "beneficial": 988032,
      "harmful": 918886,
      "wrong_to_wrong": 1323770,
      "base_confusion": [
        [
          10907187,
          6551598,
          2362979,
          1464214,
          2320375,
          2030925,
          201284
        ],
        [
          45,
          1765289,
          38,
          122,
          1025,
          104784,
          46
        ],
        [
          689,
          9205,
          1213794,
          19036,
          81234,
          167,
          0
        ],
        [
          3709316,
          899749,
          455269,
          14486350,
          800569,
          88769,
          295397
        ],
        [
          0,
          88,
          2,
          9,
          226695,
          0,
          0
        ],
        [
          3657627,
          5293801,
          7224,
          15373,
          44405,
          17693075,
          0
        ],
        [
          9744124,
          210860,
          866660,
          1121244,
          505150,
          184555,
          6659653
        ]
      ],
      "proposal_confusion": [
        [
          11021330,
          6407948,
          2329649,
          1474514,
          2431281,
          1972157,
          201683
        ],
        [
          92,
          1763700,
          47,
          128,
          1209,
          106127,
          46
        ],
        [
          685,
          9255,
          1202661,
          18538,
          92810,
          176,
          0
        ],
        [
          3566727,
          879481,
          446150,
          14577101,
          877185,
          94291,
          294484
        ],
        [
          0,
          170,
          1,
          1,
          226622,
          0,
          0
        ],
        [
          3881395,
          5234333,
          7110,
          15328,
          58857,
          17514482,
          0
        ],
        [
          9635813,
          203482,
          1025581,
          1020393,
          492406,
          199278,
          6715293
        ]
      ]
    }
  }
}
```

### potsdam/potsdam

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | impervious surface | 51.6875 | 75.3909 | 62.1781 | 29.8644 |
| Geometry | building | 78.2026 | 80.6769 | 96.2262 | 14.5546 |
| Geometry | low vegetation | 33.4221 | 72.4703 | 38.2826 | 10.1329 |
| Geometry | tree | 62.5813 | 90.6538 | 66.8975 | 17.6734 |
| Geometry | car | 11.6582 | 11.6716 | 99.0262 | 19.8762 |
| Geometry | clutter | 4.5652 | 7.7447 | 10.0073 | 7.8986 |
| SCLIP_Two | impervious surface | 57.4153 | 78.1204 | 68.4172 | 31.7129 |
| SCLIP_Two | building | 70.5782 | 72.0195 | 97.2425 | 16.4763 |
| SCLIP_Two | low vegetation | 43.8570 | 70.7653 | 53.5614 | 14.5185 |
| SCLIP_Two | tree | 71.4698 | 87.2620 | 79.7946 | 21.9000 |
| SCLIP_Two | car | 17.4846 | 17.5371 | 98.3192 | 13.1340 |
| SCLIP_Two | clutter | 3.5103 | 12.5709 | 4.6442 | 2.2583 |
| VIPProxy_Two | impervious surface | 51.6478 | 80.5953 | 58.9823 | 26.5001 |
| VIPProxy_Two | building | 76.2272 | 77.9255 | 97.2204 | 15.2241 |
| VIPProxy_Two | low vegetation | 45.5683 | 71.2365 | 55.8431 | 15.0369 |
| VIPProxy_Two | tree | 67.1931 | 91.3751 | 71.7434 | 18.8040 |
| VIPProxy_Two | car | 10.8043 | 10.8078 | 99.7039 | 21.6117 |
| VIPProxy_Two | clutter | 4.9857 | 15.0312 | 6.9422 | 2.8232 |
| CSA_SameShell | impervious surface | 51.3505 | 74.2507 | 62.4761 | 30.4683 |
| CSA_SameShell | building | 78.1370 | 81.3675 | 95.1645 | 14.2718 |
| CSA_SameShell | low vegetation | 30.3483 | 67.7598 | 35.4701 | 10.0411 |
| CSA_SameShell | tree | 58.1182 | 89.9943 | 62.1330 | 16.5350 |
| CSA_SameShell | car | 11.1386 | 11.1500 | 99.0892 | 20.8193 |
| CSA_SameShell | clutter | 4.7302 | 8.0271 | 10.3276 | 7.8646 |
| Proxy_SameShell | impervious surface | 51.4312 | 74.8997 | 62.1416 | 30.0426 |
| Proxy_SameShell | building | 78.6903 | 80.8239 | 96.7542 | 14.6078 |
| Proxy_SameShell | low vegetation | 32.7271 | 73.0968 | 37.2089 | 9.7643 |
| Proxy_SameShell | tree | 63.2857 | 90.7172 | 67.6677 | 17.8644 |
| Proxy_SameShell | car | 11.5787 | 11.5920 | 99.0166 | 20.0107 |
| Proxy_SameShell | clutter | 4.5049 | 7.7283 | 9.7480 | 7.7103 |
| Geometry_SemanticBudget | impervious surface | 51.6452 | 75.4163 | 62.0996 | 29.8166 |
| Geometry_SemanticBudget | building | 77.9288 | 80.8190 | 95.6124 | 14.4363 |
| Geometry_SemanticBudget | low vegetation | 34.1416 | 72.2771 | 39.2864 | 10.4263 |
| Geometry_SemanticBudget | tree | 61.5538 | 89.8752 | 66.1402 | 17.6247 |
| Geometry_SemanticBudget | car | 11.5228 | 11.5355 | 99.0545 | 20.1165 |
| Geometry_SemanticBudget | clutter | 4.7713 | 8.2268 | 10.2008 | 7.5795 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 72,
    "Geometry_SemanticBudget__block0_value_read_norm": 22.214293347464668,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 67.06671158472697,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 10.974658025635613,
    "Geometry_SemanticBudget__block0_block_output_norm": 20.46340062883165,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5687102634045813,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8539886474609375,
    "Geometry_SemanticBudget__block0_dual_mean": 0.1992291369371944,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.219251275062561,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.8563658561971452,
    "Geometry_SemanticBudget__block0_geometry_budget": 1.9515852944718466,
    "Geometry_SemanticBudget__block0_constraint_violation": 6.639295154147678e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.751814736260308e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.37900759279727936,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.793206850687662e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 16.569419569439358,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 51.38418165842692,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 32.410208728578354,
    "Geometry_SemanticBudget__block1_block_output_norm": 454.5770547654894,
    "Geometry_SemanticBudget__block1_patch_mass": 0.36637522611353135,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9593777126736112,
    "Geometry_SemanticBudget__block1_dual_mean": 0.4700324485699336,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.465309984154171,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 1.9297217544582155,
    "Geometry_SemanticBudget__block1_geometry_budget": 1.9515852944718466,
    "Geometry_SemanticBudget__block1_constraint_violation": 3.195471233791775e-07,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.859434233771431e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.8632963912354575,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.983610577053494e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 20.954740524291992,
    "Geometry__block0_projected_increment_norm": 63.404365592532685,
    "Geometry__block0_attention_residual_norm": 10.826265931129456,
    "Geometry__block0_block_output_norm": 20.230289101600647,
    "Geometry__block0_patch_mass": 0.5687102634045813,
    "Geometry__block1_value_read_norm": 15.340062843428719,
    "Geometry__block1_projected_increment_norm": 48.2835709783766,
    "Geometry__block1_attention_residual_norm": 32.09179457028707,
    "Geometry__block1_block_output_norm": 450.475515153673,
    "Geometry__block1_patch_mass": 0.3716236592994796,
    "CSA_SameShell__block0_value_read_norm": 20.78235591782464,
    "CSA_SameShell__block0_projected_increment_norm": 63.83076307508681,
    "CSA_SameShell__block0_attention_residual_norm": 10.87654181321462,
    "CSA_SameShell__block0_block_output_norm": 20.43586892551846,
    "CSA_SameShell__block0_patch_mass": 0.5687102634045813,
    "CSA_SameShell__block0_row_mass_error": 4.544854164123535e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.936941610442268,
    "CSA_SameShell__block1_projected_increment_norm": 49.62127452426486,
    "CSA_SameShell__block1_attention_residual_norm": 32.40650020705329,
    "CSA_SameShell__block1_block_output_norm": 456.8888808356391,
    "CSA_SameShell__block1_patch_mass": 0.3670387425356441,
    "CSA_SameShell__block1_row_mass_error": 4.5034620496961804e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 19.671365790896946,
    "Proxy_SameShell__block0_projected_increment_norm": 60.44476980633206,
    "Proxy_SameShell__block0_attention_residual_norm": 10.759644773271349,
    "Proxy_SameShell__block0_block_output_norm": 20.27402114868164,
    "Proxy_SameShell__block0_patch_mass": 0.5687102634045813,
    "Proxy_SameShell__block0_row_mass_error": 2.1937820646497937e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 15.130773292647469,
    "Proxy_SameShell__block1_projected_increment_norm": 47.636519485049774,
    "Proxy_SameShell__block1_attention_residual_norm": 32.17512506908841,
    "Proxy_SameShell__block1_block_output_norm": 453.61373647054035,
    "Proxy_SameShell__block1_patch_mass": 0.3762037360833751,
    "Proxy_SameShell__block1_row_mass_error": 2.1937820646497937e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "potsdam__Geometry_SemanticBudget__top2_margin": 0.014762499848277204,
    "potsdam__Geometry__top2_margin": 0.014888499611212561,
    "potsdam__CSA_SameShell__top2_margin": 0.013279318958262188,
    "potsdam__Proxy_SameShell__top2_margin": 0.015014621668443497,
    "potsdam__SCLIP_Two__top2_margin": 0.016754726821091026,
    "potsdam__VIPProxy_Two__top2_margin": 0.019461873297890026
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 8000000,
      "changed": 1536541,
      "beneficial": 830940,
      "harmful": 186270,
      "wrong_to_wrong": 519331,
      "base_confusion": [
        [
          1801201,
          70872,
          2616,
          20088,
          970554,
          31511
        ],
        [
          6681,
          939373,
          12114,
          5629,
          7174,
          5242
        ],
        [
          247621,
          56128,
          587465,
          103653,
          164761,
          374920
        ],
        [
          99190,
          26399,
          107346,
          1281729,
          230034,
          171262
        ],
        [
          20,
          1786,
          0,
          2,
          185590,
          17
        ],
        [
          234435,
          69806,
          101088,
          2771,
          31984,
          48938
        ]
      ],
      "proposal_confusion": [
        [
          1981938,
          128437,
          11648,
          39520,
          729121,
          6178
        ],
        [
          6798,
          949294,
          12883,
          3311,
          3834,
          93
        ],
        [
          264842,
          107624,
          821926,
          176669,
          51307,
          112180
        ],
        [
          89864,
          42299,
          159753,
          1528832,
          55711,
          39501
        ],
        [
          189,
          2342,
          0,
          619,
          184265,
          0
        ],
        [
          193398,
          88110,
          155271,
          3052,
          26480,
          22711
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 8000000,
      "changed": 1258001,
      "beneficial": 554846,
      "harmful": 289117,
      "wrong_to_wrong": 414038,
      "base_confusion": [
        [
          1801201,
          70872,
          2616,
          20088,
          970554,
          31511
        ],
        [
          6681,
          939373,
          12114,
          5629,
          7174,
          5242
        ],
        [
          247621,
          56128,
          587465,
          103653,
          164761,
          374920
        ],
        [
          99190,
          26399,
          107346,
          1281729,
          230034,
          171262
        ],
        [
          20,
          1786,
          0,
          2,
          185590,
          17
        ],
        [
          234435,
          69806,
          101088,
          2771,
          31984,
          48938
        ]
      ],
      "proposal_confusion": [
        [
          1708625,
          82111,
          5927,
          24389,
          1061791,
          13999
        ],
        [
          5773,
          949078,
          12230,
          535,
          5685,
          2912
        ],
        [
          164381,
          73003,
          856939,
          103106,
          201675,
          135444
        ],
        [
          63175,
          30618,
          170987,
          1374574,
          237053,
          39553
        ],
        [
          30,
          255,
          0,
          270,
          186860,
          0
        ],
        [
          178021,
          82865,
          156867,
          1446,
          35874,
          33949
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 8000000,
      "changed": 557328,
      "beneficial": 113972,
      "harmful": 248465,
      "wrong_to_wrong": 194891,
      "base_confusion": [
        [
          1801201,
          70872,
          2616,
          20088,
          970554,
          31511
        ],
        [
          6681,
          939373,
          12114,
          5629,
          7174,
          5242
        ],
        [
          247621,
          56128,
          587465,
          103653,
          164761,
          374920
        ],
        [
          99190,
          26399,
          107346,
          1281729,
          230034,
          171262
        ],
        [
          20,
          1786,
          0,
          2,
          185590,
          17
        ],
        [
          234435,
          69806,
          101088,
          2771,
          31984,
          48938
        ]
      ],
      "proposal_confusion": [
        [
          1809833,
          68301,
          2376,
          21656,
          958409,
          36267
        ],
        [
          11957,
          929008,
          12123,
          9979,
          8135,
          5011
        ],
        [
          280502,
          51917,
          544306,
          98690,
          189629,
          369504
        ],
        [
          113256,
          24275,
          127203,
          1190444,
          292922,
          167860
        ],
        [
          29,
          1658,
          0,
          0,
          185708,
          20
        ],
        [
          221884,
          66584,
          117280,
          2031,
          30739,
          50504
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 8000000,
      "changed": 327909,
      "beneficial": 106196,
      "harmful": 105103,
      "wrong_to_wrong": 116610,
      "base_confusion": [
        [
          1801201,
          70872,
          2616,
          20088,
          970554,
          31511
        ],
        [
          6681,
          939373,
          12114,
          5629,
          7174,
          5242
        ],
        [
          247621,
          56128,
          587465,
          103653,
          164761,
          374920
        ],
        [
          99190,
          26399,
          107346,
          1281729,
          230034,
          171262
        ],
        [
          20,
          1786,
          0,
          2,
          185590,
          17
        ],
        [
          234435,
          69806,
          101088,
          2771,
          31984,
          48938
        ]
      ],
      "proposal_confusion": [
        [
          1800145,
          69399,
          2244,
          20972,
          974087,
          29995
        ],
        [
          6646,
          944527,
          11525,
          2390,
          6589,
          4536
        ],
        [
          259744,
          56659,
          570989,
          106891,
          167881,
          372384
        ],
        [
          96665,
          24930,
          99944,
          1296486,
          235703,
          162232
        ],
        [
          34,
          1804,
          0,
          0,
          185572,
          5
        ],
        [
          240172,
          71304,
          96439,
          2412,
          31025,
          47670
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 8000000,
      "changed": 291644,
      "beneficial": 92707,
      "harmful": 99080,
      "wrong_to_wrong": 99857,
      "base_confusion": [
        [
          1801201,
          70872,
          2616,
          20088,
          970554,
          31511
        ],
        [
          6681,
          939373,
          12114,
          5629,
          7174,
          5242
        ],
        [
          247621,
          56128,
          587465,
          103653,
          164761,
          374920
        ],
        [
          99190,
          26399,
          107346,
          1281729,
          230034,
          171262
        ],
        [
          20,
          1786,
          0,
          2,
          185590,
          17
        ],
        [
          234435,
          69806,
          101088,
          2771,
          31984,
          48938
        ]
      ],
      "proposal_confusion": [
        [
          1798928,
          69810,
          3242,
          22496,
          970365,
          32001
        ],
        [
          7695,
          933381,
          12315,
          9798,
          7955,
          5069
        ],
        [
          239468,
          54822,
          602868,
          107842,
          174566,
          354982
        ],
        [
          103480,
          26528,
          116155,
          1267219,
          238169,
          164409
        ],
        [
          21,
          1733,
          0,
          2,
          185643,
          16
        ],
        [
          235738,
          68629,
          99527,
          2619,
          32625,
          49884
        ]
      ]
    }
  }
}
```

### udd5/udd5

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | vegetation | 82.4937 | 96.4862 | 85.0488 | 26.1079 |
| Geometry | building | 83.9035 | 88.3604 | 94.3293 | 41.9106 |
| Geometry | road | 45.7856 | 70.6780 | 56.5218 | 10.7242 |
| Geometry | vehicle | 10.0092 | 10.0323 | 97.7477 | 7.7994 |
| Geometry | other | 30.5846 | 52.8537 | 42.0591 | 13.4578 |
| SCLIP_Two | vegetation | 82.5592 | 95.3084 | 86.0565 | 26.7437 |
| SCLIP_Two | building | 79.4095 | 80.3363 | 98.5680 | 48.1680 |
| SCLIP_Two | road | 48.0798 | 68.5748 | 61.6669 | 12.0593 |
| SCLIP_Two | vehicle | 17.3622 | 17.4634 | 96.7719 | 4.4359 |
| SCLIP_Two | other | 24.0076 | 57.4612 | 29.1967 | 8.5931 |
| VIPProxy_Two | vegetation | 83.7009 | 95.7948 | 86.8936 | 26.8668 |
| VIPProxy_Two | building | 82.9515 | 84.2758 | 98.1408 | 45.7173 |
| VIPProxy_Two | road | 47.8302 | 72.3449 | 58.5322 | 10.8498 |
| VIPProxy_Two | vehicle | 9.8318 | 9.8574 | 97.4315 | 7.9121 |
| VIPProxy_Two | other | 25.2212 | 59.5020 | 30.4478 | 8.6540 |
| CSA_SameShell | vegetation | 81.0999 | 96.5528 | 83.5181 | 25.6204 |
| CSA_SameShell | building | 83.7739 | 88.2935 | 94.2415 | 41.9033 |
| CSA_SameShell | road | 44.4665 | 69.8325 | 55.0392 | 10.5694 |
| CSA_SameShell | vehicle | 9.1264 | 9.1438 | 97.9599 | 8.5758 |
| CSA_SameShell | other | 30.1770 | 52.5895 | 41.4548 | 13.3311 |
| Proxy_SameShell | vegetation | 82.7936 | 96.5111 | 85.3481 | 26.1930 |
| Proxy_SameShell | building | 84.3474 | 88.2819 | 94.9814 | 42.2378 |
| Proxy_SameShell | road | 46.4248 | 70.7501 | 57.4516 | 10.8895 |
| Proxy_SameShell | vehicle | 9.8340 | 9.8572 | 97.6612 | 7.9309 |
| Proxy_SameShell | other | 30.2767 | 54.0698 | 40.7597 | 12.7487 |
| Geometry_SemanticBudget | vegetation | 82.3689 | 96.4487 | 84.9452 | 26.0862 |
| Geometry_SemanticBudget | building | 83.4670 | 88.5708 | 93.5420 | 41.4620 |
| Geometry_SemanticBudget | road | 45.5757 | 70.5943 | 56.2554 | 10.6863 |
| Geometry_SemanticBudget | vehicle | 9.7975 | 9.8190 | 97.8154 | 7.9744 |
| Geometry_SemanticBudget | other | 30.5013 | 52.0337 | 42.4318 | 13.7910 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 3232,
    "Geometry_SemanticBudget__block0_value_read_norm": 23.915867723450802,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 78.56766326238613,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 12.876118818132005,
    "Geometry_SemanticBudget__block0_block_output_norm": 29.949789089731652,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5021936073346008,
    "Geometry_SemanticBudget__block0_active_fraction": 0.853644588206074,
    "Geometry_SemanticBudget__block0_dual_mean": 0.22077563818599474,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.071190045464157,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.8331360324184494,
    "Geometry_SemanticBudget__block0_geometry_budget": 1.9204774923106231,
    "Geometry_SemanticBudget__block0_constraint_violation": 6.627697165649716e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.544485323499925e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.3761468736713033,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.6243393185115097e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 16.02207889326728,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 50.637640459702745,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 43.203131443203084,
    "Geometry_SemanticBudget__block1_block_output_norm": 554.2950383460168,
    "Geometry_SemanticBudget__block1_patch_mass": 0.3629856176563714,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9666563732789295,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5693484843883774,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.752895817898287,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 1.9031311330376286,
    "Geometry_SemanticBudget__block1_geometry_budget": 1.9204774923106231,
    "Geometry_SemanticBudget__block1_constraint_violation": 2.0345248798332593e-07,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.7569375226993373e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.9681279267009237,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.815214341229732e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 22.9041705751183,
    "Geometry__block0_projected_increment_norm": 75.60602682179744,
    "Geometry__block0_attention_residual_norm": 12.749062500967838,
    "Geometry__block0_block_output_norm": 29.853998131681198,
    "Geometry__block0_patch_mass": 0.5021936073346008,
    "Geometry__block1_value_read_norm": 14.87665915518704,
    "Geometry__block1_projected_increment_norm": 47.877318335051584,
    "Geometry__block1_attention_residual_norm": 42.94696671655863,
    "Geometry__block1_block_output_norm": 548.7238740543328,
    "Geometry__block1_patch_mass": 0.36966455165794726,
    "CSA_SameShell__block0_value_read_norm": 22.919990154776244,
    "CSA_SameShell__block0_projected_increment_norm": 76.39307206337995,
    "CSA_SameShell__block0_attention_residual_norm": 12.812259357459475,
    "CSA_SameShell__block0_block_output_norm": 29.93883735472613,
    "CSA_SameShell__block0_patch_mass": 0.5021936073346008,
    "CSA_SameShell__block0_row_mass_error": 4.455041472274478e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.730930889892106,
    "CSA_SameShell__block1_projected_increment_norm": 49.64263077891699,
    "CSA_SameShell__block1_attention_residual_norm": 43.17846157645235,
    "CSA_SameShell__block1_block_output_norm": 556.2763818608652,
    "CSA_SameShell__block1_patch_mass": 0.36445960992172655,
    "CSA_SameShell__block1_row_mass_error": 4.3790603038107994e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 22.24554317835534,
    "Proxy_SameShell__block0_projected_increment_norm": 74.1322803544526,
    "Proxy_SameShell__block0_attention_residual_norm": 12.721558544010218,
    "Proxy_SameShell__block0_block_output_norm": 29.854359564804795,
    "Proxy_SameShell__block0_patch_mass": 0.5021936073346008,
    "Proxy_SameShell__block0_row_mass_error": 2.124153151370511e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.855260555401888,
    "Proxy_SameShell__block1_projected_increment_norm": 47.76976835609663,
    "Proxy_SameShell__block1_attention_residual_norm": 42.9668276374883,
    "Proxy_SameShell__block1_block_output_norm": 550.1482682652993,
    "Proxy_SameShell__block1_patch_mass": 0.37237667189081114,
    "Proxy_SameShell__block1_row_mass_error": 2.124153151370511e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "udd5__Geometry_SemanticBudget__top2_margin": 0.017884879992839116,
    "udd5__Geometry__top2_margin": 0.018074508735867988,
    "udd5__CSA_SameShell__top2_margin": 0.017424842324364344,
    "udd5__Proxy_SameShell__top2_margin": 0.018383530249224404,
    "udd5__SCLIP_Two__top2_margin": 0.022837328298487367,
    "udd5__VIPProxy_Two__top2_margin": 0.02612360679444518
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 439956480,
      "changed": 55893218,
      "beneficial": 19525900,
      "harmful": 17460627,
      "wrong_to_wrong": 18906691,
      "base_confusion": [
        [
          110827432,
          4488169,
          3210091,
          4600571,
          7184181
        ],
        [
          297922,
          162926222,
          433754,
          1101020,
          7961716
        ],
        [
          415179,
          2322716,
          33347264,
          10149123,
          12764612
        ],
        [
          48200,
          15194,
          11731,
          3442486,
          4195
        ],
        [
          3274822,
          14636079,
          10179123,
          15020747,
          31293931
        ]
      ],
      "proposal_confusion": [
        [
          112140627,
          6554470,
          2756636,
          2186860,
          6671851
        ],
        [
          258091,
          170247343,
          319859,
          562456,
          1332885
        ],
        [
          565458,
          9179048,
          36382812,
          4795159,
          8076417
        ],
        [
          54236,
          27404,
          31027,
          3408117,
          1022
        ],
        [
          4642400,
          25910086,
          13565287,
          8563220,
          21723709
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 439956480,
      "changed": 46484892,
      "beneficial": 17302278,
      "harmful": 15779463,
      "wrong_to_wrong": 13403151,
      "base_confusion": [
        [
          110827432,
          4488169,
          3210091,
          4600571,
          7184181
        ],
        [
          297922,
          162926222,
          433754,
          1101020,
          7961716
        ],
        [
          415179,
          2322716,
          33347264,
          10149123,
          12764612
        ],
        [
          48200,
          15194,
          11731,
          3442486,
          4195
        ],
        [
          3274822,
          14636079,
          10179123,
          15020747,
          31293931
        ]
      ],
      "proposal_confusion": [
        [
          113231475,
          4852085,
          2245909,
          3682557,
          6298418
        ],
        [
          279354,
          169509347,
          240100,
          982341,
          1709492
        ],
        [
          468914,
          6346377,
          34533370,
          10239562,
          7410671
        ],
        [
          52836,
          20324,
          16806,
          3431348,
          492
        ],
        [
          4169521,
          20408212,
          10698186,
          16474173,
          22654610
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 439956480,
      "changed": 16462724,
      "beneficial": 4552514,
      "harmful": 8015588,
      "wrong_to_wrong": 3894622,
      "base_confusion": [
        [
          110827432,
          4488169,
          3210091,
          4600571,
          7184181
        ],
        [
          297922,
          162926222,
          433754,
          1101020,
          7961716
        ],
        [
          415179,
          2322716,
          33347264,
          10149123,
          12764612
        ],
        [
          48200,
          15194,
          11731,
          3442486,
          4195
        ],
        [
          3274822,
          14636079,
          10179123,
          15020747,
          31293931
        ]
      ],
      "proposal_confusion": [
        [
          108832841,
          4748978,
          3902087,
          6062603,
          6763935
        ],
        [
          268560,
          162774568,
          410813,
          1241303,
          8025390
        ],
        [
          396318,
          2310809,
          32472547,
          10805294,
          13013926
        ],
        [
          41708,
          14130,
          12430,
          3449959,
          3579
        ],
        [
          3179073,
          14507760,
          9702730,
          16170793,
          30844346
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 439956480,
      "changed": 8875045,
      "beneficial": 3796102,
      "harmful": 2701251,
      "wrong_to_wrong": 2377692,
      "base_confusion": [
        [
          110827432,
          4488169,
          3210091,
          4600571,
          7184181
        ],
        [
          297922,
          162926222,
          433754,
          1101020,
          7961716
        ],
        [
          415179,
          2322716,
          33347264,
          10149123,
          12764612
        ],
        [
          48200,
          15194,
          11731,
          3442486,
          4195
        ],
        [
          3274822,
          14636079,
          10179123,
          15020747,
          31293931
        ]
      ],
      "proposal_confusion": [
        [
          111217428,
          4500566,
          3146035,
          4647492,
          6798923
        ],
        [
          283978,
          164052391,
          405125,
          1076842,
          6902298
        ],
        [
          406912,
          2328859,
          33895791,
          10311001,
          12056331
        ],
        [
          49650,
          15690,
          12841,
          3439438,
          4187
        ],
        [
          3280015,
          14930347,
          10449374,
          15417828,
          30327138
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 439956480,
      "changed": 8691877,
      "beneficial": 2521778,
      "harmful": 3894199,
      "wrong_to_wrong": 2275900,
      "base_confusion": [
        [
          110827432,
          4488169,
          3210091,
          4600571,
          7184181
        ],
        [
          297922,
          162926222,
          433754,
          1101020,
          7961716
        ],
        [
          415179,
          2322716,
          33347264,
          10149123,
          12764612
        ],
        [
          48200,
          15194,
          11731,
          3442486,
          4195
        ],
        [
          3274822,
          14636079,
          10179123,
          15020747,
          31293931
        ]
      ],
      "proposal_confusion": [
        [
          110692403,
          4462202,
          3254718,
          4972866,
          6928255
        ],
        [
          308989,
          161566319,
          450412,
          1184593,
          9210321
        ],
        [
          403692,
          2207630,
          33190071,
          10236360,
          12961141
        ],
        [
          48383,
          14431,
          10496,
          3444868,
          3628
        ],
        [
          3314662,
          14164240,
          10109520,
          15245027,
          31571253
        ]
      ]
    }
  }
}
```

### oem/oem

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | bareland | 0.0000 | 0.0000 | NA | 10.4255 |
| Geometry | rangeland | 47.7780 | 65.2945 | 64.0414 | 18.7786 |
| Geometry | developed space | 28.5294 | 61.5617 | 34.7130 | 12.6303 |
| Geometry | road | 40.8291 | 45.7530 | 79.1399 | 7.8925 |
| Geometry | tree | 56.0657 | 87.4475 | 60.9727 | 17.7308 |
| Geometry | water | 0.0000 | 0.0000 | 0.0000 | 0.0647 |
| Geometry | agriculture land | 79.1353 | 90.6413 | 86.1766 | 15.9338 |
| Geometry | building | 62.4971 | 65.5844 | 92.9954 | 16.5437 |
| SCLIP_Two | bareland | 0.0000 | 0.0000 | NA | 12.8127 |
| SCLIP_Two | rangeland | 49.3017 | 67.4631 | 64.6816 | 18.3566 |
| SCLIP_Two | developed space | 18.6439 | 66.9189 | 20.5366 | 6.8741 |
| SCLIP_Two | road | 45.0181 | 52.1441 | 76.7127 | 6.7128 |
| SCLIP_Two | tree | 52.9925 | 86.9020 | 57.5925 | 16.8530 |
| SCLIP_Two | water | 0.0000 | 0.0000 | 0.0000 | 0.0266 |
| SCLIP_Two | agriculture land | 71.0357 | 74.2694 | 94.2246 | 21.2623 |
| SCLIP_Two | building | 61.5041 | 64.0625 | 93.9027 | 17.1019 |
| VIPProxy_Two | bareland | 0.0000 | 0.0000 | NA | 12.7114 |
| VIPProxy_Two | rangeland | 50.9587 | 69.8417 | 65.3353 | 17.9107 |
| VIPProxy_Two | developed space | 20.2836 | 66.1441 | 22.6334 | 7.6646 |
| VIPProxy_Two | road | 40.8836 | 45.4639 | 80.2296 | 8.0520 |
| VIPProxy_Two | tree | 58.2973 | 86.5758 | 64.0907 | 18.8252 |
| VIPProxy_Two | water | 0.0000 | 0.0000 | 0.0000 | 0.0231 |
| VIPProxy_Two | agriculture land | 77.9662 | 84.5386 | 90.9326 | 18.0269 |
| VIPProxy_Two | building | 63.5299 | 65.8517 | 94.7420 | 16.7860 |
| CSA_SameShell | bareland | 0.0000 | 0.0000 | NA | 9.9496 |
| CSA_SameShell | rangeland | 47.4716 | 63.4232 | 65.3674 | 19.7330 |
| CSA_SameShell | developed space | 30.2842 | 61.2922 | 37.4459 | 13.6846 |
| CSA_SameShell | road | 42.0180 | 47.8262 | 77.5780 | 7.4014 |
| CSA_SameShell | tree | 55.0387 | 87.9519 | 59.5267 | 17.2111 |
| CSA_SameShell | water | 0.0000 | 0.0000 | 0.0000 | 0.0616 |
| CSA_SameShell | agriculture land | 78.2836 | 90.2704 | 85.4976 | 15.8732 |
| CSA_SameShell | building | 62.9308 | 66.6395 | 91.8751 | 16.0856 |
| Proxy_SameShell | bareland | 0.0000 | 0.0000 | NA | 10.5711 |
| Proxy_SameShell | rangeland | 47.6056 | 65.4440 | 63.5901 | 18.6037 |
| Proxy_SameShell | developed space | 27.8791 | 61.7798 | 33.6898 | 12.2147 |
| Proxy_SameShell | road | 40.5450 | 45.4641 | 78.9358 | 7.9222 |
| Proxy_SameShell | tree | 56.8005 | 87.0006 | 62.0682 | 18.1421 |
| Proxy_SameShell | water | 0.0000 | 0.0000 | 0.0000 | 0.0579 |
| Proxy_SameShell | agriculture land | 78.9768 | 90.6616 | 85.9704 | 15.8921 |
| Proxy_SameShell | building | 62.9984 | 65.8210 | 93.6268 | 16.5961 |
| Geometry_SemanticBudget | bareland | 0.0000 | 0.0000 | NA | 10.1942 |
| Geometry_SemanticBudget | rangeland | 47.3546 | 63.8972 | 64.6532 | 19.3726 |
| Geometry_SemanticBudget | developed space | 29.6326 | 63.0857 | 35.8485 | 12.7284 |
| Geometry_SemanticBudget | road | 41.1116 | 45.8890 | 79.7937 | 7.9341 |
| Geometry_SemanticBudget | tree | 56.2520 | 87.3978 | 61.2175 | 17.8121 |
| Geometry_SemanticBudget | water | 0.0000 | 0.0000 | 0.0000 | 0.0684 |
| Geometry_SemanticBudget | agriculture land | 76.6484 | 90.5542 | 83.3092 | 15.4184 |
| Geometry_SemanticBudget | building | 62.5391 | 65.7302 | 92.7964 | 16.4717 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 67,
    "Geometry_SemanticBudget__block0_value_read_norm": 23.134835399798494,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 70.61629856166554,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 11.90111304041165,
    "Geometry_SemanticBudget__block0_block_output_norm": 24.56378270618951,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5609431346850609,
    "Geometry_SemanticBudget__block0_active_fraction": 0.858523240729944,
    "Geometry_SemanticBudget__block0_dual_mean": 0.20784926392249206,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.411906640921066,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 2.0415785917595253,
    "Geometry_SemanticBudget__block0_geometry_budget": 2.1435859843866147,
    "Geometry_SemanticBudget__block0_constraint_violation": 5.586823420738106e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.643824563097598e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.3930539328660538,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.7238905038406597e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 15.437304795677981,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 47.21902055882696,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 34.68611415464487,
    "Geometry_SemanticBudget__block1_block_output_norm": 474.6192235234958,
    "Geometry_SemanticBudget__block1_patch_mass": 0.3509156410373859,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9686543478894589,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5091371175958149,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.642889926682657,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 2.125923496573719,
    "Geometry_SemanticBudget__block1_geometry_budget": 2.1435859843866147,
    "Geometry_SemanticBudget__block1_constraint_violation": 2.490940378673041e-07,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.928503463517374e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.898996024879057,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.857333738412431e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 21.972687934761616,
    "Geometry__block0_projected_increment_norm": 66.99199727755874,
    "Geometry__block0_attention_residual_norm": 11.773802671859514,
    "Geometry__block0_block_output_norm": 24.33236347027679,
    "Geometry__block0_patch_mass": 0.5609431346850609,
    "Geometry__block1_value_read_norm": 14.172577687163852,
    "Geometry__block1_projected_increment_norm": 44.13133809103895,
    "Geometry__block1_attention_residual_norm": 34.430028630726376,
    "Geometry__block1_block_output_norm": 470.48079077877213,
    "Geometry__block1_patch_mass": 0.3528768478044823,
    "CSA_SameShell__block0_value_read_norm": 21.99998098344945,
    "CSA_SameShell__block0_projected_increment_norm": 67.88459476072397,
    "CSA_SameShell__block0_attention_residual_norm": 11.826734642484295,
    "CSA_SameShell__block0_block_output_norm": 24.548287007346083,
    "CSA_SameShell__block0_patch_mass": 0.5609431346850609,
    "CSA_SameShell__block0_row_mass_error": 4.572654837992654e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.067655648758162,
    "CSA_SameShell__block1_projected_increment_norm": 45.945343017578125,
    "CSA_SameShell__block1_attention_residual_norm": 34.6246054919798,
    "CSA_SameShell__block1_block_output_norm": 477.0591457993237,
    "CSA_SameShell__block1_patch_mass": 0.3506236659057105,
    "CSA_SameShell__block1_row_mass_error": 4.465900250335238e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 21.759429618493833,
    "Proxy_SameShell__block0_projected_increment_norm": 66.48564034077658,
    "Proxy_SameShell__block0_attention_residual_norm": 11.773679249322237,
    "Proxy_SameShell__block0_block_output_norm": 24.323044421067877,
    "Proxy_SameShell__block0_patch_mass": 0.5609431346850609,
    "Proxy_SameShell__block0_row_mass_error": 2.108403106233967e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.233915471318943,
    "Proxy_SameShell__block1_projected_increment_norm": 44.22372197393161,
    "Proxy_SameShell__block1_attention_residual_norm": 34.43794745829568,
    "Proxy_SameShell__block1_block_output_norm": 470.9356880757346,
    "Proxy_SameShell__block1_patch_mass": 0.35411021513725394,
    "Proxy_SameShell__block1_row_mass_error": 2.108403106233967e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "oem__Geometry_SemanticBudget__top2_margin": 0.017117049934259103,
    "oem__Geometry__top2_margin": 0.016927472141974452,
    "oem__CSA_SameShell__top2_margin": 0.016356409576012573,
    "oem__Proxy_SameShell__top2_margin": 0.017145861155673195,
    "oem__SCLIP_Two__top2_margin": 0.016659801916233195,
    "oem__VIPProxy_Two__top2_margin": 0.020902642938516923
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 7676869,
      "changed": 1203086,
      "beneficial": 327137,
      "harmful": 524319,
      "wrong_to_wrong": 351630,
      "base_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          86334,
          941291,
          122607,
          80142,
          116356,
          3342,
          54144,
          65600
        ],
        [
          616637,
          3678,
          596910,
          187902,
          25181,
          81,
          6484,
          282684
        ],
        [
          3772,
          14,
          43377,
          277216,
          8510,
          0,
          0,
          17397
        ],
        [
          20224,
          420079,
          149810,
          46618,
          1190311,
          1077,
          53402,
          70682
        ],
        [
          1347,
          0,
          561,
          272,
          43,
          0,
          416,
          90
        ],
        [
          71315,
          76475,
          4417,
          7720,
          16815,
          470,
          1108741,
          639
        ],
        [
          721,
          72,
          51931,
          6027,
          3956,
          0,
          32,
          832947
        ]
      ],
      "proposal_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          94939,
          950700,
          46086,
          41360,
          109913,
          1869,
          151878,
          73071
        ],
        [
          804281,
          4499,
          353139,
          173077,
          36252,
          62,
          40963,
          307284
        ],
        [
          8278,
          49,
          41506,
          268714,
          11168,
          0,
          1508,
          19063
        ],
        [
          12046,
          449661,
          47872,
          24843,
          1124322,
          110,
          221607,
          71742
        ],
        [
          238,
          0,
          0,
          0,
          0,
          0,
          2417,
          74
        ],
        [
          61589,
          4164,
          579,
          2450,
          4937,
          0,
          1212286,
          587
        ],
        [
          2243,
          142,
          38530,
          4886,
          7189,
          0,
          1623,
          841073
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 7676869,
      "changed": 839660,
      "beneficial": 287878,
      "harmful": 335054,
      "wrong_to_wrong": 216728,
      "base_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          86334,
          941291,
          122607,
          80142,
          116356,
          3342,
          54144,
          65600
        ],
        [
          616637,
          3678,
          596910,
          187902,
          25181,
          81,
          6484,
          282684
        ],
        [
          3772,
          14,
          43377,
          277216,
          8510,
          0,
          0,
          17397
        ],
        [
          20224,
          420079,
          149810,
          46618,
          1190311,
          1077,
          53402,
          70682
        ],
        [
          1347,
          0,
          561,
          272,
          43,
          0,
          416,
          90
        ],
        [
          71315,
          76475,
          4417,
          7720,
          16815,
          470,
          1108741,
          639
        ],
        [
          721,
          72,
          51931,
          6027,
          3956,
          0,
          32,
          832947
        ]
      ],
      "proposal_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          97312,
          960309,
          67760,
          73977,
          123274,
          1605,
          79610,
          65969
        ],
        [
          760566,
          5277,
          389194,
          216863,
          37797,
          129,
          16125,
          293606
        ],
        [
          4935,
          0,
          35544,
          281033,
          10857,
          0,
          246,
          17671
        ],
        [
          22437,
          400934,
          63600,
          35046,
          1251181,
          40,
          116638,
          62327
        ],
        [
          942,
          0,
          229,
          34,
          203,
          0,
          1224,
          97
        ],
        [
          87493,
          8281,
          697,
          5117,
          14693,
          0,
          1169932,
          379
        ],
        [
          2155,
          178,
          31379,
          6075,
          7180,
          0,
          128,
          848591
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 7676869,
      "changed": 376594,
      "beneficial": 154938,
      "harmful": 140925,
      "wrong_to_wrong": 80731,
      "base_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          86334,
          941291,
          122607,
          80142,
          116356,
          3342,
          54144,
          65600
        ],
        [
          616637,
          3678,
          596910,
          187902,
          25181,
          81,
          6484,
          282684
        ],
        [
          3772,
          14,
          43377,
          277216,
          8510,
          0,
          0,
          17397
        ],
        [
          20224,
          420079,
          149810,
          46618,
          1190311,
          1077,
          53402,
          70682
        ],
        [
          1347,
          0,
          561,
          272,
          43,
          0,
          416,
          90
        ],
        [
          71315,
          76475,
          4417,
          7720,
          16815,
          470,
          1108741,
          639
        ],
        [
          721,
          72,
          51931,
          6027,
          3956,
          0,
          32,
          832947
        ]
      ],
      "proposal_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          77717,
          960781,
          125968,
          70822,
          112468,
          3067,
          56639,
          62354
        ],
        [
          605983,
          4022,
          643904,
          169500,
          21509,
          134,
          7793,
          266712
        ],
        [
          4764,
          7,
          48707,
          271745,
          7947,
          0,
          35,
          17081
        ],
        [
          18763,
          444395,
          162880,
          44389,
          1162082,
          1109,
          53673,
          64912
        ],
        [
          1418,
          0,
          607,
          184,
          34,
          0,
          397,
          89
        ],
        [
          54347,
          105505,
          5551,
          6468,
          13482,
          422,
          1100005,
          812
        ],
        [
          823,
          163,
          62931,
          5085,
          3748,
          0,
          24,
          822912
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 7676869,
      "changed": 182131,
      "beneficial": 69652,
      "harmful": 70206,
      "wrong_to_wrong": 42273,
      "base_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          86334,
          941291,
          122607,
          80142,
          116356,
          3342,
          54144,
          65600
        ],
        [
          616637,
          3678,
          596910,
          187902,
          25181,
          81,
          6484,
          282684
        ],
        [
          3772,
          14,
          43377,
          277216,
          8510,
          0,
          0,
          17397
        ],
        [
          20224,
          420079,
          149810,
          46618,
          1190311,
          1077,
          53402,
          70682
        ],
        [
          1347,
          0,
          561,
          272,
          43,
          0,
          416,
          90
        ],
        [
          71315,
          76475,
          4417,
          7720,
          16815,
          470,
          1108741,
          639
        ],
        [
          721,
          72,
          51931,
          6027,
          3956,
          0,
          32,
          832947
        ]
      ],
      "proposal_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          85168,
          934658,
          121128,
          82803,
          121823,
          2971,
          54738,
          66527
        ],
        [
          629226,
          3740,
          579315,
          190290,
          27348,
          84,
          6405,
          283149
        ],
        [
          3799,
          0,
          44207,
          276501,
          8941,
          0,
          0,
          16838
        ],
        [
          20605,
          411050,
          142666,
          44637,
          1211698,
          911,
          52334,
          68302
        ],
        [
          1364,
          0,
          527,
          172,
          150,
          0,
          429,
          87
        ],
        [
          70646,
          78650,
          3936,
          7757,
          18474,
          482,
          1106088,
          559
        ],
        [
          721,
          82,
          45930,
          6015,
          4312,
          0,
          24,
          838602
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 7676869,
      "changed": 235036,
      "beneficial": 90133,
      "harmful": 93221,
      "wrong_to_wrong": 51682,
      "base_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          86334,
          941291,
          122607,
          80142,
          116356,
          3342,
          54144,
          65600
        ],
        [
          616637,
          3678,
          596910,
          187902,
          25181,
          81,
          6484,
          282684
        ],
        [
          3772,
          14,
          43377,
          277216,
          8510,
          0,
          0,
          17397
        ],
        [
          20224,
          420079,
          149810,
          46618,
          1190311,
          1077,
          53402,
          70682
        ],
        [
          1347,
          0,
          561,
          272,
          43,
          0,
          416,
          90
        ],
        [
          71315,
          76475,
          4417,
          7720,
          16815,
          470,
          1108741,
          639
        ],
        [
          721,
          72,
          51931,
          6027,
          3956,
          0,
          32,
          832947
        ]
      ],
      "proposal_confusion": [
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          84939,
          950283,
          116743,
          78138,
          115416,
          3476,
          53864,
          66957
        ],
        [
          595828,
          3833,
          616436,
          191610,
          25596,
          121,
          6735,
          279398
        ],
        [
          3708,
          11,
          40930,
          279506,
          8848,
          0,
          35,
          17248
        ],
        [
          20046,
          425927,
          144011,
          46429,
          1195090,
          1179,
          50735,
          68786
        ],
        [
          1329,
          0,
          589,
          262,
          27,
          0,
          416,
          106
        ],
        [
          75973,
          107081,
          5229,
          7262,
          17872,
          477,
          1071849,
          849
        ],
        [
          776,
          72,
          53203,
          5885,
          4566,
          0,
          20,
          831164
        ]
      ]
    }
  }
}
```

### loveda/P

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | building | 64.4987 | 65.3671 | 97.9819 | 0.9799 |
| Geometry | road | 66.9348 | 67.6522 | 98.4404 | 10.3611 |
| Geometry | water | 81.1851 | 86.7638 | 92.6613 | 22.3129 |
| Geometry | barren | 24.3403 | 63.8735 | 28.2261 | 3.2301 |
| Geometry | tree | 53.7307 | 64.5184 | 76.2666 | 13.0831 |
| Geometry | farm | 86.2626 | 95.3302 | 90.0685 | 50.0330 |
| SCLIP_Two | building | 81.9391 | 87.2865 | 93.0435 | 0.6969 |
| SCLIP_Two | road | 76.4665 | 78.4733 | 96.7639 | 8.7802 |
| SCLIP_Two | water | 80.2856 | 86.3214 | 91.9886 | 22.2644 |
| SCLIP_Two | barren | 17.0004 | 90.4737 | 17.3102 | 1.3985 |
| SCLIP_Two | tree | 65.3470 | 95.2548 | 67.5458 | 7.8482 |
| SCLIP_Two | farm | 88.6741 | 89.1740 | 99.3719 | 59.0118 |
| VIPProxy_Two | building | 73.9883 | 76.3965 | 95.9137 | 0.8208 |
| VIPProxy_Two | road | 73.6544 | 74.8283 | 97.9146 | 9.3174 |
| VIPProxy_Two | water | 81.5227 | 86.4865 | 93.4228 | 22.5683 |
| VIPProxy_Two | barren | 21.9817 | 79.5581 | 23.2976 | 2.1405 |
| VIPProxy_Two | tree | 68.0388 | 87.2678 | 75.5372 | 9.5800 |
| VIPProxy_Two | farm | 92.2293 | 93.6980 | 98.3288 | 55.5730 |
| CSA_SameShell | building | 61.2582 | 62.1391 | 97.7380 | 1.0283 |
| CSA_SameShell | road | 66.3845 | 67.0915 | 98.4373 | 10.4473 |
| CSA_SameShell | water | 79.5166 | 86.0333 | 91.3027 | 22.1724 |
| CSA_SameShell | barren | 24.3703 | 66.9118 | 27.7097 | 3.0270 |
| CSA_SameShell | tree | 48.3333 | 57.4978 | 75.2011 | 14.4755 |
| CSA_SameShell | farm | 84.4123 | 95.3951 | 87.9980 | 48.8495 |
| Proxy_SameShell | building | 64.9951 | 66.0060 | 97.6979 | 0.9677 |
| Proxy_SameShell | road | 67.9789 | 68.7157 | 98.4471 | 10.2014 |
| Proxy_SameShell | water | 81.4974 | 86.8428 | 92.9777 | 22.3687 |
| Proxy_SameShell | barren | 24.4744 | 63.0115 | 28.5805 | 3.3154 |
| Proxy_SameShell | tree | 54.6839 | 65.4452 | 76.8820 | 13.0019 |
| Proxy_SameShell | farm | 86.4849 | 95.3523 | 90.2911 | 50.1450 |
| Geometry_SemanticBudget | building | 61.5336 | 62.2736 | 98.1055 | 1.0299 |
| Geometry_SemanticBudget | road | 66.4943 | 67.2349 | 98.3704 | 10.4180 |
| Geometry_SemanticBudget | water | 80.6190 | 86.8272 | 91.8536 | 22.1022 |
| Geometry_SemanticBudget | barren | 24.3964 | 61.3227 | 28.8330 | 3.4368 |
| Geometry_SemanticBudget | tree | 49.3120 | 58.2938 | 76.1930 | 14.4662 |
| Geometry_SemanticBudget | farm | 84.0600 | 95.4873 | 87.5375 | 48.5470 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 72,
    "Geometry_SemanticBudget__block0_value_read_norm": 23.58332856496175,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 76.64720641242133,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 12.983729150560167,
    "Geometry_SemanticBudget__block0_block_output_norm": 29.42327446407742,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5334829708768262,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8768870035807291,
    "Geometry_SemanticBudget__block0_dual_mean": 0.22446523047983646,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.325243890285492,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.9363527811235852,
    "Geometry_SemanticBudget__block0_geometry_budget": 2.017666972345776,
    "Geometry_SemanticBudget__block0_constraint_violation": 6.424056159125434e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.2799446317884656e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.41133425757288933,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.271666208902995e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 15.857200900713602,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 50.41946252187093,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 43.87646582391527,
    "Geometry_SemanticBudget__block1_block_output_norm": 538.5160098605686,
    "Geometry_SemanticBudget__block1_patch_mass": 0.35465586102671093,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9799202813042535,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5554580063455634,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.696582780943976,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 2.007908042934206,
    "Geometry_SemanticBudget__block1_geometry_budget": 2.017666972345776,
    "Geometry_SemanticBudget__block1_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.478626781039768e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.9689203856719865,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.4620699352688257e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 22.432170814938015,
    "Geometry__block0_projected_increment_norm": 72.8132922914293,
    "Geometry__block0_attention_residual_norm": 12.827689939075047,
    "Geometry__block0_block_output_norm": 29.28160614437527,
    "Geometry__block0_patch_mass": 0.5334829708768262,
    "Geometry__block1_value_read_norm": 14.804318692949083,
    "Geometry__block1_projected_increment_norm": 47.94527451197306,
    "Geometry__block1_attention_residual_norm": 43.67246174812317,
    "Geometry__block1_block_output_norm": 533.6929228040907,
    "Geometry__block1_patch_mass": 0.35707269691758686,
    "CSA_SameShell__block0_value_read_norm": 22.489052030775284,
    "CSA_SameShell__block0_projected_increment_norm": 73.99394920137193,
    "CSA_SameShell__block0_attention_residual_norm": 12.893738945325216,
    "CSA_SameShell__block0_block_output_norm": 29.390914254718357,
    "CSA_SameShell__block0_patch_mass": 0.5334829708768262,
    "CSA_SameShell__block0_row_mass_error": 4.0315919452243383e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.590613643328348,
    "CSA_SameShell__block1_projected_increment_norm": 49.561045699649384,
    "CSA_SameShell__block1_attention_residual_norm": 43.843663136164345,
    "CSA_SameShell__block1_block_output_norm": 540.6719754536947,
    "CSA_SameShell__block1_patch_mass": 0.3556673973798752,
    "CSA_SameShell__block1_row_mass_error": 4.056427213880751e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 22.513049920399983,
    "Proxy_SameShell__block0_projected_increment_norm": 72.8556867705451,
    "Proxy_SameShell__block0_attention_residual_norm": 12.827329052819145,
    "Proxy_SameShell__block0_block_output_norm": 29.22859552171495,
    "Proxy_SameShell__block0_patch_mass": 0.5334829708768262,
    "Proxy_SameShell__block0_row_mass_error": 2.1275546815660266e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.917470071050856,
    "Proxy_SameShell__block1_projected_increment_norm": 48.227918518914116,
    "Proxy_SameShell__block1_attention_residual_norm": 43.64742475085788,
    "Proxy_SameShell__block1_block_output_norm": 533.8818007575142,
    "Proxy_SameShell__block1_patch_mass": 0.35808451515105033,
    "Proxy_SameShell__block1_row_mass_error": 2.1275546815660266e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "P__Geometry_SemanticBudget__top2_margin": 0.017104596158282623,
    "P__Geometry__top2_margin": 0.01698107534321025,
    "P__CSA_SameShell__top2_margin": 0.016249815073226474,
    "P__Proxy_SameShell__top2_margin": 0.017331883537634794,
    "P__SCLIP_Two__top2_margin": 0.02144083958895256,
    "P__VIPProxy_Two__top2_margin": 0.022840312513936725,
    "D__Geometry_SemanticBudget__top2_margin": 0.013676064218290977,
    "D__Geometry__top2_margin": 0.013350982982147899,
    "D__CSA_SameShell__top2_margin": 0.013013227875085754,
    "D__Proxy_SameShell__top2_margin": 0.013630153372004215,
    "D__SCLIP_Two__top2_margin": 0.020226401360964194,
    "D__VIPProxy_Two__top2_margin": 0.021011004819431238
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 4577998,
      "changed": 480472,
      "beneficial": 254714,
      "harmful": 123264,
      "wrong_to_wrong": 102494,
      "base_confusion": [
        [
          29325,
          0,
          84,
          4,
          508,
          8
        ],
        [
          515,
          320894,
          0,
          7,
          2248,
          2314
        ],
        [
          147,
          19720,
          886277,
          4659,
          8098,
          37568
        ],
        [
          8676,
          18959,
          85065,
          94451,
          62851,
          64621
        ],
        [
          2294,
          82686,
          31287,
          1535,
          386430,
          2451
        ],
        [
          3905,
          32070,
          18769,
          47216,
          138810,
          2183546
        ]
      ],
      "proposal_confusion": [
        [
          27847,
          0,
          76,
          40,
          376,
          1590
        ],
        [
          541,
          315429,
          0,
          516,
          4343,
          5149
        ],
        [
          5,
          14655,
          879842,
          2102,
          2581,
          57284
        ],
        [
          1744,
          8345,
          92332,
          57924,
          7239,
          167039
        ],
        [
          1511,
          62592,
          37932,
          995,
          342243,
          61410
        ],
        [
          255,
          936,
          9081,
          2446,
          2510,
          2409088
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 4577998,
      "changed": 322985,
      "beneficial": 226462,
      "harmful": 41445,
      "wrong_to_wrong": 55078,
      "base_confusion": [
        [
          29325,
          0,
          84,
          4,
          508,
          8
        ],
        [
          515,
          320894,
          0,
          7,
          2248,
          2314
        ],
        [
          147,
          19720,
          886277,
          4659,
          8098,
          37568
        ],
        [
          8676,
          18959,
          85065,
          94451,
          62851,
          64621
        ],
        [
          2294,
          82686,
          31287,
          1535,
          386430,
          2451
        ],
        [
          3905,
          32070,
          18769,
          47216,
          138810,
          2183546
        ]
      ],
      "proposal_confusion": [
        [
          28706,
          24,
          74,
          87,
          920,
          118
        ],
        [
          557,
          319180,
          0,
          286,
          2871,
          3084
        ],
        [
          0,
          13372,
          893560,
          3611,
          3249,
          42677
        ],
        [
          2675,
          14789,
          97848,
          77959,
          41627,
          99725
        ],
        [
          1287,
          74969,
          31258,
          1709,
          382734,
          14726
        ],
        [
          4350,
          4216,
          10438,
          14338,
          7173,
          2383801
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 4577998,
      "changed": 151721,
      "beneficial": 22084,
      "harmful": 92485,
      "wrong_to_wrong": 37152,
      "base_confusion": [
        [
          29325,
          0,
          84,
          4,
          508,
          8
        ],
        [
          515,
          320894,
          0,
          7,
          2248,
          2314
        ],
        [
          147,
          19720,
          886277,
          4659,
          8098,
          37568
        ],
        [
          8676,
          18959,
          85065,
          94451,
          62851,
          64621
        ],
        [
          2294,
          82686,
          31287,
          1535,
          386430,
          2451
        ],
        [
          3905,
          32070,
          18769,
          47216,
          138810,
          2183546
        ]
      ],
      "proposal_confusion": [
        [
          29252,
          0,
          53,
          12,
          604,
          8
        ],
        [
          466,
          320884,
          0,
          26,
          2209,
          2393
        ],
        [
          222,
          22819,
          873282,
          4901,
          20825,
          34420
        ],
        [
          10436,
          17830,
          90460,
          92723,
          59254,
          63920
        ],
        [
          2805,
          84985,
          34153,
          1469,
          381031,
          2240
        ],
        [
          3894,
          31760,
          17103,
          39444,
          198765,
          2133350
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 4577998,
      "changed": 60889,
      "beneficial": 29716,
      "harmful": 17053,
      "wrong_to_wrong": 14120,
      "base_confusion": [
        [
          29325,
          0,
          84,
          4,
          508,
          8
        ],
        [
          515,
          320894,
          0,
          7,
          2248,
          2314
        ],
        [
          147,
          19720,
          886277,
          4659,
          8098,
          37568
        ],
        [
          8676,
          18959,
          85065,
          94451,
          62851,
          64621
        ],
        [
          2294,
          82686,
          31287,
          1535,
          386430,
          2451
        ],
        [
          3905,
          32070,
          18769,
          47216,
          138810,
          2183546
        ]
      ],
      "proposal_confusion": [
        [
          29240,
          0,
          86,
          8,
          577,
          18
        ],
        [
          532,
          320916,
          18,
          10,
          2183,
          2319
        ],
        [
          120,
          18646,
          889303,
          5725,
          6303,
          36372
        ],
        [
          8054,
          17096,
          87221,
          95637,
          61065,
          65550
        ],
        [
          2144,
          80704,
          30446,
          1406,
          389548,
          2435
        ],
        [
          4209,
          29658,
          16964,
          48991,
          135552,
          2188942
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 4577998,
      "changed": 120275,
      "beneficial": 15209,
      "harmful": 82828,
      "wrong_to_wrong": 22238,
      "base_confusion": [
        [
          29325,
          0,
          84,
          4,
          508,
          8
        ],
        [
          515,
          320894,
          0,
          7,
          2248,
          2314
        ],
        [
          147,
          19720,
          886277,
          4659,
          8098,
          37568
        ],
        [
          8676,
          18959,
          85065,
          94451,
          62851,
          64621
        ],
        [
          2294,
          82686,
          31287,
          1535,
          386430,
          2451
        ],
        [
          3905,
          32070,
          18769,
          47216,
          138810,
          2183546
        ]
      ],
      "proposal_confusion": [
        [
          29362,
          0,
          47,
          5,
          507,
          8
        ],
        [
          411,
          320666,
          0,
          27,
          2480,
          2394
        ],
        [
          209,
          21676,
          878551,
          7196,
          15376,
          33461
        ],
        [
          9373,
          20433,
          81704,
          96482,
          64797,
          61834
        ],
        [
          3020,
          82301,
          31118,
          1591,
          386057,
          2596
        ],
        [
          4775,
          31858,
          20419,
          52034,
          193044,
          2122186
        ]
      ]
    }
  }
}
```

### loveda/D

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | background | 34.1983 | 65.7504 | 41.6109 | 28.2483 |
| Geometry | building | 31.1862 | 31.8582 | 93.6650 | 1.0641 |
| Geometry | road | 45.4818 | 45.8348 | 98.3352 | 8.4577 |
| Geometry | water | 61.9193 | 65.9551 | 91.0065 | 15.9605 |
| Geometry | barren | 14.5296 | 59.0631 | 16.1567 | 1.1070 |
| Geometry | tree | 26.8904 | 30.0530 | 71.8726 | 14.6543 |
| Geometry | farm | 54.9851 | 69.5721 | 72.3948 | 30.5080 |
| SCLIP_Two | background | 4.2646 | 81.0134 | 4.3077 | 2.3734 |
| SCLIP_Two | building | 34.3661 | 35.2758 | 93.0201 | 0.9544 |
| SCLIP_Two | road | 58.1585 | 59.3120 | 96.7642 | 6.4315 |
| SCLIP_Two | water | 62.7413 | 66.3750 | 91.9746 | 16.0283 |
| SCLIP_Two | barren | 15.1893 | 55.8776 | 17.2594 | 1.2500 |
| SCLIP_Two | tree | 35.8056 | 43.3994 | 67.1736 | 9.4843 |
| SCLIP_Two | farm | 45.0045 | 45.3715 | 98.2348 | 63.4782 |
| VIPProxy_Two | background | 8.0534 | 81.4682 | 8.2036 | 4.4947 |
| VIPProxy_Two | building | 31.9498 | 32.6046 | 94.0860 | 1.0445 |
| VIPProxy_Two | road | 49.6986 | 50.2313 | 97.9109 | 7.6842 |
| VIPProxy_Two | water | 61.4631 | 64.2668 | 93.3724 | 16.8056 |
| VIPProxy_Two | barren | 19.1467 | 54.0645 | 22.8666 | 1.7116 |
| VIPProxy_Two | tree | 25.4426 | 27.7908 | 75.0694 | 16.5520 |
| VIPProxy_Two | farm | 53.9147 | 54.8907 | 96.8076 | 51.7074 |
| CSA_SameShell | background | 35.6517 | 64.6929 | 44.2645 | 30.5410 |
| CSA_SameShell | building | 30.9665 | 31.6989 | 93.0569 | 1.0625 |
| CSA_SameShell | road | 46.2367 | 46.6011 | 98.3370 | 8.3188 |
| CSA_SameShell | water | 60.3122 | 65.5624 | 88.2788 | 15.5749 |
| CSA_SameShell | barren | 13.0414 | 61.3714 | 14.2076 | 0.9368 |
| CSA_SameShell | tree | 26.3491 | 29.8077 | 69.4272 | 14.2722 |
| CSA_SameShell | farm | 54.2948 | 70.4078 | 70.3483 | 29.2937 |
| Proxy_SameShell | background | 33.1467 | 65.6084 | 40.1172 | 27.2933 |
| Proxy_SameShell | building | 31.2439 | 32.0645 | 92.4287 | 1.0433 |
| Proxy_SameShell | road | 45.4800 | 45.8318 | 98.3404 | 8.4587 |
| Proxy_SameShell | water | 62.3500 | 65.8614 | 92.1226 | 16.1792 |
| Proxy_SameShell | barren | 14.5207 | 58.7771 | 16.1671 | 1.1131 |
| Proxy_SameShell | tree | 26.7701 | 29.7193 | 72.9559 | 15.0422 |
| Proxy_SameShell | farm | 54.9305 | 69.1278 | 72.7863 | 30.8701 |
| Geometry_SemanticBudget | background | 34.7027 | 64.5584 | 42.8700 | 29.6405 |
| Geometry_SemanticBudget | building | 30.3131 | 30.8764 | 94.3232 | 1.1057 |
| Geometry_SemanticBudget | road | 45.2612 | 45.6225 | 98.2803 | 8.4923 |
| Geometry_SemanticBudget | water | 61.3456 | 65.8648 | 89.9404 | 15.7952 |
| Geometry_SemanticBudget | barren | 14.5988 | 58.9580 | 16.2502 | 1.1154 |
| Geometry_SemanticBudget | tree | 26.5677 | 29.7218 | 71.4573 | 14.7319 |
| Geometry_SemanticBudget | farm | 53.8571 | 70.2491 | 69.7710 | 29.1190 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 72,
    "Geometry_SemanticBudget__block0_value_read_norm": 23.58332856496175,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 76.64720641242133,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 12.983729150560167,
    "Geometry_SemanticBudget__block0_block_output_norm": 29.42327446407742,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5334829708768262,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8768870035807291,
    "Geometry_SemanticBudget__block0_dual_mean": 0.22446523047983646,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.325243890285492,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.9363527811235852,
    "Geometry_SemanticBudget__block0_geometry_budget": 2.017666972345776,
    "Geometry_SemanticBudget__block0_constraint_violation": 6.424056159125434e-07,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.2799446317884656e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.41133425757288933,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.271666208902995e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 15.857200900713602,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 50.41946252187093,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 43.87646582391527,
    "Geometry_SemanticBudget__block1_block_output_norm": 538.5160098605686,
    "Geometry_SemanticBudget__block1_patch_mass": 0.35465586102671093,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9799202813042535,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5554580063455634,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.696582780943976,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 2.007908042934206,
    "Geometry_SemanticBudget__block1_geometry_budget": 2.017666972345776,
    "Geometry_SemanticBudget__block1_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.478626781039768e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.9689203856719865,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.4620699352688257e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 22.432170814938015,
    "Geometry__block0_projected_increment_norm": 72.8132922914293,
    "Geometry__block0_attention_residual_norm": 12.827689939075047,
    "Geometry__block0_block_output_norm": 29.28160614437527,
    "Geometry__block0_patch_mass": 0.5334829708768262,
    "Geometry__block1_value_read_norm": 14.804318692949083,
    "Geometry__block1_projected_increment_norm": 47.94527451197306,
    "Geometry__block1_attention_residual_norm": 43.67246174812317,
    "Geometry__block1_block_output_norm": 533.6929228040907,
    "Geometry__block1_patch_mass": 0.35707269691758686,
    "CSA_SameShell__block0_value_read_norm": 22.489052030775284,
    "CSA_SameShell__block0_projected_increment_norm": 73.99394920137193,
    "CSA_SameShell__block0_attention_residual_norm": 12.893738945325216,
    "CSA_SameShell__block0_block_output_norm": 29.390914254718357,
    "CSA_SameShell__block0_patch_mass": 0.5334829708768262,
    "CSA_SameShell__block0_row_mass_error": 4.0315919452243383e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.590613643328348,
    "CSA_SameShell__block1_projected_increment_norm": 49.561045699649384,
    "CSA_SameShell__block1_attention_residual_norm": 43.843663136164345,
    "CSA_SameShell__block1_block_output_norm": 540.6719754536947,
    "CSA_SameShell__block1_patch_mass": 0.3556673973798752,
    "CSA_SameShell__block1_row_mass_error": 4.056427213880751e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 22.513049920399983,
    "Proxy_SameShell__block0_projected_increment_norm": 72.8556867705451,
    "Proxy_SameShell__block0_attention_residual_norm": 12.827329052819145,
    "Proxy_SameShell__block0_block_output_norm": 29.22859552171495,
    "Proxy_SameShell__block0_patch_mass": 0.5334829708768262,
    "Proxy_SameShell__block0_row_mass_error": 2.1275546815660266e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.917470071050856,
    "Proxy_SameShell__block1_projected_increment_norm": 48.227918518914116,
    "Proxy_SameShell__block1_attention_residual_norm": 43.64742475085788,
    "Proxy_SameShell__block1_block_output_norm": 533.8818007575142,
    "Proxy_SameShell__block1_patch_mass": 0.35808451515105033,
    "Proxy_SameShell__block1_row_mass_error": 2.1275546815660266e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "P__Geometry_SemanticBudget__top2_margin": 0.017104596158282623,
    "P__Geometry__top2_margin": 0.01698107534321025,
    "P__CSA_SameShell__top2_margin": 0.016249815073226474,
    "P__Proxy_SameShell__top2_margin": 0.017331883537634794,
    "P__SCLIP_Two__top2_margin": 0.02144083958895256,
    "P__VIPProxy_Two__top2_margin": 0.022840312513936725,
    "D__Geometry_SemanticBudget__top2_margin": 0.013676064218290977,
    "D__Geometry__top2_margin": 0.013350982982147899,
    "D__CSA_SameShell__top2_margin": 0.013013227875085754,
    "D__Proxy_SameShell__top2_margin": 0.013630153372004215,
    "D__SCLIP_Two__top2_margin": 0.020226401360964194,
    "D__VIPProxy_Two__top2_margin": 0.021011004819431238
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 8268895,
      "changed": 3076906,
      "beneficial": 710321,
      "harmful": 1476872,
      "wrong_to_wrong": 889713,
      "base_confusion": [
        [
          1535814,
          53276,
          254700,
          333937,
          27917,
          780369,
          704884
        ],
        [
          1571,
          28033,
          0,
          83,
          0,
          242,
          0
        ],
        [
          640,
          502,
          320551,
          0,
          0,
          2191,
          2094
        ],
        [
          49411,
          91,
          18944,
          870449,
          613,
          3335,
          13626
        ],
        [
          99191,
          3269,
          14882,
          70767,
          54064,
          46554,
          45896
        ],
        [
          30422,
          1486,
          79391,
          29891,
          229,
          364166,
          1098
        ],
        [
          618774,
          1336,
          10894,
          14633,
          8713,
          14888,
          1755078
        ]
      ],
      "proposal_confusion": [
        [
          158992,
          47055,
          130840,
          306985,
          41793,
          427588,
          2577644
        ],
        [
          45,
          27840,
          0,
          76,
          38,
          376,
          1554
        ],
        [
          0,
          541,
          315430,
          0,
          515,
          4343,
          5149
        ],
        [
          663,
          5,
          14515,
          879709,
          2099,
          2447,
          57031
        ],
        [
          1102,
          1744,
          8340,
          92096,
          57754,
          7243,
          166344
        ],
        [
          5640,
          1486,
          61754,
          37419,
          328,
          340357,
          59699
        ],
        [
          29812,
          250,
          936,
          9076,
          831,
          1889,
          2381522
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 8268895,
      "changed": 2362042,
      "beneficial": 682543,
      "harmful": 1263703,
      "wrong_to_wrong": 415796,
      "base_confusion": [
        [
          1535814,
          53276,
          254700,
          333937,
          27917,
          780369,
          704884
        ],
        [
          1571,
          28033,
          0,
          83,
          0,
          242,
          0
        ],
        [
          640,
          502,
          320551,
          0,
          0,
          2191,
          2094
        ],
        [
          49411,
          91,
          18944,
          870449,
          613,
          3335,
          13626
        ],
        [
          99191,
          3269,
          14882,
          70767,
          54064,
          46554,
          45896
        ],
        [
          30422,
          1486,
          79391,
          29891,
          229,
          364166,
          1098
        ],
        [
          618774,
          1336,
          10894,
          14633,
          8713,
          14888,
          1755078
        ]
      ],
      "proposal_confusion": [
        [
          302788,
          52203,
          210820,
          358379,
          58564,
          934999,
          1773144
        ],
        [
          739,
          28159,
          3,
          67,
          86,
          827,
          48
        ],
        [
          151,
          557,
          319168,
          0,
          286,
          2872,
          2944
        ],
        [
          2598,
          0,
          13164,
          893078,
          2615,
          3227,
          41787
        ],
        [
          5940,
          2388,
          14753,
          97264,
          76517,
          41447,
          96314
        ],
        [
          5505,
          1147,
          73763,
          30495,
          938,
          380364,
          14471
        ],
        [
          53943,
          1911,
          3726,
          10358,
          2523,
          4934,
          2346921
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 8268895,
      "changed": 495253,
      "beneficial": 202061,
      "harmful": 198911,
      "wrong_to_wrong": 94281,
      "base_confusion": [
        [
          1535814,
          53276,
          254700,
          333937,
          27917,
          780369,
          704884
        ],
        [
          1571,
          28033,
          0,
          83,
          0,
          242,
          0
        ],
        [
          640,
          502,
          320551,
          0,
          0,
          2191,
          2094
        ],
        [
          49411,
          91,
          18944,
          870449,
          613,
          3335,
          13626
        ],
        [
          99191,
          3269,
          14882,
          70767,
          54064,
          46554,
          45896
        ],
        [
          30422,
          1486,
          79391,
          29891,
          229,
          364166,
          1098
        ],
        [
          618774,
          1336,
          10894,
          14633,
          8713,
          14888,
          1755078
        ]
      ],
      "proposal_confusion": [
        [
          1633756,
          52026,
          241503,
          320472,
          22574,
          763809,
          656757
        ],
        [
          1773,
          27851,
          0,
          53,
          0,
          252,
          0
        ],
        [
          829,
          460,
          320557,
          0,
          25,
          2165,
          1942
        ],
        [
          70598,
          141,
          20905,
          844359,
          548,
          7090,
          12828
        ],
        [
          109083,
          3994,
          14540,
          76130,
          47542,
          39330,
          44004
        ],
        [
          37106,
          1992,
          81541,
          32824,
          173,
          351776,
          1271
        ],
        [
          672259,
          1397,
          8829,
          14033,
          6604,
          15730,
          1705464
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 8268895,
      "changed": 216805,
      "beneficial": 72729,
      "harmful": 102521,
      "wrong_to_wrong": 41555,
      "base_confusion": [
        [
          1535814,
          53276,
          254700,
          333937,
          27917,
          780369,
          704884
        ],
        [
          1571,
          28033,
          0,
          83,
          0,
          242,
          0
        ],
        [
          640,
          502,
          320551,
          0,
          0,
          2191,
          2094
        ],
        [
          49411,
          91,
          18944,
          870449,
          613,
          3335,
          13626
        ],
        [
          99191,
          3269,
          14882,
          70767,
          54064,
          46554,
          45896
        ],
        [
          30422,
          1486,
          79391,
          29891,
          229,
          364166,
          1098
        ],
        [
          618774,
          1336,
          10894,
          14633,
          8713,
          14888,
          1755078
        ]
      ],
      "proposal_confusion": [
        [
          1480684,
          52331,
          259123,
          340383,
          29344,
          808030,
          721002
        ],
        [
          1897,
          27663,
          0,
          83,
          0,
          281,
          5
        ],
        [
          588,
          511,
          320568,
          18,
          0,
          2139,
          2154
        ],
        [
          39158,
          76,
          18170,
          881124,
          474,
          3014,
          14453
        ],
        [
          93379,
          3064,
          14635,
          73091,
          54099,
          47148,
          49207
        ],
        [
          27731,
          1346,
          77587,
          28937,
          198,
          369655,
          1229
        ],
        [
          613414,
          1282,
          9362,
          14209,
          7926,
          13553,
          1764570
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 8268895,
      "changed": 344126,
      "beneficial": 127109,
      "harmful": 156213,
      "wrong_to_wrong": 60804,
      "base_confusion": [
        [
          1535814,
          53276,
          254700,
          333937,
          27917,
          780369,
          704884
        ],
        [
          1571,
          28033,
          0,
          83,
          0,
          242,
          0
        ],
        [
          640,
          502,
          320551,
          0,
          0,
          2191,
          2094
        ],
        [
          49411,
          91,
          18944,
          870449,
          613,
          3335,
          13626
        ],
        [
          99191,
          3269,
          14882,
          70767,
          54064,
          46554,
          45896
        ],
        [
          30422,
          1486,
          79391,
          29891,
          229,
          364166,
          1098
        ],
        [
          618774,
          1336,
          10894,
          14633,
          8713,
          14888,
          1755078
        ]
      ],
      "proposal_confusion": [
        [
          1582288,
          55057,
          255982,
          332352,
          28676,
          779725,
          656817
        ],
        [
          1382,
          28230,
          0,
          47,
          0,
          270,
          0
        ],
        [
          682,
          411,
          320372,
          0,
          20,
          2437,
          2056
        ],
        [
          56089,
          132,
          19944,
          860252,
          738,
          6761,
          12553
        ],
        [
          100935,
          3927,
          15513,
          68475,
          54377,
          47752,
          43644
        ],
        [
          31694,
          2189,
          79849,
          29423,
          188,
          362062,
          1278
        ],
        [
          677870,
          1483,
          10563,
          15538,
          8231,
          19161,
          1691470
        ]
      ]
    }
  }
}
```

### vaihingen/vaihingen

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | impervious surface | 49.2002 | 82.3971 | 54.9789 | 20.4283 |
| Geometry | building | 74.6610 | 75.5001 | 98.5334 | 28.1288 |
| Geometry | low vegetation | 46.7115 | 92.0732 | 48.6687 | 13.4530 |
| Geometry | tree | 71.8329 | 82.5509 | 84.6923 | 21.4401 |
| Geometry | car | 8.7566 | 8.7725 | 97.9748 | 16.5499 |
| SCLIP_Two | impervious surface | 59.7107 | 77.2669 | 72.4361 | 28.7018 |
| SCLIP_Two | building | 75.0953 | 75.5964 | 99.1250 | 28.2616 |
| SCLIP_Two | low vegetation | 38.0841 | 90.7828 | 39.6159 | 11.1063 |
| SCLIP_Two | tree | 69.6927 | 77.1815 | 87.7790 | 23.7674 |
| SCLIP_Two | car | 17.8532 | 17.8987 | 98.5964 | 8.1629 |
| VIPProxy_Two | impervious surface | 48.0542 | 83.9554 | 52.9136 | 19.2959 |
| VIPProxy_Two | building | 77.4157 | 77.9089 | 99.1889 | 27.4405 |
| VIPProxy_Two | low vegetation | 46.6803 | 93.2045 | 48.3250 | 13.1959 |
| VIPProxy_Two | tree | 71.9355 | 79.7531 | 88.0077 | 23.0609 |
| VIPProxy_Two | car | 8.6109 | 8.6190 | 98.9182 | 17.0068 |
| CSA_SameShell | impervious surface | 48.8467 | 77.4530 | 56.9438 | 22.5090 |
| CSA_SameShell | building | 74.0064 | 75.6839 | 97.0921 | 27.6500 |
| CSA_SameShell | low vegetation | 42.5891 | 92.9668 | 44.0070 | 12.0475 |
| CSA_SameShell | tree | 72.1908 | 84.8160 | 82.9052 | 20.4272 |
| CSA_SameShell | car | 8.3478 | 8.3621 | 97.9986 | 17.3664 |
| Proxy_SameShell | impervious surface | 49.0833 | 83.8046 | 54.2270 | 19.8105 |
| Proxy_SameShell | building | 75.4368 | 76.0519 | 98.9392 | 28.0397 |
| Proxy_SameShell | low vegetation | 46.3746 | 93.1654 | 48.0079 | 13.1148 |
| Proxy_SameShell | tree | 72.5312 | 81.7709 | 86.5210 | 22.1119 |
| Proxy_SameShell | car | 8.5462 | 8.5627 | 97.7888 | 16.9232 |
| Geometry_SemanticBudget | impervious surface | 50.2689 | 81.5470 | 56.7210 | 21.2953 |
| Geometry_SemanticBudget | building | 74.8478 | 75.9599 | 98.0814 | 27.8303 |
| Geometry_SemanticBudget | low vegetation | 46.6601 | 92.3619 | 48.5328 | 13.3735 |
| Geometry_SemanticBudget | tree | 71.5425 | 82.7143 | 84.1191 | 21.2529 |
| Geometry_SemanticBudget | car | 8.9269 | 8.9428 | 98.0546 | 16.2481 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 72,
    "Geometry_SemanticBudget__block0_value_read_norm": 22.543510489993626,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 67.92763132519192,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 11.007022711965773,
    "Geometry_SemanticBudget__block0_block_output_norm": 21.817564911312527,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5822423017687268,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8552508884006076,
    "Geometry_SemanticBudget__block0_dual_mean": 0.19226728359030354,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.261717269817988,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.8807797729969025,
    "Geometry_SemanticBudget__block0_geometry_budget": 1.9822881801260843,
    "Geometry_SemanticBudget__block0_constraint_violation": 1.006656222873264e-06,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.942218462626139e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.37504783231351113,
    "Geometry_SemanticBudget__block0_row_mass_error": 5.049837960137261e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 17.75395917892456,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 53.79280943340726,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 32.81473244561089,
    "Geometry_SemanticBudget__block1_block_output_norm": 458.59498511420355,
    "Geometry_SemanticBudget__block1_patch_mass": 0.4254739027884271,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9551671346028646,
    "Geometry_SemanticBudget__block1_dual_mean": 0.45790519523951745,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.671645475758447,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 1.9562575452857547,
    "Geometry_SemanticBudget__block1_geometry_budget": 1.9822881801260843,
    "Geometry_SemanticBudget__block1_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 5.231963263617622e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.8802611968583531,
    "Geometry_SemanticBudget__block1_row_mass_error": 5.347861184014214e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 21.244371202256943,
    "Geometry__block0_projected_increment_norm": 64.08103656768799,
    "Geometry__block0_attention_residual_norm": 10.853283339076572,
    "Geometry__block0_block_output_norm": 21.597646686765884,
    "Geometry__block0_patch_mass": 0.5822423017687268,
    "Geometry__block1_value_read_norm": 16.085408051808674,
    "Geometry__block1_projected_increment_norm": 49.706981129116485,
    "Geometry__block1_attention_residual_norm": 32.47320890426636,
    "Geometry__block1_block_output_norm": 453.4621760050456,
    "Geometry__block1_patch_mass": 0.42847733696301776,
    "CSA_SameShell__block0_value_read_norm": 21.070738368564182,
    "CSA_SameShell__block0_projected_increment_norm": 64.59587245517307,
    "CSA_SameShell__block0_attention_residual_norm": 10.90876951482561,
    "CSA_SameShell__block0_block_output_norm": 21.806080102920532,
    "CSA_SameShell__block0_patch_mass": 0.5822423017687268,
    "CSA_SameShell__block0_row_mass_error": 4.884269502427843e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 16.79817023542192,
    "CSA_SameShell__block1_projected_increment_norm": 51.149819956885445,
    "CSA_SameShell__block1_attention_residual_norm": 32.72847557067871,
    "CSA_SameShell__block1_block_output_norm": 460.594480726454,
    "CSA_SameShell__block1_patch_mass": 0.42570532320274246,
    "CSA_SameShell__block1_row_mass_error": 4.851155810885959e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 19.781888167063396,
    "Proxy_SameShell__block0_projected_increment_norm": 60.653639263576935,
    "Proxy_SameShell__block0_attention_residual_norm": 10.78741529252794,
    "Proxy_SameShell__block0_block_output_norm": 21.692755142847698,
    "Proxy_SameShell__block0_patch_mass": 0.5822423017687268,
    "Proxy_SameShell__block0_row_mass_error": 2.1109978357950845e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 15.724407156308493,
    "Proxy_SameShell__block1_projected_increment_norm": 48.61370664172702,
    "Proxy_SameShell__block1_attention_residual_norm": 32.562858210669624,
    "Proxy_SameShell__block1_block_output_norm": 457.42355982462567,
    "Proxy_SameShell__block1_patch_mass": 0.43207106532322037,
    "Proxy_SameShell__block1_row_mass_error": 2.1109978357950845e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "vaihingen__Geometry_SemanticBudget__top2_margin": 0.01880562358484086,
    "vaihingen__Geometry__top2_margin": 0.018853321542135544,
    "vaihingen__CSA_SameShell__top2_margin": 0.016875430902776618,
    "vaihingen__Proxy_SameShell__top2_margin": 0.019146965894227225,
    "vaihingen__SCLIP_Two__top2_margin": 0.01986545727898677,
    "vaihingen__VIPProxy_Two__top2_margin": 0.023893216767141387
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 7947220,
      "changed": 1294393,
      "beneficial": 641830,
      "harmful": 338052,
      "wrong_to_wrong": 314511,
      "base_confusion": [
        [
          1337700,
          189335,
          5437,
          57143,
          843499
        ],
        [
          13611,
          1687772,
          553,
          8093,
          2865
        ],
        [
          237382,
          284408,
          984393,
          232053,
          284403
        ],
        [
          34769,
          71598,
          78759,
          1406575,
          69106
        ],
        [
          17,
          2344,
          0,
          24,
          115381
        ]
      ],
      "proposal_confusion": [
        [
          1762453,
          188786,
          9479,
          76880,
          395516
        ],
        [
          4590,
          1697906,
          639,
          8969,
          790
        ],
        [
          469102,
          291383,
          801287,
          345039,
          115828
        ],
        [
          44307,
          66969,
          71215,
          1457840,
          20476
        ],
        [
          542,
          971,
          22,
          118,
          116113
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 7947220,
      "changed": 852451,
      "beneficial": 316630,
      "harmful": 306432,
      "wrong_to_wrong": 229389,
      "base_confusion": [
        [
          1337700,
          189335,
          5437,
          57143,
          843499
        ],
        [
          13611,
          1687772,
          553,
          8093,
          2865
        ],
        [
          237382,
          284408,
          984393,
          232053,
          284403
        ],
        [
          34769,
          71598,
          78759,
          1406575,
          69106
        ],
        [
          17,
          2344,
          0,
          24,
          115381
        ]
      ],
      "proposal_confusion": [
        [
          1287448,
          168402,
          8013,
          60602,
          908649
        ],
        [
          4064,
          1699001,
          569,
          7395,
          1865
        ],
        [
          220335,
          250765,
          977440,
          302805,
          271294
        ],
        [
          21629,
          61589,
          62683,
          1461638,
          53268
        ],
        [
          15,
          996,
          0,
          263,
          116492
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 7947220,
      "changed": 495276,
      "beneficial": 132583,
      "harmful": 233405,
      "wrong_to_wrong": 129288,
      "base_confusion": [
        [
          1337700,
          189335,
          5437,
          57143,
          843499
        ],
        [
          13611,
          1687772,
          553,
          8093,
          2865
        ],
        [
          237382,
          284408,
          984393,
          232053,
          284403
        ],
        [
          34769,
          71598,
          78759,
          1406575,
          69106
        ],
        [
          17,
          2344,
          0,
          24,
          115381
        ]
      ],
      "proposal_confusion": [
        [
          1385508,
          186886,
          3413,
          37838,
          819469
        ],
        [
          36348,
          1663084,
          430,
          9688,
          3344
        ],
        [
          305576,
          272097,
          890102,
          198957,
          355907
        ],
        [
          61379,
          73021,
          63496,
          1376896,
          86015
        ],
        [
          26,
          2319,
          0,
          12,
          115409
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 7947220,
      "changed": 317734,
      "beneficial": 122798,
      "harmful": 117358,
      "wrong_to_wrong": 77578,
      "base_confusion": [
        [
          1337700,
          189335,
          5437,
          57143,
          843499
        ],
        [
          13611,
          1687772,
          553,
          8093,
          2865
        ],
        [
          237382,
          284408,
          984393,
          232053,
          284403
        ],
        [
          34769,
          71598,
          78759,
          1406575,
          69106
        ],
        [
          17,
          2344,
          0,
          24,
          115381
        ]
      ],
      "proposal_confusion": [
        [
          1319404,
          189373,
          4935,
          55221,
          864181
        ],
        [
          7357,
          1694723,
          458,
          7854,
          2502
        ],
        [
          217274,
          275089,
          971026,
          257219,
          302031
        ],
        [
          30313,
          66662,
          65841,
          1436946,
          61045
        ],
        [
          33,
          2529,
          0,
          42,
          115162
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 7947220,
      "changed": 250624,
      "beneficial": 101097,
      "harmful": 78626,
      "wrong_to_wrong": 70901,
      "base_confusion": [
        [
          1337700,
          189335,
          5437,
          57143,
          843499
        ],
        [
          13611,
          1687772,
          553,
          8093,
          2865
        ],
        [
          237382,
          284408,
          984393,
          232053,
          284403
        ],
        [
          34769,
          71598,
          78759,
          1406575,
          69106
        ],
        [
          17,
          2344,
          0,
          24,
          115381
        ]
      ],
      "proposal_confusion": [
        [
          1380087,
          183943,
          4939,
          51959,
          812186
        ],
        [
          19573,
          1680031,
          738,
          9252,
          3300
        ],
        [
          252645,
          274399,
          981643,
          230730,
          283222
        ],
        [
          40069,
          71094,
          75502,
          1397056,
          77086
        ],
        [
          8,
          2267,
          0,
          16,
          115475
        ]
      ]
    }
  }
}
```

### landcoverai/landcoverai

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | background | 82.6090 | 96.6505 | 85.0437 | 59.6226 |
| Geometry | building | 34.9562 | 35.0436 | 99.2919 | 4.2927 |
| Geometry | woodland | 78.3638 | 86.4992 | 89.2841 | 22.0960 |
| Geometry | water | 93.6635 | 93.6635 | 100.0000 | 8.8309 |
| Geometry | road | 14.9318 | 15.6288 | 77.0019 | 5.1578 |
| SCLIP_Two | background | 80.8528 | 97.4390 | 82.6083 | 57.4466 |
| SCLIP_Two | building | 37.9242 | 38.1118 | 98.7190 | 3.9244 |
| SCLIP_Two | woodland | 70.8717 | 74.7543 | 93.1720 | 26.6809 |
| SCLIP_Two | water | 97.3685 | 97.3685 | 100.0000 | 8.4949 |
| SCLIP_Two | road | 21.5677 | 23.1196 | 76.2640 | 3.4532 |
| VIPProxy_Two | background | 77.8803 | 96.4572 | 80.1736 | 56.3210 |
| VIPProxy_Two | building | 38.4022 | 38.5624 | 98.9299 | 3.8868 |
| VIPProxy_Two | woodland | 68.0829 | 73.3083 | 90.5226 | 26.4336 |
| VIPProxy_Two | water | 96.1301 | 96.1301 | 100.0000 | 8.6043 |
| VIPProxy_Two | road | 16.0411 | 16.8674 | 76.6056 | 4.7544 |
| CSA_SameShell | background | 83.4950 | 97.1470 | 85.5938 | 59.7016 |
| CSA_SameShell | building | 34.3642 | 34.4411 | 99.3548 | 4.3706 |
| CSA_SameShell | woodland | 80.0636 | 87.0965 | 90.8385 | 22.3265 |
| CSA_SameShell | water | 93.5357 | 93.5357 | 100.0000 | 8.8429 |
| CSA_SameShell | road | 16.1337 | 16.9488 | 77.0384 | 4.7583 |
| Proxy_SameShell | background | 82.4782 | 96.4837 | 85.0342 | 59.7190 |
| Proxy_SameShell | building | 34.9422 | 35.0405 | 99.2037 | 4.2893 |
| Proxy_SameShell | woodland | 78.0985 | 86.4993 | 88.9398 | 22.0108 |
| Proxy_SameShell | water | 93.5453 | 93.5453 | 100.0000 | 8.8420 |
| Proxy_SameShell | road | 14.9995 | 15.7001 | 77.0702 | 5.1389 |
| Geometry_SemanticBudget | background | 82.4251 | 96.9511 | 84.6184 | 59.1405 |
| Geometry_SemanticBudget | building | 34.9005 | 34.9840 | 99.3202 | 4.3013 |
| Geometry_SemanticBudget | woodland | 78.3658 | 85.6996 | 90.1551 | 22.5197 |
| Geometry_SemanticBudget | water | 93.4334 | 93.4334 | 100.0000 | 8.8526 |
| Geometry_SemanticBudget | road | 14.8605 | 15.5496 | 77.0292 | 5.1858 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 8,
    "Geometry_SemanticBudget__block0_value_read_norm": 23.82059669494629,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 76.47942447662354,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 12.846817016601562,
    "Geometry_SemanticBudget__block0_block_output_norm": 29.12572717666626,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5525957457721233,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8824691772460938,
    "Geometry_SemanticBudget__block0_dual_mean": 0.2456204928457737,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.222423881292343,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.8434410393238068,
    "Geometry_SemanticBudget__block0_geometry_budget": 1.9213405549526215,
    "Geometry_SemanticBudget__block0_constraint_violation": 2.294778823852539e-06,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.6193599700927734e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.4248914606869221,
    "Geometry_SemanticBudget__block0_row_mass_error": 3.8743019104003906e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 14.92591381072998,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 47.36440706253052,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 41.79762840270996,
    "Geometry_SemanticBudget__block1_block_output_norm": 536.5879592895508,
    "Geometry_SemanticBudget__block1_patch_mass": 0.33214840292930603,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9805145263671875,
    "Geometry_SemanticBudget__block1_dual_mean": 0.5692699514329433,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.484035491943359,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 1.9118211716413498,
    "Geometry_SemanticBudget__block1_geometry_budget": 1.9213405549526215,
    "Geometry_SemanticBudget__block1_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.544854164123535e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.9553000703454018,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.470348358154297e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 22.849595546722412,
    "Geometry__block0_projected_increment_norm": 73.22582006454468,
    "Geometry__block0_attention_residual_norm": 12.714707136154175,
    "Geometry__block0_block_output_norm": 28.94874668121338,
    "Geometry__block0_patch_mass": 0.5525957457721233,
    "Geometry__block1_value_read_norm": 13.933656930923462,
    "Geometry__block1_projected_increment_norm": 45.04781627655029,
    "Geometry__block1_attention_residual_norm": 41.539236068725586,
    "Geometry__block1_block_output_norm": 532.0254058837891,
    "Geometry__block1_patch_mass": 0.3345654495060444,
    "CSA_SameShell__block0_value_read_norm": 22.82975673675537,
    "CSA_SameShell__block0_projected_increment_norm": 73.99795532226562,
    "CSA_SameShell__block0_attention_residual_norm": 12.765495419502258,
    "CSA_SameShell__block0_block_output_norm": 29.11055898666382,
    "CSA_SameShell__block0_patch_mass": 0.5525957457721233,
    "CSA_SameShell__block0_row_mass_error": 3.725290298461914e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 14.757202982902527,
    "CSA_SameShell__block1_projected_increment_norm": 46.66096496582031,
    "CSA_SameShell__block1_attention_residual_norm": 41.80715751647949,
    "CSA_SameShell__block1_block_output_norm": 539.9289054870605,
    "CSA_SameShell__block1_patch_mass": 0.33261243253946304,
    "CSA_SameShell__block1_row_mass_error": 3.8743019104003906e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 23.13618516921997,
    "Proxy_SameShell__block0_projected_increment_norm": 73.75806999206543,
    "Proxy_SameShell__block0_attention_residual_norm": 12.732538342475891,
    "Proxy_SameShell__block0_block_output_norm": 28.903944492340088,
    "Proxy_SameShell__block0_patch_mass": 0.5525957457721233,
    "Proxy_SameShell__block0_row_mass_error": 2.086162567138672e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.040627479553223,
    "Proxy_SameShell__block1_projected_increment_norm": 45.36738300323486,
    "Proxy_SameShell__block1_attention_residual_norm": 41.51300001144409,
    "Proxy_SameShell__block1_block_output_norm": 532.295337677002,
    "Proxy_SameShell__block1_patch_mass": 0.3339708149433136,
    "Proxy_SameShell__block1_row_mass_error": 2.086162567138672e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "landcoverai__Geometry_SemanticBudget__top2_margin": 0.025725910672917962,
    "landcoverai__Geometry__top2_margin": 0.02549038454890251,
    "landcoverai__CSA_SameShell__top2_margin": 0.025736724957823753,
    "landcoverai__Proxy_SameShell__top2_margin": 0.02563068433664739,
    "landcoverai__SCLIP_Two__top2_margin": 0.026832716772332788,
    "landcoverai__VIPProxy_Two__top2_margin": 0.03082150430418551
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 2097152,
      "changed": 178185,
      "beneficial": 71675,
      "harmful": 89172,
      "wrong_to_wrong": 17338,
      "base_confusion": [
        [
          1208496,
          57056,
          62068,
          11735,
          81675
        ],
        [
          17,
          31548,
          204,
          0,
          4
        ],
        [
          37145,
          1380,
          400826,
          0,
          9582
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4719,
          41,
          289,
          0,
          16905
        ]
      ],
      "proposal_confusion": [
        [
          1173889,
          50193,
          138562,
          4688,
          53698
        ],
        [
          150,
          31366,
          257,
          0,
          0
        ],
        [
          28046,
          629,
          418280,
          0,
          1978
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          2658,
          112,
          2441,
          0,
          16743
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 2097152,
      "changed": 163835,
      "beneficial": 42644,
      "harmful": 106491,
      "wrong_to_wrong": 14700,
      "base_confusion": [
        [
          1208496,
          57056,
          62068,
          11735,
          81675
        ],
        [
          17,
          31548,
          204,
          0,
          4
        ],
        [
          37145,
          1380,
          400826,
          0,
          9582
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4719,
          41,
          289,
          0,
          16905
        ]
      ],
      "proposal_confusion": [
        [
          1139291,
          49888,
          144847,
          6983,
          80021
        ],
        [
          87,
          31433,
          253,
          0,
          0
        ],
        [
          39528,
          151,
          406386,
          0,
          2868
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          2230,
          40,
          2866,
          0,
          16818
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 2097152,
      "changed": 44888,
      "beneficial": 27877,
      "harmful": 13053,
      "wrong_to_wrong": 3958,
      "base_confusion": [
        [
          1208496,
          57056,
          62068,
          11735,
          81675
        ],
        [
          17,
          31548,
          204,
          0,
          4
        ],
        [
          37145,
          1380,
          400826,
          0,
          9582
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4719,
          41,
          289,
          0,
          16905
        ]
      ],
      "proposal_confusion": [
        [
          1216314,
          58476,
          60030,
          11988,
          74222
        ],
        [
          2,
          31568,
          203,
          0,
          0
        ],
        [
          30903,
          1572,
          407804,
          0,
          8654
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4815,
          42,
          184,
          0,
          16913
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 2097152,
      "changed": 24250,
      "beneficial": 9908,
      "harmful": 11602,
      "wrong_to_wrong": 2740,
      "base_confusion": [
        [
          1208496,
          57056,
          62068,
          11735,
          81675
        ],
        [
          17,
          31548,
          204,
          0,
          4
        ],
        [
          37145,
          1380,
          400826,
          0,
          9582
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4719,
          41,
          289,
          0,
          16905
        ]
      ],
      "proposal_confusion": [
        [
          1208361,
          56955,
          61975,
          11969,
          81770
        ],
        [
          25,
          31520,
          215,
          0,
          13
        ],
        [
          39124,
          1462,
          399280,
          0,
          9067
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4889,
          16,
          129,
          0,
          16920
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 2097152,
      "changed": 27869,
      "beneficial": 11038,
      "harmful": 13156,
      "wrong_to_wrong": 3675,
      "base_confusion": [
        [
          1208496,
          57056,
          62068,
          11735,
          81675
        ],
        [
          17,
          31548,
          204,
          0,
          4
        ],
        [
          37145,
          1380,
          400826,
          0,
          9582
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4719,
          41,
          289,
          0,
          16905
        ]
      ],
      "proposal_confusion": [
        [
          1202453,
          57215,
          66872,
          12191,
          82299
        ],
        [
          6,
          31557,
          210,
          0,
          0
        ],
        [
          33271,
          1381,
          404736,
          0,
          9545
        ],
        [
          0,
          0,
          0,
          173462,
          0
        ],
        [
          4537,
          51,
          455,
          0,
          16911
        ]
      ]
    }
  }
}
```

### flair1/flair1

| Method | Class | IoU | Precision | Recall | Prediction area |
|---|---|---:|---:|---:|---:|
| Geometry | building | 49.6979 | 50.7258 | 96.0825 | 13.5991 |
| Geometry | pervious surface | 57.0621 | 92.1299 | 59.9861 | 11.1728 |
| Geometry | impervious surface | 52.6509 | 62.6245 | 76.7765 | 20.0846 |
| Geometry | bare soil | 0.0000 | 0.0000 | NA | 3.4411 |
| Geometry | water | 73.2328 | 77.1432 | 93.5263 | 5.3781 |
| Geometry | coniferous | 43.0233 | 61.7114 | 58.6897 | 0.5346 |
| Geometry | deciduous | 54.0229 | 78.9723 | 63.0995 | 13.6004 |
| Geometry | brushwood | 18.6136 | 23.4212 | 47.5559 | 9.9866 |
| Geometry | vineyard | NA | NA | NA | 0.0000 |
| Geometry | herbaceous vegetation | 60.2329 | 94.3169 | 62.5013 | 21.2289 |
| Geometry | agricultural land | 18.7339 | 21.6468 | 58.1977 | 0.8198 |
| Geometry | plowed land | 0.0000 | 0.0000 | NA | 0.1540 |
| SCLIP_Two | building | 58.7412 | 59.2443 | 98.5748 | 11.9458 |
| SCLIP_Two | pervious surface | 71.8874 | 97.3725 | 73.3095 | 12.9192 |
| SCLIP_Two | impervious surface | 53.2118 | 60.6638 | 81.2445 | 21.9403 |
| SCLIP_Two | bare soil | 0.0000 | 0.0000 | NA | 1.1428 |
| SCLIP_Two | water | 76.2445 | 81.2193 | 92.5639 | 5.0556 |
| SCLIP_Two | coniferous | 37.9276 | 70.3038 | 45.1629 | 0.3611 |
| SCLIP_Two | deciduous | 61.0920 | 71.0513 | 81.3379 | 19.4860 |
| SCLIP_Two | brushwood | 12.3306 | 57.5218 | 13.5659 | 1.1600 |
| SCLIP_Two | vineyard | 0.0000 | 0.0000 | NA | 0.2892 |
| SCLIP_Two | herbaceous vegetation | 60.9120 | 87.5959 | 66.6620 | 24.3793 |
| SCLIP_Two | agricultural land | 19.1230 | 22.1679 | 58.1977 | 0.8005 |
| SCLIP_Two | plowed land | 0.0000 | 0.0000 | NA | 0.5203 |
| VIPProxy_Two | building | 57.2651 | 57.5377 | 99.1794 | 12.3755 |
| VIPProxy_Two | pervious surface | 63.9225 | 94.7644 | 66.2626 | 11.9987 |
| VIPProxy_Two | impervious surface | 54.3336 | 64.0426 | 78.1847 | 20.0000 |
| VIPProxy_Two | bare soil | 0.0000 | 0.0000 | NA | 2.2913 |
| VIPProxy_Two | water | 76.3777 | 80.7757 | 93.3457 | 5.1263 |
| VIPProxy_Two | coniferous | 40.1241 | 62.7566 | 52.6646 | 0.4717 |
| VIPProxy_Two | deciduous | 65.3024 | 75.1364 | 83.3038 | 18.8719 |
| VIPProxy_Two | brushwood | 31.0695 | 57.4354 | 40.3631 | 3.4564 |
| VIPProxy_Two | vineyard | 0.0000 | 0.0000 | NA | 0.0007 |
| VIPProxy_Two | herbaceous vegetation | 64.8727 | 90.6076 | 69.5498 | 24.5900 |
| VIPProxy_Two | agricultural land | 18.8555 | 21.8092 | 58.1977 | 0.8137 |
| VIPProxy_Two | plowed land | 0.0000 | 0.0000 | NA | 0.0037 |
| CSA_SameShell | building | 49.3186 | 50.6532 | 94.9284 | 13.4550 |
| CSA_SameShell | pervious surface | 60.5468 | 93.2537 | 63.3204 | 11.6517 |
| CSA_SameShell | impervious surface | 50.5355 | 59.4473 | 77.1222 | 21.2532 |
| CSA_SameShell | bare soil | 0.0000 | 0.0000 | NA | 2.7727 |
| CSA_SameShell | water | 72.9628 | 77.4892 | 92.5875 | 5.3003 |
| CSA_SameShell | coniferous | 41.1898 | 68.4848 | 50.8232 | 0.4172 |
| CSA_SameShell | deciduous | 46.0969 | 79.9396 | 52.1268 | 11.0994 |
| CSA_SameShell | brushwood | 15.0905 | 18.3382 | 46.0069 | 12.3392 |
| CSA_SameShell | vineyard | NA | NA | NA | 0.0000 |
| CSA_SameShell | herbaceous vegetation | 58.1349 | 93.2881 | 60.6726 | 20.8350 |
| CSA_SameShell | agricultural land | 20.1801 | 23.6011 | 58.1977 | 0.7519 |
| CSA_SameShell | plowed land | 0.0000 | 0.0000 | NA | 0.1243 |
| Proxy_SameShell | building | 50.3181 | 51.2432 | 96.5363 | 13.5254 |
| Proxy_SameShell | pervious surface | 56.2119 | 91.3357 | 59.3781 | 11.1557 |
| Proxy_SameShell | impervious surface | 52.2851 | 62.2264 | 76.5957 | 20.1654 |
| Proxy_SameShell | bare soil | 0.0000 | 0.0000 | NA | 3.5353 |
| Proxy_SameShell | water | 73.3565 | 77.0769 | 93.8264 | 5.4000 |
| Proxy_SameShell | coniferous | 39.9766 | 63.1514 | 52.1385 | 0.4641 |
| Proxy_SameShell | deciduous | 55.6361 | 79.0950 | 65.2277 | 14.0373 |
| Proxy_SameShell | brushwood | 18.3783 | 23.3489 | 46.3319 | 9.7597 |
| Proxy_SameShell | vineyard | NA | NA | NA | 0.0000 |
| Proxy_SameShell | herbaceous vegetation | 59.4614 | 94.1200 | 61.7555 | 21.0194 |
| Proxy_SameShell | agricultural land | 18.7879 | 21.7188 | 58.1977 | 0.8171 |
| Proxy_SameShell | plowed land | 0.0000 | 0.0000 | NA | 0.1205 |
| Geometry_SemanticBudget | building | 49.3502 | 50.4890 | 95.6294 | 13.5985 |
| Geometry_SemanticBudget | pervious surface | 55.7840 | 92.5966 | 58.3882 | 10.8204 |
| Geometry_SemanticBudget | impervious surface | 51.9337 | 61.5976 | 76.7995 | 20.4255 |
| Geometry_SemanticBudget | bare soil | 0.0000 | 0.0000 | NA | 3.5197 |
| Geometry_SemanticBudget | water | 73.3999 | 76.9606 | 94.0705 | 5.4222 |
| Geometry_SemanticBudget | coniferous | 43.3251 | 64.4898 | 56.8992 | 0.4960 |
| Geometry_SemanticBudget | deciduous | 52.4983 | 78.9706 | 61.0304 | 13.1547 |
| Geometry_SemanticBudget | brushwood | 17.6543 | 22.2640 | 46.0234 | 10.1671 |
| Geometry_SemanticBudget | vineyard | NA | NA | NA | 0.0000 |
| Geometry_SemanticBudget | herbaceous vegetation | 60.6663 | 94.2517 | 62.9972 | 21.4121 |
| Geometry_SemanticBudget | agricultural land | 18.6147 | 21.4834 | 58.2290 | 0.8265 |
| Geometry_SemanticBudget | plowed land | 0.0000 | 0.0000 | NA | 0.1573 |

Transitions and per-block norms/constraint statistics:
```json
{
  "diagnostics": {
    "tiles": 8,
    "Geometry_SemanticBudget__block0_value_read_norm": 22.442606925964355,
    "Geometry_SemanticBudget__block0_projected_increment_norm": 68.59308815002441,
    "Geometry_SemanticBudget__block0_attention_residual_norm": 11.449345588684082,
    "Geometry_SemanticBudget__block0_block_output_norm": 23.715266942977905,
    "Geometry_SemanticBudget__block0_patch_mass": 0.5554644018411636,
    "Geometry_SemanticBudget__block0_active_fraction": 0.8598709106445312,
    "Geometry_SemanticBudget__block0_dual_mean": 0.20032159984111786,
    "Geometry_SemanticBudget__block0_semantic_expected_cost": 3.318765699863434,
    "Geometry_SemanticBudget__block0_projected_expected_cost": 1.9505142271518707,
    "Geometry_SemanticBudget__block0_geometry_budget": 2.045718938112259,
    "Geometry_SemanticBudget__block0_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block0_conditional_mass_error": 4.544854164123535e-07,
    "Geometry_SemanticBudget__block0_projection_l1": 0.3833649940788746,
    "Geometry_SemanticBudget__block0_row_mass_error": 4.6193599700927734e-07,
    "Geometry_SemanticBudget__block0_invalid_edge_error": 0.0,
    "Geometry_SemanticBudget__block1_value_read_norm": 15.81825315952301,
    "Geometry_SemanticBudget__block1_projected_increment_norm": 49.34305953979492,
    "Geometry_SemanticBudget__block1_attention_residual_norm": 35.4437530040741,
    "Geometry_SemanticBudget__block1_block_output_norm": 478.58732986450195,
    "Geometry_SemanticBudget__block1_patch_mass": 0.3427733704447746,
    "Geometry_SemanticBudget__block1_active_fraction": 0.9578094482421875,
    "Geometry_SemanticBudget__block1_dual_mean": 0.4825572520494461,
    "Geometry_SemanticBudget__block1_semantic_expected_cost": 5.50743168592453,
    "Geometry_SemanticBudget__block1_projected_expected_cost": 2.022415205836296,
    "Geometry_SemanticBudget__block1_geometry_budget": 2.045718938112259,
    "Geometry_SemanticBudget__block1_constraint_violation": 0.0,
    "Geometry_SemanticBudget__block1_conditional_mass_error": 4.470348358154297e-07,
    "Geometry_SemanticBudget__block1_projection_l1": 0.8724752962589264,
    "Geometry_SemanticBudget__block1_row_mass_error": 4.917383193969727e-07,
    "Geometry_SemanticBudget__block1_invalid_edge_error": 0.0,
    "Geometry__block0_value_read_norm": 21.23013687133789,
    "Geometry__block0_projected_increment_norm": 64.85752820968628,
    "Geometry__block0_attention_residual_norm": 11.314592838287354,
    "Geometry__block0_block_output_norm": 23.543901681900024,
    "Geometry__block0_patch_mass": 0.5554644018411636,
    "Geometry__block1_value_read_norm": 14.561819911003113,
    "Geometry__block1_projected_increment_norm": 46.163630962371826,
    "Geometry__block1_attention_residual_norm": 35.15885663032532,
    "Geometry__block1_block_output_norm": 474.0485649108887,
    "Geometry__block1_patch_mass": 0.344815943390131,
    "CSA_SameShell__block0_value_read_norm": 21.064675331115723,
    "CSA_SameShell__block0_projected_increment_norm": 65.3883318901062,
    "CSA_SameShell__block0_attention_residual_norm": 11.352115750312805,
    "CSA_SameShell__block0_block_output_norm": 23.659273624420166,
    "CSA_SameShell__block0_patch_mass": 0.5554644018411636,
    "CSA_SameShell__block0_row_mass_error": 4.3958425521850586e-07,
    "CSA_SameShell__block0_invalid_edge_error": 0.0,
    "CSA_SameShell__block1_value_read_norm": 15.271198153495789,
    "CSA_SameShell__block1_projected_increment_norm": 47.66200542449951,
    "CSA_SameShell__block1_attention_residual_norm": 35.31043219566345,
    "CSA_SameShell__block1_block_output_norm": 479.5742530822754,
    "CSA_SameShell__block1_patch_mass": 0.34182558581233025,
    "CSA_SameShell__block1_row_mass_error": 4.246830940246582e-07,
    "CSA_SameShell__block1_invalid_edge_error": 0.0,
    "Proxy_SameShell__block0_value_read_norm": 20.492660522460938,
    "Proxy_SameShell__block0_projected_increment_norm": 63.11829996109009,
    "Proxy_SameShell__block0_attention_residual_norm": 11.280076026916504,
    "Proxy_SameShell__block0_block_output_norm": 23.534013032913208,
    "Proxy_SameShell__block0_patch_mass": 0.5554644018411636,
    "Proxy_SameShell__block0_row_mass_error": 2.0116567611694336e-07,
    "Proxy_SameShell__block0_invalid_edge_error": 0.0,
    "Proxy_SameShell__block1_value_read_norm": 14.437993049621582,
    "Proxy_SameShell__block1_projected_increment_norm": 45.71464538574219,
    "Proxy_SameShell__block1_attention_residual_norm": 35.13810396194458,
    "Proxy_SameShell__block1_block_output_norm": 474.69698333740234,
    "Proxy_SameShell__block1_patch_mass": 0.34697510674595833,
    "Proxy_SameShell__block1_row_mass_error": 2.0116567611694336e-07,
    "Proxy_SameShell__block1_invalid_edge_error": 0.0,
    "flair1__Geometry_SemanticBudget__top2_margin": 0.017164101940579712,
    "flair1__Geometry__top2_margin": 0.01692103617824614,
    "flair1__CSA_SameShell__top2_margin": 0.016181627986952662,
    "flair1__Proxy_SameShell__top2_margin": 0.01699714921414852,
    "flair1__SCLIP_Two__top2_margin": 0.015585974208079278,
    "flair1__VIPProxy_Two__top2_margin": 0.018984334310516715
  },
  "transitions": {
    "SCLIP_Two": {
      "valid": 2096296,
      "changed": 412163,
      "beneficial": 214137,
      "harmful": 91629,
      "wrong_to_wrong": 106397,
      "base_confusion": [
        [
          144608,
          0,
          5843,
          0,
          0,
          0,
          22,
          0,
          0,
          31,
          0,
          0
        ],
        [
          14149,
          215782,
          53518,
          68933,
          1619,
          2,
          1716,
          2872,
          0,
          538,
          0,
          591
        ],
        [
          71259,
          1220,
          263669,
          0,
          346,
          0,
          2974,
          2494,
          0,
          1462,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          231,
          0,
          1410,
          0,
          86972,
          0,
          3533,
          846,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          869,
          0,
          0,
          6916,
          3190,
          755,
          0,
          30,
          0,
          0
        ],
        [
          8452,
          2066,
          20027,
          0,
          1177,
          1331,
          225154,
          82514,
          0,
          15820,
          283,
          0
        ],
        [
          4330,
          948,
          13345,
          0,
          17898,
          0,
          9944,
          49032,
          0,
          5263,
          2344,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42013,
          13952,
          62085,
          3202,
          4729,
          2958,
          38572,
          70836,
          0,
          419729,
          10838,
          2638
        ],
        [
          12,
          247,
          266,
          0,
          0,
          0,
          0,
          0,
          0,
          2147,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ],
      "proposal_confusion": [
        [
          148359,
          0,
          2052,
          0,
          0,
          0,
          20,
          0,
          0,
          73,
          0,
          0
        ],
        [
          10194,
          263709,
          62590,
          15925,
          310,
          0,
          2296,
          412,
          42,
          3650,
          0,
          592
        ],
        [
          54992,
          403,
          279013,
          0,
          1006,
          0,
          3572,
          132,
          43,
          4263,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          0,
          0,
          522,
          0,
          86077,
          0,
          6306,
          0,
          2,
          31,
          54,
          0
        ],
        [
          59,
          0,
          463,
          0,
          0,
          5322,
          5382,
          462,
          0,
          96,
          0,
          0
        ],
        [
          6165,
          489,
          15460,
          0,
          513,
          711,
          290233,
          3309,
          1227,
          38255,
          462,
          0
        ],
        [
          2775,
          194,
          10585,
          0,
          13180,
          0,
          40034,
          13987,
          4067,
          14483,
          3799,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          27875,
          6030,
          89248,
          8031,
          4895,
          1537,
          60641,
          6014,
          551,
          447670,
          8746,
          10314
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          130,
          2542,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ]
    },
    "VIPProxy_Two": {
      "valid": 2096296,
      "changed": 306118,
      "beneficial": 186828,
      "harmful": 43619,
      "wrong_to_wrong": 75671,
      "base_confusion": [
        [
          144608,
          0,
          5843,
          0,
          0,
          0,
          22,
          0,
          0,
          31,
          0,
          0
        ],
        [
          14149,
          215782,
          53518,
          68933,
          1619,
          2,
          1716,
          2872,
          0,
          538,
          0,
          591
        ],
        [
          71259,
          1220,
          263669,
          0,
          346,
          0,
          2974,
          2494,
          0,
          1462,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          231,
          0,
          1410,
          0,
          86972,
          0,
          3533,
          846,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          869,
          0,
          0,
          6916,
          3190,
          755,
          0,
          30,
          0,
          0
        ],
        [
          8452,
          2066,
          20027,
          0,
          1177,
          1331,
          225154,
          82514,
          0,
          15820,
          283,
          0
        ],
        [
          4330,
          948,
          13345,
          0,
          17898,
          0,
          9944,
          49032,
          0,
          5263,
          2344,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42013,
          13952,
          62085,
          3202,
          4729,
          2958,
          38572,
          70836,
          0,
          419729,
          10838,
          2638
        ],
        [
          12,
          247,
          266,
          0,
          0,
          0,
          0,
          0,
          0,
          2147,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ],
      "proposal_confusion": [
        [
          149269,
          0,
          1111,
          0,
          0,
          0,
          35,
          0,
          0,
          89,
          0,
          0
        ],
        [
          10133,
          238360,
          56905,
          43058,
          566,
          0,
          4612,
          1232,
          0,
          4854,
          0,
          0
        ],
        [
          65221,
          684,
          268505,
          0,
          422,
          0,
          4027,
          686,
          0,
          3879,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          40,
          0,
          1223,
          0,
          86804,
          0,
          4879,
          0,
          0,
          46,
          0,
          0
        ],
        [
          0,
          0,
          453,
          0,
          0,
          6206,
          4801,
          262,
          0,
          62,
          0,
          0
        ],
        [
          4620,
          671,
          13675,
          0,
          1127,
          895,
          297248,
          12384,
          0,
          25694,
          510,
          0
        ],
        [
          1710,
          850,
          11536,
          0,
          13085,
          0,
          19622,
          41616,
          0,
          11120,
          3565,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          28435,
          10964,
          65852,
          4974,
          5459,
          2788,
          60387,
          16277,
          14,
          467063,
          9262,
          77
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          2672,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ]
    },
    "CSA_SameShell": {
      "valid": 2096296,
      "changed": 149324,
      "beneficial": 34082,
      "harmful": 77469,
      "wrong_to_wrong": 37773,
      "base_confusion": [
        [
          144608,
          0,
          5843,
          0,
          0,
          0,
          22,
          0,
          0,
          31,
          0,
          0
        ],
        [
          14149,
          215782,
          53518,
          68933,
          1619,
          2,
          1716,
          2872,
          0,
          538,
          0,
          591
        ],
        [
          71259,
          1220,
          263669,
          0,
          346,
          0,
          2974,
          2494,
          0,
          1462,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          231,
          0,
          1410,
          0,
          86972,
          0,
          3533,
          846,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          869,
          0,
          0,
          6916,
          3190,
          755,
          0,
          30,
          0,
          0
        ],
        [
          8452,
          2066,
          20027,
          0,
          1177,
          1331,
          225154,
          82514,
          0,
          15820,
          283,
          0
        ],
        [
          4330,
          948,
          13345,
          0,
          17898,
          0,
          9944,
          49032,
          0,
          5263,
          2344,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42013,
          13952,
          62085,
          3202,
          4729,
          2958,
          38572,
          70836,
          0,
          419729,
          10838,
          2638
        ],
        [
          12,
          247,
          266,
          0,
          0,
          0,
          0,
          0,
          0,
          2147,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ],
      "proposal_confusion": [
        [
          142871,
          0,
          7548,
          0,
          0,
          0,
          22,
          4,
          0,
          59,
          0,
          0
        ],
        [
          13883,
          227776,
          56163,
          55701,
          1329,
          0,
          1291,
          1612,
          0,
          1378,
          0,
          587
        ],
        [
          69560,
          2117,
          264856,
          0,
          316,
          0,
          2452,
          2763,
          0,
          1360,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          254,
          0,
          2107,
          0,
          86099,
          0,
          2939,
          1593,
          0,
          0,
          0,
          0
        ],
        [
          185,
          0,
          920,
          0,
          0,
          5989,
          2588,
          2101,
          0,
          1,
          0,
          0
        ],
        [
          8601,
          1839,
          25024,
          24,
          1012,
          809,
          186001,
          114674,
          0,
          18722,
          118,
          0
        ],
        [
          3691,
          834,
          14015,
          0,
          17679,
          0,
          11501,
          47435,
          0,
          5622,
          2327,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42911,
          11523,
          74669,
          2398,
          4676,
          1947,
          25883,
          88485,
          0,
          407448,
          9597,
          2015
        ],
        [
          101,
          165,
          229,
          0,
          0,
          0,
          0,
          0,
          0,
          2173,
          3720,
          4
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ]
    },
    "Proxy_SameShell": {
      "valid": 2096296,
      "changed": 68841,
      "beneficial": 23756,
      "harmful": 25051,
      "wrong_to_wrong": 20034,
      "base_confusion": [
        [
          144608,
          0,
          5843,
          0,
          0,
          0,
          22,
          0,
          0,
          31,
          0,
          0
        ],
        [
          14149,
          215782,
          53518,
          68933,
          1619,
          2,
          1716,
          2872,
          0,
          538,
          0,
          591
        ],
        [
          71259,
          1220,
          263669,
          0,
          346,
          0,
          2974,
          2494,
          0,
          1462,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          231,
          0,
          1410,
          0,
          86972,
          0,
          3533,
          846,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          869,
          0,
          0,
          6916,
          3190,
          755,
          0,
          30,
          0,
          0
        ],
        [
          8452,
          2066,
          20027,
          0,
          1177,
          1331,
          225154,
          82514,
          0,
          15820,
          283,
          0
        ],
        [
          4330,
          948,
          13345,
          0,
          17898,
          0,
          9944,
          49032,
          0,
          5263,
          2344,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42013,
          13952,
          62085,
          3202,
          4729,
          2958,
          38572,
          70836,
          0,
          419729,
          10838,
          2638
        ],
        [
          12,
          247,
          266,
          0,
          0,
          0,
          0,
          0,
          0,
          2147,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ],
      "proposal_confusion": [
        [
          145291,
          0,
          5162,
          0,
          0,
          0,
          22,
          4,
          0,
          25,
          0,
          0
        ],
        [
          13557,
          213595,
          53593,
          71060,
          1454,
          0,
          1973,
          2952,
          0,
          948,
          0,
          588
        ],
        [
          71459,
          1551,
          263048,
          0,
          348,
          0,
          2931,
          2102,
          0,
          1985,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          204,
          0,
          1502,
          0,
          87251,
          0,
          3571,
          464,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          720,
          0,
          0,
          6144,
          4011,
          872,
          0,
          13,
          0,
          0
        ],
        [
          7455,
          1851,
          20395,
          0,
          1353,
          776,
          232748,
          76398,
          0,
          15538,
          310,
          0
        ],
        [
          3420,
          1072,
          13355,
          0,
          17859,
          0,
          12097,
          47770,
          0,
          5116,
          2415,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42091,
          15559,
          64824,
          3051,
          4935,
          2809,
          36911,
          74030,
          0,
          414720,
          10683,
          1939
        ],
        [
          31,
          229,
          128,
          0,
          0,
          0,
          0,
          0,
          0,
          2284,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ]
    },
    "Geometry_SemanticBudget": {
      "valid": 2096296,
      "changed": 68427,
      "beneficial": 18320,
      "harmful": 30007,
      "wrong_to_wrong": 20100,
      "base_confusion": [
        [
          144608,
          0,
          5843,
          0,
          0,
          0,
          22,
          0,
          0,
          31,
          0,
          0
        ],
        [
          14149,
          215782,
          53518,
          68933,
          1619,
          2,
          1716,
          2872,
          0,
          538,
          0,
          591
        ],
        [
          71259,
          1220,
          263669,
          0,
          346,
          0,
          2974,
          2494,
          0,
          1462,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          231,
          0,
          1410,
          0,
          86972,
          0,
          3533,
          846,
          0,
          0,
          0,
          0
        ],
        [
          24,
          0,
          869,
          0,
          0,
          6916,
          3190,
          755,
          0,
          30,
          0,
          0
        ],
        [
          8452,
          2066,
          20027,
          0,
          1177,
          1331,
          225154,
          82514,
          0,
          15820,
          283,
          0
        ],
        [
          4330,
          948,
          13345,
          0,
          17898,
          0,
          9944,
          49032,
          0,
          5263,
          2344,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42013,
          13952,
          62085,
          3202,
          4729,
          2958,
          38572,
          70836,
          0,
          419729,
          10838,
          2638
        ],
        [
          12,
          247,
          266,
          0,
          0,
          0,
          0,
          0,
          0,
          2147,
          3720,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ],
      "proposal_confusion": [
        [
          143926,
          0,
          6508,
          0,
          0,
          0,
          26,
          0,
          0,
          44,
          0,
          0
        ],
        [
          13860,
          210034,
          56788,
          70948,
          1854,
          9,
          1649,
          3129,
          0,
          861,
          0,
          588
        ],
        [
          70894,
          1733,
          263748,
          0,
          369,
          0,
          2806,
          2671,
          0,
          1203,
          0,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          138,
          0,
          1473,
          0,
          87478,
          0,
          3070,
          833,
          0,
          0,
          0,
          0
        ],
        [
          234,
          0,
          908,
          0,
          0,
          6705,
          3103,
          809,
          0,
          25,
          0,
          0
        ],
        [
          9023,
          2031,
          23266,
          12,
          1109,
          1249,
          217771,
          85903,
          0,
          16220,
          240,
          0
        ],
        [
          4192,
          910,
          13893,
          0,
          17981,
          0,
          10841,
          47452,
          0,
          5300,
          2535,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ],
        [
          42697,
          11954,
          61339,
          2824,
          4875,
          2434,
          36496,
          72336,
          0,
          423059,
          10828,
          2710
        ],
        [
          100,
          165,
          256,
          0,
          0,
          0,
          0,
          0,
          0,
          2149,
          3722,
          0
        ],
        [
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0
        ]
      ]
    }
  }
}
```

