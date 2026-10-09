# Stratified branch-aligned soft aliases: verified screen

## Outcome Interpretation

The v2 candidate did not pass the predeclared advancement gate. Equal-domain mIoU is 45.677024%, versus 46.058780% for unscreened all20 coupling (-0.381756pp), 46.058052% for soft-v1 (-0.381028pp), and 44.447574% for hard coupling (+1.229450pp). It improves four of eight domains, but loses 2.042960pp on Potsdam. LoveDA P additionally loses 1.684573pp. No full rollout was launched; retain the unchanged unscreened coupling as the stronger reference in this comparison, not v2 as the final model.

The matched shuffled control scores 45.654736%, only 0.022288pp below v2. One shuffled seed and eight images per non-UDD5 domain do not establish reliable semantic utility or statistical significance. The four gains are exploratory. Soft-v1 is numerically almost identical to unscreened coupling in this matched screen, so its apparent advantage over hard deletion is not evidence of substantial useful alias intervention.

Two distinct failure risks were observed:

1. Reference contamination: local/broad leave-out agreement does not imply correctness. On Potsdam, unique declared-car witnesses have 46.1538% patch-center purity, and clutter witnesses 5.5363%. These are candidate-pool audits, not query-specific retrieved-reference purity; causal contributions were not isolated.
2. Class-score calibration: multiplying normalized reliability into alias scores before exponential aggregation can raise the class score merely through weight dispersion. The pair-margin bound limits magnitude but does not remove this bias.

Potsdam car IoU changes from 24.2128% to 17.4996%; precision from 24.3214% to 17.5430%; recall from 98.1896% to 98.6068%; predicted area from 9.4578% to 13.1679%. The main observed car failure is increased false activation, not lost recall. UDD5 vehicle shows the same pattern: IoU 20.4213% to 15.6983%, area 3.4003% to 4.5237%, precision 20.9505% to 15.9693%. UDD5 road improves slightly (39.8083% to 39.9650%), but does not compensate for vehicle loss. Its non-residual mIoU falls from 52.7852% to 51.9328%.

### Unlabeled Constant-Response Check

With K=20, all raw responses equal to 3, uniform original salience, and reliability rho = 0.5/K + 0.5*softmax([2]*10 + [-2]*10):

| Aggregation | Score | Change |
| --- | ---: | ---: |
| Original logsumexp | 5.995732 | 0 |
| V2 reliability-scaled logits | 6.802605 | +0.806872 |
| Reliability outside exponent | 5.995732 | 0 |

The v2 increase still induces a bounded pair correction of 0.461856 when only one rival receives this change. This is an unlabeled mathematical diagnostic, not an estimate of how much of the Potsdam loss it caused.

A next candidate should keep original profiled evidence fixed: let r_a = K*p_a*z_a, with p_a the original salience, and use S_rho = logsumexp(tau*r_a + log(K*rho_a))/tau. Uniform rho replays the original score, and identical profiled evidence is invariant to normalized reliability. Identical raw z_a alone does not guarantee identical profiled r_a when salience differs. This outside-exponent proposal is NOT implemented or evaluated in this run; reference contamination remains a separate unresolved problem.

All main results and two audits were downloaded and independently verified. All gssa02 experiment sessions ended; GPUs 0-7 had no compute processes at the final check. Unrelated tmux sessions were left unchanged. The 1131.5905-second suite wall and 5492.1602-6238.5161 MiB peak allocated memory describe a shared seven-arm workload, not standalone v2 inference cost. BroadVIP here is the matched unscreened 20-alias branch, not an official-short-vocabulary full VIP reproduction.

Same96 unique complete images: UDD5 full40; seven other domains8 each. LoveDA P/D share images; means count D once. Corrected IRRG Vaihingen; LandCover.ai substitutes for unlabeled iSAID. Development data, not full eight-dataset or independent validation.

Frozen weights, fixed exactly20 aliases/class, image-only single-image references, no target-label tuning. Image-wide independent class pools, one candidate per64px cell, query cell excluded. Broad leave-out utility and actual salience-weighted broad aggregation; original Geometry anchor. Correlated aliases and jointly mistaken witnesses remain possible. VIP broad observer/readout is borrowed and disclosed.

12 core tests passed locally/remotely. Mask-free smoke verified exact uniform fallback, raw uniform stencil within1e-4 and unchanged frozen heads. Five historical controls replay exact per-image confusions. Unique coverage, source identity, matrix sums, transition endpoints, pair partitions and shuffled spectra independently verified.

## Complete-image mIoU

| Dataset/protocol | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 51.1203 | 31.9462 | 57.0012 | 56.0239 | 57.0011 | 55.8551 | 56.2345 |
| potsdam/potsdam | 40.5048 | 40.3528 | 41.7806 | 38.0306 | 41.7795 | 39.7376 | 40.2190 |
| udd5/udd5 | 45.2814 | 50.5553 | 49.2117 | 47.2076 | 49.2114 | 48.4086 | 49.5207 |
| oem/oem | 28.1181 | 39.3543 | 31.9533 | 32.5631 | 31.9535 | 32.3778 | 31.2635 |
| loveda/P | 56.3776 | 62.8254 | 62.4863 | 64.2089 | 62.4868 | 60.8018 | 61.9720 |
| loveda/D | 33.2861 | 38.4558 | 36.5600 | 39.6539 | 36.5593 | 36.2811 | 36.5636 |
| vaihingen/vaihingen | 47.6306 | 50.2325 | 51.4491 | 46.9131 | 51.4474 | 51.5201 | 50.5168 |
| landcoverai/landcoverai | 65.8638 | 60.9049 | 66.9060 | 65.4367 | 66.9012 | 67.4765 | 67.3081 |
| flair1/flair1 | 29.7067 | 38.8428 | 33.6084 | 29.7516 | 33.6110 | 33.7594 | 33.6116 |
| Equal-domain mean, LoveDA D once | 42.688973 | 43.830569 | 46.058780 | 44.447574 | 46.058052 | 45.677024 | 45.654736 |

Decision: {"promising": false, "winning_domains": 4, "worst_delta_vs_all20_coupling": -2.0429595336545034, "gate": {"minimum_mean_gain_vs_all20_pp": 0.1, "minimum_winning_domains": 5, "maximum_protocol_loss_vs_all20_pp": 1.0, "must_beat_matched_shuffled_mean": true, "mean_protocol": "LoveDA D once and seven other domains once", "automatic_full_rollout": false}}

Primary delta vs all20: -0.381756pp; vs soft-v1: -0.381028pp; vs hard: +1.229450pp; vs shuffled: +0.022288pp.

One shuffled seed; gate is a scheduling rule, not a statistical significance test. Reference retrieval, spatial de-duplication and branch alignment change together; this is a coherent candidate comparison, not isolated causal attribution. No retrospective config changes or automatic full rollout.

## Direct Correction Accounting

| Dataset/protocol | Contrast | Beneficial | Harmful | Net correct pixels |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | new_vs_all20 | 658931 | 960181 | -301250 |
| vdd/vdd | new_vs_old_soft | 659024 | 960217 | -301193 |
| vdd/vdd | new_vs_hard | 2028377 | 2032309 | -3932 |
| vdd/vdd | new_vs_shuffled | 683679 | 450299 | +233380 |
| potsdam/potsdam | new_vs_all20 | 46354 | 298623 | -252269 |
| potsdam/potsdam | new_vs_old_soft | 46258 | 298398 | -252140 |
| potsdam/potsdam | new_vs_hard | 367889 | 213784 | +154105 |
| potsdam/potsdam | new_vs_shuffled | 58422 | 152713 | -94291 |
| udd5/udd5 | new_vs_all20 | 5954042 | 6579466 | -625424 |
| udd5/udd5 | new_vs_old_soft | 5954388 | 6578906 | -624518 |
| udd5/udd5 | new_vs_hard | 21295487 | 10785239 | +10510248 |
| udd5/udd5 | new_vs_shuffled | 6996021 | 7771786 | -775765 |
| oem/oem | new_vs_all20 | 174407 | 76203 | +98204 |
| oem/oem | new_vs_old_soft | 174359 | 76222 | +98137 |
| oem/oem | new_vs_hard | 358783 | 589508 | -230725 |
| oem/oem | new_vs_shuffled | 320917 | 104515 | +216402 |
| loveda/P | new_vs_all20 | 3436 | 31289 | -27853 |
| loveda/P | new_vs_old_soft | 3422 | 31283 | -27861 |
| loveda/P | new_vs_hard | 10628 | 85989 | -75361 |
| loveda/P | new_vs_shuffled | 12543 | 32167 | -19624 |
| loveda/D | new_vs_all20 | 4266 | 21211 | -16945 |
| loveda/D | new_vs_old_soft | 4250 | 21196 | -16946 |
| loveda/D | new_vs_hard | 139605 | 741082 | -601477 |
| loveda/D | new_vs_shuffled | 5115 | 19681 | -14566 |
| vaihingen/vaihingen | new_vs_all20 | 82761 | 75660 | +7101 |
| vaihingen/vaihingen | new_vs_old_soft | 82719 | 75507 | +7212 |
| vaihingen/vaihingen | new_vs_hard | 455181 | 72991 | +382190 |
| vaihingen/vaihingen | new_vs_shuffled | 107042 | 32099 | +74943 |
| landcoverai/landcoverai | new_vs_all20 | 6585 | 1555 | +5030 |
| landcoverai/landcoverai | new_vs_old_soft | 6656 | 1542 | +5114 |
| landcoverai/landcoverai | new_vs_hard | 63845 | 20000 | +43845 |
| landcoverai/landcoverai | new_vs_shuffled | 8328 | 11269 | -2941 |
| flair1/flair1 | new_vs_all20 | 12352 | 5443 | +6909 |
| flair1/flair1 | new_vs_old_soft | 12255 | 5422 | +6833 |
| flair1/flair1 | new_vs_hard | 202233 | 50029 | +152204 |
| flair1/flair1 | new_vs_shuffled | 13295 | 6839 | +6456 |

## vdd/vdd

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 48.3490 | 25.3949 | 53.4471 | 54.9224 | 53.4472 | 52.8009 | 52.4734 |
| wall | 43.4246 | 11.8982 | 57.7209 | 51.7781 | 57.7204 | 53.7949 | 54.6765 |
| road | 17.7913 | 24.1970 | 25.5594 | 26.2140 | 25.5594 | 26.0053 | 26.1343 |
| vegetation | 53.3537 | 62.0257 | 60.0299 | 53.9394 | 60.0290 | 59.6645 | 58.8521 |
| vehicle | 24.8750 | 5.6965 | 25.3443 | 26.0569 | 25.3446 | 22.0876 | 25.9206 |
| roof | 79.6473 | 60.7577 | 84.4472 | 87.2233 | 84.4475 | 83.9998 | 83.3222 |
| water | 90.4010 | 33.6534 | 92.4595 | 92.0328 | 92.4594 | 92.6324 | 92.2626 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| other | 24.671744 | 24.696326 | 72.829106 | 72.215385 | 66.758676 | 66.262066 |
| wall | 2.241210 | 2.394586 | 68.427449 | 63.452588 | 78.673673 | 77.946444 |
| road | 5.201042 | 4.954428 | 25.754857 | 26.383889 | 97.116209 | 94.770811 |
| vegetation | 13.648080 | 13.527826 | 96.877545 | 97.033868 | 61.214331 | 60.772874 |
| vehicle | 0.898265 | 1.038456 | 25.537553 | 22.207343 | 97.100893 | 97.616780 |
| roof | 32.627365 | 32.719806 | 84.828280 | 84.474041 | 99.470782 | 99.336046 |
| water | 20.712295 | 20.668571 | 94.652753 | 94.843363 | 97.555085 | 97.545185 |

Diagnostics (image/window averages):
```json
{
  "tiles": 88.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 1764.0,
  "reference_views": 48.0,
  "reference_seconds": 2.7031835463421885,
  "reference_supported_alias_fraction": 0.6984020645216162,
  "mean_reference_trust": 0.18716583824411884,
  "effective_alias_count": 19.56881495124914,
  "mean_weight_kl_from_uniform": 0.022032687675403763,
  "minimum_alias_weight": 0.03472719605567611,
  "maximum_alias_weight": 0.07971925622339107,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.7367609197443182,
  "mean_absolute_margin_correction": 0.21452665000446236,
  "maximum_absolute_margin_correction": 0.44075434227190935,
  "pair_partition_error": 5.025755275379528e-07,
  "image_reference_candidates": 46750.0,
  "image_reference_selected_tokens": 439.0,
  "nonempty_reference_pool_fraction": 0.7181122079491615,
  "solver_mean_absolute_innovation": 1.5512135502966968,
  "solver_changed_patch_fraction": 0.4303685968572443,
  "solver_solver_relative_residual": 5.43213199230311e-07,
  "solver_solver_iterations": 7.998579545454546,
  "solver_energy_before": 38244.17479081587,
  "solver_energy_after": 19033.179219332607,
  "changed_observation_patch_fraction": 0.021426114169034095
}
```

## potsdam/potsdam

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 61.7181 | 51.6875 | 62.1577 | 47.5918 | 62.1533 | 56.2867 | 58.3560 |
| building | 71.1846 | 78.2026 | 75.5787 | 71.7127 | 75.5782 | 75.3732 | 75.2515 |
| low vegetation | 18.5600 | 33.4221 | 21.2306 | 24.4527 | 21.2339 | 22.9861 | 20.3610 |
| tree | 61.7470 | 62.5813 | 64.8712 | 65.5946 | 64.8726 | 63.5421 | 64.5756 |
| car | 27.3355 | 11.6582 | 24.2128 | 15.3844 | 24.2058 | 17.4996 | 20.0819 |
| clutter | 2.4836 | 4.5652 | 2.6325 | 3.4472 | 2.6331 | 2.7380 | 2.6879 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 39.274187 | 35.194637 | 73.673103 | 73.069619 | 79.906360 | 71.019648 |
| building | 15.458913 | 15.543437 | 77.023934 | 76.719998 | 97.577578 | 97.723960 |
| low vegetation | 5.198900 | 5.688162 | 82.126988 | 81.717294 | 22.259063 | 24.232347 |
| tree | 17.729750 | 17.316075 | 92.496299 | 92.591349 | 68.474759 | 66.945813 |
| car | 9.457812 | 13.167925 | 24.321427 | 17.543007 | 98.189579 | 98.606835 |
| clutter | 12.880437 | 13.089763 | 3.782286 | 3.909639 | 7.969785 | 8.372016 |

Diagnostics (image/window averages):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "reference_views": 4.0,
  "reference_seconds": 0.133780980395386,
  "reference_supported_alias_fraction": 0.75904576067761,
  "mean_reference_trust": 0.16875682285656995,
  "effective_alias_count": 19.669419729047352,
  "mean_weight_kl_from_uniform": 0.016857403996558183,
  "minimum_alias_weight": 0.03397039611202975,
  "maximum_alias_weight": 0.08472742150641151,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.8374294704861112,
  "mean_absolute_margin_correction": 0.2146411714428622,
  "maximum_absolute_margin_correction": 0.485868186586433,
  "pair_partition_error": 4.900826348198784e-07,
  "image_reference_candidates": 3844.0,
  "image_reference_selected_tokens": 372.5,
  "nonempty_reference_pool_fraction": 0.9173611253499985,
  "solver_mean_absolute_innovation": 1.7002332922485142,
  "solver_changed_patch_fraction": 0.2085639105902778,
  "solver_solver_relative_residual": 5.311337151820478e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 37499.16205512153,
  "solver_energy_after": 18641.57466634115,
  "changed_observation_patch_fraction": 0.07185872395833334
}
```

## udd5/udd5

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 61.1183 | 82.4937 | 68.3846 | 60.5494 | 68.3844 | 69.4036 | 67.9912 |
| building | 80.0717 | 83.9035 | 82.5264 | 83.5084 | 82.5263 | 82.6643 | 81.9520 |
| road | 34.6267 | 45.7856 | 39.8083 | 35.7577 | 39.8079 | 39.9650 | 40.8260 |
| vehicle | 17.2973 | 10.0092 | 20.4213 | 22.2983 | 20.4208 | 15.6983 | 22.0884 |
| other | 33.2928 | 30.5846 | 34.9177 | 33.9243 | 34.9174 | 34.3118 | 34.7459 |

non_residual_mean_iou_percent: BroadVIP=48.2785, Geometry=55.5480, Anchored_VIP=52.7852, PairAlias_Coupled=50.5284, SoftCounterfactual_Coupled=52.7849, StratifiedAlias_Coupled=51.9328, ShuffledStratified_Coupled=53.2144

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vegetation | 21.171638 | 21.506297 | 97.428230 | 97.393297 | 69.641764 | 70.717224 |
| building | 46.143661 | 46.084111 | 83.680536 | 83.806810 | 98.356258 | 98.377554 |
| road | 9.943371 | 9.302609 | 66.874410 | 69.714889 | 49.586043 | 48.361098 |
| vehicle | 3.400252 | 4.523717 | 20.950516 | 15.969273 | 88.991813 | 90.245346 |
| other | 19.341078 | 18.583266 | 48.510892 | 48.795008 | 55.479084 | 53.617527 |

Diagnostics (image/window averages):
```json
{
  "tiles": 80.8,
  "observer_empty_rows": 0.0,
  "observer_rows": 1763.9999999999998,
  "reference_views": 45.6,
  "reference_seconds": 2.469219615327893,
  "reference_supported_alias_fraction": 0.8395046931951912,
  "mean_reference_trust": 0.22391160267995266,
  "effective_alias_count": 19.462173214645095,
  "mean_weight_kl_from_uniform": 0.027568688455135645,
  "minimum_alias_weight": 0.03302487330408907,
  "maximum_alias_weight": 0.09286251484853864,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.9041255141749528,
  "mean_absolute_margin_correction": 0.3300817192848849,
  "maximum_absolute_margin_correction": 0.49608832060867414,
  "pair_partition_error": 4.88026575608687e-07,
  "image_reference_candidates": 42877.0,
  "image_reference_selected_tokens": 524.6,
  "nonempty_reference_pool_fraction": 0.9383000150322913,
  "solver_mean_absolute_innovation": 1.6208436912648154,
  "solver_changed_patch_fraction": 0.21130615234375,
  "solver_solver_relative_residual": 5.858441811495804e-07,
  "solver_solver_iterations": 7.996306818181816,
  "solver_energy_before": 29693.688211651417,
  "solver_energy_after": 14764.269120594367,
  "changed_observation_patch_fraction": 0.046754538796164785
}
```

## oem/oem

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rangeland | 32.8318 | 47.7780 | 37.5959 | 29.2852 | 37.5964 | 38.0057 | 37.2357 |
| developed space | 24.3872 | 28.5294 | 24.0531 | 40.5269 | 24.0555 | 26.4819 | 20.7856 |
| road | 37.8035 | 40.8291 | 42.1205 | 43.0614 | 42.1210 | 42.2917 | 41.5465 |
| tree | 34.2920 | 56.0657 | 38.7562 | 41.8149 | 38.7560 | 39.8649 | 39.9254 |
| water | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| agriculture land | 61.7665 | 79.1353 | 68.8612 | 68.3751 | 68.8610 | 68.8673 | 68.7260 |
| building | 33.8638 | 62.4971 | 44.2393 | 37.4412 | 44.2379 | 43.5108 | 41.8885 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bareland | 11.658568 | 10.495750 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| rangeland | 15.596476 | 15.636778 | 60.865298 | 61.258898 | 49.581240 | 50.030820 |
| developed space | 25.372753 | 25.685654 | 36.506364 | 39.195650 | 41.352686 | 44.946518 |
| road | 5.610920 | 5.422979 | 53.738540 | 54.729712 | 66.081716 | 65.046276 |
| tree | 10.957814 | 11.354538 | 92.750622 | 92.336690 | 39.966848 | 41.229012 |
| water | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| agriculture land | 23.086886 | 23.083382 | 70.382769 | 70.390970 | 96.956067 | 96.952647 |
| building | 7.716583 | 8.320918 | 77.044423 | 72.830964 | 50.955915 | 51.941640 |

Diagnostics (image/window averages):
```json
{
  "tiles": 8.375,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "reference_views": 4.0,
  "reference_seconds": 0.14202675927663222,
  "reference_supported_alias_fraction": 0.6929111139066259,
  "mean_reference_trust": 0.11776833970442466,
  "effective_alias_count": 19.85295988122622,
  "mean_weight_kl_from_uniform": 0.007404900725507307,
  "minimum_alias_weight": 0.034908763598650694,
  "maximum_alias_weight": 0.06906991066514619,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.7757839626736113,
  "mean_absolute_margin_correction": 0.17690217802333388,
  "maximum_absolute_margin_correction": 0.4825602130343517,
  "pair_partition_error": 5.03328111436632e-07,
  "image_reference_candidates": 3794.125,
  "image_reference_selected_tokens": 312.125,
  "nonempty_reference_pool_fraction": 0.6888671889901161,
  "solver_mean_absolute_innovation": 1.7886901427474287,
  "solver_changed_patch_fraction": 0.2547234429253472,
  "solver_solver_relative_residual": 4.9348670739925e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 55908.31218804253,
  "solver_energy_after": 27741.13454861111,
  "changed_observation_patch_fraction": 0.06429714626736112
}
```

## loveda/P

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 66.1383 | 64.4987 | 79.5953 | 77.8370 | 79.5953 | 77.4746 | 78.9195 |
| road | 68.0178 | 66.9348 | 68.0988 | 68.9912 | 68.1018 | 66.4204 | 66.5109 |
| water | 75.8814 | 81.1851 | 79.3816 | 79.7179 | 79.3821 | 79.3683 | 79.3147 |
| barren | 22.9545 | 24.3403 | 23.8204 | 25.1252 | 23.8205 | 22.3852 | 25.0542 |
| tree | 22.3040 | 53.7307 | 36.7931 | 44.9897 | 36.7924 | 32.4110 | 34.6885 |
| farm | 82.9699 | 86.2626 | 87.2288 | 88.5925 | 87.2288 | 86.7511 | 87.3445 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 0.559983 | 0.538401 | 96.060228 | 96.660987 | 82.281399 | 79.605065 |
| road | 10.079340 | 10.329078 | 69.130229 | 67.424741 | 97.855990 | 97.806600 |
| water | 22.353417 | 22.345663 | 85.614151 | 85.620498 | 91.599728 | 91.574740 |
| barren | 2.541220 | 2.431478 | 74.572148 | 73.275359 | 25.926192 | 24.375192 |
| tree | 4.126717 | 3.648931 | 99.033988 | 98.721924 | 36.925652 | 32.547569 |
| farm | 60.339323 | 60.706449 | 87.477867 | 86.974784 | 99.674712 | 99.704453 |

Diagnostics (image/window averages):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "reference_views": 4.0,
  "reference_seconds": 0.1527759819291532,
  "reference_supported_alias_fraction": 0.25216064818252765,
  "mean_reference_trust": 0.06503092405792031,
  "effective_alias_count": 19.865856601132286,
  "mean_weight_kl_from_uniform": 0.006844420095504233,
  "minimum_alias_weight": 0.03338387362762458,
  "maximum_alias_weight": 0.0778318572168549,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.30962456597222227,
  "mean_absolute_margin_correction": 0.09352479230575109,
  "maximum_absolute_margin_correction": 0.4817233697718216,
  "pair_partition_error": 4.3710072835286457e-07,
  "image_reference_candidates": 4096.0,
  "image_reference_selected_tokens": 200.125,
  "nonempty_reference_pool_fraction": 0.6855902969837189,
  "solver_mean_absolute_innovation": 1.8344749559958775,
  "solver_changed_patch_fraction": 0.34941948784722227,
  "solver_solver_relative_residual": 4.4490453237560564e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 43962.03255208333,
  "solver_energy_after": 21819.170884874133,
  "changed_observation_patch_fraction": 0.01117621527777778
}
```

## loveda/D

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 1.8793 | 34.1983 | 6.4379 | 23.4754 | 6.4377 | 6.0960 | 6.3870 |
| building | 33.5727 | 31.1862 | 36.3815 | 35.0399 | 36.3765 | 36.4123 | 36.4374 |
| road | 55.9896 | 45.4818 | 56.4727 | 60.6495 | 56.4727 | 55.7409 | 56.4817 |
| water | 61.1534 | 61.9193 | 62.7476 | 63.0229 | 62.7482 | 62.8147 | 62.7514 |
| barren | 18.9446 | 14.5296 | 20.0167 | 20.8316 | 20.0166 | 19.8416 | 20.0147 |
| tree | 21.5237 | 26.8904 | 31.2898 | 27.3199 | 31.2900 | 30.5827 | 31.3297 |
| farm | 39.9394 | 54.9851 | 42.5737 | 47.2383 | 42.5737 | 42.4791 | 42.5432 |

foreground_mean_iou_percent: BroadVIP=38.5206, Geometry=39.1654, Anchored_VIP=41.5803, PairAlias_Coupled=42.3504, SoftCounterfactual_Coupled=41.5796, StratifiedAlias_Coupled=41.3119, ShuffledStratified_Coupled=41.5930

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 3.743826 | 3.570792 | 78.161855 | 77.568963 | 6.555805 | 6.205375 |
| building | 0.731428 | 0.728392 | 39.876986 | 39.956832 | 80.584049 | 80.410304 |
| road | 6.745859 | 6.838000 | 57.182400 | 56.424790 | 97.849548 | 97.871942 |
| water | 15.892232 | 15.899150 | 66.617305 | 66.648868 | 91.526856 | 91.610078 |
| barren | 2.131796 | 2.115277 | 48.338401 | 48.231090 | 25.464179 | 25.210760 |
| tree | 2.304516 | 2.237748 | 87.202322 | 87.551138 | 32.795851 | 31.973048 |
| farm | 68.450343 | 68.610643 | 42.650758 | 42.554441 | 99.577365 | 99.585161 |

Diagnostics (image/window averages):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "reference_views": 4.0,
  "reference_seconds": 0.1527759819291532,
  "reference_supported_alias_fraction": 0.30074632921125366,
  "mean_reference_trust": 0.04645107844567696,
  "effective_alias_count": 19.938745786746342,
  "mean_weight_kl_from_uniform": 0.003096505021238174,
  "minimum_alias_weight": 0.034988833073940545,
  "maximum_alias_weight": 0.07319200292436612,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.4607611762152778,
  "mean_absolute_margin_correction": 0.11376269963251041,
  "maximum_absolute_margin_correction": 0.4605601190382408,
  "pair_partition_error": 5.231963263617621e-07,
  "image_reference_candidates": 4096.0,
  "image_reference_selected_tokens": 233.125,
  "nonempty_reference_pool_fraction": 0.7126275263726711,
  "solver_mean_absolute_innovation": 1.8597678525580303,
  "solver_changed_patch_fraction": 0.42659505208333337,
  "solver_solver_relative_residual": 4.519672782205614e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 52486.19276258681,
  "solver_energy_after": 26048.058213975695,
  "changed_observation_patch_fraction": 0.005479600694444445
}
```

## vaihingen/vaihingen

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 56.0846 | 49.2002 | 60.0516 | 49.1039 | 60.0475 | 60.0366 | 58.3710 |
| building | 64.8387 | 74.6610 | 68.2387 | 67.5033 | 68.2380 | 67.2152 | 67.3678 |
| low vegetation | 31.7052 | 46.7115 | 35.3912 | 33.4317 | 35.3935 | 38.3522 | 34.6568 |
| tree | 65.1005 | 71.8329 | 71.8529 | 71.6802 | 71.8527 | 71.8211 | 71.7240 |
| car | 20.4239 | 8.7566 | 21.7111 | 12.8463 | 21.7054 | 20.1756 | 20.4646 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| impervious surface | 30.987112 | 29.098024 | 74.590862 | 76.985484 | 75.495230 | 73.168664 |
| building | 31.327659 | 31.853403 | 68.466284 | 67.395732 | 99.515206 | 99.603128 |
| low vegetation | 9.462479 | 10.221650 | 96.447758 | 96.742485 | 35.858648 | 38.853943 |
| tree | 21.803020 | 21.828941 | 81.885794 | 81.817149 | 85.432142 | 85.462007 |
| car | 6.419729 | 6.997982 | 21.955742 | 20.343436 | 95.117436 | 96.071022 |

Diagnostics (image/window averages):
```json
{
  "tiles": 9.0,
  "observer_empty_rows": 0.0,
  "observer_rows": 3528.0,
  "reference_views": 4.0,
  "reference_seconds": 0.14451736604678445,
  "reference_supported_alias_fraction": 0.7859192023364205,
  "mean_reference_trust": 0.20082346223191255,
  "effective_alias_count": 19.581477327479256,
  "mean_weight_kl_from_uniform": 0.02130255409873725,
  "minimum_alias_weight": 0.03237744001671672,
  "maximum_alias_weight": 0.0891900999057624,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.8608127170138888,
  "mean_absolute_margin_correction": 0.2724811606038,
  "maximum_absolute_margin_correction": 0.49926381392611396,
  "pair_partition_error": 4.76837158203125e-07,
  "image_reference_candidates": 3844.0,
  "image_reference_selected_tokens": 354.75,
  "nonempty_reference_pool_fraction": 1.0,
  "solver_mean_absolute_innovation": 1.5279615173737209,
  "solver_changed_patch_fraction": 0.1953396267361111,
  "solver_solver_relative_residual": 6.719379604823594e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 25449.939643012156,
  "solver_energy_after": 12662.355278862848,
  "changed_observation_patch_fraction": 0.030409071180555556
}
```

## landcoverai/landcoverai

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 86.0090 | 82.6090 | 87.4477 | 85.4383 | 87.4432 | 87.7779 | 87.8877 |
| building | 48.3391 | 34.9562 | 44.4672 | 48.1165 | 44.4603 | 45.3290 | 44.6304 |
| woodland | 71.8857 | 78.3638 | 78.6026 | 67.4444 | 78.5815 | 78.6381 | 80.6309 |
| water | 98.3440 | 93.6635 | 97.6178 | 97.8977 | 97.6173 | 97.8248 | 97.5179 |
| road | 24.7413 | 14.9318 | 26.3947 | 28.2869 | 26.4039 | 27.8129 | 25.8738 |

foreground_mean_iou_percent: BroadVIP=60.8275, Geometry=55.4788, Anchored_VIP=61.7706, PairAlias_Coupled=60.4364, SoftCounterfactual_Coupled=61.7657, StratifiedAlias_Coupled=62.4012, ShuffledStratified_Coupled=62.1633

non_residual_mean_iou_percent: BroadVIP=60.8275, Geometry=55.4788, Anchored_VIP=61.7706, PairAlias_Coupled=60.4364, SoftCounterfactual_Coupled=61.7657, StratifiedAlias_Coupled=62.4012, ShuffledStratified_Coupled=62.1632

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| background | 66.968584 | 67.165995 | 93.854887 | 93.904611 | 92.758703 | 93.081427 |
| building | 3.323317 | 3.261328 | 44.812397 | 45.680240 | 98.297296 | 98.331917 |
| woodland | 18.492031 | 18.530273 | 94.956499 | 94.875299 | 82.027162 | 82.126509 |
| water | 8.473158 | 8.455229 | 97.617828 | 97.824824 | 100.000000 | 100.000000 |
| road | 2.742910 | 2.587175 | 28.852807 | 30.565641 | 75.598980 | 75.539765 |

Diagnostics (image/window averages):
```json
{
  "tiles": 1.0,
  "observer_empty_rows": 1984.5,
  "observer_rows": 3528.0,
  "reference_views": 1.0,
  "reference_seconds": 0.04148209007689729,
  "reference_supported_alias_fraction": 0.2998565718212376,
  "mean_reference_trust": 0.08638888650580157,
  "effective_alias_count": 19.83044320344925,
  "mean_weight_kl_from_uniform": 0.00864550515331075,
  "minimum_alias_weight": 0.03681263606995344,
  "maximum_alias_weight": 0.0731443502008915,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.354248046875,
  "mean_absolute_margin_correction": 0.09766800036959467,
  "maximum_absolute_margin_correction": 0.3663575202226639,
  "pair_partition_error": 3.5762786865234375e-07,
  "image_reference_candidates": 1024.0,
  "image_reference_selected_tokens": 107.125,
  "nonempty_reference_pool_fraction": 0.5272500310093164,
  "solver_mean_absolute_innovation": 1.6790449172258377,
  "solver_changed_patch_fraction": 0.085205078125,
  "solver_solver_relative_residual": 4.553739358925668e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 30784.520751953125,
  "solver_energy_after": 15246.23828125,
  "changed_observation_patch_fraction": 0.007080078125
}
```

## flair1/flair1

| Class | VIP20 | Geometry | All20 coupled | Hard coupled | Soft-v1 | Stratified-v2 | Shuffled-v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 55.6097 | 49.6979 | 56.7755 | 50.8733 | 56.7841 | 57.1605 | 57.2427 |
| pervious surface | 39.6383 | 57.0621 | 47.2756 | 35.6016 | 47.2880 | 48.2994 | 48.4482 |
| impervious surface | 50.9692 | 52.6509 | 51.9858 | 43.3158 | 51.9861 | 51.9118 | 51.8494 |
| bare soil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| water | 57.7226 | 73.2328 | 61.4565 | 61.6412 | 61.4638 | 61.6249 | 61.5149 |
| coniferous | 33.4100 | 43.0233 | 43.5282 | 38.3688 | 43.5282 | 43.5490 | 43.5139 |
| deciduous | 55.8985 | 54.0229 | 59.5237 | 61.5086 | 59.5268 | 59.9343 | 58.6180 |
| brushwood | 7.5457 | 18.6136 | 12.3287 | 8.3147 | 12.3291 | 12.1299 | 11.6238 |
| vineyard | 0.0000 | -- | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| herbaceous vegetation | 49.0459 | 60.2329 | 57.4725 | 47.1791 | 57.4717 | 57.5627 | 57.5612 |
| agricultural land | 6.6409 | 18.7339 | 12.9546 | 10.2167 | 12.9546 | 12.9403 | 12.9671 |
| plowed land | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Class | All20 area % | V2 area % | All20 precision % | V2 precision % | All20 recall % | V2 recall % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| building | 11.913060 | 12.156585 | 58.039586 | 57.850870 | 96.305746 | 97.954872 |
| pervious surface | 9.242159 | 9.390182 | 91.699829 | 92.085691 | 49.388969 | 50.391138 |
| impervious surface | 21.911219 | 21.687491 | 59.778065 | 59.985615 | 79.952187 | 79.410583 |
| bare soil | 7.147416 | 6.960849 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| water | 6.704063 | 6.685077 | 63.250247 | 63.429167 | 95.588868 | 95.587792 |
| coniferous | 0.481516 | 0.481325 | 65.732118 | 65.768087 | 56.305160 | 56.313646 |
| deciduous | 15.308859 | 15.477060 | 78.801504 | 78.688530 | 70.872195 | 71.548158 |
| brushwood | 3.540817 | 3.327822 | 26.221270 | 26.805808 | 18.877056 | 18.137027 |
| vineyard | 0.001622 | 0.001622 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| herbaceous vegetation | 21.520673 | 21.603438 | 90.825182 | 90.707308 | 61.014784 | 61.169947 |
| agricultural land | 1.429951 | 1.427899 | 13.914465 | 13.904386 | 65.253442 | 65.112641 |
| plowed land | 0.798647 | 0.800650 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Diagnostics (image/window averages):
```json
{
  "tiles": 1.0,
  "observer_empty_rows": 441.0,
  "observer_rows": 3528.0,
  "reference_views": 1.0,
  "reference_seconds": 0.045343013014644384,
  "reference_supported_alias_fraction": 0.3196594309556531,
  "mean_reference_trust": 0.05722458051423018,
  "effective_alias_count": 19.904009133577347,
  "mean_weight_kl_from_uniform": 0.004899445083140108,
  "minimum_alias_weight": 0.03533815569244325,
  "maximum_alias_weight": 0.07978834304958582,
  "shuffled_weight_spectrum_error": 0.0,
  "active_patch_fraction": 0.4703369140625,
  "mean_absolute_margin_correction": 0.06884709235964692,
  "maximum_absolute_margin_correction": 0.44173385202884674,
  "pair_partition_error": 4.76837158203125e-07,
  "image_reference_candidates": 1024.0,
  "image_reference_selected_tokens": 169.625,
  "nonempty_reference_pool_fraction": 0.500651054084301,
  "solver_mean_absolute_innovation": 1.547217920422554,
  "solver_changed_patch_fraction": 0.1671142578125,
  "solver_solver_relative_residual": 4.805117832518135e-07,
  "solver_solver_iterations": 8.0,
  "solver_energy_before": 62744.00732421875,
  "solver_energy_after": 31107.3046875,
  "changed_observation_patch_fraction": 0.01904296875
}
```

## Combined Seven-Arm Cost

Suite wall seconds: 1131.5905

| Dataset | Worker evaluation seconds | Peak allocated CUDA MiB |
| --- | ---: | ---: |
| vdd | 253.1645 | 6238.5161 |
| potsdam | 26.7432 | 5617.8750 |
| udd5 | 1050.4310 | 5993.0161 |
| oem | 26.5254 | 5658.9531 |
| loveda | 50.0717 | 5770.1934 |
| vaihingen | 26.7637 | 5597.8413 |
| landcoverai | 4.6888 | 5492.1602 |
| flair1 | 5.3144 | 5628.7959 |

Combined workload, not standalone model latency. Image-wide reference collection adds Geometry views. Original models/results preserved. No full eight-dataset rollout launched.

## Post-Selection Reference Purity Audit

Potsdam/OEM fixed8 images each. Masks loaded only after frozen image-only reference construction. Audit-only, never used to modify the rule. These are candidate-pool patch-center labels, not query-specific retrieved purity, majority-patch truth, or independent-object correctness.

| Dataset | Declared class | Pool instances | Pool purity % | Confidence-weighted purity % | Unique class witnesses | Unique purity % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| potsdam/potsdam | impervious surface | 57007 | 79.8148 | 87.1266 | 724 | 77.0718 |
| potsdam/potsdam | building | 36880 | 82.2587 | 91.9102 | 459 | 80.1743 |
| potsdam/potsdam | low vegetation | 21548 | 73.0601 | 73.5027 | 338 | 76.0355 |
| potsdam/potsdam | tree | 49937 | 92.8210 | 93.6722 | 518 | 92.6641 |
| potsdam/potsdam | car | 29420 | 52.2706 | 67.3778 | 364 | 46.1538 |
| potsdam/potsdam | clutter | 34027 | 8.3375 | 12.0955 | 578 | 5.5363 |
| oem/oem | bareland | 31945 | 0.0000 | 0.0000 | 303 | 0.0000 |
| oem/oem | rangeland | 18615 | 97.3355 | 98.3952 | 189 | 95.7672 |
| oem/oem | developed space | 61467 | 50.8012 | 52.2927 | 738 | 53.6585 |
| oem/oem | road | 42233 | 73.1513 | 82.4363 | 324 | 71.9136 |
| oem/oem | tree | 36463 | 98.6013 | 99.0128 | 306 | 97.7124 |
| oem/oem | water | 0 | -- | -- | 0 | -- |
| oem/oem | agriculture land | 26883 | 82.7363 | 94.8000 | 254 | 74.8031 |
| oem/oem | building | 33017 | 94.1303 | 96.2008 | 350 | 93.1429 |

The audit can reveal jointly wrong witnesses; it does not isolate their causal contribution to a score change.
