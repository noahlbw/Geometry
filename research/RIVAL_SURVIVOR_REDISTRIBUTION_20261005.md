# Survivor-mass and canonical-prior-preserving alias redistribution

Same64 developed512 windows and five frozen vocabularies. Geometry20, wide/fine observations, hard support, projection/posterior/H remain unchanged. The preceding matched source u is unchanged; noncanonical survivor weight w=|S|*u/sum_S(u). Canonical weight1 and rejected weight0. Positive weights can exceed1, but original survivor mass/canonical prior and K/remaining normalizer stay fixed. No new visual/text forward, fitted coefficient or domain/count routing. Not full datasets or independent validation.

Three original stress scores/per-image confusions and preceding support/matched per-image confusions replay exactly. Every class-mean score is bitwise identical to projected hard. Primary singleton scores match; canonical prior/total mass and identity-null spectra verify. All scores precede masks. Response weights are not calibrated semantic correctness.

| Scenario | Projected hard | Prior matched | Redistribution | Absolute redistribution | Text redistribution | Identity-null mean | Candidate-projected pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k20 | 45.4187 | 45.4293 | 45.4534 | 45.3756 | 45.3871 | 45.4550 | +0.0346 |
| k30 | 45.3651 | 45.7740 | 45.8423 | 45.4592 | 45.3014 | 45.3977 | +0.4772 |
| k40 | 44.2044 | 44.6498 | 44.7500 | 44.4679 | 44.0993 | 44.1970 | +0.5456 |
| wrong_parent | 44.5969 | 44.5533 | 44.5669 | 44.2496 | 44.6055 | 44.6261 | -0.0301 |
| paraphrase | 45.9874 | 45.9468 | 45.9915 | 45.8826 | 45.9615 | 45.9594 | +0.0041 |

## Frozen Accuracy And Identity Checks

```json
{
  "clean_gain_at_least_point1pp": false,
  "all_stress_means_above_projected": false,
  "worst_protocol_loss_within1pp": true,
  "wrong_parent_damage_no_worse": false,
  "clean_and_wrong_parent_above_matched_identity_and_source_controls": false,
  "worst_protocol_delta_pp": -0.9804421384827862,
  "this40_pool_harmless_vs20": false,
  "frozen_advancement_gate": false
}
```

## Per-Protocol Results

| Dataset/protocol | Scenario | Projected hard | Redistribution | Delta pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 54.1183 | 54.1480 | +0.0297 |
| vdd/vdd | k30 | 59.7020 | 59.7288 | +0.0268 |
| vdd/vdd | k40 | 57.4750 | 58.0117 | +0.5367 |
| vdd/vdd | wrong_parent | 49.5236 | 49.8036 | +0.2799 |
| vdd/vdd | paraphrase | 53.5458 | 53.5622 | +0.0164 |
| potsdam/potsdam | k20 | 41.1231 | 41.3622 | +0.2391 |
| potsdam/potsdam | k30 | 42.3003 | 43.4372 | +1.1369 |
| potsdam/potsdam | k40 | 40.1319 | 41.6072 | +1.4753 |
| potsdam/potsdam | wrong_parent | 38.0572 | 38.2139 | +0.1567 |
| potsdam/potsdam | paraphrase | 44.0639 | 44.5969 | +0.5330 |
| udd5/udd5 | k20 | 34.6174 | 34.6423 | +0.0249 |
| udd5/udd5 | k30 | 38.4878 | 38.4503 | -0.0375 |
| udd5/udd5 | k40 | 36.5185 | 36.3258 | -0.1927 |
| udd5/udd5 | wrong_parent | 31.7720 | 31.9441 | +0.1721 |
| udd5/udd5 | paraphrase | 36.4067 | 36.5426 | +0.1359 |
| oem/oem | k20 | 40.4484 | 40.2482 | -0.2003 |
| oem/oem | k30 | 33.7154 | 34.7570 | +1.0416 |
| oem/oem | k40 | 32.8811 | 33.6268 | +0.7457 |
| oem/oem | wrong_parent | 41.6142 | 41.0730 | -0.5412 |
| oem/oem | paraphrase | 40.9347 | 40.8418 | -0.0929 |
| loveda/P | k20 | 52.6436 | 52.5648 | -0.0789 |
| loveda/D | k20 | 37.6023 | 37.3739 | -0.2284 |
| loveda/P | k30 | 54.1237 | 54.2840 | +0.1602 |
| loveda/D | k30 | 37.5578 | 37.7347 | +0.1769 |
| loveda/P | k40 | 56.0367 | 55.5379 | -0.4989 |
| loveda/D | k40 | 37.6099 | 37.3050 | -0.3049 |
| loveda/P | wrong_parent | 53.8062 | 53.5479 | -0.2583 |
| loveda/D | wrong_parent | 37.2366 | 37.7325 | +0.4959 |
| loveda/P | paraphrase | 50.2645 | 49.2841 | -0.9804 |
| loveda/D | paraphrase | 36.9648 | 36.3018 | -0.6630 |
| vaihingen/vaihingen | k20 | 53.3059 | 53.6510 | +0.3451 |
| vaihingen/vaihingen | k30 | 50.5371 | 51.5396 | +1.0025 |
| vaihingen/vaihingen | k40 | 49.4814 | 50.5419 | +1.0605 |
| vaihingen/vaihingen | wrong_parent | 55.0127 | 54.9446 | -0.0681 |
| vaihingen/vaihingen | paraphrase | 54.3697 | 54.6565 | +0.2868 |
| landcoverai/landcoverai | k20 | 67.7836 | 67.9538 | +0.1702 |
| landcoverai/landcoverai | k30 | 67.5423 | 67.7796 | +0.2373 |
| landcoverai/landcoverai | k40 | 67.3024 | 67.6160 | +0.3135 |
| landcoverai/landcoverai | wrong_parent | 69.3972 | 69.1900 | -0.2072 |
| landcoverai/landcoverai | paraphrase | 66.2833 | 66.1462 | -0.1370 |
| flair1/flair1 | k20 | 34.3507 | 34.2476 | -0.1031 |
| flair1/flair1 | k30 | 33.0785 | 33.3112 | +0.2327 |
| flair1/flair1 | k40 | 32.2348 | 32.9654 | +0.7306 |
| flair1/flair1 | wrong_parent | 34.1619 | 33.6331 | -0.5287 |
| flair1/flair1 | paraphrase | 35.3300 | 35.2840 | -0.0460 |

## Source Redistribution

Each original noncanonical survivor mass is conserved. Fractions below count position/alias/rival comparisons, not distinct pixels or per-phrase causal usefulness.

| Dataset/protocol | Scenario | Mean survivor weight | Fraction above1 | Fraction below1 |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 1.000000 | 0.487390 | 0.512553 |
| vdd/vdd | k30 | 1.000000 | 0.492066 | 0.507924 |
| vdd/vdd | k40 | 1.000000 | 0.487651 | 0.512347 |
| vdd/vdd | wrong_parent | 1.000000 | 0.469817 | 0.530181 |
| vdd/vdd | paraphrase | 1.000000 | 0.487590 | 0.512389 |
| potsdam/potsdam | k20 | 1.000000 | 0.490274 | 0.509706 |
| potsdam/potsdam | k30 | 1.000000 | 0.482552 | 0.517447 |
| potsdam/potsdam | k40 | 1.000000 | 0.482502 | 0.517497 |
| potsdam/potsdam | wrong_parent | 1.000000 | 0.475515 | 0.524479 |
| potsdam/potsdam | paraphrase | 1.000000 | 0.482476 | 0.517514 |
| udd5/udd5 | k20 | 1.000000 | 0.480638 | 0.519357 |
| udd5/udd5 | k30 | 1.000000 | 0.484857 | 0.515138 |
| udd5/udd5 | k40 | 1.000000 | 0.481968 | 0.518032 |
| udd5/udd5 | wrong_parent | 1.000000 | 0.452069 | 0.547931 |
| udd5/udd5 | paraphrase | 1.000000 | 0.481171 | 0.518827 |
| oem/oem | k20 | 1.000000 | 0.480422 | 0.519567 |
| oem/oem | k30 | 1.000000 | 0.472356 | 0.527641 |
| oem/oem | k40 | 1.000000 | 0.469855 | 0.530144 |
| oem/oem | wrong_parent | 1.000000 | 0.455960 | 0.544035 |
| oem/oem | paraphrase | 1.000000 | 0.477302 | 0.522691 |
| loveda/P | k20 | 1.000000 | 0.484543 | 0.515295 |
| loveda/D | k20 | 1.000000 | 0.478184 | 0.521699 |
| loveda/P | k30 | 1.000000 | 0.477987 | 0.521989 |
| loveda/D | k30 | 1.000000 | 0.474272 | 0.525711 |
| loveda/P | k40 | 1.000000 | 0.478775 | 0.521222 |
| loveda/D | k40 | 1.000000 | 0.477234 | 0.522763 |
| loveda/P | wrong_parent | 1.000000 | 0.481065 | 0.518935 |
| loveda/D | wrong_parent | 1.000000 | 0.472343 | 0.527657 |
| loveda/P | paraphrase | 1.000000 | 0.482765 | 0.517095 |
| loveda/D | paraphrase | 1.000000 | 0.478578 | 0.521322 |
| vaihingen/vaihingen | k20 | 1.000000 | 0.493772 | 0.506194 |
| vaihingen/vaihingen | k30 | 1.000000 | 0.479171 | 0.520827 |
| vaihingen/vaihingen | k40 | 1.000000 | 0.478845 | 0.521154 |
| vaihingen/vaihingen | wrong_parent | 1.000000 | 0.471485 | 0.528507 |
| vaihingen/vaihingen | paraphrase | 1.000000 | 0.484468 | 0.515516 |
| landcoverai/landcoverai | k20 | 1.000000 | 0.494294 | 0.505656 |
| landcoverai/landcoverai | k30 | 1.000000 | 0.494721 | 0.505276 |
| landcoverai/landcoverai | k40 | 1.000000 | 0.484346 | 0.515654 |
| landcoverai/landcoverai | wrong_parent | 1.000000 | 0.472132 | 0.527859 |
| landcoverai/landcoverai | paraphrase | 1.000000 | 0.482891 | 0.517072 |
| flair1/flair1 | k20 | 1.000000 | 0.487039 | 0.512958 |
| flair1/flair1 | k30 | 1.000000 | 0.478356 | 0.521644 |
| flair1/flair1 | k40 | 1.000000 | 0.470517 | 0.529483 |
| flair1/flair1 | wrong_parent | 1.000000 | 0.482218 | 0.517781 |
| flair1/flair1 | paraphrase | 1.000000 | 0.489939 | 0.510059 |

All exact class metrics and controls are in summary.json/per_image_confusions.npz. Original scored supports are fixed; LoveDA D counts once/P separately. No post-result control promotion, coefficient/count/domain changes or full rollout. No-extra-forwards is not a standalone latency measurement. A failed gate ends advancement of this candidate; all earlier schemes and this implementation/results remain preserved.

## Interpretation And Retained Decision

Mass preservation improves this particular matched source modestly over its
preceding pure-attenuation variant in every scenario mean.30/40 gains versus
projected hard become0.4772/0.5456pp (7/8 and6/8 main-domain wins). Their gaps
above the same-spectrum alias-null means are0.4446/0.5530pp. Therefore their
word-identity benefit is not explained solely by increased canonical prior or
changed total survivor mass. These are developed-window, pool-specific findings,
not untouched validation or proof that40-word expansion is harmless.

Clean20 gains only0.0346pp, with5/8 main-domain wins. It is0.00160pp BELOW the
alias-null mean. Wrong-parent loses0.0301pp versus projected and0.0592pp versus
the null mean. Paraphrase improves0.0041pp; worst protocol loss remains LoveDA P
paraphrase-0.9804pp. The frozen advancement gate FAILS. Do not select a count,
domain or shuffle seed, relax the gate, or relabel a control as the final model.

|40-word class|Projected IoU|Redistributed IoU|Accompanying evidence|
|---|---:|---:|---|
|Potsdam car|18.6362|20.3866|Precision18.6372->20.3898%; area19.2335->17.5718%; recall99.9707->99.9229%|
|Potsdam low vegetation|26.0957|28.6215|Recall28.4642->31.4793%|
|OEM building|7.6778|12.2461|Recall7.7172->12.3216%; most correct coverage still missing|
|FLAIR herbaceous vegetation|38.2996|44.0185|Recall38.7136->44.6846%|
|VDD wall|37.2223|37.9303|Recall84.4623->85.8799%; area slightly increases|
|VDD vehicle|55.6585|54.5230|Precision worsens; recall remains100%|

Across real inputs/controls, maximum total-mass error is2.132e-14 and maximum
canonical-prior error8.327e-17. Every real class-mean endpoint score is bitwise
equal to original projected hard, and its per-image confusions are exact.
The mechanism really redistributes word mass; it does not restore output
equivalence by hiding behind matched scalar metrics.13 local/remote tests pass
(five new and eight imported reader regressions). Fixed words/observers,
historical endpoints, primary singleton scores and control spectra verify.

Keep this mass-preserving option and all prior schemes, without promoting it
over the frozen20 projected transfer candidate or the established full-suite
hard model. No standalone timing follows the failed accuracy gate. All eight
workers completed; no full20092 rollout or monitor was created.

The next useful diagnostic is whether requested word corrections are blocked
by the unchanged, UNSCREENED fine-minus-wide class target used for projection.
Word weighting reaches jointly positive responses, but the class-level target
still comes from the unmodified pool. This is a specific possible mismatch,
not a demonstrated cause of the remaining errors. It has NOT been audited or
repaired here; do not claim another sigmoid/normalizer will solve it.
