# Whole-rival text-span positive residual alias support

Same64 developed512 windows, eight/domain/five frozen vocabularies; not full or independent validation. Original Geometry20, visual observations, hard support, canonical/survivor mass, T0 projection/posterior/H remain fixed. Only the alias source changes: ridge1 decomposition of a unit query over the entire rival text dictionary, then positive-exclusive response divided by positive-exclusive+positive-shared response. Fine raw scores before salience, original template means/norms and prior epsilon are reused. No additional visual/text forwards, fitted threshold or count/domain routing. This is a regularized decomposition, not exact orthogonal projection or semantic correctness probability.

All historical three stress scores/confusions and preceding survivor per-image confusions are exact. Class-mean scores exactly recover projected hard. Canonical1/rejected0, original survivor mass, null spectra and primary singleton/source identities verify. Scores persist before masks.

| Scenario | Projected hard | Prior survivor | Residual source | Pure-text control | Identity-null mean | Candidate-projected pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k20 | 45.4187 | 45.4534 | 45.2100 | 45.2976 | 45.8649 | -0.2087 |
| k30 | 45.3651 | 45.8423 | 45.2075 | 45.1328 | 45.7961 | -0.1576 |
| k40 | 44.2044 | 44.7500 | 44.1128 | 43.7202 | 44.5016 | -0.0916 |
| wrong_parent | 44.5969 | 44.5669 | 44.0736 | 44.5837 | 44.8523 | -0.5233 |
| paraphrase | 45.9874 | 45.9915 | 45.5321 | 45.7646 | 46.1353 | -0.4553 |

## Result And Interpretation

The candidate FAILS all frozen advancement checks. Means change versus projected
hard by-0.2087pp at20,-0.1576 at30,-0.0916 at40,-0.5233 for wrong-parent and-0.4553
for paraphrase. Four/eight clean20 main domains improve, but LoveDA D loses1.9195pp.
Worst protocol loss is LoveDA D paraphrase-3.1314pp.40 stays1.0972pp below own20.
Its wrong-parent damage relative to own20 is1.1364pp versus projected hard0.8218pp.

The matched identity-null deficit is0.6549pp at20 and0.7787pp for wrong-parent;
every other regime also loses its matched null mean. Pure-text control also beats
the visual source at20, wrong-parent and paraphrase. Preserving support, canonical
prior, survivor mass and weight spectra does not rescue this source. Do not promote
the randomized controls or adjust ridge/weight strength after these outcomes.

Summing the eight main protocols (LoveDA D once),41.19% of eligible source
position/alias/rival comparisons are at the epsilon floor at20;43.79% for
wrong-parent and45.02% at40. Their source-weight means are0.2647/0.2157/0.2207.
Jointly wide/fine-positive source-weight means are0.5667/0.5027/0.4993. These are
source comparisons, not word deletion counts or pixel error counts; survivor
mass is redistributed afterward and some allocated weights exceed1.

At20, Potsdam car IoU25.1793->26.3960 improves but low vegetation21.7324->18.9682
and recall22.8310%->19.8208% worsen. VDD vehicle38.2597->41.5974 improves while
wall53.1449->52.1211, roof89.8815->88.6861 and water66.5525->65.2666 worsen.
In wrong-parent, LandCover.ai building50.8759->46.1433 while recall remains near
98.4%; increased false positives are not solved. Useful class coverage and rival
overactivation are both affected. A local car/vehicle recovery is not a general
selector success.

The decomposition has a valid algebraic conditional-response interpretation, but
positive response outside the rival span is not a demonstrated utility estimate.
Shared semantic/visual components need not be harmful, and exclusive components
need not be correct. The current predictions and identity controls contradict
using this particular response fraction as a reliable all-domain source. They
do not prove all residual methods, soft weighting or alias screening impossible.

Keep established full-suite RivalFineHard_Exact, the frozen20 projected transfer
candidate and the equivalent crop-graph speed backend. Twelve local/remote tests
pass; every class-mean score recovers projected hard exactly, historical scores/
confusions and all64 input/word/model identities verify. Actual survivor mass
error<=7.106e-14. All eight workers completed; GPU0-7 idle. No isolated latency
claim or full20092 rollout follows the failed accuracy screen. All schemes remain.

## Frozen Advancement Checks

```json
{
  "clean_gain_at_least_point1pp": false,
  "all_stress_means_above_projected": false,
  "worst_protocol_loss_within1pp": false,
  "wrong_parent_damage_no_worse": false,
  "clean_and_wrong_parent_above_class_text_and_identity_controls": false,
  "worst_protocol_delta_pp": -3.131393647985874,
  "this40_pool_harmless_vs20": false,
  "frozen_advancement_gate": false
}
```

## Per-Protocol Results

| Dataset/protocol | Scenario | Projected | Residual source | Delta pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 54.1183 | 54.5225 | +0.4042 |
| vdd/vdd | k30 | 59.7020 | 57.4640 | -2.2380 |
| vdd/vdd | k40 | 57.4750 | 54.9273 | -2.5477 |
| vdd/vdd | wrong_parent | 49.5236 | 48.9336 | -0.5901 |
| vdd/vdd | paraphrase | 53.5458 | 53.7583 | +0.2124 |
| potsdam/potsdam | k20 | 41.1231 | 41.1008 | -0.0223 |
| potsdam/potsdam | k30 | 42.3003 | 43.2054 | +0.9051 |
| potsdam/potsdam | k40 | 40.1319 | 41.7070 | +1.5751 |
| potsdam/potsdam | wrong_parent | 38.0572 | 38.1219 | +0.0647 |
| potsdam/potsdam | paraphrase | 44.0639 | 44.1185 | +0.0546 |
| udd5/udd5 | k20 | 34.6174 | 34.1338 | -0.4837 |
| udd5/udd5 | k30 | 38.4878 | 38.4909 | +0.0031 |
| udd5/udd5 | k40 | 36.5185 | 36.2465 | -0.2720 |
| udd5/udd5 | wrong_parent | 31.7720 | 30.8429 | -0.9291 |
| udd5/udd5 | paraphrase | 36.4067 | 35.9464 | -0.4603 |
| oem/oem | k20 | 40.4484 | 40.5826 | +0.1341 |
| oem/oem | k30 | 33.7154 | 34.7622 | +1.0468 |
| oem/oem | k40 | 32.8811 | 34.0919 | +1.2108 |
| oem/oem | wrong_parent | 41.6142 | 41.1531 | -0.4611 |
| oem/oem | paraphrase | 40.9347 | 41.1438 | +0.2091 |
| loveda/P | k20 | 52.6436 | 51.0660 | -1.5776 |
| loveda/D | k20 | 37.6023 | 35.6828 | -1.9195 |
| loveda/P | k30 | 54.1237 | 53.1154 | -1.0083 |
| loveda/D | k30 | 37.5578 | 35.6573 | -1.9005 |
| loveda/P | k40 | 56.0367 | 52.9914 | -3.0453 |
| loveda/D | k40 | 37.6099 | 35.4785 | -2.1313 |
| loveda/P | wrong_parent | 53.8062 | 51.1986 | -2.6077 |
| loveda/D | wrong_parent | 37.2366 | 36.8565 | -0.3801 |
| loveda/P | paraphrase | 50.2645 | 48.2698 | -1.9948 |
| loveda/D | paraphrase | 36.9648 | 33.8334 | -3.1314 |
| vaihingen/vaihingen | k20 | 53.3059 | 53.5479 | +0.2420 |
| vaihingen/vaihingen | k30 | 50.5371 | 51.0615 | +0.5245 |
| vaihingen/vaihingen | k40 | 49.4814 | 49.9857 | +0.5043 |
| vaihingen/vaihingen | wrong_parent | 55.0127 | 54.4414 | -0.5713 |
| vaihingen/vaihingen | paraphrase | 54.3697 | 54.2399 | -0.1298 |
| landcoverai/landcoverai | k20 | 67.7836 | 67.9330 | +0.1494 |
| landcoverai/landcoverai | k30 | 67.5423 | 67.9769 | +0.4347 |
| landcoverai/landcoverai | k40 | 67.3024 | 67.8283 | +0.5258 |
| landcoverai/landcoverai | wrong_parent | 69.3972 | 68.5571 | -0.8401 |
| landcoverai/landcoverai | paraphrase | 66.2833 | 66.0722 | -0.2111 |
| flair1/flair1 | k20 | 34.3507 | 34.1766 | -0.1741 |
| flair1/flair1 | k30 | 33.0785 | 33.0421 | -0.0364 |
| flair1/flair1 | k40 | 32.2348 | 32.6373 | +0.4024 |
| flair1/flair1 | wrong_parent | 34.1619 | 33.6825 | -0.4793 |
| flair1/flair1 | paraphrase | 35.3300 | 35.1440 | -0.1859 |

## Source Response

Fractions and means form from summed source comparisons, not averaged window ratios. They do not count pixel errors or quantify individual phrase causal usefulness. After source formation the writer redistributes original noncanonical survivor mass, so source and allocated weights differ.

| Dataset/protocol | Scenario | Mean source weight | Source at epsilon fraction | Joint-positive mean source weight |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 0.274272 | 0.346933 | 0.534780 |
| vdd/vdd | k30 | 0.244799 | 0.375789 | 0.500197 |
| vdd/vdd | k40 | 0.228347 | 0.396129 | 0.477625 |
| vdd/vdd | wrong_parent | 0.197727 | 0.395222 | 0.434684 |
| vdd/vdd | paraphrase | 0.273945 | 0.347586 | 0.540360 |
| potsdam/potsdam | k20 | 0.278780 | 0.395857 | 0.587677 |
| potsdam/potsdam | k30 | 0.248210 | 0.425856 | 0.554366 |
| potsdam/potsdam | k40 | 0.237107 | 0.438459 | 0.538447 |
| potsdam/potsdam | wrong_parent | 0.230834 | 0.419416 | 0.515139 |
| potsdam/potsdam | paraphrase | 0.278189 | 0.392163 | 0.585731 |
| udd5/udd5 | k20 | 0.278473 | 0.413830 | 0.591796 |
| udd5/udd5 | k30 | 0.255103 | 0.438539 | 0.567784 |
| udd5/udd5 | k40 | 0.226749 | 0.453251 | 0.518036 |
| udd5/udd5 | wrong_parent | 0.176076 | 0.458893 | 0.440974 |
| udd5/udd5 | paraphrase | 0.276426 | 0.415769 | 0.601730 |
| oem/oem | k20 | 0.255233 | 0.441220 | 0.558993 |
| oem/oem | k30 | 0.236257 | 0.453323 | 0.525109 |
| oem/oem | k40 | 0.232014 | 0.468781 | 0.525212 |
| oem/oem | wrong_parent | 0.197876 | 0.466133 | 0.476497 |
| oem/oem | paraphrase | 0.253985 | 0.432740 | 0.554259 |
| loveda/P | k20 | 0.193854 | 0.362285 | 0.406900 |
| loveda/D | k20 | 0.186247 | 0.369421 | 0.389505 |
| loveda/P | k30 | 0.178163 | 0.397503 | 0.390139 |
| loveda/D | k30 | 0.173402 | 0.401223 | 0.374512 |
| loveda/P | k40 | 0.170590 | 0.417313 | 0.377598 |
| loveda/D | k40 | 0.166076 | 0.418593 | 0.363494 |
| loveda/P | wrong_parent | 0.166561 | 0.385120 | 0.381650 |
| loveda/D | wrong_parent | 0.161439 | 0.397224 | 0.372329 |
| loveda/P | paraphrase | 0.199587 | 0.351576 | 0.416613 |
| loveda/D | paraphrase | 0.192017 | 0.359258 | 0.399660 |
| vaihingen/vaihingen | k20 | 0.279896 | 0.399252 | 0.584861 |
| vaihingen/vaihingen | k30 | 0.244968 | 0.436291 | 0.559002 |
| vaihingen/vaihingen | k40 | 0.231946 | 0.450234 | 0.545725 |
| vaihingen/vaihingen | wrong_parent | 0.210945 | 0.453755 | 0.504098 |
| vaihingen/vaihingen | paraphrase | 0.275638 | 0.395158 | 0.577623 |
| landcoverai/landcoverai | k20 | 0.252052 | 0.410862 | 0.527244 |
| landcoverai/landcoverai | k30 | 0.217629 | 0.438572 | 0.484065 |
| landcoverai/landcoverai | k40 | 0.201167 | 0.445817 | 0.463916 |
| landcoverai/landcoverai | wrong_parent | 0.186885 | 0.459338 | 0.455395 |
| landcoverai/landcoverai | paraphrase | 0.249217 | 0.399833 | 0.522231 |
| flair1/flair1 | k20 | 0.283816 | 0.437278 | 0.622462 |
| flair1/flair1 | k30 | 0.252344 | 0.456662 | 0.570741 |
| flair1/flair1 | k40 | 0.226757 | 0.471316 | 0.521843 |
| flair1/flair1 | wrong_parent | 0.252744 | 0.446635 | 0.571211 |
| flair1/flair1 | paraphrase | 0.280884 | 0.425854 | 0.612603 |

Exact per-class IoU/precision/recall/area and every control are in summary.json and per_image_confusions.npz. Prior fixed supports; LoveDA D once/P separately. All source/outputs remain. A failed gate ends this rule without tuning ridge/epsilon or promoting a control/count. A pass requires frozen complete-input transfer and matched independent timing. No automatic full20092 run follows this screen.
