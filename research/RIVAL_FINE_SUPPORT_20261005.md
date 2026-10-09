# Fine support weights on original conditional hard admission

Same64 developed512 windows, original Geometry20, finite wide/fine observations and reconstruction. All five pools/stresses share the original visual observations. Only surviving alias weights change: w=old_hard_keep*sigmoid(beta*native_alias_rival_margin), with numerical floor and canonical/self-rival protection. Original projection target and no-admission posterior remain. No labels fit weights or select counts. Not full datasets or independent validation.

This is fine response responsibility, not semantic correctness probability. It can reweight low-response words left untouched by the earlier contradiction rule; high jointly wrong responses need not be rejected. Every historical endpoint score/per-image confusion is exact. Primary singleton scores match. Scores persist before masks; fixed hard support and weight-control spectra/mass are verified.

| Scenario | No admission | Original hard | Projected hard | Support candidate | Class-mean control | Three-seed identity-null mean | Candidate-projected pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k20 | 42.8679 | 45.0622 | 45.4187 | 45.2175 | 45.2538 | 45.3427 | -0.2012 |
| k30 | 43.0357 | 45.1366 | 45.3651 | 45.5166 | 45.3332 | 45.3949 | +0.1515 |
| k40 | 41.7421 | 43.9695 | 44.2044 | 44.3148 | 43.9499 | 43.9461 | +0.1104 |
| wrong_parent | 41.9892 | 44.2848 | 44.5969 | 44.4042 | 44.5997 | 44.6615 | -0.1927 |
| paraphrase | 43.6469 | 45.6935 | 45.9874 | 45.7910 | 45.8916 | 45.8606 | -0.1963 |

## Frozen Checks

```json
{
  "clean_gain_at_least_point1pp": false,
  "all_stress_means_above_projected": false,
  "worst_protocol_loss_within1pp": false,
  "wrong_parent_damage_no_worse": true,
  "clean_and_wrong_parent_above_class_mean_and_identity_null": false,
  "worst_protocol_delta_pp": -2.1228012625099026,
  "this40_pool_harmless_vs20": false,
  "frozen_advancement_gate": false
}
```

## Interpretation And Decision

The frozen advancement gate FAILS. Mean change versus projected hard is-0.2012pp
at20 words,-0.1927pp under wrong-parent replacements and-0.1963pp under legitimate
paraphrases. The worst protocol loss is2.1228pp on LoveDA D paraphrases. Do not
replace the retained20 transfer candidate, promote an identity-null control or
switch rules by count/dataset. Accuracy did not justify a standalone timing study;
there is no measured no-latency-regression claim for this candidate.

There is a narrower positive result:30/40 gain0.1515/0.1104pp versus projected hard.
At40 the real weights exceed class-mean by0.3648pp and the identity-null mean by
0.3687pp; at30 the corresponding gains are0.1834/0.1216pp. Thus this weight source
contains useful conditional identity information in these expanded pools, but not
universally. At20 it LOSES to class mean by0.0363pp and null mean by0.1252pp; under
wrong-parent replacements it loses0.1955/0.2573pp. An absolute fine-response
responsibility is not a reliable estimate of the useful contribution of a word.

The class tradeoffs are concrete. VDD20 improves1.3559pp overall; wall IoU rises
53.1449->59.0757 and vehicle38.2597->41.5399. But VDD40 loses0.9580pp overall;
vehicle IoU55.6585->52.7647. Potsdam40 improves1.2982pp; car IoU18.6362->20.9210
and low vegetation26.0957->26.7939. At20, car improves25.1793->26.2242 while low
vegetation falls21.7324->19.5019. OEM40 building IoU7.6778->9.8772 and recall
7.7172%->9.9330% barely repair the coverage collapse; at20 building IoU instead
falls54.5491->51.4685. These are developed-window outcomes, not full datasets.

This rejects the specified all-surviving-word fine-support weighting as the new
complete selector. It does not prove soft weighting impossible or justify more
parameter curves on these masks. Both low-response coverage dilution and high
jointly-wrong alias responses remain concerns; the first has a limited benefit
here, while the second is not certified by the source. A next candidate must
estimate conditional marginal discriminative contribution, not equate strong
fine response or observer agreement with correctness. All outputs, the old full
model, exact acceleration and frozen20 projected candidate remain preserved.
Related pathway diagnosis: RIVAL_EXPANSION_ATTRIBUTION_20261005.md.

## Per-Protocol Results

| Dataset/protocol | Scenario | Projected hard | Support candidate | Delta pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 54.1183 | 55.4742 | +1.3559 |
| vdd/vdd | k30 | 59.7020 | 59.2368 | -0.4652 |
| vdd/vdd | k40 | 57.4750 | 56.5170 | -0.9580 |
| vdd/vdd | wrong_parent | 49.5236 | 50.9195 | +1.3958 |
| vdd/vdd | paraphrase | 53.5458 | 54.8884 | +1.3426 |
| potsdam/potsdam | k20 | 41.1231 | 41.0548 | -0.0683 |
| potsdam/potsdam | k30 | 42.3003 | 43.0382 | +0.7380 |
| potsdam/potsdam | k40 | 40.1319 | 41.4301 | +1.2982 |
| potsdam/potsdam | wrong_parent | 38.0572 | 38.0302 | -0.0270 |
| potsdam/potsdam | paraphrase | 44.0639 | 44.3071 | +0.2433 |
| udd5/udd5 | k20 | 34.6174 | 34.3021 | -0.3153 |
| udd5/udd5 | k30 | 38.4878 | 39.0741 | +0.5863 |
| udd5/udd5 | k40 | 36.5185 | 37.4980 | +0.9796 |
| udd5/udd5 | wrong_parent | 31.7720 | 31.7572 | -0.0148 |
| udd5/udd5 | paraphrase | 36.4067 | 36.5862 | +0.1795 |
| oem/oem | k20 | 40.4484 | 39.8216 | -0.6268 |
| oem/oem | k30 | 33.7154 | 33.8935 | +0.1781 |
| oem/oem | k40 | 32.8811 | 33.2056 | +0.3245 |
| oem/oem | wrong_parent | 41.6142 | 40.9594 | -0.6548 |
| oem/oem | paraphrase | 40.9347 | 40.3683 | -0.5664 |
| loveda/P | k20 | 52.6436 | 51.5947 | -1.0489 |
| loveda/D | k20 | 37.6023 | 36.2543 | -1.3480 |
| loveda/P | k30 | 54.1237 | 53.8241 | -0.2996 |
| loveda/D | k30 | 37.5578 | 37.4093 | -0.1486 |
| loveda/P | k40 | 56.0367 | 54.8883 | -1.1484 |
| loveda/D | k40 | 37.6099 | 35.7437 | -1.8661 |
| loveda/P | wrong_parent | 53.8062 | 52.5649 | -1.2413 |
| loveda/D | wrong_parent | 37.2366 | 36.6446 | -0.5920 |
| loveda/P | paraphrase | 50.2645 | 48.7965 | -1.4681 |
| loveda/D | paraphrase | 36.9648 | 34.8420 | -2.1228 |
| vaihingen/vaihingen | k20 | 53.3059 | 53.2089 | -0.0970 |
| vaihingen/vaihingen | k30 | 50.5371 | 50.7046 | +0.1676 |
| vaihingen/vaihingen | k40 | 49.4814 | 49.7939 | +0.3125 |
| vaihingen/vaihingen | wrong_parent | 55.0127 | 54.5335 | -0.4792 |
| vaihingen/vaihingen | paraphrase | 54.3697 | 54.3380 | -0.0316 |
| landcoverai/landcoverai | k20 | 67.7836 | 67.7251 | -0.0585 |
| landcoverai/landcoverai | k30 | 67.5423 | 67.7962 | +0.2539 |
| landcoverai/landcoverai | k40 | 67.3024 | 67.7096 | +0.4072 |
| landcoverai/landcoverai | wrong_parent | 69.3972 | 68.8074 | -0.5898 |
| landcoverai/landcoverai | paraphrase | 66.2833 | 66.0086 | -0.2747 |
| flair1/flair1 | k20 | 34.3507 | 33.8990 | -0.4517 |
| flair1/flair1 | k30 | 33.0785 | 32.9801 | -0.0984 |
| flair1/flair1 | k40 | 32.2348 | 32.6202 | +0.3853 |
| flair1/flair1 | wrong_parent | 34.1619 | 33.5817 | -0.5802 |
| flair1/flair1 | paraphrase | 35.3300 | 34.9896 | -0.3404 |

Exact class IoU/precision/recall/area and all controls are retained in summary.json and per_image_confusions.npz. Scored supports are fixed from the preceding five-scenario study; LoveDA D counts once and P separately. No automatic full rollout/control promotion or post-result parameter changes. No extra visual forwards does not prove zero latency overhead; standalone timing must follow a passed accuracy gate before making that claim. All prior models and outputs remain.
