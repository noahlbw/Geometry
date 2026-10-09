# Semantic-matched rival soft weights

Same64 developed512 windows and five frozen vocabularies. Geometry20, wide/fine visual observations, hard admission support, fine-target projection, unscreened posterior and H remain unchanged. Each alias selects its nearest ORIGINAL template-mean text rival; survivor weight is 1-positive_cosine*sigmoid(beta*(native_rival_alias-native_own_alias)). No new visual forward, text/template change, fitted parameters or target-driven domain/count selection. This is a two-phrase response allocation, not calibrated semantic correctness. Not full datasets or independent validation.

Three historical stress endpoint scores/per-image confusions and the preceding fine-support per-image confusions replay exactly. Primary singleton scores match. Scores persist before masks; canonical/hard protection and control spectra/mass verify.

| Scenario | Projected hard | Previous fine support | Matched candidate | Class mean | Identity-null mean | Text only | Candidate-projected pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k20 | 45.4187 | 45.2175 | 45.4293 | 45.4132 | 45.4288 | 45.3413 | +0.0105 |
| k30 | 45.3651 | 45.5166 | 45.7740 | 45.2942 | 45.3290 | 45.2551 | +0.4089 |
| k40 | 44.2044 | 44.3148 | 44.6498 | 44.0772 | 44.0749 | 44.0456 | +0.4454 |
| wrong_parent | 44.5969 | 44.4042 | 44.5533 | 44.5920 | 44.6126 | 44.6125 | -0.0436 |
| paraphrase | 45.9874 | 45.7910 | 45.9468 | 45.9423 | 45.9485 | 45.9275 | -0.0406 |

## Frozen Accuracy And Identity Checks

```json
{
  "clean_gain_at_least_point1pp": false,
  "all_stress_means_above_projected": false,
  "worst_protocol_loss_within1pp": false,
  "wrong_parent_damage_no_worse": false,
  "clean_and_wrong_parent_above_class_mean_and_identity_null": false,
  "worst_protocol_delta_pp": -1.0974144356889255,
  "this40_pool_harmless_vs20": false,
  "frozen_advancement_gate": false
}
```

## Per-Protocol Results

| Dataset/protocol | Scenario | Projected hard | Matched candidate | Delta pp |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 54.1183 | 54.6394 | +0.5211 |
| vdd/vdd | k30 | 59.7020 | 59.7585 | +0.0565 |
| vdd/vdd | k40 | 57.4750 | 57.8744 | +0.3994 |
| vdd/vdd | wrong_parent | 49.5236 | 50.5805 | +1.0569 |
| vdd/vdd | paraphrase | 53.5458 | 54.0024 | +0.4566 |
| potsdam/potsdam | k20 | 41.1231 | 41.2637 | +0.1406 |
| potsdam/potsdam | k30 | 42.3003 | 43.3195 | +1.0193 |
| potsdam/potsdam | k40 | 40.1319 | 41.4888 | +1.3568 |
| potsdam/potsdam | wrong_parent | 38.0572 | 38.2295 | +0.1723 |
| potsdam/potsdam | paraphrase | 44.0639 | 44.4972 | +0.4333 |
| udd5/udd5 | k20 | 34.6174 | 34.4827 | -0.1347 |
| udd5/udd5 | k30 | 38.4878 | 38.3598 | -0.1280 |
| udd5/udd5 | k40 | 36.5185 | 36.2632 | -0.2553 |
| udd5/udd5 | wrong_parent | 31.7720 | 31.8884 | +0.1164 |
| udd5/udd5 | paraphrase | 36.4067 | 36.5569 | +0.1501 |
| oem/oem | k20 | 40.4484 | 39.9166 | -0.5318 |
| oem/oem | k30 | 33.7154 | 34.5811 | +0.8658 |
| oem/oem | k40 | 32.8811 | 33.4658 | +0.5847 |
| oem/oem | wrong_parent | 41.6142 | 40.8979 | -0.7163 |
| oem/oem | paraphrase | 40.9347 | 40.5629 | -0.3718 |
| loveda/P | k20 | 52.6436 | 52.1559 | -0.4877 |
| loveda/D | k20 | 37.6023 | 37.8496 | +0.2473 |
| loveda/P | k30 | 54.1237 | 54.2297 | +0.1060 |
| loveda/D | k30 | 37.5578 | 37.9024 | +0.3446 |
| loveda/P | k40 | 56.0367 | 55.5734 | -0.4633 |
| loveda/D | k40 | 37.6099 | 37.2183 | -0.3916 |
| loveda/P | wrong_parent | 53.8062 | 53.3736 | -0.4326 |
| loveda/D | wrong_parent | 37.2366 | 37.3080 | +0.0715 |
| loveda/P | paraphrase | 50.2645 | 49.1671 | -1.0974 |
| loveda/D | paraphrase | 36.9648 | 36.1807 | -0.7841 |
| vaihingen/vaihingen | k20 | 53.3059 | 53.2724 | -0.0335 |
| vaihingen/vaihingen | k30 | 50.5371 | 51.3786 | +0.8415 |
| vaihingen/vaihingen | k40 | 49.4814 | 50.4252 | +0.9438 |
| vaihingen/vaihingen | wrong_parent | 55.0127 | 54.8949 | -0.1178 |
| vaihingen/vaihingen | paraphrase | 54.3697 | 54.4424 | +0.0728 |
| landcoverai/landcoverai | k20 | 67.7836 | 67.9281 | +0.1445 |
| landcoverai/landcoverai | k30 | 67.5423 | 67.6907 | +0.1484 |
| landcoverai/landcoverai | k40 | 67.3024 | 67.5488 | +0.2464 |
| landcoverai/landcoverai | wrong_parent | 69.3972 | 69.0616 | -0.3356 |
| landcoverai/landcoverai | paraphrase | 66.2833 | 66.1829 | -0.1003 |
| flair1/flair1 | k20 | 34.3507 | 34.0817 | -0.2690 |
| flair1/flair1 | k30 | 33.0785 | 33.2016 | +0.1231 |
| flair1/flair1 | k40 | 32.2348 | 32.9136 | +0.6788 |
| flair1/flair1 | wrong_parent | 34.1619 | 33.5657 | -0.5961 |
| flair1/flair1 | paraphrase | 35.3300 | 35.1487 | -0.1813 |

## Source Weight Outcomes

These are position/alias/rival source comparisons, not distinct pixels or causal per-word error attribution.

| Dataset/protocol | Scenario | Survivor mean weight | Broad-positive/fine-positive mean weight | Joint comparisons |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | k20 | 0.599614 | 0.767615 | 2055443 |
| vdd/vdd | k30 | 0.609594 | 0.765052 | 2967554 |
| vdd/vdd | k40 | 0.605012 | 0.759085 | 3882530 |
| vdd/vdd | wrong_parent | 0.560491 | 0.716903 | 1668215 |
| vdd/vdd | paraphrase | 0.599932 | 0.770347 | 1922930 |
| potsdam/potsdam | k20 | 0.589607 | 0.762146 | 1629354 |
| potsdam/potsdam | k30 | 0.586524 | 0.753519 | 2286213 |
| potsdam/potsdam | k40 | 0.584928 | 0.750438 | 3000010 |
| potsdam/potsdam | wrong_parent | 0.582615 | 0.734214 | 1464586 |
| potsdam/potsdam | paraphrase | 0.595299 | 0.773355 | 1557790 |
| udd5/udd5 | k20 | 0.587674 | 0.759042 | 945416 |
| udd5/udd5 | k30 | 0.569845 | 0.722567 | 1324132 |
| udd5/udd5 | k40 | 0.564337 | 0.710548 | 1721712 |
| udd5/udd5 | wrong_parent | 0.532500 | 0.658650 | 676283 |
| udd5/udd5 | paraphrase | 0.594054 | 0.768143 | 861594 |
| oem/oem | k20 | 0.561517 | 0.765363 | 3058487 |
| oem/oem | k30 | 0.553370 | 0.740691 | 4553339 |
| oem/oem | k40 | 0.551294 | 0.730779 | 6040366 |
| oem/oem | wrong_parent | 0.531758 | 0.712254 | 2542786 |
| oem/oem | paraphrase | 0.570748 | 0.779480 | 3014736 |
| loveda/P | k20 | 0.563853 | 0.712454 | 1334872 |
| loveda/D | k20 | 0.551188 | 0.695854 | 1916743 |
| loveda/P | k30 | 0.556828 | 0.708554 | 2030217 |
| loveda/D | k30 | 0.547021 | 0.692727 | 2970393 |
| loveda/P | k40 | 0.552003 | 0.699274 | 2741093 |
| loveda/D | k40 | 0.546116 | 0.687331 | 3949090 |
| loveda/P | wrong_parent | 0.559740 | 0.704309 | 1156893 |
| loveda/D | wrong_parent | 0.545162 | 0.694257 | 1676750 |
| loveda/P | paraphrase | 0.573490 | 0.728911 | 1308718 |
| loveda/D | paraphrase | 0.562236 | 0.715983 | 1872654 |
| vaihingen/vaihingen | k20 | 0.594109 | 0.774865 | 1044208 |
| vaihingen/vaihingen | k30 | 0.586649 | 0.769899 | 1434550 |
| vaihingen/vaihingen | k40 | 0.582283 | 0.763512 | 1819068 |
| vaihingen/vaihingen | wrong_parent | 0.573453 | 0.737957 | 898035 |
| vaihingen/vaihingen | paraphrase | 0.596859 | 0.781060 | 980353 |
| landcoverai/landcoverai | k20 | 0.584975 | 0.775767 | 1163977 |
| landcoverai/landcoverai | k30 | 0.578531 | 0.749207 | 1678485 |
| landcoverai/landcoverai | k40 | 0.576759 | 0.744626 | 2110864 |
| landcoverai/landcoverai | wrong_parent | 0.568145 | 0.750691 | 966154 |
| landcoverai/landcoverai | paraphrase | 0.588429 | 0.783586 | 1108895 |
| flair1/flair1 | k20 | 0.587284 | 0.764705 | 7132130 |
| flair1/flair1 | k30 | 0.578655 | 0.753136 | 10310584 |
| flair1/flair1 | k40 | 0.565410 | 0.722040 | 13461248 |
| flair1/flair1 | wrong_parent | 0.586977 | 0.759017 | 6810117 |
| flair1/flair1 | paraphrase | 0.592455 | 0.773846 | 6981940 |

Exact per-class IoU/precision/recall/area and all controls are retained in summary.json and per_image_confusions.npz. Original scored supports remain fixed; LoveDA D counts once and P separately. No post-result control promotion, parameter changes or full rollout. No-extra-forwards does not establish zero latency overhead. A failed accuracy gate ends advancement of this specific candidate; all prior options and this implementation/results remain preserved.

## Interpretation And Retained Decision

The candidate is a useful expansion experiment, not a replacement of the retained
20-word model. It improves30/40 means by0.4089/0.4454pp over projected hard,
with7/8 and6/8 main-domain wins respectively. Its gaps above the identity-null
means are0.4450/0.5748pp, versus only0.00043pp at20. Thus phrase identity carries
useful information in these particular expanded pools, but no meaningful clean20
identity advantage is established. Wrong-parent and paraphrase both lose to their
null means. Do not select a count/domain/seed from these outcomes.

|40-word class|Projected IoU|Matched IoU|Main accompanying change|
|---|---:|---:|---|
|Potsdam car|18.6362|20.3375|Precision18.6372->20.3407%; area19.2335->17.6142%; recall99.9707->99.9229%|
|Potsdam low vegetation|26.0957|28.2340|Recall28.4642->31.0121%|
|OEM building|7.6778|12.1701|Recall7.7172->12.2451%; most target coverage still absent|
|FLAIR herbaceous vegetation|38.2996|43.5728|Recall38.7136->44.1848%|
|VDD wall|37.2223|38.3927|Precision39.9584->41.0619%|
|VDD vehicle|55.6585|54.6222|Precision worsens with recall still100%|

The worst protocol loss is LoveDA P paraphrase-1.0974pp. In that protocol
building precision/IoU38.6650->34.8468 accompanies area0.06853->0.07604% with
100% recall; only307 target pixels exist. Elsewhere clean LoveDA D tree recall
50.4327->46.2008% falls despite a higher building IoU. These are competition
tradeoffs, not uniform word-error suppression. None identifies which phrase
caused a particular final error.

Surviving noncanonical weight means are roughly0.53-0.61, while jointly
broad-positive/fine-positive comparisons average0.66-0.78. Therefore the source
does reach the old sign-rule blind spot; reaching it does not imply correct
rejection. Semantic proximity and two-phrase responses are not a reliable
semantic correctness witness across clean/replacement conditions. Canonical
protection and normalization also change relative prior mass when words are
attenuated. The class-mean/identity controls separate some of this mechanism,
but do not causally identify the reason for each domain loss.

Keep the existing full-suite hard model and the frozen20 projected transfer
candidate; preserve this new implementation/results without promotion or
post-result formula tuning. Fourteen local/remote tests pass, including six
new tests and eight imported cached-reader regressions. All64 keys, fixed
observers/words, historical endpoints and control identities verify. No
standalone timing follows this failed accuracy gate. All eight workers completed.

A next distinct test can hold the original survivor mass and canonical prior
fixed while redistributing weight among noncanonical survivors. That would
separate conditional word discrimination from changed class/anchor prior mass;
it has NOT run and is not already supported as a better model. Existing
matching, support, hard and projected options remain unchanged.
