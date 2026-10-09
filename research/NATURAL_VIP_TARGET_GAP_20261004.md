# Natural-image target and verified gaps

The target values below are transcribed from the user-provided VIP paper table.
Published numbers and local runs are separate columns. A local numerical match
on VOC does not certify reproduction of every paper protocol or its timing.
The main Geometry/wide/fine-rival/coupled equations remain unchanged.

## Full results only

| Protocol | Paper VIP target | Fixed20 ImageNet coupling | Variable semantic pool, default | Previous proxy-calibrated pool | Local official finite VIP | New sense/actual-reader calibration |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VOC21 | 73.2 | 30.2858 | 57.3617 | 63.6504 | 73.2591 | default59.0436 / calibrated62.5527, complete |
| Context60 | 41.2 | not evaluated | not evaluated | not evaluated | not evaluated | frozen transfer queue waiting |
| COCO-Object81 | 47.4 | 24.9252 | 43.2917 | 47.1188 | 48.9955 | queued |
| VOC20 | 92.5 | 89.7780 | 91.8146 | 89.6651 | 92.5051 | default91.8121 / calibrated89.6655, complete |
| Context59 | 46.5 | not evaluated | not evaluated | not evaluated | not evaluated | frozen transfer queue waiting |
| COCO-Stuff171 | 33.3 | 29.5178 | running | running | running | full shard0 running |
| Cityscapes19 | 55.7 | not evaluated | not evaluated | not evaluated | not evaluated | manual download needed |
| ADE150 | 29.1 | 25.0705 | 28.7845 | 28.8354 | 29.1387 | 1/2 full shards complete |

The five fixed20 runs have verified complete unique coverage: VOC1449 images
for each protocol, ADE2000, Stuff5000 and Object5000. The completed variable
pool runs have identical VOC/ADE/COCO-Object global sample-key signatures. Other running
datasets are not given scores based on partial results.

The v2 VOC20 full run is complete with1449 unique images, verified per-image
confusions and exact agreement of its Source confusion with the previous run.
New words at the old parameters score89.6655 (+0.0004pp over89.6651); default
parameters score91.8121; the frozen calibration returns89.6655, losing2.1466pp
against its own default arm. Its shared multi-arm wall time is2779.077s and peak
CUDA allocation6716.287MiB, not standalone inference cost. This is a calibration
regression, not evidence that the unchanged core model needs another module.

The v2 VOC21 full run is now also complete:1449 unique images, verified per-image
confusions, and exact Source confusion agreement with the original run. Words at
source parameters score63.6490 versus the original63.6504. Default settings score
59.0436 (57.3772 without rejection); the frozen class-threshold calibration scores
62.5527. It remains approximately1.10pp below the original calibrated pool and
10.7064pp below local finite VIP73.2591. Shared multi-arm wall=3242.829s and peak
CUDA allocation=6716.287MiB. The negative result is retained.

The VOC21 full confusion matrices locate a concrete failure: foreground expansion onto
scored background, not simply inadequate foreground recall. The calibrated model
predicts62.8859% background area versus VIP72.2112%, with background recall84.1215%
versus94.7269%. Selected foreground examples follow; these labels diagnose the
already frozen run and do not select the new vocabulary.

| Class | Our precision / VIP (%) | Our recall / VIP (%) | Our predicted area / VIP (%) | Fraction of our false positives whose GT is background (%) |
| --- | ---: | ---: | ---: | ---: |
| aeroplane | 36.0068 / 74.6603 | 99.4852 / 93.0149 | 2.1706 / 0.9787 | 99.8126 |
| dog | 60.9741 / 93.9420 | 98.2011 / 93.4700 | 3.3109 / 2.0455 | 99.0081 |
| train | 50.6221 / 86.1117 | 95.7384 / 86.3946 | 3.0266 / 1.6056 | 99.8793 |

High VOC20 foreground-only scores therefore do not prove that scored-background
competition is solved. This evidence motivates an actual foreground-versus-background
calibration objective; foreground retention alone cannot establish improved precision.
It does not prove which individual alias caused those false positives.

## Completed COCO-Object Full Comparison

COCO-Object now has5000 unique full images, verified per-image confusions,
identical native20 sample-key SHA/checkpoint identities, and all81 classes in
common support. Each paired arm and the native20 baseline score exactly the same
1312660894 target pixels. Target masks did not select the frozen inputs.

The variable-pool default fine-rival result is43.2917, versus fixed20/ImageNet
24.9252 (+18.3665pp). The frozen proxy-calibrated result is47.1188 (+22.1936pp
over fixed20, +3.8271pp over its own default), versus local finite VIP48.9955
(-1.8767pp). These are jointly changed words/templates/background representation
and calibration, not a pure word-count or deletion ablation. The selector retained
324 of325 pool aliases, removing only one person alias; there is no isolated
evidence here that this deletion caused the gain.

At the frozen calibrated settings, removing background rejection gives38.2471;
restoring its frozen threshold gives47.1188 (+8.8717pp). Thresholding helps its
own unthresholded output, but the remaining failure is over-rejection of genuine
foreground, unlike VOC21's foreground expansion. Predicted background area is
77.5712%, versus GT69.2309% and VIP68.4347%. Foreground micro precision/recall is
74.8061%/54.5292%, versus VIP68.0646%/69.8259%. Foreground mIoU is46.7061 versus
48.5939; background IoU is80.1355 versus81.1218.

| Class | Our IoU / VIP | Our precision / VIP (%) | Our recall / VIP (%) |
| --- | ---: | ---: | ---: |
| person | 39.7715 / 66.0598 | 91.8254 / 80.4132 | 41.2314 / 78.7277 |
| donut | 11.1696 / 40.3985 | 91.1951 / 91.0346 | 11.2913 / 42.0724 |
| scissors | 45.9836 / 65.9977 | 92.7480 / 82.8922 | 47.6986 / 76.4048 |

The calibrated model wins31 classes and loses50 against the local VIP. These
post-freeze labels diagnose coverage loss; they do not identify harmful words
or choose a new threshold. A uniform background boost is therefore not a
cross-protocol remedy. The already frozen actual-reader/qualified-word queues
remain unchanged and are needed to separate lexical improvements from calibration.
Shared seven-arm wall time is29643.663s, aggregate GPU time58864.203s, and peak
CUDA allocation6716.287MiB; these are not standalone model inference costs.
Full matrices are in `research/natural_text_adaptation_20261003/coco_object81/full_merged.json`.

## Completed ADE Alias And Count Controls

The frozen-reader ADE controls have2000 unique images, all150 classes in common
support,448102043 identical target pixels, and matching sample/checkpoint
identities against the original variable-pool and native20 runs. Source
confusions exactly reproduce the original calibrated output. Reader equations,
source templates, tau/tem and rejection remain fixed in this comparison.

| Frozen arm | Full mIoU | Delta against Source |
| --- | ---: | ---: |
| Source | 28.8354 | 0.0000 |
| Image-only selected count normalization, strength1 | 28.2600 | -0.5754 |
| Curated word-deletion control | 28.8891 | +0.0537 |
| Matched local finite VIP | 29.1387 | +0.3032 |

The count rule selected strength1 from64 unlabeled images: pseudo-witness error
fell2.5328pp, with upper95 paired delta-0.9782pp. Its full GT mIoU nevertheless
falls0.5754pp. This is another objective mismatch, not a successful adaptive
calibration. The recorded frozen choice is retained; strength0 is not substituted
after seeing the labels. Count normalization wins61 classes and loses89. For
example, barrel recall changes96.2112% to97.0190%, but precision falls34.5143%
to19.0966% and IoU34.0515 to18.9852. Barrel's own one-alias group is unchanged:
changing the competition can create false positives without removing its words.

A paired image bootstrap now quantifies these full ADE differences rather than
treating a small positive point estimate as a stable improvement. It uses2000
replicates, seed20261004, the same2000 images and fixed150-class common support.
Every loaded per-image confusion sum exactly reproduces the verified full
result, and each arm scores identical target pixels in every paired image.

| Frozen arm versus Source | Full delta (pp) | Paired percentile95% interval (pp) |
| --- | ---: | --- |
| Image-only selected count normalization | -0.5754 | [-0.7740, -0.3414] |
| Curated word deletion | +0.0537 | [-0.1163, +0.2105] |

The count-rule loss is consistently negative under this image-resampling
analysis. The deletion interval crosses zero, so its tiny aggregate gain does
not establish a reliable screening improvement. These exploratory intervals
do not correct for development reuse, multiple comparisons or unobserved
scene dependence; they do not establish independent generalization. Neither
result changes a frozen vocabulary or promotes a different calibration arm.
The next existing test remains positive category-qualified replacements rather
than a deletion quota. Reproduce with
`F:/APP/codetool/anaconda3/python.exe tools/report_natural_alias_uncertainty.py --dataset ade150 --replicates 2000`.
Exact values are in
`research/natural_count_calibration_20261004/full/ade150/paired_uncertainty.json`;
four focused tests verify paired identical-arm intervals, sufficient-count
resampling equivalence, fixed support and rejected-pixel false negatives.

The word-only control removes stall/sales booth from base and clothes closet/
clothespress from apparel, reducing382 queries to378. Its global gain is small,
and the class outcomes expose a coverage-versus-competition tradeoff:

| Class | Source IoU | Curated IoU | Source recall (%) | Curated recall (%) |
| --- | ---: | ---: | ---: | ---: |
| base | 6.5837 | 0.8964 | 35.1859 | 2.2820 |
| apparel | 14.1783 | 18.7223 | 91.6801 | 32.7813 |
| wardrobe | 36.5143 | 48.7302 | 49.1126 | 72.6794 |

GT wardrobe pixels predicted as apparel decrease489188 to25 under the combined
two-class deletion control. Apparel precision improves14.3632% to30.3885%, but
its recall falls substantially. Base loses most of its correct coverage. This
does not identify each deleted alias's separate causal effect, or prove the
deleted terms are appropriate taxonomy synonyms. It supports testing specific
positive replacements and preserving valid subtype coverage, rather than
deletion-only quotas or uniform count penalties. The already frozen qualified-
query comparison tests actual replacements/removals before all branches; no
words or thresholds are now selected from these diagnostic GT outcomes.

Shared eight-arm wall/GPU time is23744.782s and peak CUDA allocation7251.281MiB,
not standalone inference cost. The collection SSH read timed out during the
large merge, but the original merge finished with verified coverage/confusion
sums. Its verified artifact was reused without rerunning inference or merging.
Results: `research/natural_count_calibration_20261004/full/ade150/merged.json`.

### Frozen Input Coverage Check

A metadata-only check confirms the already queued ADE qualified manifest can
be reconstructed exactly from its frozen v2 source with the original builder.
No candidate words changed after the completed ADE deletion/count diagnosis.
The deletion control and the queued candidate have materially different inputs:

| Class | Deletion-control retained queries | Frozen qualified queries |
| --- | --- | --- |
| base | base; stand | pedestal; pedestal base; pedestal stand |
| apparel | apparel | clothing; apparel; wearing apparel; dress; clothes |
| wardrobe | wardrobe; clothes wardrobe; clothing cabinet | wardrobe; clothes wardrobe; clothing cabinet; clothes closet; clothing wardrobe |

This checks that specific positive query coverage is present, not that those
queries improve IoU. Apparel and wardrobe already have these same query sets in
the v2 baseline, so their repair must not be credited as a new direct change in
the qualified-versus-v2 pair. Base qualification does change that paired input:
unqualified base/stand are replaced by pedestal base/pedestal stand and duplicate
queries collapse. The older-source-versus-v2 arms separately test the broader
input repair. These are existing frozen comparisons, not additional GT-selected
vocabularies or new modules. ADE has458 v2 queries and433 qualified queries.
Input: `research/qualified_sense_words_20261004/vocabularies/ade150.json`.

## Comparison Scope

Fixed20 uses OpenAI ImageNet templates, linguistic paraphrases and the original
model-native field of view without an explicit background rejection threshold.
The variable pool combines official query meanings and semantic synonyms, with
actual unequal group sizes and segmentation templates. It is not a pure alias
count ablation: the words, templates and background representation also change.
The local VIP comparator uses official queries/settings/resize with the explicit
empty-row Self-Value numerical repair; these are local implementation results,
not the published numbers themselves.

## Evidence guiding the current experiment

- On VOC20 the previous image selector deleted no alias. The default semantic
  pool scored91.8146, but changing tau/tem to5/.3 reduced it to89.6651. That loss
  is evidence against that calibration choice, not against alias selection.
- On VOC21 the previous calibrated output improved over its own no-threshold
  output,48.8813 to63.6504, but remained9.6087pp below the local finite VIP.
  The canonical cross-view witness NLL used a no-admission proxy, so it did not
  test the exact deployed fine-rival/full-image confidence distribution.
- ADE's full variable-pool Geometry scored26.3876, unscreened coupling28.7972,
  and fine-rival coupling28.7845. The previous calibrated pool scored28.8354,
  just0.3033pp below the local finite VIP29.1387. This narrows the original
  fixed20/ImageNet25.0705 gap, but jointly changes words/templates/aggregation;
  it is not proof that hard alias admission caused that gain.
- Semantic repairs must distinguish true annotation meanings from text-medoid
  centrality. ADE step means a stair step, not a pedestal; base is a pedestal;
  apparel is clothing, not a closet. Wrong-sense words are different from useful
  subtype descriptions such as several animal species within a single animal class.
- More words can change both visual evidence and aggregation/count offsets.
  A larger pool or higher pseudo-label confidence is not proof of higher IoU.

## Frozen new candidate

New independent root: `results/natural_sense_adaptation_v2_20261004`.
Only physical GPUs4-7 are used for new workers. Existing workers are untouched.

| Protocol | Frozen candidate aliases | Class count range | Image-only chosen profile | Source rejection | New global foreground guard |
| --- | ---: | ---: | --- | ---: | ---: |
| VOC20 | 116 | 3-23 | source tau5/tem.3 | 0 | 0 |
| VOC21 | 172 | 3-56 | default tau1/tem1 | .237790 | .132531 |
| ADE150 | 458 | 1-24 | source tau5/tem.3 | 0 | 0 |
| COCO-Stuff171 | 644 | 1-18 | source tau5/tem.3 | 0 | 0 |
| COCO-Object81 | 325 | 3-32 | default tau1/tem1 | .093264 | .046046 |

All five one-image source-reader equivalence checks completed successfully.
The64-image mask-free calibrations for all five protocols are frozen. Witness
probabilities come from actual accumulated full-image outputs, including the
fine-rival potential. Profile selection uses paired image-block classification
error, not confidence NLL. Background thresholds use correctly predicted
image-only foreground witnesses and float confidence, not uint8 confidence.
The pseudo-witness rule can still disagree with GT outcomes; the labeled full
evaluation is required before any performance claim.

The paired arms separate new words at old parameters, new words at default
parameters, and the frozen image-only calibrated output. Unthresholded controls
are retained. Labeled pilot results do not select a winner or change full-run
settings. Negative results will be kept rather than replaced by another arm.

## New Paired Diagnostics, Not Full Results

The following completed16-image diagnostics use common class support across
paired methods, including the same-sample original suite. Sparse/absent classes
make these inappropriate substitutes for full benchmark scores. The settings
were already selected without masks; these scores do not change the full run.

| Protocol | Images | Source | New words, source settings | New words, default settings | Image-only calibrated | Local finite VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VOC20 | 16 | 59.8371 | 59.8468 | 59.8880 | 59.8468 | 61.7996 |
| VOC21 | 16 | 40.5066 | 40.5044 | 35.8177 | 42.5238 | 47.9158 |
| ADE150 | 16 | 13.0995 | 13.1523 | 13.7548 | 13.1523 | 10.9411 |
| COCO-Stuff171 | 16 | 16.1602 | 16.0631 | 16.2905 | 16.0631 | 18.1272 |
| COCO-Object81 | 16 | 14.2865 | 14.2928 | 14.9160 | 15.9865 | 19.8103 |

Source confusions for all five protocols match the original same-sample
evaluation exactly. The small word-only deltas do not support attributing the
large fixed20-to-variable-pool gain to this additional sense-repair pass.
The VOC21 calibrated delta also changes background rejection. ADE's calibrated
profile is not its highest-scoring diagnostic arm; that negative evidence is
preserved, not used to switch the already frozen full-run setting.
COCO-Stuff's word-only/calibrated candidate regresses by0.0971pp on these16
images, while its default-setting arm improves by0.1303pp. This distinguishes
parameter effects from word repair rather than presenting either as a full
benchmark gain. COCO-Object's word-only delta is0.0063pp, while default
parameters and new residual rejection give a combined1.7000pp delta. The
word-only deltas average-0.0061pp across these five sparse diagnostics; the
combined calibrated deltas average0.7365pp. Neither is a full-domain result.
This evidence does not justify attributing the combined gain to new synonyms
or claiming that additional word counts solve the remaining VIP gap. All five
pilots are complete and VOC20/VOC21 full are verified. At the2026-10-04
02:40UTC queue check, ADE150 full shard0 had completed1000/1000 images and the
healthy controller had accepted its output; shard1 remained active on GPU4.
GPU6 had automatically started COCO-Stuff full shard0, whose live worker had
processed5/2500 images with its source-reader first-image equivalence verified.
The remaining COCO-Stuff shard and COCO-Object shards queue on idle physical
GPUs4-7. These are queue milestones, not new whole-dataset benchmark scores.

## Post-Freeze Witness Scope Diagnosis

A completed CPU-only diagnostic compared the saved64-image calibration witness
coordinates with GT, after the vocabulary/profile/thresholds had been frozen.
No selector, candidate or full-run setting was changed using these labels.
The following are coordinate-level diagnostics, not mIoU or alias utility scores.

| Protocol | Trusted points outside scored support (%) | Scored pseudo-label accuracy (%) | Default GT witness error (%) | Frozen GT witness error (%) |
| --- | ---: | ---: | ---: | ---: |
| VOC20 | 58.1855 | 99.3501 | 5.1465 | 5.4755 |
| VOC21 | 4.1029 | 94.4417 | 13.2157 | 14.1686 |
| ADE150 | 8.2873 | 82.1544 | 44.8528 | 45.7630 |
| COCO-Stuff171 | 2.7804 | 68.4095 | 52.6314 | 53.1944 |
| COCO-Object81 | 2.5453 | 91.8548 | 21.4278 | 22.6697 |

VOC20 has22937 trusted points, of which13346 are ignored by this foreground-only
protocol. They account for48.5500% of witness quality weight. Frozen-minus-default
pseudo-error is-6.0965pp over all witnesses, but+0.3515pp on scored witnesses and
-6.7434pp on ignored witnesses. Thus the all-point calibration preference can
reverse the scored-foreground preference. The full VOC20 regression is consistent
with this objective mismatch, rather than a demonstrated vocabulary benefit.

ADE and Stuff also have substantial pseudo-label noise. Cross-view agreement is
not equivalent to a correct semantic label. The justified next calibration
hypothesis is to account for task scope using image/text information, not GT
support masks, and to avoid treating self-agreement as a guaranteed IoU objective.
Genuine sense qualification/replacement is a separate input-level hypothesis.
The original frozen queue remains unchanged; follow-up replay and lexical inputs
are evaluated separately below.
The completed audit snapshot is retained in
`research/natural_sense_adaptation_v2_20261004/witness_scope_audit.json`.

A subsequent label-free CPU replay tested a concrete task-scope hypothesis on
VOC20 without altering its running or completed outputs. VOC20 and VOC21 have
identical64 calibration images and witness coordinates; their foreground category
names also align exactly. Requiring trusted VOC21 foreground of the same semantic
class leaves12810 of22937 original trusted points. Another4845 original trusted
points meet trusted VOC21 background evidence and are excluded from the foreground
objective. This replay reads only the saved image-derived labels/quality/probabilities,
never GT masks.

The original profile rule gives frozen-minus-default pseudo-error=-0.060965 with
upper95=-0.026725 and selects the frozen profile. Scoped witnesses give=-0.004996
with upper95=+0.012656, so the same conservative rule retains default. The exact
matching existing full VOC20 output is therefore91.8121 instead of89.6655,
a2.1466pp difference; no additional full inference was required. This supports
studying open-world background challenges for foreground-only calibration, not
claiming a new independent benchmark result. The scope rule was motivated by the
prior labeled diagnostic. Snapshot:
`research/natural_sense_adaptation_v2_20261004/task_scope_replay.json`.

## Frozen Qualified-Query Comparison

A separate lexical-input candidate now replaces ambiguous strings throughout the
scored pool, not merely in the protected first alias. It uses no images, masks,
pixel frequencies or IoU-based ranking. Valid synonyms/subtypes remain and no
alias quota is imposed. Both old/new arms keep segmentation templates, tau=tem=1,
the same frozen source rejection and the unchanged deployed reader equations.

| Protocol | Previous v2 aliases | Qualified aliases | Changed class query sets |
| --- | ---: | ---: | ---: |
| VOC20 | 116 | 107 | 5 |
| VOC21 | 172 | 163 | 6 |
| ADE150 | 458 | 433 | 20 |
| COCO-Stuff171 | 644 | 607 | 42 |
| COCO-Object81 | 325 | 319 | 8 |
| Context59 | 263 | 257 | 9 |
| Context60 | 266 | 260 | 9 |

Examples include computer mouse without the unqualified mouse, drinking glass
without glass for ADE's vessel category, and no skylight/fanlight in ADE's light
source category. Person apparel phrases now explicitly name the person wearing
the garment. Natural-language representatives replace extra annotation-code
queries such as wall-brick. These changes are hypotheses, not proven harmful-alias
identifications; counts and words change together.

Controller `gqs04_controller` now waits for each protocol's own verified full
baseline, rather than requiring the whole Context suite to finish first. Only
the unstarted waiting controller was transitioned; original vocabularies,
protocol, evaluator and existing GPU workers remain unchanged. VOC20/VOC21
baselines are ready. Existing upstream pending launches retain priority, and
word comparisons can use spare capacity while other full baselines continue.
It will run mask-free one-image paired/standalone-reader equivalence
checks, then full fixed-setting pairs on idle physical GPUs4-7 only. The old v2
arm must exactly reproduce its prior per-image confusion matrices, not just its
mean IoU. No labeled smoke result chooses parameters or a domain-specific winner.
Ten local tests and four remote fixed-setting contract tests passed. Four new
scheduling tests also pass both locally and on A800, covering per-protocol
readiness, required confusion artifacts, unstarted-controller transition and
upstream dispatch priority. New word-comparison image inference has not started
yet; all allowed GPUs remain occupied. No existing model, vocabulary or GPU
worker was replaced.
Inputs and results:
`research/QUALIFIED_SENSE_WORDS_20261004.md` and
`research/QUALIFIED_SENSE_WORD_RESULTS_20261004.md`.

## What The Current Word Repair Actually Changes

A text-only audit of the frozen v2 manifests finds that all62 changed
representatives retain their original query as another alias. This checks the
inputs, not ground-truth alias utility. Some retained terms are valid synonyms
or category spellings; it would be incorrect to call all62 harmful.

| Protocol | Changed representatives | Old representative still in the scored pool |
| --- | ---: | ---: |
| VOC20 | 3 | 3 |
| VOC21 | 3 | 3 |
| ADE150 | 19 | 19 |
| COCO-Stuff171 | 33 | 33 |
| COCO-Object81 | 4 | 4 |

The underlying query-set changes, ignoring order, are distinct from representative
changes. VOC20 and VOC21 add/remove no strings in any class; they are reanchoring
controls. ADE adds85/removes9 strings across51 classes, Stuff adds9/removes3 across8
classes, and Object adds1/removes0 in1 class. Thus the tiny VOC word-only difference
cannot be presented as a successful removal or replacement of bad synonyms.

Concrete examples are computer mouse plus mouse, orange fruit plus orange,
drinking glass plus glass, and projection screen plus screen. ADE's light
source still includes skylight/fanlight, although objectInfo150.txt category83
defines light/light source rather than a window. The metadata identifies a
word-sense question, not a measured IoU loss. No images, masks or taxonomy
class-frequency fields were used for this audit.

The implementation explains why reanchoring need not fix the whole score:
Geometry uses temperature-scaled log-mean-exp over all parent aliases
(`geometry_readout_trace.alias_class_scores`); wide scoring uses salience-
profiled log-sum-exp (`natural_text_adaptation.class_logits`). Merely permuting
an unchanged query set does not change either aggregate. The first query has a
separate protected role in the rival admission and pseudo-witness rules.
Rival admission modifies the coupled potential, not the original Geometry
alias pool (`natural_variable_alias_reader.retained_variable_scores`). Thus a
new protected representative does not by itself prevent an old ambiguous term
from affecting another semantic path.

The next word-focused hypothesis should therefore distinguish reanchoring from
actual sense qualification/replacement before every branch, with valid synonyms
and subtypes retained. This is not an argument for deleting every short noun,
cosine-ranking every alias, or imposing equal quotas. Existing full-run words
and rules remain unchanged; no new candidate is promoted from this text audit.

## Background Calibration Feasibility

Two CPU-only probes now make the background hypothesis more specific, without
loading GT, selecting a new calibration, or changing any active evaluation.
They use existing64-image records and are reproducible with
`tools/background_replay_feasibility.py --collect`.

First, the actual accumulated full-image probabilities were used to find the
largest per-class confidence threshold retaining at least95% of the
baseline-surviving pseudo true positives. Each contributing image receives equal
total weight in the positive and pseudo-background-error pools. At this in-sample
frontier, VOC21 dog loses4.6850% of trusted positive weight but removes only1.1745%
of pseudo background false positives. Train loses4.8296% and removes17.8538%; car
cannot increase its guard without exceeding the retention budget and removes0%.
Most classes lack repeated independent-image support: only2/20 VOC21 classes and
1/80 COCO-Object classes have at least8 positive witness images. This is not a
held-out success metric and does not justify fitting thresholds for sparse classes.

Second, an exact central-tile counterfactual tested full residual-background local
count compensation. For the unchanged reader S=G+H(W-G+V), admission V depends on
the wide/fine observations, not G. Replacing only the background local
log-mean-exp normalizer by log-sum-exp therefore adds
`(1 - H @ ones) * log(n_background)` to its final score. This fixes an explicit
normalization term, rather than searching a labeled class intercept. Other words,
features, temperatures and reader equations remain fixed in the replay.

| Protocol | Rejection | Original class-balanced pseudo error (%) | Full union compensation (%) | Paired delta (pp) |
| --- | --- | ---: | ---: | ---: |
| VOC21 | off | 4.0949 | 47.6961 | +43.6012 |
| VOC21 | original source | 20.7375 | 48.0649 | +27.3274 |
| COCO-Object81 | off | 9.4310 | 51.7180 | +42.2870 |
| COCO-Object81 | original source | 34.2086 | 52.1796 | +17.9710 |

The background-only pseudo error decreases substantially, but foreground damage
dominates the combined objective. Full compensation is not promoted to a new GPU
run. These values use the original semantic pool's centered-tile logits, not the
v2 full-image probability records used in the threshold probe; neither table is
full mIoU. Weaker or spatially conditional compensation is not ruled out by this
single full-compensation counterfactual.

The evidence supports separating actual background-versus-foreground lexical
discriminativeness from a uniform score boost or a foreground-only confidence
guard. The already frozen category-qualified word comparison remains the next
full test, with no extra module appended or label-selected setting substituted.
Four focused tests cover strict threshold ties, image balancing, missing positive
support and equality of the algebraic shift to the original linear reader.
Detailed per-class results and inputs are retained in
`research/NATURAL_BACKGROUND_REPLAY_FEASIBILITY_20261004.md` and
`research/natural_background_replay_feasibility_20261004/`.

## Reproducible Scope Replay And Its Negative Check

`tools/task_scope_witness_replay.py --pair voc --collect` now exactly reproduces
the earlier label-free task-scope snapshot from the frozen paired64-image
witnesses, without modifying any model, vocabulary or deployed selection.
Eight image-block cross-validation folds all retain default under the scoped
rule and all select frozen under the original rule. However, when both choices
are scored on the same held-out scoped pseudo-witnesses, original/frozen error
is1.8372% and scoped/default error is2.3368%, a0.4996pp regression. The already
verified full GT mIoU comparison instead favors default91.8121 over frozen89.6655.
The pseudo objective therefore remains misaligned with the downstream metric;
scope correction and stable choices alone do not validate it as an alias-utility
or IoU surrogate. No new calibration is deployed from this replay. Five focused
tests pass. Exact inputs/errors/folds and interpretation are retained in
`research/NATURAL_TASK_SCOPE_REPLAY_20261004.md` and
`research/natural_task_scope_replay_20261004/voc_results.json`.

## Remaining coverage

Context59/60 now have5105 verified validation pairs, prepared from official
Stanford full-category MAT ground truth. All10103 MAT IDs agree with the official
VOC2010 train/validation union. Fixed MMSeg category IDs/order were checked
against the archive label dictionary; raw ID255 is money and is not an ignore ID.
The existing five-protocol queue was not changed after launch. A separate
Context controller is now live and waiting for that suite to finish, then uses
only idle physical GPUs4-7. No Context image inference has started yet.

Separate metadata-only Context inputs are frozen at
`research/context_sense_vocabulary_20261004/vocabularies/`: Context59 has263
aliases and Context60 has266; each class has1-18 actual aliases. The existing
natural sense policy and standard category order are transferred without new
image-selected pruning rules or parameter fitting. These files are not results.
The Context source comparator is an explicit text-only ImageNet preset, not a
fabricated image-selected result. Its fixed source threshold is0 for Context59
and.1 for Context60; the new mask-free64-image calibration then transfers the
same actual-reader rule from the five-protocol suite. Official VIP rejection on
scored foreground pixels counts as a false negative rather than disappearing
from evaluation. Local and remote scoring/source-input tests passed.

Cityscapes still needs the registered official image and fine-label archives.
Seven protocols over four underlying datasets are now prepared. VOC20/21,
Context59/60 and Stuff/Object each share images, so they are not independent
datasets. No eight-protocol mean is claimed while Context/City results are absent.

Vocabulary construction and calibration were motivated by developed errors;
selection is transductive and exploratory. Current results do not establish
SOTA or CVPR readiness. Paper timing100ms/memory1650MB also cannot be compared
directly with shared multi-arm evaluation wall time or peak memory.

## Artifacts

- `research/rival_fine_natural20_full_20261003_qchunk128/DATASET/merged.json`
- `research/natural_text_adaptation_20261003/DATASET/full_merged.json`
- `research/natural_sense_adaptation_v2_20261004/DATASET/`
- `research/natural_sense_adaptation_v2_20261004/task_scope_replay.json`
- `research/NATURAL_TASK_SCOPE_REPLAY_20261004.md`
- `research/natural_task_scope_replay_20261004/voc_results.json`
- `research/qualified_sense_words_20261004/`
- `research/QUALIFIED_SENSE_WORDS_20261004.md`
- `research/QUALIFIED_SENSE_WORD_RESULTS_20261004.md`
- `research/NATURAL_BACKGROUND_REPLAY_FEASIBILITY_20261004.md`
- `research/natural_background_replay_feasibility_20261004/`
- `research/NATURAL_SENSE_ADAPTATION_V2_20261004.md`
- `research/CONTEXT_SENSE_ADAPTATION_20261004.md`
- `research/context_sense_adaptation_20261004/`
- `research/vip_natural_datasets_20261003/context_ground_truth_preparation_20261004.json`
- `datasets/VIP_NATURAL_DATASETS_20261003.md`
