# Alias screening: exact speedup and competition-action follow-up

## Decision

Keep the established full-suite RivalFineHard model unchanged, with its opt-in cached
equivalent reader. Retain all soft and competition-action implementations/results.
The best new development-panel mean belongs to FineRivalProjected_Exact; the primary
FineRivalPosterior_Exact is slightly better on full UDD5, including road and vehicle.
Neither is automatically promoted or switched by dataset. No new full20092 suite
or monitoring automation was launched.

Latest update: the frozen20/30/40/wrong-parent/paraphrase stress study is complete.
The projected candidate improves every scenario mean versus matched hard but FAILS
the frozen robust-advancement gate. Its existing20-word transfer evidence remains;
it is not a promoted arbitrary-expansion selector. See the final section below.

Further update: fixed class-field attribution and conditional fine-support soft
weighting are complete. The new weights help30/40 slightly but fail clean20,
stability and identity gates. Preserve them without model replacement. Details
are in the two final sections below; the research objective remains open.

## What Was Fixed

Original Geometry, finite VIP wide observer,20 aliases/class, four physical256-to512
fine views per local tile, original fine contradiction risk/canonical protection,
count-normalized hard writer and Geometry reconstruction. Only pair-action
reconciliation/projection changed. No temperature, threshold, template or domain
routing was fitted. The new methods do not add visual forwards, but still require
the original fine observations.

## Exact Acceleration

The preceding cached implementation reduced mean independently measured
window-context latency by22.4% versus the original reader. Original/cached scores,
risks and diagnostics are bitwise equal on64 developed windows; a complete
3000x4000 VDD image has zero prediction mismatches. This follow-up also verifies
all three historical complete-image controls per image on96 samples.

Acceleration is an execution result, not a new semantic contribution. Reusing only
existing local features was cheaper but recovered little screening benefit in the
preceding pilot; it does not establish equivalence to the fine witness.

## New Rules

Let D be the unchanged directed hard-screening action, E=D-D^T, and Z0 the unchanged
unscreened coupled scores. Every output remains Z0+H*u with the same reconstruction H.

- FineRivalPosterior_Exact: p=softmax(Z0); u_c=sum_d p_d*E_cd, then remove class mean.
  This is weighted pair least squares with edge weights p_c*p_d. The weights are
  predictions, not calibrated correctness probabilities.
- FineActionProjected_Exact: project each E_cd onto the segment0..T_cd, where
  T_cd=(F_c-B_c)-(F_d-B_d) comes from the same fine/wide class profiles. Opposing
  requests are refused; aligned source requests are bounded. Use the original solver.
- FineRivalProjected_Exact: project first, then use the posterior-weighted solver.

Projection is a source-action bound before H, not a guarantee on final dense margins.
All three are still conditionally hard alias screening. Rival weights are soft;
they must not be described as new continuous per-alias1-risk weights.

## Verified Results

The64-window pilot uses eight developed top-left512 windows/domain. The complete
panel uses full UDD5(40) plus the same developed eight complete images in each of
seven other domains:96 unique images. Neither is full eight-dataset evaluation or
untouched validation. LoveDA D counts once; P separately. Common scored-class
denominators are used within each domain/protocol.

| Method | Developed64 window mean | Complete96 panel mean |
| --- | ---: | ---: |
| No admission | 42.8679 | 46.0588 |
| Original/cached RivalFineHard | 45.0622 | 47.6844 |
| FineRivalPosterior_Exact | 45.3333 | 48.0055 |
| FineActionProjected_Exact | 45.1606 | 47.8548 |
| FineRivalProjected_Exact | 45.4187 | 48.1196 |
| Strength-matched original direction control | 45.1621 | 47.7598 |
| Fine-direction/alias-budget diagnostic control | 45.3240 | 48.0235 |

On the complete panel, original screening adds1.6256pp over no admission; the best
candidate adds another0.4352pp over hard, and improves seven/eight domains. Its VDD
delta is-0.2637pp. The primary adds0.3211pp, also wins seven/eight, with VDD-0.1583pp.
All three meet the predeclared mean-positive/worst-loss<=1pp development gate.
This does not establish independent statistical significance or full-suite superiority.

The strength-matched control recovers0.0754pp, so the best candidate's mean gain is
not explained by its amplitude alone in this panel. However, the fine-direction
control reaches48.0235, only0.0961pp below the best candidate and slightly above
the primary. It retains alias-derived action magnitudes, so it is not alias-free;
nevertheless, the new alias-requested-direction advantage is small.

## UDD5 And Remaining Errors

| Full UDD5 metric | Original hard | Posterior | Posterior+projection |
| --- | ---: | ---: | ---: |
| mIoU | 52.9563 | 53.2342 | 53.1947 |
| Residual-excluded mIoU | 56.9706 | 57.3236 | 57.2321 |
| Road IoU | 43.6524 | 43.7995 | 43.6159 |
| Vehicle IoU | 26.0759 | 26.7132 | 25.8748 |

Projection increases road precision but lowers its recall, and worsens vehicle IoU
despite a higher overall mean. Hence constraints and stronger competition are not
universally beneficial. Potsdam's primary raises low-vegetation IoU23.9760->24.5344
and car24.7736->25.0038, but car precision remains25.1113% and vegetation recall
25.9108%. Screening has not repaired the underlying semantic bias. On OEM and
FLAIR-1, the coupling remains below Geometry alone even after these improvements.

## Independent Cost

Same-process warmed/alternated three-repeat window-context timing includes the
two resident backbones and original fine views, excluding load/text encoding/GT.

| Method | Mean seconds | Mean paired latency ratio vs cached hard |
| --- | ---: | ---: |
| Cached hard | 0.290633 | 1.0000 |
| Posterior | 0.291710 | 1.0037 |
| Projection | 0.292707 | 1.0071 |
| Posterior+projection | 0.294247 | 1.0124 |

These small overheads are not extra visual observations. Complete-panel wall/memory
measurements share nine dense endpoints and are not isolated model latency.

## Paper Interpretation

The defensible target is **where and against which rival an alias may contribute**,
not permanent good-word/bad-word classification. The existing rule already accepts
different subsets by position/class/rival; mean retained counts vary roughly17.7-18.9,
with no quota. A word rejected against one rival may still contribute against another.
This does not prove every alias is useful, nor that VIP global distillation is
universally ineffective. Repeated fine/wide agreement can still be jointly wrong.

The next discriminating evidence is a frozen broader/full-suite test and robustness
to30/40-word pools or wrong-parent aliases, plus alias-free/identity-shuffled controls.
Do not claim a unique screening contribution merely because one reweighted candidate
beats no admission. All developed panels and the earlier labeled diagnostics must
remain disclosed; the established model is not replaced by this pilot winner.

## Reproduction

- `RIVAL_COMPETITION_ADMISSION_20261005.md`: nine-arm64-window results, exact source
  replay, class outcomes, pixel transitions and independent latency/memory.
- `RIVAL_COMPETITION_COMPLETE_20261005.md`:96 complete images, exact historical
  per-image controls, common-class results, residual metrics, diagnostics and costs.
- Raw data: `research/rival_competition_admission_20261005/` and
  `research/rival_competition_complete_20261005/`.
- New reader: `DINOtool/dinotool/rival_competition_admission.py`; evaluator:
  `DINOtool/scripts/eval_rival_competition_admission.py`.
- Managers: `tools/rival_competition_admission_experiment.py` and
  `tools/rival_competition_complete_experiment.py`; reuse checked SSH and refuse
  occupied GPUs or existing outputs.18 local/remote reader and manifest tests pass.

Checkpoints, vocabulary/class identities, frozen weights,96 unique keys, scored
target counts, per-image confusion sums and transition endpoints are verified.
Scores persist before masks in the64 pilot; the complete-image evaluator loads
masks only after predictions. All original/best and unsuccessful alternatives remain.

## Subsequent Attribution And Mass Admission

The completed attribution study uses the same64 developed windows. The frozen
FineRivalProjected candidate scores45.4187. Removing alias identity with matched
per-query/class/rival rejected counts gives43.3962 across three seeds; removing
location gives44.1413; removing rival identity through degree-preserving binary
switches gives44.7492. This supports conditional word/position/rival information,
not the claim that every word is useful. Direct fine-class reconstruction reaches
46.1541 on average but loses severely on VDD and LoveDA P.

The subsequent alias-response-mass experiment is fully completed and verified.
DirectedMassFine scores44.6300, below45.4187 projected and45.0622 cached hard.
It exceeds count-only44.2182, strength-matched count44.5515 and alias-shuffled
mean44.1875, but fails the frozen accuracy/worst-protocol advancement gate.
PairMassFine reaches45.7109 and improves seven/eight primary domain windows,
yet VDD drops54.1183->49.2494 and LoveDA P52.6436->45.0101; do not promote it
by mean alone. Alignment with the old hard direction scores44.5277 and does not
solve this problem. These are not full-dataset estimates.

Actual rejected logsumexp mass is a response fraction, not permission to replace
an entire class's wide evidence with fine evidence. VDD pair-mass wall precision
drops63.37%->30.49% and its predicted area11.08%->23.13% versus hard; vehicle
precision drops42.18%->31.51% although recall remains100%. In LoveDA P, building
IoU53.97%->1.33% accompanies predicted area0.0388%->1.9901%. Thus the action
introduces cross-class false positives, rather than simply deleting useful words.
The tiny building support in these windows also limits full-domain interpretation.

Keep the cached exact hard model as the full-suite established option and
FineRivalProjected as the stronger development/complete-panel candidate.
Do not replace an alias's counterfactual competitive action by unrestricted class
fine-minus-wide innovation. A future soft rule should preserve those conditional
actions and establish benefit beyond equal-strength and matched-word controls;
feature-only reuse remains a distinct, previously weaker speed/accuracy option.
No further full20092 rollout or automatic model replacement was launched.

Details: RIVAL_ALIAS_ATTRIBUTION_20261005.md and RIVAL_MASS_ADMISSION_20261005.md.
All source files/raw scores/controls are preserved. Sixteen mass-reader tests pass;
all five frozen endpoint scores and per-image predictions replay exactly across64
keys. Every candidate's standalone checkpoint-based score equals its all-arm score
bitwise in four checks/domain. Mass-reader timing shows no meaningful extra gain
over caching, and does not remove fine observations.

## Subsequent Margin-Unit Soft Attenuation

The completed14-arm margin-unit study retains the same64 developed windows,
five exact earlier scores/per-image controls, risk support, all20 pool, observers
and reconstruction. Only risk-positive per-alias/per-rival weights change.
Primary log(w)=beta*(f-b) transfers the sampled wide alias margin b to fine f
before normalization; neutral log(w)=-beta*b and deficit log(w)=beta*f are
declared alternatives. All use the current fine-target projection/posterior solver.
This is not the failed whole-class mass-admission writer.

| Method | Developed64 mean |
| --- | ---: |
| No admission |42.8679|
| Cached hard |45.0622|
| Projected/posterior hard |45.4187|
| Margin transfer soft |44.7412|
| Margin neutral soft |44.0781|
| Fine-deficit soft |43.8766|
| Original1-risk with matched projection/solver |43.8153|
| Class-mean transfer control |42.9348|
| Equal-strength projected hard direction |44.7941|
| Three-seed transfer alias-null mean |43.1882|

Transfer improves VDD54.1183->54.4115 but loses on seven other main domains;
its worst protocol loss is1.6535pp (LoveDA D). All three candidates fail their
predeclared accuracy gates, so no complete-image/full-suite promotion follows.
The source-margin transfer identity holds<=1e-10, canonical risks remain0,
null log-weight spectra match exactly, and real standalone/all-arm scores are
bitwise equal in four checks per candidate/domain. Sixteen reader tests pass.

Between30.6% and53.0% of eligible comparisons retain a positive sampled wide
alias margin under original1-risk weights. Thus the old mapping often leaves
the source advantage in place. However, forcing that advantage to0 or f does
not improve the strongest hard endpoint. The transfer identity describes only
the source discount: weighted class normalization, antisymmetric reconciliation,
Geometry H and dense interpolation still affect actual decisions.

Transfer's1.5530pp advantage over the matched alias-null supports word identity;
its0.0529pp deficit against an equal-strength hard direction provides no new
directional advantage over hard. It improves ratio-soft by0.9260pp, not hard.
Do not explain all remaining losses as too-small attenuation or conclude that
continuous weights are universally impossible. Candidate alternative timings
also include computing the primary transfer correction internally; these are
current implementation costs, not lower bounds for optimized deployment.

Retain cached hard and the projected/posterior candidate as the stronger options.
The next discriminating writer question is whether renormalizing surviving alias
weight amplifies unrelated words and introduces unwanted class competition.
Testing a fixed-mass/rival-neutral replacement against ordinary count-normalized
hard would isolate this effect; this writer experiment has not yet run.
No new visual forward, dataset-specific tuning or full20092 rollout occurred.

Details/raw outcomes: RIVAL_MARGIN_ATTENUATION_20261005.md and
research/rival_margin_attenuation_20261005/. All earlier schemes remain preserved.

## Subsequent Rival-Neutral Replacement

The completed13-arm source-writer study changes neither risk support nor any
visual observation. RivalNeutral caps an eligible crop alias at the ORIGINAL
rival class log-mean-exp while retaining K slots. FineCap adds the sampled fine
alias/rival margin to that cap. Both preserve the current projection/posterior/H.
HardFixedMass is a diagnostic removing the same aliases with no survivor-count
normalizer; it is not a new alias selector and is not promoted after results.

| Writer | Developed64 mean |
| --- | ---: |
| Projected/posterior hard |45.4187|
| Rival-neutral candidate |45.1221|
| Fine-cap candidate |45.6523|
| Hard fixed-mass diagnostic |46.2516|
| Neutral without projection |45.1179|
| Equal-strength hard direction |45.3389|
| Three-seed neutral alias-null mean |44.1599|

Both declared candidates fail their frozen gates. Neutral loses1.0577pp on UDD5
and trails hard/strongest controls. FineCap has a higher mean but loses3.3300pp
on VDD and6.4201pp on LoveDA P, so it is not a stable replacement. The matched
neutral word-null gap is0.9623pp: identity matters, but novelty/accuracy leadership
does not follow from that alone.

Normalizing by remaining alias count is a real causal factor in this fixed
comparison: D_count=D_fixed+coverage*log(K/remaining)/beta holds<=1e-10.
Approximately29.5%-48.2% of active old directed source actions increase their
own class after normalization. This is not automatically a harmful boost.
Removing the term raises seven/eight primary domains, with Potsdam+2.8071,
UDD5+1.8003 and Vaihingen+2.3290pp, but VDD-4.6353 and LoveDA P-6.9364pp.
It therefore cannot be uniformly disabled or credited as a new selector.

In VDD fixed-mass wall precision57.04%->30.54% and predicted area12.36%->23.19%
accompany IoU53.14%->29.43%. Vehicle precision/IoU38.26%->31.83% while recall
stays100%. Potsdam low-vegetation IoU21.73%->34.50% and recall22.83%->37.86%
improve. LoveDA P building area0.0515%->2.3235% and IoU48.93%->1.14% illustrate
competition instability on tiny support. Thus removing only nonpositive source
class evidence does not guarantee harmless final updates after pair/H coupling.

Keep the cached established hard model and previous projected/posterior candidate.
No full-suite promotion follows from this study. A useful next normalization test
is retained observer alias-prior mass, not only cardinality: compare the original
uniform K/remaining correction with original wide salience and already-computed
fine salience surviving mass. This would preserve conditional hard decisions
and actual per-alias counterfactual evidence, unlike whole-class fine replacement.
Do not fit a domain-specific normalization coefficient to these masks. This
salience-mass experiment has not yet run.

Sixteen reader tests pass. Five frozen endpoint scores/per-image predictions,
64 unique keys, fixed identities/weights, target/confusion/transition sums,
nonboosting source deltas, normalizer decomposition, matched rejected counts
and four real singleton/all-arm checks per benchmark/domain are verified.
Current fixed-mass timing includes its normalizer-decomposition diagnostics
and the old hard writer; it is not optimized standalone fixed-mass latency.
Details: RIVAL_NEUTRAL_REPLACEMENT_20261005.md and its retained raw directory.

## Subsequent Salience-Mass Normalization

The completed18-arm study retains the same64 developed windows, hard decisions,
all observers and per-alias counterfactual writer. Only count compensation changes
to negative log surviving original fine/wide salience-prior mass. Both candidates
fail the predeclared accuracy and mechanism gates; no model is promoted.

Fine salience mass scores45.4002 and wide salience mass45.3861, below the retained
projected hard45.4187 by0.0185 and0.0326pp. Each wins only VDD among eight main
domains and is below its own class-mean and three-seed prior-identity-null mean.
Fine's gain over equal-strength hard is only0.000061pp; wide also loses that control.
Original normalized salience entropy/log K is0.999873-0.999952, almost uniform:
this prior supplies negligible useful word ranking. Do not sharpen its temperature
post hoc merely to rescue the hypothesis.

Uniform/count identity, five frozen score/per-image endpoints,64 unique keys,
checkpoint/vocabulary/Geometry identities, target/confusion/transition sums and
four real singleton/all-arm score checks per candidate/domain are verified.
Sixteen tests pass. Candidates keep the original fine observations and do not
provide a measured speed gain. All schemes remain available.

Details: RIVAL_SALIENCE_MASS_20261005.md and its raw result directory.
The next scoped experiment is execution-only acceleration of the original fine
observer, first removing unused per-layer empty-row CPU counts, then testing a
batch-one CUDA graph. This preserves arithmetic and adds no visual information.
Actual feature/risk/score/prediction equality must be established before calling it
equivalent. No new soft rule, full20092 rollout or model promotion follows here.

## Completed Fine-Observer Execution Acceleration

The execution-only study above is complete. All64 developed windows across eight
domains pass exact feature/risk/score/diagnostic/prediction checks. Each backend
checks256 actual fine crops, with different inputs. The complete3000x4000 VDD image
also has zero prediction mismatch for Geometry, no admission and original hard.
All three historical per-image endpoint confusions replay exactly. Eleven remote
execution/finite-proxy/fine-reference tests pass; original defaults are unchanged.

| Execution of unchanged cached hard | Mean window-context seconds | Reduction |
| --- | ---: | ---: |
| Original fine eager |0.306597|0|
| Omit unused per-layer CPU statistics |0.305821|0.25%|
| Original batch-one CUDA graph |0.279341|8.89%|

Seven warmed, synchronized, alternating repetitions exclude equality audits,
loading/text encoding/decoding/GT. Scope is one512 local window with full-image
wide context, not full-image throughput. Graph gains6.78%-10.22% total across
domains and25.19%-29.03% in the fine stage. Graph setup costs0.206-0.261s once
per observer; measured peak allocated memory rises26.09-35.42MiB. All four fine
visual observations and per-crop finite-value validation remain. Do not compound
these timings with the earlier22.4% reader-cache gain measured in another session.

Opt-in full evaluator: DINOtool/scripts/eval_rival_fine_graph.py, with the retained
full evaluator's arguments and output-existence protection. It combines the exact
cached reader and fine graph; model signature/rules/weights remain original, while
the result records the execution backend. It does not change the established
evaluator or automatically deploy a semantic candidate/full20092 evaluation.

The current decision remains to preserve conditional screening and its fine
witness. Existing-feature-only weighting is faster but substantially weaker in
the developed comparison; none of the subsequent soft writers establishes a
stable improvement over the strongest projected hard candidate. This study
supports a lower-cost unchanged selector, not a new soft-selector contribution.
FineRivalProjected remains a stronger development/complete-panel candidate,
not an independently validated replacement. Do not route candidates by domain.

Details/raw data: FINE_OBSERVER_EXECUTION_20261005.md and
research/fine_observer_execution_20261005/. All experiments finished; GPUs0-7
were idle after collection. All model schemes and outputs remain preserved.

## Frozen Key-Disjoint Complete-Image Transfer

The next panel is complete:112 new complete inputs,16 per each of seven domains,
key-disjoint from the current developed64/complete96 panel. Full UDD5's40 already
examined results are reused unchanged. Random manifests were frozen from global
keys without scores or labels. This is exploratory transfer evidence, not
scene-disjoint or untouched independent validation. No rule was changed.

| Protocol | Original hard | FineRivalProjected | Gain pp |
| --- | ---: | ---: | ---: |
| VDD new16 |55.2145|56.1993|+0.9848|
| Potsdam new16 |41.8861|42.1555|+0.2693|
| OEM new16 |38.7700|39.6540|+0.8840|
| LoveDA P new16 |65.4682|66.6116|+1.1433|
| LoveDA D new16 |43.1298|44.2120|+1.0822|
| Vaihingen new16 |50.4574|50.6879|+0.2305|
| LandCover.ai new16 |67.6718|67.8106|+0.1388|
| FLAIR-1 new16 |40.6560|40.7028|+0.0468|
| UDD5 reused full40 |52.9563|53.1947|+0.2384|

Seven NEW domain mean (LoveDA D once):48.2551->48.7746,+0.5195pp,7/7 wins.
Candidate gains1.6477pp over no admission47.1268. The empirical paired image
bootstrap95% mean-gain interval is[+0.3734,+0.7332]pp (2000 replicates, fixed
scored-class support); both frozen transfer gates pass. These intervals do not
correct prior method selection or scene correlations. Eight-domain descriptive
mean including reused UDD5 is48.8427->49.3271,+0.4844pp, not full eight datasets.

Same-process independent window-context means: eager cached hard0.296301s,
graph hard0.268146s, graph projected0.270855s. Candidate is8.59% lower-latency
than eager cached hard, but1.01% slower than equally graph-accelerated hard.
The added projection/posterior step has no new visual forwards. Seven warmed
alternating repetitions exclude model/text/decoding/GT. Do not call these
whole-image throughput or claim it is free relative to the matched backend.

All three historical full per-image controls replay exactly on the152 descriptive
inputs, with identical scored targets, fixed checkpoint/vocabulary/model states,
verified112 new unique keys and masks only after predictions. Five manifest/
bootstrap/metric tests and18 existing competition-reader tests pass. A report-only
precision/recall field omission from the merger was resolved using exact confusion
matrices; model outputs were not changed or rerun.

Remaining errors matter: VDD vehicle IoU23.5596->26.7712, but roof86.4160->85.4038.
Potsdam car IoU24.0341->24.4134 and precision24.1290%->24.5094% still indicate
strong false activation despite98.42% recall; low-vegetation recall only27.76%.
FLAIR-1 pervious surface loses1.7952pp despite the tiny overall gain. OEM coupling
remains below Geometry44.9899 versus candidate39.6540. This transfer result
does not establish that screening fixes the entire coupled readout's deficits.

FineBudgetOnly reaches48.6931 on the seven new domains, only0.0815pp below the
candidate. It retains alias-derived correction amplitudes and is not alias-free,
but the alias-requested-direction increment remains small. Prior word/location/
rival-null evidence still supports conditional alias information; do not assign
the entire1.6477pp gain to the new projection or treat this as a new continuous
per-alias soft writer. Source rejection is still hard, with soft rival coordination.

FineRivalProjected+equivalent execution is now the preferred frozen candidate for
a later full suite, not an automatic replacement of the established full20092
model. Opt-in complete/sharded evaluator: DINOtool/scripts/eval_rival_projected_graph.py.
It accepts the retained full evaluator's arguments and refuses existing output.
All schemes remain preserved; no dataset routing, parameter tuning, new heartbeat
or full20092 rollout occurred. Report/raw: RIVAL_COMPETITION_TRANSFER_20261005.md
and research/rival_competition_transfer_20261005/.

## Completed Frozen Vocabulary Stress

All64 developed512 windows (eight/domain), five fixed vocabularies and four frozen
readers are complete. Each scenario shares the same Geometry20 local scores and
operator, wide visual observations and original four fine observations through
the equivalent graph. Masks enter after scores/flags persist. Every historical
scenario no-admission/hard per-image confusion and clean20 projected score matches
exactly. Class supports are fixed across scenarios/arms; LoveDA D counts once.
These are neither full datasets nor independent validation.

| Pool/stress | No admission | Original hard | Projected candidate |
| --- | ---: | ---: | ---: |
|20|42.8679|45.0622|45.4187|
|30|43.0357|45.1366|45.3651|
|40|41.7421|43.9695|44.2044|
|Wrong-parent replacements|41.9892|44.2848|44.5969|
|Legitimate paraphrases|43.6469|45.6935|45.9874|

The candidate wins every scenario mean, and beats no admission in all eight main
domains for20/30/40/wrong-parent and seven for paraphrases. Nevertheless it fails
the frozen stability gate: LoveDA P30 loses2.7580pp versus matched hard, P40 loses
1.3796pp. Wrong-parent damage is0.8218pp versus hard0.7774pp, slightly worse rather
than no worse. Candidate40 remains1.2143pp below candidate20. Do not promote a
new rule/count choice or claim arbitrary expansion is harmless from this test.

The sign-source blind spot is measured, not solved: added aliases retain90.09%-94.40%
of eligible position/alias/rival comparisons, and71.33%-85.15% of broad-positive
comparisons are uncontradicted by fine. Agreement across observers can be jointly
wrong. Audit targets establish wrong-parent source advantages only AFTER prediction;
they do not establish the contribution of each phrase to final pixel errors.

OEM building20->40 recall falls70.73%->7.72%, area11.96%->0.98%, IoU54.55->7.68,
while precision rises70.45%->93.77%. Thus expansion can reduce correct coverage
through class competition, not only create same-class false positives. VDD wall
loses15.92pp while vehicle gains17.40pp; UDD5/VDD improve their overall40-word
means. Phrase identity and count are confounded, so there is no demonstrated
universal word-count optimum or global helpful/harmful-word partition.

Keep the original full-suite model and the frozen20 transfer candidate with this
narrower claim. Stronger suppression, different temperatures or a renamed failed
ownership source are not justified by these targets. A future source must distinguish
conditional contributions beyond the joint-agreement blind spot and pass prediction
and matched-identity controls. This study does not settle that successful formula,
nor prove all soft rules impossible or every word useful. Equivalent acceleration
remains available without changing the original source. All schemes/outputs remain.

Report/raw: RIVAL_PROJECTED_VOCABULARY_STRESS_20261005.md and
research/rival_projected_vocabulary_stress_20261005/. The local report includes exact
per-class IoU/precision/recall/area and source audits. All workers completed, all
eight GPUs are idle, and no full20092 rollout or new monitor was launched. Goal open.

## Completed Class-Field Expansion Attribution

Zero image/text network forwards; reuse all64 developed windows' exact stress
scores. Removing the theoretical equal-count log2 offset changes zero predictions.
Class-mean gauge is fixed before own20/rivals40 and own40/rivals20 restorations.
Raw/centered endpoints and historical per-image confusions are exact; all scores
persist before audit masks. The experiments intervene on centered FINAL class
fields, not individual aliases or a newly evaluated mixed vocabulary. Pair actions
already mix classes and the gauge depends on all classes; do not identify word
ownership causally from these score interventions or add their gains as percentages.

VDD wall40 IoU37.22 becomes54.40 when its own20 field restores, but35.17 when only
rivals restore. Potsdam car40 IoU18.64 becomes17.05 with own restoration and26.65
with rival restoration. OEM building40 IoU7.68 becomes42.98/44.12 with own/rival
restoration. FLAIR herbaceous vegetation40 IoU38.30 becomes61.11/34.48. Similar
patterns exist without admission. This rules out a pure common count offset and
isolates different score pathways; it does not produce an oracle word selector.

Report/raw: RIVAL_EXPANSION_ATTRIBUTION_20261005.md and its matching raw directory.
Four mathematical tests pass locally/remotely. All64 complete inputs, targets,
confusions and source identities verify. No existing model or output changed.

## Completed Conditional Fine-Support Soft Weighting

The new source preserves original hard rejection, then sets each surviving alias's
weight to sigmoid(beta*native alias-versus-rival margin), numerical epsilon floor,
canonical/self-rival protection. Original fine-target projection, no-admission
posterior, Geometry H and all visual observations remain unchanged. Unlike old
1-risk weighting, old zero-risk weak aliases are also reweighted. This is response
responsibility, NOT correctness probability. No visual forward is added.

| Pool/stress | Projected hard | New fine support | Delta pp | Class-mean control | Identity-null mean |
| --- | ---: | ---: | ---: | ---: | ---: |
|20|45.4187|45.2175|-0.2012|45.2538|45.3427|
|30|45.3651|45.5166|+0.1515|45.3332|45.3949|
|40|44.2044|44.3148|+0.1104|43.9499|43.9461|
|Wrong-parent|44.5969|44.4042|-0.1927|44.5997|44.6615|
|Paraphrase|45.9874|45.7910|-0.1963|45.8916|45.8606|

The candidate FAILS frozen clean/stability/identity gates. Worst protocol loss is
2.1228pp on LoveDA D paraphrases.40 remains0.9027pp below its own20; do not claim
arbitrary expansion safe. Matched mass/spectrum controls support word-conditional
information on30/40 but contradict an across-regime useful weight estimator. No
control promotion, count/dataset switching or post-result temperature adjustment.

The same redistribution helps VDD20(+1.3559pp) but harms VDD40(-0.9580pp). Potsdam40
gains1.2982pp and UDD5's40 windows gain0.9796pp; LoveDA D40 loses1.8661pp. Potsdam20
car IoU improves25.18->26.22 while low vegetation falls21.73->19.50. OEM40 building
recall7.72%->9.93% leaves most target coverage missing; clean20 building worsens.
Do not tune class-specific repairs from these audit outcomes.

All64 developed windows/five regimes, exact historical scores/per-image controls,
weight spectra/fixed hard support/class-mean mass, frozen weights/checkpoints/words,
primary standalone/all-arm scores, targets and coverage verify. Twelve local/remote
reader tests pass (four new plus eight imported cached-reader tests). Scores persist
before masks. All schemes/source files remain. No standalone latency measurement
follows a failed accuracy gate; no-extra-forwards is not a zero-overhead measurement.

Keep the established full-suite hard option and frozen20 projected transfer candidate.
The next source should target conditional marginal discrimination/coverage rather
than simply absolute fine response or joint agreement. This study does not settle
that successful estimator, nor show soft weighting universally impossible. Report/
raw: RIVAL_FINE_SUPPORT_20261005.md and research/rival_fine_support_20261005/.
All workers completed; no full20092 rollout or monitoring automation was launched.

## Completed Equivalent Burst Execution

The opt-in four-crop burst backend is verified without changing batch-one
arithmetic, Geometry, observers, fixed20 words or either retained hard/projected
rule. All64 developed windows pass bitwise feature/risk/score checks and exact
historical per-image confusions for four endpoints.40 real RGB fragments cover
one/two/four crops; a complete3000x4000 VDD image has zero pixel mismatches and
identical diagnostics. Three real-CUDA burst tests and three full-reader regressions
pass. The initial tuple-versus-JSON-list signature-check failure preceded inference;
its log is preserved, with completed results in a separate r2 root.

| Endpoint | Eager s | Existing crop graph s | Burst graph s |
| --- | ---: | ---: | ---: |
| Original hard |0.299288|0.271553|0.271653|
| Frozen projected candidate |0.302269|0.274750|0.274112|

These are same-process seven-repeat warmed, synchronized, alternating independent
window-context timings, not full-image throughput. Original hard crop graph is
about9.27% faster than eager. Burst is0.0367% slower than crop graph for hard and
only0.2323% faster for projected, with mixed domain signs. No stable practical
burst gain is established; keep the simpler existing crop graph and retain burst
as an optional tested backend. Do not compound older timing gains.

Four-crop burst setup is0.426-0.489s. Co-resident measured hard peaks are identical
across the two graphs, but these are not isolated deployment memory comparisons.
The four visual observations still cost compute; removing launch/sync overhead
alone has reached little additional benefit here. No projected-burst full entry,
new full20092 rollout or automatic model replacement follows.

Source research remains open: absolute fine response/salience and joint observer
agreement have not yielded a robust soft selector. A future existing-feature-only
rule must estimate conditional alias discrimination/coverage, rather than assume
every alias is useful or that high agreement establishes semantic correctness.
Report/raw: FINE_OBSERVER_BURST_20261005.md and
research/fine_observer_burst_20261005/. All eight workers completed.

## Completed Semantic-Matched Rival Soft Support

One new source compares each surviving alias with its nearest original-template
text rival and uses w=1-positive_cosine*sigmoid(beta*(native_rival-native_alias)).
Original hard support/canonical protection, all observers, Geometry20, projection,
posterior and H remain fixed. No new forwards, templates, numerical fitting or
count/domain routing. Same64 developed windows and five frozen word regimes;
scores precede masks. Fourteen local/remote tests pass. All historical stress
score/per-image endpoints and preceding fine-support confusions replay exactly;
primary singleton scores and matched control spectra/mass verify.

|Pool/stress|Projected hard|Matched soft|Delta pp|Identity-null mean|
|---|---:|---:|---:|---:|
|20|45.4187|45.4293|+0.0105|45.4288|
|30|45.3651|45.7740|+0.4089|45.3290|
|40|44.2044|44.6498|+0.4454|44.0749|
|Wrong parent|44.5969|44.5533|-0.0436|44.6126|
|Paraphrase|45.9874|45.9468|-0.0406|45.9485|

The frozen accuracy/identity/stability gate FAILS.30/40 show7/8 and6/8 main-domain
wins and useful identity gaps0.4450/0.5748pp, but20's identity gap is only0.00043pp.
Wrong-parent and paraphrase do not improve on projected or their identity-null
means. Worst protocol loss is LoveDA P paraphrase-1.0974pp. No model/control/count
is promoted, and no standalone timing or full20092 rollout follows.

At40, Potsdam car IoU18.6362->20.3375 improves precision and reduces area while
recall stays near100%; low-vegetation recall28.4642->31.0121 also improves.
OEM building IoU7.6778->12.1701 and FLAIR herbaceous vegetation38.2996->43.5728
recover some coverage but remain well below their clean20 outcomes. VDD vehicle
precision worsens. Clean LoveDA tree recall falls while building benefits on
only307 target pixels. The outcome is partial expansion repair, not a general
positive/negative word classifier or proof that all aliases are useful.

The source does reweight jointly broad-positive/fine-positive comparisons
(mean0.66-0.78), so failure is not solely lack of access to the old sign blind
spot. Semantic neighbors and two-phrase responses remain insufficient as a
uniform correctness source. Relative canonical/class prior mass also changes;
holding original survivor/canonical mass fixed while redistributing word weights
is a distinct, not-yet-run test of conditional discrimination versus class prior.
Do not retune this failed rule's sigmoid/affinity strength from these targets.

Keep the full-suite hard option and frozen20 projected transfer candidate.
Report/raw: RIVAL_MATCHED_SUPPORT_20261005.md and
research/rival_matched_support_20261005/. All schemes remain; all eight workers
completed. The requested general useful selector/CVPR objective remains open.

## Completed Original-Survivor-Mass Redistribution

The preceding semantic-matched raw source is unchanged. Redistribute its mass
among ORIGINAL noncanonical hard survivors, w=|S|*u/sum_S(u), canonical1,
rejected0. Weights may exceed1; total survivor and canonical prior mass match
original hard. A positive-weight writer retains the original K/remaining
normalizer, stencils, evidence units and projection/posterior/H. All observers,
Geometry20 and frozen five word regimes remain unchanged. No new forwards or
fitted parameters. Same64 developed windows, not full/independent validation.

|Pool/stress|Projected hard|Prior matched|Redistribution|Identity-null mean|Delta pp|
|---|---:|---:|---:|---:|---:|
|20|45.4187|45.4293|45.4534|45.4550|+0.0346|
|30|45.3651|45.7740|45.8423|45.3977|+0.4772|
|40|44.2044|44.6498|44.7500|44.1970|+0.5456|
|Wrong parent|44.5969|44.5533|44.5669|44.6261|-0.0301|
|Paraphrase|45.9874|45.9468|45.9915|45.9594|+0.0041|

The gate FAILS. Clean20 identity gap is-0.00160pp; wrong-parent loses projected,
text-only redistribution and identity-null mean. Worst protocol loss is LoveDA P
paraphrase-0.9804pp.30/40 benefit survives fixed priors with7/8 and6/8 domain wins
and identity gaps0.4446/0.5530pp. This supports conditional information for these
expanded pools, not a general clean20 quality estimator. No count/domain/control
promotion, coefficient tuning or standalone timing follows the failed gate.

At40, Potsdam car18.6362->20.3866 improves precision while area falls and recall
stays near100%; low-vegetation recall28.4642->31.4793 also improves. OEM building
7.6778->12.2461 and FLAIR herbaceous vegetation38.2996->44.0185 recover some
coverage but substantial expansion damage remains. VDD vehicle precision loses.
Do not infer a universal useful/harmful phrase partition from these class outcomes.

Every real class-mean score exactly recovers projected hard. Actual total-mass
error<=2.132e-14, canonical-prior error<=8.327e-17. All historical stress scores/
confusions, preceding fine-support/matched confusions, singleton scores and
control spectra verify.13 local/remote tests pass. Scores precede masks.
All schemes are preserved; all eight workers complete. No full20092 or monitor.

The next scoped diagnostic should check the unchanged unscreened fine-minus-wide
projection target: it may refuse a conditional word correction although that
correction changes the pool's class evidence. This possible source/target
mismatch is NOT established by the current aggregate metrics and has not run.
Keep the established full-suite hard and frozen20 projected transfer options.
Report/raw: RIVAL_SURVIVOR_REDISTRIBUTION_20261005.md and
research/rival_survivor_redistribution_20261005/. Goal remains open.

## Completed Fixed-Source Projection Counterfactual

The proposed test is complete, without a new weight curve or visual observation.
Freeze survivor/canonical mass, original hard support, matched weights, posterior
and Geometry H. The primary target is T1=T0+D_F-D_F^T, where the exact same
conditional alias allocations act on fine crop evidence. Compare with T0 and no
projection, screened hard controls and three within-survivor word-identity nulls.
All64 developed windows/five regimes replay original scores/per-image endpoints,
survivor confusions and clean20 historical unprojected hard exactly. Weights/models
and word identities verify; masks follow score persistence.12 remote tests pass.

|Regime|Projected hard|Original survivor|Screened target|No projection|Primary delta pp|
|---|---:|---:|---:|---:|---:|
|20|45.4187|45.4534|45.5478|45.3257|+0.1291|
|30|45.3651|45.8423|46.0407|46.0107|+0.6755|
|40|44.2044|44.7500|44.9901|44.9848|+0.7857|
|Wrong parent|44.5969|44.5669|44.2987|43.3823|-0.2982|
|Paraphrase|45.9874|45.9915|46.0859|45.8420|+0.0986|

The advancement gate FAILS. Clean20 word-null advantage0.0718pp, wrong-parent
word-null deficit0.2516pp; worst protocol loss LandCover.ai wrong-parent-1.0068pp.
The changed target admits more posterior-weighted action both at20(63.84%->68.12%)
and wrong-parent(54.28%->63.79%), with opposite accuracy outcomes. These are summed
pair-action diagnostics, not pixel error counts. Source/target consistency changes
predictions but is not enough for a correctness source; shared fine evidence can
self-reinforce wrong words. Removing projection altogether is not a robust repair.

Retain this candidate diagnostically, without promoting a control or tuning its
bound. Preserve original full-suite hard, frozen20 projected transfer and exact
crop-graph speed backend. No general soft selector, arbitrary-expansion safety or
universal alias usefulness is established. Next source work must establish actual
conditional discrimination under wrong-parent as well as clean/expanded pools,
not merely admit more corrections or increase response confidence. A new source
is not implemented in this round. Report/raw: RIVAL_PROJECTION_AUDIT_20261005.md
and research/rival_projection_audit_20261005/. All eight jobs complete; GPUs idle.
No full20092 or new monitor. Goal remains open; all schemes remain for user choice.

## Completed Whole-Rival Response Residual Source

One new source was implemented and tested, keeping the original T0 projection,
all observers, Geometry20, posterior/H, hard support and survivor/canonical mass
fixed. An alias's original-template mean is represented in the entire rival text
dictionary by stable ridge1 on unit directions. Reuse raw fine responses before
salience to subtract this explained response. Source weight is positive residual
over positive residual+positive shared response, with original epsilon. This is
a regularized decomposition, not an exact projector or correctness probability.
No new neural forward, fitted parameter or count/domain routing.

|Regime|Projected hard|Residual source|Pure-text control|Identity-null mean|Delta pp|
|---|---:|---:|---:|---:|---:|
|20|45.4187|45.2100|45.2976|45.8649|-0.2087|
|30|45.3651|45.2075|45.1328|45.7961|-0.1576|
|40|44.2044|44.1128|43.7202|44.5016|-0.0916|
|Wrong parent|44.5969|44.0736|44.5837|44.8523|-0.5233|
|Paraphrase|45.9874|45.5321|45.7646|46.1353|-0.4553|

The gate FAILS. Identity deficits0.6549pp at20 and0.7787 for wrong-parent contradict
this source's allocation direction, not merely a too-small attenuation. Worst loss
LoveDA D paraphrase-3.1314pp.41.19% of eligible source comparisons reach epsilon at20
(not pixel errors or deleted-word counts). Potsdam car and VDD vehicle benefit,
but Potsdam low-vegetation recall and VDD wall/roof/water lose; LandCover.ai wrong-parent
building false positives remain worse. Shared response is not proven harmful and
exclusive response is not a correctness teacher. Do not tune ridge, invert the
rule from labels or promote the random controls after observing these results.

All64 developed windows/five pools complete;12 local/remote tests, original scores/
confusions, survivor replay, class-mean exact projected recovery, primary singleton,
support/canonical/mass/spectrum and frozen identities verify. Actual mass error
<=7.106e-14. All workers complete and GPU0-7 idle. Code/protocol/results are retained
at RIVAL_SUBSPACE_SUPPORT_20261005.md and research/rival_subspace_support_20261005/.
No standalone timing/full rollout. Retain the established full hard, frozen20
projected transfer and exact acceleration. The general screening goal remains open.

For subsequent source design, a text-unique or confident response is insufficient;
the source must distinguish the alias's contribution to current class competition
without treating shared class evidence as automatically bad. Existing held-out
ownership/canonical witnesses also failed, so do not rename those as a new solution.
The next source is not yet implemented. Preserve all options for user selection.

## Completed Conditional Responsibility Intersection

One new source was completed, rather than tuning the failed text residual. At
each original observed token, compute p(alias|own,rival)=exp(beta*profiled_alias)/
(Z_own+Z_rival); mix probabilities through original crop stencils in log space.
Raw log weight=min(0,log(p_fine)-log(p_wide)). Original hard rejection, canonical
protection, survivor/canonical mass and T0 projection/posterior/H stay fixed.
No additional neural forward or fitted parameter; four original fine forwards
remain. Algebraic responsibility intersection holds BEFORE epsilon/protection/
redistribution, not for final pixel decisions or correctness.

|Regime|Projected hard|Pair intersection|Within-class ratio|Fine-only control|Own-primary null mean|
|---|---:|---:|---:|---:|---:|
|20|45.4187|45.5699|45.3473|46.0243|45.4544|
|30|45.3651|45.5542|45.3598|46.1437|45.3985|
|40|44.2044|44.2921|44.0317|45.1318|44.2035|
|Wrong parent|44.5969|44.7116|44.4806|44.7752|44.6284|
|Paraphrase|45.9874|45.5590|45.4018|46.1762|45.9644|

Primary FAILS its frozen gate. Six/eight clean20 domains win, identity advantages
0.1155pp at20 and0.0832 for wrong-parent, and gains versus within-class ratio
0.2226/0.2309pp support conditional information there. But paraphrase loses0.4284pp
versus projected and0.4054pp versus own null; worst VDD paraphrase-3.3701pp. It
also loses the fine-only control and suppresses VDD20 water recall95.11%->87.22%.
Keep this diagnostic, not a final model or permission to tune its clipping ratio.

The predeclared FineResponsibilityOnly control gains0.6055/0.7786/0.9274/0.1783/
0.1889pp over projected for20/30/40/wrong-parent/paraphrase. This is a development
LEAD, not automatic control promotion: it has no own identity nulls in this run;
primary nulls are inapplicable. Six/eight clean20 wins, worst protocol loss1.6943pp,
own wrong-parent damage1.2491pp. Both shared errors and class coverage losses remain.

Next freeze this exact already-recorded fine-only source for an OWN mechanistic
comparison: alias-identity nulls preserving support/mass/spectra; wide-only
responsibility sharpening control; fine within-class source without rival
conditioning; source-position null. A constant rival factor cancels during
noncanonical mass redistribution, and identical-view responsibility weighting
can sharpen evidence. These alternatives must be separated before claiming new
alias reliability or competition specificity. Keep coefficients/counts/Geometry/
observers/T0/posterior/H unchanged. Do not simply invert a failed source or apply
primary nulls to the control. A favorable new comparison then needs frozen
complete-input transfer and independent matched timing, not full20092 promotion.

All64 developed windows/five regimes complete;13 local/remote tests and original
scores/confusions/survivor replay/class-mean exactness verify. Actual mass error
<=2.843e-14, raw log identity<=8.882e-16. All outputs/schemes retained at
RIVAL_RESPONSIBILITY_INTERSECTION_20261005.md and its raw directory. GPU0-7 idle.
No full20092/new monitor or standalone latency. The goal remains open.

## Completed Frozen Fine-Responsibility Own Audit

The proposed OWN mechanism audit is complete on the same64 developed windows,
eight/domain/five frozen word regimes. Freeze the exact previous fine-only source,
original Geometry20, hard/canonical/survivor support/mass and T0/posterior/H.
No extra visual forward, fitted coefficient/count/class/domain routing or main-model
replacement. All original scores/confusions and previous fine-only scores/confusions
replay exactly; own identity-null spectra/mass, uniform hard recovery and singleton
checks pass. Ten local/remote tests pass. All eight workers complete; GPU0-7 idle.

|Regime|Projected|Fine-only|Own identity-null mean|Within-class|Wide-only|
|---|---:|---:|---:|---:|---:|
|20|45.4187|46.0243|45.6224|46.0240|45.8011|
|30|45.3651|46.1437|45.4950|46.1436|46.1274|
|40|44.2044|45.1318|44.1761|45.1363|45.3149|
|Wrong parent|44.5969|44.7752|44.7363|44.7792|44.5291|
|Paraphrase|45.9874|46.1762|45.8432|46.1861|46.7365|

Fine-only has positive own word/location-null gaps at20 and wrong-parent, but the
incremental rival-denominator benefit fails. Removing that denominator changes the
clean20 mean by only0.0003pp and every clean20 protocol by<0.009pp. Factorized
responsibility matches within-class mIoU in all regimes; max allocated difference
1.421e-14 and all eligible epsilon-floor fractions0. A common pair factor really
cancels here. Primary allocations can differ substantially through covariance, but
the fixed final readout converts almost none of it into extra accuracy. Do not claim
a new competition-specific reliability source from the fine-only mean advantage.

The frozen advancement gate FAILS: worst protocol loss LoveDA D paraphrase-1.6943pp,
wrong-parent damage1.2491pp vs projected0.8218pp, and no advantage above within-class
or factorized on both20/wrong-parent.40 remains0.8925pp below its own20; wide-only
exceeds fine-only at40 and paraphrase. No source-per-domain routing/control promotion.
Potsdam car IoU25.1793->27.5389 improves while low-vegetation21.7324->19.7584 falls.
Conditional reweighting still has competition-induced coverage errors.

Keep all schemes and the established full hard/frozen20 projected transfer/exact
speed backends. Fine-only remains a development accuracy lead, not a universal
selector or new final model. Next useful evidence must distinguish true alias
discrimination from confident/shared response, rather than another denominator
that cancels after normalization. Reuse-only local/wide weighting remains distinct
and must earn its own accuracy evidence. No standalone timing/full20092 rollout or
new automation follows this failed gate. Report/raw: FINE_RESPONSIBILITY_AUDIT_20261005.md
and research/fine_responsibility_audit_20261005/. Goal remains active for user choice.

## Completed Geometry-Referenced Conditional Response Density

One distinct source is complete on the same64 developed windows/five pools. The
current fine alias response is matched to its leave-query-out class-conditioned
response distributions, with the original Geometry20 local posterior as a soft
reference. Equal class priors remove a class-area prior; missing numerical support
uses neutral1/2. Original beta/epsilon, no fitted parameter or neural forward.
Primary multiplies the frozen fine-only allocation by own-versus-rival density
responsibility and conserves original hard/canonical/survivor mass and T0/posterior/H.
Unlike previous anchored global means or common rival factors, the new source is
conditional on current response value. It is not a correctness teacher, semantic
family holdout, or independent scene reference.

Exact sorted log-prefix/suffix computation matches dense kernel sums; no NxNxalias
tensor. Ten local/remote tests pass, including ties/extremes, own-query exclusion,
class balance, exact old/fine-only scores, mass/spectra and singleton. All64 keys,
five word identities/checkpoints/frozen states, per-image confusions and target sums
verify; scores persist before masks. All eight workers complete and GPUs idle.

|Regime|Fine-only|Density primary|Increment pp|Density-only|Reference-null|
|---|---:|---:|---:|---:|---:|
|20|46.0243|46.0248|+0.0005|45.4617|46.0236|
|30|46.1437|46.1604|+0.0166|45.4494|46.1435|
|40|45.1318|45.1472|+0.0155|44.3018|45.1310|
|Wrong parent|44.7752|44.7770|+0.0018|44.6553|44.7754|
|Paraphrase|46.1762|46.1569|-0.0193|45.9956|46.1760|

Gate FAILS: no0.1pp increment at20, worst vs projected LoveDA D paraphrase-1.6857pp.
Own allocation identity nulls remove the earlier fine-only identity as well; their
large gaps cannot validate the NEW density contribution. Reference/position controls
preserve fine-only, and their20 gaps are only0.0011/0.0013pp. Query exclusion and
class balance do not demonstrate benefits at both20/wrong-parent. Eligibility
availability is1 throughout, source mean0.49767..0.50020, raw rival range max0.29733.
There is actual source variation, not useful final discrimination. Do not fit kernel
beta, add class/domain routing or promote a control to rescue this failed increment.

Retain this implementation/protocol/raw data as a negative result; do not stack it
onto the final model. Keep full hard/projected20 transfer/fine-only accuracy lead
and exact acceleration. Rather than another confidence factor, next characterize
the already frozen fine-only lead on complete inputs and independently matched cost,
without erasing its failed robustness gate or promoting it automatically. No timing
claim, full20092 rollout, automation or unrelated resumption. Report/raw:
RIVAL_CONDITIONAL_DENSITY_20261005.md and research/rival_conditional_density_20261005/.
Goal remains active; all schemes remain for the user's choice.

## Completed Frozen Fine-Only Complete-Input Characterization

The already recorded fine-only rule, not another fitted formula, is now verified
on152 complete inputs: full UDD5=40 plus16/domain in the seven other domains.
Reused transfer manifests are identical. Four original Geometry/no-admission/hard/
projected controls replay exactly PER IMAGE; fixed20 vocabularies/checkpoints,
observer/risk/canonical/survivor support/mass, T0/posterior/H and weights verify.
Independent singleton/all-arm real scores are bitwise equal. This remains an
exploratory previously examined panel, not untouched/full20092 evaluation.

Eight-domain means: projected49.3271->fine-only49.8854 (+0.5583pp), versus hard
48.8427 and no admission47.3874. Seven non-UDD5 gain0.5897pp with5/7 wins, worst
protocol-0.1668pp, own identity-null gap0.3779pp, empirical paired bootstrap95%
interval[0.3493,0.8161]pp. The complete-input fixed20 characterization gate passes;
the earlier wrong-parent/paraphrase robustness failure remains failed.

Within-class49.8886 versus primary49.8854 still does not support the novel rival
denominator. Existing hard rejection remains competition-conditioned. Soft source
allocation is among those original survivors, not a no-screening method. Wide-only
49.9021 also retains fine observations for frozen support and projection. Do not
promote either control after seeing results or claim every alias useful.

Class tradeoffs remain: UDD5 road-2.6941pp despite mean+0.3389; VDD mean+0.0116 with
vehicle+1.5120 offset by other/road; OEM rangeland-4.1489. Potsdam's six classes
improve on these16 images, car+2.2822 and low vegetation+2.8422, but car precision
is still26.82%. LandCover.ai/FLAIR-1 means lose0.1277/0.1668pp versus projected.

Matched independent seven-repeat window-context latency: cached hard eager
0.300743s, graph hard0.273154, projected graph0.276175, fine-only graph0.283774.
Equivalent graph hard is9.17% faster than equally cached eager; primary is2.75%
slower than graph projected, with unchanged measured co-resident allocated peak.
Four original fine forwards remain. These are not whole-image/isolated deployment
costs, nor speedups to compound with earlier sessions.

Keep the established full hard model, projected20 transfer option, fine-only
accuracy candidate and all alternatives. The positive fixed20 accuracy result
does not authorize replacing the retained model, arbitrary expansion, label-based
count/domain routing or a full20092 rollout. Existing reuse-only sources have not
earned comparable accuracy, so deleting fine observations is still a separate
unresolved speed/accuracy problem. A successful competition-specific soft source
must add discrimination beyond class-internal response sharpening, not another
normalizer that nearly cancels. No new source/monitor was launched after this run.

Report/raw: FINE_RESPONSIBILITY_COMPLETE_20261005.md and
research/fine_responsibility_complete_20261005/. All workers finished, GPUs0-7 idle,
and the research goal remains active for user choice.
