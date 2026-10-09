# Alias screening: retained accuracy, exact acceleration, soft alternatives

## Decision

Retain the original Geometry + finite VIP wide observer + coupling and the existing
RivalFineHard rule. Use the separate opt-in cached reader when speed matters.
It changes execution, not admission equations, observations, vocabulary or predictions.
All alternative source files and experimental outputs are preserved for later choice.
No full20k rerun, automatic model replacement or monitoring automation was started.

## Verified Pilot Results

Two studies reused the same64 developed top-left512 windows, eight/domain, with
unchanged full-image wide context and fixed20 aliases/class. They are NOT full
eight-dataset results. LoveDA D counts once in the mean; P is reported separately.
Scored-class denominators are common across arms within each domain/protocol.

| Reader | Eight-domain mIoU | Gain vs no admission, pp | Mean window-context seconds |
| --- | ---: | ---: | ---: |
| No admission | 42.8679 | 0 | 0.1556 |
| Original RivalFineHard | 45.0622 | +2.1942 | 0.3784 |
| Cached, same RivalFineHard | 45.0622 | +2.1942 | 0.2936 |
| Fine risk, normalized1-r weights | 43.5555 | +0.6876 | 0.2935 |
| Fine risk, excess-only attenuation | 43.6513 | +0.7833 | 0.2911 |
| Fine risk, responsibility-aware weights | 43.6100 | +0.7420 | 0.3191* |
| Reused local features, normalized weights | 42.9927 | +0.1248 | 0.1810 |
| Reused local features, excess-only attenuation | 43.0730 | +0.2051 | 0.1749 |
| Reused local features, class-mean risk control | 42.8616 | -0.0064 | Not timed independently |

*Responsibility-aware cost was measured in a later session; do not interpret small
differences as a controlled speed comparison. Other timings were alternated across
warmed arms, with three synchronized repeats. All exclude loading, encoding,
decoding and GT. They include one local512 window plus full-image wide context,
not whole-image throughput or shared multi-arm evaluation wall time.

Cached hard is1.115-1.515x faster across these contexts; ratio of mean times1.289x,
about22.4% lower mean latency. Both hard and the two ordinary fine soft methods
improve every main domain versus no admission. Hard remains best in every main
domain in this panel; none of the new soft rules improved on it.

## What This Establishes

- Exact acceleration works: original/cached scores, risks and diagnostics are
  bitwise identical on64 windows. A complete3000x4000 VDD image also has zero
  prediction differences for all three retained endpoints.
- Fine observations remain valuable: the extra fine stage costs about0.10s/window,
  but removing it loses most screening gain. Reusing local features is a cheaper
  distinct rule, not equivalent acceleration.
- Soft handling works versus no admission, but is not yet the accuracy leader.
  Raw risk is not a calibrated probability of an alias being wrong. A normalized
 1-r weight often changes actual log-mean-exp evidence much less than hard rejection.
- Simply adding readout responsibility to the weight was insufficient: it improves
  the ordinary normalized rule by only0.0544pp and is below excess-only soft by0.0413pp.
- Alias-specific reuse weights exceed the same-budget class-mean control by0.1312pp;
  this is a small, nonuniform effect, not proof of a complete publication contribution.
- VDD illustrates overcorrection: hard gains vegetation/water but loses road/vehicle
  on these windows. Soft retains more vehicle/road performance but recovers less
  vegetation/water. UDD5 road is0 for every arm here; these top-left windows cannot
  resolve whole-dataset road performance.

## Paper Direction

The supported framing is **competition-conditioned alias admission**, not a permanent
good-word/bad-word list and not "every alias is useful". Current hard admission is
already query/patch-and-rival-dependent; it does not globally blacklist a word.
For the same alias, reliability can differ across regions and rival classes.

The cache implementation is an efficiency result, not a novel semantic contribution.
New soft curves must establish an accuracy/robustness benefit beyond unchanged
coupling, class-only calibration and the retained conditional hard rule. Developed
pilot gains do not establish SOTA, cross-domain generalization or CVPR acceptance.

The next informative experiment is to calibrate suppression against its actual
counterfactual pair-margin effect while retaining the fine witness, with matched
action-strength controls. Independently freezing/evaluating that rule and its
30/40-word or wrong-parent robustness would test generalization. These experiments
were NOT launched here; keep the accuracy-leading cached hard model meanwhile.

## Artifacts And Reproduction

- `research/RIVAL_ALIAS_SPEED_SOFT_20261005.md`: full domain table, per-class metrics,
  precision/recall/area, stages, memory, historical replay and Geometry writer audit.
- `research/RIVAL_ALIAS_INFLUENCE_SOFT_20261005.md`: separate responsibility-aware
  follow-up, motivated by the preceding labeled development panel, not held-out data.
- Raw results: `research/rival_alias_speed_soft_20261005/` and
  `research/rival_alias_influence_soft_20261005/`.
- Execution-only entry: `DINOtool/scripts/eval_rival_fine_fast.py` accepts the same
  arguments as the retained full evaluator and refuses existing outputs.
- Managers: `tools/rival_alias_speed_soft_experiment.py` and
  `tools/rival_alias_influence_soft_experiment.py`, using the existing checked SSH
  connection and GPU-idle checks. Original/best evaluators remain unchanged.

Sixteen reader tests plus three original full-reader regression tests pass. The
legacy window Geometry writer interpolated temperature-scaled scores before
upsampling; the retained full evaluator scales after upsampling. Stored-score audits
explain all10 L1 confusion-count differences and reproduce old per-image controls;
these are not acceleration errors. No admission and hard historical controls are
exact per image. The responsibility follow-up retains exact unscreened scores and
per-image confusions. Its initial wrapper-default failure preceded prediction;
failed logs and separate r2 outputs are preserved remotely.

## Follow-Up Status

Competition-action projection/posterior reconciliation reaches45.4187 on the same
developed64 windows and48.1196 on96 complete images (full UDD5 plus eight developed
images from each other domain), versus original hard45.0622/47.6844. The complete
panel gains0.4352pp, seven/eight wins, VDD-0.2637pp. This is not untouched/full-suite
validation. The original established model is still preserved.

Subsequent mass admission, margin-unit soft attenuation, rival-neutral replacement
and salience-mass normalization fail their frozen accuracy/stability/mechanism
gates; no automatic candidate promotion follows. Fine salience mass45.4002 and
wide45.3861 do not beat45.4187, and their near-uniform priors show no useful new
alias-identity advantage. All implementations/results remain available. See
ALIAS_SCREENING_NEXT_DECISION_20261005.md for exact decisions and linked reports.

Further execution-only acceleration is complete: original batch-one512 CUDA graph
plus cached hard averages0.279341s versus0.306597s fine-eager cached hard,8.89%
lower window-context latency. Fine stage falls25.19%-29.03%; peak allocated memory
rises26.09-35.42MiB. All64 windows' features/risks/scores/diagnostics/predictions
and one complete3000x4000 VDD image are exactly preserved. These timings are not
compounded with the earlier cache gain across sessions. All fine observations
remain; no semantic model change or full-suite rollout occurred.

Opt-in entry: DINOtool/scripts/eval_rival_fine_graph.py. Full report/raw results:
FINE_OBSERVER_EXECUTION_20261005.md and research/fine_observer_execution_20261005/.
No active workers remained after verified collection.

## Key-Disjoint Transfer Update

FineRivalProjected is now frozen and evaluated on112 NEW complete inputs,16 in
each of seven domains, key-disjoint from the current developed64/complete96 panel.
UDD5's already-examined full40 results are reused separately. No rules, templates,
parameters or domain routing changed; this remains exploratory rather than
scene-disjoint/untouched independent validation.

Seven new-domain means: no admission47.1268, original hard48.2551,
FineRivalProjected48.7746. Candidate improves all7 domains,+0.5195pp versus hard
and+1.6477pp versus no admission; empirical paired bootstrap95% mean-gain interval
is[+0.3734,+0.7332]. Eight-domain descriptive mean including reused UDD5 gains
0.4844pp. All historical Geometry/no-admission/hard per-image controls match exactly.

Matched window-context timings: eager cached hard0.296301s, graph hard0.268146s,
graph projected0.270855s. Candidate is8.59% faster than eager, but1.01% slower than
matched graph hard. This satisfies the current practical accuracy/speed direction
without adding observations; these are not whole-image throughput measurements.

FineBudgetOnly diagnostic is48.6931, only0.0815pp below the candidate. VDD roof
still loses1.0122pp; Potsdam car precision is only24.51%, and OEM coupling remains
below Geometry alone. Keep these limitations, the original full-suite model and
all alternatives. Source alias rejection remains hard, with soft rival coordination,
not a new continuous per-alias writer.

Preferred frozen candidate entry: DINOtool/scripts/eval_rival_projected_graph.py.
Report/raw data: RIVAL_COMPETITION_TRANSFER_20261005.md and
research/rival_competition_transfer_20261005/. No automatic model promotion or
full20092 rollout occurred. All workers completed and results were verified.

## Frozen Vocabulary Stress Update

The five-scenario64-window study is now complete, with exact historical per-image
controls, clean20 projected scores, Geometry/operator identities and source persistence
before audit masks. These are developed windows, not full datasets. Every scenario
shares visual observations; only contextual words differ, while local Geometry20
remains unchanged. Common scored-class supports prevent vocabulary-specific denominators.

| Pool/stress | No admission | Original hard | Projected candidate |
| --- | ---: | ---: | ---: |
|20|42.8679|45.0622|45.4187|
|30|43.0357|45.1366|45.3651|
|40|41.7421|43.9695|44.2044|
|Wrong-parent replacements|41.9892|44.2848|44.5969|
|Legitimate paraphrases|43.6469|45.6935|45.9874|

All scenario means improve over matched hard, but the frozen robust-advancement gate
FAILS: LoveDA P30/40 lose2.7580/1.3796pp versus matched hard; candidate wrong-parent
damage0.8218pp is slightly worse than hard0.7774pp.40 loses1.2143pp versus candidate20.
Keep the preferred20 transfer candidate with its limited claim and original full-suite
model; no automatic promotion, count routing or full20092 rollout follows.

Added aliases retain90.09%-94.40% of eligible query/alias/rival comparisons. The old
rule cannot reject broad-positive/fine-nonnegative evidence, including joint errors.
OEM building expansion loses coverage (recall70.73%->7.72%, IoU54.55->7.68) while
VDD vehicle improves17.40pp and wall loses15.92pp. Counts and phrase identities change
together; this is not proof every word is useful or stronger deletion always helps.
Prior soft/ownership failures remain failures. Conditional contribution admission,
not universal permanent deletion, remains the supported research framing.

Report/raw: RIVAL_PROJECTED_VOCABULARY_STRESS_20261005.md and
research/rival_projected_vocabulary_stress_20261005/. Exact class metrics/source audits
are retained. All workers finished, GPUs0-7 are idle, and every scheme remains available.
The successful general selector/CVPR goal remains open.

## Expansion Pathway And Fine-Support Update

The zero-network-forward class-field attribution verifies64 developed windows,
exact original/centered endpoints and unchanged predictions after removing the
equal-count common log2 term. VDD wall40 IoU37.22 rises to54.40 when its own20
centered field restores; Potsdam car40 IoU18.64 rises to26.65 when rivals restore.
OEM building40 IoU7.68 rises to42.98/44.12 with own/rival restoration. These are
defined final-score interventions, not causal identification of bad aliases or
a selectable hybrid-vocabulary model. Report: RIVAL_EXPANSION_ATTRIBUTION_20261005.md.

The subsequent continuous fine-support weights retain original hard rejection,
all observers/projection/posterior/H and canonical protection. Surviving weights
are sigmoid(beta*native alias/rival margin). No new visual forward or target fitting.
This is a different source from earlier1-risk attenuation: weak old-zero-risk
words also change. Class-mean mass and three within-support weight-spectrum
identity-null controls are matched exactly.

| Pool/stress | Projected hard | Fine support | Delta pp |
| --- | ---: | ---: | ---: |
|20|45.4187|45.2175|-0.2012|
|30|45.3651|45.5166|+0.1515|
|40|44.2044|44.3148|+0.1104|
|Wrong-parent|44.5969|44.4042|-0.1927|
|Paraphrase|45.9874|45.7910|-0.1963|

The frozen advancement gate FAILS. Worst protocol loss2.1228pp (LoveDA D paraphrase).
It beats mass/identity controls on30/40 but loses both at20/wrong-parent, so absolute
fine support is not a reliable general contribution-quality estimator. Do not tune
temperatures or select a control/count/domain winner after this test. VDD20 improves
1.3559pp but40 worsens0.9580pp; Potsdam40 improves1.2982pp, yet clean20 vegetation
coverage falls. All per-class/source outputs and candidates remain preserved.

Twelve reader tests pass locally/remotely, including four new tests. All64 keys,
all historical score/per-image endpoints, source/checkpoint/word/model identities,
control spectra/mass, primary singleton scores and target/confusion sums verify.
No independent latency claim follows a failed accuracy gate. Retain original exact
acceleration and frozen20 projected candidate; report/raw: RIVAL_FINE_SUPPORT_20261005.md
and research/rival_fine_support_20261005/. All jobs finished, no full20092 or monitor
was started. The requested useful general selector/CVPR goal remains open.

## Equivalent Burst Execution Update

All64 developed windows,40 real-RGB fragment cases and the complete3000x4000 VDD
image pass unchanged hard/projected execution checks. Features, risks and window
scores are bitwise exact; historical per-image confusions and full-image predictions
are unchanged. Three real-CUDA burst unit tests and three full-reader regressions
pass. The first signature-check formatting failure and all older outputs remain
preserved; the completed run uses an independent r2 root.

Same-session independent seven-repeat means: hard eager0.299288s, existing
crop graph0.271553s, burst0.271653s; projected eager0.302269s, crop graph0.274750s,
burst0.274112s. Scope remains one512 local window with full-image wide context,
not whole-image throughput. Hard crop graph saves about9.27% versus eager;
burst adds no meaningful gain (hard-0.0367% speedup, projected+0.2323%).
Keep the simpler existing graph; all backends remain available. Four-crop setup
costs0.426-0.489s, and co-resident memory figures do not measure isolated backend
deployment. No compounded speed claim or model/full-suite promotion follows.

The four extra visual observations remain. To avoid them, a new dynamic source
still needs accuracy and matched-identity evidence; previous reuse weights recover
little of hard screening's benefit. The successful universal soft selector is not
established. Report/raw: FINE_OBSERVER_BURST_20261005.md and
research/fine_observer_burst_20261005/. All eight workers finished; goal remains open.

## Semantic-Matched Rival Support Update

The next frozen soft source is complete on the same64 developed windows/five
word regimes. Each alias compares against its nearest original-template semantic
rival; weights depend on that specific rival alias's native response, not only
an aggregate rival class response. Original hard support/protection and all
Geometry/visual/projection/posterior/H components remain unchanged. No extra
forward or fitted coefficient. Fourteen local/remote tests and exact historical
scores/confusions, singleton scores and matched controls pass.

Projected->matched means:20 45.4187->45.4293;30 45.3651->45.7740;
40 44.2044->44.6498; wrong-parent44.5969->44.5533;
paraphrase45.9874->45.9468.30/40 beat matched identity-null means by0.4450/0.5748pp,
but20's gap is only0.00043pp. The gate FAILS: clean gain negligible, wrong-parent
damage worse, worst protocol loss LoveDA P paraphrase-1.0974pp, no robust
clean/wrong-parent identity advantage. Preserve, do not replace the20-word model.

Potsdam40 car improves18.64->20.34 with fewer false positives, and low vegetation
coverage increases. OEM40 building improves7.68->12.17 and FLAIR herbaceous
vegetation38.30->43.57, but major expansion damage remains. It reaches the old
joint-positive blind spot, yet semantic proximity/response agreement is not a
uniform semantic correctness witness. No standalone timing/full rollout follows
the failed accuracy gate; all workers completed and all options remain.

Report/raw: RIVAL_MATCHED_SUPPORT_20261005.md and
research/rival_matched_support_20261005/. A future mass-preserving survivor
redistribution test is recorded but has not run. Goal remains open.

## Original-Survivor-Mass Redistribution Update

The recorded test is now complete. Keep the same semantic-matched raw source,
hard support and canonical weight1, but redistribute original noncanonical
survivor mass with w=|S|*u/sum_S(u). Positive weights can exceed1; original
survivor/canonical prior and K/remaining normalizer remain fixed. All observers,
Geometry/projection/posterior/H and five word regimes remain unchanged.

Means against projected hard:20 +0.0346pp;30 +0.4772;40 +0.5456;
wrong-parent-0.0301; paraphrase+0.0041.30/40 gains exceed identity-null means
by0.4446/0.5530pp, showing a pool-specific word effect beyond changed prior mass.
Clean20 remains0.00160pp below its null mean; wrong-parent also loses its null.
Worst protocol loss LoveDA P paraphrase-0.9804pp. The advancement gate FAILS.
Keep all implementations/results without count/domain/control promotion.

All64 developed windows verify actual mass<=2.132e-14 and canonical-prior
error<=8.327e-17. Every real class-mean score exactly equals projected hard.
Historical endpoints, per-image confusions, primary singleton and control
spectra verify;13 local/remote tests pass. Scores precede masks. This is not
full/independent validation or a measured speed gain; no timing/full rollout
follows the failed accuracy gate. All eight workers finished.

Keep the full-suite hard and frozen20 projected transfer candidate. Report/raw:
RIVAL_SURVIVOR_REDISTRIBUTION_20261005.md and its retained raw directory.
The next source/target projection diagnostic is only proposed, not executed:
the projection reference still uses unscreened fine/wide class fields while
the requested correction reweights words. Whether this blocks useful actions
needs direct evidence, not another untested weight curve. Goal remains open.

## Completed Projection-Target Counterfactual

The proposed fixed-source diagnostic is complete. Geometry20, observers, hard
support, survivor/canonical mass, posterior and H stay fixed. The primary uses
the same alias allocation's fine change to replace T0 by T0+D_F-D_F^T. No projection,
hard class controls and three matched word-identity nulls distinguish the effects.
Same64 developed windows/five pools, not full or independent validation. Original
scores/confusions and survivor predictions replay exactly; clean20 unprojected
hard recovers the prior posterior endpoint.12 remote tests pass, actual mass error
<=2.132e-14; source/checkpoint/words/weights/targets verify and scores precede masks.

Projected->primary means:20 45.4187->45.5478;30 45.3651->46.0407;
40 44.2044->44.9901; wrong-parent44.5969->44.2987;
paraphrase45.9874->46.0859. Five/eight clean20 domains improve. Gate FAILS:
wrong-parent loses projected and its null, worst loss LandCover.ai wrong-parent
-1.0068pp, and40 remains0.5577pp below own20. Clean20's word-null gap is0.0718pp;
30/40 gaps0.6113/0.7815pp. Do not promote a regime/control or tune the bound.

Summed main-protocol retained posterior action mass rises63.84%->68.12% at20 and
54.28%->63.79% for wrong-parent. More action is not correctness: the latter mean
gets worse. Without projection20/wrong-parent are45.3257/43.3823. The old target
is a causal component, but self-consistent fine-weight evidence is not sufficient
to reject shared semantic mistakes. Potsdam40 car/low-vegetation improve; wrong-parent
LandCover.ai building false positives and OEM developed-space missed coverage worsen.

Keep established full-suite hard, frozen20 projected transfer and exact crop graph.
All schemes/results remain. All jobs complete/GPU0-7 idle. No standalone timing,
full20092 promotion or automation follows this failed screen. Report/raw:
RIVAL_PROJECTION_AUDIT_20261005.md and research/rival_projection_audit_20261005/.
General conditional useful-weight estimation remains open; do not claim every alias
useful, arbitrary expansion harmless, or a CVPR contribution solved from these means.

## Whole-Rival Positive Residual Support Update

Completed another fixed-source comparison: original Geometry20, wide/fine visual
observations, T0 projection/posterior/H, hard support and canonical/survivor mass
unchanged. Ridge1 represents each unit original-template alias in the entire
rival text dictionary. Reuse raw fine responses to split shared/exclusive parts;
weight=positive exclusive/(positive exclusive+positive shared), original epsilon.
Redistribute original noncanonical survivor mass. No extra neural forward/fitted
coefficient/count routing. Regularized decomposition is not exact projection or
a calibrated correctness source.

Means projected->primary:20 45.4187->45.2100;30 45.3651->45.2075;
40 44.2044->44.1128; wrong-parent44.5969->44.0736;
paraphrase45.9874->45.5321. All frozen gates FAIL. Every regime loses its matched
word-identity-null mean;20/wrong-parent deficits0.6549/0.7787pp. Worst protocol
loss LoveDA D paraphrase-3.1314pp. Four/eight clean20 main domains improve but
mean falls. Do not promote shuffled controls or fit ridge/strength from targets.

Potsdam car25.1793->26.3960 improves but low-vegetation21.7324->18.9682 and recall
22.8310%->19.8208% fall. VDD vehicle improves while wall/roof/water lose. This
supports rejecting this SOURCE, not a universal verdict on soft weighting. Shared
response may be useful and exclusive response wrong; spectral/text separation
alone does not establish conditional pixel utility.

All64 developed windows/five pools, exact old scores/confusions and preceding
survivor replay verify; every class-mean score is exact projected hard. Twelve
local/remote tests and primary singleton pass; actual mass error<=7.106e-14.
Scores precede masks. All jobs complete, GPU0-7 idle. All implementations/results
preserved, no standalone latency/full20092 rollout. Report/raw:
RIVAL_SUBSPACE_SUPPORT_20261005.md and research/rival_subspace_support_20261005/.
Keep established full hard, frozen20 projected transfer and equivalent crop graph.
The requested useful all-domain screening/CVPR objective is not yet achieved.

## Conditional Cross-View Responsibility Intersection Update

Completed a new actual-readout-unit source. The pair alias responsibility is
exp(beta*profiled_alias)/(Z_own+Z_rival), mixed through original crop stencils in
log space. Raw source=min(1,p_fine/p_wide), followed by original hard/canonical
protection, epsilon and survivor/canonical mass conservation. T0 projection,
posterior/H, Geometry20 and every visual observation stay fixed. No new neural
forward/fitted parameter; original four fine observations still required.

Primary gains versus projected:20+0.1512pp,30+0.1891,40+0.0877,wrong-parent+0.1146,
paraphrase-0.4284. Clean20 six/eight wins. Matched source identity advantages
0.1155/0.0832pp on20/wrong-parent, and gains0.2226/0.2309 over within-class ratio.
Unlike the residual source, this shows conditional information in those regimes.
Nevertheless the gate FAILS: primary trails fine-only, VDD paraphrase-3.3701pp,
and paraphrase loses own null0.4054pp. VDD20 water recall95.11%->87.22% also falls.

The predeclared fine-only CONTROL is a stronger next lead, not promoted: gains
0.6055/0.7786/0.9274/0.1783/0.1889pp for20/30/40/wrong-parent/paraphrase; six/eight
clean20 wins. Worst protocol loss LoveDA D paraphrase1.6943pp, own wrong-parent
damage1.2491pp. It has no own identity nulls yet; primary nulls cannot validate it.
Next freeze its exact rule and test OWN identity nulls, wide-only sharpening,
fine within-class (no rival) and source-position controls. This must distinguish
cross-view useful evidence from sharpening/rival-factor cancellation before a
complete-input transfer and same-session independent timing test. No parameter
or class/count routing should be fitted to current outcomes.

All64 developed windows/five pools and13 local/remote tests pass source/execution
verification; scores precede masks, old scores/confusions/survivor replay and
class-mean exact projected recovery verify. Mass error<=2.843e-14, pre-protection
intersection-log error<=8.882e-16. All jobs complete and GPU0-7 idle. Preserve all
schemes, original full hard/frozen20 projected and exact graph acceleration.
Report/raw: RIVAL_RESPONSIBILITY_INTERSECTION_20261005.md and its matching raw
directory. No standalone latency/full20092/control promotion. Goal still open.

## Frozen Fine-Only Source: Own Mechanism Audit

Completed64 developed windows/eight domains/five pools with original Geometry20,
all observers, original hard/canonical/survivor mass and T0 projection/posterior/H.
The existing fine-only formula is frozen, not refitted. Three own alias-identity
nulls, valid-query position shuffle, wide-only responsibility, within-class source,
factorized source and uniform hard recovery are tested. Original three endpoints
and previous fine-only scores/confusions replay exactly; mass/spectra/singleton
checks pass. Ten local/remote tests pass; all jobs complete and GPU0-7 idle.

The fine-only means remain46.0243/46.1437/45.1318/44.7752/46.1762 for20/30/40/
wrong-parent/paraphrase, all above projected hard. Own identity advantages are
0.4019pp at20 and0.0389pp wrong-parent; own position-null advantages0.4870/0.3252pp.
This supports word/location-specific information in these developed windows.

But rival specificity is not established: within-class without a rival denominator
scores46.0240/46.1436/45.1363/44.7792/46.1861. Every clean20 protocol differs by
<0.009pp. Factorized responsibility matches within-class mIoU in every regime;
maximum allocated difference1.421e-14, all eligible floor fractions0. The common
class-pair factor cancels in actual data. Primary's covariance changes allocations
but delivers almost no extra accuracy through the fixed final readout. Do not call
the current mean improvement a verified new competition-specific reliability source.

The frozen gate FAILS: worst LoveDA D paraphrase-1.6943pp; own wrong-parent damage
1.2491pp versus projected0.8218pp; primary does not beat within/factorized at both
20 and wrong-parent. Wide-only wins40/paraphrase. Potsdam car25.1793->27.5389 improves
while low-vegetation21.7324->19.7584 loses recall.40 still trails its own20 by0.8925pp.
No post-result coefficient/class/domain routing, control promotion or full rollout.

Preserve the established full hard, projected20 transfer and equivalent execution
backends, plus this soft development lead and all unsuccessful schemes. A genuine
next source must distinguish alias competition utility from confident/shared response;
merely inserting another common denominator is insufficient. Reuse-only local/wide
features are a distinct speed/accuracy path and cannot inherit fine-only evidence.
No isolated timing is claimed. Report/raw: FINE_RESPONSIBILITY_AUDIT_20261005.md and
research/fine_responsibility_audit_20261005/. Research goal stays active for user choice.

## Geometry-Referenced Conditional Alias Density Update

Completed a distinct response-value-conditioned source on the same64 developed
windows/five regimes, preserving original Geometry20, observers, hard/canonical/
survivor support/mass, T0/posterior/H. A leave-query-out Laplace density references
each scalar fine response to Geometry20 soft class posteriors, balanced by class
mass. Primary multiplies frozen fine-only weights by own-vs-rival conditional
density responsibility. Original beta/epsilon, no fitted parameter or extra visual
forward. Exact sorted prefix/suffix kernel sums avoid NxNxalias storage and match
dense sums, including ties/extremes. Ten local/remote tests pass. Old/fine-only
scores/confusions, uniform recovery, own null spectra, mass/singleton, frozen
identities and64 unique keys verify; predictions persist before masks.

Fine-only->primary increments are +0.0005/+0.0166/+0.0155/+0.0018/-0.0193pp for
20/30/40/wrong-parent/paraphrase. The frozen gate FAILS: no0.1pp clean increment,
worst vs projected LoveDA D paraphrase-1.6857pp. Density-only20 is45.4617, far below
fine-only46.0243. Joint identity-null gaps largely inherit fine-only identity and
cannot isolate the new source. Reference/position nulls preserving fine-only have
20 gaps only0.0011/0.0013pp. No accuracy benefit established for query exclusion
or class balance on both20/wrong-parent. Availability is1, source means near1/2,
raw rival range max0.29733: numerical missing support is not the main explanation.

Preserve this source as a negative result, not another final-model module. Do not
fit bandwidth from these targets or route by class/domain. Keep all earlier schemes,
established full hard, projected20 transfer, fine-only development lead and exact
acceleration. Next characterize the frozen fine-only lead on complete inputs and
independently matched cost; its robustness failure remains, with no automatic final
promotion or full20092 launch. All workers completed and GPUs idle. Report/raw:
RIVAL_CONDITIONAL_DENSITY_20261005.md and research/rival_conditional_density_20261005/.
Goal remains active for the user's eventual choice; no new automation or speed claim.

## Frozen Fine-Only Complete-Input And Matched-Cost Update

The frozen fine-only source has completed152 unique complete inputs: full UDD5=40,
plus the original16-image transfer manifest in each other domain. Keys/manifests,
vocabularies/checkpoints, original Geometry20, observer/risk/projection/posterior/H
and model weights are unchanged. All four historical complete per-image controls
match exactly. Ten reader tests already passed locally/remotely; actual requested
endpoint and all-arm scores also match bitwise. These are exploratory previously
examined complete inputs, not untouched validation or full20092 results.

Eight-domain means (LoveDA D once): no admission47.3874, original hard48.8427,
projected49.3271, fine-only49.8854. Fine-only gains0.5583pp over projected with6/8
main-domain wins. Seven non-UDD5 gain0.5897pp,5/7 wins, own identity-null advantage
0.3779pp, empirical paired bootstrap95% interval[0.3493,0.8161]pp. The fixed20
characterization gate passes, but the previous wrong-parent/paraphrase robustness
failure remains. Keep this as a candidate, not a universal selector or automatic
final-model promotion.

Within-class49.8886 is effectively identical to primary49.8854. The additional
rival denominator is still not an established contribution. Wide-only49.9021 is
a source control and still needs fine RGB for frozen hard support/projection.
Soft means survivor redistribution AFTER original conditional hard rejection;
it does not establish removal of screening or the four extra observations.

Potsdam car/low vegetation improve2.2822/2.8422pp; UDD5 road loses2.6941pp despite
overall+0.3389pp. VDD vehicle+1.5120pp is offset by other/road losses, total+0.0116pp.
OEM rangeland loses4.1489pp; LandCover.ai/FLAIR-1 overall lose0.1277/0.1668pp.
Do not infer every alias useful, all classes improved or VIP distillation invalid.

Seven warmed alternating independent repeats/domain: cached eager hard0.300743s,
graph hard0.273154s, graph projected0.276175s, graph fine-only0.283774s. Graph hard
reduces latency9.17% versus equally cached eager; fine-only adds2.75% versus matched
projected, with unchanged measured co-resident allocated peaks. Scope is one512
window+full-image wide context, not whole-image throughput. Four fine forwards
remain; do not compound cross-session speedups. All schemes/results remain; all
workers finished and GPUs0-7 idle. No full-suite rollout/new monitor follows.

Report/raw: FINE_RESPONSIBILITY_COMPLETE_20261005.md and
research/fine_responsibility_complete_20261005/. The research goal remains active.
