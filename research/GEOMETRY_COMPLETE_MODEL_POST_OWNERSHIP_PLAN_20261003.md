# Complete Geometry Model: Decision After Ownership Experiments

Follow-up coherent admission is now complete, with positive multiclass gains
but a failed alias-specificity gate. Current planning/results:
GEOMETRY_COMPLETE_MODEL_POST_COHERENT_PLAN_20261003.md and
COHERENT_NATIVE_ADMISSION_RESULTS_20261003.md. This historical ownership result
and its failed gate remain unchanged; no final controller is promoted.

## Outcome

The planned source-feasibility and actual-prediction experiments are complete.
The native family-excluded class reference passes its mask-free source gate on
8/8 developed domains, but its fixed prediction gate FAILS. It is not a promoted
alias controller, and no complete-image/full-dataset rollout follows this gate.
The useful complete alias-controller/CVPR research objective remains open.

Same64 developed top-left512 windows, eight/domain, exact20 aliases/class. Only
contextual vocabulary changes under stress; local original20 Geometry remains
fixed. Weights are frozen. Masks enter only after prediction/source persistence;
no target-label numerical fitting, seed selection or dataset routing. LoveDA D
enters the equal-domain mean once; P is separate. LandCover.ai replaces unlabeled
iSAID. These are not full-dataset scores or untouched validation.

| Regime | Original coupling | Prior view soft | New joint ownership soft | Ownership-only | New same-source hard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical20 |42.867949|43.241696|43.226730|42.858905|43.599125|
| Four wrong-parent replacements |41.989160|42.251500|42.241704|41.997862|42.752057|
| Four legitimate paraphrases |43.646857|43.955612|43.931355|43.627668|43.967904|

New primary improves original coupling by0.358781pp clean and0.252544pp
wrong-parent, with5/8 wrong-parent domain wins and worst clean/paraphrase change
-0.051002pp. It nevertheless trails the preceding soft, preceding hard,
two preceding random-deletion controls and all three new alias-risk shuffles
on the wrong-parent mean. Source separation is not prediction specificity.
All17 historical numerical/per-image endpoints replay exactly. Eleven targeted
tests pass locally/remotely; mask-free checkpoint smoke and complete unique
coverage/confusion/transition reconstruction pass.

## What Was Learned

The source asks two different questions: does a phrase visually match this query,
and does it support its assigned class rather than a rival? The second question
adds real attachment information in this constructed audit: foreign mean risk
is0.0603-0.1623 versus legitimate-paraphrase0-0.0017; it captures42.68-68.88%
of the view-blind comparisons. These are query/alias comparisons, not harmful
pixel percentages or proof of real LLM vocabulary robustness.

However, ownership-only mean changes are just-0.009044pp clean and+0.008702pp
wrong-parent. This is NOT because the source performs no action. Wrong-parent
UDD5 mean directed suppression rises from0.01843947 to0.03623900 after union;
its primary mIoU instead falls27.616556->27.532093. These are valid-query,
class/rival averages, not semantic-correctness probabilities. A class-reference
check can recognize wrong attachment without supplying a useful final decision.

Holdout adds only+0.000275pp clean and+0.012458pp wrong-parent compared with
the same joint rule without family exclusion. Exact mathematical independence
is verified, but those small effects do not establish a strong new contribution.
Other correlated synonyms can still confirm each other; the native class
reference is itself a frozen-model prediction, not an independent truth source.

On clean windows VDD vehicle IoU50.3084->47.7857, despite its total mIoU gain.
Potsdam car24.5378->24.5704, precision24.6242%, recall99.1183%; low vegetation
recall is still15.5410%. UDD5 road IoU remains0. These observations do not show
that the new controller solves small-object precision or missed-class coverage.
They must not be generalized to full-dataset values.

A correctly attached 'car' phrase can fire on pavement with no text-attachment
conflict. The new source then supplies no rejection through that pathway.
Canonical words are additionally protected. Neither near-text family removal
nor coarse/fine agreement can generally certify those visual false positives.
The experiment does not prove all training-free semantic control impossible;
it rejects this specified controller and its promotion claim.

## Retained Complete Model

Keep the current executable model unchanged:

```text
image + declared ontology/aliases
                |
       frozen DINOv3 + DINO.text
                |
       +--------+-----------------------------+
       |                                      |
1. Geometry local reader               attributed VIP wide observer
fine descriptor Yg                     contextual alias field
local scores g                         original templates/profile scoring
patch relation G                       broad class observation b
       +------------------+-------------------+
                          |
2. local-fidelity Geometry reconstruction
   z = g + (I + G^T G)^(-1) G^T G (b-g)
                          |
              original dense tile assembly
```

Geometry uses normalized frozen backbone patch similarity and a positional
penalty to form the conditional patch relation. It replaces patch-to-patch
reading in the frozen head while preserving native prefix interactions/mass,
head transforms and the declared two-block depth. It produces descriptors and
relations, not ground-truth class ownership. Source: tcpr.py/parallel_readout.py.

The local20 readout is normalized log-mean-exp. Its equal prior over words does
not imply equal pixelwise responsibility: response-dependent responsibilities
already exist. The wide observer additionally profiles words by crop salience.
Neither dynamic response weighting is a semantic-reliability estimator.

The reconstruction is the inference objective

```text
min_z 0.5 ||z-g||_F^2 + 0.5 ||G(z-b)||_F^2.
```

There is no supervised training loss, network optimizer or target-label loss.
This is classical least-squares reconstruction, not a newly invented solver.
The VIP observation path is attributed, not renamed as our visual innovation.
The retained model has NO promoted semantic alias controller.

## Target Final Architecture

Keep two mechanisms, not a separate sequence of text filter, calibration,
detector, segmentation model and confidence gates. Integrate any successful
alias admission inside the second mechanism:

```text
Geometry -> local descriptor, relation and fidelity anchor
wide observation + physically fine witness + ontology
        -> query x alias x rival evidence admission
        -> accountable contextual class field b*
Geometry + b* -> local-fidelity reconstruction -> prediction
```

The desired contract is more specific than 'select good words': at query i,
does alias a provide discriminative evidence for class c AGAINST rival d?
Shared meaning, weak support and missing coverage are unknown, not automatically
bad. The same phrase may help against one rival and remain neutral against
another. Do not force15/20, apply a global blacklist or treat high response as
correctness. Preserve useful context and original local detail.

For the executed experimental interface, gamma(i,a,d) is a bounded risk. The
unchanged excess writer removes only the response excess above the uniform
within-class share, forms coherent class increments and admits b*=b+v:

```text
q_a = softmax(beta * profiled_alias_evidence)_a
delta(c|d) = log(1 - sum_a gamma(i,a,d) max(q_a-1/K,0)) / beta
e(c,d) = delta(c|d) - delta(d|c)
v_c = mean_d e(c,d)
z = g + (I + G^T G)^(-1) G^T G (b+v-g).
```

These equations describe the tested failed candidate, NOT a new production
default. Positive v is relative competition after rival suppression, not proof
that a beneficial phrase was recognized and positively boosted. Zero risk
recovers original coupling. Missing source evidence must remain neutral.
The fixed20 implementation has not demonstrated arbitrary-size vocabulary
admission, and the context-only controller has not cleansed local aliases.

The next successful admission source must address correctly attached visual
false activation as well as attachment errors. Current sources do not supply
that certificate. Its formula is not settled by this report: writing another
confidence product without a useful observation would repeat the same failure.
Do not conceal this remaining design gap by naming the failed source a final
module or using a winning control as the new method.

## Next Decision Sequence

1. Preserve original Geometry/coupling and freeze this negative result. Do not
   tune writer magnitude, family cosine, hard/soft mixture or per-class routing
   from these target results. Earlier SharedFamily/leave-out/writer failures
   remain failures rather than new names for the same mechanism.
2. Specify a NEW complete evidence-admission candidate only when its native
   observation can distinguish true object/class evidence from shared surface,
   scene association and wrong attachment. The acceptance question must be
   useful competing-class prediction, not construction-name classification.
   Require explicit unknown coverage and actual contribution-level measurement.
3. Use one predeclared whole-candidate comparison with original coupling,
   same-source fusion, unmodified admission and matched shuffled evidence/action
   controls. Test attachment stress and correctly attached visual false positives
   separately. No more source-only ranking is enough to promote a reader.
4. After prediction promotion, freeze the full-image implementation and all
   inputs once. Run the eight complete labeled protocols and fair complete
   official VIP, including LoveDA P/D/foreground, UDD5/LandCover.ai foreground,
   per-class precision/recall/area, object-size errors and scene-level uncertainty.
   Do not compare current window scores with historical full-image VIP scores.
5. Validate on real provider-provenance raw LLM aliases and a held-out vocabulary
   or region not inspected during development. Existing historical LLM-style
   banks are not verified raw outputs; LoveDA style is identity. A model can
   improve noisy vocabularies without deleting useful curated words everywhere.
6. Measure standalone latency/memory for all witness observations, not the
   shared33-arm evaluator. If claiming general OVSS, add natural images under
   a common declared template/readout protocol. Replacing RS with ImageNet
   templates is not itself evidence of natural-image generalization.

This sequence is a plan, not a claim that an unimplemented successor will win.
No new full run was launched after the failed prediction gate.

## CVPR Claim Boundary

The defensible target is Geometry-guided control of contextual semantic
contributions with local fidelity. Geometry's distinctiveness still requires
matched nearest-method evidence and attribution. A second contribution needs
useful alias/query/rival allocation beyond shuffled controls, not merely a
dynamic-weight formula. Classical aggregation/reconstruction and VIP observation
are reused components. Current results do not establish all-eight SOTA, a useful
universal alias selector or CVPR readiness. Removing words is not a publication
requirement; demonstrated accountable use of semantic evidence is the objective.

## Artifacts And Cost

Source report: NATIVE_CLASS_OWNERSHIP_FEASIBILITY_20261003.md. Prediction report:
NATIVE_CLASS_OWNERSHIP_PREDICTION_20261003.md. Verified results/per-image matrices:
native_class_ownership_prediction_r2_20261003/. Source166.0633s, shared predictor
suite457.7261s, combined peak5596.97-5940.35MiB. Neither is standalone latency.
The first prediction smoke failed before mask loading because of an FP32/FP64
joint-risk replay discrepancy; its output/log remain preserved. The r2 repair
reconstructs the same FP64 source before the final cast, without changing the
source equation, gate, words or inference objective.

Code: dinotool/native_class_ownership.py, native_ownership_reader.py;
scripts/run_native_ownership_feasibility.py, eval_native_ownership_reader.py,
run_native_ownership_reader.py; tools/native_ownership_feasibility.py and
native_ownership_reader_experiment.py. Older models and outputs are unchanged.

## Follow-Up Action-Path Evidence

The additional fixed diagnostic is now COMPLETE, not an unexecuted plan.
It reuses all64 windows/three regimes, zero new image/text network forwards,
CPU-only55.6766s. Five mathematical tests pass locally/remotely. Derived fields
precede audit masks; exact old/new class-score and per-image confusion replay,
orthogonal decomposition and fixed coverage pass. No predictor or numerical
parameter changed. Full report: OWNERSHIP_ACTION_PATH_AUDIT_20261003.md.

This separates the ADDED ownership action from prior view-soft, rather than
claiming its whole gain over original coupling. Wrong-parent incremental mIoU:

| Domain/protocol | Sampled admitted broad field, before writeback | Coupled output |
| --- | ---: | ---: |
| VDD |+1.817169|+0.286862|
| Potsdam |+0.048220|-0.231440|
| UDD5 |+0.040814|-0.024257|
| OEM |+0.515265|+0.014442|
| LoveDA P |-0.347234|-0.522873|
| LoveDA D |-0.350619|-0.055947|
| Vaihingen |+0.353531|-0.055408|
| LandCover.ai |+1.326339|+0.097939|
| FLAIR-1 |-0.101007|-0.110558|

Before-writeback fields are sampled query fields, not full broad-view predictions
or final-model scores. Their old starting fields differ from old coupled fields.
These deltas do not justify removing Geometry or selecting a stage per domain.

The evidence rules out a single universal explanation. LoveDA/FLAIR-1 correction
is already harmful before writeback; it is not solely spoiled by Geometry.
Conversely, VDD/OEM/LandCover.ai lose much of the incremental mIoU gain after
writeback, so a source-only attachment ranking is insufficient for a coupled
reader. Potsdam's prewrite5662 beneficial/7121 harmful changes show that its
small positive mIoU delta does not certify a clean correction signal.

UDD5 coupled changes gain1246 correct pixels and lose6, yet mIoU falls because
class-balanced competition also changes;8752 changes are wrong-to-wrong. Do not
optimize or gate a selector only by net correct-pixel count. Per-class IoU and
FP redistribution must remain outcomes, not target-fitted control inputs.

Only27.77-54.12% of wrong-parent added antisymmetric edge energy survives the
class-potential projection. The residual is a non-integrable cycle component,
not known useful semantic information. Directly increasing its weight or
swapping solvers cannot be justified by this energy diagnostic.

The resulting next architecture requirement is stronger: formulate contextual
admission as ONE coherent multiclass semantic observation before Geometry
writeback, with explicit unknown evidence and no forced deletion quota. A
query/alias/rival test can be an internal witness, but incompatible requests
must not be sold as independently realizable class corrections. That coherent
observation must itself improve actual disputed visual decisions and then retain
its usefulness under the unchanged local-fidelity reconstruction. The present
audit does not settle a successful source formula; another arbitrary confidence
product, retrospective class switch or renamed least-squares solver would not
fill that gap. Original Geometry/coupling remain retained, and the complete
useful-controller/CVPR objective is still active.
