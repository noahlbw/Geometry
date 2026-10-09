# Complete Model Planning After Coherent Admission

The subsequent mask-free dependency audit and role-separated prediction study
are now complete. Current decision: GEOMETRY_COMPLETE_MODEL_POST_ATTACHMENT_PLAN_20261003.md.
The coherent gate below remains failed; no old control or source is promoted.

## Experimental Decision

The requested coherent multiclass candidate is IMPLEMENTED and its actual
prediction experiment is COMPLETE on A800 GPUs0-7. It improves original coupling
on all8 developed domains in clean and wrong-parent regimes, but FAILS the
prospective alias-specificity gate. Do not deploy it as a verified final alias
selector or start full rollout under this failed protocol. Preserve this useful
performance candidate and all earlier models/results. The complete useful
alias-controller/CVPR objective remains open; this report does not complete it.

Fixed64 developed top-left512 windows, eight/domain; NOT full images/datasets,
untouched validation, or a matched official-VIP comparison. Frozen weights,
fixed20 aliases/class, original Geometry/local20/operator. Target masks are
loaded only after prediction persistence; no numerical target-label fitting.
The rule was motivated by previous labelled development audits, not independently
designed on an untouched benchmark. LandCover.ai replaces unlabeled iSAID.

| Condition | Original coupling | Prior view-soft | New coherent candidate | Gain over original pp |
| --- | ---: | ---: | ---: | ---: |
| Historical20 |42.867949|43.241696|44.161110|+1.293161|
| Four wrong-parent replacements |41.989160|42.251500|42.934464|+0.945304|
| Four legitimate paraphrases |43.646857|43.955612|44.631331|+0.984474|

LoveDA D counts once in the equal-domain mean; P is separate. Worst protocol
change over all clean/paraphrase endpoints is+0.003870pp, not a loss. All33
historical numerical/per-image endpoints replay exactly, all64 sample windows
and their matrices/transitions verify. Eight new unit tests plus16 retained
mathematical tests pass locally; the8 new tests pass remotely. Actual checkpoint
smoke loads no masks. Suite563.3034s; combined allocated peak5598.96-5948.32MiB
includes all50 shared arms/regimes, not standalone candidate latency. All
experiment sessions are terminal; unrelated tmux sessions remain untouched.

| Domain/protocol | Original clean | Candidate clean | Gain pp |
| --- | ---: | ---: | ---: |
| VDD |53.746954|55.927447|+2.180493|
| Potsdam |38.920258|39.525433|+0.605175|
| UDD5 |28.175811|30.794864|+2.619054|
| OEM |39.023152|39.353115|+0.329962|
| LoveDA P |50.706552|53.281856|+2.575303|
| LoveDA D |30.736030|34.135199|+3.399169|
| Vaihingen |51.826962|52.865911|+1.038950|
| LandCover.ai |66.906020|66.949114|+0.043094|
| FLAIR-1 |33.608404|33.737800|+0.129396|

## What Actually Changed

For each alias attached to class c, its held-out native class field asks whether
a known rival beats c at the same physical query. Unlike preceding ownership,
this does not require a text-attachment conflict and does not protect canonical
words in the primary. Missing references yield zero risk, not rejection.

    gamma_native(i,a) = 1 - exp(min(0, r_minus_family(i,c) - max_known_rival r_minus_family(i,d)))
    gamma = gamma_view + (1-gamma_view)*gamma_native

At each exact wide-crop contribution, let q be the original within-class alias
softmax, K the original count, and beta1. The writer removes only excess above
the uniform floor:

    q_used(a) = q(a) - gamma(a)*max(q(a)-1/K,0)
    delta_c = log(sum_a q_used(a))/beta
    b_star = b + delta
    z = g + H_G*(b_star-g).

The equivalent multiplicative contribution weight is

    w_a = 1 - gamma_a*max(q_a-1/K,0)/q_a.

Thus dynamic alias weights are EXECUTABLE, not a fixed equal-weight vocabulary
or hard deletion. Low-response shares are preserved; active high shares can be
attenuated. However, gamma is observed disagreement, NOT a validated good/bad
word probability. No positive word is explicitly rewarded: relative class gains
can arise from rival suppression. Word weights act in the context branch;
the original local20 anchor is unchanged. Claims about arbitrary vocabulary size
or cleaning local aliases are not established by fixed20 context-only tests.

The new writer directly supplies one class vector, so its class margins are
coherent before Geometry writeback. The previous pair-cycle reconciliation is
absent. This is an implemented candidate, not a claim of a novel log aggregation
operator or an Astra-endorsed exact formula.

## The Remaining Failure Is Specific

| Wrong-parent control | Mean mIoU |
| --- | ---: |
| Candidate |42.934464|
| No family holdout |42.855418|
| Native only |42.882385|
| Coherent view only |42.244447|
| Same-source mean fusion |42.121261|
| Class-mean action |41.476822|
| Reference shuffles0/1/2 |41.867648 /41.767648 /41.775658|
| Spatial action shuffles0/1/2 |41.515747 /41.381101 /41.463908|
| Within-class alias-risk shuffles0/1/2 |43.071997 /43.504088 /43.082810|

The rule exceeds all listed controls EXCEPT every alias-risk shuffle. Their
mean43.219631 exceeds primary by0.285167pp. Clean shuffle mean44.155055 is
almost primary44.161110; paraphrase mean44.611249 is also close44.631331.
No post-result shuffled seed is selected as a new model.

This supports investigation of spatially assigned multiclass observations,
not a claim that per-word reliability has been identified. Reference/action
allocation matters here; which member of a class receives rejection is not yet
usefully resolved. These controls are not identical-strength causal tests:
alias shuffles preserve risk spectra but can change total rejected contribution;
spatial action shuffles preserve per-class action budgets/spectra but can exceed
the source's local capacities. Those asymmetries are reported, not hidden.

Family exclusion contributes only+0.069279pp clean and+0.079046pp wrong-parent;
it is-0.026044pp on paraphrases. A family-excluded class score is not an alias's
marginal correctness: removing an irreplaceable useful family can itself reduce
own-class support and increase its measured risk. This is a plausible failure
mechanism, not a proven attribution from the current metrics.

Small-object errors are NOT solved: clean VDD vehicle IoU50.3084->49.3852;
Potsdam car24.5378->24.7945, precision24.8239%, recall99.5239%; low vegetation
recall is16.7682%. UDD5 road IoU remains0 on these windows. LoveDA D tree and
barren IoUs fall despite total mIoU improvement. No full-dataset claims follow
from these sparse windows or net beneficial-pixel totals.

## Final Architecture To Converge On

Keep TWO mechanisms. Alias processing is an internal contribution decision of
the second mechanism, not an independently optimized blacklist:

    frozen DINOv3 + DINO.text
             |
    1. Geometry local reading
       fine descriptor Y_G, conditional relation G, local semantic anchor g
             |
    2. Accountable contextual observation and Geometry reconstruction
       wide contextual observation + physically fine class witness + ontology
       -> coherent class evidence with unknown/rejectable contributions
       -> b_star
       -> min_z 0.5||z-g||^2 + 0.5||G(z-b_star)||^2
             |
       original dense tile assembly and prediction

Geometry retains normalized backbone similarity, positional penalty, native
prefix interactions/mass and frozen head transforms, with declared two-block
reading. Do not change it while interpreting this admission experiment.
The wide observer remains the attributed VIP observation path. Reconstruction
is classical least squares with H_G=(I+G^T G)^(-1)G^T G; no supervised training
loss, learned gate, network optimizer or target-label loss is introduced.
VIP observation and the solver must not be renamed as original inventions.

The retained deployable model still uses b_star=b. The new executable candidate
uses the equations above but is NOT promoted under the failed specificity gate.
The final useful alias-specific source is therefore UNRESOLVED. The architecture
is fixed enough to focus research; the second contribution is not yet proven.

## Next Experiment Boundary

Do not continue coefficient/threshold/canonical-protection sweeps from these
labels, repeat fixed15 deletion, or select one successful domain. Separate two
questions before another full model run:

1. Can an image-conditioned CLASS observation reliably control contextual class
   excess while preserving local detail? The current result gives positive
   development evidence. A prospective study may test this narrower claim with
   a freshly frozen class-level rule, independent vocabulary/regions and full
   image protocols. The existing no-holdout control is not silently promoted.
2. Does a named alias supply positive discriminative information beyond that
   class observation? It needs an affirmative, query-owned contribution witness,
   not merely low held-out parent confidence or large activation. Useful,
   ambiguous, unsupported and wrong-parent contributions must remain distinct.
   Until such a witness is specified, do not invent another confidence product.

At fixed query/crop/class, the tested writer depends only on the total rejected
excess sum_a gamma_a*max(q_a-1/K,0). Matching that total exactly makes every
alias allocation produce the SAME class score. An exact-action-matched alias
shuffle cannot prove good-word recognition through final mIoU; it is a
mathematical identity, not a useful prediction test. Allocation must instead be
assessed with prospectively defined nuisance/legitimate-family interventions
and mask-free contribution evidence, with labels used only for independent
outcome audits. Do not use the existing wrong-parent construction names as
selector input, or current labelled per-alias audits to fit it.

That independent evidence must survive correctly attached visual false
activation as well as wrong attachment; native reference agreement alone cannot
certify correctness. A universal full-image implementation must also preserve
physical16px query correspondence and use bounded-cost native observations
across tiles, not accidentally re-encode the same overlapping crop many times.

After a NEW prospective useful-controller pass, freeze one complete-image model,
all vocabularies and config, then run full eight labelled protocols alongside
fair full official VIP and closest Geometry neighbors. Include LoveDA P/D and
D foreground, UDD5/LandCover.ai foreground, per-class IoU/precision/recall/area,
object-size errors, independent region/vocabulary validation and standalone
latency/peak memory. Verified provider-provenance raw aliases are still needed;
historical style banks are not sufficient. Natural-image validation requires a
shared declared template/protocol, not just changing to ImageNet templates.

## Paper Decision

Do not force the claim 'we screen bad aliases' because publication is desired.
The promising story is Geometry-grounded organization of contextual semantic
contributions with local fidelity; a word may be useful in one query/competition
and neutral in another. This round moves the useful multiclass control forward,
but does not establish reliable word identity allocation, independent novelty,
all-eight SOTA or CVPR readiness. Geometry's matched nearest-method evidence and
the wide-reader attribution remain necessary. If alias-specific evidence stays
indistinguishable from shuffled allocation, omit that unproven contribution
instead of presenting dynamic weights alone as semantic correctness.

## Artifacts

Prospective protocol: COHERENT_NATIVE_ADMISSION_PROTOCOL_20261003.md.
Exact tables, all50 arms, class metrics, transitions and cost:
COHERENT_NATIVE_ADMISSION_RESULTS_20261003.md.
Verified local data: coherent_native_admission_20261003/.
Code: dinotool/coherent_native_admission.py;
scripts/eval_coherent_native_admission.py, run_coherent_native_admission.py;
tests/test_coherent_native_admission.py;
tools/coherent_native_admission_experiment.py.
