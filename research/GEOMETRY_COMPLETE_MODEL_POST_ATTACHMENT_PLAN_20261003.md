# Complete Model: Decision After Class/Attachment Separation

## Current Decision

Two additional authorized steps are complete: a mask-free family-dependency
diagnostic and a new actual-prediction candidate separating class disagreement
from word attachment. This is progress toward the original useful-alias-control
goal, NOT completion. The new prediction gate FAILS; no full rollout, winning
control promotion or target-label numerical fitting follows. Preserve original
Geometry/coupling and the earlier coherent performance candidate.

Same64 developed top-left512 windows, eight/domain; NOT full datasets,
independent untouched validation, or matched official-VIP scores. Frozen weights,
fixed20 aliases/class, original local20/Geometry/operator. All50 prior numerical
and per-image endpoints replay exactly. LoveDA D enters equal-domain means once,
P separate; LandCover.ai substitutes for unlabeled iSAID. Historical style banks
are not verified provider raw outputs. Rule design used prior development
evidence, though no numerical parameters were fitted to target labels.

## Family-Dependency Evidence

The CPU-only cache audit loads no images or masks and adds no network forwards.
All64 source sequences verify. Native-risk formula replay is exact; saved GPU
versus CPU expm1 agrees within an explicit FP64 machine guard. R1's2.22e-16
replay failure and log remain preserved; R2 changed only the diagnostic check.
Four targeted tests pass locally/remotely. Audit112.3571s, no GPU allocation.

For the old held-family source, gamma=1-exp(min(0,m_minus_family)). Every observed
risk increase after family removal tracks a loss of own-class model margin.
On clean banks this affects about30-35% of query/alias comparisons, carrying
about50-58% of native family responsibility. These are not harmful-pixel counts
or proof those words are semantically correct. A wrongly attached family can
also inflate its incorrect parent. Simply reversing the omission sign is not a
valid selector. The diagnostic identifies dependence, not independent utility.

Report: NATIVE_FAMILY_DEPENDENCY_AUDIT_20261003.md.
Verified data: native_family_dependency_audit_r2_20261003/.

## Implemented Role-Separated Candidate

The full-bank native class field supplies shared class disagreement. The previous
view source remains. Extra word rejection requires BOTH an affirmative known
rival in the independently held-family reference and a text attachment conflict:

    R_class = native_risk(full native20 class field)
    R_view = max_rival(previous view risk)
    R_base = R_view + (1-R_view)*R_class
    R_attach = max_rival(positive held-family contrast * text attachment conflict)
    R = R_base + (1-R_base)*R_attach
    delta_c = log(1-sum_a R_a*max(q_a-1/K,0))/beta
    z = g + H_G*(b+delta-g).

No learned mixing strength, new threshold or dataset routing. Canonical words
retain the unprotected class/view base; their EXTRA attachment risk is zero.
Unknown reference evidence remains neutral. The bounded coherent writer and
Geometry reconstruction are unchanged. This uses tested ingredients in separate
roles, not a new independent observation or a claimed novel probability rule.

## Verified Actual Prediction Outcome

| Condition | Original coupling | Earlier coherent | Class/view base | New role-separated | New extra over base pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical20 |42.867949|44.161110|44.091831|44.082486|-0.009346|
| Wrong-parent |41.989160|42.934464|42.855418|42.895130|+0.039712|
| Legitimate paraphrases |43.646857|44.631331|44.657375|44.600696|-0.056679|

The new complete candidate gains1.214537pp clean and0.905970pp wrong-parent over
original, with8/8 wrong-parent wins and worst clean/paraphrase protocol delta
-0.061634pp. Those broad gains largely belong to the class/view mechanism, not
the extra word term. It trails preceding coherent prediction by0.078625pp clean
and0.039334pp wrong-parent. Its prospective gate stays FAILED.

| Extra-source control, wrong-parent | Mean mIoU |
| --- | ---: |
| New role-separated |42.895130|
| Class/view base |42.855418|
| Alias shuffles0/1/2 |42.874098 /42.879552 /42.878942|
| Reference shuffles0/1/2 |42.834430 /42.841144 /42.845468|
| Unconditional text conflict on same base |43.528885|

It now exceeds every EXTRA-alias and EXTRA-reference shuffle on the wrong-parent
mean, unlike the preceding whole-risk alias allocation. However, the extra gain
over class/view is just0.039712pp and the gap over alias-shuffle mean is about
0.017599pp. No independence, statistical certainty or universal useful-word
identification follows from this developed small window screen. On clean and
paraphrase banks, every extra-alias shuffle is better than the extra word term.

Unconditional text conflict is much stronger on wrong-parent noise but weaker
on legitimate banks: clean43.521285 and paraphrase43.415082, below the grounded
candidate44.082486/44.600696. It also causes clean LandCover.ai loss greater
than1pp versus original. It is not promoted as a final selector or chosen for
noisy datasets. Increasing the current word term until it resembles that control
would be label-motivated strength selection, not a new evidence source.

Five new tests pass locally/remotely. Mask-free actual-checkpoint smoke passes;
class/view base, zero-risk identity, extra-canonical identity, coverage, matrix
and transition reconstruction verify. Suite638.8893s, combined peak
5600.25-5953.48MiB includes all61 shared arms/regimes and persistence; NOT
standalone candidate latency. Every experiment session is terminal. The prior
coherent suite has50 methods, not51; its plan's count was corrected without
changing predictions, protocol, gate or scores.

## Final Architecture Remains Two Mechanisms

    image + declared classes and candidate aliases
                       |
               frozen DINOv3 + DINO.text
                       |
    1. Geometry: local descriptors Y_G, relation G and anchor g
                       |
    2. Accountable contextual observation/reconstruction
       wide observation + physically fine class evidence
       + independent semantic membership evidence for alias contributions
       -> one coherent class observation b_star
       -> local-fidelity Geometry reconstruction
                       |
               original dense tile assembly

Keep the current Geometry two-block implementation, normalized backbone patch
relations, positional penalty, prefix interactions/mass and frozen head
transforms. Attribute the current wide observer to VIP. The inference objective
is classical least squares, not a newly invented solver or supervised loss:

    min_z 0.5||z-g||^2 + 0.5||G(z-b_star)||^2
    H_G=(I+G^T G)^(-1)G^T G.

The retained deployed model still uses b_star=b. The executed coherent and
role-separated candidates are preserved research models, not promoted universal
selectors. Neither cleans the unchanged local20 bank. Their dynamic context
weights exist, but do not themselves prove word correctness or arbitrary-size
vocabulary robustness. Do not relabel VIP observation or the solver as our
visual novelty. The final useful alias source is still unresolved.

## Next Concrete Source Requirement

Stop coefficient/threshold sweeps of this text-cosine-times-pseudo-reference
source. The next source must distinguish what an alias MEANS from the visual
model's propensity to activate it. Single canonical text cosine and votes from
the remaining correlated aliases have not supplied enough semantic evidence.

The next safe implementation step is to specify and inspect an independently
frozen semantic-membership source using only declared class names/ontology and
candidate words. Possible sources are a fixed entailment/contradiction/unknown
text encoder or a provenance-recorded language-model word-role judgment. Neither
exists in this experiment yet. First check authorized project runtime/assets
and public source availability; do not add an unneeded dependency or assume
provider credentials. A public frozen text model may be a vocabulary-time
auxiliary inside mechanism2, not a new per-image detector, calibrated visual
branch or independently tuned third module.

Its contract is more specific than 'a word is close to a canonical embedding':
does this phrase directly denote a subtype/appearance of c, denote a competitor,
describe shared material, describe a part, or only co-occur in its scene? Shared,
ambiguous and unsupported descriptions must remain unknown, not automatically
bad. A roof phrase can be legitimate for a building ontology and wrong for a
wall/roof mutually exclusive ontology. Public declared class definitions are
allowed; no observed masks or labelled per-alias marginal scores may decide
these assignments. Any source probabilities are model outputs, not ground truth.

Freeze that source, prompts/provenance and one complete contribution rule BEFORE
new target metrics. Test the actual whole candidate against the retained coupling,
current coherent candidate, class/view base, semantic-only control and matched
word/reference controls under both legitimate and nuisance vocabularies. A
source-only rank/AUC pass cannot promote it. No post-result seed selection,
dataset switches, winner mixtures or rewriting old failed gates.

Semantic membership alone cannot identify a correctly named car firing on road;
that remains the visual/class contribution problem. The complete model must
retain useful contextual corrections under Geometry and must report small-object
precision/recall and missed-class coverage rather than only average gains.

After an actually useful-controller pass, freeze full-image physical-coordinate
handling, bounded-cost native observations and all inputs, then run the complete
eight labelled protocols with fair full VIP/nearest-method comparisons. Add
independent vocabulary/region validation, verified raw-LLM provenance, standalone
latency/memory and natural-image validation under a common declared template and
protocol. Nothing here establishes all-eight SOTA or CVPR readiness. The original
goal remains active, not replaced by successful class confidence alone.

## Artifacts

CLASS_ATTACHMENT_ADMISSION_PROTOCOL_20261003.md;
CLASS_ATTACHMENT_ADMISSION_RESULTS_20261003.md;
class_attachment_admission_20261003/.
Code: dinotool/class_attachment_admission.py;
scripts/eval_class_attachment_admission.py, run_class_attachment_admission.py;
tests/test_class_attachment_admission.py;
tools/class_attachment_admission_experiment.py.
