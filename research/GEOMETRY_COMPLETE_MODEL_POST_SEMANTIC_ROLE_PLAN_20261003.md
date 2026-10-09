# Complete Model After Frozen Semantic-Role Admission

## Successor Status

The proposed semantic-admissible native reference experiment below is now
implemented and COMPLETE. Its frozen gate FAILS. Wrong-parent reference-only
42.887448 adds0.032030pp over the old base; complete42.871833 adds0.016415pp,
with attachment subtracting0.015615pp. No controller or full rollout is promoted.
Current planning: `GEOMETRY_COMPLETE_MODEL_POST_REFERENCE_PLAN_20261003.md`.
Verified successor results: `SEMANTIC_REFERENCE_ADMISSION_RESULTS_20261003.md`.
The prospective section below records its prior motivation, not current pending
work. Original models, this experiment and all earlier failed gates stay intact.

## Decision

The requested source replacement and actual prediction experiment are complete.
Signature `geometry-frozen-semantic-role-admission-v1-20261003`. The prospective
promotion gate FAILS. Preserve original Geometry/coupling and every prior model;
do not promote a control, tune rejection strength or start a full rollout.
The useful alias-controller/CVPR goal remains open. This is a complete-model
planning decision, not a claim that a final universal controller is ready.

Frozen checkpoints and20 aliases/class; same64 development top-left512 windows,
eight/domain; LoveDA D counts once in means, P separately. LandCover.ai replaces
unlabeled iSAID. Historical style vocabularies are not verified provider raw
outputs. Only the context bank is perturbed; original local20 stays fixed.
These are not full-image/full-eight results or matched official-VIP comparisons.
No numerical target-label fitting occurred, but prior development results
motivated the design; do not call these untouched independent validations.

## What Was Implemented

Reuse the already staged frozen Qwen2.5-VL-7B-Instruct, author revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, for vocabulary-time TEXT-ONLY role
judgment. Declared class names and words are the only semantic inputs. Forward
and reverse option orders distinguish direct meaning, shared material, part,
scene co-occurrence, unknown, and exclusive competitor denotation. A rejection
requires both orders to name the same competitor. Its strength is their minimum
restricted option score, not a calibrated correctness probability.

All source contracts and five sentinels pass on eight workers. Seven clean banks
have no admitted competitor conflicts; FLAIR-1 has2/240. The source itself can
be wrong: it assigns building's `agricultural silo` to agricultural land,
apparently confusing an adjective/scene with object denotation. Sentinel success
is not vocabulary-wide semantic accuracy. No prompt was retuned after outcomes.

The joint model unions this global word-membership risk with the original
class/view base, then uses the unchanged excess-contribution writer and Geometry
reconstruction. It adds no per-image language forward. All61 previous score
and per-image endpoints replay exactly. Six unit tests and mask-free actual
checkpoint smoke pass;64 unique coverage, matrices and transitions verify.

## Verified Outcome

| Regime | Original coupling | Class/view base | New semantic joint | Word increment over base |
| --- | ---: | ---: | ---: | ---: |
| Historical20 |42.867949|44.091831|44.088403|-0.003429|
| Wrong-parent |41.989160|42.855418|42.589952|-0.265466|
| Legitimate paraphrase |43.646857|44.657375|44.654522|-0.002853|

New joint beats original on6/8 wrong-parent domains, but improves its class/view
base only on Potsdam and FLAIR-1. Semantic-only wrong-parent41.509023 is below
original41.989160. Binary semantic excess-risk42.243275 is worse than soft
42.589952; this is not full vocabulary deletion. Soft being less harmful than
binary rejection does not establish useful word control.

Visual-supported semantic control42.863682 exceeds the new joint but adds only
0.008264pp over class/view base. Earlier coherent42.934464 and class/attachment
42.895130 also exceed it. Semantic alias shuffles1/2 score42.661983/42.610170,
above primary42.589952. No specificity or final-selector claim follows.

| Wrong-parent domain/protocol | Base | New joint | Extra word effect pp |
| --- | ---: | ---: | ---: |
| VDD |49.718132|49.628514|-0.089618|
| Potsdam |36.118895|37.283986|+1.165091|
| UDD5 |28.317171|27.301082|-1.016089|
| OEM |40.711109|40.619348|-0.091761|
| LoveDA P |50.287711|49.742447|-0.545264|
| LoveDA D |31.569316|31.289952|-0.279364|
| Vaihingen |54.201623|52.691449|-1.510174|
| LandCover.ai |68.758172|68.185930|-0.572242|
| FLAIR-1 |33.448926|33.719356|+0.270430|

Examples versus the SAME visual base, not versus a different Geometry protocol:

- UDD5 vegetation TP-3978/FP-1099; vehicle TP0/FP+6659. Removing some false
  vegetation occupation did not preserve correct coverage or car competition.
- VDD wall TP-474/FP-4522, while road TP+39/FP+4369 and vehicle FP+265.
  A local class precision benefit can displace false occupation into competitors.
- Potsdam tree TP+35482/FP+1510; low vegetation TP-16479/FP-57292. Tree gains
  coexist with lost low-vegetation coverage, not a universal suppression benefit.
- Vaihingen low vegetation TP-40585/FP-29801; car TP+20/FP+16269. Reduced area
  in one class can increase another class's false detections substantially.

## What The Evidence Changes

1. Meaning membership and pixel contribution utility are different quantities.
   Even exact identification of a constructed wrong-parent phrase would not
   prove that globally suppressing its observed response improves class margins.
2. Most curated20 words are not explicitly contradictory according to this
   source. This does not prove every word useful. It does mean that a forced
   deletion quota or a claimed clean-bank semantic-filter gain is unsupported.
3. Good-word gains over original belong mostly to visual class/view control.
   Do not relabel its1.22pp gain as word-selection gain.
4. A global negative class offset changes all rivals' argmax competition and
   can sacrifice true coverage. Word count, class normalization, original
   salience and local/broad units matter as well as phrase identity.
5. The current full-bank native reference also includes the perturbed words.
   It may be contaminated by the same attachment error the context writer is
   asked to reject. Reference contamination is a plausible next hypothesis,
   NOT a cause isolated by this experiment. The same caution applies to the
   relative contributions of source errors, class shifts and writeback.

## Complete Architecture Target: Two Mechanisms

    image + declared ontology + candidate vocabulary
                         |
                 frozen DINOv3/DINO.text
                         |
    1. Geometry local reader: fine descriptors, local anchor g, relation G
                         |
    2. Accountable contextual reading and reconstruction
       wide observer b, with explicit VIP attribution
       physically fine competing-class reference
       semantic membership constrains who may claim reference evidence
       image-conditioned competition determines accepted contributions
       Geometry writes one coherent contextual innovation, anchored at g
                         |
                original dense assembly/prediction

Semantic membership is a constraint INSIDE mechanism2, not an independent
global TopK filter, extra detector or per-image language branch. Unknown/shared
meaning remains neutral. A correct car name firing on road is a visual-ownership
problem; it cannot be fixed by word denotation alone. Good aliases retain their
dynamic original responsibility; no tested source yet justifies positive
amplification of an alias merely because a language model approves its meaning.

Retained deployment still uses b_star=b. Experimental admission uses:

    q_a = softmax(beta*r)_a
    delta_c = log(1-sum_a R(i,a)*max(q_a-1/K,0))/beta
    b_star = b + delta
    min_z 0.5||z-g||^2 + 0.5||G(z-b_star)||^2
    z = g + H_G(b_star-g), H_G=(I+G^T G)^(-1)G^T G.

The responsibility floor bounds attenuation by log(K)/beta. Zero risk recovers
original coupling exactly. Dynamic contribution is not semantic truth. The
least-squares objective/solver is classical and VIP's observer is borrowed;
neither is renamed as a new visual invention. Geometry's distinct mechanism
must be demonstrated against the closest matched readouts independently.

## Next Single Complete Candidate, Not A Module Sweep

Test **semantic-admissible native references**, rather than making a globally
wrong-looking word trigger unconditional class suppression. Keep this semantic
source/prompt fixed: do not strengthen or manually correct it from these labels.

For each alias a, define vocabulary-time admissibility w_a=1-max_d S(a,d).
Direct/shared/part/scene/unknown assignments give w1. In each existing native
crop, use the original profiled evidence r and its existing beta, but compute:

    F_ref(c) = log(sum_a w_a*exp(beta*r_a))/beta
               - log(sum_a w_a)/beta.

Apply the original physical crop stencils to this field. With w1 this recovers
the original full-bank native reference. A class with zero supported mass is
UNKNOWN, not negative evidence. Use the corresponding held-family weighted
reference for word attachment, with the same no-self-confirmation contract.
Do not refit salience, temperature, background threshold or class area priors.

Then use that reference in the already specified image-conditioned class/view
and rival-supported attachment rule; keep the coherent excess writer and
Geometry reconstruction. No unconditional semantic penalty in the primary.
This is one replacement of the evidence source inside the complete model, not
new branch stacking or promotion of the visual-supported winning control.

Its prospective test must compare original coupling, the unchanged class/view
base, current failed semantic joint, reference-only change, complete candidate,
and matched semantic assignment shuffles. Require actual incremental pixel
benefit and retained correct coverage, not semantic sentinel accuracy or AUC.
Specify executable gate and freeze all details BEFORE its target metrics.
This proposal was not implemented by the semantic-role experiment itself; its
separate completed successor and failed decision are recorded at the top.

## Paper And Rollout Requirements

Only after a useful-controller gate passes: freeze full-image coordinates,
bounded native-crop computation and ONE global rule, then run full eight labeled
protocols with original official VIP and closest matched methods. Report per-class
precision/recall, small-object misses, large-class coverage, LoveDA P/D and
foreground protocols. Measure standalone deployed cost, not the69-arm suite.

Use provenance-recorded raw-LLM candidates, legitimate ambiguity and misplaced
word stress separately. Apply perturbations to all relevant model inputs in
independent validation; fixed clean local20 plus noisy context is only a
context-robustness test. Include VIP with the same semantic source to distinguish
generic language filtering from Geometry-specific coupling. Include natural
images/common declared templates if claiming general OVSS, and independent
regions/vocabularies not used in these repeated development iterations.

Do not promise all-eight SOTA, claim ordinary word-role prompting as CVPR novelty,
or force three contributions because a narrative needs them. Current results
support the first Geometry investigation and useful contextual reconstruction;
a specific, transferable alias-controller contribution remains unproved.

## Deliverables And Cost

Protocol: SEMANTIC_ROLE_ADMISSION_PROTOCOL_20261003.md.
Results: SEMANTIC_ROLE_ADMISSION_RESULTS_20261003.md.
Raw role observations, matrices, source decisions and exact derived extra effects:
semantic_role_admission_20261003/.

Vocabulary-source suite40.5705s parallel; source worker16.3116-32.9749s including
loading, peak16558.5791-16963.6230MiB. Prediction suite704.3076s, combined
multi-arm peak5601.1846-5957.2285MiB. These are NOT standalone deployment costs.
All workers/controllers are terminal and A800 GPUs0-7 are idle; no automation
was created and unrelated server jobs were preserved.
