# Complete Model Decision After Semantic-Reference Admission

## Successor Status

The physical8 reference proposal below is now implemented and COMPLETE. Its
frozen alias gate FAILS despite useful reference gains: clean44.850639 and
wrong-parent43.924949; reference-only adds0.759433/1.066708pp over the old base,
while extra attachment adds only-0.000626/+0.002824pp. Current planning:
`GEOMETRY_COMPLETE_MODEL_POST_FINE_REFERENCE_PLAN_20261003.md`; verified results:
`FINE_REFERENCE_ADMISSION_RESULTS_20261003.md`. No controller/full rollout is
promoted. The prospective section below records its motivation, not pending work.

## Outcome And Retained Model

The semantic-admissible reference successor is implemented and complete.
Signature: `geometry-semantic-admissible-reference-v1-20261003`.
All eight workers are terminal, with no failures or queued work. The frozen
promotion gate FAILED. No alias controller is promoted and no full-dataset
rollout follows this screen. Preserve the original Geometry, attributed VIP
wide observation, reconstruction, and every earlier result.

This document completes the requested post-experiment model planning. It does
not claim that Astra prescribed these exact successor equations, that a useful
universal alias controller is ready, or that the CVPR research goal is achieved.

The screen uses 64 fixed development windows: eight per domain, one top-left
512 window per image. LoveDA D enters the equal-domain mean once; P is separate.
LandCover.ai substitutes for unlabeled iSAID. These are not full eight-dataset
or independent validation results. Weights remain frozen, each class has 20
aliases, and target masks did not enter the source or select numerical parameters.
Previous labeled development results motivated this design. Only context
vocabularies are perturbed; local20 stays fixed. Historical LLM-style banks are
not provenance-verified raw provider outputs.

## What The Latest Experiment Established

The new source reweights the native reference, rather than imposing an
unconditional semantic penalty on contextual class scores:

```text
w_a = 1 - max_d S(a,d)
F_ref(c) = [log sum_a w_a exp(beta*r_a) - log sum_a w_a] / beta
```

Here S is the previously frozen text-only Qwen role judgment, not semantic
truth. The original native profiled evidence r, crop coordinates, temperatures,
Geometry, local20, wide observer and writer remain unchanged. An unsupported
class reference is UNKNOWN. Extra attachment additionally requires held-family
visual rival evidence. The experiment replays 69 earlier endpoints and adds
10 arms; it is not 79 independent new model designs.

| Context-bank regime | Original coupling | Old class/view base | Reference-only | Complete candidate |
| --- | ---: | ---: | ---: | ---: |
| Historical20 | 42.867949 | 44.091831 | 44.091316 | 44.090745 |
| Wrong-parent | 41.989160 | 42.855418 | 42.887448 | 42.871833 |
| Legitimate paraphrase | 43.646857 | 44.657375 | 44.657218 | 44.656655 |

Wrong-parent reference cleaning adds 0.032030pp over the old base. The extra
attachment subtracts 0.015615pp, leaving only 0.016415pp attributable to the
complete new intervention. The apparent 0.882673pp gain over original coupling
is predominantly the already existing class/view correction, not new screening.

The primary wins 8/8 wrong-parent domains against original coupling but loses
to reference-only, the earlier coherent candidate, the new cosine control, and
semantic assignment shuffle2 (42.970401). Shuffles preserve classwise semantic
weight spectra, not final intervention strength. Their results fail the declared
specificity gate; they do not by themselves prove that word meaning is useless.

| Wrong-parent protocol | Complete increment over old class/view base, pp |
| --- | ---: |
| VDD | +0.072575 |
| Potsdam | +1.058719 |
| UDD5 | -0.228863 |
| OEM | -0.052717 |
| LoveDA P | -0.094720 |
| LoveDA D | -0.154984 |
| Vaihingen | -0.378887 |
| LandCover.ai | -0.359550 |
| FLAIR-1 | +0.175027 |

The suite took 789.6501s. Combined 79-arm worker peaks were
5602.3564-5961.9160MiB; these are not standalone deployment costs. Earlier
numerical and per-image endpoints, unique coverage, confusion sums, transitions,
zero-semantic identity and held-family independence were verified. Passing
execution tests is not evidence that screening helps.

## Diagnosis: What Remains Unresolved

Three questions must remain distinct:

1. Does the phrase denote its assigned class? Language role judgment addresses
   this imperfectly; it can confuse `agricultural silo` with agricultural land.
2. Does the observed content at this query support that phrase/class? A correct
   vehicle phrase can still respond to road. Geometry consistency and a high
   response do not establish semantic ownership.
3. Does changing this contribution improve the complete competing-class margin?
   Removing evidence can reduce both false and true coverage, or move false
   occupation into another class. Normalization and writeback also affect it.

This experiment does not isolate which of source error, reference ambiguity,
class offsets or writeback causes the remaining failure. Missing spatially
resolved semantic evidence is a hypothesis, not a proven universal explanation.
Do not replace this uncertainty with another fitted rejection coefficient.

Normalized log-mean-exp already gives dynamic alias responsibility:
`q_a = softmax(beta*r)_a`. Its prior is uniform; its actual contribution is not.
q measures current response, not reliability. Consequently the model is neither
a proven reliable selector nor literally giving every word equal pixel influence.

## Complete Architecture: Two Coupled Mechanisms

```text
image + declared ontology + candidate aliases
                      |
             frozen DINOv3 / DINO.text
                      |
1. Geometry local reader
   local descriptors, local anchor g, relation G, coordinates and validity
                      |
2. Accountable contextual observation and reconstruction
   attributed VIP wide observation b and per-alias contributions
   physically fine, query-local competing-class reference [proposed]
   alias reliability attached to query and rival [not yet established]
   one admitted contextual field b_star
   Geometry-supported innovation with local fidelity
                      |
            original dense assembly / prediction
```

Alias control is an internal part of mechanism2, not a third global TopK stage.
No new encoder, detector, learned class prior or dataset routing is adopted.
Vocabulary-time language membership is an optional prior, not a pixel teacher.
Do not use it alone to decide which pixels must lose a class score.

The retained executable model has `b_star=b`:

```text
min_z 0.5*||z-g||_F^2 + 0.5*||G(z-b_star)||_F^2
H_G = (I + G^T G)^(-1) G^T G
z = g + H_G(b_star-g)
```

This is a classical inference objective, not a new solver or supervised loss.
G controls where a contextual innovation can be reconstructed; it does not
certify its semantic correctness. VIP's wide observer must retain attribution.

The existing experimental bounded writer can stay fixed while testing a new
source, so that a changed solver does not obscure source utility:

```text
delta_c = log(1 - sum_a R(i,a)*max(q_a-1/K,0)) / beta
b_star = b + delta
```

With bounded R, attenuation is bounded by log(K)/beta; R=0 recovers the original
coupling. UNKNOWN means no extra intervention, not compulsory deletion. Useful
words retain their original dynamic response; this writer does not demonstrate
positive amplification. Pair-conditioned evidence may vary with the rival, but
the final coherent class field still changes multiclass competition. It cannot
promise that an A/B adjustment leaves A/C decisions unchanged.

## Next Proposed Study: Change The Observation, Not Rejection Strength

The next worthwhile hypothesis is whether physically finer RGB observations
provide better query-local alias ownership than the existing native reference.
It was a proposal at this decision point, not an Astra-endorsed equation. The
separate subsequently frozen/completed experiment is recorded at the top.

Previous native-query witnesses used unresized 336 crops at 224 stride, at most
four per 512 window: 16 original pixels/token. They changed view/context but did
not physically magnify detail. Native donor re-acquisition reread cached tokens;
it also did not acquire new RGB detail. Conversely, the earlier cross-scale
model did use zoomed detail, but changed fine/coarse relation transfer and lost
useful coarse content. Its failure is not evidence that a zoomed reference will
work, nor a direct test of that narrower ownership question.

Proposed observation: four fixed 256-to-512 RGB crops covering the 512 window,
giving 8 original pixels/token with the current 16-pixel patch encoder. Retain
validity and exact original-image footprints. Keep the original wide observation,
local Geometry and writer unchanged. Replace only the native reference path,
including its held-family reference, rather than adding another prediction
branch or arbitrary logit average. No detail RGB is synthesized by interpolation:
the benefit sought is more encoder samples of the existing real image content.

The concrete candidate must freeze score units, support pooling, boundary
handling and full-image crop budget before target metrics. Finer tokens alone
do not calibrate class scores, separate correlated concepts or guarantee useful
alias decisions. Do not add gains or class thresholds to make its scores fit.

Use a small, matched-budget complete-candidate comparison, not another expansion
of all historical arms:

- Same fine observation with reference/class-view correction but no alias-specific
  attachment: separates new visual information from screening.
- Complete query/rival-conditioned attachment with the same observation and writer.
- Predeclared within-parent alias assignment shuffles, with intervention strength
  reported or matched: tests whether identity adds useful discrimination.
- Coordinate/support shuffle with the same observation budget: tests localization.
- Preserved original coupling and existing hard-deletion reference, where protocol
  identities permit reuse: anchors overall performance and the deletion question.

Require an actual incremental benefit over the same-observation no-attachment
model, not only over original coupling. Also require retained clean/paraphrase
coverage, reliable wrong-parent correction, and superiority to declared semantic
and spatial controls. Report TP/FP/FN and wrong-to-wrong transitions, including
car/vehicle precision and water/low-vegetation recall. The gate and seeds must be
declared before results; failure must not trigger per-domain routing or selecting
the best control. A shuffle comparison alone is not a substitute for uncertainty
or independent validation.

If finer observation cannot supply useful identity-specific correction, stop
this same-source alias-controller route. Options then are a paper centered on
the independently supported Geometry contribution, or a separately justified
new pretrained/learned observation mechanism with an explicit budget/protocol.
Do not assert that all training-free improvement is impossible, or add an
unsupported third contribution merely to complete a story.

## CVPR Claim And Completion Requirements

The defensible target is contextual evidence admission grounded in Geometry,
not the slogan that more words are always better or that every input bank must
lose a fixed fraction of words. A valid selector can keep all words on a clean
bank. It must show useful, identity-specific restraint when evidence is harmful.

Current evidence does not yet establish that restraint as a universal contribution.
The existing wide-view gain and classical reconstruction cannot be renamed as
a wholly new visual observer. Geometry needs its matched nearest-method boundary;
the second mechanism needs independent incremental evidence. CVPR acceptance and
all-eight SOTA cannot be inferred from these development screens.

After a controller succeeds: freeze one full-image rule, evaluate the eight full
labeled protocols, compare matched complete official VIP and closest methods,
measure standalone latency/memory, and validate fresh regions/vocabularies.
Apply vocabulary stress to every relevant branch, not only context with clean
local20. Use provenance-recorded raw LLM output. If claiming general OVSS, include
natural images with common templates and the same fixed inference rule.

## Artifacts

- Protocol: `SEMANTIC_REFERENCE_ADMISSION_PROTOCOL_20261003.md`.
- Exact results: `SEMANTIC_REFERENCE_ADMISSION_RESULTS_20261003.md`.
- Structured summary and per-image arrays: `semantic_reference_admission_20261003/`.
- Implementation: `../DINOtool/dinotool/semantic_reference_admission.py`.
- Experiment utility: `../tools/semantic_reference_admission_experiment.py`.

Original models and results remain intact. No automation, parameter sweep,
winning-control promotion or new GPU run is created by this document.
