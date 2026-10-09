# Geometry-grounded competitive alias ownership: model design

2026-09-30. Design proposal; no new experiments or performance claims. The user
asked to consolidate the mechanism and architecture, not launch another screen.

## Verified diagnosis motivating the model

Full UDD5 results, GEAR-OV, rows = ground truth and columns = predictions:

| Vocabulary | road TP | road FP | road FN | precision % | recall % | IoU % |
|---|---:|---:|---:|---:|---:|---:|
| all20 | 31,305,474 | 12,039,029 | 27,693,420 | 72.22 | 53.06 | 44.07 |
| fixed15 | 18,250,784 | 4,248,550 | 40,748,110 | 81.12 | 30.93 | 28.86 |
| road20/others15 | 24,922,781 | 8,176,561 | 34,076,113 | 75.30 | 42.24 | 37.10 |
| road15/others20 | 20,780,479 | 5,532,104 | 38,218,415 | 78.98 | 35.22 | 32.20 |
| dynamic | 20,811,701 | 5,510,043 | 38,187,193 | 79.07 | 35.27 | 32.26 |

Restoring non-road words raises road recall and also increases false positives.
It does not establish that all corrections are beneficial. Overall mIoU drops
48.1441 -> 47.3736 in this contrast. Marginal confusion counts are not paired
pixel transitions. Geometry alone also improves road IoU 29.90 -> 34.63;
ordinary class-score competition therefore suffices for part of the effect.

The actual `class_scores` in `dinotool/gear_ov.py` is log-mean-exp. With raw
similarities s and temperature tau:

S_c = tau log[(1/K_c) sum_(a in A_c) exp(s_a/tau)].

If E_c = exp(S_c/tau), adding one word produces
E'_c = [K_c E_c + exp(s_new/tau)]/(K_c+1).
Adding a below-average word lowers that class's evidence at the pixel; removing
one raises it. Selection changes semantic content AND calibration. A weak word
on road pixels may usefully restrain a competing class, even if it does not
directly identify road. The restored `other` words were `other land cover`,
`construction area`, `water surface`, `agricultural field`, `sports field`.
Their individual causal contributions are not known from the saved aggregate
results; dilution of stronger `pavement`/`parking area` responses is a plausible
explanation, not a verified word-level attribution.

GEAR also rebuilds the full text center, SVD basis, and whitening matrix when
the candidate vocabulary changes (`build_text_basis`). This can change road
reconstruction even if its aliases are unchanged. The proposed method freezes
the full candidate bank and its observation coordinates to eliminate that
additional source of change.

## One core hypothesis

The relevant unit of trust is (visual location, alias, target class), not a
single good/bad flag on a word. Infer which target class explains the visual
support of a phrase, and use the same latent ownership variable to determine
its contribution to the pixel prediction. A phrase can be locally useful,
ambiguous, or wrongly assigned, depending on the target ontology and image.

`roof -> wall` is an invalid assignment if wall and roof are mutually exclusive
classes. `roof -> building` can be valid when building includes rooftops. An
ontology is input metadata; geometry cannot discover the dataset's intended
label definition on its own. Do not automatically remap phrases to arbitrary
classes or call a phrase universally harmful.

## Architecture

1. Frozen DINOv3 and DINO.text produce native image structure, Geometry-aligned
   token features, and the full alias score tensor. Keep the existing local,
   detail, and bounded context observations and their spatial coordinates.
2. One competitive ownership readout infers class ownership on a sparse
   geometry graph and converts it directly to corrected class scores. The
   geometry graph and the ownership variables are shared across scales.

There are two functional blocks: Geometry observation and competitive ownership
readout. Reference construction, graph consistency, and aggregation below are
parts of the one inference problem, not separately trained networks or a stack
of successive correction modules. In the intended architecture this readout
replaces the old alias-selection/reconstruction decision, rather than appending
GAR, TextGraph, CIDER, and another gate in series.

## Reference without direct self-confirmation

Group exact duplicate and near-synonymous phrases into text-only families using
a fixed rule shared across datasets. When judging a phrase in family g, omit
that whole family from the *semantic reference* in all classes. Do not change
the image features, graph, full reconstruction basis, or final evidence bank.
Remaining families provide Q^(-g), a soft class reference. Average families
rather than raw synonym counts in this reference so repeats do not fabricate
independent votes. Canonical names may contribute but are not the sole judge.

Use aligned views and multiple remaining families to distinguish affirmative
rival support from missing evidence. If no usable family remains, views
disagree, or the supposed rival lacks support, assign an unresolved state.
Leaving a family out reduces direct leakage; correlated errors can still occur.

For alias a at location i, construct qbar_(i,a) over C target classes plus an
unresolved state. The class entries combine Q^(-g(a)) and a weak text-ontology
compatibility prior. Their total probability is the confidence in corroborated
support; the rest is unresolved. This prior is not an oracle or a hard
text-similarity filter. Confidence and family grouping must be fixed rules,
not learned from target labels.

## Coupled inference objective

Let R_(i,a) lie on the probability simplex over C classes and unresolved. Let
h_(i,a) indicate alias support strength; normalize support within a phrase's
local footprint so small valid objects are not penalized merely for their area.
Let W_ij be sparse, boundary-sensitive raw-DINO geometry with explicit spatial
and cross-view support. Infer:

E(R) = sum_(i,a) h_(i,a) KL(R_(i,a) || qbar_(i,a))
       + (lambda/2) sum_(a,(i,j)) W_ij sqrt(h_(i,a) h_(j,a))
                                      ||R_(i,a)-R_(j,a)||^2.

Observations, qbar, h and W remain fixed during this optimization. The objective
is convex in R (with positive, smoothed reference probabilities); its terms
enforce semantic support and geometric consistency. The first implementation
can use a small fixed budget of simplex-preserving mirror-descent updates.
Network parameters never receive updates. Optimize image-specific inference
states only; report this inference cost when calling the model training-free.

Class competition occurs within R: evidence assigned to roof cannot at the same
time have full ownership by wall. Unresolved evidence does not justify deletion.
The graph smooths *ownership of an observed phrase response*, not ground-truth
labels or a final hard region mask. Fine pixel/token positions remain intact.
High graph affinity alone never proves a semantic class.

For an alias whose supplied parent is c, define retention
r_(i,a,c) = R_(i,a,c) + R_(i,a,unresolved).
Only affirmative competing-class ownership reduces its contribution. Broad
support, small support, and absent-class uncertainty do not by themselves
force deletion. KL inference gives soft selection, not exactly sparse K-word
lists. This is intentional: the model dynamically selects usable evidence at
locations, without imposing a global count or physically deleting uncertain
phrases from the candidate bank.

## Class-score readout without count-induced inflation

Let z_(i,a) = s_(i,a)/tau and e_(i,a) = exp(z_(i,a)). Let b_(i,c)^(-g(a)) be
the class score from the fixed family-excluded reference in the same score
units, and e_ref = exp(b). Keep the original number K_c of candidate slots.
Use:

e_tilde_(i,a,c) = min(e_(i,a), e_ref)
                 + r_(i,a,c) [e_(i,a)-e_ref]_+.

S_tilde_(i,c) = tau log[(1/K_c) sum_(a in A_c) e_tilde_(i,a,c)].

This suppresses only an unsupported *positive excess* over the reference.
When ownership is correct or unresolved (r=1), it exactly preserves the
original alias evidence. When a rival is supported (r<1), it can lower a
misassigned high response. Crucially, it never raises an individual class
score solely because a weak alias was rejected. Weak responses and the original
denominator remain, avoiding the observed inflation from re-averaging a shorter
list. A symmetric replacement of every rejected score by e_ref would sometimes
raise a weak score; the positive-part constraint avoids that failure.

Implement this in a numerically stable log domain; do not materialize unstable
large exponentials. Missing reliable reference implies r=1. For a one-class
vocabulary there is no rival and no competitive suppression. Exact duplicates
must share ownership computation rather than pretending to be independent
evidence; keep the original per-class slot weights for compatibility.

All-r=1 must recover the chosen full-bank Geometry/multiscale readout exactly.
Changing inference variables never recomputes full-bank whitening or SVD.
If integrating initially with GEAR observations, construct its basis once from
all candidates and keep it fixed; do not describe physical bank pruning as the
same intervention. The final architecture uses the ownership solver as its
single context-aware semantic readout instead of stacking both optimizers.

## Expected behavior and limits

- Valid `narrow road`: the phrase need only explain its own road-shaped support,
  not all road pixels. Reliable own-class support or unresolved ownership keeps
  it; rare area is not a penalty.
- Wrong `roof` in wall: other phrase families and the ontology indicate roof on
  geometrically consistent roof tokens. Its positive wall excess is reduced;
  the correct roof class can win without inventing new labels.
- Broad `pavement` in other: only locations with corroborated road ownership
  suppress its other-class excess. It can remain useful on actual other ground.
- Low-scoring other words on road: keep their existing counterweight rather
  than removing slots and accidentally strengthening `pavement`.

Improved margins are not guaranteed improved mIoU: an incorrect reference can
still suppress the correct class, and coherent texture can span different
semantic classes. The unresolved state and one-sided correction limit specific
failure modes but do not solve identifiability or guarantee accuracy. If the
entire vocabulary and ontology anchor agree on a wrong class, frozen image
structure alone cannot recover the missing semantics.

## Implementation surface

- Reuse `prepare_views` and text-independent `build_support` observations.
- Add a single ownership readout implementation with `build_family_reference`,
  `solve_ownership`, and `ownership_class_scores` functions; these are steps of
  the same objective, not independent prediction heads.
- Cache full-bank alias maps and family-excluded reductions; no extra DINO
  forwards per alias. For C about 5–12 and M about 100–240, chunk the ownership
  tensor by alias family on each tile and keep the graph sparse.
- Keep ontology metadata explicit. Avoid per-dataset K, class-specific manual
  thresholds, a supervised router, class-area quotas, or reassignment based
  solely on text cosine.

## Paper positioning

Candidate thesis: vocabulary expansion changes cross-class evidence competition;
visual activation quality alone cannot establish correct semantic ownership.
Infer ownership using geometry and competing explanations, then use that same
variable to govern the classifier's evidence without renormalization drift.

The proposed contribution is the formulation and coupled readout, not a claim
that KL, graph Laplacians, leave-out reasoning, or DINO attention are new. Novelty
against existing work has not been established by a literature review in this
turn. The current UDD5 results motivate the mechanism but do not validate this
unimplemented architecture or guarantee acceptance at CVPR.
