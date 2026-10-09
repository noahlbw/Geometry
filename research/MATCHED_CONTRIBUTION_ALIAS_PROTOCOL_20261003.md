# Frozen Matched-Contribution Alias Source Screen

Implementation `geometry-matched-contribution-alias-source-v1-20261003`.
One changed reliability observation path. Original Geometry, shared128px
supports, two fills, crop/stencil/overlap, all20 aliases, weights, templates,
bounded excess writer, classical rival potential and reconstruction stay fixed.
Same64 developed image IDs, eight/domain, one top-left512 window/image, original
whole-image broad observation. LoveDA P/D share images and mean counts D once.
Corrected IRRG; LandCover.ai substitutes for unlabeled iSAID. Not full-dataset
or untouched validation. No target-label selection or post-result routing.

## Exact Changed Source

Compute the SAME per-template mean logits and crop salience used by the original
wide observer. After original salience profiling, let r(t,c,a) be alias evidence.
At each wide token compare an alias with the actual rival class's normalized
log-mean-exp, not a raw mean-template cosine against one canonical phrase:

```text
l_d = logsumexp(beta*r_d)/beta - log(K)/beta
m(t,a,d) = r(t,a) - l_d
logmeanexp_a(beta*m(t,c,a,d))/beta = class_score_c - class_score_d
```

K20 and beta1 remain fixed. Aggregate each margin via the exact original crop
stencil only AFTER crop profiling and rival aggregation. This interpolation of
per-alias margins does not generally retain the log-mean-exp identity after
averaging multiple tokens/crops; the identity is asserted at wide tokens only.
The actual full class-score field must replay the unchanged broad prediction.

At every query use the same declared reversal rule, minimum across both fills:
full margin positive, kept margin negative, removed margin positive;
gamma=clamp(-m_kept/max(m_removed-m_kept,1e-6),0,1). Canonical/self/unsupported
risks are zero. Unknown remains retained. No survivor renormalization, hard
deletion quota, promoted alias or fitted gain. The source is query/rival
conditioned; the unchanged writer applies the query risk across its exact wide
crop contributions. This does not claim per-stencil risk admission or semantic
truth from same-encoder interventions.

The writer remains d(c|d)=log(1-sum_a gamma*positive(q_a-1/K))/beta,
e=d-d^T, v_c=sum_d e(c,d)/C and z=z_original+H_G*v. It has the prior directed
capacity bounds and standard least-squares consistency, not a new solver.
Controllers affect broad aliases only; local20 remains unchanged.

## Controls And Checks

Retain all29 historical arms as cached endpoints, with actual Geometry and
unmasked wide-score replay. Add primary, same-source fixed fusion, shuffled
support source, template-matched text-only source, three canonical-protected
alias/rival-vector shuffles, directed-capacity-balanced mean and three
directed spatial shuffles. Alias shuffles preserve each rival's risk spectrum;
directional controls preserve pair budgets and capacities, spatial controls
also preserve each directed field's spectrum, not final-potential covariance.
Shuffled support can change risk/action strength; report it, do not infer
support causality from unmatched strength alone.

Text-only control uses the original wide query template features averaged per
alias and normalized, not the RS bank. It has no masked-image observation and
does not include image salience. Its conflict heuristic remains the existing
text-only definition, not the intervention's numerical margin formula.

Ten targeted tests plus the retained40 source/reader/allocation tests and
mask-free real-checkpoint smoke precede labels. Check actual alias identities,
checkpoint/config/coverage, unchanged Geometry operator and shared masks,
full-class-score replay, zero-risk writer identity, canonical protection,
finite risks/actions, directed capacities, normal equations and budgets.
Record source-derived observations/actions before loading masks. Verify all29
historical per-image endpoints exactly and every confusion/transition endpoint.

## Prospective Promotion Gate

Primary equal-domain gain>=0.1pp vs original coupled reference, at least5/8
domain wins, no protocol loss>1pp including LoveDA P. Mean must exceed both
prior ContrastReversal_Exact and RivalPreserving_Exact, every newly declared
control and prior RivalShuffledSupport_Exact, all three prior rival-alias
shuffles, prior directed mean/spatial shuffles. These comparisons are frozen
before any GPU result. No favorable seed/fill/domain choice afterward.

A failed gate keeps the retained original model and ends promotion of this
specific matched reversal source. Passing requires subsequent complete-image,
vocabulary-noise and standalone-cost validation before full promotion. Do not
equate raw ranking or small window gains with useful universal screening.

## Deployment And Reporting

Remote root `results/matched_contribution_alias_screen_20261003`, sessions
`gmca03_*`. Only authorized physical GPUs0-7 after occupancy checks. Preserve
all prior outputs and unrelated sessions. Report mIoU/per-class metrics,
beneficial/harmful changes, source/action strength, unknown fraction, costs
including all new masked forwards. Combined40-arm timing is not deployed
primary latency; old cached observations are not charged as new forwards.
