# Frozen ADE generated-expression diagnostic

The VDD/ADE goal remains open: useful processing of the unchanged twenty aliases,
full mIoU above saved VIP numeric targets54.3/29.1, and complete-image singleton
latency no greater than6x both matched-input and official-configuration VIP.
Previous template-relative and fine-average trials failed to establish alias
benefit. This diagnostic tests a different, documented source of information.

ADE's existing curated20 strings were generated from safe semantic roots under
neutral expression patterns. Exact pattern-major generation, normalization,
deduplication and twenty-slot truncation recover every string in order. Root
counts are1:73,2:31,3:19,4:26,5:1. Group identity comes from this frozen provenance,
not embeddings, visual predictions, masks or guessed suffix stripping.

For the original profiled alias logits e (including unchanged mean-one salience
amplification), replace only the wide aggregation:

    Q = log20 + LSE_g(mean(e within generated family g)) - logG.

This tests within-family expression-response inflation; inverse family-size
weights alone would collapse to Base on131/150 classes. A common log20 offset
prevents family count from becoming a class prior when all e are equal. Q is
computed at crop tokens before interpolation and Geometry writeback. It is not
yet a proven conditional semantic reliability mechanism or a final paper model.

| Arm | Local text | Wide text/aggregation |
| --- | --- | --- |
| CC | Curated20 | Curated20, inherited |
| HH | Historical433 | Historical433, inherited |
| HC | Historical433 | Curated20, inherited |
| CH | Curated20 | Historical433, inherited |
| CH-flat | Curated20 | Historical433 W minus logK plus log20 |
| CQ | Curated20 | Generated-family Q |
| CQ-shuffle | Curated20 | Same Q, shuffled family labels |
| C-mean | Curated20 | Mean(e) plus log20 |

Shuffle preserves family sizes and canonical family assignment, moving only
nineteen noncanonical labels, not alias responses/salience. On the73 single-root
classes, CQ/CQ-shuffle/C-mean class logits coincide; their IoU can still change
because competing class scores differ. Such gains do not identify their own
family assignment as useful.
Historical433 remains a diagnostic control, never a fixed20 candidate.

All arms share bounded Geometry/wide RGB observations. No fine views, new visual
head, class-rival C-square tensor, new words or fitted coefficients. Keep actual
ADE profile: original Geometry, local T=.07, g=.5, seg6, short336/cap672 wide.
H uses the historical PackedAliasReadout.uniform path, with soft=False; C keeps
the existing scalar LME path. Stale historical profile metadata is not executed.

First/middle/last complete images must replay CC and packed HH independently,
match all singleton/joint predictions, and pass CQ's6x gate against both VIPs
on every image. CQ timing includes only C, Q and original reconstruction. Full
evaluation then covers2000 unique images on four shards. Verify exact aggregate
and per-image CC/HH confusion against preserved references, scored target counts,
checkpoint/vocabulary identity and all downloaded archive sums.

CQ must beat CC, CQ-shuffle and C-mean with positive paired image-bootstrap95%
intervals and exceed29.1. Otherwise reject this attribution without coefficient
sweeps or promotion of a control. HC/CH/HH local/wide effects and their mIoU
interaction are descriptive causal contrasts, not additive percentages.

VDD original20 has no verified generated-family provenance. Its strong55.0754
Base remains preserved; no invented groups or VDD word substitution are allowed
by this diagnostic. A successful ADE diagnosis would still require a grounded
VDD-compatible mechanism and independent budget/accuracy verification.

Existing word pools/profiles and method selection used prior development, so
results are exploratory. Saved VIP paper numbers are targets, not certified
paper-protocol reproductions. Manager: tools/generation_family_alias_experiment.py.
