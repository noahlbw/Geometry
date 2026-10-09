# Bounded Physical Coupling Candidate

The preceding448 full evaluation failed on four of eight domains. The next single
candidate changes only the visual sampling budget: long-edge896 Geometry with
512 crops/384 stride and independent original-RGB long-edge448 wide observations.
Both branches are bounded to at most four actual visual forwards. No native-size
windows, fine observations, new vocabulary, per-domain routing, threshold tuning,
new alias rule or reconstruction change is allowed. Output probabilities are
restored to the original mask dimensions before argmax.

Signature: `geometry-shared-rival-soft-bounded896-v1-20261005`. Geometry depth2,
fixed20 words/class, local RS/wide ImageNet templates, canonical protection and
SharedRivalSoft remain unchanged. This is a sampling/efficiency candidate, not a
new innovation claim. The purpose is to recover local/wide physical-scale
complementarity without88 original-resolution Geometry forwards.

First gate: full VDD80 and Potsdam504, plus warmed same-image fixed20 comparison
against finite VIP on first/middle/last images. The primary must exceed VIP_All20
on BOTH domains before a full eight-domain run is considered. Whole-image median
latency must remain below1000ms on each measured input; report the actual VIP
ratio rather than assume it. At most3x VIP is the preferred efficiency target,
not an achieved claim. Acceptance of a final model still requires full eight-domain
performance, per-class/small-object outcomes, independent inference cost and an
honest comparison to official-query/distilled VIP as separate protocols.

All five original simultaneous readouts are retained for the accuracy measurement.
Timing is singleton inference, three warm repeats, serial workers with no other
GPU jobs. Evaluation worker wall time is not inference latency. Smoke checks
verify unchanged Geometry preparation, singleton equality, frozen weights and
actual encoding budgets without loading target masks. Prior validation results
motivated this candidate; it is development, not untouched independent validation.
