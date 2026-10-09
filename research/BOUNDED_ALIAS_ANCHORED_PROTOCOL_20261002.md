# Bounded alias aggregation with Anchored reconstruction

Prospective configuration, 2026-10-02. Implementation:
`geometry-bounded-alias-anchored-v2-exact-inactive-20261002`.

The primary candidate limits alias responsibility in both original local
Geometry and the existing guarded broad VIP observation. It then uses the
unchanged Anchored reconstruction. Frozen weights, all20 aliases, original
text families, scoring temperatures, physical views and image assembly remain.
No target-label parameter selection. VIP is an explicitly borrowed source.

## Aggregation

For effective alias logits s, solve
`max_w <w,s> - T KL(w||uniform)` with `sum(w)=1` and `0<=w<=rho/K`.
Fix rho=4 on every dataset. K=20, so each responsibility is at most0.2.
All aliases remain in the bank; no fixed retained-word count or semantic
correctness claim. This constrains a derivative with respect to effective
alias logits, not the complete image/text encoder or VIP salience chain.

Local T=.07, count-normalized entropy objective. Broad VIP retains the
original template similarities, salience rescaling, scale40, tau=tem=1,
and the LSE count offset; T=1/tau for final aggregation. Local RS and broad
ImageNet representations intentionally match the historical Anchored source.
rho>=K exactly replays original LME/LSE. rho=1 yields uniform averaging.
The numerical saturation solve uses float64 to avoid ambiguous tail ties;
reported scores/weights return to source precision.

Numerical repeat r2: the initial mask-free smoke found a tiny rounding change
when the constraint was inactive. That run was stopped before inspecting
target metrics. Inactive rows now return the original score and responsibility
exactly. The corrected run uses a separate `bounded_alias_anchored_screen_r2_20261002`
directory. No rho, words, sources or fixed samples changed; initial outputs
are preserved as an aborted numerical preflight, not mixed into this result.

The same fixed unit-weight Anchored objective receives bounded g and b:
`min_z .5||z-g||^2 + .5||A(z-b)||^2`.
This is inference reconstruction, not a network training loss. Entropic
caps and quadratic reconstruction have precedent; no verified novelty claim.

## Four matched factorial arms

- Original aggregation + MeanLogit_VIP.
- Bounded aggregation + Bounded_MeanLogit.
- Original aggregation + Anchored_VIP.
- Bounded aggregation + Bounded_Anchored (primary).

Additional controls: Geometry, Bounded_Local, BroadVIP, Bounded_BroadVIP,
and original MeanProb_VIP. Direct old/new/truth transitions for all six
factorial contrasts distinguish correct coverage loss from reduced false
activation. No new manual word cleanup or injected corruption in this run.

## Development screen

Physical A800 GPUs4-7. Exact existing96 COMPLETE images: UDD5 full40 and
eight fixed images each for VDD, Potsdam, OEM, LoveDA, corrected IRRG
Vaihingen, LandCover.ai and FLAIR-1. LoveDA P/D share images; D enters the
equal-domain mean once. These are development sets, not untouched validation.
LandCover.ai substitutes for unlabeled iSAID.

Require local mathematical tests and a mask-free actual-checkpoint smoke.
Every dataset must exactly replay five original controls per image against
the saved paired-context evaluation. Verify unique complete keys, fixed
vocabulary/checkpoints, confusion sums and factorial transition endpoints.

Promotion: primary mean exceeds original Geometry, both original fusion
controls, original Anchored and bounded simple fusion. No reported protocol
loses more than1pp versus Anchored; VDD/Potsdam retain Anchored scores. This
screen only decides whether the frozen rule merits full evaluation; no
automatic full rollout, per-domain routing or retrospective rho tuning.

Risks: a rare valid appearance may depend on one alias and be suppressed;
many correlated wrong aliases can survive the cap. The cap is a robustness
hypothesis and cannot certify which word is semantically correct. Report all
domain/class outcomes, interventions and combined runtime/memory.
