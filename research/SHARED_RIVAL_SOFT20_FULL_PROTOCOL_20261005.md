# Frozen Shared-Rival Soft20 Full Protocol

Authorized eight-domain full evaluation, with original fixed20 aliases/class and
original Geometry depth2/wide observer/reconstruction. Preserve prior slow models
and outputs. Expected unique images: LoveDA1669, UDD540, OEM384, VDD80, Potsdam504,
corrected-input Vaihingen113, LandCover.ai1602, FLAIR-115700; total20092.

Signature: geometry-shared-rival-soft-alias-v1-20261005. No new fine-view or other
visual encoder forward. Existing native and Geometry projected features both
provide local evidence using the unchanged ImageNet query bank/profile. Local
Geometry's original RS text bank/readout remains the fidelity anchor.

For each view, class evidence is count-normalized alias log-mean-exp. Other
classes are summarized by stable leave-class-out log-mean-exp using prefix/suffix
log-cumulative sums. This is one scalar/class/location; it retains all class
support but is not exact pair-by-pair rival conditioning. No alias x all-rival
array or fixed top2 candidate truncation.

Wide positive margin is positive(tanh(alias-rival)), interpolated through the
original crop stencils. Local contradiction is the minimum of
positive(tanh(rival-alias)) from native and Geometry features. Risk is their
product, not a calibrated correctness probability. Weight=1-risk, floored1e-6;
canonical names and invalid positions receive weight1. Beta1 is inherited.

Keep original salience-profiled crop evidence e. Delta is the difference between
normalized weighted and unweighted LME:

    delta = log(sum_a w_a exp(e_a)/sum_a w_a)
            - log(sum_a exp(e_a)/K).

Original interpolation/overlap carries the delta to donor positions. The
prediction is z_original + H delta_b using the original Geometry operator. Weight
identity restores original scores; equal profiled evidence is invariant to
weight dispersion. This is different from multiplying weights into logits.

Simultaneous full arms: Geometry, NoAdmission_Exact, SharedRivalSoft (primary),
SharedRivalHard (same source, weight>=0.5), SharedRivalShuffle (fixed20261005
within-class noncanonical permutation, exactly preserved weight spectrum).
Hard threshold is a fixed diagnostic convention, not a fitted parameter.

Pruned Geometry preparation skips an intermediate result overwritten in the
legacy depth2 route. Require bitwise equality of all real prepared fields plus
original Geometry/no-admission prediction replay in mask-free smokes. Grouped
class scoring and device stitching use already verified execution patterns.

Freeze the entire rule before new masks. Prior experiments motivated it, so
these are exploratory developed-data full validations, not untouched independent
test results. No post-result rule/threshold/dataset routing or control promotion.
Report per-class IoU/precision/recall/area and primary gains against no admission,
hard and spectrum shuffle, even if negative. Retain slow full Hard20 reference;
do not extrapolate partial fine-only scores to full20092.

Reuse already completed numerically repaired VIP_All20 full results on seven
domains. Only missing corrected-input Vaihingen All20 is evaluated once. VIP
retains its official/default settings and background rules; this is same-word
whole-method comparison. Also label its historical official-short result.
Do not use historical NaN-contaminated VIP as primary. LandCover.ai substitutes
for unlabeled iSAID.

After full evaluations, serial GPU7 timing uses first/middle/last complete inputs
from each frozen sequence, three warmed synchronized rotated singleton repeats
for NoAdmission_Exact, soft, hard and finite VIP_All20. Include every local window,
wide observations, stitching and argmax; exclude loading/text encoding/decoding/
masks. These24 images are a timing panel, not full-domain mean throughput. Shared
resident allocation is not standalone model memory. Report100-1000ms budget
violations without changing full image size or coverage.

Manager: tools/shared_rival_soft_full_experiment.py. Remote controller:
scripts/run_shared_rival_soft_full.py; new results root:
results/shared_rival_soft20_full_20261005. Completed results/report are collected
under research/shared_rival_soft20_full_20261005 and
research/SHARED_RIVAL_SOFT20_FULL_VS_VIP_20261005.md.
