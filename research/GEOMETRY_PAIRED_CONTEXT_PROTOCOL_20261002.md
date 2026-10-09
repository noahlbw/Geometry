# Geometry-supported paired contextual readout

Prospective frozen experiment: `geometry-paired-context-readout-v1-20261002`.
Primary: `Geometry_PairedContext`. Physical A800 GPUs 4-7 only.

Keep original Geometry, all20 aliases, local512/128, and guarded VIP wide448,
336/112, ImageNet templates, tau=tem=1, scale40, no background threshold.
The borrowed VIP observation remains attributed. No new checkpoint or training.

For each actual local512 tile, threshold its label-independent Geometry
conditional donor probabilities strictly above the uniform probability among
valid donors. Map query/donor supports into each wide21x21 grid using fractional
overlap of actual patch footprints in original image coordinates. This is a
fixed natural reference, not a calibrated correctness threshold or object mask.
Unmapped wide queries retain full donors. Self Value fallback handles empty rows.

Encode each wide crop once. Full and reference heads start from identical
backbone tokens; reference intersects both VIP head blocks' donor supports with
the transported Geometry support. Residual/MLP/projection remain frozen. Both
reads use the full read's fixed alias-salience profile and the exact same text.
Backbone context and indirect two-block influences remain in the reference.
Thus the difference measures this head intervention, not all contextual content
or ground-truth semantic correctness.

Let d be sampled full-minus-reference class logits on each local grid. Primary
solves `min_delta .5||delta||^2+.5||A(delta-d)||^2`, using the unchanged anchored
solver and adding delta to original Geometry logits. Both positive and negative
increments remain. Zero d exactly recovers Geometry. With A=I this exactly equals
the same-information `g+.5d` control. No per-class coefficients or sign clipping.

Controls: Geometry, BroadVIP, MeanProb_VIP, MeanLogit_VIP, Anchored_VIP,
MeanLogit_Context, ShuffledContextWriteback (same measured d, permuted writeback
relation). The shuffled control isolates writeback, not support construction.

Screen: same fixed96 image IDs as the prior SAT screen: full40 UDD5 plus8 each
of the other seven datasets. These are complete-image predictions, unlike the
later detector cached-window experiments. Corrected IRRG Vaihingen mandatory.
LoveDA P/D share images; D enters equal-domain mean once. OEM available384;
LandCover.ai substitutes for unlabeled iSAID. All domains informed development.

Promotion requires mean above original Geometry, Anchored_VIP and both ordinary
VIP fusions; above same-information half-context mean; no Geometry protocol loss
greater than1pp; VDD/Potsdam above both Geometry and Anchored_VIP. This stringent
prospective gate prevents an average-only gain being called cross-domain success.
Failure ends this candidate; no target-label tuning or per-domain winner routing.
Run correctness tests and a mask-free real-checkpoint smoke before the screen.
Verify unique complete coverage and exact original Geometry per-image confusions.
