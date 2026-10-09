# Geometry / VIPProxy factor transplantation

Primary: `Geometry_VIPSupport`, frozen before masks or new results. This is a
mechanism experiment toward Geometry optimization, not a novel CVPR module by
renaming a borrowed operator. Original/best models remain unchanged.

## Fixed scope

Use the existing fixed96 complete-image development manifest: UDD5 full40,
eight complete images in each other domain. Same20 aliases/class, six RS text
templates, normalized LME .07, native512/128/Hann probability assembly. One
backbone and one deployed head, no additional encoder or physical view.
Corrected Vaihingen input; LandCover.ai substitutes for unlabeled iSAID. LoveDA
P/D share images; eight-domain mean counts D once. Not untouched validation.

## Hypothesis and controls

Let S be original raw DINO cosine and G the original spatial Geometry relation.
Pinned VIP support is `S > 1.5*mean(S)` using the whole crop mean, not the
previous Geometry sparse implementation's per-row mean. Only empty VIP rows
receive self-Value fallback. The primary renormalizes G on this support and
retains native special contribution, native patch mass and special queries.

- Geometry / BlockPrefix / SCLIP_Two / VIPProxy_Two: exact historical controls.
- Geometry_VIPSupport: support only; frozen primary.
- Geometry_VIPWeights: cosine*.5 rather than cosine/.10, same spatial term.
- Geometry_VIPSupportWeights: both changes, same spatial term/native pathway.
- VIPRelation_GeometryPath: exact VIP relation, native Geometry allocation.
- Geometry_VIPPath: original G with VIP unit patch/normalized-prefix pathway.

The weighting pair includes a change in feature-versus-distance balance.
The difference from exact VIP relation isolates the spatial term under that
weighting. VIPPath changes both prefix evolution and patch allocation; prior
staged allocation controls already isolate other parts of that bundle.
Final mIoU contrasts are nonlinear responses, not additive causal components.

Failure predictions: a visually coherent but semantically wrong donor survives
the mask; empty rows can discard useful context; weaker weighting can spread
small-target activations; changing patch mass can trade recall for precision.
No class-certification or small-object improvement guarantee is asserted.

## Verification and decision

Synthetic tests: exact pinned support/relation, finite empty-row fallback,
native group mass and special-query preservation, exact historical operators,
VIP pathway identity and unchanged inputs/weights. Real-checkpoint mask-free
fp32/bf16 smoke must replay all four baselines and selected singleton path.
Measure independent512-window latency/memory excluding comparison heads.

All96 unique keys, frozen vocab/checkpoints, original Geometry configuration
and all four historical per-image confusion matrices must replay. Report all
arms, per-class IoU/precision/recall/area, beneficial/harmful transitions,
support activity and deleted Geometry mass. Source masks are evaluation-only.

Frozen advancement gate: primary mean greater than Geometry, BlockPrefix,
SCLIP_Two and VIPProxy_Two; VDD/Potsdam no loss versus Geometry; every protocol
loss at most1pp; singleton latency at most1.1x. Do not promote a control or
retune rules after observing results. Small-panel ranks can reverse on full
data; passing is reason for broader locked validation, not final superiority.
No new heartbeat, no paused automation restart, no duplicate jobs or output
overwrites. Revalidate idle physical GPUs0-7 before launching.
