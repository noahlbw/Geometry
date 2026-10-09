# Query-anchored frozen semantic response transport

Primary `Geometry_ResponseTransport`, frozen before new labels/results. Same
original Geometry relation and frozen weights. Preserve original/best systems.
This is an unverified mechanism hypothesis toward CVPR, not a novelty guarantee.
Parallel literature lookup is unavailable because its CLI is not authenticated;
local prior implementations are inspected, but no exhaustive external priority
claim is made. No unpublished results were sent to an external service.

## Derivation

For each head block, recompute native attention on its evolving state x. Let
u_j be the frozen projected/LayerScaled conditional self-patch attention read:
native special contribution plus native patch mass times V_j. Prefix queries
retain full native attention. Let f(x)=ls2*MLP(norm2(x)).

Primary patch update:

`x'_i = x_i + f(x_i) + sum_j G_ij [u_j + f(x_j+u_j)-f(x_j)]`.

The query's semantic baseline is not averaged across donors. Transport the
response induced by a visual read, including its frozen nonlinear MLP change.
Prefix state follows x+u+f(x+u) without transport. Repeat at both blocks; apply
unchanged final normalization/projection. Compute small differences and transport
in fp32, with original bf16 head AMP. No label or text feeds the visual path.

Same-attention before control:
`x'_i = x_i + sum_j G_ij u_j + f(x_i+sum_j G_ij u_j)`.

Their difference is
`G[f(x+u)-f(x)] - [f(x+G*u)-f(x)]`.
It vanishes for any affine f even when query/donor x differ, and at G=identity.
This is NOT a correctness guarantee or an invented learned loss. f includes
nonlinear normalization and MLP; do not attribute the difference to MLP alone.

## Controls

- Original Geometry, BlockPrefix, SCLIP_Two and VIPProxy_Two: exact references.
- Geometry_DonorBefore: same u/G, transport before the nonlinear response.
- Geometry_FullDonor: x+G*(u+f(x+u)); tests transporting donor baseline semantics.
- Geometry_QueryResponse: preserve original Geometry attention increment u_G,
  then x+u_G+f(x)+G*(f(x+u_G)-f(x)); removes donor native-mass confound.
- Geometry_ResponseUniform/Spatial: same primary donors, uniform/Gaussian-only
  relation instead of raw-feature Geometry; tests meaningful relation use.
- Geometry_SelfConditional: x+u+f(x+u), no geometric transport.

Original Geometry vs DonorBefore includes query-owned versus donor-owned special
and patch allocation differences. Only primary vs DonorBefore at identical
input states isolates response ordering. Full two-block mIoU differences also
include later state evolution. Do not call all primary improvements nonlinear.
QueryResponse has a different attention ownership rule from the primary.

Unlike prior FrozenPath/EvidenceGeometry, this transports induced MLP responses
and feeds the changed state into the next block. Unlike post-head smoothing,
the query residual and its baseline are retained. Similar algebra or generic
graph message passing can have precedents; public novelty remains to verify.

## Fixed experiment and decision

Existing fixed96 complete-image development manifest: UDD5 full40, other seven
domains8 each. Same20 aliases/class, six RS templates, normalized LME .07,
native512/128/Hann, corrected Vaihingen input; LandCover.ai substitutes for
unlabeled iSAID. LoveDA P/D both reported; eight-domain mean counts D once.
No source selection, threshold changes, target-label fitting or extra encoder.

Tests: native conditional-self allocation, identity/self replay, affine
commutation, nonlinear difference, zero attention/MLP, prefix preservation,
historical exact operators, finite controls and unchanged inputs/weights.
Mask-free fp32/bf16 actual-checkpoint smoke: exact four baselines and singleton
outputs; independent one-arm latency/memory. Diagnostic extra f calls are not
part of the deployed cost. Full96 unique/per-image coverage and frozen source
identities required. Original Geometry confusion must exactly replay.

Frozen advancement gate: primary mean above Geometry/BlockPrefix/SCLIP_Two/
VIPProxy_Two AND DonorBefore/ResponseUniform/ResponseSpatial; VDD/Potsdam no
loss vs Geometry; every protocol loss at most1pp;
singleton cost at most1.15x. Passing warrants broader locked validation, not
final superiority/CVPR readiness. Reject primary if gate fails; no control-to-
primary switching, per-domain rule selection or parameter adjustment.

Report all arms, per-class IoU/precision/recall/area, transitions, response and
nonlinear-order activity, source identities and deployed cost. Small-panel ranks
can reverse on full sets. Use only currently idle physical GPUs0-7. No heartbeat,
duplicate/restarted evaluations, output overwrite or paused automation resume.
