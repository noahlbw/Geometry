# Geometry conservative contrast: frozen mechanism study

This follows the completed CSA allocation study, not a successful final model.
The new rule is motivated by those labeled development results; no independent
validation or original-priority claim. Preserve original Geometry, best coupled
models, historical outputs, unrelated sessions and paused automations.

## Hypothesis

Stronger patch reads improve car precision and vegetation coverage but can
remove useful residual/background evidence. CSA patch allocations are almost
always1, not adaptive semantic certification. Test whether compensating only
surviving centered patch differences can improve Geometry without moving its
aggregate Value read at the same block input. Attention/read norm is not class
correctness; contrast enhancement has standard signal-processing precedents.

## Primary rule

At each frozen head block let v_i be patch Values, mu their per-head spatial
mean, p_i=(normalized G)v_i, m_i the unchanged native QK patch mass, and
o_i=special_i+m_i G_i v the ORIGINAL Geometry read (unedited AMP path).

E0=mean_i ||v_i-mu||^2; E1=mean_i ||p_i-mean(p)||^2.
g=max(0,1-E1/E0), with g=0 when E0=0. Thus g in[0,1].
d_i=g*m_i*(p_i-mu); primary o'_i=o_i+d_i-mean_i(d_i).

Use fp32 for extra statistics/correction, cast to original Value dtype before
the unchanged projection. Both blocks recompute their native Q/K/V and gain
from their own evolving states. Native prefix query reads stay unchanged.
No text-based gain, extra encoder, view, MLP, solver or fitted threshold.

At fixed inputs, the additional read has zero window mean, is invariant to a
common patch-Value shift, and vanishes for identity G or fully collapsed uniform
contrast. It does not recover information erased by smoothing. Total signed
coefficients can be negative; this is NOT a probability attention distribution.
Finite AMP casts approximate intermediate mean preservation. The residual/MLP,
second block and final normalization need not preserve class areas or mIoU.
Means include all patches of the unchanged padded crop; no new padding mask
is introduced. This boundary is recorded rather than silently changing support.

## Controls and locked data

Four exact historical operators, Geometry_DoubleRow and the prior rejected
Geometry_CSAAllocationSum, plus four declared new controls:
Uncentered keeps the same gain/d but does not subtract the correction mean;
Fixed sets gain1 for the same centered correction; GainOnly multiplies the
entire original patch-query read by1+g; HeadShuffle rolls the per-head gains
by half the head count, preserving their multiset, NOT correction norm budgets.
Controls are not retrospectively promoted or used for domain-specific routing.

Reuse the same developed264 keys: UDD5 full40, other seven domains32 each.
Fixed20 aliases/six RS templates/normalized LME .07/native512/overlap128
(step384)/Hann. Unchanged checkpoints, vocabulary, ontology, scoring/targets.
Masks enter only after whole-image prediction. All six old arms must replay
exact per-image confusion matrices from the completed CSA allocation study.

Advance only if primary mean exceeds Geometry, SCLIP_Two, VIPProxy_Two,
Geometry_DoubleRow, Geometry_CSAAllocationSum, GainOnly, Fixed and HeadShuffle;
VDD/Potsdam no loss versus Geometry; every protocol including LoveDA P loses
at most1pp; independent512-window median latency <=1.10x Geometry. LoveDA D
counts once. Foreground/non-residual and class P/R/area are reported separately.
No automatic full rollout or claim of significance from the advancement gate.

Twelve synthetic invariants and real-checkpoint fp32/bf16 mask-free exact
replay/finite/singleton/frozen-weight checks precede label scoring. Schedule
only idle GPUs0-7. Preserve partial outputs on any failure. Paired uncertainty
is conditional on reused development and correlated filename source groups.
