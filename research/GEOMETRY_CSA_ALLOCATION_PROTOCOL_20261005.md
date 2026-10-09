# Geometry / CSA allocation: frozen mechanism study

This study addresses the unisolated difference between Geometry and the
strongest matched SCLIP_Two operator. It is not an original CSA claim or a
validated CVPR module. Keep original/best models, prior outputs and automations.

## Hypothesis and primary

SCLIP sums softmax(QQ*scale) and softmax(KK*scale), yielding row mass2.
Geometry retains native QK special contributions and native patch mass, yielding
row mass1. Previously moving conditional CSA into Geometry's shell did not
recover SCLIP's benefit; relation quality alone is therefore not established.

Primary Geometry_CSAAllocationSum uses the current frozen block's CSA special-key
coefficients and total CSA patch allocation, but replaces its conditional patch
distribution with the unchanged raw Geometry G. Prefix query rows stay native.
Patch read: `CSA_iS V_S + sum(CSA_iP) * G_i V_P`.
Keep frozen projection/LayerScale/query residual/MLP and evolving two-block state.
Each primary patch row sums to2 in exact arithmetic; it is NOT a probability row.
CSA is explicitly attributed to SCLIP, including its published sum convention.

## Disentangling controls

Exact historical Geometry, BlockPrefix, SCLIP_Two and VIPProxy_Two.
Geometry_DoubleRow doubles original patch-query Value read before projection,
not projection bias/residual or native prefix-query read.
CSA_NativePrefixSum/Mean use full CSA patch-query rows at sum2/mean1 strength,
keeping native prefix queries. Geometry_CSAAllocationMean halves the primary
read. Geometry_CSAMassSum/Mean use CSA group allocations but native conditional
special content and unchanged G. Geometry_CSAConditional uses CSA conditional
patch relation with original native special content and native patch mass.
All controls run under their own evolving states; whole-system deltas are not
linear decompositions. Mean/sum controls do not change softmax temperature.

The new group-mass construction is not native mass as semantic reliability.
CSA group allocations are also not certified class correctness. Use no numerical
selection threshold; empty own special groups use uniform special conditional,
and empty CSA patch groups fall back to G. These are numerical conventions only.

## Locked data, protocol and decision

Reuse the exact264 image keys from geometry_row_composition_screen_20261005:
UDD5 full40, each other domain32. They exclude prior exact8 keys except UDD5,
but the264 results have now informed development. NOT untouched validation.
Fixed20 aliases/six RS templates/normalized LME .07/native512 crops/overlap128
(nominal step384)/Hann probability assembly. Keep all checkpoints and settings.
Masks enter only after whole-image prediction. No domain-specific arm selection.
Verify exact four historical per-image confusion controls and264 unique images.

Primary advances only if its eight-domain mean beats Geometry, SCLIP_Two,
VIPProxy_Two, CSA_NativePrefixSum and Geometry_DoubleRow; VDD/Potsdam retain
Geometry; every protocol including LoveDA P loses at most1pp versus Geometry;
independent deployed median window latency is at most1.15x Geometry. LoveDA D
counts once; foreground/non-residual/class tradeoffs are separately reported.
Five warmups/15 synchronized timing repeats, one backbone and one selected head,
no diagnostic/control heads. No automatic full rollout or control promotion.

Before labels: small synthetic invariants plus mask-free actual-checkpoint
fp32/bf16 replay, unchanged weights and finite singleton outputs. Schedule only
idle GPUs0-7, preserving unrelated sessions. Report failures and safe recovery.

## Publication boundary

The aim is to decide whether the nearest-method gap concerns conditional donor
relations, semantic key-group allocation, prefix evolution or Value-read strength.
It does not establish priority for attention modification, CSA or simple scaling.
If a constant multiplier explains a gain, report calibration rather than an
invented semantic-reliability mechanism. If replacing CSA patch conditionals
with G is helpful under identical CSA allocation, it is evidence for Geometry's
conditional relation, still requiring broader fixed/full and independent tests.
