# Native-row composition: mechanism and claim boundary

Candidate implementation: geometry-native-row-composition-v1-20261005.
This is a prospective development candidate, not a validated final model.

## Derivation

For each frozen head block and attention head, native attention A assigns
patch query i special-key weights a_i and patch mass m_i. Geometry G is a
row-normalized raw-feature/spatial relation shared by attention heads.
Let D_self have special weights a_i and patch weight m_i only on its own patch.
The candidate's patch attention rows are G D_self; special query rows remain A.

Original: o_i = a_i V_S + m_i sum_j G_ij V_j.
Candidate: o'_i = sum_j G_ij (a_j V_S + m_j V_j).
Thus delta_i = sum_j G_ij a_j V_S - a_i V_S
               + sum_j G_ij (m_j-m_i) V_j.

Effective group allocations: m'_i = sum_j G_ij m_j;
a'_i = sum_j G_ij a_j. Since every D_self row sums to1 and every G row sums
to1, the composed row sums to1. Conditional patch weights are G_ij*m_j/m'_i,
not simply G_ij. The unchanged projection, LayerScale, query residual and MLP
follow this read; both blocks recompute native A and Values from evolving states.

The change is not post-head smoothing, a second semantic encoder or an added
MLP. It is also not unrelated to the developed DonorBefore control: transporting
its projected self increments approximately commutes with affine projection
under normalized G. Finite-precision normalization and casts differ explicitly.

## Important non-invariance

Adding a common vector to EVERY Value leaves candidate-minus-original unchanged,
because both total rows sum to1. Adding a vector b to PATCH Values alone changes
that difference by (m'_i-m_i)b. Unlike the original conditional replacement,
moving group allocations can modify common patch-semantic directions. It is
not guaranteed to remove bias or preserve all true coverage. This is an
analytical risk, not evidence that native m is semantic reliability.

G is row-stochastic, generally not column-stochastic. It preserves local total
row mass, but not necessarily crop-wide average patch allocation or class area.
Text competition, final normalization, evolving Q/K/V and MLPs remain nonlinear.

## What must be demonstrated

The primary must beat FPRead and closest matched operators under one frozen rule,
not merely the native head. Factor contrasts must distinguish budget movement,
special-content movement and conditional donor weighting. Car/vehicle precision
gains must not conceal excessive water/low-vegetation or road coverage losses.
Independent singleton cost is measured separately from multi-arm evaluation.

The broader264-image panel is developed confirmation, not untouched validation.
Any full evaluation and coupled-model replacement needs a new locked decision;
do not promote a control or select a different arm per domain. Row composition
has broad precedents; do not claim a new principle merely by naming G D_self.
External novelty verification and independent evaluation remain open.
