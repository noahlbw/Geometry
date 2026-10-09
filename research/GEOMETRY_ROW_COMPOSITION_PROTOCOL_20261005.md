# Geometry native-row composition: prospective development protocol

This study follows the positive, previously developed DonorBefore control. That
control is not retroactively promoted or claimed original. Preserve original
Geometry, best coupled models, historical outputs and paused automations.

## Hypothesis and fixed candidate

Original patch read: `special_i + m_i * sum_j G_ij V_j`.
Primary: `sum_j G_ij (special_j + m_j V_j)` before unchanged projection,
LayerScale, query residual and MLP. Native prefix queries remain native at each
evolving input. Both frozen DINO.text blocks use the same raw Geometry relation.
Effective special weights are `G @ A_special`; patch weights are `G_ij*m_j`.
Whole rows conserve probability mass in exact arithmetic. Native allocation
`m_j` is NOT a semantic correctness/reliability estimate.

FP32 native softmax, normalized Geometry rows and FP32 read are shared by all
new factor arms; attended features are cast to native Value dtype before the
unchanged projection. Geometry_FPRead isolates these numerical differences.
Zero own special groups use uniform special-key conditional; zero mixed special
groups use own conditional; zero mixed patch groups use original G conditional.
No fitted threshold, quota, text changes, second encoder, view or additional MLP.

## Controls

Exact historical Geometry, Geometry_BlockPrefix, SCLIP_Two, VIPProxy_Two;
Geometry_FPRead; primary Geometry_RowCompose; BudgetOnly; SpecialOnly;
GroupCompose; ConditionalOnly; GlobalBudget; exact prior Geometry_DonorBefore.
The budget/special 2x2 keeps original patch conditional G. Full versus
GroupCompose tests donor conditional reweighting at composed group allocations.
This is not a complete three-factor interaction decomposition.
GlobalBudget uses crop/head mean native patch mass, own conditional special
content and original G. Prior DonorBefore is an attributed developed control;
it is algebraically related to the primary, not an independent novel method.

## Coverage and locked gate

UDD5 uses all40 already-developed validation images. Each other domain uses32
images, excluding the exact8 keys from the previous96-image panel. Total264.
Choose from full reference key lists without opening masks or reading metrics,
round-robin among shuffled source/geography groups with seed20261005. Retain
original reference ordering. Parent groups may overlap prior panels. Prior full
results have already been inspected. This is broader mechanism confirmation,
NOT untouched independent validation.

Fixed20 aliases/class, six RS templates, normalized LME .07, native512 crops,
overlap128 (nominal step384), Hann probability blending; masks only after whole-image prediction.
Verify264 unique keys, four exact historical per-image confusion references,
fixed checkpoint/vocabulary/config identities and per-image confusion sums.

Advance only if primary eight-domain mean exceeds Geometry, FPRead,
BlockPrefix, SCLIP_Two, VIPProxy_Two and GlobalBudget; VDD/Potsdam do not decline
versus Geometry; every protocol including LoveDA P loses at most1pp versus
Geometry; independent singleton median latency is at most1.10x Geometry.
LoveDA D counts once in eight-domain mean. Report LoveDA P separately, fixed
scored-class metrics and foreground/non-residual metrics. Do not select winners
per dataset, rename factor controls as successful primary or automatically roll
out to full data. Cost: five warmups/15 synchronized512-window repeats with only
one backbone and the selected head, diagnostics disabled.

## Mechanism and publication interpretation

Report per-class IoU/precision/recall/predicted area, beneficial/harmful changes,
native/moved patch mass variation and whole-row mass error. Compute the
budget/special finite-difference contrasts, explicitly acknowledging nonlinear
two-block state evolution and mIoU nonlinearity. Do not interpret them as pixel
causal decomposition. In particular inspect VDD vehicle/water, Potsdam
car/low-vegetation and UDD5 road/vehicle tradeoffs, not just domain means.

Composition has broad precedents; external novelty has not been established.
Any positive development finding requires a separately locked larger/full
study and fair nearest-operator/coupling comparisons. CVPR acceptance and SOTA
are not implied by a passing gate.

Documentation correction: the initially deployed protocol.md called overlap128
"stride128". The unchanged evaluator calls tile_starts(length,512,128), whose
third argument is overlap. Actual inference and all frozen rules are unchanged.
