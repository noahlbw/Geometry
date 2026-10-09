# Geometry-balanced SAT feature transport: prospective protocol

2026-10-02. One exploratory complete candidate; not a successful final model or a novelty claim.
Physical A800 GPUs 4-7 only. Original Geometry and historical best results are preserved.

## Architecture and question

Frozen LVD backbone produces native patch states L, prefix states and original Geometry relation G.
Frozen SAT-493M, with its own pretrained satellite normalization, observes the identical RGB512 patch grid S.
For each window, compute valid-patch incoming mass from G and set correspondence weights proportional
to its reciprocal (mean one). Solve weighted orthogonal Procrustes between centered S and L:

`min_R sum_i w_i ||(S_i - mu_S) R - (L_i - mu_L)||^2, R^T R = I`.

The transported SAT states `(S - mu_S) R + mu_L` replace valid native patch states. Invalid/padding patches
and native prefix inputs retain their original values. The unchanged two-block Geometry head reads these
states using the original G. Fixed20 aliases, six RS templates, normalized LME(.07), native512 windows,
128 overlap and Hann probability assembly stay unchanged. There is one primary output, no score gate,
class-specific calibration, alias pruning or dataset-specific method switching.

The fitting target is the actual pretrained backbone state, not its L2-normalized relation descriptor.
SAT and LVD network weights are frozen. Rotations and means are image-specific inference adaptation
states; this is not adaptation-free. It uses an additional pretrained aerial backbone. Orthogonal
Procrustes and inverse-density weighting are established operations, not claimed inventions.
Alignment uses FP32 SVD (CUDA gesvd)/matmul with TF32 disabled only inside the fit; original readout
precision is restored. The high-dimensional centered covariance is rank-deficient; the default CUDA
Jacobi SVD failed the real-checkpoint orthogonality check, so the more accurate standard driver is used.
SAT is instantiated through the pinned official factory with SAT-specific untied norms and an explicit
strict local checkpoint load, bypassing the project's text-only hub wrapper.

Hypothesis: aerial features add useful local distinctions; original Geometry can balance which matched
correspondences establish the bridge into the language-aligned head. Coherence, lower fitting error or
small rotations do not establish semantic correctness. A1024-patch/1024-channel fit is high-dimensional;
overfitting native errors, imperfect coordinate semantics and the additional backbone cost are risks.

## Complete readout arms

Primary `Geometry_SATTransport`; controls `Geometry`, `SCLIP_Two`, `VIPProxy_Two`, `SAT_Relation`,
`SAT_Unaligned`, `SAT_UniformTransport`, and `MeanLogit_SATTransport`. The last is the equal mean of
Geometry and uniform-transport class logits. Uniform transport uses the same SAT/LVD observations,
head and text but no Geometry correspondence balancing. All arms use identical alias sets and views.
SCLIP/VIP controls are matched operator adaptations, not complete official reproductions.

## Execution checks before metrics

Real-checkpoint FP32 and bf16 smoke must show exact zero error for same-source head recovery,
original matched-head features, native cached states, repeat Geometry and padding-state retention.
All features must be finite, grids match and SAT weights remain frozen. No target mask is loaded.

## Fixed eight-domain screen and promotion gate

Choose eight images per dataset (full40 for UDD5), preserving reference order after fixed-seed
`20261002` sampling of the verified complete sample sequence. Total96 unique images. No mask is used
to select the samples. Reference per-image confusion arrays reconstruct the identical screened
Geometry/SCLIP/VIPProxy metrics and must match bit-for-bit. Full counts: LoveDA1669, UDD540, OEM384,
VDD80, Potsdam504, Vaihingen113 with corrected IRRG inputs, LandCover.ai1602, FLAIR-115700.

Predeclared promotion gate, before viewing any candidate metric:

- All eight screens complete with unique coverage, matching vocab/checkpoints and exact baseline confusions.
- Equal-domain primary mean exceeds Geometry, VIPProxy_Two, uniform transport and mean-logit transport.
  LoveDA D counts once in this mean; P is also reported and checked for regression.
- No reported protocol loses more than1.0 percentage point versus Geometry.
- VDD and Potsdam primary each exceed Geometry and same-source mean-logit transport.

Passing permits full evaluation with exactly the same implementation/config; it does not establish SOTA.
Failure rejects promotion of this candidate, not all training-free methods. Do not change the gate or fit
parameters from these results. Dataset-level mIoU is the gate; beneficial/harmful pixels are diagnostics,
not an interchangeable objective. All eight datasets informed earlier development: none is untouched.

## Reporting and publication boundaries

Save complete per-image confusions, original/new/truth transition counts, per-class IoU/P/R/area,
alignment diagnostics, runtime and peak allocated memory. Full rollout uses the same GPU4-7 queue and
requires exact reference coverage and all three baseline confusions. A gain over a weak unaligned SAT
control is insufficient; Geometry balancing must improve over uniform transport and simple fusion.

Historical official-configured VIP VDD52.0647 and Potsdam44.0643 are distinct protocol references, not
matched-control victories. Invalid historical black-input Vaihingen results are excluded. iSAID lacks
semantic masks here, so LandCover.ai is the eighth labeled dataset. Standalone inference costs must
separate primary execution from the multi-arm evaluator before any efficiency claim. No CVPR readiness
or originality claim follows merely from implementing this candidate.
