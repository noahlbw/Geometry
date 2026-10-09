# Geometry-conditioned bounded semantic residual metric

Prospective complete candidate, not a final model or verified novelty claim.
Signature `geometry-residual-semantic-metric-v1-20261002`; primary
`Geometry_ResidualMetric`. Preserve original Geometry and all historical models.
Physical A800 GPUs0-7 only after occupancy checks.

## Hypothesis And Boundary

The previous matched fine/context Value innovation fails its fixed eight-domain
gate. It improves five domains but its mean is below Geometry and same-source
fusion, with major Potsdam/Vaihingen regressions. Learned view partition
functions did not establish semantic trust; its effective fine coefficients
could become negative. Do not repeat that operator with a new coefficient.

This candidate changes the semantic metric, not the visual encoder, attention
head or spatial class propagation. Image-only Geometry residuals might identify
directions sensitive to nuisance variation. They can also contain true class
differences and small-object detail, so that interpretation is explicitly a
falsifiable hypothesis. It does not supply an independent semantic observation
or certify which class/alias is correct. A constant wrong semantic mode is not
removed; this is not an estimated class intercept or globally centered feature.

Mahalanobis similarity, whitening and graph residuals have precedents. GEAR-OV
previously used fixed text-correlation whitening inside reconstruction; this
candidate instead estimates a bounded visual covariance from original Geometry
and jointly changes visual/text cosine. This implementation distinction alone
does not establish a new scholarly contribution or CVPR readiness.

## One Complete Readout

Frozen DINOv3/DINO.text -> original two-block Geometry -> normalized fine512
descriptors Y and original raw-DINO relation G -> bounded metric -> all20 alias
cosines -> unchanged normalized LME(.07) -> original Hann probability assembly.
One backbone call per original tile; no VIP/external teacher/context view.

1. Build an uncentered orthonormal basis B of the union of all original encoded
   aliases. Only numerical SVD rank tolerance1e-6 is used, not alias deletion or
   semantic rank selection. Text reconstruction error must be below2e-5. LoveDA
   P/D share one union basis, covariance and original Geometry observations.
2. Exclude padded donors from G and normalize image-valid rows. Empty rows act
   as identity, not absence evidence. Let U=YB and R=(I-G_valid)U on valid rows.
   C=R^T R/n_valid is positive semidefinite and uses no target labels.
3. Set M=I+C/trace(C). Inverse metric eigenvalues lie in[1/2,1]. Exact zero
   covariance, identity G or empty validity gives the original alias scores.
   Covariance trace below64*fp32_eps^2*mean projected squared norm is numerical
   roundoff and is treated as zero; this is not a semantic confidence threshold.
4. Apply M^-1 consistently to BOTH visual and text directions in the B span,
   retaining the orthogonal descriptor component, then normalize the resulting
   cosine. Cholesky triangular solves implement the same full-space metric.
   No neighbor descriptor/class score is written back. No class-specific weight,
   threshold, hard rejection, optimization solver or dataset winner routing.
5. Keep all20 aliases, six RS templates, uniform normalized LME temperature.07,
   tile512/overlap128/temperature-after-interpolation/Hann assembly unchanged.

The network and encoded text weights are frozen, but M is window-specific
inference adaptation, not adaptation-free inference. Test correctness invariants
do not guarantee final class preservation or better mIoU.

## Controls And Decision

Eight arms: Geometry, depth-matched SCLIP_Two/VIPProxy_Two, UniformMetric,
SpatialMetric, ShuffledMetric, MeanLogit_UniformMetric, primary. Uniform uses
global residual covariance from the same Y; Spatial uses the same sigma.25
Gaussian but no raw-DINO appearance relation; Shuffle permutes valid Geometry
rows/columns relative to fixed Y without moving padding into valid members.
The fusion control is the fixed half mean of Geometry and UniformMetric logits.
The nearest heads are attributed adaptations, not complete official systems.

After17 focused tests and a real-checkpoint mask-free smoke, use exactly the
previous96 COMPLETE-IMAGE IDs: UDD5 full40, other seven domains8 each. These are
not window-center statistics or eight full-dataset totals. Corrected IRRG
Vaihingen; LandCover.ai replaces unlabeled iSAID; all domains are development.

Require complete unique coverage, unchanged vocab/checkpoints/config and exact
per-image Geometry/SCLIP/VIPProxy replay against full matched references.
Per-image sums and old/new/truth transitions must reproduce endpoint matrices.
Labels are loaded only after predictions. No metric strength, span rank, class
rule or covariance estimator is selected from results.

Promotion requires retaining every Geometry protocol, improving the eight-domain
mean over Geometry, both nearest heads, Uniform/Spatial/Shuffle and same-source
fusion, and beating Geometry/fusion on VDD and Potsdam. LoveDA D enters the mean
once; P is a separate retention check. A pass permits unchanged full follow-up,
not SOTA or CVPR completion. Failure retains outputs and rejects this fixed rule
without retrospective tuning or per-domain switching.

Report per-class IoU/P/R/area, true/false-positive changes, transition counts,
covariance/alias-change diagnostics and independent one-arm descriptor cost.
All-arm evaluation timing is not deployment latency. The larger research goal
of a useful original complete coupled model remains unachieved until verified.
