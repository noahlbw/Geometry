# Frozen Patch-Only2: Same-Field Coupling Audit

The full frozen candidate already completed 20092 images on eight domains.
Its equal-domain mIoU is 47.6399 versus 44.0039 for the stronger of the two
already measured VIP query protocols. All eight domain mIoUs and both LoveDA
protocols exceed that comparator. This does not establish every-class
superiority, published-benchmark SOTA, or originality of the coupling.

## Fixed Source And Controls

Keep original20 aliases/class, checkpoints, Geometry896/wide448 inputs,
maximum four Geometry and four wide encodings, text banks, temperatures,
padding, interpolation, probability stitching and original-size argmax.
No native-resolution or fine RGB encodings, alias admission, learned or fitted
coefficient, domain routing, or post-result winner switching.

One local patch-only2 head produces L; the existing VIP observer produces B.
W is the original valid-token Geometry relation, never the doubled head read.
H = (I + W^T W)^(-1) W^T W. The primary remains Z = L + H(B - L).

| Arm | Score Field |
| --- | --- |
| Geometry | Original local Geometry, retained endpoint |
| NoAdmission_Exact | Original local head with original reconstruction |
| Geometry_PatchOnly2Coupled | Exact frozen full-suite candidate |
| PatchOnly2_Local | L |
| WideOnly_Lifted | B, with the same output grid and stitching |
| MeanLogit_PatchOnly2 | (L + B) / 2 |
| PermutedH_PatchOnly2 | L + P^T H P(B - L) |

P permutes valid tokens only, with seed20261005, and leaves padded tokens
fixed. The permutation preserves H's spectrum, symmetry and padding; it
changes correspondence, not observations. Every arm uses identical L and B.
These are mechanism controls, not automatically promoted model candidates.
WideOnly_Lifted is not the standalone upstream VIP pipeline.

## Samples And Verification

UDD5 uses all40 images. Each other domain uses eight complete images at
indices round(i * (N - 1) / 7), i=0..7, of the existing full sample order.
There are96 unique images, not a full eight-domain benchmark. LoveDA P/D
share image IDs; only D enters the eight-domain mean. These are developed
domains, not untouched independent validation. No masks choose sample IDs,
controls, weights, aliases or thresholds; masks enter after prediction only.

CPU algebra tests cover the identity-ridge/equal-mean equivalence, local and
wide endpoints, duplicate-source equality, deterministic spectrum-preserving
permutation and unchanged padding. Mask-free VDD/Potsdam smoke checks require
original endpoints, singleton/all-arm equality and unchanged frozen weights.
The96-image primary per-image confusion matrices must match their existing
full-suite counterparts exactly, with identical vocabularies/checkpoints.

Compare the primary against the equal mean, local-only and wide-only scores,
then the correspondence permutation. A permutation advantage alone does not
show that coupling beats simple fusion. Report all eight outcomes, not only
the average or favorable domains. No rule/parameter change follows from this
panel; any later change requires a separate declared experiment.

The ridge solver, twofold read strength and inherited VIP observer are not
claimed as mathematical innovations. A positive result can support the use
of a shared Geometry relation for cross-view correction, not establish
priority or CVPR acceptance.
