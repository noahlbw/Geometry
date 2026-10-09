# Stratified, branch-aligned soft alias protocol

Signature: geometry-stratified-wide-alias-v2-20261002. Primary: StratifiedAlias_Coupled.

Prospective development screen: the same96 unique complete images as soft-v1, UDD5 full40 and seven domains8 each. LoveDA P/D share images; equal-domain means count D once. Corrected IRRG Vaihingen. LandCover.ai substitutes for unlabeled iSAID. All eight domains are development data. Physical A800 GPUs0-7. No automatic full rollout or target-label parameter tuning.

## Frozen rule

- Original Geometry, checkpoints, vocabularies (exactly20 aliases/class), templates, broad observer, tiling and reconstruction remain matched to soft-v1.
- First collect image-wide reference features using nonoverlapping512px Geometry views. No masks are loaded. Raw normalized backbone tokens provide feature similarity; original Geometry class evidence provides local witness scores.
- For each tested alias, remove it from local normalized-LME and broad salience/readout aggregation. Reference winners must agree between these two leave-out readouts. Their geometric-mean margin confidence ranks references. Remaining correlated aliases can still self-confirm.
- For each alias and class, retain the best witness in each fixed64px spatial cell, then up to64 cells by confidence. The query's cell is excluded. Cells are a spatial de-duplication approximation, not independent semantic objects.
- Current original Geometry top2 specifies A/B. Independently retrieve up to32 references from each class pool using frozen feature/position structural logits (temperature0.10, spatial sigma0.25; positions normalized over the image).
- Weight references by structural similarity and witness confidence, independently normalize the two pools, and require4 effective references per side. There is no additional penalty for the relative total area/mass of A versus B.
- Utility is the difference of the signed broad-branch deletion marginal means across the two pools, standardized by reference variance/effective count, clipped to+-2, and attenuated by geometric-mean reference confidence.
- Mix half uniform alias allocation with half utility softmax. Every alias remains present and weights are at least0.025 for20 aliases.
- Within each broad crop, multiply original per-class salience by reliability weights, renormalize, and run the original count-scaled nonlinear aggregation. Compose the original21-to336 interpolation, crop-overlap mean, and query sampling exactly; uniform stencil replay tolerance1e-4, exact uniform correction fallback.
- Recompute the broad A/B margin, limit its change to0.5*tanh(delta/0.5), preserve the pair partition mass, and feed the corrected observation into the unchanged original-Geometry anchored reconstruction.
- One within-class shuffled reliability control has an exactly matched weight spectrum, seed20261002. All configuration values are common to all datasets and fixed before evaluation.

## Arms and checks

Geometry, BroadVIP (matched20 fixed profile), Anchored_VIP (internal unscreened coupling), PairAlias_Coupled (historical hard selector), SoftCounterfactual_Coupled (soft-v1), StratifiedAlias_Coupled (primary), ShuffledStratified_Coupled.

Five historical arms must replay exact per-image confusion matrices. Independently verify unique complete sample coverage, unchanged sources/configuration, per-image confusion sums, transition endpoints, weight spectra and pair partitions. Mask-free smoke checks frozen heads, raw uniform broad replay and exact uniform fallback.

Record mIoU/per-class IoU, P/D, foreground/non-residual metrics, car/vehicle area and precision/recall, road/low-vegetation coverage, direct beneficial/harmful changes, reference support/trust, weight divergence/effective count, reference preparation cost, combined wall time and peak allocated CUDA memory. Combined seven-arm runtime is not standalone inference latency.

Predeclared promising gate: mean gain>=0.1 percentage points over unscreened coupling, positive mean over matched shuffled weights, wins on at least5/8 domain protocols (LoveDA D once), and worst loss<=1 point across all protocols including LoveDA P. This is a research scheduling criterion, not a statistical significance test or SOTA guarantee. No retrospective parameter change.

The experiment changes reference retrieval, spatial de-duplication and branch alignment together. It tests one coherent candidate; a gain would still require later contribution analysis. Borrowed VIP broad features/readout remain disclosed. Agreement of frozen predictors does not guarantee correct semantic witnesses. This module can address competition among detected classes but cannot guarantee recovery of targets missing from both readouts.
