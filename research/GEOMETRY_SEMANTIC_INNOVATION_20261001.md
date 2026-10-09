# Geometry semantic innovation: fixed experiment

## Hypothesis

The verified 8-image observer diagnostic establishes useful additional semantic content, not a reliable semantic router. VDD Geometry/VIP-wide/MeanProb50: 46.5339/52.6237/55.5817. Potsdam: 42.5695/44.8563/45.3252. Native local scores are substantially worse; they are not treated as a teacher.

Let g be local Geometry class logits, b a broad semantic observation, and A the original raw-DINO Geometry relation. Test the single coupled readout:

`z = (g - A g) + A b = g + A(b-g)`.

This retains local evidence outside the support readout and replaces the supported semantic component. It is neither output routing nor a convex probability average. It is a hypothesis, not evidence that A identifies true semantic classes. A can propagate errors.

## Frozen Protocol

Local Geometry: unchanged 512/128, depth 2, RS text, normalized alias LME .07, historical Hann probability blending. Exactly the existing 20 aliases per class. Padding donors excluded from innovation; invalid queries unchanged.

Broad observation: 448 long edge, 336/112 windows, pinned ImageNet templates/profile scorer. Fixed tau=1, salience temperature=1, logit scale=40, no background threshold on every dataset. Two information-source arms: original Geometry head and pinned VIP head. The VIP arm is explicitly borrowed, not a claimed new contribution. Separate frozen model copies preserve original Geometry precision and weights.

For each source compare broad alone, MeanProb50, MeanLogit50, and innovation. No label-based alias deletion, class weights, gates, dataset-specific parameters or threshold search. Labels loaded after predictions. Existing datasets are exploratory development.

First run the same VDD/Potsdam 8-image sequences. Verify original Geometry confusion matrices exactly. Full VDD80/Potsdam504 follows if the mechanism is useful on both; transfer unchanged to the six other datasets rather than choosing per-domain winners. Do not claim a CVPR-ready contribution from a small-set gain or from beating Geometry while losing to same-source fusion.

## V1 Result And V2 Hypothesis

V1 direct VIP innovation gives VDD54.0748 and Potsdam46.0530, versus same-source MeanLogit56.3759/45.4451 and MeanProb55.5817/45.4713. The own-Geometry observer is weaker: innovation42.7126/38.1568, below original46.5339/42.5695. The new content cannot be replaced merely by changing Geometry's view/text profile.

The V1 Geometry control had a tiny numerical deviation (confusion L1 mass26/4) because temperature division moved before interpolation. V2 restores the original interpolation-before-temperature order; V1 is not an exactly matched control despite unchanged rounded mIoUs. Preserve V1 outputs.

V2 adds one predeclared fidelity-constrained alternative, uniformly across domains:

`min_z 0.5||z-g||^2 + 0.5||A(z-b)||^2`.

Equivalently `(I+A^T A) delta = A^T A(b-g)`. This is a fixed unit-weight quadratic readout, solved by conjugate gradients with purely numerical tolerance. Geometry-nullspace local detail is unchanged; inconsistent semantic observations cannot fully replace local scores. Energy reduction proves numerical correctness, not segmentation correctness. Compare both source arms and both fusion controls; do not present this solver as a training loss or a proven semantic reliability mechanism.

## Verified V2 Diagnostic

Original Geometry confusion matrices exactly match both prior diagnostic runs. Unique 8/8 coverage per dataset and all transition tensors reproduce both endpoint confusion matrices. Eleven CPU tests pass, including comparison to a direct linear-system solve. The numerical solver reaches about 5e-7 relative residual in about 8 iterations.

| Method | VDD, 8 images | Potsdam, 8 images |
|---|---:|---:|
| Geometry | 46.5339 | 42.5695 |
| BroadGeometry | 41.8094 | 36.7048 |
| BroadVIP | 52.6237 | 45.0377 |
| MeanProb_Geo | 48.9626 | 39.7668 |
| MeanLogit_Geo | 50.7970 | 40.5133 |
| Innovation_Geo | 42.7126 | 38.1568 |
| Anchored_Geo | 51.1507 | 41.4715 |
| MeanProb_VIP | 55.5817 | 45.4713 |
| MeanLogit_VIP | 56.3759 | 45.4451 |
| Innovation_VIP | 54.0748 | 46.0530 |
| Anchored_VIP | 57.3350 | 45.8940 |

Anchored_VIP beats the stronger same-source simple fusion by 0.9591/0.4227 points on these diagnostic sets. VDD wall/vegetation/roof/water benefit versus logit fusion; car IoU is not better on Potsdam. Therefore do not claim this has solved small-object recognition. The weak own-Geometry broad semantic stream still harms Potsdam.

Freeze Anchored_VIP as the primary eight-domain candidate; no per-dataset winner selection. Preserve direct innovation and own-Geometry-source arms as information/mechanism controls. Full batch: GPUs0-3 evaluate VDD, Potsdam, UDD5, OEM, Vaihingen, LandCover.ai, LoveDA sequentially, four shards each; GPUs4-7 evaluate FLAIR-1, four shards. Each batch stops on a failed shard, retains logs, and merges only verified complete coverage. No other jobs are stopped. Eight-domain outcomes, not this small screen, determine whether the mechanism survives.

## Full-Run Numerical Failure And V3

FLAIR-1 V2 shards stopped before producing valid results: `ValueError: Scores and nonnegative relations must be finite.` A label-free reinspection of the first sample confirms finite RGB, text queries, local Geometry features/relations, and broad Geometry logits, but nonfinite broad VIP logits. Pinned proxy masks `similarity <= 0` after subtracting 1.5 times global mean similarity; an empty support row softmaxes all negative infinity.

V3 preserves the anchored rule and all parameters, adds self-Value fallback only when this proxy support is empty, and records empty-row counts. This is a numerically guarded borrowed observation, **not identical official VIP** on empty rows. Normal rows use unchanged pinned equations. Do not pool V2 and V3 results; retain V2 failure logs and any completed V2 dataset. The full suite is restarted under new V3 paths, not over existing outputs.

V2 VDD also exited on the nonfinite-score guard; its incomplete results are retained, not merged as full. The label-free failure-source inspection described above was on FLAIR, not every VDD failure. V3 full VDD reports zero fallback rows and matches the prior broad VIP profile mIoU; do not attribute a VDD gain to the fallback.

## Verified Full VDD80, V3

Coverage is complete and unique, all four signatures match, all transition tensors reproduce endpoint confusion matrices, and the global sample sequence and vocabulary SHA match the historical full Geometry run. Local Geometry remains38.8511. Original confusion matrix equality is checked separately from rounded metrics.

| Method | Full VDD mIoU |
|---|---:|
| Geometry | 38.8511 |
| BroadGeometry | 41.7671 |
| BroadVIP, guarded observer | 51.0697 |
| MeanProb_Geo | 47.9208 |
| MeanLogit_Geo | 48.6819 |
| Innovation_Geo | 42.5772 |
| Anchored_Geo | 49.3315 |
| MeanProb_VIP | 53.9073 |
| MeanLogit_VIP | 54.7118 |
| Innovation_VIP | 52.1727 |
| Anchored_VIP | 55.4446 |

Primary candidate improves Geometry by16.5935 points, but most of that is the additional semantic information: the attributable increment over the stronger same-source fusion is0.7328. The own-Geometry observer gives a0.6496 increment over its logit-fusion control. Do not assign the entire16.59-point gain to the new module.

| Class | Geometry | MeanLogit_VIP | Anchored_VIP |
|---|---:|---:|---:|
| other | 15.2481 | 34.8651 | 35.2730 |
| wall | 22.9675 | 46.7330 | 48.3011 |
| road | 48.5684 | 41.1802 | 41.8864 |
| vegetation | 70.0250 | 65.6924 | 66.5737 |
| vehicle | 9.4839 | 25.3907 | 26.2841 |
| roof | 65.8483 | 82.9585 | 83.6516 |
| water | 39.8168 | 86.1631 | 86.1421 |

Vehicle precision/recall: Geometry 9.5584/92.4087, anchored 27.8852/82.0715 percent. Water recall rises from 41.6736 to 95.0888 percent. Road and vegetation remain below the original Geometry despite improving over simple fusion. These are real cross-class tradeoffs.

Changes relative to Geometry: logit fusion beneficial204,597,210/harmful65,801,938; anchored beneficial202,140,156/harmful58,750,469. Thus anchored sacrifices2,457,054 beneficial changes but eliminates7,051,469 harmful changes versus the fusion counts, for4,594,415 extra correct pixels. This is aggregate comparison relative to Geometry, not a saved direct fusion-versus-anchored transition tensor.

Parallel wall441.6942 seconds, peak allocated CUDA5805.0752 MiB; timings cover all eleven arms together, not standalone candidate latency. Local result: `research/geometry_semantic_innovation_v3_vdd_full_20261001_results.json`.

## Verified Full Potsdam504, V3

Four shards uniquely cover all 504 tiles. Sample sequence, vocabulary SHA, class order and original Geometry confusion matrix exactly match the historical control. The tiles come from 14 parent scenes, not 504 independent scenes.

| Method | Full Potsdam mIoU |
|---|---:|
| Geometry | 40.6892 |
| BroadGeometry | 36.0078 |
| BroadVIP, guarded observer | 43.1513 |
| MeanProb_Geo | 37.9208 |
| MeanLogit_Geo | 38.2139 |
| Innovation_Geo | 37.0913 |
| Anchored_Geo | 38.5263 |
| MeanProb_VIP | 43.5224 |
| MeanLogit_VIP | 43.4407 |
| Innovation_VIP | 44.1683 |
| Anchored_VIP | 43.7334 |

The frozen primary Anchored_VIP improves Geometry by 3.0442 and stronger same-source fusion by 0.2110. Direct innovation is better on Potsdam but worse on VDD: **do not switch the primary method separately by dataset**. Primary car IoU rises 10.8915 to 26.7830, with recall effectively retained (99.2513 to 99.2291 percent); however low-vegetation IoU falls 39.9270 to 34.1710 and recall 41.4373 to 35.0748. The broad semantic stream still loses genuine low-vegetation coverage.

Changes relative to Geometry: MeanProb beneficial 46,494,936/harmful 25,870,243; anchored beneficial 42,445,912/harmful 20,724,302. Again, it gives up some beneficial changes to reject more harmful ones. Parallel wall 176.7170 seconds, peak 5520.3848 MiB, combined eleven-arm workload. Guarded observation uses self fallback on 0.6448 percent of rows. Local result: `research/geometry_semantic_innovation_v3_potsdam_full_20261001_results.json`.

The remaining six full datasets are running/queued under the unchanged V3 rule. No eight-dataset aggregate or publication-ready claim is warranted until those results are verified. FLAIR-1's partial first-shard empty-support fraction is about 31 percent; this is a running diagnostic, not its final full rate or mIoU.

## Final full-domain audit, 2026-10-01

All eight original runs completed; LoveDA1669 was recovered by the path-only loader fix. Complete unique keys, vocabulary SHA and exact original Geometry confusion matrices were verified against historical controls, including both LoveDA protocols. This verifies inference consistency, not dataset correctness.

| Dataset/protocol | Images | Geometry | Stronger same-source simple fusion | Anchored_VIP | Delta vs Geometry |
|---|---:|---:|---:|---:|---:|
| VDD | 80 | 38.8511 | 54.7118 | 55.4446 | +16.5935 |
| Potsdam | 504 | 40.6892 | 43.5224 | 43.7334 | +3.0442 |
| UDD5 | 40 | 50.5553 | 48.8694 | 49.2117 | -1.3436 |
| OEM | 384 available | 44.6097 | 37.6302 | 37.8977 | -6.7120 |
| Vaihingen, corrected input | 113 | 48.8009 | 49.8567 | 51.4079 | +2.6070 |
| LandCover.ai | 1602 | 59.2340 | 62.1248 | 63.7049 | +4.4709 |
| FLAIR-1 | 15700 | 43.8071 | 42.0188 | 42.5381 | -1.2690 |
| LoveDA P | 1669 | 64.9557 | 62.7796 | 62.7149 | -2.2408 |
| LoveDA D | 1669 | 42.6589 | 40.1569 | 40.2519 | -2.4070 |

The fusion column is a diagnostic comparison against the stronger of fixed MeanProb50 and MeanLogit50, not a dataset-selected deployable model. LoveDA D foreground: Geometry44.8329, Anchored_VIP44.7484. Its tree IoU drops42.1096 to26.8155 and background29.6149 to13.2730. LoveDA P tree56.0994 to33.6546. UDD5 vegetation82.4937 to68.3846, road45.7856 to39.8083. OEM tree57.0767 to37.0260 and building62.8216 to45.4447. These outweigh gains in other classes. The frozen primary is **not a reliable cross-domain final model**.

VDD's large gain mostly reflects additional wide-view/ImageNet/guarded-proxy semantic content: BroadVIP51.0697, simple logit fusion54.7118, coupled55.4446. The incremental coupling gain is0.7328. Potsdam gains0.2110 over its stronger simple fusion. Geometry controls the reading support, but spatial support cannot certify semantic correctness. Reject V3 as a universal correction; retain its difficult-domain complementarity as evidence for future design.

### Vaihingen input invalidation and repair

An input audit found all113 historical prepared images were single-channel, with62 entirely black. Upstream `split_tiff` remapped both images and labels regardless of `is_label`. The old Geometry4.8494 and Anchored_VIP6.1667 scores, and other methods measured on that input folder, must not be used as valid remote-sensing results.

Rebuilt a new folder `/data/test/datasets/Vaihingen/preprocessed_corrected_20261001` from the original three-band IRRG sources. Same113 keys and historical1000px offsets, unchanged band order, source-crop pixel equality and unchanged five-class masks were verified. Old files remain preserved. Corrected full Geometry48.8009, Multiscale47.8063, GEAR_OV48.6696. This jump is a data repair, not method progress. Manifest: `research/vaihingen_input_repair_20261001.json`; corrected baseline: `research/gear_vaihingen_input_corrected_full_20261001_results.json`.

The V3 corrected-input repeat is now complete: all113 unique keys, exact repaired Geometry confusion matrix, unchanged vocabulary and transition endpoint equality verified. BroadVIP47.4938, MeanProb_VIP49.6716, MeanLogit_VIP49.8567, Anchored_VIP51.4079. Coupling gains2.6070 over Geometry and1.5512 over the stronger simple fusion. Class IoUs, Geometry to Anchored_VIP: impervious surface49.9570 to61.3360; building73.3086 to68.0397; low vegetation40.6120 to33.2475; tree70.1372 to69.1603; car9.9897 to25.2560. This is a valid but mixed-class improvement, not a universal reliability claim. Local repeat: `research/geometry_semantic_innovation_v3_vaihingen_inputcorrected_full_20261001_results.json`. Across eight domains the primary improves four (VDD, Potsdam, corrected Vaihingen, LandCover.ai) and harms four (UDD5, OEM, LoveDA, FLAIR-1).
