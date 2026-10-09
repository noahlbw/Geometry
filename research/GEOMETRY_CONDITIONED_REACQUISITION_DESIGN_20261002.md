# Geometry-conditioned feature re-acquisition: prospective design

2026-10-02. The prospective architecture below is implemented and its fixed full eight-domain trial is complete. It does not achieve the intended eight-domain superiority. This is not a verified successful final model or an originality claim. The original Geometry and historical coupled models remain unchanged.

## Implementation and completed verification

- Model: `DINOtool/dinotool/geometry_reacquisition.py`; implementation `geometry-native-donor-reacquisition-v1-last4-20261002`.
- Evaluator: `DINOtool/scripts/eval_geometry_reacquisition.py`; primary `Reacquired_Geometry`.
- Controller: `DINOtool/scripts/run_geometry_reacquisition_suite.py`; physical A800 GPUs4-7 only.
- Full root: `results/geometry_reacquisition_full_20261002`; controller session `grq02_suite`, workers `grq02_DATASET_sN`.
- Local collection: `research/geometry_reacquisition_full_20261002`; collector `tools/collect_geometry_reacquisition.py`.

Fourteen local tests passed, including eight new tests against actual pinned upstream blocks and six existing matched-readout tests. The eight new tests also passed on A800 using `unittest` (its environment does not have pytest installed). Real-checkpoint FP32 and bf16 execution checks on an original UDD5 tile both give exactly0 maximum error for uniform-relation backbone replay and the resulting original Geometry head. Capturing the native stream also changes original Geometry features by exactly0. No target mask was loaded for this execution check.

The implementation uses `native_attention_output + native_patch_mass * (conditioned_patch_read - native_patch_read)`. This is algebraically the special-path-preserving definition below and retains the original fused attention output when displacement is zero. This numerical form passed the explicit special/patch formula test; it is not a new inference objective or a score fusion.

Full evaluation arms: Geometry, SCLIP_Two, VIPProxy_Two, Reacquired_Native, MeanLogit_Reacquired and Reacquired_Geometry. The mean-logit arm uses the same original Geometry and reread/native-head observations. No arm is selected per dataset. All shards save per-image confusion matrices and original/new/truth transitions. The controller requires exact original confusion matrices for all three matched baselines, unique complete sample coverage, identical checkpoints and vocabulary, and exactly20 aliases per class.

All20 shards completed without failures in2725.909 seconds on physical GPUs4-7. Unique complete coverage, checkpoint/vocabulary/sample identity, per-image confusion sums and exact replay of all three earlier matched baselines were verified on all eight datasets. No rule was changed after observing an outcome.

| Dataset/protocol | Images | Original Geometry | Reacquired_Geometry | Delta |
| --- | ---: | ---: | ---: | ---: |
| LoveDA P | 1669 | 64.9557 | 62.7304 | -2.2253 |
| LoveDA D | 1669 | 42.6589 | 41.1009 | -1.5580 |
| UDD5 | 40 | 50.5553 | 49.5202 | -1.0351 |
| OEM | 384 | 44.6097 | 45.5864 | +0.9767 |
| VDD | 80 | 38.8511 | 38.5793 | -0.2718 |
| Potsdam | 504 | 40.6892 | 43.4517 | +2.7625 |
| Vaihingen, corrected IRRG | 113 | 48.8009 | 47.6738 | -1.1271 |
| LandCover.ai | 1602 | 59.2340 | 58.7913 | -0.4427 |
| FLAIR-1 | 15700 | 43.8071 | 43.5633 | -0.2438 |

The primary wins2/8 versus Geometry and2/8 versus matched VIPProxy_Two. Equal-domain mean, counting LoveDA D once, is46.0334 versus Geometry46.1508, SCLIP_Two47.2215 and VIPProxy_Two46.8876. It wins7/8 against its same-source mean-logit control, but that does not make it better than original Geometry. Historical upstream-configured VIP references remain stronger on VDD(52.0647) and Potsdam(44.0643).

The completed independent cost test on idle physical GPU4 confirmed exact equality of standalone and evaluator primary features. On one1024x1024 LoveDA image, Geometry/new median times are0.323385/0.405549 seconds(1.2541x), with22.1250/30.7006ms windows(1.3876x). Peak allocated image memory is3605.347/3625.570MiB. This is a one-image deployed comparison, not full eight-domain throughput.

Potsdam's gain mainly comes from low vegetation(+9.4217 IoU) and tree(+5.5474); car FP falls8,061,616 while TP falls6,798. VDD instead gains12,186,403 vehicle FP and loses2.0190 vehicle IoU, while water recall also declines. Thus feature re-acquisition is not a domain-independent semantic correction mechanism. Preserve these negative outcomes; do not tune depth/strength per dataset or promote this candidate as the final model.

Full outputs, per-class errors, paired bootstrap comparisons, diagnostics and costs are recorded in `research/GEOMETRY_REACQUISITION_EIGHT_DATASET_20261002.md` and `research/geometry_reacquisition_full_20261002/analysis.json`. The matched proxy and official-system VIP comparisons remain distinct; the invalid historical black-input Vaihingen VIP result is excluded from victory claims. All current domains are development data, not independent validation.

## Decision and research question

Retain Geometry as the first structural component. The next complete candidate should test whether its relations can improve the formation of the actual patch features entering the language-aligned head, rather than certify an already computed semantic score.

**Research question:** Can a Geometry-constrained second reading of frozen backbone content reduce cross-support semantic interference while preserving native scene pathways, correct local detail and background competition?

This is a hypothesis, not a diagnosed universal cause. Earlier intermediate probes passed through the final head normalization/projection and do not prove that backbone attention causes the errors. The candidate changes the information-processing path, not the input pixels, model weights or amount of independent semantic evidence.

## Evidence motivating the target

The complete matched audit covers 20,092 images across eight datasets. Geometry's equal-domain mean is 46.1508, versus 47.2215 for SCLIP_Two and 46.8876 for VIPProxy_Two. These are matched DINO.text operator adaptations, not complete official systems or SOTA rankings.

Potsdam car precision/recall are 10.9005%/99.2513%; VDD vehicle precision/recall are 9.5584%/92.4087%. VDD water recall is 41.6736%, and Potsdam low-vegetation recall is 41.4373%. Therefore the objective must address both excessive false activation and lost correct coverage. It cannot be expressed as suppressing every uncertain score or increasing every small-class response.

In the bounded prior support audit, 316/318 other-to-vehicle error centres retained the erroneous vehicle preference in Geometry neighbours. Geometry coherence is not semantic correctness. SigLIP2 RegionJoint beat its same-source fusion on all screened protocols, yet lost against Geometry on seven of eight primary dataset entries. A better inference solver does not make a weak observer reliable.

Relevant evidence: `geometry_publication_20261001/NOVELTY_AND_MECHANISM_AUDIT.md`, `geometry_publication_20261001/ERROR_ANALYSIS.md`, `GEOMETRY_REGION_SEMANTIC_READOUT_20261001.md`, and `FROZEN_SEMANTIC_PATH_GEOMETRY_20261001.md`.

## One candidate, two coupled functions

1. The normal frozen backbone provides its native token stream and the unchanged original Geometry relation G.
2. A separate stream of real patch queries rereads the last four backbone blocks using native cached donors and G. Its result enters the original Geometry head, using the same G and the original text/readout protocol.

The native donor/special stream never receives writes from the reread stream. There is one final segmentation output; the proposed primary does not blend original and reread logits or select an arm by dataset. Four replay blocks is a prospective common engineering choice, not a theoretically optimal depth or a target-label-selected value. Do not silently change it after examining results.

### Native pass

Run the complete original backbone once. For its last four blocks cache native K/V, the native patch-to-special attention contribution, native patch attention mass, and the starting real patch state. Preserve the original coordinates, RoPE, padding and checkpoint normalizations.

Compute the existing relation from final native raw patch descriptors:

`G_ij = softmax_j(cos(F_i,F_j)/0.10 - ||u_i-u_j||^2/(2*0.25^2))`.

Cache G once; do not recompute it from altered queries or predicted labels. Do not introduce alias selection, class thresholds or a global semantic subtraction in this candidate.

### Real-query replay

Let Z^l denote the cached native state and H^l the evolving reread patch state. Initialize H at the input to the last four blocks with the corresponding native patch state. For each attention head and replay block:

`B_ij^l = softmax_(j in patches)(q(H_i^l)^T k(Z_j^l)/sqrt(d) + log G_ij)`.

Use the exact pretrained attention scaling and positional transformation, rather than the shorthand dot product where the implementation differs. Let m_i^l and the special-key contribution come from the unmodified native pass:

`o_i^l = sum_(s in special) A_native,is^l V_native,s^l + m_native,i^l sum_(j in patches) B_ij^l V_native,j^l`.

Apply the original output projection, LayerScale, residual and MLP, evaluating the latter on the evolving H state in the original order. H changes; donor K/V and the native special contribution do not. Keep real query locations and their own residual features. No region receives a broadcast hard class.

Apply the original final backbone normalization. Feed the reread patch tokens and original native special tokens into the original Geometry head with the same fixed G. The head's own states, attention masses, residuals and MLPs then evolve as in the original Geometry implementation. Prefix preservation in the replay is not a guarantee that final background scores remain unchanged.

### Text and output

Keep each dataset's existing fixed20 aliases, six templates, normalized log-mean-exp and window assembly exactly unchanged. Text is the final open-vocabulary recognition interface. The present candidate does not claim to solve malformed class-to-alias assignments such as assigning roof to wall. Introducing an alias selector now would confound the test of feature formation.

Geometry has two explicit roles: its relation constrains the upstream query's content acquisition and also defines the downstream semantic-head patch reading. This is a two-pass acyclic computation, not a converged self-training loop.

## Invariants and limits

- A uniform relation in the replay must exactly recover the native backbone and original Geometry output, subject to a documented numerical tolerance for changed kernels. This follows by induction: native initial queries plus native donors reproduce native conditional patch attention; native group masses and prefix contributions reproduce the full original attention output.
- Native donor and special-token caches remain identical to the first pass.
- All classes and all domains use the same replay rule and depth.
- There is no training loss and no learned new parameter. A row of B can be viewed as the solution of `min_p KL(p || a_query,native_donor) - sum_j p_j log G_ij`; this is a standard conditional attention formulation, not an IoU objective or a novelty claim.
- Keeping attention mass and a query residual constrains the intervention. Neither proves unchanged background calibration, correct labels, preserved boundaries or higher mIoU.

Expected positive mechanism, if the hypothesis holds: reduce attention to unrelated donors before the language head, while retaining supported donor detail. This could reduce coherent false car responses caused by upstream mixing and strengthen true water/vegetation content previously diluted by context.

Explicit failure modes: the class error already exists in the query/self features; donors inside G carry the same wrong semantic direction; G groups inappropriate content; the fixed special-token contribution contains the problematic cue; later semantic-head transformations still suppress the correct class; or restricted context removes genuinely necessary recognition information. The candidate has no independent semantic teacher that guarantees recovery in these cases.

## What differs from previous failures

FrozenPathGeometry changed feedback inside the two language-aligned head blocks. This candidate recomputes the actual features entering that head, with native donor/special caches and evolving real queries in late backbone blocks. It is not virtual region CLS, crop-global classification, erasure-score attribution, regional posterior fusion, or confidence gating of the same output.

Moving G earlier, dual streams, frozen caches and multiplicative attention priors are not automatically new ideas. A defensible contribution would require a reproducible failure mechanism, evidence that this conditional re-acquisition corrects it, and gains over nearest matched alternatives. Full literature priority remains to be checked before naming it a novel method.

## Objectives and a bounded complete-candidate evaluation

Final aspiration: one frozen model beats strong official full-system comparisons on all eight domains under reconciled protocols. This is a target, not a guarantee or something optimized directly without labels.

The next candidate should be evaluated as a complete architecture across all eight domains, with no per-domain winner selection. Reuse existing verified Geometry and matched near-neighbour results when inputs/protocols remain exact. A small common screen may reject a broken hypothesis, but its winner is not a final eight-domain result.

Required decisions:

1. Does the full eight-domain vector improve over Geometry, with a better equal-domain mean and no concealed material losses? Use LoveDA D once in the mean; report LoveDA P and D foreground separately.
2. Does it beat strong matched SCLIP_Two/VIPProxy_Two and the preserved historical coupled reference where relevant? A +0.2 gain over Geometry alone is insufficient against an existing +1.07 mean nearest-operator gap.
3. Is the extra computation useful beyond an equal-budget, predeclared ordinary rereading/augmentation control? Exact native replay is an implementation invariant, not a useful independent observation or TTA control.
4. Is there actual coupling? Keep the same reread features and remove the downstream Geometry relation; keep the downstream Geometry head and replace the replay relation with the matched spatial-only or nearest-operator relation. Report the interaction, not just individual gains. Use one small, predeclared control set rather than a broad parameter sweep.
5. Do class transitions show car/vehicle FP falling without erasing their TP, and water/low-vegetation TP increasing without uncontrolled FP? Preserve the LoveDA background audit. Measure actual instance/connected-component size and boundary errors before making small-object claims; a weak class IoU alone does not establish object-size failure.
6. Does independent deployed latency/memory remain reasonable? One native pass plus four replay blocks has a different cost from evaluating all arms together. A roughly 1.3x original runtime is a practical design aspiration, not a measured figure.

Use scene-paired uncertainty where metadata allows. Total beneficial-minus-harmful pixels is a pixel-accuracy change, not an mIoU objective. For a class with fixed target area T, IoU improves iff `DeltaTP*(T+FP) > TP*DeltaFP`. Report the complete confusion matrices, class areas and wrong-to-wrong transitions.

Do not change replay depth, relation strength or class-specific decisions using these results and then describe the same data as independent validation. All eight current domains have informed development. Freeze any successful method before a genuinely new region, split or official held-out test.

## Stop rule and the more radical alternative

If meaningful correction still fails to separate true from false semantic responses, do not add another gate/solver to the same reread. That outcome would reject this hypothesis; it would not mathematically prove that all training-free methods are at a ceiling.

A more radical future direction is Geometry-conditioned semantic explanation with a frozen conditional generative model. The model would score the denoising/reconstruction compatibility of the actual supported content under competing class text, using matched noise/timesteps and the same external context. This changes the pretrained observation objective, unlike another discriminative pooled-region observer. Diffusion classification itself has precedent; no originality or reliability is claimed here.

Such scores are compatibility proxies, not calibrated posteriors. Use the checkpoint's actual trained noise/inpainting protocol, not an invented partially noised latent distribution. Key obstacles are aerial/IRRG domain mismatch, tiny-object latent resolution, text-class energy calibration and support-by-class-by-noise cost. It is a separate high-risk research route, not an extra block to stack into the next candidate.

Paired context swaps are also not a free calibration solution: their measurements constrain `(h_r-h_s)^T(Z_c-Z_d)`. They leave classwise constant offsets unidentifiable. Setting `h_r^T(Z_c-Z_d)` equal to those relative measurements would repeat the invalid conversion of influence into an absolute semantic posterior.

## Paper positioning

The prospective story is: structural relations should constrain how local semantic evidence is acquired, not only how completed semantic scores are redistributed. Geometry supplies a bounded conditional readout; the second component tests whether the same structure can repair upstream evidence formation while retaining native scene pathways.

This is a research program, not a CVPR-readiness claim. The paper still needs closest-method superiority, causal/mechanistic controls, fair complete-system comparisons, independent validation and transparent compute. Preserve and cite borrowed VIP components in historical results; do not rename them as a new observer.
