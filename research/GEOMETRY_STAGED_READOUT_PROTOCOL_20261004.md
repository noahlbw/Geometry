# Geometry staged readout: frozen protocol

## Hypothesis And Boundary

Keep the original Geometry relation G fixed. The first block retains native patch/special allocation,
so context can condition semantic Value features. The final block reads unit-mass patches only,
so its dense evidence increment does not inherit native special-token allocation.
Prefix queries continue their original native attention in both blocks; residuals and MLPs remain unchanged.
This is not VIP's sparse proxy relation or normalized-prefix attention increment.
Patch-only readout and stage-specific processing have precedents; this arrangement alone does not establish CVPR novelty.
Its purpose is to determine whether Geometry's performance gap is in its readout allocation rather than its relation.

Original: a_special V_special + m_patch G V_patch.
Final patch-only: G V_patch. Both still use the existing residual and MLP.

Primary: Geometry_Staged = preserve / block.
Controls: Geometry = preserve / preserve; Geometry_BlockPrefix = block / block;
Geometry_ReverseStage = block / preserve; Geometry_PrefixUnit = (special + unit-patch) in both blocks;
Geometry_PatchMass = mass-patch-only in both blocks. Also replay SCLIP_Two and VIPProxy_Two.
The allocation factorial separates special-token removal from patch-increment strength; stage reversal tests ordering.
Do not select another control as the primary after seeing scores.

## Evaluation

Reuse existing fixed96 complete-image manifests: UDD5 full40 and seven domains8 each.
Unchanged20 aliases/class, six RS templates, normalized LME .07, native512/stride128/Hann.
One frozen DINO backbone, no extra views, fitted thresholds, alias changes or wide-view coupling changes.
Mask-free actual-checkpoint smoke must verify fp32/bf16 exact original and BlockPrefix replay, unchanged weights and caches.
Per-image historical confusion replay and unique complete coverage are required before interpreting scores.
Report LoveDA P/D separately; the eight-domain mean counts D once.
Measure per-class precision/recall/area, correction transitions, native patch mass and stage feature norms.
Benchmark standalone512-window latency with two warmups/five synchronized repeats, one backbone and one selected head.

All images are developed validation. This is not untouched validation or official VIP/SCLIP benchmark reproduction.
Corrected-input Vaihingen and LandCover.ai replacing unlabeled iSAID remain unchanged.

Promotion requires the fixed primary to beat Geometry, BlockPrefix, SCLIP_Two and VIPProxy_Two in the domain mean;
not degrade VDD/Potsdam; incur no protocol loss above1 pp; and take at most1.1x Geometry single-window median latency.
No automatic full dataset rollout in this screen. Original model and old paused jobs remain untouched.
