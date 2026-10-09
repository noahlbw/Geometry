# Geometry-defined semantic units with retained fine detail

Prospective candidate: `Geometry_UnitDetail`, signature `geometry-semantic-unit-detail-lift-v1-196-20261002`.
Only authorized physical A800 GPUs4-7. Original Geometry and every historical result remain unchanged.

## Hypothesis, not a verified diagnosis

Earlier frozen-head relation reweighting, donor quotas, virtual regional CLS and borrowed regional classifiers failed their fixed gates. This candidate changes the actual physical units entering the frozen semantic head, rather than merely reweighting the same completed Values or logits. It tests whether structure-aligned unit encoding can provide useful semantic content while retaining fine Geometry detail. It does not add an independent semantic source, and coherent wrong semantics may persist.

The pinned author DINO.text builder/config specifies nominal224-pixel training images and patch16, hence196 nominal patch tokens. The unit budget is fixed to196 before outcomes, shared by all domains. This config alone does not establish that a512-window head failure is caused by token count, nor that pooled units match the training distribution. No224/336/512 target-score search is performed.

## Complete coupled computation

1. Run the original frozen DINOv3 and Geometry head on each original512 window. Cache native backbone tokens, original relation G and unnormalized projected Geometry descriptors Yg.
2. Form a4-connected minimum spanning forest using symmetric original-G edge costs `-.5(log Gij + log Gji)`. Retain the strongest forest edges until196 connected units remain. If fewer valid patches exist, use singletons; disconnected image-valid components are never joined across padding.
3. Membership U broadcasts a unit back to its original patches. Within each unit, S uses normalized original-G incoming support. Thus `S U = I`. Only image validity and G enter grouping; text predictions and masks do not.
4. Pool native backbone content `Xunits = S Xnative`. Push the same Geometry relation into the unit basis: `Gunits = row_normalize(S G U)`. Starting native prefix tokens and pooled patch tokens pass through both original frozen Geometry head blocks, residuals, MLP, LN and projection to produce Yunits. Prefix/head attention allocations evolve on this new unit stream; they are not falsely claimed identical to the fine head's allocations.
5. Produce one final dense descriptor field: `Y = Yg + U(Yunits - S Yg)`. Then apply the original L2 descriptor normalization, unchanged20 aliases/class, six RS templates, normalized LME .07 and Hann window probability assembly.

Before final L2 normalization, `S Y = Yunits` and within-unit descriptor differences are exactly those of Yg, up to stated floating-point error. These are implementation invariants, not guarantees of correct classes or preserved final predictions. Invalid patches receive no displacement. Singleton/no-valid-unit cases return original Geometry exactly. No extra backbone, image crop, teacher, alias deletion, fitted confidence threshold, class quota or per-domain branch.

MST clustering, content-adaptive token merging and coarse/detail reconstruction have precedent. Their use is not itself claimed as a new mathematical invention or a confirmed CVPR contribution. The prospective distinction is re-encoding geometry-defined semantic units and using a defined coarse/fine reconstruction that retains local detail, not renamed score fusion.

## Controls, frozen screen and promotion

Evaluate the complete primary against original Geometry, matched SCLIP_Two/VIPProxy_Two, UnitOnly broadcast, equal-information MeanLogit_Unit, and Spatial_UnitDetail. The spatial control changes grouping to a fixed14x14 spatial partition while keeping Geometry pooling, compressed-head readout and detail reconstruction. This tests content-aligned grouping against generic compression, though irregular valid boundaries can give different active unit counts.

Use the existing fixed96-image screen: UDD5 full40 plus eight images per other domain. Verify exact matched per-image baselines, complete unique coverage, checkpoints and all20 vocabularies. LoveDA D enters the eight-domain mean once; P and D foreground are separate. Corrected IRRG Vaihingen is authoritative. LandCover.ai replaces unlabeled iSAID. All domains are development data; VIPProxy is not official full VIP.

Promote only if the primary retains every Geometry protocol, beats Geometry/SCLIP/VIPProxy/mean-logit/spatial controls in equal-domain mean, and beats Geometry/VIPProxy/mean-logit/spatial controls on VDD and Potsdam. A pass authorizes the same frozen full eight-domain rule; it does not prove SOTA. No unit budget, edge rule, lift coefficient or class-specific choice is tuned after outcomes.

Real-checkpoint FP32/bf16 smoke must load no target masks, preserve native caches and exact singleton Geometry, reconstruct unit means within2e-4, produce finite fields and preserve invalid queries. Record independent synchronized one-window costs including CPU/GPU partition transfer; evaluator all-arm timings are not deployed latency.

## Implemented execution and control correction

Model: `DINOtool/dinotool/geometry_unit_readout.py`. Evaluator, smoke and controller are `scripts/eval_geometry_unit_readout.py`, `smoke_geometry_unit_readout.py`, and `run_geometry_unit_suite.py`.21 local and remote tests pass, including actual pinned upstream block execution and a regression that gives matched proxy operators the exact unnormalized native backbone patch cache.

Real-checkpoint FP32/bf16 singleton Geometry/cache errors0; unit mean error1.19209e-7; invalid displacement0; all weights frozen; no target masks loaded. Actual model nominal configuration is224, budget196. Median window Geometry/primary times0.024748/0.030845 seconds, about1.2464x, including partition and excluding control heads. Both peak allocations3715.736MiB. MST grouping is not invariant to small precision changes: the smoke's largest FP32/bf16 units contain238/277 patches despite the same196-unit budget. Do not claim identical grouping across precision modes.

The first screen root `results/geometry_unit_readout_screen_20261002` is retained as a failed control-replay record. The VIP proxy was incorrectly passed `prepared.raw_patch_tokens`, which is already normalized; its author adapter normalizes again. This caused small but nonidentical reference confusions. The verifier refused promotion and cleared its remaining queue. All four already-launched runs finished; their primary algorithm did not depend on this control input and already showed substantial losses. No primary coefficient, grouping, unit budget or checkpoint was changed.

Only the control cache interface was corrected to use `prepared.backbone_tokens[:, prefix:]`. A clean fixed repeat uses `results/geometry_unit_readout_screen_20261002_r2`, controller `gur02_suite`, with the same screen sequence and primary rule. The old outputs are neither overwritten nor treated as verified matched results. If its unchanged prospective gate passes, the controller may launch `results/geometry_unit_readout_full_20261002_r2`; otherwise no full rollout. The exact repeat of original primary confusions should also be checked on the four repeated datasets.

## Verified final decision

The clean repeat has finished on all eight domains, with 96 unique images in total. Local collection verifies complete per-image coverage, unchanged checkpoint/vocabulary identities, exact replay of Geometry/SCLIP_Two/VIPProxy_Two reference confusions, and agreement between per-image arrays and every merged confusion matrix. All suites and workers are terminal. A subsequent read-only GPU check found physical GPUs4-7 idle; unrelated tmux sessions were preserved.

The predeclared promotion gate fails. Equal-domain mean is26.874787 for Geometry_UnitDetail versus43.830569 for Geometry, a16.955782pp loss, with0/8 domain wins. UDD5 full40 is30.4186 versus50.5553; Potsdam8 is34.7371 versus40.3528; VDD8 is17.7412 versus31.9462. No full rollout was launched and no strength, grouping or budget was tuned from these outcomes.

The control-recovery audit confirms exact original Geometry and primary per-image confusion matrices on VDD, Potsdam, OEM and LoveDA P/D before and after correcting the VIP proxy cache. Thus that correction did not change the primary's negative outcome. The completed results are in `research/GEOMETRY_UNIT_READOUT_SCREEN_20261002.md`; the audit is `research/geometry_unit_readout_screen_20261002_r2/control_repeat_audit.json`.

This result rejects the fixed semantic-unit/detail-lift hypothesis, not every Geometry coupling. Pre-normalization unit means and within-unit differences do not guarantee correct text-relative directions after normalization. Connected supports also are not semantically certified regions. The detailed report distinguishes these architectural limitations from causally established error sources. The broader objective of an original useful coupling and eight-domain official-VIP superiority remains unachieved.
