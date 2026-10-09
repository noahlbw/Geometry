# Geometry-conditioned position-language observation

Prospective signature `geometry-position-language-observation-v1-20261002`. Use physical A800 GPUs4-7 only. Preserve Geometry, historical models and every negative result. This is a new source-viability diagnostic, not a verified successful coupling or a claim of original mathematics.

## Why this differs from earlier trials

Earlier frozen discriminative regional pools, virtual CLS, native donor replay, denoising-loss compatibility and unit-detail lifting failed their fixed rules. Here an independently pretrained instruction-tuned VLM answers an explicit location-recognition question from actual RGB. Its training objective and localization supervision differ from a pooled image-text cosine. Geometry supplies both the queried locations and each foveal view's physical extent, not a semantic truth certificate.

The additional source is Qwen/Qwen2.5-VL-7B-Instruct, fixed revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. Public ungated model; local inference; no images are uploaded. About16.6GB of source files, with author card/license and pinned file-size/LFS metadata preserved. No new weights are trained, no remote model code is trusted, and no gated contract is accepted. This changes the model budget and supervision source; it is not a matched single-backbone comparison with VIP. Pretraining/evaluation overlap is not independently audited.

## One fixed observation and rejection rule

Reuse the original VDD/Potsdam eight-image diagnostic and all16 saved512 windows per dataset. Use original class names, exactly20 aliases/class and original Geometry scores/relation, without modifying any checkpoint, text bank, threshold or source window.

Within each currently predicted class, choose up to four evenly spaced margin-rank query positions, including extrema. This is a diagnostic sampling design, not a test-time class area quota. It uses no target masks. High and low ranks can expose correct detections and false activation; prior labeled audits motivated this question, so these are development data, not untouched validation.

Read two unmasked RGB views for every query: the original512 context and a crop centered at the query that encloses95% of its image-valid Geometry support in Chebyshev distance. Crop extent is bounded64..512 pixels and rounded to patch16. A hollow red marker indicates the exact queried location, leaving its center unobscured. Annotation is an intervention to the VLM input; it does not modify original Geometry. It does not isolate an object from context or certify that Geometry support has a single semantic class.

Every class option lists all original20 aliases. Score the exact next-token letters with both forward and reverse option orders, restoring class identity before any comparison. Include the model's explicit Z abstention. Conditional option distributions are compatibility scores, not calibrated correctness probabilities. No view, order, phrase or coefficient is selected from target outcomes.

Primary `GroundedConsensus`: all four view/order argmax choices must agree on the same non-abstaining class; otherwise keep the original Geometry query prediction exactly. Unanimity is only an unverified rejectable-evidence rule, not proof of correctness. Controls expose context-only, foveal-only and their matched normalized log-probability mean. The complete observations are saved before cached target labels are consulted for the audit. Cached label tensors are resident because they share the original snapshot file, but none is passed to the observer. Do not claim that no label was loaded into memory.

## Evidence and decision

Report metrics on identical selected query centers, not full windows/images. An abstention is a false negative for the true semantic class; it is not dropped and does not add an extra class to the dataset mean. Save exact query identities, all four raw option-score vectors, physical crop extents, legal-answer token mass, agreement/abstention, original/new/truth changes, per-class TP/FP/FN, time and peak memory.

This fixed diagnostic can justify a subsequent dense-coupling trial only if the primary retains Geometry query mIoU and has more beneficial than harmful changes on both datasets, lowers vehicle/car FP without losing their TP, and preserves VDD water/Potsdam low-vegetation TP. Passing does not authorize a SOTA claim, full-system victory or completion of the eight-domain goal. Any subsequent dense transport/reconstruction must be prospectively specified and tested across all eight domains with matched information controls and standalone cost.

Failure rejects this particular language question/view/consensus design; do not tune wording, crop mass, ranks, per-class rules or confidence thresholds on these cached labels. Keep the broader objective active. Ordinary VLM prompting, foveation and consensus all have precedent; an independent CVPR contribution has not been established by assembling them.

## Completed verification and decision

All four workers on physical GPUs4-7 completed; all32 original windows and572 unique contextual query identities were verified. Ten local/remote CPU tests pass. Source weights were frozen, and original Geometry sampling/predictions and saved language-derived choices reproduce the original caches/raw option scores exactly on the same CUDA arithmetic path. The first CPU control reconstruction exposed tied-score log-softmax reduction sensitivity; the exact CUDA audit passed without changing inference outputs, raw scores, primary rule or source inputs.

On the268 VDD selected queries, Geometry/primary mIoU is21.398155/22.510969, with11 beneficial and2 harmful changes. Vehicle FP39->33, TP2->2; water TP4->4. The VDD diagnostic gate passes, with very small true-vehicle support and no full-image performance implication.

On the304 Potsdam selected queries, Geometry/primary mIoU is30.420924/30.059465, with7 beneficial and6 harmful changes. Car FP60->60 and TP4->4; building TP17->14. The Potsdam gate fails, hence combined viability fails. Direct context/foveal language observations are materially weaker than Geometry on these Potsdam queries. No full eight-domain rollout was launched or supported by this gate.

Max shard times are64.942s VDD and70.639s Potsdam; peak allocations16011.200/16003.876MiB for the diagnostic observer. These are not final dense model cost or comparisons against standalone Geometry latency. The full report and raw observations are in `research/GEOMETRY_LANGUAGE_OBSERVATION_RESULTS_20261002.md` and `research/geometry_language_diagnostic_20261002/`. All workers are terminal; the final read-only server check found GPUs4-7 idle and preserved unrelated tmux sessions. The original eight-domain/CVPR objective remains unachieved.
