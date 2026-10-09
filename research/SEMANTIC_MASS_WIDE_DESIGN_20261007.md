# Variable vocabulary: separate semantic mass from query multiplicity

## Evidence and decision

The retained wide score is `LSE(tau * K * softmax(s/tem) * a)/tau` within each class. Whole-group replication adds `log(replications)/tau`. Replicating only one alias also changes the amplification of other aliases through K and the salience denominator. Previously tested count normalization removes only the first effect and degraded full natural results; it is not the chosen solution.

An experimental operator is implemented in `DINOtool/dinotool/semantic_mass_wide.py`. It is not connected to production inference and has no measured segmentation improvement yet. Existing best outputs/settings are unchanged.

## Proposed operator

For declared semantic groups G, use group-mean salience and group softmax q. Each group gets amplification `M*q_g`, where M is number of semantic groups. Average exponentiated alias evidence inside a group, sum across groups, and add the frozen offset `log(K_reference/M)/tau`.

Thus singleton groups with K_reference equal to original K reproduce the retained score. Replicating an identical observation within its declared group leaves the score invariant, without removing the original class-count prior. Distinct new concepts can still affect evidence. K_reference must be frozen before candidate evaluation, never selected from target IoU. This preserves rather than eliminates historical vocabulary-prior provenance.

This does not establish invariance to arbitrary nonidentical paraphrases: group-mean salience/evidence can change when a different embedding is added. False group assignments can discard discriminative evidence. Residual/background unions are sets of different concepts, not synonyms; they require separate ontology treatment. The operator does not repair background semantics or guarantee fair variable-count calibration on its own.

## Verification and next experiment

Five CPU float64 tests passed: singleton identity at three tau values; individual identical alias replication; whole-bank replication; sensitivity to a distinct new concept; invalid inputs. These are algebraic tests, not full-model probability replay, timing or mIoU evidence.

Before a five-target evaluation, declare groups from class definitions/provenance without masks, retain canonical aliases, and smoke-test unchanged singleton predictions on real cached observations. Then compare singleton inherited versus grouped evidence using identical vocabulary, views, residual route and frozen read profiles. Only afterwards compare variable generated pools. This isolates grouping from word quality and parameter changes; no additional image encodings are needed by the operator. A production implementation may need packed groups for efficient text scoring.

Do not claim a discovered adaptive parameter law or promote the operator based on the passing algebra tests. VOC/PC performance gaps and cross-task generalization remain unresolved.

## First verified full result: VDD

All80 images merged with unique coverage, exact selected-task Frozen confusion replay, matching checkpoint/classes/sample keys and paired scored targets. Frozen55.2382 versus SemanticMass55.8414 mIoU (+0.6032pp). Class IoU deltas: other+0.9092, wall+0.9769, road+0.9731, vegetation+0.1478, vehicle+0.9283, roof+0.3180, water-0.0311pp.

Road precision42.7382→43.7964%, recall96.2384→95.9882%; vehicle precision31.9787→33.1796%, recall74.3063→74.0466%. Gains therefore include reduced false activations with a small recall cost, rather than uniform foreground recall recovery. Wall recall53.6417→54.9643% improves despite precision80.1213→79.9834%. Changed competition affects non-grouped classes too; these post-freeze outcomes do not identify individual-alias causality.

The frozen language-only groups use all the same words, not hard deletion. Singleton probability identity passed on one mask-free real image for each of the five targets before full runs. Remaining targets are still running; the VDD gain alone does not justify promotion. Raw verified VDD results: `semantic_mass_wide_20261007/vdd/merged.json`. Full report will supersede this partial outcome.

## Additional verified full results: Potsdam and VOC21

Potsdam504 images:49.7888→50.0944 (+0.3056pp), with car36.7792→37.9182, building80.5641→80.9609, tree61.8622→61.7942. VOC21 all1449:70.4508→69.4710 (-0.9798pp). Both unique coverage, paired targets and original selected-task Frozen confusion replay pass.

VOC motorbike IoU72.0045→62.4687: recall77.5319→67.2691%, precision90.9909→89.7477%. Car IoU69.6258→64.7523: recall83.2451→75.1504%, precision80.9732→82.3939%. Person recall81.1687→76.4721%, while precision86.7518→86.9526%. Sofa IoU65.6431→68.1494 improves. Aeroplane recall95.8508→97.2809% but precision66.6904→64.8848% declines.

Thus the new rule is already unsupported as a nondegrading universal upgrade. Algebraic identical-observation duplication invariance is not evidence that nonidentical synonym embeddings can be averaged without losing visual coverage. The candidate simultaneously changes group salience, amplification and within-group reduction; these confusions cannot isolate which causes the losses. Do not invent a class-wise exception from the target scores. Keep the frozen full PC60/ADE150 evaluation running and retain historical deployment. A later mechanism diagnostic must isolate these score changes on matched observations rather than assume lexical equivalence implies interchangeable visual responses.
