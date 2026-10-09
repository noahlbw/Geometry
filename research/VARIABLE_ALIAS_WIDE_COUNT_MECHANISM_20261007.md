# Variable alias count and inherited wide-view score

## Verified implementation, not a measured causal attribution

The frozen candidate's local uniform reader uses count-normalized alias evidence. Its inherited wide branch calls `eval_matched_head_text.imagenet_geometry_logits`, which does not subtract log alias count. For class c with K queries:

`q = softmax(global_salience)` within class; `scaled_j = K*q_j*alias_logit_j`; `B_c = logsumexp(tau*scaled_j)/tau`.

Local/semantic query banks are variable length on natural tasks. Thus variable query support and aggregation-count offsets enter together. The current words/templates/profile comparisons cannot isolate these effects. This observation does not establish that quantity bias explains any specific false positive or the aggregate VIP gap.

## Exact duplication consequence

Duplicate every query/template feature in one class once. Each global q halves; K doubles; K*q and each per-alias scaled logit remain unchanged. The log-sum-exp contains twice as many identical terms, so B_c increases by `log(2)/tau` at every wide patch. Dense interpolation and crop averaging preserve that additive offset. No new visual or semantic information was supplied.

Subtracting `log(K)/tau` removes this whole-group duplication offset. It does not make arbitrary individual duplicates or correlated paraphrases harmless, because duplicating a subset also changes global q and therefore K*q. It also does not resolve wrong-but-responsive concepts. A normalized aggregation is a testable calibration control, not proof of better IoU or a novel semantic selector.

All classes with the same K receive a common class offset; this alone does not change branch softmax. Unequal natural counts can introduce unequal offsets. PC60's explicit residual maximum uses a different aggregation and401 extra concepts: its information budget and maximum-union behavior must remain separate from alias-count correction.

## Next isolated test, after the live scale study

The isolated full test has now completed on all9138 protocol images. Frozen/count-normalized mIoU: VDD55.2382/55.2382, Potsdam49.7888/49.7888, VOC21 70.3179/29.1475, PC60 41.5558/35.6789, ADE150 31.1076/28.8973. Exact Frozen confusion replay verified. The count-only correction is rejected as a deployment upgrade; algebraic duplicate invariance does not establish useful semantic calibration.

VOC background recall collapses93.3811→14.3900%; PC60 background prediction area expands7.2329→43.9720%. VOC residual concepts mixed into an alias pool and PC60's separate residual maximum cannot be treated as identical synonym ensembles. See `COUNT_NORMALIZED_WIDE_20261007.md` for full-matrix diagnostics. The following paragraph records the original pre-test protocol, not pending work.

Keep words, visual observations, profile, temperatures, residual handling and retained Geometry correction fixed. Compare the current wide branch with count-normalized wide aggregation. Verify whole-group duplication invariance and equal-count prediction equivalence, then test all five full protocols. Do not choose normalization separately from labelled per-dataset gains or infer success from the algebra alone.

This note does not change the live natural_wide_resolution_20261007 experiment or promote a new model.
