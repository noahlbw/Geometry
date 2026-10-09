# Frozen adaptive model candidate: current evidence and remaining verification

## Candidate definition

Retain Geometry, bounded wide observations and the original geometric reconstruction. No fine observations. The readout consumes at most four local and four wide encodings. Model weights are frozen.

Task input includes the supplied class taxonomy, declared natural/remote-sensing family and residual-class definition. It does not include masks or a dataset-name lookup of best coefficients.

- Natural tasks use the centered canonical-text effective-rank rule to select the previously declared correction menu. Shared segmentation templates, semantic variable-count banks and temperature0.07. Foreground aliases use count-normalized uniform aggregation.
- Remote sensing uses the previously frozen canonical-margin route selection on existing original Geometry features: original20/ImageNet/strength1 versus focused20/RS6/strength3, shared correction0.5. The bounded half-uniform competitive alias weights are enabled for this family.
- A natural task with a supplied full residual ontology may explicitly enable residual protected readout. The complement of scored names forms a residual union; maximum evidence replaces the background channel. Local residual scores are conservatively suppressed relative to the original coupled foreground rival. Wide residual evidence remains unchanged. This route currently has full evidence only on PC60 with official459-class metadata. Do not enable it on unrelated background definitions by default.

The residual path accepts preencoded `[query,template,dimension]` features in `TaxonomyInference(..., soft=False, residual_features=..., residual_mode='protected')`. The implementation accepts a generic background index. It rejects unverified simultaneous foreground soft weights and residual replacement. Existing API defaults remain unchanged; construction must explicitly choose this candidate's family policy.

## Full evaluation evidence for the selected routes

| Protocol | Images | Original reference | Candidate route | Difference |
| --- | ---: | ---: | ---: | ---: |
| VDD | 80 | 53.4600 | 55.2382 | +1.7782 |
| Potsdam | 504 | 45.9339 | 49.7888 | +3.8549 |
| VOC21 | 1449 | 29.6765 | 70.3179 | +40.6414 |
| PC60 | 5105 | 36.7248 | 41.5558 | +4.8310 |
| ADE150 | 2000 | 24.6903 | 31.1076 | +6.4173 |

These are route results from separate verified frozen full evaluations, not a newly completed single deployment suite. Original reference versus candidate changes words/templates/readout jointly. The table cannot isolate Geometry, adaptation or weighting contributions. Full scalar evaluation results do not establish complete vectorized-deployment confusion equality.

Source evidence: `coverage_taxonomy_readout_20261007` for remote sensing; `taxonomy_readout_20261007` for VOC21/ADE150; `residual_rival_protection_context60_20261007` for PC60. Each has unique full coverage and verified scored target counts. Reference replays and unchanged-arm replays are checked in the relevant collectors.

## Real-image deployment checks

The optional residual max/protected API reuses existing visual observations. Four inference API tests passed on A800. Both modes were compared to their scalar full evaluators on three fixed spread PC60 images without loading target masks. All three predictions matched exactly for each mode; maximum probability error was8.94e-8. This is limited-image numerical agreement, not full-dataset equivalence.

On GPU4 A800, warmed synchronized three repeats per image, the protected route averaged200.52ms/image versus198.75ms for the original uniform automatic route, averaging each image's median. Peak allocated memory differed by about33–37MiB. This includes encoding/readout/stitching/restoration/CPU argmax but excludes model/text loading and disk I/O. It is not a matched VIP speed comparison or a full-dataset latency estimate.

Raw timing: `residual_protected_inference_benchmark_20261007.json`; residual-max timing: `residual_max_inference_benchmark_20261007.json`. Non-residual family timing is reported in `TAXONOMY_COVERAGE_SYNTHESIS_20261007.md`.

## Interpretation for a CVPR submission

The claim supported so far is task-conditioned use of Geometry and semantic evidence, not parameter-free optimization or a guarantee of SOTA. Rank quantization, the two RS candidate routes and the0.5 soft floor retain labelled-development provenance. Online decisions are mask-free; this does not erase development provenance.

The extra PC60 ontology is an additional information budget. A fair baseline must receive the same ontology if the comparison attributes improvement to the reader. The protected route restores some foreground precision but still loses food/cup/sign IoU relative to the original automatic route. It is not a universal background solution.

The wide vision reader is inherited from VIP. Its origin must remain explicit. Geometry and the coupling/semantic use mechanisms require independent mechanism and matched-operator evidence; vocabulary construction alone is not an independent architectural contribution.

## Next required verification

Freeze the above family policy and run the unified deployed implementation on all9138 protocol images without further parameter selection. Check per-image probability/argmax differences against the corresponding frozen scalar routes, record tie-only changes and resulting full confusion matrices, and preserve old results. Then use untouched task transfer to assess whether the rank/margin selection generalizes. Do not report this candidate as completed independent validation before those steps.

The five-protocol deployed verification has now been launched under `results/final_adaptive_deployment_20261007`, controller `gfad07_controller`. It first runs mask-free smoke checks and then queues only idle GPUs. It checks the full original reference and the previously frozen scalar confusion matrices independently, and saves actual deployed confusion matrices rather than assuming equality. Manager: `tools/final_adaptive_deployment_experiment.py`. No model rules or text banks were changed for this verification.

The first completed VDD worker exposed a missing `diagnostics.tiles` field in the new evaluator's output schema. Inference completed normally; the shared merger failed on this metadata requirement. Existing workers continue without source changes. `recover_final_adaptive_merges.py` provides merge-only recovery after the original controller becomes terminal: derive tiles from saved per-image geometry encoding counts in separate adapter copies, retain original worker results/logs, and rerun all coverage/confusion checks. The local evaluator is corrected for future runs, but this suite will not be relaunched. Recovery is not yet claimed complete.

The original suite is now terminal. Four protocols completed inference, but PC60 shard1 stopped after1890 saved progress on a strict probability comparison (first failing image1894, `2010_000822`). A mask-free reproduction located one rival flip at a scalar foreground gap1.33e-7, amplifying the residual raw score by0.0018909 and final probabilities by0.0008841. The failing image's final argmax labels were unchanged, but full probability equivalence is disproven. This is not a reason to loosen tolerances or report full5105 deployed coverage.

The repair preserves the frozen scalar reduction order before the discontinuous residual rival argmax and caches class indices/Python count logs, avoiding per-class count synchronizations. It does not change the semantic weighting rule. Original deployed modules are backed up remotely before the repair. Three alias-group tests and four inference API tests pass. The failed full outputs are preserved; only four completed protocols can be merge-recovered without inference. A corrected PC60 deployment verification needs a separate output root after regression checking, not an overwrite or a fabricated resume from missing per-image checkpoints.

The exact-group regression now passes all ten images near the failure with zero probability difference, including `2010_000822`. Four completed protocols were merge-recovered with full coverage, exact original/scalar confusion replay, per-image sums and paired target counts: VDD55.2382, Potsdam49.7888, VOC2170.3179, ADE15031.1076. Their near-tie deployed argmax changes total32/15/1/14 pixels respectively; rounded full mIoU is unchanged. Downloaded merged files are in `final_adaptive_deployment_20261007/{dataset}/merged.json`.

Corrected PC60-only full verification is running in the separate `results/final_adaptive_deployment_pc_repair_20261007` root, controller `gfad07_pc_repair_controller`, four shards. Model rule, texts and strict comparison tolerance remain frozen. Exact-group deployment timing is separately running on GPU4; the earlier201ms protected timing describes the superseded packed-gate implementation and must not be quoted as the final repaired latency.

The corrected timing has completed and been collected in `residual_protected_exact_inference_benchmark_20261007.json`. Three fixed PC60 images, warmed synchronized three repeats: average per-image median220.27ms for protected versus197.84ms for uniform, +22.43ms. All three probabilities equal the scalar reference exactly (maximum error0), with unchanged weights and no target masks. Cached scalar-order aggregation is retained only before the protected residual hard rival decision; other routes retain the packed backend. This removes per-class count synchronizations but still launches per-class reductions. The cost is accepted for fidelity; no lower-latency guarantee or full-dataset average is inferred from the sample.

Corrected PC60 full deployment is now complete and collected:5105 unique images, exact original/scalar full confusion replay, paired targets and per-image sums. Deployed41.5558, scalar41.5558, changed prediction pixels0 and maximum probability difference0. Four other recovered protocols plus this repaired result cover all9138 requested protocol images. The consolidated report is `FINAL_ADAPTIVE_DEPLOYMENT_20261007.md`; publication/transfer limitations remain explicitly open rather than treated as SOTA proof.
