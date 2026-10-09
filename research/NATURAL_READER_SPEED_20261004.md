# Exact Reader Acceleration

This is an opt-in execution optimization, not a new model or vocabulary candidate.
Existing running evaluations and their frozen source files are unchanged.

## Changes

- Cache each crop's profiled alias logits and class log-mean-exp references once per tile/readout, not once per query block.
- Gather patch logits and class references before forming sampled alias/rival margins, avoiding repeated full-crop alias-by-class tensors. Subtraction still precedes interpolation.
- Cache crop interpolation stencils and reuse retained-count normalizers across wide crops. Keep the original crop/reduction order and FP64 competitive reconstruction.
- Omit unused historical wide class maps and auxiliary Geometry-text scores. The same visual features, alias logits and salience are retained.

Geometry, crop locations/resolutions, text templates, temperatures, thresholding,
competitive admission and reconstruction equations are unchanged. The timing
benchmark loads no masks by default; optional `--score` loads GT only after all
frozen predictions and timing repetitions, solely to report single-image IoU.

## Verified Tests

Six focused local CPU tests and eight A800 CPU tests pass. They require bitwise
score/diagnostic equality for unequal/equal alias counts, singleton classes,
different beta values/chunk sizes, overlapping crops and invalid coordinates.
Wide crop logits/salience/count maps and visual forward counts also match exactly;
the opt-in predictor's dependency substitution is scoped to its own call.

## CPU Reader Microbenchmark

Windows CPU, PyTorch2.10.0+cpu, two threads,128 sampled points,query chunk32,
one wide and one fine observation,three timed repeats after equality/warm-up.
Class counts and per-class alias counts come from the frozen v2 manifests; image
features/logits are synthetic. The timings include neither visual encoders nor
full-image stitching and are not end-to-end GPU inference measurements.

| Vocabulary | Classes | Aliases | Reference Median | Cached Median | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| ADE150 v2 | 150 | 458 | 2.1909s | 1.3282s | 1.650x |
| COCO-Stuff171 v2 | 171 | 644 | 3.6282s | 2.0877s | 1.738x |

Both benchmark outputs require exact score and diagnostic equality before timing.
Raw repetitions are in `natural_reader_speed_20261004/cpu_DATASET/benchmark.json`.

## GPU And Full-Image Check

Physical GPUs4-7 remain occupied by frozen evaluations. After GPU3 proved occupied
by another project, the user authorized a free card; the fixed-image CUDA check
completed on GPU0. On ADE_val_00000001 (512x683), original single-profile inference
median is6.750437s and cached median3.341751s, a2.020x speedup. Every full-image
prediction variant has zero pixel mismatches and diagnostics are identical.
Current workers/controllers were not restarted or switched; GPU0 was released.
Timing commands refuse other processes on the selected GPU and require an explicit
single physical GPU via `CUDA_VISIBLE_DEVICES`. Full timing/IoU details are in
`SINGLE_IMAGE_SPEED_VS_VIP_20261004.md`; one image does not establish dataset-wide
latency or accuracy equivalence.

`DINOtool/scripts/benchmark_natural_sense_fast.py` provides two modes: synthetic
reader timing (`--micro`) and a full-image, single-profile reference/fast/VIP
comparison using frozen model/input paths. Full-image mode verifies
identical predictions/diagnostics and frozen model state, warms each path, then
synchronizes CUDA around each timed repetition. Its memory numbers include shared
resident models/caches, not standalone VIP memory. Use an independent output
directory rather than any existing evaluation output.

With `--without-fine`, an additional ablation skips fine visual forwards and
their rival-admission correction, while retaining Geometry/wide reconstruction
and frozen thresholds. `--vip-matched-words` adds VIP at its original visual and
scoring settings with the current model's text query bank. These are fixed
diagnostic comparisons, not newly selected final models. Optional image scores
report both each method's own union support and a fixed shared class support;
rejected predictions count as false negatives on valid target pixels.

`DINOtool/scripts/eval_natural_sense_fast.py` is the opt-in evaluator for subsequent
verified runs. It reuses the frozen predictor, records
`execution_backend=cached-exact-v1` in the configuration, and does not alter the
current queues.

## Parallelism Boundary

Wide, Geometry and fine visual features have no cross-branch dependency until
fusion, so branch concurrency is possible. Separate CUDA streams on one card do
not eliminate FLOPs and can contend for resources. Two-card branch parallelism
may reduce single-image latency but consumes two cards per image; it need not
improve full-dataset throughput. The current measured change removes redundant
work before introducing branch scheduling.

For each full512 tile the model runs one Geometry extraction plus up to four
256-physical-pixel fine crops enlarged to512, as well as image-level wide crops.
VIP's local official inference uses its own336-pixel crop protocol. This extra
visual work remains after exact reader caching; paper single-crop latency is not
the runtime of this full-resolution coupled evaluation.
