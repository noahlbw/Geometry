# Fixed20 one-pass input generation: strict-format failure

The frozen `semantic_coverage_input_vdd_ade_20261009` generation trial is terminal.
All four live tmux handles have exited; all157 classes are covered exactly once.
The154 eligible classes received exactly one greedy metadata-only Qwen request,
and the three protected VDD groups received none. No target image or mask was
loaded and no new segmentation score was produced.

All154 replies returned a Markdown JSON fence despite the prompt requiring raw
JSON. Strict whole-reply parsing therefore failed with `Expecting value: line 1
column 1 (char 0)` for every request. Under the frozen no-salvage rule, every
class retains its entire existing20 group. There are zero eligible review items
and no implemented semantic-coverage treatment. Do not run identical candidate
arms, call this an alias-selection accuracy failure, or claim the hypothesis was
tested by segmentation.

The raw replies and failed protocol remain intact. Generation wall seconds per
shard:255.0055,264.3970,182.4921,226.0995; sum928.5941 GPU-worker seconds,
maximum264.3970 seconds. Peak allocated memory:16772.32,16771.38,16775.89,
16773.28 MiB. These are offline Qwen costs, not image inference latency.

Sampled raw replies also contain semantic errors: bare `concrete`/`brick` for
VDD wall, scored `skyscraper` for ADE building, broad furniture terms for sofa,
and scored-object parts such as `streetlight pole` for pole. Formatting repair
alone does not establish usable words. Every surviving proposal would still
need a complete text-only ownership/relation review before segmentation.

Any separately declared single-JSON-fence compatibility diagnostic must preserve
this strict trial as failed, reuse raw replies without regeneration or selective
schema repair, and disclose that its transport handling was revised after
observing generation outputs. It is not the original frozen protocol and is not
a new conditional-alias module. The VDD/ADE accuracy/speed objective is unmet.
