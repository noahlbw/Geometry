# Frozen-Backbone Vocabulary And Readout Development

User-authorized priorities: VDD/Potsdam twenty-alias inputs and Geometry read
strength; VOC21/Context60/COCO-Object background repair; ADE150 >=30,
Context59 >=49, VOC20 >=92. Also evaluate COCO-Stuff. Targets are goals, not
promised results. This round does not modify retained model source or weights.

## Inputs And Parameter Scope

Original20 remains the reference. VDD/Potsdam candidates preserve exactly20
strings/class and make limited taxonomy-specific replacements; an ImageNet-local
template control separates words from templates. Natural candidates compare the
original ImageNet20, identical words under segmentation templates, existing
qualified compact semantic pools, and explicit concrete residual-background
queries. Natural compact/background pools have actual, potentially unequal counts.

Tune the patch-only read strength1/2/3 and original Geometry route, correction
gain0.5/1/2, local score temperature0.05/0.07/0.1, wide aggregation tau1/4 and
salience temperature0.3/1/3 using a finite declared profile grid. The grid is not
the full Cartesian product of all parameters. Scored residual categories additionally
test probability-space background log-odds offsets0/-2/2/4/8 and confidence rejection
thresholds0/0.1/0.2/0.25/0.3/0.35/0.4/0.45/0.5/0.6/0.7/0.8/0.9/0.95.
Background offsets operate after probability stitching/restoration, not as
arbitrary individual foreground class biases. Foreground-only protocols have no
background rejection. Category order, masks, checkpoints and input workload stay fixed.

## Development And Validation

ADE uses96 image-label pairs from its20210-image training split. All other
protocols use a deterministic SHA-ranked development subset of validation images,
at most64 and at most20% except the minimum8-image small-set allowance.
VDD has16 development and64 complement images; Potsdam64 and440.
These labels explicitly select hyperparameters: fixed weights are not a claim of
label-free adaptation. The remaining validation images do not select the setting.
Prior development also prevents calling these untouched independent test domains.

Select once by development aggregate-confusion mIoU on target-present classes,
with identical class support across candidates. Stable ties favor the retained
reference, zero bias and the least rejection. Freeze one configuration per protocol
before full evaluation. Report full results, including tuning images where applicable,
AND their disjoint non-development complements; do not promote a retrospective
full-set maximum or silently replace a failed candidate. Report separate
foreground/background precision, recall and IoU.

## Execution

Original RGB independently feeds bounded896 Geometry and bounded448 wide views:
at most4 local encodings and4 wide encodings, zero fine forwards. Search reuses
per-image visual observations across every numerical/text candidate. Frozen full
evaluation uses only reference and selected profiles. Source smoke must reproduce
retained predictions exactly without masks. Full merges verify unique coverage,
per-image confusion sums, paired target counts and exact original20 baseline replay.

The controller uses only idle physical GPUs0-7, preserves existing jobs/outputs,
stops new dispatch on a worker failure, and records the exact log. It schedules9
protocols: VDD, Potsdam, VOC21, Context60, COCO-Object81, ADE150, Context59,
VOC20 and COCO-Stuff171. Shared-source protocol pairs are not independent datasets.
No unrelated paused monitor is resumed. If numerical/text optimization misses
the stated targets or hurts the disjoint complement, report that result before
designing a further module.

Manager: `tools/development_readout_experiment.py`.
Results: `research/development_readout_20261006/` and
`research/DEVELOPMENT_READOUT_20261006.md`.
