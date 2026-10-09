# Frozen 20-alias mixture experiment: VDD and ADE150

The target is a useful alias module in a Geometry-centered coupled model:
VDD mIoU >54.3, ADE150 >29.1, and full-image singleton inference <=6x
both matched20/template VIP and official-query/configuration VIP. These paper
numbers are targets; crossing them is not certification of matched published
benchmark protocols or independent validation. Existing development was labelled.

The preceding complete trials did not establish a successful alias module.
SharedStar: VDD53.1345 versus same-observation mean53.6976; ADE25.7319 versus
mean25.5635 but below pooled25.8000. Canonical-rival component penalty:
VDD54.4902 versus strong Base55.0754; ADE26.8493 versus Base26.8763.
The latest method costs334/249ms, about2.58/2.25x official VIP, so its failure
is evidence utility rather than latency. Full results and negative controls
remain in their separate reports and result directories.

## One new hypothesis

Keep every original alias string and all20 slots. The old wide reader uses
LSE(tau *20*softmax(s/tem)*z)/tau. The new reader instead uses
LSE(tau*z+log(20*softmax(s/tem)))/tau, with tau=tem=1.
Salience is a mixture prior, rather than an alias-specific logit temperature.
It does not certify that a word is semantically correct. Positive weights
depend on the already observed wide crop, without additional visual inference.
Uniform salience recovers the old scorer, including its log20 count gauge.

Four predeclared arms: unchanged task Base; Mix (the sole primary); UniformMix
(all20 priors uniform); ShuffleMix (all20 priors permuted within each class,
seed20261008). No canonical exception is added. The fifth accuracy comparator
is unchanged matched20/template VIP; official-query/configuration VIP is also
measured for latency. Controls cannot be promoted after viewing results.

The local Geometry readout, temperature0.07, original H, coupling0.5 and
task text profiles remain unchanged. The only semantic change is the wide
aggregator. Final scores are L+0.5H(Wmix-L). Mix corrections therefore travel
through the same Geometry writer, rather than directly covering local scores.
VDD retains ImageNet80/patch-only strength1/wide long448; ADE retains the pinned
seg template/original Geometry/wide short336 with long-edge cap672.

## Cost, controls and verification

At most4local512 and4wide336 encodings; zero fine or additional RGB/semantic
heads. Alias aggregation is on21x21 wide tokens before interpolation, with
maximum [441,classes,20] alias work, not a pixel/alias/all-rival tensor. For
ADE, one float32 tensor of this size is about5.3MB; this is not total memory.
Multi-arm execution shares observations; deployment and timing run singleton
arms. Model/text initialization and decode are excluded from latency, while
resize, encoders, scores, alias aggregation, H, restoration and argmax are included.

Two remote CPU tests verify uniform identity, score shifts, weight alignment
and the exact prior-bound inequality. Astra high reviewed scorer units,
AMP ordering, salience dimensions, all20 shuffle, unchanged local/writer and
the singleton paths. Every worker must also pass mask-free Base=existing
task profile and joint=singleton checks on a real image.

Full VDD80/ADE2000 unique sample coverage, exact Base confusion replay against
the preceding completed fixed20 task trial, checkpoint/word identities,
paired scored targets and frozen weights are required. Per-image confusions
support1000 paired image bootstrap resamples (seed20261008). Full joint
diagnostics record actual prior range, Mix-vs-Uniform wide/writeback magnitude
and final prediction changes; singleton absent comparisons are not evidence.

Timing uses first/middle/last complete images and5 rotated, warmed,
synchronized singleton repetitions on an exclusive physical GPU, checked
before and after each measurement. Shared-resident memory is reported and
does not establish isolated VIP deployment memory. Existing GPU workers are
preserved; the new controller waits for idle cards and restores the old queue
dispatcher after this suite. No recurring heartbeat was created.

## Interpretation and a separate semantic-input diagnostic

Mix must improve its same strong Base and both identity controls on both
datasets, pass the latency limits and cross both numeric targets. A gain only
over Base is evidence of readout calibration, not useful word-specific selection.
Zero-crossing paired intervals are inconclusive. Negative results remain visible.

ADE original20 also lacks some class disambiguation present in the historical
433-query pool: pot/flowerpot, glass/drinking glass, hood/range hood and
washer/washing machine. Historical31.1862 used that different pool and cannot
be inherited by this trial. The existing, previously frozen curated20 pool
contains those definitions and can later support an original/curated x
Base/Mix factorial. Input gains must be separated from same-input module gains,
with curated Uniform/Shuffle controls before claiming word-identity utility.
No curated words or target-mask-based tuning enter the current trial.
