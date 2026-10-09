# Frozen one-sided soft alias execution experiment

Execution-only candidate. Preserve the developed OneSide_Soft rule, fixed20
vocabularies, checkpoints, Geometry patch-only2, bounded896/448 inputs, at most
4 Geometry/4 wide/16 fine encodings, positive fine observation, FP64 reader,
fixed-slot weights, canonical protection and original H. No parameter fitting,
new risk score, sparse rivals or additional visual observation.

Cache each crop's word softmax before stencil gathering; process queries in
blocks of1024 and rivals in blocks of16. Keep original crop summation order and
underflow fallback. Kernel batching can change FP64 rounding: require maximum
patch-score error<=1e-10 and identical COMPLETE-image predictions. Report
bitwise equality separately; numerical tolerance alone is not exact prediction.

Test CPU/CUDA synthetic stencils including extreme logits and rival blocks.
Then mask-free VDD/Potsdam replay. Then same96 developed complete inputs:
UDD5 full40 and8 evenly spaced images in each other remote-sensing domain.
Compare original Geometry/no-admission/patch2, same-fine no attenuation,
original OneSide_Soft and new execution. Load labels only after predictions.
Require per-image original confusion replay against the archived one-sided
audit and pixelwise candidate/reference equality. No full20092 relaunch.

Measure3 fixed complete images/domain,5 rotated warmed singleton repeats, on
one idle GPU with the other GPUs idle. Compare current OneSide_Soft, new stream
execution, same-fine no attenuation, bounded patch2 baseline and VIP_All20.
Both soft arms already use cached-burst fine execution. Record graph setup
separately, full original-resolution restoration, prediction repeatability,
stage microbenchmarks and shared-resident memory (not standalone deployment).
No target masks in timing. No attribution of old cache/burst gains to this work.

Preserve old outputs and unrelated jobs. Report any speed regression. This
experiment cannot establish a new lexical mechanism or independent accuracy
gain; the previous weak word-specific attribution remains unresolved.

## Environment amendment during execution

After VDD/Potsdam/UDD5 timing completed, the all-server-idle scheduler guard
stopped because unrelated jobs started on GPUs0,1,3. Preserve all prior results
and do not interfere. Complete the remaining paired rotated timing on exclusive
GPU7, allowing other GPU jobs and recording their state. Label this changed
environment explicitly: host/power contention can affect small speed differences.
This changes the timing isolation condition only, not observations or model rules.
