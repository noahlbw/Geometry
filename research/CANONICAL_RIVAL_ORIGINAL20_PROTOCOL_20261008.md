# Frozen original20 VDD/ADE alias trial

Goal: a credible conditional alias treatment within6x VIP whole-image
latency, crossing saved paper numeric targets VDD54.3 and ADE15029.1. Crossing
these numbers alone is not a certified reproduction or an alias-module gain.

Original20 strings and class order remain unchanged, including ADE whitespace.
VDD uses the prior-developed ImageNet/strength1/g0.5 profile whose original20
full result was55.0754; exact full-confusion replay is required. ADE uses pinned
segmentation templates, original Geometry readout, g0.5, wide short336/cap672.
This ADE20 hypothesis cannot inherit the historical variable433-word score.
Local long896/crop512/max4 encodes; wide crop336/max4; no fine RGB or extra head.
No mask-based configuration or penalty fitting is performed in this trial;
profiles came from prior labelled development, so results remain exploratory.

For each local normalized alias t, a is its parent canonical and b is the
nearest other canonical under text cosine (stable category-order ties).
u=(b-(a^T b)a)/norm; alpha=max(t^T u,0). For existing Geometry feature f,
q=alpha*max(f^T u,0). Canonical q is explicitly zero. Norm<=1e-6 receives zero
penalty as an unidentifiable direction; no alternate-rival search. Coefficient1
is frozen; neither the changed text nor surviving alias mass is renormalized.

Local L'=log(sum_k exp((f^T t_k-q_k)/0.07)/20). The equivalent gate is
exp(-q/0.07), with canonical gate1 and all20 original denominator slots.
Wide salience/readout and all wide aliases remain unchanged. The final score is
L'+0.5H(W-L')=Z0+(I-0.5H)(L'-L), using the original Geometry reconstruction H.
Parent orthogonality is an algebraic constraint, not semantic ground truth.
True subclasses and shared parts may be wrongly suppressed.

Four frozen arms: Base/q0; Penalty/original q; Pooled/class mean over19
noncanonicals; AliasShuffle/fixed noncanonical permutation seed20261008.
Pooled preserves total cosine penalty, not exact LME change or writer norm.
Shuffle preserves its multiset and canonical protection. Same20/template VIP
is a fifth comparator, retaining its dataset inference settings and explicit
empty-row numerical repair. All predictions precede mask loading.

The new online penalty maximum shape is[patch,class,20], using canonical cosine
gathers already available from the local alias dot products. Offline text setup
uses[alias,class]; no online[patch,alias,class] competitive tensor exists.
Unit checks cover dense-vector/gather agreement, canonical protection,
fixed-slot identity/nonincrease, exact coupled writer, singleton/joint equality
and pooled/shuffle controls. Real-image mask-free smoke precedes evaluation.

Full coverage: VDD80, ADE2000. Save per-image confusion for1000 paired image
bootstrap resamples, seed20261008. Primary must improve Base and both controls;
tiny gains with zero-crossing intervals are inconclusive. Both paper numeric
targets and<=6x speed against both same20/template VIP and official-query/
official-configuration VIP are required before declaring the requested target met.
No control is promoted as a replacement primary; failures are preserved.

Timing uses first/middle/last complete images,5 rotated warmed synchronized
singleton repetitions and physical-GPU occupancy checks before/after each.
Include resize, encoders, semantic scoring, alias/writer, restoration and
argmax; exclude decode and model/text/plan initialization. Shared-resident peak
memory is recorded and is not an isolated VIP deployment memory comparison.
The ADE official speed reference uses its pinned seg/tau5/tem5/threshold0.07
configuration and short336/max2048 view; VDD uses pinned official short queries.
Only one official prediction is restored/returned in its timed arm; mask-free
smoke checks its equality to the previously preserved official implementation.

Astra high's read-only code review found the same-space units, canonical gather,
fixed20 LME, I-gH writer and four-arm separation consistent with this protocol.
It did not establish a positive alias effect. Extra semantic heads=0 is relative
to this task Base, not a claim that Geometry executes only one head. Full-stage
maximum_penalty_elements is the average of each image's maximum, not a global peak.

SIGSTOP reservation proved ineffective because tmux continued its stopped pane.
The incumbent dispatcher alone was terminated with checked process identity;
all independently running GPU workers and its queue were preserved. One new
controller reuses the valid VDD exclusive timing from v2, recovers ADE timing in
benchmark_exclusive_v3, then runs these trials on naturally idle cards. In
finally it resumes the incumbent queue with existing-worker/output detection.
No worker or model result was removed; all failed timing attempts remain.
