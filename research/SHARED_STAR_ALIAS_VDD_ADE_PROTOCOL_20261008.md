# Fixed-budget conditional alias attenuation on VDD and ADE150

Freeze original20 vocabularies, checkpoints, ImageNet template banks, Geometry
patch-only strength2, bounded local896/wide448 observation policy, and original
coupled writer. Use the existing SharedLocal VIPProxy_Two local semantic head on
complete Geometry backbone tokens; zero extra RGB/fine encodings. Its inherited
head is attributed to VIP, not claimed as a new operator.

Primary: SharedStar2_Soft. Predeclared budget sensitivity: SharedStar4_Soft.
Controls: unchanged Patch2, same-source ObservationMean, same-star pooled-class
attenuation, same-star within-class noncanonical risk shuffle (seed20261008), and
matched20 finite-row-repaired VIP. No dataset-specific choice between budgets.

At each valid query, the unmodified same-source observation mean selects top1
as a center and the next2 or4 classes as opponents. Observe only the two
directions of these edges. Original all-class scores remain available.
Keep the original wide-positive/local-negative continuous N/(P+N) risk,
canonical protection, and fixed-slot outside-exponent attenuation. Crop
logmean/softmax precedes stencil interpolation. No survivor redistribution.

For edge j, t_j=action(center->rival_j)-action(rival_j->center). The minimum-norm
edge-consistent class correction is delta_center=sum(t)/(B+1),
delta_rival_j=delta_center-t_j; graph-external delta=0. The correction is written
as ObservationMean + .5 H delta. Missing edges are absent constraints, not zero
observations. This is a sparse approximation and can amplify individual edge
actions compared with dense all-class averaging; it is not exact acceleration.

Existing evidence tensors [query,class,20] are allowed. Additional risk is
[query,2B,20], stencil temporaries [query,4,2B,20], and writeback [query,class].
No additional [query,all_aliases,class] or [query,class,class] is permitted.
Original Geometry H and all-class text scoring still have their own costs.

VDD80 and ADE1502000 complete validation images, exact unique coverage and
global sample sequence. Match the archived Patch2 full confusion and all arms'
scored target counts. Save per-image confusion and correct->wrong/wrong->correct
pixel changes against ObservationMean. Masks enter only after predictions.
These are developed datasets, not untouched independent validation.

Timing: three fixed complete images (first/middle/last), five rotated warmed
synchronized singleton repetitions, including resize, all encoders, heads,
alias processing, coupled writer, restoration and argmax. Exclude image decode,
model/text initialization. Exclusive target GPU; other GPU tasks allowed.
Models and frozen head snapshots are jointly resident; reported allocation is
not standalone VIP deployment memory. Multi-arm full-evaluation wall time is
not singleton inference latency.

Primary goal: <=6x matched20 VIP full-image time and mIoU above VIP paper numeric
targets VDD54.3/ADE29.1, transcribed in existing research reports. Paper protocols
are not certified identical to these local protocols. Local official-query VIP
VDD52.0647/ADE29.1387 is a separate historical comparator. Numeric target crossing
does not establish word-level causal value: report primary vs mean, pooled and
shuffle on both datasets, without promoting a control after seeing outcomes.

The previous ADE31.1862 used a variable433-query semantic pool, not fixed20.
VDD55.2382 used two routed20-word banks and different read strengths. Both are
whole-system historical references, not results inherited by this candidate.

Prior attempt stopped in the mask-free VDD smoke on a diagnostic interface
KeyError('maximum_risk_elements'); logs and failed source snapshots are retained.
The private diagnostic dictionary repair changes no model equation or incumbent
file. Current outputs are in shared_star_alias_vdd_ade_20261008_r2.

Existing full-suite workers finish normally. Its dispatcher is temporarily
stopped only while reserving two naturally idle GPU slots, then resumed as soon
as the new workers hold their devices. A finite local watcher collects the
completed report and exits; no paused heartbeat is resumed.
