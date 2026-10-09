# Final-Model Decision: Verified Evidence And Open Requirements

## Retained Performance Candidate

Retain `Geometry_PatchOnly2Coupled`, signature
`geometry-bounded896-patch-only-strength2-coupled-v1-20261005`, as the frozen
performance reference. This is a provisional paper candidate, not a declaration
that every contribution or the active research goal is complete.

The original RGB independently feeds bounded896 Geometry and bounded448 wide
observations, at most four actual crops per branch. All20 aliases/class remain
available. There are no fine/native-resolution forwards or alias admission.
Patch queries read geometric patch Values with strength2 and no prefix Values;
prefix queries stay native. This differs from original Geometry's conditional
native-group-mass-preserving operator. Do not transfer that invariant to this
patch-only configuration. The reconstruction uses original, undoubled Geometry
relations: `H=(I+W^T W)^(-1)W^T W`, `Z=L+H(B-L)`.

All eight full datasets cover20092 unique images. The candidate mean47.6399
exceeds the stronger of the measured repaired VIP20 and official/distilled VIP
protocols, mean44.0039, by3.6360pp; all eight domains and LoveDA P improve.
Three warmed fixed complete images/domain yield primary domain-average times
228.09-401.76ms and VIP20 ratios1.255-2.617x. These24-image measurements are not
full-dataset throughput; shared-resident peaks are not standalone memory.

## Full Same-Information Coupling Attribution

The unchanged controls completed all20092 images. Source vocabularies,
checkpoints, full unique coverage and every primary per-image confusion agree
with the original candidate suite. Controls reuse identical local/wide
observations, with no additional encoder, word rule or fitted coefficient.

| Comparator | Eight-domain mean | Candidate gain pp | Conditional paired95% interval pp |
| --- | ---: | ---: | --- |
| Patch-only2 local |47.0452|+0.5947|[+0.0765,+1.1966]|
| Lifted wide |44.1723|+3.4676|[+3.2687,+3.6755]|
| Equal logit mean |46.7713|+0.8686|[+0.7850,+0.9685]|
| Correspondence-permuted H |43.6647|+3.9752|[+3.5238,+4.4223]|

LoveDA D counts once; P is separate. The candidate exceeds equal mean on8/8
domain primary point estimates, but LoveDA P loses0.1020pp. OEM and LoveDA D
intervals against equal mean include zero. Candidate-minus-local is negative
on Potsdam, UDD5, OEM, LoveDA and FLAIR-1; OEM loses7.6297pp and LoveDA D
loses1.9295pp. LandCover.ai correspondence permutation is0.4804pp higher, with
an interval spanning zero. Do not claim universal positive transfer or that
correct spatial correspondence benefits every domain.

The2000 paired bootstrap replicates resample existing filename-based source
groups, recompute aggregate confusion mIoU, and average independent within-domain
draws. These are conditional development intervals, not method-selection-adjusted
uncertainty or verified independent acquisition/test evidence. The group rules
and per-domain intervals are retained in the paired report.

## Alias Contribution Is Not Yet Part Of This Candidate

The old native-resolution RivalFineHard20 gains1.4129pp over its matched
unscreened full coupling, improving all eight domain point estimates. It uses
additional physical fine observations. Its gain and cost cannot be assigned to
the current bounded candidate. Earlier count-matched random controls support
conditional admission on developed windows, not the present bounded model.

Bounded SharedRivalSoft instead changes mean47.2986 to47.2694 (-0.0292pp),
improves only UDD5, and fails to beat its weight-shuffle mean. It is not an
effective module based on these results. This does not prove all aliases are
useful, nor that every hard/soft selection method is ineffective.

The next alias test must isolate conditional word use from additional visual
information: identical observations with/without admission, class-only weights,
and alias-identity-shuffled weights; clean20 and vocabulary pollution/30/40
stress; corrected coverage versus reduced competitor activation. Any extra
observation must have an explicit per-image encoding cap and measured whole-image
cost. No failed nearest-rival/shared-risk variant is promoted by this report.

## Crop-Head Witness: Verified Rejection

The frozen `CropHead_RivalSoft` pilot completed96 unique complete images:
UDD5 full40 and eight evenly spaced images in each other domain. Every original
Geometry, no-admission and patch-only2 per-image confusion matches the preceding
same-field panel. All arms have identical per-image scored target counts;
checkpoints, vocabulary and ordered sample identities agree. This is developed
pilot evidence, not the complete20092-image suite.

Mean46.6450 becomes46.5911 (-0.0539pp), with only2/8 primary domain wins. The
primary is below class-mean46.6354 and alias-identity-shuffle46.6348. The hard
control reaches46.4049 and is not promoted. No threshold, weight rule or
domain-dependent route was changed after these results.

All eight independent serial timing panels are complete: three fixed full images
per domain and three warmed synchronized repetitions. The primary costs
559.73-1123.15ms per domain-average image; the mean of paired domain latency
ratios is2.7058, versus the predeclared maximum1.3. LoveDA exceeds1000ms.
Zero extra RGB/backbone encodings does not remove additional frozen semantic-head
and alias/rival action computation. Both the accuracy and cost gates fail.

The losses are not explained by a universal inability to suppress competitors.
On the VDD pilot vehicle IoU rises23.8613->24.7810, with precision24.52->25.51,
but vegetation IoU falls45.8052->45.3270. Potsdam low-vegetation recall falls
62.6009->61.7524 and correct coverage loses18645 pixels despite reduced false
positives. LoveDA D road gains623 true-positive pixels but adds20087 false
positives, reducing IoU63.0492->62.0848. Suppression must be evaluated alongside
coverage and effects on competing classes, not weight sparsity alone.

Reject this implementation and retain the existing performance reference; do not
launch its full evaluation or tune it into a different primary. These results
do not establish that every alias is useful or that physical fine observation
is universally necessary. They do rule out this cropped-head reuse as a verified
substitute for the earlier physical witness. A further candidate needs both a
defensible reliability source and bounded alias/rival computation; renaming the
same contradiction rule or repeating the prior per-window budget1 observer is
not evidence of a new effective mechanism. The three-part research goal remains
unachieved.

## Whole-Image Detail2 Witness: Verified Rejection

The frozen `Detail2_RivalSoft` candidate completed the same96 developed complete
inputs, with independent paired NPZ verification of all first3 endpoints, image
keys, scored target counts, vocabularies and checkpoints. Two globally selected
detail regions are acquired directly from original RGB; the witness uses native
DINO.text rather than VIP proxy. Actual detail encodings never exceed two per
whole image. This changes acquisition/source/sparse action together; failure
does not isolate their individual causes.

Mean46.6450 becomes46.6778 (+0.0328pp), with3/8 domain wins. The advantage over
class-mean is0.0337pp and over alias-identity shuffle0.0084pp; neither reaches
the frozen0.05pp requirement. The same-observation mean reaches46.8270, above
the primary. The hard control is not promoted. Accuracy criteria fail.

All eight serial timing panels are complete: primary364.81-494.96ms per
domain-average complete image. Every domain mean is below1000ms, but the mean
paired domain latency ratio exceeds the frozen1.3 cap. These are fixed3-input,
3-repeat warmed synchronized timings, not full-dataset throughput or standalone
deployment memory. The lower cost than cropped-head reuse does not establish
more reliable conditional word use.

The VDD pilot vehicle IoU improves23.8613->24.8682 chiefly through reduced FP,
while vegetation IoU45.8052->45.0352 loses219849 TP pixels. Potsdam car IoU falls
31.4528->30.9613 while gaining8 TP and8393 FP pixels. Useful and harmful class
pathways coexist. Retain the frozen performance reference; do not launch full
evaluation or retrospectively relax this candidate's gate. The three-part
Geometry/conditional-alias/coupled-write goal remains unachieved.

## Canonical-Reference Alias Contrast: Verified Rejection

The frozen `CanonicalPair_RivalSoft` repair uses the same bounded base with no
extra RGB encodings or semantic head calls. Canonical-only local/wide agreement
labels pseudo-references, query-excluded class response moments define own/rival
alias contrasts, original G controls dispute action, and original H writes the
outside-exponent weighted-LME potential. This revisits stratified/canonical
reference ideas rather than claiming a new mathematical weighting operator.

All96 developed complete inputs, three original per-image endpoints, target
counts, image/vocabulary/checkpoint identities and actual forward caps verify.
Seven focused CPU tests and VDD/Potsdam mask-free singleton smoke checks pass.
Mean46.6450 becomes46.4056 (-0.2395pp), winning only2/8 domains. Primary is below
class-mean46.6317, alias-identity shuffle46.5450, response-position shuffle46.5719
and alias-free ClassGate46.6866. All accuracy/mechanism advancement checks fail.

Potsdam car IoU31.4528->28.4056, with20 additional TP and56615 additional FP
pixels, while impervious surface loses54996 TP pixels. Vaihingen car loses
3.3601pp and adds66999 FP pixels. This repair aggravates small-class false
activation rather than resolving it. These confusion outcomes do not isolate
reference contamination, source estimation and action selection as causes.

Eight serial timing panels complete: primary328.31-497.48ms per domain-average
complete image; mean paired latency ratio1.4256 fails the frozen1.3 cap. Zero
additional visual forwards does not mean zero weighting/stencil/action overhead.
Retain the performance reference, preserve outputs and reject full rollout.
Neither the favorable FLAIR-1/LandCover.ai entries nor a better control is
promoted. Canonical agreement is not semantic truth; the required conditional
alias contribution remains unachieved.

## Native Detail1 Projected Alias Admission: Verified Rejection

The frozen Detail1_ProjectedSoft candidate uses at most one globally selected
original-RGB detail observation per complete image. Its native-only DINO.text
reader replays legacy native_projected descriptors exactly while removing unused
detail Geometry computation. Alias actions operate on observed, Geometry-supported
donors and baseline top2 competitors; source-level projection limits them to the
native-minus-wide semantic segment before original H writing. This is a bounded
adaptation of earlier projection, not a new projection theorem.

All96 developed complete images, original three per-image confusion endpoints,
ordered image/vocabulary/checkpoint identities and scored target counts verify.
Eight CPU tests and both mask-free VDD/Potsdam smoke checks pass, including frozen
weights, native descriptor replay, singleton equality and actual4/4/1 maximum
Geometry/wide/detail encodings. The initial test failure was an illegal
depth2/head-block0 fixture combination, not a model-rule failure; only that test
combination was excluded before deployment.

Mean46.6450 becomes46.6865 (+0.0415pp), with4/8 domain wins. Primary exceeds
class-mean by0.0410pp and alias-identity shuffle by0.0201pp, below both frozen
0.05pp requirements. Same-detail observation mean46.8056 and pair observation
46.7660 exceed the primary, as does AliasBudgetOnly46.6925. The latter retains
alias-derived magnitudes and is not word independent. Projection adds only
0.0060pp over unprojected soft action. No control is retrospectively promoted.

VDD vehicle gains0.9362pp primarily through39786 fewer false positives, while
vegetation loses57375 true positives. Potsdam car adds735 false positives with
unchanged true positives and loses0.0438pp; low vegetation loses1552 true
positives. Vaihingen car gains0.6443pp through lower false activation. These
mixed outcomes do not establish a reliable conditional alias module.

Eight serial timing panels complete: primary333.96-538.35ms per domain-average
complete image, with mean paired latency ratio1.5099. All domain means are below
1000ms but the frozen1.3 ratio criterion fails; accuracy and mechanism criteria
also fail independently of speed. Shared-resident primary peak allocation is
5857.78-6107.25MiB, not standalone deployment memory. Every worker exited after
complete output. Retain the existing performance reference; do not launch
vocabulary stress/full evaluation or revise the failed gate. The three-part
Geometry/conditional-alias/coupled-write contribution remains unachieved.

## Conditional Covariance: Verified Rejection

The frozen `GeometryCov_RivalSoft` completed the same96 developed complete
inputs. Original three per-image confusion endpoints, ordered sample keys,
vocabulary/checkpoint identities and all-arm scored targets verify. Ten CPU
tests and both mask-free smoke checks pass. The pre-label batch-axis wiring
repair changed no model equation; its original failed smoke root is preserved.

Mean46.6450 becomes46.1547 (-0.4903pp), with2/8 domain wins. Class-mean46.2776,
alias-identity shuffle46.2509, response-position shuffle46.3411 and normalized
pool46.4843 all exceed the primary. Conditional response correlation does not
establish transferable alias utility. Potsdam gains0.7241pp but its class-mean
control is higher; Vaihingen loses2.6961pp and car false positives rise197060.

All eight serial timing panels complete: primary322.84-438.52ms per domain
average; the mean paired latency ratio is1.3752. Every domain is below1000ms,
but both the original cost and accuracy/mechanism gates fail. Shared-resident
memory is not standalone deployment memory. Preserve the reference and reject
this candidate's stress/full rollout. Evidence:
`BOUNDED_COVARIANCE_ALIAS_20261006.md` and
`bounded_covariance_alias_screen_r2_20261006/summary.json`.

## Counterfactual Lexical Binding: Verified Rejection

LexicalBinding_RivalSoft tests a different text measurement: each original alias
is described under own/rival class names, then the existing native descriptor
and original-G supported descriptor compare these frozen text hypotheses.
Original20 scoring banks, local/wide observations, original H and calibration
stay unchanged. All96 developed complete inputs, original per-image endpoints,
NPZ sums, target counts and source identities verify. Nine CPU tests and both
mask-free smoke checks pass; all accuracy/timing workers have exited.

Mean46.6450 becomes46.6112 (-0.0338pp), with4/8 domain wins. Class-mean advantage
is only0.0027pp; word-identity shuffle46.6213 and canonical-only46.6135 are
stronger. Removing Geometry support changes mean by only0.0018pp. Thus this
particular lexical contrast does not establish a conditional word contribution.
Potsdam car gains0.3637pp but low vegetation loses coverage; UDD5 road loses
441002 TP pixels and vehicle gains95372 FP pixels.

Serial singleton domain-average latency296.86-496.99ms, mean paired ratio1.2786:
the frozen cost gate PASSES. The accuracy/mechanism gates FAIL independently of
cost. Extra frozen text setup costs1.67-9.70s/protocol during concurrent accuracy
workers, with1.95-11.25MiB retained direction banks; it is excluded from warmed
inference and not an isolated cold-start measurement. There are no extra RGB
or semantic-head calls. Shared-resident peaks are not standalone memory.

Reject this primary without tuning, stress/full rollout or control promotion.
The retained performance reference and three-part research goal are unchanged.
Evidence: `BOUNDED_BINDING_ALIAS_20261006.md` and
`bounded_binding_alias_screen_20261006/summary.json`.

## Bounded Complete-Image Alias Path Audit

One fixed-source diagnostic completed the same96 developed complete inputs,
including UDD5 full40. Nine CPU tests and two mask-free original-endpoint and
counterfactual batch/singleton replays pass. Every original Geometry,
no-admission and patch-only2 per-image confusion matches the preceding panel;
ordered sample/checkpoint/vocabulary identities and all-action target counts
verify. All eight workers exited after complete output. The original pre-label
WideCrop metadata wiring failure is preserved; r2 changed no equation.

All original20 aliases are audited at fixed original salience, with local,
wide, joint and writeback-bypass paths and two normalization choices. Explicitly
LABEL-ASSISTED canonical-protected donor actions yield local48.7393,
wide49.9044 and joint51.8140 versus baseline46.6450. These are feasible
privileged audit controls, not deployable selector scores, untouched validation
or global attainable upper bounds. Joint changes the acted-on source/budget;
its advantage cannot be assigned to an unlabeled controller.

Wide-only action space is not empty, but local aliases are not certified
harmless. Survivor-normalized policies are below fixed-slot counterparts for
all four paths. Under wide fixed-slot single-word actions896/1159 noncanonical
dataset/protocol/class/word entries have both beneficial and harmful image-level
changes. This supports contextual action utility, not intrinsic semantic verdicts
or a deployable way to identify utility. Prior failed fixed-slot covariance
remains failed; changed normalization is not sufficient by itself.

DirectWide labeled55.0443 versus H-routed wide49.9044 changes correction
magnitude and spatial mixing together. Without matched-magnitude controls it
does not prove H is harmful and cannot promote bypassing Geometry as the final
writer. No label-derived word ranking, threshold or domain route is deployed.
The effective lightweight conditional-alias mechanism remains missing.
Evidence: `BOUNDED_ALIAS_PATH_AUDIT_20261006.md` and
`bounded_alias_path_audit_r2_20261006/summary.json`.

## Independent SigLIP Witness: Verified Rejection

The frozen SigWitness_JointSoft tests one256x256 independently pretrained
SigLIP2-base whole-image observation and at most128 global supported reads.
Original G rows are projected through the actual observer patch footprints;
trained MAP K/V are cached once. Canonical-protected contradiction weights
suppress at most one alias slot per query/class/source, without survivor mass
redistribution. Both local and wide deltas use unchanged original H. Independent
pretraining/support do not certify semantic correctness or constitute a new
encoder contribution.

Ten focused CPU tests and both mask-free real-checkpoint smokes pass, including
native MAP descriptor replay, directly instrumented one-forward execution,
unchanged frozen semantic-head parameters and primary singleton equality.
All96 developed complete images, original three per-image confusion endpoints,
ordered sample/vocabulary/checkpoint identities and all-arm scored targets verify.

Mean46.6450 becomes46.6346 (-0.0104pp), with3/8 domain wins. The primary's
class-mean and alias-shuffle advantages are only0.0011/0.0027pp. NoGeometrySupport
46.6372 is slightly stronger. Beating ObservationMean46.3674 does not establish
word/support attribution or improvement over the unchanged model. VDD vehicle
loses0.4325pp and adds19793 FP pixels; Potsdam car loses0.1326pp and adds2317 FP,
while low vegetation gains0.0855pp through2055 additional TP pixels.

All eight serial timing panels complete: primary389.30-555.06ms per domain
average, mean paired ratio1.6282; every domain is below1000ms but the frozen1.3
ratio criterion fails. Shared-resident peaks7131.42-7281.64MiB include all three
models, not standalone deployment memory. Initialization/text costs4.58-5.03s
were recorded in concurrent accuracy workers and excluded from warmed timing.

Reject the primary without tuning, stress/full rollout or control promotion.
All workers exited and GPU0-7 compute processes are absent; unrelated tmux
sessions and paused automations remain untouched. This does not prove no useful
conditional alias module exists. Fixed-action, magnitude/locality-matched
writing diagnosis remains a defensible next distinction before blaming the
source alone or abandoning H. Retain the established performance reference;
the requested three-part final-model contribution is still incomplete.
Evidence: BOUNDED_SIGLIP_ALIAS_20261006.md and
bounded_siglip_alias_screen_20261006/summary.json.

## Fixed-Action, Magnitude-Matched Writing Diagnosis

One additional diagnosis completed the same96 developed complete inputs.
Original20 words, local/wide views, G/H, salience and native-size probability
assembly remain unchanged. Every Wide FixedSlots single removal is frozen
before masks. ONE label-assisted canonical/true-class/unknown-protected donor
policy supplies the same actions for all writers. Whole-image valid-tile field
norms match per intervention column; this is not native-pixel margin matching.
DirectLegacy alone retains historical padded-field writing.

Seventeen CPU tests and both mask-free endpoint/zero-action/batch-singleton
smokes pass. Every original per-image endpoint, ordered unique key,
vocabulary/checkpoint identity, all-action scored target count and NPZ aggregate
verifies. Historical H/DirectLegacy single-word and privileged-policy confusions
replay exactly. Maximum relative squared-norm error6.5229e-16, zero fallbacks.

All numbers here are LABEL-ASSISTED DIAGNOSIS, not deployed model gains.
Privileged H49.9044, TwoH52.8874, DirectLegacy55.0443, DirectMatched50.7285,
GeometryMatched50.0562, PositiveHMatched49.8979. TwoH gains2.9829pp and matched
direct gains0.8240pp across8/8 domains. This shows amplitude is an important
alternative explanation for the old unmatched direct gap, while localization
also changes the utility of these sparse donor interventions. It does not
establish an unlabeled action estimator or a universally superior writer.

Same H-chosen global/image deletions are different: DirectMatched47.8334/48.6122
are below H47.8547/48.6278. H-based selection favors H; no alternative picks its
own better word. Negative-entry removal is nearly neutral and slightly harms
the privileged policy. Do not blame H signs, abandon Geometry, or promote a
generic gain coefficient from this diagnosis.

Matched direct versus H on the same privileged donor actions increases VDD
vehicle TP2515 and removes18546 FP; Potsdam car TP332/FP-4336 and low vegetation
TP15055/FP-9092. These are available localized-writing effects, not selector
performance. Next separate writing a broad observation from writing a localized
competitive intervention; reliable unlabeled identification still remains
necessary. Retain the established model, make no stress/full rollout or label
tuning, and leave the actual three-part final-model objective open. Every worker
exited; no GPU compute process or paused-automation change remains.
Evidence: BOUNDED_ALIAS_WRITE_MATCH_AUDIT_20261006.md and
bounded_alias_write_match_audit_20261006/summary.json.

## Bounded Legacy Fine-Coverage Transfer Diagnosis

The established legacy fine proxy/admission/writer now has one bounded complete-
image transfer diagnosis, not a new final proposal. Geometry896/wide448,
patch-only2 and original G/H stay fixed. Four256-to512 quadrants per bounded
Geometry tile give at most16 fine encodings/image, with no native sliding windows.
The source reads resized896 RGB; pitch8 is not native-image physical8. No graph
setup forwards are hidden and no model/template/risk/write parameter is fitted.

All96 developed complete inputs, original three per-image endpoints, unique sample
keys, vocabulary/checkpoint identities and all-arm scored targets verify. Ten
real-dependency remote CPU tests and both mask-free source/singleton/frozen-head
smokes pass. Mean46.6450 becomes47.7361 (+1.0911pp),7/8 wins; class-only and
mean identity-null gaps1.0588/1.0155pp support conditional word-identity utility.
FLAIR-1 loses0.2891pp. These are not full20092 or untouched validation results.

The same-fine-information observation mean49.8154 is2.0793pp above the primary
and exceeds it on all domains and LoveDA P. It is a diagnostic control, not an
automatically promoted candidate. The legacy rule uses fine contradiction to
modify wide words; this does not directly exploit all positive fine innovation.
Coverage, reader and physical-scale differences from preceding sparse failures
remain confounded. Do not claim the new result proves fine observations necessary.

Eight serial complete-image timing panels give primary843.12-1622.53ms versus
paired reference228.31-324.05ms; mean paired ratio4.5359. Both the predeclared2
cost ratio and1000ms cap fail, as does superiority to same-information mean.
Old failed1.3 gates remain failed. All workers exited; GPUs0-7 are idle.
Preserve the reference and reject full rollout. Effective evidence has now been
located, but bounded acquisition/computation and conditional writing are still
unresolved. Report/raw: BOUNDED_FINE_COVERAGE_20261006.md and
bounded_fine_coverage_20261006/. The three-part goal remains unachieved.

## Exact Fine Execution: Completed, No Semantic Change

All eight mask-free serial profiles completed on24 fixed complete inputs. Cached
risks, hard-reader scores, predictions and actual caps agree bitwise with legacy
execution. Domain-average warm times fall from960.62-1609.17ms to678.84-845.88ms;
mean paired ratio0.635977, a36.40% reduction. Graph setup0.460-0.509s and16 real
setup forwards are recorded separately. Seventeen CPU tests pass locally with
an import-only cv2 stand-in and remotely with real dependencies.

These are exact execution improvements, not new semantic contributions, full-
suite throughput or rescue of the previous coverage candidate's failed gates.
They also are not timings of the subsequent joint-read candidate. Report:
BOUNDED_FINE_EXECUTION_20261006.md. The original full model remains retained.

## Supported Positive Alias: Useful Word Action, Unsupported Extra G Check

One frozen joint-read primary completed the same96 developed complete inputs.
Fixed20 words, original G/H, local/wide/fine source and calibration stay fixed.
Positive fine and wide class observations are equally combined; the established
wide hard intervention contributes half its potential because it acts on the
wide half. Unlike the old selector, positive fine class evidence is written too.
An extra eligibility condition requires original-G supported fine margins to
agree with the query's contradiction. This is a testable extension of established
components, not a new theorem or independent observer.

Twelve CPU tests and both mask-free original/positive/singleton/frozen-head smokes
pass. Every original three per-image confusion AND same-fine-information mean
matches prior outputs exactly. Ordered96 unique complete keys, source identities,
all-arm scored targets and independently recomputed aggregate mIoU verify.

Mean46.6450 becomes50.0742 (+3.4292pp), but positive observation alone49.8154
accounts for most of this difference. The actual SAME-INFORMATION word-action
gain is0.2588pp,7/8 wins, with class-only/mean word-identity-null gaps0.2426/0.2508pp.
FLAIR-1 loses0.2055pp; LoveDA P is almost neutral (-0.0036pp). Thus conditional
word identity helps this joint observation on developed inputs; it is not just
extra visual information or class-level rescaling.

Removing G ONLY from risk estimation yields50.1639,0.0897pp above primary and
better on7/8 primary domains. Therefore the newly proposed Geometry-supported
word-eligibility condition fails its frozen attribution check. This does not
reject Geometry's local reader or reconstruction writer: those are unchanged
in the control. Permuting ONLY intervention writeback correspondence yields
49.9613,0.1130pp below primary, also7/8 lower; positive observation writing stays
matched. This is limited correspondence evidence, not universal writer novelty.

Compared with same-information mean, VDD vehicle IoU26.8392->27.1147 removes10565
FP but loses154 TP. Potsdam overall gains0.0752pp, yet car33.0103->32.9775 adds
534 FP with10 additional TP. UDD5 vehicle19.2408->19.7048 removes406058 FP but
loses8570 TP. Conditional suppression improves some competition pathways but
does not universally repair small-object correct coverage.

Reject this frozen primary's advancement, with no timing/stress/full rollout,
post-result tuning or control promotion. Its cost is not measured; earlier
execution times must not be assigned to it. Record both positive word-identity
evidence and failed extra-G attribution. Geometry should organize visual reading
and writing, not be assumed to certify every semantic contradiction. Report/raw:
SUPPORTED_POSITIVE_ALIAS_20261006.md and supported_positive_alias_20261006/.
All eight workers exited; retained full reference and goal remain unchanged.

## Joint Competitive Constraints: Word Utility, No Verified Solver Advantage

A frozen primary transports antisymmetric word-intervention pair targets through
original Geometry W BEFORE class projection, then jointly solves class/spatial
constraints with the existing float64 CG. Geometry local logits define a
competition graph, normalized to trace C-1 per valid query; they do not certify
word truth. Positive wide/fine observation and the established fine contradiction
rule remain fixed. This tests joint use/writing of known components, not invention
of weighted graph least squares or a pretrained observer.

All96 developed complete inputs finish. Twenty-seven CPU tests pass locally
with an import-only cv2 stand-in and remotely with real dependencies; both
mask-free original/positive/strong-uniform/singleton smokes pass. Every original
three per-image confusion, same-fine observation mean AND prior NoGeometryRisk
uniform comparator replay exactly. Ordered unique coverage, source identities,
all-arm target counts and independently recomputed confusion mIoU verify.
Mean CG iterations12.03-13.75 across domain/protocol tile averages; explicit
residuals satisfy1e-7, graph/class gauge and same-base word-write norm controls
verify, with no empty-prior numerical fallback.

Primary50.1958 versus same-graph NoAlias49.8074 gives0.3884pp word utility,7/8
domain wins. Class-only/mean identity-null gaps0.3653/0.3715pp support specific
conditional word actions. Geometry-prior position shuffle50.0280 is0.1678pp
lower, but prior assignment is not universally beneficial; LandCover.ai improves
under permutation. These are developed point estimates, not independent evidence.

The stronger exact uniform comparator50.1639 is only0.0319pp lower, with4/8
primary domain wins. Existing separated posterior projection50.1853 is only
0.0105pp lower. Same-NoAlias-base, Frobenius-norm-matched uniform-H intervention
50.1542 is only0.0416pp lower. All three advantages fail their frozen minimum
attribution/advancement checks. Changing class/spatial solve ordering therefore
does not establish the intended useful joint-solver contribution in this test.

Against strong uniform, Potsdam car32.9462->33.3004 gains94 TP and removes5107
FP, but building83.7755->83.4536 loses1592 TP and adds3101 FP. VDD vehicle
27.0151->27.1192 gains495 TP/removes1932 FP, while road16.4450->16.2777 adds
129602 FP. UDD5 vehicle gains0.3484pp through276567 fewer FP but loses4652 TP.
Class improvements coexist with other competition/coverage losses; do not claim
an all-domain, all-class correction mechanism from selected favorable classes.

Reject primary advancement without timing/stress/full rollout, retrospective
tuning or favorable-control promotion. No new singleton latency is measured;
shared multi-arm wall time and previous source profiles are not its latency.
The actual word-specific utility remains useful evidence; extra joint decoder
complexity is not yet justified. Preserve the retained full model and focus
subsequent work on reliable conditional evidence rather than treating a more
elaborate solve as novelty by itself. Report/raw:
COMPETITIVE_CONSTRAINT_COUPLING_20261006.md and
competitive_constraint_coupling_20261006/. All workers exited; goal remains open.

## Reciprocal Soft Admission: Cost Passes, Reverse Veto Loses Coverage

The frozen reciprocal candidate completes the same96 developed full inputs.
Both semantic observers can be contradicted by the other; continuous1-risk
weights use fixed20-slot LME without survivor-mass compensation. Positive fine
information, original Geometry local reader and original H remain fixed, with
no G truth gate, extra visual source, fitted parameter or new joint solver.
There are at most4 Geometry/4 wide/16 fine encodings, not native sliding windows.

Fourteen focused CPU tests and12 positive-source regression tests pass with real
remote dependencies. Both mask-free replay smokes pass. All96 unique ordered
inputs, every original/positive/strong-hard per-image confusion, scored targets,
vocabulary/checkpoint identities and independent mIoU calculations verify.

Primary49.8710 versus same-information ObservationMean49.8154 gives only0.0556pp.
Class-only/mean identity-null gaps0.0802/0.0965pp are positive, but the primary
loses0.2929pp versus established single-sided hard50.1639 and wins only2/8 domains.
LoveDA D loses1.7917pp. Accuracy advancement fails; the retained full model
and previous failed gates remain unchanged. No favorable control is promoted.

The matched single-sided soft control50.2181 differs only by omitting reverse
fine intervention; reciprocal addition loses0.3471pp and loses7/8 domains.
This isolates harmful reverse use under this rule, not a theorem that every
bidirectional mechanism fails. Fixed-slot versus survivor-normalized reciprocal
soft is-0.0229pp, so normalization alone does not rescue it. Word-write H
correspondence gives0.0513pp, not universal spatial or solver superiority.

Versus single-sided soft, VDD vehicle gains0.3446pp through16656 fewer FP while
losing1131 TP; vegetation loses113747 TP. Potsdam car gains0.1040pp through1733
fewer FP while losing52 TP; low vegetation loses12979 TP and0.4609pp IoU.
UDD5 road loses576355 TP and0.7714pp IoU. LoveDA D tree loses41844 TP and2.0343pp;
farm gains19922 TP but adds227938 FP and loses2.6588pp IoU. Suppression can
improve precision for small classes while worsening overall correct coverage.

All eight isolated serial timing panels finish: actual primary660.56-906.40ms
per full-image domain average. Mean paired primary/hard ratio1.040567 passes
the frozen cost condition. Graph setup0.462-0.495s includes16 real setup
forwards; shared-resident peaks5741.60-5910.21MiB are not standalone memory.
These24 fixed-image warm timings are not full-suite throughput. Cost does not
justify the accuracy loss; there is no stress/full rollout.

The next reliability source must distinguish missing observational information
from valid negative evidence. Contradiction alone does not justify symmetric
veto, especially when a wider/lower-resolution view cannot see fine detail.
Neither adding another confidence multiplier nor promoting the favorable
single-sided control establishes that reliability. Every worker and timing
session exited; paused automations and unrelated jobs remain untouched.
Report/raw: RECIPROCAL_ALIAS_ADMISSION_20261006.md and
reciprocal_alias_admission_20261006/. The three-part objective remains open.

## Pool-Isolated Witness: Independent Reference, Unsupported Word Utility

The frozen `PoolIsolated_Canonical` completes the same96 developed complete
inputs. Only judgment margins change to existing raw alias response minus the
rival canonical response, on both wide and fine scales. Prediction retains the
original profiled20-word observations and calibration. The one-sided continuous
wide intervention, positive fine information, local Geometry and original H stay
fixed. No extra RGB/head encoding or fitted parameter is added. A canonical
reference removes noncanonical-pool dependence, not semantic ambiguity.

Thirteen focused CPU tests and26 source regression tests pass remotely. Both
mask-free original/positive/hard/soft/singleton smokes pass. All96 ordered unique
keys, every original/positive/hard/soft per-image confusion, scored target counts,
checkpoint/vocabulary identities and independent confusion mIoU verify. There
are at most4 Geometry/4 wide/16 bounded fine encodings, not native sliding windows.

Primary50.1706 exceeds same-information ObservationMean49.8154 by0.3552pp, but
loses0.0476pp versus the exact previous single-sided soft replay50.2181 and wins
only2/8 domains. Matched-risk-mass ClassMean50.2318 and mean identity-null50.2355
are higher by0.0612/0.0649pp. Therefore this candidate does not establish specific
conditional word utility. LoveDA D improves0.5199pp, while Potsdam loses0.1768pp,
Vaihingen loses0.3036pp and FLAIR-1 loses0.3421pp; favorable entries are not promoted.

The raw/profiled x canonical/pooled source factorial is also unsupported:
primary-minus-RawPool is-0.0607pp, and primary-minus-ProfileCanonical is only
0.0144pp. This does not attribute either claimed isolation benefit or establish
why a canonical reference fails. FineOnly50.0109 does not rescue it. Permuting
only intervention H correspondence lowers mean0.1640pp on7/8 domains, supporting
conditional writing correspondence here, not reliable word identification or
universal spatial superiority. LandCover.ai improves under that permutation.

Versus prior single-sided soft, VDD vehicle gains0.0700pp through3880 fewer FP
while losing354 TP; vegetation loses97027 TP. Potsdam car loses97 TP and adds215
FP; low vegetation loses6854 TP and0.3151pp IoU. UDD5 road loses138320 TP despite
264051 fewer FP. Suppression still trades correct coverage against competition;
pool independence alone does not resolve that tradeoff.

All eight serial singleton timing panels complete: primary640.82-895.76ms per
domain-average complete image; mean paired primary/ProfilePool ratio1.054144.
Both frozen cost checks pass. Graph setup0.459-0.501s includes16 actual setup
forwards; shared-resident peaks5743.50-5926.83MiB are not standalone memory.
These24 fixed-image warmed timings are not full-suite throughput. Accuracy and
word/source attribution fail independently of cost.

Reject this primary without tuning, vocabulary-stress/full rollout or favorable
control promotion. Retain the existing full performance reference. The result
rejects this isolated-canonical witness, not all conditional soft alias use;
the requested three-part contribution remains incomplete. Report/raw:
POOL_ISOLATED_ALIAS_20261006.md and pool_isolated_alias_20261006/.

## Evidence-Local Writeback: Cost Passes, Pair Restriction Adds Little

The frozen `EvidenceLocal_Soft` finishes all96 developed complete inputs and all
eight serial singleton timing panels. Original20 words, positive wide/fine
observations, one-sided continuous risk, local Geometry and original H stay fixed.
Only competitive word writing changes: antisymmetric pair actions are transported
through H, restricted to observed pair support, then projected to class potentials.
One scalar matches the original soft class-field Frobenius norm PER Geometry
tile/protocol, before the unchanged half-strength addition. This is not native-
pixel probability matching or a whole-image fidelity guarantee.

Fifteen focused tests and26 source regressions pass with real remote dependencies;
both mask-free smokes verify singleton equality, frozen heads and original six
endpoints. Collection verifies96 ordered unique keys, every original/positive/
soft/hard per-image confusion, scored target counts, checkpoints/vocabularies,
support/strength invariants and independent aggregate mIoU. No output or retained
model is overwritten, and no previous favorable control is promoted.

Mean50.2216 exceeds same-information ObservationMean49.8154 by0.4062pp, but
the actual new writing gain over GlobalSoft50.2181 is only0.0035pp,4/8 domain
wins. Matched ClassMean50.2108 is only0.0108pp lower, and the mean of three
matched identity nulls50.1993 is only0.0223pp lower. QueryOnly50.2181 has the
same displayed per-domain scores as GlobalSoft. Norm compensation contributes
0.0156pp; primary-minus-Direct is0.0404pp and primary-minus-ShuffledWrite0.0734pp.
These do not establish the intended useful new pair-support mechanism.

The new offline reporter uses2000 paired filename-source-group bootstrap draws
and independently recomputes each aggregate confusion mIoU. Null-model scores
are averaged on the SAME resample, rather than pooling their confusions into an
unevaluated ensemble. Four focused coverage/pairing/metric/null-mean tests pass.
LoveDA D counts once; P stays separate. Conditional development intervals are:

| Comparator | Primary gain pp | Conditional paired95% interval pp |
| --- | ---: | --- |
| Same-information ObservationMean |+0.4061|[+0.2418,+0.5416]|
| Exact previous GlobalSoft |+0.0034|[-0.0037,+0.0096]|
| Matched ClassMean |+0.0108|[-0.0084,+0.0347]|
| Mean alias-identity null |+0.0223|[+0.0004,+0.0492]|
| QueryOnly support |+0.0034|[-0.0037,+0.0096]|
| Matched direct writing |+0.0404|[+0.0065,+0.0751]|
| Intervention-only shuffled H |+0.0734|[+0.0171,+0.1426]|

The word-identity gap is weakly positive under this conditional resampling, not
zero or proof that all words are equally useful. The class-only gap spans zero;
the new writing gap also spans zero. These intervals are not adjusted for method
selection/multiple comparisons and do not turn this developed panel into
independent validation. Frozen minimum-gain gates remain failed, independently
of these statistical descriptions; engineering gates are not CVPR acceptance
rules.

Recorded pair support covers63.77-81.48% of valid pairs across domain/protocol
means. Unrestricted CLASS-field correction power at token/classes with NO
incident support is only0.0092-0.6623%. This is not unsupported pair-edge arrival
power or an error rate. The protected no-incident region carries little existing
intervention power; together with the matched QueryOnly result, this explains
why token eligibility alone offers little leverage here. Pair support still
changes the direction and norm compensation, but its measured accuracy gain is
negligible. These diagnostics do not identify semantic truth or prove a cause
for the remaining weak word-identity advantage.

Versus exact GlobalSoft, VDD vehicle gains0.0168pp through896 fewer FP but loses
75 TP; wall loses3340 TP and0.0655pp IoU. Potsdam car gains0.0081pp through123
fewer FP with unchanged TP; low vegetation gains41 TP and0.0026pp. UDD5 road
gains9679 TP but adds9312 FP, yielding only0.0075pp; vehicle gains156 TP and
removes19925 FP. Small class-specific improvements are not broad error repair.

Actual primary costs666.95-853.62ms per domain-average complete image, with mean
paired primary/GlobalSoft ratio1.006854. Both frozen cost checks pass. Setup
0.463-0.496s includes16 real setup forwards; shared-resident peaks5741.60-
5910.21MiB are not standalone memory. These24 fixed-input warm timings are not
full-suite throughput. At most4 Geometry/4 wide/16 fine encodings operate on
bounded resized inputs, not native sliding windows.

Reject advancement of this added pair restriction without parameter tuning or
stress/full rollout. Preserve the existing full performance reference. The next
missing evidence is reliable, competition-specific word action beyond matched
class calibration, not another unsupported-region mask or a more elaborate
solver. The preserved source and writing controls provide a reference for that
question; they are not newly promoted final models. All evaluation and timing
workers exited. Report/raw: EVIDENCE_LOCAL_ALIAS_20261006.md and
evidence_local_alias_20261006/, including PAIRED_UNCERTAINTY.md. The full three-
part objective remains unachieved.

## Mask-Free Word-Action Path: Variation Exists, No Preferential H Loss

The completed audit replays the existing single-sided soft predictor on16 fixed
first/last developed inputs, two/domain. No masks are loaded, model rules change,
or accuracy is measured. Original score arithmetic and complete predictions match
exactly; frozen weights, checkpoint/vocabulary identities and20 aliases verify.

Within-class word-risk power fraction is0.6877-0.7994; word-specific directed
relative norm is0.3543-0.5259. Thus there is nontrivial source variation, not
identical weights for every alias. Complete-class gradient projection retains
0.3713-0.5891 of antisymmetric word-specific pair power. Original H retains
0.1073-0.1655 of word-potential squared power, versus0.0984-0.1647 for its common
class component, with no consistent preferential word attenuation. Relative
word/total written norm remains0.3254-0.5322, while patch argmax differs from
class-only action on only0.0610-0.9521% of positions across protocol tile means.

These are numerical field quantities, not semantic utility, accuracy contribution,
object recall or native-pixel error. They do not justify blaming H alone or
amplifying the word action. Together with the matched accuracy evidence, the
next source needs a defensible distinction between unsupported lexical response
and correct coverage, not another support mask or more elaborate solve.

One specific untested property is that existing eligible retention P/(P+N)
increases with disputed wide advantage P while negative fine witness N stays
fixed. The new frozen witness-cap source tests exp(fine margin) on IDENTICAL
eligible slots, holding observations/writing fixed and adding own class/identity
and magnitude-matched previous-source controls. This is a source hypothesis,
not an observed cause, a promoted final model or a new exponential invention.
Report/raw: ALIAS_WORD_PATH_AUDIT_20261006.md and alias_word_path_audit_20261006/.
New pre-label protocol: WITNESS_CAP_ALIAS_PROTOCOL_20261006.md. Retained full
reference and earlier failed gates remain unchanged.

## Fine-Witness Likelihood Cap: Clearer Word Attribution, No Best-Model Gain

The frozen source removes broad response amplitude from eligible retention,
using exp(fine margin) on identical protected wide-positive/fine-negative slots.
All96 developed complete inputs finish. Twelve focused tests,26 source regression
tests and both mask-free smokes pass with real remote dependencies. Every original,
positive, previous-soft and hard per-image endpoint, all-arm scored targets,
checkpoint/vocabulary identities and independently computed aggregate mIoU verify.
Canonical/self/invalid protection and eligibility are unchanged; previous-soft
interventions match the new valid-tile class-field norm with no impossible field.

Primary50.1909 versus no-attenuation49.8154 gives0.3755pp. Risk-budget-matched
class-mean and mean identity-null gaps0.0777/0.0796pp initially support word
action on this panel; these controls do NOT match the final written-field norm.
The later audit below supplies that distinction and supersedes stronger lexical
allocation claims, while preserving all historical results and failed gates.
The2000 paired source-group intervals are[+0.0421,+0.1185]/[+0.0422,+0.1215]pp.
These are conditional development intervals, not selection-adjusted/independent
evidence. Original H correspondence gives0.1908pp,7/8 main-domain wins, but is
not universally beneficial; LandCover.ai improves under permutation.

Exact previous soft50.2181 remains0.0272pp higher, with only2/8 primary domain
wins. The paired primary-minus-previous interval[-0.0685,+0.0114]pp spans zero.
Magnitude-matched previous50.1712 is0.0197pp lower, interval[+0.0034,+0.0400]pp;
this small positive source benefit fails the frozen0.05pp practical criterion.
Do not describe it as no word signal or a verified best-model upgrade.

UDD5 vehicle gains982 TP/removes30805 FP, but road loses56943 TP and0.0606pp IoU.
Potsdam low vegetation loses2568 TP despite557 fewer FP; car adds969 FP with2 TP.
VDD vehicle loses135 TP/adds1498 FP. Better attribution does not guarantee better
overall coverage. Removing analytic broad-amplitude protection does not establish
that the original protection was the cause of the weak word advantage.

Reject advancement of this frozen source without tuning, timing, stress/full
rollout or control promotion. No new singleton cost is measured; previous source
timings do not belong to it. Retained full reference and earlier gates remain
unchanged. All workers exited. Report/raw: WITNESS_CAP_ALIAS_20261006.md and
witness_cap_alias_20261006/, including paired statistics. The full Geometry /
conditional-alias / coupled-write objective remains open.

## Actual-Footprint Negative Witness: Stricter Evidence Does Not Improve Utility

The frozen FootprintWitness_RivalSoft source completes the same96 developed
complete inputs. It examines every positive-coefficient fine contributor in the
original interpolation/overlap stencil. Mixed/incomplete footprints stay neutral;
all-negative upper endpoints replace mean negative magnitudes, with unchanged
P/(P+N), canonical protection, positive observations, Geometry and H. There are
no additional vision/head/text calls beyond the existing4/4/16 bounded caps.

The14 source tests and26 source regressions passed before launch; both no-mask
smokes and every original/positive/previous-soft/previous-hard per-image endpoint
verify. Collection confirms ordered96 unique keys, targets, identities,
independent confusions and no support expansion/increased risk/protected action.
Four offline pairing/statistics tests pass. All workers exited after completion.

Primary50.1065 gains0.2910pp over same-information no attenuation49.8154, but
loses0.1117pp versus exact previous soft50.2181, with0/8 main-domain wins and a
conditional paired95% interval[-0.1454,-0.0804]pp. GuardOnly50.1660 also loses
0.0522pp versus previous soft; upper-endpoint weakening costs another0.0595pp.
Neither added conservatism improves this developed panel.

Matched class-mean and mean identity-null advantages remain positive at
0.0433/0.0465pp, intervals[+0.0261,+0.0654]/[+0.0270,+0.0708]pp, but below the
frozen0.05pp practical criterion. Magnitude-matched previous is0.0099pp stronger.
Original H correspondence gives0.1448pp on7/8 domains, with LandCover.ai the
exception. These conditional development results are not selection-adjusted
independent evidence; null scores are averaged on the same resample.

Mixed contributors occur in31.39-46.58% of old eligible comparisons under the
evaluator's domain/protocol averaging, not an error rate. UDD5 vehicle loses9321
TP/adds11239 FP; road loses197801 TP despite fewer FP. Potsdam low vegetation
loses4480 TP and car loses23 TP/adds539 FP. VDD roof adds42796 FP. Interpolation
sign unanimity does not identify reliable lexical action or guarantee fidelity.

Reject advancement without threshold/source changes, control promotion, timing,
stress or full rollout. No candidate latency was measured; previous costs do
not belong to it. The retained full reference is unchanged. Report/raw:
FOOTPRINT_WITNESS_ALIAS_20261006.md and footprint_witness_alias_20261006/,
including paired statistics from ../tools/report_footprint_witness_alias.py.
The three-part objective remains open; reliable competition-specific word
judgment is still the missing mechanism, not another support mask or solver.

## Rival-Word Counterfactual: Reference Stability Does Not Improve Judgment

The frozen RivalOmission_RivalSoft tests the competing lexical reference rather
than spatial footprint unanimity. Every single rival-word omission is computed
from original cached fine evidence with K-1 normalization; ONE word identity is
shared across all actual interpolation contributors before minimizing its score.
Prediction retains all20 words, original salience, positive information, P/(P+N),
canonical protection, Geometry and H. There are no new visual/head/text calls;
the existing4/4/16 bounded observation budget still applies.

Sixteen mathematical/source tests and26 regressions pass remotely; both no-mask
smokes verify frozen heads and singleton equality. All96 developed complete inputs,
six prior per-image endpoints, scored targets, identities and independent confusions
verify. Risk support cannot expand and increases stay below1e-5 arithmetic tolerance;
unknown/protected slots remain neutral. Four offline pairing/statistics tests pass,
and all workers exited after complete output.

Primary50.1474 gains0.3320pp over same-information no attenuation49.8154 but loses
0.0707pp versus previous soft50.2181 on ALL eight primary domains. LoveDA P gains
0.0372pp and is not counted twice. Conditional paired95% intervals are
[+0.1847,+0.4467]pp against no attenuation and[-0.0940,-0.0492]pp against old soft.
GuardOnly50.1950 also loses0.0231pp versus old soft; replacing its negative magnitude
costs another0.0475pp. Neither conservative eligibility nor magnitude improves this
panel. A magnitude-matched previous field is0.0244pp higher than the new source.

Class-only and mean identity-null gaps are0.0423/0.0436pp, conditional intervals
[+0.0252,+0.0646]/[+0.0233,+0.0686]pp. There is a small word signal, but both gaps
fail the frozen0.05pp practical requirement. Rival-reference identity shuffle is
0.0128pp HIGHER, so this experiment does not establish its claimed rival-reference
judgment. Original H correspondence gains0.1678pp on7/8 domains; LandCover.ai is
the exception. Null scores share the bootstrap draw; these developed intervals
are not independent or adjusted for repeated candidate selection.

The source withdraws13.48-22.50% of old eligible comparisons under evaluator
domain/protocol averaging, not a semantic-error rate. UDD5 vehicle adds74999 FP,
building adds225391 FP, and vegetation loses203994 TP. Potsdam low vegetation loses
3692 TP and car adds932 FP. A single influential competing word can be useful:
deletion stability and spatial sign unanimity do not themselves identify harmful
alias contributions. Increasing those restrictions weakens useful action here.

Reject advancement without post-result source/threshold changes, control promotion,
timing, stress or full rollout. New candidate latency is not measured. Keep the
full performance reference unchanged; the requested three-part goal remains open.
Report/raw: RIVAL_OMISSION_ALIAS_20261006.md and rival_omission_alias_20261006/,
including paired statistics from ../tools/report_rival_omission_alias.py.

## Local Lexical Fidelity: Better Mean, Unsupported Word/Writer Contribution

The frozen LocalPath_JointSoft completes the same96 developed complete inputs.
Existing fine features read the SAME normalized local Geometry text bank,
without new salience, RGB/head/text encoding or a fitted coefficient. Protected
fixed-slot outside-exponent local alias actions yield u; changing the anchor
from L to L+u under the original fidelity objective adds exactly(I-H)u. Exact
previous wide soft action, positive observations, original G and H stay fixed.
This tests a new action PATH, not invention of complement/ridge algebra.

14 focused tests and26 source regressions pass with real remote dependencies;
four offline paired tests pass. Both mask-free real-checkpoint smokes verify
frozen weights, primary singleton equality and old six prediction endpoints.
All96 ordered unique images, per-image old six confusion endpoints, scored
targets, vocab/checkpoint identities, independent mIoU and source/action bounds
verify. Every vocabulary keeps20 aliases/class; retained_count_mean counts fully
unattenuated slots per comparison, not soft mass or permanent deletion. All
workers exit normally.

Primary50.3143 exceeds previous soft50.2181 by0.0962pp, with5/8 domain wins.
Its conditional95% paired interval[-0.2767,+0.2569] includes zero; Potsdam loses
0.6451pp, Vaihingen0.2201pp, FLAIR-10.0915pp and separate LoveDA P1.4500pp.
Same-information no attenuation49.8154 is0.4989pp lower. However, the new local
class-mean control50.3215 is higher, and local word-identity nulls average50.3095:
both attribution intervals include zero. The established wide intervention is
held fixed in these controls, so this failure concerns the NEW local word path.

Magnitude-matched previous soft50.8993 is0.5850pp higher, with paired interval
[-0.8779,-0.3928]. DirectMatched50.3319 is also higher; correspondence shuffle
50.2832 is only0.0311pp lower, with interval spanning zero. Fine-reference
position matters in aggregate(+1.9096pp), but does not identify word-specific
utility. The local/wide2x2 has negative mIoU difference-of-differences-0.0767pp;
this is metric-level evidence, not an algebraic nonadditivity claim.

Potsdam car adds86 TP but29339 FP and loses1.8057pp; Vaihingen car adds1003 TP
but43006 FP. UDD5 road gains0.6405pp, yet vehicle adds566873 FP and loses0.5866pp.
This candidate does not repair balanced small-object fidelity. Frozen accuracy/
mechanism criteria fail independently of cost. No timing, stress/full rollout,
post-result tuning or magnitude-control promotion follows. The retained full
performance reference and three-part research goal remain unchanged. Evidence:
LOCAL_ALIAS_PATH_20261006.md and local_alias_path_20261006/.

## Semantic Rival Reference: Word Attribution, No Verified Matching Advantage

The frozen SemanticRival_Soft completes the same96 developed complete inputs.
Existing normalized mean VIP query text vectors define a0.07-scale semantic
softmax over each rival class20 words, separately for every source alias.
Weighted rival log evidence replaces the uniform pool ONLY in the wide/fine
one-sided judgment. Prediction scores/salience, positive observations, local
anchor, original G/H and fixed-slot soft writer remain. There are no new RGB,
semantic-head or text encodings; text matching/weighted LME are known operations.

17 focused tests and26 original regressions pass remotely. Both mask-free real
checkpoint smokes verify frozen weights, actual caps, old six endpoints and
singleton equality. All96 ordered unique inputs, old six per-image confusions,
scored targets, vocabulary/checkpoint identities, independent mIoU and source/
action invariants verify. Every worker exits normally.

Primary50.2261 gives0.4107pp over same-information no attenuation49.8154. It
exceeds matched class-mean by0.0869pp and mean risk identity null by0.0846pp;
their conditional paired intervals are[+0.0570,+0.1253] and[+0.0532,+0.1266].
Thus assigning weights to specific source words has supported utility here.
However, NEW reference gain versus previous soft is only0.0080pp,3/8 wins,
with interval[-0.0088,+0.0318]. The magnitude-matched old reference is only
0.0044pp lower, also spanning zero. Separate LoveDA P loses0.1521pp.

Correct semantic keys beat shuffled keys by only0.0229pp and magnitude-matched
keys by0.0165pp; BOTH intervals include zero. Risk identity attribution is not
semantic-reference key attribution. Existing H correspondence gives0.2177pp
relative to its intervention-only permutation, but matched direct writing is
only0.0335pp lower with interval including zero. Entropies2.7281-2.8654 remain
close to uniform log20=2.9957; risk-support changes are1.29-2.58% of valid
alias/rival comparisons, not pixel/error fractions. No sharpening is fitted.

VDD vehicle adds2679 FP, Potsdam car938 FP, UDD5 vehicle128845 FP and Vaihingen
car3898 FP; all four corresponding IoUs fall relative to old soft. The new
reference does not establish balanced small-object repair. Frozen accuracy/
mechanism criteria fail; no timing, vocabulary stress, full rollout or post-result
source/control promotion follows. No new latency is borrowed. Preserve the full
reference and leave the complete goal active. Evidence:
SEMANTIC_RIVAL_ALIAS_20261006.md and semantic_rival_alias_20261006/.

## Incumbent One-Sided Rule: Missing Attribution Controls Completed

The exact existing single-sided soft rule was frozen as an attribution probe,
not invented, tuned or promoted as a final model. Twelve tests plus26 regressions
and four offline paired tests pass. Both real-checkpoint no-mask smokes, all96
unique developed complete inputs, six old per-image endpoints including the
PRIMARY, source identities, exact20 counts, targets, confusions and4/4/16 caps
verify. Every accuracy and serial timing worker exits after complete output.

Primary50.2181 exceeds same-fine-information no attenuation49.8154 by0.4027pp on
8/8 main domains, conditional95% interval[+0.2386,+0.5357]. Fine positive evidence
alone accounts for3.1704pp of its3.5731pp pilot gain over the bounded base. This
is not full20092 or independent/selection-adjusted evidence. Soft-over-hard
0.0543pp has an interval spanning zero.

At identical valid class-field write norm per tile, competitor-conditioned risk
beats rival-collapsed risk by0.1239pp[+0.0800,+0.1752], and correct word-write
correspondence beats its permutation by0.0964pp[+0.0353,+0.1741]. Direct matched
writing is0.0370pp lower with a positive interval, but below the0.05pp practical
criterion. These are existing competitive/write effects, not new mathematics.

Word allocation beyond class calibration is NOT verified: matched ClassMean
is only0.0064pp lower[-0.0120,+0.0309], matched word identity null only0.0190pp
lower[-0.0021,+0.0461]. Unmatched ClassMean is0.0550pp lower with a positive
interval. Matching inherits the primary WORD-DERIVED total correction norm;
these nulls are not deployable word-free replacements or proof that every word
is useful. The evidence separates competitive context from budget-preserving
lexical allocation. Full-objective word attribution remains open.

Eight isolated serial timing panels give primary664.88-854.57ms, mean paired
ratio1.157599 versus SAME-FINE no attenuation. Both frozen cost checks pass;
LoveDA alone has29.25% word overhead. VIP20 is96.36-203.03ms, primary/VIP4.156-
6.900x. These24 complete-input warm timings are not full throughput or standalone
memory. Prior timing is not substituted. The matched word-allocation checks
fail independently of cost; no tuning/control promotion or automatic full/stress
launch follows.

Equal fusion's large LandCover.ai pilot point loss is partly a union-mIoU class
participation discontinuity:9 building false positives with no building targets
add a fifth scored class. Its across-domain interval spans zero. Do not transfer
the verified full bounded-base equal-fusion result to this fine-source route.
Report/raw: ONE_SIDED_ALIAS_ATTRIBUTION_20261006.md and
one_sided_alias_attribution_20261006/, including paired claim/cost checks.

## Incumbent Vocabulary Stress: Useful Calibration, Missing Lexical Allocation

An intentional diagnostic of the unchanged one-sided rule completed all96
developed complete inputs under existing20/30/40, wrong-parent and legitimate
paraphrase pools. Local Geometry20 stays fixed; all scenarios share4/4/16 inference
observations. Both no-mask smokes replay independently running all13 endpoints
under every scenario and primary singleton. Every clean20 per-image endpoint
replays the preceding audit. Unique coverage, original/checkpoint/context/pool
SHA identities, scored targets and aggregate confusions verify. Five new source
tests,12 existing rule tests and three offline support/statistic tests pass.

At common scored-class support, soft exceeds same-information no attenuation
by0.3997/0.4292/0.4186/0.3257/0.3509pp for20/30/40/wrong-parent/paraphrase, all
with positive conditional paired intervals. Original native metrics are retained;
common support adds LandCover.ai's absent building class through another arm's
false predictions, not a change in clean primary outputs. These are developed
pilot diagnostics, not full20092 results or independent validation.

Wrong-parent matched class-mean and word-identity nulls are0.0250/0.0191pp HIGHER
than primary, with intervals spanning zero.40 also has no word allocation gain;
paraphrase's0.0311pp identity gain does not resolve uncertain class attribution.
Matched nulls inherit primary word-derived written norm and do not demonstrate a
deployable word-free replacement. Competitor conditioning remains supported in
all scenarios (0.1064-0.1639pp); write correspondence except40. Relative wrong-parent
protection versus no attenuation is-0.0740pp[-0.1838,+0.0612], not verified worse.
Wrong-parent's own primary mean INCREASES0.4093pp, so its construction name is not
an empirical bad-word label. No favorable control or pool is promoted.

OEM20->40 building IoU63.4347->12.1722, losing1831094 TP and routing many target
building pixels to developed space. UDD5 instead improves51.2305->53.7309.
The immutable rule never attenuates fine-nonnegative entries, including jointly
wrong broad/fine advantages. This is a code-level scope limit, not an isolated
causal explanation of every expansion loss. A final repair needs more than a
stronger consistency veto and must separate positive-observation contamination
from additive word action. Existing clean timings cannot become30/40 latency
claims. All workers exit; outputs/reference/paused automations are preserved.
Report: ONE_SIDED_ALIAS_STRESS_20261006.md and one_sided_alias_stress_20261006/.

## Held-Out Cross-Image Alias Reference: Mean Gain, Unsupported Source Alignment

The frozen CrossRef_Soft adds a disjoint RGB calibration reference to the
unchanged one-sided rule, ONLY on wide-positive/fine-nonnegative comparisons.
Original20 scoring pools, positive fine/wide observation, Geometry and H remain.
Per-alias scalar rank neighborhoods are balanced by canonical-only pseudo-class
mass. These rank/density operations have precedents; cross-image source
provenance is the tested factor, not a new mathematical theorem.

All eight references freeze64 RGB images/domain before pilot masks: UDD5 train
RGB, remaining evaluation RGB elsewhere, explicitly transductive.512 unique
reference images are disjoint from pilot/timing files.16 new tests,26 original
regressions,4 offline paired-engine tests and both real-checkpoint mask-free
smokes pass. All96 developed complete inputs, six retained per-image endpoints,
target counts, source identities, independent confusions/mIoU, actual4/4/16
query caps and matched norms verify. The pre-RGB UDD5 duplicated-root failure
is preserved; its wiring repair changes no model rule. All workers exit.

Primary50.3510 gives+0.1328pp versus old soft50.2181,6/8 domain wins and
conditional95% interval[+0.0681,+0.1905]. Same-information no attenuation
49.8154 is0.5355pp lower. However, matched class-mean advantage0.0203pp is
uncertain; matched word-identity advantage0.0372pp is positive but below the
frozen0.05pp practical requirement. The old soft direction at primary written
norm is0.0853pp HIGHER, with interval[-0.1627,-0.0125] for primary-minus-control.
Reference-position shuffle is0.1075pp HIGHER, primary interval
[-0.1840,-0.0301]. Thus mean gain does not verify beneficial corpus category
alignment or better lexical allocation at fixed strength. Nulls inherit primary
word-derived total norm and are not word-independent deployable replacements.

Rival conditioning(+0.1344pp) and H word-write correspondence(+0.1398pp)
remain supported. In-image reference is0.0146pp higher, uncertain; direct
matched writing is only0.0119pp lower, uncertain. UDD5 road gains1.3165pp but
vehicle loses0.6922pp/adds688165 FP. VDD vehicle loses0.5410pp/adds23126 FP;
Potsdam car loses0.0459pp/adds721 FP. Balanced small-target repair is unproven.

Reject advancement without post-result tuning, control promotion or timing/
stress/full rollout. Calibration costs20.73-37.76 seconds/domain with872-1024
real fine encodings, separately recorded graph and index costs; no new query
visual encoding does not make lookup free. No primary latency is measured or
borrowed. Keep the full reference unchanged and leave the complete goal active.
Report/raw: CROSS_IMAGE_ALIAS_REFERENCE_20261006.md and
cross_image_alias_reference_20261006/.

## Pair-Likelihood Consumer: Word Gain, Missing New Retention Attribution

The frozen PairLikelihood_Soft keeps the exact prior one-sided risk, fixed20
outside-exponent action, positive broad/fine observations and original Geometry
relations. It changes only the consumer: directed A/B actions enter target pair
probabilities before local-fidelity/Geometry-constrained likelihood reading.
Bradley-Terry, convex fidelity and accelerated gradient are known tools, not
claimed inventions.24 fixed steps and the4/C coefficient are not label fitted.

15 focused math/source tests,26 regressions, four offline paired tests and both
real-checkpoint no-mask smokes pass. All96 developed complete inputs, six old
per-image endpoints, identities, exact20 vocabularies, scored targets, independent
confusion/mIoU and actual4/4/16 caps verify. All workers exit normally.

Primary50.2526 exceeds new same-information NoAlias49.8594 by0.3932pp,8/8 domain
wins, conditional paired95% interval[+0.2745,+0.5322]. Its old-soft gain0.0345pp
has interval[-0.0766,+0.1541],5/8 wins. LoveDA P separately gains0.0585pp.
Unmatched ProjectFirst gain0.0276pp is positive, but matching NEW primary-minus-
NoAlias word-intervention norm reduces it to0.0065pp[-0.0005,+0.0149]. Matched
old-direction gain0.0062pp is positive but below the0.05pp practical requirement.
Matched class gain0.0117pp is uncertain; identity-null gain0.0246pp is positive
but does not establish class attribution. These norm controls differ from old
TOTAL-correction matching and still inherit a word-derived budget.

Competitor conditioning0.1176pp[+0.0745,+0.1669] and word-action spatial
correspondence0.1376pp[+0.0733,+0.2324] remain supported. Direct matched writing
is0.0115pp HIGHER, with negative primary interval. Target pair probabilities
are sufficient only through their class row sums in this complete-graph loss;
the new exact test verifies equal-row-sum invariance. Nonlinear aggregation
before compression is not full edge preservation through the final solve.

Final image/tile-average gradient diagnostics are approximately1e-12, so observed
nonconvergence is not an explanation supported by this panel. VDD/Potsdam/
Vaihingen car or vehicle IoU lose1.1971/1.3422/1.8718pp, adding58584/21916/38176
FP. UDD5 vehicle adds551824 FP. Jointly wrong positive sources remain uncorrected.

Reject advancement without new timing, stress/full rollout, tuning or favorable
control promotion. No latency claim is borrowed. Keep the full reference and
three-part research goal unchanged. Evidence: PAIR_LIKELIHOOD_COUPLING_20261006.md,
pair_likelihood_coupling_20261006/ and its frozen protocol. This consumer-only
test redirects the next action to missing conditional lexical evidence.

## Unchanged Likelihood-Cap Audit: Competition, Not Verified Word Allocation

All96 developed complete inputs and eight serial timing panels complete.
Twelve earlier per-image endpoints replay exactly. Coverage, target counts,
checkpoints, vocabulary20/class, independently recomputed confusion mIoU,
canonical protection, class gauge and actual4/4/16 encoding caps verify.
Ten new audit tests,50 existing regressions, four paired-statistics tests and
both real-checkpoint mask-free smokes pass. The cap itself is unchanged.

Primary50.1909 exceeds same-fine no attenuation49.8154 by0.3754pp
[+0.1737,+0.5220],8/8 primary domain wins. It remains0.0273pp below previous
soft50.2181, interval[-0.0685,+0.0114]. After matching the WORD-ONLY valid
class-field norm after H, class-mean advantage shrinks to0.0097pp
[-0.0084,+0.0280] and mean word-identity-null advantage to0.0214pp
[-0.0010,+0.0429]. Neither allocation claim passes. These nulls still inherit
primary word-derived budgets; they are not deployable word-free models.

Matched rival collapse is0.1078pp lower [+0.0569,+0.1562]; matched word-write
permutation0.0900pp lower [+0.0302,+0.1592]. Thus competition and correspondence
remain supported on this development panel, not semantic truth or new algebra.
Equal fusion's1.9511pp point gap has an interval spanning zero and the known
LandCover.ai absent-building participation discontinuity.

Actual primary means664.97-844.26ms; paired ratio1.0011 versus old soft,
1.1588 versus same-fine no attenuation,5.8791 versus VIP20. LoveDA alone has
29.25% word overhead. Shared-resident peaks5741.60-5910.21MiB are not standalone
memory; graph setup0.458-0.506s has16 real setup forwards. All workers exit.
The original cap rejection remains; no model tuning or full rollout follows.
Report/raw: WITNESS_CAP_ATTRIBUTION_20261006.md and
witness_cap_attribution_20261006/. The full three-part goal remains active.

## Deployable Class Calibration: Existing Word Dose Is Not Necessary Here

One frozen diagnostic completes the same96 developed complete inputs. It adds
NO new primary rule: all original/positive/soft/hard and previous audit endpoints
replay exactly. The new control computes P=max(B_c-B_r,0), N=max(F_r-F_c,0)
from existing wide/fine CLASS logits; positive-wide/negative-fine comparisons
use N/(P+N), uniformly over noncanonical slots. Canonical1 and original fixed
slots/pair projection/H stay. An analytic canonical/complement mass read equals
explicit uniform alias weights, including underflow and arbitrary canonical
positions. It never reads primary word risks, means, write norms or fitted dose.
It is not alias-free visual scoring: the original all20 bank is still scored.

Twelve new tests,38 source regressions, four paired-engine tests and both
mask-free real-checkpoint smokes pass. Unique full-input coverage, identities,
target counts,20/class, independent confusion mIoU and4/4/16 caps verify.
All eight workers exit with complete results; GPUs0-7 idle. No retained model,
prior output or paused automation changes.

Existing one-sided soft50.2181 versus deployable pooled50.5108 gives-0.2926pp
[-0.4270,-0.1707], only1/8 primary domain wins. The specific word-DOSE contrast
fails, not only norm-matched word allocation. Matching the pooled post-H word
write to primary yields50.2212, primary delta-0.0030pp[-0.0526,+0.0396]. All
fields match with approximately1e-16 norm error. Raw pooled/word write-norm
ratios1.81-2.06 and directional cosines0.75-0.91 retain evaluator averaging.
Thus correction strength explains much of this raw gain, not a demonstrated
fine-grained lexical judgment. Neither every alias nor every class-only model
is proved equivalent. Rival/spatial attribution remains supported.

The raw pooled control improves UDD5 road0.7674pp and vehicle1.0390pp, Potsdam
low vegetation0.5842pp, but VDD vehicle loses0.2887pp, Potsdam car0.0798pp and
Vaihingen car1.0086pp, each adding FP. A mean winner is not balanced small-object
repair. This is development diagnosis, not full20092 or independent validation.
No timing or full rollout of the new control, dose tuning or control promotion
follows. Next word evidence must address jointly positive misleading response
and exceed this deployable calibration, not only the older weaker no-attenuation
baseline. Report/raw: ALIAS_BUDGET_ATTRIBUTION_20261006.md and
alias_budget_attribution_20261006/. Full three-part objective remains active.

## Own-Class Single-Word Exclusion: Mean Gain, No Lexical Allocation Advantage

The frozen OwnPeer_Soft completed96 developed complete inputs, unchanged20
words, positive wide/fine information, Geometry and H. The tested word is
excluded across every contributing fine token/crop with frozen original
salience; wide CLASS advantage and negative own-peer fine CLASS margin set
1-N/(P+N). Canonical/self/unknown entries stay protected. There are no extra
RGB/head calls beyond the incumbent4/4/16 observer, fitted parameters or quotas.
Leave-word-out arithmetic is not a new invention or semantic-truth certificate.

Both mask-free real-checkpoint smokes pass. All original/positive/previous
soft/hard/deployable pooled-class per-image endpoints replay exactly; the new
primary is not unchanged. Unique coverage, identities, targets, independent
aggregate mIoU and encoding caps verify. All eight workers exit after complete
outputs. Source tests passed before launch; existing paired statistics verify
the collected panel. No original source rule or retained full model is changed.

Primary50.5130 versus old soft50.2181 gains0.2948pp[+0.1642,+0.4370], and
same-information49.8154 gains0.6975pp[+0.4285,+0.9477]. But deployable pooled
class50.5108 is essentially tied: primary+0.0022pp[-0.0113,+0.0190]. Primary
wins5/8 pooled point estimates; worst protocol delta-0.0844pp. All norm matches
are possible with errors about1e-16. Primary loses0.0149pp to matched pooled,
0.0121pp to matched class mean and0.0130pp to mean3 matched identity nulls.
Lexical allocation/source advancement fails. Matched rival conditioning gains
0.1475pp[+0.0717,+0.2224], spatial word-write correspondence0.1958pp
[+0.0600,+0.4024], and original H versus direct writing0.1467pp
[+0.0716,+0.2137] remain supported on this development panel.

Mean absolute omission shift0.0230-0.0309;19.70-22.75% of acted word/rival
entries have positive wide AND fine word margins, reaching the old blind spot.
Only2.72-4.38% have nonnegative full fine CLASS margins, so most action is still
class-level disagreement. These are averaged support diagnostics, not semantic
error fractions. The new wide class margin and fine reference change together;
old-soft gains do not isolate omission. Peer redundancy cannot certify truth.

UDD5 road44.7628->45.8878 gains1032010 TP but448177 FP versus old soft.
VDD vehicle26.8570->26.5502 adds12263 FP, Potsdam car33.1180->32.9556 adds2630
FP, and Vaihingen car27.2431->26.0316 adds23034 FP. Improved mean is not a
balanced small-object correction. No timing, stress/full rollout, post-result
tuning or favorable-control promotion follows the failed frozen source gate.
The three-part objective remains open. Report: OWN_PEER_ALIAS_20261006.md;
verified raw/statistics: own_peer_alias_20261006/.

## Word-Specific Competitive Excess: Mean Gain, No Source Advancement

The frozen `ExcessCap_Soft` completes96 developed complete images, unchanged20
words, original Geometry/W/H and positive wide/fine observations. It uses
w=exp(-max(b-f,0)) on wide-positive sampled alias/rival margins, including
jointly positive cases; canonical/self/invalid entries are protected. All
slots and source salience stay fixed, with no survivor redistribution or new
RGB/head calls beyond4/4/16. This is a scoped fixed-slot test of the old blind
spot, not new exponential/ridge mathematics or a semantic-truth certificate.

Eleven new source tests,36 existing regressions and four offline statistics
tests pass. Both real-checkpoint no-mask smokes and primary singleton parity
pass. Original/positive/previous soft/hard/OwnPeer/pooled-class per-image
confusions replay exactly. Unique ordered coverage, checkpoints/vocabularies,
exact20 counts, paired targets, independent aggregate mIoU and encoding caps
verify. All eight workers exit complete; retained full20092 is unchanged.

Primary mean50.6787 exceeds same-information49.8154 by0.8632pp
[+0.1563,+1.3645], old soft50.2181 by0.4605pp[-0.1518,+0.8828] and OwnPeer50.5130
by0.1657pp[-0.4092,+0.5432]. But same-formula deployable class excess50.6469 is
nearly tied: primary+0.0318pp[-0.3575,+0.1051]. Old pooled50.5108 is0.1679pp
lower[-0.4077,+0.5487]. Primary wins5/8 main domain points versus class excess,
6/8 versus pooled, with worst protocol loss2.2787pp versus pooled on LoveDA P.
These2000 paired filename-source-group intervals are conditional development
evidence, not full-suite, independent or method-selection-adjusted validation.

All class norm matches are possible with error below1.90e-16. Against matched
pooled, class excess, own-source class mean and mean3 word-identity nulls,
primary deltas are-0.1638,+0.0171,+0.0218,+0.0891pp; every interval includes
zero. Matched rival collapse+0.0636pp also crosses zero. Matched word-write
permutation+0.2434pp[+0.0292,+0.5374] is supported, while direct writing is
0.0165pp higher with an interval spanning zero. Raw class controls do not
borrow primary risk/dose; matched controls are diagnostic, inheriting its
word-only written norm. Word allocation and source advancement fail.

About62.54-75.24% of acted word/rival entries are individually wide AND fine
positive, confirming expanded support, not semantic correctness. Versus old
soft, VDD vehicle26.8570->26.0141 adds40022 FP, Potsdam car33.1180->32.2937
adds13482 FP and Vaihingen car27.2431->25.5025 adds36510 FP. VDD water92.5582
->89.4376 loses763649 TP; water-to-other errors increase762653. LoveDA P
barren68.0139->48.3424 adds82823 FP, including72040 more farm-to-barren errors.
UDD5 road44.7628->46.6666 gains2375027 TP but2092701 FP. Mean gain is not
balanced small-object repair; aggregate routes do not identify causal words.

Reject advancement under the frozen protocol. No timing/stress/full rollout,
retrospective tuning or favorable-control promotion follows; earlier latency
does not characterize this candidate. The three-part objective remains open.
Report/raw: COMPETITIVE_EXCESS_ALIAS_20261006.md and
competitive_excess_alias_20261006/.

## Geometry-Supported Word Response Collisions: Capacity, No Utility

The frozen `ResponseCollision_Soft` completes96 developed complete inputs
with unchanged20 vocabularies, bounded896 patch-only2 Geometry, independent448
VIP wide and original W/H. No fine/RGB/head/text extras: actual caps4/4/0.
Source weights use positive squared cross-WORD response correlation in32
original-G neighbours, weighted by rival-word softmax of donor raw-logit
means. Current query/padding are excluded from the source. Canonical/unknown
entries are protected; original slots/salience stay fixed without survivor
redistribution. This differs from canonical-margin covariance and static
rival text-span projection, not a new correlation/ridge theorem.

Eleven new source tests,17 prior regressions and four offline statistics tests
pass. Both mask-free real-checkpoint smokes and primary singleton checks pass.
Original Geometry/no-admission/patch-only2 per-image confusions replay exactly.
Ordered unique coverage, identities, exact20 counts, paired target counts and
independent aggregate mIoU verify. All eight workers exit after completion;
retained full20092 is unchanged.

Primary46.6565 versus same-information bounded46.6450 gains0.0115pp
[-0.1388,+0.1433], with2/8 domain wins and worst protocol loss0.4378pp on
Vaihingen. Raw pooled pattern46.7542 and class mean46.7715 are higher. Matched
pooled and class mean beat primary by0.0475pp[+0.0108,+0.0924] and0.1020pp
[+0.0425,+0.1784]. Mean3 matched identity nulls are0.0871pp higher with an
interval spanning zero. All norm matches are possible, error below2.54e-16.
Raw pooled pattern reads no primary word risk/dose; matched nulls inherit its
word-only post-H norm and are diagnostic, not deployable substitutes.

Same-budget global support is0.0287pp higher; rival-response position null is
only0.0069pp lower. Both differences cross zero. Matched word-write permutation
is0.0413pp higher[+0.0077,+0.0777]; matched direct writing is0.0130pp higher
with an interval spanning zero. Do not promote these controls or transfer the
retained full coupling's writer evidence to this poorly judged semantic source.
The2000 paired filename-source-group intervals are conditional development
evidence, not full/independent or method-selection-adjusted confirmation.

Known noncanonical fraction95%, active soft weights90.49-93.96%, mean risks
0.2344-0.4261 and mean absolute antisymmetric potential0.00465-0.00719 in
existing units. These are image/tile-averaged diagnostics, not semantic errors.
Constructed tests establish word information beyond identical class means,
but not useful real-data allocation. Symmetric co-activation cannot determine
semantic ownership; valid context can collide. Pre-projection cancellation
was not measured and cannot be reconstructed from these averages alone.

Versus baseline, Potsdam car31.4528->31.2128 adds4645 FP, UDD5 vehicle16.9950
->16.6584 adds452548 FP and Vaihingen car26.4838->25.4418 adds19966 FP. UDD5
road37.2416->37.1445 loses67447 TP and adds9239 FP. VDD vehicle gains0.0473pp,
but water90.6042->90.5498 loses51842 TP. This is not balanced small-object
repair. Source advancement fails: no timing/stress/full rollout, tuning or
favorable-control promotion. An encoding cap is not latency evidence.

Preserve the retained model and keep the three-part objective active. The
remaining gap is signed competitive discrimination, not merely increased
collision coverage or attenuation. Report/raw: RESPONSE_COLLISION_ALIAS_20261006.md
and response_collision_alias_20261006/.

## Vocabulary-Time NLI Membership: Independent Source, No Useful Allocation

The frozen SemanticMembership_Soft implements the previously unexecuted public
semantic-membership source: cross-encoder/nli-MiniLM2-L6-H768, pinned revision
b95119ce93d3e065de6214e38cd4a97b0f2f2c6d, unchanged premise/hypothesis template
`The object or surface is {phrase}.` SNLI/MultiNLI-supervised pretraining is
external semantic knowledge, not target-data fitting, and must be attributed.
No definitions, word exceptions, target images/masks or labeled alias rankings
enter its6774-pair cache. Setup12.1916s includes10.5834s loading and1.6083s NLI
forward; no per-image NLI or extra visual/head observations. This source is
distinct from frozen DINO.text cosine/lexical binding/response collision, not
an architectural innovation merely because another pretrained model is added.

At baseline top2 competitors, own/rival entailments o/r define unknown
u=(1-o)(1-r) and weight=(o+u)/(o+r+u), numerical floor1e-6. Canonical/self/padding
are protected. Original20 slots/salience, bounded896/448 sources and original
antisymmetric H writing stay. Semantic membership cannot tell whether a
correctly named car has fired on the wrong road pixel.

Nine new source tests,21 prior source regressions, four paired-statistics tests,
both real-checkpoint mask-free smokes and singleton parity pass. All96 developed
complete inputs, exact original three per-image endpoints, source identities,
exact20 counts, paired targets and independent aggregate mIoU verify. Actual
caps4/4/0 plus0 per-image NLI hold; all workers exit complete. The retained
full20092-image reference and paused automations remain unchanged.

Primary46.7673 versus baseline46.6450 gains0.1223pp[-0.3260,+0.4421], with
only3/8 main-domain wins and Potsdam worst loss1.1440pp. Raw pooled NLI46.8831
is higher by0.1158pp[+0.0030,+0.2034]. Primary-minus-matched pooled, class mean
and mean3 identity nulls are-0.0976,-0.0793,-0.0990pp; all corresponding paired
conditional intervals are below zero. Norm matches are feasible. Rival
conditioning and writing/direct comparisons have intervals spanning zero.
NLI source capacity is not useful lexical allocation on this developed panel.

Potsdam car gains0.4929pp through8303 fewer FP and54 fewer TP, but impervious
surface loses2.4167pp and clutter3.0319pp. VDD vegetation loses853758 TP; UDD5
road loses551274 TP and0.5031pp. These aggregate routes do not identify causal
words. FLAIR-1/OEM gains cannot justify per-domain source switching.

Source advancement fails. No singleton timing, vocabulary stress, full rollout,
prompt/formula/strength tuning or favorable-control promotion follows. New
protocol's prospective1.5x/1000ms budget reflects moderate-cost tolerance, not
CVPR acceptance rules; it does not change any historical failed gate. Preserve
this frozen rejection rather than replacing semantic membership with truth.
Report/raw: SEMANTIC_MEMBERSHIP_ALIAS_20261006.md and
semantic_membership_alias_20261006/. The three-part goal remains active.

## CVPR Claim Boundary

Useful supported framing: visual relationships organize local frozen-head
reading and constrain where a complementary semantic observation can revise it.
The full same-source evidence distinguishes the current correction from simple
equal-logit fusion on these developed domains.

The VIP wide observer is inherited and must be attributed. The ridge solver,
twofold read strength, DINO-relation replacement and spatial locality have
precedents. Matched SCLIP coupling is only0.3075pp lower in mean (six domain
wins); it is a DINO.text adaptation, not an official-system reproduction, and
its prefix handling differs. Neither the ridge algebra nor renaming VIP proves
an independent innovation. An effective, attributable conditional alias module
and/or stronger independent Geometry-coupling evidence remains necessary for
the desired three-part contribution.

Eight-domain means are not all-class superiority. VDD vehicle and Potsdam low
vegetation remain weak relative to particular VIP protocols. VIP results are
repaired local adaptations, not published benchmark numbers. LandCover.ai
substitutes for unlabeled iSAID. These domains informed development; an unchanged
final protocol on genuinely new regions or natural-image data remains open.

## Reproducible Artifacts

- Candidate accuracy and timing: `BOUNDED_PATCH_ONLY_FULL_VS_VIP_20261005.md`.
- Full controls: `BOUNDED_COUPLING_SAME_FIELD_FULL_20261005.md`.
- Per-domain paired evidence: `bounded_coupling_controls_full_20261005/PAIRED_UNCERTAINTY.md`.
- Statistics script: `../tools/report_bounded_coupling_controls.py`.
- Collector: `../tools/bounded_coupling_controls_full_experiment.py collect`.
- Four offline statistics/coverage tests: `../tools/test_report_bounded_coupling_controls.py`.
- Rejected crop-head pilot, timing and per-class coverage: `BOUNDED_CROP_HEAD_ALIAS_20261005.md`.
- Crop-head results and frozen gate: `bounded_crop_head_alias_screen_20261005/summary.json`.
- Rejected global detail2 pilot and timing: `BOUNDED_DETAIL_PAIR_ALIAS_20261005.md`.
- Global detail2 paired evidence: `bounded_detail_pair_alias_screen_20261005/summary.json`.
- Rejected canonical-reference repair: `BOUNDED_CANONICAL_PAIR_ALIAS_20261005.md`.
- Canonical-reference paired evidence: `bounded_canonical_pair_alias_screen_20261005/summary.json`.
- Rejected native detail1 projected alias pilot: `BOUNDED_DETAIL_PROJECTED_ALIAS_20261006.md`.
- Detail1 paired evidence and timings: `bounded_detail_projected_alias_screen_20261006/summary.json`.
- Rejected covariance source: `BOUNDED_COVARIANCE_ALIAS_20261006.md`.
- Covariance paired evidence and timings: `bounded_covariance_alias_screen_r2_20261006/summary.json`.
- Rejected lexical binding source: `BOUNDED_BINDING_ALIAS_20261006.md`.
- Binding paired evidence, extra text setup and timings: `bounded_binding_alias_screen_20261006/summary.json`.
- Fixed-source complete-image path audit: `BOUNDED_ALIAS_PATH_AUDIT_20261006.md`.
- Path-audit marginals and labeled-control evidence: `bounded_alias_path_audit_r2_20261006/`.
- Rejected independent witness, timing and source verification: `BOUNDED_SIGLIP_ALIAS_20261006.md`.
- Independent-witness paired artifacts: `bounded_siglip_alias_screen_20261006/`.
- Magnitude-matched same-action writing evidence: `BOUNDED_ALIAS_WRITE_MATCH_AUDIT_20261006.md`.
- Complete paired writing counterfactuals: `bounded_alias_write_match_audit_20261006/`.
- Pool-isolated source factorial, coverage and actual singleton cost: `POOL_ISOLATED_ALIAS_20261006.md`.
- Verified pool-isolated confusions, timing and failed attribution: `pool_isolated_alias_20261006/summary.json`.
- Frozen evidence-local writing, real cost and class coverage: `EVIDENCE_LOCAL_ALIAS_20261006.md`.
- Evidence-local per-image confusions, timing and failed factor checks: `evidence_local_alias_20261006/summary.json`.
- Conditional word/class/writing intervals and support diagnostics: `evidence_local_alias_20261006/PAIRED_UNCERTAINTY.md`.
- Offline paired reporter and four tests: `../tools/report_evidence_local_alias.py` and `../tools/test_report_evidence_local_alias.py`.
- Mask-free word-action decomposition: `ALIAS_WORD_PATH_AUDIT_20261006.md` and `alias_word_path_audit_20261006/`.
- Frozen likelihood-cap source, exact controls and per-class coverage: `WITNESS_CAP_ALIAS_20261006.md`.
- Word/class/magnitude/write paired intervals: `witness_cap_alias_20261006/PAIRED_UNCERTAINTY.md`.
- Paired-statistics wrapper using the tested existing engine: `../tools/report_witness_cap_alias.py`.
- Frozen actual-footprint witness, coverage and failed source attribution: `FOOTPRINT_WITNESS_ALIAS_20261006.md`.
- Actual-footprint paired statistics and verified raw controls: `footprint_witness_alias_20261006/`.
- Actual-footprint paired reporter reusing the tested engine: `../tools/report_footprint_witness_alias.py`.
- Rival-word omission source, exact controls and failed reference attribution: `RIVAL_OMISSION_ALIAS_20261006.md`.
- Rival-word omission per-image evidence and paired intervals: `rival_omission_alias_20261006/`.
- Rival-word omission paired reporter using the same tested engine: `../tools/report_rival_omission_alias.py`.
- Frozen same-bank local lexical path, failed identity/writer attribution: `LOCAL_ALIAS_PATH_20261006.md`.
- Verified local-path per-image controls and paired intervals: `local_alias_path_20261006/`.
- Local-path reproducible manager/statistics: `../tools/local_alias_path_experiment.py` and `../tools/report_local_alias_path.py`.
- Semantic-rival reference and source-word versus rival-key attribution: `SEMANTIC_RIVAL_ALIAS_20261006.md`.
- Verified semantic-reference per-image controls and paired statistics: `semantic_rival_alias_20261006/`.
- Semantic-reference manager/statistics: `../tools/semantic_rival_alias_experiment.py` and `../tools/report_semantic_rival_alias.py`.

No model rule, checkpoint, vocabulary, completed evaluation or paused automation
was changed to obtain this evidence. The final-model research goal remains active.
