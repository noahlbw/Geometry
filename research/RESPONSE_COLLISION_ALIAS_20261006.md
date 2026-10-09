# Response Collision: Frozen Bounded Complete-Image Candidate

Same96 developed inputs, fixed20 words and original Geometry/W/H. No fine or extra RGB/head/text observations.

## Decision

Reject advancement of `ResponseCollision_Soft`. The frozen source has a word
dependence in construction, but does not identify useful lexical allocation
on these inputs. Primary mean46.6565 versus same-information bounded46.6450
gains only0.0115pp[-0.1388,+0.1433], with2/8 main-domain wins. Class-mean and
same-formula pooled-pattern controls are higher. No tuning, timing, vocabulary
stress, full rollout or favorable-control promotion follows the failed gate.
The retained full20092-image performance model remains unchanged.

These are UDD5 full40 plus8 complete images/other domain, not full20092 or
independent validation. LoveDA D counts once and P stays separate. Intervals
use2000 paired filename-source-group draws, conditional on developed inputs,
not independent acquisition units or method-selection-adjusted uncertainty.

| Domain/protocol | Bounded baseline | Word collision | Pooled pattern | Class mean | Delta vs baseline pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.8468 | 47.1233 | 47.3859 | 47.4156 | +0.2765 |
| potsdam/potsdam | 51.5168 | 51.4031 | 51.4414 | 51.6121 | -0.1137 |
| udd5/udd5 | 47.0786 | 46.9602 | 47.0448 | 47.1072 | -0.1184 |
| oem/oem | 25.2530 | 25.0922 | 24.9289 | 24.9610 | -0.1608 |
| loveda/P | 67.2104 | 67.2200 | 67.4806 | 66.0530 | +0.0096 |
| loveda/D | 39.4705 | 39.3314 | 39.2494 | 39.3576 | -0.1391 |
| vaihingen/vaihingen | 49.1468 | 48.7090 | 48.5519 | 49.4004 | -0.4378 |
| landcoverai/landcoverai | 76.2441 | 76.2257 | 76.2630 | 76.3081 | -0.0184 |
| flair1/flair1 | 37.6036 | 38.4072 | 39.1683 | 38.0103 | +0.8036 |

## Domain Mean

LoveDA D once; P separate. Developed inputs, not full20092.

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| ResponseCollision_Soft | 46.6565 |
| ResponseCollision_PooledPattern | 46.7542 |
| ResponseCollision_MatchedPooledPattern | 46.7040 |
| ResponseCollision_ClassMean | 46.7715 |
| ResponseCollision_MatchedClassMean | 46.7585 |
| ResponseCollision_AliasShuffle0 | 46.4926 |
| ResponseCollision_AliasShuffle1 | 46.9273 |
| ResponseCollision_AliasShuffle2 | 46.9806 |
| ResponseCollision_MatchedAliasShuffle0 | 46.6199 |
| ResponseCollision_MatchedAliasShuffle1 | 46.7943 |
| ResponseCollision_MatchedAliasShuffle2 | 46.8165 |
| ResponseCollision_GlobalSupport | 46.6852 |
| ResponseCollision_PositionNull | 46.6496 |
| ResponseCollision_MatchedShuffledWrite | 46.6978 |
| ResponseCollision_DirectMatched | 46.6695 |

## Timing

No candidate singleton timings collected. Multi-arm worker duration is not inference latency.
Actual inference uses at most4 Geometry/4 wide encodings and0 fine encodings.
This proves the observation cap, not a measured latency improvement. The
failed source gate stops serial timing; no earlier timing is assigned here.

## Source Advancement

Paired source checks are complete: see
response_collision_alias_20261006/PAIRED_UNCERTAINTY.md and CLAIM_CHECKS.md.
Accuracy/source advancement fails; research thresholds are not CVPR acceptance
criteria. Do not promote a higher control or revise the source after this audit.

## Verified Implementation And Source

Eleven source tests,17 prior covariance/canonical regressions and four offline
paired-statistics tests pass. Both real-checkpoint mask-free smokes replay
original Geometry/no-admission/patch-only2 predictions and verify primary
singleton equality. All96 unique ordered images, source vocabularies,
checkpoint identities, exact20 counts/class, paired target counts, per-image
aggregate confusions and independent aggregate mIoU verify. All eight workers
exit after complete output; no unrelated jobs or retained models are changed.

The source compares original wide WORD response curves across the current
baseline top2 competitors in32 original-G neighbours. Self/padding are
excluded. Weighted Pearson rho and rival-word softmax of donor mean raw logits
define risk=sum_b q_r[b]*max(rho[a,b],0)^2. Outside-exponent weight=1-risk,
floor1e-6, canonical protected, unknown neutral, original20 slots and salience
kept. Original H writes the antisymmetric fixed-slot pair potential. No new
RGB, semantic-head, text, fine observation or fitted parameter is introduced.

This differs from canonical-margin covariance and static rival text-span
projection. Constructed tests show word patterns can differ when class means
agree, and the same word can collide with one rival but not another. These
facts establish representational capacity, not real-world semantic judgment,
new correlation/ridge mathematics or a publication novelty claim.

## Conditional Attribution

| Comparator | Primary-minus-comparator pp | Conditional95% interval pp |
| --- | ---: | --- |
| Same-information bounded baseline |+0.0115|[-0.1388,+0.1433]|
| Raw pooled pattern |-0.0977|[-0.2762,+0.1015]|
| Matched pooled pattern |-0.0475|[-0.0924,-0.0108]|
| Raw class mean |-0.1150|[-0.2938,+0.0178]|
| Matched class mean |-0.1020|[-0.1784,-0.0425]|
| Mean3 matched word-identity nulls |-0.0871|[-0.1951,+0.0039]|
| Same-budget global support |-0.0287|[-0.2111,+0.1386]|
| Rival-response position null |+0.0069|[-0.1515,+0.1463]|
| Matched word-write permutation |-0.0413|[-0.0777,-0.0077]|
| Matched direct writing |-0.0130|[-0.0458,+0.0165]|

Raw pooled-pattern calibration is independently deployable and does not read
primary word risks/dose. Raw class-mean and identity controls inherit primary
weights; matched controls additionally inherit primary word-only post-H norm.
They are diagnostic controls, not deployment candidates. All matches are
possible, relative errors below2.54e-16. Worst protocol loss versus baseline
is0.4378pp on Vaihingen. Small degradation does not establish a useful module.

Known noncanonical entries average95%; active weight fractions90.49-93.96%
and average collision risks0.2344-0.4261. Antisymmetric potential mean absolute
values are0.00465-0.00719 in existing score units. These are evaluator image/
tile-averaged diagnostics, not percentages of harmful words or corrected
pixels. The source acts broadly, not just on rare ambiguous aliases.

Colliding patterns can represent valid co-occurrence/context. Correlation is
symmetric and does not specify the correct semantic owner. A lack of raw
class-mean equivalence in a fixture does not overcome the real-data result:
averaging word risks produces a better matched correction direction here.
Changing only attenuation amount cannot explain away that attribution failure.
The raw pre-projection suppression/cancellation ratio was not measured, so do
not infer it from mean risk and final antisymmetric potential alone.

Correct H word-write correspondence is worse than its permutation for THIS
source. This concerns a poorly judged semantic intervention, not the retained
full same-information coupling result or every Geometry writer. Do not promote
the shuffled control or transfer earlier candidates' writer attribution.

## Coverage And Competition

The audit below compares with the exact same-source bounded baseline, not
with prior fine-source candidates or full-domain VIP. It is not used to tune.

| Domain/class | Baseline IoU | Collision IoU | Delta TP | Delta FP |
| --- | ---: | ---: | ---: | ---: |
| VDD vehicle |23.8613|23.9086|+386|-460|
| VDD water |90.6042|90.5498|-51842|-44786|
| Potsdam car |31.4528|31.2128|+185|+4645|
| Potsdam low vegetation |60.1752|60.2676|+2722|+1012|
| UDD5 road |37.2416|37.1445|-67447|+9239|
| UDD5 vehicle |16.9950|16.6584|+20737|+452548|
| Vaihingen car |26.4838|25.4418|+366|+19966|
| LoveDA P barren |70.3607|70.7555|-404|-1755|
| FLAIR-1 bare soil |21.3471|20.7646|-1|+2203|

Potsdam/UDD5/Vaihingen vehicle classes add false positives despite added TP;
the source does not repair balanced small-object fidelity. VDD water loses
correct coverage while false activation falls. These aggregate counts do not
identify a causal alias or certify word correctness.

The next information gap is signed competitive discriminability beyond
co-activation/redundancy and pooled class calibration, with source-specific
word and write attribution. Neither more response collisions nor a stronger
penalty is justified by this result. Preserve this frozen failure and the
retained full model; the complete three-part objective remains active.

Collector: ../tools/response_collision_alias_experiment.py. Paired reporter:
../tools/report_response_collision_alias.py. Raw/results/statistics:
response_collision_alias_20261006/.
