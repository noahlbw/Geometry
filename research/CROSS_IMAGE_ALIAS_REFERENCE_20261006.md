# Held-Out RGB Alias Reference: Frozen Eight-Domain Pilot

Same96 developed complete inputs, fixed20 aliases and original G/H.64 disjoint RGB references/domain; UDD5 train RGB, other domains transductive evaluation RGB. Canonical pseudo-categories are fallible. No labels loaded for reference construction. No query encoder added. Prior developed labels motivated the hypothesis; not untouched independent validation.

| Domain/protocol | CrossRef_Soft | CrossRef_ObservationMean | CrossRef_PreviousSoft | CrossRef_PreviousHard | CrossRef_MatchedPrevious | CrossRef_MatchedClassMean | CrossRef_MatchedAliasShuffle0 | CrossRef_MatchedAliasShuffle1 | CrossRef_MatchedAliasShuffle2 | CrossRef_MatchedReferenceShuffle | CrossRef_MatchedInImage | CrossRef_MatchedRivalCollapsed | CrossRef_MatchedShuffledWrite | CrossRef_DirectMatched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 50.3016 | 49.5849 | 49.8732 | 49.7970 | 50.0953 | 50.2725 | 50.2767 | 50.2707 | 50.2716 | 50.0721 | 50.2101 | 50.3317 | 50.0092 | 50.2350 |
| potsdam/potsdam | 55.2941 | 54.8216 | 55.1778 | 54.9119 | 55.2619 | 55.2700 | 55.2732 | 55.2820 | 55.3318 | 55.3028 | 55.2785 | 55.1977 | 55.0843 | 55.3447 |
| udd5/udd5 | 51.3990 | 50.7777 | 51.2305 | 51.2136 | 51.6092 | 51.3320 | 51.1512 | 51.3499 | 51.2911 | 51.6063 | 51.7698 | 51.1045 | 50.8184 | 51.4523 |
| oem/oem | 30.1649 | 29.5309 | 29.9714 | 29.8628 | 30.4265 | 30.1436 | 30.0886 | 30.1263 | 30.1999 | 30.4550 | 30.3509 | 29.8623 | 30.0717 | 30.2496 |
| loveda/P | 75.5136 | 75.2590 | 75.5356 | 75.2137 | 75.6188 | 75.4330 | 75.6274 | 75.5513 | 75.3547 | 75.6283 | 75.5162 | 75.2128 | 75.6872 | 75.4616 |
| loveda/D | 46.7309 | 45.5946 | 46.6718 | 47.0139 | 47.0257 | 46.7360 | 46.7013 | 46.6482 | 46.6458 | 47.0751 | 46.4860 | 46.5860 | 46.8579 | 46.5736 |
| vaihingen/vaihingen | 52.0065 | 51.5130 | 51.8593 | 51.9034 | 51.9517 | 51.8677 | 51.7843 | 51.9138 | 51.9058 | 52.0301 | 51.6970 | 51.9155 | 52.0029 | 51.9022 |
| landcoverai/landcoverai | 78.0166 | 77.9179 | 78.0381 | 77.9643 | 78.0881 | 77.9857 | 77.9754 | 77.9888 | 77.9681 | 78.0907 | 78.0327 | 77.9612 | 78.0728 | 78.0702 |
| flair1/flair1 | 38.8943 | 38.7829 | 38.9230 | 38.6442 | 39.0320 | 39.0378 | 39.0288 | 39.0106 | 39.0475 | 39.0358 | 39.0997 | 38.7739 | 38.7724 | 38.8851 |

## Domain Mean

LoveDA D once; P separately.

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| CrossRef_Soft | 50.3510 |
| CrossRef_ObservationMean | 49.8154 |
| CrossRef_PreviousSoft | 50.2181 |
| CrossRef_PreviousHard | 50.1639 |
| CrossRef_MatchedPrevious | 50.4363 |
| CrossRef_MatchedClassMean | 50.3307 |
| CrossRef_MatchedAliasShuffle0 | 50.2849 |
| CrossRef_MatchedAliasShuffle1 | 50.3238 |
| CrossRef_MatchedAliasShuffle2 | 50.3327 |
| CrossRef_MatchedReferenceShuffle | 50.4585 |
| CrossRef_MatchedInImage | 50.3656 |
| CrossRef_MatchedRivalCollapsed | 50.2166 |
| CrossRef_MatchedShuffledWrite | 50.2112 |
| CrossRef_DirectMatched | 50.3391 |

## Reference Cost

| Domain | RGB images | Fine encodings | Calibration seconds | Rows by protocol |
| --- | ---: | ---: | ---: | --- |
| vdd | 64 | 1024 | 37.76 | {'vdd': 16384} |
| potsdam | 64 | 1024 | 23.13 | {'potsdam': 16384} |
| udd5 | 64 | 872 | 32.58 | {'udd5': 13744} |
| oem | 64 | 1024 | 21.64 | {'oem': 16384} |
| loveda | 64 | 1024 | 23.91 | {'P': 16384, 'D': 16384} |
| vaihingen | 64 | 1024 | 20.96 | {'vaihingen': 16384} |
| landcoverai | 64 | 1024 | 20.73 | {'landcoverai': 16384} |
| flair1 | 64 | 1024 | 21.23 | {'flair1': 16384} |

Calibration workers run concurrently; these are real elapsed source-construction costs, not isolated throughput. Graph setup is recorded separately. Index construction and primary inference cost are distinct; no historical latency is assigned to this candidate.

## Coverage And Competitive Activation

| Domain/protocol | Class | Previous soft IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 21.0681 | 22.2868 | +359553 | +67049 |
| vdd/vdd | wall | 55.6308 | 56.2306 | +63659 | +64431 |
| vdd/vdd | road | 16.5334 | 16.8620 | -1052 | -183938 |
| vdd/vdd | vegetation | 48.9341 | 49.1512 | +55623 | -11549 |
| vdd/vdd | vehicle | 26.8570 | 26.3160 | +747 | +23126 |
| vdd/vdd | roof | 87.5307 | 88.4782 | -7312 | -336447 |
| vdd/vdd | water | 92.5582 | 92.7863 | -21066 | -72824 |
| potsdam/potsdam | impervious surface | 77.3746 | 77.1986 | +270 | +8614 |
| potsdam/potsdam | building | 83.9116 | 84.0440 | +64 | -1959 |
| potsdam/potsdam | low vegetation | 65.3738 | 65.6330 | +8281 | +3461 |
| potsdam/potsdam | tree | 62.6535 | 62.6664 | -838 | -1633 |
| potsdam/potsdam | car | 33.1180 | 33.0721 | +7 | +721 |
| potsdam/potsdam | clutter | 8.6353 | 9.1505 | +108 | -17096 |
| udd5/udd5 | vegetation | 73.1690 | 73.5209 | +541621 | +94421 |
| udd5/udd5 | building | 83.7786 | 83.7673 | -62314 | -47124 |
| udd5/udd5 | road | 44.7628 | 46.0793 | +1063139 | +208551 |
| udd5/udd5 | vehicle | 19.4049 | 18.7127 | +22483 | +688165 |
| udd5/udd5 | other | 35.0373 | 34.9148 | -749698 | -1759244 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +2459 |
| oem/oem | rangeland | 6.8975 | 7.2494 | +4967 | +1913 |
| oem/oem | developed space | 22.7224 | 22.8443 | -806 | -21386 |
| oem/oem | road | 29.6258 | 29.6339 | -2967 | -10205 |
| oem/oem | tree | 27.0871 | 27.4940 | +5717 | +3492 |
| oem/oem | water | 17.0185 | 17.0485 | +180 | +818 |
| oem/oem | agriculture land | 72.9856 | 73.5037 | -178 | -7052 |
| oem/oem | building | 63.4347 | 63.5450 | +11270 | +11778 |
| loveda/P | building | 89.6315 | 89.4881 | +125 | +716 |
| loveda/P | road | 85.1967 | 85.2031 | -167 | -265 |
| loveda/P | water | 72.6192 | 73.3328 | +10674 | +741 |
| loveda/P | barren | 68.0139 | 66.4664 | +84 | +5202 |
| loveda/P | tree | 64.6523 | 65.1200 | +5774 | +834 |
| loveda/P | farm | 73.1002 | 73.4711 | -4818 | -18900 |
| loveda/D | background | 14.6919 | 14.2194 | -13823 | -3619 |
| loveda/D | building | 48.4265 | 48.3532 | +111 | +1234 |
| loveda/D | road | 65.9654 | 66.1137 | -847 | -3929 |
| loveda/D | water | 67.7771 | 68.2825 | +8966 | +1984 |
| loveda/D | barren | 26.2285 | 25.8791 | +1421 | +12995 |
| loveda/D | tree | 52.2329 | 52.6596 | +10596 | +9397 |
| loveda/D | farm | 51.3805 | 51.6089 | -3171 | -21315 |
| vaihingen/vaihingen | impervious surface | 63.9288 | 63.8424 | +1632 | +6613 |
| vaihingen/vaihingen | building | 70.2291 | 70.5371 | -728 | -13913 |
| vaihingen/vaihingen | low vegetation | 29.7808 | 30.3369 | +8945 | +1683 |
| vaihingen/vaihingen | tree | 68.1148 | 68.2341 | -1803 | -6296 |
| vaihingen/vaihingen | car | 27.2431 | 27.0818 | +242 | +3625 |
| landcoverai/landcoverai | background | 85.1704 | 85.1901 | +474 | +383 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7243 | 92.7709 | -403 | -967 |
| landcoverai/landcoverai | water | 93.8996 | 93.7974 | +29 | +418 |
| landcoverai/landcoverai | road | 40.3582 | 40.3080 | +1 | +65 |
| flair1/flair1 | building | 59.6299 | 60.0082 | +26 | -1018 |
| flair1/flair1 | pervious surface | 28.8614 | 28.7390 | -32 | +142 |
| flair1/flair1 | impervious surface | 69.6020 | 69.5465 | +472 | +1012 |
| flair1/flair1 | bare soil | 11.4686 | 11.1694 | +0 | +3942 |
| flair1/flair1 | water | 95.6854 | 95.7691 | +47 | +0 |
| flair1/flair1 | coniferous | 2.8284 | 2.7732 | -2 | +72 |
| flair1/flair1 | deciduous | 66.8983 | 66.8484 | -295 | -215 |
| flair1/flair1 | brushwood | 8.3984 | 8.4318 | +154 | +1215 |
| flair1/flair1 | vineyard | 56.9048 | 57.5202 | -47 | -2246 |
| flair1/flair1 | herbaceous vegetation | 35.3889 | 34.6743 | -4705 | -86 |
| flair1/flair1 | agricultural land | 31.4099 | 31.2518 | -330 | +1586 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | +308 |

## Decision

The paired reporter completed against the frozen protocol. Accuracy improves,
but lexical/class/source attribution fails. No post-result tuning, control
promotion, timing, vocabulary stress or full rollout. Retained full model unchanged.

## Interpretation And Verification

All96 ordered unique complete inputs finish, including full UDD540. Original
Geometry/NoAdmission/PatchOnly2 and previous positive/soft/hard per-image
confusions replay exactly. All-arm scored targets, original20 vocabulary and
checkpoint identities, NPZ aggregate sums and independently recomputed mIoU
verify. Canonical/self/invalid protection, nonpositive directed attenuation,
normal-equation/gauge checks and matched-field errors verify. No norm-control
field is unmatchable. Every worker exits after complete output.

All eight frozen references contain64 unique RGB files disjoint from the
developed pilot and timing inputs:512 source images total. UDD5 uses train/src,
the other domains remaining evaluation RGB, explicitly transductive. No
reference masks are loaded. The first UDD5 calibration stopped before any RGB
because data-root already included the extracted/UDD/UDD5 suffix; correcting
that duplicated path changes no source rule. Its s0.path_failure.log is
preserved.16 new source/math tests,26 original regressions and4 existing offline
paired-engine tests pass; both VDD/Potsdam real-checkpoint mask-free smokes
verify old endpoints, frozen weights and singleton equality.

Primary50.3510 exceeds old soft50.2181 by0.1328pp on6/8 main domains,
conditional95% paired interval[+0.0681,+0.1905]. Against identical visual
information without attenuation49.8154, gain0.5355pp has interval
[+0.3432,+0.6855]. These are96 developed complete inputs, NOT full20092 or
independent/selection-adjusted evidence. Separate LoveDA P loses0.0220pp;
LandCover.ai loses0.0215pp and FLAIR-10.0287pp versus old soft.

| Matched comparator | Primary gain pp | Conditional95% interval pp |
| --- | ---: | --- |
| Previous soft at primary written norm | -0.0853 | [-0.1627,-0.0125] |
| Class-mean risk | +0.0203 | [-0.0100,+0.0531] |
| Mean word-identity null | +0.0372 | [+0.0056,+0.0719] |
| Reference-position shuffle | -0.1075 | [-0.1840,-0.0301] |
| In-image leave-query-out reference | -0.0146 | [-0.0981,+0.1057] |
| Rival-collapsed risk | +0.1344 | [+0.0868,+0.1957] |
| Intervention-only H correspondence shuffle | +0.1398 | [+0.0544,+0.2470] |
| Direct writing | +0.0119 | [-0.0308,+0.0430] |

Matched nulls inherit the primary WORD-DERIVED total valid class-field norm per
tile; they are not deployable word-independent replacements. There is a small
word-identity signal, below the frozen0.05pp practical requirement, while
class-only attribution is uncertain. Most importantly, shuffling the reference
pseudo-categories away from their responses is HIGHER at matched norm. This
source therefore does not establish beneficial cross-image category alignment.
The old rule at matched strength is also stronger: a positive source point
gain alone cannot be assigned to better lexical judgment. This does not prove
all corpus references or all alias information useless. Canonical-only
pseudo-categories remain fallible, including those constructed across images.

Competitor identity and existing spatial word-write correspondence retain
conditional evidence. However, direct matched writing is nearly equal and
cross-image reference is not better than the in-image source. No new density,
solver or universal fidelity theorem follows.

UDD5 road IoU44.7628->46.0793 gains1063139 TP with208551 FP, while vehicle
IoU19.4049->18.7127 adds688165 FP despite22483 more TP. VDD vehicle
IoU26.8570->26.3160 adds23126 FP with747 TP; Potsdam car33.1180->33.0721
adds721 FP with7 TP. These mixed pathways do not establish balanced small-target
repair. They do not uniquely isolate reference contamination, risk support or
the class-potential projection as the cause.

Calibration costs20.73-37.76 seconds/domain in concurrent source workers,
872-1024 real fine encodings/domain, plus separately recorded graph/setup and
index-construction costs. Query execution adds no visual/head encoding, but
does add reference lookup and memory. Primary singleton latency is NOT measured;
do not borrow old timing or declare the source fast. Frozen accuracy/mechanism
checks fail independently of cost, so no timing/stress/full rollout follows.

The full Geometry_PatchOnly2Coupled reference remains unchanged. The requested
Geometry/conditional-word/coupled-write goal remains active. Verified raw
references, source manifests, per-image confusions and paired statistics are in
cross_image_alias_reference_20261006/. New source:
../DINOtool/dinotool/cross_image_alias_reference.py; manager:
../tools/cross_image_alias_reference_experiment.py; paired reporter:
../tools/report_cross_image_alias_reference.py.
