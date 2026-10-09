# Existing One-Sided Alias: Matched Attribution

Exact previously developed one-sided soft rule; not a new model. Same96 developed COMPLETE inputs, unchanged20 words and original sources. No threshold/source/Geometry changes. Matching uses valid whole-tile class-field norms, not pixelwise margins.

| Domain/protocol | OneSide_Soft | OneSide_ObservationMean | OneSide_Hard | OneSide_ClassMean | OneSide_MatchedClassMean | OneSide_AliasShuffle0 | OneSide_AliasShuffle1 | OneSide_AliasShuffle2 | OneSide_MatchedAliasShuffle0 | OneSide_MatchedAliasShuffle1 | OneSide_MatchedAliasShuffle2 | OneSide_RivalCollapsed | OneSide_MatchedRivalCollapsed | OneSide_ShuffledWrite | OneSide_MatchedShuffledWrite | OneSide_DirectMatched | OneSide_EqualLogitMean | OneSide_EqualLogitMeanNoAlias | OneSide_AllCorrespondenceShuffle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 49.8732 | 49.5849 | 49.7970 | 49.8083 | 49.8375 | 49.8130 | 49.7838 | 49.7934 | 49.8368 | 49.8120 | 49.8117 | 49.7108 | 49.7893 | 49.7476 | 49.8547 | 49.8011 | 49.9490 | 49.7020 | 43.5661 |
| potsdam/potsdam | 55.1778 | 54.8216 | 54.9119 | 55.1785 | 55.2076 | 55.1906 | 55.1801 | 55.2456 | 55.2185 | 55.2271 | 55.2653 | 54.9989 | 55.0610 | 54.9160 | 54.9704 | 55.2075 | 54.6560 | 54.3347 | 48.4787 |
| udd5/udd5 | 51.2305 | 50.7777 | 51.2136 | 51.1644 | 51.2752 | 51.0466 | 51.1169 | 51.1525 | 51.1116 | 51.2053 | 51.2359 | 50.9693 | 51.0807 | 50.8967 | 51.0522 | 51.1904 | 51.4513 | 50.9365 | 39.8034 |
| oem/oem | 29.9714 | 29.5309 | 29.8628 | 29.9289 | 29.9796 | 29.9029 | 29.8938 | 29.9651 | 29.9353 | 29.9334 | 30.0045 | 29.6136 | 29.6942 | 29.7442 | 29.8369 | 29.9887 | 31.5197 | 30.9291 | 25.4862 |
| loveda/P | 75.5356 | 75.2590 | 75.2137 | 75.4297 | 75.4389 | 75.5321 | 75.4950 | 75.3913 | 75.5424 | 75.5209 | 75.3744 | 75.3854 | 75.3168 | 75.5922 | 75.5485 | 75.5356 | 73.6555 | 73.5138 | 68.3469 |
| loveda/D | 46.6718 | 45.5946 | 47.0139 | 46.5009 | 46.6291 | 46.5049 | 46.5022 | 46.4955 | 46.6257 | 46.5742 | 46.5858 | 46.1442 | 46.4980 | 46.5035 | 46.8286 | 46.4981 | 46.3297 | 45.3674 | 44.1777 |
| vaihingen/vaihingen | 51.8593 | 51.5130 | 51.9034 | 51.7123 | 51.7421 | 51.6951 | 51.7547 | 51.7606 | 51.7165 | 51.8127 | 51.7939 | 51.6794 | 51.7682 | 51.5421 | 51.5733 | 51.7865 | 51.4654 | 51.0516 | 44.8651 |
| landcoverai/landcoverai | 78.0381 | 77.9179 | 77.9643 | 78.0184 | 78.0200 | 78.0223 | 78.0306 | 78.0114 | 78.0101 | 78.0272 | 78.0140 | 77.9836 | 78.0298 | 78.0665 | 78.1053 | 78.0799 | 61.9505 | 77.2244 | 79.3731 |
| flair1/flair1 | 38.9230 | 38.7829 | 38.6442 | 38.9936 | 39.0031 | 38.9859 | 39.0009 | 39.0274 | 38.9900 | 38.9958 | 39.0357 | 38.8005 | 38.8329 | 38.7174 | 38.7530 | 38.8971 | 38.7015 | 38.5734 | 32.6430 |

## Domain Mean

LoveDA D once; P separately. These are developed inputs.

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| OneSide_Soft | 50.2181 |
| OneSide_ObservationMean | 49.8154 |
| OneSide_Hard | 50.1639 |
| OneSide_ClassMean | 50.1632 |
| OneSide_MatchedClassMean | 50.2118 |
| OneSide_AliasShuffle0 | 50.1452 |
| OneSide_AliasShuffle1 | 50.1579 |
| OneSide_AliasShuffle2 | 50.1814 |
| OneSide_MatchedAliasShuffle0 | 50.1806 |
| OneSide_MatchedAliasShuffle1 | 50.1985 |
| OneSide_MatchedAliasShuffle2 | 50.2184 |
| OneSide_RivalCollapsed | 49.9875 |
| OneSide_MatchedRivalCollapsed | 50.0943 |
| OneSide_ShuffledWrite | 50.0168 |
| OneSide_MatchedShuffledWrite | 50.1218 |
| OneSide_DirectMatched | 50.1812 |
| OneSide_EqualLogitMean | 48.2529 |
| OneSide_EqualLogitMeanNoAlias | 49.7649 |
| OneSide_AllCorrespondenceShuffle | 44.7992 |

## Actual Complete-Image Warm Cost

| Domain | Bounded baseline ms | Same-information no attenuation ms | Soft ms | VIP20 ms | Soft/no attenuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd | 446.73 | 796.05 | 854.57 | 186.14 | 1.0735x |
| potsdam | 239.35 | 588.66 | 682.87 | 102.95 | 1.1600x |
| udd5 | 397.63 | 684.92 | 730.35 | 170.11 | 1.0663x |
| oem | 240.76 | 589.14 | 686.26 | 103.93 | 1.1649x |
| loveda | 251.63 | 652.87 | 843.82 | 203.03 | 1.2925x |
| vaihingen | 237.22 | 583.10 | 675.65 | 101.07 | 1.1587x |
| landcoverai | 228.09 | 573.51 | 664.88 | 96.36 | 1.1593x |
| flair1 | 236.76 | 588.43 | 697.61 | 103.38 | 1.1856x |

Three fixed complete inputs/domain, three synchronized warmed singleton repeats. Serial idle GPU, no masks; graph setup in timing.json. Shared-resident peaks are not standalone memory. Not full-dataset throughput. No prior timing substitution.

## Coverage And Competitive Activation

| Domain/protocol | Class | No attenuation IoU | Soft IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.9792 | 21.0681 | +13460 | -55704 |
| vdd/vdd | wall | 54.7467 | 55.6308 | +50410 | +18235 |
| vdd/vdd | road | 16.4903 | 16.5334 | +9696 | +34935 |
| vdd/vdd | vegetation | 48.5575 | 48.9341 | +105445 | -1717 |
| vdd/vdd | vehicle | 26.8392 | 26.8570 | +962 | +2927 |
| vdd/vdd | roof | 87.1274 | 87.5307 | +926 | -140792 |
| vdd/vdd | water | 92.3539 | 92.5582 | +3041 | -41824 |
| potsdam/potsdam | impervious surface | 77.2062 | 77.3746 | -1484 | -9833 |
| potsdam/potsdam | building | 83.9723 | 83.9116 | -4204 | -4073 |
| potsdam/potsdam | low vegetation | 64.6930 | 65.3738 | +19531 | +5795 |
| potsdam/potsdam | tree | 62.2148 | 62.6535 | +3854 | -3920 |
| potsdam/potsdam | car | 33.0103 | 33.1180 | +46 | -1505 |
| potsdam/potsdam | clutter | 7.8333 | 8.6353 | +2109 | -6316 |
| udd5/udd5 | vegetation | 72.3957 | 73.1690 | +1115220 | +106915 |
| udd5/udd5 | building | 83.4492 | 83.7786 | -42360 | -852786 |
| udd5/udd5 | road | 43.9514 | 44.7628 | +723890 | +290903 |
| udd5/udd5 | vehicle | 19.2408 | 19.4049 | +17071 | -42241 |
| udd5/udd5 | other | 34.8514 | 35.0373 | -187781 | -1128831 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +1539 |
| oem/oem | rangeland | 6.7226 | 6.8975 | +2517 | +1759 |
| oem/oem | developed space | 22.6743 | 22.7224 | -17828 | -85730 |
| oem/oem | road | 29.2393 | 29.6258 | +3315 | +2086 |
| oem/oem | tree | 25.1199 | 27.0871 | +26606 | +14384 |
| oem/oem | water | 17.1844 | 17.0185 | +215 | +2557 |
| oem/oem | agriculture land | 72.6534 | 72.9856 | -2267 | -7538 |
| oem/oem | building | 62.6536 | 63.4347 | +38971 | +19414 |
| loveda/P | building | 89.6217 | 89.6315 | -125 | -179 |
| loveda/P | road | 84.6982 | 85.1967 | +396 | -4933 |
| loveda/P | water | 71.5268 | 72.6192 | +16356 | +1185 |
| loveda/P | barren | 70.8717 | 68.0139 | -344 | +8305 |
| loveda/P | tree | 62.7904 | 64.6523 | +21919 | +1744 |
| loveda/P | farm | 72.0453 | 73.1002 | -3570 | -40754 |
| loveda/D | background | 11.5187 | 14.6919 | +91797 | +21072 |
| loveda/D | building | 48.0033 | 48.4265 | -135 | -6124 |
| loveda/D | road | 65.5973 | 65.9654 | +385 | -6039 |
| loveda/D | water | 67.1491 | 67.7771 | +10548 | +1622 |
| loveda/D | barren | 26.3807 | 26.2285 | +56 | +3421 |
| loveda/D | tree | 51.0505 | 52.2329 | +27820 | +23839 |
| loveda/D | farm | 49.4626 | 51.3805 | -11701 | -156561 |
| vaihingen/vaihingen | impervious surface | 63.5007 | 63.9288 | +7034 | -9127 |
| vaihingen/vaihingen | building | 69.5201 | 70.2291 | -1471 | -32201 |
| vaihingen/vaihingen | low vegetation | 29.3648 | 29.7808 | +5367 | -3208 |
| vaihingen/vaihingen | tree | 68.0361 | 68.1148 | +14089 | +18291 |
| vaihingen/vaihingen | car | 27.1435 | 27.2431 | +621 | +605 |
| landcoverai/landcoverai | background | 85.0344 | 85.1704 | -107 | -1326 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7416 | 92.7243 | +1181 | +1471 |
| landcoverai/landcoverai | water | 93.6191 | 93.8996 | -46 | -1113 |
| landcoverai/landcoverai | road | 40.2765 | 40.3582 | +12 | -72 |
| flair1/flair1 | building | 58.8079 | 59.6299 | +61 | -2249 |
| flair1/flair1 | pervious surface | 27.6578 | 28.8614 | +775 | +212 |
| flair1/flair1 | impervious surface | 69.3510 | 69.6020 | +67 | -1414 |
| flair1/flair1 | bare soil | 12.6674 | 11.4686 | +2 | +13942 |
| flair1/flair1 | water | 95.7515 | 95.6854 | -39 | -2 |
| flair1/flair1 | coniferous | 3.1536 | 2.8284 | -19 | +145 |
| flair1/flair1 | deciduous | 66.8952 | 66.8983 | +891 | +1318 |
| flair1/flair1 | brushwood | 7.0324 | 8.3984 | +1961 | -2103 |
| flair1/flair1 | vineyard | 55.9338 | 56.9048 | -47 | -3596 |
| flair1/flair1 | herbaceous vegetation | 35.8534 | 35.3889 | -3063 | -65 |
| flair1/flair1 | agricultural land | 32.2904 | 31.4099 | -3793 | +2495 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | -5479 |

## Claim Status

Paired attribution and prospective thresholds are evaluated by report_one_sided_alias_audit.py after collection. No control promotion, threshold tuning or automatic full-suite launch. The existing full20092 reference remains unchanged.

## Interpretation And Decision

All96 ordered unique complete inputs finish, including full UDD540. Both real
checkpoint mask-free smokes pass; original Geometry/NoAdmission/PatchOnly2,
positive observation, previous hard and PRIMARY soft per-image confusions replay
exactly. Source vocabulary/checkpoint identities, exact20 counts/class, scored
targets, independently recomputed aggregate mIoU, actual4/4/16 observation caps
and control budgets/norms verify. Twelve source tests plus26 regressions and
four existing offline paired-statistics tests pass. All workers and all eight
serial timing jobs exit after complete output. No primary equation is changed.

The incumbent50.2181 exceeds same-information no attenuation49.8154 by0.4027pp
on8/8 main domains and LoveDA P. The conditional2000 paired source-group interval
is[+0.2386,+0.5357]pp. Additional positive fine information accounts for3.1704pp
of the3.5731pp pilot gain over bounded patch-only2; that is NOT alias efficacy.
Soft-minus-hard is0.0543pp with interval[-0.0189,+0.1171], not verified hard
admission superiority. These developed96 inputs are not the full20092 suite,
untouched validation or method-selection-adjusted evidence.

| Matched comparator | Primary gain pp | Conditional95% interval pp |
| --- | ---: | --- |
| Class-mean risk, no norm matching |+0.0550|[+0.0334,+0.0814]|
| Class-mean risk, matched written norm |+0.0064|[-0.0120,+0.0309]|
| Mean word-identity null, matched written norm |+0.0190|[-0.0021,+0.0461]|
| Rival-collapsed risk, matched written norm |+0.1239|[+0.0800,+0.1752]|
| Word-write correspondence shuffle, matched written norm |+0.0964|[+0.0353,+0.1741]|
| Direct word writing, matched written norm |+0.0370|[+0.0033,+0.0714]|

The matched fields inherit the PRIMARY word-derived total valid class-field
norm per tile. They test word allocation/spatial direction beyond that budget;
they are NOT independently deployable word-free models. Near-equality therefore
does not prove that all word information is useless, or that every alias is good.
It shows the current word identity advantage is mainly compatible with a change
in overall correction strength rather than a verified better allocation at
fixed strength. Competitor identity remains useful even after that matching.
Spatial correspondence also has conditional evidence. The direct-write gain
is positive but below the frozen0.05pp practical criterion; neither a new solver
nor universal local correctness preservation is established.

Equal-logit fusion with identical word action is1.9653pp lower in mean, but its
interval[-0.2255,+2.5670] includes zero. LandCover.ai has ZERO building targets on
this small panel: equal fusion predicts9 building pixels, adding a fifth class
to union-scored mIoU. Its61.9505 versus77.2244 no-word score is largely affected
by this participation discontinuity, not a corresponding large pixelwise collapse.
Do not sell that contrast as universal coupling superiority. Primary scoring
is not changed. Complete20092 same-field base coupling evidence remains separate.

Primary domain-average warm latency664.88-854.57ms; mean paired primary/same-fine
no-attenuation ratio1.157599 passes the frozen cost check. Absolute domain means
are all below1000ms, but LoveDA word overhead is29.25%, so the average is not an
all-domain15.8% guarantee. Paired VIP20 times96.36-203.03ms imply4.156-6.900x
primary/VIP ratios. These24 fixed complete images/three warmed repeats are not
full throughput or standalone memory. Graph setup and real setup forwards are
recorded separately in each timing.json; historical timings are not substituted.

Coverage remains mixed. Potsdam car gains46 TP/removes1505 FP and low vegetation
gains19531 TP; building loses4204 TP. UDD5 vehicle gains17071 TP/removes42241 FP,
and road gains723890 TP but adds290903 FP. VDD vehicle gains962 TP but adds2927 FP.
LoveDA P barren loses344 TP/adds8305 FP; FLAIR bare soil adds13942 FP. Improved
overall mIoU is not universal good-word/bad-word identification or error repair.

Same-information gain, matched competitor conditioning, matched spatial word
writing and both cost checks pass. Matched class/word-allocation attribution
FAILS. Preserve the full performance reference; no threshold/source tuning,
favorable-control promotion or automatic vocabulary stress/full rollout follows.
The next missing distinction is useful word-derived budget versus genuinely
better conditional lexical allocation, not another visual-consistency veto or
local-anchor route. The complete Geometry/conditional-word/coupled-write goal
remains active.

Raw: one_sided_alias_attribution_20261006/{summary.json,paired_statistics.json,
PAIRED_UNCERTAINTY.md,claim_checks.json,CLAIM_CHECKS.md} and per-domain merged
confusions/timing.json. Entries: ../tools/one_sided_alias_audit_experiment.py
and ../tools/report_one_sided_alias_audit.py. Re-running collect regenerates
the tables; preserve this interpretation separately when intentionally doing so.
