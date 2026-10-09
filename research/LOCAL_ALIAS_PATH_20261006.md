# Local Lexical Fidelity Path: Frozen Eight-Domain Pilot

Same96 developed complete inputs and unchanged20 words. Existing fine features judge local aliases with the SAME normalized local text bank. Local fixed-slot soft actions change anchor L to L+u, written by original (I-H); exact previous wide soft action and H remain. No additional RGB/head/text encoding; at most4 Geometry/4 wide/16 bounded fine observations. Corrected-anchor fidelity is not a no-harm guarantee to original scores.

| Domain/protocol | LocalPath_JointSoft | LocalPath_ObservationMean | LocalPath_PreviousSoft | LocalPath_PreviousHard | LocalPath_MatchedPrevious | LocalPath_ClassMean | LocalPath_AliasShuffle0 | LocalPath_AliasShuffle1 | LocalPath_AliasShuffle2 | LocalPath_ShuffledWrite | LocalPath_LocalOnly | LocalPath_ReferenceShuffle | LocalPath_DirectMatched | Primary-PreviousSoft pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 50.0413 | 49.5849 | 49.8732 | 49.7970 | 50.3205 | 50.0516 | 50.0474 | 50.0506 | 50.0362 | 49.8366 | 49.7697 | 50.0846 | 50.0668 | +0.1681 |
| potsdam/potsdam | 54.5327 | 54.8216 | 55.1778 | 54.9119 | 55.6091 | 54.5612 | 54.5295 | 54.5664 | 54.5102 | 54.1817 | 54.2051 | 51.5181 | 54.2593 | -0.6451 |
| udd5/udd5 | 51.3657 | 50.7777 | 51.2305 | 51.2136 | 52.1953 | 51.3878 | 51.4119 | 51.3556 | 51.3924 | 51.5294 | 50.9257 | 47.3466 | 51.3585 | +0.1352 |
| oem/oem | 30.4122 | 29.5309 | 29.9714 | 29.8628 | 31.0237 | 30.3579 | 30.3620 | 30.3643 | 30.3763 | 30.5582 | 29.9424 | 28.0693 | 30.3947 | +0.4408 |
| loveda/P | 74.0856 | 75.2590 | 75.5356 | 75.2137 | 75.0472 | 74.1293 | 74.0502 | 74.2025 | 74.2299 | 73.0466 | 74.3456 | 70.7465 | 73.7954 | -1.4500 |
| loveda/D | 47.2603 | 45.5946 | 46.6718 | 47.0139 | 48.1696 | 47.3584 | 47.3768 | 47.3453 | 47.3254 | 46.7177 | 46.6650 | 46.4901 | 47.6691 | +0.5885 |
| vaihingen/vaihingen | 51.6392 | 51.5130 | 51.8593 | 51.9034 | 52.4688 | 51.6595 | 51.5649 | 51.5934 | 51.5925 | 51.9823 | 51.3786 | 47.2991 | 51.5892 | -0.2201 |
| landcoverai/landcoverai | 78.4317 | 77.9179 | 78.0381 | 77.9643 | 78.3370 | 78.4059 | 78.4056 | 78.4234 | 78.4225 | 78.5354 | 78.3032 | 79.1812 | 78.4599 | +0.3936 |
| flair1/flair1 | 38.8315 | 38.7829 | 38.9230 | 38.6442 | 39.0706 | 38.7894 | 38.7814 | 38.7735 | 38.8209 | 38.9245 | 38.7172 | 37.2485 | 38.8574 | -0.0915 |

| Method | Eight-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| LocalPath_JointSoft | 50.3143 |
| LocalPath_ObservationMean | 49.8154 |
| LocalPath_PreviousSoft | 50.2181 |
| LocalPath_PreviousHard | 50.1639 |
| LocalPath_MatchedPrevious | 50.8993 |
| LocalPath_ClassMean | 50.3215 |
| LocalPath_AliasShuffle0 | 50.3099 |
| LocalPath_AliasShuffle1 | 50.3091 |
| LocalPath_AliasShuffle2 | 50.3096 |
| LocalPath_ShuffledWrite | 50.2832 |
| LocalPath_LocalOnly | 49.9884 |
| LocalPath_ReferenceShuffle | 48.4047 |
| LocalPath_DirectMatched | 50.3319 |

## Actual Complete-Image Warm Cost

Not measured. Previous source timings are not measurements of this new candidate.

Three fixed complete inputs/domain, three warmed synchronized singleton repeats when acquired. Graph setup and actual setup forwards are separate; shared peaks are not standalone memory.

14 focused tests and26 source regression tests; both mask-free previous/positive/hard/soft/singleton smokes and all96 per-image endpoint replays must verify, with unique coverage, targets and identities.

## Frozen Advancement Checks

```json
{
  "means": {
    "Geometry": 44.683575,
    "NoAdmission_Exact": 46.3359625,
    "Geometry_PatchOnly2Coupled": 46.645025000000004,
    "LocalPath_JointSoft": 50.314325000000004,
    "LocalPath_ObservationMean": 49.8154375,
    "LocalPath_PreviousSoft": 50.2181375,
    "LocalPath_PreviousHard": 50.1638875,
    "LocalPath_MatchedPrevious": 50.899325,
    "LocalPath_ClassMean": 50.321462499999996,
    "LocalPath_AliasShuffle0": 50.309937500000004,
    "LocalPath_AliasShuffle1": 50.3090625,
    "LocalPath_AliasShuffle2": 50.30955,
    "LocalPath_ShuffledWrite": 50.283225,
    "LocalPath_LocalOnly": 49.988362499999994,
    "LocalPath_ReferenceShuffle": 48.404687499999994,
    "LocalPath_DirectMatched": 50.3318625
  },
  "speed_ms": {},
  "graph_setup": {},
  "shared_peak_mib": {},
  "unique_images": 96,
  "domain_wins": 5,
  "exact_original_positive_soft_hard_replay": true,
  "accuracy_mechanism_passed": false,
  "component_deltas_pp": {
    "source": 0.09618750000000631,
    "matched_source": -0.5849999999999937,
    "word_identity": 0.004808333333336634,
    "class_only": -0.007137499999991803,
    "write_correspondence": 0.031100000000002126
  },
  "gate": {
    "passed": false,
    "checks": {
      "mean_gain": false,
      "domain_wins": true,
      "worst_protocol_loss": false,
      "word_gain": true,
      "above_class_mean": false,
      "above_identity_null": false,
      "above_matched_previous": false,
      "no_unmatchable_field": true,
      "above_LocalPath_ReferenceShuffle": true,
      "above_LocalPath_LocalOnly": true,
      "above_LocalPath_ShuffledWrite": false,
      "above_LocalPath_DirectMatched": false
    }
  }
}
```

No post-result threshold, source or domain-route changes and no control promotion. A failed source gets no timing/stress/full rollout. These are developed inputs, not independent validation.

## Interpretation And Decision

All96 ordered unique complete inputs finish, including UDD5 full40. Every
original Geometry/NoAdmission/PatchOnly2 and previous positive-mean/soft/hard
per-image confusion replays exactly against the preceding collected experiment.
Every scored target, checkpoint/vocabulary identity, independent aggregate mIoU
and source/action invariant verifies. All vocabulary counts remain exactly20;
the diagnostic retained_count_mean counts fully unattenuated slots per comparison,
not effective soft mass or permanently deleted words.
14 new tests and26 source regressions pass remotely with real dependencies;
four offline statistics tests pass. Both real-checkpoint mask-free smokes pass
with frozen weights, exact legacy endpoints and primary singleton equality.
Every worker exited normally. No original source or retained model was changed.

Primary50.3143 exceeds previous soft50.2181 by0.0962pp and wins5/8 domain point
estimates, but Potsdam loses0.6451pp, Vaihingen0.2201pp and FLAIR-10.0915pp.
LoveDA P loses1.4500pp and is not included in the eight-domain mean. The frozen
accuracy/mechanism advancement check FAILS, not just the0.1pp mean criterion.

| Comparator | Primary gain pp | Conditional95% paired interval pp |
| --- | ---: | --- |
| Same-information no attenuation |+0.4989|[+0.0891,+0.7504]|
| Exact previous soft |+0.0962|[-0.2767,+0.2569]|
| Matched local class-mean |-0.0071|[-0.0200,+0.0140]|
| Mean local alias-identity null |+0.0048|[-0.0096,+0.0270]|
| Magnitude-matched previous soft |-0.5850|[-0.8779,-0.3928]|
| Local intervention-only correspondence shuffle |+0.0311|[-0.0594,+0.1344]|
| Local-only word action |+0.3260|[+0.1814,+0.4456]|
| Fine same-bank reference position shuffle |+1.9096|[+1.4682,+2.5113]|
| Magnitude-matched direct local writing |-0.0175|[-0.1260,+0.0467]|

These2000 paired filename-source-group intervals are conditional development
evidence, not independent or method-selection-adjusted uncertainty. Identity
nulls average scores on the SAME resample, not pooled confusion matrices.
ClassMean and identity shuffles change ONLY the new local risk; the established
wide word action is held fixed. Their near-equality rejects attribution of the
NEW local contribution to useful word identity, not all prior wide alias use.
Correct fine-reference position is useful in aggregate, but does not certify
word correctness; VDD and LandCover.ai improve under reference permutation.

The no-attenuation/previous-soft/local-only/joint2x2 mIoU means are49.8154,
50.2181,49.9884,50.3143. Adding local action without old wide action gives
0.1729pp; adding it WITH old wide action gives0.0962pp. Retaining wide action
with local action gives0.3260pp, versus0.4027pp without local action. The mIoU
difference-of-differences is-0.0767pp; nonlinear mIoU does not prove a general
algebraic interaction. Logit-space additive factorial identity is tested exactly.

Matching the previous wide correction to the primary TOTAL class-field norm
yields50.8993,0.5850pp above primary. Domain/protocol-averaged matching scales
are3.5039-4.9002, with zero unmatchable fields and numerical norm errors near
1e-16. This is a predeclared diagnostic, NOT a new selected final model. It
shows that this new correction direction is inferior to the existing wide
direction at matched strength on this panel; a positive total mean alone is
insufficient source/word attribution. Original(I-H) also does not beat matched
direct writing or establish useful new spatial correspondence in aggregate.

Compared with previous soft, Potsdam car adds86 TP but29339 FP and loses1.8057pp;
tree loses31898 TP. Vaihingen car adds1003 TP but43006 FP and loses2.1364pp.
UDD5 vehicle adds16592 TP but566873 FP, losing0.5866pp despite road gaining
0.6405pp. VDD vehicle also loses0.6075pp while road gains0.4397pp. Thus the
new local correction aggravates small-class false activation; improved large
class coverage does not establish the desired balanced error repair. LoveDA D
gains0.5885pp while P loses1.4500pp, showing protocol-specific tradeoffs.

Reject advancement with no coefficient/threshold/domain-route tuning, no
promotion of the stronger magnitude control, and no timing/stress/full rollout.
No new latency is measured or borrowed from the preceding source. Preserve
Geometry_PatchOnly2Coupled as the full performance reference. This exact
same-bank local contradiction rule and complement writing are not the final
three-part model. The missing evidence remains useful word-specific competitive
action beyond class recalibration, with a spatial write route that earns its
cost and preserves correct local coverage. Merely adding another path, stricter
eligibility test or greater correction amplitude is not sufficient.

Raw evidence: local_alias_path_20261006/PAIRED_UNCERTAINTY.md,
paired_statistics.json, summary.json and per-image confusions.
Reproducible entries: ../tools/local_alias_path_experiment.py and
../tools/report_local_alias_path.py. The full research goal remains active.

## Correct Coverage And Competitor Activation

| Domain/protocol | Class | PreviousSoft IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 21.0681 | 21.8294 | +213159 | -9640 |
| vdd/vdd | wall | 55.6308 | 55.8872 | +20241 | +15241 |
| vdd/vdd | road | 16.5334 | 16.9731 | +8116 | -188391 |
| vdd/vdd | vegetation | 48.9341 | 49.1290 | +31154 | -48574 |
| vdd/vdd | vehicle | 26.8570 | 26.2495 | +1910 | +30115 |
| vdd/vdd | roof | 87.5307 | 87.8057 | +1667 | -94075 |
| vdd/vdd | water | 92.5582 | 92.4154 | -5968 | +25045 |
| potsdam/potsdam | impervious surface | 77.3746 | 77.5703 | -14059 | -27272 |
| potsdam/potsdam | building | 83.9116 | 83.0917 | -6921 | +4426 |
| potsdam/potsdam | low vegetation | 65.3738 | 65.5464 | +12661 | +13211 |
| potsdam/potsdam | tree | 62.6535 | 61.4389 | -31898 | -23559 |
| potsdam/potsdam | car | 33.1180 | 31.3123 | +86 | +29339 |
| potsdam/potsdam | clutter | 8.6353 | 8.2368 | +2152 | +41834 |
| udd5/udd5 | vegetation | 73.1690 | 73.6730 | +675333 | -1273 |
| udd5/udd5 | building | 83.7786 | 83.7655 | +111614 | +164910 |
| udd5/udd5 | road | 44.7628 | 45.4033 | +481698 | +24740 |
| udd5/udd5 | vehicle | 19.4049 | 18.8183 | +16592 | +566873 |
| udd5/udd5 | other | 35.0373 | 35.1683 | -423696 | -1616791 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +34673 |
| oem/oem | rangeland | 6.8975 | 6.9578 | +1500 | +9652 |
| oem/oem | developed space | 22.7224 | 23.1387 | +3334 | -45789 |
| oem/oem | road | 29.6258 | 29.9196 | +1200 | -2864 |
| oem/oem | tree | 27.0871 | 27.8671 | +10262 | +4105 |
| oem/oem | water | 17.0185 | 17.6705 | +831 | -289 |
| oem/oem | agriculture land | 72.9856 | 73.7662 | -6554 | -19108 |
| oem/oem | building | 63.4347 | 63.9773 | +14888 | -5841 |
| loveda/P | building | 89.6315 | 89.6101 | +597 | +752 |
| loveda/P | road | 85.1967 | 84.4133 | +816 | +9483 |
| loveda/P | water | 72.6192 | 71.9588 | -3454 | +8228 |
| loveda/P | barren | 68.0139 | 59.7020 | -675 | +29220 |
| loveda/P | tree | 64.6523 | 65.3886 | +14806 | +10049 |
| loveda/P | farm | 73.1002 | 73.4411 | -24760 | -45062 |
| loveda/D | background | 14.6919 | 19.2820 | +146694 | +90339 |
| loveda/D | building | 48.4265 | 48.6214 | +596 | -1430 |
| loveda/D | road | 65.9654 | 65.6681 | +1199 | +7171 |
| loveda/D | water | 67.7771 | 66.5348 | -10200 | +12792 |
| loveda/D | barren | 26.2285 | 25.4451 | -1038 | +13033 |
| loveda/D | tree | 52.2329 | 52.0763 | +10395 | +23941 |
| loveda/D | farm | 51.3805 | 53.1941 | -61331 | -232161 |
| vaihingen/vaihingen | impervious surface | 63.9288 | 63.5693 | -23994 | -20794 |
| vaihingen/vaihingen | building | 70.2291 | 70.1329 | +1853 | +6688 |
| vaihingen/vaihingen | low vegetation | 29.7808 | 30.7053 | +13715 | -998 |
| vaihingen/vaihingen | tree | 68.1148 | 68.6817 | -1315 | -19164 |
| vaihingen/vaihingen | car | 27.2431 | 25.1067 | +1003 | +43006 |
| landcoverai/landcoverai | background | 85.1704 | 85.1715 | +482 | +556 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7243 | 92.6331 | +148 | +1205 |
| landcoverai/landcoverai | water | 93.8996 | 94.4037 | +118 | -1771 |
| landcoverai/landcoverai | road | 40.3582 | 41.5185 | +195 | -933 |
| flair1/flair1 | building | 59.6299 | 59.1428 | +276 | +1853 |
| flair1/flair1 | pervious surface | 28.8614 | 30.4332 | +875 | -199 |
| flair1/flair1 | impervious surface | 69.6020 | 68.3540 | -205 | +7320 |
| flair1/flair1 | bare soil | 11.4686 | 9.8789 | +26 | +23942 |
| flair1/flair1 | water | 95.6854 | 95.5389 | -107 | -26 |
| flair1/flair1 | coniferous | 2.8284 | 0.9897 | -128 | +532 |
| flair1/flair1 | deciduous | 66.8983 | 66.9079 | -1687 | -2565 |
| flair1/flair1 | brushwood | 8.3984 | 10.3384 | +3326 | +3200 |
| flair1/flair1 | vineyard | 56.9048 | 58.2067 | -56 | -4621 |
| flair1/flair1 | herbaceous vegetation | 35.3889 | 34.8651 | -3492 | -186 |
| flair1/flair1 | agricultural land | 31.4099 | 31.3226 | +252 | +2259 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | -30589 |
