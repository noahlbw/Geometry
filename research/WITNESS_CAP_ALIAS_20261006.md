# Fine-Witness Likelihood Cap: Frozen Eight-Domain Pilot

Same96 developed complete inputs and unchanged20 words. Only attenuation magnitude changes on identical protected wide-positive/fine-negative slots. Original Geometry, positive wide/fine information and H stay fixed. Capped likelihood is not semantic correctness. At most4 Geometry/4 wide/16 bounded fine encodings; no new observation or native sliding windows.

| Domain/protocol | WitnessCap_RivalSoft | WitnessCap_ObservationMean | WitnessCap_PreviousSoft | WitnessCap_PreviousHard | WitnessCap_MatchedPrevious | WitnessCap_ClassMean | WitnessCap_AliasShuffle0 | WitnessCap_AliasShuffle1 | WitnessCap_AliasShuffle2 | WitnessCap_ShuffledWrite | Primary-PreviousSoft pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 49.8228 | 49.5849 | 49.8732 | 49.7970 | 49.8115 | 49.7573 | 49.7609 | 49.7408 | 49.7462 | 49.7155 | -0.0504 |
| potsdam/potsdam | 55.0254 | 54.8216 | 55.1778 | 54.9119 | 55.0670 | 55.0302 | 55.0099 | 55.0212 | 55.0652 | 54.8641 | -0.1524 |
| udd5/udd5 | 51.2275 | 50.7777 | 51.2305 | 51.2136 | 51.1997 | 51.1210 | 51.0259 | 51.1115 | 51.1036 | 50.9063 | -0.0030 |
| oem/oem | 30.0087 | 29.5309 | 29.9714 | 29.8628 | 29.9260 | 29.9074 | 29.8852 | 29.8947 | 29.9314 | 29.7417 | +0.0373 |
| loveda/P | 75.2791 | 75.2590 | 75.5356 | 75.2137 | 75.4308 | 75.2455 | 75.3929 | 75.3127 | 75.2218 | 75.5929 | -0.2565 |
| loveda/D | 46.7896 | 45.5946 | 46.6718 | 47.0139 | 46.7513 | 46.5122 | 46.5214 | 46.5569 | 46.4350 | 46.5509 | +0.1178 |
| vaihingen/vaihingen | 51.8278 | 51.5130 | 51.8593 | 51.9034 | 51.8026 | 51.6954 | 51.7075 | 51.7283 | 51.7437 | 51.5402 | -0.0315 |
| landcoverai/landcoverai | 78.0238 | 77.9179 | 78.0381 | 77.9643 | 78.0223 | 78.0061 | 78.0103 | 78.0238 | 77.9974 | 78.0361 | -0.0143 |
| flair1/flair1 | 38.8016 | 38.7829 | 38.9230 | 38.6442 | 38.7894 | 38.8762 | 38.8594 | 38.9041 | 38.8863 | 38.6461 | -0.1214 |

| Method | Eight-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| WitnessCap_RivalSoft | 50.1909 |
| WitnessCap_ObservationMean | 49.8154 |
| WitnessCap_PreviousSoft | 50.2181 |
| WitnessCap_PreviousHard | 50.1639 |
| WitnessCap_MatchedPrevious | 50.1712 |
| WitnessCap_ClassMean | 50.1132 |
| WitnessCap_AliasShuffle0 | 50.0976 |
| WitnessCap_AliasShuffle1 | 50.1227 |
| WitnessCap_AliasShuffle2 | 50.1136 |
| WitnessCap_ShuffledWrite | 50.0001 |

## Actual Complete-Image Warm Cost

Not measured. Previous source timings are not measurements of this new candidate.

Three fixed complete inputs/domain, three warmed synchronized singleton repeats when acquired. Graph setup and actual setup forwards are separate; shared peaks are not standalone memory.

12 focused tests and26 source regression tests; both mask-free previous/positive/hard/soft/singleton smokes and all96 per-image endpoint replays must verify, with unique coverage, targets and identities.

## Frozen Advancement Checks

```json
{
  "means": {
    "Geometry": 44.683575,
    "NoAdmission_Exact": 46.3359625,
    "Geometry_PatchOnly2Coupled": 46.645025000000004,
    "WitnessCap_RivalSoft": 50.1909,
    "WitnessCap_ObservationMean": 49.8154375,
    "WitnessCap_PreviousSoft": 50.2181375,
    "WitnessCap_PreviousHard": 50.1638875,
    "WitnessCap_MatchedPrevious": 50.171225,
    "WitnessCap_ClassMean": 50.113225,
    "WitnessCap_AliasShuffle0": 50.0975625,
    "WitnessCap_AliasShuffle1": 50.122662500000004,
    "WitnessCap_AliasShuffle2": 50.1136,
    "WitnessCap_ShuffledWrite": 50.0001125
  },
  "speed_ms": {},
  "graph_setup": {},
  "shared_peak_mib": {},
  "unique_images": 96,
  "domain_wins": 2,
  "exact_original_positive_soft_hard_replay": true,
  "accuracy_mechanism_passed": false,
  "component_deltas_pp": {
    "source": -0.02723749999999825,
    "matched_source": 0.019674999999999443,
    "word_identity": 0.07962500000000006,
    "class_only": 0.07767499999999927,
    "write_correspondence": 0.190787499999999
  },
  "gate": {
    "passed": false,
    "checks": {
      "mean_gain": false,
      "domain_wins": false,
      "worst_protocol_loss": true,
      "word_gain": true,
      "above_class_mean": true,
      "above_identity_null": true,
      "above_matched_previous": false,
      "no_unmatchable_field": true
    }
  }
}
```

No post-result threshold, source or domain-route changes and no control promotion. A failed source gets no timing/stress/full rollout. These are developed inputs, not independent validation.

## Interpretation And Research Decision

All96 ordered complete inputs, original/positive/previous-soft/previous-hard
per-image endpoints, all-arm scored targets, vocabulary/checkpoint identities,
zero protected-slot action, unchanged eligible support and matched action norms
verify. The12 focused tests and26 source regressions pass with real remote
dependencies. Both no-mask smokes pass; every worker exited after complete output.

Primary50.1909 exceeds same-information no-attenuation49.8154 by0.3755pp and
previous hard50.1639 by0.0270pp, but is0.0272pp below exact previous soft50.2181,
with only2/8 main-domain wins. LoveDA D improves0.1178pp while P loses0.2565pp.
The predeclared accuracy/mechanism advancement gate FAILS. No threshold/source
revision, favorable-control promotion, timing, vocabulary stress or full rollout
follows. Preserve the established full performance reference.

The source nevertheless provides specific word evidence rather than only class
calibration: primary exceeds matched class-mean by0.0777pp and mean of three
identity nulls by0.0796pp. Per-image aggregate recomputation and2000 paired
filename-source-group draws give the following conditional development intervals:

| Comparator | Primary delta pp | Conditional95% paired interval pp |
| --- | ---: | --- |
| Same-information no attenuation |+0.3754|[+0.1737,+0.5220]|
| Exact previous soft |-0.0273|[-0.0685,+0.0114]|
| Matched class-mean |+0.0777|[+0.0421,+0.1185]|
| Mean alias-identity null |+0.0796|[+0.0422,+0.1215]|
| Magnitude-matched previous soft |+0.0197|[+0.0034,+0.0400]|
| Intervention-only H shuffle |+0.1908|[+0.1251,+0.2703]|

Mean null scores use the same bootstrap draw, not pooled confusions. These
intervals are not adjusted for method selection or independent validation.
The small magnitude-matched source benefit is positive here, not zero, but
does not reach the frozen0.05pp practical-advancement threshold. H correspondence
helps7/8 main domains under this intervention; LandCover.ai is the exception.
This is not a proof of universal transfer or a new solver contribution.

Compared with previous soft, UDD5 vehicle gains982 TP and removes30805 FP pixels,
but road loses56943 TP and0.0606pp IoU despite27695 fewer FP. Potsdam car adds2 TP
and969 FP, losing0.0631pp; low vegetation loses2568 TP and0.0951pp despite557
fewer FP. VDD vehicle loses135 TP and adds1498 FP. The analytic removal of broad
amplitude protection does not produce universally beneficial suppression.

The specific word source has clearer attribution than the preceding soft source,
but not a verified best-model improvement. These results do not establish that
the previous P/(P+N) dependence caused its weak identity advantage, nor that
fine negative evidence is always trustworthy. The next reliability distinction
remains useful contradiction versus incomplete/mismatched semantic observation;
another amplitude-only formula is not justified as a final innovation.
Raw paired statistics: witness_cap_alias_20261006/PAIRED_UNCERTAINTY.md and
paired_statistics.json; reproducible wrapper: ../tools/report_witness_cap_alias.py.

## Correct Coverage And Competitor Activation

| Domain/protocol | Class | PreviousSoft IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 21.0681 | 20.9747 | -24928 | +7079 |
| vdd/vdd | wall | 55.6308 | 55.4910 | -7777 | -2495 |
| vdd/vdd | road | 16.5334 | 16.5594 | +925 | -8711 |
| vdd/vdd | vegetation | 48.9341 | 48.9392 | +2855 | +2846 |
| vdd/vdd | vehicle | 26.8570 | 26.8027 | -135 | +1498 |
| vdd/vdd | roof | 87.5307 | 87.4517 | -3 | +27687 |
| vdd/vdd | water | 92.5582 | 92.5408 | -1291 | +2450 |
| potsdam/potsdam | impervious surface | 77.3746 | 77.3326 | -623 | +1161 |
| potsdam/potsdam | building | 83.9116 | 83.5439 | -3360 | +1668 |
| potsdam/potsdam | low vegetation | 65.3738 | 65.2787 | -2568 | -557 |
| potsdam/potsdam | tree | 62.6535 | 62.5851 | -590 | +625 |
| potsdam/potsdam | car | 33.1180 | 33.0549 | +2 | +969 |
| potsdam/potsdam | clutter | 8.6353 | 8.3575 | -580 | +3853 |
| udd5/udd5 | vegetation | 73.1690 | 73.1622 | -9755 | -848 |
| udd5/udd5 | building | 83.7786 | 83.7720 | +6196 | +23255 |
| udd5/udd5 | road | 44.7628 | 44.7022 | -56943 | -27695 |
| udd5/udd5 | vehicle | 19.4049 | 19.4503 | +982 | -30805 |
| udd5/udd5 | other | 35.0373 | 35.0505 | +35677 | +59936 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | -1049 |
| oem/oem | rangeland | 6.8975 | 6.8803 | -254 | -278 |
| oem/oem | developed space | 22.7224 | 22.7523 | +2551 | +6812 |
| oem/oem | road | 29.6258 | 29.6581 | +146 | -271 |
| oem/oem | tree | 27.0871 | 27.2797 | +2386 | +494 |
| oem/oem | water | 17.0185 | 17.2034 | +2 | -1442 |
| oem/oem | agriculture land | 72.9856 | 72.9719 | +504 | +872 |
| oem/oem | building | 63.4347 | 63.3241 | -6383 | -4090 |
| loveda/P | building | 89.6315 | 89.6426 | -49 | -99 |
| loveda/P | road | 85.1967 | 85.1573 | +94 | +535 |
| loveda/P | water | 72.6192 | 72.6704 | +576 | -209 |
| loveda/P | barren | 68.0139 | 66.6415 | -131 | +4293 |
| loveda/P | tree | 64.6523 | 64.5211 | -1459 | +12 |
| loveda/P | farm | 73.1002 | 73.0414 | -2335 | -1228 |
| loveda/D | background | 14.6919 | 15.4924 | +23903 | +8771 |
| loveda/D | building | 48.4265 | 48.3450 | -73 | +966 |
| loveda/D | road | 65.9654 | 65.9263 | +43 | +767 |
| loveda/D | water | 67.7771 | 67.6672 | -1687 | -47 |
| loveda/D | barren | 26.2285 | 26.3606 | -87 | -3117 |
| loveda/D | tree | 52.2329 | 52.1208 | -1528 | -84 |
| loveda/D | farm | 51.3805 | 51.6149 | -4174 | -23653 |
| vaihingen/vaihingen | impervious surface | 63.9288 | 63.8892 | -657 | +827 |
| vaihingen/vaihingen | building | 70.2291 | 70.2007 | +2 | +1197 |
| vaihingen/vaihingen | low vegetation | 29.7808 | 29.7113 | -1072 | -63 |
| vaihingen/vaihingen | tree | 68.1148 | 68.1101 | -292 | -284 |
| vaihingen/vaihingen | car | 27.2431 | 27.2278 | +18 | +324 |
| landcoverai/landcoverai | background | 85.1704 | 85.1298 | -180 | +146 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 92.7243 | 92.7163 | -85 | +0 |
| landcoverai/landcoverai | water | 93.8996 | 93.8527 | +8 | +186 |
| landcoverai/landcoverai | road | 40.3582 | 40.3962 | -8 | -67 |
| flair1/flair1 | building | 59.6299 | 59.5983 | -30 | +39 |
| flair1/flair1 | pervious surface | 28.8614 | 28.8854 | +12 | -8 |
| flair1/flair1 | impervious surface | 69.6020 | 69.1330 | -12 | +2814 |
| flair1/flair1 | bare soil | 11.4686 | 11.1454 | +0 | +4267 |
| flair1/flair1 | water | 95.6854 | 95.6871 | +1 | +0 |
| flair1/flair1 | coniferous | 2.8284 | 2.7789 | -4 | -15 |
| flair1/flair1 | deciduous | 66.8983 | 66.8931 | +335 | +524 |
| flair1/flair1 | brushwood | 8.3984 | 8.3763 | -207 | -2064 |
| flair1/flair1 | vineyard | 56.9048 | 56.6374 | +17 | +985 |
| flair1/flair1 | herbaceous vegetation | 35.3889 | 35.4937 | +710 | +68 |
| flair1/flair1 | agricultural land | 31.4099 | 30.9901 | -2452 | -838 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | -4142 |
