# Semantic Membership: Frozen Vocabulary-Time Source

Same96 developed complete inputs, unchanged20 words and bounded Geometry/wide/H. Frozen SNLI/MultiNLI-pretrained NLI runs only at vocabulary setup, not per image. No target image/mask entered its source cache. This is not a novelty claim from adding a text model.

## Decision

Reject advancement of SemanticMembership_Soft. Mean46.7673 versus unchanged
same-information46.6450 gains0.1223pp, conditional95% interval
[-0.3260,+0.4421]. Only3/8 main domains improve; worst protocol loss1.1440pp
is Potsdam. This is96 developed complete images, not full20092 or independent
validation. No prompt, posterior formula, strength, class exception or seed is
changed after this result. No favorable control is promoted.

Word-specific use is worse than the raw pooled NLI source by0.1158pp
[-0.2034,-0.0030], matched pooled by0.0976pp[-0.1619,-0.0200], matched class
mean by0.0793pp[-0.1438,-0.0007], and mean3 matched identity nulls by0.0990pp
[-0.1605,-0.0262]. Every norm match is feasible. Full conditional statistics
and frozen source checks are in semantic_membership_alias_20261006/
PAIRED_UNCERTAINTY.md and CLAIM_CHECKS.md. NLI model posteriors differ between
own/rival words, but this does not establish useful pixel-level alias judgment.

Potsdam car improves31.4528->31.9457 through8303 fewer FP and54 fewer TP, but
impervious surface loses2.4167pp and clutter loses3.0319pp. VDD vegetation
loses853758 TP; UDD5 road loses551274 TP and0.5031pp. These are aggregate class
pathways, not proof that a named individual alias caused the losses. FLAIR-1's
2.2095pp gain does not justify a domain-specific switch.

The source cache freezes6774 text-only pairs at the public pinned revision;
load10.5834s plus NLI forward1.6083s totals12.1916s, peak374.26MiB during setup.
No target pixels or masks enter the cache. Real-checkpoint VDD/Potsdam mask-free
smokes, primary singleton equality, nine new source tests,21 source regressions
and four offline paired-statistics tests pass. All96 ordered unique images,
original three per-image endpoints, checkpoints/vocabularies, exact20 counts,
paired targets and independently recomputed aggregate mIoU verify. Every worker
exits complete; observation caps are4 Geometry/4 wide/0 fine and0 per-image NLI.

There is no candidate singleton timing: source advancement fails before timing,
vocabulary stress or full rollout. Encoding counts and vocabulary-time NLI cost
are not evidence of per-image latency. Preserve the full retained performance
model unchanged. This result rejects this frozen membership/action combination,
not all NLI or all conditional alias handling. It does establish that static
semantic membership alone has not supplied the missing signed competitive
pixel-utility estimate; source posteriors are neither semantic truth nor pixel
correctness. The original three-part final-model objective remains active.

| Domain/protocol | Baseline | NLI word use | Pooled NLI source | Class mean | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.8468 | 46.3773 | 46.4729 | 46.4660 | -0.4695 |
| potsdam/potsdam | 51.5168 | 50.3728 | 50.7312 | 50.6770 | -1.1440 |
| udd5/udd5 | 47.0786 | 46.7323 | 46.8905 | 46.8678 | -0.3463 |
| oem/oem | 25.2530 | 26.4411 | 26.3375 | 26.2862 | +1.1881 |
| loveda/P | 67.2104 | 67.3071 | 67.2566 | 67.2572 | +0.0967 |
| loveda/D | 39.4705 | 39.7386 | 39.7135 | 39.6916 | +0.2681 |
| vaihingen/vaihingen | 49.1468 | 48.9886 | 49.1476 | 49.1387 | -0.1582 |
| landcoverai/landcoverai | 76.2441 | 75.6743 | 75.4790 | 75.6377 | -0.5698 |
| flair1/flair1 | 37.6036 | 39.8131 | 40.2926 | 40.2773 | +2.2095 |

## Domain Mean

LoveDA D once; P separate. Developed96, not full20092.

| Method | mIoU |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| SemanticMembership_Soft | 46.7673 |
| SemanticMembership_PooledSource | 46.8831 |
| SemanticMembership_MatchedPooledSource | 46.8649 |
| SemanticMembership_ClassMean | 46.8803 |
| SemanticMembership_MatchedClassMean | 46.8466 |
| SemanticMembership_AliasShuffle0 | 46.8922 |
| SemanticMembership_AliasShuffle1 | 46.8577 |
| SemanticMembership_AliasShuffle2 | 46.7828 |
| SemanticMembership_MatchedAliasShuffle0 | 46.8640 |
| SemanticMembership_MatchedAliasShuffle1 | 46.8800 |
| SemanticMembership_MatchedAliasShuffle2 | 46.8548 |
| SemanticMembership_RivalCollapsed | 46.6375 |
| SemanticMembership_MatchedRivalCollapsed | 46.5359 |
| SemanticMembership_MatchedShuffledWrite | 46.9135 |
| SemanticMembership_DirectMatched | 46.7357 |

## Semantic Setup And Cost

```json
{
  "load_seconds": 10.583355989074335,
  "inference_seconds": 1.6082907798700035,
  "total_seconds": 12.191647344036028,
  "text_pairs": 6774,
  "gpu": 0,
  "peak_allocated_mib": 374.2587890625
}
```

NLI load/forwards are vocabulary-time costs. No singleton timing collected until source advancement; multi-arm evaluation duration is not primary latency.

## Correct Coverage And Competitive Activation

| Domain/protocol | Class | Baseline IoU | Primary IoU | Delta TP | Delta FP |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.6232 | 20.5522 | +182790 | +988857 |
| vdd/vdd | wall | 46.3090 | 46.0377 | -16402 | -8708 |
| vdd/vdd | road | 15.2940 | 15.4321 | -12241 | -160431 |
| vdd/vdd | vegetation | 45.8052 | 42.8260 | -853758 | -29491 |
| vdd/vdd | vehicle | 23.8613 | 23.5587 | -1178 | +8475 |
| vdd/vdd | roof | 85.4306 | 85.5176 | +978 | -30704 |
| vdd/vdd | water | 90.6042 | 90.7166 | -20189 | -47998 |
| potsdam/potsdam | impervious surface | 75.3698 | 72.9531 | -12420 | +105527 |
| potsdam/potsdam | building | 80.4415 | 78.5851 | +111 | +32170 |
| potsdam/potsdam | low vegetation | 60.1752 | 60.6123 | +21766 | +19424 |
| potsdam/potsdam | tree | 57.3938 | 56.9046 | -16997 | -17286 |
| potsdam/potsdam | car | 31.4528 | 31.9457 | -54 | -8303 |
| potsdam/potsdam | clutter | 4.2680 | 1.2361 | -13213 | -110725 |
| udd5/udd5 | vegetation | 66.2953 | 65.6931 | -898479 | -144044 |
| udd5/udd5 | building | 81.2771 | 81.0485 | +444554 | +1138883 |
| udd5/udd5 | road | 37.2416 | 36.7385 | -551274 | -500650 |
| udd5/udd5 | vehicle | 16.9950 | 17.1286 | -941 | -132152 |
| udd5/udd5 | other | 33.5843 | 33.0530 | -315920 | +960023 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0 | +24533 |
| oem/oem | rangeland | 4.3949 | 4.2687 | -1841 | -3395 |
| oem/oem | developed space | 21.7107 | 19.4625 | -287806 | -990208 |
| oem/oem | road | 26.4384 | 26.6402 | +4265 | +10848 |
| oem/oem | tree | 10.8604 | 11.4500 | +6618 | +2253 |
| oem/oem | water | 18.0795 | 18.3376 | +140 | -846 |
| oem/oem | agriculture land | 69.4583 | 68.8836 | +117 | +8834 |
| oem/oem | building | 51.0820 | 62.4863 | +701916 | +524572 |
| loveda/P | building | 88.2412 | 88.0117 | +202 | +1183 |
| loveda/P | road | 79.8873 | 80.0553 | +380 | -1567 |
| loveda/P | water | 62.0850 | 62.6228 | +7752 | +266 |
| loveda/P | barren | 70.3607 | 70.9059 | -115 | -1794 |
| loveda/P | tree | 39.2222 | 38.7549 | -5322 | -373 |
| loveda/P | farm | 63.4659 | 63.4921 | +219 | -831 |
| loveda/D | background | 4.0203 | 7.0722 | +92064 | +101062 |
| loveda/D | building | 45.7347 | 45.9401 | -1052 | -5422 |
| loveda/D | road | 63.0492 | 63.6681 | -703 | -13046 |
| loveda/D | water | 58.9731 | 59.4786 | +8557 | +1770 |
| loveda/D | barren | 27.9609 | 26.5113 | -23905 | -61220 |
| loveda/D | tree | 34.5990 | 32.9313 | -20944 | -4021 |
| loveda/D | farm | 41.9559 | 42.5689 | -3467 | -69673 |
| vaihingen/vaihingen | impervious surface | 60.8940 | 60.6430 | -16266 | -14202 |
| vaihingen/vaihingen | building | 65.1989 | 64.2352 | +1772 | +50540 |
| vaihingen/vaihingen | low vegetation | 27.2861 | 27.3390 | +1894 | +3935 |
| vaihingen/vaihingen | tree | 65.8712 | 65.8061 | -8409 | -10814 |
| vaihingen/vaihingen | car | 26.4838 | 26.9198 | -238 | -8212 |
| landcoverai/landcoverai | background | 82.5166 | 81.0161 | +3686 | +18877 |
| landcoverai/landcoverai | building | N/A | N/A | +0 | +0 |
| landcoverai/landcoverai | woodland | 91.5012 | 90.0244 | -17750 | -2465 |
| landcoverai/landcoverai | water | 91.7035 | 91.9302 | +0 | -897 |
| landcoverai/landcoverai | road | 39.2550 | 39.7267 | -240 | -1211 |
| flair1/flair1 | building | 49.1283 | 48.4928 | -676 | +1159 |
| flair1/flair1 | pervious surface | 17.8467 | 18.9908 | +627 | -86 |
| flair1/flair1 | impervious surface | 66.4519 | 66.7288 | -358 | -2324 |
| flair1/flair1 | bare soil | 21.3471 | 20.8798 | +1 | +1766 |
| flair1/flair1 | water | 94.9931 | 94.9966 | +1 | -1 |
| flair1/flair1 | coniferous | 5.9298 | 7.3674 | +119 | +503 |
| flair1/flair1 | deciduous | 64.3305 | 63.4527 | -3546 | -1655 |
| flair1/flair1 | brushwood | 3.3685 | 1.9313 | -3729 | -66939 |
| flair1/flair1 | vineyard | 51.9276 | 60.0756 | -522 | -31025 |
| flair1/flair1 | herbaceous vegetation | 37.8267 | 47.8132 | +69760 | +12072 |
| flair1/flair1 | agricultural land | 38.0928 | 47.0281 | +66724 | +43498 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0 | -85369 |
