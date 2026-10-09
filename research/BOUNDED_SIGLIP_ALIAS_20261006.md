# Independent Bounded SigLIP Witness: Frozen Joint Alias Pilot

Same96 developed complete inputs, unchanged20 vocabularies, bounded896 patch-only2 Geometry/448 VIP wide, original G/H and output protocol. One pretrained SigLIP2-base256 whole-image observation; at most128 global supported query reads. Original trained MAP K/V are cached once and support probes are batched. Canonical-protected contradiction weights suppress at most one alias slot/class/source without survivor redistribution; joint local/wide changes use original H. Independent pretraining/support are not semantic correctness. NoGeometrySupport and direct observation mean preserve extra RGB/query information. This is exploratory development, not independent validation, a new MAP head or theorem.

| Dataset/protocol | Geometry | NoAdmission_Exact | Geometry_PatchOnly2Coupled | SigWitness_JointSoft | SigWitness_ClassMean | SigWitness_AliasShuffle | SigWitness_NoGeometrySupport | SigWitness_ObservationMean | Primary-baseline pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 42.7217 | 46.3491 | 46.8468 | 46.8303 | 46.8010 | 46.8052 | 46.8576 | 46.7689 | -0.0165 |
| potsdam/potsdam | 45.6466 | 49.7562 | 51.5168 | 51.5358 | 51.5502 | 51.5447 | 51.5252 | 50.3191 | +0.0190 |
| udd5/udd5 | 47.1219 | 47.0096 | 47.0786 | 47.0564 | 47.0511 | 47.0260 | 47.0963 | 46.6577 | -0.0222 |
| oem/oem | 31.4297 | 25.7311 | 25.2530 | 25.2685 | 25.2586 | 25.2596 | 25.2729 | 25.6276 | +0.0155 |
| loveda/P | 70.8274 | 67.0933 | 67.2104 | 67.2216 | 67.2204 | 67.2183 | 67.2485 | 67.9370 | +0.0112 |
| loveda/D | 46.6369 | 41.0700 | 39.4705 | 39.4088 | 39.4153 | 39.4188 | 39.4233 | 39.5309 | -0.0617 |
| vaihingen/vaihingen | 46.0831 | 48.3789 | 49.1468 | 49.0686 | 49.0968 | 49.1007 | 49.0289 | 47.8067 | -0.0782 |
| landcoverai/landcoverai | 57.9352 | 75.1447 | 76.2441 | 76.3192 | 76.3287 | 76.3456 | 76.2910 | 76.3903 | +0.0751 |
| flair1/flair1 | 39.8935 | 37.2481 | 37.6036 | 37.5892 | 37.5664 | 37.5546 | 37.6025 | 37.8380 | -0.0144 |

| Arm | Equal-domain mean, LoveDA D once |
| --- | ---: |
| Geometry | 44.6836 |
| NoAdmission_Exact | 46.3360 |
| Geometry_PatchOnly2Coupled | 46.6450 |
| SigWitness_JointSoft | 46.6346 |
| SigWitness_ClassMean | 46.6335 |
| SigWitness_AliasShuffle | 46.6319 |
| SigWitness_NoGeometrySupport | 46.6372 |
| SigWitness_ObservationMean | 46.3674 |

## Matched Complete-Image Timing

| Domain | Baseline ms | Primary ms | VIP20 ms | Primary/baseline |
| --- | ---: | ---: | ---: | ---: |
| vdd | 433.43 | 536.36 | 178.74 | 1.237x |
| potsdam | 238.23 | 394.44 | 102.58 | 1.656x |
| udd5 | 300.49 | 390.28 | 120.55 | 1.299x |
| oem | 246.69 | 407.58 | 107.34 | 1.652x |
| loveda | 252.33 | 555.06 | 205.27 | 2.200x |
| vaihingen | 241.60 | 401.65 | 104.04 | 1.662x |
| landcoverai | 233.17 | 389.30 | 99.24 | 1.670x |
| flair1 | 244.11 | 402.71 | 107.56 | 1.650x |

Three fixed complete inputs/domain, three synchronized singleton repeats, serial idle-GPU timing. Includes all views, alias action, H writing, restoration and argmax; excludes loading/text/decoding/masks. Shared-resident peaks are not standalone memory.

## Frozen Advancement Gate

```json
{
  "passed": false,
  "checks": {
    "mean_gain": false,
    "domain_wins": false,
    "worst_protocol_loss": true,
    "above_class_mean": false,
    "above_alias_shuffle": false,
    "above_same_source_mean": true,
    "all_eight_timings": true,
    "above_SigWitness_NoGeometrySupport": false,
    "mean_cost_ratio": false,
    "domain_mean_below1000ms": true
  },
  "domain_wins": 3,
  "mean_gain_pp": -0.010424999999997908,
  "worst_protocol_delta_pp": -0.07819999999999538
}
```

No post-result primary, threshold or domain-route changes and no automatic full20092 promotion. A passed developed pilot still requires vocabulary stress and unchanged full-domain validation. A failed gate rejects this implementation, not every possible conditional alias mechanism.

## Interpretation And Decision

Reject this frozen primary and retain Geometry_PatchOnly2Coupled. Mean46.6450
becomes46.6346 (-0.0104pp), with3/8 main-domain wins. The word-specific primary
exceeds class-mean by only0.0011pp and alias-identity shuffle by0.0027pp;
NoGeometrySupport46.6372 is slightly higher. Its advantage over the same-source
ObservationMean46.3674 does not establish useful admission when it fails the
unchanged baseline and the word/support attribution checks.

VDD vehicle loses0.4325pp, adds100 TP and19793 FP pixels; Potsdam car loses
0.1326pp, adds27 TP and2317 FP. Low vegetation gains0.0855pp with2055 more TP
pixels. UDD5 road gains0.0682pp, but both its TP and FP increase. Even though
fixed-slot attenuation cannot increase its acted-on source class score, other
classes and signed H writeback can change final class competition. These
outcomes do not isolate semantic-source error, action choice and writing as
separate causes; they do reject the combined mechanism on this panel.

All eight serial timing panels complete: primary389.30-555.06ms per domain
average, mean paired latency ratio1.6282. Every domain is below1000ms but the
frozen1.3 ratio gate fails. Shared-resident peaks7131.42-7281.64MiB include all
three models. Observer initialization/text costs4.58-5.03s in concurrent accuracy
workers and are excluded from warmed inference. Neither a single extra RGB
forward nor cached K/V implies negligible support/action overhead.

Ten focused CPU tests and both mask-free real-checkpoint smoke checks passed.
All96 unique complete inputs, three original per-image confusion endpoints,
scored per-image target counts, vocabulary/base-checkpoint identities and the
pinned independent source verify. Every worker exited and GPUs0-7 have no
compute processes. No model-rule change, threshold tuning, word-regime stress,
full20092 promotion or paused-automation change follows this failed result.

The result does not prove conditional alias use is impossible. It shows that
independent low-resolution MAP evidence plus the existing contradiction/budget
rule is not a verified reliability source. Do not respond by merely changing
its sigmoid, temperature, suppression budget or selecting a favorable control.
The preceding alias-path audit leaves a source-versus-writing ambiguity: a
fixed-action, magnitude/locality-matched writeback diagnosis should isolate
that ambiguity before another source is declared necessary or H is discarded.
Any labels used there remain diagnostic only, not selector fitting. The full
three-part accuracy/efficiency research goal remains open.

## Coverage And Competition Outcomes

Confusion rows are targets and columns are predictions. Delta TP counts changes in correct coverage; delta FP counts changes in competitor activation. Negative delta FP is beneficial, but does not establish useful suppression if correct coverage also falls.

| Dataset/protocol | Class | Baseline IoU | Soft IoU | Delta pp | Delta TP pixels | Delta FP pixels | Baseline precision | Soft precision | Baseline recall | Soft recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 20.6232 | 20.5485 | -0.0747 | -41315 | -96341 | 32.3110 | 32.3283 | 36.3108 | 36.0584 |
| vdd/vdd | wall | 46.3090 | 46.4744 | +0.1654 | +8329 | +1663 | 84.1267 | 84.1236 | 50.7428 | 50.9425 |
| vdd/vdd | road | 15.2940 | 15.2630 | -0.0310 | +230 | +19911 | 15.7348 | 15.7015 | 84.5199 | 84.5339 |
| vdd/vdd | vegetation | 45.8052 | 45.9628 | +0.1576 | +44950 | +959 | 96.1118 | 96.1179 | 46.6700 | 46.8322 |
| vdd/vdd | vehicle | 23.8613 | 23.4288 | -0.4325 | +100 | +19793 | 24.5176 | 24.0586 | 89.9132 | 89.9491 |
| vdd/vdd | roof | 85.4306 | 85.4139 | -0.0167 | -450 | +5586 | 85.8504 | 85.8348 | 99.4309 | 99.4292 |
| vdd/vdd | water | 90.6042 | 90.7204 | +0.1162 | +30065 | +6520 | 91.1218 | 91.1060 | 99.3769 | 99.5357 |
| potsdam/potsdam | impervious surface | 75.3698 | 75.2844 | -0.0854 | -2253 | +1205 | 84.9896 | 84.9481 | 86.9432 | 86.8729 |
| potsdam/potsdam | building | 80.4415 | 80.5227 | +0.0812 | +57 | -1296 | 80.4757 | 80.5536 | 99.9471 | 99.9523 |
| potsdam/potsdam | low vegetation | 60.1752 | 60.2607 | +0.0855 | +2055 | +166 | 93.9502 | 93.9480 | 62.6009 | 62.6944 |
| potsdam/potsdam | tree | 57.3938 | 57.5158 | +0.1220 | +2235 | +779 | 77.3966 | 77.3877 | 68.9511 | 69.1345 |
| potsdam/potsdam | car | 31.4528 | 31.3202 | -0.1326 | +27 | +2317 | 31.5636 | 31.4285 | 98.8959 | 98.9120 |
| potsdam/potsdam | clutter | 4.2680 | 4.3112 | +0.0432 | -57 | -5235 | 5.7679 | 5.8554 | 14.0993 | 14.0511 |
| udd5/udd5 | vegetation | 66.2953 | 66.2952 | -0.0001 | +20851 | +31737 | 96.5410 | 96.5084 | 67.9084 | 67.9244 |
| udd5/udd5 | building | 81.2771 | 81.1770 | -0.1001 | +32302 | +297775 | 82.2868 | 82.1713 | 98.5127 | 98.5314 |
| udd5/udd5 | road | 37.2416 | 37.3098 | +0.0682 | +99684 | +133589 | 65.9709 | 65.8402 | 46.0967 | 46.2656 |
| udd5/udd5 | vehicle | 16.9950 | 17.0646 | +0.0696 | +6048 | -30816 | 17.8319 | 17.8997 | 78.3585 | 78.5303 |
| udd5/udd5 | other | 33.5843 | 33.4354 | -0.1489 | -281112 | -310058 | 47.1961 | 47.1936 | 53.7992 | 53.4214 |
| oem/oem | bareland | 0.0000 | 0.0000 | +0.0000 | +0 | +2496 | 0.0000 | 0.0000 | N/A | N/A |
| oem/oem | rangeland | 4.3949 | 4.4206 | +0.0257 | +390 | +1002 | 66.4403 | 65.8481 | 4.4947 | 4.5243 |
| oem/oem | developed space | 21.7107 | 21.7248 | +0.0141 | -1287 | -8678 | 24.6999 | 24.7316 | 64.2081 | 64.1181 |
| oem/oem | road | 26.4384 | 26.4508 | +0.0124 | +276 | +725 | 45.0353 | 44.9917 | 39.0337 | 39.0935 |
| oem/oem | tree | 10.8604 | 10.9186 | +0.0582 | +722 | +871 | 86.7166 | 86.2342 | 11.0442 | 11.1122 |
| oem/oem | water | 18.0795 | 18.0779 | -0.0016 | +8 | +54 | 22.5533 | 22.5468 | 47.6827 | 47.7011 |
| oem/oem | agriculture land | 69.4583 | 69.4284 | -0.0299 | -556 | -353 | 69.6777 | 69.6852 | 99.5487 | 99.4720 |
| oem/oem | building | 51.0820 | 51.1266 | +0.0446 | +2433 | +1897 | 78.2527 | 78.2081 | 59.5335 | 59.6200 |
| loveda/P | building | 88.2412 | 88.2764 | +0.0352 | +44 | -96 | 92.8501 | 92.8767 | 94.6743 | 94.6872 |
| loveda/P | road | 79.8873 | 79.7751 | -0.1122 | +365 | +1825 | 83.5543 | 83.3972 | 94.7923 | 94.8368 |
| loveda/P | water | 62.0850 | 61.9567 | -0.1283 | -1767 | +68 | 98.5503 | 98.5399 | 62.6572 | 62.5308 |
| loveda/P | barren | 70.3607 | 70.5958 | +0.2351 | -118 | -874 | 71.3382 | 71.6213 | 98.0897 | 98.0122 |
| loveda/P | tree | 39.2222 | 39.2678 | +0.0456 | +543 | +99 | 94.6144 | 94.6003 | 40.1179 | 40.1681 |
| loveda/P | farm | 63.4659 | 63.4577 | -0.0082 | -177 | +88 | 64.3610 | 64.3568 | 97.8555 | 97.8459 |
| loveda/D | background | 4.0203 | 3.8709 | -0.1494 | -4475 | -8195 | 49.2011 | 50.0203 | 4.1944 | 4.0266 |
| loveda/D | building | 45.7347 | 45.6556 | -0.0791 | +114 | +1463 | 47.1134 | 47.0211 | 93.9861 | 94.0195 |
| loveda/D | road | 63.0492 | 62.8655 | -0.1837 | +373 | +4183 | 65.4549 | 65.2353 | 94.4918 | 94.5373 |
| loveda/D | water | 58.9731 | 58.8287 | -0.1444 | -1883 | +445 | 90.9392 | 90.8794 | 62.6546 | 62.5199 |
| loveda/D | barren | 27.9609 | 27.8854 | -0.0755 | -125 | +985 | 28.1841 | 28.1142 | 97.2456 | 97.1635 |
| loveda/D | tree | 34.5990 | 34.8286 | +0.2296 | +3031 | +945 | 81.3407 | 81.3005 | 37.5818 | 37.8617 |
| loveda/D | farm | 41.9559 | 41.9266 | -0.0293 | +44 | +3095 | 42.4483 | 42.4178 | 97.3098 | 97.3122 |
| vaihingen/vaihingen | impervious surface | 60.8940 | 60.8488 | -0.0452 | -2018 | -1048 | 84.1446 | 84.1701 | 68.7868 | 68.7120 |
| vaihingen/vaihingen | building | 65.1989 | 65.1584 | -0.0405 | +210 | +2304 | 65.4089 | 65.3638 | 99.5099 | 99.5200 |
| vaihingen/vaihingen | low vegetation | 27.2861 | 27.0772 | -0.2089 | -3456 | -837 | 84.6746 | 84.7109 | 28.7036 | 28.4685 |
| vaihingen/vaihingen | tree | 65.8712 | 65.8205 | -0.0507 | +807 | +2756 | 77.8059 | 77.6893 | 81.1120 | 81.1620 |
| vaihingen/vaihingen | car | 26.4838 | 26.4382 | -0.0456 | +105 | +1177 | 27.2384 | 27.1830 | 90.5299 | 90.6093 |
| landcoverai/landcoverai | background | 82.5166 | 82.7641 | +0.2475 | -192 | -2546 | 93.9248 | 94.2763 | 87.1691 | 87.1428 |
| landcoverai/landcoverai | building | N/A | N/A | N/A | +0 | +0 | N/A | N/A | N/A | N/A |
| landcoverai/landcoverai | woodland | 91.5012 | 91.6887 | +0.1875 | +2259 | +313 | 95.2495 | 95.2307 | 95.8766 | 96.1017 |
| landcoverai/landcoverai | water | 91.7035 | 91.7433 | +0.0398 | +0 | -158 | 91.7116 | 91.7514 | 99.9904 | 99.9904 |
| landcoverai/landcoverai | road | 39.2550 | 39.0807 | -0.1743 | +27 | +297 | 46.0574 | 45.7789 | 72.6614 | 72.7592 |
| flair1/flair1 | building | 49.1283 | 49.4376 | +0.3093 | +814 | +428 | 51.3696 | 51.4634 | 91.8434 | 92.6246 |
| flair1/flair1 | pervious surface | 17.8467 | 17.8078 | -0.0389 | -11 | +61 | 64.2446 | 63.9694 | 19.8148 | 19.7931 |
| flair1/flair1 | impervious surface | 66.4519 | 66.5740 | +0.1221 | -64 | -886 | 74.8732 | 75.0429 | 85.5244 | 85.5053 |
| flair1/flair1 | bare soil | 21.3471 | 21.1039 | -0.2432 | +0 | +907 | 21.3761 | 21.1323 | 99.3672 | 99.3672 |
| flair1/flair1 | water | 94.9931 | 94.9789 | -0.0142 | -8 | +0 | 99.7998 | 99.7998 | 95.1745 | 95.1602 |
| flair1/flair1 | coniferous | 5.9298 | 5.9112 | -0.0186 | +0 | +18 | 7.4090 | 7.3799 | 22.8997 | 22.8997 |
| flair1/flair1 | deciduous | 64.3305 | 64.2729 | -0.0576 | +90 | +395 | 81.0241 | 80.8901 | 75.7421 | 75.7793 |
| flair1/flair1 | brushwood | 3.3685 | 3.3373 | -0.0312 | -52 | +26 | 6.8848 | 6.8243 | 6.1874 | 6.1311 |
| flair1/flair1 | vineyard | 51.9276 | 51.8526 | -0.0750 | -1 | +320 | 52.0611 | 51.9859 | 99.5087 | 99.5079 |
| flair1/flair1 | herbaceous vegetation | 37.8267 | 37.6508 | -0.1759 | -1129 | -5 | 86.6034 | 86.5507 | 40.1776 | 39.9905 |
| flair1/flair1 | agricultural land | 38.0928 | 38.1433 | +0.0505 | -24 | -749 | 83.6119 | 83.8766 | 41.1664 | 41.1614 |
| flair1/flair1 | plowed land | 0.0000 | 0.0000 | +0.0000 | +0 | -130 | 0.0000 | 0.0000 | N/A | N/A |

## Independent Source And Setup Cost

Pinned SigLIP2-base-patch16-256 source identity and both real-checkpoint MAP descriptor replays verify. Each smoke directly instruments one observer forward per inference and checks unchanged frozen semantic-head parameters. Per-image inference enforces the global128-query cap and frozen source. The observer is additional pretrained information, not an original encoder or semantic-correctness guarantee.

| Domain | Observer initialization/text seconds | Peak shared-resident primary MiB |
| --- | ---: | ---: |
| vdd | 4.58 | 7281.64 |
| potsdam | 4.80 | 7137.12 |
| udd5 | 4.84 | 7165.72 |
| oem | 4.90 | 7145.33 |
| loveda | 4.93 | 7169.35 |
| vaihingen | 4.84 | 7131.42 |
| landcoverai | 4.88 | 7131.42 |
| flair1 | 5.03 | 7166.10 |

Setup is recorded during concurrent accuracy workers, not isolated cold-start timing. Warmed inference excludes initialization/text and includes all three resident models. The peak is not standalone deployment memory.
