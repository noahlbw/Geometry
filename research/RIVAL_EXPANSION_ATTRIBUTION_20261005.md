# Expansion attribution with fixed class-field interventions

Same64 developed512 windows; zero image/text network forwards. Not a new model or independent validation. All intervention scores persist before masks; raw/centered20/40 predictions and historical per-image confusions match exactly. Removing the common log2 count term changes zero predictions.

Class mean is subtracted at each query before restoration. Restore only the own class20 field (rivals remain40), or restore all rivals20 (own remains40). This is a fixed final-score intervention, not a hybrid-vocabulary rerun or an individual-word correctness test. The unchanged linear H permits the same column intervention on admitted observations with the local anchor fixed. Argmax/IoU interactions make the two gains nonadditive; no class-wise oracle model is selected.

## Findings And Next Candidate

The pure common count-offset hypothesis is ruled out for these equal-count,
fixed-readout endpoints: every removed-offset prediction is unchanged. This does
not rule out vocabulary-dependent salience, alias response, admission or rival
coordination effects. Only the additive log K component is a common gauge.

The observed pathways differ. VDD wall40 IoU37.2223 rises to54.3991 when its own
centered20 score field is restored, but falls to35.1650 when only rivals restore.
Potsdam car40 IoU18.6362 instead rises to26.6523 when rivals restore; restoring its
own field yields17.0520. OEM building40 IoU7.6778 rises to42.9850/44.1179 under
own/rival restorations respectively. FLAIR herbaceous vegetation40 IoU38.2996
rises to61.1050 under own restoration but falls to34.4752 under rival restoration.
These are defined centered SCORE-field interventions, not causal identification
of which parent class's words are wrong: coupled fields already mix pair actions,
and the gauge itself depends on all classes. The two effects are not additive.

Comparable patterns exist without admission: VDD wall20/40 IoU59.6816/35.7086,
Potsdam car24.5378/18.2124 and OEM building47.0287/6.0097. Thus the expansion losses
were not introduced solely by the projected screening writer. On the other hand,
VDD vehicle improves38.2597->55.6585 with100% recall, reflecting reduced false
positives rather than more target coverage. Globally rejecting all added words
would remove this benefit. These developed windows remain insufficient to choose
word lists or establish all-dataset effects.

The next frozen soft candidate keeps original hard rejection and weights every
surviving alias by its existing fine alias/rival response responsibility. Unlike
old1-risk attenuation, this also reweights weak words on old zero-risk support.
Its purpose is to test coverage/competition redistribution, not certify joint
observer errors as correct. Canonical protection, original fine-target projection,
posterior and H remain. Class-mass and within-support weight-spectrum identity
controls are predeclared. No restored field or audit label becomes a model rule.
Protocol: RIVAL_FINE_SUPPORT_PROTOCOL_20261005.md.

## Projected Candidate Class Outcomes

| Dataset/protocol | Class | Target pixels | All20 IoU | All40 IoU | Restore own20 IoU | Restore rivals20 IoU | Restore own gain pp | Restore rivals gain pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | other | 1070714 | 55.0600 | 58.8770 | 57.3359 | 57.4570 | -1.5411 | -1.4200 |
| vdd/vdd | wall | 166833 | 53.1449 | 37.2223 | 54.3991 | 35.1650 | +17.1768 | -2.0573 |
| vdd/vdd | road | 69605 | 21.6618 | 31.7393 | 32.9681 | 22.5377 | +1.2288 | -9.2015 |
| vdd/vdd | vegetation | 483080 | 54.2677 | 65.0435 | 63.3429 | 60.7386 | -1.7006 | -4.3049 |
| vdd/vdd | vehicle | 7510 | 38.2597 | 55.6585 | 40.7002 | 52.7314 | -14.9583 | -2.9271 |
| vdd/vdd | roof | 126362 | 89.8815 | 90.4365 | 89.1382 | 92.1018 | -1.2983 | +1.6653 |
| vdd/vdd | water | 173048 | 66.5525 | 63.3479 | 64.6939 | 62.4536 | +1.3460 | -0.8943 |
| potsdam/potsdam | impervious surface | 857241 | 66.3665 | 57.9011 | 68.3244 | 54.5591 | +10.4234 | -3.3420 |
| potsdam/potsdam | building | 338121 | 72.8663 | 79.2487 | 75.4589 | 75.2808 | -3.7898 | -3.9678 |
| potsdam/potsdam | low vegetation | 363664 | 21.7324 | 26.0957 | 17.7623 | 29.0880 | -8.3333 | +2.9923 |
| potsdam/potsdam | tree | 370013 | 58.0561 | 55.7699 | 57.7928 | 56.1142 | +2.0229 | +0.3443 |
| potsdam/potsdam | car | 75196 | 25.1793 | 18.6362 | 17.0520 | 26.6523 | -1.5841 | +8.0162 |
| potsdam/potsdam | clutter | 92917 | 2.5381 | 3.1399 | 3.0891 | 2.6163 | -0.0508 | -0.5237 |
| udd5/udd5 | vegetation | 64744 | 68.5613 | 71.7674 | 69.3222 | 70.7470 | -2.4453 | -1.0204 |
| udd5/udd5 | building | 1508385 | 85.8800 | 84.5009 | 85.7504 | 84.9496 | +1.2495 | +0.4486 |
| udd5/udd5 | road | 89116 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| udd5/udd5 | vehicle | 8808 | 6.6312 | 9.3786 | 6.4907 | 9.3035 | -2.8878 | -0.0751 |
| udd5/udd5 | other | 426099 | 12.0146 | 16.9453 | 16.0264 | 12.2212 | -0.9190 | -4.7242 |
| oem/oem | bareland | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| oem/oem | rangeland | 297086 | 49.8592 | 47.0310 | 48.5722 | 48.7137 | +1.5412 | +1.6826 |
| oem/oem | developed space | 400255 | 23.4870 | 20.0355 | 25.4226 | 20.4414 | +5.3871 | +0.4059 |
| oem/oem | road | 128408 | 55.4142 | 56.3151 | 54.6047 | 55.9940 | -1.7104 | -0.3211 |
| oem/oem | tree | 539563 | 57.7324 | 53.6803 | 56.1531 | 55.4677 | +2.4728 | +1.7874 |
| oem/oem | water | 503 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| oem/oem | agriculture land | 433260 | 82.5457 | 78.3091 | 82.5202 | 79.0740 | +4.2112 | +0.7649 |
| oem/oem | building | 243289 | 54.5491 | 7.6778 | 42.9850 | 44.1179 | +35.3072 | +36.4401 |
| loveda/P | building | 307 | 48.9292 | 66.5944 | 55.6159 | 47.9789 | -10.9784 | -18.6154 |
| loveda/P | road | 96423 | 60.1611 | 58.3656 | 57.9559 | 60.4235 | -0.4098 | +2.0579 |
| loveda/P | water | 170486 | 62.9242 | 63.5146 | 62.2277 | 63.9664 | -1.2868 | +0.4519 |
| loveda/P | barren | 128461 | 1.6724 | 5.4887 | 2.2423 | 3.2501 | -3.2464 | -2.2386 |
| loveda/P | tree | 190765 | 56.5911 | 55.0111 | 57.3894 | 54.3334 | +2.3784 | -0.6776 |
| loveda/P | farm | 572179 | 85.5838 | 87.2460 | 85.4800 | 87.0379 | -1.7661 | -0.2081 |
| loveda/D | background | 938531 | 30.6934 | 36.6338 | 34.8182 | 32.4941 | -1.8156 | -4.1397 |
| loveda/D | building | 307 | 31.5625 | 28.5714 | 27.8332 | 29.2159 | -0.7382 | +0.6445 |
| loveda/D | road | 96423 | 49.2751 | 48.0196 | 47.7277 | 49.6209 | -0.2920 | +1.6013 |
| loveda/D | water | 170486 | 55.0011 | 57.5468 | 55.1203 | 56.7375 | -2.4265 | -0.8093 |
| loveda/D | barren | 128461 | 1.6177 | 2.1425 | 1.7826 | 1.8914 | -0.3599 | -0.2511 |
| loveda/D | tree | 190765 | 48.4719 | 43.1800 | 48.3629 | 43.6092 | +5.1829 | +0.4291 |
| loveda/D | farm | 572179 | 46.5944 | 47.1749 | 45.6265 | 48.5089 | -1.5483 | +1.3341 |
| vaihingen/vaihingen | impervious surface | 572872 | 59.3766 | 47.0780 | 54.7102 | 55.0330 | +7.6322 | +7.9551 |
| vaihingen/vaihingen | building | 432655 | 67.4674 | 70.3103 | 67.4267 | 71.5395 | -2.8836 | +1.2292 |
| vaihingen/vaihingen | low vegetation | 620087 | 45.7742 | 47.7033 | 48.2892 | 44.7968 | +0.5859 | -2.9064 |
| vaihingen/vaihingen | tree | 432823 | 67.6134 | 67.4349 | 67.4839 | 67.5488 | +0.0491 | +0.1139 |
| vaihingen/vaihingen | car | 38715 | 26.2980 | 14.8806 | 19.0653 | 21.3271 | +4.1848 | +6.4465 |
| landcoverai/landcoverai | background | 1421030 | 88.2370 | 87.9126 | 88.8921 | 87.5746 | +0.9795 | -0.3380 |
| landcoverai/landcoverai | building | 31773 | 44.5937 | 44.4773 | 42.6393 | 46.4062 | -1.8379 | +1.9290 |
| landcoverai/landcoverai | woodland | 448933 | 81.5413 | 85.7366 | 84.0258 | 84.4186 | -1.7108 | -1.3180 |
| landcoverai/landcoverai | water | 173462 | 97.6079 | 97.0639 | 97.1781 | 97.5509 | +0.1142 | +0.4869 |
| landcoverai/landcoverai | road | 21954 | 26.9379 | 21.3216 | 22.7597 | 25.2344 | +1.4380 | +3.9128 |
| flair1/flair1 | building | 150504 | 57.4041 | 58.7902 | 52.4022 | 61.6122 | -6.3881 | +2.8220 |
| flair1/flair1 | pervious surface | 359720 | 46.8072 | 39.9642 | 53.3721 | 22.3437 | +13.4079 | -17.6205 |
| flair1/flair1 | impervious surface | 343424 | 52.9006 | 58.4585 | 49.8512 | 54.5873 | -8.6074 | -3.8712 |
| flair1/flair1 | bare soil | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| flair1/flair1 | water | 92992 | 64.7837 | 67.7312 | 65.2763 | 65.4646 | -2.4549 | -2.2665 |
| flair1/flair1 | coniferous | 11784 | 42.7265 | 43.5182 | 41.5688 | 44.3899 | -1.9495 | +0.8717 |
| flair1/flair1 | deciduous | 356824 | 60.9048 | 57.8800 | 60.9053 | 58.2541 | +3.0254 | +0.3742 |
| flair1/flair1 | brushwood | 103104 | 15.6306 | 14.7428 | 12.5986 | 18.6988 | -2.1443 | +3.9560 |
| flair1/flair1 | vineyard | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| flair1/flair1 | herbaceous vegetation | 671552 | 57.6203 | 38.2996 | 61.1050 | 34.4752 | +22.8054 | -3.8245 |
| flair1/flair1 | agricultural land | 6392 | 13.4305 | 7.4334 | 5.7450 | 11.1987 | -1.6885 | +3.7653 |
| flair1/flair1 | plowed land | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

Exact precision/recall/area, all three reader controls, all restoration mIoUs and per-image confusions are under research/rival_expansion_attribution_20261005/. These interventions establish effects of the specified centered class fields, not globally good/bad word lists. Numerical class-count offset cannot explain the observed equal-count expansion losses; word response/profile/conditional admission and competition still can. Prior schemes remain preserved. No weights or selectors are chosen from this audit.
