# Shared-Rival Soft20: Full Eight-Dataset Results

Frozen original20 words/class, Geometry depth2, wide VIP observer and reconstruction. No fine visual forwards. A stable leave-class-out LME summarizes all competing classes once; positive wide margin and simultaneous native/Geometry contradiction attenuate noncanonical aliases. Normalized weights enter outside the exponent. The risk is an untrained heuristic, not a correctness probability. Hard control keeps weights>=0.5; shuffle retains each class weight spectrum and canonical protection. Rule was fixed before these masks, but motivated by developed experiments: exploratory full validation.

| Dataset/protocol | Images | Finite VIP All20 | Geometry | No admission | Soft primary | Hard control | Shuffled weights | Soft - VIP | Soft - no admission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 1669 | 56.3110 | 64.9557 | 62.7149 | 62.6510 | 62.7350 | 62.7679 | +6.3400 | -0.0639 |
| loveda/D | 1669 | 36.1457 | 42.6589 | 40.2519 | 40.2095 | 40.2877 | 40.2316 | +4.0638 | -0.0424 |
| udd5/udd5 | 40 | 41.4336 | 50.5553 | 49.2117 | 49.3542 | 49.3240 | 49.1458 | +7.9206 | +0.1425 |
| oem/oem | 384 | 33.2064 | 44.6097 | 37.8977 | 37.8676 | 37.8981 | 37.8896 | +4.6612 | -0.0301 |
| vdd/vdd | 80 | 51.4827 | 38.8511 | 55.4446 | 55.4932 | 55.5124 | 55.4046 | +4.0105 | +0.0486 |
| potsdam/potsdam | 504 | 42.3028 | 40.6892 | 43.7334 | 43.6900 | 43.7340 | 43.6840 | +1.3872 | -0.0434 |
| vaihingen/vaihingen | 113 | 47.3620 | 48.8009 | 51.4079 | 51.3649 | 51.4167 | 51.3631 | +4.0029 | -0.0430 |
| landcoverai/landcoverai | 1602 | 54.9420 | 59.2340 | 63.7049 | 63.6909 | 63.7101 | 63.7120 | +8.7489 | -0.0140 |
| flair1/flair1 | 15700 | 35.2977 | 43.8071 | 42.5381 | 42.5105 | 42.5387 | 42.5771 | +7.2128 | -0.0276 |
| Eight-domain mean, LoveDA D once | 20092 | 42.7716 | 46.1508 | 48.0238 | 48.0226 | 48.0527 | 48.0010 | +5.2510 | -0.0012 |

Soft gain versus no admission: -0.0012pp; versus shuffled weights: +0.0216pp; versus retained full slow hard20: -1.4141pp. The slow fine-only redistribution was measured on a different partial panel and is not assigned a full20092 score.

VIP uses the already completed finite-row All20 evaluations on seven domains. The missing corrected-input Vaihingen All20 is filled once. Official settings/background thresholds remain in VIP; our coupling uses its frozen protocol. These are same-word whole-method comparisons, not identical-readout ablations or published VIP benchmark numbers. LandCover.ai substitutes for unlabeled iSAID.

## Matched Complete-Image Timing

| Dataset | VIP All20 ms | No admission ms | Soft ms | Hard ms | Soft overhead | Soft/VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda | 202.66 | 401.12 | 886.89 | 887.89 | +121.10% | 4.376x |
| udd5 | 120.45 | 4531.00 | 5901.43 | 5891.56 | +30.25% | 48.994x |
| oem | 101.30 | 378.45 | 619.34 | 616.17 | +63.65% | 6.114x |
| vdd | 130.14 | 4926.15 | 6430.59 | 6436.02 | +30.54% | 49.414x |
| potsdam | 101.42 | 375.79 | 617.94 | 618.88 | +64.44% | 6.093x |
| vaihingen | 98.56 | 371.09 | 610.74 | 611.64 | +64.58% | 6.197x |
| landcoverai | 93.79 | 125.96 | 154.51 | 154.07 | +22.66% | 1.647x |
| flair1 | 102.19 | 137.31 | 166.73 | 166.18 | +21.43% | 1.632x |

Three fixed COMPLETE inputs/domain (first, middle, last of the frozen sequence); three warmed CUDA-synchronized rotated singleton repeats/image, serial on GPU7 after full evaluation. Includes method views, every local window, probability stitching and argmax. Excludes initialization/text encoding/image decoding/masks. This is a24-image timing panel, not full-domain average throughput. No resizing of our original full evaluation inputs to meet the100-1000ms target. Shared-resident allocated peaks are in timing.json and are not standalone model memory.

## Accuracy And Cost Interpretation

The frozen soft primary improves over no admission in only UDD5 and VDD; the
other six domain means decrease. Its eight-domain mean48.0226 is effectively
unchanged from no admission48.0238, and the0.0216pp mean advantage over shuffled
weights does not establish a useful or statistically significant screening gain.
The5.2510pp advantage over finite VIP All20 comes from the coupled subject, not
an added average gain from this soft rule. It also loses1.4141pp to the retained
full slow Hard20 model. Do not promote the hard control after inspecting results.

The input-resolution strategy is the main reason that cancelling fine views
does not make every complete image fast. VIP first resizes the long edge to448,
then uses336 crops with112 stride. Our subject keeps original-resolution512
Geometry windows with128 overlap. VDD3000x4000 therefore has88 local Geometry
windows, while VIP has two336 windows after resizing to336x448. The same original
input, fixed20 aliases and output size do not imply the same visual workload.

| Timing domain | Original input H x W | Geometry windows per input |
| --- | --- | ---: |
| LoveDA | 1024x1024 | 9 |
| UDD5 | 2160x4096 / 3000x4000 / 3000x4000 | 66 / 88 / 88 |
| OEM | 1000x1000 / 1024x1024 / 1024x1024 | 9 / 9 / 9 |
| VDD | 3000x4000 | 88 |
| Potsdam | 1000x1000 | 9 |
| Vaihingen | 1000x1000 | 9 |
| LandCover.ai | 512x512 | 1 |
| FLAIR-1 | 512x512 | 1 |

LoveDA timing computes both P and D readouts together for each method. Each
domain time is the mean of its three per-image medians, not its full validation
mean. The model still uses the original wide observer in addition to the listed
Geometry windows; fine visual forwards are zero.

For VDD, no admission already costs4926.15ms; soft adds1504.45ms (+30.54%).
Removing the new weights alone cannot meet a1000ms budget. For Potsdam, no
admission costs375.79ms and soft adds242.15ms (+64.44%), without improving mIoU.
Both512-input domains cost154.51-166.73ms with soft, but high-resolution VDD and
UDD5 cost6430.59ms and5901.43ms. This is not a universally sub-second model.

The earlier939.60ms without-fine result was one512x683 ADE image with a different
class/text profile, recorded in SINGLE_IMAGE_SPEED_VS_VIP_20261004.md. It was not
a whole-image VDD or eight-domain speed guarantee. Earlier equivalent-device
measurements already recorded VDD5738.1ms with88 windows; the high-resolution
bottleneck was known before this suite.

Equivalent execution changes such as batching existing local windows, caching
layout/profile metadata, and removing diagnostic GPU synchronizations are
possible next optimizations. Their gains and prediction equivalence are not
verified here. If these are insufficient, a hard per-image visual-work budget
requires changing the input/window strategy and revalidating small-object
accuracy; no lossless1000ms claim follows from this experiment.

## Per-Class Outcomes

| Protocol | Class | VIP All20 IoU | No admission IoU | Soft IoU | Soft precision | Soft recall | Soft predicted area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | building | 81.4273 | 88.7804 | 88.7605 | 92.4904 | 95.6541 | 11.6622 |
| loveda/P | road | 63.5157 | 68.4225 | 68.3958 | 80.1035 | 82.3931 | 7.5172 |
| loveda/P | water | 65.2910 | 70.2657 | 70.3325 | 94.4468 | 73.3664 | 14.2349 |
| loveda/P | barren | 34.6101 | 41.2310 | 41.1390 | 60.6495 | 56.1179 | 6.3198 |
| loveda/P | tree | 21.2621 | 33.6546 | 33.3535 | 90.3992 | 34.5784 | 4.4120 |
| loveda/P | farm | 71.7595 | 73.9354 | 73.9245 | 76.5392 | 95.5831 | 55.8539 |
| loveda/D | background | 11.9386 | 13.2730 | 13.1988 | 61.7762 | 14.3725 | 8.3988 |
| loveda/D | building | 47.6742 | 52.1072 | 52.0637 | 53.6514 | 94.6218 | 12.7082 |
| loveda/D | road | 50.1601 | 51.6805 | 51.6682 | 58.5678 | 81.4331 | 6.4932 |
| loveda/D | water | 55.8824 | 59.3552 | 59.3715 | 79.2593 | 70.2925 | 10.3849 |
| loveda/D | barren | 22.9091 | 29.8529 | 29.8463 | 45.9465 | 45.9970 | 4.3692 |
| loveda/D | tree | 16.4150 | 26.8155 | 26.6504 | 66.5086 | 30.7813 | 3.4112 |
| loveda/D | farm | 48.0404 | 48.6788 | 48.6673 | 49.9863 | 94.8568 | 54.2344 |
| udd5/udd5 | vegetation | 64.3148 | 68.3847 | 68.3339 | 97.4778 | 69.5639 | 21.1372 |
| udd5/udd5 | building | 81.1240 | 82.5264 | 82.5379 | 83.6801 | 98.3732 | 46.1519 |
| udd5/udd5 | road | 19.3962 | 39.8083 | 39.9653 | 66.5887 | 49.9896 | 10.0673 |
| udd5/udd5 | vehicle | 11.8369 | 20.4213 | 20.9299 | 21.4437 | 89.7275 | 3.3495 |
| udd5/udd5 | other | 30.4958 | 34.9177 | 35.0039 | 48.6547 | 55.5085 | 19.2941 |
| oem/oem | bareland | 6.9087 | 9.9789 | 9.9333 | 10.1935 | 79.5563 | 10.0725 |
| oem/oem | rangeland | 9.0957 | 17.9245 | 17.8451 | 67.3564 | 19.5345 | 6.1061 |
| oem/oem | developed space | 22.7255 | 28.2491 | 28.1972 | 33.2060 | 65.1485 | 38.8646 |
| oem/oem | road | 30.3281 | 34.4170 | 34.3982 | 50.5898 | 51.8016 | 7.2323 |
| oem/oem | tree | 31.4655 | 37.0260 | 36.9726 | 89.6303 | 38.6248 | 8.0969 |
| oem/oem | water | 61.2792 | 68.3274 | 68.3011 | 78.1210 | 84.4567 | 2.5519 |
| oem/oem | agriculture land | 61.9014 | 61.8142 | 61.7309 | 65.7708 | 90.9503 | 16.3535 |
| oem/oem | building | 41.9472 | 45.4447 | 45.5628 | 83.2855 | 50.1484 | 10.7222 |
| vdd/vdd | other | 36.5655 | 35.2730 | 35.4141 | 53.1973 | 51.4420 | 18.1905 |
| vdd/vdd | wall | 29.0582 | 48.3011 | 48.3571 | 74.6588 | 57.8529 | 2.5072 |
| vdd/vdd | road | 45.0432 | 41.8864 | 41.8065 | 42.7545 | 94.9637 | 12.1923 |
| vdd/vdd | vegetation | 50.2515 | 66.5737 | 66.6633 | 97.1406 | 67.9976 | 25.0001 |
| vdd/vdd | vehicle | 29.6526 | 26.2841 | 26.3114 | 27.8934 | 82.2664 | 1.5355 |
| vdd/vdd | roof | 85.5153 | 83.6516 | 83.6761 | 84.5840 | 98.7336 | 24.5618 |
| vdd/vdd | water | 84.2928 | 86.1421 | 86.2240 | 90.2144 | 95.1204 | 16.0127 |
| potsdam/potsdam | impervious surface | 61.2488 | 61.4711 | 61.3570 | 72.7054 | 79.7199 | 34.5074 |
| potsdam/potsdam | building | 77.2218 | 79.0047 | 79.0066 | 81.5193 | 96.2451 | 28.2403 |
| potsdam/potsdam | low vegetation | 28.5736 | 34.1710 | 34.1299 | 93.0171 | 35.0274 | 7.9113 |
| potsdam/potsdam | tree | 50.9752 | 54.6857 | 54.7287 | 83.6509 | 61.2839 | 12.4847 |
| potsdam/potsdam | car | 30.4756 | 26.7830 | 26.6547 | 26.7098 | 99.2328 | 7.3001 |
| potsdam/potsdam | clutter | 5.3220 | 6.2848 | 6.2632 | 8.7278 | 18.1536 | 9.5562 |
| vaihingen/vaihingen | impervious surface | 56.9979 | 61.3360 | 61.2549 | 79.2487 | 72.9569 | 26.1949 |
| vaihingen/vaihingen | building | 64.9474 | 68.0397 | 68.0060 | 68.6211 | 98.6990 | 37.8904 |
| vaihingen/vaihingen | low vegetation | 30.2954 | 33.2475 | 33.1817 | 88.4537 | 34.6840 | 8.4025 |
| vaihingen/vaihingen | tree | 62.0184 | 69.1603 | 69.1505 | 80.9953 | 82.5436 | 22.8979 |
| vaihingen/vaihingen | car | 22.5506 | 25.2560 | 25.2312 | 25.8476 | 91.3648 | 4.6143 |
| landcoverai/landcoverai | background | 74.1810 | 82.5407 | 82.5181 | 87.1228 | 93.9806 | 64.7476 |
| landcoverai/landcoverai | building | 44.5106 | 47.8814 | 47.8759 | 49.3643 | 94.0754 | 1.7440 |
| landcoverai/landcoverai | woodland | 52.0468 | 73.5558 | 73.5000 | 95.6175 | 76.0624 | 25.1476 |
| landcoverai/landcoverai | water | 64.8284 | 76.3963 | 76.4114 | 88.3087 | 85.0113 | 5.7016 |
| landcoverai/landcoverai | road | 39.1431 | 38.1505 | 38.1494 | 43.4651 | 75.7244 | 2.6593 |
| flair1/flair1 | building | 60.2824 | 61.5881 | 61.5362 | 63.5489 | 95.1050 | 12.9109 |
| flair1/flair1 | pervious surface | 32.1094 | 34.8700 | 34.8516 | 69.8716 | 41.0154 | 4.3235 |
| flair1/flair1 | impervious surface | 56.6589 | 59.1593 | 59.1300 | 69.7258 | 79.5545 | 17.1418 |
| flair1/flair1 | bare soil | 23.1537 | 36.9224 | 36.8814 | 38.6492 | 88.9666 | 10.0624 |
| flair1/flair1 | water | 74.1419 | 80.1186 | 80.1481 | 87.5974 | 90.4074 | 6.1935 |
| flair1/flair1 | coniferous | 14.4692 | 15.5480 | 15.5778 | 45.7200 | 19.1126 | 1.0025 |
| flair1/flair1 | deciduous | 48.7353 | 55.9329 | 55.9525 | 77.0492 | 67.1431 | 12.1557 |
| flair1/flair1 | brushwood | 11.1440 | 26.8597 | 26.8455 | 44.0016 | 40.7769 | 6.4189 |
| flair1/flair1 | vineyard | 57.6837 | 63.9027 | 63.9008 | 78.4183 | 77.5366 | 3.8367 |
| flair1/flair1 | herbaceous vegetation | 18.8786 | 31.6081 | 31.5795 | 79.5596 | 34.3680 | 9.6034 |
| flair1/flair1 | agricultural land | 16.1240 | 23.5354 | 23.4237 | 37.1696 | 38.7775 | 7.2678 |
| flair1/flair1 | plowed land | 10.1912 | 20.4121 | 20.2991 | 21.0720 | 84.6965 | 9.0831 |

Every downloaded per-image confusion reconstructs its full aggregate, scored target counts match VIP, original Geometry/no-admission full confusions match history, and vocabulary/checkpoint/global coverage identities are verified. No control is promoted or model rule changed after results.
