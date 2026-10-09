# Bounded-Resolution Coupling: Full-Image Accuracy

Frozen signature `geometry-shared-rival-soft-bounded896-v1-20261005`. Completed 8/8 domains; only completed full-domain evaluations appear below. No partial-shard mIoU is used as a full score.

Geometry is independently resized to long edge896, with at most four real512 crops/384 stride and32x32 tokens. Original RGB is independently resized to long edge448 for at most four336 wide crops/112 stride and21x21 tokens. Frozen Geometry depth2, fixed20 words/class, local RS/wide ImageNet text banks, SharedRivalSoft, reconstruction and Hann probability stitching. Zero native/fine encodings; restore probabilities to original-size masks before argmax. This changes only the visual sampling protocol, not the frozen heads/alias rule.

| Dataset/protocol | Images | Finite VIP All20 | New Geometry | New no admission | New soft primary | Soft - VIP | Soft - no admission | Old native soft | New - old native soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 1669 | 56.3110 | 66.5715 | 62.5150 | 62.4389 | +6.1279 | -0.0761 | 62.6510 | -0.2121 |
| loveda/D | 1669 | 36.1457 | 42.8650 | 40.0210 | 39.9584 | +3.8127 | -0.0626 | 40.2095 | -0.2511 |
| udd5/udd5 | 40 | 41.4336 | 47.1219 | 47.0096 | 47.0188 | +5.5852 | +0.0092 | 49.3542 | -2.3354 |
| oem/oem | 384 | 33.2064 | 43.9751 | 37.2104 | 37.1865 | +3.9801 | -0.0239 | 37.8676 | -0.6811 |
| vdd/vdd | 80 | 51.4827 | 46.6010 | 53.8090 | 53.7886 | +2.3059 | -0.0204 | 55.4932 | -1.7046 |
| potsdam/potsdam | 504 | 42.3028 | 40.3779 | 43.4301 | 43.3830 | +1.0802 | -0.0471 | 43.6900 | -0.3070 |
| vaihingen/vaihingen | 113 | 47.3620 | 48.2368 | 50.7949 | 50.7477 | +3.3857 | -0.0472 | 51.3649 | -0.6172 |
| landcoverai/landcoverai | 1602 | 54.9420 | 57.0988 | 63.6099 | 63.5915 | +8.6495 | -0.0184 | 63.6909 | -0.0994 |
| flair1/flair1 | 15700 | 35.2977 | 40.5208 | 42.5038 | 42.4808 | +7.1831 | -0.0230 | 42.5105 | -0.0297 |
| Eight-domain mean (LoveDA D once) | 20092 | 42.7716 | 45.8497 | 47.2986 | 47.2694 | +4.4978 | -0.0292 | 48.0226 | -0.7532 |

VIP comparators are the completed finite-row-repaired All20 evaluations, including corrected-input Vaihingen. Original official short words, historical NaN-contaminated VIP scores and distilled words are not substituted. Matched words, masks and checkpoints do not imply identical view resolution, templates, visual readout, stitching or background thresholds. These are local method evaluations, not published VIP benchmark numbers. LandCover.ai substitutes for unlabeled iSAID. Method development used prior validation windows; this is exploratory, not independent validation.

## Admission Controls

| Protocol | No admission | Soft | Hard | Shuffled | Mean noncanonical risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda/P | 62.5150 | 62.4389 | 62.5247 | 62.5539 | 0.007773 |
| loveda/D | 40.0210 | 39.9584 | 40.0288 | 40.0155 | 0.007423 |
| udd5/udd5 | 47.0096 | 47.0188 | 47.0197 | 46.9949 | 0.007374 |
| oem/oem | 37.2104 | 37.1865 | 37.2101 | 37.2002 | 0.008292 |
| vdd/vdd | 53.8090 | 53.7886 | 53.8099 | 53.7950 | 0.007003 |
| potsdam/potsdam | 43.4301 | 43.3830 | 43.4290 | 43.3804 | 0.009894 |
| vaihingen/vaihingen | 50.7949 | 50.7477 | 50.8010 | 50.7522 | 0.011310 |
| landcoverai/landcoverai | 63.6099 | 63.5915 | 63.6058 | 63.6081 | 0.004524 |
| flair1/flair1 | 42.5038 | 42.4808 | 42.5061 | 42.5543 | 0.007203 |

Primary beats VIP on 8/8 domains; mean delta +4.4978pp. Soft versus matched no-admission mean delta -0.0292pp; versus shuffled weights -0.0182pp. No post-result rule/parameter change or control promotion was made.

## Foreground And Non-Residual IoU

| Protocol (excluded class) | VIP | No admission | Soft |
| --- | ---: | ---: | ---: |
| loveda/D (background) | 40.1802 | 44.5342 | 44.4862 |
| udd5/udd5 (other) | 44.1680 | 50.3444 | 50.3579 |
| vdd/vdd (other) | 53.9689 | 57.0078 | 56.9746 |
| landcoverai/landcoverai (background) | 50.1322 | 58.9920 | 58.9743 |

## Per-Class Outcomes

| Protocol | Class | VIP IoU | Old native soft IoU | New soft IoU | New - VIP | Precision | Recall | Predicted area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | building | 81.4273 | 88.7605 | 88.4354 | 7.0081 | 92.1863 | 95.6014 | 11.6942 |
| loveda/P | road | 63.5157 | 68.3958 | 67.8608 | 4.3451 | 79.7652 | 81.9722 | 7.5105 |
| loveda/P | water | 65.2910 | 70.3325 | 70.6713 | 5.3803 | 94.4189 | 73.7522 | 14.3140 |
| loveda/P | barren | 34.6101 | 41.1390 | 40.9259 | 6.3158 | 60.9964 | 55.4323 | 6.2071 |
| loveda/P | tree | 21.2621 | 33.3535 | 32.8514 | 11.5893 | 90.8459 | 33.9761 | 4.3139 |
| loveda/P | farm | 71.7595 | 73.9245 | 73.8888 | 2.1293 | 76.4532 | 95.6576 | 55.9604 |
| loveda/D | background | 11.9386 | 13.1988 | 12.7912 | 0.8526 | 61.6949 | 13.8947 | 8.1303 |
| loveda/D | building | 47.6742 | 52.0637 | 51.6415 | 3.9673 | 53.2221 | 94.5617 | 12.8026 |
| loveda/D | road | 50.1601 | 51.6682 | 51.2604 | 1.1003 | 58.2503 | 81.0311 | 6.4964 |
| loveda/D | water | 55.8824 | 59.3715 | 59.2689 | 3.3865 | 78.3974 | 70.8379 | 10.5806 |
| loveda/D | barren | 22.9091 | 29.8463 | 29.7914 | 6.8823 | 46.1515 | 45.6643 | 4.3184 |
| loveda/D | tree | 16.4150 | 26.6504 | 26.3109 | 9.8959 | 66.9570 | 30.2370 | 3.3284 |
| loveda/D | farm | 48.0404 | 48.6673 | 48.6443 | 0.6039 | 49.9358 | 94.9514 | 54.3434 |
| udd5/udd5 | vegetation | 64.3148 | 68.3339 | 66.3541 | 2.0393 | 96.7390 | 67.8722 | 20.7807 |
| udd5/udd5 | building | 81.1240 | 82.5379 | 80.9082 | -0.2158 | 81.8769 | 98.5589 | 47.2573 |
| udd5/udd5 | road | 19.3962 | 39.9653 | 36.8110 | 17.4148 | 65.7465 | 45.5460 | 9.2899 |
| udd5/udd5 | vehicle | 11.8369 | 20.9299 | 17.3584 | 5.5215 | 18.0634 | 81.6428 | 3.6181 |
| udd5/udd5 | other | 30.4958 | 35.0039 | 33.6622 | 3.1664 | 47.5376 | 53.5592 | 19.0540 |
| oem/oem | bareland | 6.9087 | 9.9333 | 9.9677 | 3.0590 | 10.2370 | 79.1207 | 9.9748 |
| oem/oem | rangeland | 9.0957 | 17.8451 | 17.6262 | 8.5305 | 66.8853 | 19.3115 | 6.0789 |
| oem/oem | developed space | 22.7255 | 28.1972 | 27.9783 | 5.2528 | 32.7383 | 65.8034 | 39.8160 |
| oem/oem | road | 30.3281 | 34.3982 | 33.4495 | 3.1214 | 50.0355 | 50.2260 | 7.0901 |
| oem/oem | tree | 31.4655 | 36.9726 | 36.0179 | 4.5524 | 89.6158 | 37.5866 | 7.8805 |
| oem/oem | water | 61.2792 | 68.3011 | 67.8420 | 6.5628 | 77.7785 | 84.1531 | 2.5540 |
| oem/oem | agriculture land | 61.9014 | 61.7309 | 61.4685 | -0.4329 | 65.4372 | 91.0196 | 16.4494 |
| oem/oem | building | 41.9472 | 45.5628 | 43.1415 | 1.1943 | 82.9821 | 47.3289 | 10.1564 |
| vdd/vdd | other | 36.5655 | 35.4141 | 34.6725 | -1.8930 | 52.2929 | 50.7144 | 18.2434 |
| vdd/vdd | wall | 29.0582 | 48.3571 | 44.2644 | 15.2062 | 76.2403 | 51.3477 | 2.1791 |
| vdd/vdd | road | 45.0432 | 41.8065 | 41.3118 | -3.7314 | 42.5075 | 93.6249 | 12.0902 |
| vdd/vdd | vegetation | 50.2515 | 66.6633 | 65.1044 | 14.8529 | 96.7800 | 66.5458 | 24.5575 |
| vdd/vdd | vehicle | 29.6526 | 26.3114 | 24.0637 | -5.5889 | 26.1920 | 74.7569 | 1.4860 |
| vdd/vdd | roof | 85.5153 | 83.6761 | 82.5731 | -2.9422 | 83.2531 | 99.0205 | 25.0269 |
| vdd/vdd | water | 84.2928 | 86.2240 | 84.5302 | 0.2374 | 88.1842 | 95.3272 | 16.4170 |
| potsdam/potsdam | impervious surface | 61.2488 | 61.3570 | 60.9712 | -0.2776 | 72.5450 | 79.2604 | 34.3844 |
| potsdam/potsdam | building | 77.2218 | 79.0066 | 78.5858 | 1.3640 | 81.0135 | 96.3269 | 28.4408 |
| potsdam/potsdam | low vegetation | 28.5736 | 34.1299 | 33.5932 | 5.0196 | 93.0714 | 34.4549 | 7.7774 |
| potsdam/potsdam | tree | 50.9752 | 54.7287 | 54.4453 | 3.4701 | 83.4140 | 61.0551 | 12.4734 |
| potsdam/potsdam | car | 30.4756 | 26.6547 | 26.4974 | -3.9782 | 26.5623 | 99.0851 | 7.3297 |
| potsdam/potsdam | clutter | 5.3220 | 6.2632 | 6.2053 | 0.8833 | 8.6407 | 18.0441 | 9.5943 |
| vaihingen/vaihingen | impervious surface | 56.9979 | 61.2549 | 60.7368 | 3.7389 | 78.7633 | 72.6311 | 26.2386 |
| vaihingen/vaihingen | building | 64.9474 | 68.0060 | 67.3884 | 2.4410 | 68.0059 | 98.6704 | 38.2221 |
| vaihingen/vaihingen | low vegetation | 30.2954 | 33.1817 | 32.2895 | 1.9941 | 88.6940 | 33.6756 | 8.1361 |
| vaihingen/vaihingen | tree | 62.0184 | 69.1505 | 68.4905 | 6.4721 | 80.6739 | 81.9338 | 22.8193 |
| vaihingen/vaihingen | car | 22.5506 | 25.2312 | 24.8335 | 2.2829 | 25.5586 | 89.7474 | 4.5839 |
| landcoverai/landcoverai | background | 74.1810 | 82.5181 | 82.0601 | 7.8791 | 86.8628 | 93.6876 | 64.7389 |
| landcoverai/landcoverai | building | 44.5106 | 47.8759 | 47.7569 | 3.2463 | 48.5760 | 96.5895 | 1.8197 |
| landcoverai/landcoverai | woodland | 52.0468 | 73.5000 | 72.9876 | 20.9408 | 95.0339 | 75.8818 | 25.2419 |
| landcoverai/landcoverai | water | 64.8284 | 76.4114 | 74.8865 | 10.0581 | 89.1247 | 82.4177 | 5.4770 |
| landcoverai/landcoverai | road | 39.1431 | 38.1494 | 40.2663 | 1.1232 | 44.8024 | 79.9078 | 2.7225 |
| flair1/flair1 | building | 60.2824 | 61.5362 | 62.7026 | 2.4202 | 64.6704 | 95.3719 | 12.7226 |
| flair1/flair1 | pervious surface | 32.1094 | 34.8516 | 33.6647 | 1.5553 | 69.8535 | 39.3871 | 4.1530 |
| flair1/flair1 | impervious surface | 56.6589 | 59.1300 | 60.3703 | 3.7114 | 71.1690 | 79.9145 | 16.8701 |
| flair1/flair1 | bare soil | 23.1537 | 36.8814 | 35.3552 | 12.2015 | 37.0465 | 88.5640 | 10.4502 |
| flair1/flair1 | water | 74.1419 | 80.1481 | 80.3025 | 6.1606 | 88.2439 | 89.9225 | 6.1151 |
| flair1/flair1 | coniferous | 14.4692 | 15.5778 | 15.9311 | 1.4619 | 44.1066 | 19.9609 | 1.0853 |
| flair1/flair1 | deciduous | 48.7353 | 55.9525 | 56.3355 | 7.6002 | 77.2496 | 67.5414 | 12.1961 |
| flair1/flair1 | brushwood | 11.1440 | 26.8455 | 26.5580 | 15.4140 | 42.3527 | 41.5936 | 6.8023 |
| flair1/flair1 | vineyard | 57.6837 | 63.9008 | 62.9809 | 5.2972 | 80.1560 | 74.6149 | 3.6120 |
| flair1/flair1 | herbaceous vegetation | 18.8786 | 31.5795 | 31.8476 | 12.9690 | 79.4604 | 34.7046 | 9.7096 |
| flair1/flair1 | agricultural land | 16.1240 | 23.4237 | 23.7343 | 7.6103 | 37.6172 | 39.1397 | 7.2484 |
| flair1/flair1 | plowed land | 10.1912 | 20.2991 | 19.9865 | 9.7953 | 20.8235 | 83.2571 | 9.0353 |

## Evaluation Cost

| Dataset | All-control worker wall s | Aggregate worker s | Peak allocated MiB | Geometry encodings/image | Wide encodings/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda | 302.32 | 1190.84 | 5940.08 | 4 | 4 |
| udd5 | 52.77 | 52.77 | 5736.34 | 4 | 2 |
| oem | 181.48 | 181.48 | 5827.61 | 4 | 4 |
| vdd | 60.90 | 121.53 | 5867.25 | 4 | 2 |
| potsdam | 63.43 | 251.46 | 5781.72 | 4 | 4 |
| vaihingen | 52.42 | 52.42 | 5759.90 | 4 | 4 |
| landcoverai | 674.40 | 674.40 | 5759.90 | 4 | 4 |
| flair1 | 1703.06 | 6744.25 | 5916.14 | 4 | 4 |

These worker costs include all five simultaneous readouts, image decoding, masks, original-size output restoration, confusion accumulation and checkpointing. They are NOT warmed primary-only inference latency. Old native-resolution timings are not assigned to this resized model.

Verified 20092 unique full images, per-image confusion sums, matching scored target counts and class order versus VIP, frozen checkpoints/vocabulary and bounded encodings. Reducing resolution changes visual sampling as well as branch geometry; differences from native results do not isolate only a resize operator.

## Matched Complete-Image Timing

| Dataset | VIP All20 ms | No admission ms | Soft ms | Soft/VIP | Maximum soft ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda | 200.62 | 242.68 | 463.20 | 2.309x | 464.00 |
| udd5 | 167.75 | 387.64 | 444.47 | 2.650x | 497.53 |
| oem | 102.98 | 231.33 | 340.34 | 3.305x | 341.65 |
| vdd | 127.19 | 323.77 | 391.99 | 3.082x | 392.99 |
| potsdam | 103.96 | 235.09 | 346.74 | 3.335x | 348.38 |
| vaihingen | 100.67 | 229.45 | 339.05 | 3.368x | 339.77 |
| landcoverai | 96.85 | 222.75 | 331.95 | 3.427x | 332.82 |
| flair1 | 104.09 | 232.60 | 344.57 | 3.310x | 345.71 |

Three fixed complete inputs/domain, three warmed synchronized rotated singleton repeats per input, serial workers. Includes all views, reconstruction, soft admission and original-size output restoration/stitching/argmax. Excludes initialization/text/image loading and masks. This24-image panel is not full-domain average throughput. Peak allocation in timing.json has both models resident, not isolated deployment memory.

## Official-Query / Distilled VIP Comparator

Separate whole-method comparison, not a same-word ablation. The repaired VIP paper-rule adaptation uses official short queries on VDD/Potsdam/corrected Vaihingen and previously frozen image-only distilled words on the other five domains. These remain local reimplementations, not published benchmark scores.

| Dataset/protocol | Official-short or distilled VIP | Bounded soft | Delta pp |
| --- | ---: | ---: | ---: |
| loveda/P | 55.0261 | 62.4389 | +7.4128 |
| loveda/D | 35.3436 | 39.9584 | +4.6148 |
| udd5/udd5 | 44.9713 | 47.0188 | +2.0475 |
| oem/oem | 35.0388 | 37.1865 | +2.1477 |
| vdd/vdd | 52.0647 | 53.7886 | +1.7239 |
| potsdam/potsdam | 44.8995 | 43.3830 | -1.5165 |
| vaihingen/vaihingen | 41.9062 | 50.7477 | +8.8415 |
| landcoverai/landcoverai | 50.2980 | 63.5915 | +13.2935 |
| flair1/flair1 | 36.6074 | 42.4808 | +5.8734 |

## Final Selection Decision

| Dataset | Best measured VIP query protocol | Bounded soft | Delta pp |
| --- | ---: | ---: | ---: |
| loveda | 36.1457 | 39.9584 | +3.8127 |
| udd5 | 44.9713 | 47.0188 | +2.0475 |
| oem | 35.0388 | 37.1865 | +2.1477 |
| vdd | 52.0647 | 53.7886 | +1.7239 |
| potsdam | 44.8995 | 43.3830 | -1.5165 |
| vaihingen | 47.3620 | 50.7477 | +3.3857 |
| landcoverai | 54.9420 | 63.5915 | +8.6495 |
| flair1 | 36.6074 | 42.4808 | +5.8734 |
| Eight-domain mean (LoveDA D once) | 44.0039 | 47.2694 | +3.2655 |

The soft primary beats the strongest already measured VIP query protocol on 7/8 domains, not all eight. Remaining domains: potsdam. Soft weighting improves matched no-admission on only 1/8 domains; mean change -0.0292pp. Best measured VIP here means the maximum of the explicitly listed completed All20 and official/distilled adaptations, not a claim of the best possible published VIP.

This candidate is NOT selected as the final model: the joint all-domain accuracy/latency objective remains unachieved. Sampling acceleration does not establish alias-module efficacy.
