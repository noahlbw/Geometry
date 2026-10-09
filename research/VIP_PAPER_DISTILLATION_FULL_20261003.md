# VIP Paper-Rule Vocabulary Distillation: Full Five-Domain Paired Evaluation

Same fixed20 candidate pools, pinned VIP proxy-attention visual readout, 80 ImageNet templates, checkpoints, input images, repository resize/crop protocol and masks. No vocabulary regeneration or target-label fitting. Each candidate replaces only its own canonical class, all rivals remain canonical. VG/SC are computed from the resulting multiclass softmax map on >=0.4 high-activation patches, with attention power2 and two walks. Equal-image averaging skips absent high regions. Retain canonical plus aliases with cosine>=0.7, VG higher and entropy lower than canonical. No quota.

This is transductive, using ALL unlabeled evaluation images before freezing each dataset vocabulary. Public pinned VIP does not implement distillation; this is the paper-rule reimplementation, not execution of unavailable upstream selection code. Fixed attention-prefix, text-cosine and sliding-crop pooling conventions are recorded in protocol.json. The pinned long-edge448/crop336/stride112 protocol is preserved rather than replacing it with the appendix shorter-edge336/crop224 setting.

## Explicit Numerical Repair

Unmodified proxy attention can mask EVERY entry in a row (similarity -1.5*global_mean <=0), so softmax(all -inf) returns NaN. Earlier historical VIP evaluations did not reject these predictions. The pre-evaluation image-only audit found nonfinite crops in LoveDA 2/32, OEM 1/32, LandCover.ai 29/32, FLAIR-1 14/32 and UDD5 0/16 (first8 images/domain). Every originally finite crop was unchanged exactly by the fixed repair; every repaired feature was finite. All runs here use the same explicit self-Value fallback ONLY for all-masked rows. This is numerically repaired VIP, not bitwise upstream VIP on undefined cases. Both All20 and distilled arms use this source; the three official-query domains are rerun with the same repair. Historical raw-VIP results are not the primary comparator.

## Paired VIP Results

| Dataset/protocol | Images | All20 | Distilled | Delta pp | Retained aliases/class |
| --- | ---: | ---: | ---: | ---: | --- |
| loveda/P | 1669 | 56.3110 | 55.0261 | -1.2849 | [13, 2, 7, 13, 13, 11] |
| loveda/D | 1669 | 36.1457 | 35.3436 | -0.8020 | [15, 14, 3, 7, 13, 12, 11] |
| udd5/udd5 | 40 | 41.4336 | 44.9713 | +3.5378 | [14, 10, 12, 13, 13] |
| oem/oem | 384 | 33.2064 | 35.0388 | +1.8324 | [11, 19, 13, 5, 12, 10, 4, 15] |
| landcoverai/landcoverai | 1602 | 54.9420 | 50.2980 | -4.6440 | [19, 13, 12, 1, 6] |
| flair1/flair1 | 15700 | 35.2977 | 36.6074 | +1.3097 | [12, 5, 17, 5, 3, 11, 16, 10, 5, 18, 4, 10] |

Distillation improves on All20 in 3/5 domains; equal-five-domain mean change +0.2468pp. LoveDA D counted once.

## Updated Eight-Domain Comparator

VDD/Potsdam/Vaihingen retain pinned official short queries/settings but use finite-row repair; Vaihingen uses corrected IRRG input. The other five domains use the newly distilled fixed20 external pools. These are method adaptations, not VIP published benchmark results. LandCover.ai substitutes for unlabeled iSAID. Our retained model was selected on developed windows; comparison remains exploratory full validation.

| Dataset/protocol | VIP comparator | RivalFineHard20 | Delta pp |
| --- | ---: | ---: | ---: |
| loveda/P | 55.0261 | 65.2685 | +10.2424 |
| loveda/D | 35.3436 | 42.7491 | +7.4055 |
| udd5/udd5 | 44.9713 | 52.9563 | +7.9849 |
| oem/oem | 35.0388 | 39.2472 | +4.2084 |
| vdd/vdd | 52.0647 | 56.6428 | +4.5781 |
| potsdam/potsdam | 44.8995 | 44.7208 | -0.1787 |
| vaihingen/vaihingen | 41.9062 | 52.4691 | +10.5629 |
| landcoverai/landcoverai | 50.2980 | 64.0676 | +13.7696 |
| flair1/flair1 | 36.6074 | 42.6407 | +6.0333 |
| Eight-domain mean (LoveDA D once) | 42.6412 | 49.4367 | +6.7955 |

## Historical Versus Repaired All20

| Dataset/protocol | Historical raw VIP | Repaired All20 | Numeric repair delta pp | Empty proxy rows / observed rows |
| --- | ---: | ---: | ---: | --- |
| loveda/P | 42.3170 | 56.3110 | +13.9940 | 710932/5888232 |
| loveda/D | 33.9828 | 36.1457 | +2.1629 | 710932/5888232 |
| udd5/udd5 | 41.4336 | 41.4336 | +0.0000 | 0/70560 |
| oem/oem | 28.9516 | 33.2064 | +4.2548 | 76756/1354752 |
| landcoverai/landcoverai | 37.3519 | 54.9420 | +17.5901 | 3491922/5651856 |
| flair1/flair1 | 19.3010 | 35.2977 | +15.9967 | 15921674/55389600 |

## Classwise Outcomes

| Dataset/protocol | Class | All20 IoU | Distilled IoU | Delta pp | Precision | Recall | Predicted area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loveda/P | building | 81.4273 | 79.3618 | -2.0655 | 85.2657 | 91.9754 | 12.1638 |
| loveda/P | road | 63.5157 | 61.2826 | -2.2331 | 85.9698 | 68.0927 | 5.7886 |
| loveda/P | water | 65.2910 | 62.3187 | -2.9723 | 93.0426 | 65.3646 | 12.8738 |
| loveda/P | barren | 34.6101 | 33.9002 | -0.7099 | 43.9968 | 59.6324 | 9.2574 |
| loveda/P | tree | 21.2621 | 21.6331 | 0.3710 | 75.8160 | 23.2365 | 3.5351 |
| loveda/P | farm | 71.7595 | 71.6600 | -0.0995 | 74.8607 | 94.3696 | 56.3813 |
| loveda/D | background | 11.9386 | 11.9665 | 0.0279 | 51.1257 | 13.5122 | 9.5410 |
| loveda/D | building | 47.6742 | 46.5717 | -1.1025 | 49.5685 | 88.5098 | 12.8665 |
| loveda/D | road | 50.1601 | 51.7150 | 1.5549 | 69.3253 | 67.0601 | 4.5174 |
| loveda/D | water | 55.8824 | 49.7909 | -6.0915 | 76.3158 | 58.8909 | 9.0361 |
| loveda/D | barren | 22.9091 | 21.9642 | -0.9449 | 29.0677 | 47.3347 | 7.1072 |
| loveda/D | tree | 16.4150 | 17.7414 | 1.3264 | 50.3753 | 21.4988 | 3.1455 |
| loveda/D | farm | 48.0404 | 47.6557 | -0.3847 | 49.4243 | 93.0154 | 53.7863 |
| loveda/D | foreground_mean_iou_percent | 40.1802 | 39.2398 | -- | -- | -- | -- |
| udd5/udd5 | vegetation | 64.3148 | 67.0805 | 2.7657 | 93.5741 | 70.3198 | 22.2583 |
| udd5/udd5 | building | 81.1240 | 79.9297 | -1.1943 | 82.0560 | 96.8599 | 46.3413 |
| udd5/udd5 | road | 19.3962 | 32.1817 | 12.7855 | 50.8014 | 46.7529 | 12.3415 |
| udd5/udd5 | vehicle | 11.8369 | 15.4845 | 3.6476 | 17.3658 | 58.8362 | 2.7121 |
| udd5/udd5 | other | 30.4958 | 30.1803 | -0.3155 | 47.1682 | 45.5924 | 16.3469 |
| udd5/udd5 | non_residual_mean_iou_percent | 44.1680 | 48.6691 | -- | -- | -- | -- |
| oem/oem | bareland | 6.9087 | 6.6615 | -0.2472 | 6.9775 | 59.5262 | 11.0101 |
| oem/oem | rangeland | 9.0957 | 12.3559 | 3.2602 | 56.4549 | 13.6575 | 5.0934 |
| oem/oem | developed space | 22.7255 | 22.1125 | -0.6130 | 27.3809 | 53.4716 | 38.6849 |
| oem/oem | road | 30.3281 | 29.2458 | -1.0823 | 46.0591 | 44.4806 | 6.8211 |
| oem/oem | tree | 31.4655 | 33.1620 | 1.6965 | 83.2263 | 35.5372 | 8.0228 |
| oem/oem | water | 61.2792 | 64.2911 | 3.0119 | 80.0928 | 76.5185 | 2.2552 |
| oem/oem | agriculture land | 61.9014 | 68.4754 | 6.5740 | 83.1128 | 79.5422 | 11.3180 |
| oem/oem | building | 41.9472 | 44.0064 | 2.0592 | 62.9601 | 59.3792 | 16.7945 |
| landcoverai/landcoverai | background | 74.1810 | 72.6430 | -1.5380 | 74.6995 | 96.3486 | 77.4185 |
| landcoverai/landcoverai | building | 44.5106 | 50.8202 | 6.3096 | 53.8116 | 90.1399 | 1.5329 |
| landcoverai/landcoverai | woodland | 52.0468 | 49.3624 | -2.6844 | 94.9030 | 50.7067 | 16.8907 |
| landcoverai/landcoverai | water | 64.8284 | 38.9371 | -25.8913 | 98.9391 | 39.1003 | 2.3406 |
| landcoverai/landcoverai | road | 39.1431 | 39.7273 | 0.5842 | 52.3144 | 62.2803 | 1.8172 |
| landcoverai/landcoverai | foreground_mean_iou_percent | 50.1322 | 44.7117 | -- | -- | -- | -- |
| flair1/flair1 | building | 60.2824 | 58.6727 | -1.6097 | 62.1492 | 91.2960 | 12.6729 |
| flair1/flair1 | pervious surface | 32.1094 | 7.0286 | -25.0808 | 47.1442 | 7.6298 | 1.1920 |
| flair1/flair1 | impervious surface | 56.6589 | 45.8843 | -10.7746 | 51.0920 | 81.8237 | 24.0608 |
| flair1/flair1 | bare soil | 23.1537 | 24.4057 | 1.2520 | 41.6864 | 37.0570 | 3.8859 |
| flair1/flair1 | water | 74.1419 | 67.1946 | -6.9473 | 89.8708 | 72.7004 | 4.8545 |
| flair1/flair1 | coniferous | 14.4692 | 14.9779 | 0.5087 | 43.0053 | 18.6874 | 1.0421 |
| flair1/flair1 | deciduous | 48.7353 | 53.6978 | 4.9625 | 73.5507 | 66.5483 | 12.6211 |
| flair1/flair1 | brushwood | 11.1440 | 22.1228 | 10.9788 | 43.2520 | 31.1703 | 4.9917 |
| flair1/flair1 | vineyard | 57.6837 | 59.3478 | 1.6641 | 85.3796 | 66.0614 | 3.0023 |
| flair1/flair1 | herbaceous vegetation | 18.8786 | 41.9934 | 23.1148 | 67.7829 | 52.4651 | 17.2073 |
| flair1/flair1 | agricultural land | 16.1240 | 22.6933 | 6.5693 | 35.3367 | 38.8097 | 7.6511 |
| flair1/flair1 | plowed land | 10.1912 | 21.2695 | 11.0783 | 23.3520 | 70.4580 | 6.8183 |

## Runtime And Memory

Selection timings include frozen model forward, all-layer attention capture, scoring and compressed cache writes. Evaluation timings cover cached All20 AND distilled aggregation, decompression and metric computation; they are not fresh-image standalone inference timings. Initialization/text encoding is excluded from worker timers. The suite wall includes these initialization steps.

| Dataset | Selection parallel s | Selection aggregate GPU s | Selection peak MiB | Paired aggregation parallel s | Aggregation peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda | 403.525 | 806.379 | 1804.268 | 117.585 | 74.755 |
| udd5 | 16.504 | 16.504 | 1757.070 | 16.076 | 566.361 |
| oem | 137.031 | 137.031 | 1799.942 | 36.675 | 83.339 |
| landcoverai | 435.997 | 435.997 | 1757.070 | 88.415 | 19.152 |
| flair1 | 1770.412 | 7066.617 | 1853.464 | 398.902 | 42.187 |

| Official-query dataset | Inference seconds | Peak MiB | Empty proxy rows / observed rows |
| --- | ---: | ---: | --- |
| vdd | 49.470 | 2332.553 | 0/141120 |
| potsdam | 83.976 | 1739.281 | 11466/1778112 |
| vaihingen | 14.996 | 1739.281 | 0/398664 |

## Eight-Domain Per-Class Comparison

| Dataset/protocol | Class | Repaired VIP comparator IoU | Retained RivalFineHard20 IoU | Delta pp |
| --- | --- | ---: | ---: | ---: |
| loveda/P | building | 79.3618 | 89.3451 | 9.9833 |
| loveda/P | road | 61.2826 | 69.7269 | 8.4443 |
| loveda/P | water | 62.3187 | 73.8795 | 11.5608 |
| loveda/P | barren | 33.9002 | 41.4407 | 7.5405 |
| loveda/P | tree | 21.6331 | 40.9761 | 19.3430 |
| loveda/P | farm | 71.6600 | 76.2426 | 4.5826 |
| loveda/D | background | 11.9665 | 19.2269 | 7.2604 |
| loveda/D | building | 46.5717 | 52.9455 | 6.3738 |
| loveda/D | road | 51.7150 | 52.3694 | 0.6544 |
| loveda/D | water | 49.7909 | 61.4183 | 11.6274 |
| loveda/D | barren | 21.9642 | 30.0830 | 8.1188 |
| loveda/D | tree | 17.7414 | 31.3071 | 13.5657 |
| loveda/D | farm | 47.6557 | 51.8936 | 4.2379 |
| udd5/udd5 | vegetation | 67.0805 | 73.4613 | 6.3808 |
| udd5/udd5 | building | 79.9297 | 84.6928 | 4.7631 |
| udd5/udd5 | road | 32.1817 | 43.6524 | 11.4707 |
| udd5/udd5 | vehicle | 15.4845 | 26.0759 | 10.5914 |
| udd5/udd5 | other | 30.1803 | 36.8990 | 6.7187 |
| oem/oem | bareland | 6.6615 | 9.8186 | 3.1571 |
| oem/oem | rangeland | 12.3559 | 18.2996 | 5.9437 |
| oem/oem | developed space | 22.1125 | 28.6950 | 6.5825 |
| oem/oem | road | 29.2458 | 35.6272 | 6.3814 |
| oem/oem | tree | 33.1620 | 40.3417 | 7.1797 |
| oem/oem | water | 64.2911 | 69.1635 | 4.8724 |
| oem/oem | agriculture land | 68.4754 | 62.6488 | -5.8266 |
| oem/oem | building | 44.0064 | 49.3832 | 5.3768 |
| landcoverai/landcoverai | background | 72.6430 | 82.8183 | 10.1753 |
| landcoverai/landcoverai | building | 50.8202 | 48.2488 | -2.5714 |
| landcoverai/landcoverai | woodland | 49.3624 | 74.5308 | 25.1684 |
| landcoverai/landcoverai | water | 38.9371 | 76.2586 | 37.3215 |
| landcoverai/landcoverai | road | 39.7273 | 38.4815 | -1.2458 |
| flair1/flair1 | building | 58.6727 | 61.9912 | 3.3185 |
| flair1/flair1 | pervious surface | 7.0286 | 34.5619 | 27.5333 |
| flair1/flair1 | impervious surface | 45.8843 | 59.4722 | 13.5879 |
| flair1/flair1 | bare soil | 24.4057 | 36.6620 | 12.2563 |
| flair1/flair1 | water | 67.1946 | 80.5043 | 13.3097 |
| flair1/flair1 | coniferous | 14.9779 | 15.6864 | 0.7085 |
| flair1/flair1 | deciduous | 53.6978 | 56.3862 | 2.6884 |
| flair1/flair1 | brushwood | 22.1228 | 26.9633 | 4.8405 |
| flair1/flair1 | vineyard | 59.3478 | 63.9613 | 4.6135 |
| flair1/flair1 | herbaceous vegetation | 41.9934 | 31.7266 | -10.2668 |
| flair1/flair1 | agricultural land | 22.6933 | 23.7156 | 1.0223 |
| flair1/flair1 | plowed land | 21.2695 | 20.0577 | -1.2118 |
| vdd/vdd | other | 35.5829 | 34.8894 | -0.6935 |
| vdd/vdd | wall | 40.1862 | 51.7045 | 11.5183 |
| vdd/vdd | road | 40.5200 | 41.4658 | 0.9458 |
| vdd/vdd | vegetation | 52.5014 | 68.5503 | 16.0489 |
| vdd/vdd | vehicle | 25.2671 | 27.9274 | 2.6603 |
| vdd/vdd | roof | 81.6539 | 84.4787 | 2.8248 |
| vdd/vdd | water | 88.7412 | 87.4835 | -1.2577 |
| potsdam/potsdam | impervious surface | 60.7547 | 62.4545 | 1.6998 |
| potsdam/potsdam | building | 71.0553 | 80.0106 | 8.9553 |
| potsdam/potsdam | low vegetation | 52.7964 | 35.5348 | -17.2616 |
| potsdam/potsdam | tree | 56.8492 | 56.8659 | 0.0167 |
| potsdam/potsdam | car | 16.8657 | 27.1159 | 10.2502 |
| potsdam/potsdam | clutter | 11.0760 | 6.3433 | -4.7327 |
| vaihingen/vaihingen | impervious surface | 45.4945 | 62.4942 | 16.9997 |
| vaihingen/vaihingen | building | 60.4878 | 69.6630 | 9.1752 |
| vaihingen/vaihingen | low vegetation | 22.1971 | 33.9315 | 11.7344 |
| vaihingen/vaihingen | tree | 56.9623 | 69.8625 | 12.9002 |
| vaihingen/vaihingen | car | 24.3892 | 26.3943 | 2.0051 |

Suite elapsed 2239.423s; all20092 unique full images verified (19395 with distillation). Scored targets match the historical comparator; All20 is recomputed after explicit finite-row repair.

Selected.json files contain every candidate alias, canonical flag, cosine, VG, SC, observed-image count and retention reason. Source logits are cached remotely; no masks enter vocabulary selection.

Paper: https://arxiv.org/html/2605.12325v2
