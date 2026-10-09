# RivalFineHard20: Full Eight-Dataset Comparison With VIP

Fixed20 aliases/class; frozen DINOv3/DINO.text, original Geometry, physical256-to512 fine observations, query/class/rival hard admission and exact Geometry reconstruction. No dataset-specific numerical tuning. The method was selected on developed windows; these are full exploratory validation results, not untouched test or published VIP benchmark numbers.

LandCover.ai explicitly substitutes for iSAID, whose labeled validation is unavailable here. LoveDA P and D use the same1669 images. All20092 images are evaluated at full dimensions.

## Main Comparison

| Dataset/protocol | Images | Historical VIP | Matched-input VIP | Geometry | Unscreened coupling | RivalFineHard20 | Delta vs matched VIP | Delta vs unscreened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoveDA P | 1669 | 42.3170 | 42.3170 | 64.9557 | 62.7149 | 65.2685 | +22.9515 | +2.5536 |
| LoveDA D | 1669 | 33.9828 | 33.9828 | 42.6589 | 40.2519 | 42.7491 | +8.7663 | +2.4972 |
| udd5 | 40 | 41.4336 | 41.4336 | 50.5553 | 49.2117 | 52.9563 | +11.5227 | +3.7446 |
| oem | 384 | 28.9516 | 28.9516 | 44.6097 | 37.8977 | 39.2472 | +10.2956 | +1.3495 |
| vdd | 80 | 52.0647 | 52.0647 | 38.8511 | 55.4446 | 56.6428 | +4.5781 | +1.1982 |
| potsdam | 504 | 44.0643 | 44.0643 | 40.6892 | 43.7334 | 44.7208 | +0.6566 | +0.9874 |
| vaihingen | 113 | 5.6908 | 41.9062 | 48.8009 | 51.4079 | 52.4691 | +10.5629 | +1.0612 |
| landcoverai | 1602 | 37.3519 | 37.3519 | 59.2340 | 63.7049 | 64.0676 | +26.7157 | +0.3627 |
| flair1 | 15700 | 19.3010 | 19.3010 | 43.8071 | 42.5381 | 42.6407 | +23.3397 | +0.1026 |
| Eight-domain mean (LoveDA D once) | 20092 | -- | 37.3820 | 46.1508 | 48.0238 | 49.4367 | +12.0547 | +1.4129 |

## Outcome

The frozen primary improves on the matched unscreened coupling on 8/8 domains, with an equal-domain mean gain of 1.4129pp. It exceeds the protocol-specific VIP comparator on 8/8 domains. This supports conditional admission on these full validation sets, not universal alias correctness or an all-benchmark SOTA claim.

The complete model still falls below original Geometry on OEM by5.3625pp and FLAIR-1 by1.1664pp. Screening helps both relative to unscreened coupling but does not repair the entire contextual-reader loss. Potsdam low vegetation remains35.5348 versus VIP50.6618, despite the higher overall mIoU. No dataset-specific winner switching or post-result parameter change was applied.

VIP uses official short queries/settings/background rules on VDD, Potsdam and Vaihingen. On the five unsupported domains it is a fixed20-vocabulary constructor-default adaptation, not a reproduction of unpublished VIP distilled vocabularies. Our uniform20 method is a whole-method comparison, not a matched-word-only ablation. Historical Vaihingen used old first-three-band input; the matched VIP column reruns unchanged VIP on the same corrected IRRG input as our model.

## Foreground And Non-Residual Metrics

| Protocol | VIP | RivalFineHard20 |
| --- | ---: | ---: |
| loveda/D | 36.3863 | 46.6695 |
| udd5/udd5 | 44.1680 | 56.9706 |
| landcoverai/landcoverai | 31.1774 | 59.3799 |

## Per-Class IoU

| Protocol | Class | VIP | RivalFineHard20 | Delta pp |
| --- | --- | ---: | ---: | ---: |
| loveda/P | building | 34.2803 | 89.3451 | +55.0648 |
| loveda/P | road | 63.0261 | 69.7269 | +6.7008 |
| loveda/P | water | 50.2333 | 73.8795 | +23.6462 |
| loveda/P | barren | 34.3073 | 41.4407 | +7.1334 |
| loveda/P | tree | 20.4350 | 40.9761 | +20.5411 |
| loveda/P | farm | 51.6200 | 76.2426 | +24.6226 |
| loveda/D | background | 19.5618 | 19.2269 | -0.3349 |
| loveda/D | building | 47.5121 | 52.9455 | +5.4334 |
| loveda/D | road | 50.3943 | 52.3694 | +1.9751 |
| loveda/D | water | 44.0357 | 61.4183 | +17.3826 |
| loveda/D | barren | 22.8765 | 30.0830 | +7.2065 |
| loveda/D | tree | 16.1073 | 31.3071 | +15.1998 |
| loveda/D | farm | 37.3918 | 51.8936 | +14.5018 |
| udd5/udd5 | vegetation | 64.3148 | 73.4613 | +9.1465 |
| udd5/udd5 | building | 81.1240 | 84.6928 | +3.5688 |
| udd5/udd5 | road | 19.3962 | 43.6524 | +24.2562 |
| udd5/udd5 | vehicle | 11.8369 | 26.0759 | +14.2390 |
| udd5/udd5 | other | 30.4958 | 36.8990 | +6.4032 |
| oem/oem | bareland | 4.1503 | 9.8186 | +5.6683 |
| oem/oem | rangeland | 6.9879 | 18.2996 | +11.3117 |
| oem/oem | developed space | 22.4503 | 28.6950 | +6.2447 |
| oem/oem | road | 29.8397 | 35.6272 | +5.7875 |
| oem/oem | tree | 24.1271 | 40.3417 | +16.2146 |
| oem/oem | water | 53.5637 | 69.1635 | +15.5998 |
| oem/oem | agriculture land | 51.5215 | 62.6488 | +11.1273 |
| oem/oem | building | 38.9718 | 49.3832 | +10.4114 |
| vdd/vdd | other | 35.5829 | 34.8894 | -0.6935 |
| vdd/vdd | wall | 40.1862 | 51.7045 | +11.5183 |
| vdd/vdd | road | 40.5200 | 41.4658 | +0.9458 |
| vdd/vdd | vegetation | 52.5014 | 68.5503 | +16.0489 |
| vdd/vdd | vehicle | 25.2671 | 27.9274 | +2.6603 |
| vdd/vdd | roof | 81.6539 | 84.4787 | +2.8248 |
| vdd/vdd | water | 88.7412 | 87.4835 | -1.2577 |
| potsdam/potsdam | impervious surface | 59.5516 | 62.4545 | +2.9029 |
| potsdam/potsdam | building | 71.0761 | 80.0106 | +8.9345 |
| potsdam/potsdam | low vegetation | 50.6618 | 35.5348 | -15.1270 |
| potsdam/potsdam | tree | 56.7532 | 56.8659 | +0.1127 |
| potsdam/potsdam | car | 16.8901 | 27.1159 | +10.2258 |
| potsdam/potsdam | clutter | 9.4527 | 6.3433 | -3.1094 |
| vaihingen/vaihingen | impervious surface | 45.4945 | 62.4942 | +16.9997 |
| vaihingen/vaihingen | building | 60.4878 | 69.6630 | +9.1752 |
| vaihingen/vaihingen | low vegetation | 22.1971 | 33.9315 | +11.7344 |
| vaihingen/vaihingen | tree | 56.9623 | 69.8625 | +12.9002 |
| vaihingen/vaihingen | car | 24.3892 | 26.3943 | +2.0051 |
| landcoverai/landcoverai | background | 62.0497 | 82.8183 | +20.7686 |
| landcoverai/landcoverai | building | 46.2994 | 48.2488 | +1.9494 |
| landcoverai/landcoverai | woodland | 8.7210 | 74.5308 | +65.8098 |
| landcoverai/landcoverai | water | 35.4498 | 76.2586 | +40.8088 |
| landcoverai/landcoverai | road | 34.2395 | 38.4815 | +4.2420 |
| flair1/flair1 | building | 16.7925 | 61.9912 | +45.1987 |
| flair1/flair1 | pervious surface | 27.4408 | 34.5619 | +7.1211 |
| flair1/flair1 | impervious surface | 56.7782 | 59.4722 | +2.6940 |
| flair1/flair1 | bare soil | 7.8128 | 36.6620 | +28.8492 |
| flair1/flair1 | water | 44.9114 | 80.5043 | +35.5929 |
| flair1/flair1 | coniferous | 4.7435 | 15.6864 | +10.9429 |
| flair1/flair1 | deciduous | 29.2411 | 56.3862 | +27.1451 |
| flair1/flair1 | brushwood | 3.5975 | 26.9633 | +23.3658 |
| flair1/flair1 | vineyard | 17.4552 | 63.9613 | +46.5061 |
| flair1/flair1 | herbaceous vegetation | 13.5540 | 31.7266 | +18.1726 |
| flair1/flair1 | agricultural land | 4.6208 | 23.7156 | +19.0948 |
| flair1/flair1 | plowed land | 4.6640 | 20.0577 | +15.3937 |

## Full-Run Cost

| Dataset | Model wall s | Model aggregate GPU s | Model peak MiB | VIP wall s | VIP peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| loveda | 1733.05 | 6821.77 | 5615.61 | 117.40 | 1821.66 |
| udd5 | 855.02 | 855.02 | 5575.76 | 22.30 | 2265.28 |
| oem | 1030.82 | 1030.82 | 5597.16 | 52.91 | 1819.40 |
| vdd | 1984.97 | 1984.97 | 5587.52 | 50.71 | 2527.10 |
| potsdam | 1384.11 | 1384.11 | 5583.62 | 80.08 | 1754.38 |
| vaihingen | 293.69 | 293.69 | 5578.04 | 14.17 | 1739.10 |
| landcoverai | 638.44 | 638.44 | 5559.25 | 179.65 | 1773.01 |
| flair1 | 1794.82 | 7106.20 | 5737.62 | 277.93 | 1878.05 |

These full-run model costs include shared Geometry and unscreened control accumulation, text encoding, fine observations and exact reconstruction; they are not isolated primary-only latency. Historical VIP sharding differs. Per-image confusion sums, transition endpoints, global coverage, checkpoint/vocabulary/config identity and64 mask-free historical score replays were verified.
