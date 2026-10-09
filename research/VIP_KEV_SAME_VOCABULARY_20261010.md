# Native VIP with the frozen Kev-run language inputs

Status: complete; 15/15 protocols.

The source words/parameters were selected on labeled development and prior developed data. These are frozen exploratory matched-language controls, not independent validation or published VIP numbers. VOC20/21 and PC59/60 share images. PC59 previously lacked a full VIP comparator.

VIP native fixed tau/tem/background settings; natural short-edge336/maxlong2048, RS long448, crop336/stride112. No transfer of our selected calibration, Geometry, writeback, or new distillation. Self-Value repair only on empty proxy rows.

PC60 background uses the same401 explicit residual concepts as the source model, aggregated by standard VIP; no protected Geometry residual gate. LoveDA P uses its independent six-class query bank. Vaihingen retains VIP threshold0.1/index5; same five-class bank has max probability>=0.2, so rejection cannot occur.

Query-parsing correction: the completed Kev source normalized official text files by splitting ALL commas. Pinned VIP splits only comma followed by space. For VDD this changes `surface,other` and `grassland,forest area` from one query each to two; for Potsdam it splits `farmland,forest area`. The official_* source bank names therefore mean normalized official-file words, not byte-identical upstream query groups. Frozen predictions and words are preserved. Historical exact replay is required only when the actual groups match; the first merge assertion incorrectly assumed they always matched and was repaired without rerunning inference.

| Protocol | Our frozen result | Historical VIP | VIP same text | Our − same text pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: | ---: |
| flair1 | 43.6579 | 36.6074 | 35.2977 | +8.3602 | 35.2977 |
| vdd | 60.4352 | 52.0647 | 52.5765 | +7.8587 | 52.5765 |
| potsdam | 56.1934 | 44.8995 | 42.8721 | +13.3213 | 42.8721 |
| udd5 | 55.8266 | 44.9713 | 23.4923 | +32.3343 | 24.2481 |
| vaihingen | 58.8562 | 41.9062 | 49.1039 | +9.7523 | 49.1039 |
| oem | 45.1052 | 35.0388 | 33.2064 | +11.8988 | 33.2064 |
| landcoverai | 65.5400 | 50.298 | 54.9420 | +10.5980 | 54.9420 |
| loveda P | 69.6117 | 55.0261 | 56.3110 | +13.3007 | 56.3110 |
| loveda D | 45.3171 | 35.3436 | 36.1457 | +9.1714 | 36.1457 |
| voc20 | 90.7871 | 92.5051 | 92.5051 | -1.7180 | 92.5051 |
| voc21 | 69.4757 | 73.2591 | 68.4548 | +1.0209 | 68.4548 |
| context59 | 46.3075 | not measured | 45.9359 | +0.3716 | 45.9317 |
| context60 | 41.6004 | 42.5987 | 34.7996 | +6.8008 | 33.7258 |
| ade150 | 31.3398 | 29.1387 | 29.4630 | +1.8768 | 29.4630 |
| coco_object81 | 50.3999 | 48.9955 | 48.0664 | +2.3335 | 48.9955 |
| coco_stuff171 | 33.7661 | 33.4867 | 32.8295 | +0.9366 | 33.4940 |

Eight remote domains; LoveDA D once: `{"ours": 53.86645, "historical_vip": 42.6411875, "same_text": 40.954575, "ours_minus_same_text": 12.911875, "official_template": 41.04905}`.

Matched words do not make observation scales or tau/tem/thresholds identical: this is a native VIP language-input control, not a same-visual-observation operator ablation. Selected source words retain labeled development provenance. Do not attribute all historical-to-matched changes to semantic screening; changed counts and templates affect aggregation.

## Standalone cost

| Protocol | VIP same text ms | Our prior measured ms | Our / same text | Historical VIP ms | VIP peak MiB | Our peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flair1 | 111.14 | 247.00 | 2.222 | 109.08 | 1868.4 | 5548.5 |
| vdd | 190.85 | 494.15 | 2.589 | 191.50 | 2332.9 | 5993.5 |
| potsdam | 104.11 | 259.20 | 2.490 | 109.47 | 1739.2 | 5515.8 |
| udd5 | 191.15 | 500.18 | 2.617 | 187.57 | 2147.0 | 5825.4 |
| vaihingen | 108.55 | 238.85 | 2.200 | 107.71 | 1747.8 | 5519.8 |
| oem | 105.38 | 249.53 | 2.368 | 108.07 | 1810.0 | 5540.3 |
| landcoverai | 102.71 | 233.39 | 2.272 | 106.84 | 1762.6 | 5529.2 |
| loveda | 105.86 | 243.35 | 2.299 | 110.02 | 1811.1 | 5532.1 |
| voc20 | 58.16 | 193.58 | 3.328 | 58.61 | 1743.1 | 5514.3 |
| voc21 | 59.50 | 196.12 | 3.296 | 60.74 | 1745.6 | 5517.0 |
| context59 | 111.84 | 247.86 | 2.216 | 111.35 | 1804.3 | 5567.8 |
| context60 | 112.06 | 259.70 | 2.317 | 112.81 | 1811.4 | 5620.4 |
| ade150 | 113.98 | 293.66 | 2.576 | 111.30 | 2174.8 | 6433.2 |
| coco_object81 | 131.46 | 283.66 | 2.158 | 132.91 | 1944.1 | 5799.4 |
| coco_stuff171 | 185.87 | 366.10 | 1.970 | 183.69 | 2187.2 | 6537.0 |

Primary same-text VIP only, one backbone and one active query bank; seven fixed whole inputs and seven synchronized warmed repeats.
Our/historical timings reuse the previous matched-input measurements; fresh same-text VIP costs are separate processes. Full per-class precision/recall/area, foreground metrics and confusion matrices are in each merged.json.

## Foreground and non-residual metrics

| Protocol | Metric | Our frozen | VIP same text | VIP same words/official template |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | non_residual_mean_iou_percent | 62.9550 | 54.5119 | 54.5119 |
| udd5/udd5 | non_residual_mean_iou_percent | 60.4556 | 23.7115 | 24.5175 |
| landcoverai/landcoverai | foreground_mean_iou_percent | 61.1921 | 50.1322 | 50.1322 |
| loveda/D | foreground_mean_iou_percent | 46.4307 | 40.1802 | 40.1802 |
| voc21/voc21 | foreground_mean_iou_percent | 68.4901 | 67.4041 | 67.4041 |
| context60/context60 | foreground_mean_iou_percent | 41.9465 | 35.1399 | 34.0631 |
| coco_object81/coco_object81 | foreground_mean_iou_percent | 50.0487 | 47.6615 | 48.5939 |

## Interpretation

The frozen coupled model exceeds same-text VIP on14/15 dataset protocols; VOC20 remains lower (90.7871 versus92.5051). This does not overturn the stronger historical VIP results on VOC21 (73.2591) and PC60 (42.5987), which still exceed our69.4757 and41.6004. The matched inputs were optimized for our model and can hurt native VIP.

Language input changes are not uniformly beneficial to VIP: UDD5 drops44.9713 ->23.4923; Vaihingen rises41.9062 ->49.1039, but changes both word groups and the unscored clutter-query membership. VDD changes52.0647 ->52.5765; Potsdam44.8995 ->42.8721. The latter two specifically expose official-file comma parsing differences, not Kev screening gains.

The UDD5 matched bank counts are[7,1,2,1,20]. Native VIP predicts other on67.7532% of scored pixels, while its true area is16.9118%; road/vehicle recalls are2.1498%/1.6126%. This is severe class competition bias. Unequal alias counts and inherited unnormalized log-sum-exp are plausible contributors, but this experiment does not isolate them causally. The32.3343pp whole-model gap is not evidence of a32pp standalone Geometry contribution.

Changing only the template, with aliases and VIP settings fixed, gives COCO Object 48.0664 ->48.9955 (+0.9291pp), COCO Stuff32.8295 ->33.4940 (+0.6645pp), PC6034.7996 ->33.7258 (-1.0738pp), and UDD523.4923 ->24.2481 (+0.7558pp). Both template arms coincide for VDD/Potsdam and several other domains.

Remaining gaps include Geometry/local evidence, coupled writeback, view budgets and selected calibration. This comparison controls language inputs, not all those factors simultaneously. A paper should report both native historical and matched-language VIP, retain the VOC20 failure, and avoid claims that Kev universally helps or that this establishes published-benchmark SOTA.

## flair1

Bank: `original`; counts: `[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| flair1/building | 61.4040 | 60.2824 | +1.1216 | 60.2824 |
| flair1/pervious surface | 33.7509 | 32.1094 | +1.6415 | 32.1094 |
| flair1/impervious surface | 60.0740 | 56.6589 | +3.4151 | 56.6589 |
| flair1/bare soil | 39.6440 | 23.1537 | +16.4903 | 23.1537 |
| flair1/water | 78.6820 | 74.1419 | +4.5401 | 74.1419 |
| flair1/coniferous | 13.9215 | 14.4692 | -0.5477 | 14.4692 |
| flair1/deciduous | 54.0531 | 48.7353 | +5.3178 | 48.7353 |
| flair1/brushwood | 27.2315 | 11.1440 | +16.0875 | 11.1440 |
| flair1/vineyard | 58.4955 | 57.6837 | +0.8118 | 57.6837 |
| flair1/herbaceous vegetation | 39.8215 | 18.8786 | +20.9429 | 18.8786 |
| flair1/agricultural land | 32.0117 | 16.1240 | +15.8877 | 16.1240 |
| flair1/plowed land | 24.8056 | 10.1912 | +14.6144 | 10.1912 |

## vdd

Bank: `official_imagenet`; counts: `[2, 1, 1, 2, 1, 1, 1]`.

VIP settings: `{"tau": 1.0, "tem": 1.0, "prob_thd": 0.35, "background": true, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| vdd/other | 45.3164 | 40.9640 | +4.3524 | 40.9640 |
| vdd/wall | 48.0048 | 35.9683 | +12.0365 | 35.9683 |
| vdd/road | 49.7305 | 50.0878 | -0.3573 | 50.0878 |
| vdd/vegetation | 79.4081 | 73.9611 | +5.4470 | 73.9611 |
| vdd/vehicle | 36.7663 | 17.1158 | +19.6505 | 17.1158 |
| vdd/roof | 82.5380 | 72.9294 | +9.6086 | 72.9294 |
| vdd/water | 81.2826 | 77.0093 | +4.2733 | 77.0093 |

## potsdam

Bank: `official_imagenet`; counts: `[2, 1, 4, 2, 1, 1]`.

VIP settings: `{"tau": 1.0, "tem": 2.0, "prob_thd": 0.25, "background": true, "bg_idx": 5, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| potsdam/impervious surface | 65.6164 | 62.3405 | +3.2759 | 62.3405 |
| potsdam/building | 77.9794 | 71.0669 | +6.9125 | 71.0669 |
| potsdam/low vegetation | 60.0738 | 53.0014 | +7.0724 | 53.0014 |
| potsdam/tree | 66.1729 | 46.3191 | +19.8538 | 46.3191 |
| potsdam/car | 46.4693 | 16.8687 | +29.6006 | 16.8687 |
| potsdam/clutter | 20.8484 | 7.6361 | +13.2123 | 7.6361 |

## udd5

Bank: `kev60_segmentation`; counts: `[7, 1, 2, 1, 20]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| udd5/vegetation | 79.2404 | 68.3982 | +10.8422 | 64.2025 |
| udd5/building | 85.3634 | 22.6962 | +62.6672 | 30.4280 |
| udd5/road | 37.3689 | 2.1466 | +35.2223 | 1.8496 |
| udd5/vehicle | 39.8496 | 1.6049 | +38.2447 | 1.5898 |
| udd5/other | 37.3109 | 22.6158 | +14.6951 | 23.1707 |

## vaihingen

Bank: `kev40_imagenet`; counts: `[16, 16, 17, 12, 8]`.

VIP settings: `{"tau": 1.0, "tem": 10.0, "prob_thd": 0.1, "background": true, "bg_idx": 5, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| vaihingen/impervious surface | 66.0449 | 60.4679 | +5.5770 | 60.4679 |
| vaihingen/building | 72.7879 | 62.8241 | +9.9638 | 62.8241 |
| vaihingen/low vegetation | 40.6235 | 35.6116 | +5.0119 | 35.6116 |
| vaihingen/tree | 69.5008 | 56.8126 | +12.6882 | 56.8126 |
| vaihingen/car | 45.3241 | 29.8032 | +15.5209 | 29.8032 |

## oem

Bank: `original`; counts: `[20, 20, 20, 20, 20, 20, 20, 20]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| oem/bareland | 10.4191 | 6.9087 | +3.5104 | 6.9087 |
| oem/rangeland | 32.0863 | 9.0957 | +22.9906 | 9.0957 |
| oem/developed space | 27.5374 | 22.7255 | +4.8119 | 22.7255 |
| oem/road | 38.0586 | 30.3281 | +7.7305 | 30.3281 |
| oem/tree | 54.9661 | 31.4655 | +23.5006 | 31.4655 |
| oem/water | 68.9790 | 61.2792 | +7.6998 | 61.2792 |
| oem/agriculture land | 66.0801 | 61.9014 | +4.1787 | 61.9014 |
| oem/building | 62.7148 | 41.9472 | +20.7676 | 41.9472 |

## landcoverai

Bank: `original_imagenet`; counts: `[20, 20, 20, 20, 20]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| landcoverai/background | 82.9318 | 74.1810 | +8.7508 | 74.1810 |
| landcoverai/building | 60.9726 | 44.5106 | +16.4620 | 44.5106 |
| landcoverai/woodland | 77.1198 | 52.0468 | +25.0730 | 52.0468 |
| landcoverai/water | 64.1632 | 64.8284 | -0.6652 | 64.8284 |
| landcoverai/road | 42.5126 | 39.1431 | +3.3695 | 39.1431 |

## loveda

Bank: `original`; counts: `[20, 20, 20, 20, 20, 20, 20]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| P/building | 88.4885 | 81.4273 | +7.0612 | 81.4273 |
| P/road | 65.2094 | 63.5157 | +1.6937 | 63.5157 |
| P/water | 75.6226 | 65.2910 | +10.3316 | 65.2910 |
| P/barren | 43.9516 | 34.6101 | +9.3415 | 34.6101 |
| P/tree | 63.1547 | 21.2621 | +41.8926 | 21.2621 |
| P/farm | 81.2434 | 71.7595 | +9.4839 | 71.7595 |
| D/background | 38.6355 | 11.9386 | +26.6969 | 11.9386 |
| D/building | 56.6110 | 47.6742 | +8.9368 | 47.6742 |
| D/road | 52.3713 | 50.1601 | +2.2112 | 50.1601 |
| D/water | 45.6562 | 55.8824 | -10.2262 | 55.8824 |
| D/barren | 25.3890 | 22.9091 | +2.4799 | 22.9091 |
| D/tree | 37.7511 | 16.4150 | +21.3361 | 16.4150 |
| D/farm | 60.8057 | 48.0404 | +12.7653 | 48.0404 |

## voc20

Bank: `official_segmentation`; counts: `[2, 1, 1, 1, 2, 3, 1, 1, 7, 4, 2, 1, 2, 2, 21, 4, 1, 2, 1, 4]`.

VIP settings: `{"tau": 4.0, "tem": 1.0, "prob_thd": 0.0, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| voc20/aeroplane | 98.7269 | 98.7858 | -0.0589 | 98.7858 |
| voc20/bicycle | 90.6731 | 78.5410 | +12.1321 | 78.5410 |
| voc20/bird | 88.1562 | 98.4408 | -10.2846 | 98.4408 |
| voc20/boat | 95.1173 | 96.4059 | -1.2886 | 96.4059 |
| voc20/bottle | 79.0976 | 88.8427 | -9.7451 | 88.8427 |
| voc20/bus | 98.0618 | 98.4189 | -0.3571 | 98.4189 |
| voc20/car | 92.6579 | 97.1708 | -4.5129 | 97.1708 |
| voc20/cat | 98.5062 | 98.7458 | -0.2396 | 98.7458 |
| voc20/chair | 72.2228 | 69.9349 | +2.2879 | 69.9349 |
| voc20/cow | 97.1445 | 99.1495 | -2.0050 | 99.1495 |
| voc20/diningtable | 69.7871 | 77.8509 | -8.0638 | 77.8509 |
| voc20/dog | 96.8559 | 97.2815 | -0.4256 | 97.2815 |
| voc20/horse | 96.8034 | 97.1777 | -0.3743 | 97.1777 |
| voc20/motorbike | 92.7933 | 91.3165 | +1.4768 | 91.3165 |
| voc20/person | 82.7224 | 89.1567 | -6.4343 | 89.1567 |
| voc20/pottedplant | 94.6293 | 97.2341 | -2.6048 | 97.2341 |
| voc20/sheep | 96.0959 | 98.7102 | -2.6143 | 98.7102 |
| voc20/sofa | 87.0033 | 87.9112 | -0.9079 | 87.9112 |
| voc20/train | 99.0004 | 99.2316 | -0.2312 | 99.2316 |
| voc20/tvmonitor | 89.6874 | 89.7960 | -0.1086 | 89.7960 |

## voc21

Bank: `semantic_segmentation`; counts: `[56, 4, 3, 3, 4, 6, 6, 4, 4, 9, 7, 4, 4, 4, 3, 18, 5, 4, 5, 4, 6]`.

VIP settings: `{"tau": 3.0, "tem": 0.3, "prob_thd": 0.28, "background": true, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| voc21/background | 89.1870 | 89.4670 | -0.2800 | 89.4670 |
| voc21/aeroplane | 69.5204 | 58.1221 | +11.3983 | 58.1221 |
| voc21/bicycle | 45.5807 | 42.4406 | +3.1401 | 42.4406 |
| voc21/bird | 89.7051 | 73.6781 | +16.0270 | 73.6781 |
| voc21/boat | 42.0732 | 56.1415 | -14.0683 | 56.1415 |
| voc21/bottle | 55.0314 | 63.6065 | -8.5751 | 63.6065 |
| voc21/bus | 84.9358 | 83.1880 | +1.7478 | 83.1880 |
| voc21/car | 71.5430 | 64.1126 | +7.4304 | 64.1126 |
| voc21/cat | 89.5052 | 87.5922 | +1.9130 | 87.5922 |
| voc21/chair | 47.2028 | 45.6920 | +1.5108 | 45.6920 |
| voc21/cow | 84.3789 | 81.7656 | +2.6133 | 81.7656 |
| voc21/diningtable | 57.1652 | 51.9687 | +5.1965 | 51.9687 |
| voc21/dog | 87.7978 | 83.1495 | +4.6483 | 83.1495 |
| voc21/horse | 85.8893 | 86.6160 | -0.7267 | 86.6160 |
| voc21/motorbike | 59.0673 | 70.4001 | -11.3328 | 70.4001 |
| voc21/person | 69.4979 | 78.7076 | -9.2097 | 78.7076 |
| voc21/pottedplant | 55.6155 | 56.9504 | -1.3349 | 56.9504 |
| voc21/sheep | 91.6522 | 84.6447 | +7.0075 | 84.6447 |
| voc21/sofa | 68.9450 | 62.8695 | +6.0755 | 62.8695 |
| voc21/train | 54.2061 | 60.3190 | -6.1129 | 60.3190 |
| voc21/tvmonitor | 60.4897 | 56.1179 | +4.3718 | 56.1179 |

## context59

Bank: `official_segmentation`; counts: `[1, 3, 1, 3, 2, 1, 1, 1, 1, 3, 4, 1, 2, 2, 1, 1, 6, 8, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 4, 1, 2, 1, 1, 16, 5, 1, 3, 2, 1, 1, 2, 1, 1, 7, 1, 1, 1, 1, 2, 3, 1, 3, 3, 9, 2, 5]`.

VIP settings: `{"tau": 2.0, "tem": 1.0, "prob_thd": 0.1, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| context59/aeroplane | 67.3383 | 61.2938 | +6.0445 | 61.5367 |
| context59/bag | 44.8271 | 45.6415 | -0.8144 | 45.4144 |
| context59/bed | 6.8257 | 11.5929 | -4.7672 | 9.1873 |
| context59/bedclothes | 26.9829 | 29.3116 | -2.3287 | 28.5878 |
| context59/bench | 18.2440 | 25.2274 | -6.9834 | 30.7866 |
| context59/bicycle | 69.3990 | 69.5085 | -0.1095 | 66.8136 |
| context59/bird | 70.3675 | 62.9168 | +7.4507 | 64.0673 |
| context59/boat | 68.6008 | 63.2728 | +5.3280 | 63.5905 |
| context59/book | 6.5114 | 8.5253 | -2.0139 | 11.7494 |
| context59/bottle | 71.2428 | 70.0653 | +1.1775 | 65.0155 |
| context59/building | 36.9821 | 29.6215 | +7.3606 | 21.0019 |
| context59/bus | 80.2299 | 78.5611 | +1.6688 | 75.2774 |
| context59/cabinet | 36.8496 | 33.0901 | +3.7595 | 39.4304 |
| context59/car | 64.8208 | 67.1370 | -2.3162 | 62.8157 |
| context59/cat | 87.9852 | 87.8096 | +0.1756 | 87.4784 |
| context59/ceiling | 48.0722 | 44.6127 | +3.4595 | 47.7476 |
| context59/chair | 43.2269 | 49.7340 | -6.5071 | 47.5953 |
| context59/cloth | 16.9566 | 21.5695 | -4.6129 | 21.6736 |
| context59/computer | 26.6571 | 28.4793 | -1.8222 | 31.8943 |
| context59/cow | 83.9745 | 82.1743 | +1.8002 | 82.5012 |
| context59/cup | 31.5851 | 25.2603 | +6.3248 | 24.1676 |
| context59/curtain | 57.4631 | 55.4709 | +1.9922 | 55.1846 |
| context59/dog | 82.2504 | 82.3715 | -0.1211 | 82.7180 |
| context59/door | 27.4833 | 26.6459 | +0.8374 | 25.7601 |
| context59/fence | 38.4962 | 38.2413 | +0.2549 | 38.8542 |
| context59/floor | 50.5707 | 51.3052 | -0.7345 | 44.2891 |
| context59/flower | 24.5721 | 23.9660 | +0.6061 | 26.5115 |
| context59/food | 50.2279 | 45.6887 | +4.5392 | 33.2901 |
| context59/grass | 71.6279 | 61.7668 | +9.8611 | 66.1694 |
| context59/ground | 12.1732 | 9.0362 | +3.1370 | 9.9575 |
| context59/horse | 83.9426 | 80.4582 | +3.4844 | 80.0458 |
| context59/keyboard | 52.1329 | 62.8262 | -10.6933 | 66.9861 |
| context59/light | 24.1319 | 21.4321 | +2.6998 | 19.5284 |
| context59/motorbike | 73.6571 | 75.6372 | -1.9801 | 76.2161 |
| context59/mountain | 35.5850 | 34.3189 | +1.2661 | 43.6250 |
| context59/mouse | 62.1596 | 51.1336 | +11.0260 | 43.8437 |
| context59/person | 66.1820 | 76.4441 | -10.2621 | 78.6398 |
| context59/plate | 31.8412 | 39.4752 | -7.6340 | 32.2638 |
| context59/platform | 9.2000 | 11.4482 | -2.2482 | 13.8374 |
| context59/pottedplant | 49.8043 | 51.9620 | -2.1577 | 49.3159 |
| context59/road | 37.2111 | 35.5485 | +1.6626 | 33.5370 |
| context59/rock | 21.0625 | 22.2236 | -1.1611 | 43.0233 |
| context59/sheep | 85.6449 | 82.7748 | +2.8701 | 83.6483 |
| context59/shelves | 18.7586 | 24.2935 | -5.5349 | 23.3008 |
| context59/sidewalk | 9.4003 | 10.9879 | -1.5876 | 14.0155 |
| context59/sign | 42.5840 | 41.4765 | +1.1075 | 39.6430 |
| context59/sky | 81.5902 | 76.4618 | +5.1284 | 78.5545 |
| context59/snow | 64.2701 | 62.6514 | +1.6187 | 60.9538 |
| context59/sofa | 65.0447 | 64.1321 | +0.9126 | 63.8578 |
| context59/table | 46.5805 | 43.8505 | +2.7300 | 41.5012 |
| context59/track | 5.0192 | 8.6218 | -3.6026 | 14.2280 |
| context59/train | 49.3242 | 52.0964 | -2.7722 | 57.5569 |
| context59/tree | 64.1148 | 53.1130 | +11.0018 | 56.6758 |
| context59/truck | 23.4607 | 19.0877 | +4.3730 | 15.6009 |
| context59/tvmonitor | 50.2867 | 58.4174 | -8.1307 | 61.8621 |
| context59/wall | 30.6628 | 29.1542 | +1.5086 | 22.1983 |
| context59/water | 73.9754 | 76.4435 | -2.4681 | 73.7073 |
| context59/window | 36.7030 | 35.4660 | +1.2370 | 34.2209 |
| context59/wood | 15.2672 | 18.3864 | -3.1192 | 16.5166 |

## context60

Bank: `semantic_segmentation`; counts: `[401, 4, 5, 4, 4, 4, 3, 3, 4, 3, 5, 8, 6, 4, 4, 4, 4, 9, 8, 3, 6, 3, 4, 4, 4, 5, 5, 1, 1, 7, 2, 3, 4, 1, 3, 5, 2, 18, 5, 1, 3, 4, 4, 3, 2, 4, 1, 9, 4, 4, 3, 1, 5, 6, 4, 5, 6, 11, 2, 5]`.

VIP settings: `{"tau": 2.2, "tem": 1.0, "prob_thd": 0.1, "background": true, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| context60/background | 21.1851 | 14.7188 | +6.4663 | 13.8283 |
| context60/aeroplane | 46.8727 | 58.6131 | -11.7404 | 60.7257 |
| context60/bag | 29.6155 | 23.1581 | +6.4574 | 20.3955 |
| context60/bed | 10.6082 | 14.7997 | -4.1915 | 11.6869 |
| context60/bedclothes | 29.3743 | 25.3619 | +4.0124 | 28.5122 |
| context60/bench | 18.2378 | 29.9499 | -11.7121 | 32.2255 |
| context60/bicycle | 66.1364 | 43.6553 | +22.4811 | 30.4779 |
| context60/bird | 51.6946 | 64.0196 | -12.3250 | 67.5419 |
| context60/boat | 55.8018 | 48.6782 | +7.1236 | 45.4733 |
| context60/book | 9.0906 | 7.9688 | +1.1218 | 12.8905 |
| context60/bottle | 67.1933 | 38.8195 | +28.3738 | 30.0389 |
| context60/building | 37.1816 | 18.1169 | +19.0647 | 10.6117 |
| context60/bus | 74.3023 | 75.3987 | -1.0964 | 72.2377 |
| context60/cabinet | 38.1727 | 15.2347 | +22.9380 | 21.2025 |
| context60/car | 67.0412 | 55.8367 | +11.2045 | 28.2503 |
| context60/cat | 76.3283 | 77.4925 | -1.1642 | 71.8935 |
| context60/ceiling | 43.9380 | 40.9201 | +3.0179 | 42.2386 |
| context60/chair | 44.4900 | 44.3407 | +0.1493 | 46.3158 |
| context60/cloth | 17.8501 | 9.2220 | +8.6281 | 10.1527 |
| context60/computer | 15.4523 | 18.9113 | -3.4590 | 13.0882 |
| context60/cow | 78.7635 | 73.5746 | +5.1889 | 60.7891 |
| context60/cup | 27.5236 | 15.4413 | +12.0823 | 13.5088 |
| context60/curtain | 50.3095 | 50.3203 | -0.0108 | 50.8527 |
| context60/dog | 63.0602 | 76.1437 | -13.0835 | 74.7272 |
| context60/door | 26.8700 | 20.9085 | +5.9615 | 19.8233 |
| context60/fence | 30.2049 | 31.4144 | -1.2095 | 32.8490 |
| context60/floor | 47.0226 | 50.4973 | -3.4747 | 44.5223 |
| context60/flower | 18.2155 | 2.9950 | +15.2205 | 8.6537 |
| context60/food | 26.7004 | 11.4866 | +15.2138 | 5.5771 |
| context60/grass | 67.5062 | 47.1716 | +20.3346 | 49.3581 |
| context60/ground | 4.5260 | 1.3056 | +3.2204 | 0.8036 |
| context60/horse | 78.6718 | 71.8876 | +6.7842 | 70.0016 |
| context60/keyboard | 27.1064 | 23.0974 | +4.0090 | 41.3642 |
| context60/light | 13.5404 | 3.7716 | +9.7688 | 5.6855 |
| context60/motorbike | 72.1840 | 48.5594 | +23.6246 | 31.4983 |
| context60/mountain | 43.4988 | 25.2576 | +18.2412 | 39.6620 |
| context60/mouse | 43.9825 | 24.0140 | +19.9685 | 32.4650 |
| context60/person | 69.3020 | 56.3562 | +12.9458 | 45.8749 |
| context60/plate | 36.5697 | 45.0820 | -8.5123 | 44.4735 |
| context60/platform | 7.3692 | 12.1376 | -4.7684 | 12.7845 |
| context60/pottedplant | 50.4633 | 26.8914 | +23.5719 | 17.4811 |
| context60/road | 42.2402 | 32.7793 | +9.4609 | 25.7875 |
| context60/rock | 41.0284 | 19.5791 | +21.4493 | 37.1149 |
| context60/sheep | 81.0117 | 74.0138 | +6.9979 | 56.4962 |
| context60/shelves | 20.6856 | 20.1586 | +0.5270 | 19.3253 |
| context60/sidewalk | 12.4965 | 16.9057 | -4.4092 | 19.0265 |
| context60/sign | 32.9258 | 19.3842 | +13.5416 | 15.3521 |
| context60/sky | 75.1054 | 54.7447 | +20.3607 | 58.3009 |
| context60/snow | 58.9894 | 60.8220 | -1.8326 | 58.5386 |
| context60/sofa | 62.2704 | 61.3425 | +0.9279 | 60.7222 |
| context60/table | 46.9374 | 37.1240 | +9.8134 | 37.5903 |
| context60/track | 0.0002 | 0.0291 | -0.0289 | 0.3008 |
| context60/train | 41.8386 | 53.3451 | -11.5065 | 50.4541 |
| context60/tree | 58.1826 | 30.7534 | +27.4292 | 36.3577 |
| context60/truck | 15.8215 | 12.1091 | +3.7124 | 14.8651 |
| context60/tvmonitor | 41.7021 | 44.0462 | -2.3441 | 44.1512 |
| context60/wall | 38.3291 | 17.3213 | +21.0078 | 13.5686 |
| context60/water | 73.0353 | 60.1174 | +12.9179 | 69.9781 |
| context60/window | 29.8805 | 21.9049 | +7.9756 | 18.0673 |
| context60/wood | 19.5885 | 7.9654 | +11.6231 | 15.0104 |

## ade150

Bank: `semantic_segmentation`; counts: `[5, 8, 9, 5, 6, 6, 4, 1, 4, 4, 4, 5, 18, 5, 5, 3, 4, 5, 4, 4, 4, 5, 3, 5, 3, 1, 4, 1, 4, 1, 3, 1, 4, 4, 4, 5, 4, 3, 3, 1, 3, 1, 3, 3, 3, 2, 4, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 2, 2, 4, 1, 1, 1, 2, 3, 1, 3, 2, 3, 1, 5, 4, 1, 4, 1, 4, 1, 1, 4, 4, 1, 2, 4, 1, 2, 3, 2, 5, 4, 4, 1, 5, 1, 1, 5, 2, 6, 4, 3, 3, 1, 1, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 2, 3, 3, 3, 3, 1, 1, 2, 2, 2, 3, 2, 4, 3, 1, 3, 1, 2, 1, 3, 1, 3, 3, 1, 9, 1, 3, 3, 2, 4, 2, 1, 1, 1, 3, 1]`.

VIP settings: `{"tau": 5.0, "tem": 5.0, "prob_thd": 0.07, "background": true, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| ade150/wall | 37.1842 | 34.8189 | +2.3653 | 34.8189 |
| ade150/building | 62.7147 | 60.1792 | +2.5355 | 60.1792 |
| ade150/sky | 79.9823 | 69.1292 | +10.8531 | 69.1292 |
| ade150/floor | 60.3097 | 48.9899 | +11.3198 | 48.9899 |
| ade150/tree | 62.0770 | 45.0974 | +16.9796 | 45.0974 |
| ade150/ceiling | 55.7382 | 64.2957 | -8.5575 | 64.2957 |
| ade150/road | 70.0590 | 66.1513 | +3.9077 | 66.1513 |
| ade150/bed | 66.8879 | 61.0341 | +5.8538 | 61.0341 |
| ade150/windowpane | 41.5302 | 34.5766 | +6.9536 | 34.5766 |
| ade150/grass | 49.3493 | 46.4007 | +2.9486 | 46.4007 |
| ade150/cabinet | 47.2816 | 25.8457 | +21.4359 | 25.8457 |
| ade150/sidewalk | 50.3189 | 48.5600 | +1.7589 | 48.5600 |
| ade150/person | 60.6360 | 61.4546 | -0.8186 | 61.4546 |
| ade150/earth | 12.3410 | 6.6742 | +5.6668 | 6.6742 |
| ade150/door | 37.1846 | 36.2172 | +0.9674 | 36.2172 |
| ade150/table | 41.0312 | 34.3324 | +6.6988 | 34.3324 |
| ade150/mountain | 38.9642 | 32.8285 | +6.1357 | 32.8285 |
| ade150/plant | 37.6397 | 24.7201 | +12.9196 | 24.7201 |
| ade150/curtain | 64.0450 | 58.0846 | +5.9604 | 58.0846 |
| ade150/chair | 44.2905 | 24.5118 | +19.7787 | 24.5118 |
| ade150/car | 63.9871 | 63.0810 | +0.9061 | 63.0810 |
| ade150/water | 45.3726 | 37.8916 | +7.4810 | 37.8916 |
| ade150/painting | 29.6231 | 6.3906 | +23.2325 | 6.3906 |
| ade150/sofa | 50.6249 | 52.0296 | -1.4047 | 52.0296 |
| ade150/shelf | 32.5025 | 31.9228 | +0.5797 | 31.9228 |
| ade150/house | 21.0911 | 25.4648 | -4.3737 | 25.4648 |
| ade150/sea | 28.5027 | 25.4117 | +3.0910 | 25.4117 |
| ade150/mirror | 32.7875 | 27.5355 | +5.2520 | 27.5355 |
| ade150/rug | 33.4448 | 25.6054 | +7.8394 | 25.6054 |
| ade150/field | 18.9190 | 16.6488 | +2.2702 | 16.6488 |
| ade150/armchair | 10.0973 | 20.9182 | -10.8209 | 20.9182 |
| ade150/seat | 27.3993 | 18.2668 | +9.1325 | 18.2668 |
| ade150/fence | 25.0113 | 28.1535 | -3.1422 | 28.1535 |
| ade150/desk | 32.6296 | 34.1075 | -1.4779 | 34.1075 |
| ade150/rock | 33.5568 | 20.7643 | +12.7925 | 20.7643 |
| ade150/wardrobe | 45.7128 | 47.9260 | -2.2132 | 47.9260 |
| ade150/lamp | 30.7669 | 31.5679 | -0.8010 | 31.5679 |
| ade150/bathtub | 48.6249 | 58.1024 | -9.4775 | 58.1024 |
| ade150/railing | 19.4873 | 24.3447 | -4.8574 | 24.3447 |
| ade150/cushion | 37.6780 | 15.8751 | +21.8029 | 15.8751 |
| ade150/base | 3.3420 | 0.0344 | +3.3076 | 0.0344 |
| ade150/box | 23.1800 | 14.1677 | +9.0123 | 14.1677 |
| ade150/column | 37.3675 | 36.0005 | +1.3670 | 36.0005 |
| ade150/signboard | 26.5261 | 25.2221 | +1.3040 | 25.2221 |
| ade150/chest of drawers | 29.6321 | 35.3418 | -5.7097 | 35.3418 |
| ade150/counter | 37.8705 | 40.4719 | -2.6014 | 40.4719 |
| ade150/sand | 38.7158 | 47.6340 | -8.9182 | 47.6340 |
| ade150/sink | 40.4072 | 47.3571 | -6.9499 | 47.3571 |
| ade150/skyscraper | 22.5813 | 24.3355 | -1.7542 | 24.3355 |
| ade150/fireplace | 48.5427 | 53.3752 | -4.8325 | 53.3752 |
| ade150/refrigerator | 64.7569 | 60.8553 | +3.9016 | 60.8553 |
| ade150/grandstand | 23.9350 | 26.7897 | -2.8547 | 26.7897 |
| ade150/path | 6.1373 | 6.1815 | -0.0442 | 6.1815 |
| ade150/stairs | 34.9794 | 28.1353 | +6.8441 | 28.1353 |
| ade150/runway | 40.7494 | 29.9181 | +10.8313 | 29.9181 |
| ade150/case | 22.7332 | 33.3673 | -10.6341 | 33.3673 |
| ade150/pool table | 81.3563 | 79.9201 | +1.4362 | 79.9201 |
| ade150/pillow | 35.9296 | 21.1485 | +14.7811 | 21.1485 |
| ade150/screen door | 0.0388 | 0.0259 | +0.0129 | 0.0259 |
| ade150/stairway | 18.3541 | 14.5874 | +3.7667 | 14.5874 |
| ade150/river | 15.8254 | 16.9373 | -1.1119 | 16.9373 |
| ade150/bridge | 26.8326 | 25.9411 | +0.8915 | 25.9411 |
| ade150/bookcase | 17.2964 | 21.1093 | -3.8129 | 21.1093 |
| ade150/blind | 32.7460 | 35.2448 | -2.4988 | 35.2448 |
| ade150/coffee table | 49.8514 | 49.3487 | +0.5027 | 49.3487 |
| ade150/toilet | 53.5176 | 61.2265 | -7.7089 | 61.2265 |
| ade150/flower | 13.7263 | 1.0474 | +12.6789 | 1.0474 |
| ade150/book | 13.3161 | 2.4528 | +10.8633 | 2.4528 |
| ade150/hill | 5.4293 | 1.5212 | +3.9081 | 1.5212 |
| ade150/bench | 38.0810 | 35.1188 | +2.9622 | 35.1188 |
| ade150/countertop | 17.9637 | 18.9644 | -1.0007 | 18.9644 |
| ade150/stove | 26.4129 | 39.8129 | -13.4000 | 39.8129 |
| ade150/palm | 24.5032 | 26.1765 | -1.6733 | 26.1765 |
| ade150/kitchen island | 20.9910 | 19.0201 | +1.9709 | 19.0201 |
| ade150/computer | 40.7204 | 44.1654 | -3.4450 | 44.1654 |
| ade150/swivel chair | 9.9967 | 15.6096 | -5.6129 | 15.6096 |
| ade150/boat | 35.2331 | 34.4555 | +0.7776 | 34.4555 |
| ade150/bar | 22.4898 | 25.8413 | -3.3515 | 25.8413 |
| ade150/arcade machine | 12.5977 | 11.5484 | +1.0493 | 11.5484 |
| ade150/hovel | 11.0229 | 16.9199 | -5.8970 | 16.9199 |
| ade150/bus | 62.8865 | 67.0555 | -4.1690 | 67.0555 |
| ade150/towel | 55.2839 | 51.1505 | +4.1334 | 51.1505 |
| ade150/light | 7.5858 | 6.6191 | +0.9667 | 6.6191 |
| ade150/truck | 17.9129 | 22.4947 | -4.5818 | 22.4947 |
| ade150/tower | 14.7726 | 17.4999 | -2.7273 | 17.4999 |
| ade150/chandelier | 42.5879 | 42.6738 | -0.0859 | 42.6738 |
| ade150/awning | 21.8359 | 24.5103 | -2.6744 | 24.5103 |
| ade150/streetlight | 22.3405 | 15.7154 | +6.6251 | 15.7154 |
| ade150/booth | 8.9948 | 8.5942 | +0.4006 | 8.5942 |
| ade150/television receiver | 27.2471 | 34.0409 | -6.7938 | 34.0409 |
| ade150/airplane | 15.7427 | 35.5368 | -19.7941 | 35.5368 |
| ade150/dirt track | 2.2681 | 5.8868 | -3.6187 | 5.8868 |
| ade150/apparel | 5.9156 | 12.9073 | -6.9917 | 12.9073 |
| ade150/pole | 12.1618 | 14.9364 | -2.7746 | 14.9364 |
| ade150/land | 2.8619 | 0.9431 | +1.9188 | 0.9431 |
| ade150/bannister | 14.8594 | 11.4216 | +3.4378 | 11.4216 |
| ade150/escalator | 40.5218 | 40.0735 | +0.4483 | 40.0735 |
| ade150/ottoman | 37.1205 | 39.3174 | -2.1969 | 39.3174 |
| ade150/bottle | 32.6859 | 21.8970 | +10.7889 | 21.8970 |
| ade150/buffet | 7.7425 | 27.0374 | -19.2949 | 27.0374 |
| ade150/poster | 6.9058 | 5.2695 | +1.6363 | 5.2695 |
| ade150/stage | 7.9432 | 13.8156 | -5.8724 | 13.8156 |
| ade150/van | 36.9570 | 36.1778 | +0.7792 | 36.1778 |
| ade150/ship | 7.9054 | 8.3664 | -0.4610 | 8.3664 |
| ade150/fountain | 33.9808 | 45.8453 | -11.8645 | 45.8453 |
| ade150/conveyer belt | 45.7465 | 7.8554 | +37.8911 | 7.8554 |
| ade150/canopy | 17.5233 | 13.8933 | +3.6300 | 13.8933 |
| ade150/washer | 67.3043 | 36.1517 | +31.1526 | 36.1517 |
| ade150/plaything | 8.6042 | 5.4271 | +3.1771 | 5.4271 |
| ade150/swimming pool | 33.7992 | 31.1706 | +2.6286 | 31.1706 |
| ade150/stool | 24.6744 | 15.0488 | +9.6256 | 15.0488 |
| ade150/barrel | 29.2598 | 56.2696 | -27.0098 | 56.2696 |
| ade150/basket | 35.4013 | 35.4577 | -0.0564 | 35.4577 |
| ade150/waterfall | 21.3209 | 24.6026 | -3.2817 | 24.6026 |
| ade150/tent | 44.2048 | 42.8397 | +1.3651 | 42.8397 |
| ade150/bag | 25.3773 | 19.7155 | +5.6618 | 19.7155 |
| ade150/minibike | 63.5767 | 53.6269 | +9.9498 | 53.6269 |
| ade150/cradle | 49.2217 | 43.3196 | +5.9021 | 43.3196 |
| ade150/oven | 26.9018 | 34.0071 | -7.1053 | 34.0071 |
| ade150/ball | 19.8401 | 35.7320 | -15.8919 | 35.7320 |
| ade150/food | 42.5130 | 24.4435 | +18.0695 | 24.4435 |
| ade150/step | 2.8648 | 0.3474 | +2.5174 | 0.3474 |
| ade150/tank | 22.5529 | 8.0133 | +14.5396 | 8.0133 |
| ade150/trade name | 2.6907 | 2.1894 | +0.5013 | 2.1894 |
| ade150/microwave | 71.9927 | 64.0065 | +7.9862 | 64.0065 |
| ade150/pot | 35.4284 | 38.1864 | -2.7580 | 38.1864 |
| ade150/animal | 63.1460 | 58.8094 | +4.3366 | 58.8094 |
| ade150/bicycle | 46.5265 | 47.9723 | -1.4458 | 47.9723 |
| ade150/lake | 4.1765 | 6.6527 | -2.4762 | 6.6527 |
| ade150/dishwasher | 51.4241 | 28.2754 | +23.1487 | 28.2754 |
| ade150/screen | 36.9948 | 37.4722 | -0.4774 | 37.4722 |
| ade150/blanket | 11.1223 | 7.2378 | +3.8845 | 7.2378 |
| ade150/sculpture | 42.7199 | 46.4562 | -3.7363 | 46.4562 |
| ade150/hood | 33.9919 | 39.7265 | -5.7346 | 39.7265 |
| ade150/sconce | 20.5007 | 20.7517 | -0.2510 | 20.7517 |
| ade150/vase | 16.4284 | 14.1863 | +2.2421 | 14.1863 |
| ade150/traffic light | 20.5725 | 16.8219 | +3.7506 | 16.8219 |
| ade150/tray | 13.1233 | 10.0448 | +3.0785 | 10.0448 |
| ade150/ashcan | 13.9012 | 29.2209 | -15.3197 | 29.2209 |
| ade150/fan | 46.5336 | 35.8146 | +10.7190 | 35.8146 |
| ade150/pier | 3.8774 | 5.9731 | -2.0957 | 5.9731 |
| ade150/crt screen | 0.3714 | 0.3022 | +0.0692 | 0.3022 |
| ade150/plate | 34.3786 | 40.6058 | -6.2272 | 40.6058 |
| ade150/monitor | 18.7229 | 4.1025 | +14.6204 | 4.1025 |
| ade150/bulletin board | 6.5619 | 11.0734 | -4.5115 | 11.0734 |
| ade150/shower | 0.9691 | 1.1990 | -0.2299 | 1.1990 |
| ade150/radiator | 55.0117 | 52.0833 | +2.9284 | 52.0833 |
| ade150/glass | 8.7374 | 3.2398 | +5.4976 | 3.2398 |
| ade150/clock | 44.4135 | 21.8943 | +22.5192 | 21.8943 |
| ade150/flag | 55.8652 | 54.2080 | +1.6572 | 54.2080 |

## coco_object81

Bank: `official_imagenet`; counts: `[32, 14, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 6, 1, 1, 3, 1, 1, 1]`.

VIP settings: `{"tau": 6.0, "tem": 0.3, "prob_thd": 0.13, "background": true, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| coco_object81/background | 78.4969 | 80.4594 | -1.9625 | 81.1218 |
| coco_object81/person | 30.5662 | 58.0253 | -27.4591 | 66.0598 |
| coco_object81/bicycle | 59.5600 | 58.4978 | +1.0622 | 60.9191 |
| coco_object81/car | 34.9615 | 36.9466 | -1.9851 | 46.5366 |
| coco_object81/motorcycle | 75.2194 | 74.1251 | +1.0943 | 75.5922 |
| coco_object81/airplane | 59.3917 | 56.5034 | +2.8883 | 56.2385 |
| coco_object81/bus | 70.8357 | 72.4653 | -1.6296 | 73.4548 |
| coco_object81/train | 48.8879 | 64.6228 | -15.7349 | 63.0998 |
| coco_object81/truck | 41.1901 | 43.6005 | -2.4104 | 49.7471 |
| coco_object81/boat | 45.6119 | 49.9450 | -4.3331 | 49.5350 |
| coco_object81/traffic light | 20.8175 | 29.5740 | -8.7565 | 38.4214 |
| coco_object81/fire hydrant | 71.8424 | 66.2435 | +5.5989 | 68.4892 |
| coco_object81/stop sign | 39.2949 | 39.9851 | -0.6902 | 36.1821 |
| coco_object81/parking meter | 61.9053 | 55.1696 | +6.7357 | 43.8043 |
| coco_object81/bench | 38.8993 | 38.9256 | -0.0263 | 37.1161 |
| coco_object81/bird | 58.0272 | 55.2590 | +2.7682 | 59.4721 |
| coco_object81/cat | 80.4076 | 80.2516 | +0.1560 | 80.5686 |
| coco_object81/dog | 68.6549 | 65.6089 | +3.0460 | 73.1549 |
| coco_object81/horse | 70.7875 | 69.5357 | +1.2518 | 70.6005 |
| coco_object81/sheep | 80.6673 | 78.4667 | +2.2006 | 80.3604 |
| coco_object81/cow | 78.9477 | 73.7637 | +5.1840 | 76.2228 |
| coco_object81/elephant | 85.8584 | 84.8812 | +0.9772 | 85.2633 |
| coco_object81/bear | 81.2511 | 80.6980 | +0.5531 | 82.2705 |
| coco_object81/zebra | 86.3780 | 83.1227 | +3.2553 | 84.6382 |
| coco_object81/giraffe | 72.9500 | 73.3036 | -0.3536 | 76.0103 |
| coco_object81/backpack | 34.5770 | 30.4048 | +4.1722 | 30.5867 |
| coco_object81/umbrella | 77.6635 | 71.4981 | +6.1654 | 73.3071 |
| coco_object81/handbag | 40.6348 | 27.4392 | +13.1956 | 24.0236 |
| coco_object81/tie | 7.0701 | 8.2801 | -1.2100 | 8.7081 |
| coco_object81/suitcase | 62.9106 | 65.9459 | -3.0353 | 64.1597 |
| coco_object81/frisbee | 65.5646 | 50.1667 | +15.3979 | 46.7297 |
| coco_object81/skis | 26.2490 | 20.1338 | +6.1152 | 15.8767 |
| coco_object81/snowboard | 34.4714 | 36.4298 | -1.9584 | 22.3684 |
| coco_object81/sports ball | 25.2606 | 23.8072 | +1.4534 | 21.7798 |
| coco_object81/kite | 36.3375 | 38.5443 | -2.2068 | 40.3539 |
| coco_object81/baseball bat | 19.9199 | 21.0239 | -1.1040 | 12.4354 |
| coco_object81/baseball glove | 67.7804 | 62.0929 | +5.6875 | 63.5062 |
| coco_object81/skateboard | 16.7904 | 14.9712 | +1.8192 | 15.4085 |
| coco_object81/surfboard | 68.8998 | 64.3853 | +4.5145 | 65.5371 |
| coco_object81/tennis racket | 59.7305 | 54.8524 | +4.8781 | 54.9841 |
| coco_object81/bottle | 31.6962 | 28.7939 | +2.9023 | 37.3390 |
| coco_object81/wine glass | 49.4358 | 44.9157 | +4.5201 | 42.3648 |
| coco_object81/cup | 25.9533 | 22.6757 | +3.2776 | 23.8184 |
| coco_object81/fork | 36.9940 | 36.8633 | +0.1307 | 41.4354 |
| coco_object81/knife | 31.5128 | 26.2229 | +5.2899 | 24.5864 |
| coco_object81/spoon | 39.4681 | 36.7999 | +2.6682 | 38.1642 |
| coco_object81/bowl | 20.5100 | 19.6786 | +0.8314 | 21.0656 |
| coco_object81/banana | 56.4732 | 54.9267 | +1.5465 | 55.2990 |
| coco_object81/apple | 54.5282 | 52.0049 | +2.5233 | 53.1994 |
| coco_object81/sandwich | 34.2403 | 36.8842 | -2.6439 | 41.4584 |
| coco_object81/orange | 76.0882 | 74.5630 | +1.5252 | 73.1030 |
| coco_object81/broccoli | 56.7400 | 54.1429 | +2.5971 | 55.0654 |
| coco_object81/carrot | 54.0844 | 50.3192 | +3.7652 | 50.9897 |
| coco_object81/hot dog | 41.8737 | 42.5445 | -0.6708 | 47.1200 |
| coco_object81/pizza | 57.3303 | 57.5306 | -0.2003 | 60.6512 |
| coco_object81/donut | 41.0093 | 41.3054 | -0.2961 | 40.3985 |
| coco_object81/cake | 32.4226 | 35.4948 | -3.0722 | 33.6642 |
| coco_object81/chair | 27.8546 | 22.6866 | +5.1680 | 21.9885 |
| coco_object81/couch | 58.0652 | 54.6036 | +3.4616 | 54.9620 |
| coco_object81/potted plant | 25.7798 | 23.9146 | +1.8652 | 22.5977 |
| coco_object81/bed | 56.7126 | 0.0150 | +56.6976 | 0.0138 |
| coco_object81/dining table | 28.0638 | 30.3196 | -2.2558 | 31.7376 |
| coco_object81/toilet | 51.4335 | 46.4765 | +4.9570 | 46.3017 |
| coco_object81/tv | 33.4142 | 28.3256 | +5.0886 | 29.8473 |
| coco_object81/laptop | 52.8100 | 51.8221 | +0.9879 | 54.2117 |
| coco_object81/mouse | 71.0951 | 46.7711 | +24.3240 | 49.2470 |
| coco_object81/remote | 47.1827 | 48.3546 | -1.1719 | 50.8786 |
| coco_object81/keyboard | 62.4311 | 30.0685 | +32.3626 | 31.2719 |
| coco_object81/cell phone | 32.8066 | 35.8325 | -3.0259 | 37.6283 |
| coco_object81/microwave | 48.6815 | 49.3363 | -0.6548 | 47.2161 |
| coco_object81/oven | 41.8611 | 42.8477 | -0.9866 | 43.2177 |
| coco_object81/toaster | 73.1087 | 71.0561 | +2.0526 | 73.1906 |
| coco_object81/sink | 45.6466 | 41.9691 | +3.6775 | 42.3849 |
| coco_object81/refrigerator | 61.7637 | 69.7407 | -7.9770 | 69.4354 |
| coco_object81/book | 39.1545 | 23.6119 | +15.5426 | 26.4029 |
| coco_object81/clock | 65.6551 | 64.4021 | +1.2530 | 65.1494 |
| coco_object81/vase | 31.9337 | 33.5775 | -1.6438 | 31.1688 |
| coco_object81/scissors | 63.9161 | 66.0184 | -2.1023 | 65.9977 |
| coco_object81/teddy bear | 73.1671 | 72.2795 | +0.8876 | 73.8343 |
| coco_object81/hair drier | 24.6937 | 25.5565 | -0.8628 | 47.1047 |
| coco_object81/toothbrush | 39.5334 | 38.1726 | +1.3608 | 38.4776 |

## coco_stuff171

Bank: `official_segmentation`; counts: `[15, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 3, 1, 1, 1, 1, 1, 4, 1, 5, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 2, 2, 6, 1, 1, 3, 1, 1, 1, 1, 1, 6, 1, 5, 4, 1, 2, 3, 1, 3, 2, 6, 6, 2, 5, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 3, 3, 4, 4, 2, 5, 4, 4, 5, 5, 3, 3, 5, 2, 4, 6, 4, 4, 5, 1, 1, 4, 2, 1, 4, 1, 1, 6, 5, 1, 1, 3, 5, 5, 3, 1, 3, 7, 2, 6, 2, 1, 3, 1, 2, 1, 3, 1, 1, 4, 1, 3, 6, 2, 2, 3, 2, 1, 1, 1, 3, 4, 5, 1, 4]`.

VIP settings: `{"tau": 5.0, "tem": 10.0, "prob_thd": 0.05, "background": false, "bg_idx": 0, "resize_long_edge": 448, "slide_crop": 336, "slide_stride": 112, "logit_scale": 40.0}`.

| Class | Our IoU | VIP same text | Difference pp | VIP same words/official template |
| --- | ---: | ---: | ---: | ---: |
| coco_stuff171/person | 57.5809 | 64.3063 | -6.7254 | 58.7158 |
| coco_stuff171/bicycle | 63.3126 | 63.0549 | +0.2577 | 61.5557 |
| coco_stuff171/car | 44.0760 | 48.0155 | -3.9395 | 42.6711 |
| coco_stuff171/motorcycle | 72.4010 | 74.9446 | -2.5436 | 71.6700 |
| coco_stuff171/airplane | 44.3283 | 48.3673 | -4.0390 | 54.5248 |
| coco_stuff171/bus | 68.1715 | 70.7913 | -2.6198 | 69.6259 |
| coco_stuff171/train | 61.4632 | 58.7067 | +2.7565 | 66.7723 |
| coco_stuff171/truck | 45.0251 | 48.0586 | -3.0335 | 40.7544 |
| coco_stuff171/boat | 42.9889 | 40.7064 | +2.2825 | 43.8585 |
| coco_stuff171/traffic light | 33.1438 | 36.4864 | -3.3426 | 46.9170 |
| coco_stuff171/fire hydrant | 72.9176 | 71.9791 | +0.9385 | 73.6294 |
| coco_stuff171/stop sign | 29.6526 | 36.2386 | -6.5860 | 36.1166 |
| coco_stuff171/parking meter | 39.6603 | 44.0782 | -4.4179 | 45.3708 |
| coco_stuff171/bench | 37.5523 | 41.2030 | -3.6507 | 43.9844 |
| coco_stuff171/bird | 43.4981 | 46.2881 | -2.7900 | 46.6671 |
| coco_stuff171/cat | 78.1456 | 78.0664 | +0.0792 | 78.2158 |
| coco_stuff171/dog | 66.6913 | 70.5100 | -3.8187 | 72.8848 |
| coco_stuff171/horse | 73.8030 | 71.8051 | +1.9979 | 72.0603 |
| coco_stuff171/sheep | 80.1818 | 80.0505 | +0.1313 | 79.6475 |
| coco_stuff171/cow | 81.0151 | 77.1162 | +3.8989 | 71.2698 |
| coco_stuff171/elephant | 85.7154 | 83.8128 | +1.9026 | 81.9566 |
| coco_stuff171/bear | 86.3392 | 82.7207 | +3.6185 | 80.5390 |
| coco_stuff171/zebra | 84.6159 | 83.7204 | +0.8955 | 82.0703 |
| coco_stuff171/giraffe | 80.4785 | 76.8572 | +3.6213 | 76.0263 |
| coco_stuff171/backpack | 25.5907 | 26.8180 | -1.2273 | 30.7291 |
| coco_stuff171/umbrella | 74.7691 | 72.3473 | +2.4218 | 72.0527 |
| coco_stuff171/handbag | 32.3317 | 23.9315 | +8.4002 | 24.5374 |
| coco_stuff171/tie | 5.8119 | 6.5520 | -0.7401 | 5.8455 |
| coco_stuff171/suitcase | 63.9715 | 65.9443 | -1.9728 | 65.2466 |
| coco_stuff171/frisbee | 34.5490 | 44.6947 | -10.1457 | 30.8044 |
| coco_stuff171/skis | 18.6954 | 15.0628 | +3.6326 | 9.5744 |
| coco_stuff171/snowboard | 12.0402 | 17.2822 | -5.2420 | 12.5959 |
| coco_stuff171/sports ball | 5.7220 | 7.8376 | -2.1156 | 13.6101 |
| coco_stuff171/kite | 37.8577 | 31.7150 | +6.1427 | 32.2783 |
| coco_stuff171/baseball bat | 33.4065 | 19.9686 | +13.4379 | 26.9992 |
| coco_stuff171/baseball glove | 44.9972 | 60.4605 | -15.4633 | 61.6158 |
| coco_stuff171/skateboard | 17.3265 | 16.8572 | +0.4693 | 22.4274 |
| coco_stuff171/surfboard | 65.0422 | 57.0899 | +7.9523 | 63.4198 |
| coco_stuff171/tennis racket | 36.3872 | 21.4753 | +14.9119 | 47.5811 |
| coco_stuff171/bottle | 42.8993 | 41.4961 | +1.4032 | 41.0464 |
| coco_stuff171/wine glass | 48.8847 | 44.8520 | +4.0327 | 46.1536 |
| coco_stuff171/cup | 29.9140 | 26.5364 | +3.3776 | 26.8203 |
| coco_stuff171/fork | 40.0532 | 39.0543 | +0.9989 | 38.2880 |
| coco_stuff171/knife | 29.6198 | 24.6295 | +4.9903 | 27.1465 |
| coco_stuff171/spoon | 37.9700 | 38.7318 | -0.7618 | 35.2000 |
| coco_stuff171/bowl | 19.3120 | 17.1858 | +2.1262 | 15.7876 |
| coco_stuff171/banana | 42.6392 | 43.0246 | -0.3854 | 48.2219 |
| coco_stuff171/apple | 34.1342 | 45.3229 | -11.1887 | 46.5922 |
| coco_stuff171/sandwich | 34.4433 | 32.7938 | +1.6495 | 29.4927 |
| coco_stuff171/orange | 19.7292 | 24.0812 | -4.3520 | 50.5433 |
| coco_stuff171/broccoli | 58.7404 | 57.4380 | +1.3024 | 56.6833 |
| coco_stuff171/carrot | 41.0854 | 23.1555 | +17.9299 | 40.5291 |
| coco_stuff171/hot dog | 39.4482 | 39.0480 | +0.4002 | 39.2964 |
| coco_stuff171/pizza | 56.3030 | 56.8220 | -0.5190 | 56.4570 |
| coco_stuff171/donut | 61.4860 | 48.5484 | +12.9376 | 51.7971 |
| coco_stuff171/cake | 49.3841 | 38.1974 | +11.1867 | 45.1927 |
| coco_stuff171/chair | 32.9560 | 29.0632 | +3.8928 | 28.9058 |
| coco_stuff171/couch | 56.0970 | 56.7946 | -0.6976 | 55.6435 |
| coco_stuff171/potted plant | 22.9245 | 21.4923 | +1.4322 | 17.7472 |
| coco_stuff171/bed | 55.1228 | 48.1458 | +6.9770 | 38.9583 |
| coco_stuff171/dining table | 23.2100 | 22.1012 | +1.1088 | 19.9779 |
| coco_stuff171/toilet | 50.3888 | 52.2182 | -1.8294 | 44.8148 |
| coco_stuff171/tv | 22.8686 | 35.2837 | -12.4151 | 39.4489 |
| coco_stuff171/laptop | 59.9377 | 60.4523 | -0.5146 | 57.0127 |
| coco_stuff171/mouse | 46.0413 | 46.7282 | -0.6869 | 48.6730 |
| coco_stuff171/remote | 50.2981 | 56.8964 | -6.5983 | 51.1790 |
| coco_stuff171/keyboard | 57.1124 | 58.6725 | -1.5601 | 58.7696 |
| coco_stuff171/cell phone | 36.5486 | 39.4135 | -2.8649 | 30.6678 |
| coco_stuff171/microwave | 48.6009 | 53.4602 | -4.8593 | 57.7107 |
| coco_stuff171/oven | 38.1105 | 43.4660 | -5.3555 | 41.5959 |
| coco_stuff171/toaster | 47.0347 | 55.0230 | -7.9883 | 54.6232 |
| coco_stuff171/sink | 46.4514 | 47.8906 | -1.4392 | 44.2192 |
| coco_stuff171/refrigerator | 64.7964 | 68.3538 | -3.5574 | 69.3741 |
| coco_stuff171/book | 33.4470 | 28.3180 | +5.1290 | 31.0821 |
| coco_stuff171/clock | 67.5045 | 64.7415 | +2.7630 | 64.2323 |
| coco_stuff171/vase | 39.3780 | 31.6282 | +7.7498 | 33.7913 |
| coco_stuff171/scissors | 64.7500 | 65.6361 | -0.8861 | 65.8662 |
| coco_stuff171/teddy bear | 71.5062 | 71.8553 | -0.3491 | 70.4742 |
| coco_stuff171/hair drier | 42.5422 | 53.4573 | -10.9151 | 44.9652 |
| coco_stuff171/toothbrush | 30.8565 | 36.0549 | -5.1984 | 34.2818 |
| coco_stuff171/banner | 18.3424 | 22.7464 | -4.4040 | 22.5919 |
| coco_stuff171/blanket | 14.7191 | 0.0085 | +14.7106 | 0.0062 |
| coco_stuff171/branch | 3.9419 | 3.4026 | +0.5393 | 3.3532 |
| coco_stuff171/bridge | 32.4322 | 32.4201 | +0.0121 | 37.0942 |
| coco_stuff171/building-other | 40.5898 | 35.9167 | +4.6731 | 30.9056 |
| coco_stuff171/bush | 22.5058 | 23.2708 | -0.7650 | 23.0870 |
| coco_stuff171/cabinet | 24.1562 | 16.0453 | +8.1109 | 23.0254 |
| coco_stuff171/cage | 12.5920 | 18.1595 | -5.5675 | 16.4029 |
| coco_stuff171/cardboard | 47.8522 | 44.1002 | +3.7520 | 41.4919 |
| coco_stuff171/carpet | 45.8000 | 44.1657 | +1.6343 | 42.7332 |
| coco_stuff171/ceiling-other | 43.5931 | 43.4765 | +0.1166 | 45.5570 |
| coco_stuff171/ceiling-tile | 10.7145 | 16.4217 | -5.7072 | 14.5088 |
| coco_stuff171/cloth | 1.4253 | 2.6070 | -1.1817 | 2.7662 |
| coco_stuff171/clothes | 5.1107 | 5.5244 | -0.4137 | 5.3779 |
| coco_stuff171/clouds | 39.6868 | 39.9376 | -0.2508 | 42.0684 |
| coco_stuff171/counter | 12.1516 | 11.1022 | +1.0494 | 11.5221 |
| coco_stuff171/cupboard | 5.8880 | 7.0245 | -1.1365 | 5.8429 |
| coco_stuff171/curtain | 51.1676 | 49.9206 | +1.2470 | 55.1060 |
| coco_stuff171/desk-stuff | 24.9269 | 23.4336 | +1.4933 | 24.7665 |
| coco_stuff171/dirt | 4.0745 | 3.7633 | +0.3112 | 9.2112 |
| coco_stuff171/door-stuff | 35.2382 | 34.8279 | +0.4103 | 33.3696 |
| coco_stuff171/fence | 30.8998 | 26.6824 | +4.2174 | 28.6375 |
| coco_stuff171/floor-marble | 6.7468 | 7.4947 | -0.7479 | 7.8394 |
| coco_stuff171/floor-other | 14.0472 | 11.3453 | +2.7019 | 10.0547 |
| coco_stuff171/floor-stone | 4.8881 | 3.2201 | +1.6680 | 3.9338 |
| coco_stuff171/floor-tile | 45.6892 | 46.9311 | -1.2419 | 46.5204 |
| coco_stuff171/floor-wood | 34.8019 | 39.4737 | -4.6718 | 39.5506 |
| coco_stuff171/flower | 34.5559 | 32.8727 | +1.6832 | 32.2019 |
| coco_stuff171/fog | 8.2171 | 8.9558 | -0.7387 | 7.9127 |
| coco_stuff171/food-other | 14.6011 | 14.1791 | +0.4220 | 13.5533 |
| coco_stuff171/fruit | 19.6917 | 20.9444 | -1.2527 | 25.6902 |
| coco_stuff171/furniture-other | 1.3013 | 2.1648 | -0.8635 | 2.8380 |
| coco_stuff171/grass | 56.0949 | 44.4004 | +11.6945 | 50.0404 |
| coco_stuff171/gravel | 17.3015 | 14.8530 | +2.4485 | 15.7344 |
| coco_stuff171/ground-other | 5.1897 | 4.3045 | +0.8852 | 4.0659 |
| coco_stuff171/hill | 13.7304 | 11.5749 | +2.1555 | 9.8892 |
| coco_stuff171/house | 12.5723 | 13.1577 | -0.5854 | 12.2203 |
| coco_stuff171/leaves | 14.0270 | 12.2877 | +1.7393 | 8.5322 |
| coco_stuff171/light | 23.8591 | 21.3725 | +2.4866 | 18.5401 |
| coco_stuff171/mat | 6.9947 | 7.1160 | -0.1213 | 3.8579 |
| coco_stuff171/metal | 8.1187 | 4.3738 | +3.7449 | 5.0874 |
| coco_stuff171/mirror-stuff | 19.5749 | 21.7388 | -2.1639 | 22.9730 |
| coco_stuff171/moss | 8.1250 | 6.3386 | +1.7864 | 6.7278 |
| coco_stuff171/mountain | 42.9898 | 42.1245 | +0.8653 | 41.4530 |
| coco_stuff171/mud | 4.8737 | 8.2077 | -3.3340 | 6.7105 |
| coco_stuff171/napkin | 19.5786 | 19.4880 | +0.0906 | 17.4445 |
| coco_stuff171/net | 24.7054 | 17.6878 | +7.0176 | 25.1868 |
| coco_stuff171/paper | 25.2443 | 16.8471 | +8.3972 | 17.8236 |
| coco_stuff171/pavement | 39.3016 | 33.9701 | +5.3315 | 31.3009 |
| coco_stuff171/pillow | 2.7608 | 1.6917 | +1.0691 | 6.9230 |
| coco_stuff171/plant-other | 5.0735 | 4.6778 | +0.3957 | 8.0991 |
| coco_stuff171/plastic | 6.9078 | 1.2414 | +5.6664 | 3.8093 |
| coco_stuff171/platform | 8.1302 | 10.1165 | -1.9863 | 10.7339 |
| coco_stuff171/playingfield | 42.4445 | 25.6166 | +16.8279 | 23.0902 |
| coco_stuff171/railing | 4.1258 | 4.2414 | -0.1156 | 4.5584 |
| coco_stuff171/railroad | 46.0180 | 45.1877 | +0.8303 | 46.6091 |
| coco_stuff171/river | 24.9676 | 21.6314 | +3.3362 | 21.0443 |
| coco_stuff171/road | 41.2090 | 33.9774 | +7.2316 | 39.6227 |
| coco_stuff171/rock | 35.1781 | 19.1832 | +15.9949 | 33.9294 |
| coco_stuff171/roof | 14.8308 | 14.5612 | +0.2696 | 14.4839 |
| coco_stuff171/rug | 23.5546 | 26.9603 | -3.4057 | 26.8253 |
| coco_stuff171/salad | 8.7054 | 9.3027 | -0.5973 | 8.7765 |
| coco_stuff171/sand | 45.8609 | 41.3565 | +4.5044 | 55.5594 |
| coco_stuff171/sea | 74.8899 | 71.7297 | +3.1602 | 74.5402 |
| coco_stuff171/shelf | 22.7316 | 25.8243 | -3.0927 | 25.8698 |
| coco_stuff171/sky-other | 59.0040 | 51.7083 | +7.2957 | 51.2453 |
| coco_stuff171/skyscraper | 20.5779 | 17.5177 | +3.0602 | 16.7156 |
| coco_stuff171/snow | 78.6620 | 73.3852 | +5.2768 | 69.0111 |
| coco_stuff171/solid-other | 0.5720 | 1.0829 | -0.5109 | 1.4054 |
| coco_stuff171/stairs | 21.8787 | 18.9950 | +2.8837 | 21.7564 |
| coco_stuff171/stone | 3.0037 | 0.3583 | +2.6454 | 4.4959 |
| coco_stuff171/straw | 21.1509 | 16.7528 | +4.3981 | 22.1882 |
| coco_stuff171/structural-other | 0.0064 | 0.0000 | +0.0064 | 0.0000 |
| coco_stuff171/table | 2.5187 | 1.1771 | +1.3416 | 1.9135 |
| coco_stuff171/tent | 6.5271 | 6.7748 | -0.2477 | 7.3961 |
| coco_stuff171/textile-other | 2.7027 | 4.9413 | -2.2386 | 5.5785 |
| coco_stuff171/towel | 32.4807 | 33.0396 | -0.5589 | 34.4590 |
| coco_stuff171/tree | 34.9507 | 14.9005 | +20.0502 | 21.9417 |
| coco_stuff171/vegetable | 16.9679 | 22.5740 | -5.6061 | 21.1306 |
| coco_stuff171/wall-brick | 40.6537 | 39.0378 | +1.6159 | 38.1037 |
| coco_stuff171/wall-concrete | 3.3729 | 2.5793 | +0.7936 | 2.1104 |
| coco_stuff171/wall-other | 8.6951 | 6.8835 | +1.8116 | 6.2384 |
| coco_stuff171/wall-panel | 4.0234 | 2.9898 | +1.0336 | 3.1837 |
| coco_stuff171/wall-stone | 22.3365 | 21.7355 | +0.6010 | 21.4476 |
| coco_stuff171/wall-tile | 46.1225 | 41.6988 | +4.4237 | 36.9570 |
| coco_stuff171/wall-wood | 28.7199 | 25.6940 | +3.0259 | 24.2030 |
| coco_stuff171/water-other | 24.4989 | 21.9699 | +2.5290 | 22.5777 |
| coco_stuff171/waterdrops | 0.7850 | 0.6359 | +0.1491 | 0.4682 |
| coco_stuff171/window-blind | 13.6706 | 23.4924 | -9.8218 | 20.5162 |
| coco_stuff171/window-other | 29.3908 | 31.3151 | -1.9243 | 27.3746 |
| coco_stuff171/wood | 12.9317 | 8.9449 | +3.9868 | 17.3953 |
