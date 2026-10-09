# Kev vocabulary/readout development and frozen full evaluation

Status: all15 protocols complete;45200 protocol images verified.

Labeled development-selected configuration, not a label-free selector or untouched independent test. Full scores and non-development complement; ADE uses training development. Kev is an additional frozen pretrained language prior.

Kev judgments use taxonomy text only. Per-dataset thresholds, templates and readout settings were then selected with labeled development images. This is task-specific development, not proof of a universal adaptive or label-free selector. Original architecture/checkpoints preserved.

| Protocol | Images | Chosen | Previous final | Δ pp | VIP declared | VIP20 | Complement chosen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 80 | 60.4352 | 55.1305 | +5.3047 | 52.0647 | 51.4827 | 60.3729 |
| potsdam | 504 | 56.1934 | 49.7278 | +6.4656 | 44.8995 | 42.3028 | 55.6258 |
| voc20 | 1449 | 90.7871 | 92.1747 | -1.3876 | 92.5051 | not measured | 90.6361 |
| voc21 | 1449 | 69.4757 | 70.4508 | -0.9751 | 73.2591 | not measured | 69.3657 |
| ade150 | 2000 | 31.3398 | 31.1862 | +0.1536 | 29.1387 | not measured | 31.3398 |
| context60 | 5105 | 41.6004 | 41.6004 | +0.0000 | 42.5987 | not measured | 41.5898 |
| udd5 | 40 | 55.8266 | 48.3506 | +7.4760 | 44.9713 | 41.4336 | 55.8854 |
| oem | 384 | 45.1052 | 39.5582 | +5.5470 | 35.0388 | 33.2064 | 45.3432 |
| vaihingen | 113 | 58.8562 | 52.3535 | +6.5027 | 41.9062 | 47.362 | 58.7807 |
| landcoverai | 1602 | 65.5400 | 65.5954 | -0.0554 | 50.298 | 54.942 | 65.4455 |
| loveda P | 1669 | 69.6117 | 64.4340 | +5.1777 | 55.0261 | 56.311 | 69.4923 |
| loveda D | 1669 | 45.3171 | 40.3047 | +5.0124 | 35.3436 | 36.1457 | 45.2830 |
| context59 | 5105 | 46.3075 | 45.1338 | +1.1737 | not measured | not measured | 46.3075 |
| coco_object81 | 5000 | 50.3999 | 44.2396 | +6.1603 | 48.9955 | not measured | 50.3563 |
| coco_stuff171 | 5000 | 33.7661 | 32.8235 | +0.9426 | 33.4867 | not measured | 33.7527 |
| flair1 | 15700 | 43.6579 | 42.4460 | +1.2119 | 36.6074 | 35.2977 | 43.6545 |

## Admission control and method

Each alias is judged against the complete competing class taxonomy with four outcomes: specific, shared, wrong, unknown. The frozen judge probabilities are averaged over two option rotations. Keep the canonical class name; admit additional aliases when `p(specific) + 0.5*p(shared) >= t`, with development-selected `t` from0.4/0.6/0.8. Residual ontologies are protected. There is no per-class quota. Candidate words are the existing original, semantic and official vocabulary union; Kev judges and does not generate new words. Selected words affect both local and wide readouts. The inference architecture/checkpoints and bounded RGB observation budget remain fixed.

| Protocol | Chosen bank | Counts | Same-parameter unscreened | Admission Δ pp |
| --- | --- | --- | ---: | ---: |
| vdd/vdd | official_imagenet | [2, 1, 1, 2, 1, 1, 1] | 60.4352 | +0.0000 |
| potsdam/potsdam | official_imagenet | [2, 1, 4, 2, 1, 1] | 56.1934 | +0.0000 |
| voc20/voc20 | official_segmentation | [2, 1, 1, 1, 2, 3, 1, 1, 7, 4, 2, 1, 2, 2, 21, 4, 1, 2, 1, 4] | 90.7871 | +0.0000 |
| voc21/voc21 | semantic_segmentation | [56, 4, 3, 3, 4, 6, 6, 4, 4, 9, 7, 4, 4, 4, 3, 18, 5, 4, 5, 4, 6] | 69.4757 | +0.0000 |
| ade150/ade150 | semantic_segmentation | [5, 8, 9, 5, 6, 6, 4, 1, 4, 4, 4, 5, 18, 5, 5, 3, 4, 5, 4, 4, 4, 5, 3, 5, 3, 1, 4, 1, 4, 1, 3, 1, 4, 4, 4, 5, 4, 3, 3, 1, 3, 1, 3, 3, 3, 2, 4, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 2, 2, 4, 1, 1, 1, 2, 3, 1, 3, 2, 3, 1, 5, 4, 1, 4, 1, 4, 1, 1, 4, 4, 1, 2, 4, 1, 2, 3, 2, 5, 4, 4, 1, 5, 1, 1, 5, 2, 6, 4, 3, 3, 1, 1, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 2, 3, 3, 3, 3, 1, 1, 2, 2, 2, 3, 2, 4, 3, 1, 3, 1, 2, 1, 3, 1, 3, 3, 1, 9, 1, 3, 3, 2, 4, 2, 1, 1, 1, 3, 1] | 31.3398 | +0.0000 |
| context60/context60 | __previous_final__ | inherited previous policy | 41.6004 | +0.0000 |
| udd5/udd5 | kev60_segmentation | [7, 1, 2, 1, 20] | 53.7654 | +2.0612 |
| oem/oem | original | [20, 20, 20, 20, 20, 20, 20, 20] | 45.1052 | +0.0000 |
| vaihingen/vaihingen | kev40_imagenet | [16, 16, 17, 12, 8] | 57.2895 | +1.5667 |
| landcoverai/landcoverai | original_imagenet | [20, 20, 20, 20, 20] | 65.5400 | +0.0000 |
| loveda/P | original | [20, 20, 20, 20, 20, 20, 20] | 69.6117 | +0.0000 |
| loveda/D | original | [20, 20, 20, 20, 20, 20, 20] | 45.3171 | +0.0000 |
| context59/context59 | official_segmentation | [1, 3, 1, 3, 2, 1, 1, 1, 1, 3, 4, 1, 2, 2, 1, 1, 6, 8, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 4, 1, 2, 1, 1, 16, 5, 1, 3, 2, 1, 1, 2, 1, 1, 7, 1, 1, 1, 1, 2, 3, 1, 3, 3, 9, 2, 5] | 46.3075 | +0.0000 |
| coco_object81/coco_object81 | official_imagenet | [32, 14, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 6, 1, 1, 3, 1, 1, 1] | 50.3999 | +0.0000 |
| coco_stuff171/coco_stuff171 | official_segmentation | [15, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 3, 1, 1, 1, 1, 1, 4, 1, 5, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 2, 2, 6, 1, 1, 3, 1, 1, 1, 1, 1, 6, 1, 5, 4, 1, 2, 3, 1, 3, 2, 6, 6, 2, 5, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 3, 3, 4, 4, 2, 5, 4, 4, 5, 5, 3, 3, 5, 2, 4, 6, 4, 4, 5, 1, 1, 4, 2, 1, 4, 1, 1, 6, 5, 1, 1, 3, 5, 5, 3, 1, 3, 7, 2, 6, 2, 1, 3, 1, 2, 1, 3, 1, 1, 4, 1, 3, 6, 2, 2, 3, 2, 1, 1, 1, 3, 4, 5, 1, 4] | 33.7661 | +0.0000 |
| flair1/flair1 | original | [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20] | 43.6579 | +0.0000 |

Only a selected `kev*` bank has a true screened-versus-unscreened-pool control here; other banks repeat the selected readout and have zero admission delta.

Changing alias count also changes the inherited, unnormalized wide log-sum-exp class offset. A gain cannot automatically be attributed to removing semantically bad words. The same-parameter unscreened pool control is included. An original-bank or original-model winner is reported as such, not called a Kev gain.

The old-model replay is checked against exact historical confusion and target counts. CPU/GPU float32 text-mean roundoff may change a few predictions; the text error, confusion L1 and mIoU error are saved explicitly. Summary comparisons use the exact historical final-model result; no bit-exact replay is claimed when it differs.

VIP references are the verified local finite-row repaired reimplementations, not published benchmark claims. Official short vocabularies/settings are used where provided; other remote domains use external/distilled vocabularies. PC59 has no completed VIP full comparator and remains unfilled.

Cityscapes not included in the available frozen inventory; iSAID unlabeled mirror replaced by LandCover.ai. No fabricated metrics.

VDD and Potsdam chose official short banks, so their gains are vocabulary/readout development gains, not Kev admission gains. Parsing clarification from the matched-VIP control: these official_* banks split all commas, whereas pinned VIP splits only comma-space. They are normalized official-file words, not identical upstream query groups (VDD surface,other and grassland,forest area; Potsdam farmland,forest area). Only UDD5 and Vaihingen chose Kev banks. Negative VOC20/VOC21 results and the small LandCover.ai decline are retained. This run does not establish universal screening benefits or SOTA against published VIP numbers.

Full results reuse previously developed data. VOC20/21 and PC59/60 share RGB images: 45,200 is a protocol-image count, not a count of independent images. No architecture or frozen choice was replaced after observing full scores.

## Standalone online cost and offline language cost

| Protocol | Chosen median ms | p95 ms | Prior VIP median ms | Ratio | Peak MiB | Offline Kev seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 494.15 | 559.43 | 191.50 | 2.580 | 5993.5 | 7.84 |
| potsdam | 259.20 | 274.03 | 109.47 | 2.368 | 5515.8 | 8.05 |
| voc20 | 193.58 | 218.37 | 58.61 | 3.303 | 5514.3 | 27.84 |
| voc21 | 196.12 | 223.91 | 60.74 | 3.229 | 5517.0 | 27.40 |
| ade150 | 293.66 | 347.34 | 111.30 | 2.638 | 6433.2 | 219.94 |
| context60 | 259.70 | 262.92 | 112.81 | 2.302 | 5620.4 | 83.50 |
| udd5 | 500.18 | 526.92 | 187.57 | 2.667 | 5825.4 | 5.30 |
| oem | 249.53 | 255.02 | 108.07 | 2.309 | 5540.3 | 10.32 |
| vaihingen | 238.85 | 250.81 | 107.71 | 2.218 | 5519.8 | 6.70 |
| landcoverai | 233.39 | 237.80 | 106.84 | 2.184 | 5529.2 | 5.33 |
| loveda | 243.35 | 247.96 | 110.02 | 2.212 | 5532.1 | 7.84 |
| context59 | 247.86 | 251.45 | 111.35 | 2.226 | 5567.8 | 86.10 |
| coco_object81 | 283.66 | 287.52 | 132.91 | 2.134 | 5799.4 | 111.22 |
| coco_stuff171 | 366.10 | 368.10 | 183.69 | 1.993 | 6537.0 | 254.53 |
| flair1 | 247.00 | 253.10 | 109.08 | 2.264 | 5548.5 | 16.19 |

Seven deterministic whole inputs per protocol, two warmups and seven timed repeats; CUDA synchronized. Includes RGB resize, visual encoding, alias aggregation, coupled writeback, original-size restoration and CPU prediction. Excludes decode, initialization, text encoding, offline Kev judgment and labeled development search. This is a sampled warmed latency benchmark, not full-dataset mean latency. VIP measurements reuse earlier matched inputs/settings; PC59 has a timing reference but no full VIP accuracy reference. All measured forwards obey Geometry <=4, wide <=4, fine=0. Kev judgment time excludes checkpoint initialization and is paid offline; the 4B judge is not loaded for per-image segmentation.

## Foreground and non-residual outcomes

| Protocol | Metric | Chosen | Exact previous | VIP declared |
| --- | --- | ---: | ---: | ---: |
| vdd/vdd | non_residual_mean_iou_percent | 62.9550 | 58.4474 | 54.8116 |
| voc21/voc21 | foreground_mean_iou_percent | 68.4901 | 69.4894 | 72.3574 |
| context60/context60 | foreground_mean_iou_percent | 41.9465 | 41.9465 | 43.0323 |
| udd5/udd5 | non_residual_mean_iou_percent | 60.4556 | 52.0914 | 48.6691 |
| landcoverai/landcoverai | foreground_mean_iou_percent | 61.1921 | 61.0976 | 44.7117 |
| loveda/D | foreground_mean_iou_percent | 46.4307 | 45.9680 | 39.2398 |
| coco_object81/coco_object81 | foreground_mean_iou_percent | 50.0487 | 43.9101 | 48.5939 |

Eight remote domains (LoveDA D once): `{"selected": 53.86645, "previous": 49.1833375, "delta": 4.683112500000003, "vip": 42.6411875, "vip20": 42.7716125}`.

## vdd

Choice: `{"profile": {"bank": "official_imagenet", "strength": "original", "coupling": 1.0, "temperature": 0.07, "tau": 1.0, "tem": 0.3, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[2, 1, 1, 2, 1, 1, 1]`. Frozen Kev judgment compute 7.84 s (excludes checkpoint initialization); 242 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 494.15 ms; p95 559.43 ms; peak 5993.5 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/other | 45.3164 | 35.2289 | +10.0875 | 35.5829 | +9.7335 |
| vdd/wall | 48.0048 | 47.3161 | +0.6887 | 40.1862 | +7.8186 |
| vdd/road | 49.7305 | 41.8638 | +7.8667 | 40.5200 | +9.2105 |
| vdd/vegetation | 79.4081 | 66.4698 | +12.9383 | 52.5014 | +26.9067 |
| vdd/vehicle | 36.7663 | 28.7672 | +7.9991 | 25.2671 | +11.4992 |
| vdd/roof | 82.5380 | 83.2093 | -0.6713 | 81.6539 | +0.8841 |
| vdd/water | 81.2826 | 83.0581 | -1.7755 | 88.7412 | -7.4586 |

Prior measured VIP declared-setting median 191.50 ms; chosen/prior-VIP 2.580×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## potsdam

Choice: `{"profile": {"bank": "official_imagenet", "strength": 3.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.2}`.

Alias counts: `[2, 1, 4, 2, 1, 1]`. Frozen Kev judgment compute 8.05 s (excludes checkpoint initialization); 242 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 259.20 ms; p95 274.03 ms; peak 5515.8 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| potsdam/impervious surface | 65.6164 | 67.1860 | -1.5696 | 60.7547 | +4.8617 |
| potsdam/building | 77.9794 | 80.5867 | -2.6073 | 71.0553 | +6.9241 |
| potsdam/low vegetation | 60.0738 | 43.0884 | +16.9854 | 52.7964 | +7.2774 |
| potsdam/tree | 66.1729 | 61.9424 | +4.2305 | 56.8492 | +9.3237 |
| potsdam/car | 46.4693 | 36.6319 | +9.8374 | 16.8657 | +29.6036 |
| potsdam/clutter | 20.8484 | 8.9316 | +11.9168 | 11.0760 | +9.7724 |

Prior measured VIP declared-setting median 109.47 ms; chosen/prior-VIP 2.368×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## voc20

Choice: `{"profile": {"bank": "official_segmentation", "strength": 1.0, "coupling": 1.0, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[2, 1, 1, 1, 2, 3, 1, 1, 7, 4, 2, 1, 2, 2, 21, 4, 1, 2, 1, 4]`. Frozen Kev judgment compute 27.84 s (excludes checkpoint initialization); 852 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 193.58 ms; p95 218.37 ms; peak 5514.3 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| voc20/aeroplane | 98.7269 | 99.2713 | -0.5444 | 98.7858 | -0.0589 |
| voc20/bicycle | 90.6731 | 87.4408 | +3.2323 | 78.5410 | +12.1321 |
| voc20/bird | 88.1562 | 97.0117 | -8.8555 | 98.4408 | -10.2846 |
| voc20/boat | 95.1173 | 96.0777 | -0.9604 | 96.4059 | -1.2886 |
| voc20/bottle | 79.0976 | 82.5277 | -3.4301 | 88.8427 | -9.7451 |
| voc20/bus | 98.0618 | 97.1739 | +0.8879 | 98.4189 | -0.3571 |
| voc20/car | 92.6579 | 92.9760 | -0.3181 | 97.1708 | -4.5129 |
| voc20/cat | 98.5062 | 98.9997 | -0.4935 | 98.7458 | -0.2396 |
| voc20/chair | 72.2228 | 68.3467 | +3.8761 | 69.9349 | +2.2879 |
| voc20/cow | 97.1445 | 98.7528 | -1.6083 | 99.1495 | -2.0050 |
| voc20/diningtable | 69.7871 | 80.8718 | -11.0847 | 77.8509 | -8.0638 |
| voc20/dog | 96.8559 | 97.5571 | -0.7012 | 97.2815 | -0.4256 |
| voc20/horse | 96.8034 | 97.5286 | -0.7252 | 97.1777 | -0.3743 |
| voc20/motorbike | 92.7933 | 89.5460 | +3.2473 | 91.3165 | +1.4768 |
| voc20/person | 82.7224 | 88.9842 | -6.2618 | 89.1567 | -6.4343 |
| voc20/pottedplant | 94.6293 | 92.8874 | +1.7419 | 97.2341 | -2.6048 |
| voc20/sheep | 96.0959 | 99.1383 | -3.0424 | 98.7102 | -2.6143 |
| voc20/sofa | 87.0033 | 88.1771 | -1.1738 | 87.9112 | -0.9079 |
| voc20/train | 99.0004 | 97.5485 | +1.4519 | 99.2316 | -0.2312 |
| voc20/tvmonitor | 89.6874 | 92.6770 | -2.9896 | 89.7960 | -0.1086 |

Prior measured VIP declared-setting median 58.61 ms; chosen/prior-VIP 3.303×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## voc21

Choice: `{"profile": {"bank": "semantic_segmentation", "strength": 2.0, "coupling": 1.0, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": -1.0, "background_threshold": 0.2}`.

Alias counts: `[56, 4, 3, 3, 4, 6, 6, 4, 4, 9, 7, 4, 4, 4, 3, 18, 5, 4, 5, 4, 6]`. Frozen Kev judgment compute 27.40 s (excludes checkpoint initialization); 854 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 196.12 ms; p95 223.91 ms; peak 5517.0 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| voc21/background | 89.1870 | 89.6771 | -0.4901 | 91.2911 | -2.1041 |
| voc21/aeroplane | 69.5204 | 64.8191 | +4.7013 | 70.6965 | -1.1761 |
| voc21/bicycle | 45.5807 | 45.9316 | -0.3509 | 46.2336 | -0.6529 |
| voc21/bird | 89.7051 | 83.0559 | +6.6492 | 73.4782 | +16.2269 |
| voc21/boat | 42.0732 | 59.2355 | -17.1623 | 64.0359 | -21.9627 |
| voc21/bottle | 55.0314 | 54.4634 | +0.5680 | 65.5658 | -10.5344 |
| voc21/bus | 84.9358 | 87.2961 | -2.3603 | 88.9076 | -3.9718 |
| voc21/car | 71.5430 | 69.6258 | +1.9172 | 69.6774 | +1.8656 |
| voc21/cat | 89.5052 | 89.5423 | -0.0371 | 89.6118 | -0.1066 |
| voc21/chair | 47.2028 | 47.1759 | +0.0269 | 50.9725 | -3.7697 |
| voc21/cow | 84.3789 | 91.3160 | -6.9371 | 88.4002 | -4.0213 |
| voc21/diningtable | 57.1652 | 56.3292 | +0.8360 | 50.6817 | +6.4835 |
| voc21/dog | 87.7978 | 83.7058 | +4.0920 | 88.1563 | -0.3585 |
| voc21/horse | 85.8893 | 87.7509 | -1.8616 | 86.6532 | -0.7639 |
| voc21/motorbike | 59.0673 | 72.0045 | -12.9372 | 80.2067 | -21.1394 |
| voc21/person | 69.4979 | 72.2169 | -2.7190 | 79.4624 | -9.9645 |
| voc21/pottedplant | 55.6155 | 49.5259 | +6.0896 | 58.1966 | -2.5811 |
| voc21/sheep | 91.6522 | 89.6548 | +1.9974 | 89.8704 | +1.7818 |
| voc21/sofa | 68.9450 | 65.6431 | +3.3019 | 67.4159 | +1.5291 |
| voc21/train | 54.2061 | 60.2222 | -6.0161 | 75.8286 | -21.6225 |
| voc21/tvmonitor | 60.4897 | 60.2738 | +0.2159 | 63.0978 | -2.6081 |

Prior measured VIP declared-setting median 60.74 ms; chosen/prior-VIP 3.229×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## ade150

Choice: `{"profile": {"bank": "semantic_segmentation", "strength": "original", "coupling": 0.25, "temperature": 0.14, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[5, 8, 9, 5, 6, 6, 4, 1, 4, 4, 4, 5, 18, 5, 5, 3, 4, 5, 4, 4, 4, 5, 3, 5, 3, 1, 4, 1, 4, 1, 3, 1, 4, 4, 4, 5, 4, 3, 3, 1, 3, 1, 3, 3, 3, 2, 4, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 2, 2, 2, 4, 1, 1, 1, 2, 3, 1, 3, 2, 3, 1, 5, 4, 1, 4, 1, 4, 1, 1, 4, 4, 1, 2, 4, 1, 2, 3, 2, 5, 4, 4, 1, 5, 1, 1, 5, 2, 6, 4, 3, 3, 1, 1, 1, 1, 3, 2, 2, 2, 1, 1, 2, 2, 1, 2, 3, 3, 3, 3, 1, 1, 2, 2, 2, 3, 2, 4, 3, 1, 3, 1, 2, 1, 3, 1, 3, 3, 1, 9, 1, 3, 3, 2, 4, 2, 1, 1, 1, 3, 1]`. Frozen Kev judgment compute 219.94 s (excludes checkpoint initialization); 6092 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 293.66 ms; p95 347.34 ms; peak 6433.2 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| ade150/wall | 37.1842 | 37.3253 | -0.1411 | 33.6049 | +3.5793 |
| ade150/building | 62.7147 | 62.9592 | -0.2445 | 50.9390 | +11.7757 |
| ade150/sky | 79.9823 | 80.1794 | -0.1971 | 71.5092 | +8.4731 |
| ade150/floor | 60.3097 | 60.2454 | +0.0643 | 54.1747 | +6.1350 |
| ade150/tree | 62.0770 | 62.0207 | +0.0563 | 50.2204 | +11.8566 |
| ade150/ceiling | 55.7382 | 55.6719 | +0.0663 | 58.0268 | -2.2886 |
| ade150/road | 70.0590 | 69.8901 | +0.1689 | 58.3584 | +11.7006 |
| ade150/bed | 66.8879 | 66.3467 | +0.5412 | 72.2027 | -5.3148 |
| ade150/windowpane | 41.5302 | 41.6596 | -0.1294 | 26.2347 | +15.2955 |
| ade150/grass | 49.3493 | 50.5323 | -1.1830 | 24.5573 | +24.7920 |
| ade150/cabinet | 47.2816 | 47.4834 | -0.2018 | 13.8135 | +33.4681 |
| ade150/sidewalk | 50.3189 | 50.0149 | +0.3040 | 43.6731 | +6.6458 |
| ade150/person | 60.6360 | 61.8620 | -1.2260 | 65.8754 | -5.2394 |
| ade150/earth | 12.3410 | 12.8921 | -0.5511 | 5.1821 | +7.1589 |
| ade150/door | 37.1846 | 37.2379 | -0.0533 | 32.1803 | +5.0043 |
| ade150/table | 41.0312 | 40.9099 | +0.1213 | 33.3149 | +7.7163 |
| ade150/mountain | 38.9642 | 39.4936 | -0.5294 | 32.2016 | +6.7626 |
| ade150/plant | 37.6397 | 37.9407 | -0.3010 | 23.2837 | +14.3560 |
| ade150/curtain | 64.0450 | 63.6802 | +0.3648 | 60.9081 | +3.1369 |
| ade150/chair | 44.2905 | 44.1368 | +0.1537 | 21.5236 | +22.7669 |
| ade150/car | 63.9871 | 63.7777 | +0.2094 | 58.3539 | +5.6332 |
| ade150/water | 45.3726 | 46.0791 | -0.7065 | 38.0562 | +7.3164 |
| ade150/painting | 29.6231 | 29.1571 | +0.4660 | 2.9117 | +26.7114 |
| ade150/sofa | 50.6249 | 49.9858 | +0.6391 | 34.0010 | +16.6239 |
| ade150/shelf | 32.5025 | 32.6892 | -0.1867 | 25.6425 | +6.8600 |
| ade150/house | 21.0911 | 20.1022 | +0.9889 | 22.0073 | -0.9162 |
| ade150/sea | 28.5027 | 28.3612 | +0.1415 | 29.7409 | -1.2382 |
| ade150/mirror | 32.7875 | 32.2911 | +0.4964 | 29.3583 | +3.4292 |
| ade150/rug | 33.4448 | 33.4445 | +0.0003 | 55.8700 | -22.4252 |
| ade150/field | 18.9190 | 19.2950 | -0.3760 | 15.8208 | +3.0982 |
| ade150/armchair | 10.0973 | 9.7383 | +0.3590 | 16.0391 | -5.9418 |
| ade150/seat | 27.3993 | 25.0714 | +2.3279 | 24.2219 | +3.1774 |
| ade150/fence | 25.0113 | 24.8270 | +0.1843 | 31.8857 | -6.8744 |
| ade150/desk | 32.6296 | 32.4501 | +0.1795 | 32.2176 | +0.4120 |
| ade150/rock | 33.5568 | 33.2688 | +0.2880 | 12.2054 | +21.3514 |
| ade150/wardrobe | 45.7128 | 45.8662 | -0.1534 | 32.4940 | +13.2188 |
| ade150/lamp | 30.7669 | 30.5940 | +0.1729 | 31.4436 | -0.6767 |
| ade150/bathtub | 48.6249 | 48.0472 | +0.5777 | 57.4341 | -8.8092 |
| ade150/railing | 19.4873 | 19.4972 | -0.0099 | 28.6014 | -9.1141 |
| ade150/cushion | 37.6780 | 33.1600 | +4.5180 | 25.9376 | +11.7404 |
| ade150/base | 3.3420 | 3.1573 | +0.1847 | 11.0130 | -7.6710 |
| ade150/box | 23.1800 | 22.2996 | +0.8804 | 16.3459 | +6.8341 |
| ade150/column | 37.3675 | 37.9400 | -0.5725 | 32.4128 | +4.9547 |
| ade150/signboard | 26.5261 | 26.5231 | +0.0030 | 23.6085 | +2.9176 |
| ade150/chest of drawers | 29.6321 | 29.6543 | -0.0222 | 33.4508 | -3.8187 |
| ade150/counter | 37.8705 | 38.0033 | -0.1328 | 34.4847 | +3.3858 |
| ade150/sand | 38.7158 | 38.6898 | +0.0260 | 31.4002 | +7.3156 |
| ade150/sink | 40.4072 | 40.0990 | +0.3082 | 48.0042 | -7.5970 |
| ade150/skyscraper | 22.5813 | 21.8831 | +0.6982 | 27.5566 | -4.9753 |
| ade150/fireplace | 48.5427 | 48.6242 | -0.0815 | 49.1975 | -0.6548 |
| ade150/refrigerator | 64.7569 | 64.3650 | +0.3919 | 69.3618 | -4.6049 |
| ade150/grandstand | 23.9350 | 24.2368 | -0.3018 | 34.3236 | -10.3886 |
| ade150/path | 6.1373 | 6.0593 | +0.0780 | 5.4519 | +0.6854 |
| ade150/stairs | 34.9794 | 34.9303 | +0.0491 | 24.9586 | +10.0208 |
| ade150/runway | 40.7494 | 39.2958 | +1.4536 | 31.5672 | +9.1822 |
| ade150/case | 22.7332 | 23.1101 | -0.3769 | 29.0650 | -6.3318 |
| ade150/pool table | 81.3563 | 81.2019 | +0.1544 | 80.1147 | +1.2416 |
| ade150/pillow | 35.9296 | 35.3421 | +0.5875 | 24.2932 | +11.6364 |
| ade150/screen door | 0.0388 | 0.0445 | -0.0057 | 25.3420 | -25.3032 |
| ade150/stairway | 18.3541 | 18.4301 | -0.0760 | 5.2061 | +13.1480 |
| ade150/river | 15.8254 | 15.4090 | +0.4164 | 16.9605 | -1.1351 |
| ade150/bridge | 26.8326 | 27.4070 | -0.5744 | 22.6622 | +4.1704 |
| ade150/bookcase | 17.2964 | 17.2000 | +0.0964 | 21.2196 | -3.9232 |
| ade150/blind | 32.7460 | 32.2193 | +0.5267 | 31.8235 | +0.9225 |
| ade150/coffee table | 49.8514 | 49.7018 | +0.1496 | 46.4380 | +3.4134 |
| ade150/toilet | 53.5176 | 53.1572 | +0.3604 | 58.6530 | -5.1354 |
| ade150/flower | 13.7263 | 12.9713 | +0.7550 | 7.2161 | +6.5102 |
| ade150/book | 13.3161 | 13.0109 | +0.3052 | 2.8344 | +10.4817 |
| ade150/hill | 5.4293 | 5.3212 | +0.1081 | 1.5756 | +3.8537 |
| ade150/bench | 38.0810 | 38.0570 | +0.0240 | 34.6372 | +3.4438 |
| ade150/countertop | 17.9637 | 18.3208 | -0.3571 | 17.5772 | +0.3865 |
| ade150/stove | 26.4129 | 26.0226 | +0.3903 | 29.8495 | -3.4366 |
| ade150/palm | 24.5032 | 24.3528 | +0.1504 | 21.7076 | +2.7956 |
| ade150/kitchen island | 20.9910 | 21.5936 | -0.6026 | 14.4231 | +6.5679 |
| ade150/computer | 40.7204 | 40.5404 | +0.1800 | 56.9835 | -16.2631 |
| ade150/swivel chair | 9.9967 | 8.6743 | +1.3224 | 24.0413 | -14.0446 |
| ade150/boat | 35.2331 | 34.4732 | +0.7599 | 42.7689 | -7.5358 |
| ade150/bar | 22.4898 | 23.2187 | -0.7289 | 26.3874 | -3.8976 |
| ade150/arcade machine | 12.5977 | 11.9737 | +0.6240 | 12.9519 | -0.3542 |
| ade150/hovel | 11.0229 | 10.7100 | +0.3129 | 16.5820 | -5.5591 |
| ade150/bus | 62.8865 | 61.9450 | +0.9415 | 66.5772 | -3.6907 |
| ade150/towel | 55.2839 | 55.2035 | +0.0804 | 50.6688 | +4.6151 |
| ade150/light | 7.5858 | 7.3185 | +0.2673 | 1.0767 | +6.5091 |
| ade150/truck | 17.9129 | 17.5602 | +0.3527 | 24.9793 | -7.0664 |
| ade150/tower | 14.7726 | 15.9063 | -1.1337 | 12.1470 | +2.6256 |
| ade150/chandelier | 42.5879 | 42.7096 | -0.1217 | 43.2300 | -0.6421 |
| ade150/awning | 21.8359 | 22.1805 | -0.3446 | 24.2781 | -2.4422 |
| ade150/streetlight | 22.3405 | 22.0518 | +0.2887 | 14.8397 | +7.5008 |
| ade150/booth | 8.9948 | 8.5967 | +0.3981 | 11.7137 | -2.7189 |
| ade150/television receiver | 27.2471 | 26.6117 | +0.6354 | 50.3859 | -23.1388 |
| ade150/airplane | 15.7427 | 15.6370 | +0.1057 | 46.1580 | -30.4153 |
| ade150/dirt track | 2.2681 | 2.5262 | -0.2581 | 3.0693 | -0.8012 |
| ade150/apparel | 5.9156 | 6.2818 | -0.3662 | 11.6094 | -5.6938 |
| ade150/pole | 12.1618 | 13.1073 | -0.9455 | 14.2534 | -2.0916 |
| ade150/land | 2.8619 | 2.8655 | -0.0036 | 1.5839 | +1.2780 |
| ade150/bannister | 14.8594 | 14.5678 | +0.2916 | 8.8627 | +5.9967 |
| ade150/escalator | 40.5218 | 40.7057 | -0.1839 | 26.2465 | +14.2753 |
| ade150/ottoman | 37.1205 | 36.1429 | +0.9776 | 35.3837 | +1.7368 |
| ade150/bottle | 32.6859 | 32.8267 | -0.1408 | 21.4942 | +11.1917 |
| ade150/buffet | 7.7425 | 8.0749 | -0.3324 | 17.9028 | -10.1603 |
| ade150/poster | 6.9058 | 6.8099 | +0.0959 | 10.6086 | -3.7028 |
| ade150/stage | 7.9432 | 8.6387 | -0.6955 | 11.7086 | -3.7654 |
| ade150/van | 36.9570 | 37.7073 | -0.7503 | 29.4580 | +7.4990 |
| ade150/ship | 7.9054 | 7.8980 | +0.0074 | 7.8074 | +0.0980 |
| ade150/fountain | 33.9808 | 33.7591 | +0.2217 | 46.3913 | -12.4105 |
| ade150/conveyer belt | 45.7465 | 45.4572 | +0.2893 | 1.1773 | +44.5692 |
| ade150/canopy | 17.5233 | 17.5745 | -0.0512 | 24.7804 | -7.2571 |
| ade150/washer | 67.3043 | 65.9225 | +1.3818 | 49.0956 | +18.2087 |
| ade150/plaything | 8.6042 | 8.7554 | -0.1512 | 4.1830 | +4.4212 |
| ade150/swimming pool | 33.7992 | 34.3988 | -0.5996 | 28.9062 | +4.8930 |
| ade150/stool | 24.6744 | 24.0736 | +0.6008 | 15.0101 | +9.6643 |
| ade150/barrel | 29.2598 | 28.9321 | +0.3277 | 53.8995 | -24.6397 |
| ade150/basket | 35.4013 | 35.8607 | -0.4594 | 32.8277 | +2.5736 |
| ade150/waterfall | 21.3209 | 21.6488 | -0.3279 | 24.8013 | -3.4804 |
| ade150/tent | 44.2048 | 44.2451 | -0.0403 | 40.1862 | +4.0186 |
| ade150/bag | 25.3773 | 25.1346 | +0.2427 | 19.4450 | +5.9323 |
| ade150/minibike | 63.5767 | 63.3895 | +0.1872 | 60.3257 | +3.2510 |
| ade150/cradle | 49.2217 | 47.4411 | +1.7806 | 41.4424 | +7.7793 |
| ade150/oven | 26.9018 | 27.2376 | -0.3358 | 43.3981 | -16.4963 |
| ade150/ball | 19.8401 | 20.8735 | -1.0334 | 29.8872 | -10.0471 |
| ade150/food | 42.5130 | 41.0184 | +1.4946 | 30.1175 | +12.3955 |
| ade150/step | 2.8648 | 2.8769 | -0.0121 | 4.7751 | -1.9103 |
| ade150/tank | 22.5529 | 21.2511 | +1.3018 | 0.0000 | +22.5529 |
| ade150/trade name | 2.6907 | 2.6622 | +0.0285 | 5.0025 | -2.3118 |
| ade150/microwave | 71.9927 | 71.1240 | +0.8687 | 70.5793 | +1.4134 |
| ade150/pot | 35.4284 | 35.7577 | -0.3293 | 10.3428 | +25.0856 |
| ade150/animal | 63.1460 | 62.4980 | +0.6480 | 57.4733 | +5.6727 |
| ade150/bicycle | 46.5265 | 46.5675 | -0.0410 | 50.4003 | -3.8738 |
| ade150/lake | 4.1765 | 3.6004 | +0.5761 | 6.3345 | -2.1580 |
| ade150/dishwasher | 51.4241 | 50.6453 | +0.7788 | 31.5145 | +19.9096 |
| ade150/screen | 36.9948 | 37.7049 | -0.7101 | 27.3365 | +9.6583 |
| ade150/blanket | 11.1223 | 10.3351 | +0.7872 | 27.1230 | -16.0007 |
| ade150/sculpture | 42.7199 | 43.8291 | -1.1092 | 49.9499 | -7.2300 |
| ade150/hood | 33.9919 | 33.6148 | +0.3771 | 49.3380 | -15.3461 |
| ade150/sconce | 20.5007 | 20.6106 | -0.1099 | 18.9622 | +1.5385 |
| ade150/vase | 16.4284 | 16.0901 | +0.3383 | 15.8009 | +0.6275 |
| ade150/traffic light | 20.5725 | 20.6410 | -0.0685 | 19.0139 | +1.5586 |
| ade150/tray | 13.1233 | 13.2434 | -0.1201 | 10.1659 | +2.9574 |
| ade150/ashcan | 13.9012 | 12.9070 | +0.9942 | 21.7705 | -7.8693 |
| ade150/fan | 46.5336 | 44.6165 | +1.9171 | 41.2839 | +5.2497 |
| ade150/pier | 3.8774 | 3.8598 | +0.0176 | 12.7750 | -8.8976 |
| ade150/crt screen | 0.3714 | 0.3820 | -0.0106 | 0.7872 | -0.4158 |
| ade150/plate | 34.3786 | 34.3556 | +0.0230 | 28.4089 | +5.9697 |
| ade150/monitor | 18.7229 | 18.4030 | +0.3199 | 6.3062 | +12.4167 |
| ade150/bulletin board | 6.5619 | 6.5556 | +0.0063 | 5.1615 | +1.4004 |
| ade150/shower | 0.9691 | 1.0226 | -0.0535 | 0.0000 | +0.9691 |
| ade150/radiator | 55.0117 | 56.0047 | -0.9930 | 50.0881 | +4.9236 |
| ade150/glass | 8.7374 | 7.9863 | +0.7511 | 4.5640 | +4.1734 |
| ade150/clock | 44.4135 | 44.3758 | +0.0377 | 20.4810 | +23.9325 |
| ade150/flag | 55.8652 | 56.4051 | -0.5399 | 52.6647 | +3.2005 |

Prior measured VIP declared-setting median 111.30 ms; chosen/prior-VIP 2.638×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## context60

Choice: `{"profile": {"bank": "__previous_final__", "strength": 2.0, "coupling": 1.0, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "residual_protected"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `null`. Frozen Kev judgment compute 83.50 s (excludes checkpoint initialization); 2422 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 259.70 ms; p95 262.92 ms; peak 5620.4 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| context60/background | 21.1851 | 21.1851 | +0.0000 | 17.0214 | +4.1637 |
| context60/aeroplane | 46.8727 | 46.8727 | +0.0000 | 59.9399 | -13.0672 |
| context60/bag | 29.6155 | 29.6155 | +0.0000 | 30.6280 | -1.0125 |
| context60/bed | 10.6082 | 10.6082 | +0.0000 | 8.4253 | +2.1829 |
| context60/bedclothes | 29.3743 | 29.3743 | +0.0000 | 26.2230 | +3.1513 |
| context60/bench | 18.2378 | 18.2378 | +0.0000 | 28.9763 | -10.7385 |
| context60/bicycle | 66.1364 | 66.1364 | +0.0000 | 65.6614 | +0.4750 |
| context60/bird | 51.6946 | 51.6946 | +0.0000 | 50.6011 | +1.0935 |
| context60/boat | 55.8018 | 55.8018 | +0.0000 | 60.0529 | -4.2511 |
| context60/book | 9.0906 | 9.0906 | +0.0000 | 12.0033 | -2.9127 |
| context60/bottle | 67.1933 | 67.1933 | +0.0000 | 58.8178 | +8.3755 |
| context60/building | 37.1816 | 37.1816 | +0.0000 | 19.8263 | +17.3553 |
| context60/bus | 74.3023 | 74.3023 | +0.0000 | 77.1473 | -2.8450 |
| context60/cabinet | 38.1727 | 38.1727 | +0.0000 | 37.2566 | +0.9161 |
| context60/car | 67.0412 | 67.0412 | +0.0000 | 61.6446 | +5.3966 |
| context60/cat | 76.3283 | 76.3283 | +0.0000 | 83.9833 | -7.6550 |
| context60/ceiling | 43.9380 | 43.9380 | +0.0000 | 46.6205 | -2.6825 |
| context60/chair | 44.4900 | 44.4900 | +0.0000 | 44.8920 | -0.4020 |
| context60/cloth | 17.8501 | 17.8501 | +0.0000 | 18.8909 | -1.0408 |
| context60/computer | 15.4523 | 15.4523 | +0.0000 | 21.8456 | -6.3933 |
| context60/cow | 78.7635 | 78.7635 | +0.0000 | 81.6405 | -2.8770 |
| context60/cup | 27.5236 | 27.5236 | +0.0000 | 21.4151 | +6.1085 |
| context60/curtain | 50.3095 | 50.3095 | +0.0000 | 51.5946 | -1.2851 |
| context60/dog | 63.0602 | 63.0602 | +0.0000 | 81.0355 | -17.9753 |
| context60/door | 26.8700 | 26.8700 | +0.0000 | 25.4135 | +1.4565 |
| context60/fence | 30.2049 | 30.2049 | +0.0000 | 36.0234 | -5.8185 |
| context60/floor | 47.0226 | 47.0226 | +0.0000 | 43.4426 | +3.5800 |
| context60/flower | 18.2155 | 18.2155 | +0.0000 | 24.3442 | -6.1287 |
| context60/food | 26.7004 | 26.7004 | +0.0000 | 30.1804 | -3.4800 |
| context60/grass | 67.5062 | 67.5062 | +0.0000 | 64.3105 | +3.1957 |
| context60/ground | 4.5260 | 4.5260 | +0.0000 | 9.4880 | -4.9620 |
| context60/horse | 78.6718 | 78.6718 | +0.0000 | 79.5262 | -0.8544 |
| context60/keyboard | 27.1064 | 27.1064 | +0.0000 | 57.1077 | -30.0013 |
| context60/light | 13.5404 | 13.5404 | +0.0000 | 18.0883 | -4.5479 |
| context60/motorbike | 72.1840 | 72.1840 | +0.0000 | 74.6608 | -2.4768 |
| context60/mountain | 43.4988 | 43.4989 | -0.0001 | 44.2328 | -0.7340 |
| context60/mouse | 43.9825 | 43.9825 | +0.0000 | 25.4688 | +18.5137 |
| context60/person | 69.3020 | 69.3020 | +0.0000 | 76.2353 | -6.9333 |
| context60/plate | 36.5697 | 36.5697 | +0.0000 | 23.8684 | +12.7013 |
| context60/platform | 7.3692 | 7.3692 | +0.0000 | 13.0139 | -5.6447 |
| context60/pottedplant | 50.4633 | 50.4633 | +0.0000 | 45.7282 | +4.7351 |
| context60/road | 42.2402 | 42.2402 | +0.0000 | 32.6118 | +9.6284 |
| context60/rock | 41.0284 | 41.0284 | +0.0000 | 42.9908 | -1.9624 |
| context60/sheep | 81.0117 | 81.0117 | +0.0000 | 82.2482 | -1.2365 |
| context60/shelves | 20.6856 | 20.6856 | +0.0000 | 18.9536 | +1.7320 |
| context60/sidewalk | 12.4965 | 12.4965 | +0.0000 | 13.8311 | -1.3346 |
| context60/sign | 32.9258 | 32.9258 | +0.0000 | 35.2234 | -2.2976 |
| context60/sky | 75.1054 | 75.1054 | +0.0000 | 75.9825 | -0.8771 |
| context60/snow | 58.9894 | 58.9894 | +0.0000 | 58.3262 | +0.6632 |
| context60/sofa | 62.2704 | 62.2704 | +0.0000 | 61.7744 | +0.4960 |
| context60/table | 46.9374 | 46.9374 | +0.0000 | 39.5401 | +7.3973 |
| context60/track | 0.0002 | 0.0002 | +0.0000 | 14.4066 | -14.4064 |
| context60/train | 41.8386 | 41.8386 | +0.0000 | 55.8805 | -14.0419 |
| context60/tree | 58.1826 | 58.1826 | +0.0000 | 54.7273 | +3.4553 |
| context60/truck | 15.8215 | 15.8215 | +0.0000 | 15.2330 | +0.5885 |
| context60/tvmonitor | 41.7021 | 41.7021 | +0.0000 | 56.6260 | -14.9239 |
| context60/wall | 38.3291 | 38.3291 | +0.0000 | 21.3076 | +17.0215 |
| context60/water | 73.0353 | 73.0353 | +0.0000 | 70.4771 | +2.5582 |
| context60/window | 29.8805 | 29.8805 | +0.0000 | 32.7139 | -2.8334 |
| context60/wood | 19.5885 | 19.5885 | +0.0000 | 15.7940 | +3.7945 |

Prior measured VIP declared-setting median 112.81 ms; chosen/prior-VIP 2.302×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## udd5

Choice: `{"profile": {"bank": "kev60_segmentation", "strength": "original", "coupling": 0.5, "temperature": 0.05, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": -1.0, "background_threshold": 0.3}`.

Alias counts: `[7, 1, 2, 1, 20]`. Frozen Kev judgment compute 5.30 s (excludes checkpoint initialization); 152 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 500.18 ms; p95 526.92 ms; peak 5825.4 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| udd5/vegetation | 79.2404 | 68.2122 | +11.0282 | 67.0805 | +12.1599 |
| udd5/building | 85.3634 | 81.7249 | +3.6385 | 79.9297 | +5.4337 |
| udd5/road | 37.3689 | 42.0333 | -4.6644 | 32.1817 | +5.1872 |
| udd5/vehicle | 39.8496 | 16.3953 | +23.4543 | 15.4845 | +24.3651 |
| udd5/other | 37.3109 | 33.3873 | +3.9236 | 30.1803 | +7.1306 |

Prior measured VIP declared-setting median 187.57 ms; chosen/prior-VIP 2.667×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## oem

Choice: `{"profile": {"bank": "original", "strength": 1.0, "coupling": 0.25, "temperature": 0.03, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[20, 20, 20, 20, 20, 20, 20, 20]`. Frozen Kev judgment compute 10.32 s (excludes checkpoint initialization); 304 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 249.53 ms; p95 255.02 ms; peak 5540.3 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| oem/bareland | 10.4191 | 9.6171 | +0.8020 | 6.6615 | +3.7576 |
| oem/rangeland | 32.0863 | 20.4012 | +11.6851 | 12.3559 | +19.7304 |
| oem/developed space | 27.5374 | 28.6598 | -1.1224 | 22.1125 | +5.4249 |
| oem/road | 38.0586 | 36.7057 | +1.3529 | 29.2458 | +8.8128 |
| oem/tree | 54.9661 | 44.1703 | +10.7958 | 33.1620 | +21.8041 |
| oem/water | 68.9790 | 69.4172 | -0.4382 | 64.2911 | +4.6879 |
| oem/agriculture land | 66.0801 | 57.4537 | +8.6264 | 68.4754 | -2.3953 |
| oem/building | 62.7148 | 50.0407 | +12.6741 | 44.0064 | +18.7084 |

Prior measured VIP declared-setting median 108.07 ms; chosen/prior-VIP 2.309×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## vaihingen

Choice: `{"profile": {"bank": "kev40_imagenet", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[16, 16, 17, 12, 8]`. Frozen Kev judgment compute 6.70 s (excludes checkpoint initialization); 196 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 238.85 ms; p95 250.81 ms; peak 5519.8 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vaihingen/impervious surface | 66.0449 | 61.6239 | +4.4210 | 45.4945 | +20.5504 |
| vaihingen/building | 72.7879 | 72.3801 | +0.4078 | 60.4878 | +12.3001 |
| vaihingen/low vegetation | 40.6235 | 35.3293 | +5.2942 | 22.1971 | +18.4264 |
| vaihingen/tree | 69.5008 | 68.0322 | +1.4686 | 56.9623 | +12.5385 |
| vaihingen/car | 45.3241 | 24.4020 | +20.9221 | 24.3892 | +20.9349 |

Prior measured VIP declared-setting median 107.71 ms; chosen/prior-VIP 2.218×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## landcoverai

Choice: `{"profile": {"bank": "original_imagenet", "strength": 2.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.3}`.

Alias counts: `[20, 20, 20, 20, 20]`. Frozen Kev judgment compute 5.33 s (excludes checkpoint initialization); 152 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 233.39 ms; p95 237.80 ms; peak 5529.2 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| landcoverai/background | 82.9318 | 83.5869 | -0.6551 | 72.6430 | +10.2888 |
| landcoverai/building | 60.9726 | 49.9808 | +10.9918 | 50.8202 | +10.1524 |
| landcoverai/woodland | 77.1198 | 77.2998 | -0.1800 | 49.3624 | +27.7574 |
| landcoverai/water | 64.1632 | 76.9689 | -12.8057 | 38.9371 | +25.2261 |
| landcoverai/road | 42.5126 | 40.1409 | +2.3717 | 39.7273 | +2.7853 |

Prior measured VIP declared-setting median 106.84 ms; chosen/prior-VIP 2.184×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## loveda

Choice: `{"profile": {"bank": "original", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.2}`.

Alias counts: `[20, 20, 20, 20, 20, 20, 20]`. Frozen Kev judgment compute 7.84 s (excludes checkpoint initialization); 228 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 243.35 ms; p95 247.96 ms; peak 5532.1 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| P/building | 88.4885 | 89.7022 | -1.2137 | 79.3618 | +9.1267 |
| P/road | 65.2094 | 70.0351 | -4.8257 | 61.2826 | +3.9268 |
| P/water | 75.6226 | 72.2122 | +3.4104 | 62.3187 | +13.3039 |
| P/barren | 43.9516 | 39.0996 | +4.8520 | 33.9002 | +10.0514 |
| P/tree | 63.1547 | 40.9863 | +22.1684 | 21.6331 | +41.5216 |
| P/farm | 81.2434 | 74.5687 | +6.6747 | 71.6600 | +9.5834 |
| D/background | 38.6355 | 6.3249 | +32.3106 | 11.9665 | +26.6690 |
| D/building | 56.6110 | 52.6895 | +3.9215 | 46.5717 | +10.0393 |
| D/road | 52.3713 | 52.0423 | +0.3290 | 51.7150 | +0.6563 |
| D/water | 45.6562 | 61.2619 | -15.6057 | 49.7909 | -4.1347 |
| D/barren | 25.3890 | 28.1006 | -2.7116 | 21.9642 | +3.4248 |
| D/tree | 37.7511 | 34.6204 | +3.1307 | 17.7414 | +20.0097 |
| D/farm | 60.8057 | 47.0932 | +13.7125 | 47.6557 | +13.1500 |

Prior measured VIP declared-setting median 110.02 ms; chosen/prior-VIP 2.212×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## context59

Choice: `{"profile": {"bank": "official_segmentation", "strength": 3.0, "coupling": 0.75, "temperature": 0.07, "tau": 1.0, "tem": 10.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[1, 3, 1, 3, 2, 1, 1, 1, 1, 3, 4, 1, 2, 2, 1, 1, 6, 8, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 2, 4, 1, 2, 1, 1, 16, 5, 1, 3, 2, 1, 1, 2, 1, 1, 7, 1, 1, 1, 1, 2, 3, 1, 3, 3, 9, 2, 5]`. Frozen Kev judgment compute 86.10 s (excludes checkpoint initialization); 2418 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 247.86 ms; p95 251.45 ms; peak 5567.8 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| context59/aeroplane | 67.3383 | 45.9148 | +21.4235 | not measured | not measured |
| context59/bag | 44.8271 | 46.2192 | -1.3921 | not measured | not measured |
| context59/bed | 6.8257 | 12.0683 | -5.2426 | not measured | not measured |
| context59/bedclothes | 26.9829 | 30.1800 | -3.1971 | not measured | not measured |
| context59/bench | 18.2440 | 17.2077 | +1.0363 | not measured | not measured |
| context59/bicycle | 69.3990 | 67.4389 | +1.9601 | not measured | not measured |
| context59/bird | 70.3675 | 64.0619 | +6.3056 | not measured | not measured |
| context59/boat | 68.6008 | 57.9108 | +10.6900 | not measured | not measured |
| context59/book | 6.5114 | 9.7892 | -3.2778 | not measured | not measured |
| context59/bottle | 71.2428 | 74.4220 | -3.1792 | not measured | not measured |
| context59/building | 36.9821 | 40.0331 | -3.0510 | not measured | not measured |
| context59/bus | 80.2299 | 75.6961 | +4.5338 | not measured | not measured |
| context59/cabinet | 36.8496 | 40.6516 | -3.8020 | not measured | not measured |
| context59/car | 64.8208 | 68.1628 | -3.3420 | not measured | not measured |
| context59/cat | 87.9852 | 81.5895 | +6.3957 | not measured | not measured |
| context59/ceiling | 48.0722 | 45.9536 | +2.1186 | not measured | not measured |
| context59/chair | 43.2269 | 47.1607 | -3.9338 | not measured | not measured |
| context59/cloth | 16.9566 | 21.0622 | -4.1056 | not measured | not measured |
| context59/computer | 26.6571 | 21.4125 | +5.2446 | not measured | not measured |
| context59/cow | 83.9745 | 79.5071 | +4.4674 | not measured | not measured |
| context59/cup | 31.5851 | 43.5801 | -11.9950 | not measured | not measured |
| context59/curtain | 57.4631 | 49.1481 | +8.3150 | not measured | not measured |
| context59/dog | 82.2504 | 68.5610 | +13.6894 | not measured | not measured |
| context59/door | 27.4833 | 27.6190 | -0.1357 | not measured | not measured |
| context59/fence | 38.4962 | 33.7289 | +4.7673 | not measured | not measured |
| context59/floor | 50.5707 | 49.9022 | +0.6685 | not measured | not measured |
| context59/flower | 24.5721 | 20.8343 | +3.7378 | not measured | not measured |
| context59/food | 50.2279 | 37.6058 | +12.6221 | not measured | not measured |
| context59/grass | 71.6279 | 69.8380 | +1.7899 | not measured | not measured |
| context59/ground | 12.1732 | 4.7906 | +7.3826 | not measured | not measured |
| context59/horse | 83.9426 | 79.3161 | +4.6265 | not measured | not measured |
| context59/keyboard | 52.1329 | 32.1188 | +20.0141 | not measured | not measured |
| context59/light | 24.1319 | 13.9203 | +10.2116 | not measured | not measured |
| context59/motorbike | 73.6571 | 74.6576 | -1.0005 | not measured | not measured |
| context59/mountain | 35.5850 | 43.1859 | -7.6009 | not measured | not measured |
| context59/mouse | 62.1596 | 51.3202 | +10.8394 | not measured | not measured |
| context59/person | 66.1820 | 72.8506 | -6.6686 | not measured | not measured |
| context59/plate | 31.8412 | 41.2347 | -9.3935 | not measured | not measured |
| context59/platform | 9.2000 | 7.3115 | +1.8885 | not measured | not measured |
| context59/pottedplant | 49.8043 | 55.4874 | -5.6831 | not measured | not measured |
| context59/road | 37.2111 | 42.4205 | -5.2094 | not measured | not measured |
| context59/rock | 21.0625 | 42.0975 | -21.0350 | not measured | not measured |
| context59/sheep | 85.6449 | 80.5793 | +5.0656 | not measured | not measured |
| context59/shelves | 18.7586 | 24.7405 | -5.9819 | not measured | not measured |
| context59/sidewalk | 9.4003 | 11.9407 | -2.5404 | not measured | not measured |
| context59/sign | 42.5840 | 42.0746 | +0.5094 | not measured | not measured |
| context59/sky | 81.5902 | 76.8040 | +4.7862 | not measured | not measured |
| context59/snow | 64.2701 | 61.6620 | +2.6081 | not measured | not measured |
| context59/sofa | 65.0447 | 66.0248 | -0.9801 | not measured | not measured |
| context59/table | 46.5805 | 51.5159 | -4.9354 | not measured | not measured |
| context59/track | 5.0192 | 0.0002 | +5.0190 | not measured | not measured |
| context59/train | 49.3242 | 42.7732 | +6.5510 | not measured | not measured |
| context59/tree | 64.1148 | 60.6261 | +3.4887 | not measured | not measured |
| context59/truck | 23.4607 | 18.8669 | +4.5938 | not measured | not measured |
| context59/tvmonitor | 50.2867 | 47.0427 | +3.2440 | not measured | not measured |
| context59/wall | 30.6628 | 40.4935 | -9.8307 | not measured | not measured |
| context59/water | 73.9754 | 73.5905 | +0.3849 | not measured | not measured |
| context59/window | 36.7030 | 34.3746 | +2.3284 | not measured | not measured |
| context59/wood | 15.2672 | 21.8147 | -6.5475 | not measured | not measured |

Prior measured VIP declared-setting median 111.35 ms; chosen/prior-VIP 2.226×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## coco_object81

Choice: `{"profile": {"bank": "official_imagenet", "strength": 1.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": -4.0, "background_threshold": 0.05}`.

Alias counts: `[32, 14, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 6, 1, 1, 3, 1, 1, 1]`. Frozen Kev judgment compute 111.22 s (excludes checkpoint initialization); 3112 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 283.66 ms; p95 287.52 ms; peak 5799.4 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| coco_object81/background | 78.4969 | 70.5970 | +7.8999 | 81.1218 | -2.6249 |
| coco_object81/person | 30.5662 | 65.9064 | -35.3402 | 66.0598 | -35.4936 |
| coco_object81/bicycle | 59.5600 | 59.3661 | +0.1939 | 60.9191 | -1.3591 |
| coco_object81/car | 34.9615 | 41.9558 | -6.9943 | 46.5366 | -11.5751 |
| coco_object81/motorcycle | 75.2194 | 71.0537 | +4.1657 | 75.5922 | -0.3728 |
| coco_object81/airplane | 59.3917 | 35.2395 | +24.1522 | 56.2385 | +3.1532 |
| coco_object81/bus | 70.8357 | 61.1795 | +9.6562 | 73.4548 | -2.6191 |
| coco_object81/train | 48.8879 | 37.5608 | +11.3271 | 63.0998 | -14.2119 |
| coco_object81/truck | 41.1901 | 42.3441 | -1.1540 | 49.7471 | -8.5570 |
| coco_object81/boat | 45.6119 | 19.1325 | +26.4794 | 49.5350 | -3.9231 |
| coco_object81/traffic light | 20.8175 | 8.6200 | +12.1975 | 38.4214 | -17.6039 |
| coco_object81/fire hydrant | 71.8424 | 37.1474 | +34.6950 | 68.4892 | +3.3532 |
| coco_object81/stop sign | 39.2949 | 17.3440 | +21.9509 | 36.1821 | +3.1128 |
| coco_object81/parking meter | 61.9053 | 23.4712 | +38.4341 | 43.8043 | +18.1010 |
| coco_object81/bench | 38.8993 | 24.8159 | +14.0834 | 37.1161 | +1.7832 |
| coco_object81/bird | 58.0272 | 43.1870 | +14.8402 | 59.4721 | -1.4449 |
| coco_object81/cat | 80.4076 | 73.9729 | +6.4347 | 80.5686 | -0.1610 |
| coco_object81/dog | 68.6549 | 41.9836 | +26.6713 | 73.1549 | -4.5000 |
| coco_object81/horse | 70.7875 | 70.4722 | +0.3153 | 70.6005 | +0.1870 |
| coco_object81/sheep | 80.6673 | 69.2813 | +11.3860 | 80.3604 | +0.3069 |
| coco_object81/cow | 78.9477 | 66.1398 | +12.8079 | 76.2228 | +2.7249 |
| coco_object81/elephant | 85.8584 | 82.4181 | +3.4403 | 85.2633 | +0.5951 |
| coco_object81/bear | 81.2511 | 80.8578 | +0.3933 | 82.2705 | -1.0194 |
| coco_object81/zebra | 86.3780 | 83.4742 | +2.9038 | 84.6382 | +1.7398 |
| coco_object81/giraffe | 72.9500 | 78.5911 | -5.6411 | 76.0103 | -3.0603 |
| coco_object81/backpack | 34.5770 | 25.0550 | +9.5220 | 30.5867 | +3.9903 |
| coco_object81/umbrella | 77.6635 | 67.9278 | +9.7357 | 73.3071 | +4.3564 |
| coco_object81/handbag | 40.6348 | 30.2870 | +10.3478 | 24.0236 | +16.6112 |
| coco_object81/tie | 7.0701 | 8.9293 | -1.8592 | 8.7081 | -1.6380 |
| coco_object81/suitcase | 62.9106 | 64.8792 | -1.9686 | 64.1597 | -1.2491 |
| coco_object81/frisbee | 65.5646 | 42.3862 | +23.1784 | 46.7297 | +18.8349 |
| coco_object81/skis | 26.2490 | 5.4224 | +20.8266 | 15.8767 | +10.3723 |
| coco_object81/snowboard | 34.4714 | 2.0664 | +32.4050 | 22.3684 | +12.1030 |
| coco_object81/sports ball | 25.2606 | 1.3155 | +23.9451 | 21.7798 | +3.4808 |
| coco_object81/kite | 36.3375 | 47.3393 | -11.0018 | 40.3539 | -4.0164 |
| coco_object81/baseball bat | 19.9199 | 36.9413 | -17.0214 | 12.4354 | +7.4845 |
| coco_object81/baseball glove | 67.7804 | 34.9038 | +32.8766 | 63.5062 | +4.2742 |
| coco_object81/skateboard | 16.7904 | 13.5204 | +3.2700 | 15.4085 | +1.3819 |
| coco_object81/surfboard | 68.8998 | 20.5989 | +48.3009 | 65.5371 | +3.3627 |
| coco_object81/tennis racket | 59.7305 | 31.0550 | +28.6755 | 54.9841 | +4.7464 |
| coco_object81/bottle | 31.6962 | 43.5886 | -11.8924 | 37.3390 | -5.6428 |
| coco_object81/wine glass | 49.4358 | 47.8599 | +1.5759 | 42.3648 | +7.0710 |
| coco_object81/cup | 25.9533 | 33.6312 | -7.6779 | 23.8184 | +2.1349 |
| coco_object81/fork | 36.9940 | 40.9899 | -3.9959 | 41.4354 | -4.4414 |
| coco_object81/knife | 31.5128 | 33.3404 | -1.8276 | 24.5864 | +6.9264 |
| coco_object81/spoon | 39.4681 | 36.2665 | +3.2016 | 38.1642 | +1.3039 |
| coco_object81/bowl | 20.5100 | 32.5721 | -12.0621 | 21.0656 | -0.5556 |
| coco_object81/banana | 56.4732 | 53.5640 | +2.9092 | 55.2990 | +1.1742 |
| coco_object81/apple | 54.5282 | 44.5738 | +9.9544 | 53.1994 | +1.3288 |
| coco_object81/sandwich | 34.2403 | 35.4063 | -1.1660 | 41.4584 | -7.2181 |
| coco_object81/orange | 76.0882 | 72.0443 | +4.0439 | 73.1030 | +2.9852 |
| coco_object81/broccoli | 56.7400 | 51.6390 | +5.1010 | 55.0654 | +1.6746 |
| coco_object81/carrot | 54.0844 | 53.7707 | +0.3137 | 50.9897 | +3.0947 |
| coco_object81/hot dog | 41.8737 | 41.8020 | +0.0717 | 47.1200 | -5.2463 |
| coco_object81/pizza | 57.3303 | 57.9889 | -0.6586 | 60.6512 | -3.3209 |
| coco_object81/donut | 41.0093 | 60.5520 | -19.5427 | 40.3985 | +0.6108 |
| coco_object81/cake | 32.4226 | 45.1281 | -12.7055 | 33.6642 | -1.2416 |
| coco_object81/chair | 27.8546 | 32.8568 | -5.0022 | 21.9885 | +5.8661 |
| coco_object81/couch | 58.0652 | 52.8099 | +5.2553 | 54.9620 | +3.1032 |
| coco_object81/potted plant | 25.7798 | 22.0359 | +3.7439 | 22.5977 | +3.1821 |
| coco_object81/bed | 56.7126 | 52.5372 | +4.1754 | 0.0138 | +56.6988 |
| coco_object81/dining table | 28.0638 | 33.4001 | -5.3363 | 31.7376 | -3.6738 |
| coco_object81/toilet | 51.4335 | 45.8450 | +5.5885 | 46.3017 | +5.1318 |
| coco_object81/tv | 33.4142 | 15.5667 | +17.8475 | 29.8473 | +3.5669 |
| coco_object81/laptop | 52.8100 | 61.7042 | -8.8942 | 54.2117 | -1.4017 |
| coco_object81/mouse | 71.0951 | 31.9050 | +39.1901 | 49.2470 | +21.8481 |
| coco_object81/remote | 47.1827 | 45.2235 | +1.9592 | 50.8786 | -3.6959 |
| coco_object81/keyboard | 62.4311 | 57.3269 | +5.1042 | 31.2719 | +31.1592 |
| coco_object81/cell phone | 32.8066 | 45.9613 | -13.1547 | 37.6283 | -4.8217 |
| coco_object81/microwave | 48.6815 | 44.6905 | +3.9910 | 47.2161 | +1.4654 |
| coco_object81/oven | 41.8611 | 33.3531 | +8.5080 | 43.2177 | -1.3566 |
| coco_object81/toaster | 73.1087 | 30.8319 | +42.2768 | 73.1906 | -0.0819 |
| coco_object81/sink | 45.6466 | 35.6941 | +9.9525 | 42.3849 | +3.2617 |
| coco_object81/refrigerator | 61.7637 | 50.8347 | +10.9290 | 69.4354 | -7.6717 |
| coco_object81/book | 39.1545 | 41.8291 | -2.6746 | 26.4029 | +12.7516 |
| coco_object81/clock | 65.6551 | 65.1122 | +0.5429 | 65.1494 | +0.5057 |
| coco_object81/vase | 31.9337 | 38.2624 | -6.3287 | 31.1688 | +0.7649 |
| coco_object81/scissors | 63.9161 | 67.7027 | -3.7866 | 65.9977 | -2.0816 |
| coco_object81/teddy bear | 73.1671 | 72.7475 | +0.4196 | 73.8343 | -0.6672 |
| coco_object81/hair drier | 24.6937 | 42.3180 | -17.6243 | 47.1047 | -22.4110 |
| coco_object81/toothbrush | 39.5334 | 39.3978 | +0.1356 | 38.4776 | +1.0558 |

Prior measured VIP declared-setting median 132.91 ms; chosen/prior-VIP 2.134×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## coco_stuff171

Choice: `{"profile": {"bank": "official_segmentation", "strength": 1.0, "coupling": 0.5, "temperature": 0.05, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[15, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 3, 1, 1, 1, 1, 1, 4, 1, 5, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 4, 1, 4, 2, 6, 1, 1, 2, 1, 1, 1, 3, 1, 2, 2, 6, 1, 1, 3, 1, 1, 1, 1, 1, 6, 1, 5, 4, 1, 2, 3, 1, 3, 2, 6, 6, 2, 5, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 3, 3, 4, 4, 2, 5, 4, 4, 5, 5, 3, 3, 5, 2, 4, 6, 4, 4, 5, 1, 1, 4, 2, 1, 4, 1, 1, 6, 5, 1, 1, 3, 5, 5, 3, 1, 3, 7, 2, 6, 2, 1, 3, 1, 2, 1, 3, 1, 1, 4, 1, 3, 6, 2, 2, 3, 2, 1, 1, 1, 3, 4, 5, 1, 4]`. Frozen Kev judgment compute 254.53 s (excludes checkpoint initialization); 7040 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 366.10 ms; p95 368.10 ms; peak 6537.0 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| coco_stuff171/person | 57.5809 | 62.2812 | -4.7003 | 58.1776 | -0.5967 |
| coco_stuff171/bicycle | 63.3126 | 58.2230 | +5.0896 | 61.4111 | +1.9015 |
| coco_stuff171/car | 44.0760 | 39.3538 | +4.7222 | 42.2503 | +1.8257 |
| coco_stuff171/motorcycle | 72.4010 | 69.9253 | +2.4757 | 71.4488 | +0.9522 |
| coco_stuff171/airplane | 44.3283 | 26.3459 | +17.9824 | 54.7370 | -10.4087 |
| coco_stuff171/bus | 68.1715 | 67.7113 | +0.4602 | 69.5749 | -1.4034 |
| coco_stuff171/train | 61.4632 | 50.1994 | +11.2638 | 66.9000 | -5.4368 |
| coco_stuff171/truck | 45.0251 | 41.4408 | +3.5843 | 40.4874 | +4.5377 |
| coco_stuff171/boat | 42.9889 | 33.7766 | +9.2123 | 43.5502 | -0.5613 |
| coco_stuff171/traffic light | 33.1438 | 25.1384 | +8.0054 | 47.2329 | -14.0891 |
| coco_stuff171/fire hydrant | 72.9176 | 61.0320 | +11.8856 | 73.4965 | -0.5789 |
| coco_stuff171/stop sign | 29.6526 | 30.4878 | -0.8352 | 36.3943 | -6.7417 |
| coco_stuff171/parking meter | 39.6603 | 33.6517 | +6.0086 | 45.7356 | -6.0753 |
| coco_stuff171/bench | 37.5523 | 36.2169 | +1.3354 | 44.0598 | -6.5075 |
| coco_stuff171/bird | 43.4981 | 36.4110 | +7.0871 | 47.0951 | -3.5970 |
| coco_stuff171/cat | 78.1456 | 74.2462 | +3.8994 | 78.2986 | -0.1530 |
| coco_stuff171/dog | 66.6913 | 47.8947 | +18.7966 | 72.7931 | -6.1018 |
| coco_stuff171/horse | 73.8030 | 64.8472 | +8.9558 | 72.2405 | +1.5625 |
| coco_stuff171/sheep | 80.1818 | 71.6154 | +8.5664 | 79.5992 | +0.5826 |
| coco_stuff171/cow | 81.0151 | 77.1240 | +3.8911 | 71.2424 | +9.7727 |
| coco_stuff171/elephant | 85.7154 | 82.2743 | +3.4411 | 81.8500 | +3.8654 |
| coco_stuff171/bear | 86.3392 | 84.9081 | +1.4311 | 80.3308 | +6.0084 |
| coco_stuff171/zebra | 84.6159 | 81.2999 | +3.3160 | 82.2761 | +2.3398 |
| coco_stuff171/giraffe | 80.4785 | 77.0780 | +3.4005 | 75.9788 | +4.4997 |
| coco_stuff171/backpack | 25.5907 | 21.1803 | +4.4104 | 30.8001 | -5.2094 |
| coco_stuff171/umbrella | 74.7691 | 74.1904 | +0.5787 | 72.1070 | +2.6621 |
| coco_stuff171/handbag | 32.3317 | 31.0992 | +1.2325 | 24.5610 | +7.7707 |
| coco_stuff171/tie | 5.8119 | 7.2046 | -1.3927 | 5.8480 | -0.0361 |
| coco_stuff171/suitcase | 63.9715 | 63.2797 | +0.6918 | 65.1294 | -1.1579 |
| coco_stuff171/frisbee | 34.5490 | 26.4980 | +8.0510 | 31.2655 | +3.2835 |
| coco_stuff171/skis | 18.6954 | 15.4646 | +3.2308 | 9.6548 | +9.0406 |
| coco_stuff171/snowboard | 12.0402 | 10.0079 | +2.0323 | 12.8196 | -0.7794 |
| coco_stuff171/sports ball | 5.7220 | 8.9742 | -3.2522 | 14.5597 | -8.8377 |
| coco_stuff171/kite | 37.8577 | 40.1328 | -2.2751 | 32.3666 | +5.4911 |
| coco_stuff171/baseball bat | 33.4065 | 33.3564 | +0.0501 | 26.6307 | +6.7758 |
| coco_stuff171/baseball glove | 44.9972 | 24.2563 | +20.7409 | 62.1374 | -17.1402 |
| coco_stuff171/skateboard | 17.3265 | 21.2836 | -3.9571 | 22.6364 | -5.3099 |
| coco_stuff171/surfboard | 65.0422 | 54.2007 | +10.8415 | 63.5705 | +1.4717 |
| coco_stuff171/tennis racket | 36.3872 | 43.9051 | -7.5179 | 47.7681 | -11.3809 |
| coco_stuff171/bottle | 42.8993 | 44.7463 | -1.8470 | 40.8237 | +2.0756 |
| coco_stuff171/wine glass | 48.8847 | 50.8447 | -1.9600 | 46.1462 | +2.7385 |
| coco_stuff171/cup | 29.9140 | 34.0587 | -4.1447 | 26.7933 | +3.1207 |
| coco_stuff171/fork | 40.0532 | 36.9974 | +3.0558 | 38.2651 | +1.7881 |
| coco_stuff171/knife | 29.6198 | 36.2588 | -6.6390 | 27.1669 | +2.4529 |
| coco_stuff171/spoon | 37.9700 | 32.4428 | +5.5272 | 35.2995 | +2.6705 |
| coco_stuff171/bowl | 19.3120 | 25.7623 | -6.4503 | 15.7820 | +3.5300 |
| coco_stuff171/banana | 42.6392 | 48.5597 | -5.9205 | 48.2587 | -5.6195 |
| coco_stuff171/apple | 34.1342 | 26.4520 | +7.6822 | 46.6795 | -12.5453 |
| coco_stuff171/sandwich | 34.4433 | 38.3464 | -3.9031 | 29.4956 | +4.9477 |
| coco_stuff171/orange | 19.7292 | 23.2805 | -3.5513 | 50.6227 | -30.8935 |
| coco_stuff171/broccoli | 58.7404 | 57.7318 | +1.0086 | 56.6859 | +2.0545 |
| coco_stuff171/carrot | 41.0854 | 45.9651 | -4.8797 | 40.5753 | +0.5101 |
| coco_stuff171/hot dog | 39.4482 | 41.0525 | -1.6043 | 39.2655 | +0.1827 |
| coco_stuff171/pizza | 56.3030 | 61.1842 | -4.8812 | 56.4602 | -0.1572 |
| coco_stuff171/donut | 61.4860 | 53.0483 | +8.4377 | 50.9258 | +10.5602 |
| coco_stuff171/cake | 49.3841 | 53.6617 | -4.2776 | 45.1257 | +4.2584 |
| coco_stuff171/chair | 32.9560 | 36.5383 | -3.5823 | 28.7918 | +4.1642 |
| coco_stuff171/couch | 56.0970 | 53.0967 | +3.0003 | 55.6650 | +0.4320 |
| coco_stuff171/potted plant | 22.9245 | 22.1581 | +0.7664 | 17.7146 | +5.2099 |
| coco_stuff171/bed | 55.1228 | 50.6193 | +4.5035 | 38.9988 | +16.1240 |
| coco_stuff171/dining table | 23.2100 | 25.1459 | -1.9359 | 19.9895 | +3.2205 |
| coco_stuff171/toilet | 50.3888 | 49.3868 | +1.0020 | 44.9789 | +5.4099 |
| coco_stuff171/tv | 22.8686 | 21.3065 | +1.5621 | 39.3757 | -16.5071 |
| coco_stuff171/laptop | 59.9377 | 61.3385 | -1.4008 | 56.8465 | +3.0912 |
| coco_stuff171/mouse | 46.0413 | 37.9916 | +8.0497 | 48.9537 | -2.9124 |
| coco_stuff171/remote | 50.2981 | 54.6748 | -4.3767 | 51.7195 | -1.4214 |
| coco_stuff171/keyboard | 57.1124 | 59.1083 | -1.9959 | 58.7835 | -1.6711 |
| coco_stuff171/cell phone | 36.5486 | 39.7874 | -3.2388 | 30.4748 | +6.0738 |
| coco_stuff171/microwave | 48.6009 | 57.1579 | -8.5570 | 57.7223 | -9.1214 |
| coco_stuff171/oven | 38.1105 | 38.3476 | -0.2371 | 41.7004 | -3.5899 |
| coco_stuff171/toaster | 47.0347 | 42.1504 | +4.8843 | 55.0708 | -8.0361 |
| coco_stuff171/sink | 46.4514 | 41.5367 | +4.9147 | 44.3552 | +2.0962 |
| coco_stuff171/refrigerator | 64.7964 | 65.9868 | -1.1904 | 69.3664 | -4.5700 |
| coco_stuff171/book | 33.4470 | 35.7945 | -2.3475 | 30.9912 | +2.4558 |
| coco_stuff171/clock | 67.5045 | 62.4699 | +5.0346 | 64.2751 | +3.2294 |
| coco_stuff171/vase | 39.3780 | 44.1324 | -4.7544 | 33.5037 | +5.8743 |
| coco_stuff171/scissors | 64.7500 | 61.7964 | +2.9536 | 65.8353 | -1.0853 |
| coco_stuff171/teddy bear | 71.5062 | 70.7125 | +0.7937 | 70.5152 | +0.9910 |
| coco_stuff171/hair drier | 42.5422 | 29.6403 | +12.9019 | 45.1536 | -2.6114 |
| coco_stuff171/toothbrush | 30.8565 | 30.2015 | +0.6550 | 34.4264 | -3.5699 |
| coco_stuff171/banner | 18.3424 | 22.0108 | -3.6684 | 22.5472 | -4.2048 |
| coco_stuff171/blanket | 14.7191 | 16.6886 | -1.9695 | 0.0062 | +14.7129 |
| coco_stuff171/branch | 3.9419 | 7.5331 | -3.5912 | 3.3628 | +0.5791 |
| coco_stuff171/bridge | 32.4322 | 33.1432 | -0.7110 | 37.0067 | -4.5745 |
| coco_stuff171/building-other | 40.5898 | 39.7139 | +0.8759 | 30.7980 | +9.7918 |
| coco_stuff171/bush | 22.5058 | 21.4998 | +1.0060 | 23.0826 | -0.5768 |
| coco_stuff171/cabinet | 24.1562 | 31.9854 | -7.8292 | 22.9714 | +1.1848 |
| coco_stuff171/cage | 12.5920 | 7.5926 | +4.9994 | 16.4373 | -3.8453 |
| coco_stuff171/cardboard | 47.8522 | 46.8066 | +1.0456 | 41.4786 | +6.3736 |
| coco_stuff171/carpet | 45.8000 | 41.3687 | +4.4313 | 42.6739 | +3.1261 |
| coco_stuff171/ceiling-other | 43.5931 | 42.7339 | +0.8592 | 45.5122 | -1.9191 |
| coco_stuff171/ceiling-tile | 10.7145 | 0.4960 | +10.2185 | 14.6043 | -3.8898 |
| coco_stuff171/cloth | 1.4253 | 2.3513 | -0.9260 | 2.7794 | -1.3541 |
| coco_stuff171/clothes | 5.1107 | 5.1394 | -0.0287 | 5.3893 | -0.2786 |
| coco_stuff171/clouds | 39.6868 | 33.3122 | +6.3746 | 42.0480 | -2.3612 |
| coco_stuff171/counter | 12.1516 | 11.3382 | +0.8134 | 11.5759 | +0.5757 |
| coco_stuff171/cupboard | 5.8880 | 4.0591 | +1.8289 | 5.8529 | +0.0351 |
| coco_stuff171/curtain | 51.1676 | 58.8338 | -7.6662 | 55.0244 | -3.8568 |
| coco_stuff171/desk-stuff | 24.9269 | 29.2409 | -4.3140 | 24.7070 | +0.2199 |
| coco_stuff171/dirt | 4.0745 | 2.7092 | +1.3653 | 9.0991 | -5.0246 |
| coco_stuff171/door-stuff | 35.2382 | 35.6175 | -0.3793 | 33.3227 | +1.9155 |
| coco_stuff171/fence | 30.8998 | 27.3491 | +3.5507 | 28.6182 | +2.2816 |
| coco_stuff171/floor-marble | 6.7468 | 6.8527 | -0.1059 | 6.5222 | +0.2246 |
| coco_stuff171/floor-other | 14.0472 | 14.0103 | +0.0369 | 10.0121 | +4.0351 |
| coco_stuff171/floor-stone | 4.8881 | 5.4079 | -0.5198 | 3.9210 | +0.9671 |
| coco_stuff171/floor-tile | 45.6892 | 41.7857 | +3.9035 | 46.4409 | -0.7517 |
| coco_stuff171/floor-wood | 34.8019 | 35.7516 | -0.9497 | 39.5792 | -4.7773 |
| coco_stuff171/flower | 34.5559 | 34.3349 | +0.2210 | 32.2863 | +2.2696 |
| coco_stuff171/fog | 8.2171 | 9.5182 | -1.3011 | 7.8118 | +0.4053 |
| coco_stuff171/food-other | 14.6011 | 16.3816 | -1.7805 | 13.5457 | +1.0554 |
| coco_stuff171/fruit | 19.6917 | 20.5839 | -0.8922 | 25.7319 | -6.0402 |
| coco_stuff171/furniture-other | 1.3013 | 0.6379 | +0.6634 | 2.8217 | -1.5204 |
| coco_stuff171/grass | 56.0949 | 56.2834 | -0.1885 | 49.8899 | +6.2050 |
| coco_stuff171/gravel | 17.3015 | 19.0697 | -1.7682 | 15.7521 | +1.5494 |
| coco_stuff171/ground-other | 5.1897 | 5.8364 | -0.6467 | 4.0444 | +1.1453 |
| coco_stuff171/hill | 13.7304 | 13.0714 | +0.6590 | 9.5987 | +4.1317 |
| coco_stuff171/house | 12.5723 | 12.3402 | +0.2321 | 12.0964 | +0.4759 |
| coco_stuff171/leaves | 14.0270 | 12.5497 | +1.4773 | 8.4982 | +5.5288 |
| coco_stuff171/light | 23.8591 | 26.7769 | -2.9178 | 18.6507 | +5.2084 |
| coco_stuff171/mat | 6.9947 | 4.8742 | +2.1205 | 3.8623 | +3.1324 |
| coco_stuff171/metal | 8.1187 | 8.1945 | -0.0758 | 4.9744 | +3.1443 |
| coco_stuff171/mirror-stuff | 19.5749 | 21.7816 | -2.2067 | 22.9572 | -3.3823 |
| coco_stuff171/moss | 8.1250 | 12.2250 | -4.1000 | 6.8034 | +1.3216 |
| coco_stuff171/mountain | 42.9898 | 40.2693 | +2.7205 | 41.3951 | +1.5947 |
| coco_stuff171/mud | 4.8737 | 5.5774 | -0.7037 | 6.7652 | -1.8915 |
| coco_stuff171/napkin | 19.5786 | 19.8945 | -0.3159 | 17.4573 | +2.1213 |
| coco_stuff171/net | 24.7054 | 25.8987 | -1.1933 | 25.2561 | -0.5507 |
| coco_stuff171/paper | 25.2443 | 26.1420 | -0.8977 | 17.7441 | +7.5002 |
| coco_stuff171/pavement | 39.3016 | 37.8808 | +1.4208 | 31.2412 | +8.0604 |
| coco_stuff171/pillow | 2.7608 | 9.4266 | -6.6658 | 6.9202 | -4.1594 |
| coco_stuff171/plant-other | 5.0735 | 5.5181 | -0.4446 | 8.0302 | -2.9567 |
| coco_stuff171/plastic | 6.9078 | 4.1879 | +2.7199 | 3.7602 | +3.1476 |
| coco_stuff171/platform | 8.1302 | 10.5575 | -2.4273 | 10.8179 | -2.6877 |
| coco_stuff171/playingfield | 42.4445 | 37.1027 | +5.3418 | 22.8172 | +19.6273 |
| coco_stuff171/railing | 4.1258 | 4.0608 | +0.0650 | 4.5582 | -0.4324 |
| coco_stuff171/railroad | 46.0180 | 31.3810 | +14.6370 | 46.7353 | -0.7173 |
| coco_stuff171/river | 24.9676 | 29.2017 | -4.2341 | 21.0040 | +3.9636 |
| coco_stuff171/road | 41.2090 | 42.0671 | -0.8581 | 39.6204 | +1.5886 |
| coco_stuff171/rock | 35.1781 | 37.6126 | -2.4345 | 33.8852 | +1.2929 |
| coco_stuff171/roof | 14.8308 | 15.1596 | -0.3288 | 14.5147 | +0.3161 |
| coco_stuff171/rug | 23.5546 | 27.8059 | -4.2513 | 26.8320 | -3.2774 |
| coco_stuff171/salad | 8.7054 | 9.0823 | -0.3769 | 8.7785 | -0.0731 |
| coco_stuff171/sand | 45.8609 | 52.6011 | -6.7402 | 55.5704 | -9.7095 |
| coco_stuff171/sea | 74.8899 | 74.0609 | +0.8290 | 74.6278 | +0.2621 |
| coco_stuff171/shelf | 22.7316 | 22.9841 | -0.2525 | 25.9205 | -3.1889 |
| coco_stuff171/sky-other | 59.0040 | 55.3403 | +3.6637 | 50.9892 | +8.0148 |
| coco_stuff171/skyscraper | 20.5779 | 17.0098 | +3.5681 | 16.6971 | +3.8808 |
| coco_stuff171/snow | 78.6620 | 75.1650 | +3.4970 | 68.7907 | +9.8713 |
| coco_stuff171/solid-other | 0.5720 | 0.2011 | +0.3709 | 1.4488 | -0.8768 |
| coco_stuff171/stairs | 21.8787 | 22.1211 | -0.2424 | 21.6551 | +0.2236 |
| coco_stuff171/stone | 3.0037 | 4.7206 | -1.7169 | 4.5036 | -1.4999 |
| coco_stuff171/straw | 21.1509 | 20.8431 | +0.3078 | 22.2248 | -1.0739 |
| coco_stuff171/structural-other | 0.0064 | 0.0317 | -0.0253 | 0.0000 | +0.0064 |
| coco_stuff171/table | 2.5187 | 6.4634 | -3.9447 | 1.9078 | +0.6109 |
| coco_stuff171/tent | 6.5271 | 7.6939 | -1.1668 | 7.4074 | -0.8803 |
| coco_stuff171/textile-other | 2.7027 | 2.3149 | +0.3878 | 5.5705 | -2.8678 |
| coco_stuff171/towel | 32.4807 | 35.4349 | -2.9542 | 34.4995 | -2.0188 |
| coco_stuff171/tree | 34.9507 | 40.6205 | -5.6698 | 21.7037 | +13.2470 |
| coco_stuff171/vegetable | 16.9679 | 14.4550 | +2.5129 | 21.1056 | -4.1377 |
| coco_stuff171/wall-brick | 40.6537 | 39.1638 | +1.4899 | 38.0740 | +2.5797 |
| coco_stuff171/wall-concrete | 3.3729 | 1.9025 | +1.4704 | 2.1042 | +1.2687 |
| coco_stuff171/wall-other | 8.6951 | 9.0322 | -0.3371 | 6.1805 | +2.5146 |
| coco_stuff171/wall-panel | 4.0234 | 2.8675 | +1.1559 | 3.1441 | +0.8793 |
| coco_stuff171/wall-stone | 22.3365 | 22.2094 | +0.1271 | 21.4417 | +0.8948 |
| coco_stuff171/wall-tile | 46.1225 | 38.0945 | +8.0280 | 36.8647 | +9.2578 |
| coco_stuff171/wall-wood | 28.7199 | 24.9528 | +3.7671 | 24.1214 | +4.5985 |
| coco_stuff171/water-other | 24.4989 | 24.1712 | +0.3277 | 22.5396 | +1.9593 |
| coco_stuff171/waterdrops | 0.7850 | 0.9571 | -0.1721 | 0.4691 | +0.3159 |
| coco_stuff171/window-blind | 13.6706 | 23.0702 | -9.3996 | 20.6432 | -6.9726 |
| coco_stuff171/window-other | 29.3908 | 30.9472 | -1.5564 | 27.3138 | +2.0770 |
| coco_stuff171/wood | 12.9317 | 14.7866 | -1.8549 | 17.2856 | -4.3539 |

Prior measured VIP declared-setting median 183.69 ms; chosen/prior-VIP 1.993×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.

## flair1

Choice: `{"profile": {"bank": "original", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}, "background_bias": 0.0, "background_threshold": 0.0}`.

Alias counts: `[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]`. Frozen Kev judgment compute 16.19 s (excludes checkpoint initialization); 456 questions including two option rotations. Full language judgments are in language_selection.json.

Whole-image warmed median 247.00 ms; p95 253.10 ms; peak 5548.5 MiB. This excludes offline language/development and includes online alias aggregation.

| Class | Chosen IoU | Exact previous IoU | Δ pp | VIP declared IoU | Δ vs VIP pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| flair1/building | 61.4040 | 66.6302 | -5.2262 | 58.6727 | +2.7313 |
| flair1/pervious surface | 33.7509 | 30.5518 | +3.1991 | 7.0286 | +26.7223 |
| flair1/impervious surface | 60.0740 | 62.8242 | -2.7502 | 45.8843 | +14.1897 |
| flair1/bare soil | 39.6440 | 37.3976 | +2.2464 | 24.4057 | +15.2383 |
| flair1/water | 78.6820 | 78.5816 | +0.1004 | 67.1946 | +11.4874 |
| flair1/coniferous | 13.9215 | 14.9810 | -1.0595 | 14.9779 | -1.0564 |
| flair1/deciduous | 54.0531 | 58.1193 | -4.0662 | 53.6978 | +0.3553 |
| flair1/brushwood | 27.2315 | 25.7808 | +1.4507 | 22.1228 | +5.1087 |
| flair1/vineyard | 58.4955 | 59.2108 | -0.7153 | 59.3478 | -0.8523 |
| flair1/herbaceous vegetation | 39.8215 | 32.8667 | +6.9548 | 41.9934 | -2.1719 |
| flair1/agricultural land | 32.0117 | 23.6110 | +8.4007 | 22.6933 | +9.3184 |
| flair1/plowed land | 24.8056 | 18.7970 | +6.0086 | 21.2695 | +3.5361 |

Prior measured VIP declared-setting median 109.08 ms; chosen/prior-VIP 2.264×. VIP timing is reused from the earlier matched-input benchmark, not a simultaneous measurement.
