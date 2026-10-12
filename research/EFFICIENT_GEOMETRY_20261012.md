# Geometry: all-protocol efficient execution, latest retained configurations

The vocabulary, readout coefficients, background rules, view sizes/counts, pretrained weights and FP32/BF16/FP16/FP64 arithmetic roles are frozen. VOC20, VOC21 and PC60 use the latest retained variable-word configurations, not their older Kev trial settings. This is an execution optimization, not a new accuracy search or proof of a global optimum.

## Execution changes

- Cache the static spatial Geometry term. Keep the original FP64 reconstruction solve and omit only its unused diagnostic residual matrix multiplication.
- Reuse the same raw-patch VIP proxy relation across the two wide head blocks. Preserve the existing self-Value fallback only on empty proxy rows.
- Batch equal-size alias groups in the local log-mean-exp and wide salience/log-sum-exp. Preserve each class's alias count, inherited salience and aggregation mass; no words are deleted.
- Bound FP64 dense-softmax workspace with spatial slabs. Restore class planes in chunks when background calibration does not need a classwise sum. Keep exact original calibration in bounded spatial slabs otherwise.
- Retain the earlier compact deployment optimizations: text towers off GPU, requested head only, cached text metadata, uint8 CPU IDs. Original and first compact paths remain available.

## Full accuracy and whole-image cost

Costs use the same seven fixed whole RGB inputs per protocol on A800, two warmups and seven synchronized repeats. They include resizing, every visual encoding/readout, reconstruction, original-resolution restoration and CPU IDs. Decode, initialization, offline language selection and labeled development are excluded. Costs are scheduled one at a time within this suite; unrelated user jobs are preserved. Prior VIP timings are historical measurements on these same sample keys, not simultaneous measurements or paper numbers.

| Protocol | Images | Historical best mIoU | Compact replay | Efficient mIoU | Compact ms | Efficient ms | Compact MiB | Efficient MiB | Prior VIP ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 80 | 60.4352 | 60.4352 | 60.4352 | 330.05 | 309.98 | 2847.8 | 2622.0 | 190.85 |
| potsdam | 504 | 56.1934 | 56.1934 | 56.1934 | 214.37 | 208.24 | 2156.0 | 2159.9 | 104.11 |
| voc20 | 1449 | 92.7936 | 92.7936 | 92.7936 | 166.32 | 156.35 | 2201.8 | 2218.7 | 58.16 |
| voc21 | 1449 | 75.8586 | 75.8586 | 75.8586 | 164.24 | 153.29 | 2156.1 | 2179.7 | 59.50 |
| ade150 | 2000 | 31.3398 | 31.3398 | 31.3398 | 254.33 | 181.80 | 3295.4 | 3047.7 | 113.98 |
| context60 | 5105 | 44.7613 | 44.7613 | 44.7613 | 218.93 | 185.16 | 2479.3 | 2409.5 | 112.06 |
| udd5 | 40 | 55.8266 | 55.8266 | 55.8266 | 306.73 | 303.05 | 2677.5 | 2256.3 | 191.15 |
| oem | 384 | 45.1052 | 45.1052 | 45.1052 | 210.82 | 201.27 | 2227.7 | 2231.7 | 105.38 |
| vaihingen | 113 | 58.8562 | 58.8562 | 58.8562 | 209.37 | 205.13 | 2183.0 | 2187.0 | 108.55 |
| landcoverai | 1602 | 65.5400 | 65.5400 | 65.5400 | 202.28 | 193.48 | 2199.0 | 2203.0 | 102.71 |
| loveda P | 1669 | 69.6117 | 69.6117 | 69.6117 | 214.19 | 207.31 | 2275.9 | 2279.9 | 105.86 |
| loveda D | 1669 | 45.3171 | 45.3171 | 45.3171 | 215.10 | 207.03 | 2275.9 | 2279.9 | 105.86 |
| context59 | 5105 | 46.3075 | 46.3075 | 46.3075 | 219.14 | 181.78 | 2424.2 | 2332.2 | 111.84 |
| coco_object81 | 5000 | 50.3999 | 50.3999 | 50.3999 | 237.41 | 186.78 | 2700.1 | 2620.2 | 131.46 |
| coco_stuff171 | 5000 | 33.7661 | 33.7661 | 33.7661 | 299.55 | 201.94 | 3399.6 | 3025.5 | 185.87 |
| flair1 | 15700 | 43.6579 | 43.6579 | 43.6579 | 218.65 | 204.19 | 2266.7 | 2270.7 | 111.14 |

## Verification and limitations

15/15 protocols have both completed paired full evaluation and standalone costs. Every complete image is inferred with the old compact and new efficient backends, and changed prediction pixels are counted directly. Per-image confusion matrices sum to the reported aggregates. Complete unique sample coverage and scored target counts are checked against the already verified local VIP inputs.

The frozen protocol totals45,200 protocol images; VOC20/21 and PC59/60 share photographs. UDD5 here is the established40-image five-class validation protocol, not the older540-image UDD run. LoveDA P and D are independent six-/seven-class readouts; eight-remote means count D once. PC60 retains its401-concept residual ontology. LandCover.ai substitutes for unlabeled iSAID.

The retained configurations used labeled development and previously viewed full validation results. These results remain exploratory. Training-free means no weight fitting, not zero inference cost. Latency and peak CUDA allocated memory are reported together with mIoU; no universal venue-specific latency or VRAM cutoff is assumed. No semantic accuracy gain is attributed to execution changes. Tiny floating-point differences, if any, must be reported and cannot be called bit-exact replay.

Completed standalone cost protocols: 15.
Small peak-memory increases (MiB): `{"potsdam": 3.8779296875, "voc20": 16.87646484375, "voc21": 23.623046875, "oem": 4.0, "vaihingen": 4.00537109375, "landcoverai": 4.00048828125, "loveda": 3.9990234375, "flair1": 3.9990234375}`. The compact backend remains available where minimum memory matters more than these latency savings.

The preserved preliminary `profile/` results include a larger finite-check workspace (ADE150 efficient3766.5MiB versus compact3295.4MiB). The final `cost/` backend checks finite values in slabs and removes that regression. These are different execution revisions, not selected repetitions of an unchanged implementation. Preliminary measurements are not substituted for final standalone costs.

Median paired latency reduction: 6.08%.
Total changed prediction pixels: 0.

Eight-remote mean, LoveDA D counted once: `{"compact": 53.86645, "efficient": 53.86645}`.

## vdd

Frozen profile: `{"bank": "official_imagenet", "strength": "original", "coupling": 1.0, "temperature": 0.07, "tau": 1.0, "tem": 0.3, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"vdd": 0}`. Historical confusion agreement: `{"compact__vdd": true, "efficient__vdd": true}`.

Protocol vdd:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| other | 45.3164 | 45.3164 | +0.0000 |
| wall | 48.0048 | 48.0048 | +0.0000 |
| road | 49.7305 | 49.7305 | +0.0000 |
| vegetation | 79.4081 | 79.4081 | +0.0000 |
| vehicle | 36.7663 | 36.7663 | +0.0000 |
| roof | 82.5380 | 82.5380 | +0.0000 |
| water | 81.2826 | 81.2826 | +0.0000 |

## potsdam

Frozen profile: `{"bank": "official_imagenet", "strength": 3.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"potsdam": 0}`. Historical confusion agreement: `{"compact__potsdam": true, "efficient__potsdam": true}`.

Protocol potsdam:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| impervious surface | 65.6164 | 65.6164 | +0.0000 |
| building | 77.9794 | 77.9794 | +0.0000 |
| low vegetation | 60.0738 | 60.0738 | +0.0000 |
| tree | 66.1729 | 66.1729 | +0.0000 |
| car | 46.4693 | 46.4693 | +0.0000 |
| clutter | 20.8484 | 20.8484 | +0.0000 |

## voc20

Frozen profile: `{"bank": "semantic_imagenet", "strength": 2.0, "coupling": 1.0, "temperature": 0.1, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Changed pixels: `{"voc20": 0}`. Historical confusion agreement: `{"compact__voc20": true, "efficient__voc20": true}`.

Protocol voc20:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| aeroplane | 99.3289 | 99.3289 | +0.0000 |
| bicycle | 84.2572 | 84.2572 | +0.0000 |
| bird | 99.5058 | 99.5058 | +0.0000 |
| boat | 95.8951 | 95.8951 | +0.0000 |
| bottle | 85.3503 | 85.3503 | +0.0000 |
| bus | 96.4546 | 96.4546 | +0.0000 |
| car | 96.0880 | 96.0880 | +0.0000 |
| cat | 99.0210 | 99.0210 | +0.0000 |
| chair | 69.0765 | 69.0765 | +0.0000 |
| cow | 98.3147 | 98.3147 | +0.0000 |
| diningtable | 82.1865 | 82.1865 | +0.0000 |
| dog | 97.6810 | 97.6810 | +0.0000 |
| horse | 97.9447 | 97.9447 | +0.0000 |
| motorbike | 90.0486 | 90.0486 | +0.0000 |
| person | 90.2744 | 90.2744 | +0.0000 |
| pottedplant | 97.7382 | 97.7382 | +0.0000 |
| sheep | 99.6062 | 99.6062 | +0.0000 |
| sofa | 87.9066 | 87.9066 | +0.0000 |
| train | 97.8185 | 97.8185 | +0.0000 |
| tvmonitor | 91.3760 | 91.3760 | +0.0000 |

## voc21

Frozen profile: `{"bank": "official_segmentation", "strength": 1.0, "coupling": 0.75, "temperature": 0.07, "tau": 1.0, "tem": 0.3, "wide_policy": "natural_short336_cap672", "background_rule": "union_max"}`.

Changed pixels: `{"voc21": 0}`. Historical confusion agreement: `{"compact__voc21": true, "efficient__voc21": true}`.

Protocol voc21:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| background | 92.7412 | 92.7412 | +0.0000 |
| aeroplane | 78.8993 | 78.8993 | +0.0000 |
| bicycle | 52.6900 | 52.6900 | +0.0000 |
| bird | 82.1239 | 82.1239 | +0.0000 |
| boat | 69.0906 | 69.0906 | +0.0000 |
| bottle | 64.5972 | 64.5972 | +0.0000 |
| bus | 90.1574 | 90.1574 | +0.0000 |
| car | 62.1772 | 62.1772 | +0.0000 |
| cat | 91.2084 | 91.2084 | +0.0000 |
| chair | 54.6892 | 54.6892 | +0.0000 |
| cow | 92.7476 | 92.7476 | +0.0000 |
| diningtable | 46.1760 | 46.1760 | +0.0000 |
| dog | 90.9827 | 90.9827 | +0.0000 |
| horse | 88.3334 | 88.3334 | +0.0000 |
| motorbike | 82.4196 | 82.4196 | +0.0000 |
| person | 81.5668 | 81.5668 | +0.0000 |
| pottedplant | 59.6098 | 59.6098 | +0.0000 |
| sheep | 93.4085 | 93.4085 | +0.0000 |
| sofa | 70.0068 | 70.0068 | +0.0000 |
| train | 83.0908 | 83.0908 | +0.0000 |
| tvmonitor | 66.3134 | 66.3134 | +0.0000 |

## ade150

Frozen profile: `{"bank": "semantic_segmentation", "strength": "original", "coupling": 0.25, "temperature": 0.14, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Changed pixels: `{"ade150": 0}`. Historical confusion agreement: `{"compact__ade150": true, "efficient__ade150": true}`.

Protocol ade150:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| wall | 37.1842 | 37.1842 | +0.0000 |
| building | 62.7147 | 62.7147 | +0.0000 |
| sky | 79.9823 | 79.9823 | +0.0000 |
| floor | 60.3097 | 60.3097 | +0.0000 |
| tree | 62.0770 | 62.0770 | +0.0000 |
| ceiling | 55.7382 | 55.7382 | +0.0000 |
| road | 70.0590 | 70.0590 | +0.0000 |
| bed | 66.8879 | 66.8879 | +0.0000 |
| windowpane | 41.5302 | 41.5302 | +0.0000 |
| grass | 49.3493 | 49.3493 | +0.0000 |
| cabinet | 47.2816 | 47.2816 | +0.0000 |
| sidewalk | 50.3189 | 50.3189 | +0.0000 |
| person | 60.6360 | 60.6360 | +0.0000 |
| earth | 12.3410 | 12.3410 | +0.0000 |
| door | 37.1846 | 37.1846 | +0.0000 |
| table | 41.0312 | 41.0312 | +0.0000 |
| mountain | 38.9642 | 38.9642 | +0.0000 |
| plant | 37.6397 | 37.6397 | +0.0000 |
| curtain | 64.0450 | 64.0450 | +0.0000 |
| chair | 44.2905 | 44.2905 | +0.0000 |
| car | 63.9871 | 63.9871 | +0.0000 |
| water | 45.3726 | 45.3726 | +0.0000 |
| painting | 29.6231 | 29.6231 | +0.0000 |
| sofa | 50.6249 | 50.6249 | +0.0000 |
| shelf | 32.5025 | 32.5025 | +0.0000 |
| house | 21.0911 | 21.0911 | +0.0000 |
| sea | 28.5027 | 28.5027 | +0.0000 |
| mirror | 32.7875 | 32.7875 | +0.0000 |
| rug | 33.4448 | 33.4448 | +0.0000 |
| field | 18.9190 | 18.9190 | +0.0000 |
| armchair | 10.0973 | 10.0973 | +0.0000 |
| seat | 27.3993 | 27.3993 | +0.0000 |
| fence | 25.0113 | 25.0113 | +0.0000 |
| desk | 32.6296 | 32.6296 | +0.0000 |
| rock | 33.5568 | 33.5568 | +0.0000 |
| wardrobe | 45.7128 | 45.7128 | +0.0000 |
| lamp | 30.7669 | 30.7669 | +0.0000 |
| bathtub | 48.6249 | 48.6249 | +0.0000 |
| railing | 19.4873 | 19.4873 | +0.0000 |
| cushion | 37.6780 | 37.6780 | +0.0000 |
| base | 3.3420 | 3.3420 | +0.0000 |
| box | 23.1800 | 23.1800 | +0.0000 |
| column | 37.3675 | 37.3675 | +0.0000 |
| signboard | 26.5261 | 26.5261 | +0.0000 |
| chest of drawers | 29.6321 | 29.6321 | +0.0000 |
| counter | 37.8705 | 37.8705 | +0.0000 |
| sand | 38.7158 | 38.7158 | +0.0000 |
| sink | 40.4072 | 40.4072 | +0.0000 |
| skyscraper | 22.5813 | 22.5813 | +0.0000 |
| fireplace | 48.5427 | 48.5427 | +0.0000 |
| refrigerator | 64.7569 | 64.7569 | +0.0000 |
| grandstand | 23.9350 | 23.9350 | +0.0000 |
| path | 6.1373 | 6.1373 | +0.0000 |
| stairs | 34.9794 | 34.9794 | +0.0000 |
| runway | 40.7494 | 40.7494 | +0.0000 |
| case | 22.7332 | 22.7332 | +0.0000 |
| pool table | 81.3563 | 81.3563 | +0.0000 |
| pillow | 35.9296 | 35.9296 | +0.0000 |
| screen door | 0.0388 | 0.0388 | +0.0000 |
| stairway | 18.3541 | 18.3541 | +0.0000 |
| river | 15.8254 | 15.8254 | +0.0000 |
| bridge | 26.8326 | 26.8326 | +0.0000 |
| bookcase | 17.2964 | 17.2964 | +0.0000 |
| blind | 32.7460 | 32.7460 | +0.0000 |
| coffee table | 49.8514 | 49.8514 | +0.0000 |
| toilet | 53.5176 | 53.5176 | +0.0000 |
| flower | 13.7263 | 13.7263 | +0.0000 |
| book | 13.3161 | 13.3161 | +0.0000 |
| hill | 5.4293 | 5.4293 | +0.0000 |
| bench | 38.0810 | 38.0810 | +0.0000 |
| countertop | 17.9637 | 17.9637 | +0.0000 |
| stove | 26.4129 | 26.4129 | +0.0000 |
| palm | 24.5032 | 24.5032 | +0.0000 |
| kitchen island | 20.9910 | 20.9910 | +0.0000 |
| computer | 40.7204 | 40.7204 | +0.0000 |
| swivel chair | 9.9967 | 9.9967 | +0.0000 |
| boat | 35.2331 | 35.2331 | +0.0000 |
| bar | 22.4898 | 22.4898 | +0.0000 |
| arcade machine | 12.5977 | 12.5977 | +0.0000 |
| hovel | 11.0229 | 11.0229 | +0.0000 |
| bus | 62.8865 | 62.8865 | +0.0000 |
| towel | 55.2839 | 55.2839 | +0.0000 |
| light | 7.5858 | 7.5858 | +0.0000 |
| truck | 17.9129 | 17.9129 | +0.0000 |
| tower | 14.7726 | 14.7726 | +0.0000 |
| chandelier | 42.5879 | 42.5879 | +0.0000 |
| awning | 21.8359 | 21.8359 | +0.0000 |
| streetlight | 22.3405 | 22.3405 | +0.0000 |
| booth | 8.9948 | 8.9948 | +0.0000 |
| television receiver | 27.2471 | 27.2471 | +0.0000 |
| airplane | 15.7427 | 15.7427 | +0.0000 |
| dirt track | 2.2681 | 2.2681 | +0.0000 |
| apparel | 5.9156 | 5.9156 | +0.0000 |
| pole | 12.1618 | 12.1618 | +0.0000 |
| land | 2.8619 | 2.8619 | +0.0000 |
| bannister | 14.8594 | 14.8594 | +0.0000 |
| escalator | 40.5218 | 40.5218 | +0.0000 |
| ottoman | 37.1205 | 37.1205 | +0.0000 |
| bottle | 32.6859 | 32.6859 | +0.0000 |
| buffet | 7.7425 | 7.7425 | +0.0000 |
| poster | 6.9058 | 6.9058 | +0.0000 |
| stage | 7.9432 | 7.9432 | +0.0000 |
| van | 36.9570 | 36.9570 | +0.0000 |
| ship | 7.9054 | 7.9054 | +0.0000 |
| fountain | 33.9808 | 33.9808 | +0.0000 |
| conveyer belt | 45.7465 | 45.7465 | +0.0000 |
| canopy | 17.5233 | 17.5233 | +0.0000 |
| washer | 67.3043 | 67.3043 | +0.0000 |
| plaything | 8.6042 | 8.6042 | +0.0000 |
| swimming pool | 33.7992 | 33.7992 | +0.0000 |
| stool | 24.6744 | 24.6744 | +0.0000 |
| barrel | 29.2598 | 29.2598 | +0.0000 |
| basket | 35.4013 | 35.4013 | +0.0000 |
| waterfall | 21.3209 | 21.3209 | +0.0000 |
| tent | 44.2048 | 44.2048 | +0.0000 |
| bag | 25.3773 | 25.3773 | +0.0000 |
| minibike | 63.5767 | 63.5767 | +0.0000 |
| cradle | 49.2217 | 49.2217 | +0.0000 |
| oven | 26.9018 | 26.9018 | +0.0000 |
| ball | 19.8401 | 19.8401 | +0.0000 |
| food | 42.5130 | 42.5130 | +0.0000 |
| step | 2.8648 | 2.8648 | +0.0000 |
| tank | 22.5529 | 22.5529 | +0.0000 |
| trade name | 2.6907 | 2.6907 | +0.0000 |
| microwave | 71.9927 | 71.9927 | +0.0000 |
| pot | 35.4284 | 35.4284 | +0.0000 |
| animal | 63.1460 | 63.1460 | +0.0000 |
| bicycle | 46.5265 | 46.5265 | +0.0000 |
| lake | 4.1765 | 4.1765 | +0.0000 |
| dishwasher | 51.4241 | 51.4241 | +0.0000 |
| screen | 36.9948 | 36.9948 | +0.0000 |
| blanket | 11.1223 | 11.1223 | +0.0000 |
| sculpture | 42.7199 | 42.7199 | +0.0000 |
| hood | 33.9919 | 33.9919 | +0.0000 |
| sconce | 20.5007 | 20.5007 | +0.0000 |
| vase | 16.4284 | 16.4284 | +0.0000 |
| traffic light | 20.5725 | 20.5725 | +0.0000 |
| tray | 13.1233 | 13.1233 | +0.0000 |
| ashcan | 13.9012 | 13.9012 | +0.0000 |
| fan | 46.5336 | 46.5336 | +0.0000 |
| pier | 3.8774 | 3.8774 | +0.0000 |
| crt screen | 0.3714 | 0.3714 | +0.0000 |
| plate | 34.3786 | 34.3786 | +0.0000 |
| monitor | 18.7229 | 18.7229 | +0.0000 |
| bulletin board | 6.5619 | 6.5619 | +0.0000 |
| shower | 0.9691 | 0.9691 | +0.0000 |
| radiator | 55.0117 | 55.0117 | +0.0000 |
| glass | 8.7374 | 8.7374 | +0.0000 |
| clock | 44.4135 | 44.4135 | +0.0000 |
| flag | 55.8652 | 55.8652 | +0.0000 |

## context60

Frozen profile: `{"bank": "mix_f6f525631e7f743d226c", "strength": 2.0, "coupling": 0.75, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "residual_max"}`.

Changed pixels: `{"context60": 0}`. Historical confusion agreement: `{"compact__context60": true, "efficient__context60": true}`.

Protocol context60:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| background | 20.7389 | 20.7389 | +0.0000 |
| aeroplane | 68.7764 | 68.7764 | +0.0000 |
| bag | 29.4065 | 29.4065 | +0.0000 |
| bed | 12.1528 | 12.1528 | +0.0000 |
| bedclothes | 26.4340 | 26.4340 | +0.0000 |
| bench | 19.8780 | 19.8780 | +0.0000 |
| bicycle | 67.0011 | 67.0011 | +0.0000 |
| bird | 67.0847 | 67.0847 | +0.0000 |
| boat | 51.2771 | 51.2771 | +0.0000 |
| book | 37.4128 | 37.4128 | +0.0000 |
| bottle | 63.1870 | 63.1870 | +0.0000 |
| building | 33.3275 | 33.3275 | +0.0000 |
| bus | 76.4648 | 76.4648 | +0.0000 |
| cabinet | 34.4024 | 34.4024 | +0.0000 |
| car | 68.4574 | 68.4574 | +0.0000 |
| cat | 86.6568 | 86.6568 | +0.0000 |
| ceiling | 47.7303 | 47.7303 | +0.0000 |
| chair | 41.2580 | 41.2580 | +0.0000 |
| cloth | 16.6705 | 16.6705 | +0.0000 |
| computer | 22.5117 | 22.5117 | +0.0000 |
| cow | 84.5857 | 84.5857 | +0.0000 |
| cup | 24.2944 | 24.2944 | +0.0000 |
| curtain | 55.2001 | 55.2001 | +0.0000 |
| dog | 86.1203 | 86.1203 | +0.0000 |
| door | 23.4358 | 23.4358 | +0.0000 |
| fence | 34.7057 | 34.7057 | +0.0000 |
| floor | 48.8606 | 48.8606 | +0.0000 |
| flower | 27.5004 | 27.5004 | +0.0000 |
| food | 34.6374 | 34.6374 | +0.0000 |
| grass | 67.8552 | 67.8552 | +0.0000 |
| ground | 10.2002 | 10.2002 | +0.0000 |
| horse | 77.8086 | 77.8086 | +0.0000 |
| keyboard | 54.8674 | 54.8674 | +0.0000 |
| light | 32.2097 | 32.2097 | +0.0000 |
| motorbike | 64.2913 | 64.2913 | +0.0000 |
| mountain | 38.9106 | 38.9106 | +0.0000 |
| mouse | 55.8503 | 55.8503 | +0.0000 |
| person | 76.0512 | 76.0512 | +0.0000 |
| plate | 44.7232 | 44.7232 | +0.0000 |
| platform | 21.8104 | 21.8104 | +0.0000 |
| pottedplant | 44.1617 | 44.1617 | +0.0000 |
| road | 42.3314 | 42.3314 | +0.0000 |
| rock | 29.1788 | 29.1788 | +0.0000 |
| sheep | 84.5030 | 84.5030 | +0.0000 |
| shelves | 23.8299 | 23.8299 | +0.0000 |
| sidewalk | 15.0331 | 15.0331 | +0.0000 |
| sign | 35.5181 | 35.5181 | +0.0000 |
| sky | 72.0540 | 72.0540 | +0.0000 |
| snow | 64.7348 | 64.7348 | +0.0000 |
| sofa | 61.0352 | 61.0352 | +0.0000 |
| table | 47.7483 | 47.7483 | +0.0000 |
| track | 1.9396 | 1.9396 | +0.0000 |
| train | 47.2403 | 47.2403 | +0.0000 |
| tree | 52.9710 | 52.9710 | +0.0000 |
| truck | 17.0885 | 17.0885 | +0.0000 |
| tvmonitor | 50.9884 | 50.9884 | +0.0000 |
| wall | 28.9571 | 28.9571 | +0.0000 |
| water | 76.3571 | 76.3571 | +0.0000 |
| window | 29.9164 | 29.9164 | +0.0000 |
| wood | 5.3425 | 5.3425 | +0.0000 |

## udd5

Frozen profile: `{"bank": "kev60_segmentation", "strength": "original", "coupling": 0.5, "temperature": 0.05, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"udd5": 0}`. Historical confusion agreement: `{"compact__udd5": true, "efficient__udd5": true}`.

Protocol udd5:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| vegetation | 79.2404 | 79.2404 | +0.0000 |
| building | 85.3634 | 85.3634 | +0.0000 |
| road | 37.3689 | 37.3689 | +0.0000 |
| vehicle | 39.8496 | 39.8496 | +0.0000 |
| other | 37.3109 | 37.3109 | +0.0000 |

## oem

Frozen profile: `{"bank": "original", "strength": 1.0, "coupling": 0.25, "temperature": 0.03, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"oem": 0}`. Historical confusion agreement: `{"compact__oem": true, "efficient__oem": true}`.

Protocol oem:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| bareland | 10.4191 | 10.4191 | +0.0000 |
| rangeland | 32.0863 | 32.0863 | +0.0000 |
| developed space | 27.5374 | 27.5374 | +0.0000 |
| road | 38.0586 | 38.0586 | +0.0000 |
| tree | 54.9661 | 54.9661 | +0.0000 |
| water | 68.9790 | 68.9790 | +0.0000 |
| agriculture land | 66.0801 | 66.0801 | +0.0000 |
| building | 62.7148 | 62.7148 | +0.0000 |

## vaihingen

Frozen profile: `{"bank": "kev40_imagenet", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"vaihingen": 0}`. Historical confusion agreement: `{"compact__vaihingen": true, "efficient__vaihingen": true}`.

Protocol vaihingen:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| impervious surface | 66.0449 | 66.0449 | +0.0000 |
| building | 72.7879 | 72.7879 | +0.0000 |
| low vegetation | 40.6235 | 40.6235 | +0.0000 |
| tree | 69.5008 | 69.5008 | +0.0000 |
| car | 45.3241 | 45.3241 | +0.0000 |

## landcoverai

Frozen profile: `{"bank": "original_imagenet", "strength": 2.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"landcoverai": 0}`. Historical confusion agreement: `{"compact__landcoverai": true, "efficient__landcoverai": true}`.

Protocol landcoverai:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| background | 82.9318 | 82.9318 | +0.0000 |
| building | 60.9726 | 60.9726 | +0.0000 |
| woodland | 77.1198 | 77.1198 | +0.0000 |
| water | 64.1632 | 64.1632 | +0.0000 |
| road | 42.5126 | 42.5126 | +0.0000 |

## loveda

Frozen profile: `{"bank": "original", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"P": 0, "D": 0}`. Historical confusion agreement: `{"compact__P": true, "compact__D": true, "efficient__P": true, "efficient__D": true}`.

Protocol P:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| building | 88.4885 | 88.4885 | +0.0000 |
| road | 65.2094 | 65.2094 | +0.0000 |
| water | 75.6226 | 75.6226 | +0.0000 |
| barren | 43.9516 | 43.9516 | +0.0000 |
| tree | 63.1547 | 63.1547 | +0.0000 |
| farm | 81.2434 | 81.2434 | +0.0000 |

Protocol D:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| background | 38.6355 | 38.6355 | +0.0000 |
| building | 56.6110 | 56.6110 | +0.0000 |
| road | 52.3713 | 52.3713 | +0.0000 |
| water | 45.6562 | 45.6562 | +0.0000 |
| barren | 25.3890 | 25.3890 | +0.0000 |
| tree | 37.7511 | 37.7511 | +0.0000 |
| farm | 60.8057 | 60.8057 | +0.0000 |

## context59

Frozen profile: `{"bank": "official_segmentation", "strength": 3.0, "coupling": 0.75, "temperature": 0.07, "tau": 1.0, "tem": 10.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Changed pixels: `{"context59": 0}`. Historical confusion agreement: `{"compact__context59": true, "efficient__context59": true}`.

Protocol context59:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| aeroplane | 67.3383 | 67.3383 | +0.0000 |
| bag | 44.8271 | 44.8271 | +0.0000 |
| bed | 6.8257 | 6.8257 | +0.0000 |
| bedclothes | 26.9829 | 26.9829 | +0.0000 |
| bench | 18.2440 | 18.2440 | +0.0000 |
| bicycle | 69.3990 | 69.3990 | +0.0000 |
| bird | 70.3675 | 70.3675 | +0.0000 |
| boat | 68.6008 | 68.6008 | +0.0000 |
| book | 6.5114 | 6.5114 | +0.0000 |
| bottle | 71.2428 | 71.2428 | +0.0000 |
| building | 36.9821 | 36.9821 | +0.0000 |
| bus | 80.2299 | 80.2299 | +0.0000 |
| cabinet | 36.8496 | 36.8496 | +0.0000 |
| car | 64.8208 | 64.8208 | +0.0000 |
| cat | 87.9852 | 87.9852 | +0.0000 |
| ceiling | 48.0722 | 48.0722 | +0.0000 |
| chair | 43.2269 | 43.2269 | +0.0000 |
| cloth | 16.9566 | 16.9566 | +0.0000 |
| computer | 26.6571 | 26.6571 | +0.0000 |
| cow | 83.9745 | 83.9745 | +0.0000 |
| cup | 31.5851 | 31.5851 | +0.0000 |
| curtain | 57.4631 | 57.4631 | +0.0000 |
| dog | 82.2504 | 82.2504 | +0.0000 |
| door | 27.4833 | 27.4833 | +0.0000 |
| fence | 38.4962 | 38.4962 | +0.0000 |
| floor | 50.5707 | 50.5707 | +0.0000 |
| flower | 24.5721 | 24.5721 | +0.0000 |
| food | 50.2279 | 50.2279 | +0.0000 |
| grass | 71.6279 | 71.6279 | +0.0000 |
| ground | 12.1732 | 12.1732 | +0.0000 |
| horse | 83.9426 | 83.9426 | +0.0000 |
| keyboard | 52.1329 | 52.1329 | +0.0000 |
| light | 24.1319 | 24.1319 | +0.0000 |
| motorbike | 73.6571 | 73.6571 | +0.0000 |
| mountain | 35.5850 | 35.5850 | +0.0000 |
| mouse | 62.1596 | 62.1596 | +0.0000 |
| person | 66.1820 | 66.1820 | +0.0000 |
| plate | 31.8412 | 31.8412 | +0.0000 |
| platform | 9.2000 | 9.2000 | +0.0000 |
| pottedplant | 49.8043 | 49.8043 | +0.0000 |
| road | 37.2111 | 37.2111 | +0.0000 |
| rock | 21.0625 | 21.0625 | +0.0000 |
| sheep | 85.6449 | 85.6449 | +0.0000 |
| shelves | 18.7586 | 18.7586 | +0.0000 |
| sidewalk | 9.4003 | 9.4003 | +0.0000 |
| sign | 42.5840 | 42.5840 | +0.0000 |
| sky | 81.5902 | 81.5902 | +0.0000 |
| snow | 64.2701 | 64.2701 | +0.0000 |
| sofa | 65.0447 | 65.0447 | +0.0000 |
| table | 46.5805 | 46.5805 | +0.0000 |
| track | 5.0192 | 5.0192 | +0.0000 |
| train | 49.3242 | 49.3242 | +0.0000 |
| tree | 64.1148 | 64.1148 | +0.0000 |
| truck | 23.4607 | 23.4607 | +0.0000 |
| tvmonitor | 50.2867 | 50.2867 | +0.0000 |
| wall | 30.6628 | 30.6628 | +0.0000 |
| water | 73.9754 | 73.9754 | +0.0000 |
| window | 36.7030 | 36.7030 | +0.0000 |
| wood | 15.2672 | 15.2672 | +0.0000 |

## coco_object81

Frozen profile: `{"bank": "official_imagenet", "strength": 1.0, "coupling": 0.5, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Changed pixels: `{"coco_object81": 0}`. Historical confusion agreement: `{"compact__coco_object81": true, "efficient__coco_object81": true}`.

Protocol coco_object81:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| background | 78.4969 | 78.4969 | +0.0000 |
| person | 30.5662 | 30.5662 | +0.0000 |
| bicycle | 59.5600 | 59.5600 | +0.0000 |
| car | 34.9615 | 34.9615 | +0.0000 |
| motorcycle | 75.2194 | 75.2194 | +0.0000 |
| airplane | 59.3917 | 59.3917 | +0.0000 |
| bus | 70.8357 | 70.8357 | +0.0000 |
| train | 48.8879 | 48.8879 | +0.0000 |
| truck | 41.1901 | 41.1901 | +0.0000 |
| boat | 45.6119 | 45.6119 | +0.0000 |
| traffic light | 20.8175 | 20.8175 | +0.0000 |
| fire hydrant | 71.8424 | 71.8424 | +0.0000 |
| stop sign | 39.2949 | 39.2949 | +0.0000 |
| parking meter | 61.9053 | 61.9053 | +0.0000 |
| bench | 38.8993 | 38.8993 | +0.0000 |
| bird | 58.0272 | 58.0272 | +0.0000 |
| cat | 80.4076 | 80.4076 | +0.0000 |
| dog | 68.6549 | 68.6549 | +0.0000 |
| horse | 70.7875 | 70.7875 | +0.0000 |
| sheep | 80.6673 | 80.6673 | +0.0000 |
| cow | 78.9477 | 78.9477 | +0.0000 |
| elephant | 85.8584 | 85.8584 | +0.0000 |
| bear | 81.2511 | 81.2511 | +0.0000 |
| zebra | 86.3780 | 86.3780 | +0.0000 |
| giraffe | 72.9500 | 72.9500 | +0.0000 |
| backpack | 34.5770 | 34.5770 | +0.0000 |
| umbrella | 77.6635 | 77.6635 | +0.0000 |
| handbag | 40.6348 | 40.6348 | +0.0000 |
| tie | 7.0701 | 7.0701 | +0.0000 |
| suitcase | 62.9106 | 62.9106 | +0.0000 |
| frisbee | 65.5646 | 65.5646 | +0.0000 |
| skis | 26.2490 | 26.2490 | +0.0000 |
| snowboard | 34.4714 | 34.4714 | +0.0000 |
| sports ball | 25.2606 | 25.2606 | +0.0000 |
| kite | 36.3375 | 36.3375 | +0.0000 |
| baseball bat | 19.9199 | 19.9199 | +0.0000 |
| baseball glove | 67.7804 | 67.7804 | +0.0000 |
| skateboard | 16.7904 | 16.7904 | +0.0000 |
| surfboard | 68.8998 | 68.8998 | +0.0000 |
| tennis racket | 59.7305 | 59.7305 | +0.0000 |
| bottle | 31.6962 | 31.6962 | +0.0000 |
| wine glass | 49.4358 | 49.4358 | +0.0000 |
| cup | 25.9533 | 25.9533 | +0.0000 |
| fork | 36.9940 | 36.9940 | +0.0000 |
| knife | 31.5128 | 31.5128 | +0.0000 |
| spoon | 39.4681 | 39.4681 | +0.0000 |
| bowl | 20.5100 | 20.5100 | +0.0000 |
| banana | 56.4732 | 56.4732 | +0.0000 |
| apple | 54.5282 | 54.5282 | +0.0000 |
| sandwich | 34.2403 | 34.2403 | +0.0000 |
| orange | 76.0882 | 76.0882 | +0.0000 |
| broccoli | 56.7400 | 56.7400 | +0.0000 |
| carrot | 54.0844 | 54.0844 | +0.0000 |
| hot dog | 41.8737 | 41.8737 | +0.0000 |
| pizza | 57.3303 | 57.3303 | +0.0000 |
| donut | 41.0093 | 41.0093 | +0.0000 |
| cake | 32.4226 | 32.4226 | +0.0000 |
| chair | 27.8546 | 27.8546 | +0.0000 |
| couch | 58.0652 | 58.0652 | +0.0000 |
| potted plant | 25.7798 | 25.7798 | +0.0000 |
| bed | 56.7126 | 56.7126 | +0.0000 |
| dining table | 28.0638 | 28.0638 | +0.0000 |
| toilet | 51.4335 | 51.4335 | +0.0000 |
| tv | 33.4142 | 33.4142 | +0.0000 |
| laptop | 52.8100 | 52.8100 | +0.0000 |
| mouse | 71.0951 | 71.0951 | +0.0000 |
| remote | 47.1827 | 47.1827 | +0.0000 |
| keyboard | 62.4311 | 62.4311 | +0.0000 |
| cell phone | 32.8066 | 32.8066 | +0.0000 |
| microwave | 48.6815 | 48.6815 | +0.0000 |
| oven | 41.8611 | 41.8611 | +0.0000 |
| toaster | 73.1087 | 73.1087 | +0.0000 |
| sink | 45.6466 | 45.6466 | +0.0000 |
| refrigerator | 61.7637 | 61.7637 | +0.0000 |
| book | 39.1545 | 39.1545 | +0.0000 |
| clock | 65.6551 | 65.6551 | +0.0000 |
| vase | 31.9337 | 31.9337 | +0.0000 |
| scissors | 63.9161 | 63.9161 | +0.0000 |
| teddy bear | 73.1671 | 73.1671 | +0.0000 |
| hair drier | 24.6937 | 24.6937 | +0.0000 |
| toothbrush | 39.5334 | 39.5334 | +0.0000 |

## coco_stuff171

Frozen profile: `{"bank": "official_segmentation", "strength": 1.0, "coupling": 0.5, "temperature": 0.05, "tau": 1.0, "tem": 1.0, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Changed pixels: `{"coco_stuff171": 0}`. Historical confusion agreement: `{"compact__coco_stuff171": true, "efficient__coco_stuff171": true}`.

Protocol coco_stuff171:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| person | 57.5809 | 57.5809 | +0.0000 |
| bicycle | 63.3126 | 63.3126 | +0.0000 |
| car | 44.0760 | 44.0760 | +0.0000 |
| motorcycle | 72.4010 | 72.4010 | +0.0000 |
| airplane | 44.3283 | 44.3283 | +0.0000 |
| bus | 68.1715 | 68.1715 | +0.0000 |
| train | 61.4632 | 61.4632 | +0.0000 |
| truck | 45.0251 | 45.0251 | +0.0000 |
| boat | 42.9889 | 42.9889 | +0.0000 |
| traffic light | 33.1438 | 33.1438 | +0.0000 |
| fire hydrant | 72.9176 | 72.9176 | +0.0000 |
| stop sign | 29.6526 | 29.6526 | +0.0000 |
| parking meter | 39.6603 | 39.6603 | +0.0000 |
| bench | 37.5523 | 37.5523 | +0.0000 |
| bird | 43.4981 | 43.4981 | +0.0000 |
| cat | 78.1456 | 78.1456 | +0.0000 |
| dog | 66.6913 | 66.6913 | +0.0000 |
| horse | 73.8030 | 73.8030 | +0.0000 |
| sheep | 80.1818 | 80.1818 | +0.0000 |
| cow | 81.0151 | 81.0151 | +0.0000 |
| elephant | 85.7154 | 85.7154 | +0.0000 |
| bear | 86.3392 | 86.3392 | +0.0000 |
| zebra | 84.6159 | 84.6159 | +0.0000 |
| giraffe | 80.4785 | 80.4785 | +0.0000 |
| backpack | 25.5907 | 25.5907 | +0.0000 |
| umbrella | 74.7691 | 74.7691 | +0.0000 |
| handbag | 32.3317 | 32.3317 | +0.0000 |
| tie | 5.8119 | 5.8119 | +0.0000 |
| suitcase | 63.9715 | 63.9715 | +0.0000 |
| frisbee | 34.5490 | 34.5490 | +0.0000 |
| skis | 18.6954 | 18.6954 | +0.0000 |
| snowboard | 12.0402 | 12.0402 | +0.0000 |
| sports ball | 5.7220 | 5.7220 | +0.0000 |
| kite | 37.8577 | 37.8577 | +0.0000 |
| baseball bat | 33.4065 | 33.4065 | +0.0000 |
| baseball glove | 44.9972 | 44.9972 | +0.0000 |
| skateboard | 17.3265 | 17.3265 | +0.0000 |
| surfboard | 65.0422 | 65.0422 | +0.0000 |
| tennis racket | 36.3872 | 36.3872 | +0.0000 |
| bottle | 42.8993 | 42.8993 | +0.0000 |
| wine glass | 48.8847 | 48.8847 | +0.0000 |
| cup | 29.9140 | 29.9140 | +0.0000 |
| fork | 40.0532 | 40.0532 | +0.0000 |
| knife | 29.6198 | 29.6198 | +0.0000 |
| spoon | 37.9700 | 37.9700 | +0.0000 |
| bowl | 19.3120 | 19.3120 | +0.0000 |
| banana | 42.6392 | 42.6392 | +0.0000 |
| apple | 34.1342 | 34.1342 | +0.0000 |
| sandwich | 34.4433 | 34.4433 | +0.0000 |
| orange | 19.7292 | 19.7292 | +0.0000 |
| broccoli | 58.7404 | 58.7404 | +0.0000 |
| carrot | 41.0854 | 41.0854 | +0.0000 |
| hot dog | 39.4482 | 39.4482 | +0.0000 |
| pizza | 56.3030 | 56.3030 | +0.0000 |
| donut | 61.4860 | 61.4860 | +0.0000 |
| cake | 49.3841 | 49.3841 | +0.0000 |
| chair | 32.9560 | 32.9560 | +0.0000 |
| couch | 56.0970 | 56.0970 | +0.0000 |
| potted plant | 22.9245 | 22.9245 | +0.0000 |
| bed | 55.1228 | 55.1228 | +0.0000 |
| dining table | 23.2100 | 23.2100 | +0.0000 |
| toilet | 50.3888 | 50.3888 | +0.0000 |
| tv | 22.8686 | 22.8686 | +0.0000 |
| laptop | 59.9377 | 59.9377 | +0.0000 |
| mouse | 46.0413 | 46.0413 | +0.0000 |
| remote | 50.2981 | 50.2981 | +0.0000 |
| keyboard | 57.1124 | 57.1124 | +0.0000 |
| cell phone | 36.5486 | 36.5486 | +0.0000 |
| microwave | 48.6009 | 48.6009 | +0.0000 |
| oven | 38.1105 | 38.1105 | +0.0000 |
| toaster | 47.0347 | 47.0347 | +0.0000 |
| sink | 46.4514 | 46.4514 | +0.0000 |
| refrigerator | 64.7964 | 64.7964 | +0.0000 |
| book | 33.4470 | 33.4470 | +0.0000 |
| clock | 67.5045 | 67.5045 | +0.0000 |
| vase | 39.3780 | 39.3780 | +0.0000 |
| scissors | 64.7500 | 64.7500 | +0.0000 |
| teddy bear | 71.5062 | 71.5062 | +0.0000 |
| hair drier | 42.5422 | 42.5422 | +0.0000 |
| toothbrush | 30.8565 | 30.8565 | +0.0000 |
| banner | 18.3424 | 18.3424 | +0.0000 |
| blanket | 14.7191 | 14.7191 | +0.0000 |
| branch | 3.9419 | 3.9419 | +0.0000 |
| bridge | 32.4322 | 32.4322 | +0.0000 |
| building-other | 40.5898 | 40.5898 | +0.0000 |
| bush | 22.5058 | 22.5058 | +0.0000 |
| cabinet | 24.1562 | 24.1562 | +0.0000 |
| cage | 12.5920 | 12.5920 | +0.0000 |
| cardboard | 47.8522 | 47.8522 | +0.0000 |
| carpet | 45.8000 | 45.8000 | +0.0000 |
| ceiling-other | 43.5931 | 43.5931 | +0.0000 |
| ceiling-tile | 10.7145 | 10.7145 | +0.0000 |
| cloth | 1.4253 | 1.4253 | +0.0000 |
| clothes | 5.1107 | 5.1107 | +0.0000 |
| clouds | 39.6868 | 39.6868 | +0.0000 |
| counter | 12.1516 | 12.1516 | +0.0000 |
| cupboard | 5.8880 | 5.8880 | +0.0000 |
| curtain | 51.1676 | 51.1676 | +0.0000 |
| desk-stuff | 24.9269 | 24.9269 | +0.0000 |
| dirt | 4.0745 | 4.0745 | +0.0000 |
| door-stuff | 35.2382 | 35.2382 | +0.0000 |
| fence | 30.8998 | 30.8998 | +0.0000 |
| floor-marble | 6.7468 | 6.7468 | +0.0000 |
| floor-other | 14.0472 | 14.0472 | +0.0000 |
| floor-stone | 4.8881 | 4.8881 | +0.0000 |
| floor-tile | 45.6892 | 45.6892 | +0.0000 |
| floor-wood | 34.8019 | 34.8019 | +0.0000 |
| flower | 34.5559 | 34.5559 | +0.0000 |
| fog | 8.2171 | 8.2171 | +0.0000 |
| food-other | 14.6011 | 14.6011 | +0.0000 |
| fruit | 19.6917 | 19.6917 | +0.0000 |
| furniture-other | 1.3013 | 1.3013 | +0.0000 |
| grass | 56.0949 | 56.0949 | +0.0000 |
| gravel | 17.3015 | 17.3015 | +0.0000 |
| ground-other | 5.1897 | 5.1897 | +0.0000 |
| hill | 13.7304 | 13.7304 | +0.0000 |
| house | 12.5723 | 12.5723 | +0.0000 |
| leaves | 14.0270 | 14.0270 | +0.0000 |
| light | 23.8591 | 23.8591 | +0.0000 |
| mat | 6.9947 | 6.9947 | +0.0000 |
| metal | 8.1187 | 8.1187 | +0.0000 |
| mirror-stuff | 19.5749 | 19.5749 | +0.0000 |
| moss | 8.1250 | 8.1250 | +0.0000 |
| mountain | 42.9898 | 42.9898 | +0.0000 |
| mud | 4.8737 | 4.8737 | +0.0000 |
| napkin | 19.5786 | 19.5786 | +0.0000 |
| net | 24.7054 | 24.7054 | +0.0000 |
| paper | 25.2443 | 25.2443 | +0.0000 |
| pavement | 39.3016 | 39.3016 | +0.0000 |
| pillow | 2.7608 | 2.7608 | +0.0000 |
| plant-other | 5.0735 | 5.0735 | +0.0000 |
| plastic | 6.9078 | 6.9078 | +0.0000 |
| platform | 8.1302 | 8.1302 | +0.0000 |
| playingfield | 42.4445 | 42.4445 | +0.0000 |
| railing | 4.1258 | 4.1258 | +0.0000 |
| railroad | 46.0180 | 46.0180 | +0.0000 |
| river | 24.9676 | 24.9676 | +0.0000 |
| road | 41.2090 | 41.2090 | +0.0000 |
| rock | 35.1781 | 35.1781 | +0.0000 |
| roof | 14.8308 | 14.8308 | +0.0000 |
| rug | 23.5546 | 23.5546 | +0.0000 |
| salad | 8.7054 | 8.7054 | +0.0000 |
| sand | 45.8609 | 45.8609 | +0.0000 |
| sea | 74.8899 | 74.8899 | +0.0000 |
| shelf | 22.7316 | 22.7316 | +0.0000 |
| sky-other | 59.0040 | 59.0040 | +0.0000 |
| skyscraper | 20.5779 | 20.5779 | +0.0000 |
| snow | 78.6620 | 78.6620 | +0.0000 |
| solid-other | 0.5720 | 0.5720 | +0.0000 |
| stairs | 21.8787 | 21.8787 | +0.0000 |
| stone | 3.0037 | 3.0037 | +0.0000 |
| straw | 21.1509 | 21.1509 | +0.0000 |
| structural-other | 0.0064 | 0.0064 | +0.0000 |
| table | 2.5187 | 2.5187 | +0.0000 |
| tent | 6.5271 | 6.5271 | +0.0000 |
| textile-other | 2.7027 | 2.7027 | +0.0000 |
| towel | 32.4807 | 32.4807 | +0.0000 |
| tree | 34.9507 | 34.9507 | +0.0000 |
| vegetable | 16.9679 | 16.9679 | +0.0000 |
| wall-brick | 40.6537 | 40.6537 | +0.0000 |
| wall-concrete | 3.3729 | 3.3729 | +0.0000 |
| wall-other | 8.6951 | 8.6951 | +0.0000 |
| wall-panel | 4.0234 | 4.0234 | +0.0000 |
| wall-stone | 22.3365 | 22.3365 | +0.0000 |
| wall-tile | 46.1225 | 46.1225 | +0.0000 |
| wall-wood | 28.7199 | 28.7199 | +0.0000 |
| water-other | 24.4989 | 24.4989 | +0.0000 |
| waterdrops | 0.7850 | 0.7850 | +0.0000 |
| window-blind | 13.6706 | 13.6706 | +0.0000 |
| window-other | 29.3908 | 29.3908 | +0.0000 |
| wood | 12.9317 | 12.9317 | +0.0000 |

## flair1

Frozen profile: `{"bank": "original", "strength": "original", "coupling": 0.25, "temperature": 0.07, "tau": 1.0, "tem": 1.0, "wide_policy": "long448", "background_rule": "retained"}`.

Changed pixels: `{"flair1": 0}`. Historical confusion agreement: `{"compact__flair1": true, "efficient__flair1": true}`.

Protocol flair1:

| Class | Historical IoU | Efficient IoU | Δ pp |
| --- | ---: | ---: | ---: |
| building | 61.4040 | 61.4040 | +0.0000 |
| pervious surface | 33.7509 | 33.7509 | +0.0000 |
| impervious surface | 60.0740 | 60.0740 | +0.0000 |
| bare soil | 39.6440 | 39.6440 | +0.0000 |
| water | 78.6820 | 78.6820 | +0.0000 |
| coniferous | 13.9215 | 13.9215 | +0.0000 |
| deciduous | 54.0531 | 54.0531 | +0.0000 |
| brushwood | 27.2315 | 27.2315 | +0.0000 |
| vineyard | 58.4955 | 58.4955 | +0.0000 |
| herbaceous vegetation | 39.8215 | 39.8215 | +0.0000 |
| agricultural land | 32.0117 | 32.0117 | +0.0000 |
| plowed land | 24.8056 | 24.8056 | +0.0000 |
