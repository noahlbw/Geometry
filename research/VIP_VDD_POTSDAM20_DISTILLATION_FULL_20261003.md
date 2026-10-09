# VDD/Potsdam Full Fixed20 VIP Paper-Rule Distillation

Same frozen20 candidate pools as RivalFineHard20. Same pinned VIP readout and official dataset settings for both All20 and Distilled; 80 ImageNet templates, long-edge448/crop336/stride112, logit scale40. VDD: tau1/tem1/background threshold0.35/index0. Potsdam: tau1/tem2/threshold0.25/index5. Self-Value numerical fallback ONLY on completely masked proxy rows, identical to the corrected official-short comparator.

Selection is the unchanged VIP paper-rule reimplementation: replace only the parent canonical query; multiclass softmax, all-layer attention, affinity power2/two walks, high probability>=0.4, text cosine>=0.7, strictly higher VG and lower entropy than canonical. Keep canonical; no quota. Use ALL unlabeled evaluation images, globally freeze selection before loading masks (transductive). Public pinned VIP does not include the distillation code. Official-short versus All20 changes the words/count; only All20 versus Distilled isolates selection.

| Dataset | Images | VIP official short | VIP All20 | VIP Distilled | Distilled - All20 pp | Distilled - official pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 80 | 52.0647 | 51.4827 | 50.9480 | -0.5347 | -1.1166 |
| potsdam | 504 | 44.8995 | 42.3028 | 37.0558 | -5.2471 | -7.8438 |

## VDD

Selected aliases/class: [13, 18, 3, 12, 15, 11, 8].

| Class | Official IoU | All20 IoU | Distilled IoU | Selection delta pp |
| --- | ---: | ---: | ---: | ---: |
| other | 35.5829 | 36.5655 | 38.6322 | 2.0667 |
| wall | 40.1862 | 29.0582 | 40.4930 | 11.4348 |
| road | 40.5200 | 45.0432 | 33.5352 | -11.5080 |
| vegetation | 52.5014 | 50.2515 | 55.7460 | 5.4945 |
| vehicle | 25.2671 | 29.6526 | 22.4267 | -7.2259 |
| roof | 81.6539 | 85.5153 | 85.6912 | 0.1759 |
| water | 88.7412 | 84.2928 | 80.1118 | -4.1810 |

| Class | Arm | Precision % | Recall % | Predicted area % of scored pixels |
| --- | --- | ---: | ---: | ---: |
| vehicle | Official | 32.1751 | 54.0622 | 0.8748 |
| water | Official | 94.6192 | 93.4576 | 15.0004 |
| vehicle | VIP_All20 | 39.6519 | 54.0414 | 0.7096 |
| water | VIP_All20 | 90.9998 | 91.9593 | 15.3470 |
| vehicle | VIP_Distilled | 24.3578 | 73.8825 | 1.5792 |
| water | VIP_Distilled | 94.4808 | 84.0449 | 13.5094 |

Retained aliases:

- other: other, other area, other land cover, miscellaneous area, miscellaneous ground, unclassified area, unlabeled area, mixed ground, unknown area, non target area, unassigned area, other ground, remaining area
- wall: wall, facade, facades, building facade, exterior wall, building wall, house wall, vertical wall, vertical building facade, visible building side, front facade, side facade, concrete wall, brick wall, apartment facade, commercial facade, residential facade, building exterior wall
- road: road, roadway, road lane
- vegetation: vegetation, trees, tree canopy, forest, shrubs, bushes, greenery, leafy cover, natural vegetation, vegetated ground, park vegetation, vegetation patch
- vehicle: vehicle, vehicles, car, cars, automobile, truck, bus, moving car, parked car, moving vehicle, road vehicle, traffic vehicle, large vehicle, motor vehicle, transport vehicle
- roof: roof, roofs, rooftop, rooftops, building roof, blue roof, flat roof, visible roof, urban roof, large roof, covered roof
- water: water, water body, lake, pond, lake surface, pond water, inland water, water area

Selection parallel 17.930s; aggregate 35.570 GPU-s; peak 1784.641MiB. Paired cached aggregation parallel 19.231s; peak 844.480MiB. These timers exclude initialization; cached paired aggregation is not standalone fresh-image inference cost.

## POTSDAM

Selected aliases/class: [15, 8, 10, 2, 5, 4].

| Class | Official IoU | All20 IoU | Distilled IoU | Selection delta pp |
| --- | ---: | ---: | ---: | ---: |
| impervious surface | 60.7547 | 61.2488 | 49.7563 | -11.4925 |
| building | 71.0553 | 77.2218 | 79.0348 | 1.8130 |
| low vegetation | 52.7964 | 28.5736 | 41.3200 | 12.7464 |
| tree | 56.8492 | 50.9752 | 4.5346 | -46.4406 |
| car | 16.8657 | 30.4756 | 44.8385 | 14.3629 |
| clutter | 11.0760 | 5.3220 | 2.8503 | -2.4717 |

| Class | Arm | Precision % | Recall % | Predicted area % of scored pixels |
| --- | --- | ---: | ---: | ---: |
| low vegetation | Official | 77.0456 | 62.6513 | 17.0838 |
| car | Official | 59.1739 | 19.0866 | 0.6338 |
| low vegetation | VIP_All20 | 92.9464 | 29.2069 | 6.6017 |
| car | VIP_All20 | 30.7704 | 96.9522 | 6.1911 |
| low vegetation | VIP_Distilled | 79.0913 | 46.3871 | 12.3217 |
| car | VIP_Distilled | 50.1357 | 80.9298 | 3.1718 |

Retained aliases:

- impervious surface: impervious surface, paved surface, pavement, asphalt surface, concrete surface, road pavement, parking pavement, sealed ground, built ground, urban pavement, sidewalk, paved area, gray pavement, street surface, concrete ground
- building: building, buildings, house, houses, rooftop, roof, building roof, urban building
- low vegetation: low vegetation, grass, grassland, short grass, herbaceous cover, lawn, green field, low greenery, vegetated ground, grass cover
- tree: tree, evergreen trees
- car: car, cars, vehicles, automobile, parked car
- clutter: clutter, mixed clutter, construction debris, miscellaneous ground

Selection parallel 42.798s; aggregate 170.795 GPU-s; peak 1770.180MiB. Paired cached aggregation parallel 10.798s; peak 69.935MiB. These timers exclude initialization; cached paired aggregation is not standalone fresh-image inference cost.
