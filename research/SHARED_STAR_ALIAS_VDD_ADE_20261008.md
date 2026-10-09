# Fixed20 shared-star alias experiment: VDD and ADE150

Developed model/datasets; exploratory complete validation, not untouched independent validation.

Prior mask-free smoke stopped on missing new diagnostic fields (KeyError maximum_risk_elements). The new wrapper uses a private extensible diagnostic dictionary; original source module is untouched. Failed outputs/logs preserved at /data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926/results/shared_star_alias_vdd_ade_20261008

Primary2 and sensitivity4 rivals are frozen. Same SharedLocal semantic observation, continuous canonical-protected fixed-slot attenuation, minimum-norm star edge projection, original H. No added fine image encodings. This is a sparse approximation, not an equivalent dense acceleration.

VIP paper numeric targets are VDD54.3 and ADE15029.1 from the saved table transcriptions; these local protocols do not certify paper reproduction. Matched20 VIP is rerun on identical inputs/words; official-query local historical VIP VDD52.0647/ADE29.1387 is a separate comparison.

Initial co-resident timing is retained as invalid acceptance data. The first exclusive attempt also stopped on GPU occupancy and is preserved at benchmark_exclusive. v2 recovered VDD, but its ADE timing stopped on GPU occupancy as tmux continued the suspended dispatcher. v3 exits only that dispatcher, preserves active GPU workers, and reruns ADE timing. The table below accepts only exclusive remeasurements with before/after process checks on every repetition.

| Dataset | Images | Patch2 | Same-source mean | Star2 | Star4 | Pooled class | Alias shuffle | Matched20 VIP | Paper target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 80 | 53.4600 | 53.6976 | 53.1345 | 53.1359 | 53.3182 | 53.0792 | 51.4827 | 54.3 |
| ade150 | 2000 | 24.6903 | 25.5635 | 25.7319 | 25.7478 | 25.8000 | 25.7136 | 23.5584 | 29.1 |

| Dataset | Patch2 ms | Same-source mean ms | Star2 ms | Star4 ms | VIP ms | Star2/VIP | Star2 alias+head overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 451.54 | 463.74 | 492.15 | 490.14 | 186.80 | 2.635x | 8.99% |
| ade150 | 272.22 | 383.20 | 413.66 | 411.14 | 121.61 | 3.402x | 51.96% |

Three fixed complete images,5 rotated synchronized warmed singleton repeats; whole-image inference including resize/all branches/alias/writeback/restoration/argmax. Excludes decode/initialization. One exclusive GPU, other GPU workers recorded as allowed. Shared Geometry/VIP and frozen head snapshots resident; memory is not standalone VIP deployment memory. Full multi-arm wall time is not singleton latency.

## vdd

{"miou": {"Geometry_PatchOnly2Coupled": 53.46, "SharedLocal_ObservationMean": 53.6976, "SharedStar2_Soft": 53.1345, "SharedStar4_Soft": 53.1359, "SharedStar2_PooledClass": 53.3182, "SharedStar2_AliasShuffle": 53.0792, "VIP_Matched20": 51.4827}, "above_paper_target": false, "above_mean": false, "above_pooled": false, "above_shuffle": true, "below6x": true, "paired_image_bootstrap": {"SharedLocal_ObservationMean": {"delta_pp": -0.5630999999999986, "interval95_pp": [-1.2916432979381773, 0.03731027629077194]}, "SharedStar2_PooledClass": {"delta_pp": -0.18369999999999465, "interval95_pp": [-0.506587076776583, 0.11618083129748838]}, "SharedStar2_AliasShuffle": {"delta_pp": 0.05530000000000257, "interval95_pp": [-0.014735726093342904, 0.11962485772436218]}}}

| Class | Patch2 IoU | Mean IoU | Star2 IoU | Star4 IoU | VIP IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| other | 32.7096 | 30.1261 | 29.0832 | 29.0980 | 36.5655 |
| wall | 43.4422 | 46.1911 | 46.4519 | 46.4376 | 29.0582 |
| road | 40.6842 | 40.0263 | 39.7728 | 39.7467 | 45.0432 |
| vegetation | 68.1015 | 70.6941 | 71.2021 | 71.1978 | 50.2515 |
| vehicle | 22.7793 | 23.8422 | 23.8243 | 23.8098 | 29.6526 |
| roof | 81.4826 | 82.5622 | 82.8307 | 82.8648 | 85.5153 |
| water | 85.0207 | 82.4409 | 78.7768 | 78.7967 | 84.2928 |

Full evaluation peak allocation (shared multi-arm execution): 5891.2333984375 MiB.
Unique coverage verified; matched archived Patch2 confusion replayed; all paired scored target counts equal.
Full-image pixel flips against same-source mean: {"SharedStar2_Soft": {"wrong_to_correct": 4474319, "correct_to_wrong": 9711448, "changed": 15788787}, "SharedStar4_Soft": {"wrong_to_correct": 4531083, "correct_to_wrong": 9771399, "changed": 16007012}, "SharedStar2_PooledClass": {"wrong_to_correct": 11598794, "correct_to_wrong": 16943275, "changed": 32851343}, "SharedStar2_AliasShuffle": {"wrong_to_correct": 4437548, "correct_to_wrong": 10358715, "changed": 16403058}}.
Paired complete-image bootstrap:1000 resamples, seed20261008; intervals describe these developed validation images, not independent model-selection evidence.

## ade150

{"miou": {"Geometry_PatchOnly2Coupled": 24.6903, "SharedLocal_ObservationMean": 25.5635, "SharedStar2_Soft": 25.7319, "SharedStar4_Soft": 25.7478, "SharedStar2_PooledClass": 25.8, "SharedStar2_AliasShuffle": 25.7136, "VIP_Matched20": 23.5584}, "above_paper_target": false, "above_mean": true, "above_pooled": false, "above_shuffle": true, "below6x": true, "paired_image_bootstrap": {"SharedLocal_ObservationMean": {"delta_pp": 0.16839999999999833, "interval95_pp": [0.0890263879225996, 0.22168235680619972]}, "SharedStar2_PooledClass": {"delta_pp": -0.06810000000000116, "interval95_pp": [-0.11976002185689341, -0.004092205427958185]}, "SharedStar2_AliasShuffle": {"delta_pp": 0.018299999999999983, "interval95_pp": [0.010625490768435598, 0.02782297336415409]}}}

| Class | Patch2 IoU | Mean IoU | Star2 IoU | Star4 IoU | VIP IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | 26.2872 | 30.1089 | 31.3000 | 31.3767 | 25.8631 |
| building | 55.4727 | 55.1151 | 54.9651 | 54.9706 | 53.3015 |
| sky | 65.1888 | 69.0051 | 70.1536 | 70.2369 | 63.6441 |
| floor | 50.5038 | 51.3205 | 51.1416 | 51.1525 | 44.9483 |
| tree | 48.1482 | 54.3610 | 56.1755 | 56.2546 | 45.6540 |
| ceiling | 62.7281 | 64.4182 | 64.8267 | 64.8211 | 60.0070 |
| road | 70.4884 | 72.5074 | 72.7604 | 72.7739 | 60.8633 |
| bed | 54.7096 | 58.1934 | 59.5098 | 59.5934 | 56.0503 |
| windowpane | 40.1672 | 39.7799 | 39.4796 | 39.4710 | 36.6402 |
| grass | 45.9499 | 48.8201 | 49.9457 | 49.9814 | 46.9592 |
| cabinet | 39.7252 | 42.9327 | 43.9987 | 44.0083 | 39.0322 |
| sidewalk | 49.0140 | 50.7909 | 50.9613 | 50.9823 | 45.5705 |
| person | 39.0287 | 41.6132 | 42.5177 | 42.6331 | 48.7294 |
| earth | 6.6422 | 7.0084 | 7.2751 | 7.2973 | 8.8234 |
| door | 34.9703 | 35.1841 | 35.0488 | 35.0481 | 32.6800 |
| table | 30.5306 | 32.5542 | 33.1593 | 33.1976 | 28.9759 |
| mountain | 31.1963 | 33.0374 | 33.6533 | 33.6428 | 30.4821 |
| plant | 20.6914 | 25.2663 | 26.5275 | 26.5944 | 17.3886 |
| curtain | 60.4051 | 61.8644 | 61.9609 | 61.9624 | 52.2927 |
| chair | 28.3288 | 31.4094 | 31.9365 | 31.9960 | 22.4116 |
| car | 62.3336 | 64.0181 | 64.2786 | 64.2987 | 55.9464 |
| water | 42.2883 | 44.1955 | 44.7146 | 44.6854 | 40.1327 |
| painting | 30.2791 | 32.0962 | 32.7266 | 32.7355 | 30.3312 |
| sofa | 52.9647 | 54.3831 | 54.7181 | 54.7468 | 53.2343 |
| shelf | 28.3151 | 28.2365 | 28.1058 | 28.0738 | 27.4937 |
| house | 18.9319 | 20.5653 | 20.8279 | 20.8883 | 19.6189 |
| sea | 25.0470 | 25.9166 | 26.1031 | 26.1529 | 23.0917 |
| mirror | 35.5499 | 34.6795 | 34.2575 | 34.2030 | 33.6224 |
| rug | 26.4859 | 25.4647 | 24.9417 | 24.9358 | 22.1157 |
| field | 18.3966 | 19.2603 | 19.7916 | 19.8120 | 19.5365 |
| armchair | 21.3202 | 21.2401 | 21.9587 | 21.9939 | 28.0530 |
| seat | 35.7894 | 35.4368 | 35.4067 | 35.3634 | 37.1662 |
| fence | 25.7123 | 25.9620 | 25.6369 | 25.6308 | 22.7075 |
| desk | 33.7214 | 34.6350 | 34.4925 | 34.5312 | 31.4378 |
| rock | 36.7569 | 37.7893 | 37.7932 | 37.7704 | 33.3710 |
| wardrobe | 45.2308 | 45.4046 | 44.9766 | 44.9664 | 43.2449 |
| lamp | 29.2906 | 29.7353 | 30.0113 | 29.9903 | 25.2911 |
| bathtub | 49.2460 | 52.4428 | 53.5803 | 53.6925 | 50.4130 |
| railing | 21.1772 | 19.4530 | 18.9393 | 18.9099 | 20.8623 |
| cushion | 28.3476 | 28.7877 | 27.8808 | 27.8844 | 21.3525 |
| base | 0.0469 | 0.0357 | 0.0331 | 0.0283 | 0.0612 |
| box | 24.2532 | 25.6871 | 25.7757 | 25.7848 | 19.7734 |
| column | 35.6703 | 34.2643 | 33.6704 | 33.6143 | 34.1766 |
| signboard | 21.9836 | 22.6726 | 22.7137 | 22.7087 | 20.4579 |
| chest of drawers | 27.9888 | 27.2288 | 26.9703 | 26.9745 | 30.7292 |
| counter | 25.1012 | 24.1817 | 23.8123 | 23.8413 | 24.3403 |
| sand | 37.9052 | 35.9981 | 35.1366 | 35.0850 | 37.7386 |
| sink | 36.8591 | 41.8245 | 43.6147 | 43.6995 | 36.1642 |
| skyscraper | 19.9369 | 19.1789 | 18.5903 | 18.5515 | 21.1517 |
| fireplace | 38.3817 | 39.5548 | 39.6817 | 39.7246 | 38.7919 |
| refrigerator | 49.0827 | 49.3904 | 49.0276 | 49.0664 | 46.8673 |
| grandstand | 29.7858 | 31.5432 | 31.9928 | 32.0029 | 30.3176 |
| path | 3.1151 | 3.2195 | 3.2484 | 3.2687 | 3.6144 |
| stairs | 25.0795 | 26.6879 | 27.2454 | 27.2644 | 22.5852 |
| runway | 34.1910 | 35.6411 | 36.2369 | 36.2791 | 33.3931 |
| case | 2.0391 | 2.6739 | 2.9098 | 2.9221 | 2.5977 |
| pool table | 68.1225 | 71.4312 | 72.2339 | 72.3655 | 68.5410 |
| pillow | 1.6365 | 2.7612 | 3.6265 | 3.6251 | 2.9169 |
| screen door | 0.2115 | 0.2456 | 0.2430 | 0.2423 | 0.3032 |
| stairway | 2.9227 | 3.0391 | 3.1806 | 3.2157 | 4.8492 |
| river | 18.6316 | 18.0850 | 17.4370 | 17.4038 | 16.2521 |
| bridge | 22.4870 | 23.5800 | 24.8032 | 24.8016 | 21.1776 |
| bookcase | 17.4085 | 16.4394 | 16.1745 | 16.1566 | 17.1624 |
| blind | 22.9662 | 23.0866 | 23.3065 | 23.2930 | 25.1398 |
| coffee table | 37.9830 | 40.4676 | 41.2060 | 41.2304 | 37.6577 |
| toilet | 43.8863 | 46.3579 | 46.8913 | 46.9582 | 38.7720 |
| flower | 10.9567 | 11.6302 | 11.5087 | 11.4998 | 10.3229 |
| book | 4.8397 | 5.8248 | 6.3015 | 6.3170 | 6.6451 |
| hill | 2.6531 | 3.3550 | 3.5846 | 3.6102 | 2.5161 |
| bench | 19.5263 | 20.1027 | 20.3891 | 20.4049 | 22.9069 |
| countertop | 13.1506 | 13.8133 | 14.0001 | 14.0330 | 12.5151 |
| stove | 27.8596 | 26.6984 | 26.9477 | 26.9927 | 35.4350 |
| palm | 20.7827 | 21.3125 | 21.4012 | 21.3861 | 19.3777 |
| kitchen island | 7.2913 | 8.8439 | 9.6275 | 9.6628 | 8.4105 |
| computer | 35.0416 | 35.0808 | 34.5897 | 34.6263 | 33.7831 |
| swivel chair | 18.3521 | 17.1337 | 16.9017 | 16.8560 | 19.8216 |
| boat | 42.4942 | 44.4383 | 43.5214 | 43.4983 | 33.9027 |
| bar | 17.1000 | 18.1689 | 18.4914 | 18.5522 | 17.1979 |
| arcade machine | 17.9777 | 16.7084 | 16.4400 | 16.3963 | 21.6014 |
| hovel | 6.4820 | 6.8653 | 6.7889 | 6.8046 | 6.7407 |
| bus | 59.7157 | 59.2776 | 59.4621 | 59.4132 | 62.2771 |
| towel | 38.1900 | 41.3539 | 41.3279 | 41.3895 | 30.4985 |
| light | 6.2769 | 6.3320 | 6.4795 | 6.4409 | 4.3852 |
| truck | 16.8151 | 17.4768 | 17.5234 | 17.5132 | 13.6913 |
| tower | 6.0355 | 5.7512 | 5.7006 | 5.6891 | 6.5894 |
| chandelier | 17.7555 | 18.2841 | 18.6165 | 18.6054 | 19.5829 |
| awning | 13.9899 | 12.4836 | 12.4328 | 12.4457 | 15.1988 |
| streetlight | 7.0489 | 7.7728 | 8.1778 | 8.1890 | 8.0364 |
| booth | 3.4656 | 3.5809 | 3.7352 | 3.7439 | 4.6637 |
| television receiver | 31.8155 | 29.0379 | 27.3696 | 27.4373 | 28.6540 |
| airplane | 16.6806 | 18.4553 | 19.2317 | 19.2959 | 16.0018 |
| dirt track | 0.8308 | 0.9189 | 0.9754 | 0.9830 | 1.1527 |
| apparel | 4.1816 | 4.5904 | 4.6505 | 4.6450 | 5.7030 |
| pole | 7.1115 | 7.1061 | 7.0192 | 6.9872 | 6.1566 |
| land | 2.0624 | 2.2645 | 2.4176 | 2.4247 | 2.2566 |
| bannister | 6.5286 | 6.8059 | 6.8397 | 6.7572 | 6.6971 |
| escalator | 34.9032 | 36.5766 | 37.5346 | 37.4152 | 33.9924 |
| ottoman | 45.8392 | 45.1271 | 44.3678 | 44.3329 | 40.1127 |
| bottle | 28.3776 | 30.1391 | 30.4388 | 30.4655 | 30.7252 |
| buffet | 0.1756 | 0.2112 | 0.2234 | 0.2267 | 0.1923 |
| poster | 7.7774 | 7.7569 | 7.6329 | 7.6470 | 9.5588 |
| stage | 6.6257 | 7.5381 | 7.9605 | 8.0318 | 6.6852 |
| van | 22.6616 | 21.9419 | 21.9640 | 21.8929 | 21.4639 |
| ship | 3.3744 | 3.7174 | 4.0079 | 4.0163 | 5.1776 |
| fountain | 21.6272 | 21.5020 | 21.2692 | 21.3418 | 21.4403 |
| conveyer belt | 42.9349 | 41.4348 | 40.6595 | 40.5057 | 40.6773 |
| canopy | 3.2268 | 3.2528 | 3.4009 | 3.4012 | 1.3104 |
| washer | 78.2916 | 75.6598 | 73.4175 | 73.4714 | 74.3316 |
| plaything | 4.4674 | 4.5540 | 4.3602 | 4.4375 | 3.6703 |
| swimming pool | 16.4347 | 18.2442 | 18.8861 | 18.9584 | 16.8388 |
| stool | 13.2908 | 14.9412 | 15.2831 | 15.2949 | 11.7514 |
| barrel | 4.2828 | 4.3998 | 4.4298 | 4.4385 | 3.7162 |
| basket | 33.2152 | 32.7495 | 32.2233 | 32.2057 | 27.4059 |
| waterfall | 19.8193 | 21.4097 | 21.1099 | 21.1481 | 18.8599 |
| tent | 31.9791 | 38.0290 | 40.2999 | 40.4114 | 36.7223 |
| bag | 24.6185 | 24.6167 | 24.6701 | 24.7093 | 22.5168 |
| minibike | 0.2054 | 0.3544 | 0.4104 | 0.4096 | 0.4633 |
| cradle | 42.5587 | 42.3305 | 40.8171 | 40.8594 | 40.5457 |
| oven | 13.4868 | 15.4791 | 16.1057 | 16.1538 | 9.6694 |
| ball | 10.1176 | 12.0696 | 13.0331 | 13.1280 | 9.1423 |
| food | 39.7605 | 43.3928 | 45.7229 | 45.7569 | 37.4312 |
| step | 1.1587 | 1.0955 | 1.0381 | 1.0523 | 1.0003 |
| tank | 1.0683 | 1.6207 | 1.7517 | 1.7454 | 0.7053 |
| trade name | 2.9905 | 3.1406 | 2.9844 | 3.0065 | 3.4578 |
| microwave | 53.4153 | 55.6116 | 55.5145 | 55.5210 | 46.2885 |
| pot | 22.9999 | 24.1209 | 24.5616 | 24.5422 | 18.3340 |
| animal | 47.7141 | 51.2937 | 52.2486 | 52.3676 | 41.2381 |
| bicycle | 31.7816 | 32.3221 | 32.2251 | 32.2074 | 26.4538 |
| lake | 7.9593 | 6.1803 | 4.9921 | 4.9796 | 6.7962 |
| dishwasher | 40.4913 | 37.3647 | 36.4802 | 36.3918 | 36.6096 |
| screen | 32.3341 | 35.5432 | 35.7934 | 35.8948 | 25.9348 |
| blanket | 21.5664 | 25.7119 | 25.6522 | 25.6962 | 20.6301 |
| sculpture | 15.8014 | 15.9810 | 15.8694 | 15.8360 | 14.0652 |
| hood | 7.4941 | 8.7218 | 9.2634 | 9.3260 | 8.2073 |
| sconce | 3.4967 | 2.8751 | 2.7367 | 2.7167 | 3.6446 |
| vase | 12.6567 | 13.4135 | 13.1970 | 13.1879 | 9.8040 |
| traffic light | 7.8906 | 9.0901 | 9.5973 | 9.6051 | 8.7389 |
| tray | 6.5102 | 8.4496 | 8.7798 | 8.8575 | 4.4258 |
| ashcan | 0.3798 | 0.5028 | 0.5629 | 0.5706 | 0.0804 |
| fan | 38.4791 | 39.6001 | 39.9601 | 40.0223 | 28.8258 |
| pier | 5.3278 | 5.9598 | 6.2922 | 6.3209 | 5.4721 |
| crt screen | 0.2904 | 0.1609 | 0.1524 | 0.1521 | 0.4657 |
| plate | 16.9826 | 22.6297 | 24.1372 | 24.3406 | 13.6323 |
| monitor | 9.5388 | 9.7374 | 9.6582 | 9.6744 | 10.6096 |
| bulletin board | 2.3192 | 2.0566 | 1.9566 | 1.9530 | 2.5507 |
| shower | 0.4332 | 0.4549 | 0.4709 | 0.4712 | 0.4593 |
| radiator | 37.6359 | 38.3496 | 38.6699 | 38.7065 | 34.3641 |
| glass | 3.6965 | 4.7223 | 5.0577 | 5.0688 | 2.6584 |
| clock | 40.2114 | 44.6600 | 44.8926 | 44.9489 | 33.1601 |
| flag | 22.8165 | 23.9865 | 24.3987 | 24.3721 | 23.1466 |

Full evaluation peak allocation (shared multi-arm execution): 10988.89013671875 MiB.
Unique coverage verified; matched archived Patch2 confusion replayed; all paired scored target counts equal.
Full-image pixel flips against same-source mean: {"SharedStar2_Soft": {"wrong_to_correct": 4885326, "correct_to_wrong": 2268988, "changed": 12998439}, "SharedStar4_Soft": {"wrong_to_correct": 5149455, "correct_to_wrong": 2384736, "changed": 14021743}, "SharedStar2_PooledClass": {"wrong_to_correct": 7802920, "correct_to_wrong": 4586132, "changed": 22300096}, "SharedStar2_AliasShuffle": {"wrong_to_correct": 5100351, "correct_to_wrong": 2522807, "changed": 13870101}}.
Paired complete-image bootstrap:1000 resamples, seed20261008; intervals describe these developed validation images, not independent model-selection evidence.

All two full datasets and timing complete: True.
A target crossing alone does not establish alias-specific value or CVPR novelty. Budget4 is sensitivity and is not silently substituted for the primary. Negative controls/results are retained.
