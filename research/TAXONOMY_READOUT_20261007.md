# Taxonomy-driven readout: frozen full comparison

Canonical-rank/per-image bank decisions use no target masks. Candidate banks and RS priors retain prior labelled-development provenance. Developed domains, exploratory results. No background confidence rejection.

| Domain | Reference | Taxonomy adaptive | Taxonomy + soft alias | Prior per-protocol tuned |
| --- | ---: | ---: | ---: | ---: |
| vdd | 53.4600 | 55.1305 | 53.2906 | 56.2807 |
| potsdam | 45.9339 | 49.7278 | 42.8772 | 51.1828 |
| voc21 | 29.6765 | 70.3179 | 70.5277 | 70.5894 |
| context60 | 36.7248 | 40.2354 | 40.5552 | 40.2949 |
| ade150 | 24.6903 | 31.1076 | 31.3379 | 31.1076 |

## Nondevelopment complements

| Domain | Images | Reference | Taxonomy adaptive | Taxonomy + soft alias |
| --- | ---: | ---: | ---: | ---: |
| vdd | 64 | 54.6531 | 55.7513 | 53.7805 |
| potsdam | 440 | 45.9512 | 49.7879 | 42.9465 |
| voc21 | 1385 | 29.6305 | 70.3522 | 70.5676 |
| context60 | 5041 | 36.7208 | 40.2366 | 40.5465 |
| ade150 | 2000 | 24.6903 | 31.1076 | 31.3379 |

## Selected routes

- vdd: {"natural_static": null, "profile_counts": {"{\"bank\": \"focused20\", \"coupling\": 0.5, \"strength\": 3.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 1, "{\"bank\": \"original_imagenet\", \"coupling\": 0.5, \"strength\": 1.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 79}}
- potsdam: {"natural_static": null, "profile_counts": {"{\"bank\": \"focused20\", \"coupling\": 0.5, \"strength\": 3.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 371, "{\"bank\": \"original_imagenet\", \"coupling\": 0.5, \"strength\": 1.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 133}}
- voc21: {"natural_static": {"canonical_rank_fraction": 0.6243211030960083, "rank_odds": 1.6618476796037853, "selected_gain": 2.0, "selected_strength": 2.0}, "profile_counts": {"{\"bank\": \"semantic_segmentation\", \"coupling\": 2.0, \"strength\": 2.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 1449}}
- context60: {"natural_static": {"canonical_rank_fraction": 0.5224696397781372, "rank_odds": 1.0941076909442897, "selected_gain": 1.0, "selected_strength": 2.0}, "profile_counts": {"{\"bank\": \"semantic_segmentation\", \"coupling\": 1.0, \"strength\": 2.0, \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 5105}}
- ade150: {"natural_static": {"canonical_rank_fraction": 0.3416299819946289, "rank_odds": 0.5189027031176894, "selected_gain": 0.5, "selected_strength": "original"}, "profile_counts": {"{\"bank\": \"semantic_segmentation\", \"coupling\": 0.5, \"strength\": \"original\", \"tau\": 1.0, \"tem\": 1.0, \"temperature\": 0.07}": 2000}}

## Residual coverage

| Domain | Method | Residual IoU | Precision | Recall | Predicted area | GT area |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd | Reference | 32.7096 | 53.4974 | 45.7046 | 16.0711 | 18.8112 |
| vdd | TaxonomyAdaptive | 35.2289 | 52.1050 | 52.1001 | 18.8095 | 18.8112 |
| vdd | TaxonomySoft | 36.7391 | 52.4376 | 55.1004 | 19.7665 | 18.8112 |
| potsdam | Reference | 6.5825 | 9.9072 | 16.3986 | 7.6047 | 4.5944 |
| potsdam | TaxonomyAdaptive | 8.9316 | 14.0279 | 19.7332 | 6.4630 | 4.5944 |
| potsdam | TaxonomySoft | 8.5840 | 12.7596 | 20.7800 | 7.4823 | 4.5944 |
| voc21 | Reference | 22.9584 | 98.6270 | 23.0321 | 17.1217 | 73.3180 |
| voc21 | TaxonomyAdaptive | 89.5149 | 95.5793 | 93.3811 | 71.6318 | 73.3180 |
| voc21 | TaxonomySoft | 89.6175 | 95.5859 | 93.4864 | 71.7076 | 73.3180 |
| context60 | Reference | 4.3841 | 41.3200 | 4.6751 | 0.9419 | 8.3252 |
| context60 | TaxonomyAdaptive | 0.0701 | 37.5498 | 0.0701 | 0.0156 | 8.3252 |
| context60 | TaxonomySoft | 0.0565 | 39.9521 | 0.0565 | 0.0118 | 8.3252 |

## Per-class outcomes

| Domain | Class | Reference | Taxonomy adaptive | Taxonomy + soft alias |
| --- | --- | ---: | ---: | ---: |
| vdd | other | 32.7096 | 35.2289 | 36.7391 |
| vdd | wall | 43.4422 | 47.3161 | 34.1353 |
| vdd | road | 40.6842 | 41.8638 | 41.6257 |
| vdd | vegetation | 68.1015 | 66.4698 | 66.6516 |
| vdd | vehicle | 22.7793 | 28.7672 | 27.9540 |
| vdd | roof | 81.4826 | 83.2093 | 82.2937 |
| vdd | water | 85.0207 | 83.0581 | 83.6346 |
| potsdam | impervious surface | 62.7987 | 67.1860 | 58.0967 |
| potsdam | building | 79.0175 | 80.5867 | 75.4058 |
| potsdam | low vegetation | 38.5018 | 43.0884 | 27.5335 |
| potsdam | tree | 58.5794 | 61.9424 | 60.6439 |
| potsdam | car | 30.1233 | 36.6319 | 26.9996 |
| potsdam | clutter | 6.5825 | 8.9316 | 8.5840 |
| voc21 | background | 22.9584 | 89.5149 | 89.6175 |
| voc21 | aeroplane | 14.5656 | 62.3288 | 62.8423 |
| voc21 | bicycle | 19.4848 | 46.0954 | 46.3007 |
| voc21 | bird | 12.5075 | 83.5171 | 84.1732 |
| voc21 | boat | 12.4851 | 59.0349 | 59.1063 |
| voc21 | bottle | 47.9232 | 53.9320 | 54.0948 |
| voc21 | bus | 49.0436 | 87.3569 | 87.4101 |
| voc21 | car | 29.3124 | 70.1532 | 70.0943 |
| voc21 | cat | 48.8822 | 89.6694 | 89.9483 |
| voc21 | chair | 13.9618 | 46.9872 | 47.0824 |
| voc21 | cow | 48.6032 | 91.3593 | 91.6024 |
| voc21 | diningtable | 23.2006 | 56.7746 | 56.8213 |
| voc21 | dog | 25.7741 | 83.9697 | 84.3881 |
| voc21 | horse | 38.2864 | 88.1481 | 88.4024 |
| voc21 | motorbike | 31.4640 | 72.3352 | 72.7379 |
| voc21 | person | 64.3244 | 71.7803 | 71.9279 |
| voc21 | pottedplant | 21.0998 | 49.6908 | 49.7463 |
| voc21 | sheep | 20.6334 | 89.2267 | 89.4936 |
| voc21 | sofa | 30.5125 | 65.2962 | 65.3460 |
| voc21 | train | 27.3487 | 59.3679 | 59.5245 |
| voc21 | tvmonitor | 20.8346 | 60.1370 | 60.4222 |
| context60 | background | 4.3841 | 0.0701 | 0.0565 |
| context60 | aeroplane | 35.3679 | 44.0004 | 45.4242 |
| context60 | bag | 23.1372 | 29.0784 | 29.5627 |
| context60 | bed | 8.8728 | 10.5073 | 10.8123 |
| context60 | bedclothes | 33.8275 | 29.1640 | 29.5954 |
| context60 | bench | 7.4305 | 15.5909 | 15.5357 |
| context60 | bicycle | 59.4554 | 65.3354 | 66.8019 |
| context60 | bird | 28.2136 | 47.4603 | 47.2938 |
| context60 | boat | 37.4254 | 52.9120 | 53.0627 |
| context60 | book | 11.5701 | 9.3387 | 9.8407 |
| context60 | bottle | 62.6554 | 63.9968 | 63.9714 |
| context60 | building | 29.3643 | 38.6137 | 38.3754 |
| context60 | bus | 72.2486 | 72.8009 | 73.5541 |
| context60 | cabinet | 36.8981 | 38.8309 | 38.8501 |
| context60 | car | 62.1021 | 66.7816 | 67.3110 |
| context60 | cat | 74.6854 | 74.7236 | 75.1437 |
| context60 | ceiling | 40.6359 | 43.0718 | 44.4377 |
| context60 | chair | 40.6779 | 42.6595 | 42.5093 |
| context60 | cloth | 15.2680 | 18.4231 | 18.4986 |
| context60 | computer | 9.7609 | 13.0613 | 12.7088 |
| context60 | cow | 74.3238 | 78.1062 | 78.6635 |
| context60 | cup | 19.7938 | 30.5911 | 30.5104 |
| context60 | curtain | 46.0393 | 45.6036 | 46.8026 |
| context60 | dog | 54.6244 | 61.6250 | 62.9169 |
| context60 | door | 22.5693 | 26.8036 | 27.0017 |
| context60 | fence | 27.6626 | 29.5518 | 29.8552 |
| context60 | floor | 49.5913 | 44.6993 | 45.4486 |
| context60 | flower | 20.8316 | 19.7180 | 19.7412 |
| context60 | food | 27.2259 | 35.1602 | 34.1896 |
| context60 | grass | 60.3038 | 68.3456 | 68.4898 |
| context60 | ground | 19.5547 | 4.6345 | 5.0418 |
| context60 | horse | 66.9648 | 78.5004 | 79.0742 |
| context60 | keyboard | 59.5343 | 29.4418 | 32.9292 |
| context60 | light | 8.8714 | 12.8069 | 12.7049 |
| context60 | motorbike | 65.4901 | 72.1627 | 74.2161 |
| context60 | mountain | 44.2573 | 42.3822 | 43.1355 |
| context60 | mouse | 6.6641 | 23.2091 | 25.8026 |
| context60 | person | 72.9186 | 65.5667 | 65.4885 |
| context60 | plate | 14.2587 | 27.9976 | 27.0357 |
| context60 | platform | 8.4134 | 6.9664 | 6.5442 |
| context60 | pottedplant | 43.7746 | 51.8741 | 51.9658 |
| context60 | road | 36.9049 | 41.6723 | 41.2173 |
| context60 | rock | 49.4701 | 42.1911 | 41.7108 |
| context60 | sheep | 59.3685 | 78.5288 | 78.8271 |
| context60 | shelves | 16.5476 | 20.0915 | 20.3912 |
| context60 | sidewalk | 11.9931 | 11.6314 | 10.8454 |
| context60 | sign | 24.2449 | 36.9910 | 36.7476 |
| context60 | sky | 63.1374 | 75.0290 | 75.5009 |
| context60 | snow | 52.3368 | 58.0375 | 57.7322 |
| context60 | sofa | 58.8020 | 62.4637 | 62.5896 |
| context60 | table | 41.2301 | 45.9313 | 45.6660 |
| context60 | track | 4.3575 | 0.0002 | 0.0008 |
| context60 | train | 45.7787 | 40.4126 | 40.1748 |
| context60 | tree | 50.2358 | 58.9155 | 58.9765 |
| context60 | truck | 11.6246 | 13.6624 | 13.9874 |
| context60 | tvmonitor | 29.4547 | 35.9414 | 35.2236 |
| context60 | wall | 32.9629 | 38.5200 | 40.1289 |
| context60 | water | 60.6108 | 68.9724 | 69.5664 |
| context60 | window | 28.2347 | 32.8990 | 33.4203 |
| context60 | wood | 18.5420 | 20.0666 | 19.7008 |
| ade150 | wall | 26.2872 | 37.1495 | 40.9090 |
| ade150 | building | 55.4727 | 62.9711 | 62.3117 |
| ade150 | sky | 65.1888 | 80.2752 | 79.6137 |
| ade150 | floor | 50.5038 | 59.8859 | 63.1444 |
| ade150 | tree | 48.1482 | 61.9465 | 61.5260 |
| ade150 | ceiling | 62.7281 | 55.5834 | 58.8079 |
| ade150 | road | 70.4884 | 69.9955 | 69.2559 |
| ade150 | bed | 54.7096 | 65.9810 | 68.5281 |
| ade150 | windowpane | 40.1672 | 41.6400 | 41.9907 |
| ade150 | grass | 45.9499 | 50.4808 | 48.7509 |
| ade150 | cabinet | 39.7252 | 47.2305 | 47.9387 |
| ade150 | sidewalk | 49.0140 | 49.9836 | 48.5606 |
| ade150 | person | 39.0287 | 61.8811 | 62.9086 |
| ade150 | earth | 6.6422 | 12.7637 | 6.7073 |
| ade150 | door | 34.9703 | 37.0933 | 37.0749 |
| ade150 | table | 30.5306 | 40.7606 | 40.9724 |
| ade150 | mountain | 31.1963 | 39.3888 | 40.1550 |
| ade150 | plant | 20.6914 | 38.0044 | 38.2520 |
| ade150 | curtain | 60.4051 | 63.5588 | 64.6912 |
| ade150 | chair | 28.3288 | 43.9664 | 44.6191 |
| ade150 | car | 62.3336 | 63.7243 | 64.7275 |
| ade150 | water | 42.2883 | 45.7248 | 47.7833 |
| ade150 | painting | 30.2791 | 29.1889 | 31.1013 |
| ade150 | sofa | 52.9647 | 49.6811 | 50.2626 |
| ade150 | shelf | 28.3151 | 32.6615 | 33.0521 |
| ade150 | house | 18.9319 | 20.3601 | 18.2413 |
| ade150 | sea | 25.0470 | 27.9419 | 27.0851 |
| ade150 | mirror | 35.5499 | 32.2526 | 32.4507 |
| ade150 | rug | 26.4859 | 32.9227 | 38.3124 |
| ade150 | field | 18.3966 | 19.2716 | 19.2553 |
| ade150 | armchair | 21.3202 | 9.2313 | 12.8725 |
| ade150 | seat | 35.7894 | 25.1913 | 26.7764 |
| ade150 | fence | 25.7123 | 24.7684 | 25.9567 |
| ade150 | desk | 33.7214 | 32.5294 | 32.5755 |
| ade150 | rock | 36.7569 | 34.0254 | 33.6548 |
| ade150 | wardrobe | 45.2308 | 45.7903 | 46.6434 |
| ade150 | lamp | 29.2906 | 30.6021 | 31.3209 |
| ade150 | bathtub | 49.2460 | 47.9108 | 50.4697 |
| ade150 | railing | 21.1772 | 19.3640 | 20.0760 |
| ade150 | cushion | 28.3476 | 32.6346 | 26.9235 |
| ade150 | base | 0.0469 | 3.0700 | 3.3691 |
| ade150 | box | 24.2532 | 22.1655 | 22.5024 |
| ade150 | column | 35.6703 | 38.2057 | 38.5875 |
| ade150 | signboard | 21.9836 | 26.5056 | 26.9656 |
| ade150 | chest of drawers | 27.9888 | 29.8974 | 29.7769 |
| ade150 | counter | 25.1012 | 37.3936 | 33.2465 |
| ade150 | sand | 37.9052 | 38.8412 | 41.5995 |
| ade150 | sink | 36.8591 | 39.8383 | 40.1016 |
| ade150 | skyscraper | 19.9369 | 22.1702 | 20.4210 |
| ade150 | fireplace | 38.3817 | 48.8580 | 49.2284 |
| ade150 | refrigerator | 49.0827 | 63.7717 | 64.7911 |
| ade150 | grandstand | 29.7858 | 24.1050 | 29.0527 |
| ade150 | path | 3.1151 | 5.7229 | 5.6182 |
| ade150 | stairs | 25.0795 | 34.9042 | 29.8949 |
| ade150 | runway | 34.1910 | 39.7429 | 39.8737 |
| ade150 | case | 2.0391 | 23.0877 | 24.5353 |
| ade150 | pool table | 68.1225 | 80.8930 | 80.2604 |
| ade150 | pillow | 1.6365 | 34.3821 | 28.5520 |
| ade150 | screen door | 0.2115 | 0.1369 | 0.1744 |
| ade150 | stairway | 2.9227 | 18.6171 | 16.4635 |
| ade150 | river | 18.6316 | 15.3116 | 13.9877 |
| ade150 | bridge | 22.4870 | 27.2096 | 27.8537 |
| ade150 | bookcase | 17.4085 | 17.3446 | 16.7631 |
| ade150 | blind | 22.9662 | 32.3831 | 33.2676 |
| ade150 | coffee table | 37.9830 | 49.7550 | 51.9133 |
| ade150 | toilet | 43.8863 | 52.7842 | 48.2379 |
| ade150 | flower | 10.9567 | 11.9936 | 14.0904 |
| ade150 | book | 4.8397 | 12.9715 | 13.8714 |
| ade150 | hill | 2.6531 | 5.7445 | 5.4765 |
| ade150 | bench | 19.5263 | 38.3662 | 39.7414 |
| ade150 | countertop | 13.1506 | 18.2453 | 19.1935 |
| ade150 | stove | 27.8596 | 25.8314 | 30.3383 |
| ade150 | palm | 20.7827 | 24.3811 | 24.1110 |
| ade150 | kitchen island | 7.2913 | 21.4112 | 21.8627 |
| ade150 | computer | 35.0416 | 40.3654 | 41.1619 |
| ade150 | swivel chair | 18.3521 | 8.3683 | 7.2691 |
| ade150 | boat | 42.4942 | 34.6580 | 33.5040 |
| ade150 | bar | 17.1000 | 23.3031 | 23.3747 |
| ade150 | arcade machine | 17.9777 | 13.0477 | 10.6813 |
| ade150 | hovel | 6.4820 | 10.8124 | 10.1593 |
| ade150 | bus | 59.7157 | 62.9664 | 58.6028 |
| ade150 | towel | 38.1900 | 54.7895 | 55.0516 |
| ade150 | light | 6.2769 | 7.3429 | 9.0347 |
| ade150 | truck | 16.8151 | 17.9700 | 16.5700 |
| ade150 | tower | 6.0355 | 16.1756 | 17.3325 |
| ade150 | chandelier | 17.7555 | 42.1338 | 42.8002 |
| ade150 | awning | 13.9899 | 22.3292 | 20.8674 |
| ade150 | streetlight | 7.0489 | 21.9955 | 21.7615 |
| ade150 | booth | 3.4656 | 8.5843 | 6.5432 |
| ade150 | television receiver | 31.8155 | 26.3064 | 32.6536 |
| ade150 | airplane | 16.6806 | 15.8925 | 17.1627 |
| ade150 | dirt track | 0.8308 | 2.7596 | 2.0052 |
| ade150 | apparel | 4.1816 | 6.2295 | 8.6197 |
| ade150 | pole | 7.1115 | 13.1591 | 13.5430 |
| ade150 | land | 2.0624 | 2.6949 | 2.4888 |
| ade150 | bannister | 6.5286 | 14.5149 | 11.6507 |
| ade150 | escalator | 34.9032 | 40.7161 | 37.0552 |
| ade150 | ottoman | 45.8392 | 35.8553 | 34.8944 |
| ade150 | bottle | 28.3776 | 33.4163 | 35.8048 |
| ade150 | buffet | 0.1756 | 7.4221 | 3.1381 |
| ade150 | poster | 7.7774 | 6.8041 | 6.8517 |
| ade150 | stage | 6.6257 | 8.4117 | 9.2727 |
| ade150 | van | 22.6616 | 37.6618 | 39.3679 |
| ade150 | ship | 3.3744 | 8.3479 | 8.3750 |
| ade150 | fountain | 21.6272 | 33.8311 | 30.1139 |
| ade150 | conveyer belt | 42.9349 | 45.8174 | 44.6656 |
| ade150 | canopy | 3.2268 | 17.8104 | 15.3430 |
| ade150 | washer | 78.2916 | 66.5821 | 71.2136 |
| ade150 | plaything | 4.4674 | 8.6752 | 12.4268 |
| ade150 | swimming pool | 16.4347 | 33.8712 | 34.2547 |
| ade150 | stool | 13.2908 | 24.8166 | 25.1204 |
| ade150 | barrel | 4.2828 | 29.5177 | 29.2131 |
| ade150 | basket | 33.2152 | 35.5176 | 35.6703 |
| ade150 | waterfall | 19.8193 | 21.7144 | 23.1360 |
| ade150 | tent | 31.9791 | 44.2472 | 42.4100 |
| ade150 | bag | 24.6185 | 25.0176 | 25.9898 |
| ade150 | minibike | 0.2054 | 63.5523 | 59.5756 |
| ade150 | cradle | 42.5587 | 47.1309 | 54.6289 |
| ade150 | oven | 13.4868 | 27.1456 | 26.3534 |
| ade150 | ball | 10.1176 | 20.7427 | 28.6319 |
| ade150 | food | 39.7605 | 40.7910 | 42.1301 |
| ade150 | step | 1.1587 | 2.7171 | 3.3615 |
| ade150 | tank | 1.0683 | 20.7715 | 20.5790 |
| ade150 | trade name | 2.9905 | 2.6172 | 2.3674 |
| ade150 | microwave | 53.4153 | 70.9409 | 73.0244 |
| ade150 | pot | 22.9999 | 35.2839 | 34.0940 |
| ade150 | animal | 47.7141 | 62.9997 | 56.8711 |
| ade150 | bicycle | 31.7816 | 46.6620 | 47.8259 |
| ade150 | lake | 7.9593 | 4.0759 | 2.8967 |
| ade150 | dishwasher | 40.4913 | 45.1659 | 47.0963 |
| ade150 | screen | 32.3341 | 37.5349 | 37.9221 |
| ade150 | blanket | 21.5664 | 10.2167 | 12.0350 |
| ade150 | sculpture | 15.8014 | 43.2244 | 41.0415 |
| ade150 | hood | 7.4941 | 33.7729 | 32.1807 |
| ade150 | sconce | 3.4967 | 20.4888 | 20.8420 |
| ade150 | vase | 12.6567 | 15.8600 | 18.2979 |
| ade150 | traffic light | 7.8906 | 20.7618 | 21.1701 |
| ade150 | tray | 6.5102 | 12.8933 | 13.4994 |
| ade150 | ashcan | 0.3798 | 13.0143 | 15.5793 |
| ade150 | fan | 38.4791 | 43.1426 | 43.4805 |
| ade150 | pier | 5.3278 | 3.9099 | 4.7178 |
| ade150 | crt screen | 0.2904 | 0.3844 | 0.3710 |
| ade150 | plate | 16.9826 | 34.0783 | 38.4579 |
| ade150 | monitor | 9.5388 | 19.9107 | 18.4904 |
| ade150 | bulletin board | 2.3192 | 6.4384 | 6.2742 |
| ade150 | shower | 0.4332 | 1.0456 | 1.0741 |
| ade150 | radiator | 37.6359 | 55.5340 | 56.1973 |
| ade150 | glass | 3.6965 | 8.3100 | 7.9569 |
| ade150 | clock | 40.2114 | 44.2053 | 45.2162 |
| ade150 | flag | 22.8165 | 56.0003 | 57.3087 |
