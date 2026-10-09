# Frozen RivalFineHard20 Natural-Image Evaluation

Retained Geometry + physical8 rival-conditioned alias admission + wide-view exact coupled reconstruction. No core rule or numerical hyperparameter changes. Both text paths use the same OpenAI ImageNet templates and semantic-only20 candidate lists. Candidates include curated synonyms and neutral linguistic paraphrases; they are not twenty independent visual concepts. No image/label-conditioned vocabulary generation or target-label tuning.

Full validation labels use standard class/ignore mappings and original RGB image dimensions. This preserves our model-native field of view and is NOT a reproduction of VIP's resize/background protocol. There is no new standalone VIP comparator in this run.

| Protocol | Full images | Geometry | NoAdmission coupling | RivalFineHard20 | Delta vs coupling |
| --- | ---: | ---: | ---: | ---: | ---: |
| voc20 | 1449 | 88.7009 | 89.7717 | 89.7780 | +0.0063 |
| voc21 | 1449 | 31.4696 | 30.2074 | 30.2858 | +0.0784 |
| ade150 | 2000 | 25.1003 | 25.0651 | 25.0705 | +0.0054 |
| coco_stuff171 | 5000 | 28.0282 | 29.5383 | 29.5178 | -0.0205 |
| coco_object81 | 5000 | 24.4626 | 24.9508 | 24.9252 | -0.0256 |

## Unavailable Protocols

- context59: [Errno 2] No such file or directory: '/data/test/datasets/VIP_natural/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'
- context60: [Errno 2] No such file or directory: '/data/test/datasets/VIP_natural/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'
- cityscapes19: Incorrect full validation coverage: cityscapes19

## Per-Class IoU

| Protocol | Class | Geometry | NoAdmission | RivalFineHard20 |
| --- | --- | ---: | ---: | ---: |
| voc20 | aeroplane | 97.5208 | 98.3959 | 98.4322 |
| voc20 | bicycle | 79.7614 | 70.6127 | 70.5780 |
| voc20 | bird | 97.5621 | 98.9607 | 98.9569 |
| voc20 | boat | 95.0018 | 94.1596 | 94.0936 |
| voc20 | bottle | 85.6004 | 89.9750 | 90.0212 |
| voc20 | bus | 91.2997 | 95.3916 | 95.3909 |
| voc20 | car | 84.3502 | 93.0681 | 93.0831 |
| voc20 | cat | 97.6627 | 98.4352 | 98.4222 |
| voc20 | chair | 58.5678 | 55.1759 | 55.1799 |
| voc20 | cow | 96.1298 | 97.6934 | 97.7997 |
| voc20 | diningtable | 76.6965 | 75.7629 | 75.8130 |
| voc20 | dog | 95.4803 | 96.8751 | 96.8815 |
| voc20 | horse | 96.5015 | 95.6255 | 95.6422 |
| voc20 | motorbike | 85.9946 | 84.2299 | 84.2340 |
| voc20 | person | 86.1636 | 84.8014 | 84.7907 |
| voc20 | pottedplant | 94.0355 | 96.9786 | 96.9002 |
| voc20 | sheep | 96.4873 | 97.5958 | 97.7380 |
| voc20 | sofa | 80.7292 | 82.6632 | 82.5966 |
| voc20 | train | 92.5226 | 98.0601 | 98.0324 |
| voc20 | tvmonitor | 85.9494 | 90.9740 | 90.9733 |
| voc21 | background | 23.0398 | 23.5502 | 23.7618 |
| voc21 | aeroplane | 11.1510 | 13.4109 | 13.4262 |
| voc21 | bicycle | 23.5743 | 20.3889 | 20.3228 |
| voc21 | bird | 10.8386 | 12.0970 | 12.0766 |
| voc21 | boat | 14.2758 | 13.1721 | 13.2018 |
| voc21 | bottle | 43.5486 | 48.9253 | 49.0278 |
| voc21 | bus | 62.1127 | 55.1178 | 55.1321 |
| voc21 | car | 25.2698 | 25.0296 | 25.1470 |
| voc21 | cat | 65.6523 | 50.1931 | 50.1791 |
| voc21 | chair | 14.9613 | 17.5200 | 17.5851 |
| voc21 | cow | 61.1544 | 52.3454 | 52.7209 |
| voc21 | diningtable | 26.0381 | 23.2875 | 23.2513 |
| voc21 | dog | 20.5778 | 25.5415 | 25.7521 |
| voc21 | horse | 32.9586 | 32.8994 | 33.2591 |
| voc21 | motorbike | 36.2843 | 33.6453 | 33.7813 |
| voc21 | person | 60.0415 | 61.9019 | 62.2053 |
| voc21 | pottedplant | 16.9094 | 20.6598 | 20.4740 |
| voc21 | sheep | 28.1981 | 21.7533 | 21.4757 |
| voc21 | sofa | 30.5711 | 30.8802 | 30.9102 |
| voc21 | train | 30.2688 | 28.7653 | 28.6783 |
| voc21 | tvmonitor | 23.4351 | 23.2719 | 23.6336 |
| ade150 | wall | 31.2137 | 26.6339 | 26.6294 |
| ade150 | building | 38.0709 | 52.5424 | 52.4931 |
| ade150 | sky | 66.7477 | 66.1766 | 66.1472 |
| ade150 | floor | 54.8192 | 52.0621 | 52.0156 |
| ade150 | tree | 45.6729 | 46.6493 | 46.7208 |
| ade150 | ceiling | 63.5004 | 62.4940 | 62.5067 |
| ade150 | road | 70.9473 | 69.0048 | 68.6101 |
| ade150 | bed | 65.0027 | 56.4631 | 56.5251 |
| ade150 | windowpane | 34.6238 | 40.2387 | 40.1590 |
| ade150 | grass | 41.6061 | 43.9284 | 43.9632 |
| ade150 | cabinet | 40.4916 | 39.9172 | 39.8527 |
| ade150 | sidewalk | 49.0784 | 48.5461 | 48.5270 |
| ade150 | person | 33.7733 | 38.6297 | 38.6544 |
| ade150 | earth | 6.0848 | 6.9499 | 7.0087 |
| ade150 | door | 32.3644 | 34.0812 | 33.9988 |
| ade150 | table | 37.1321 | 31.5090 | 31.5148 |
| ade150 | mountain | 30.0716 | 30.8858 | 30.8744 |
| ade150 | plant | 25.2722 | 19.8811 | 19.9348 |
| ade150 | curtain | 63.7408 | 60.1956 | 60.0408 |
| ade150 | chair | 26.3867 | 28.2030 | 28.2161 |
| ade150 | car | 57.1708 | 60.9142 | 60.8099 |
| ade150 | water | 48.0563 | 44.0785 | 44.1720 |
| ade150 | painting | 43.1291 | 38.5026 | 38.5093 |
| ade150 | sofa | 55.9599 | 53.7052 | 53.6693 |
| ade150 | shelf | 26.2657 | 27.8975 | 27.8829 |
| ade150 | house | 19.2832 | 18.8919 | 18.9455 |
| ade150 | sea | 25.4580 | 24.4385 | 24.4340 |
| ade150 | mirror | 24.3800 | 34.0552 | 34.0237 |
| ade150 | rug | 26.8244 | 26.5733 | 26.5213 |
| ade150 | field | 15.3562 | 17.2729 | 17.2959 |
| ade150 | armchair | 21.6549 | 23.5462 | 23.5626 |
| ade150 | seat | 18.0343 | 31.5229 | 31.6079 |
| ade150 | fence | 26.9000 | 25.7572 | 25.6589 |
| ade150 | desk | 29.9405 | 32.2259 | 32.2058 |
| ade150 | rock | 32.3955 | 34.4587 | 34.4745 |
| ade150 | wardrobe | 34.0476 | 44.7615 | 44.6037 |
| ade150 | lamp | 32.3082 | 29.0708 | 29.0073 |
| ade150 | bathtub | 51.9630 | 52.1395 | 52.3507 |
| ade150 | railing | 15.9200 | 19.8186 | 19.8146 |
| ade150 | cushion | 49.0260 | 33.7431 | 33.7593 |
| ade150 | base | 0.0801 | 0.0524 | 0.0516 |
| ade150 | box | 25.0736 | 23.4901 | 23.4834 |
| ade150 | column | 17.3449 | 30.8641 | 30.8961 |
| ade150 | signboard | 22.4871 | 22.2416 | 22.2612 |
| ade150 | chest of drawers | 26.5235 | 29.0643 | 29.0904 |
| ade150 | counter | 16.4360 | 21.5845 | 21.6248 |
| ade150 | sand | 35.6980 | 41.3149 | 41.3492 |
| ade150 | sink | 46.4881 | 37.7755 | 37.8345 |
| ade150 | skyscraper | 21.3094 | 20.5986 | 20.6550 |
| ade150 | fireplace | 43.0077 | 38.8004 | 38.7766 |
| ade150 | refrigerator | 65.6097 | 55.0241 | 55.1040 |
| ade150 | grandstand | 27.1233 | 29.8721 | 29.8920 |
| ade150 | path | 6.7669 | 4.2415 | 4.2625 |
| ade150 | stairs | 26.5245 | 25.6573 | 25.6859 |
| ade150 | runway | 38.2657 | 34.7397 | 34.7167 |
| ade150 | case | 1.2400 | 1.1636 | 1.1618 |
| ade150 | pool table | 73.5406 | 72.0204 | 72.0916 |
| ade150 | pillow | 36.3542 | 5.0647 | 5.0743 |
| ade150 | screen door | 0.1728 | 0.2426 | 0.2346 |
| ade150 | stairway | 8.0240 | 4.4657 | 4.4696 |
| ade150 | river | 18.9112 | 19.1065 | 19.0446 |
| ade150 | bridge | 21.3787 | 22.6042 | 22.6719 |
| ade150 | bookcase | 15.2141 | 17.4476 | 17.4652 |
| ade150 | blind | 21.7452 | 26.5900 | 26.6631 |
| ade150 | coffee table | 40.1528 | 38.0687 | 38.1116 |
| ade150 | toilet | 46.0507 | 42.5716 | 42.2396 |
| ade150 | flower | 12.7122 | 10.3587 | 10.3765 |
| ade150 | book | 13.9710 | 7.9795 | 8.0282 |
| ade150 | hill | 5.4906 | 3.4550 | 3.4529 |
| ade150 | bench | 22.7466 | 25.3591 | 25.4112 |
| ade150 | countertop | 11.3660 | 11.1933 | 11.2243 |
| ade150 | stove | 31.1216 | 32.6922 | 32.7242 |
| ade150 | palm | 18.2676 | 20.2925 | 20.2722 |
| ade150 | kitchen island | 11.6837 | 8.3091 | 8.3276 |
| ade150 | computer | 42.8319 | 40.0767 | 40.0320 |
| ade150 | swivel chair | 14.7410 | 21.1121 | 21.1505 |
| ade150 | boat | 44.9531 | 42.8924 | 42.8129 |
| ade150 | bar | 15.6178 | 15.7217 | 15.7511 |
| ade150 | arcade machine | 18.6493 | 21.7752 | 21.7985 |
| ade150 | hovel | 9.5676 | 8.8890 | 8.8855 |
| ade150 | bus | 71.1684 | 64.2044 | 64.2408 |
| ade150 | towel | 38.7279 | 37.3612 | 37.3103 |
| ade150 | light | 4.1117 | 5.4955 | 5.5039 |
| ade150 | truck | 24.8487 | 18.0820 | 17.9606 |
| ade150 | tower | 6.3771 | 6.4252 | 6.4499 |
| ade150 | chandelier | 26.3016 | 20.4545 | 20.5570 |
| ade150 | awning | 6.8582 | 11.6875 | 11.7277 |
| ade150 | streetlight | 13.3235 | 8.3879 | 8.4885 |
| ade150 | booth | 3.7492 | 4.4613 | 4.4766 |
| ade150 | television receiver | 21.5592 | 34.5348 | 34.6109 |
| ade150 | airplane | 16.6154 | 13.9594 | 13.8728 |
| ade150 | dirt track | 1.7826 | 1.0650 | 1.0672 |
| ade150 | apparel | 5.3591 | 4.3340 | 4.3535 |
| ade150 | pole | 4.7537 | 5.8847 | 5.8728 |
| ade150 | land | 2.3873 | 2.2173 | 2.2247 |
| ade150 | bannister | 5.6684 | 6.9269 | 6.9342 |
| ade150 | escalator | 27.2226 | 31.9067 | 31.9850 |
| ade150 | ottoman | 31.9895 | 45.4252 | 45.4038 |
| ade150 | bottle | 23.8081 | 28.8224 | 29.2108 |
| ade150 | buffet | 0.7243 | 0.3139 | 0.3108 |
| ade150 | poster | 11.5473 | 11.0694 | 11.0627 |
| ade150 | stage | 10.3349 | 7.1414 | 7.1565 |
| ade150 | van | 18.4156 | 21.1951 | 21.3584 |
| ade150 | ship | 4.5352 | 4.1358 | 4.1492 |
| ade150 | fountain | 15.4500 | 20.2180 | 20.2541 |
| ade150 | conveyer belt | 42.2280 | 42.1796 | 42.1583 |
| ade150 | canopy | 2.4624 | 4.4731 | 4.4487 |
| ade150 | washer | 66.8095 | 76.5882 | 76.6334 |
| ade150 | plaything | 3.2372 | 4.3093 | 4.2827 |
| ade150 | swimming pool | 22.6177 | 17.5461 | 17.5521 |
| ade150 | stool | 13.2475 | 13.5905 | 13.6407 |
| ade150 | barrel | 6.0008 | 3.9098 | 3.9382 |
| ade150 | basket | 16.6497 | 26.8183 | 26.8422 |
| ade150 | waterfall | 17.3752 | 18.1624 | 18.1967 |
| ade150 | tent | 41.2881 | 34.4245 | 34.6022 |
| ade150 | bag | 21.3685 | 24.4731 | 24.5724 |
| ade150 | minibike | 22.7863 | 6.7076 | 6.5611 |
| ade150 | cradle | 37.2431 | 47.9213 | 47.9162 |
| ade150 | oven | 11.9312 | 12.1470 | 12.0843 |
| ade150 | ball | 11.4265 | 10.3506 | 10.4802 |
| ade150 | food | 49.1109 | 41.9967 | 41.9649 |
| ade150 | step | 0.7337 | 1.0249 | 1.0202 |
| ade150 | tank | 0.1316 | 0.3036 | 0.3000 |
| ade150 | trade name | 4.6928 | 3.8969 | 3.9033 |
| ade150 | microwave | 71.3560 | 67.4121 | 67.3292 |
| ade150 | pot | 22.1060 | 20.4046 | 20.2394 |
| ade150 | animal | 40.6890 | 42.6681 | 42.7730 |
| ade150 | bicycle | 45.5789 | 35.0614 | 34.9302 |
| ade150 | lake | 3.7303 | 6.4895 | 6.4487 |
| ade150 | dishwasher | 54.2166 | 40.7321 | 40.7313 |
| ade150 | screen | 26.9149 | 25.7687 | 25.8302 |
| ade150 | blanket | 15.2888 | 23.9537 | 24.0415 |
| ade150 | sculpture | 19.5978 | 17.7397 | 17.8474 |
| ade150 | hood | 6.7327 | 8.2700 | 8.3176 |
| ade150 | sconce | 3.1168 | 3.6909 | 3.6896 |
| ade150 | vase | 13.5305 | 11.4359 | 11.4296 |
| ade150 | traffic light | 7.7151 | 7.3061 | 7.4013 |
| ade150 | tray | 9.1345 | 7.0249 | 7.0000 |
| ade150 | ashcan | 0.4972 | 0.3133 | 0.3278 |
| ade150 | fan | 15.5864 | 30.8140 | 30.6874 |
| ade150 | pier | 7.5518 | 5.8291 | 5.8463 |
| ade150 | crt screen | 0.2142 | 0.1953 | 0.1968 |
| ade150 | plate | 18.8077 | 15.7856 | 15.7599 |
| ade150 | monitor | 8.6911 | 11.0291 | 11.0016 |
| ade150 | bulletin board | 3.3150 | 3.1202 | 3.1302 |
| ade150 | shower | 0.5721 | 0.4525 | 0.4533 |
| ade150 | radiator | 19.3786 | 33.6932 | 33.8231 |
| ade150 | glass | 4.7547 | 3.1763 | 3.1715 |
| ade150 | clock | 41.1743 | 40.3408 | 40.1951 |
| ade150 | flag | 29.9197 | 27.3825 | 27.4719 |
| coco_stuff171 | person | 44.5265 | 48.8639 | 48.7488 |
| coco_stuff171 | bicycle | 55.7024 | 58.9606 | 58.9192 |
| coco_stuff171 | car | 37.1377 | 40.4385 | 40.4086 |
| coco_stuff171 | motorcycle | 64.0326 | 66.4266 | 66.4244 |
| coco_stuff171 | airplane | 27.1106 | 28.6748 | 28.5151 |
| coco_stuff171 | bus | 65.0862 | 66.0763 | 66.0875 |
| coco_stuff171 | train | 52.4526 | 53.7706 | 53.5682 |
| coco_stuff171 | truck | 40.4721 | 45.8792 | 45.8816 |
| coco_stuff171 | boat | 43.4403 | 38.8621 | 38.8020 |
| coco_stuff171 | traffic light | 13.3383 | 11.0477 | 11.1062 |
| coco_stuff171 | fire hydrant | 49.6668 | 52.4895 | 52.7045 |
| coco_stuff171 | stop sign | 25.4893 | 23.7534 | 23.7294 |
| coco_stuff171 | parking meter | 29.4145 | 24.4870 | 24.3688 |
| coco_stuff171 | bench | 30.7295 | 26.3088 | 26.3158 |
| coco_stuff171 | bird | 24.6256 | 31.4911 | 31.4869 |
| coco_stuff171 | cat | 78.0403 | 75.9533 | 75.9559 |
| coco_stuff171 | dog | 50.6333 | 56.5067 | 56.5139 |
| coco_stuff171 | horse | 61.3436 | 62.3640 | 62.3758 |
| coco_stuff171 | sheep | 68.4950 | 67.9713 | 67.6589 |
| coco_stuff171 | cow | 71.6907 | 75.9686 | 75.9878 |
| coco_stuff171 | elephant | 75.6429 | 76.0416 | 76.0435 |
| coco_stuff171 | bear | 76.9533 | 77.4904 | 77.5190 |
| coco_stuff171 | zebra | 66.5884 | 68.1009 | 68.0255 |
| coco_stuff171 | giraffe | 70.2360 | 73.4276 | 73.3714 |
| coco_stuff171 | backpack | 20.9712 | 22.1560 | 22.1353 |
| coco_stuff171 | umbrella | 60.3474 | 58.6006 | 58.3136 |
| coco_stuff171 | handbag | 23.6238 | 29.4995 | 29.4864 |
| coco_stuff171 | tie | 3.8457 | 3.3137 | 3.3138 |
| coco_stuff171 | suitcase | 61.9936 | 65.6016 | 65.5736 |
| coco_stuff171 | frisbee | 12.0647 | 15.8376 | 15.4819 |
| coco_stuff171 | skis | 9.3129 | 8.1039 | 8.1239 |
| coco_stuff171 | snowboard | 8.9657 | 5.3783 | 5.4395 |
| coco_stuff171 | sports ball | 4.3367 | 2.4153 | 2.4330 |
| coco_stuff171 | kite | 31.4817 | 32.0285 | 31.6804 |
| coco_stuff171 | baseball bat | 15.2391 | 16.4974 | 16.4893 |
| coco_stuff171 | baseball glove | 13.2993 | 8.5699 | 8.4381 |
| coco_stuff171 | skateboard | 18.6302 | 11.7654 | 11.7104 |
| coco_stuff171 | surfboard | 42.3130 | 29.3436 | 29.4144 |
| coco_stuff171 | tennis racket | 17.1826 | 13.8695 | 13.8397 |
| coco_stuff171 | bottle | 35.2601 | 39.4116 | 39.4288 |
| coco_stuff171 | wine glass | 44.1084 | 42.7976 | 42.7805 |
| coco_stuff171 | cup | 34.7004 | 35.4387 | 35.4633 |
| coco_stuff171 | fork | 15.4542 | 24.1209 | 24.1421 |
| coco_stuff171 | knife | 22.3446 | 20.2589 | 20.3232 |
| coco_stuff171 | spoon | 17.3772 | 21.0169 | 20.7493 |
| coco_stuff171 | bowl | 23.0631 | 21.5831 | 21.5800 |
| coco_stuff171 | banana | 47.6180 | 52.1567 | 52.1105 |
| coco_stuff171 | apple | 23.0830 | 31.0112 | 30.9937 |
| coco_stuff171 | sandwich | 30.1382 | 34.5594 | 34.5406 |
| coco_stuff171 | orange | 45.5283 | 55.2343 | 55.0777 |
| coco_stuff171 | broccoli | 53.2795 | 55.7299 | 55.6606 |
| coco_stuff171 | carrot | 38.3568 | 40.9738 | 40.9311 |
| coco_stuff171 | hot dog | 24.7497 | 29.0623 | 28.7263 |
| coco_stuff171 | pizza | 52.0264 | 54.2427 | 54.2034 |
| coco_stuff171 | donut | 21.1588 | 26.2409 | 26.2439 |
| coco_stuff171 | cake | 46.5681 | 49.0013 | 48.9150 |
| coco_stuff171 | chair | 32.4126 | 31.5087 | 31.5369 |
| coco_stuff171 | couch | 49.8654 | 51.1096 | 50.9882 |
| coco_stuff171 | potted plant | 16.6172 | 19.2569 | 19.2593 |
| coco_stuff171 | bed | 39.3950 | 47.9858 | 47.9549 |
| coco_stuff171 | dining table | 13.2856 | 23.1455 | 23.1720 |
| coco_stuff171 | toilet | 45.6517 | 44.8633 | 44.7090 |
| coco_stuff171 | tv | 26.6372 | 26.9099 | 26.9266 |
| coco_stuff171 | laptop | 57.5877 | 61.8222 | 61.8395 |
| coco_stuff171 | mouse | 40.4348 | 35.1137 | 34.8555 |
| coco_stuff171 | remote | 39.9400 | 51.5537 | 51.4410 |
| coco_stuff171 | keyboard | 53.6790 | 51.4438 | 51.4094 |
| coco_stuff171 | cell phone | 31.2939 | 39.4808 | 39.5089 |
| coco_stuff171 | microwave | 51.2447 | 56.1218 | 56.1762 |
| coco_stuff171 | oven | 35.2272 | 34.7845 | 34.7740 |
| coco_stuff171 | toaster | 27.9826 | 22.7647 | 22.6085 |
| coco_stuff171 | sink | 41.4658 | 37.2873 | 37.2281 |
| coco_stuff171 | refrigerator | 60.8518 | 60.1566 | 60.1911 |
| coco_stuff171 | book | 30.1058 | 26.6828 | 26.7489 |
| coco_stuff171 | clock | 55.7434 | 55.5154 | 55.3484 |
| coco_stuff171 | vase | 36.1192 | 38.9319 | 38.9894 |
| coco_stuff171 | scissors | 39.4553 | 60.8755 | 61.2150 |
| coco_stuff171 | teddy bear | 68.7868 | 68.8937 | 68.9408 |
| coco_stuff171 | hair drier | 9.4239 | 21.5590 | 21.2972 |
| coco_stuff171 | toothbrush | 13.1396 | 23.8370 | 23.7192 |
| coco_stuff171 | banner | 14.9792 | 18.2059 | 18.2410 |
| coco_stuff171 | blanket | 14.8342 | 16.1956 | 16.2348 |
| coco_stuff171 | branch | 7.7044 | 8.4096 | 8.4358 |
| coco_stuff171 | bridge | 23.0881 | 28.6865 | 28.7959 |
| coco_stuff171 | building-other | 30.9639 | 40.3834 | 40.3183 |
| coco_stuff171 | bush | 15.3007 | 15.4104 | 15.4214 |
| coco_stuff171 | cabinet | 31.5828 | 32.1838 | 32.2434 |
| coco_stuff171 | cage | 9.1683 | 8.5671 | 8.5654 |
| coco_stuff171 | cardboard | 37.8527 | 33.6245 | 33.6348 |
| coco_stuff171 | carpet | 41.9833 | 44.9631 | 44.9543 |
| coco_stuff171 | ceiling-other | 42.5235 | 44.6545 | 44.6611 |
| coco_stuff171 | ceiling-tile | 6.9414 | 15.0273 | 15.0110 |
| coco_stuff171 | cloth | 1.6015 | 1.1462 | 1.1611 |
| coco_stuff171 | clothes | 6.0706 | 7.0380 | 7.0375 |
| coco_stuff171 | clouds | 35.8611 | 40.8932 | 40.8917 |
| coco_stuff171 | counter | 10.9482 | 14.4041 | 14.3995 |
| coco_stuff171 | cupboard | 2.8489 | 3.3246 | 3.3396 |
| coco_stuff171 | curtain | 58.8382 | 58.2229 | 58.1136 |
| coco_stuff171 | desk-stuff | 16.3849 | 17.2373 | 17.2221 |
| coco_stuff171 | dirt | 10.1544 | 14.6304 | 14.6201 |
| coco_stuff171 | door-stuff | 26.4897 | 28.3265 | 28.3265 |
| coco_stuff171 | fence | 29.5128 | 28.8568 | 28.8212 |
| coco_stuff171 | floor-marble | 5.3806 | 5.5022 | 5.5015 |
| coco_stuff171 | floor-other | 13.1821 | 14.2300 | 14.2196 |
| coco_stuff171 | floor-stone | 4.6192 | 4.3351 | 4.3328 |
| coco_stuff171 | floor-tile | 40.8560 | 44.4284 | 44.4262 |
| coco_stuff171 | floor-wood | 28.8014 | 33.2150 | 33.2585 |
| coco_stuff171 | flower | 33.3974 | 34.6371 | 34.6067 |
| coco_stuff171 | fog | 8.9042 | 10.8631 | 10.8615 |
| coco_stuff171 | food-other | 14.0595 | 13.3497 | 13.3642 |
| coco_stuff171 | fruit | 22.3517 | 26.4463 | 26.4888 |
| coco_stuff171 | furniture-other | 3.4903 | 3.5093 | 3.4956 |
| coco_stuff171 | grass | 43.6055 | 45.1615 | 45.1844 |
| coco_stuff171 | gravel | 20.0686 | 17.2556 | 17.2622 |
| coco_stuff171 | ground-other | 4.0071 | 4.8707 | 4.8803 |
| coco_stuff171 | hill | 12.2442 | 11.2869 | 11.2796 |
| coco_stuff171 | house | 10.2031 | 13.9552 | 13.9432 |
| coco_stuff171 | leaves | 11.9684 | 9.8792 | 9.9048 |
| coco_stuff171 | light | 7.3734 | 10.5673 | 10.5929 |
| coco_stuff171 | mat | 1.4402 | 1.9231 | 1.9180 |
| coco_stuff171 | metal | 5.3047 | 3.2480 | 3.2532 |
| coco_stuff171 | mirror-stuff | 11.2761 | 18.7597 | 18.7582 |
| coco_stuff171 | moss | 8.9484 | 12.3715 | 12.3579 |
| coco_stuff171 | mountain | 40.5903 | 40.4804 | 40.4185 |
| coco_stuff171 | mud | 5.7371 | 6.8978 | 6.8899 |
| coco_stuff171 | napkin | 10.4583 | 11.6171 | 11.6169 |
| coco_stuff171 | net | 19.6825 | 18.0478 | 18.0340 |
| coco_stuff171 | paper | 14.8579 | 8.3333 | 8.3391 |
| coco_stuff171 | pavement | 38.2126 | 39.2558 | 39.2611 |
| coco_stuff171 | pillow | 6.5127 | 9.0442 | 9.0635 |
| coco_stuff171 | plant-other | 8.3506 | 7.5343 | 7.5395 |
| coco_stuff171 | plastic | 9.8923 | 8.9605 | 8.9590 |
| coco_stuff171 | platform | 9.7797 | 9.3505 | 9.3601 |
| coco_stuff171 | playingfield | 9.6760 | 12.1918 | 12.1717 |
| coco_stuff171 | railing | 3.9683 | 4.6105 | 4.6181 |
| coco_stuff171 | railroad | 29.5818 | 30.0213 | 30.1813 |
| coco_stuff171 | river | 8.3939 | 12.1840 | 12.1847 |
| coco_stuff171 | road | 27.0281 | 33.2760 | 33.1453 |
| coco_stuff171 | rock | 15.5362 | 14.3399 | 14.4591 |
| coco_stuff171 | roof | 18.2299 | 19.7567 | 19.7472 |
| coco_stuff171 | rug | 24.4206 | 22.7436 | 22.7400 |
| coco_stuff171 | salad | 8.3099 | 8.3564 | 8.3469 |
| coco_stuff171 | sand | 52.9745 | 54.7297 | 54.7149 |
| coco_stuff171 | sea | 52.2179 | 63.6686 | 63.6805 |
| coco_stuff171 | shelf | 23.6175 | 25.8165 | 25.7452 |
| coco_stuff171 | sky-other | 49.1638 | 52.5215 | 52.4266 |
| coco_stuff171 | skyscraper | 19.8823 | 18.2591 | 18.2737 |
| coco_stuff171 | snow | 69.6253 | 69.3307 | 69.3581 |
| coco_stuff171 | solid-other | 0.0603 | 0.0369 | 0.0371 |
| coco_stuff171 | stairs | 22.2638 | 26.2874 | 26.2884 |
| coco_stuff171 | stone | 10.8284 | 11.7523 | 11.7505 |
| coco_stuff171 | straw | 17.0715 | 17.9540 | 17.9663 |
| coco_stuff171 | structural-other | 0.1947 | 0.0081 | 0.0083 |
| coco_stuff171 | table | 9.4484 | 9.9559 | 10.0057 |
| coco_stuff171 | tent | 7.2179 | 6.7349 | 6.7343 |
| coco_stuff171 | textile-other | 2.4107 | 2.7112 | 2.7102 |
| coco_stuff171 | towel | 20.3352 | 25.9345 | 25.9442 |
| coco_stuff171 | tree | 30.6704 | 30.7806 | 30.8263 |
| coco_stuff171 | vegetable | 23.0002 | 17.6565 | 17.6484 |
| coco_stuff171 | wall-brick | 31.4184 | 35.1428 | 35.1390 |
| coco_stuff171 | wall-concrete | 5.5890 | 5.6552 | 5.6589 |
| coco_stuff171 | wall-other | 8.4925 | 8.3369 | 8.3204 |
| coco_stuff171 | wall-panel | 1.3106 | 1.8369 | 1.8311 |
| coco_stuff171 | wall-stone | 19.2939 | 19.8418 | 19.8489 |
| coco_stuff171 | wall-tile | 48.4745 | 48.7923 | 48.7670 |
| coco_stuff171 | wall-wood | 22.2571 | 20.2767 | 20.2640 |
| coco_stuff171 | water-other | 16.0840 | 20.6835 | 20.6964 |
| coco_stuff171 | waterdrops | 0.7707 | 0.8594 | 0.8607 |
| coco_stuff171 | window-blind | 31.7843 | 32.2570 | 32.2593 |
| coco_stuff171 | window-other | 28.7629 | 34.7321 | 34.7512 |
| coco_stuff171 | wood | 13.6573 | 12.5239 | 12.5144 |
| coco_object81 | background | 20.6752 | 18.2140 | 18.2772 |
| coco_object81 | person | 52.9671 | 52.3363 | 52.3148 |
| coco_object81 | bicycle | 49.2006 | 49.7180 | 49.6657 |
| coco_object81 | car | 25.9618 | 25.2108 | 25.2448 |
| coco_object81 | motorcycle | 55.2790 | 46.8121 | 46.9032 |
| coco_object81 | airplane | 5.8632 | 8.8475 | 8.8697 |
| coco_object81 | bus | 55.4941 | 47.9190 | 47.9910 |
| coco_object81 | train | 29.5544 | 28.3047 | 28.1899 |
| coco_object81 | truck | 33.1672 | 36.1864 | 36.1525 |
| coco_object81 | boat | 10.3345 | 10.0191 | 10.0367 |
| coco_object81 | traffic light | 2.9950 | 2.5872 | 2.5950 |
| coco_object81 | fire hydrant | 8.7870 | 10.3213 | 10.4354 |
| coco_object81 | stop sign | 8.9881 | 8.7002 | 8.6294 |
| coco_object81 | parking meter | 9.1556 | 7.0328 | 6.9926 |
| coco_object81 | bench | 6.6308 | 6.1420 | 6.1726 |
| coco_object81 | bird | 4.1169 | 4.9384 | 4.9674 |
| coco_object81 | cat | 66.6238 | 53.1995 | 53.0321 |
| coco_object81 | dog | 16.8051 | 24.0344 | 24.2273 |
| coco_object81 | horse | 31.6180 | 32.6379 | 32.9255 |
| coco_object81 | sheep | 30.8156 | 25.6039 | 24.9229 |
| coco_object81 | cow | 58.0746 | 50.3174 | 50.2654 |
| coco_object81 | elephant | 42.3627 | 35.3470 | 35.4783 |
| coco_object81 | bear | 38.5114 | 31.5402 | 31.6748 |
| coco_object81 | zebra | 29.7841 | 28.0539 | 27.9628 |
| coco_object81 | giraffe | 30.4617 | 42.0108 | 41.9525 |
| coco_object81 | backpack | 12.5992 | 16.5396 | 16.5841 |
| coco_object81 | umbrella | 24.0185 | 23.4756 | 23.2513 |
| coco_object81 | handbag | 11.6755 | 16.9659 | 16.9919 |
| coco_object81 | tie | 1.8596 | 1.8934 | 1.8988 |
| coco_object81 | suitcase | 55.5238 | 56.7944 | 56.7076 |
| coco_object81 | frisbee | 3.6158 | 6.2689 | 6.1112 |
| coco_object81 | skis | 1.1198 | 1.0783 | 1.0833 |
| coco_object81 | snowboard | 2.0712 | 1.4724 | 1.4710 |
| coco_object81 | sports ball | 0.5081 | 0.4425 | 0.4458 |
| coco_object81 | kite | 7.6117 | 7.9898 | 7.8759 |
| coco_object81 | baseball bat | 3.8074 | 5.8618 | 5.9019 |
| coco_object81 | baseball glove | 6.1747 | 4.0199 | 3.9474 |
| coco_object81 | skateboard | 9.3093 | 7.2955 | 7.2801 |
| coco_object81 | surfboard | 6.2020 | 5.4040 | 5.4312 |
| coco_object81 | tennis racket | 10.3697 | 7.7482 | 7.7450 |
| coco_object81 | bottle | 30.5838 | 31.6433 | 31.8697 |
| coco_object81 | wine glass | 30.7011 | 30.5309 | 30.5028 |
| coco_object81 | cup | 24.5190 | 28.1501 | 28.3137 |
| coco_object81 | fork | 3.8889 | 8.9022 | 8.9206 |
| coco_object81 | knife | 9.3001 | 10.0384 | 10.2105 |
| coco_object81 | spoon | 11.0337 | 12.6205 | 12.5045 |
| coco_object81 | bowl | 22.7732 | 30.1837 | 29.8933 |
| coco_object81 | banana | 44.7813 | 41.5156 | 41.4242 |
| coco_object81 | apple | 28.4626 | 36.4516 | 36.1022 |
| coco_object81 | sandwich | 27.2654 | 25.3922 | 25.2746 |
| coco_object81 | orange | 17.2676 | 25.0488 | 24.0343 |
| coco_object81 | broccoli | 42.5245 | 39.4702 | 39.3615 |
| coco_object81 | carrot | 38.2957 | 43.6248 | 43.5117 |
| coco_object81 | hot dog | 22.2044 | 17.6202 | 17.2376 |
| coco_object81 | pizza | 48.8255 | 49.9453 | 49.7854 |
| coco_object81 | donut | 42.7666 | 48.3386 | 48.1569 |
| coco_object81 | cake | 42.4114 | 39.8400 | 39.6905 |
| coco_object81 | chair | 19.3545 | 23.0333 | 23.2701 |
| coco_object81 | couch | 35.9340 | 37.4468 | 37.4012 |
| coco_object81 | potted plant | 10.5044 | 12.4239 | 12.4019 |
| coco_object81 | bed | 36.1199 | 29.6956 | 29.8780 |
| coco_object81 | dining table | 26.9304 | 29.9380 | 29.8786 |
| coco_object81 | toilet | 17.6026 | 17.1571 | 17.0650 |
| coco_object81 | tv | 13.6837 | 12.2566 | 12.4260 |
| coco_object81 | laptop | 55.5973 | 56.7840 | 56.8119 |
| coco_object81 | mouse | 10.2810 | 5.6939 | 5.6082 |
| coco_object81 | remote | 30.2986 | 47.0371 | 47.0709 |
| coco_object81 | keyboard | 40.3422 | 47.6796 | 47.8449 |
| coco_object81 | cell phone | 26.3698 | 33.4035 | 33.5416 |
| coco_object81 | microwave | 31.6502 | 29.4122 | 29.5365 |
| coco_object81 | oven | 16.7778 | 20.4270 | 20.5149 |
| coco_object81 | toaster | 10.2624 | 7.0357 | 6.9179 |
| coco_object81 | sink | 15.3416 | 14.5416 | 14.5926 |
| coco_object81 | refrigerator | 29.5362 | 24.1773 | 24.3816 |
| coco_object81 | book | 35.5492 | 41.0197 | 40.9340 |
| coco_object81 | clock | 20.8908 | 20.9776 | 20.8309 |
| coco_object81 | vase | 18.1871 | 20.9662 | 21.0857 |
| coco_object81 | scissors | 36.7127 | 53.5925 | 53.7361 |
| coco_object81 | teddy bear | 60.3967 | 51.8832 | 52.2252 |
| coco_object81 | hair drier | 3.6475 | 3.9950 | 3.9649 |
| coco_object81 | toothbrush | 11.1994 | 11.8401 | 11.6248 |

## Measured Cost

| Protocol | Parallel wall seconds | Aggregate GPU seconds | Peak MiB |
| --- | ---: | ---: | ---: |
| voc20 | 889.76 | 889.76 | 5709.79 |
| voc21 | 762.26 | 762.26 | 5749.41 |
| ade150 | 4526.98 | 8928.02 | 24312.33 |
| coco_stuff171 | 16844.90 | 33612.53 | 29846.86 |
| coco_object81 | 6444.45 | 12135.73 | 11095.88 |

Full-run costs include the three shared readouts and model loading. Text cache creation is measured in the integration runs, not billed as isolated primary-model latency. VOC20/21 share images, as do the two COCO taxonomies; protocol image totals must not be called unique underlying images.
