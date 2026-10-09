# RivalFineHard: Verified Nested20/30/40 Context Results

Same64 developed top-left512 windows,8/domain. Weights and RivalFineHard rule frozen; Geometry local20 unchanged; only context candidates change. Nested historical20 prefix plus unfiltered Qwen2.5-VL7B additions. Actual prompts/responses/source manifest are saved. Only duplicate/format checks, no image/label/semantic filtering in generation. LoveDA D counts once, P separately; LandCover.ai substitutes for unlabeled iSAID. Not full-dataset or untouched validation.

## Equal-Domain Means

| Context count | Unscreened | RivalFineHard | Gain pp | Random deletion | Independent all | Independent screened |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 42.867949 | 45.062165 | 2.194216 | 42.975203 | 39.128143 | 43.384785 |
| 30 | 43.035710 | 45.136642 | 2.100932 | 42.838784 | 38.240808 | 43.037324 |
| 40 | 41.742094 | 43.969479 | 2.227385 | 41.700926 | 36.677867 | 41.116261 |

## Interpretation

All three sizes improve all eight domain means over their own unscreened and one count-matched random control. Screened30 is only0.074477pp above screened20 in the equal-domain mean; screened40 is1.092686pp below it. The screening mechanism remains useful at larger counts, but does not make arbitrary expansion harmless. Thirty improves VDD/Potsdam/UDD5 while OEM/Vaihingen/FLAIR decline versus screened20. LoveDA P improves and is reported separately, not counted twice in the domain mean. Keep the original20 deployment; do not select different counts per dataset from this development screen.

Alias use is query/class/rival-conditioned, not a permanently shortened vocabulary. The frozen rule rejects cross-view competitive contradiction, not all semantic mistakes. The retention figures below do not establish that retained phrases are correct; consistent cross-view errors can escape rejection.


```json
{
  "k20": {
    "screening_gain_pp": 2.194216064949032,
    "above_count_matched_random_pp": 2.0869620079452886,
    "screened_change_vs20_pp": 0.0,
    "unscreened_change_vs20_pp": 0.0,
    "independent_screening_gain_pp": 4.256642186674156,
    "domain_wins_vs_unscreened": 8,
    "domain_wins_vs_random": 8
  },
  "k30": {
    "screening_gain_pp": 2.1009321183412837,
    "above_count_matched_random_pp": 2.2978581292351947,
    "screened_change_vs20_pp": 0.07447747023080353,
    "unscreened_change_vs20_pp": 0.16776141683855172,
    "independent_screening_gain_pp": 4.796516755232389,
    "domain_wins_vs_unscreened": 8,
    "domain_wins_vs_random": 8
  },
  "k40": {
    "screening_gain_pp": 2.2273851434275187,
    "above_count_matched_random_pp": 2.2685529000378537,
    "screened_change_vs20_pp": -1.0926859557104862,
    "unscreened_change_vs20_pp": -1.125855034188973,
    "independent_screening_gain_pp": 4.43839448596799,
    "domain_wins_vs_unscreened": 8,
    "domain_wins_vs_random": 8
  }
}
```

## Per-Domain Matched Results

| Dataset/protocol |20 all |20 screened |30 all |30 screened |40 all |40 screened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 53.746954 | 54.344136 | 56.342673 | 59.479844 | 53.838233 | 56.896133 |
| potsdam/potsdam | 38.920258 | 40.617274 | 40.488414 | 42.037699 | 37.994503 | 39.686363 |
| udd5/udd5 | 28.175811 | 34.039388 | 33.171658 | 38.576771 | 31.077372 | 36.590851 |
| oem/oem | 39.023152 | 39.790613 | 32.557674 | 33.307929 | 31.644780 | 32.487401 |
| loveda/P | 50.706552 | 52.587883 | 53.766575 | 56.881713 | 53.461851 | 57.416353 |
| loveda/D | 30.736030 | 37.215569 | 33.618585 | 37.157210 | 32.886529 | 37.457146 |
| vaihingen/vaihingen | 51.826962 | 52.944605 | 48.694363 | 50.258702 | 47.703096 | 49.275088 |
| landcoverai/landcoverai | 66.906020 | 67.483370 | 67.104015 | 67.437928 | 67.044557 | 67.250958 |
| flair1/flair1 | 33.608404 | 34.062366 | 32.308301 | 32.837057 | 31.747681 | 32.111892 |

## Per-Class Screening Effects

| Dataset/protocol | Count | Class | All IoU | Screened IoU | Delta pp |
| --- | ---: | --- | ---: | ---: | ---: |
| vdd/vdd | 20 | other | 59.7766 | 57.2425 | -2.5341 |
| vdd/vdd | 20 | wall | 59.6816 | 58.4390 | -1.2426 |
| vdd/vdd | 20 | road | 23.2250 | 21.3171 | -1.9079 |
| vdd/vdd | 20 | vegetation | 44.1426 | 50.4296 | 6.2870 |
| vdd/vdd | 20 | vehicle | 50.3084 | 42.1815 | -8.1269 |
| vdd/vdd | 20 | roof | 86.2607 | 89.7289 | 3.4682 |
| vdd/vdd | 20 | water | 52.8339 | 61.0704 | 8.2365 |
| vdd/vdd | 30 | other | 64.4934 | 63.8205 | -0.6729 |
| vdd/vdd | 30 | wall | 55.2097 | 58.5129 | 3.3032 |
| vdd/vdd | 30 | road | 22.2954 | 22.3585 | 0.0631 |
| vdd/vdd | 30 | vegetation | 56.6587 | 65.4790 | 8.8203 |
| vdd/vdd | 30 | vehicle | 60.2957 | 55.0546 | -5.2411 |
| vdd/vdd | 30 | roof | 81.5528 | 88.0723 | 6.5195 |
| vdd/vdd | 30 | water | 53.8929 | 63.0611 | 9.1682 |
| vdd/vdd | 40 | other | 61.1784 | 60.4904 | -0.6880 |
| vdd/vdd | 40 | wall | 35.7086 | 41.9407 | 6.2321 |
| vdd/vdd | 40 | road | 29.6075 | 26.8076 | -2.7999 |
| vdd/vdd | 40 | vegetation | 53.6668 | 62.6867 | 9.0199 |
| vdd/vdd | 40 | vehicle | 59.7555 | 57.9008 | -1.8547 |
| vdd/vdd | 40 | roof | 82.2517 | 88.7265 | 6.4748 |
| vdd/vdd | 40 | water | 54.6992 | 59.7203 | 5.0211 |
| potsdam/potsdam | 20 | impervious surface | 65.8977 | 66.2446 | 0.3469 |
| potsdam/potsdam | 20 | building | 71.0529 | 72.5275 | 1.4746 |
| potsdam/potsdam | 20 | low vegetation | 14.4124 | 19.6733 | 5.2609 |
| potsdam/potsdam | 20 | tree | 55.5064 | 57.9390 | 2.4326 |
| potsdam/potsdam | 20 | car | 24.5378 | 24.9082 | 0.3704 |
| potsdam/potsdam | 20 | clutter | 2.1143 | 2.4111 | 0.2968 |
| potsdam/potsdam | 30 | impervious surface | 59.9065 | 60.8771 | 0.9706 |
| potsdam/potsdam | 30 | building | 75.7298 | 77.6742 | 1.9444 |
| potsdam/potsdam | 30 | low vegetation | 29.2859 | 32.8221 | 3.5362 |
| potsdam/potsdam | 30 | tree | 55.6904 | 57.7302 | 2.0398 |
| potsdam/potsdam | 30 | car | 19.5819 | 20.0561 | 0.4742 |
| potsdam/potsdam | 30 | clutter | 2.7361 | 3.0665 | 0.3304 |
| potsdam/potsdam | 40 | impervious surface | 56.3182 | 57.5743 | 1.2561 |
| potsdam/potsdam | 40 | building | 76.6916 | 78.9812 | 2.2896 |
| potsdam/potsdam | 40 | low vegetation | 19.9363 | 24.5027 | 4.5664 |
| potsdam/potsdam | 40 | tree | 54.1800 | 55.4947 | 1.3147 |
| potsdam/potsdam | 40 | car | 18.2124 | 18.6020 | 0.3896 |
| potsdam/potsdam | 40 | clutter | 2.6284 | 2.9633 | 0.3349 |
| udd5/udd5 | 20 | vegetation | 46.2799 | 67.6962 | 21.4163 |
| udd5/udd5 | 20 | building | 83.4871 | 85.5987 | 2.1116 |
| udd5/udd5 | 20 | road | 0.0000 | 0.0000 | 0.0000 |
| udd5/udd5 | 20 | vehicle | 5.3645 | 7.0104 | 1.6459 |
| udd5/udd5 | 20 | other | 5.7476 | 9.8917 | 4.1441 |
| udd5/udd5 | 30 | vegetation | 51.6449 | 69.7258 | 18.0809 |
| udd5/udd5 | 30 | building | 84.3361 | 85.9091 | 1.5730 |
| udd5/udd5 | 30 | road | 0.0000 | 0.0000 | 0.0000 |
| udd5/udd5 | 30 | vehicle | 10.9993 | 11.6225 | 0.6232 |
| udd5/udd5 | 30 | other | 18.8780 | 25.6264 | 6.7484 |
| udd5/udd5 | 40 | vegetation | 56.1012 | 71.0307 | 14.9295 |
| udd5/udd5 | 40 | building | 81.4500 | 84.3415 | 2.8915 |
| udd5/udd5 | 40 | road | 0.0000 | 0.0000 | 0.0000 |
| udd5/udd5 | 40 | vehicle | 10.1303 | 10.0602 | -0.0701 |
| udd5/udd5 | 40 | other | 7.7054 | 17.5219 | 9.8165 |
| oem/oem | 20 | bareland | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 20 | rangeland | 49.5261 | 49.6933 | 0.1672 |
| oem/oem | 20 | developed space | 23.7669 | 23.8549 | 0.0880 |
| oem/oem | 20 | road | 54.5529 | 54.9225 | 0.3696 |
| oem/oem | 20 | tree | 55.6605 | 56.9837 | 1.3232 |
| oem/oem | 20 | water | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 20 | agriculture land | 81.6500 | 82.0069 | 0.3569 |
| oem/oem | 20 | building | 47.0287 | 50.8636 | 3.8349 |
| oem/oem | 30 | bareland | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 30 | rangeland | 48.6056 | 48.8021 | 0.1965 |
| oem/oem | 30 | developed space | 22.2950 | 22.8153 | 0.5203 |
| oem/oem | 30 | road | 52.5545 | 53.2714 | 0.7169 |
| oem/oem | 30 | tree | 50.2766 | 53.0830 | 2.8064 |
| oem/oem | 30 | water | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 30 | agriculture land | 79.1320 | 79.9648 | 0.8328 |
| oem/oem | 30 | building | 7.5977 | 8.5268 | 0.9291 |
| oem/oem | 40 | bareland | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 40 | rangeland | 46.2375 | 46.6698 | 0.4323 |
| oem/oem | 40 | developed space | 19.9391 | 20.4093 | 0.4702 |
| oem/oem | 40 | road | 55.5133 | 55.9670 | 0.4537 |
| oem/oem | 40 | tree | 49.4183 | 52.4450 | 3.0267 |
| oem/oem | 40 | water | 0.0000 | 0.0000 | 0.0000 |
| oem/oem | 40 | agriculture land | 76.0404 | 77.3255 | 1.2851 |
| oem/oem | 40 | building | 6.0097 | 7.0826 | 1.0729 |
| loveda/P | 20 | building | 60.8479 | 53.9715 | -6.8764 |
| loveda/P | 20 | road | 53.7553 | 58.5446 | 4.7893 |
| loveda/P | 20 | water | 62.4091 | 62.5489 | 0.1398 |
| loveda/P | 20 | barren | 4.3215 | 2.3466 | -1.9749 |
| loveda/P | 20 | tree | 39.9553 | 53.2126 | 13.2573 |
| loveda/P | 20 | farm | 82.9502 | 84.9031 | 1.9529 |
| loveda/D | 20 | background | 7.3224 | 25.9735 | 18.6511 |
| loveda/D | 20 | building | 31.2821 | 38.0886 | 6.8065 |
| loveda/D | 20 | road | 44.1270 | 48.5879 | 4.4609 |
| loveda/D | 20 | water | 54.4379 | 54.7088 | 0.2709 |
| loveda/D | 20 | barren | 3.2414 | 1.7765 | -1.4649 |
| loveda/D | 20 | tree | 35.8674 | 45.9895 | 10.1221 |
| loveda/D | 20 | farm | 38.8741 | 45.3841 | 6.5100 |
| loveda/P | 30 | building | 80.9783 | 78.0612 | -2.9171 |
| loveda/P | 30 | road | 51.7001 | 56.5965 | 4.8964 |
| loveda/P | 30 | water | 60.8726 | 61.8570 | 0.9844 |
| loveda/P | 30 | barren | 4.4950 | 3.2039 | -1.2911 |
| loveda/P | 30 | tree | 40.3072 | 54.6888 | 14.3816 |
| loveda/P | 30 | farm | 84.2463 | 86.8829 | 2.6366 |
| loveda/D | 30 | background | 19.0231 | 33.3174 | 14.2943 |
| loveda/D | 30 | building | 46.5686 | 40.4591 | -6.1095 |
| loveda/D | 30 | road | 45.1544 | 48.3255 | 3.1711 |
| loveda/D | 30 | water | 54.7976 | 56.1870 | 1.3894 |
| loveda/D | 30 | barren | 3.2240 | 1.8282 | -1.3958 |
| loveda/D | 30 | tree | 23.3427 | 32.8531 | 9.5104 |
| loveda/D | 30 | farm | 43.2197 | 47.1301 | 3.9104 |
| loveda/P | 40 | building | 80.0532 | 78.0612 | -1.9920 |
| loveda/P | 40 | road | 51.0321 | 57.5059 | 6.4738 |
| loveda/P | 40 | water | 61.6426 | 63.2126 | 1.5700 |
| loveda/P | 40 | barren | 5.8374 | 6.1655 | 0.3281 |
| loveda/P | 40 | tree | 37.8742 | 52.2571 | 14.3829 |
| loveda/P | 40 | farm | 84.3315 | 87.2959 | 2.9644 |
| loveda/D | 40 | background | 17.2595 | 32.0359 | 14.7764 |
| loveda/D | 40 | building | 35.8173 | 33.3333 | -2.4840 |
| loveda/D | 40 | road | 43.2810 | 47.6762 | 4.3952 |
| loveda/D | 40 | water | 55.7298 | 57.3131 | 1.5833 |
| loveda/D | 40 | barren | 4.7648 | 3.2972 | -1.4676 |
| loveda/D | 40 | tree | 30.9531 | 40.9420 | 9.9889 |
| loveda/D | 40 | farm | 42.4002 | 47.6023 | 5.2021 |
| vaihingen/vaihingen | 20 | impervious surface | 57.3052 | 58.8103 | 1.5051 |
| vaihingen/vaihingen | 20 | building | 65.7437 | 67.0764 | 1.3327 |
| vaihingen/vaihingen | 20 | low vegetation | 43.7153 | 44.9502 | 1.2349 |
| vaihingen/vaihingen | 20 | tree | 67.2612 | 67.5208 | 0.2596 |
| vaihingen/vaihingen | 20 | car | 25.1095 | 26.3654 | 1.2559 |
| vaihingen/vaihingen | 30 | impervious surface | 45.6855 | 48.8028 | 3.1173 |
| vaihingen/vaihingen | 30 | building | 69.0365 | 70.8142 | 1.7777 |
| vaihingen/vaihingen | 30 | low vegetation | 47.2725 | 48.8878 | 1.6153 |
| vaihingen/vaihingen | 30 | tree | 67.4294 | 67.4207 | -0.0087 |
| vaihingen/vaihingen | 30 | car | 14.0480 | 15.3680 | 1.3200 |
| vaihingen/vaihingen | 40 | impervious surface | 43.1441 | 46.9164 | 3.7723 |
| vaihingen/vaihingen | 40 | building | 69.0169 | 70.0014 | 0.9845 |
| vaihingen/vaihingen | 40 | low vegetation | 45.4683 | 47.2823 | 1.8140 |
| vaihingen/vaihingen | 40 | tree | 67.5991 | 67.3715 | -0.2276 |
| vaihingen/vaihingen | 40 | car | 13.2870 | 14.8039 | 1.5169 |
| landcoverai/landcoverai | 20 | background | 87.4477 | 87.9828 | 0.5351 |
| landcoverai/landcoverai | 20 | building | 44.4672 | 44.5590 | 0.0918 |
| landcoverai/landcoverai | 20 | woodland | 78.6026 | 80.5571 | 1.9545 |
| landcoverai/landcoverai | 20 | water | 97.6178 | 97.6134 | -0.0044 |
| landcoverai/landcoverai | 20 | road | 26.3947 | 26.7045 | 0.3098 |
| landcoverai/landcoverai | 30 | background | 87.8452 | 88.1214 | 0.2762 |
| landcoverai/landcoverai | 30 | building | 43.6788 | 43.8490 | 0.1702 |
| landcoverai/landcoverai | 30 | woodland | 83.6138 | 84.3587 | 0.7449 |
| landcoverai/landcoverai | 30 | water | 97.6343 | 97.6404 | 0.0061 |
| landcoverai/landcoverai | 30 | road | 22.7480 | 23.2201 | 0.4721 |
| landcoverai/landcoverai | 40 | background | 87.7170 | 87.8795 | 0.1625 |
| landcoverai/landcoverai | 40 | building | 44.3938 | 44.5567 | 0.1629 |
| landcoverai/landcoverai | 40 | woodland | 85.3754 | 85.6155 | 0.2401 |
| landcoverai/landcoverai | 40 | water | 97.0400 | 97.0455 | 0.0055 |
| landcoverai/landcoverai | 40 | road | 20.6965 | 21.1576 | 0.4611 |
| flair1/flair1 | 20 | building | 56.7755 | 57.1752 | 0.3997 |
| flair1/flair1 | 20 | pervious surface | 47.2753 | 46.6274 | -0.6479 |
| flair1/flair1 | 20 | impervious surface | 51.9858 | 52.5746 | 0.5888 |
| flair1/flair1 | 20 | bare soil | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 20 | water | 61.4565 | 63.6878 | 2.2313 |
| flair1/flair1 | 20 | coniferous | 43.5282 | 42.7771 | -0.7511 |
| flair1/flair1 | 20 | deciduous | 59.5237 | 60.3808 | 0.8571 |
| flair1/flair1 | 20 | brushwood | 12.3287 | 14.8835 | 2.5548 |
| flair1/flair1 | 20 | vineyard | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 20 | herbaceous vegetation | 57.4725 | 57.5245 | 0.0520 |
| flair1/flair1 | 20 | agricultural land | 12.9546 | 13.1175 | 0.1629 |
| flair1/flair1 | 20 | plowed land | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 30 | building | 56.0942 | 56.6445 | 0.5503 |
| flair1/flair1 | 30 | pervious surface | 44.2154 | 44.1556 | -0.0598 |
| flair1/flair1 | 30 | impervious surface | 54.8734 | 55.6509 | 0.7775 |
| flair1/flair1 | 30 | bare soil | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 30 | water | 61.0597 | 62.6809 | 1.6212 |
| flair1/flair1 | 30 | coniferous | 41.9349 | 41.3575 | -0.5774 |
| flair1/flair1 | 30 | deciduous | 56.1805 | 58.1441 | 1.9636 |
| flair1/flair1 | 30 | brushwood | 13.0272 | 14.5021 | 1.4749 |
| flair1/flair1 | 30 | vineyard | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 30 | herbaceous vegetation | 50.7481 | 50.8840 | 0.1359 |
| flair1/flair1 | 30 | agricultural land | 9.5662 | 10.0250 | 0.4588 |
| flair1/flair1 | 30 | plowed land | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 40 | building | 58.1241 | 58.6612 | 0.5371 |
| flair1/flair1 | 40 | pervious surface | 39.6518 | 39.8636 | 0.2118 |
| flair1/flair1 | 40 | impervious surface | 57.7004 | 58.3287 | 0.6283 |
| flair1/flair1 | 40 | bare soil | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 40 | water | 65.8654 | 66.9900 | 1.1246 |
| flair1/flair1 | 40 | coniferous | 43.3541 | 43.3732 | 0.0191 |
| flair1/flair1 | 40 | deciduous | 55.3096 | 57.0630 | 1.7534 |
| flair1/flair1 | 40 | brushwood | 12.6656 | 14.0276 | 1.3620 |
| flair1/flair1 | 40 | vineyard | 0.0000 | 0.0000 | 0.0000 |
| flair1/flair1 | 40 | herbaceous vegetation | 40.1684 | 39.1899 | -0.9785 |
| flair1/flair1 | 40 | agricultural land | 8.1328 | 7.8454 | -0.2874 |
| flair1/flair1 | 40 | plowed land | 0.0000 | 0.0000 | 0.0000 |

## Correction Transitions

Wrong-to-correct and correct-to-wrong pixels compare each screened endpoint with its same-count unscreened control. Wrong-to-wrong changes are listed separately; these counts are not equivalent to mIoU.

| Dataset/protocol | Count | Changed | Beneficial | Harmful | Wrong to wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 20 | 116675 | 49565 | 46562 | 20548 |
| vdd/vdd | 30 | 133700 | 78185 | 34067 | 21448 |
| vdd/vdd | 40 | 157409 | 75417 | 35634 | 46358 |
| potsdam/potsdam | 20 | 71867 | 43913 | 7377 | 20577 |
| potsdam/potsdam | 30 | 68183 | 42053 | 6249 | 19881 |
| potsdam/potsdam | 40 | 73603 | 47262 | 8935 | 17406 |
| udd5/udd5 | 20 | 125038 | 45375 | 5731 | 73932 |
| udd5/udd5 | 30 | 82553 | 54589 | 8269 | 19695 |
| udd5/udd5 | 40 | 120067 | 63081 | 5017 | 51969 |
| oem/oem | 20 | 38843 | 25549 | 7646 | 5648 |
| oem/oem | 30 | 33444 | 24355 | 2724 | 6365 |
| oem/oem | 40 | 40428 | 26056 | 2469 | 11903 |
| loveda/P | 20 | 37023 | 27988 | 2706 | 6329 |
| loveda/D | 20 | 245652 | 207690 | 2693 | 35269 |
| loveda/P | 30 | 45416 | 30527 | 2066 | 12823 |
| loveda/D | 30 | 238120 | 180375 | 32492 | 25253 |
| loveda/P | 40 | 56706 | 34455 | 2360 | 19891 |
| loveda/D | 40 | 235059 | 180962 | 17616 | 36481 |
| vaihingen/vaihingen | 20 | 42922 | 24562 | 4429 | 13931 |
| vaihingen/vaihingen | 30 | 64248 | 39542 | 5100 | 19606 |
| vaihingen/vaihingen | 40 | 78568 | 47529 | 11364 | 19675 |
| landcoverai/landcoverai | 20 | 16064 | 12707 | 3271 | 86 |
| landcoverai/landcoverai | 30 | 11884 | 8275 | 3458 | 151 |
| landcoverai/landcoverai | 40 | 8501 | 5524 | 2722 | 255 |
| flair1/flair1 | 20 | 34223 | 14251 | 8634 | 11338 |
| flair1/flair1 | 30 | 45891 | 20979 | 8049 | 16863 |
| flair1/flair1 | 40 | 49720 | 19307 | 13947 | 16466 |

## Retention

Equal-domain means use LoveDA D once; averaging is over valid queries and class/rival pairs, then images and domains.

| Context count | Mean retained count | Original20 retained % | Added words retained % |
| ---: | ---: | ---: | ---: |
| 20 | 18.5106 | 92.5529 | -- |
| 30 | 27.8338 | 92.6556 | 93.0269 |
| 40 | 37.1591 | 92.6436 | 93.1519 |

```json
{
  "vdd": {
    "k20": {
      "vdd": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.006231398809522,
        "old20_retention_fraction": 0.9003115699404763,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "vdd": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 27.034734816778272,
        "old20_retention_fraction": 0.9020016624813988,
        "added_retention_fraction": 0.89947015671503
      }
    },
    "k40": {
      "vdd": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 36.165931338355655,
        "old20_retention_fraction": 0.9034705752418155,
        "added_retention_fraction": 0.9048259916759674
      }
    }
  },
  "potsdam": {
    "k20": {
      "potsdam": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.684269205729166,
        "old20_retention_fraction": 0.9342134602864584,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "potsdam": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 28.078584798177083,
        "old20_retention_fraction": 0.9353350830078125,
        "added_retention_fraction": 0.9371883138020833
      }
    },
    "k40": {
      "potsdam": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 37.4288330078125,
        "old20_retention_fraction": 0.9338527425130211,
        "added_retention_fraction": 0.9375889078776042
      }
    }
  },
  "udd5": {
    "k20": {
      "udd5": {
        "minimum": 2,
        "maximum": 20,
        "image_mean": 18.368023681640626,
        "old20_retention_fraction": 0.9184011840820313,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "udd5": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 27.819329833984376,
        "old20_retention_fraction": 0.9164248657226564,
        "added_retention_fraction": 0.9490832519531252
      }
    },
    "k40": {
      "udd5": {
        "minimum": 4,
        "maximum": 40,
        "image_mean": 37.1401611328125,
        "old20_retention_fraction": 0.9157806396484375,
        "added_retention_fraction": 0.9412274169921876
      }
    }
  },
  "oem": {
    "k20": {
      "oem": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.904854910714285,
        "old20_retention_fraction": 0.9452427455357143,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "oem": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 28.286383492606024,
        "old20_retention_fraction": 0.9452352251325336,
        "added_retention_fraction": 0.9381678989955358
      }
    },
    "k40": {
      "oem": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 37.77573503766741,
        "old20_retention_fraction": 0.9448150634765626,
        "added_retention_fraction": 0.9439716884068081
      }
    }
  },
  "loveda": {
    "k20": {
      "P": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 17.834236653645835,
        "old20_retention_fraction": 0.8917118326822916,
        "added_retention_fraction": null
      },
      "D": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 17.938191731770832,
        "old20_retention_fraction": 0.8969095865885418,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "P": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 26.804134114583334,
        "old20_retention_fraction": 0.8934222412109376,
        "added_retention_fraction": 0.8935689290364582
      },
      "D": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 27.015648251488095,
        "old20_retention_fraction": 0.899189685639881,
        "added_retention_fraction": 0.9031854538690478
      }
    },
    "k40": {
      "P": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 35.968754069010416,
        "old20_retention_fraction": 0.8975801595052083,
        "added_retention_fraction": 0.9008575439453125
      },
      "D": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 36.18592761811756,
        "old20_retention_fraction": 0.9029129754929316,
        "added_retention_fraction": 0.9063834054129465
      }
    }
  },
  "vaihingen": {
    "k20": {
      "vaihingen": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.43699951171875,
        "old20_retention_fraction": 0.9218499755859375,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "vaihingen": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 27.7818603515625,
        "old20_retention_fraction": 0.9216244506835938,
        "added_retention_fraction": 0.9349371337890624
      }
    },
    "k40": {
      "vaihingen": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 37.13667602539063,
        "old20_retention_fraction": 0.9191717529296873,
        "added_retention_fraction": 0.9376620483398437
      }
    }
  },
  "landcoverai": {
    "k20": {
      "landcoverai": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.836230468750003,
        "old20_retention_fraction": 0.9418115234375002,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "landcoverai": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 28.303375244140625,
        "old20_retention_fraction": 0.9451412963867188,
        "added_retention_fraction": 0.9400549316406248
      }
    },
    "k40": {
      "landcoverai": {
        "minimum": 3,
        "maximum": 40,
        "image_mean": 37.657012939453125,
        "old20_retention_fraction": 0.9434725952148437,
        "added_retention_fraction": 0.9393780517578125
      }
    }
  },
  "flair1": {
    "k20": {
      "flair1": {
        "minimum": 1,
        "maximum": 20,
        "image_mean": 18.909873268821023,
        "old20_retention_fraction": 0.9454936634410512,
        "added_retention_fraction": null
      }
    },
    "k30": {
      "flair1": {
        "minimum": 1,
        "maximum": 30,
        "image_mean": 28.350557269472066,
        "old20_retention_fraction": 0.9474942756421637,
        "added_retention_fraction": 0.9400671756628789
      }
    },
    "k40": {
      "flair1": {
        "minimum": 1,
        "maximum": 40,
        "image_mean": 37.782490123401985,
        "old20_retention_fraction": 0.9480115948301373,
        "added_retention_fraction": 0.9411129113399621
      }
    }
  }
}
```

## Cost And Limits

Suite wall 201.2641s; shared18-endpoint evaluation, not per-model latency. The source generation first greedy attempt failed structural uniqueness and is preserved; the complete bank uses fixed seed20261003,temperature0.7,top_p0.9,repetition_penalty1.1. No segmentation results selected words. Count and added phrase identity change together; this is not a universal claim about every30/40 bank. The local Geometry vocabulary did not change.

| Dataset | Shared worker seconds | Peak allocated MiB |
| --- | ---: | ---: |
| vdd | 22.3320 | 5696.0552 |
| potsdam | 21.0032 | 5630.6011 |
| udd5 | 17.8870 | 5617.1440 |
| oem | 25.8556 | 5805.8589 |
| loveda | 39.3237 | 5815.0718 |
| vaihingen | 18.9500 | 5617.1440 |
| landcoverai | 18.6337 | 5617.1440 |
| flair1 | 35.8903 | 6329.1704 |
