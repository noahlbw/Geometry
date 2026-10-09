# Frozen full RS/natural alias comparison

Matched20 PatchOnly2 isolates candidate additions. Specified historical five-target model is a whole-system comparison with different task/word/background choices; do not call it a pure alias ablation.

Full unique coverage within each protocol;45,200 protocol-image evaluations, 33,646 distinct dataset/image identities after shared VOC/PC/COCO protocols. Developed datasets and prior labeled method selection; exploratory, not untouched validation.

ADE original alias bed has trailing whitespace. Exact words/features preserved; canonical display label maps to its unique whitespace-equivalent original alias. Failed r1 outputs/logs preserved.

32-query interpolation and risk chunks; exact per-crop margins generated lazily. Original FP32/FP64 crop order, scalar equations and fixed-slot writer preserved. r2 ADE/COCO171 OOM logs preserved; all15 mask-free checks rerun for this execution path.

| Dataset/protocol | Full images | Matched20 PatchOnly2 | Lightweight | Retained fine | Specified historical | Light minus retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5/udd5 | 40 | 47.0786 | 49.5374 | 51.2305 | — | -1.6931 |
| vdd/vdd | 80 | 53.4600 | 53.4134 | 54.5409 | 55.2382 | -1.1275 |
| potsdam/potsdam | 504 | 45.9339 | 49.0237 | 49.1176 | 49.7888 | -0.0939 |
| oem/oem | 384 | 36.8400 | 41.1911 | 42.2557 | — | -1.0646 |
| voc21/voc21 | 1449 | 29.6765 | 30.5665 | 32.9048 | 71.0273 | -2.3383 |
| ade150 | 2000 | pending | pending | pending | pending | pending |
| context60 | 5105 | pending | pending | pending | pending | pending |
| vaihingen/vaihingen | 113 | 51.4282 | 53.9270 | 54.0722 | — | -0.1452 |
| voc20/voc20 | 1449 | 89.3515 | 89.2631 | 89.5870 | — | -0.3239 |
| loveda | 1669 | pending | pending | pending | pending | pending |
| landcoverai | 1602 | pending | pending | pending | pending | pending |
| context59 | 5105 | pending | pending | pending | pending | pending |
| coco_object81 | 5000 | pending | pending | pending | pending | pending |
| coco_stuff171 | 5000 | pending | pending | pending | pending | pending |
| flair1 | 15700 | pending | pending | pending | pending | pending |

| Protocol | Matched Patch2 ms | Light ms | Retained ms | Historical ms | Matched VIP20 ms | Light/VIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| udd5 | 412.74 | 521.53 | 841.87 | — | 173.77 | 3.001x |
| vdd | 455.29 | 591.23 | 959.53 | 489.40 | 190.15 | 3.109x |
| potsdam | 242.44 | 410.65 | 781.60 | 245.62 | 100.89 | 4.070x |
| oem | 246.39 | 419.28 | 793.34 | — | 104.77 | 4.002x |
| voc21 | 215.28 | 381.51 | 767.96 | 187.52 | 76.34 | 4.997x |
| ade150 | 268.16 | 1929.59 | 3728.61 | 241.59 | 120.08 | 16.069x |
| context60 | 214.89 | 563.69 | 1172.80 | 236.12 | 75.83 | 7.434x |
| vaihingen | 240.94 | 408.06 | 778.61 | — | 100.33 | 4.067x |
| voc20 | 210.63 | 372.64 | 757.00 | — | 74.44 | 5.006x |
| loveda | 258.13 | 600.97 | 1058.31 | — | 206.34 | 2.912x |
| landcoverai | 235.58 | 406.11 | 779.23 | — | 98.08 | 4.141x |
| context59 | 218.91 | 560.88 | 1165.42 | — | 75.68 | 7.411x |
| coco_object81 | 260.30 | 930.75 | 1736.62 | — | 117.24 | 7.939x |
| coco_stuff171 | 328.95 | 2893.70 | 5109.64 | — | 174.35 | 16.597x |
| flair1 | 241.48 | 419.33 | 795.70 | — | 103.48 | 4.052x |

3 fixed complete images/protocol,5 synchronized warmed rotated singleton repeats on exclusive GPU7; other GPU full workers allowed and recorded; includes resize/branches/readers/restoration/argmax, excludes disk decode/model/text/initial graph setup. Models jointly resident; memory not standalone.

The full-run aggregate GPU seconds include multiple arms and setup; they are not individual model latency. Per-class IoU/confusions and memory reside in each merged/timing JSON. Specified historical exists for the five identified protocols only; other protocols retain the matched20 PatchOnly2 control. Full accuracy is pending until coverage checks complete. Neither candidate is promoted or tuned by this suite.

Blocked: {"cityscapes19": "Incorrect full validation coverage: cityscapes19"}

Completed full protocols 7/15; timing panels 15/15.

Completed protocol/domain means (LoveDA D once; natural taxonomies remain separate): {"remote_sensing": {"Geometry_PatchOnly2Coupled": 46.94814, "SharedLocal_Soft": 49.41852, "OneSide_Stream": 50.24338}, "natural": {"Geometry_PatchOnly2Coupled": 59.514, "SharedLocal_Soft": 59.9148, "OneSide_Stream": 61.245900000000006}}

## udd5/udd5

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| vegetation | 66.2953 | 72.3672 | 73.1690 | N/A |
| building | 81.2771 | 82.6404 | 83.7786 | N/A |
| road | 37.2416 | 43.4736 | 44.7628 | N/A |
| vehicle | 16.9950 | 16.5053 | 19.4049 | N/A |
| other | 33.5843 | 32.7005 | 35.0373 | N/A |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 50.4522.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 53.7466.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 55.2788.

## vdd/vdd

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| other | 32.7096 | 29.5049 | 30.8967 | 35.5306 |
| wall | 43.4422 | 46.2906 | 48.8507 | 47.3411 |
| road | 40.6842 | 39.8020 | 41.2913 | 42.0360 |
| vegetation | 68.1015 | 71.0007 | 71.0937 | 66.7112 |
| vehicle | 22.7793 | 23.8292 | 25.2359 | 28.7947 |
| roof | 81.4826 | 82.7815 | 83.4638 | 83.2067 |
| water | 85.0207 | 80.6851 | 80.9539 | 83.0469 |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 56.9184.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 57.3982.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 58.4815.
Foreground/non-residual (explicit exclusion of background/other/clutter), PatchOnly2_SpecifiedHistorical: 58.5228.

## vaihingen/vaihingen

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| impervious surface | 61.5534 | 64.6595 | 64.4450 | N/A |
| building | 68.7479 | 72.6701 | 73.0297 | N/A |
| low vegetation | 32.1553 | 36.2372 | 35.1913 | N/A |
| tree | 67.7382 | 69.0065 | 69.1148 | N/A |
| car | 26.9462 | 27.0619 | 28.5801 | N/A |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 51.4282.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 53.9270.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 54.0722.

## oem/oem

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| bareland | 9.9823 | 9.8064 | 9.4165 | N/A |
| rangeland | 17.7049 | 21.6175 | 22.1836 | N/A |
| developed space | 27.9752 | 28.5580 | 29.0708 | N/A |
| road | 33.0407 | 37.2488 | 37.5029 | N/A |
| tree | 36.4308 | 46.5813 | 48.4634 | N/A |
| water | 67.9548 | 69.5448 | 69.4595 | N/A |
| agriculture land | 57.4689 | 60.5187 | 62.3254 | N/A |
| building | 44.1626 | 55.6534 | 59.6233 | N/A |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 36.8400.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 41.1911.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 42.2557.

## potsdam/potsdam

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| impervious surface | 62.7987 | 65.6165 | 65.1943 | 67.2087 |
| building | 79.0175 | 80.9405 | 81.2736 | 80.5641 |
| low vegetation | 38.5018 | 45.7869 | 45.8386 | 43.1983 |
| tree | 58.5794 | 63.5326 | 63.3542 | 61.8622 |
| car | 30.1233 | 30.0414 | 30.9856 | 36.7792 |
| clutter | 6.5825 | 8.2241 | 8.0591 | 9.1203 |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 53.8041.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 57.1836.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 57.3293.
Foreground/non-residual (explicit exclusion of background/other/clutter), PatchOnly2_SpecifiedHistorical: 57.9225.

## voc20/voc20

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| aeroplane | 98.7766 | 98.9884 | 98.9059 | N/A |
| bicycle | 68.8449 | 69.6716 | 70.4306 | N/A |
| bird | 98.9581 | 99.4413 | 99.5259 | N/A |
| boat | 92.6764 | 93.5370 | 94.2811 | N/A |
| bottle | 88.1792 | 87.8788 | 88.3977 | N/A |
| bus | 95.1836 | 94.1305 | 93.8788 | N/A |
| car | 94.3833 | 93.9080 | 93.0340 | N/A |
| cat | 98.5117 | 97.6762 | 97.1958 | N/A |
| chair | 54.3144 | 55.7895 | 57.7596 | N/A |
| cow | 96.0981 | 93.7647 | 94.3557 | N/A |
| diningtable | 76.4594 | 79.0886 | 81.0177 | N/A |
| dog | 96.3880 | 95.1505 | 94.3225 | N/A |
| horse | 95.4535 | 96.2012 | 96.6001 | N/A |
| motorbike | 83.5861 | 83.4977 | 84.0500 | N/A |
| person | 85.2902 | 87.0182 | 88.1035 | N/A |
| pottedplant | 96.2507 | 96.4513 | 96.2075 | N/A |
| sheep | 96.4166 | 93.7567 | 94.6900 | N/A |
| sofa | 83.4404 | 83.4567 | 83.6211 | N/A |
| train | 97.6368 | 96.4582 | 96.6706 | N/A |
| tvmonitor | 90.1829 | 89.3978 | 88.6922 | N/A |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 89.3515.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 89.2631.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 89.5870.

## voc21/voc21

| Class | Matched Patch2 IoU | Light IoU | Retained IoU | Historical IoU |
| --- | ---: | ---: | ---: | ---: |
| background | 22.9584 | 23.7174 | 30.8624 | 89.9681 |
| aeroplane | 14.5656 | 15.0992 | 15.5220 | 66.1426 |
| bicycle | 19.4848 | 20.6232 | 21.5374 | 46.5040 |
| bird | 12.5075 | 11.7392 | 12.3521 | 84.0958 |
| boat | 12.4851 | 12.7963 | 13.3436 | 60.0232 |
| bottle | 47.9232 | 46.7412 | 46.4902 | 54.8669 |
| bus | 49.0436 | 46.3046 | 49.6765 | 87.5949 |
| car | 29.3124 | 29.1786 | 32.1685 | 70.1323 |
| cat | 48.8822 | 54.4957 | 59.1460 | 90.3724 |
| chair | 13.9618 | 12.1949 | 12.9726 | 47.5988 |
| cow | 48.6032 | 52.7292 | 59.6717 | 91.8548 |
| diningtable | 23.2006 | 25.8858 | 28.8536 | 56.4400 |
| dog | 25.7741 | 22.6063 | 25.4440 | 84.8279 |
| horse | 38.2864 | 43.2996 | 45.7973 | 88.4919 |
| motorbike | 31.4640 | 32.7846 | 33.7094 | 72.7795 |
| person | 64.3244 | 67.9236 | 69.9673 | 72.6754 |
| pottedplant | 21.0998 | 20.2357 | 19.3201 | 49.8223 |
| sheep | 20.6334 | 21.9082 | 25.4166 | 90.3836 |
| sofa | 30.5125 | 31.3635 | 32.8498 | 66.1113 |
| train | 27.3487 | 27.7492 | 29.3736 | 60.1649 |
| tvmonitor | 20.8346 | 22.5214 | 26.5251 | 60.7224 |
Foreground/non-residual (explicit exclusion of background/other/clutter), Geometry_PatchOnly2Coupled: 30.0124.
Foreground/non-residual (explicit exclusion of background/other/clutter), SharedLocal_Soft: 30.9090.
Foreground/non-residual (explicit exclusion of background/other/clutter), OneSide_Stream: 33.0069.
Foreground/non-residual (explicit exclusion of background/other/clutter), PatchOnly2_SpecifiedHistorical: 70.0802.
