# Bounded896 Geometry Coupling: Two-Domain Gate

Fixed20 words/class and unchanged Geometry/soft rule/wide head/reconstruction. Geometry receives long-edge896/512 crops, wide receives independently resized original RGB long-edge448/336 crops. Both branches enforce at most4 actual calls; no native/fine windows. Full original-size masks, restored probabilities before argmax. Exploratory development, not independent validation or a new contribution claim.

| Dataset | VIP All20 | Geometry | No admission | Soft primary | Soft - VIP | Soft - no admission | Native soft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 51.4827 | 46.6010 | 53.8090 | 53.7886 | +2.3059 | -0.0204 | 55.4932 |
| potsdam | 42.3028 | 40.3779 | 43.4301 | 43.3830 | +1.0802 | -0.0471 | 43.6900 |

VIP is the already completed repaired All20 comparator, not historical NaN outputs, official short queries or published paper numbers. Same20 words, scored masks and checkpoints are verified; templates/readout/background settings differ by method.

## Complete-Image Timing

| Dataset | Sample | HxW | VIP ms | No admission ms | Soft ms | Soft/VIP | Peak shared MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd | DJI_0007.JPG | 3000x4000 | 128.00 | 324.63 | 392.17 | 3.064x | 5792.39 |
| vdd | DJI_0826.JPG | 3000x4000 | 126.45 | 323.12 | 390.82 | 3.091x | 5792.39 |
| vdd | DJI_10708.JPG | 3000x4000 | 127.12 | 323.56 | 392.99 | 3.092x | 5792.39 |
| potsdam | top_potsdam_2_13_RGB_y00_x00.tif | 1000x1000 | 103.90 | 235.29 | 346.38 | 3.334x | 5696.90 |
| potsdam | top_potsdam_5_13_RGB_y00_x00.tif | 1000x1000 | 103.85 | 235.09 | 345.45 | 3.326x | 5696.90 |
| potsdam | top_potsdam_7_13_RGB_y05_x05.tif | 1000x1000 | 104.14 | 234.90 | 348.38 | 3.345x | 5696.90 |

First/middle/last COMPLETE images, three warmed synchronized rotated singleton repeats per image, serial VDD then Potsdam without other GPU jobs. Includes every window, shared alias correction, output interpolation/stitching/argmax. Excludes model/text/image loading and masks. This six-image panel is not full-domain average throughput. Peaks have both backbones resident and are not standalone deployment memory.

## Per-Class Results

| Dataset | Class | VIP IoU | Soft IoU | Precision | Recall | Predicted area % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd | other | 36.5655 | 34.6725 | 52.2929 | 50.7144 | 18.2434 |
| vdd | wall | 29.0582 | 44.2644 | 76.2403 | 51.3477 | 2.1791 |
| vdd | road | 45.0432 | 41.3118 | 42.5075 | 93.6249 | 12.0902 |
| vdd | vegetation | 50.2515 | 65.1044 | 96.7800 | 66.5458 | 24.5575 |
| vdd | vehicle | 29.6526 | 24.0637 | 26.1920 | 74.7569 | 1.4860 |
| vdd | roof | 85.5153 | 82.5731 | 83.2531 | 99.0205 | 25.0269 |
| vdd | water | 84.2928 | 84.5302 | 88.1842 | 95.3272 | 16.4170 |
| potsdam | impervious surface | 61.2488 | 60.9712 | 72.5450 | 79.2604 | 34.3844 |
| potsdam | building | 77.2218 | 78.5858 | 81.0135 | 96.3269 | 28.4408 |
| potsdam | low vegetation | 28.5736 | 33.5932 | 93.0714 | 34.4549 | 7.7774 |
| potsdam | tree | 50.9752 | 54.4453 | 83.4140 | 61.0551 | 12.4734 |
| potsdam | car | 30.4756 | 26.4974 | 26.5623 | 99.0851 | 7.3297 |
| potsdam | clutter | 5.3220 | 6.2053 | 8.6407 | 18.0441 | 9.5943 |

Gate: both-domain accuracy=True; all measured images below1000ms=True; preferred <=3x VIP=False. No control promotion or parameter change. Passing this gate does not prove all-eight performance or the complete CVPR objective. Both source datasets and all584 unique images/per-image confusion sums are verified.
