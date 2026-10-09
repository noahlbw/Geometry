# Bounded zero-DC coupling: full four-domain mechanism test

Fixed20 words/class, unchanged masks/checkpoints/Geometry/wide text and bounded896/448 views. Primary L + P H P(B-L) removes valid-window mean class offsets before and after Geometry transport. No alias admission, extra RGB encodings, threshold fitting or domain routing. SCLIP is a published matched local-head control, not our innovation. Developed validation domains; exploratory, not untouched validation.

| Dataset | VIP All20 | Geometry | Coupled | Zero-DC coupled | SCLIP coupled | SCLIP zero-DC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 51.4827 | 46.6010 | 53.8090 | 46.0956 | 53.5257 | 44.8421 |
| potsdam | 42.3028 | 40.3779 | 43.4301 | 45.0251 | 45.6185 | 47.9042 |
| udd5 | 41.4336 | 47.1219 | 47.0096 | 48.3931 | 46.7075 | 48.9586 |
| oem | 33.2064 | 43.9751 | 37.2104 | 41.8793 | 36.1612 | 42.1482 |

All1008 unique images and original Geometry/no-admission confusion sums verified. This study changes score coupling, not just execution; zero intermediate mean does not guarantee class-area or mIoU preservation.

## Per-Class IoU

| Domain | Class | Coupled | Zero-DC | SCLIP coupled | SCLIP zero-DC |
| --- | --- | ---: | ---: | ---: | ---: |
| vdd | other | 34.6165 | 18.8329 | 32.7789 | 11.1847 |
| vdd | wall | 44.4124 | 34.6975 | 43.3988 | 29.0370 |
| vdd | road | 41.3459 | 36.8833 | 40.4711 | 33.1789 |
| vdd | vegetation | 65.0501 | 68.5862 | 68.5395 | 73.3681 |
| vdd | vehicle | 24.1562 | 10.7123 | 23.1693 | 13.3988 |
| vdd | roof | 82.5973 | 80.5634 | 81.2545 | 77.9368 |
| vdd | water | 84.4846 | 72.3938 | 85.0677 | 75.7902 |
| potsdam | impervious surface | 61.0891 | 54.1765 | 62.7376 | 59.8867 |
| potsdam | building | 78.6001 | 81.6290 | 77.8507 | 79.8840 |
| potsdam | low vegetation | 33.6290 | 49.7614 | 36.8971 | 53.7261 |
| potsdam | tree | 54.4128 | 60.6747 | 58.2877 | 65.4324 |
| potsdam | car | 26.6264 | 13.2787 | 31.9433 | 20.2951 |
| potsdam | clutter | 6.2234 | 10.6306 | 5.9946 | 8.2007 |
| udd5 | vegetation | 66.3933 | 74.7053 | 66.0396 | 74.9819 |
| udd5 | building | 80.9728 | 86.2923 | 80.3869 | 85.7899 |
| udd5 | road | 36.8085 | 42.1611 | 36.5998 | 46.0342 |
| udd5 | vehicle | 17.2031 | 7.8535 | 17.8845 | 9.1284 |
| udd5 | other | 33.6700 | 30.9533 | 32.6271 | 28.8583 |
| oem | bareland | 10.0142 | 9.7268 | 10.1246 | 9.3241 |
| oem | rangeland | 17.6999 | 28.8578 | 17.3615 | 31.7004 |
| oem | developed space | 28.0239 | 22.7104 | 27.9050 | 18.0059 |
| oem | road | 33.4645 | 35.4783 | 32.4560 | 36.0109 |
| oem | tree | 36.0697 | 54.5122 | 34.8510 | 54.5603 |
| oem | water | 67.8635 | 63.9846 | 67.9855 | 66.9843 |
| oem | agriculture land | 61.5442 | 62.3686 | 56.8213 | 64.0328 |
| oem | building | 43.0029 | 57.3955 | 41.7847 | 56.5674 |

## Competition Outcomes

| Domain | Class | Arm | IoU | Precision | Recall | Predicted area % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| vdd | vehicle | NoAdmission_Exact | 24.1562 | 26.3075 | 74.7086 | 1.4785 |
| vdd | wall | NoAdmission_Exact | 44.4124 | 75.9742 | 51.6693 | 2.2004 |
| vdd | water | NoAdmission_Exact | 84.4846 | 88.1803 | 95.2737 | 16.4085 |
| vdd | vehicle | Geometry_ContrastCoupled | 10.7123 | 10.8701 | 88.0703 | 4.2182 |
| vdd | wall | Geometry_ContrastCoupled | 34.6975 | 39.7448 | 73.2065 | 5.9595 |
| vdd | water | Geometry_ContrastCoupled | 72.3938 | 89.8517 | 78.8403 | 13.3257 |
| vdd | vehicle | SCLIP_NoAdmission | 23.1693 | 25.3408 | 73.0003 | 1.4998 |
| vdd | wall | SCLIP_NoAdmission | 43.3988 | 70.9938 | 52.7526 | 2.4042 |
| vdd | water | SCLIP_NoAdmission | 85.0677 | 88.4496 | 95.6987 | 16.4315 |
| vdd | vehicle | SCLIP_ContrastCoupled | 13.3988 | 13.6840 | 86.5400 | 3.2925 |
| vdd | wall | SCLIP_ContrastCoupled | 29.0370 | 31.7954 | 76.9956 | 7.8351 |
| vdd | water | SCLIP_ContrastCoupled | 75.7902 | 90.5460 | 82.3031 | 13.8043 |
| potsdam | car | NoAdmission_Exact | 26.6264 | 26.6925 | 99.0780 | 7.2934 |
| potsdam | low vegetation | NoAdmission_Exact | 33.6290 | 93.0373 | 34.4972 | 7.7898 |
| potsdam | car | Geometry_ContrastCoupled | 13.2787 | 13.2840 | 99.7019 | 14.7475 |
| potsdam | low vegetation | Geometry_ContrastCoupled | 49.7614 | 88.4807 | 53.2085 | 12.6338 |
| potsdam | car | SCLIP_NoAdmission | 31.9433 | 32.0562 | 98.9099 | 6.0627 |
| potsdam | low vegetation | SCLIP_NoAdmission | 36.8971 | 92.3868 | 38.0541 | 8.6535 |
| potsdam | car | SCLIP_ContrastCoupled | 20.2951 | 20.3100 | 99.6393 | 9.6397 |
| potsdam | low vegetation | SCLIP_ContrastCoupled | 53.7261 | 87.8308 | 58.0471 | 13.8847 |

## Decision

Do NOT select the zero-DC primary or roll it out as the final eight-domain model. It improves Potsdam, UDD5 and OEM but loses7.7134pp on VDD. On Potsdam the total 45.0251 exceeds official-short VIP44.8995, yet car IoU drops26.6264 to13.2787 and predicted car area7.2934% to14.7475%. The aggregate gain comes with worse small-object competition; class-wide broad information is not uniformly harmful.

The same-view published SCLIP local-head control, with the original unscreened reconstruction, reaches Potsdam45.6185 and car31.9433 (versus coupled43.4301 and car26.6264). It changes no RGB encoding budget. This is evidence of a local semantic-readout opportunity, not an independent Geometry innovation or proof of all-eight performance. Removing class-wide offsets from SCLIP increases Potsdam total further to47.9042 but drops car to20.2951 and VDD to44.8421.

Next design constraint: preserve useful broad class calibration while recovering discriminative local contrast. Mean-only gating, per-domain choice and label-chosen class exceptions are not justified by this experiment. No new complete-image singleton timing is claimed for these rejected candidates; existing no-admission timing must not be relabeled as SCLIP or zero-DC timing.
