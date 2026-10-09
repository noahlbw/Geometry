# Bounded local contrast: fixed full-domain pilot

Frozen bounded896 Geometry/448 wide views, original H, unchanged20 words/class and broad class calibration. Only local readout changes. The primary adds mean-zero, input-derived per-head contrast compensation; DoubleRow and GainOnly are generic amplification controls; SCLIP is published prior art. No fitted threshold, domain routing, extra RGB or fine encoding. Developed domains; exploratory, not independent validation.

| Domain | VIP20 | Official/distilled VIP | Original coupled | Conservative | DoubleRow | GainOnly | SCLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 51.4827 | 52.0647 | 53.8090 | 53.9372 | 53.9690 | 53.8975 | 53.5257 |
| potsdam | 42.3028 | 44.8995 | 43.4301 | 44.2659 | 44.5348 | 44.0559 | 45.6185 |
| udd5 | 41.4336 | 44.9713 | 47.0096 | 47.6229 | 47.4622 | 47.2620 | 46.7075 |
| oem | 33.2064 | 35.0388 | 37.2104 | 38.1889 | 37.6046 | 37.4670 | 36.1612 |

All1008 unique images, frozen identities, original endpoints and per-image confusion sums verified.

## Per-Class Outcomes

| Domain | Class | Arm | IoU | Precision | Recall | Predicted area % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vdd | other | NoAdmission_Exact | 34.6165 | 52.1847 | 50.6965 | 18.2748 |
| vdd | wall | NoAdmission_Exact | 44.4124 | 75.9742 | 51.6693 | 2.2004 |
| vdd | road | NoAdmission_Exact | 41.3459 | 42.5370 | 93.6568 | 12.0859 |
| vdd | vegetation | NoAdmission_Exact | 65.0501 | 96.7735 | 66.4922 | 24.5393 |
| vdd | vehicle | NoAdmission_Exact | 24.1562 | 26.3075 | 74.7086 | 1.4785 |
| vdd | roof | NoAdmission_Exact | 82.5973 | 83.2883 | 99.0055 | 25.0126 |
| vdd | water | NoAdmission_Exact | 84.4846 | 88.1803 | 95.2737 | 16.4085 |
| vdd | other | Geometry_ConservativeCoupled | 34.0888 | 51.9572 | 49.7798 | 18.0229 |
| vdd | wall | Geometry_ConservativeCoupled | 45.3953 | 75.4015 | 53.2868 | 2.2866 |
| vdd | road | Geometry_ConservativeCoupled | 40.9775 | 42.0613 | 94.0836 | 12.2783 |
| vdd | vegetation | Geometry_ConservativeCoupled | 65.6265 | 96.7494 | 67.1062 | 24.7721 |
| vdd | vehicle | Geometry_ConservativeCoupled | 23.6928 | 25.8056 | 74.3184 | 1.4994 |
| vdd | roof | Geometry_ConservativeCoupled | 82.8697 | 83.6580 | 98.8757 | 24.8694 |
| vdd | water | Geometry_ConservativeCoupled | 84.9096 | 88.7782 | 95.1184 | 16.2714 |
| vdd | other | Geometry_DoubleRowCoupled | 34.3489 | 53.3347 | 49.1074 | 17.3202 |
| vdd | wall | Geometry_DoubleRowCoupled | 43.7970 | 76.3053 | 50.6910 | 2.1494 |
| vdd | road | Geometry_DoubleRowCoupled | 41.3216 | 42.4043 | 94.1805 | 12.1915 |
| vdd | vegetation | Geometry_DoubleRowCoupled | 67.0314 | 96.6386 | 68.6316 | 25.3642 |
| vdd | vehicle | Geometry_DoubleRowCoupled | 24.1405 | 26.4944 | 73.0978 | 1.4364 |
| vdd | roof | Geometry_DoubleRowCoupled | 82.0970 | 82.6948 | 99.1271 | 25.2230 |
| vdd | water | Geometry_DoubleRowCoupled | 85.0470 | 88.7408 | 95.3341 | 16.3152 |
| vdd | other | Geometry_GainOnlyCoupled | 34.5446 | 52.7713 | 50.0039 | 17.8247 |
| vdd | wall | Geometry_GainOnlyCoupled | 44.0080 | 76.4920 | 50.8909 | 2.1526 |
| vdd | road | Geometry_GainOnlyCoupled | 41.3478 | 42.4876 | 93.9073 | 12.1324 |
| vdd | vegetation | Geometry_GainOnlyCoupled | 66.0897 | 96.7287 | 67.6006 | 24.9600 |
| vdd | vehicle | Geometry_GainOnlyCoupled | 24.1527 | 26.4538 | 73.5214 | 1.4469 |
| vdd | roof | Geometry_GainOnlyCoupled | 82.3041 | 82.9374 | 99.0807 | 25.1375 |
| vdd | water | Geometry_GainOnlyCoupled | 84.8359 | 88.5412 | 95.2990 | 16.3459 |
| vdd | other | SCLIP_Coupled | 32.7789 | 54.3222 | 45.2513 | 15.6701 |
| vdd | wall | SCLIP_Coupled | 43.3988 | 70.9938 | 52.7526 | 2.4042 |
| vdd | road | SCLIP_Coupled | 40.4711 | 41.4265 | 94.6085 | 12.5360 |
| vdd | vegetation | SCLIP_Coupled | 68.5395 | 96.4570 | 70.3096 | 26.0333 |
| vdd | vehicle | SCLIP_Coupled | 23.1693 | 25.3408 | 73.0003 | 1.4998 |
| vdd | roof | SCLIP_Coupled | 81.2545 | 81.9293 | 98.9965 | 25.4251 |
| vdd | water | SCLIP_Coupled | 85.0677 | 88.4496 | 95.6987 | 16.4315 |
| potsdam | impervious surface | NoAdmission_Exact | 61.0891 | 72.6216 | 79.3681 | 34.3948 |
| potsdam | building | NoAdmission_Exact | 78.6001 | 81.0594 | 96.2834 | 28.4118 |
| potsdam | low vegetation | NoAdmission_Exact | 33.6290 | 93.0373 | 34.4972 | 7.7898 |
| potsdam | tree | NoAdmission_Exact | 54.4128 | 83.4512 | 60.9944 | 12.4555 |
| potsdam | car | NoAdmission_Exact | 26.6264 | 26.6925 | 99.0780 | 7.2934 |
| potsdam | clutter | NoAdmission_Exact | 6.2234 | 8.6467 | 18.1704 | 9.6547 |
| potsdam | impervious surface | Geometry_ConservativeCoupled | 61.3816 | 73.3511 | 78.9985 | 33.8941 |
| potsdam | building | Geometry_ConservativeCoupled | 79.0520 | 81.5554 | 96.2621 | 28.2328 |
| potsdam | low vegetation | Geometry_ConservativeCoupled | 35.8999 | 92.8790 | 36.9160 | 8.3502 |
| potsdam | tree | Geometry_ConservativeCoupled | 56.2229 | 83.3513 | 63.3355 | 12.9490 |
| potsdam | car | Geometry_ConservativeCoupled | 26.4992 | 26.5626 | 99.1072 | 7.3312 |
| potsdam | clutter | Geometry_ConservativeCoupled | 6.5398 | 9.1897 | 18.4870 | 9.2426 |
| potsdam | impervious surface | Geometry_DoubleRowCoupled | 61.6762 | 72.5294 | 80.4752 | 34.9189 |
| potsdam | building | Geometry_DoubleRowCoupled | 78.8720 | 81.2800 | 96.3798 | 28.3631 |
| potsdam | low vegetation | Geometry_DoubleRowCoupled | 35.2374 | 93.0246 | 36.1938 | 8.1741 |
| potsdam | tree | Geometry_DoubleRowCoupled | 56.1105 | 82.2905 | 63.8166 | 13.2156 |
| potsdam | car | Geometry_DoubleRowCoupled | 28.7798 | 28.8667 | 98.9648 | 6.7364 |
| potsdam | clutter | Geometry_DoubleRowCoupled | 6.5327 | 9.4111 | 17.5998 | 8.5920 |
| potsdam | impervious surface | Geometry_GainOnlyCoupled | 61.4072 | 72.4525 | 80.1116 | 34.7980 |
| potsdam | building | Geometry_GainOnlyCoupled | 78.7855 | 81.2205 | 96.3342 | 28.3704 |
| potsdam | low vegetation | Geometry_GainOnlyCoupled | 34.4841 | 93.0757 | 35.3920 | 7.9886 |
| potsdam | tree | Geometry_GainOnlyCoupled | 55.3114 | 82.7579 | 62.5156 | 12.8731 |
| potsdam | car | Geometry_GainOnlyCoupled | 27.9577 | 28.0365 | 99.0043 | 6.9386 |
| potsdam | clutter | Geometry_GainOnlyCoupled | 6.3892 | 9.0606 | 17.8106 | 9.0313 |
| potsdam | impervious surface | SCLIP_Coupled | 62.7376 | 73.5266 | 81.0446 | 34.6890 |
| potsdam | building | SCLIP_Coupled | 77.8507 | 79.7518 | 97.0290 | 29.1013 |
| potsdam | low vegetation | SCLIP_Coupled | 36.8971 | 92.3868 | 38.0541 | 8.6535 |
| potsdam | tree | SCLIP_Coupled | 58.2877 | 81.7349 | 67.0170 | 13.9727 |
| potsdam | car | SCLIP_Coupled | 31.9433 | 32.0562 | 98.9099 | 6.0627 |
| potsdam | clutter | SCLIP_Coupled | 5.9946 | 9.1105 | 14.9133 | 7.5207 |
| udd5 | vegetation | NoAdmission_Exact | 66.3933 | 96.7293 | 67.9181 | 20.7968 |
| udd5 | building | NoAdmission_Exact | 80.9728 | 81.9480 | 98.5516 | 47.2128 |
| udd5 | road | NoAdmission_Exact | 36.8085 | 65.8275 | 45.5034 | 9.2698 |
| udd5 | vehicle | NoAdmission_Exact | 17.2031 | 17.8933 | 81.6859 | 3.6544 |
| udd5 | other | NoAdmission_Exact | 33.6700 | 47.5315 | 53.5866 | 19.0662 |
| udd5 | vegetation | Geometry_ConservativeCoupled | 67.5871 | 96.7357 | 69.1645 | 21.1771 |
| udd5 | building | Geometry_ConservativeCoupled | 81.7482 | 82.7558 | 98.5324 | 46.7428 |
| udd5 | road | Geometry_ConservativeCoupled | 37.9550 | 66.2425 | 47.0567 | 9.5262 |
| udd5 | vehicle | Geometry_ConservativeCoupled | 16.8533 | 17.5156 | 81.6765 | 3.7327 |
| udd5 | other | Geometry_ConservativeCoupled | 33.9707 | 48.1413 | 53.5764 | 18.8212 |
| udd5 | vegetation | Geometry_DoubleRowCoupled | 67.3117 | 96.6236 | 68.9331 | 21.1307 |
| udd5 | building | Geometry_DoubleRowCoupled | 81.1887 | 82.1201 | 98.6223 | 47.1477 |
| udd5 | road | Geometry_DoubleRowCoupled | 37.3828 | 65.6393 | 46.4781 | 9.4955 |
| udd5 | vehicle | Geometry_DoubleRowCoupled | 17.7316 | 18.5617 | 79.8589 | 3.4440 |
| udd5 | other | Geometry_DoubleRowCoupled | 33.6961 | 47.8973 | 53.1943 | 18.7821 |
| udd5 | vegetation | Geometry_GainOnlyCoupled | 66.8673 | 96.6748 | 68.4413 | 20.9688 |
| udd5 | building | Geometry_GainOnlyCoupled | 81.0711 | 82.0207 | 98.5920 | 47.1903 |
| udd5 | road | Geometry_GainOnlyCoupled | 37.1641 | 65.7477 | 46.0872 | 9.4001 |
| udd5 | vehicle | Geometry_GainOnlyCoupled | 17.5490 | 18.3249 | 80.5620 | 3.5192 |
| udd5 | other | Geometry_GainOnlyCoupled | 33.6583 | 47.6900 | 53.3572 | 18.9215 |
| udd5 | vegetation | SCLIP_Coupled | 66.0396 | 96.6488 | 67.5872 | 20.7128 |
| udd5 | building | SCLIP_Coupled | 80.3869 | 81.2347 | 98.7183 | 47.7080 |
| udd5 | road | SCLIP_Coupled | 36.5998 | 65.1202 | 45.5241 | 9.3748 |
| udd5 | vehicle | SCLIP_Coupled | 17.8845 | 18.8274 | 78.1225 | 3.3216 |
| udd5 | other | SCLIP_Coupled | 32.6271 | 46.6333 | 52.0686 | 18.8830 |
| oem | bareland | NoAdmission_Exact | 10.0142 | 10.2866 | 79.0896 | 9.9228 |
| oem | rangeland | NoAdmission_Exact | 17.6999 | 66.9109 | 19.3978 | 6.1037 |
| oem | developed space | NoAdmission_Exact | 28.0239 | 32.7422 | 66.0406 | 39.9548 |
| oem | road | NoAdmission_Exact | 33.4645 | 50.0977 | 50.1971 | 7.0772 |
| oem | tree | NoAdmission_Exact | 36.0697 | 89.6185 | 37.6425 | 7.8920 |
| oem | water | NoAdmission_Exact | 67.8635 | 77.7888 | 84.1741 | 2.5543 |
| oem | agriculture land | NoAdmission_Exact | 61.5442 | 65.5575 | 90.9529 | 16.4072 |
| oem | building | NoAdmission_Exact | 43.0029 | 83.1526 | 47.1072 | 10.0881 |
| oem | bareland | Geometry_ConservativeCoupled | 9.9432 | 10.2170 | 78.7706 | 9.9501 |
| oem | rangeland | Geometry_ConservativeCoupled | 18.7935 | 68.0900 | 20.6086 | 6.3724 |
| oem | developed space | Geometry_ConservativeCoupled | 28.2521 | 33.5137 | 64.2795 | 37.9940 |
| oem | road | Geometry_ConservativeCoupled | 34.4958 | 49.8376 | 52.8433 | 7.4891 |
| oem | tree | Geometry_ConservativeCoupled | 38.0082 | 88.6724 | 39.9479 | 8.4647 |
| oem | water | Geometry_ConservativeCoupled | 68.1138 | 77.2683 | 85.1833 | 2.6023 |
| oem | agriculture land | Geometry_ConservativeCoupled | 62.0204 | 66.2877 | 90.5964 | 16.1629 |
| oem | building | Geometry_ConservativeCoupled | 45.8845 | 82.5344 | 50.8190 | 10.9645 |
| oem | bareland | Geometry_DoubleRowCoupled | 10.4158 | 10.7188 | 78.6514 | 9.4698 |
| oem | rangeland | Geometry_DoubleRowCoupled | 18.3186 | 67.2421 | 20.1136 | 6.2978 |
| oem | developed space | Geometry_DoubleRowCoupled | 28.3954 | 33.3359 | 65.7064 | 39.0447 |
| oem | road | Geometry_DoubleRowCoupled | 33.7927 | 50.5522 | 50.4778 | 7.0528 |
| oem | tree | Geometry_DoubleRowCoupled | 36.0498 | 88.9878 | 37.7331 | 7.9671 |
| oem | water | Geometry_DoubleRowCoupled | 68.1308 | 78.5977 | 83.6496 | 2.5122 |
| oem | agriculture land | Geometry_DoubleRowCoupled | 60.5755 | 64.1686 | 91.5385 | 16.8703 |
| oem | building | Geometry_DoubleRowCoupled | 45.1583 | 82.4733 | 49.9521 | 10.7854 |
| oem | bareland | Geometry_GainOnlyCoupled | 10.3175 | 10.6119 | 78.8087 | 9.5844 |
| oem | rangeland | Geometry_GainOnlyCoupled | 18.1115 | 67.0852 | 19.8779 | 6.2385 |
| oem | developed space | Geometry_GainOnlyCoupled | 28.2708 | 33.1136 | 65.9064 | 39.4265 |
| oem | road | Geometry_GainOnlyCoupled | 33.6626 | 50.4011 | 50.3379 | 7.0543 |
| oem | tree | Geometry_GainOnlyCoupled | 35.9513 | 89.2310 | 37.5820 | 7.9135 |
| oem | water | Geometry_GainOnlyCoupled | 68.0534 | 78.4037 | 83.7532 | 2.5216 |
| oem | agriculture land | Geometry_GainOnlyCoupled | 60.9099 | 64.6379 | 91.3502 | 16.7133 |
| oem | building | Geometry_GainOnlyCoupled | 44.4591 | 82.7336 | 49.0061 | 10.5479 |
| oem | bareland | SCLIP_Coupled | 10.1246 | 10.4054 | 78.9594 | 9.7933 |
| oem | rangeland | SCLIP_Coupled | 17.3615 | 67.3565 | 18.9566 | 5.9254 |
| oem | developed space | SCLIP_Coupled | 27.9050 | 32.9404 | 64.6080 | 38.8530 |
| oem | road | SCLIP_Coupled | 32.4560 | 50.7496 | 47.3791 | 6.5941 |
| oem | tree | SCLIP_Coupled | 34.8510 | 89.6556 | 36.3111 | 7.6097 |
| oem | water | SCLIP_Coupled | 67.9855 | 78.8534 | 83.1445 | 2.4889 |
| oem | agriculture land | SCLIP_Coupled | 56.8213 | 59.2043 | 93.3850 | 18.6537 |
| oem | building | SCLIP_Coupled | 41.7847 | 81.5231 | 46.1558 | 10.0819 |

## Matched Complete-Image Timing

| Domain | VIP20 ms | Coupled ms | Primary ms | Primary/VIP |
| --- | ---: | ---: | ---: | ---: |
| vdd | 124.71 | 317.52 | 334.52 | 2.682x |

Three fixed COMPLETE images/domain, three warmed synchronized rotated singleton repeats. All GPUs idle before timing launch; no overlapping workers. Includes original-size restoration, stitching and argmax; excludes initialization, text/image loading and masks. Shared-resident memory is not standalone deployment memory. This is not full-domain average throughput.

## Decision

Do not claim a final model from this four-domain pilot. The requested all-eight accuracy/latency objective still needs full evidence. A published SCLIP control or simple amplitude gain is not automatically promoted into our innovation.
