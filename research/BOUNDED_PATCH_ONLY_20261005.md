# Bounded Geometry patch-only route: fixed full-domain pilot

Same bounded896/448 views, fixed20 words/class, checkpoints, original Geometry H and wide class calibration. Patch queries read original G at strength1 or2; prefix queries remain native. No added encoder, fine view, learned operator, threshold or domain routing. Strength and key-group route are configuration choices, not a new innovation claim. SCLIP is a published matched control. These are previously developed domains, not independent validation.

| Domain | VIP20 | Official/distilled VIP | Original coupled | Patch-only1 | Patch-only2 | SCLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 51.4827 | 52.0647 | 53.8090 | 53.5838 | 53.4600 | 53.5257 |
| potsdam | 42.3028 | 44.8995 | 43.4301 | 44.5000 | 45.9339 | 45.6185 |
| udd5 | 41.4336 | 44.9713 | 47.0096 | 47.0654 | 47.0786 | 46.7075 |
| oem | 33.2064 | 35.0388 | 37.2104 | 37.8737 | 36.8400 | 36.1612 |

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
| vdd | other | Geometry_PatchOnlyCoupled | 32.7650 | 53.5356 | 45.7850 | 16.0878 |
| vdd | wall | Geometry_PatchOnlyCoupled | 45.0761 | 71.6048 | 54.8873 | 2.4801 |
| vdd | road | Geometry_PatchOnlyCoupled | 41.0323 | 42.0198 | 94.5827 | 12.3556 |
| vdd | vegetation | Geometry_PatchOnlyCoupled | 67.9454 | 96.5158 | 69.6539 | 25.7748 |
| vdd | vehicle | Geometry_PatchOnlyCoupled | 21.5497 | 23.1816 | 75.3768 | 1.6929 |
| vdd | roof | Geometry_PatchOnlyCoupled | 81.9146 | 82.5673 | 99.0441 | 25.2408 |
| vdd | water | Geometry_PatchOnlyCoupled | 84.8038 | 88.4658 | 95.3460 | 16.3679 |
| vdd | other | Geometry_PatchOnly2Coupled | 32.7096 | 53.4974 | 45.7046 | 16.0711 |
| vdd | wall | Geometry_PatchOnly2Coupled | 43.4422 | 71.6955 | 52.4351 | 2.3663 |
| vdd | road | Geometry_PatchOnly2Coupled | 40.6842 | 41.6059 | 94.8360 | 12.5120 |
| vdd | vegetation | Geometry_PatchOnly2Coupled | 68.1015 | 96.4819 | 69.8357 | 25.8512 |
| vdd | vehicle | Geometry_PatchOnly2Coupled | 22.7793 | 24.9941 | 71.9936 | 1.4996 |
| vdd | roof | Geometry_PatchOnly2Coupled | 81.4826 | 82.1567 | 99.0030 | 25.3564 |
| vdd | water | Geometry_PatchOnly2Coupled | 85.0207 | 88.6520 | 95.4036 | 16.3434 |
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
| potsdam | impervious surface | Geometry_PatchOnlyCoupled | 61.6735 | 74.5142 | 78.1607 | 33.0112 |
| potsdam | building | Geometry_PatchOnlyCoupled | 78.4107 | 80.4402 | 96.8826 | 28.8087 |
| potsdam | low vegetation | Geometry_PatchOnlyCoupled | 38.3805 | 92.1931 | 39.6698 | 9.0399 |
| potsdam | tree | Geometry_PatchOnlyCoupled | 57.2066 | 82.8889 | 64.8670 | 13.3361 |
| potsdam | car | Geometry_PatchOnlyCoupled | 24.8875 | 24.9324 | 99.2817 | 7.8243 |
| potsdam | clutter | Geometry_PatchOnlyCoupled | 6.4411 | 9.5354 | 16.5615 | 7.9797 |
| potsdam | impervious surface | Geometry_PatchOnly2Coupled | 62.7987 | 73.9621 | 80.6228 | 34.3053 |
| potsdam | building | Geometry_PatchOnly2Coupled | 79.0175 | 81.0699 | 96.8956 | 28.5888 |
| potsdam | low vegetation | Geometry_PatchOnly2Coupled | 38.5018 | 92.0555 | 39.8250 | 9.0888 |
| potsdam | tree | Geometry_PatchOnly2Coupled | 58.5794 | 81.9854 | 67.2335 | 13.9750 |
| potsdam | car | Geometry_PatchOnly2Coupled | 30.1233 | 30.2159 | 98.9927 | 6.4374 |
| potsdam | clutter | Geometry_PatchOnly2Coupled | 6.5825 | 9.9072 | 16.3986 | 7.6047 |
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
| udd5 | vegetation | Geometry_PatchOnlyCoupled | 67.1655 | 96.5814 | 68.8012 | 21.0995 |
| udd5 | building | Geometry_PatchOnlyCoupled | 81.0641 | 81.9730 | 98.6507 | 47.2458 |
| udd5 | road | Geometry_PatchOnlyCoupled | 37.9035 | 66.6524 | 46.7736 | 9.4106 |
| udd5 | vehicle | Geometry_PatchOnlyCoupled | 15.9502 | 16.5240 | 82.1212 | 3.9783 |
| udd5 | other | Geometry_PatchOnlyCoupled | 33.2438 | 48.0499 | 51.8966 | 18.2657 |
| udd5 | vegetation | Geometry_PatchOnly2Coupled | 66.2953 | 96.5410 | 67.9084 | 20.8344 |
| udd5 | building | Geometry_PatchOnly2Coupled | 81.2771 | 82.2868 | 98.5127 | 46.9999 |
| udd5 | road | Geometry_PatchOnly2Coupled | 37.2416 | 65.9709 | 46.0967 | 9.3702 |
| udd5 | vehicle | Geometry_PatchOnly2Coupled | 16.9950 | 17.8319 | 78.3585 | 3.5176 |
| udd5 | other | Geometry_PatchOnly2Coupled | 33.5843 | 47.1961 | 53.7992 | 19.2779 |
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
| oem | bareland | Geometry_PatchOnlyCoupled | 9.8971 | 10.1640 | 79.0340 | 10.0354 |
| oem | rangeland | Geometry_PatchOnlyCoupled | 18.9754 | 68.3469 | 20.8036 | 6.4085 |
| oem | developed space | Geometry_PatchOnlyCoupled | 28.2872 | 33.9073 | 63.0539 | 36.8371 |
| oem | road | Geometry_PatchOnlyCoupled | 33.8037 | 49.9409 | 51.1277 | 7.2310 |
| oem | tree | Geometry_PatchOnlyCoupled | 38.0960 | 89.0420 | 39.9700 | 8.4342 |
| oem | water | Geometry_PatchOnlyCoupled | 67.8326 | 78.6530 | 83.1386 | 2.4951 |
| oem | agriculture land | Geometry_PatchOnlyCoupled | 59.8642 | 63.1554 | 91.9920 | 17.2258 |
| oem | building | Geometry_PatchOnlyCoupled | 46.2332 | 81.2946 | 51.7370 | 11.3328 |
| oem | bareland | Geometry_PatchOnly2Coupled | 9.9823 | 10.2547 | 78.9879 | 9.9408 |
| oem | rangeland | Geometry_PatchOnly2Coupled | 17.7049 | 68.4987 | 19.2742 | 5.9242 |
| oem | developed space | Geometry_PatchOnly2Coupled | 27.9752 | 33.3700 | 63.3759 | 37.6213 |
| oem | road | Geometry_PatchOnly2Coupled | 33.0407 | 50.3852 | 48.9750 | 6.8655 |
| oem | tree | Geometry_PatchOnly2Coupled | 36.4308 | 88.9660 | 38.1548 | 8.0581 |
| oem | water | Geometry_PatchOnly2Coupled | 67.9548 | 78.9922 | 82.9449 | 2.4786 |
| oem | agriculture land | Geometry_PatchOnly2Coupled | 57.4689 | 60.0952 | 92.9327 | 18.2881 |
| oem | building | Geometry_PatchOnly2Coupled | 44.1626 | 81.0347 | 49.2533 | 10.8234 |
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
| vdd | 127.47 | 323.23 | 333.62 | 2.617x |

Three fixed COMPLETE images/domain, three warmed synchronized rotated singleton repeats. All GPUs idle before timing launch; no overlapping workers. Includes original-size restoration, stitching and argmax; excludes initialization, text/image loading and masks. Shared-resident memory is not standalone deployment memory. This is not full-domain average throughput.

## Decision

Do not claim a final model from this four-domain pilot. The requested all-eight accuracy/latency objective still needs full evidence. A published SCLIP control or simple amplitude gain is not automatically promoted into our innovation.
