# PC60 conservative local rival protection

All401 residual queries, old foreground banks, automatic readout and wide scores unchanged. Only local residual evidence receives bounded rival-conditioned suppression. Labels motivated this development candidate; online weights use no masks.

| Method | mIoU | Foreground mIoU | Background IoU | Background precision | Background recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| TaxonomyAdaptive | 40.2354 | 40.9162 | 0.0701 | 37.5498 | 0.0701 |
| ResidualMax | 41.4029 | 41.7416 | 21.4163 | 34.2319 | 36.3888 |
| ResidualProtected | 41.5558 | 41.9000 | 21.2474 | 37.6944 | 32.7488 |

## Per-class IoU

| Class | Automatic | Residual max | Protected |
| --- | ---: | ---: | ---: |
| background | 0.0701 | 21.4163 | 21.2474 |
| aeroplane | 44.0004 | 46.9412 | 46.4670 |
| bag | 29.0784 | 28.9965 | 29.4926 |
| bed | 10.5073 | 10.7126 | 10.7273 |
| bedclothes | 29.1640 | 29.7124 | 29.6723 |
| bench | 15.5909 | 19.2097 | 18.6901 |
| bicycle | 65.3354 | 65.5787 | 66.0382 |
| bird | 47.4603 | 51.5031 | 51.4757 |
| boat | 52.9120 | 55.6676 | 55.2883 |
| book | 9.3387 | 8.7705 | 9.0575 |
| bottle | 63.9968 | 66.2459 | 67.0179 |
| building | 38.6137 | 36.5945 | 37.2274 |
| bus | 72.8009 | 74.0634 | 73.9217 |
| cabinet | 38.8309 | 37.6352 | 38.6180 |
| car | 66.7816 | 67.0307 | 67.0957 |
| cat | 74.7236 | 76.2376 | 76.1993 |
| ceiling | 43.0718 | 44.0814 | 44.0481 |
| chair | 42.6595 | 44.8152 | 44.5302 |
| cloth | 18.4231 | 16.9463 | 18.0743 |
| computer | 13.0613 | 16.5520 | 15.7693 |
| cow | 78.1062 | 78.9757 | 78.8427 |
| cup | 30.5911 | 26.0866 | 27.4814 |
| curtain | 45.6036 | 50.4056 | 50.2271 |
| dog | 61.6250 | 63.6615 | 63.4629 |
| door | 26.8036 | 26.7215 | 26.7679 |
| fence | 29.5518 | 29.8207 | 30.0987 |
| floor | 44.6993 | 47.6102 | 46.9081 |
| flower | 19.7180 | 16.4680 | 17.9637 |
| food | 35.1602 | 25.8697 | 26.6261 |
| grass | 68.3456 | 67.4160 | 67.5040 |
| ground | 4.6345 | 4.2273 | 4.4180 |
| horse | 78.5004 | 78.2593 | 78.6403 |
| keyboard | 29.4418 | 27.4717 | 27.4110 |
| light | 12.8069 | 13.1098 | 13.1666 |
| motorbike | 72.1627 | 71.8762 | 72.2429 |
| mountain | 42.3822 | 43.1297 | 43.0664 |
| mouse | 23.2091 | 42.5704 | 43.4396 |
| person | 65.5667 | 69.2706 | 69.1043 |
| plate | 27.9976 | 37.3232 | 36.4517 |
| platform | 6.9664 | 7.4188 | 7.3230 |
| pottedplant | 51.8741 | 49.7763 | 50.6789 |
| road | 41.6723 | 42.0632 | 42.1347 |
| rock | 42.1911 | 41.3717 | 41.5261 |
| sheep | 78.5288 | 80.6238 | 80.6865 |
| shelves | 20.0915 | 20.8399 | 20.7407 |
| sidewalk | 11.6314 | 12.9596 | 12.6381 |
| sign | 36.9910 | 31.6239 | 33.1373 |
| sky | 75.0290 | 74.8572 | 74.9265 |
| snow | 58.0375 | 58.9797 | 58.8632 |
| sofa | 62.4637 | 62.2184 | 62.4811 |
| table | 45.9313 | 46.5005 | 46.6909 |
| track | 0.0002 | 0.0002 | 0.0002 |
| train | 40.4126 | 42.0160 | 41.7181 |
| tree | 58.9155 | 57.6328 | 58.0406 |
| truck | 13.6624 | 15.3180 | 15.7638 |
| tvmonitor | 35.9414 | 41.9837 | 41.5370 |
| wall | 38.5200 | 38.0101 | 38.1443 |
| water | 68.9724 | 73.0861 | 72.6620 |
| window | 32.8990 | 28.6391 | 29.8165 |
| wood | 20.0666 | 19.2686 | 19.3589 |

Full5105 unique coverage, frozen weights, per-image confusion sums, paired target counts and exact replay of both unchanged arms verified. The0.5 floor is a fixed development prior, not a parameter-free theorem. The score bound does not guarantee IoU.
