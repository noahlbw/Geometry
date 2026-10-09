# PC60 residual-ontology full comparison

Foreground words, automatic readout profile and visual observations are unchanged. Only residual semantics/aggregation change. Full459 category names provide additional ontology information; no masks select the401 residual queries. No fitted background threshold. Developed protocol, exploratory results.

| Method | mIoU | Foreground mIoU | Background IoU | Precision | Recall | Predicted area | GT area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaxonomyAdaptive | 40.2354 | 40.9162 | 0.0701 | 37.5498 | 0.0701 | 0.0156 | 8.3252 |
| ResidualMean | 40.2320 | 40.9139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8.3252 |
| ResidualMax | 41.4029 | 41.7416 | 21.4163 | 34.2319 | 36.3888 | 8.8497 | 8.3252 |

## Nondevelopment complement

5041 images; developed domain, not untouched model selection.

| Method | mIoU | Background IoU |
| --- | ---: | ---: |
| TaxonomyAdaptive | 40.2366 | 0.0706 |
| ResidualMean | 40.2331 | 0.0000 |
| ResidualMax | 41.3961 | 21.4685 |

## Per-class IoU

| Class | Automatic | Residual mean | Residual max |
| --- | ---: | ---: | ---: |
| background | 0.0701 | 0.0000 | 21.4163 |
| aeroplane | 44.0004 | 44.0005 | 46.9412 |
| bag | 29.0784 | 29.0774 | 28.9965 |
| bed | 10.5073 | 10.5073 | 10.7126 |
| bedclothes | 29.1640 | 29.1639 | 29.7124 |
| bench | 15.5909 | 15.5877 | 19.2097 |
| bicycle | 65.3354 | 65.3355 | 65.5787 |
| bird | 47.4603 | 47.4593 | 51.5031 |
| boat | 52.9120 | 52.9120 | 55.6676 |
| book | 9.3387 | 9.3387 | 8.7705 |
| bottle | 63.9968 | 64.0033 | 66.2459 |
| building | 38.6137 | 38.6112 | 36.5945 |
| bus | 72.8009 | 72.8011 | 74.0634 |
| cabinet | 38.8309 | 38.8253 | 37.6352 |
| car | 66.7816 | 66.7807 | 67.0307 |
| cat | 74.7236 | 74.7234 | 76.2376 |
| ceiling | 43.0718 | 43.0717 | 44.0814 |
| chair | 42.6595 | 42.6577 | 44.8152 |
| cloth | 18.4231 | 18.3875 | 16.9463 |
| computer | 13.0613 | 13.0551 | 16.5520 |
| cow | 78.1062 | 78.1062 | 78.9757 |
| cup | 30.5911 | 30.5883 | 26.0866 |
| curtain | 45.6036 | 45.6036 | 50.4056 |
| dog | 61.6250 | 61.6248 | 63.6615 |
| door | 26.8036 | 26.7977 | 26.7215 |
| fence | 29.5518 | 29.5509 | 29.8207 |
| floor | 44.6993 | 44.6846 | 47.6102 |
| flower | 19.7180 | 19.7031 | 16.4680 |
| food | 35.1602 | 35.1528 | 25.8697 |
| grass | 68.3456 | 68.3451 | 67.4160 |
| ground | 4.6345 | 4.6487 | 4.2273 |
| horse | 78.5004 | 78.5003 | 78.2593 |
| keyboard | 29.4418 | 29.4414 | 27.4717 |
| light | 12.8069 | 12.8066 | 13.1098 |
| motorbike | 72.1627 | 72.1626 | 71.8762 |
| mountain | 42.3822 | 42.3741 | 43.1297 |
| mouse | 23.2091 | 23.2091 | 42.5704 |
| person | 65.5667 | 65.5569 | 69.2706 |
| plate | 27.9976 | 27.9961 | 37.3232 |
| platform | 6.9664 | 6.9665 | 7.4188 |
| pottedplant | 51.8741 | 51.8925 | 49.7763 |
| road | 41.6723 | 41.6709 | 42.0632 |
| rock | 42.1911 | 42.1904 | 41.3717 |
| sheep | 78.5288 | 78.5287 | 80.6238 |
| shelves | 20.0915 | 20.0872 | 20.8399 |
| sidewalk | 11.6314 | 11.6310 | 12.9596 |
| sign | 36.9910 | 36.9910 | 31.6239 |
| sky | 75.0290 | 75.0288 | 74.8572 |
| snow | 58.0375 | 58.0349 | 58.9797 |
| sofa | 62.4637 | 62.4638 | 62.2184 |
| table | 45.9313 | 45.9295 | 46.5005 |
| track | 0.0002 | 0.0002 | 0.0002 |
| train | 40.4126 | 40.4126 | 42.0160 |
| tree | 58.9155 | 58.9159 | 57.6328 |
| truck | 13.6624 | 13.6612 | 15.3180 |
| tvmonitor | 35.9414 | 35.9136 | 41.9837 |
| wall | 38.5200 | 38.5107 | 38.0101 |
| water | 68.9724 | 68.9720 | 73.0861 |
| window | 32.8990 | 32.8986 | 28.6391 |
| wood | 20.0666 | 20.0663 | 19.2686 |

Verified unique full5105 coverage, per-image confusion sums, unchanged automatic confusion exact replay, checkpoint identity and paired target counts. The max variant is a semantic union, not a normalized synonym mean; sensitivity to residual candidate count remains a mechanism question.

Parallel evaluation wall seconds: 467.17; not single-model inference time.
