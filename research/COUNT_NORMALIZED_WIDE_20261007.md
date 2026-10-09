# Wide-view query-count normalization: frozen full comparison

## Decision and full-confusion diagnosis

All 9138 protocol images completed with verified full unique coverage, paired target counts and exact Frozen confusion replay. Do not adopt count-normalized wide aggregation: it decreases all three natural protocols and is only an equivalent common gauge on the two equal-count remote-sensing banks. The inherited route remains the default.

The following diagnostics are calculated from the saved full confusion matrices (rows are targets, columns predictions); no additional evaluation or fitted offset was used.

| Protocol / arm | Background IoU | Background precision | Background recall | Predicted background area | Foreground mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| VOC21 Frozen | 89.5149 | 95.5793 | 93.3811 | 71.6318% | 69.3580 |
| VOC21 count-normalized | 14.3866 | 99.8372 | 14.3900 | 10.5677% | 29.8855 |
| PC60 Frozen | 21.2474 | 37.6944 | 32.7488 | 7.2329% | 41.9000 |
| PC60 count-normalized | 14.1611 | 14.7530 | 77.9229 | 43.9720% | 36.0436 |

VOC loses background coverage while PC60 massively expands background. The two opposite directions are consistent with distinct aggregation structures: VOC's 56 background queries mix synonyms with different residual concepts; PC60 uses a separate maximum union of 401 residual queries, while foreground class LSE is count-corrected. The measured test isolates the score correction, not the relative causal contributions of concept coverage and multiplicity within the original vocabulary.

A variable vocabulary therefore needs explicit distinction between same-concept paraphrases and different residual concepts. Neither restoring a dataset-specific offset nor exempting background based on these observed scores would establish a general adaptive solution. No such repair was fitted here.

ADE150 has no scored generic background class in this comparison. Of150 classes,52 improve and98 decline. Largest drops include person61.8811→32.1163, sculpture43.2244→17.2902, van37.6618→17.0564 and water45.7248→26.0737. Gains include cushion32.6346→54.9727 and ashcan13.0143→30.8434. Thus the failure is not confined to background handling; unequal-count foreground calibration also changes class competition. These are diagnostic outcomes, not a rule for selecting exceptions.

Inherited wide LSE minus log(actual class query count)/tau. No new fitted strength, words, thresholds, vision forwards, geometry gain or resolution changes. Old long448 wide policy retained. Equal counts without separate residual replacement retain the identical common gauge. PC60 residual max/protection rule and extra401 concepts remain unchanged; foreground rival assignment can change as an intended consequence of score calibration.
Entire query-group duplication invariance is not invariance to arbitrary correlated paraphrases or wrong concepts. Default deployed model remains inherited until the whole comparison is evaluated. Previously developed targets; exploratory.

| Protocol | Images | Frozen | Count-normalized wide | Delta |
| --- | ---: | ---: | ---: | ---: |
| vdd | 80 | 55.2382 | 55.2382 | +0.0000 |
| potsdam | 504 | 49.7888 | 49.7888 | +0.0000 |
| voc21 | 1449 | 70.3179 | 29.1475 | -41.1704 |
| context60 | 5105 | 41.5558 | 35.6789 | -5.8769 |
| ade150 | 2000 | 31.1076 | 28.8973 | -2.2103 |
