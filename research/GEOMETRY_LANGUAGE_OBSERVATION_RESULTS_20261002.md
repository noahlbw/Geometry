# Geometry position-language observation: verified diagnostic

Signature `geometry-position-language-observation-v1-20261002`. All four A800 workers on physical GPUs4-7 completed. Original Geometry and all prior models remain unchanged.

This uses an additional frozen Qwen2.5-VL-7B-Instruct, pinned author revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. About16.6GB of source assets, original 20 aliases/class, local RGB inference and no uploaded images. No target-label fitting. Its pretrained instruction/localization supervision and compute are extra, not a matched single-backbone VIP comparison or an originality claim.

## Scope and computation

Each dataset uses its earlier eight development images/all16 saved windows. Query centers are sampled from original predicted-class margin ranks. Geometry's95% support defines a64..512-pixel foveal crop. Original context and foveal RGB each receive a hollow location marker and forward/reverse class-option questions. All four must agree on a non-abstaining class for the primary to replace Geometry.

Metrics describe exactly the common selected query centers, not full-image or full-patch mIoU. The sampling is not a representative prevalence estimate; overlapping source windows correlate observations. Abstentions are counted as FN rather than discarded. Cached labels were resident but never passed to the observer, and observations were saved before those labels were consulted for metrics.

## Selected-query mIoU

| Dataset | Queries | Geometry | LanguageContext | LanguageFoveal | LanguageGrounded | GroundedConsensus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 268 | 21.398155 | 24.521277 | 25.216331 | 21.186703 | 22.510969 |
| potsdam | 304 | 30.420924 | 15.466160 | 20.755698 | 18.587075 | 30.059465 |

## vdd changes

Primary query mIoU delta: +1.112815pp. Diagnostic gate passed: True.

| Method | Beneficial | Harmful | Wrong-to-wrong | Abstentions |
| --- | ---: | ---: | ---: | ---: |
| LanguageContext | 29 | 51 | 131 | 97 |
| LanguageFoveal | 79 | 34 | 78 | 79 |
| LanguageGrounded | 50 | 41 | 114 | 113 |
| GroundedConsensus | 11 | 2 | 10 | 0 |

| Class | Geometry IoU | Primary IoU | Geometry TP/FP/FN | Primary TP/FP/FN |
| --- | ---: | ---: | --- | --- |
| other | 8.737864 | 8.080808 | 9/37/57 | 8/33/58 |
| wall | 6.521739 | 4.545455 | 3/43/0 | 2/41/1 |
| road | 36.842105 | 40.000000 | 14/22/2 | 14/19/2 |
| vegetation | 28.965517 | 34.868421 | 42/2/101 | 53/9/90 |
| vehicle | 4.878049 | 5.714286 | 2/39/0 | 2/33/0 |
| roof | 30.508475 | 31.034483 | 18/29/12 | 18/28/12 |
| water | 33.333333 | 33.333333 | 4/4/4 | 4/4/4 |

| Frozen viability check | Passed |
| --- | --- |
| query_miou_retained | True |
| beneficial_exceeds_harmful | True |
| small_class_fp_falls | True |
| small_class_tp_retained | True |
| coverage_class_tp_retained | True |
| focus_classes_observed | True |

## potsdam changes

Primary query mIoU delta: -0.361460pp. Diagnostic gate passed: False.

| Method | Beneficial | Harmful | Wrong-to-wrong | Abstentions |
| --- | ---: | ---: | ---: | ---: |
| LanguageContext | 33 | 99 | 105 | 127 |
| LanguageFoveal | 35 | 85 | 107 | 118 |
| LanguageGrounded | 31 | 93 | 107 | 149 |
| GroundedConsensus | 7 | 6 | 3 | 0 |

| Class | Geometry IoU | Primary IoU | Geometry TP/FP/FN | Primary TP/FP/FN |
| --- | ---: | ---: | --- | --- |
| impervious surface | 44.897959 | 46.601942 | 44/16/38 | 48/21/34 |
| building | 42.500000 | 38.888889 | 17/17/6 | 14/13/9 |
| low vegetation | 37.634409 | 38.709677 | 35/5/53 | 36/5/52 |
| tree | 45.360825 | 45.360825 | 44/8/45 | 44/8/45 |
| car | 6.250000 | 6.250000 | 4/60/0 | 4/60/0 |
| clutter | 5.882353 | 4.545455 | 4/50/14 | 3/48/15 |

| Frozen viability check | Passed |
| --- | --- |
| query_miou_retained | False |
| beneficial_exceeds_harmful | True |
| small_class_fp_falls | False |
| small_class_tp_retained | True |
| coverage_class_tp_retained | True |
| focus_classes_observed | True |

## Observation stability and cost

| Dataset | Non-abstaining unanimity | Any unanimity | Mean legal-answer token mass | Mean crop pixels | Max shard seconds | Sum shard seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd | 0.216418 | 0.320896 | 0.995721 | 314.985 | 64.942 | 127.292 | 16011.200 |
| potsdam | 0.157895 | 0.276316 | 0.993434 | 291.263 | 70.639 | 136.504 | 16003.876 |

| Dataset | Context order agreement | Foveal order agreement | View agreement forward | View agreement reverse |
| --- | ---: | ---: | ---: | ---: |
| vdd | 0.779851 | 0.847015 | 0.473881 | 0.421642 |
| potsdam | 0.595395 | 0.634868 | 0.516447 | 0.526316 |

## Verification and decision

All32 source windows and unique query identities are complete. Original Geometry query predictions/sampling and every raw language score-derived prediction were reconstructed against original caches on the same CUDA arithmetic path. CPU log-softmax reduction at tied averaged control scores can change argmax; the initial CPU audit caught this and no performance outputs were overwritten. The exact CUDA audit passed. No raw model scores, source input or primary rule was changed. Local downloads preserve all raw four-way observations.

The combined predeclared viability gate passed: False. No full eight-domain rollout is supported by a failed gate. Passing one domain or correcting more pixels than harmed does not establish positive mIoU across domains. Whole-class performance must retain correct coverage as well as reject false occupation.

This rejects the tested marked-location question/two-view unanimity design, not every language model or Geometry coupling. Direct language observations are especially inferior on selected Potsdam queries. A stronger global model does not automatically provide a reliable pixel-local semantic observation. It remains necessary to validate actual spatial identity and transferable positive class competition before another dense reconstruction or fusion. The eight-domain improvement/CVPR goal remains unachieved.
