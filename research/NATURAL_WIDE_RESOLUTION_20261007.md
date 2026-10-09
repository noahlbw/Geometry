# Natural wide-view scale: one frozen task prior

Natural short-edge336/max-long672, crop336/overlap224(actual stride112), at most4 wide encodings. RS stays unchanged. All words/profiles/residuals/weights/coupling frozen; no fine views or envelope. Geometry observations shared, but natural comparison encodes both wide resolutions; paired wall time is not deployed latency.
The scale follows the official crop size and retained four-call cap. Not an unconstrained VIP official max-long2048 pipeline. Previous target-label development persists; exploratory full evaluation.

| Protocol | Images | Frozen | Natural short-edge | Delta |
| --- | ---: | ---: | ---: | ---: |
| vdd | 80 | 55.2382 | 55.2382 | +0.0000 |
| potsdam | 504 | 49.7888 | 49.7888 | +0.0000 |
| voc21 | 1449 | 70.3179 | 70.4508 | +0.1329 |
| context60 | 5105 | 41.5558 | 41.6004 | +0.0446 |
| ade150 | 2000 | 31.1076 | 31.1862 | +0.0786 |

All9138 images completed with exact frozen confusion replay, unique full coverage, paired scored targets and per-image confusion sums. The single declared natural-task scale prior is nondegrading across the five developed protocols, but its gains are small. It remains a provisional candidate until deployment-only latency/encoding checks; the default retained API policy has not changed.

Scale-stratum audit rejects a universal "more pixels is better" explanation: VOC gains mainly on reduced-resolution views, while PC60/ADE gains mainly on increased views. See `NATURAL_WIDE_RESOLUTION_STRATA_20261007.md`. Do not select the winning direction separately by target labels.
