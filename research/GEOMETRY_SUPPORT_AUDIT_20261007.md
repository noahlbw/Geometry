# Geometry relation support: fixed40-image diagnostic

Same8 metadata-selected images per task as previous audit. Squared off-diagonal H is normalized into unsigned leave-self-out support; it is not the signed reconstruction. No online labels, gain fitting or model change. Max128x128 grid; small-object coverage and pixel independence are not guaranteed.

| Task | Baseline | Probability evidence AUC | 3x3 support AUC | Geometry support AUC |
| --- | --- | ---: | ---: | ---: |
| vdd | LocalEndpoint | 0.4999 | 0.5039 | 0.4949 |
| vdd | WideEndpoint | 0.6610 | 0.6139 | 0.5465 |
| potsdam | LocalEndpoint | 0.6532 | 0.6722 | 0.7239 |
| potsdam | WideEndpoint | 0.7568 | 0.7498 | 0.7796 |
| voc21 | LocalEndpoint | 0.7184 | 0.7572 | 0.8199 |
| voc21 | WideEndpoint | 0.7083 | 0.6155 | 0.6630 |
| context60 | LocalEndpoint | 0.6664 | 0.5231 | 0.5326 |
| context60 | WideEndpoint | 0.5658 | 0.5957 | 0.7331 |
| ade150 | LocalEndpoint | 0.7361 | 0.5533 | 0.6195 |
| ade150 | WideEndpoint | 0.7494 | 0.6635 | 0.6621 |

AUC values are foreground class means for classes with both fixed and broken outcomes. Prior diagnostic outcomes and scores replay exactly. Do not interpret AUC as improved mIoU or choose task/class-specific score signs from these labels.

## Interpretation and next design constraint

Geometry support improves over3x3 support for local-baseline Potsdam (.6722→.7239) and VOC (.7572→.8199), and for wide-baseline PC (.5957→.7331). This supports studying relation-conditioned evidence rather than generic spatial smoothing. It does not establish a universal gate: VDD local .4949 remains chance-level, and ADE local evidence ratio .7361 is stronger than Geometry support .6195. The direction/baseline matters; one scalar reliability cannot be assumed interchangeable between local preservation and wide preservation.

Keep the selected model unchanged. A subsequent candidate must preserve the task/background semantics, combine complementary evidence under one frozen rule rather than per-dataset AUC winners, and be tested for actual per-class/mIoU effects on all five targets. No benefit is claimed from this diagnostic alone. The repeated cohort is intentionally a matched mechanism comparison, not new independent validation.
