# Qwen variable candidate pool: generation outcome

One frozen text-only generation completed239/240 category schema checks. Failed raw output is preserved; derivative uses canonical-only fallback for any invalid class. No images, masks or score-based selection. No segmentation performance result yet. Semantically unsafe phrases remain in the structurally accepted pool.

| Task | Generated foreground groups | Foreground alias min/max | Mean foreground count | Retained background count |
| --- | ---: | --- | ---: | ---: |
| vdd | 6 | 9/11 | 9.50 | 20 |
| potsdam | 5 | 9/9 | 9.00 | 20 |
| voc21 | 20 | 1/13 | 8.40 | 56 |
| context60 | 59 | 2/13 | 10.36 | 1 |
| ade150 | 150 | 1/13 | 9.68 | 0 |

Format fallback: [{"dataset": "voc21", "index": 2, "name": "bicycle"}]

The generated count range does not prove meaningful adaptation. Subsequent admission must distinguish legitimate synonyms/subtypes from competitor, part and mixed-scene queries. Do not run the raw pool as a claimed clean vocabulary or silently repair selected class lists. Existing best model remains unchanged.
