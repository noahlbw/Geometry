# Selected task-conditioned deployment candidate

## Candidate and measured performance

Use `TaxonomyInference.for_task(geometry,vip,banks,queries,family=...,background=...,residual_features=...)` with pre-encoded frozen banks. No dataset-name branch, target labels, fitted online offsets or new visual observations. Historical constructor defaults remain unchanged. The selected entry composes the already evaluated natural scale prior with exactly the retained words/read profiles/residual handling; it does not combine untested winners.

| Protocol | Retained long448 | Selected task candidate | Local finite VIP | Candidate minus VIP |
| --- | ---: | ---: | ---: | ---: |
| VDD | 55.2382 | 55.2382 | 52.0647 | +3.1735 |
| Potsdam | 49.7888 | 49.7888 | 44.8995 | +4.8893 |
| VOC21 | 70.3179 | 70.4508 | 73.2591 | -2.8083 |
| PC60 | 41.5558 | 41.6004 | 42.5987 | -0.9983 |
| ADE150 | 31.1076 | 31.1862 | 29.1387 | +2.0475 |

Candidate metrics come from the verified9138-image `natural_wide_resolution_20261007` paired suite, not a new evaluation of the named constructor. The factory selects identical constructor options. Nine fixed real natural images previously replayed the direct single-policy API probabilities exactly against paired observations. No full9138-image independent factory probability replay is claimed. Full original Frozen confusions, sample coverage, scored targets and checkpoint/class identities verified in that suite.

## Rules and semantic budget

- Natural: canonical-text rank menu retains VOC g2/strength2, PC g1/strength2, ADE g.5/original for these banks. Wide shortedge336/maxlong672, crop336/overlap224(actual stride112), max4 calls. Local alias aggregation stays uniform; no rejected wide competitor gate/count normalization/consistency gain/strength3.
- Remote sensing: existing image canonical margin selects original ImageNet strength1 or focused RS strength3; g.5 and conservative local soft weights unchanged. Wide long448 retained.
- Caller supplies any scored residual index explicitly. Extra residual concepts remain optional; when supplied, protected local max is used with uniform foreground. PC60 reported values use401 extra queries from the official residual ontology. The entry does not invent this ontology for other datasets.
- RS words remain two fixed20 banks; natural words have variable counts. VOC background queries are a union of different concepts, not56 synonyms. Word-count offsets in the inherited wide score are deliberately retained. Qwen generation and a reliable variable-pool normalization are not completed by this entry.

## Speed and scientific scope

Previously measured single-policy timing on GPU1 A800: three fixed-spread images/domain, one warm pass then three synchronized repeats; mean image median181.71ms VOC,235.52ms PC,238.62ms ADE. Includes visual encoding/readout/stitch/restoration/CPU argmax, excludes loading/text/cache/disk. Not full-dataset mean or matched VIP speed. Maximum4local+4wide,0fine. Deployment computes only selected wide view. Detailed timing/checks in `NATURAL_WIDE_PRODUCTION_TIMING_20261007.md`.

Task profiles/menus/words use labelled development provenance. This is training-free inference, not parameter-free or untouched evaluation. VIP comparisons use official queries/settings and explicit empty-row numerical repair; different queries/information budgets prevent a pure readout attribution. This candidate beats the local finite VIP on3/5 protocols, not all five and not certified SOTA. Remaining goal work includes natural foreground/residual competition and evidence beyond developed domains; completing this deployment entry does not complete the research objective.

Sources: `NATURAL_WIDE_RESOLUTION_20261007.md`, `NATURAL_WIDE_PRODUCTION_TIMING_20261007.md`, `FINAL_ADAPTIVE_DEPLOYMENT_20261007.md`, `CONTEXT60_OFFICIAL_FINITE_COMPARISON_20261007.md`. Failed alternatives remain separately archived and opt-in.
