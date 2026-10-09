# Task-conditioned candidate with explicit local background union

## Entry and measured full results

Use `TaxonomyInference.for_task(...,family=...,background=...,residual_features=...,local_background='union_max')` with the existing frozen banks. Historical constructor and factory defaults stay retained. This option describes taxonomy semantics, not a dataset-name winner table. It applies only to natural multiquery background without an extra residual ontology; foreground, inherited wide aggregation, Geometry and coupled reconstruction are unchanged. PC protected residual behavior is preserved. No new threshold, encoder, fine observation or fitted scalar.

| Protocol | Retained selected task | Union candidate | Local finite VIP | Candidate minus VIP |
| --- | ---: | ---: | ---: | ---: |
| VDD80 |55.2382|55.2382|52.0647|+3.1735|
| Potsdam504 |49.7888|49.7888|44.8995|+4.8893|
| VOC21 1449 |70.4508|71.0273|73.2591|-2.2318|
| PC60 5105 |41.6004|41.6004|42.5987|-0.9983|
| ADE150 2000 |31.1862|31.1862|29.1387|+2.0475|

9138 full protocol images verified: unique coverage, same scored target counts, original Frozen confusion replay. VDD/Potsdam/PC60/ADE150 candidate confusions exactly equal their controls. Twenty of21 VOC class IoUs improve; train declines -0.0573pp. Full values come from the constructor-based `local_background_union_20261007` suite, not a separate full factory rerun. Five factory-entry unit tests passed after forwarding the new option. The pre-evaluation real-image checks used the constructor, not a separate full independent factory replay.

## Profiles retained

Natural text-rank menu retains VOC g2/strength2, PC g1/strength2, ADE g.5/original. Natural wide short336/maxlong672, maximum4 calls. RS original-image canonical margin selects original ImageNet/strength1 or focused RS/strength3, g.5, wide long448. Maximum4local+4wide,0fine. No new image encodings; candidate production latency has not yet been measured, and historical speed figures are not relabelled as new measurements.

Local background max represents a disjunction of supplied concepts, not synonym averaging. It does not make unlimited wrong queries safe. Foreground natural synonym aggregation remains uniform; RS local soft weighting remains retained. Wide query-count priors remain inherited. This is not the rejected semantic-mass grouping, not a new foreground competitor gate and not Qwen-generated vocabulary evaluation.

## Research interpretation

The local background intervention improves the current VOC protocol without lowering any other target mean; preserve it as an explicit candidate. VOC/PC still trail the local finite VIP. VIP comparisons use its official words/settings plus the explicit empty-row numerical repair, so they are not same-word pure operator comparisons or certified paper benchmark reproductions. PC uses401 extra residual concepts; information budgets differ. Prior labelled development and full-result feedback make these exploratory results.

Max pooling alone is not a standalone CVPR-level innovation. It fixes a semantic-type mismatch in this model. Geometry's independent contribution and coupled evidence management still require mechanism/fair-comparison evidence. Do not claim all-five SOTA or parameter-free selection: the task menus/words have development provenance even though online inference uses no labels.

Sources: `LOCAL_BACKGROUND_UNION_20261007.md`, `READOUT_ENDPOINTS_20261007.md`, `FINAL_TASK_CONDITIONED_CANDIDATE_20261007.md`, `ADAPTIVE_TRANSPORT_TARGETS_20261007.md`.
