# Geometry leave-self-out adaptive gain: full paired comparison

One fixed rule, unchanged words/profiles/visual budget/long448/residual handling. Squared offdiagonal H induces leave-self-out neighborhoods; branch probability consistency modulates frozen g between0 and2g. Semantic correctness is not guaranteed. Prior labelled development: exploratory.

| Protocol | Frozen | GeometryConsistency | Delta |
| --- | ---: | ---: | ---: |
| vdd | 55.2382 | 54.2184 | -1.0198 |
| potsdam | 49.7888 | 50.4398 | +0.6510 |
| voc21 | 70.3179 | 42.3029 | -28.0150 |
| context60 | 41.5558 | 39.0864 | -2.4694 |
| ade150 | 31.1076 | 29.8088 | -1.2988 |

Default remains frozen; full comparison is evidence, not automatic promotion.

## Verified decision

All9138 protocol images completed without failures. Every merged protocol verifies full unique coverage, per-image confusion sums, paired scored-target equality, checkpoint/class identities and exact retained Frozen confusion replay. This rule is rejected as a universal upgrade: four of five domains decline. Existing frozen defaults and prior best outputs are retained.

## Full-confusion diagnosis

| Protocol / class | Frozen IoU | Adaptive IoU | Other observed change |
| --- | ---: | ---: | --- |
| VDD water | 83.0469 | 76.5321 | -6.5148pp |
| VDD wall | 47.3411 | 49.2977 | +1.9566pp |
| Potsdam low vegetation | 43.1983 | 48.1229 | recall44.948→50.972% |
| Potsdam car | 36.7792 | 32.3208 | precision36.976→32.459%; prediction area5.238→5.975% |
| VOC21 background | 89.5149 | 53.5419 | recall93.381→53.937%; prediction area71.632→40.086% |
| VOC21 aeroplane | 62.3288 | 21.7617 | precision63.817→21.787%; recall96.393→99.475% |
| VOC21 bottle | 53.9320 | 51.6567 | recall58.342→83.345%, precision87.707→57.603% |
| PC60 background | 21.2474 | 17.0147 | recall32.7488→60.4688%; precision37.6944→19.1441%; area7.2329→26.2959% |
| PC60 person | 69.1043 | 40.4819 | -28.6224pp |
| ADE150 person | 61.8811 | 44.1453 | -17.7358pp |

PC60 foreground mIoU41.9000→39.4605;22classes improve and38decline. ADE15050classes improve and100decline; cushion improves32.6346→56.3845 while sculpture43.2244→29.1651 and van37.6618→23.7161 decline. These class outcomes are post-evaluation diagnostics, not exceptions used to select the method.

## Interpretation and next decision

The designed error compares probabilities with their own geometric neighborhood. A uniform branch has zero error regardless of semantic correctness; this is a direct property of the rule, also exercised by the unit test that doubles gain for a consistent broad branch. It therefore measures self-consistency rather than independent reliability. The full outputs show increased foreground spread on VOC, background spread on PC60, and a mixture of gains/losses without scored generic background on ADE. The aggregate confusion matrices do not isolate whether each error arises from semantic confidence, signed transport extrapolation, or per-patch gain selection.

The factor is bounded0..2 but multiplies a developed gain; VOC g2 can reach4. A bounded factor is not a guarantee of a convex blend or monotonic accuracy for signed H. No realized gain distribution was logged, so do not assert that gain4 explains the specific VOC failure.

Do not select the adaptive route only on Potsdam, fit a new cap to these full-test scores, or call the rejected rule a successful reliability module. The next candidate needs evidence about competing semantic explanations or a independently justified calibration objective; mere smoothness/entropy/agreement is insufficient. Word-count correction failed separately, so variable word-count and branch calibration must be studied jointly without removing residual concept coverage. Any subsequent rule requires a separate frozen protocol, not an in-place repair of this completed run.
