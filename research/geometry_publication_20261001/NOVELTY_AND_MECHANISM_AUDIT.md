# Geometry: novelty boundary and mechanism audit

2026-10-01. This is a bounded comparison against verified public sources, not proof that no related operator exists. Both full matched suites are complete; the empirical verdict is in section9. The original Geometry configuration and historical coupled readouts are preserved.

## 1. What Geometry actually does

Frozen DINOv3 backbone -> raw patch relation -> conditional attention replacement in both frozen DINO.text head blocks -> normalized patch descriptors -> fixed text aliases -> normalized log-mean-exp class scores -> fixed tile assembly.

For every patch query and head, `o_G = A_special V_special + m_patch G V_patch`. Here `G` comes from raw normalized DINO features and spatial distance, while `m_patch` and `A_special` come from the native attention of the current intervened block. Native special-query rows, Values, projection, LayerScale, residual, MLP and final normalization remain active. At the next block the native attention is recomputed on the changed state. The relation is shared across heads; the native group mass is not.

Geometry currently has no image-conditioned alias selector, no text-conditioned visual relation, no optimization loss and no independent semantic-calibration guarantee. Uniform fixed20 aggregation is an input/output convention, not Geometry's innovation.

## 2. Claim-to-source boundary

| Component or possible claim | Verified precedent | Actual Geometry distinction | Allowed wording |
| --- | --- | --- | --- |
| Replace QK with visually meaningful relation | SCLIP CSA; ProxyCLIP external-feature proxy | Raw DINO cosine plus distance in a pretrained DINO.text head | Relation replacement is established, not invented here |
| Use another frozen visual representation's correspondence | ProxyCLIP; SegEarth-OV visual features | Reuses this model's own frozen backbone | Avoid an additional visual encoder; not first use of visual correspondence |
| Apply DINO affinity to two DINO.text blocks | VIP explicitly does this | Conditional replacement rather than VIP masked patch-only proxy | Two-block DINO modulation is not new |
| Preserve residual and MLP | SCLIP and VIP | Preserve the specific native patch/special-token allocation too | Frozen pathways retained, not invented |
| Encourage local spatial donors | NACLIP additive Gaussian neighbourhood kernel | Log-distance prior combined with raw-feature similarity | Spatial prior has precedent; different insertion/normalization |
| Preserve native patch mass and special-key contribution while replacing conditional patch relation | Not found in the inspected SCLIP/ProxyCLIP/NACLIP/VIP implementations | `A_special V_special + m_patch G V_patch` | Explicit conditional mass-preserving operator; priority remains bounded by inspected sources |
| Zero-sum attention displacement | Follows algebraically from two normalized rows | `(G-A_conditional)` has zero row sum before later nonlinearities | A local mathematical property, not a claim of removing all semantic bias |
| Frozen inference across eight datasets | Training-free OVSS precedent | Fixed operator and fully checked coverage | Reproducibility and domain evidence; not eight untouched datasets or SOTA |
| Combining readouts or choosing a winner by dataset | Fusion precedent and our historical experiments | No successful universal second mechanism established | Historical complementarity is evidence, not new ownership of VIP |

Primary author-code anchors (commit-pinned downloads are under `public_sources/`):

- SCLIP `clip/model.py:250,283`: final residual/MLP and `softmax(QQ)+softmax(KK)`; the sum has mass two. Using their average would silently change the operator.
- ProxyCLIP `open_clip/transformer.py:528,560`: last attention-only output; beta1.2/gamma3.0 masked external-feature proxy.
- NACLIP `clip/model.py:115,178`: full versus reduced block; KK logits plus an additive Gaussian kernel. Sigma5 is in patch units, not Geometry's normalized-coordinate sigma.
- VIP `third_party/VIP_official/dinov3/eval/text/vision_tower.py:79,89`: both head blocks, masked DINO affinity, residual/MLP, and normalized special-token inputs returned as an attention increment. Its official grid is21x21; this matched experiment adapts the proxy to32x32.
- ClearCLIP studies residual/feed-forward decomposition; SC-CLIP uses self-calibration and intermediate/multilevel organization. Neither is claimed fully reproduced by the current eight-arm suite.

Repository identities and file SHA256 values are in `public_sources/manifest.json`; VIP is pinned locally to `5bd25ee03ec25c1538622cf7da661e8c0461e769`. Papers and source excerpts are in the sibling `geometry_public_*_20261001.json` files. Public-only queries were used; unpublished mechanism details were not sent to the literature service.

## 3. Matched comparison design

**Core:** Geometry, Native, SCLIP_Last, depth-matched SCLIP_Two, ProxyCLIP_Last, VIPProxy_Two, Geometry_NoSpatial, Geometry_BlockPrefix.

**Locality:** Geometry, NACLIP_Last, Native_Spatial, SpatialOnly. Native_Spatial uses native head-specific QK with exactly Geometry's distance term and native mass/prefix handling. SpatialOnly removes raw feature similarity but retains distance and the same pathways. These separate useful backbone structure from generic smoothing.

All arms share checkpoints, encoded fixed20 text, six templates, class aggregation, native-resolution512 windows, overlap, assembly, precision and sample sets. Constants from authors' operators are retained. No per-domain tuning or post-hoc winner switching is used. Every full Geometry confusion must exactly replay its historical reference. Per-image sums and old/new/truth transition tensors must reproduce endpoint confusions.

These are matched **operator adaptations to DINO.text**, not complete official CLIP baseline systems. In particular, a poor reduced attention-only output in this head says nothing decisive about that operator's performance in its original trained CLIP head. The guard for empty VIP proxy rows reads self and is disclosed. Full official-system claims additionally require the author's weights, view, scoring, vocabulary and distillation protocol.

Metadata clarification: the shared evaluator inherited a generic `signature.note` saying "no VIP operator" from an earlier candidate. This is true of the Geometry arm, not the entire core suite. The explicit `methods`, implementation identifier, `gear.operators`, matched settings and source determine the actual suite; `VIPProxy_Two` is present and attributed. Original files are retained rather than retroactively rewritten. Combined evaluator timing fields are also relabeled accurately in the report: `parallel_wall_seconds` is the maximum shard elapsed duration, and `aggregate_gpu_seconds` is the sum of elapsed durations, not end-to-end queued wall time or measured kernel-active time.

## 4. Mechanism evidence and falsification

| Question | Evidence collected | What it can establish | What it cannot establish |
| --- | --- | --- | --- |
| Does the implementation preserve native row/group mass? | Unit test constructs explicit replaced attention and checks row sums/output | Exact per-block implementation invariant | Better class identity or final calibration |
| Does a common patch Value change the displacement? | Unit test adds a common vector and checks intervention-minus-native | Zero-sum redistribution at one attention operation | Removal of common bias from final descriptors |
| Are published operators faithfully adapted? | Four small FP32 tests call isolated pinned author attention functions | Formula, Gaussian and prefix-increment fidelity | Full official-system reproduction or GPU bit equality |
| Is the large native gain specific to Geometry? | Full matched core suite | Relative effectiveness under identical readout protocol | Priority from gain alone |
| Is geometry just spatial averaging? | SpatialOnly and Native_Spatial full suites | Value of raw relations beyond matched locality | A universal guarantee across unseen domains |
| Are spatial prior and prefix preservation necessary? | Geometry_NoSpatial and Geometry_BlockPrefix | Contribution/negative transfer under one frozen rule | Permission to drop components differently per dataset |
| Is it affordable alone? | One-arm independent synchronized benchmark | Steady-state latency and allocated GPU memory | Multiscale or official-system total cost |
| Are small objects missing? | Class TP/FP/FN and prediction area | Overactivation versus undercoverage by class | Instance-size, boundary or physical-scale attribution |

No scalar confidence score proves correct semantics. High vehicle recall with very low precision is evidence of overactivation, not failure to see vehicles. Likewise, a visually coherent water/roof region can be assigned the wrong text class. Existing readout-stage/spatial diagnostics may support more specific hypotheses, but aggregate confusion alone must not be used to claim a diagnosed causal pathway.

## 5. Independent inference cost

A800, FP32 stored weights with bf16 AMP; five window warmups,20 synchronized window trials. Whole1024 LoveDA prediction uses nine backbone calls and three timed repetitions. Geometry's independent descriptors exactly replay the evaluation path (max error0), as do Native's.

| Cost | Geometry | Native |
| --- | ---: | ---: |
|512-window median |22.3002ms |21.1115ms |
|1024-image median, including probability transfer/assembly |0.3440s |0.3314s |
|Full-predictor peak allocated GPU memory |3605.896MiB |3565.543MiB |
|Backbone calls per1024 image |9 |9 |

Text encoding/model loading/decoding/label evaluation are excluded from steady-state numbers and separately recorded where measured. These are one-arm measurements, not multi-arm evaluation wall time. A dense relation has quadratic patch storage; there is no claim of linear attention complexity. See `independent_latency.json` for all comparator timings and measurement details.

## 6. Publication decision rule

Three judgments must remain separate: identifiable implementation difference, useful increment over closest matched operators, and complete-system SOTA. The first does not imply the second, and neither implies the third.

The full suite must test a globally fixed method, report the complete eight-domain vector and paired scene/group uncertainty, and include negative results. A large gain only over Native is insufficient to sell Geometry as a strong independent CVPR centerpiece when a two-block near neighbour matches or exceeds it. A useful simple readout can remain the first module even if its current novelty/performance margin is insufficient as the sole main contribution.

Do not replace this assessment with renamed fusion, relabeled VIP attention, an invented training loss or a different winning setting per dataset. Preserve the best historical coupled model as an explicitly attributed performance reference. Freeze the eventual final method before genuinely new official test/region evaluation.

## 7. Existing stage and support evidence

The previously completed observation-only diagnostic uses VDD/Potsdam each8 fixed images, all704/72 unchanged sliding windows, exact final Geometry feature replay and score decomposition error below2e-7. High-dimensional snapshots are label-independently chosen (two windows per image). Full results and restrictions are recorded in `../GEOMETRY_READOUT_DIAGNOSTIC_8_20261001.md`; they are not substituted for the new full-domain metrics.

| Observation | Specific evidence | Interpretation boundary |
| --- | --- | --- |
| Inherited false semantic direction | Potsdam car-minus-impervious margins on2017 false-car centres are positive at the earliest final-head probe and remain positive after both blocks | Some errors precede final attention; raw features passed through a semantic projection are still a diagnostic probe, not a true teacher |
| Effective evidence can be lost along learned semantic transformations |98 cached water->other centres flip from positive water margin after block0 attention to negative after block0 MLP | Localizes one observed pathway; this group is from one image and cannot justify deleting MLP across domains |
| A coherent neighbourhood can remain semantically wrong |316/318 other->vehicle centres still have positive vehicle-minus-other Geometry-neighbour margin | Geometry support is not semantic certification; centres share model/view and are not independent observations |
| Neighbours can help particular isolated competitions |22/23 false vehicle-minus-road centres and87/98 missed water centres have the correct neighbour sign | Shows information in the relation, not a validated general rejection rule |
| Global subtraction damages true shared scene semantics | Support-minus-window response becomes nonpositive on all98 missed water centres and on about half926 correctly predicted water centres | Full-window semantic content can be the real object, not nuisance; no label-fitted subtraction gate is justified |
| Removing nonlinear feedback is not a universal remedy | FrozenPathGeometry full UDD5/OEM50.3268/44.0790 versus Geometry50.5553/44.6097; fixed VDD/Potsdam screens also decline | Rejects that particular frozen-path candidate, not every possible path interaction |

Conditional score reconstruction includes the actual normalization denominators and alias responsibilities. It is not the counterfactual effect of deleting the component: deletion changes those denominators, responsibilities and subsequent Q/K/V. High audit AUC also does not provide a natural transferable decision threshold.

## 8. Preserved coupled performance reference

The frozen Anchored_VIP candidate solves a unit-weight fidelity problem using Geometry support and a guarded wide-view VIP semantic observation. Its global rule is retained unchanged; no dataset-specific best-arm assembly is used. Valid eight-domain outcomes are in `../GEOMETRY_SEMANTIC_INNOVATION_20261001.md`, including the corrected IRRG Vaihingen repeat.

| Dataset | Geometry | Anchored_VIP | Change |
| --- | ---: | ---: | ---: |
| LoveDA D |42.6589 |40.2519 |-2.4070 |
| UDD5 |50.5553 |49.2117 |-1.3436 |
| OEM |44.6097 |37.8977 |-6.7120 |
| VDD |38.8511 |55.4446 |+16.5935 |
| Potsdam |40.6892 |43.7334 |+3.0442 |
| Vaihingen, corrected |48.8009 |51.4079 |+2.6070 |
| LandCover.ai |59.2340 |63.7049 |+4.4709 |
| FLAIR-1 |43.8071 |42.5381 |-1.2690 |

This is a strong difficult-domain performance reference, not a uniformly best eight-domain final model: four domains improve and four decline. Most VDD improvement comes from the added observation's view/text/readout; coupling improves the stronger same-source simple-fusion reference by0.7328, not16.5935. Potsdam's attributable margin is0.2110. The borrowed observation must remain explicitly credited, rather than being renamed as our new visual reader. Historical all-arm timing is not standalone latency.

## 9. Verified verdict

Both core and locality suites completely cover the same 20,092 images per suite across eight datasets (LoveDA P/D share images). All sixteen dataset merges verify exact historical Geometry confusions, fixed sample/vocabulary/checkpoint identity, unique full coverage, per-image matrix reconstruction and transition endpoints. All workers and both controllers have completed without failures. Sixteen focused CPU tests pass, covering formula invariants, author-function adaptation and report statistics. Independent latency replay errors are zero.

| Question | Verdict | Evidence |
| --- | --- | --- |
| Is Geometry useful compared with the native head? | Yes, across all eight current domains | Mean46.1508 versus33.0498; gains8.6085 to20.6640 points |
| Is it just generic spatial smoothing? | No, under the matched controls | SpatialOnly43.2429; Native_Spatial35.1753; Geometry wins every domain against both |
| Is its complete operator identical to inspected VIP/SCLIP/NACLIP? | No | Conditional native patch mass/special-key allocation differs; raw relations and frozen two-block modulation have clear precedents |
| Does the specific new distinction produce a stable nearest-method advantage? | Not demonstrated | SCLIP_Two47.2215 and VIPProxy_Two46.8876; Geometry wins only two of eight primary entries against each |
| Is preserving native group mass/prefix uniformly helpful? | No | BlockPrefix46.8188; background benefits in LoveDA D coexist with negative transfer elsewhere; this ablation bundles two changes |
| Is the spatial term uniformly necessary? | No | NoSpatial46.2753; six primary domain entries improve after removal |
| Is Geometry's own overhead modest? | Yes in the measured standalone benchmark |+5.63% window latency,+3.79% whole1024-image latency,+40.35MiB peak allocation versus Native |
| Can these results establish complete-system SOTA or an untouched cross-domain claim? | No | Operator adaptations are not complete official systems; all eight domains informed development; split/input caveats remain |
| Is Geometry a defensible first module? | Yes, with bounded contribution wording | Useful structure-conditioned frozen readout, not semantic certification |
| Is it already proven sufficient as the sole strong CVPR centerpiece? | No | Identifiable formulation exists, but stable nearest-method superiority and independent validation are missing |

**Recommended contribution wording:** "A conditional, native-mass-preserving patch readout in a frozen semantic head, with explicit pathway invariants and matched multi-domain analysis." Do not claim the first DINO affinity reader, first two-block intervention, invented residual/MLP preservation, a generally beneficial spatial prior or a solved context-trust controller.

The appropriate next research requirement is not another name for fusion: a single coupled rule must distinguish semantic misactivation from lost true coverage, beat a strong same-information-source comparison under one fixed configuration, and retain the useful background competition instead of trading it away unnoticed. Coherent Geometry support alone does not supply that distinction. This is a requirement for the next module, not a claim that a validated design already exists.

Writing is synchronized in `MANUSCRIPT_DRAFT.md`: Introduction, full Geometry definitions/proposition/no-training-loss statement, experimental protocol and verified results. `MATCHED_RESULTS.md`, `paired_statistics.json`, `ERROR_ANALYSIS.md`, `class_errors.json` and `independent_latency.json` contain reproducible quantitative evidence. The literature skill supplied public-source boundaries and citations, not our experiment results or an exhaustive priority guarantee.
