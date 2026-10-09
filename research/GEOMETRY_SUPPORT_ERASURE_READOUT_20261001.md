# Geometry-defined semantic intervention and local reconstruction

## Why this is the next observation

Independent attention-only Geometry helps three screens but harms VDD and does not beat same-source simple fusion. Neither original Geometry support, unedited native head, virtual region CLS nor whole-margin density has established which semantic direction should be trusted. A new final architecture needs actual additional semantic observations, not a relabeled solver or transplanted VIP head.

Test one shared operator H for both observing and writing semantic evidence. Geometry partitions raw DINO features into supports using the already fixed SLIC-zero32/max4 policy. For each support, average the original Geometry relation rows, exclude image padding, and normalize. The soft image erasure strength is H divided by its maximum. This preserves coordinates, resolution and physical field of view; no zoom or bounding-box context change is mixed in.

For the actual original RGB window and its support-erased counterpart, run the native frozen DINO.text image path. Replacement is the valid-window RGB mean. The full trained image descriptor includes real CLS and pooled patches, rather than virtual region queries. Read the normalized embedding difference against the full2048-dimensional encodings of the same20 aliases and six RS templates. This is a hypothesized support-conditioned semantic observation, not a calibrated class posterior. Erasure may introduce artifacts or change context outside the support through the backbone; geometric support can contain several classes.

Let g be original local Geometry class cosine/LME scores, b the support observations. Reconstruct one dense field using:

`min_z .5||z-g||^2+.5||H z-b||^2`.

Equivalently `z=g+H^T(I+HH^T)^(-1)(b-Hg)`. H has at most4 rows, so the solver is small. H both determines the image intervention and controls the dense correction. Zero response or no support gives exact original output; no alias is physically deleted or re-averaged. There is no VIP observer, class-dependent threshold, external network, label-fitted gate or learned weight. The quadratic itself is not novel; any potential contribution lies in the grounded visual measurements and shared observation/reconstruction operator, subject to literature and performance verification.

## Fixed diagnostic before promoting a full candidate

Run only on the existing16 windows per VDD/Potsdam diagnostic, GPU6/7, with primary SupportErasure. Controls SupportGlobal and SupportErasedGlobal use exactly the same supports, full text and inverse writeback but not the embedding difference. Original Geometry must exactly replay the cached class scores. Selector inputs exclude target masks, including image validity, which is recomputed from RGB bounds.

Record all class IoUs and `(old,new,GT)` counts at overlapping patch centers, response magnitudes, support coverage and time/memory. These are not full-image scores. Require positive primary net corrections and no mIoU decline in either diagnostic before implementing/launching full-image transfer evaluation. This gate rejects an information-source hypothesis; it does not prove universal reliability or originality. The eight previously developed domains are not independent validation.

## V1 outcome: positive but too little coverage

Original Geometry exactly replayed all32 cached windows. Complete VDD/Potsdam audits are downloaded locally under `research/geometry_readout_diagnostic_8_20261001/*_support_erasure_snapshot_20261001.json`.

| Fixed patch-center audit | Geometry | SupportGlobal | SupportErasedGlobal | SupportErasure | Primary beneficial/harmful |
|---|---:|---:|---:|---:|---:|
| VDD, 16 overlapping windows | 44.3340 | 44.5946 | 44.5566 | 44.4121 | 18/4 |
| Potsdam, 16 overlapping windows | 40.0511 | 40.0050 | 40.0068 | 40.0635 | 14/6 |

These small gains do not establish a useful correction mechanism. Unit-weight mean measurements are diluted by the local prior's token count: for a uniform N-token support and constant g, every token moves only `(b-g)/(N+1)`. A near-identity result is not adequate evidence for the full objective.

## V2 analytic measurement normalization

Preserve V1 results and the exact same visual observations. Normalize each H row and its corresponding b by the H row's L2 norm before the same fixed unit-weight solver. A uniform support then gives `(b-g)/2`, independent of its size. This is a geometric unit convention, not a class-specific sign, label-fitted strength, semantic confidence or statistical independence claim. All classes and all observation controls receive the same normalization.

V2 primary BalancedErasure is compared with BalancedGlobal and BalancedErasedGlobal, alongside the unchanged V1 controls. Write to new `_snapshot_v2_` directories, never over V1 outputs. Record observation/reference vectors and effective support token counts so the observer and solver are separately inspectable. Require positive net corrections and nondegradation on both fixed diagnostics before full-image promotion; this does not certify generalization.

## V2 rejected at meaningful correction coverage

| Fixed patch-center audit | Geometry | BalancedGlobal | BalancedErasedGlobal | BalancedErasure | Primary beneficial/harmful |
|---|---:|---:|---:|---:|---:|
| VDD | 44.3340 | 42.8653 | 42.8663 | 40.7794 | 344/612 |
| Potsdam | 40.0511 | 35.8022 | 35.6535 | 38.9493 | 491/471 |

All16 windows per domain exactly replay original scores, all64 supports per domain have a nonzero response. Effective support size averages100.57 tokens in VDD and89.25 in Potsdam. Primary VDD water89.1242 drops to76.4657 and roof78.1117 to70.9519; Potsdam building72.7498 drops to65.7966. Normalized descriptor differences do not identify a reliable absolute semantic prototype. Reject V2; do not promote the weak V1 by ignoring its coverage problem.

## V3: measure actual semantic score influence, not a new prototype

Use the already saved original/erased native class scores with all20 aliases: `Delta_c=S_c(original)-S_c(erased)`. Do not normalize a feature difference into a new class embedding and replace Hg with its absolute scores. The observation is an increment, so solve `min_delta .5||delta||^2+.5||D(H delta-Delta/f)||^2`, where D normalizes H rows and f is the actual soft-erasure fraction of valid image centers. Add delta to original Geometry. Compare unscaled ScoreGain with area-scaled primary ScoreInfluence. There is no tuned strength; the latter tests a per-area influence hypothesis. The nonlinear image encoder does not guarantee exact additive attribution, and small-support artifacts can be amplified.

A common gain for all classes does not alter competition; zero gain preserves the exact original prediction. Geometry defines both the counterfactual content and its inverse writeback. This is a candidate complete coupled visual readout, not another teacher-fusion gate or a claim of proven novelty.

First replay the16 cached windows per domain without new image/model forwards. Reconstruct the original supports from raw DINO caches and require exact original/V2 confusion matrices and support-reference equality within1e-6 before evaluating V3. The fixed rule uses no class-specific sign or target-label coefficient. Both domains must retain mIoU and positive net corrections before any full-image promotion. Saved sources and outcomes remain separate versioned artifacts.

## V3 diagnostic outcome and full-image decision

Both replays are complete and `source_observation_controls_exact=true`. These are overlapping patch centers from16 fixed windows per domain, not full-image results.

| Diagnostic | Geometry | BalancedErasure | ScoreGain | ScoreInfluence | Primary beneficial/harmful/wrong-to-wrong |
|---|---:|---:|---:|---:|---:|
| VDD | 44.3340 | 40.7794 | 44.8740 | 44.4767 | 593/461/505 |
| Potsdam | 40.0511 | 38.9493 | 40.1205 | 40.0981 | 619/393/592 |

The predeclared primary passes both diagnostic gates, but only by +0.1427/+0.0470 mIoU. Do not replace it with ScoreGain after examining labels. VDD water89.1242->94.0576 and vegetation68.5895->70.9534 improve, while roof78.1117->73.1338 and road63.6776->60.6512 decline. Potsdam low vegetation61.2922->65.1068 and car5.4791->6.3497 improve, while building72.7498->68.9692 and impervious surface48.5694->46.9694 decline. Positive total correction counts do not imply every class improves, and the main small-object overprediction remains.

Proceed to unchanged full UDD5(40), OEM(384), VDD(80), Potsdam(504), keeping Geometry, ScoreGain and ScoreInfluence. Use physical GPU4-7 only when idle. The RGB-aware evaluator preserves original Geometry interpolation, temperature and Hann blending. Require complete unique coverage, identical vocabulary/sample signatures, and exact baseline Geometry confusion before interpreting new metrics. No eight-domain promotion based on patch-center gains alone; full transfer evidence determines the next step.

## Full-image trials launched

The12 core solver/intervention tests and one RGB-callback integration test pass on A800. The integration test requires bit-exact unchanged Geometry predictions and the same image-valid patch centers with/without the RGB callback. Full-image baseline equality remains to be verified by the complete confusion matrices.

GPU4 was occupied during the first inspection but had become idle before the launcher rechecked occupancy. All four launches passed idle checks. GPUs0-3 were not used.

| Dataset | Physical GPU | Full image total | Session |
|---|---:|---:|---|
| UDD5 | 4 | 40 | gsi03_influence_full_udd5_s0 |
| OEM | 5 | 384 | gsi03_influence_full_oem_s0 |
| VDD | 6 | 80 | gsi03_influence_full_vdd_s0 |
| Potsdam | 7 | 504 | gsi03_influence_full_potsdam_s0 |

Remote roots are `results/geometry_support_influence_v3_full_DATASET_20261001`, each containing `s0.log` and `s0/results.json`. The bounded transfer controller `gsi03_influence_suite` merges completed trials with old/new/GT transitions and requires exact matched baseline coverage, vocabulary, checkpoints and Geometry confusion. It does not tune parameters or automatically launch eight datasets. Controller log: `results/geometry_support_influence_v3_transfer_20261001.log`.

All four were healthy at the first progress check: UDD5 5/40, OEM41/384, VDD3/80, Potsdam42/504. Primary progress metrics were lower on UDD5/OEM and close to Geometry on VDD/Potsdam; partial metrics are not final results and do not change the frozen rule. A passed patch-center diagnostic is not evidence of reliable full-image transfer. Full results and the cross-domain decision are pending.

## Measurement-direction audit

`audit_support_measurement_alignment.py` reconstructs the saved32 windows and128 supports on CPU, without new model forwards. The maximum support-reference replay error is4.17e-7 for VDD and5.22e-7 for Potsdam. Targets enter after reconstructing H, the native gain and the unchanged readout. The audit cannot select aliases, supports or coefficients.

| Existing diagnostic supports | Mean target purity of H | Geometry dominant matches | Native-gain dominant matches | Corrected dominant matches |
|---|---:|---:|---:|---:|
| VDD,64 | 93.09% | 38/64 | 20/64 | 39/64 |
| Potsdam,64 | 78.68% | 26/64 | 13/64 | 29/64 |

The mean target mass assigned to the chosen class is58.64%/38.69% for Geometry and31.85%/23.96% for native gain. These selected diagnostic supports contain no vehicle/car-dominant support; they cannot establish small-object classification reliability. A gain is an influence direction, not a posterior, but it is substantially weaker than Geometry as a detector of the dominant removed category in this audit. The hypothesis that the measured global score change is a reliable local semantic observation is unsupported even when many Geometry supports are pure.

The original native class delta averages0.001952 in VDD and0.001872 in Potsdam. Dividing by erasure area raises these to0.047479 and0.045156, approximately24 times larger. This scaling can amplify a misidentified direction; it is not the only demonstrated weakness. The RGB-mean replacement may preserve semantic appearance (e.g. water remains water-colored) or alter texture/context, so the score change is not necessarily removal of a category. This remains a proposed explanation rather than a proven fill-effect mechanism; a matched intervention control would be required.

Consequently, do not promote the readout from positive diagnostic net counts or improve it only by changing a diffusion/solver/gate. A next reliable observation must first establish what local semantic information it measures and distinguish content from replacement/context response. The complete full-image transfer trials remain unchanged. Local audit artifacts: `geometry_readout_diagnostic_8_20261001/{vdd,potsdam}_support_measurement_alignment_v3_20261001.json`.

## First verified full transfer result

OEM384 is complete and coverage-verified. The controller verified exact original Geometry confusion, sample sequence, vocabulary and checkpoints. Geometry44.6097, ScoreGain44.6252, primary ScoreInfluence42.2063(-2.4034). Primary beneficial7,587,984/harmful13,787,864/wrong-to-wrong11,445,503. This rejects the cross-domain nondegradation gate regardless of the remaining outcomes. Do not extend V3 to eight datasets. Preserve and finish the other three already running trials.

## Next bounded observation diagnostic, declared before outcomes

On the same16 windows and64 supports per domain, compare actual native full-descriptor readings of the original image, support kept against window-mean fill, and support kept against zero fill. Compute deletion influence with mean and zero fill and the exact two-coalition Shapley attribution `.5*(S(original)-S(drop)+S(keep)-S(null))` for each fill. Primary attribution probe is ZeroShapley; all observers retain the same H,512px FOV, all20 aliases and six templates. No area division, fitted weight, label-fitted threshold or new propagation is introduced. Require original Geometry and original native score replay to be exact.

This tests the observation source rather than adding a final model: support-only absolute recognition is distinct from a global score-change attribution. Neither attribution nor gray/black masking is assumed to be a calibrated posterior or in-distribution. The audit reports dominant-category matching, expected target mass and beneficial/harmful support decisions after every native observation is computed. It does not alter the ongoing full-image trials or justify selecting different rules for each dataset.
