# Geometry coupling: completed execution and decision

## Scope and outcome

All four complete candidates were implemented and evaluated only on authorized physical A800 GPUs4-7. All runs are terminal. Their prospective promotion gates failed; none was promoted to a full eight-dataset rollout. The research objective remains unachieved.

Each screen covers the same96 unique images: full40 UDD5 and eight fixed images for each other domain. Samples, checkpoint identities, vocabulary identities, baseline confusion matrices and per-image reconstruction were checked. All20 aliases per class remain. Weights are frozen; masks are evaluation-only. No per-domain parameter or model choice.

These are exploratory development screens, not full eight-dataset SOTA results. VIPProxy_Two is a matched DINO.text operator adaptation, not the complete official VIP system. Corrected IRRG Vaihingen is used. LandCover.ai replaces unlabeled iSAID.

## Implemented complete candidates

| Candidate | Coupling | Tested limitation |
| --- | --- | --- |
| DonorBudget | Geometry flow is KL-projected onto native attention donor marginals; both head blocks retain native patch/special mass and frozen transforms | Native budgets are far more concentrated; satisfying quotas is not semantic ownership |
| RoleCompatible | Geometry edges are conditioned on query attention-profile Hellinger distance; equal/uninformative profiles recover Geometry exactly | Similar learned attention roles do not supply reliable class identity |
| RegionStrong | Geometry supports control frozen SigLIP2 SO400M detail/context pooling and a joint pixel-region KL reconstruction | Borrowed region observations are too weak on most screened domains, despite gains over same-source fusion |
| Geometry_UnitDetail | Geometry-defined196 connected units are re-encoded by the original frozen head; dense detail is reconstructed as `Yg + U(Yunits - S Yg)` | Pre-normalization mean/detail invariants do not preserve useful class evidence;0/8 domain wins |

The mathematical projection, Hellinger kernels and KL solver have precedent. Neither their implementation nor the larger borrowed observer is claimed as an original invention.

## Matched screen mIoU

| Dataset/protocol | Geometry | SCLIP_Two | VIPProxy_Two | DonorBudget | RoleCompatible | RegionStrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 31.9462 | 30.6800 | 31.1056 | 27.0574 | 31.6467 | 32.9216 |
| potsdam/potsdam | 40.3528 | 44.0525 | 42.7377 | 25.1387 | 40.1809 | 38.3952 |
| udd5/udd5 | 50.5553 | 50.2837 | 49.9071 | 39.4870 | 50.1738 | 48.0467 |
| oem/oem | 39.3543 | 37.3120 | 38.9899 | 35.0790 | 39.5040 | 32.3310 |
| loveda/P | 62.8254 | 68.2854 | 68.5692 | 45.6732 | 61.5712 | 54.9680 |
| loveda/D | 38.4558 | 36.5043 | 35.6670 | 27.9732 | 38.2467 | 35.3908 |
| vaihingen/vaihingen | 50.2325 | 52.0872 | 50.5393 | 40.6944 | 50.2190 | 45.8222 |
| landcoverai/landcoverai | 60.9049 | 61.7170 | 59.3073 | 51.6082 | 60.8735 | 57.8141 |
| flair1/flair1 | 38.8428 | 37.6225 | 39.3436 | 27.8610 | 38.3479 | 31.2528 |

LoveDA D enters the equal-domain mean once; P is reported separately.

| Method | Equal-domain mean | Wins vs Geometry |
| --- | ---: | ---: |
| Geometry | 43.830569 | n/a |
| SCLIP_Two | 43.782393 | n/a |
| VIPProxy_Two | 43.449702 | n/a |
| DonorBudget | 34.362349 | 0/8 |
| RoleCompatible | 43.649058 | 1/8 |
| RegionStrong | 40.246801 | 1/8 |
| Geometry_UnitDetail | 26.874787 | 0/8 |

## Frozen gates and cost

| Candidate | Promotion passed | Window time vs Geometry | Primary peak MiB |
| --- | --- | ---: | ---: |
| DonorBudget | False | 3.0866x | 3911.070 |
| RoleCompatible | False | 1.8081x | 4285.637 |
| RegionStrong | False | 58.1795x | 8850.988 |
| Geometry_UnitDetail | False | 1.2464x | 3715.736 |

These are synchronized single-window medians, not full-image throughput. RegionStrong keeps both encoders resident during its Geometry timing; that baseline memory is not standalone Geometry-only memory. Combined-evaluator durations are not deployed-primary inference times.

## Mechanism verdict and next requirement

1. Stronger structural consistency is not a semantic correctness certificate. The results reject these specific interventions, not every possible training-free coupling.
2. RegionStrong improving its same-source fusion control does not compensate for a regional observation that is inferior to original Geometry. Bigger encoders and more computation alone did not solve that problem.
3. The next candidate needs a demonstrably useful, position-specific competing-class observation before another reconstruction solver. Coherent vehicle/car false activation and lost water/vegetation evidence must both be addressed, without blanket suppression.
4. Do not tune role scales, donor quotas, region influence or per-class thresholds from these labeled outcomes. Do not choose a different winning model for each dataset.
5. Preserve Geometry and all failed results. No successor has been selected or verified by this report. CVPR novelty and eight-domain official-VIP superiority remain open.

## Reproduction

- Model files: `DINOtool/dinotool/geometry_donor_budget.py`, `geometry_role_compatibility.py`, and `geometry_region_strong.py`.
- Detailed reports: `GEOMETRY_DONOR_BUDGET_SCREEN_20261002.md`, `GEOMETRY_ROLE_COMPATIBILITY_SCREEN_20261002.md`, and `GEOMETRY_REGION_STRONG_SCREEN_20261002.md` under research.
- Earlier native-backbone reacquisition is a separate completed full20,092-image trial, not another row in this screen. Its mean46.0334 is below full-reference Geometry46.1508; see `GEOMETRY_REACQUISITION_EIGHT_DATASET_20261002.md`.
- The fourth candidate's full class results and exact control-recovery audit are recorded in `GEOMETRY_UNIT_READOUT_SCREEN_20261002.md`. Its clean repeat verifies all three matched baselines, and original Geometry/primary per-image matrices reproduce the prior run exactly on the four repeated domains. The failed control run is retained. Physical GPUs4-7 were idle after completion; no successor has been launched.

## Later continuation: competitive semantic demixing

After the four-candidate report and the separate failed language-observation diagnostic, the pending competitive-reconstruction investigation was implemented as one complete successor. This is not a claim that Astra endorsed this particular NNLS objective or that it is a novel solver.

Signature `geometry-joint-semantic-residual-demixing-v1-20261002`. Original Geometry yields Y and G; all20 alias descriptors form T; joint nonnegative coefficients minimize `.25*(||Y-XT||^2+||G(Y-XT)||^2)+.5*.01*||X||^2`. The class readout is the norm of its explained descriptor. Geometry measures residuals rather than enforcing equal class labels. No new encoder, crop, learned weights, hard alias deletion, fitted threshold or dataset routing.

The final clean96-image eight-domain screen completed on physical GPUs4-7 only. UDD5 is full40, seven other domains8 each. Twenty tests passed locally and remotely; real-checkpoint FP32/bf16 controls passed. Complete unique coverage, unchanged all20 vocabularies/checkpoints/config, exact original Geometry/SCLIP/VIPProxy per-image matrices and confusion/transition reconstruction were verified. Earlier numerical failures remain in separate directories; their repairs changed no model objective, ridge, class scoring or data.

Equal-domain mean Geometry43.830569 -> primary42.348325 (-1.482244pp), wins3/8. TextOnlyDemix42.310300, same-source fixed logit fusion43.264592, matched VIPProxy43.449702. Primary gains VDD+0.6972, Potsdam+3.8806, Vaihingen+2.3679pp; it loses on UDD5, OEM, LoveDA P/D, LandCover.ai and FLAIR-1. The fixed prospective gate failed; no full rollout was launched.

Potsdam car11.6582 ->22.5028 IoU, but low vegetation33.4221 ->20.8390. VDD vehicle5.6965 ->13.7367, but water33.6534 ->20.8069. Sparse explanations and reduced false car activation do not certify reliable semantic correction. The residual coupling adds only+0.038025pp mean over text-only decoding and trails simple fusion by0.916266pp. Independent window latency is10.7067x Geometry, with primary peak3716.373MiB.

Detailed verified output: `GEOMETRY_SEMANTIC_DEMIXING_SCREEN_20261002.md`; protocol and numerical repairs: `GEOMETRY_SEMANTIC_DEMIXING_PROTOCOL_20261002.md`; raw merged files/per-image matrices: `geometry_semantic_demixing_screen_r2_20261002/`. This fixed successor is rejected, not promoted as a final model. The original Geometry and historical best candidates remain unchanged; the eight-domain official-VIP/CVPR objective is still unmet. Final check: all experiment workers/controller terminal, GPUs4-7 idle, unrelated tmux sessions preserved.

## Localized detector observation continuation

After demixing failed, the missing position-specific observation was tested using the frozen public GroundingDINO tiny checkpoint, pinned revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. This investigation is not attributed to an explicit Astra endorsement. It adds detection pretraining and is not a matched single-encoder system or our detector innovation. Safetensors and standard Transformers only; no external image/prompt/label upload or remote executable repository code.

Implementation `geometry-grounding-observation-diagnostic-v1-20261002`. Exact20-alias coverage is checked with tokenizer offsets and untruncated192-token round-robin captions. Seven tests passed locally and remotely, followed by a real-image mask-free smoke. The same96 fixed images were observed on physical GPUs4-7, using one or two image-only original512 windows each, NOT complete-image inference. Raw boxes/alias responses/original Geometry scores and relations were saved before the separate label audit. All eight runs completed without failure; confusion matrices and correction transitions were reconstructed and unique image coverage verified.

The fixed BoxMean raw-rasterization diagnostic failed on all eight domains and both LoveDA protocols. Window-only Geometry/BoxMean IoU: VDD37.8993/4.1291, Potsdam40.7676/8.6138, UDD545.1736/24.1372, OEM38.9652/6.9555, LoveDA P65.1016/9.9669, D35.8503/6.3642, corrected Vaihingen47.6994/21.6744, LandCover.ai60.9049/13.5520, FLAIR-138.8428/0.6587. These are not full-dataset metrics and must not be compared directly with the earlier full-image screen table.

Potsdam window car IoU improved11.7333->45.5966, while low vegetation27.6598->0 and tree63.9191->1.1537. VDD other recall48.174->98.035 while road/vegetation/vehicle/water recall collapsed to0. Detector coverage reached100% on six domains; background/residual/broad boxes can dominate the raw all-class competition. Car's gain shows some localization observations may be useful, not that all detector information is invalid. The failure rejects this fixed source/rasterization rule and its justification for immediate full rollout; it does not prove every localized observation unusable.

No confidence tuning, hard alias deletion, class-specific routing, successor solver or full rollout was performed after viewing labels. Original Geometry and historical models remain unchanged. See `GEOMETRY_GROUNDING_OBSERVATION_PROTOCOL_20261002.md`, `GEOMETRY_GROUNDING_OBSERVATION_20261002.md`, and verified local summaries under `geometry_grounding_observation_20261002/`. The useful original coupled module and eight-domain complete-official-VIP superiority remain unachieved.

## Revised localized-likelihood coupled readout

The following goal continuation classified the completed detector diagnostic as progress: it supplied useful object observations mixed with uninformative broad/background detections. A label-free raw-box extent audit established that90% of VDD other box queries cover>=95% of a window (median area0.999128), versus vehicle median area0.016676. This motivates neutralizing unlocalized responses, not confidence-threshold tuning. The original raw-box gate remains failed.

One complete revised operator was implemented as `geometry-localized-alias-likelihood-v1-20261002`: original Geometry g/G plus frozen all20-alias detector observations -> h=b*(G*1[b>0]) within each box -> D=N*h/sum(h) -> per-alias unit-uniform-pseudocount e=(1+sum(s*D))/(1+sum(s)) -> class mean of all20 e -> g'=g+.07*log(e). Whole-window and empty observations are exactly neutral; G=I exactly recovers BoxLocalLikelihood. Cutoff0.25, temperature0.07, pseudocount1 and alias groups are fixed across all domains. Bayes updates/pseudocounts have precedent; additional detection pretraining and its cost are explicit, not new detector innovation or matched single-encoder inference.

Ten core tests and a near-tie regression passed locally/remotely. The first cached suite is preserved: LoveDA failed exact Geometry replay because FP32 softmax changed one D-window near-tie pixel. A label-free numerical audit proved that difference; the clean r2 repair uses the original direct interpolated-score argmax. Candidate score arrays, equations, data and parameters are unchanged. All96 images' cached windows then completed on physical GPUs4-7; raw candidate scores were saved before masks; sample/vocabulary coverage, actual identity controls, original Geometry matrices and transitions were verified. Source weights were not rerun. These are window-only development statistics, not full-image mIoU.

Mean Geometry43.262886 -> primary42.917934 (-0.344952pp), two domain wins: Vaihingen+3.6610 and Potsdam+0.0008pp. Mean BoxLocalLikelihood42.700817, fixed probability blend43.238662, shuffled support42.746255. Geometry conditioning improves the box-only control in every domain/protocol (+0.217117pp mean), but does not beat original Geometry or same-source simple fusion. The prospective gate failed; no actual-checkpoint/full96/full20,092-image rollout was launched.

Potsdam car11.7333->15.6110 while building77.4448->68.0955. UDD5 and OEM building also lose4.4658/8.8481pp; LoveDA P building loses25.3052pp. Normalizing detection density introduces lower class likelihood outside partially detected supports. It does not certify absence there; reliable coupling still needs to distinguish positive observations from unknown coverage. This is a remaining hypothesis boundary, not a newly validated correction rule or reason to tune a coefficient from masks.

Detailed protocol/results: `GEOMETRY_LOCALIZED_LIKELIHOOD_PROTOCOL_20261002.md`, `GEOMETRY_LOCALIZED_LIKELIHOOD_CACHED_20261002.md`, verified outputs under `geometry_localized_likelihood_cached_r2_20261002/`. Scholarly Search remains unauthorized; no new verified novelty gap. SAM3 metadata is public but its checkpoint is manually gated, and no weights were downloaded or run. Original Geometry/historical best candidates remain intact. The research objective remains active and unachieved.

## Fixed sign audit: correction of the previous explanation

The completed likelihood candidate was followed by a fixed computational diagnostic, not another selected model. For both Geometry-conditioned and box-only scores, split the update at the neutral value zero: `PositiveOnly=g+max(g'-g,0)`, `NegativeOnly=g+min(g'-g,0)`. Original encoders, detector observations, all20 aliases, temperature, cutoff and96 image IDs are unchanged. Candidate arrays were saved before loading masks. All eight cached-window runs completed on physical A800 GPUs4-7. Original Geometry and complete-candidate confusions replay exactly; coverage and transition reconstruction are verified. These remain one/two512 windows per image, not full-image dataset mIoU.

Equal-domain window mean: Geometry43.262886, complete candidate42.917934, PositiveOnly_Geometry42.017725, NegativeOnly_Geometry44.495695, PositiveOnly_Box41.833011, NegativeOnly_Box44.472885. LoveDA D enters this mean once; P is reported separately. PositiveOnly loses on all eight domains and both LoveDA protocols (-1.245161pp mean). NegativeOnly improves VDD/Potsdam/UDD5/Vaihingen but loses OEM/LoveDA/LandCover.ai/FLAIR-1. Its advantage over the matched box-only NegativeOnly is only+0.022810pp mean. Neither sign split was promoted or selected by dataset.

This directly revises the preceding unproven explanation: eliminating negative increments outside incomplete detections does NOT repair the readout. Positive location boosts also damage class competition. For example, LoveDA P building falls51.7113->25.6828 with PositiveOnly, and Potsdam building77.4448->69.7717. The diagnostic does not uniquely establish whether loose spatial support, incorrect phrase responses or cross-class likelihood calibration causes those boosts; it does establish that a coverage-only repair is insufficient for this fixed candidate. Geometry cohesion and detector confidence alone are not semantic correctness certificates.

Do not turn favorable NegativeOnly results into a retrospectively selected final model, describe sign clipping as a CVPR contribution, or compare window metrics to full official VIP. A useful successor must first supply pixel-specific competing-class measurements that improve the disputed decisions, then demonstrate extra value from Geometry coupling over the same-source control. No new source, threshold search or full rollout was launched in this audit.

Implementation: `DINOtool/scripts/audit_localized_likelihood_signs.py` and `run_localized_likelihood_sign_audit.py`. Full report: `GEOMETRY_LOCAL_LIKELIHOOD_SIGN_AUDIT_20261002.md`; verified outputs: `geometry_localized_likelihood_sign_audit_20261002/`. All15 localized-likelihood/cache/sign tests were rerun and pass locally and remotely after the optional audit API change. Final read-only check confirms all audit workers/controllers terminal and physical GPUs4-7 idle; unrelated sessions `controlled_index` and `qlift_smoke_matched_20260921` are intact. Original Geometry and historical best models are preserved. The original useful-module and eight-domain complete-official-VIP objective remains active and unachieved.

## Pixel-support measurement continuation

The remaining box-versus-semantic ambiguity was tested with the public frozen SAM2.1 tiny checkpoint, revision `de431c4043854a71d8101e17995dfe596bf101a5`. This adds segmentation pretraining in addition to the cached detector's detection pretraining; SAM2/Grounded-SAM are borrowed sources, not a claim of a new detector or mask model. Safetensors and standard Transformers only; image-model loading has zero missing, unexpected, mismatched or error entries. No images, aliases or labels were uploaded. A mask-free real-window smoke passed; original Geometry and box scores reproduce exactly, and identity Geometry recovers the mask-only control exactly.

One fixed complete candidate, `geometry-grounded-mask-likelihood-v1-20261002`, replaces each original fractional box support b with `M=b*patch_mean(sigmoid(SAM2_mask_logits))`. An originally constant valid box support stays unchanged and neutral, rather than becoming an arbitrary salient-object mask. Every query over the unchanged0.25 phrase-mean cutoff is observed, without NMS, query cap, alias deletion or mask-quality selection. The original signed likelihood, temperature0.07, pseudocount1 and all20 aliases remain. Original Geometry and detector encoders are not rerun; image-only masks and candidate score arrays are saved before label evaluation. SAM2 is encoded once per physical window and shared by LoveDA P/D.

All eight fixed-window runs completed on physical GPUs4-7, with96 unique image IDs and the unchanged window sequence. Exact source score/confusion recovery, alias/source identities, complete coverage and transition reconstruction are verified. Equal-domain mean: original Geometry43.262886, original box coupling42.917934, MaskLocalLikelihood43.654426, fixed probability blend43.690578, shuffled Geometry43.673482, primary Geometry_MaskLikelihood43.707858. Primary improves Geometry by0.444972pp and original box coupling by0.789924pp, but only wins3/8 domains. Its increment over mask-only is0.053433pp and over simple fusion0.017280pp. The fixed gate fails; no full-image rollout or retrospective control promotion.

The mask-coupled primary improves the prior box-coupled readout in every protocol, establishing that support imprecision matters for this operator. It still loses on OEM, UDD5, LoveDA P/D, LandCover.ai and FLAIR-1. A separate support audit reports score-weighted supported-label precision and correct-mass retention, not object AP or exact subpatch precision. Some classes improve precision after refinement, while substantial semantic mismatches remain. Greater precision can also result from reduced coverage. This does not establish that all remaining losses are caused by one semantic calibration defect.

Implementation: `dinotool/grounded_mask_observer.py`, `scripts/eval_grounded_mask_likelihood.py`, `run_grounded_mask_likelihood_suite.py`, `audit_grounded_mask_supports.py`. Eight new mask/replay tests plus15 existing likelihood/cache/sign tests pass locally and remotely. The post-prediction support audit initially stopped on `KeyError: 'boxcorrect_mass'`, a report-field concatenation error; no support or prediction changed, no result file was written, and the repaired audit completed for all eight. Full report: `GEOMETRY_MASK_LIKELIHOOD_SCREEN_20261002.md`; protocol and verified outputs remain under research. Cached costs include SAM2 but exclude original DINO/detector forwards, so they are not deployed-primary timing. The observed support improvement is progress, not a verified CVPR contribution or achieved eight-domain superiority.

For the user's current best-model question, the corrected full eight-domain `Anchored_VIP` reference remains48.023775 mean versus Geometry46.150775 and fixed MeanLogit_VIP47.351163. It improves four domains and harms four; it is a preserved performance reference, not a universally successful final module or complete official VIP reproduction. No new window metric is substituted for those full results. Original Geometry and all historical best models remain intact.

## Paired contextual head intervention

The user subsequently authorized implementation and testing of the proposed
Geometry-supported paired contextual observation. Signature
`geometry-paired-context-readout-v1-20261002`. Reuse frozen guarded VIP, wide448,
336/112, ImageNet/profile scoring and all20 aliases. Each local Geometry row's
above-uniform visual support is mapped to wide token query/donor support by
fractional original-coordinate patch overlap. Reference restricts both frozen
head blocks' VIP donors; full and reference share initial backbone tokens and
the full read's unchanged text-salience profile. Unmapped queries retain full
support; empty rows use self Value. Reference still contains backbone context
and indirect two-block effects, not pure context-free semantics.

Sample full-minus-reference class logits on each local grid. The primary solves
`min_delta .5||delta||^2+.5||A(delta-d)||^2` and adds delta to original Geometry.
It retains both signs, has exact zero-d identity, and equals `g+.5d` when A=I.
Geometry controls observation supports and writeback; standard quadratic solving
and the borrowed VIP operator are explicitly attributed, not claimed original.

The same96 fixed image IDs were evaluated with COMPLETE image inference on
physical A800 GPUs4-7: full40 UDD5 and8 each of seven other domains. This scope
differs from the preceding detector cached-window statistics. Corrected IRRG
Vaihingen was used. Eight new tests passed remotely; local syntax compilation
passed but local unit-test import lacked the preexisting OpenCV dependency.
A mask-free real-checkpoint smoke verifies exact full-source/head/profile replay
(maximum errors0), finite reference output and frozen weights. Complete unique
coverage, endpoint transitions, per-image matrix reconstruction and exact
original Geometry per-image confusions all pass. Five full40 UDD5 source/fusion
controls exactly reproduce their historical confusion matrices.

Equal-domain mean: Geometry43.830569, MeanProb_VIP44.900830,
MeanLogit_VIP45.400028, Anchored_VIP46.058780, same-information
MeanLogit_Context43.935994, ShuffledContextWriteback43.914756,
primary43.961350. Primary beats Geometry by0.130781pp with7/8 domain gains, but
trails Anchored_VIP by2.097430pp. Its increment over half-context is0.025356pp,
over shuffled writeback0.046594pp. VDD primary32.4305 versus anchored57.0012;
Potsdam40.6508 versus41.7806. Full40 UDD550.4548 versus Geometry50.5553 and
anchored49.2117. LoveDA P62.9356/D38.4653 versus Geometry62.8254/38.4558.

The prospective promotion gate failed. No full rollout, parameter adjustment,
dataset routing or additional encoder experiment was launched. The difference
did not retain the old semantic gains (VDD water35.4850 versus anchored92.4595;
Potsdam car11.8530 versus24.2128), while reducing several old negative transfers.
This does not establish that context differences certify correct corrections.
Suite elapsed581.3291 seconds; peak combined eight-arm allocation5836.0381MiB;
these are not independent primary latency/cost. A report-only absent-class null
IoU formatting error was repaired without altering outputs or rerunning jobs.

Code: `dinotool/paired_context_readout.py`, `scripts/eval_paired_context_readout.py`,
`scripts/run_paired_context_suite.py`; local launcher/collector
`tools/paired_context_experiment.py`. Protocol and full metrics/per-class/change
counts: `GEOMETRY_PAIRED_CONTEXT_PROTOCOL_20261002.md`,
`GEOMETRY_PAIRED_CONTEXT_SCREEN_20261002.md`; verified raw outputs downloaded to
`research/geometry_paired_context_screen_20261002/`. All workers/controller
terminal, physical GPUs4-7 idle, unrelated tmux sessions preserved. Historical
Geometry/Anchored_VIP remain unchanged; the larger original useful-module and
eight-domain complete-official-VIP research objective remains unachieved.

## Native encoder-position observation continuation

Following the user's latest physical GPU0-7 authorization, completed the pending
frozen GroundingDINO encoder-position observation and its fixed Geometry coupling.
Signature `geometry-encoder-position-grounding-v1-20261002`; primary
`Geometry_EncoderCoupling`. The checkpoint/revision, exact original round-robin
captions and all20 aliases/class are unchanged. Native `enc_outputs_class` before
object-query selection supplies phrase-mean sigmoid responses, validity-weighted
level interpolation and all-alias arithmetic means on the32x32 lattice.
`b=.07*log(class_probability)` supplies the existing unit-weight anchored solver.
Numerically unobserved positions recover original Geometry exactly, including
after coupling. LoveDA P uses the same D observation with exact class/alias
mapping. No semantic threshold, label-fitted weight or dataset routing.

Ten focused tests pass remotely; local syntax compilation passes. A mask-free
real-checkpoint VDD smoke verifies actual output fields, validity geometry,
finite alias probabilities, exact Geometry score replay and no-information
identity. All eight workers complete with96 unique image IDs and the original
one/two512 windows per image (UDD5 all40 IDs, not full-image UDD5 predictions).
Exact source confusions, original window identities and transition endpoints
are verified remotely and independently checked after download. All responses
and candidate scores precede target-mask loading. Additional frozen detection
pretraining and the standard quadratic solver are borrowed, not new inventions.

Equal-domain WINDOW mean: Geometry43.26288635, encoder-only17.15245727,
logit mean35.83803752, probability mean35.79879145,
shuffled Geometry coupling43.47941279, primary38.03813361.
Primary loses5.22475274pp vs Geometry and5.44127919pp vs shuffled coupling,
despite beating the two very weak simple fusions by2.20009609/2.23934216pp.
Only Vaihingen improves; the other seven domains and both LoveDA protocols
decline. The prospective gate fails; no full-image rollout or source-scale,
level/phrase aggregation, solver-weight or confidence search follows.

VDD roof79.4937->44.1250 and wall38.3866->15.4296; wall gains738,439 false
positives while roof loses385,633 true positives. Potsdam car11.7333->39.9601
comes with439,467 fewer false positives but3,325 fewer true positives;
building77.4448->49.1691 loses19,655 true positives and adds266,921 false
positives. Thus an isolated car gain is not a demonstration of recovered small
objects. Native detection-position scores, read as dense semantic evidence,
are a poor source under this fixed all-alias measurement. The actual Geometry
structure does not repair them and is worse than spatially shuffled coupling.
This rejects the fixed source/readout route, not all frozen grounding models
or Geometry itself. It does not uniquely identify semantic alignment, calibration
or spatial interpolation as the sole cause.

Parallel suite wall80.9499s; maximum allocated1692.5430MiB across workers.
Costs include the detector's unused decoder and all cached readouts, excluding
original DINO forwards, initialization and label audit; not deployed-primary
latency/memory. Final read-only checks confirm all workers/controller terminal,
GPUs0-7 idle and unrelated tmux sessions intact. Full report:
`GEOMETRY_ENCODER_GROUNDING_SCREEN_20261002.md`; protocol:
`GEOMETRY_ENCODER_GROUNDING_PROTOCOL_20261002.md`; verified metadata:
`research/geometry_encoder_grounding_screen_20261002/`. Raw probabilities/scores
remain in the matching remote results root. Code:
`dinotool/encoder_grounding_observer.py`, `scripts/eval_encoder_grounding.py`,
`scripts/run_encoder_grounding_suite.py`, `tools/encoder_grounding_experiment.py`.
Original Geometry and the historical full-eight-domain Anchored_VIP reference
remain unchanged. The useful original coupled-module/CVPR objective is still
active and unachieved; this diagnostic does not constitute a final model.

## Cross-view matched Value innovation continuation

Implemented one VIP-independent internal candidate using the same frozen
DINOv3/DINO.text, signature
`geometry-cross-view-value-innovation-v1-20261002`; primary
`Geometry_ValueInnovation`. Each fine512 window receives one enclosing1024->512
context view. Context donors follow original Geometry; current fine Queries
read local/context learned Keys/Values conditioned on normalized valid Geometry
priors. The partition-function share multiplies their matched Value difference,
which is added to original Geometry's native-mass-scaled patch read before the
projection/residual/MLP in both blocks. One final fine descriptor, all20 aliases,
original RS/LME/assembly; no external teacher, alias deletion, dataset routing or
label-fitted coefficient. Original special pathways remain active, not invariant
to indirect changes in later blocks.

Eleven focused tests pass locally/remotely. Actual-checkpoint mask-free smoke
verifies exact duplicate/no-context identity, original context Geometry replay,
zero direct prefix/invalid-query displacement, finite output and unchanged full
head fingerprint. The first smoke's inference-tensor version-counter failure
was repaired in validation only; its log remains. This is not a model repair or
post-result parameter change.

The full-image development screen completed on physical A800 GPUs0-7: UDD5
all40 images and eight unchanged images from each of seven other domains,
96 unique complete-image predictions. These are not the detector window-only
statistics and not full eight-domain totals. Corrected IRRG Vaihingen is used;
LandCover.ai substitutes for unlabeled iSAID. All eight have informed development.
Remote and independent local checks verify exact per-image original Geometry,
SCLIP_Two and VIPProxy_Two confusion replay against the matched full references,
complete unique IDs, frozen config/vocabulary/checkpoints, per-image sums and
old/new/truth transition endpoints. Masks load only after predictions.

Equal-domain means, counting LoveDA D once: Geometry43.830569,
SCLIP_Two43.782393, VIPProxy_Two43.449702, ContextGeometry41.012051,
same-source MeanLogit_ContextGeometry43.902207, shuffled coupling35.599071,
primary43.754654. Primary loses0.075915pp versus Geometry,0.147553pp versus
simple fusion and2.304126pp versus the sample-matched historical Anchored_VIP
screen46.058780. Historical Anchored_VIP changes the view/text/readout and cost;
it is not an equal-information operator ablation or complete official VIP.

Primary improves5/8 domain scores, but loses Potsdam2.085869pp,
Vaihingen1.711530pp and FLAIR-10.588313pp. Full40 UDD5 is50.812524 versus
Geometry50.555301 and same-source fusion51.564421. LoveDA P64.910200/D39.090965
versus original62.825368/38.455828. VDD33.719331 versus original31.946180,
context alone38.656673 and simple fusion36.420913; the internal writeback does
not retain that source's useful gains. Potsdam38.266967 versus original40.352836
and fusion39.712482. The prospective frozen gate fails; no full rollout or
retrospective share/support/layer search is launched.

Potsdam car11.6582->10.1338, predicted area19.8762->22.9335%,
recall99.0262->99.2775%. It gains471 true positives and244,113 false positives.
Low vegetation33.4221->27.7111 loses114,608 true positives. UDD5 vehicle also
adds778,147 false positives for3,199 additional true positives. Thus this is
not demonstrated small-object recovery. Aggregate class statistics do not
establish the cause or instance-size dependence of every error.

The actual block algebra exposes a limitation: original conditional read
`G_f V_f` subtracts `alpha P_f V_f`, with
`P_f=softmax(log G_f+QK_f)`, not `alpha G_f V_f`. Effective fine coefficients
`G_f-alpha P_f` may be negative, and native QK is reintroduced into the path
Geometry originally rewrote. Normalized priors remove token-count bias, not
cross-scale learned-Key calibration. Partition-function shares around0.43-0.55
on average are not semantic trust measurements; averages do not show the full
querywise distribution. Beating shuffled coupling establishes sensitivity to
correspondence, not correct semantic corrections. This analysis rejects the
fixed operator without proving that Geometry or every cross-view approach has
reached a ceiling. Cross-attention/geometric priors/differential reads have
precedents; no new verified novelty gap or CVPR readiness claim follows.

Parallel suite wall931.3596s, maximum combined seven-arm allocation3957.2573MiB.
A separate mask-free one-resident-model benchmark on idle physical GPU1 uses
three warmups and20 synchronized trials: Geometry24.9397ms/3715.7246MiB,
primary84.5616ms/3827.2959MiB, descriptor latency ratio3.3906x. These timings
include as-implemented fine/context preparation but exclude text scoring,
assembly, decoding and initialization; they are neither optimized nor
whole-image latency. Final read-only checks confirm all suite/benchmark sessions
terminal, GPUs0-7 idle without compute processes, and unrelated sessions
`controlled_index` and `qlift_smoke_matched_20260921` intact.

Full report and regenerating interpretation:
`GEOMETRY_CROSS_VIEW_VALUE_SCREEN_20261002.md`,
`tools/cross_view_value_experiment.py`; protocol:
`GEOMETRY_CROSS_VIEW_VALUE_PROTOCOL_20261002.md`.
Verified merged/per-image results, manifests, logs, smoke and independent timing
are downloaded under `research/geometry_cross_view_value_screen_20261002/`.
Model/test/evaluator files are `dinotool/cross_view_value_innovation.py`,
`tests/test_cross_view_value_innovation.py`,
`scripts/eval_cross_view_value_innovation.py`,
`scripts/run_cross_view_value_suite.py`, `scripts/benchmark_cross_view_value.py`.
Original Geometry, full-eight-domain historical Anchored_VIP and all failed
outputs remain unchanged. The useful original complete-module/eight-domain
official-VIP research objective remains active and unachieved.

## Bounded Geometry residual semantic metric continuation

The preceding cross-view experiment is classified as progress: it completed a
new internal candidate, exact paired evidence and independent cost, while
rejecting its promotion. This continuation implemented a different complete
single-view readout, `geometry-residual-semantic-metric-v1-20261002`, primary
`Geometry_ResidualMetric`. It does not change the original backbone/head,
replace Values, execute VIP in the primary or acquire another context view.

Original Geometry produces normalized Y and G. The uncentered union of all
encoded20-alias banks supplies an orthonormal numerical text span B. The
image-valid residual R=(I-G_valid)YB gives C=R^T R/n. M=I+C/trace(C) changes
BOTH visual/text cosine directions within that span, preserving perpendicular
components and original uniform alias LME/assembly. Its inverse eigenvalues
lie in[1/2,1]. Zero covariance/identity Geometry/empty validity recover exact
original alias scores. A fixed fp32 roundoff bound prevents constant-field
roundoff from becoming an amplified metric. This guard was added before any
candidate metric was observed; no strength or class rule was tuned afterward.
Network/text weights are frozen, but the window covariance is inference
adaptation, not adaptation-free inference. Mahalanobis metrics, whitening and
graph residuals have precedents; no verified priority claim is made.

Seventeen focused tests pass locally and remotely, including the explicit
full-space metric, coordinate rotation, padding, duplicated alias span and
numerical neutrality. Actual-checkpoint mask-free smoke verifies zero original
score/identity-Geometry error, unchanged head fingerprint and finite output.
All eight complete-image workers finished on physical GPUs0-7: same96 IDs,
UDD5 full40 and other seven domains8 each. Corrected IRRG Vaihingen and
LandCover.ai substitution remain. The frozen screen is development data, not
eight complete-dataset SOTA results. Exact original Geometry/SCLIP/VIPProxy
per-image replay, unique coverage, all20/checkpoint/config identity, confusion
sums and transition endpoints pass remotely and independently after download.

Equal-domain mean, LoveDA D once: Geometry43.830569, SCLIP_Two43.782393,
VIPProxy_Two43.449702, UniformMetric42.971138, SpatialMetric43.137402,
ShuffledMetric43.256352, MeanLogit_UniformMetric43.500753, primary43.722164.
Primary loses0.108405pp versus Geometry, despite gaining0.751026pp over the
global metric and0.221411pp over its half-logit fusion. Only VDD(+0.056938)
and FLAIR-1(+0.001661) improve; the other six domains and LoveDA P decline.
Potsdam40.115262 versus original40.352836; UDD5 full40 is50.491959 versus
50.555301. LoveDA P62.496008/D38.320862 versus62.825368/38.455828.
Foreground: LoveDA D38.9808 versus39.1654; UDD555.4811 versus55.5480;
LandCover.ai55.0477 versus55.4788. The fixed gate fails; no full rollout,
covariance/rank/strength search or per-domain control promotion follows.

Potsdam car11.6582->11.4718 loses43 true positives and adds25,500 false
positives; low vegetation loses9,021 TP and tree15,925 TP. VDD vehicle loses
3 TP and adds124,532 FP while roof/water coverage increases. UDD5 vehicle
loses255 TP and adds865,605 FP. A small VDD average gain is not a repair of
the inherited vehicle competition bias or an independent small-object result.

This rejects the fixed hypothesis that original Geometry residual covariance
is a useful semantic nuisance metric. True class boundaries/detail can enter
the residual, and coherent wrong responses can be explained by G. A constant
wrong descriptor field has exact zero residual and unchanged predictions.
Cosine renormalization can increase as well as decrease alias scores despite
contractive feature metric bounds. Better performance than harmful global
covariance shows less damage, not improvement over the original model or a
certified semantic controller. This does not prove every metric/coupling or
training-free method has reached a ceiling; the next action must supply or
identify useful competing-class information rather than rename a weak control.

Suite wall646.1968s, maximum eight-arm allocation3822.0474MiB. A separate
mask-free single-resident-model GPU1 benchmark uses three warmups/20
synchronized trials: Geometry26.7870ms, primary28.9636ms, ratio1.0813x;
both peak3717.6416MiB. It times original512 preparation plus all20 alias
scoring/LME and the primary covariance/Cholesky/cosine, not comparator arms.
Text encoding/span initialization, decoding, dense image assembly and model
initialization are excluded; these are not whole-image or optimized timings.
Final read-only checks find all suite/benchmark sessions terminal, GPUs0-7
idle with no compute processes, and unrelated tmux sessions intact.

Code: `dinotool/geometry_residual_metric.py`,
`tests/test_geometry_residual_metric.py`,
`scripts/eval_geometry_residual_metric.py`,
`scripts/run_geometry_residual_metric_suite.py`,
`scripts/benchmark_geometry_residual_metric.py`.
Launcher/collector/independent report generator:
`tools/geometry_residual_metric_experiment.py`.
Prospective protocol and verified full result report:
`GEOMETRY_RESIDUAL_METRIC_PROTOCOL_20261002.md`,
`GEOMETRY_RESIDUAL_METRIC_SCREEN_20261002.md`.
All merged/per-image results, exact manifests, logs, smoke and cost are
downloaded under `research/geometry_residual_metric_screen_20261002/`.
Original Geometry/historical best models and failed outputs are unchanged.
The useful original complete coupled-module and CVPR objective remains active
and unachieved; this screen is evidence, not a successful final model.
