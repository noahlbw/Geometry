# Geometry-conditioned frozen regional semantic readout

Date: 2026-10-01. This is a prospective exploratory experiment, not a verified final model or SOTA claim. The user authorized A800 physical GPUs0-7. The existing Geometry and previous outputs are preserved.

## Information source and complete candidate

Use the original frozen DINOv3/DINO.text Geometry with two modified blocks, tile512/stride128, interpolation before temperature, Hann probability blending, and exactly the historical20 aliases per class. Add one independent frozen public SigLIP2 encoder: `google/siglip2-base-patch16-256`, revision `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`. Downloaded weights SHA256: `6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863`. This changes the scope to a two-encoder training-free method; the extra encoder itself is not claimed as an innovation.

Partition the original raw DINO token grid into all connected fine64 and coarse16 SLIC-zero supports. Keep even single-token supports; there is no class-based top-K region selection. Within a support, average original Geometry relation rows and restrict the resulting weights to that support. Normalize overlapping fine/coarse writeback weights per pixel, so extra regions do not create extra evidence mass.

The semantic observation uses actual unmasked RGB. For each support read the original512 context and an enclosing detail crop with minimum64px and16px margin, resized to the encoder's trained256 input. Read the support's token values through SigLIP2's pretrained probe, attention projections, residual and MLP, adding the log support weights to attention. Whole-crop descriptors are retained as a same-budget control. Tokens can still carry contextual information from the backbone; this is not an isolation or causality claim.

Encode the same20 aliases and same six RS templates through SigLIP2's text encoder. Geometry retains its0.07 temperature; SigLIP2 uses its own frozen pretrained logit scale (`4.726512908935547`, corresponding temperature about0.00886), including normalized alias log-mean-exp. This is pretrained scoring, not a coefficient fitted to target labels. The first single-image execution check omitted this source temperature; its output is preserved separately and is not used to select parameters or report performance.

Average detail/context class probabilities into regional compatibility q. Entropy and normalized Jensen-Shannon disagreement produce a continuous influence weight. They are uncertainty measures, not correctness probabilities or statistical independence. No alias is deleted.

The primary `RegionJoint` jointly infers pixel probability p and regional latent class prior z by minimizing:

`sum_i KL(p_i || g_i) + sum_r n_r KL(q_r || z_r) + sum_ri W_ri KL(p_i || z_r)`, with `n_r=sum_i W_ri`.

The alternating block updates are exact minimizers: p uses the weighted geometric mean of Geometry and region priors; z uses the weighted arithmetic mean of q and the supported pixel probabilities. Fixed32 iterations /1e-5 numerical tolerance. No target-label weights. z is a regional class prior, not a predicted pixel-area occupancy. No influence gives exact original Geometry scores. The objective is standard inference machinery; any potential contribution must come from the observation/Geometry coupling and verified gains over equal-information controls.

## Predeclared controls and screen

Methods: `Geometry`, `CropGlobalFusion`, `RegionSemantic`, `RegionFusion`, `RegionJoint` (primary). Fusion and joint use the same region observations, the same influence weights and the same computation. CropGlobalFusion reads the whole crop with the same frozen head and inputs. RegionSemantic exposes whether the observer recognizes supported content without the Geometry prior.

| Physical GPU | Dataset | Screen images | Available full images |
|---|---|---:|---:|
|0|VDD|8 prior fixed diagnostic images|80|
|1|Potsdam|8 prior fixed diagnostic tiles|504|
|2|UDD5|40, full set|40|
|3|OEM|32 fixed-seed images|384|
|4|LoveDA|16 fixed-seed images, both P/D|1669|
|5|Vaihingen|16 corrected-input tiles|113|
|6|LandCover.ai|32 fixed-seed patches|1602|
|7|FLAIR-1|64 fixed-seed images|15700|

Non-diagnostic samples use seed20261001, selected from verified full sample keys before any predictions. Each dataset has one shard. Target labels enter only after complete image prediction for metrics, old/new/truth transitions and audits of two label-independently chosen windows per image. These repeatedly used datasets are exploratory development, not untouched validation. iSAID has no labeled validation here and is not included.

The prospective promotion gate is: every screened protocol must retain Geometry mIoU and have positive net corrected pixels; the primary must beat same-information RegionFusion on both VDD and Potsdam. Do not choose different arms by domain. A failed gate stops promotion; a passed screen still requires full-set validation and independent scene testing. The controller does not automatically start full datasets.

Root: `results/geometry_region_siglip2_screen_v2_20261001`. Sessions: `grsr02_screen_DATASET`. The controller merges unique complete sample keys and transition endpoint matrices, checks vocabulary/checkpoint signatures, and verifies exact previous Geometry confusion matrices for full UDD5 and the matched VDD/Potsdam diagnostic sets. Other partial screens do not have a saved historical per-image confusion matrix; this limitation is recorded rather than claiming full baseline replay verification.

## Implementation and verification

- Model: `DINOtool/dinotool/region_semantic_readout.py`.
- Evaluator: `DINOtool/scripts/eval_region_semantic_readout.py`.
- Launch/controller: `DINOtool/scripts/run_region_semantic_suite_a800.py`.
- Nine unit tests pass: native-pool replay, support-external value invariance, empty support rejection, uniform-source influence, exact no-evidence fallback, solver energy/probabilities, class-permutation invariance, one-token support/padding preservation, and border crop/support coordinate coverage.
- First OEM execution check finished9 windows, all support crop masks usable, peak5254.96MiB. Its one-image mIoUs are not selection evidence or full results.

Signature: `geometry-regional-siglip2-trained-pool-v2-native-scale-20261001`.

## Verified completed screens

All eight datasets have complete, unique, coverage-verified merged results downloaded under `research/geometry_region_siglip2_screen_v2_20261001/`. All eight actual trained-pool full-support replay errors are exactly0 under the recorded bf16 protocol. The VDD/Potsdam original Geometry confusion matrices exactly match the earlier fixed diagnostic runs, and full UDD5 exactly matches its historical Geometry confusion matrix. Only UDD5 uses its complete40-image set; the other seven datasets are partial-set screens, not full-set benchmark numbers.

| Dataset/protocol | Images | Geometry | RegionFusion, same information | RegionJoint | Delta vs Geometry |
|---|---:|---:|---:|---:|---:|
| VDD |8|46.5339|44.0173|46.4368|-0.0971|
| Potsdam |8|42.5695|40.5963|42.8216|+0.2521|
| UDD5, full |40|50.5553|46.8464|48.3234|-2.2319|
| OEM |32|44.3423|35.3765|37.7244|-6.6179|
| LoveDA P |16|63.9207|59.7521|64.1685|+0.2478|
| LoveDA D |same16|43.2600|41.1120|42.5842|-0.6758|
| Vaihingen, corrected IRRG |16|48.8339|37.4341|40.8430|-7.9909|
| LandCover.ai |32|53.9329|48.5072|50.3626|-3.5703|
| FLAIR-1 |64|46.8183|42.7711|45.3678|-1.4505|

Full UDD5(40) has completed and is merged and verified. Its CropGlobalFusion and RegionSemantic mIoUs are44.6591 and43.0758; RegionJoint has3,027,066 more harmful than beneficial pixel changes. The controller has finished with `status=complete`, `gate_passed=false`, `failures={}` and `automatic_full_launch=false` in the downloaded `suite_results.json`. All eight runs ended normally; the final read-only server check found GPUs0-7 idle. The candidate fails the predeclared gate and is not promoted to full eight-dataset evaluation.

### Observation and correction diagnosis

On the label-independent support-audit windows, restricted regional pooling recognizes supported content more often than whole-crop classification, but still substantially less often than Geometry:

| Screen audit | Geometry dominant matches | Whole-crop observer | Regional observer | Joint regional posterior |
|---|---:|---:|---:|---:|
| Potsdam |897/1255|499/1255|718/1255|751/1255|
| LandCover.ai |2302/2499|1795/2499|1960/2499|2055/2499|
| Vaihingen |1702/2482|760/2482|1284/2482|1341/2482|
| FLAIR-1 |3453/4996|2193/4996|2321/4996|2450/4996|

These overlapping fine/coarse supports are not independent samples. Dominant matching measures the selected support's principal class, not every minority object or pixel. Their target labels were accessed only after all predictions. The counts are diagnostic and were not used to change the current rule.

Potsdam car IoU rises5.8594->9.7026, precision5.8594->9.7028%, recall100->99.9826%. Low vegetation falls65.3138->63.0169 and tree51.6910->51.4290. Beneficial668,595/harmful456,798/wrong-to-wrong435,289. This reduces some false car occupation but does not demonstrate small-target miss recovery or full504-tile improvement.

Vaihingen has422,897 beneficial versus1,584,457 harmful changes. LandCover.ai186,799 versus378,508; FLAIR-1 635,782 versus1,232,570. LoveDA P's small mIoU increase coexists with376,895 beneficial versus402,525 harmful changes, so it fails the positive-net correction condition. LoveDA P/D are two protocols on the same16 images, not separate datasets.

The joint candidate beats same-information RegionFusion on every completed protocol. That verifies better handling of this source relative to the frozen fusion control; it does not establish a universally reliable semantic observer or a successful final model. The regional source's much weaker support identity and standalone RegionSemantic segmentation are evidence against promoting this specific base256/pretrained-pool observation. They do not prove that every independent frozen encoder, larger checkpoint or alternative region representation is unusable.

No parameter was selected by dataset, no aliases were deleted, and no SOTA/CVPR-ready claim is warranted. Preserve original Geometry as the primary research model. The complete eight-domain screen rejects this particular regional observer and joint-readout candidate; no per-domain winners are selected. All currently launched runs are terminal and reported.
