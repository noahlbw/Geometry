# Pixel-support localization measurement: prospective complete candidate

This is a successor test of the remaining support ambiguity after the sign audit, not a claim of a final CVPR contribution. Only physical A800 GPUs4-7 are authorized. Original Geometry and historical results remain unchanged.

The existing all20-alias GroundingDINO queries have useful localization mixed with harmful semantic boosts. Positive-only updates fail every domain. The audit does not distinguish loose rectangular supports from incorrect phrase response or likelihood calibration. A frozen SAM2 observation tests the spatial-support part while holding original phrase confidences and the complete signed likelihood rule fixed.

## Frozen operator and controls

Use official public `facebook/sam2.1-hiera-tiny`, revision `de431c4043854a71d8101e17995dfe596bf101a5`, safetensors and standard Transformers only. It adds pretrained segmentation supervision on top of GroundingDINO's detection supervision. SAM2 and Grounded-SAM are borrowed components, not our innovation. No image, label, prompt or credential is uploaded. No weights are fitted on target images.

For each original512 window, encode the image once; decode one SAM2 mask per unchanged detector query with phrase-mean response greater than the existing0.25 threshold. No NMS, query cap, alias deletion, mask-score selection or per-domain rule. Single-mask decoding and batches of16 are fixed. Interpolate official mask logits to512, sigmoid, and average in the original16px patches. Multiply by the original fractional box intersection. A box whose valid patch support is constant is retained exactly as its original box; it supplies no localization and must not become an arbitrary salient-object mask.

With the observed support M, use exactly the existing signed readout: `h=M*(G*1[M>0])`, `D=N*h/sum(h)`, `e_a=(1+sum(s*D))/(1+sum(s))`, all20 arithmetic class mean, `g'=g+.07*log(e_c)`. No clipping or newly fitted gain. Empty/constant supports are neutral. G=I must recover the mask-only control exactly. This holds measurement identity fixed, but replacing an imprecise mask may still worsen semantic competition; a support is not a correctness certificate.

Controls: original Geometry, original box-only likelihood, original Geometry-conditioned box likelihood, MaskLocalLikelihood, fixed probability-mean blend of Geometry and MaskLocalLikelihood, spatially shuffled Geometry with the same mask observations, and primary Geometry_MaskLikelihood. This separates source utility, Geometry coupling, and simple fusion. It is not a complete official-VIP reproduction or a single-encoder fair comparison.

## Scope, audit and decision

Reuse exactly96 image IDs and one/two cached image-only windows each:40 UDD5 and eight from each other domain. This is NOT full-image dataset mIoU; windows may overlap. Corrected IRRG Vaihingen is mandatory; LandCover.ai replaces unlabeled iSAID. All eight domains are development data. Label masks are loaded only after all new supports and candidate scores have been saved. Original Geometry and both original box readouts must reproduce source confusion matrices exactly. Verify alias counts20, sample uniqueness, source revision, vocabulary SHA, coverage, transitions and finite supports. No count/threshold/source search follows labels.

Prospective promotion: primary retains Geometry mIoU on every protocol, beats mask-only and its fixed simple fusion in equal-domain mean, and beats mask-only and fusion on VDD/Potsdam. LoveDA D enters the mean once; P is separate. Beneficial/harmful and wrong-to-wrong changes are diagnostic, not an mIoU objective. A passed window screen only authorizes an actual full-image screen with the same rule, not an immediate SOTA claim. No automatic full evaluation is launched by this controller.

If the fixed primary fails, retain outputs and report whether improved spatial support repaired the positive observations; do not promote the retrospectively best control. Independent original-model/detector costs remain excluded from cached timing, while SAM2 observation cost and memory are included. Novelty requires a genuinely useful Geometry-coupled mechanism beyond this borrowed observer and ordinary likelihood update, and remains unproven.

## Verified execution

All eight fixed-window runs completed on physical GPUs4-7. Eight new mask/replay tests and15 previous likelihood/cache/sign tests pass locally/remotely. Real-image smoke loads no labels, passes exact source-score recovery and actual-mask identity Geometry, and SAM2 image-model weight loading is exact. Original Geometry and both original box-readout scores/confusions replay exactly on every protocol. Supports, scores, coverage and transitions were verified; separate label-only support auditing changes no readout.

Primary mean43.707858 versus Geometry43.262886 (+0.444972pp), original box coupling42.917934 (+0.789924pp), mask-only43.654426 (+0.053433pp), simple blend43.690578 (+0.017280pp). Three domain wins versus Geometry; five losses, including LoveDA P/D. The prospective gate failed; no full-image evaluation or control selection. Fixed-window results and complete class/support audits are in `GEOMETRY_MASK_LIKELIHOOD_SCREEN_20261002.md`. An initial support-report field-key typo was repaired without altering any observations, inference or labels. The research goal remains active and unachieved.
