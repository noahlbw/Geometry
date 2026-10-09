# Geometry-conditioned localized likelihood hypothesis

The prior raw BoxMean diagnostic FAILED; its gate is not reinterpreted as passed. A complete revised measurement/readout is now specified prospectively. The revision is motivated by the previous labeled development audit and a subsequent label-free extent audit, so it is NOT independent validation. No original threshold, alias selection or per-domain parameter tuning is introduced. The earlier raw rasterization results remain intact.

## One coupled operator

Original frozen DINOv3/DINO.text -> unchanged Geometry scores g and relation G -> fixed frozen GroundingDINO all20-alias localization observations -> Geometry-conditioned within-box spatial density -> likelihood product with the original Geometry posterior -> unchanged dense/Hann assembly.

For a box r let b_r contain its fractional intersections with valid16-pixel patches, and m_r=1[b_r>0]. Restrict G to valid patches and row-normalize. Define `h_r=b_r*(G m_r)` and `D_r=N*h_r/sum(h_r)`. D_r has spatial mean1 and no support outside that particular observed box. The no-Geometry control uses h_r=b_r. G=I recovers that control exactly. A whole-window support gives D_r=1, regardless of detection confidence.

For every alias a, keep the same detector phrase-mean score s_ra and original processor-default cutoff0.25. Its observation is `e_a=(1+sum_r s_ra*D_r)/(1+sum_r s_ra)`. The unit uniform pseudocount makes an unobserved alias exactly neutral, avoids zero likelihood and prevents a missing detection from becoming an absence veto. It is a prospective single pseudocount, not fitted reliability. All20 alias densities are arithmetically averaged for each class: e_c=mean_a e_a. Each class density retains spatial mean1, so detection counts or numerous aliases do not create a class's spatially uniform semantic boost.

The sole readout is `g'_ic=g_ic+0.07*log(e_ic)`, equivalently `p'_ic proportional to p_Geometry(ic)*e_ic`. Temperature0.07 is inherited from the original Geometry posterior, not a new fitted influence coefficient. Empty detections and constant whole-window observations preserve original scores exactly. Original aliases, text templates, crop geometry, frozen weights and label mapping remain unchanged. The detector's native processor resize remains an additional observation cost.

This is a relative localization-likelihood hypothesis. Lower likelihood outside positive detected locations does NOT certify class absence; incomplete detections can harm recall. Boxes can still mix semantic classes, false small boxes can be amplified, and class-density normalization cannot certify correctness. G only changes where observed likelihood is concentrated inside a box, not the observation's semantic identity. It is not a consistency-based correctness gate or segmentation guarantee.

## Controls and decision

Geometry; BoxLocalLikelihood (identical source/update without G); fixed50/50 probability mean of Geometry and BoxLocalLikelihood; spatially shuffled Geometry density; primary Geometry_LocalLikelihood. The uniform pseudocount, confidence cutoff, update temperature, alias aggregation and Geometry rule are frozen across all domains. Standard Bayes updates, density normalization and pseudocounts have precedent; no novelty claim is inferred from their assembly. Additional detection pretraining requires separate complete-system/resource comparisons, not matched single-encoder claims.

First use the already saved96 images' one/two original windows, without any detector reruns. Compute and persist all candidate patch scores from image-only raw data before loading any masks. Evaluate dense window predictions and exact old/new/truth transitions afterward. This is window-only development evidence, NOT full-image mIoU. Tests cover empty/uniform neutrality, G=I control, alias coverage, positivity, padding exclusion, density means and normalized posterior product.

Prospective promotion requires primary nondegradation versus Geometry in every window protocol (LoveDA P/D separately), positive net corrected pixels in every protocol, and an equal-domain mean above both BoxLocalLikelihood and the fixed probability blend, with primary gains over BoxLocalLikelihood on VDD/Potsdam. The mean counts LoveDA D once. A passed window gate requires an actual-checkpoint no-mask smoke followed by unchanged full96-image readout screening, including matched nearest controls, BEFORE full20,092-image rollout. A failed gate rejects this fixed hypothesis; do not tune pseudocount, thresholds, word counts or class exceptions from masks.

The original useful-coupling/CVPR/eight-domain complete-official-VIP objective remains unchanged and unachieved. SAM3 public metadata is gated, and no SAM3 weights were downloaded, loaded or used. Scholarly Search currently lacks authorization; its failed response is saved. No claim of a literature gap or verified originality is made.

## Numerical replay repair

The first cached suite is retained under `geometry_localized_likelihood_cached_20261002`. Its LoveDA audit failed exact original-Geometry replay. A mask-free check established that applying FP32 softmax before argmax changes exactly one D-window pixel and zero P pixels relative to the source diagnostic's direct interpolated-score argmax. The repair uses the same source direct-score argmax for Geometry and likelihood readouts; the probability-mean control still averages dense probabilities. Candidate score arrays, equations, source observations, images, aliases, pseudocount, thresholds and temperature are unchanged. A near-tie regression test covers the numerical convention. The clean repeat writes only to `geometry_localized_likelihood_cached_r2_20261002`.
