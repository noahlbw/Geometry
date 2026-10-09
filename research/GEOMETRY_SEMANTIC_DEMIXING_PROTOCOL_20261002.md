# Frozen Geometry semantic-demixing experiment

## Objective and boundary

The research goal is a useful original Geometry coupling and one fixed model that eventually surpasses complete VIP across eight remote-sensing domains. This experiment does not establish that outcome or novelty. Astra's previously tested donor-budget, role, observer and unit-lifting candidates failed; they are preserved, not relaunched.

The new hypothesis is that independent alias similarities conflate correlated class explanations. Joint nonnegative decoding might separate which class actually explains the final Geometry descriptor. Geometry adds a residual-observation metric, not a label agreement term or a majority-vote teacher. This successor operationalizes the pending competitive-reconstruction investigation; it is not a claim that Astra endorsed this exact solver.

## Complete candidate

Frozen DINOv3/DINO.text -> unchanged original Geometry -> Y, G -> joint all-alias nonnegative explanation -> norm of each class's explained descriptor -> original dense interpolation and Hann probability accumulation.

For valid patches only, Y is the original final L2-normalized descriptor, T contains all 20 normalized text aliases per class, and G is the valid-restricted original Geometry relation:

`min_X>=0 0.25*(||Y-XT||_F^2 + ||G(Y-XT)||_F^2) + 0.5*0.01*||X||_F^2`.

The class score at each query is `||X_c T_c||_2`. This is explained descriptor mass, not a calibrated probability. Coefficients can become exactly zero, but no alias is globally removed, no top-K count is enforced, and identical aliases belonging to different classes remain fundamentally ambiguous. All background/other classes remain competitors.

If Y is exactly explained, the geometric term is zero even when adjacent class assignments differ. Therefore it does not directly penalize small-object boundaries. This algebraic property does not guarantee semantic correctness or small-object improvement. G contains no new semantic information; coherent wrong features may still fail.

Fixed ridge 0.01; restarted monotone FP32 FISTA, maximum2048 steps, check every8, relative projected KKT tolerance5e-4. These are prospective numerical settings, not target-label-selected values. Failure of convergence is reported, not hidden as a completed result.

The initial mask-free actual-checkpoint smoke exhausted512 steps at KKT0.0006799167, above the unchanged0.0005 target. Before consulting any segmentation metrics the numerical iteration cap was raised to2048. Objective, ridge, scoring rule and tolerance were not changed; the failed smoke log is retained.

The first screen stopped after numerical failures (including shuffled controls at KKT0.0008273832/0.0008697945 and a LoveDA primary at0.0008717412). FP32 subtraction of nearly equal objective values caused excessive backtracking. The numerical repair uses the exact quadratic curvature inequality and FP64 scalar inner products, not differences of total objectives; KKT uses the direct active/inactive constraint residual. Model objective, scoring, ridge, data and tolerance are unchanged. A fresh r2 root preserves the initial failed screen. No performance-driven semantic change is made.

## Controls and verification

Geometry; matched SCLIP_Two and VIPProxy_Two; TextOnlyDemix (same decoder without the Geometry residual observation); fixed 50/50 Geometry/TextOnlyDemix logit average; spatially shuffled Geometry residual relation; primary Geometry_SemanticDemix.

G=I must recover TextOnlyDemix exactly. It does not recover original Geometry, since semantic decoding itself changes the readout. Zero explained evidence and invalid patches retain original Geometry scores exactly. All caches, weights, checkpoints, vocabularies and original baseline predictions must remain unchanged.

Synthetic tests compare both local and coupled solutions against SciPy NNLS. Actual-checkpoint smoke uses no masks, tests FP32/bf16, and reports primary-only synchronized cost separately from the combined evaluation.

## Prospective decision

Use the already fixed96-image development screen: full40 UDD5; eight images each from VDD, Potsdam, OEM, LoveDA, corrected-IRRG Vaihingen, LandCover.ai and FLAIR-1. Only physical A800 GPUs4-7 are authorized. No other job is stopped.

Promotion requires no loss versus Geometry in any screened protocol (LoveDA P and D separately), a higher equal-domain mean than every control, and gains over Geometry, matched VIP proxy, text-only decode and simple fusion on both VDD and Potsdam. The mean counts LoveDA D once, with P separate. If it passes, the same frozen rule is automatically promoted to full80/504/40/384/1669/113/1602/15700 coverage. Otherwise preserve the negative result and do not tune coefficients from masks.

VIPProxy_Two is a matched operator control, not complete official VIP. Full-system VIP superiority still needs reconciled complete-system results. All eight datasets are development data. LandCover.ai substitutes for unlabeled iSAID. NNLS, residual metrics and FISTA have extensive precedent; novelty cannot be inferred from assembling them.

Literature service authentication is unavailable in this session. No new verified novelty claim is made. No external teacher or new checkpoint is introduced.
