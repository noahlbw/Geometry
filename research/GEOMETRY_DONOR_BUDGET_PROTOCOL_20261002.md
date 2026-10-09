# Geometry and native visual donor budgets: prospective complete candidate

Signature: `geometry-native-donor-marginal-v1-20261002`. Primary: `Geometry_DonorBudget`.
Only A800 physical GPUs4-7, when idle; do not disturb the still-running strong-observer screen. Original Geometry and historical models/results remain unchanged.

## Question and coupled architecture

Geometry's raw relation is shared across semantic heads. It changes not only each query's conditional patch reading but also the total attention delivered to each donor. Native attention contains head-specific collective donor roles even when its query-specific localization is poor. Can Geometry organize where to read while retaining those learned roles, without a second semantic teacher or a class quota?

This is a hypothesis. Existing error audits do not prove that false vehicle responses are caused by donor hubs. Most audited false responses can already have wrong local semantic direction. The candidate can fail when native donor budgets themselves are wrong or when the retained query residual is the main problem.

At each of the two frozen head blocks, compute the actual evolving native attention A, Values V, and original raw Geometry G. Original Geometry's valid patch flow is `Q_ij = m_i G_ij`, with native patch mass m. Valid query/donor masks exclude numerical padding from the displacement. Source row budgets are Q1. Native donor budgets are the valid patch-column sums of A, rescaled only to match Q's total mass. This compatibility rescaling also handles bf16 rounding and partial valid supports; it is not learned class calibration.

Solve the standard entropic projection:

`min_T generalized_KL(T || Q), subject to T1=Q1, T^T1=native_donor_budget`.

Write `o_new = o_original_Geometry + (T-Q)V`. Preserve special-query and special-key pathways, invalid query/donor displacement, output projection, LayerScale, residual, MLP and final normalization. The next block recomputes Q/K/V from the changed state. Only one final descriptor/class output is primary; no dataset routing or confidence gate.

When the target budget already equals Geometry's incoming mass, T=Q and the actual original Geometry execution is recovered exactly. In the ideal full-valid unrounded formulation, the summed patch read equals the native prescribed donor-weighted Value sum. This is a one-attention-operation property, not guaranteed preservation of final semantic bias, background calibration or mIoU after nonlinearities. bf16 arithmetic is explicitly checked and recorded rather than claimed exact in all conservation statements.

Use all20 aliases and six RS templates, normalized LME0.07, original512/128/Hann assembly. No target masks, text-specific capacities, absent-class allocation, external encoder, training objective or new learned network parameter. Numerical maximum1024 proportional-fitting steps and1e-5 mass-weighted L1 marginal tolerance are fixed before results. The initial no-label smoke exhausted128 log-domain steps at8.59e-5 error; its log is retained. The equivalent FP32 matrix-vector solver and larger numerical ceiling repair that execution issue without changing the objective or fitting labels. Numerical nonconvergence remains an execution failure, not silently accepted approximate semantics.

## Controls and frozen evaluation

Record Geometry, SCLIP_Two, VIPProxy_Two, UniformDonorBudget, MeanLogit_DonorBudget50% and the primary. UniformDonorBudget uses the same Geometry prior/solver with equal donor mass; it tests ordinary doubly-stochastic normalization against native semantic-head donor roles. The mean-logit arm weakens the same new observation under a predeclared fixed coefficient, not a post-result fitted gate.

Use the same existing image-only fixed screen: full40 UDD5 and eight samples each for the other seven domains,96 unique images. Screen UDD5 uses four shards; all others one. After a real-checkpoint no-label smoke, schedule on idle physical GPUs4-7 only. Verify exact original per-image Geometry/SCLIP/VIPProxy confusion arrays, unique complete sequence, all20 vocabulary SHA and unchanged checkpoints.

Promotion requires equal-domain mean greater than Geometry, matched VIPProxy, UniformDonorBudget and MeanLogit; maximum1pp loss against Geometry on every protocol; VDD/Potsdam gains against Geometry and the mean-logit control. LoveDA D enters the domain mean once; P and D foreground remain reported. If passed, run full eight domains unchanged. If failed, do not tune margins, iterations, donor quotas or class-specific choices using the same target outcomes.

## Novelty and publication boundary

KL projection, Sinkhorn iterations, balanced attention and marginal conservation are established machinery, not new inventions. The investigated distinction is native head-role donor conservation coupled to the original conditional Geometry intervention and exact zero-displacement writeback, rather than uniform balancing, class assignment or an independent semantic teacher. A source-priority audit and positive same-information controls are still required before a contribution claim.

The objective is a useful complete eight-domain coupling, not merely solver convergence. All current domains are development data. LandCover.ai substitutes for unlabeled iSAID; corrected IRRG Vaihingen is used. Matched VIPProxy is not full official VIP. No superiority, SOTA or CVPR readiness is claimed before complete verified evidence.
