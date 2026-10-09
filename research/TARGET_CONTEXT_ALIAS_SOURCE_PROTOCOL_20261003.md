# Frozen Target/Context Alias Source Experiment

Signature geometry-target-context-alias-source-v1-20261003. This is a window-only source/coupling pilot, not full-dataset deployment or evidence of novelty.

Same64 images/keys and top-left512 windows as alias_action_capacity_audit_20261003; eight per domain, LoveDA P/D share images, means count D once. Clean original20 only in this first evidence test. Fixed corrected IRRG and LandCover.ai substitution. No target-label fitting; all previous domains are development domains.

Original Geometry and whole-image VIP broad observations/views/templates/salience stay unchanged. Reuse prior pre-mask exact-system H, local/broad logits and baseline/source controls, and verify replay before masks load. FP64 H is the prior numerical action diagnostic, not a new inference objective or claimed solver contribution. All primary/control candidates use the same H; the inherited old-source CG arm remains explicitly labeled.

## Frozen New Measurement

Sixteen class-agnostic support units per window: a4x4 fixed128px grid. Average the original Geometry query rows for each unit over valid queries. Normalize the valid donor weights as (w-min(w))/(max(w)-min(w)), with epsilon1e-6. Invalid donors are0. A constant field or effective support<4 for either soft support or complement is unknown. No class winners or tested aliases define these masks. Supports are not asserted to be object masks.

Interpolate32x32 masks to512 with the original bilinear/align_corners=false convention. Map into the ORIGINAL resized whole-image336 crops using pixel-centre coordinates; support outside the top-left512 window is0. No new zoom/crop/FOV advantage. Each query uses its grid unit's intervention; all needed crop stencils/overlap coefficients remain unchanged.

Two fixed RGB fills: ImageNet mean (.485,.456,.406), and the original visible crop's per-channel mean. Padding stays original. Encode target-kept M*I+(1-M)*F and target-removed (1-M)*I+M*F with the existing frozen VIP observer. All source alias scores use the SAME original ImageNet query features, FP32 dot product with their unnormalized template mean (algebraically the mean template cosine), BEFORE salience/logit scaling. The source's unmasked reference is evaluated by that same FP32 expression; the original writer's BF16 profile remains untouched. Aggregate to query coordinates with original crop stencils.

For each fill, contextual dependence is positive(s0-skeep)/(abs(s0-skeep)+abs(s0-sremove)+epsilon). Canonical target-kept responses give p=softmax(skeep/.07). Rival leakage is positive(p_d-p_c)/(p_d+p_c+epsilon). Risk is max_rival min_fill(dependence*leakage), bounded0..1. Canonical aliases stay risk0; unknown/invalid units stay risk0. ContextOnly control removes the rival factor. No canonical TEXT-cosine prerequisite, quota, global blacklist or survivor renormalization.

Matched shuffled-support control permutes the mapped soft-mask pixels inside each crop's visible rectangle, preserving exact per-unit area/value spectrum and padding; seed20261003+crop_index. Both fills use the same shuffled mask. Three alias controls permute retention within each class's NONCANONICAL slots, protecting canonical words and exact retention spectrum; seeds20261003..20261005.

Original suppression-only writer: q=softmax(beta*r); delta=log(1-sum((1-rho)*positive(q-1/K)))/beta, with original profiled r, K20, original beta/salience/assembly. Exact audit endpoint z=baseline+H*delta. MeanLogit controls use .5*(g+b) and .5*(g+b+delta), with the same original source. No new solver, encoder, salience recalibration or threshold.

All observations, masks, risk/retention and endpoints are cached before evaluation labels. No audit-discovered words are deleted. Actual confusions/transitions and useful-coverage losses matter, not ranking alone. Report standard and paired fixed-class-set mIoU; fixed set is the original exact coupling's union-positive set per protocol, zero IoU if a scored class disappears. Report activity/fallback, per-class precision/recall/area, runtime and memory. These interventions are OOD and correlated-model measurements, not ground truth or certified causal semantics.

## Gate And Scope

Source-pilot advancement: mean gain>=.1pp over original exact coupling, wins>=5/8 domains, no protocol loss>1pp including LoveDA P, mean greater than all three alias shuffles, shuffled spatial support and ContextOnly, and primary above same-source MeanLogit. These are scheduling rules, not significance. Also report incremental gain over each matched unmodified reconstruction baseline; do not attribute the pre-existing anchored-vs-mean difference to the new source.

If this gate fails, preserve results and original model; no full rollout, strength sweep, word bans or per-domain winner routing. If it passes, run the previously declared complete-image96/constructed stress gate, then real raw-LLM vocabulary validation and full20,092 only after those gates. No SOTA or CVPR-complete claim from64 windows.

Authorized idle physical A800 GPUs0-7, unique result root and session prefix gtca03. Launcher refuses existing outputs and occupied GPUs. Mask-free real-checkpoint smoke and nine focused mathematical tests precede workers. Existing unrelated jobs are preserved.

The first mask-free smoke failed before labels because an alias-shuffle index tensor was on CPU while members were on CUDA. The fix creates the permutation on the members' device and adds GPU coverage to the existing shuffle test; formulas, parameters, masks, seeds and scoring are unchanged. Preserve failed results/target_context_alias_source_20261003. Retry in results/target_context_alias_source_r2_20261003 with the same signature and protocol.
