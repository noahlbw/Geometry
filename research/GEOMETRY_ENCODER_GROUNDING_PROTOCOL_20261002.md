# Native encoder-position grounding: prospective source/coupling test

The previous mask support candidate improved the rectangular readout but remained worse than Geometry on five domains. It cannot certify semantic identity, even with precise supports. This continuation tests the existing GroundingDINO encoder-position classification, before the object queries and final boxes, as a different local semantic measurement. It does not claim Astra endorsed this exact source, a final model, or a verified novelty gap.

Physical A800 GPUs0-7, following the user's latest authorization; check occupancy before each launch. Same frozen DINO/Geometry source,20 aliases/class, exact token spans and192-token round-robin captions. Same pinned public GroundingDINO tiny safetensors revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. No additional checkpoint, detector threshold, label fitting, alias filtering, query cap or dataset routing. This adds detection pretraining; the source is not a semantic-mask-trained pixel classifier or a matched single-encoder system.

## Measurement

Native `enc_outputs_class` classifies the memory at encoder spatial positions before query selection/decoding. Source inspection confirms this exact author/Transformers path. The checkpoint's actual classification response is used, not a fabricated cosine on an unaligned hidden feature. Use the unchanged phrase mean of sigmoid token logits over exact alias spans. Capture actual feature-level shapes and validity; additionally exclude locations with nonfinite native proposal coordinates. Special/padding text tokens are not aliases.

Interpolate each level's probability numerator and validity to the original32x32 patch lattice. Divide pooled numerator by pooled validity so levels contribute equally where observed, rather than by their different native pixel counts. All20 aliases are averaged arithmetically per class. Let `b=.07*log(class_probability)` so `softmax(b/.07)` equals class-normalized native response. The FP32 tiny floor is numerical, not a fitted confidence cutoff. Unobserved/all-class-zero locations return original Geometry exactly. Missing observations do not certify class absence. The standard forward computes decoder outputs but they are not used for this measurement; account for that cost.

## Complete candidate and controls

Reuse the previous fixed Geometry fidelity operator: `min_z .5||z-g||^2+.5||A(z-b)||^2`, with original validity-restricted Geometry relation A. The quadratic solver and borrowed encoder classification are not original inventions. The source must first be useful and the complete coupling must outperform equal-source controls before developing a genuinely original final readout around it.

At numerically unobserved/all-class-zero locations, overwrite the final coupled score with the exact original Geometry score as well as applying input fallback. This prevents neighboring residuals from violating the stated no-information policy. LoveDA P reuses the same D observation and exact `(class, alias)` mapping; it does not repack a foreground-only detector caption.

Controls: Geometry; EncoderGrounding alone; fixed probability mean; fixed logit mean; identical coupling with spatially shuffled Geometry; primary Geometry_EncoderCoupling. Keep one fixed model across every domain. No automatic full rollout. Source images and raw alias probabilities/candidate scores are saved before labels.

## Fixed screen and gate

Reuse exactly96 development image IDs and the same one/two image-only512 windows per image: full40 UDD5 IDs and eight from each of seven other domains. NOT full-image dataset mIoU; overlaps may duplicate pixels. Corrected IRRG Vaihingen; LandCover.ai replaces unlabeled iSAID. LoveDA D enters the equal-domain mean once, P separately. All datasets are development, not untouched validation.

Require exact original Geometry score/confusion recovery,20 aliases/class, pinned source/config, complete unique sample/window coverage, finite native measurements, identity no-information behavior and endpoint transition reconstruction. Promote only if primary retains Geometry mIoU on every protocol, improves equal-domain mean over both simple fusions, and beats both fusions on VDD/Potsdam. A pass authorizes the same full-image screen, not CVPR readiness or official-VIP superiority. Raw beneficial/harmful counts are diagnostic, not a replacement for mIoU.

If it fails, preserve the source and results. Do not retune source scale, sigmoid/phrase aggregation, interpolation levels or reconstruction weight on these masks. Detection-position supervision may be inappropriate for dense stuff classes; this fixed test can reject that route, not every frozen grounding model.
