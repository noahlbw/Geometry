# Frozen taxonomy-driven calibration

Five full protocols: VDD80, Potsdam504, VOC211449, Context605105, ADE1502000. Reference unchanged bounded Geometry_PatchOnly2Coupled. Methods: Reference, TaxonomyAdaptive, TaxonomySoft. Same visual backbone observations across methods, no fine observations or confidence-to-background rejection.

## Natural tasks

Use the same variable-count qualified alias pool and segmentation templates on every natural task. Normalize each canonical text embedding, centre across classes and form its Gram matrix G. Effective rank is tr(G)^2 / ||G||_F^2. Divide by the number of classes to obtain r; use r/(1-r) as the candidate gain. Choose the nearest gain in logarithmic distance from the already established0.5/1/2 menu. Gain0.5 uses original Geometry (native prefix participation); gains1/2 use patch-only strength2. No image masks or per-dataset names enter this selection. Temperature0.07 and wide tau1/tem1 are shared.

The observed r values for the frozen candidate pools are VOC210.6243, Context600.5225 and ADE1500.3416. The rule therefore chooses gains2,1,0.5 respectively. This mapping is an explicit hypothesis: text redundancy may call for more local/prefix semantic fidelity and less wide-view correction. Effective rank does not measure segmentation accuracy, object size or visual class separability. The rule was devised after examining prior developed curves, so it is not independent evidence of a universal relationship.

## Remote sensing

Two fixed, developed routes: original20/ImageNet with patch strength1; focused20/RS6 with patch strength3. Both use gain0.5, local temperature0.07, wide tau1/tem1. Choose the route per image by largest mean canonical top-two cosine margin on existing original Geometry patch features, excluding residual-class queries and padded patches. This is a lexical-view selection hypothesis; a larger margin can still be a confident error. RS route strengths retain labelled-development provenance.

## Soft weights and verification

TaxonomySoft keeps the same selected route/gain and adds the previously implemented rival-conditioned, canonical-anchored soft alias weighting locally. It isolates whether that weighting helps under these new readout choices. Wide alias aggregation remains inherited VIP aggregation.

Masks are loaded only after all three predictions are frozen. Require mask-free real-image smoke checks, frozen head/backbone checks, unique full coverage, exact original confusion replay and matched scored targets. Report full and prior nondevelopment-complement scores, residual coverage and every class outcome. Extra head routes used for RS route selection and paired comparisons must be counted when benchmarking deployed cost; backbone cap alone does not prove equal runtime.

Retain all prior outputs. Candidate banks and rule design use previous research evidence; only online selection is mask-free. These full results remain exploratory, not a claim of untouched zero-shot SOTA or CVPR readiness.
