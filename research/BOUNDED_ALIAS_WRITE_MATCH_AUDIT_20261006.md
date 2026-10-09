# Fixed-Action, Magnitude-Matched Alias Writeback Audit

Same96 developed COMPLETE inputs; no deployable selector or new observations. All20 source words, salience, G/H and original full-size probability assembly stay fixed. Privileged actions use labels and are identical across writers. These are diagnostic scores, not model scores, independent validation or attainable upper bounds.

| Domain/protocol | Baseline | H | TwoH | DirectLegacy | DirectMatched | GeometryMatched | PositiveHMatched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 46.8468 | 49.6654 | 52.6061 | 54.3278 | 50.3010 | 49.7781 | 49.6593 |
| potsdam/potsdam | 51.5168 | 55.0269 | 58.2671 | 60.6149 | 55.8277 | 55.1747 | 55.0203 |
| udd5/udd5 | 47.0786 | 53.2101 | 57.8576 | 60.7355 | 54.2970 | 53.3860 | 53.2035 |
| oem/oem | 25.2530 | 26.6876 | 28.2864 | 30.7788 | 27.9119 | 26.8820 | 26.6741 |
| loveda/P | 67.2104 | 70.9122 | 73.9601 | 75.9533 | 71.5892 | 71.0972 | 70.9043 |
| loveda/D | 39.4705 | 42.6956 | 46.2012 | 48.6095 | 43.5844 | 42.8118 | 42.6859 |
| vaihingen/vaihingen | 49.1468 | 52.2969 | 55.3722 | 58.0088 | 53.2005 | 52.4543 | 52.2916 |
| landcoverai/landcoverai | 76.2441 | 77.4833 | 78.6032 | 79.1373 | 77.7820 | 77.5976 | 77.4828 |
| flair1/flair1 | 37.6036 | 42.1697 | 45.9050 | 48.1416 | 42.9233 | 42.3650 | 42.1657 |

The six privileged columns use the SAME label-assisted donor actions. Matched routes have H-matched whole-image valid-patch squared norms per action/class; this is not equality of native-pixel probabilities. DirectLegacy alone retains historical padded field writes.

| Writer | Privileged policy | Same H-chosen global deletion | Same H-chosen per-image deletion |
| --- | ---: | ---: | ---: |
| H | 49.9044 | 47.8547 | 48.6278 |
| TwoH | 52.8874 | 48.6759 | 49.3301 |
| DirectLegacy | 55.0443 | 48.6026 | 49.2635 |
| DirectMatched | 50.7285 | 47.8334 | 48.6122 |
| GeometryMatched | 50.0562 | 47.8561 | 48.6417 |
| PositiveHMatched | 49.8979 | 47.8552 | 48.6280 |

Baseline eight-domain mean: 46.645024; LoveDA D once.

No alternative writer chooses a new label-optimal word. H-chosen actions include no action and favor the H reference by construction; all selections remain privileged audit, not tuning.

All96 ordered unique complete keys, original three per-image endpoints, base checkpoint/vocabulary identities, NPZ aggregate/target counts and previous H/DirectLegacy single/privileged confusions verify. Norm errors and zero-template fallbacks are retained per protocol. Diagnostic wall time is not model latency. No selector, writer, threshold or word list is promoted.

## Interpretation

All following scores are LABEL-ASSISTED DIAGNOSIS, not improvements achieved by
a deployable model. The source actions are identical across all six writers.

1. Amplitude is an important competing explanation for the old direct-versus-H
   gap. Doubling the same H-routed donor action changes the privileged mean
   49.9044->52.8874 (+2.9829pp), with8/8 domain gains. DirectLegacy55.0443 drops
   to DirectMatched50.7285 when matching H's valid-field action norm and padding
   contract. Therefore the original5.1398pp direct-versus-H gap cannot be read
   as purely a spatial-mixing penalty. Padding and nonlinear image assembly
   remain distinctions; these numbers are not a percentage causal attribution.
2. Localization still matters for the privileged sparse donor policy.
   DirectMatched exceeds H by0.8240pp, improving8/8 domains and LoveDA P;
   GeometryMatched gains0.1517pp with the same domain pattern. The norm matching
   is accurate to6.5229e-16 relative squared-power error and no fallback occurs.
   This supports a writing-dependent opportunity for these specific labeled
   donor actions, not a generally better writer or a ready unlabeled selector.
3. The result depends on the action regime. For SAME H-chosen global removals,
   DirectMatched47.8334 is below H47.8547 (-0.0213pp); for same H-chosen per-image
   removals,48.6122 is below48.6278 (-0.0156pp). H selection favors H by
   construction. These results do not show universal spatial-writing damage,
   and cannot be exchanged for the localized privileged-policy comparison.
4. Merely dropping H's negative entries is not a useful correction here.
   PositiveHMatched privileged49.8979 is0.0065pp below H; its global/image
   single-action means differ from H by only+0.0005/+0.0002pp. Negative matrix
   entries alone do not explain the observed practical deficit.

## Coverage And False Activation

These compare the SAME label-assisted donor policy under matched direct versus
original H, after complete original-image restoration. They do not describe an
unlabeled model's gain.

| Domain | Class | H IoU | DirectMatched IoU | Delta TP pixels | Delta FP pixels |
| --- | --- | ---: | ---: | ---: | ---: |
| VDD | vehicle |26.8827|27.7309|+2515|-18546|
| VDD | road |16.5955|17.0151|+18940|-95624|
| Potsdam | car |37.2070|37.6479|+332|-4336|
| Potsdam | low vegetation |64.1450|65.0636|+15055|-9092|
| UDD5 | vehicle |20.9928|22.4643|+36093|-711314|
| UDD5 | road |52.5443|53.7315|+357943|-957597|

Localized writing can both restore true coverage and limit competitor false
activation in this privileged action space. It remains unknown how to identify
these donor actions without labels. Do not fit their word identities, action
magnitudes or class/domain routes into a selector.

## Research Decision

Retain the established frozen performance reference and original reconstruction
for ordinary local/wide fusion. H solves a regularized reconstruction, not
nonnegative probability transport: its structural eigenvalues are nu/(1+nu),
so weakly supported correction directions are attenuated. This algebra alone
does not identify which image frequencies or object sizes caused the deficit.
The useful next distinction is between writing an independent broad semantic
observation and writing a localized competitive word intervention; they need
not share an identical attenuation assumption.

Do not promote TwoH, DirectMatched or GeometryMatched, or tune a reconstruction
coefficient from these labels. A future unlabeled proposal needs both reliable
action identification and same-action magnitude/locality attribution, while
preserving local fidelity. A new sigmoid, third encoder, sign clipping or renamed
solver is not justified by this audit. All17 CPU tests and both mask-free smokes
passed; all96 complete inputs' endpoints/actions verified and all sessions exited. The
three-part accuracy/efficiency objective remains open.
