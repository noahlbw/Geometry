# Existing Soft Alias: Label-Free Word-Action Path Audit

Two existing first/last developed complete inputs/domain,16 total; LoveDA P/D share images. No masks or model changes. Exact original scores and complete predictions are verified. Unweighted valid-tile means below describe numerical fields, NOT accuracy, semantic correctness or deployment latency.

| Domain/protocol | Within-class risk power | Word/total directed norm | Gradient word power | Word H retention | Common H retention | Word/total write norm | Patch argmax changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/vdd | 0.730931 | 0.422864 | 0.392954 | 0.142715 | 0.133409 | 0.382452 | 0.001343 |
| potsdam/potsdam | 0.774627 | 0.414792 | 0.482420 | 0.134297 | 0.118674 | 0.407112 | 0.001831 |
| udd5/udd5 | 0.799384 | 0.524798 | 0.579737 | 0.115641 | 0.106143 | 0.532156 | 0.002376 |
| oem/oem | 0.778351 | 0.525917 | 0.389235 | 0.126721 | 0.100801 | 0.516685 | 0.002319 |
| loveda/P | 0.782096 | 0.478118 | 0.572395 | 0.155554 | 0.159186 | 0.449305 | 0.002930 |
| loveda/D | 0.789362 | 0.490908 | 0.523414 | 0.159303 | 0.159399 | 0.454143 | 0.006592 |
| vaihingen/vaihingen | 0.736362 | 0.354334 | 0.589075 | 0.107330 | 0.098400 | 0.356314 | 0.002319 |
| landcoverai/landcoverai | 0.694436 | 0.402889 | 0.371315 | 0.136833 | 0.155339 | 0.325409 | 0.000610 |
| flair1/flair1 | 0.687739 | 0.424459 | 0.382247 | 0.165471 | 0.164701 | 0.360457 | 0.009521 |

## Interpretation Boundary

Risk power excludes canonical/self/padded slots. Gradient/cycle decomposition is orthogonal on the complete class graph. Relative norms are not additive accuracy contributions and may exceed one. H retention is a squared-norm ratio, not object recall. Changed patch argmax is a disagreement rate before interpolation/softmax/stitching, not a native-pixel error rate. These observations can distinguish source variation from projection/write attenuation; they cannot choose a label-optimal selector, word, threshold or route.
