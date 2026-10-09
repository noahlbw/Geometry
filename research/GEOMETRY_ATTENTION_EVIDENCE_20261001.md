# Geometry attention evidence: one fixed semantic observation

The native-path transport and virtual-region CLS candidates failed their four-domain gates. Whole-margin density calibration severely damaged existing correct predictions and promoted losing classes through negative intercepts. These results do not justify another confidence gate.

Original feature audits show that residual states carry much of the car/vehicle margin even at false positives. MLP and final LayerNorm bias often have large opposing contributions. Conditional attribution is not proof that removing either is beneficial; preserving all native residual/MLP in the previous frozen-path trial did not resolve the error. Test a different actual information source rather than reinterpret the same probability as truth.

## Frozen hypothesis

Advance every head donor along the exact native path `n[l+1]=NativeBlock(n[l])`. Read `e=sum_l GeoAttention(native_l)` using the unchanged original Geometry relation, native prefix mass, frozen Value projections and LayerScale. The descriptor uses the original final LayerNorm/projection of e, not `n[L]+delta`. Residual/MLP update the donors but are not explicitly summed into this evidence readout. This is an attention-only idea related to ClearCLIP, not an established new contribution or guaranteed transfer to DINO.text.

All20 alias strings and six RS templates, .07 normalized LME, 512/128 sliding windows, Hann blending and frozen weights remain unchanged. There is no VIP observer, extra crop, image-level CLS, density fit, alias deletion, global mean subtraction, or learned router. Native replay must be exactly equal to prepared native features at every window.

Primary `AnchoredEvidence` uses the already-fixed unit-weight objective `min_z .5||z-g||^2+.5||A(z-b_attention)||^2`. Controls are original Geometry, exact Native, attention-only Native, attention-only Geometry and MeanLogitEvidence. The learned final normalization remains; retaining it may itself limit the observation. No alternative normalization or class-dependent branch is chosen from outcomes.

## Predeclared gate

Use only A800 physical GPU indices 4-7 (the fifth through eighth cards), one shard each for full UDD5 40, full OEM 384, fixed diagnostic VDD 8 and Potsdam 8. GPU indices 0-3 remain untouched. Every control must match original Geometry confusion and exact keys/vocabulary. Transitions record `(old,new,GT)`. Require primary gains on both diagnostic datasets and nondegradation on both full transfer datasets, then compare against same-source fixed fusion. Otherwise reject this hypothesis without eight-domain expansion. These are exploratory development evaluations, not independent validation.

User clarified that an original, VIP-independent complete readout is the main research priority. This attention-only candidate is a mechanism probe, not the claimed novel final architecture. Its provenance must remain explicit even if a dataset improves.

No successful reliable correction is claimed before verified outcomes. A numerical anchor or removal of explicit residuals does not certify semantic correctness.

## Verified four-domain outcomes

| Dataset | Coverage | Geometry | EvidenceGeometry | MeanLogitEvidence | AnchoredEvidence |
|---|---:|---:|---:|---:|---:|
| UDD5 | 40 full | 50.5553 | 50.2979 | 51.7544 | 51.1317 |
| OEM | 384 full | 44.6097 | 43.0625 | 44.9331 | 44.9063 |
| VDD | 8 fixed diagnostic | 46.5339 | 41.4560 | 46.3358 | 45.4774 |
| Potsdam | 8 fixed diagnostic | 42.5695 | 44.4032 | 44.8181 | 44.4948 |

All unique keys, all20 vocabulary SHA and original Geometry confusion matrices match their references; native donor replay error is exactly zero. The primary fails the VDD gate and is below the same-source MeanLogit control in all four evaluations. Do not expand this primary to eight datasets. Attention-only Geometry supplies some complementary information without VIP but does not resolve which residual semantics should be retained.

In the VDD diagnostic, primary vehicle IoU rises 13.5825 to20.0416 and vegetation75.4744 to80.0472, while water78.8724 falls to67.7623 and roof70.5414 to67.3689. This directly rejects uniform explicit-residual removal as reliable semantic correction. Potsdam car5.8594 rises to8.6009, tree51.6910 to58.1298 and lowveg65.3138 to66.2575, but building77.5183 falls to75.9449.

Primary beneficial/harmful pixel changes: UDD5 9617085/9162363; OEM 8447153/6719830; VDD 2987329/2969064; Potsdam 428816/109681. Combined six-arm wall time/peak MiB: UDD5501.63/3817.90, OEM379.18/3818.57, VDD134.59/3819.81, Potsdam7.99/3816.21. These timings are not standalone inference latency.

All four merged files are local: full UDD5/OEM under `research/geometry_attention_evidence_v1_*_full_20261001_results.json`; diagnostics under `research/geometry_readout_diagnostic_8_20261001/*_geometry_attention_evidence_v1_merged.json`. Transient OpenSSH connections ended, but authoritative tmux/result checks confirmed all runs terminal and complete; no run was restarted.
