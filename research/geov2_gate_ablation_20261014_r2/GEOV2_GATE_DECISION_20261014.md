# Geov2 structure writeback and adaptive gate decision

Geov2 keeps structure writeback as the final reader. The equal-logit mean is a matched control, while the adaptive branch changes only the step taken after the same structure-anchored target has been solved.

For local logits (L), wide logits (W), degree matrix (D), and normalized visual relation (S), the target is

\[
 Z^* = \arg\min_Z \frac12\|D^{1/2}(Z-L)\|_F^2
       + \frac{\lambda}{2}\|S D^{1/2}(Z-W)\|_F^2 .
\]

The implementation solves this target in FP32 with a Cholesky factorization. The released output then takes a bounded, image-only step:

\[
 Z = L + g(x)(Z^*-L),
\]

where, for patch (i),

\[
 r_i=\operatorname{clip}\frac{\langle Z_i^*-L_i,W_i-L_i\rangle}
 {\|W_i-L_i\|_2^2+\epsilon},\qquad
 q_i=2\sigma((m^W_i-m^L_i)/\tau)-1,
\]

\[
 g_i=\operatorname{clip}(1+\eta r_iq_i,0.5,2.0).
\]

Here (m^L,m^W) are normalized top-two response margins, (eta=0.75), and (	au=0.25). The gate uses only the two frozen observations and (S); it never reads masks, labels, class frequencies, or a development score. (r_i) measures whether the structure-supported correction agrees with the observed cross-view displacement. (q_i) increases the step when the wide view is more decisive and damps it when the local view is more decisive. If the two views agree, the correction is exactly zero and the gate is set to one.

## Matched screen

The four fixed 16-image development sets use one backbone, alias bank, background rule, 24 slots, beta 0.5 and query temperature 0.1 for every row.

| Dataset | (Z^*), fixed (g=1) | Fixed (g=2) | Adaptive gate | Equal fusion |
|---|---:|---:|---:|---:|
| VDD | 61.4631 | 59.8281 | **61.6014** | 61.1622 |
| Potsdam | 50.9296 | **52.6104** | 50.9271 | 52.3331 |
| ADE150 | 20.9202 | **23.2339** | 20.7692 | 22.6201 |
| VOC20 | **97.6090** | 78.6806 | 97.4065 | 97.3835 |
| Equal-domain macro | 57.7305 | 53.5883 | **57.6761** | **58.3747** |

The adaptive gate is within 0.0544 pp of the fixed structure target on this screen, improves VDD by 0.1383 pp, and avoids the large cross-domain failures of a global (g=2). Equal fusion is higher by 0.6987 pp on this small screen and remains reported as a control. The frozen 2026-10-09 full remote-sensing protocol separately measured structure coupling at +2.1869 pp over equal fusion; that result is the reason the structure writer remains the paper's final mechanism, while this Geov2 screen is treated as a development mechanism analysis.

The first gate formulation, which used the absolute wide-margin sigmoid as an unconditional attenuation, scored 55.3282 macro mIoU. It is retained as a failed ablation in `research/geov2_gate_ablation_20261014/`; the centered gate above removes that failure mode without changing (Z^*).

## Cost

The end-to-end Geov2 runner was also timed on the same A800 protocol (two warmups and seven synchronized repeats per image). Adaptive versus fixed-target writer was:

| Dataset | Adaptive ms / MiB | Fixed target ms / MiB | Extra time | Extra memory |
|---|---:|---:|---:|---:|
| VDD | 163.3091 / 1632.16 | 161.2219 / 1632.16 | +2.0873 ms (+1.30%) | 0 MiB |
| VOC21 | 141.6551 / 1634.48 | 137.6713 / 1634.48 | +3.9839 ms (+2.90%) | 0 MiB |

The gate-only arithmetic on an NVIDIA A800-SXM4-80GB, with 1024 patches and 21 classes, measured 0.421 ms median, 0.447 ms p95, and 0.504 MiB peak allocated over 100 synchronized repeats. The encoder and the (Z^*) solve are identical between fixed and adaptive writers; the end-to-end numbers confirm that the gate adds no measurable peak memory and only a small latency increment. Compared with the old Efficient references (VDD 309.98 ms / 2622.03 MiB and VOC21 153.29 ms / 2179.69 MiB), the released adaptive runner is 47.3% faster and 37.8% lower in allocated memory on VDD, and 7.6% faster and 25.0% lower on VOC21. These are separate implementation/protocol measurements and should be reported with their configurations.

The implementation and controls are in `DINOtool/dinotool/geov2.py`, `DINOtool/dinotool/natural_static_search.py`, and `DINOtool/scripts/eval_geov2_gate_ablation.py`. The complete screen and raw per-image confusions are in this directory.
