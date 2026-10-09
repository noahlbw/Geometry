# Family pre-salience: fixed mask-free diagnostic

h0=mean-one slot softmax(s/tem), h1=mean-one family softmax(mean_g(s)/tem) broadcast to slots. M0=G/20, M1=1/m_family. Wuv=LSE(tau*original_BF16_scaled_raw*h_u+logM_v)/tau. Four00/01/10/11, plus same-size alias-to-family identity shuffle(seed20261011). Exact neutral uniform-salience fallback. Fixed G/local/Geometry/H/g/words/templates/views.

Three frozen first/middle/last whole images/domain, no mask reads. Record amplification, wide field, H writeback and final prediction changes. Replay00/01 prior fields exactly. This panel shows action only, never mIoU/semantic usefulness.

Do not full-score a rounding-only/no-prediction-action candidate. If nonrounding11-versus01 action exists on both domains,11 is the sole predesignated full candidate: must beat00/01/same-size identity shuffle with positive paired95% lower bounds on both full domains, exceed54.3/29.1, and pass double6x singleton cost. No temperature/gain tuning or control promotion.

| Domain / image | Max amplification delta | Mean amplification RMS | Mean salience entropy |11-01 wide RMS|11-01 writeback RMS|11-01 changed pixels|
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdd/DJI_0007.JPG | 0.02441454 | 0.00319304 | 0.99991990 | 0.00252259 | 0.00057222 | 3548 |
| vdd/DJI_0826.JPG | 0.01377404 | 0.00137950 | 0.99994598 | 0.00165072 | 0.00038987 | 1057 |
| vdd/DJI_10708.JPG | 0.01475745 | 0.00239224 | 0.99994023 | 0.00140951 | 0.00031597 | 0 |
| ade150/ADE_val_00000001 | 0.04593402 | 0.00683080 | 0.99997285 | 0.00363294 | 0.00086163 | 689 |
| ade150/ADE_val_00001001 | 0.03499043 | 0.00754338 | 0.99997205 | 0.00386842 | 0.00095193 | 792 |
| ade150/ADE_val_00002000 | 0.03162658 | 0.00804869 | 0.99997165 | 0.00515782 | 0.00129957 | 297 |

Nonrounding candidate action on both domains: True.
No target masks or IoU were used. Nonzero changes do not establish usefulness. The membership shuffle preserves each family size; it changes alias identity, not just group names. No coefficient/input/readout tuning.

Sources/words/profiles retain developed-label provenance; this diagnostic does not create independent validation. Historical433ADE31.1862 remains the strong separate incumbent.
